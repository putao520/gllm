# JIT 数值诊断方法论 领域资料库

> 来源：项目 BCE 沉淀（BCE-20260706 SmolLM2 logits 发散 + BCE-20260708-GGUF-QUANT Q4_0 乱码）+ diagnostic API（client_impl.inc.rs:313-351）+ DiagnosticScratchpad（pack_observe.inc.rs:629）+ generateloop-diagnostic-semantics.md
> 建库触发：Q4_0 乱码排查中数值诊断二分法证明高效（architect 4 轮静态盲猜全错 vs 数值诊断 1 刀定位 prefill 路径）；静态读代码 + numerical_sim 单测对 JIT 数值 bug 无效，必须数值诊断
> 最后验证：2026-07-08

---

## 核心机制：三层二分法（O(log n) 定位 JIT 数值 bug）

JIT 推理乱码/logits 发散时，**禁止静态读代码盲猜**（architect 4 轮全错的教训）。按三层二分法逐层收敛：

```
乱码/logits 错
  │
  ├─ 第1刀: prefill_logits 二分（区分 prefill vs decode 坏）
  │   ├─ 范围合理（无 NaN/inf，max_abs≈60）→ prefill 路径数值错（非崩溃）
  │   ├─ 全垃圾（NaN/inf/全零）→ embed(第一个 op) 或 layer0 崩溃
  │   └─ argmax 错但范围对 → 计算路径数值错（非 decode 坏）
  │
  ├─ 第2刀: 逐层 bisection（定位首个发散层）
  │   ├─ Ring-Buffer capture 单遍捕获全 N 层（diagnostic-layer-capture feature）
  │   ├─ GLLM_SINGLE_LAYER=1 / GLLM_DEBUG_LAYERS=N 隔离层循环
  │   └─ 单 token prefill 重建（方法 A，避开 GenerateLoop M=1 错位）
  │
  └─ 第3刀: 逐算子 bisection（定位首个发散算子）
      ├─ named_offsets 读中间张量（v_proj/q_proj/up_proj offset）
      ├─ Python 参考模型逐算子导出对比（diag_layer0_divergence.py）
      └─ cosine < 0.99 即发散，首个发散算子 = 根因
```

### 第1刀：prefill_logits 范围二分（最快，1 次调用定方向）

`diagnostic_prefill_logits(tokens) → Vec<f32>`（client_impl.inc.rs:341）读 **row 0**（ARCH-DECODE-LOGITS-ROW0，非 row[prompt_len-1]）。

| prefill_logits 信号 | 结论 | 下一步 |
|---------------------|------|--------|
| NaN/inf 任意一个 | 数值崩溃 | embed(QuantGather) 或 layer0 第一个 op 坏，全链垃圾 |
| 全零 | 未写出 | 检查 logits offset / output_sink 拓扑 |
| **范围合理（无 NaN，max_abs≈10-60）+ argmax 错** | **prefill 计算路径数值错（非崩溃非 decode）** | 进第2刀逐层 |
| 范围合理 + argmax 对 + gen_text 乱码 | decode 路径坏（KV cache / 采样） | 查 decode 专用路径 |

**BCE-20260708-GGUF-QUANT 实证**：prefill_logits `nan=0 inf=0 max_abs=60.15, 值[0.70,23.5,1.86,...]`（范围合理）但 `argmax=121034`（期望 Paris≈7310）→ 1 刀锁定 prefill 路径数值错，**推翻** architect 前 3 轮"decode 坏/QuantGemm 嫌疑"的静态盲猜。

### 第2刀：逐层 bisection（定位首个发散层）

**方法 A：单 token prefill 重建**（推荐，避开 GenerateLoop M=1 错位，详见 generateloop-diagnostic-semantics.md）
- 对每位置 i：`diagnostic_prefill_scratchpad(&tokens[0..=i])` 读 row0（最后 token 输出）
- 拼 `[seq, hidden]` vs `golden hidden_layer_i`
- layer 0 ops（embedding/gather/q_proj）无 KV 依赖，单 token 可重建
- 找首个 `cosine < 0.99` 的层

**Ring-Buffer capture**（diagnostic-layer-capture feature 门控，默认关，生产零开销）
- `diagnostic_layer_capture_stride()` 返回 per-layer stride（bytes）
- 单次 forward 捕获全部 N 层输出到 capture 区
- 读第 N 层 = `capture_offset + N * stride`
- **前提**：`is_layer_group=true`（assign_homogeneous_markers 对 "layer." 前缀 op label 设置）+ close_layer_loop emit capture copy

**层循环隔离**（gllm-kernels pipeline.inc.rs:850-860）
- `GLLM_SINGLE_LAYER=1`：层循环只跑 1 次（layer 0），中间张量不被后续层覆盖
- `GLLM_DEBUG_LAYERS=N`：层循环跑 N 次（控制调试范围）
- 用途：单层诊断时让 named_offsets 中间张量（v_proj/q_proj 等）唯一 offset 可读

### 第3刀：逐算子 bisection（定位首个发散算子）

**named_offsets 中间张量读**（DiagnosticScratchpad.named_offsets，pack_observe.inc.rs:643）
- `diagnostic_tensor_offset("v_proj")` 动态查询 scratchpad byte offset
- `scratchpad.read_dtype_aware(offset, count)` dtype 感知读（BF16 vs F16 区分指数位）
- 读唯一 offset 的中间张量（共享 offset 的被后续层覆盖）

**Python 参考模型逐算子对比**（`tests/e2e_alignment/diag_layer0_divergence.py`）
- transformers 加载同模型，手动跑 layer0 各算子，导出中间结果（rmsnorm1/q/k/v/rope/attention/o_proj/resid1）
- 参考正确性验证：ref embedding vs golden h0 cos=0.99999，ref layer0 vs golden h1 cos=0.99999
- 逐算子 `gllm norm vs ref norm + cosine`，cosine < 0.99 即发散

**BCE-20260706 实证**：GLLM_SINGLE_LAYER=1 + named_offsets 对比，定位 **v_proj GEMM cos=0.0333**（几乎正交）= 首个发散算子 → 根因 `emit_gemm_trans_b_inline` 混合精度 K维 stride bug（A/B 共享 offset，A 漏读一半）。

### 三路互补诊断（architect sessionId 088a2b41 推荐）

| 路 | 方法 | 用途 | 依赖 |
|----|------|------|------|
| **A** | 单 token attention 恒等式 | seq=1 时 attention=V（softmax over 1 key=1.0），不依赖 golden 验证 attention 链 | RMSNorm 输出 norm ≈ ‖weight‖ 推断 |
| **B** | Python 参考模型补 golden | transformers 导出每算子中间结果作 ground truth | golden model 可加载 |
| **C** | 权重字节验证 | `diagnostic_weight_blob_bytes()` dump blob，搜 golden 权重字节，排除 loader 转置/偏移/错层 | golden 权重可独立获取 |

**路 C 价值**：排除 loader bug（权重字节一致 = loader 没转置/错位/偏移），保留嫌疑收敛到 JIT 计算逻辑。BCE-20260706 路 C 确认 input_norm/q/k/v_proj 权重字节全一致 → 根因在 GEMM JIT 计算逻辑非 loader。

---

## AI 易误判点（★ 核心价值，堵幻觉）

### 1. ❌ numerical_sim 单测 PASS = 算子实现对 → ✅ 零数据跑通不验证数值

- ❌ 误判：Q4_0 GEMV / GEMM scalar reference + numerical_sim unit PASS，认为算子正确
- ✅ 正解：numerical_sim 用**零数据/合成数据**跑通管线，只验证"不 panic/形状对"，**不验证真实数值**。BCE-20260708 Q4_0 GEMV unit PASS 但 E2E 乱码
- **姿势**：单测 PASS ≠ 算子数值正确。必须用真实权重 + Python 参考逐算子对比

### 2. ❌ 静态读代码能定位 JIT bug → ✅ 必须运行时数值诊断

- ❌ 误判：4 轮静态读 GEMM/QuantGemm 代码，盲猜嫌疑（"K维 stride 对"、"dtype 传播对"）
- ✅ 正解：BCE-20260708 architect 4 轮静态盲猜全错（QuantGemm 嫌疑被穷尽验证排除），数值诊断 1 刀（prefill argmax 错）定位 prefill 路径
- **姿势**：静态读代码排除"明显错"，但 JIT 机器码数值发散必须运行时诊断（dump 中间张量 + Python 对比）。架构师静态盲猜有界（≤2 次），超限必须数值诊断

### 3. ❌ logits 乱码 = 全链坏 → ✅ 二分法先判 prefill_logits 范围

- ❌ 误判：gen_text 乱码就重写整个推理路径
- ✅ 正解：先 `diagnostic_prefill_logits` 看 NaN/inf/max_abs/argmax 四指标
  - NaN/inf → 崩溃（embed/layer0 坏）
  - 范围合理 + argmax 错 → prefill 计算路径数值错（非崩溃）
  - argmax 对 + gen 乱码 → decode 坏
- **价值**：O(n) 全链排查 → O(log n) 二分，1 次调用定方向

### 4. ❌ encode_to_layer 可逐层诊断 → ✅ 返回全零，不可用

- ❌ 误判：用 `encode_to_layer(LastToken)` / `encode_to_layer(MeanPool)` 逐层 bisection
- ✅ 正解：Step8 实测 30 层全 cosine=0.0000（LastToken 返回全零）；MeanPool 错位伪信号（gllm mean=row0/5 vs golden mean=5token 真实平均）
- **姿势**：逐层 bisection 用 Ring-Buffer capture（feature 门控）或方法 A 单 token prefill 重建，**禁用** encode_to_layer

### 5. ❌ 诊断读多行比 golden → ✅ GenerateLoop M=1 覆盖 row0

- ❌ 误判：`read_dtype_aware(off, seq*hidden)` 读 row0-seq 比 golden 多行
- ✅ 正解：decoder 模型 GenerateLoop 每迭代 M=1，只写 row0（覆盖），row1-4=0 是设计。读多行 → 错位伪信号（cosine 假低）。详见 generateloop-diagnostic-semantics.md
- **姿势**：单 token prefill（prefix=[tok0..i]）读 row0 = token i 输出；logits 读 row0（ARCH-DECODE-LOGITS-ROW0）

### 6. ❌ 单 token prefill 不能重建深层（KV 依赖）→ ✅ layer 0 ops 无 KV 可重建；attention 单 token 恒等

- ❌ 误判：attention 跨 token 依赖 KV cache，单 token 不能诊断深层
- ✅ 正解：
  - layer 0 ops（embedding/gather/q_proj/v_proj）无 KV 依赖，单 token 可重建（方法 A）
  - attention 单 token（seq=1）：Q·K^T=[n_head,1], softmax over 1 key = [1.0], attention 输出 = V（路 A 恒等式）
  - 深层需逐 token 累积 prefix（prefix=tokens[0..N+1] 读 row0）

### 7. ❌ 算子发散 = 算子实现错 → ✅ 可能是 GEMM stride 或层间传递

- ❌ 误判：v_proj cos=0.03 就认定 v_proj 实现错
- ✅ 正解：BCE-20260706 两个真根因都不是 v_proj 本身：
  - **BCE-20260706-MIXED-GEMM-STRIDE**：`emit_gemm_trans_b_inline` 混合精度 A/B 共享 K维 offset，A 漏读一半（影响所有 trans_b + a_elem≠b_elem 的 GEMM）
  - **BCE-20260706-ACTSWAP-INPUT-ALIAS**：activation_alias.input_tid 走 Intermediate，ActivationSwap 失效，layer1+ 读 embedding 非 layer0 输出
- **姿势**：逐算子发散定位后，仍需溯源是算子本身还是数据传递（stride / 层间 ActivationSwap / dtype 传播）

### 8. ❌ GPU diagnostic_prefill_logits 走 GPU PTX → ✅ 走 CPU entry_fn

- ❌ 误判：GPU 测试 diagnostic_prefill_logits 失败 = GPU PTX codegen 坏
- ✅ 正解：`diagnostic_prefill_logits` 调 `mega.entry_fn`（CPU x86 机器码），**不 launch GPU PTX**。GPU 测试 NaN 真因是 CPU x86 AVX-512 `lower_broadcast_x86` 半初始化 ZMM 高 lanes（BCE-20260706 真因）
- **姿势**：GPU 路径数值 bug 用 GPU 专用诊断（非 diagnostic_prefill_logits），后者只测 CPU entry_fn

### 9. ❌ dtype 发散根因 = derive_compute_dtype 硬编码 → ✅ 数值自洽，非发散根因

- ❌ 误判：`derive_compute_dtype` 的 `BF16|F16 => F32` 硬编码（宪法 -1 违宪）导致 logits 发散
- ✅ 正解：architect sessionId 裁决——硬编码是宪法违宪（层2），但当前对 SmolLM2 数值自洽（层1），**非发散根因**。KV cache 全 F32 自洽（运行时插桩 kv_row_stride=768=buffer 非越界）
- **姿势**：违宪≠当前 bug 根因。数值发散先用二分法定位首个发散算子，再判是否 dtype 传播断链

### 10. ❌ 权重字节对 = loader 没 bug 就跳过 → ✅ 必须 dump 验证排除 loader

- ❌ 误判：假设 loader 对（safetensors/gguf 标准格式），跳过路 C 直接查 JIT
- ✅ 正解：BCE-20260706 路 C dump weight_blob 搜 golden 权重字节，确认 input_norm/q/k/v_proj 全 offset 正确字节一致 → 排除 loader 转置/偏移/错层/tied embedding 双拷贝偏移
- **姿势**：三路互补（A 恒等式 / B Python golden / C 字节验证），路 C 是排除 loader 的确定性手段，不可跳过

---

## 解决问题时参考

### 诊断流程（按序执行）

1. **第1刀**：`client.diagnostic_prefill_logits(&tokens)` 看 NaN/inf/max_abs/argmax
   - 范围合理 + argmax 错 → 进第2刀
   - 崩溃 → 查 embed/layer0
2. **第2刀**：逐层 bisection
   - 启 `diagnostic-layer-capture` feature，`diagnostic_layer_capture_stride()` 读 stride
   - 单次 forward + capture 区逐层读 `capture_off + N*stride`
   - 或方法 A：单 token prefill prefix 重建
   - 找首个 `cosine < 0.99` 的层
3. **第3刀**：逐算子 bisection
   - `GLLM_SINGLE_LAYER=1` 隔离 layer 0
   - `diagnostic_tensor_offset("v_proj")` 读 named_offsets 中间张量
   - `scratchpad.read_dtype_aware(off, count)` 读中间值
   - Python `diag_layer0_divergence.py` 导出 ref 逐算子对比
   - 找首个 `cosine < 0.99` 的算子

### diagnostic API 工具箱（client_impl.inc.rs:313-351）

| API | 返回 | 用途 |
|-----|------|------|
| `diagnostic_prefill_logits(tokens)` | `Vec<f32>` | 第1刀：读 logits row0，判范围 + argmax |
| `diagnostic_prefill_scratchpad(tokens)` | `DiagnosticScratchpad` | 第2/3刀：含 named_offsets 中间张量 + compute_dtype |
| `diagnostic_tensor_offset(name)` | `usize` | 动态查询 named tensor scratchpad offset |
| `diagnostic_weight_blob_bytes()` | `Vec<u8>` | 路 C：blob 字节验证，搜 golden 权重 |
| `diagnostic_weight_offsets()` | `Vec<(name, off, dtype)>` | 列所有 weight tensor offset + dtype |
| `diagnostic_weight_row(name, row, cols)` | `Vec<f32>` | 读 weight tensor 单行 |
| `diagnostic_layer_capture_stride()` | `usize` | Ring-Buffer 逐层 stride（feature 关=0） |
| `diagnostic_forward_only(tokens)` | `Vec<f32>` | 单遍 forward 输出（非 generate loop） |

### DiagnosticScratchpad（pack_observe.inc.rs:629）

```rust
pub struct DiagnosticScratchpad {
    pub data: Vec<u8>,           // scratchpad 原始字节
    pub logits_offset: usize,
    pub vocab_size: usize,
    pub prompt_len: usize,
    pub hidden_size: usize,
    pub compute_dtype: DType,    // ★ dtype 感知读（BF16 vs F16 指数位不同）
    pub named_offsets: Vec<(String, usize, DType)>, // 中间张量 (name, offset, dtype)
}
// 方法
scratchpad.elem_bytes()                      // compute_dtype.size_bytes()
scratchpad.read_dtype_aware(off, count)      // dtype 感知读 → Vec<f32>
```

### Python 参考模型（tests/e2e_alignment/diag_layer0_divergence.py）

- transformers 加载同模型，手动跑 layer0：rmsnorm1 → q/k/v_proj → rope → attention → o_proj → resid1 → ...
- 导出每算子中间结果 + 测试假设（no-softmax/no-scale/no-rope）
- 参考正确性：ref vs golden cos ≥ 0.99999（可作 ground truth）
- 假设测试：对比 gllm 数值模式匹配哪个"去掉某算子"的假设（如 no-rope 放大 1.75x 匹配 gllm row4 放大）

### 环境变量（gllm-kernels pipeline.inc.rs:850-860）

- `GLLM_SINGLE_LAYER=1`：层循环 bound=1（只 layer 0），中间张量不被覆盖
- `GLLM_DEBUG_LAYERS=N`：层循环 bound=N（控制调试范围）
- **注意**：这两个是编译时读取（JIT codegen 阶段读 env），改后需重新加载模型触发重编译

---

## 已知问题 / 边界

### 诊断 API 限制

- **encode_to_layer 不可用**：LastToken 返回全零（Step8 30 层 cos=0.0000），MeanPool 错位伪信号。逐层 bisection 用 Ring-Buffer capture 或方法 A
- **named_offsets coverage**：当前 weight tensor 全覆盖，intermediate tensor offset 需补（BCE-20260708 待做：embed op 输出需补 intermediate offset）
- **GPU 路径**：`diagnostic_prefill_logits` 走 CPU entry_fn 非 GPU PTX，GPU 数值 bug 需 GPU 专用诊断

### Ring-Buffer capture 前提

- `diagnostic-layer-capture` Cargo feature 门控（默认关，生产零开销）
- `is_layer_group=true`（assign_homogeneous_markers 对 "layer." 前缀 op label 设置）
- close_layer_loop / handle_standard_layer_loop 在 LoopEnd 前 emit capture copy
- capture emit 要求 `state.in_layer_loop=true`（SmolLM2 走 GroupMarker 驱动的层循环，已确认 is_layer_group=true）

### 诊断语义陷阱（详见 generateloop-diagnostic-semantics.md）

- GenerateLoop M=1：每迭代只处理 1 token，输出写 row0（覆盖），读多行错位
- logits 在 row0（ARCH-DECODE-LOGITS-ROW0），非 row[prompt_len-1]
- 单 token prefill 重建仅适用 layer 0 ops（无 KV 依赖）；深层需逐 token 累积 prefix

### 二分法有效边界

- **适用**：数值发散（cosine 低 / argmax 错 / 范围异常）
- **不适用**：崩溃（panic / segfault）— 用 debug_process DAP；性能问题 — 用 runtime profile
- **前提**：有 golden reference（Python transformers 模型）或可构造恒等式（路 A）

### 与其他资料库关系

- `generateloop-diagnostic-semantics.md`：GenerateLoop M=1 错位陷阱（本库第2刀方法 A 的语义地基）
- `mega-kernel-topology.md`：GenerateLoop/SinglePass 拓扑推导（诊断前需知模型走哪条拓扑）
- `kv-cache-dtype-dual-layer.md`：KV cache dtype 双地层陷阱（排除 KV cache 嫌疑时参考）
- `derive-compute-dtype-unconstitution.md`：dtype 硬编码违宪（易误判点 9 的详情）
- `gguf-format-spec.md`：GGUF 权重布局（路 C 字节验证时参考 offset 计算）
