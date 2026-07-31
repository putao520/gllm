# derive_compute_dtype 精度预设违宪（C-9 + 宪法 -1，堵 AI 幻觉）

> 来源：gllm-kernels dtype_chain.rs:195-210 + architect 裁决（sessionId 401396fe）+ 用户宪法 -1 指令
> 建库触发：运行时钉死 SmolLM2 compute_dtype=F32（config 是 BF16）→ 追出 derive_compute_dtype 硬编码 BF16→F32
> 重大修正：architect 裁决指出 "blob 保留 BF16" 但这本身预设 BF16 立场 → 用户宪法 -1：禁止任何精度预设
> 最后验证：2026-07-31
> **状态更正（2026-07-31）**：层2 描述的 `BF16|F16 => F32` 硬编码**已重构**。dtype_chain.rs:205-217 现为 `derive_compute_dtype(storage_dtype, device)` 的 `DotProductCap` 驱动组合判定（NativeBf16→保留 BF16；无原生累加→F32 兜底）。不再简单硬编码。本文件层2 段落保留作历史归因，但「当前仍违宪」表述已过时——见 BUG-KNOWLEDGE.md「DTYPE-HARDCODE-ASSUME-SINGLE」范式段。残留：单个算子 lowering 内仍有 ctx.dtype/F32 单一 dtype 假设位点。

## 两层事实（必须区分）

### 层1：当前 SmolLM2 路径数值上自洽（architect 裁决，源码证据）

architect 2 agent 交叉验证确认（对当前 BF16 safetensors 模型成立）：
- blob 走 raw_floats-first（pack_observe.inc.rs:206），BF16 权重拷原始 BF16 字节
- dequantize BF16→F32 循环是死代码（tensor_names ∩ quantized_tensor = ∅，BF16 在 raw_floats map）
- compute_dtype=F32 用作 KV cache 分配 + 累加器精度，不影响权重 blob
- KV cache F32(768/行) = MemCopy stride F32(768) = 自洽无越界
- **当前 SmolLM2 logits 发散根因不在 dtype 链**（候选根因 A 运行时证伪）

### 层2：derive_compute_dtype 仍违宪（宪法 -1，用户洞察）

但 architect 的 "blob 保留 BF16" 表述**本身预设了 BF16 立场**。用户宪法 -1 指出：

> 模型权重如果是 NVFP4 的呢？如果权重是混合精度的呢？我们自己代码不允许预设任何精度立场，必须严格按照权重文件和配置文件要求生成 JIT 代码。

`derive_compute_dtype` (dtype_chain.rs:198) `DType::BF16 | DType::F16 => DType::F32` **就是精度预设违宪**：
- 对 BF16 "看起来正确"（WidenCompute 累加），但这是**巧合**，不是设计正确
- 对 NVFP4 权重：`DType::F8E4M3 | ... => DType::F32`（line 200）同样硬编码 dequant F32 — 但 NVFP4 应该有原生计算路径，不该强制 dequant
- 对混合精度：整模型一个 compute_dtype 根本无法表达"部分张量 BF16 + 部分 NVFP4"
- 注释 "device parameter reserved for future" 是把违宪推迟到"未来"，非根治

**违宪本质**：代码假设"所有窄 dtype 都必须降到 F32 计算"，这是精度立场。宪法 -1 要求：JIT 看到什么 dtype 就生成什么 dtype 的代码，不预设升降级。

## 违宪源码（dtype_chain.rs:195-210）

```rust
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    match storage_dtype {
        // BF16/F16 always widen to F32 on current hardware
        DType::BF16 | DType::F16 => DType::F32,  // ← 精度预设违宪（宪法 -1）
        // Quantized types always dequant to F32
        DType::U8 | DType::F8E4M3 | DType::F8E5M2
        | DType::F6E3M2 | DType::F6E2M3 | DType::F4E2M1 => DType::F32,  // ← 同违宪
        // F32 stays F32
        DType::F32 => DType::F32,
    }
    // Note: device parameter is reserved for future hardware
}
```

## AI 易误判点（含我自己之前的错误）

| ❌ 误判 | ✅ 正解 |
|--------|---------|
| "blob 保留 BF16" 是正确表述 | 预设 BF16 立场。正确："blob 保留权重文件原始 dtype 字节"（可能是 BF16/NVFP4/混合） |
| derive_compute_dtype 对 BF16 自洽所以不违宪 | 对 BF16 巧合自洽，但对 NVFP4/混合精度必错。代码有精度预设就是违宪 |
| "BF16 always widen to F32" 是合理策略 | 精度预设。应"权重 BF16 → JIT 生成 BF16 计算代码（按配置/硬件决定累加精度）" |
| compute_dtype=F32 是累加器精度，无害 | 整模型单一 compute_dtype 无法表达混合精度，设计层面违宪 |
| 5 项联动改 BF16 能修 logits 发散 | 不能。当前发散根因不在 dtype 链（architect 裁决 + 运行时证伪） |
| derive_compute_dtype 改了会引入 768/384 越界 | 只有"只改 derive 不改 spec.dtype"才越界。宪法 -1 合规的修复须整体设计 |

## 根治方案（宪法 -1 合规，architect sessionId 401396fe 完整重构方案）

**核心裁决**：违宪的根不是 match arm 写错，是 `compute_dtype` 一个字段扛了三个正交语义（过载）。

| 语义 | 是什么 | 当前载体 | 宪法 -1 顺从数据源 | 现状 |
|------|--------|---------|------------------|------|
| A 累加器精度 | FMA acc 寄存器 dtype | compute_dtype | 硬件能力 + 配置，兜底 F32 | ❌ 硬编码 F32，device 忽略 |
| B KV cache 存储精度 | runtime 产生的 K/V | compute_dtype(buffer) + spec.dtype 硬编码 F32(stride/load) | K/V projection 输出张量 dtype | ❌ 主权分裂（split-brain 隐患）|
| C 权重存储精度 | 权重文件字节 | ~~compute_dtype~~ | weight_dtypes/tdt/b_dtype | ✅ 已 per-tensor 化，合规 |

**关键**：语义 C（权重布局）已从 compute_dtype 剥离（build_graph.inc.rs:86 tdt、lower_op.inc.rs:1357-1362 三路 b_dtype、build_graph.inc.rs:357 weight_physical_bytes）。NVFP4 原生寄存器解码已存在（emit_nvfp4_sub_block_dequant，无 F32 buffer 落地）。**本重构不碰权重侧，只拆 A+B**。

### 3 阶段重构（每阶段当前硬件行为不变，可独立回归）

**阶段 1：derive_compute_dtype → 累加器精度，顺从硬件+配置（行为不变，逻辑合规）**
- dtype_chain.rs:195-210：`match device.dot_product_cap() { NativeBf16 if BF16 => BF16, _ => F32(兜底) }`
- F32 是兜底分支（无原生累加支持时数值安全），非"BF16 always => F32"恒等预设
- 当前硬件（i9 无 AMX-BF16）→ 仍 F32，行为零变化
- 改动 1.2：统一双 compute_dtype 主权（ModelGeometry 用户 Option vs GraphDerived 硬编码）→ derive_compute_dtype 加 config_override 参数，P0 优先用户配置

**阶段 2：KV cache dtype 主权归位到 K/V projection 输出张量（消除 split-brain）**
- 改动 2.1：build_graph.inc.rs:693/1316/1523 AttentionSpec.dtype 不再硬编码 F32，从 k_out 张量 dtype 推导
- 改动 2.2：abi_types.inc.rs:469-489 KV cache 尺寸用新增 kv_dtype 字段（非全局 compute_dtype）
- 改动 2.3：types.inc.rs:211-217 kv_bytes_per_token 用 kv_dtype
- 当前全 F32 → 行为不变，但消除 split-brain，让"未来 K/V 输出 BF16 → KV cache 自动 BF16"成正确路径

**阶段 3（收尾）：解耦 TurboQuant + 清理死标签 + NVFP4 W512**
- 3.1（前置，须与阶段 1 同批）：executor_builder.rs:219 TurboQuant 开关从 `compute_dtype != F32` 改成 `storage 是否量化`
- 3.2：trace.rs:1025-1036 DequantMethod 死标签（低优先级）
- 3.3：lower_instr.rs NVFP4 W512(ZMM) 补齐（当前已是 Err，符合 NO-SILENT-FALLBACK）

### 3 个隐藏耦合点（重构必须同步处理，否则回归）

1. **TurboQuant 误触发**（executor_builder.rs:219 `!= F32` 开关）→ 累加器变 BF16 会误开，必须改看 storage 是否量化，且须与阶段 1 同批
2. **双 compute_dtype 主权分裂**（ModelGeometry 用户 Option vs GraphDerived 硬编码）→ 用户 `with_compute_dtype` 被忽略
3. **kv_bytes_per_token 用 storage dtype**（types.inc.rs:211）vs buffer 用 compute_dtype → 混合精度估算/实际不一致

### DAG 执行顺序

阶段 3.1（TurboQuant 解耦，前置）→ 阶段 1（累加器精度）→ 全量回归 → 阶段 2（KV cache 主权）→ 全量回归 → 阶段 3.2/3.3（收尾，可选）

### 与 logits 发散解耦（重要）

- **本重构对 SmolLM2 发散零帮助**（上轮已证 dtype 链自洽）
- 发散诊断必须并行另开线（Gather/decode M=1/RoPE partial）
- 重构与发散诊断分开 commit/分支（重构有回归风险，混在一起会污染发散定位）

### 宪法合规验证

| 宪法 | 重构后合规 | 证据 |
|------|-----------|------|
| -1 ARCH-NO-PRECISION-ASSUMPTION | ✅ | derive_compute_dtype 去恒等预设，device.dot_product_cap() 前置查询，F32 是兜底分支非映射 |
| 1 ARCH-BLOB-YIELDS-WEIGHT | ✅（已合规，不动）| 权重侧 per-tensor，blob 保原始 dtype |
| 2 ARCH-MEMORY-FIRST | ✅ | KV cache dtype 跟数据产生点（k_out 张量）|

完整方案详见 `docs/dtype-compute-refactor-plan.md`（architect 写入，已并入本资料库）。

## 与 logits 发散的关系（解耦）

- **derive_compute_dtype 违宪** ≠ logits 发散根因（architect 裁决：当前路径自洽）
- 发散真根因换方向：M=1 单 token prefill 逐算子 cosine 对齐 golden（架构师建议）
- **违宪仍须根治**（用户明确要求），但与发散诊断解耦，独立推进

## 历史错误记录（C-9 自我修正）

本资料库初版（commit 29f1d810）基于不完整源码阅读，错误声称：
- "executor_compile.rs:193 dequantize BF16→F32 进 blob 违宪" — 错，该路径对 BF16 死代码
- "blob 存 F32 字节" — 错，blob 走 raw_floats 保留 BF16

architect 裁决（sessionId 401396fe）+ 用户宪法 -1 双重纠正：
- architect：当前路径数值自洽（事实层）
- 用户：但 "blob 保留 BF16" 表述 + derive_compute_dtype 硬编码仍违宪（宪法 -1 层）

本资料库修正为两层事实：层1 数值自洽（非发散根因）+ 层2 精度预设违宪（须根治，独立于发散）。

## 关键代码位置

- `gllm-kernels/src/compiler/dtype_chain.rs:195-210` — derive_compute_dtype（精度预设违宪）
- `gllm-kernels/src/compiler/graph_geometry.rs:64,127,137` — from_graph 调 derive_compute_dtype
- `gllm/src/engine/mega_kernel/pack_observe.inc.rs:206-345` — raw_floats-first（BF16 原始字节，当前自洽）
- `gllm/src/engine/executor_compile.rs:185-206` — dequantize 循环（对 BF16 死代码，但路径仍违宪存在）
- `gllm-kernels/src/compiler/codegen/vm/plan_lower/context.inc.rs:167-180` — graph_dtype() F32（计算精度，正确）

## 与其他资料库关系

- `kv-cache-dtype-dual-layer.md`：候选根因 A 运行时证伪，但 KV cache dtype 跟随 compute_dtype 的设计仍需重构
- `dtype-propagation.md`：WidenCompute 是 JIT 层正确 widen，本库指出 derive_compute_dtype 是 loader 层精度预设违宪
- 本文件：derive_compute_dtype 精度预设违宪（宪法 -1）+ 两层事实修正


## 根治方案（用户要求：代码顺从数据/配置）

### 关键区分：graph_dtype() vs derive_compute_dtype（两个独立 dtype，必须解耦）

| 函数 | 位置 | 返回值 | 语义 | 是否违宪 |
|------|------|--------|------|---------|
| `graph_dtype()` | context.inc.rs:167-180 | 硬编码 F32（line 179） | **计算精度**（累加用 F32，WidenCompute 在 SIMD 层 widen BF16→F32） | ✅ 正确设计，不改 |
| `derive_compute_dtype` | dtype_chain.rs:195-210 | 硬编码 BF16→F32（line 198） | **storage/buffer 精度**（触发 loader dequant + KV cache 分配） | ❌ 违宪，须改 |

**context.inc.rs:167-180 graph_dtype() 源码**：
```rust
pub(super) fn graph_dtype(graph: &CompilerGraph) -> QuantPrecision {
    // 取第一个浮点 tensor 确认有浮点数据
    let has_float = graph.tensors.iter()...any(|qp| matches!(qp.kind, F32|BF16|F16|TF32));
    if !has_float { return QuantPrecision::F32; }
    // 计算精度统一 F32（激活累加）。存储 dtype(BF16/F16) 由 VecLoad WidenCompute
    // 在寄存器内 widen 到 F32，不在 graph_dtype 层混入存储精度。
    QuantPrecision::F32  // ← 硬编码 F32（计算精度，正确）
}
```

注释逻辑正确：计算精度 F32 + WidenCompute 在 SIMD 层 widen BF16 权重。**graph_dtype() 不用改**。

### 解耦关系（derive_compute_dtype 改了会重新引入 stride 不一致）

- `graph_dtype()` 保持 F32（计算精度）→ ctx.dtype=F32 → GEMM c_dtype=F32 → 激活 F32
- `derive_compute_dtype` 改成返回 storage_dtype(BF16) → KV cache 按 BF16(384/行)
- 但 attention `spec.dtype=F32`（build_graph.inc.rs:693 硬编码）→ MemCopy stride=768
- **768 vs 384 = stride 不一致（候选根因 A 重现）**

所以 `derive_compute_dtype` 单独改会重新引入候选根因 A。**必须连同 attention MemCopy/VecLoad 一起改**（5 项联动）。

### 方案（根治）：derive_compute_dtype 顺从 storage_dtype
```rust
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    // 代码顺从数据：compute_dtype = storage_dtype（BF16 权重就用 BF16 compute）
    // JIT 层 WidenCompute 在 SIMD 指令层 widen BF16→F32 累加（正确路径）
    // 不在 loader 层把权重字节转 F32（宪法1：blob 保留原始 dtype）
    match storage_dtype {
        DType::BF16 | DType::F16 => storage_dtype,  // 顺从, 不降级
        DType::F32 => DType::F32,
        // 量化类型仍 dequant（合法，量化本身需解码）
        DType::U8 | DType::F8E4M3 | ... => DType::F32,
    }
}
```

**配套**：
- `executor_compile.rs:185 needs_dtype_conversion` 变 false（compute_dtype==dtype==BF16）→ 不再 dequantize BF16→F32 → blob 保留 BF16 字节（宪法1恢复）
- KV cache 按 BF16(384/行) 分配 → MemCopy 需 narrow F32→BF16（k_out 是 F32，因 ctx.dtype=F32 不变）→ 触发方案 A 的 4 项联动（见 kv-cache-dtype-dual-layer.md）
- attention VecLoad 按 BF16 读 + widen
- GEMM c_dtype=BF16（因 spec.dtype 跟 compute_dtype？需确认 build_graph.inc.rs:693 AttentionSpec.dtype 来源）→ needs_narrow=true → 触发 VecNarrow → **必须先修 lane-loss bug**（emit_f32_to_bf16_ymm_to_xmm_avx2 vextracti128 取高半）

**影响面**：所有 BF16 模型（SmolLM2/Llama/Qwen 等）。需全量回归 + 5070Ti 验证。

### 待 architect 厘清

derive_compute_dtype 改 BF16 后，attention spec.dtype（build_graph.inc.rs:693 硬编码 F32）是否也要跟着改 BF16？还是 spec.dtype 跟 ctx.dtype(graph_dtype=F32)？这决定 MemCopy/VecLoad 的 stride 来源。需 architect 给 5 项精确改动清单 + 解耦关系。

## 关键代码位置

- `gllm-kernels/src/compiler/dtype_chain.rs:195-210` — derive_compute_dtype（违宪源头，BF16→F32 硬编码）
- `gllm-kernels/src/compiler/graph_geometry.rs:64,127,137` — from_graph 调 derive_compute_dtype
- `gllm/src/engine/mega_kernel/executor_core.inc.rs:371` — MegaKernelCompiled.compute_dtype 来自 GraphDerivedGeometry
- `gllm/src/engine/executor_compile.rs:185-206` — needs_dtype_conversion + dequantize_weight_to_dtype（宪法1违宪执行点）
- `gllm/src/model_config_fragments/types.inc.rs:167` — ModelGeometry.compute_dtype（纸面正确但被 GraphDerivedGeometry 覆盖）
- `gllm/src/loader/safetensors.rs:780` — cast_or_copy_f32（BF16→F32 转换实现）

## 与其他资料库关系

- `kv-cache-dtype-dual-layer.md`：候选根因 A 运行时证伪（KV cache 全 F32 自洽），但方案 A 4 项联动仍需做（derive_compute_dtype 修复后 KV cache 变 BF16）
- `dtype-propagation.md`：WidenCompute 是 JIT 层正确 widen，本库指出 loader 层 dequantize 是违宪（非 WidenCompute）
- `smollm2-135m-architecture.md`：SmolLM2 BF16 权重事实（本库是违宪检测的输入）
- 本文件：derive_compute_dtype 硬编码 BF16→F32 降级违宪铁证

## 阶段 3.1 方案修正（2026-07-06，C-9 自我修正）

**原方案问题**：KB §阶段 3.1 说"TurboQuant 开关从 `compute_dtype != F32` 改成 `storage 是否量化`"。但实测发现 `derive_storage_dtype`（graph_geometry.rs:192-215）只返回浮点 {F32, BF16, F16}，量化类型（U8/F8/F6/F4）被 `_ => {}` 忽略（测试 `storage_dtype_ignores_quantized_weight_dtypes` line 964-978 确认是设计行为）。所以"storage 是否量化"无法从 storage_dtype 判断——storage_dtype 值域不含 INT 量化。

**SPEC 契约确认**（00-PHILOSOPHY.html:157,181-188）：
- "原生混合精度路径处理模型原始 dtype，TurboQuant 处理额外量化"
- TurboQuant 量化 = "模型使用 TurboQuant 量化格式（INT4/FP4/FP6）"
- 即：BF16 权重（原生浮点）不触发 TurboQuant；INT4/FP4/FP6 权重才触发

**当前 `compute_dtype != F32` 触发逻辑错**：
- BF16 权重 + NativeBf16 硬件 → compute=BF16 → 误触发（BF16 是原生浮点不是 TurboQuant 量化）
- 当前 i9 SimdAssisted → compute=F32 → 巧合不触发（隐藏 bug）

**修正方案 B（architect infra 不可用，KB+SPEC 双源决策）**：
- `GraphDerivedGeometry`（graph_geometry.rs）新增 `is_weight_quantized: bool` 字段
- `from_graph` 扫权重 tensor，统计是否有量化 dtype（U8/F8E4M3/F8E5M2/F6E3M2/F6E2M3/F4E2M1）→ 填字段
- `default_for_simple` 填 `false`
- TurboQuant 触发（executor_builder.rs:219）改成 `g.is_weight_quantized`（非 `compute_dtype != F32`）
- **不碰 derive_storage_dtype**（隔离，下游 KV cache dtype 不受影响）

**为何选 B 不选 A/C**：
- 方案 A（修 derive_storage_dtype 识别量化）：影响下游 KV cache dtype，回归面大
- 方案 C（从 ctx/weight_dtypes 判断）：executor_builder 无法直接拿 weight_dtypes，需额外暴露，不隔离
- 方案 B：geometry 加字段隔离 + SPEC 合规 + graph tensors 可探测

**阶段 3.1 修正后 task**：跨仓（gllm-kernels geometry 加字段 + gllm executor_builder 改触发）。
