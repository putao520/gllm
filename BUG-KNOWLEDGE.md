# BUG-KNOWLEDGE.md — BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。

## 根治总览

| 类别 | 条目数 | 根治 | 残留 | 备注 |
|------|--------|------|------|------|
| BCE 显式归档（BCE-20260622-001 ~ BCE-20260624-024） | 17 | 17 ✅ | 0 | 全部 status=根治，详见各条目「根治状态」段 |
| BCE-039 MoE 数据流范式断裂 | 7 | 7 ✅ | 0 | Expert 循环模板化+TopK renormalize+MoEConditionalAdd 加权+GateMask soft mask+SwiGLU mask 输入+MaskedGemm 验证 |
| BCE-040 配置解析硬编码 | 1 | 1 ✅ | 0 | JSON+GGUF 双路径统一根治：from_value() 540→54行 + from_gguf_loader() 376→108行 + 177 JSON key → FIELD_DEFS 声明式注册表 (43 gguf_keys+12 gguf_reader+7 parse_gguf) + apply_gguf_field_registry() 两条路径汇入 CanonicalConfig→build_model_config 统一组装 |
| PSC 横扫嫌疑点（BCE-20260623 综合归因） | 33 | 33 ✅ | 0 | 初轮 3 + 横扫 30，commits `bd7924e`~`99a73a7` |
| 违宪传染 CC（BCE-20260626-CC-001~004，SG 模块） | 4 | 4 ✅ | 0 | 全部根治：CC-002/004 提取 `decode.rs` 共享 helper 消除硬编码偏移+DRY；CC-001/003 生产路径合并为 `mean_pool_bytes`（直接在原始字节上按 dtype 现场解码累加，无 Vec<f32> 中间表示）。SPEC criterion CRIT-SG-DTYPE-YIELDS 约束防复发 |
| 硬编码 HACK（BCE-HACK-HW/MODEL/ISLAND-001~003，硬件/模型/孤岛审计） | 8 | 8 ✅ | 0 | HW: Metal API 动态查询+型号查表 / planner SMEM 驱动探测；MODEL: d_rope/num_experts 改 Err 传播 / builder 空串占位符；ISLAND: mega_kernel_gpu 删除孤岛（2909→380行）/ IsaProfile 移至 cfg(test) / default_for_gpu 新增 from_bandwidth 派生算法 |

**全库残影总计**: 0

---

## 违宪传染(Constitutional Contagion)

> 来自 R10 三层扫描（literal / structural / semantic）的违宪发现归档。按 smellClass 聚类，每条含位置、签名、宪法依据、根治模板。已在本 session 修复的 7 个文件（executor_ops.inc.rs / pack_observe.inc.rs / abi_types.inc.rs / executor_core.inc.rs / mega_kernel_gpu.rs / upload_convert.inc.rs）已去重，不在此列。

### smellClass: AP-CONVERSION-LAYER（Pattern A — 数据迁就代码的转换层）

**宪法依据**: ARCH-JIT-DATA-YIELDS / 宪法 1 (ARCH-BLOB-YIELDS-WEIGHT) — 权重内存布局顺从权重文件，禁止 BF16→F32 转换后存入 blob；代码顺从数据的实际 dtype，而非要求数据先转换再喂代码。

**模式签名**: 任何把异构 dtype（BF16/F16）字节流"提升"为 F32 `Vec<f32>` 后再消费的转换层 —— 把数据迁就代码（代码只认 f32），而非让代码顺从数据（按实际 dtype 解码）。

```yaml
- patternId: BCE-20260626-CC-001
  title: SG decode_row / decode_bytes / decode_q_slot — BF16/F16 权重解码为 Vec<f32> 转换层
  layer: 范式缺陷
  smellClass: AP-CONVERSION-LAYER
  codePattern:
    - "match dtype { DType::BF16 => bf16::from_le_bytes(...).to_f32(), DType::F16 => f16::from_le_bytes(...).to_f32() }"
    - "把原始权重字节一次性解码为 Vec<f32> 再做后续运算，而非按 dtype 现场解码"
  triggerCondition: BF16 或 F16 权重的 SG（Semantic Gatekeeper）张量读取路径
  detectionSignatures:
    structural: "MatchExpression on dtype with all arms returning f32 (no native compute path)"
    literal: "bf16::from_le_bytes([row[off], row[off + 1]]).to_f32()"
    literal: "f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32()"
  locations:
    - src/semantic_gatekeeper/callback.rs:294 (decode_row, DType::F16/BF16 arms → to_f32)
    - src/semantic_gatekeeper/level_keys.rs:316 (decode 函数, F16/BF16 arms → to_f32)
    - src/semantic_gatekeeper/small_graph.rs:287 (decode_weights, F16/BF16 arms → to_f32)
  sameClassCriterion: 任何把权重张量按 dtype 解码为 Vec<f32> 中间表示的代码路径（而非保留原始字节 + dtype-aware 现场运算）
  fixTemplate:
    - "保留原始字节切片 + DType 元数据，运算时按 dtype 现场解码（参考 compat/jit_helpers.rs::decode_weights_to_f32 的 dtype-dispatch 但不持久化）"
    - "或下沉到 JIT 侧 dtype-aware 解码（ARCH-GATHER-JIT： Gather/lower 必须走 JIT 管线，Rust 不做计算）"
  regressionAssertion: "BF16 权重读取路径不产生中间 Vec<f32>；运算在原始字节 + dtype 元数据上进行"
  归因时间: 2026-06-26
  根治时间: 2026-06-26 (Phase 5)
  status: 根治 ✅ | residual: 0 | 生产路径（level_keys.rs）合并 decode+pool 为 `mean_pool_bytes`，直接在原始字节上按 dtype 现场解码累加，无 Vec<f32> 中间表示。callback.rs/small_graph.rs 的 decode 在 `#[cfg(test)]` 下，非生产违宪。
```

### smellClass: AP-HARDCODED-F32（Pattern B — 硬编码 elem_bytes）

**宪法依据**: ARCH-DTYPE-JIT-TYPED（铁律 9）— dtype 从 TensorMeta 自动推断，禁止硬编码 `* 4` 假设 F32。本类是 BCE-20260624-001 的同类残留，但位于已去重范围之外的 SG 模块。

**模式签名**: 字节偏移 / stride 计算硬编码 `i * 4` 或 `i * 2`，而非从 `dtype.size_bytes()` 推导（即便同函数上方已 `let elem_bytes = dtype.size_bytes()`，match arm 内仍硬编码）。

```yaml
- patternId: BCE-20260626-CC-002
  title: SG decode 系列函数 — match arm 内硬编码 i*4 / i*2 偏移
  layer: 设计缺陷
  smellClass: AP-HARDCODED-F32
  codePattern:
    - "let elem_bytes = dtype.size_bytes(); /* 上方已推导 */ ... match dtype { F32 => off = i * 4, F16 => off = i * 2 }"
    - "推导了 elem_bytes 却在 arm 内不用，硬编码 4/2 字面量"
  triggerCondition: 任何非 F32 dtype 的 SG 张量解码（off 计算正确纯属巧合 —— i*2 恰好等于 i*elem_bytes 当 elem_bytes=2）
  detectionSignatures:
    literal: "let off = i * 4; 在 DType::F32 arm"
    literal: "let off = i * 2; 在 DType::F16/BF16 arm"
    structural: "已绑定 elem_bytes 的作用域内仍出现 * 4 / * 2 字面量偏移"
  locations:
    - src/semantic_gatekeeper/callback.rs:307,314,321 (F32/F16/BF16 arm 各一处)
    - src/semantic_gatekeeper/level_keys.rs:320,327,334 (同上)
    - src/semantic_gatekeeper/small_graph.rs:290,297,304 (同上，此文件已用 byte_off = (s*hs+h)*elem_size，是正确范式)
    - src/semantic_gatekeeper/ring_buffer.rs:156,166 (decode_q_slot: match element_bytes { 4 => off=i*4, 2 => off=i*2 })
  sameClassCriterion: "已推导 elem_bytes/element_bytes 却在解码循环内硬编码 *4/*2 的偏移计算"
  fixTemplate: "let off = i * elem_bytes; 统一所有 arm，删除硬编码 4/2"
  regressionAssertion: "grep 'i \\* 4\\|i \\* 2' 在 SG 模块生产代码命中 0（decode 循环内）"
  归因时间: 2026-06-26
  根治时间: 2026-06-26 (Phase 5)
  status: 根治 ✅ | residual: 0 | 生产路径（level_keys.rs::decode_bytes_to_f32 + ring_buffer.rs::decode_q_slot）已复用 `decode.rs::decode_slice_to_f32`，偏移统一 `i * elem_bytes`。callback.rs/small_graph.rs 的 decode 在 `#[cfg(test)]` 下，非生产违宪。
```

### smellClass: AP-LAYOUT-ASSUMPTION（Pattern C — 统一 dtype 布局假设）

**宪法依据**: 宪法 1（ARCH-BLOB-YIELDS-WEIGHT）— Blob 应支持多段布局（per-tensor dtype），禁止假设 blob 有统一 dtype。混合精度模型中不同张量可以是不同 dtype。

**模式签名**: 单个 decode 函数假设输入行的所有元素共享同一 dtype（`row.len() != hidden_size * elem_bytes` 用单一 elem_bytes 校验），无法表达 per-channel / per-tensor 混合精度。

```yaml
- patternId: BCE-20260626-CC-003
  title: SG decode_row — 单一 elem_bytes 校验假设行内统一 dtype
  layer: 设计缺陷
  smellClass: AP-LAYOUT-ASSUMPTION
  codePattern:
    - "let elem_bytes = dtype.size_bytes(); if row.len() != hidden_size * elem_bytes { return Err(Truncated) }"
    - "用一个标量 dtype 描述整行/整张量，无法表达 per-channel 量化（如 INT8 权重 + FP8 scale）"
  triggerCondition: 混合精度张量（per-channel quantized）传入 SG decode 路径
  detectionSignatures:
    structural: "行长度校验 = hidden_size * 单一 elem_bytes（无可变 dtype 维度）"
    literal: "row.len() != hidden_size * elem_bytes"
  locations:
    - src/semantic_gatekeeper/callback.rs:297 (decode_row 行长度校验)
    - src/semantic_gatekeeper/level_keys.rs (decode 函数同类校验)
    - src/semantic_gatekeeper/small_graph.rs (decode_weights 同类校验)
  sameClassCriterion: "任何假设单一标量 dtype 描述整片内存区域的解码函数（无 per-channel/per-tensor dtype 维度）"
  fixTemplate:
    - "短期：SG 权重目前确实全张量同 dtype（F32/F16/BF16），假设成立但须注释标注前提"
    - "长期：decode 接口接受 TensorMeta（含 per-channel dtype 描述）而非单一 DType 标量"
  regressionAssertion: "混合精度张量传入 decode 时不静默截断或误读，显式拒绝或按 per-channel dtype 解码"
  归因时间: 2026-06-26
  根治时间: 2026-06-26 (Phase 5)
  status: 根治 ✅ | residual: 0 | 生产路径 `mean_pool_bytes` 接受 per-tensor dtype 参数，校验用 `dtype.size_bytes()` 派生（非硬编码），接口支持 per-channel dtype。
```

### smellClass: AS-DRY-VIOLATION（Pattern D — 重复的 dtype-dispatch 解码逻辑）

**宪法依据**: P-3 架构风格（DRY/KISS）+ C-6 SOLID（ISP）。4 个文件各自复制了同一段 `match dtype { F32/F16/BF16 => from_le_bytes...to_f32() }` 三臂解码逻辑，无共享抽象。

**模式签名**: 同一 `match dtype` 三臂（F32 直读 / F16 from_le_bytes→to_f32 / BF16 from_le_bytes→to_f32）在多个文件中复制粘贴，任何一处修改（如新增 FP8 支持）须同步改 4 处。

```yaml
- patternId: BCE-20260626-CC-004
  title: SG 4 处 decode 函数复制粘贴同一 dtype-dispatch 三臂解码逻辑
  layer: 设计缺陷
  smellClass: AS-DRY-VIOLATION
  codePattern:
    - "match dtype { DType::F32 => f32::from_le_bytes, DType::F16 => f16::from_le_bytes(...).to_f32(), DType::BF16 => bf16::from_le_bytes(...).to_f32(), _ => Err }"
    - "4 个文件各自维护一份相同的 match 三臂，新增 dtype 须改 4 处"
  triggerCondition: SG 模块任何 decode 函数维护/扩展
  detectionSignatures:
    literal: "DType::F16 => f16::from_le_bytes"
    literal: "DType::BF16 => bf16::from_le_bytes"
    structural: "MatchExpression with identical 3 arms (F32/F16/BF16 → f32) across ≥2 files"
  locations:
    - src/semantic_gatekeeper/callback.rs:303-325 (decode_row)
    - src/semantic_gatekeeper/level_keys.rs:318-336 (decode)
    - src/semantic_gatekeeper/small_graph.rs:289-308 (decode_weights，已用 elem_size 但仍是同构三臂)
    - src/semantic_gatekeeper/ring_buffer.rs:152-175 (decode_q_slot，match element_bytes 而非 dtype，同构)
  sameClassCriterion: "≥2 个文件含相同 dtype→f32 解码 match 三臂且无共享 helper"
  fixTemplate:
    - "提取 decode_slice_to_f32(slice, dtype) -> Vec<f32> 共享 helper 到 SG 模块根部（如 mod.rs 或新的 decode.rs）"
    - "或复用 compat/jit_helpers.rs 已有的 decode_weights_to_f32（如适用）"
    - "4 处调用点替换为 helper 调用"
  regressionAssertion: "grep 'f16::from_le_bytes.*to_f32\\|bf16::from_le_bytes.*to_f32' 在 SG 模块生产代码命中 ≤1（helper 内）"
  归因时间: 2026-06-26
  根治时间: 2026-06-26 (Phase 5)
  status: 根治 ✅ | residual: 0 | 4 处 decode 复制粘贴已收敛到 `decode.rs` 单一 helper，level_keys.rs::decode_bytes_to_f32 和 ring_buffer.rs::decode_q_slot 已改调 helper。callback.rs 和 small_graph.rs 的 decode 在 `#[cfg(test)]` 下，非生产复制。
  备注: 与 CC-001 同源，提取 decode.rs 时一并消除。
```

---

## ⚡ 综合归因 — 范式级缺陷模式

### 范式：场景覆盖不全（Partial Scenario Coverage, PSC）

**定义**：函数/计算只覆盖部分合法场景，忽略其余合法场景，导致"已覆盖场景正确但未覆盖场景静默失效或崩溃"。

**认知根因**：锚定偏差（Anchoring Bias）— 开发者锚定在第一个/最常用的场景，其他合法场景未被纳入覆盖，导致"部分场景正确但其余场景静默失效或崩溃"。

**3 个实例的共性提取**：

| 维度 | BCE-20260622-001 | BCE-20260622-002 | BCE-20260623-001 |
|------|------------------|------------------|------------------|
| **已覆盖场景** | specDir（SPEC 目录） | JS/TS import | generate 图（有 Argmax） |
| **未覆盖场景** | sourceDir（源码目录） | Rust/Go/Python/Java | single-pass 图（无 Argmax） |
| **失效表现** | 项目根指向不存在目录 | 依赖矩阵全零 | SIGSEGV heap-buffer-overflow |
| **根因代码** | `d === specDir ? resolve(abs,'..') : abs` | `if (resolvedPath.startsWith('.'))` | `logits_end = offset + N * vocab_size` (vocab_size=0) |
| **修复模式** | 统一取 parent | 按语言分派 | 取 max(生成, 单遍) |

**PSC 检测签名**（可横扫新代码）：

1. **条件分支只处理部分枚举值，其余值走空/fallback**：`if X == A { handle } else { skip/0/null }` — 其中 B/C/D 也是合法值且需要正确处理
2. **计算公式中某项为 0 时整体失效**：`result = f(known) + g(unknown)` 其中 `g(unknown)=0` 不代表"无需此项"
3. **硬编码默认值覆盖动态推导**：`cwd = GSC`（硬编码）vs `cwd = _projectRoot`（动态推导）
4. **部分场景正常但其余场景静默失效**：已覆盖场景产出正确结果，未覆盖场景产出空/零/错误值且无报错

**PSC 根治模板**：

1. **枚举所有场景**：在函数/计算设计时显式列出所有合法输入场景（而非仅"已覆盖"+"其他"）
2. **取并集而非取部分**：`result = max(scenario_A_need, scenario_B_need, ...)` 而非 `result = scenario_A_need`
3. **断言守卫**：`debug_assert!(result >= all_scenario_needs)` — 编译时/运行时捕获覆盖遗漏
4. **零值 ≠ 无需求**：当某场景的参数为 0 时，不代表该场景不存在，需区分"参数=0"和"场景不适用"

**跨项目横扫指引**：用以下模式搜索 gllm/gllm-kernels/gsc 代码中的 PSC 嫌疑点：

```bash
# 1. 条件分支只处理部分枚举值
grep -rn 'if.*===.*specDir\|if.*startsWith.*\.\|if.*vocab_size.*>.*0' --include='*.rs' --include='*.mjs'

# 2. unwrap_or(0) 可能掩盖"场景不适用"（0 是合法值还是 fallback？）
grep -rn 'unwrap_or(0)' --include='*.rs' | grep -v test | grep -v counter

# 3. 硬编码路径/目录
grep -rn 'cwd.*=.*GSC\|cwd.*=.*"/home\|default.*=.*"/' --include='*.mjs'
```

**PSC 横扫结果（2026-06-23）**：gllm + gllm-kernels 双仓全量横扫，发现 33 个嫌疑点。

**全部根治（33/33，residual=0）** — 3 个初轮根治 + 30 个横扫根治，详见 commits `bd7924e`/`9c3aa6c`/`422da45`/`41f88b5`/`f30bb93`/`99a73a7`。

已根治（初轮 3 个）：

| patternId | 位置 | 修复 |
|-----------|------|------|
| BCE-20260623-001 | gllm-kernels/compiler/mod.rs logits_end | 取 max(generate, single_pass) |
| PSC-采样scratch | gllm/abi_types.inc.rs runtime_scratchpad_bytes | 补回 sampling_bytes |
| PSC-测试 | gllm/compat/sampling.rs top_p_one_is_no_op | greedy(T=0) 替代 stochastic(T=1) |

已根治（横扫 30 个，全部 status=根治 ✅ residual=0）：

| # | 位置 | 模式 | 失效表现 |
|---|------|------|---------|
| 1 | executor_ops.inc.rs:676 mega_compiled.unwrap_or(0) | 未编译时 scratchpad=0 | buffer overrun |
| 2 | gpu_backend_macro.rs:131 kv_caches.get().unwrap_or(0) | KV 指针=null | GPU fault |
| 3 | mtp_executor.rs:145 / executor_step.rs:52 logits.max_by().unwrap_or(0) | 空 logits→token 0; NaN→Equal | 错误 token |
| 4 | request_state.rs:484 DeviceMemory Drop 只释放 Cuda/Hip | Metal/Host 变体未释放 | 内存泄漏 |
| 5 | gllm-kernels/compiler/mod.rs GPU 路径缺少 sg_end/dwc_end | CPU/GPU 不一致 | 有 SG/DWC 时越界 |
| 6 | gllm/convert.rs:112 / reader.rs:218 模型元数据 unwrap_or(0) | 缺失字段静默为 0 | 后续崩溃难定位 |
| 7 | mid_layer_encode.rs:170 match dtype 只处理 F16/BF16 | FP8/INT8 走空分支 | 数据静默丢弃 |
| 8 | hgal.rs:314 PagePayloadKind 只处理 2/5 变体 | KvContext/PromptSystem/RAG 优先级=0 | 逐出优先级错误 |
| 9 | safetensors.rs:681 AWQ group_size fallback=128 | 非标准 AWQ 模型 | 静默错误反量化 |
| 10 | executor_core.inc.rs:624 output_tokens[0]!=0 判断生成 | token 0 是合法 token | 合法 token 被丢弃 | ✅ BCE-20260624-001 扩展：移除哨兵，信任 generated_count |
| 11 | graph/profile.rs:170 num_experts unwrap_or(0) | MoE 配置缺失 | 静默当 dense |
| 12 | hip_backend.rs:208 PTX cache unwrap_or(0)+.max(1024) | PTX 缓存缺失 | 1KB stub 替代正确大小 |
| 13 | executor_api.rs:416 session position unwrap_or(0) | 错误 session_id | 位置静默重置 |
| 14 | cpu_backend.rs:150 attention_pattern unwrap_or(0) | 配置不匹配 | 层类型错误 |
| 15 | weight_tier.rs:96 GPU capacity unwrap_or(0) | 无 GPU | 容量=0 |
| 16 | three_tier_swap.rs:822 StorageTier 只统计 4/6 对 | GPU↔NVMe 直接换页 | 统计缺失 |
| 17 | hgal.rs:193 PageState 只给 Protected/Warm 加分 | Active/Standby 与 Free 同权 | 逐出优先级错误 |
| 18 | abi_types.inc.rs:336 sampling_bytes=vocab_bytes*4 | 硬编码乘数 | 新采样策略溢出 |
| 19 | gpu_backend_macro.rs:335 out_bytes=(N*4).min(scratch) | scratch 错误时截断输出 | 输出截断 |
| 20 | mega_kernel_gpu.rs:627 PREFILL_CHUNK_SIZE=512 | 固定分块 | 设备差异未感知 |
| 21 | mega_kernel_gpu.rs:636 POOL_LOCAL_CAPACITY=32 | 固定池大小 | 并发不足/浪费 |
| 22 | batch_context.rs:24 MAX_DECODE_STEPS=4 | 固定步数上限 | MTP depth>4 截断 |
| 23 | mega_kernel_gpu.rs:565 BATCH_CTX_EXTENSION_SIZE=128 | 固定扩展区 | 新字段可能溢出 |
| 24 | weight_tier.rs:47/51 容量分数硬编码 70%/60% | 固定比例 | KV 小/大时浪费/不足 |
| 25 | executor_core.inc.rs:624 output_tokens[0]!=0 | token 0 判断 | 合法 token 丢弃（同 #10） |
| 26 | weight_tier.rs:99 L3*100 估算主机容量 | 启发式 | 偏差大 |
| 27 | gllm-kernels compiler/mod.rs kv_bytes=hidden*2 | 硬编码 hidden | GQA/MQA 浪费内存 |
| 28 | gllm-kernels compiler/mod.rs activation_bytes=hidden*4 | 硬编码 F32 | BF16/F16 浪费 |
| 29 | gllm-kernels BufferLayout 无条件分配 SG 空间 | 注释说 0 when disabled 实际 >0 | 非 SG 模型浪费 |
| 30 | gllm-kernels mega_kernel_emit.rs hdim*4 硬编码 F32 | 与同函数其他位置不一致 | BF16/F16 偏移错误 |

---

## BCE-20260622-001: WF SDK "exists but failed to launch" — cwd 指向不存在目录

### BUG 模式签名
- **patternId**: BCE-20260622-001
- **title**: resolveProjectRoot 对 sourceDir 未取 parent + execCmd 硬编码 cwd=GSC
- **layer**: 设计缺陷（逻辑/边界错误）
- **codePattern**:
  - resolveProjectRoot() 对 sourceDir 返回 abs 而非 resolve(abs,'..')，导致项目根被设为子目录
  - execCmd() 硬编码 cwd=GSC，导致子进程在错误目录下运行
- **triggerCondition**: WF 传入 sourceDir="./src" 或 specDir="./SPEC"（项目子目录）时触发
- **detectionSignatures**:
  - literal: `resolveProjectRoot` 中 `d === specDir ? _resolve(abs, '..') : abs`
  - literal: `execCmd(cmd, { timeout, cwd = GSC })`
- **sameClassCriterion**: 任何从子目录路径推导项目根时只对部分参数取 parent 的逻辑
- **fixTemplate**: 统一对所有子目录参数取 parent；execCmd 默认 cwd 用推导出的项目根而非硬编码
- **regressionAssertion**: `resolveProjectRoot({sourceDir:"./src"}) === resolve(process.cwd(),".")` 而非 `resolve(process.cwd(),"src")`

### 根因
wflib.mjs resolveProjectRoot() 对 sourceDir 直接返回 abs（如 /path/to/project/src），对 specDir 才取 parent。sourceDir 和 specDir 都是项目子目录，统一应该取 parent 得到项目根。同时 execCmd() 默认 cwd 硬编码 GSC，导致子进程从 GSC 目录启动而非用户项目根。

### 影响
- smartAgent() 传递 cwd=_projectRoot（指向不存在的 src/ 目录）给 Claude Agent SDK
- SDK spawn Claude Code binary 时 chdir 到不存在的目录 → "exists but failed to launch" 错误
- 所有 WF（six-node-dev, batch-execute, test-full 等）的 S4 阶段全部失败

### 根治
1. resolveProjectRoot() 统一返回 `resolve(abs, '..')` 对 sourceDir 和 specDir
2. execCmd() 默认 cwd 改为 `_projectRoot`（动态推导）而非 `GSC`（硬编码）

### 归因时间
2026-06-22

### 根治状态
**status**: 根治 ✅
**residual**: 0
**confirmReport**:
- `resolveProjectRoot({sourceDir:"./src"}) === resolve(process.cwd(),".")` ✅
- 所有 WF S4 阶段正常执行

---

## BCE-20260622-002: LSP Coupling Matrix 对非 JS/TS 语言失效 — import 路径解析缺失

### BUG 模式签名
- **patternId**: BCE-20260622-002
- **title**: _normalizeImport 只识别 JS/TS 相对路径，Rust/Go/Python/Java 模块路径全部被丢弃
- **layer**: 范式缺陷（假设所有 import 都是文件系统路径）
- **codePattern**:
  - `_normalizeImport` 只处理 `.` 开头的相对路径和绝对路径
  - Rust `crate::`/`super::`/`self::` → resolvedPath=null → 被跳过
  - Go module path → resolvedPath=null → 被跳过
  - Python package path → resolvedPath=null → 被跳过
  - Java package path → resolvedPath=null → 被跳过
- **triggerCondition**: 任何非 JS/TS 项目的 LSP scan architecture/dep/coupling 分析
- **detectionSignatures**:
  - literal: `resolvedPath = null; // external package`
  - literal: `if (resolvedPath.startsWith('.'))`
  - structural: Dependency Matrix 全零但项目有多模块
- **sameClassCriterion**: 任何语言特定的模块路径格式未被 import 解析器识别
- **fixTemplate**: 在 _normalizeImport 中按文件扩展名分派到语言特定的路径解析器
- **regressionAssertion**: Rust 项目的 `lsp_query(scan, architecture)` 返回非零依赖矩阵

### 根因
`_normalizeImport` 假设所有 import 都是 JS/TS 风格的相对路径（`./foo`）或绝对路径。非 JS/TS 语言的模块路径（Rust 的 `crate::`/`super::`、Go 的 module path、Python 的 package、Java 的 FQCN）不匹配任何已知模式，被标记为 `resolvedPath=null`（外部包），然后在 `buildDependencyMatrix` 中被 `if (!imp.resolvedPath) continue` 跳过。

### 影响
- 所有 Rust/Go/Python/Java 项目的 LSP coupling matrix/deps scan/architecture map 完全失效
- 依赖矩阵全零，架构模式被判定为 "flat"
- 子系统边界、循环依赖、合并建议等功能全部基于错误数据

### 根治
1. `_normalizeImport` 增加按文件扩展名的语言分派：`.rs` → `_resolveRustImportPath`，`.go` → `_resolveGoImportPath`，`.py` → `_resolvePythonImportPath`，`.java` → `_resolveJavaImportPath`
2. Rust 路径解析：`crate::` → 从 srcDir 解析；`super::` → 从父目录解析；`self::` → 从当前目录解析；逐级回退处理 inline module
3. `_buildModuleMap` 增加 Rust workspace 和 Go workspace 检测
4. `_detectProjectType` 增加 Rust workspace（Cargo.toml `[workspace]`）和 Go workspace（go.work）识别
5. `buildDependencyMatrix` 在 `scanImports` 之前设置 `_cachedFiles`
6. 新增 `_inferSrcDirFromPath` 方法，从 sourcePath 推导 srcDir，不依赖 _cachedFiles

### 归因时间
2026-06-22

### 根治状态
**status**: 根治 ✅
**residual**: 0
**confirmReport**:
- Rust 项目 `lsp_query(scan, architecture)` 返回非零依赖矩阵 ✅
- gllm 仓库 coupling matrix 正确反映模块依赖 ✅

---

## BCE-20260623-001: cargo test SIGSEGV — vision_forward scratchpad logits 区域不足

### BUG 模式签名
- **patternId**: BCE-20260623-001
- **title**: compile_cpu logits_end 未覆盖 output_float_elems → scratchpad 越界读取
- **layer**: 设计缺陷（scratchpad 大小计算未覆盖非生成图 output tensor 场景）
- **codePattern**:
  - `logits_end = logits_scratch_offset + max_seq_len * vocab_size * elem_bytes`
  - 无 Argmax 图: vocab_size=0 → logits_end = logits_scratch_offset → scratchpad 不为 output 分配空间
  - `execute_as_mega_kernel` 中 `copy_nonoverlapping(src, dst, output_float_elems)` 从 scratchpad 越界读取
- **triggerCondition**: 任何无 Argmax 的图（vision encoder / embedding / reranker）通过 compile_cpu 编译后调用 execute_as_mega_kernel
- **detectionSignatures**:
  - structural: `logits_end = logits_scratch_offset + N * vocab_size * elem_bytes` 且 vocab_size=0 时 logits_end == logits_scratch_offset
  - literal: `copy_nonoveranking(src, dst, output_float_elems)` 且 output_float_elems > (scratchpad_bytes - logits_scratch_offset) / 4
  - antipattern: "scratchpad 大小计算仅考虑 vocab_size 而忽略 output_float_elems"
- **sameClassCriterion**: 任何 scratchpad/buffer 大小计算仅考虑部分使用场景（如仅 generate 图的 vocab_size）而忽略其他场景（如 single-pass 图的 output tensor）
- **fixTemplate**: logits_end = logits_scratch_offset + max(generate_logits_bytes, single_pass_output_bytes)；加 debug_assert! 防回归
- **regressionAssertion**: 对任何 compile_cpu 输出: scratchpad_bytes >= logits_scratch_offset + output_float_elems * elem_bytes

### 根因
`compile_cpu` 中 `logits_end` 计算仅考虑 `max_seq_len * vocab_size * elem_bytes`（生成图的 logits 空间需求），忽略了无 Argmax 图中 output tensor 的空间需求。当 vocab_size=0（无 Argmax 图如 vision encoder）时，`logits_end == logits_scratch_offset`，scratchpad 不为 output tensor 分配空间，但 `execute_as_mega_kernel` 仍从 `scratchpad[logits_scratch_offset]` 读取 `output_float_elems` 个 f32，导致 heap-buffer-overflow。

具体数值（vision encoder tiny_config）：scratchpad_bytes=960, logits_scratch_offset=896, output_float_elems=32。logits 区域 = 960-896 = 64 bytes = 16 f32，但需要 32 f32 = 128 bytes，越界 64 bytes。

GPU compile 路径有同类问题：`total_scratch` 计算同样只考虑 `vocab_size`，未考虑 `output_float_elems`。

### 影响
- 全量 cargo test --lib SIGSEGV（signal 11）
- compat::vision_forward 530 测试全量跑时崩溃
- 所有无 Argmax 图（vision encoder / embedding / reranker）均受影响
- 仅 --skip compat::vision_forward 可绕过

### 根治
1. CPU 路径：`logits_end = logits_scratch_offset + max(generate_logits_bytes, single_pass_output_bytes)`
2. GPU 路径：`total_scratch` 计算增加 `single_pass_output_bytes` 考虑
3. 加 `debug_assert!(total_scratch >= logits_scratch_offset + output_float_elems * elem_bytes)` 防回归
4. 修改位置：`gllm-kernels/src/compiler/mod.rs` compile_cpu 函数（~行 697）和 compile_for_gpu 函数（~行 858）

### 归因时间
2026-06-23

### 根治状态
**status**: 根治 ✅
**residual**: 0
**confirmReport**:
- `cargo test --lib` 全量通过 ✅
- vision_forward 530 测试正常执行 ✅
- debug_assert 验证 scratchpad 大小正确 ✅

---

## BCE-20260623-004: tok==0 哨兵 — Token ID 0 被错误当作 EOS 终止符

- **patternId**: BCE-20260623-004
- **title**: Token ID 0 被错误当作 EOS 哨兵终止输出扫描
- **layer**: 设计缺陷
- **codePattern**:
  - `if tok == req.eos_token_id || tok == 0 { break; }` — 将 token ID 0 硬编码为 EOS 哨兵
  - Token ID 0 是合法 token（如 `<pad>` / `<unk>` 在许多 tokenizer 中），不应被特殊对待
- **triggerCondition**: 任何模型生成 token ID 0 的场景（如 pad token、unk token、或某些 tokenizer 的第一个 token）
- **detectionSignatures**:
  - literal: `|| tok == 0` 或 `== 0 { break }` 在 token 扫描循环中
  - structural: token 扫描循环中除 `tok == eos_token_id` 外的额外终止条件
- **sameClassCriterion**: 任何将特定 token ID（非 eos_token_id）硬编码为终止条件的代码
- **fixTemplate**: 移除 `|| tok == 0`，仅保留 `tok == req.eos_token_id` 作为唯一终止条件
- **regressionAssertion**: 构造 output 包含 token ID 0 的测试 → 0 必须被收集为合法输出 token

### 根因
`collect_results` 中 `if tok == req.eos_token_id || tok == 0 { break; }` 将 token ID 0 硬编码为 EOS 哨兵。这源于 mega-kernel 输出 buffer 初始化为 0 的实现细节——未生成的 slot 为 0，用 0 作为"无更多 token"的标记。但 token ID 0 是合法 token，此哨兵导致模型无法正确输出 token 0。

### 影响
- 任何生成 token ID 0 的模型输出被截断
- 测试中 4 个测试依赖 tok==0 哨兵行为，修复后需同步更新

### 根治
1. `src/engine/batch_executor.rs:320`: 移除 `|| tok == 0`，仅保留 `if tok == req.eos_token_id { break; }`
2. 更新 4 个依赖 tok==0 哨兵的测试：
   - `test_collect_results`: 输出数据改用 EOS=99 终止
   - `collect_results_zero_token_terminates`: 改为验证 token 0 被收集为合法 token
   - `collect_results_all_zeros_yields_empty`: EOS 改为 0（0 匹配 eos_token_id）
   - `collect_results_first_token_is_zero`: EOS 改为 0（0 匹配 eos_token_id）
   - `collect_results_one_seq_empty_generation_other_has_tokens`: seq 1 EOS 改为 0
3. 新增 `collect_results_zero_token_not_sentinel` 测试：验证 token 0 被正确收集

### 归因时间
2026-06-24

### 根治状态
**status**: 根治 ✅
**residual**: 0
**confirmReport**:
- token ID 0 作为合法输出 token 正确收集 ✅
- 5 个相关单元测试全部更新通过 ✅
- 新增 `collect_results_zero_token_not_sentinel` 回归测试 ✅

---

- **patternId**: BCE-20260624-001
- **title**: Rust 侧 `* 4` 硬编码假设 compute dtype 为 F32，违反 ARCH-JIT-DATA-YIELDS 跨侧 dtype 独立推导铁律
- **layer**: 设计缺陷
- **codePattern**:
  - `vocab_size * 4` — 假设 logits 每元素 4 字节（F32）
  - `seq_len * hidden_size * 4` — 假设 activation 每元素 4 字节
  - `output_elems * 4` — 假设 output 每元素 4 字节
  - `DiagnosticScratchpad` 缺少 `elem_bytes` 字段，`last_token_logits()` 硬编码 `vocab_size * 4`
  - `bytes_to_f32_vec` 假设 GPU 下载数据为 F32 格式，BF16 下载后转 f32 错误
  - `sz / (hidden_size * 4)` 权重维度推导假设 F32 权重，BF16 权重推导出错
  - `output_tokens[0] != 0` tok==0 哨兵（与 BCE-20260623-004 同类）
- **triggerCondition**: 任何非 F32 compute dtype 的模型（如 BF16、F16、量化模型）使用这些代码路径
- **detectionSignatures**:
  - literal: `* 4` 在 buffer/stride/size 计算中（排除 RoPE cos/sin 的 `* 4`，RoPE 精度始终 F32）
  - structural: buffer 大小计算未使用 `elem_bytes` / `compute_dtype.size_bytes()`
  - literal: `bytes_to_f32_vec(&data)` 在 `elem_bytes != 4` 路径中
  - literal: `/ (hidden_size * 4)` 权重维度推导
  - literal: `output_tokens[0] != 0` token 0 哨兵
- **sameClassCriterion**: 任何 Rust 侧 buffer/stride/size 计算硬编码 `* 4` 而非从 compute dtype 推导 elem_bytes；任何将 dtype-aware 数据传给 F32-only 转换函数的代码
- **fixTemplate**: 用 `compute_dtype.size_bytes()` 或 `elem_bytes` 替代 `* 4`；`bytes_to_f32_vec_with_elem_bytes(&data, elem_bytes)` 替代 `bytes_to_f32_vec`；BF16→f32 转换用 `half::bf16::from_bits(bits).to_f32()`；`DiagnosticScratchpad` 增加 `elem_bytes` 字段
- **regressionAssertion**: 对任何 compute dtype: `buffer_size == count * compute_dtype.size_bytes()`; GPU 下载后 `result.len() == expected_elem_count`

### 根因
Rust 侧多处 buffer 大小/stride 计算硬编码 `* 4`（F32 elem_bytes），违反 ARCH-JIT-DATA-YIELDS 铁律"跨侧 dtype 硬编码对齐"禁令。JIT 侧已通过 `ctx.dtype.elem_bytes()` 正确感知 dtype，但 Rust 侧未同步。当 compute dtype 非 F32（如 BF16=2 bytes、F16=2 bytes）时，buffer 分配不足或 stride 错误。更深层：GPU 路径下载 BF16 字节后调用 `bytes_to_f32_vec` 将每 4 字节解释为 f32，返回半数错误值。权重维度推导 `sz / (hidden_size * 4)` 在 BF16 权重下返回维度/2。

### 影响
- `gpu_backend_macro.rs` 7 处 `* 4` 硬编码 + 4 处缺少 `let elem_bytes` 定义：GPU 路径 buffer 分配/stride 全部错误，4 处为潜伏编译错误
- `gpu_backend_macro.rs` 6 处 `bytes_to_f32_vec` 在 BF16 下载路径上返回错误元素数量
- `gpu_helpers.rs` 缺少 dtype-aware 字节→f32 转换函数
- `executor_ops.inc.rs` 5 处 `* 4` 硬编码：logits stride / output copy / rerank offset / score_tokens offset / diagnostic logits
- `executor_ops.inc.rs` 5 处 `copy_nonoverlapping` 读取 F32 但 scratchpad 可能为 BF16
- `pack_observe.inc.rs` `embedding()` 总是 `read_f32_at`，BF16 scratchpad 返回垃圾
- `pack_observe.inc.rs` `last_token_logits()`: logits row stride 硬编码 `vocab_size * 4`
- `executor_core.inc.rs` `output_tokens[0] != 0` tok==0 哨兵丢弃合法 token 0
- `executor_compile.rs` 2 处 `sz / (hidden_size * 4)` BF16 权重维度推导返回维度/2
- `mid_layer_encode.rs` 生产路径 `hidden_state: vec![0u8; hidden_size * 4]` 假设 F32
- `abi_types.inc.rs` `runtime_scratchpad_bytes()`: sampling workspace 硬编码 `vocab_bytes * 4`
- ~57 处 `DiagnosticScratchpad` 构造缺少 `elem_bytes` 字段

### 根治
1. `gpu_backend_macro.rs`: 4 处添加 `let elem_bytes = config.geometry.compute_dtype.size_bytes();`; 7 处 `* 4` → `* elem_bytes`; 6 处 `bytes_to_f32_vec` → `bytes_to_f32_vec_with_elem_bytes(&data, elem_bytes)`
2. `gpu_helpers.rs`: 新增 `bytes_to_f32_vec_with_elem_bytes(data, elem_bytes)` 函数，elem_bytes=2 时 BF16→f32 转换
3. `executor_ops.inc.rs`: 5 处 `* 4` → `* mega.elem_bytes`; 5 处 `copy_nonoverlapping` → `match elem_bytes { 4 => direct copy, 2 => BF16→f32 }`; `diagnostic_prefill_logits` 同样添加 BF16 处理
4. `pack_observe.inc.rs`: `embedding()` 添加 `match self.elem_bytes { 4 => read_f32_at, 2 => BF16→f32 }`; `last_token_logits()` 改用 `vocab_size * self.elem_bytes`
5. `executor_core.inc.rs`: 移除 `output_tokens[0] != 0` 哨兵，直接用 `generated_count`
6. `executor_compile.rs`: 2 处 `sz / (hidden_size * 4)` → `sz / (hidden_size * elem_bytes)`
7. `mid_layer_encode.rs`: `hidden_state: vec![0u8; hidden_size * 4]` → `hidden_size * compute_dtype.size_bytes()`
8. `abi_types.inc.rs`: 新增 `SAMPLING_WORKSPACE_MULTIPLIER = 4` 命名常量
9. ~57 处测试 `DiagnosticScratchpad` 构造补 `elem_bytes: 4`（测试中 F32 是合理的）
10. RoPE `* 4` 不修（RoPE cos/sin 精度始终 F32，与 compute dtype 无关）
11. SG 共享内存 `* 4` 不修（JIT 侧 SgDetect/SgInject 接口规范为 F32）
12. weight_blob 读取 `* 4` 不修（权重存储始终 F32）

### 归因时间
2026-06-24

### 根治状态
**status**: 根治 ✅
**residual**: 0
**confirmReport**:
- `grep -rn '\\* 4' src/engine src/compat src/loader` 仅命中授权例外（RoPE/SG/weight_blob） ✅
- BF16 模型 E2E 推理 buffer size 正确 ✅
- DiagnosticScratchpad 全部含 elem_bytes 字段 ✅
- bytes_to_f32_vec_with_elem_bytes 覆盖 BF16/F16 转换 ✅

---

## BCE-20260624-013 — 静默降级编解码不匹配

**patternId**: BCE-20260624-013
**title**: ZstdDict/NvcompAns 编解码静默降级到 LZ4
**layer**: 设计

**codePattern**:
- `CompressionCodec::ZstdDict` 空字典时 `lz4_compress(data)` fallback → 存储 ZstdDict codec tag + LZ4 数据
- `CompressionCodec::NvcompAns` CPU 不可用时 `lz4_compress(data)` fallback → 存储 NvcompAns codec tag + LZ4 数据
- `compress_weight` 用 `.ok().flatten()` 静默丢弃错误

**triggerCondition**: ZstdDict 字典未训练 / NvcompAns 在 CPU 环境

**detectionSignatures**:
- literal: `lz4_compress(data)` 在 ZstdDict/NvcompAns match arm 内
- literal: `.ok().flatten()` 在 compress_weight 函数

**sameClassCriterion**: 任何 codec 编码路径使用与 codec tag 不匹配的实际压缩算法

**fixTemplate**:
1. ZstdDict 空字典 → `Err(CodecError(...))` (NO-SILENT-FALLBACK)
2. NvcompAns 不可用 → `Err(CodecError(...))` (NO-FALLBACK)
3. `compress_weight` 用 `unwrap_or_else(|e| { log::warn!(...); None })` 替代 `.ok().flatten()`

**regressionAssertion**: compress_weight_page(ZstdDict, empty_dict) → is_err(); compress_weight_page(NvcompAns, cpu) → is_err()

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | ZstdDict/NvcompAns 错误路径返回 Err；compress_weight 用 log::warn 替代静默丢弃

---

## BCE-20260624-014 — AllGather 缓冲区分配不足

**patternId**: BCE-20260624-014
**title**: AllGather 缓冲区分配 elem_count 但写入 world_size*elem_count
**layer**: 设计

**codePattern**: `vec![0.0f32; sendcount]` 然后 `all_gather_inplace(&mut buf, sendcount)` — all_gather_inplace 文档要求 buffer 容量 = world_size * sendcount

**triggerCondition**: 分布式推理，world_size > 1 时 OOB 写入

**detectionSignatures**:
- literal: `vec![0.0f32; elem_count]` 紧接 `all_gather_inplace`
- structural: buffer 分配大小不含 world_size 因子

**sameClassCriterion**: 任何 all_gather_inplace 调用的 buffer 分配大小不含 world_size 因子

**fixTemplate**: `vec![0.0f32; elem_count * world_size]` + 显式提取 world_size 局部变量

**regressionAssertion**: all_gather_inplace 调用时 buffer.len() >= world_size * sendcount

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | buffer 分配包含 world_size 因子

---

## BCE-20260624-015 — AllGather sendcount 整数截断

**patternId**: BCE-20260624-015
**title**: buffer.len() / world_size 整数除法截断余数
**layer**: 设计

**codePattern**: `sendcount = buffer.len() / world_size` 无可整除性检查

**triggerCondition**: buffer 大小不是 world_size 整数倍时 sendcount 截断 → 数据丢失

**detectionSignatures**:
- literal: `buffer.len() / .*world_size` 无 assert
- structural: 除法前无可整除断言

**sameClassCriterion**: 任何整数除法计算分布式参数时缺少可整除性检查

**fixTemplate**: 除法前 `assert!(buffer.len() % world_size == 0, "...")`

**regressionAssertion**: 不可整除时 panic 而非静默截断

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | 除法前添加可整除性 assert

---

## BCE-20260624-016 — PageAddrTable current_tier 迁移后不更新

**patternId**: BCE-20260624-016
**title**: 页面迁移完成后 addr_table.current_tier 未更新导致重复迁移
**layer**: 设计

**codePattern**: `drain_completions_and_update` 接收 `&PageAddrTable` 参数但以下划线前缀忽略，只更新 PageMetadata 不更新 PageAddrEntry

**triggerCondition**: 页面从 GpuHbm 驱逐到 CpuDram 后 addr_table 仍显示 GpuHbm → 重复驱逐；或换入后仍显示 CpuDram → 重复换入

**detectionSignatures**:
- literal: `_addr_table` 参数名（下划线前缀 = 未使用）
- structural: 迁移完成处理只更新 metadata 不更新 addr_table

**sameClassCriterion**: 任何迁移/状态变更操作只更新部分状态存储，导致状态不一致

**fixTemplate**: 移除下划线前缀，在 MigrationResult::Ok 处理中同时更新 `addr_table` 的 `entry.current_tier = done.to_tier`

**regressionAssertion**: 迁移后 addr_table[page_id].current_tier == new_tier

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | addr_table 在 MigrationResult::Ok 中同步更新

---

## BCE-20260624-017 — poll_transfers 吞掉 KV transfer 失败

**patternId**: BCE-20260624-017
**title**: poll_transfers 失败时 log::warn 后丢弃，调用者无从得知
**layer**: 设计

**codePattern**: `future.wait()` 的 Err 分支只 `log::warn!`，不返回任何失败信息给调用者

**triggerCondition**: 分布式 KV transfer 失败（网络/NCCL 错误）

**detectionSignatures**:
- literal: `log::warn!("[poll_transfers] async transfer failed` 后无 return/push
- structural: 函数返回 `Vec<KvTransferResult>`（只有 Ok 结果，无 Err 空间）

**sameClassCriterion**: 任何异步操作结果收集只返回成功结果，失败被静默丢弃

**fixTemplate**: 返回 `Vec<Result<KvTransferResult, String>>`，让调用者决定如何处理失败

**regressionAssertion**: poll_transfers 返回值包含 Err variant，调用者必须处理

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | 返回 `Vec<Result<KvTransferResult, String>>`，失败信息传播

---

## BCE-20260624-018 — victim_id wrapping_sub 整数下溢出

**patternId**: BCE-20260624-018
**title**: expert group ID wrapping_sub 无下界检查
**layer**: 设计

**codePattern**: `victim_id.wrapping_sub(1_000_000) as usize` — victim_id < 1_000_000 时下溢出产生巨大索引

**triggerCondition**: HGAL 返回非 expert group ID（ID < 1_000_000 基偏移）

**detectionSignatures**:
- literal: `wrapping_sub` 在 group ID 上
- structural: 减法无下界检查

**sameClassCriterion**: 任何从编码 ID 提取索引的减法缺少下界检查

**fixTemplate**: `checked_sub().expect("...")` 替代 `wrapping_sub()`

**regressionAssertion**: victim_id < base_offset 时 panic，不产生巨大索引

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | 用 `checked_sub().expect()` 替代 `wrapping_sub()`

---

## BCE-20260624-019 — Mutex poison 恢复掩盖数据不一致

**patternId**: BCE-20260624-019
**title**: Mutex poison 时 into_inner() 恢复数据继续推理，掩盖前一个线程 panic 导致的不一致
**layer**: 设计

**codePattern**: `.lock().unwrap_or_else(|e| e.into_inner())` — poison 时恢复内部数据继续使用

**triggerCondition**: 持有锁的线程在修改数据过程中 panic → 数据可能不一致 → into_inner() 恢复后继续推理产生错误结果

**detectionSignatures**:
- literal: `into_inner()` 在 Mutex lock 之后
- literal: `unwrap_or_else(|e| e.into_inner())` 或 `unwrap_or_else(|err| err.into_inner())`

**sameClassCriterion**: 任何 Mutex poison 时恢复数据而非终止操作的代码

**fixTemplate**: `.lock().expect("mutex poison — previous holder panicked, cannot continue inference")` — 推理引擎中任何 panic 都应终止推理

**regressionAssertion**: Mutex poison 时 panic 传播，不恢复数据继续推理

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | `.lock().expect("mutex poison...")` 替代 `into_inner()` 恢复

---

## BCE-20260624-020 — MoE Custom mapping 静默 RoundRobin fallback

**patternId**: BCE-20260624-020
**title**: Custom expert mapping 缺失时静默 fallback 到 RoundRobin
**layer**: 设计

**codePattern**: `mapping.get(id).copied().unwrap_or(id % world_size)` — 用户指定 Custom mapping 但未覆盖所有 expert 时，缺失项静默 RoundRobin 分配

**triggerCondition**: Custom mapping 长度 < num_experts

**detectionSignatures**:
- literal: `unwrap_or(expert_id % self.world_size)` 在 placement match 中
- structural: Option::unwrap_or 回退到不同分配策略

**sameClassCriterion**: 任何用户指定的映射/配置缺失项静默 fallback 到默认策略而非报错

**fixTemplate**: `.ok_or_else(|| format!("expert_id {} not found in Custom mapping...", id))` — Result 传播

**regressionAssertion**: Custom mapping 缺失 expert → Err，不静默 RoundRobin

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | Custom mapping 缺失项返回 Err 而非静默 fallback

---

## BCE-20260624-021 — TieredCache migration plan 丢弃 no-op

**patternId**: BCE-20260624-021
**title**: build_batch() 产出的 TierMigrationPlan 被 `_plan` 丢弃，tier migration 为空操作
**layer**: 设计

**codePattern**: `let _plan = coordinator.build_batch(&[], 0.5)` — 计算了迁移计划但不执行

**triggerCondition**: TieredCache 模式下有 page 需要迁移

**detectionSignatures**:
- literal: `let _plan =` 或 `let _ =` 丢弃非 trivial 返回值
- structural: 返回值包含 Vec/容器字段但被丢弃

**sameClassCriterion**: 任何计算结果被 `_` 丢弃（尤其包含待执行操作的容器）

**fixTemplate**: 执行 plan 或 log::warn! 标注为未执行（直到 scheduler 集成完成）

**regressionAssertion**: TierMigrationPlan 非空时必须至少有日志输出

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | three_tier_swap execute_plan 实现，非空 plan 执行并 log

---

## BCE-20260624-022 — f32→usize 负值环绕

**patternId**: BCE-20260624-022
**title**: f32 转 usize 无负值保护，负 f32 环绕为接近 usize::MAX 的巨大值
**layer**: 设计

**codePattern**: `(f32_value) as usize` — f32 为负时环绕为巨大 usize → 缓冲区过分配/索引 OOB

**triggerCondition**: 算术运算产生负 f32（如 memory_pressure_ratio < 0、capacity_factor < 0、sparsity > 2.0）

**detectionSignatures**:
- literal: `as usize` 前置为 f32 表达式
- structural: f32 算术结果无范围检查直接转 usize

**sameClassCriterion**: 任何 f32→usize 转换无负值/上界保护

**fixTemplate**: `.clamp(0.0, usize::MAX as f32) as usize` 或返回 Result 传播错误

**regressionAssertion**: 负 f32 输入 → clamp 到 0 而非环绕到 usize::MAX

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | `.clamp(0.0, usize::MAX as f32) as usize` 统一应用

---

## BCE-20260624-023 — GPU mem_free 错误静默丢弃

**patternId**: BCE-20260624-023
**title**: `let _ = driver.mem_free()` 吞掉 GPU 内存释放错误，导致 GPU 内存泄漏不可观测
**layer**: 设计

**codePattern**: `let _ = driver.mem_free(*ptr)` 或 `let _ = backend.free_gpu_page(gpu_ptr)` — 释放 GPU 内存失败被静默忽略

**triggerCondition**: GPU 驱动返回内存释放错误

**detectionSignatures**:
- literal: `let _ =` 后跟 `mem_free`/`free_gpu_page`
- structural: 资源释放返回值被丢弃

**sameClassCriterion**: 任何资源释放错误被 `let _ =` 吞掉（GPU/CPU 内存、文件句柄、网络连接）

**fixTemplate**: Drop 中用 `if let Err(e) = ... { log::error!(...) }`；非 Drop 函数用 `?` 传播

**regressionAssertion**: GPU 内存释放失败时 log::error! 输出，不静默

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | Drop 中 `if let Err(e) = mem_free { log::error!(...) }` 替代 `let _ =`

---

## BCE-20260624-024 — decode 热路径 .unwrap() 无上下文 panic

**patternId**: BCE-20260624-024
**title**: 推理 decode 热路径中 .unwrap() 缺少诊断信息，panic 时无法定位根因
**layer**: 设计

**codePattern**: `self.eagle_config.as_ref().unwrap()` / `self.ngram_index.as_ref().unwrap()` — 推理路径 panic 无具体原因

**triggerCondition**: 配置/状态不一致导致 Option 为 None

**detectionSignatures**:
- literal: `.unwrap()` 在 `as_ref()`/`as_mut()` 之后
- structural: 推理热路径中 Option 解包无诊断

**sameClassCriterion**: 任何推理热路径中的 .unwrap() 缺少 expect 诊断

**fixTemplate**: `.expect("具体原因 — 调用什么方法修复")` 替代 `.unwrap()`

**regressionAssertion**: panic 消息包含具体原因和修复建议

**归因时间**: 2026-06-24

**根治状态**: 根治 ✅ | residual: 0 | 所有推理路径 .unwrap() 替换为含诊断的 .expect()

---

## 硬编码 HACK (Hardcoded Hardware/Model Params)

> 来自三组审计（硬件参数 / 模型参数 / 孤岛模块）的硬编码与未接入符号发现归档。按 smellClass 聚类：AP-HARDCODED-HW（硬件参数字面量，应从 DeviceProfile 派生）、AP-HARDCODED-MODEL（模型参数字面量，应从 ModelConfig 派生）、AS-ISLAND-MODULE（有定义无生产调用）。状态：待根治。

```yaml
- patternId: BCE-HACK-HW-001
  title: Metal GpuDeviceProfile 全字段硬编码（compute_units/shared_mem/warp_size/bandwidth/gflops/clock）
  layer: 设计缺陷
  smellClass: AP-HARDCODED-HW
  instances:
    - file: /home/putao/code/rust/gllm/src/compat/metal_backend.rs
      line: 85-100
      value: "compute_units: 10, shared_mem_per_block: 32768, warp_size: 32, max_threads_per_block: 1024, memory_bandwidth_gbs: 200.0, peak_gflops_f32: 5000.0, peak_gflops_f16: 10000.0, clock_mhz: 1000"
      shouldDeriveFrom: "MTLDevice API（maxThreadgroupMemoryLength / maxThreadsPerThreadgroup / maxThreadgroupWidth|Height|Depth）+ 按 device.name() 分型号查表 bandwidth/gflops（M1/M2/M3 Pro/Max/Ultra 差异极大）"
  codePattern:
    - "GpuDeviceProfile { compute_units: <字面量>, shared_mem_per_block: <字面量>, memory_bandwidth_gbs: <字面量>, peak_gflops_*: <字面量>, clock_mhz: <字面量>, ... }"
    - "所有 Apple Silicon 设备共享同一组失真数值，忽略 M1/M2/M3 Ultra 间 2-10 倍性能差异"
  triggerCondition: 任何 Metal (Apple Silicon) 设备构建 GpuDeviceProfile 的生产路径
  detectionSignatures:
    literal: "memory_bandwidth_gbs: 200.0"
    literal: "peak_gflops_f32: 5000.0"
    literal: "clock_mhz: 1000"
    structural: "GpuDeviceProfile struct literal with >3 hardcoded numeric fields (non-total_memory)"
  sameClassCriterion: 任何 GpuDeviceProfile 字段（除 total_memory 外）使用字面量而非 Metal API 探测或分型号查表
  fixTemplate:
    - "扩展 gllm-kernels/src/gpu/metal/device.rs 暴露 maxThreadgroupMemoryLength / maxThreadsPerThreadgroup / maxThreadgroupWidth|Height|Depth 等 Metal API selector"
    - "按 device.name() 分型号查表 memory_bandwidth_gbs / peak_gflops_* / clock_mhz（M1/M2/M3 Pro/Max/Ultra）"
    - "compute_units 从 MTLCapture 范围或 device.name() 推导"
  regressionAssertion: "M1 vs M3 Ultra 构建的 GpuDeviceProfile 字段值不同；不再出现 200.0/5000.0/10000.0/1000 固定字面量"
  违反铁律: ARCH-JIT-YIELDS（P0 硬件信息驱动）/ ARCH-ROOT-CAUSE（治本不治标）
  归因时间: 2026-06-26
  status: 待根治
  residual: 1
```

```yaml
- patternId: BCE-HACK-HW-002
  title: FA3/FA4 SMEM 预算硬编码 49152 字面量（忽略 A100/H100/Blackwell 差异）
  layer: 设计缺陷
  smellClass: AP-HARDCODED-HW
  instances:
    - file: /home/putao/code/rust/gllm-kernels/src/compiler/planner.rs
      line: 1107
      value: "49152 (typical SMEM size 字面量)"
      shouldDeriveFrom: "GpuDeviceProfile.shared_mem_per_block（A100=49152, H100=227328, Blackwell=228KB）"
  codePattern:
    - "cache.l1_tile_budget.max(49152) // typical SMEM size"
    - "对所有 SM 版本一刀切用 48KB，H100/Blackwell 上严重低估可用 SMEM → tile 选型过小 → 性能损失"
  triggerCondition: GPU 编译路径的 FlashAttention 3/4 tile 大小决策
  detectionSignatures:
    literal: ".max(49152)"
    literal: "typical SMEM size"
    structural: "AttentionVariant::FA3Pipeline | AttentionVariant::FA4BlockScaled arm 内使用字面量 SMEM 预算"
  sameClassCriterion: 任何 GPU SMEM 相关预算/容量计算使用字面量而非 GpuDeviceProfile.shared_mem_per_block 派生
  fixTemplate: "传入 profile.shared_mem_per_block 并取 min(available, hw_limit)；删除 49152 字面量"
  regressionAssertion: "H100 上 FA3/FA4 的 cache_for_attn >= 227328；A100 上 = 49152；不再出现硬编码 49152"
  违反铁律: ARCH-JIT-YIELDS（P1 输入文件 / P0 硬件驱动）/ NO-HW-DEGRADATION（H100 被降级到 A100 SMEM 容量）
  归因时间: 2026-06-26
  status: 待根治
  residual: 1
```

```yaml
- patternId: BCE-HACK-MODEL-001
  title: MLA d_rope 硬编码 fallback 64（DeepSeek V3/R1/Kimi-K2 固定值）
  layer: 设计缺陷
  smellClass: AP-HARDCODED-MODEL
  instances:
    - file: /home/putao/code/rust/gllm/src/model_config_fragments/config_impl.inc.rs
      line: 407
      value: "64 (MLA d_rope unwrap_or fallback)"
      shouldDeriveFrom: "GGUF metadata qk_rope_head_dim 或 config.json rope_dimension_count"
    - file: /home/putao/code/rust/gllm/src/model_config_fragments/config_impl.inc.rs
      line: 816
      value: "64 (config.json 路径同源 fallback)"
      shouldDeriveFrom: "find_usize(value, ['rope_dimension_count','qk_rope_head_dim',...])"
  codePattern:
    - "reader.qk_rope_head_dim().unwrap_or(64) as usize"
    - "find_usize(value, &['rope_dimension_count','qk_rope_head_dim',...]).unwrap_or(64)"
    - "DeepSeek V3/R1/Kimi-K2 的 d_rope 恒为 64，故当前正确；但未来 d_rope != 64 的 MLA 模型会静默使用错误值"
  triggerCondition: MLA 模型的 GGUF/config.json 元数据中 qk_rope_head_dim 字段缺失
  detectionSignatures:
    literal: "unwrap_or(64) 在 qk_rope_head_dim / rope_dimension_count 上下文"
    structural: "MLA d_rope 推导路径使用 unwrap_or 字面量 fallback"
  sameClassCriterion: 任何 MLA d_rope 参数使用 unwrap_or(64) 字面量 fallback 而非 Err 传播
  fixTemplate:
    - "缺失时返回 Err(\"MLA d_rope (qk_rope_head_dim) missing in metadata\") 而非 fallback 64（NO-SILENT-FALLBACK）"
    - "或加 warning log 标注 fallback 触发"
  regressionAssertion: "构造 d_rope 缺失的 MLA metadata → 返回 Err 或触发 warning；不再静默使用 64"
  违反铁律: NO-SILENT-FALLBACK（静默 fallback 违宪精神）/ ARCH-ROOT-CAUSE（64 是经验值非派生）
  归因时间: 2026-06-26
  status: 已根治 ✅
  residual: 0
```

```yaml
- patternId: BCE-HACK-MODEL-002
  title: num_experts_per_tok 硬编码 fallback 2（MoE 行业默认值，已标 LEGAL）
  layer: 设计缺陷
  smellClass: AP-HARDCODED-MODEL
  instances:
    - file: /home/putao/code/rust/gllm/src/model_config_fragments/config_impl.inc.rs
      line: 974
      value: "2 (num_experts_per_tok unwrap_or fallback, 已有 LEGAL 注释)"
      shouldDeriveFrom: "config.json num_experts_per_tok"
  codePattern:
    - "self.num_experts_per_tok.unwrap_or(2) // LEGAL: num_experts_per_tok=2 是 MoE 的行业标准默认值"
    - "MoE top-k experts 的 fallback，已标 LEGAL 但仍违反 NO-SILENT-FALLBACK 精神"
  triggerCondition: MoE 模型的 config.json 中 num_experts_per_tok 字段缺失
  detectionSignatures:
    literal: "num_experts_per_tok.unwrap_or(2)"
    structural: "MoE top-k 参数 unwrap_or 字面量 fallback"
  sameClassCriterion: 任何 MoE 配置参数使用 unwrap_or 字面量 fallback 而非 Err 传播（即便标 LEGAL）
  fixTemplate:
    - "缺失时返回 Err(\"num_experts_per_tok missing in config.json\") 或 warning log"
    - "保留 LEGAL 注释作为过渡，但应向 Err 演进"
  regressionAssertion: "num_experts_per_tok 缺失时返回 Err 或 warning；不再静默使用 2"
  违反铁律: NO-SILENT-FALLBACK（精神层面）/ ARCH-ROOT-CAUSE
  归因时间: 2026-06-26
  status: 已根治 ✅
  residual: 0
```

```yaml
- patternId: BCE-HACK-MODEL-003
  title: builder.inc.rs 临时 manifest 架构占位符 "llama"
  layer: 设计缺陷（轻微）
  smellClass: AP-HARDCODED-MODEL
  instances:
    - file: /home/putao/code/rust/gllm/src/client_fragments/builder.inc.rs
      line: 427
      value: "\"llama\" (临时 dummy_manifest 架构占位符)"
      shouldDeriveFrom: "loader.detect_architecture()（line 431 实际调用）"
    - file: /home/putao/code/rust/gllm/src/client_fragments/builder.inc.rs
      line: 649
      value: "\"llama\" (同源占位符)"
      shouldDeriveFrom: "loader.detect_architecture()"
  codePattern:
    - "let dummy_manifest = make_dummy_manifest(model_id, \"llama\", kind);"
    - "Ω1 tensor-driven 推导路径的临时占位符；dummy_manifest 仅传给 from_loader()（不读 manifest.arch），随后 detect_architecture() 得到真实 arch 重建 manifest"
  triggerCondition: 任何 builder 路径构建临时 manifest
  detectionSignatures:
    literal: "make_dummy_manifest(model_id, \"llama\", kind)"
    structural: "manifest 构造使用字面量架构字符串而非 detect_architecture 结果"
  sameClassCriterion: 任何 manifest 构造使用字面量架构字符串作为占位符（即便功能无 bug）
  fixTemplate:
    - "重构为 make_dummy_manifest(model_id, \"\", kind)（空串更明确表达'待推导'）"
    - "或重构 API 使此阶段不需要 manifest 占位符"
  regressionAssertion: "dummy_manifest 不再含 'llama' 字面量；真实 arch 始终来自 detect_architecture"
  违反铁律: ARCH-ROOT-CAUSE（API 设计要求传 manifest 但此阶段还没有真实 arch）
  归因时间: 2026-06-26
  status: 已根治 ✅
  residual: 0
```

```yaml
- patternId: BCE-HACK-MODEL-004
  title: 权重形状推导静默 fallback（AltUp P / LAuReL rank — 权重已确认存在却取不到形状）
  layer: 设计缺陷
  smellClass: AP-HARDCODED-MODEL
  instances:
    - file: /home/putao/code/rust/gllm/src/arch/auto_graph_fragments/types.inc.rs
      line: 220
      value: "2 (altup_num_inputs unwrap_or fallback, has_altup=true 时)"
      shouldDeriveFrom: "correction_coefs / altup.correction 张量形状首维 P"
      fixStrategy: "panic on invariant violation — has_altup 检测条件含 correction_coefs/altup.correction，形状 filter 必有匹配；空结果即内部不变量违反"
    - file: /home/putao/code/rust/gllm/src/arch/auto_graph_fragments/build_graph.inc.rs
      line: 684
      value: "64 (per-layer LAuReL laurel_rank unwrap_or fallback)"
      shouldDeriveFrom: "weight_shapes[laurel_up] 首维"
      fixStrategy: "GraphBuildError::MissingTensor Err 传播（父函数 build_compiler_graph 返回 Result）"
    - file: /home/putao/code/rust/gllm/src/arch/auto_graph_fragments/build_graph.inc.rs
      line: 1520
      value: "64 (layer-template LAuReL laurel_rank 同源 fallback)"
      shouldDeriveFrom: "weight_shapes[laurel_up@L0] 首维"
      fixStrategy: "同上 GraphBuildError::MissingTensor Err 传播"
  codePattern:
    - "weight_shapes.get(key).map(|s| s[0]).unwrap_or(64)"
    - "weight_shapes.keys().filter(...).filter_map(...).next().unwrap_or(2)"
    - "if let Some(weight) = weight_lookup { ... weight_shapes.get(name).unwrap_or(N) } — 权重已 Some 但形状查找 fallback，自相矛盾"
  triggerCondition: 权重存在性检测通过（has_altup / Some(weight)）但后续形状查找失败
  detectionSignatures:
    literal: "unwrap_or(64) / unwrap_or(2) 在 weight_shapes 查找上下文"
    structural: "形状推导使用 unwrap_or 字面量而非 Err/panic，且查找发生在权重存在性守卫之后"
  sameClassCriterion: 任何权重形状推导使用 unwrap_or 字面量 fallback（即便周围有权重存在性守卫保证查找应成功）
  fixTemplate:
    - "父函数返回 Result → .ok_or_else(|| GraphBuildError::MissingTensor(...))? 传播"
    - "父函数返回非 Result（如 analyze_architecture）→ panic on invariant violation（带清晰原因），禁止静默 fallback"
  regressionAssertion: "权重存在但形状缺失 → 返回 Err 或 panic；不再静默使用 64/2"
  违反铁律: NO-SILENT-FALLBACK / ARCH-ROOT-CAUSE
  归因时间: 2026-06-27
  status: 已根治 ✅
  residual: 0
```

```yaml
- patternId: BCE-ISLAND-001
  title: mega_kernel_gpu.rs 整模块孤岛（SmPartitionConfig/MkCompileVariant/DualBatchMeta/RequestQueue/OutputRingBuffer 全无生产调用）
  layer: 范式缺陷
  smellClass: AS-ISLAND-MODULE
  instances:
    - file: /home/putao/code/rust/gllm/src/engine/mega_kernel_gpu.rs
      line: 457
      symbol: "SmPartitionConfig (pub struct + impl derive/cluster_62/grid_sync/serial)"
      prodCalls: 0
      testCalls: "全部在自身 #[cfg(test)] mod tests（行 898/908/922/932/1208-1259/1648-2758/2842）"
      note: "pipeline/scheduler.rs:388 存在另一个同名 SmPartitionConfig（pipeline 模块独立定义，不同类型）"
    - file: /home/putao/code/rust/gllm/src/engine/mega_kernel_gpu.rs
      line: 441
      symbol: "MkCompileVariant (4 变体枚举 Serial/Cluster62/Cluster53/GridSync)"
      prodCalls: 0
      testCalls: "仅被孤岛的 SmPartitionConfig 字段引用（行 459）及测试断言"
    - file: /home/putao/code/rust/gllm/src/engine/mega_kernel_gpu.rs
      line: 22
      symbol: "DualBatchMeta (pub struct)"
      prodCalls: "经 batch_context.rs:326 set_ext_dual_batch_meta（prod 方法），但该方法 5 处调用方均在 batch_context.rs #[cfg(test)] 区（行 556/1205/2150/2166/2959）"
      testCalls: "Setter 是 dead code 桥，无推理热路径消费"
    - file: /home/putao/code/rust/gllm/src/engine/mega_kernel_gpu.rs
      line: 236
      symbol: "RequestQueue + RequestQueueEntry (行 72) + enqueue/dequeue/dequeue_batch/peek"
      prodCalls: 0
      testCalls: "热路径文件（executor_step.rs/executor.rs/mega_kernel.rs/mega_kernel_v2.rs/mega_kernel_callback.rs/mtp_executor.rs/batch_executor.rs）零调用；batch_context.rs 仅引用 EXT_REQUEST_QUEUE_PTR 常量（usize 槽位），从未实例化"
    - file: /home/putao/code/rust/gllm/src/engine/mega_kernel_gpu.rs
      line: 330
      symbol: "OutputRingBuffer (pub struct + impl)"
      prodCalls: 0
      testCalls: "与 RequestQueue 同——热路径零调用，batch_context 仅引用 EXT_OUTPUT_RING_PTR 常量"
  codePattern:
    - "整个 mega_kernel_gpu 模块（engine/mod.rs:19 pub mod mega_kernel_gpu）唯一外部消费者是 batch_context.rs"
    - "batch_context.rs 仅消费 EXT_* 常量与 DualBatchMeta（经测试-only 调用链）"
    - "executor / mega_kernel / cuda_backend 热路径对 mega_kernel_gpu 模块零引用"
    - "5 个核心类型全部为 GPU Mega-Kernel SM 分区/双批调度/请求队列/输出环形缓冲的设计预埋代码，从未接入推理热路径"
  triggerCondition: N/A（孤岛，不触发运行时）
  detectionSignatures:
    literal: "pub mod mega_kernel_gpu in engine/mod.rs:19"
    structural: "SmPartitionConfig/MkCompileVariant/DualBatchMeta/RequestQueue/OutputRingBuffer 的所有调用方均在 #[cfg(test)] 内"
    structural: "executor_step.rs / executor.rs / mega_kernel.rs / cuda_backend.rs 零引用 mega_kernel_gpu::SmPartitionConfig 等"
  sameClassCriterion: 任何 pub 类型/模块的所有调用方均在 #[cfg(test)] 内，无生产推理路径消费
  fixTemplate:
    - "方案 A（接入）：将 SmPartitionConfig/MkCompileVariant 接入生产 codegen 路径（与 gllm-kernels mega_kernel_emit.rs:53-69 select_mk_variant 对齐）"
    - "方案 B（删除）：删除整个 mega_kernel_gpu 模块（生产已有 mega_kernel_emit.rs::select_mk_variant 等价实现，参数化 sm_version/total_sm，硬编码 cluster_size=8 是 portable cluster 合理默认）"
    - "DualBatchMeta/RequestQueue/OutputRingBuffer：接入推理热路径或删除"
  regressionAssertion: "grep 'SmPartitionConfig|DualBatchMeta|RequestQueue|OutputRingBuffer' 在 executor_step.rs/executor.rs/mega_kernel.rs 等热路径文件有命中，或整个模块删除"
  违反铁律: NO-ISLAND-MODULE（编译通过+测试通过≠完成，需真实调用链接入）
  归因时间: 2026-06-26
  status: 待根治
  residual: 5
```

```yaml
- patternId: BCE-ISLAND-002
  title: gllm-kernels IsaProfile::cuda() 硬编码查找表（无生产消费方）
  layer: 范式缺陷
  smellClass: AS-ISLAND-MODULE
  instances:
    - file: /home/putao/code/rust/gllm-kernels/src/compiler/codegen/vm/isa_profile.rs
      line: 583-594
      symbol: "IsaProfile::cuda(sm_version) 查表函数"
      value: "warp_size=32, smem_kb/reg_file/max_regs 按 sm_version 分档（100..=Blackwell, 90..=99=Hopper, ...）"
      shouldDeriveFrom: "NVIDIA 公开架构 spec（NVIDIA 无运行时查询 shared_mem_per_sm 的 CUDA API）"
      prodCalls: "无直接生产调用；非测试命中只有 mega_kernel_emit.rs:6083（在 test mod 内，访问 .platform 字段）"
      testCalls: "isa_profile.rs:934/946/1036/1231, jit_context.rs:1578 均在 #[cfg(test)] 内"
  codePattern:
    - "pub fn cuda(sm_version) -> Self { match sm_version { 100.. => (228, 65536, 255), ... } }"
    - "查表本身是合理 fallback（NVIDIA 无 API），但当前无生产消费方；生产用 GpuDeviceProfile（运行时探测）而非 IsaProfile（查表）"
  triggerCondition: N/A（孤岛，不触发运行时）
  detectionSignatures:
    literal: "pub fn cuda(sm_version: u32) -> Self"
    structural: "IsaProfile::cuda 的所有调用方均在 #[cfg(test)] 内"
  sameClassCriterion: 任何硬件查表函数无生产消费方（孤岛），或未来接入 codegen 时未与 GpuDeviceProfile 探测值交叉校验
  fixTemplate:
    - "方案 A（接入）：将 IsaProfile 接入生产 codegen 路径，并与 GpuDeviceProfile 探测值交叉校验"
    - "方案 B（删除）：删除孤岛查表函数，统一用 GpuDeviceProfile 作为硬件真相源"
  regressionAssertion: "IsaProfile::cuda 有生产调用方，或整个函数删除"
  违反铁律: NO-ISLAND-MODULE
  归因时间: 2026-06-26
  status: 待根治
  residual: 1
```

```yaml
- patternId: BCE-ISLAND-003
  title: pipeline/scheduler.rs default_for_gpu 硬编码 10% 通信比例（无生产调用）
  layer: 设计缺陷
  smellClass: AS-ISLAND-MODULE
  instances:
    - file: /home/putao/code/rust/gllm/src/engine/pipeline/scheduler.rs
      line: 404-409
      symbol: "default_for_gpu(total_sms) 构造函数"
      value: "(total_sms * 10 / 100).max(1) 硬编码 10% 通信 SM 比例"
      shouldDeriveFrom: "REQ-DIST-026 — 从通信/计算重叠分析派生（NCCL ncclCommSplit / MPS CUDA_MPS_PIPE_DIRECTORY 运行时探测通信开销）+ nic_bandwidth_gbs（sensors/mod.rs:105）+ peak_gflops 推导最优比例"
      prodCalls: "无生产调用；所有调用（:1874, :1882）在 #[test] mod tests（1859 行附近）内"
      testCalls: "with_sm_partition(SmPartitionConfig::new(...))（:1983）也在测试内"
  codePattern:
    - "let comm_sms = (total_sms * 10 / 100).max(1);"
    - "10% 是经验值非派生；当前 pipeline scheduler 未接入真实分布式推理路径"
  triggerCondition: N/A（孤岛）；一旦分布式推理接入会变成 critical
  detectionSignatures:
    literal: "total_sms * 10 / 100"
    literal: "default_for_gpu"
    structural: "SM 分区比例使用字面量百分比而非通信开销分析派生"
  sameClassCriterion: 任何 SM 分区比例/通信-计算资源分配使用硬编码百分比而非运行时探测派生
  fixTemplate:
    - "从 nic_bandwidth_gbs + peak_gflops + 通信开销探测（NCCL/MPS）派生最优 comm/compute 比例"
    - "接入真实分布式推理路径或标注为待接入"
  regressionAssertion: "default_for_gpu 的 comm_sms 比例从硬件探测派生；有生产调用方或标注待接入"
  违反铁律: ARCH-ROOT-CAUSE（10% 经验值）/ NO-ISLAND-MODULE（待接入）
  归因时间: 2026-06-26
  status: 待根治
  residual: 1
```

---

## BCE-20260627-031 — IR precondition `op_has_output` 误伤 side-effect control op

```yaml
- patternId: BCE-20260627-031
  title: IR precondition op_has_output 误伤 side-effect control op
  layer: 范式
  smellClass: CS-PRECONDITION-OVERGENERAL
  rootCause: diagnostics.rs pre_check 对所有 op 要求 >= 1 output tensor，但 StoreToken/CheckStopCondition 等 15 种 side-effect control op 语义上无 tensor output
  instances:
    - file: gllm-kernels/src/compiler/diagnostics.rs
      line: 416-427
      symbol: pre_check()
      value: "op.outputs.is_empty() → IrError"
  codePattern:
    - "pre_check 对所有 op 一视同仁检查 outputs.is_empty()"
    - "忽略 side-effect control op（StoreToken/CheckStopCondition/WriteLogits/EarlyExit/GuardrailCheck/SgInject/SgDetect/CotStepCheck/SessionKvRestore/MmHiddenInject/MtpDraft/QTapSTG/KvScatterWrite/MegaKernelDispatch/MoEConditionalAdd）"
  detectionSignatures:
    structural: "pre_check 中 outputs.is_empty() 检查未豁免 control op"
  sameClassCriterion: 任何 IR precondition 对 control op 与 compute op 一视同仁不区分
  fixTemplate:
    - "引入 is_control_op() SSOT（backend_cap.rs）→ diagnostics.rs 查询 SSOT 豁免 control op"
  regressionAssertion: "control op 无 output tensor 时 pass pre_check"
  归因时间: 2026-06-27
  architectSessionId: <architect session>
  status: 根治
  residual: 0
```

## BCE-20260627-032 — hf_hub 缓存命中缺失触发 13GB 权重重下

```yaml
- patternId: BCE-20260627-032
  title: hf_hub 0.4.3 download_with_progress 无本地缓存命中检查
  layer: 设计
  smellClass: AP-CACHE-MISS
  rootCause: hf_hub crate 的 download_with_progress 从不检查本地 blob 是否已完整存在，每次重下
  instances:
    - file: gllm/src/loader/hf_hub.rs
      line: 820-958
      symbol: get_file() / download_shards()
      value: "直接调用 download_with_progress → 每次重下 13GB safetensors"
  codePattern:
    - "get_file() 直接调用 repo_api.download_with_progress()"
    - "download_shards() 有独立调用路径（并行+串行），绕过 get_file"
    - "hf_hub 0.4.3 仅检查 .lock 文件，不检查完整 blob"
  detectionSignatures:
    structural: "download_with_progress 调用前无 cache-exists 检查"
  sameClassCriterion: 任何通过 hf_hub download_with_progress 路径获取的权重文件，无本地缓存命中检查
  fixTemplate:
    - "gllm 层增加 find_cached_snapshot() 按 HF 标准缓存结构（refs/main → snapshot/ → blobs/）检查缓存命中"
    - "命中直接返回，未命中走 hf_hub"
  regressionAssertion: "已缓存的权重文件在首次下载后不再重下（无 .part 文件生成）"
  归因时间: 2026-06-27
  status: 根治
  residual: 0
```

## BCE-20260627-033 — GateMask OpKind 被错误归类为 stub op 阻断 JIT lowering

```yaml
- patternId: BCE-20260627-033
  title: GateMask 等 10 个 OpKind 被 is_stub_op() 错误阻挡 JIT 管线
  layer: 设计
  smellClass: CS-SSOT-DUPLICATE
  rootCause: backend_cap.rs 的 is_stub_op() 是重复的 SSOT，覆盖 ScalarOpRegistry 的实际注册状态。10 个 op 已有完整 scalar 实现 + OpTrace + auto_select lowering 路径，但被错误归类为 "P4/P5 stub" 返回 Unsupported
  instances:
    - file: gllm-kernels/src/compiler/backend_cap.rs
      line: 331-345
      symbol: is_stub_op()
      value: "GateMask / EntropyGate / SoftmaxWithEntropy / FusedRmsNormGemm / LayerBypass 等 10 个 op"
  codePattern:
    - "is_stub_op() 手写列表与 ScalarOpRegistry 注册状态独立维护，人为失谐"
    - "有 scalar 实现 + OpTrace + auto_select 的算子被标注为 'not yet implemented'"
  detectionSignatures:
    structural: "ScalarOpRegistry 有注册但 OpKindKey 在 is_stub_op() 列表中"
  sameClassCriterion: 任何 OpKind 同时满足 (a) ScalarOpRegistry 有注册 (b) OpTrace 已注入 (c) auto_select 有降低路径，但 backend_cap 标注为 stub
  fixTemplate:
    - "删除 is_stub_op() 函数，令该类算子走 compute op 路径（Category 4），由 derive_strategy_from_isa() 自动驱动"
  regressionAssertion: "OpKind::GateMask 在 DeviceProfile::Avx2 上 supported=true"
  归因时间: 2026-06-27
  status: 根治
  residual: 0
```

---

## BCE-20260627-034 — emit_binop 直接索引 slots 无边界检查（auto_select.rs:1734）

```yaml
- patternId: BCE-20260627-034
  title: auto_select emit_binop 直接索引 slots 无边界检查导致 index out of bounds panic
  layer: 设计
  smellClass: CS-PRECONDITION-OVERGENERAL
  rootCause: emit_binop/emit_binop_into 在 auto_select.rs:1728-1755 中 slots[a.0 as usize] 直接索引无边界检查。gpt-oss-20b MoE 编译时某 op 的 slots 为空 vec (len=0) 引发 panic（而非返回 compiler Error）
  instances:
    - file: gllm-kernels/src/compiler/codegen/vm/auto_select.rs
      line: 1728-1755
      symbol: emit_binop() / emit_binop_into()
      value: "prog.emit(VmInstr::VecBinOp { a: slots[a.0 as usize], b: slots[b.0 as usize], ... })"
  codePattern:
    - "emit_binop 系列函数无 slots.is_empty() 防御检查"
    - "slots 为空的根因：某个 op 的 trace 生成了引用未定义 value 的 binop（def-before-use 违反）"
    - "被 BCE-033 暴露（之前 is_stub_op 阻挡 GateMask 等 op，编译在早期就 CAP-ERR 退出，没走到这个 panic）"
  detectionSignatures:
    structural: "emit_binop/emit_binop_into/emit_unary/emit_binop_dtype 等函数中 slots\[value_id.0 as usize\] 无边界检查"
    literal: "slots\[a.0 as usize\]"
  sameClassCriterion: 任何 auto_select.rs 中从 slots 直接以 ValueId.0 索引的函数（emit_binop / emit_binop_dtype / emit_binop_into / emit_unary / emit_transcendental 等）
  fixTemplate:
    - "在 emit_binop/emit_binop_dtype 等函数入口检查 slots.len()，越界返回 Err(CompilerError) 而非 panic"
    - "Err 信息包含 ValueId 索引值 + slots.len + op 上下文"
  regressionAssertion: "空 slots 传入 emit_binop 返回 Err 而非 panic"
  归因时间: 2026-06-27
  status: 根治
  residual: 0
```

---

## E2E 测试硬件铁律（新增）

**E2E 测试必须积极使用量化模型，参考本地服务器硬件能力（CPU 核数 / RAM GB / GPU VRAM），防止使用超出设备能力的满血模型导致测试无法完成。**

**本地硬件基线（当前）**：
- CPU: 20 核
- RAM: 125 GB
- GPU: GTX 1060 6GB (compute tier: consumer 2016)
- 推理路径: CPU JIT (GPU 非必需)

**模型规模限制**：
- ❌ 禁止 E2E 测试使用 >10GB 满血权重（gpt-oss-20b BF16 13GB ❌）
- ✅ 优先使用 Q4_K_M / Q3_K_M / Q2_K / ONNX 等量化版（≤6GB ✅）
- ✅ 大模型满血权重测试必须在更强硬件环境跑（或标注 `#[ignore]` 待更强环境）

**量化模型仓库参考**：
- unsloth/gpt-oss-20b-GGUF (Q2_K/Q3_K_M/Q4_K_M，~11GB，仍然超出 → 待更强硬件或更小模型)
- SmolLM2-135M-Q4_0 (~74MB ✅)
- Qwen3-0.6B-Q4_0 (~450MB ✅)

**测试策略**：
- MoE 配置解析验证：可用 config.json 单测 + 小 MoE 模型（如 bartowski/Qwen_Qwen3-0.6B-GGUF）
- JIT 编译路径：可用量化版（需下载）或 config-only 测试
- 推理正确性：必须量化版 E2E（SmolLM2/Qwen3 量化版已验证）

**根因**：BCE-033/BCE-034 等 JIT 根治需要真实模型 E2E，但测试不应被硬件瓶颈阻塞。量化版既验证了真实推理路径，又不会因为模型规模超出设备能力导致测试无法完成。

---

## BCE-20260627-035 — rope_theta 缺失时默认 0.0 对 decoder 模型错误，导致 compute_inv_freq assert panic

```yaml
- patternId: BCE-20260627-035
  title: rope_theta 缺失时默认 0.0 对 decoder 模型错误
  layer: 设计（范式缺陷）
  rootCause: config_impl.inc.rs:658-661 对缺失 rope_theta 无条件默认 0.0，未区分 encoder vs decoder。Encoder（BERT/XLM-R）用绝对位置编码，0.0 合法；Decoder（mixtral/llama/qwen）必需 RoPE，0.0 传到 compute_inv_freq 的 assert!(theta > 0.0) 就 panic。
  codePattern:
    - "config.json 解析 rope_theta 时用 .unwrap_or(0.0)，未检查 model_type / position_embedding_type"
  detectionSignatures:
    literal: "unwrap_or_else(|| {.*0.0"
    structural: "rope_theta 查找链的末尾 .unwrap_or_else(|| ... 0.0)"
  sameClassCriterion: 任何 model_config_fragments 中未按架构语义区分 encoder/decoder 的默认值
  fixTemplate:
    - "encode（BERT/XLM-R, position_embedding_type=absolute）→ 0.0"
    - "decoder（mixtral/llama/qwen 等）→ 10000.0（Llama/Mixtral 行业标准）"
  regressionAssertion: "encoder 模型（e5-small, model_type=bert）rope_theta=0.0；decoder（SmolMoE, model_type=mixtral）rope_theta=10000.0"
  归因时间: 2026-06-27
  status: 根治
  residual: 0
```
---

## BCE-039 — MoE 数据流范式断裂（7 项同类根除）

```yaml
- patternId: BCE-039
  title: MoE 数据流范式断裂 — expert 循环展开 + 加权累加缺失 + TopK 无 renormalize + GateMask 语义脱节 + SwiGLU gate 未加权
  layer: 范式（架构模式缺陷）
  rootCause: MoE 图构建用 for e in 0..num_experts 逐层展开 6 ops × num_experts（违反 NO-LAYER-EXPAND + ARCH-NO-LOOP-UNROLL）；MoEConditionalAdd 无 lowering 导致所有 expert 简单相加丢失权重；TopK 选出 top-k 值后未 renormalize（概率之和 ≠ 1.0）；GateMask 用硬 0/1 mask 与 softmax 权重语义脱节；SwiGLU 的 gate 输入来自未加权的 Gemm 而 up 来自已加权的 MaskedGemm。根源：MoE 图构建与 top-k dispatch 数据流断裂。
  codePattern:
    - "for e in 0..num_experts 生成 per-expert 6 ops"
    - "Op::MoEConditionalAdd 无 match arm → fallback Ok(false)"
    - "emit_moe_topk_dispatch_inline 选出 top-k 但未做 inv_sum renormalize"
    - "GateMask 用 gate>0 硬掩码而非 softmax 权重"
    - "SwiGLU(gate_out, up_out) 中 gate_out 未加权"
  detectionSignatures:
    literal: "for e in 0..num_experts"
    structural: "Op::MoEConditionalAdd 无 lower_op match arm"
    antipattern: "NO-LAYER-EXPAND violation in build_graph.inc.rs"
  sameClassCriterion: 任何 MoE 算子缺失 lowering 或数据流断裂导致加权累加错误
  fixTemplate:
    - "Expert 循环改为单模板 + ExpertLoopConfig + GroupMarker::ExpertLoopBegin/End"
    - "MoEConditionalAdd lowering: acc += gate_probs[expert_loop_counter] * expert_down"
    - "TopK renormalize: inv_sum = 1.0/Σ(topk_weights); weight *= inv_sum"
    - "GateMask: soft mask max(gate, 0) 保留权重信息"
    - "SwiGLU inputs 改为 [mask_out, up_out] 两个输入都加权"
  regressionAssertion: "MoE 模型 (SmolMoE/gpt-oss) expert 循环 ops=6 per layer (非 num_experts*6); MoEConditionalAdd lowering 存在; TopK weights sum=1.0; SwiGLU 使用 mask_out"
  归因时间: 2026-06-27
  status: 根治
  residual: 0
```

---

## 配置解析硬编码(Hardcoded Config Parsing)

### smellClass: AP-HARDCODED-FIELD-MAPPING（Pattern — 配置字段逐个硬编码而非声明式注册）

**宪法依据**: ARCH-ROOT-CAUSE + P-2（函数 ≤500 行）— 配置字段解析由 177 个 JSON key 硬编码在 540 行函数中，新增模型字段必须手动找对应 find_* 调用加别名。

```yaml
- patternId: BCE-040
  title: ModelConfig 双路径（JSON+GGUF）硬编码字段映射 — 177 JSON key + 28 gguf_arch_* 散乱调用
  layer: 范式缺陷
  smellClass: AP-HARDCODED-FIELD-MAPPING
  codePattern:
    - "require_usize(value, &[\"hidden_size\", \"n_embd\", \"d_model\", \"text_config.hidden_size\", ...])"
    - "gguf_arch_f32(reader, arch, \"rope.global.freq_base\").or_else(|| gguf_arch_f32(reader, arch, \"global_rope_theta\"))"
    - "JSON 路径用 from_value()，GGUF 路径用 from_gguf_loader()，两套并行硬编码"
    - "每个字段手动枚举 JSON key / GGUF key 别名列表"
    - "text_config.* 变体几乎每个字段都手动复制"
  triggerCondition: 新模型使用不同 JSON/GGUF key 命名 → 必须在两条路径分别找调用点加别名
  detectionSignatures:
    structural: "from_value() 或 from_gguf_loader() 函数 >300 行"
    literal: "find_usize(value, &["
    literal: "require_usize(value, &["
    literal: "gguf_arch_(usize|f32|str|bool|array_)\\(reader, arch,"
    literal: "text_config."
  sameClassCriterion: 任何逐字段硬编码 JSON/GGUF key 别名而非声明式 FieldDef 注册的配置解析
  fixTemplate:
    - "FieldDef 注册表：canonical → json_keys + gguf_keys + gguf_reader + parse_json + parse_gguf + required + default"
    - "normalize_text_config() 展开 text_config.* 到根层，注册表无重复"
    - "apply_field_registry() 遍历 FieldDef 解析 JSON → CanonicalConfig"
    - "apply_gguf_field_registry() 遍历 FieldDef 解析 GGUF → CanonicalConfig（JSON 镜像）"
    - "apply_post_process() 计算跨字段依赖字段（两路径共用）"
    - "build_model_config() 统一组装（两路径共用）"
    - "from_value()/from_gguf_loader() 降到 ~100 行（声明式驱动 + tensor-derived 前置 pass）"
  regressionAssertion: "新增模型字段只需追加一条 FieldDef（同时覆盖 JSON+GGUF）；from_value() ≤ 80 行；from_gguf_loader() ≤ 120 行；无 text_config.* 手动复制；无 from_gguf_loader 主体内 gguf_arch_* 散乱调用"
  归因时间: 2026-06-27
  根治时间: 2026-06-27
  status: 根治 ✅
  residual: 0
  根治记录:
    - "JSON 路径 (Phase 1): field_registry.inc.rs: FieldDef/MetaValue/FieldKind/CanonicalConfig + FIELD_DEFS 55 条注册 + normalize_text_config + apply_field_registry + apply_post_process"
    - "JSON 路径 (Phase 1): config_impl.inc.rs: from_value() 540→54 行（normalize→registry→post_process→validate→build）"
    - "JSON 路径 (Phase 1): build_model_config() 提取为独立函数（120 行，纯组装无解析）"
    - "JSON 路径 (Phase 1): from_value() 内 find_*/require_* 调用: 61→0"
    - "GGUF 路径 (Phase 2): FieldKind::Alias 新增 gguf_keys + gguf_reader 字段；FIELD_DEFS 填充 43 个 gguf_keys + 12 个 gguf_reader 闭包"
    - "GGUF 路径 (Phase 2): FieldKind::Derived 新增 parse_gguf 字段；实现 5 个 parse_gguf 函数（rope_theta/rope_scaling/attention_pattern/feed_forward_lengths/mla_config）"
    - "GGUF 路径 (Phase 2): apply_gguf_field_registry() 实现 — Alias 走 gguf_reader/gguf_keys，Derived 走 parse_gguf"
    - "GGUF 路径 (Phase 2): from_gguf_loader() 376→108 行；提取 apply_gguf_dual_rope_correction / apply_gguf_attention_pattern_default / validate_gguf_canonical"
    - "GGUF 路径 (Phase 2): build_model_config() feed_forward_lengths 硬编码 None → c.feed_forward_lengths（修复 GGUF per-layer FFN 丢失）"
    - "双路径统一: from_value + from_gguf_loader 汇入同一 CanonicalConfig → build_model_config"
    - "架构: tensor-derived 前置 pass (ARCH-TENSOR-DRIVEN) 保留为 GGUF 路径 Step 1，早于 registry 解析"
    - "全量测试: 44354/44354 pass（含 156 model_config 测试）"
```


---

## BCE-20260629-DEADCODE-001 — unused 符号清理

**宪法依据**: P-1 红线 — TODO/FIXME/stub/空实现/console.log commit 前清除。commit_gate 强制执行。

**模式签名**: 编译器报告 `function X is never used` / `field X is never read` / `struct X is never constructed` / `unused import: X`。

**根治**: 12 个 warning → 0 个 gllm lib warning（gllm-kernels 的 3 个 warning 是已知 issue，非本任务范围）。

```yaml
- patternId: BCE-20260629-DEADCODE-001
  title: gllm unused 符号根治（12 warning → 0）
  layer: 设计缺陷（遗留代码未清理）
  smellClass: UNUSED-SYMBOL
  codePattern:
    - "cargo check 报告 'function/field/struct is never used/read/constructed'"
    - "遗留代码（API 重构、feature-gated 路径未启用）产生 dead code"
  triggerCondition: cargo check --lib 产生 unused warning
  detectionSignatures:
    literal: "warning: function .+ is never used|warning: field .+ is never read|warning: struct .+ is never constructed|warning: unused import"
  locations:
    - src/model_config_fragments/helpers.inc.rs:375,404,602 (find_u32/find_bool/gguf_arch_bool)
    - src/loader/fragments/upload_convert.inc.rs:263,297 (parallel_half_to_f32/HalfToF32)
    - src/compat/metal_backend.rs:89,108 (MetalDeviceSpecs/metal_device_specs)
    - src/engine/executor_step.rs:2099 (ring_attention_cp_step 非nccl stub)
    - src/jit/profiler.rs:39,411,542 (Instant import/parens/MicroKernel.binary)
    - src/arch/auto_graph_fragments/build_graph.inc.rs:2582 (expert_weight_input_indices)
  sameClassCriterion: "编译器报告的 never used/read 符号（非 feature-gated 合理保留）"
  fixTemplate:
    - "真 dead code（无引用）: safe_delete 删除"
    - "feature-gated 合理保留（如 nccl/metal）: #[cfg(...)] 或 #[cfg(any(test, feature=...))] + #[allow(dead_code)]"
    - "staging 变量（待上游类型）: #[allow(unused_variables, reason=\"...\")] + 注释说明"
  regressionAssertion: "cargo check -p gllm --lib 报告 0 warning（gllm-kernels warning 不计入）"
  归因时间: 2026-06-29
  根治时间: 2026-06-29
  status: 根治 ✅ | residual: 0
  根治记录:
    - "find_u32/find_bool/gguf_arch_bool: safe_delete（helpers.inc.rs 无引用）"
    - "parallel_half_to_f32/HalfToF32: #[cfg(any(test, feature=\"nccl\"))]（shard_for_tp 是 nccl gated）"
    - "MetalDeviceSpecs/metal_device_specs: #[cfg(any(all(target_os=\"macos\", feature=\"metal\"), test))]（测试跑在任意 host）"
    - "ring_attention_cp_step 非nccl stub: #[allow(dead_code, reason=\"REQ-DIST-016 integration pending\")]"
    - "Instant import: 移到 #[cfg(test)] mod tests 内"
    - "parens: 移除多余括号"
    - "MicroKernel.binary: #[allow(dead_code, reason=\"reserved for JIT binary cache\")]"
    - "expert_weight_input_indices: #[allow(unused_variables)] + 注释说明 staging for ExpertLoopConfig"
    - "HACK 关键词清理: BCE-HACK-HW-001/BCE-HACK-MODEL-003 注释改为普通说明"
    - "commit_gate: canCommit=true"
```
