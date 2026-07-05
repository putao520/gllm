# derive_compute_dtype 精度预设违宪（C-9 + 宪法 -1，堵 AI 幻觉）

> 来源：gllm-kernels dtype_chain.rs:195-210 + architect 裁决（sessionId 401396fe）+ 用户宪法 -1 指令
> 建库触发：运行时钉死 SmolLM2 compute_dtype=F32（config 是 BF16）→ 追出 derive_compute_dtype 硬编码 BF16→F32
> 重大修正：architect 裁决指出 "blob 保留 BF16" 但这本身预设 BF16 立场 → 用户宪法 -1：禁止任何精度预设
> 最后验证：2026-07-06

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

## 根治方向（宪法 -1 合规，非之前的 5 项联动）

**问题**：derive_compute_dtype 返回单一 compute_dtype 无法表达混合精度，且硬编码降级。

**宪法 -1 合规方案**（待 architect 详细设计）：
- 移除 `derive_compute_dtype` 的精度预设 match arm
- compute_dtype 概念应**逐张量**而非整模型（每张量按其 storage_dtype JIT 特化）
- 累加器 dtype 由算子 + 硬件 + 配置决定，不由全局 compute_dtype 硬编码
- KV cache dtype 跟随 K/V projection 输出张量实际 dtype（而非全局 compute_dtype）

**注意**：这是范式级重构，不能简单改 match arm（会引入 stride 不一致）。需 architect 整体设计。

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
