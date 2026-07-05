# RMSNorm / final norm 实现链路资料库（C-9，堵 AI 幻觉）

> 来源：gllm-kernels scalar-ops/norms.rs + defaults.inc.rs + norm_softmax_emit.rs + lower_op.inc.rs + x86_lower/lower_instr_dispatch.inc.rs + gllm build_graph.inc.rs（Explore 调研，源码事实）
> 建库触发：8 轮 SmolLM2 logits 发散诊断，RMSNorm（final norm）反复被怀疑，需确定性记录其实现正确性
> 最后验证：2026-07-05

## 核心机制（源码确认）

### scalar 参考实现（scalar-ops/src/norms.rs:8-29）
```rust
pub unsafe extern "C" fn scalar_rms_norm(x, weight, out, n, eps) {
    let mut sum_sq = 0.0_f32;
    for i in 0..n { sum_sq += x[i]*x[i]; }
    let scale = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n { out[i] = x[i] * scale * weight[i]; }
}
```
**公式**：`out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)` — 标准 RMSNorm，正确。

### eps 来源（SmolLM2 = 1e-5，非硬编码）
- config.json `rms_norm_eps: 1e-05` → field_registry.inc.rs:600 alias `rms_norm_eps → layer_norm_epsilon`
- types.inc.rs:168 `norm_eps = layer_norm_epsilon.unwrap_or(1e-12)`（fallback 是 1e-12，但 SmolLM2 显式提供 1e-5）
- build_graph.inc.rs:118 `let eps = config.norm_eps` → NormSpec.eps
- **lower_op.inc.rs:1616-1626 用 spec.eps 覆盖 registry trace 的 Const**（注释明确"禁止用 registry 默认值"）

### OpKind 注册（defaults.inc.rs:484-523）
- OpKindKey = `RmsNorm`，fn_ptr = `scalar_rms_norm`
- ComputePattern = `NormLike`（三阶段：reduce/finalize/transform）
- registry trace 的 `Const(1e-5)` 仅 symexec fallback 占位，codegen 用 spec.eps 覆盖

### JIT lowering 三阶段（norm_softmax_emit.rs:141-248 emit_normlike_one_group）
1. **Phase 1 reduce**：VecLoad(x) → x*x → Accumulate → HReduce(Sum) → sum_sq
   - reduction 维度 = **feature_dim（hidden=576）**，非 seq 维度 ✅
   - 外层 emit_loop 遍历 seq，内层 reduce 遍历 feature_dim
2. **Phase 2 finalize**：sum_sq/n → +eps → Rsqrt → scale
   - `Div(sum_sq, n) → Add(eps) → Rsqrt`（defaults.inc.rs:505-512）
   - rsqrt 用硬件近似指令：AVX-512 `vrsqrt14ps` / AVX2 `vrsqrtps`（11-bit 精度）
3. **Phase 3 transform**：VecLoad(x) → *scale → VecLoad(weight) → *weight → VecStore
   - weight 在 transform 阶段 per-element mul，非 GEMM epilogue
   - weight dtype 独立（weight_step_bytes = lanes * weight_dtype.elem_bytes()，BCE-20260703-NORM-MIXED-PRECISION-STEPBYTES 已修）

### dtype 传播
- activation dtype = `dtype`（从 NormSpec.dtype = act_dt）
- weight dtype = `weight_dtype`（从 weight tensor 独立获取）
- 累加 dtype = `dtype.accumulator_dtype()`（BF16→F32, F32→F32）
- BF16 weight widen：VecLoad weight 用 BF16，widen 到 F32 计算

## final norm vs 每层 norm（同一套代码）

**代码层面无 final_norm 专用分支**，全部走 `lower_norm_v2` 同一 lowering 路径。区别仅在：
- op label：`"final_norm"` vs `"layer.input_norm"` / `"layer.post_norm"`
- weight tensor：`final_norm`（模型级）vs `layer.{N}.input_norm`（每层独立）
- eps 相同（都从 config.norm_eps）、dtype 相同（都从 act_dt）

## SmolLM2 final norm 在 graph 里

- op label = `"final_norm"`（build_graph.inc.rs:2240）
- op 类型 = `Op::RmsNorm(NormSpec { feature_dim: hidden=576, eps: 1e-5, dtype: act_dt, has_weight: true })`
- weight tensor 名 = `"final_norm"`（build_graph.inc.rs:2237）
- 对应 HF 权重 `model.norm.weight`（weight_names.rs:236 `decoder_final_norm_aliases` 映射）

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| eps 硬编码 1e-5 | 从 config 经 spec.eps 覆盖（lower_op.inc.rs:1616-1626），SmolLM2=1e-5 |
| reduction 维度是 seq | 是 feature_dim（hidden=576），外层 seq 内层 feature |
| sqrt 算 sqrt(x^2) | 是 rsqrt(mean(x^2)+eps)，finalize trace 确证 |
| BF16 weight widen 丢符号 | vpmovzxwd+vpslld 零扩展保符号（dtype-propagation.md） |
| weight 乘法是 GEMM epilogue | 是 NormLike transform 阶段 per-element mul |
| final_norm 有专用分支 | 与每层 norm 同套代码（lower_norm_v2），仅 label/weight 不同 |
| RESIDUAL_NORM_EPSILON=1e-6 影响 RMSNorm | 仅用于 telemetry（telemetry_emit.rs），不参与 RMSNorm 计算 |

## 已修复的相关 BUG（历史，非当前根因）

- BCE-20260703-AVX512-BROADCAST-NAN（59629b4d）：AVX-512 broadcast 半初始化 ZMM 高 lanes → logits NaN。已修
- BCE-20260703-NORM-MIXED-PRECISION-STEPBYTES（2d2e5dbd/e24a2e49）：NormLike weight VecLoad 复用 input byte_off → BF16 只读一半。已修（但 SmolLM2 是 F32 模型不受影响——**注：此处的 F32 指激活，权重仍 BF16**）
- BCE-20260703-AVX512-HALF-LANES（1d2da241）：7处 reduction/scan 按 16-lane 步长跳但只处理低 8 lanes → argmax=6。已修

## 潜在精度风险（非当前根因）

- **rsqrt 无 Newton 迭代**：vrsqrtps 11-bit 精度，对 RMSNorm 通常足够，但极小 eps（1e-12）或极大激活值可能累积误差。SmolLM2 eps=1e-5 hidden=576 风险低
- **eps 覆盖逻辑过宽**（lower_op.inc.rs:1619-1623）：替换 finalize 中所有 TraceOp::Const。当前 RmsNorm finalize 只有 eps 一个 Const 安全；LayerNorm 走独立路径不受影响

## 排除结论

RMSNorm（final norm）实现**正确**：
- eps 正确（1e-5 从 config）
- reduction 维度正确（hidden）
- sqrt 正确（rsqrt(mean(x^2)+eps)）
- BF16 weight widen 正确（独立 dtype/步长）
- weight 乘法正确（transform 阶段）

**非 SmolLM2 logits 发散根因**。真因见 `kv-cache-dtype-dual-layer.md`（KV cache dtype 双地层裂开）。

## 关键代码位置

- `gllm-kernels/scalar-ops/src/norms.rs:8-29` — scalar 参考实现
- `gllm-kernels/src/compiler/registry_fragments/defaults.inc.rs:484-523` — RmsNorm 注册
- `gllm-kernels/src/compiler/codegen/vm/norm_softmax_emit.rs:39-248` — JIT 三阶段发射
- `gllm-kernels/src/compiler/codegen/vm/plan_lower/lower_op.inc.rs:52-69, 1583-1694` — lowering + eps 覆盖
- `gllm-kernels/src/compiler/codegen/vm/x86_lower/lower_instr_dispatch.inc.rs:1699-1761, 1450, 1477` — HReduce + Rsqrt
- `gllm/src/arch/auto_graph_fragments/build_graph.inc.rs:118, 2237-2244` — final_norm graph 构建
- `gllm/src/weight_names.rs:236-247` — model.norm.weight 映射

## 与其他资料库关系

- `dtype-propagation.md`：BF16 widen/narrow（本库是其 NormLike 应用）
- `kv-cache-dtype-dual-layer.md`：SmolLM2 logits 发散真因（本库排除 RMSNorm 嫌疑）
- 本文件：RMSNorm/final norm 实现正确性证据
