# Thinking + Temperature + Speculative 开发清单

> **⚠️ ARCH-RUST-IS-CODEGEN 合规更新**: 本文档已适配 mega-kernel 架构。所有采样、thinking 追踪、KV 跳过、推测解码逻辑均编译进 JIT mega-kernel，不经过 Rust。
>
> 三大功能的端到端实现计划。依赖图 + 影响文件 + 验证方法。

## 现状

| 功能 | 架构 | 接线 | 问题 |
|------|------|------|------|
| Temperature 采样 | ✅ 缩放有 | ❌ 永远 argmax | 无 multinomial 随机采样 |
| Thinking 模式 | ⚠️ 后处理 | ❌ 无前向控制 | 无 budget、无 KV 跳过 |
| EESD 自推测 | ✅ 完整 | ❌ 未接入 | 孤岛模块 |
| EAGLE | ❌ 不存在 | ❌ | 需从零实现 |
| MTP | ❌ 不存在 | ❌ | 需从零实现 |
| Thinking KV 跳过 | ❌ 不存在 | ❌ | 思考 token 不应占 KV 空间 |

## 依赖图

```
T1 Multinomial 采样 (JIT codegen) ────────────────────────────┐
                                                               │
T2 Thinking Budget API (Client 层) ──→ T3 Thinking 标记 (JIT) ──→ T4 Thinking KV 跳过 (JIT)
                                                                  │
S1 EESD 接线 (JIT variant) ──→ S2 EAGLE Draft Head ──→ S3 MTP 多 Token 预测
```

---

## T1 — Multinomial 采样 (温度真正生效)

**目标**: temperature > 0 时按概率分布随机采样，temperature = 0 时 greedy argmax。

> **铁律**: 采样在 JIT mega-kernel 内部完成（详见 `08-EXECUTOR.md §4.1.1`、`04-API-DESIGN.md §3.1.2`）。采样参数通过 `MegaKernelFn` 栈参数传入。Rust 不参与采样。

**JIT codegen 实现**:
- greedy (temperature=0): argmax（VMAXPS + VPCMPEQD）
- temperature scaling: JIT 内部除法 + softmax renorm
- top_k: partial sort + mask + sample
- top_p: sort + cumsum + mask + sample
- 多项式采样: 硬件随机源 (CPU: RDRAND; GPU: CURAND)

**依赖**: 无
**验证**: 同 prompt + temperature=0.7 跑两次，输出应不同

---

## T2 — Thinking Budget API

**目标**: `GenerationBuilder.thinking_budget(max_tokens)` 控制思考 token 上限。

> **设计**: thinking_budget 作为 `MegaKernelFn` 的栈参数传入。JIT mega-kernel 内部维护 thinking_budget 计数器，在 generate 循环中检查。

**API**:
```rust
client.generate("Solve: 2+3*4")
    .thinking_budget(1024)  // 最多 1024 个思考 token
    .max_tokens(100)         // 最多 100 个输出 token
    .generate()
```

**参数传递**: thinking_budget 加入 `MegaKernelFn` ABI 栈参数（或通过 hook_ctx 传入）。

**依赖**: 无
**验证**: thinking_budget=0 → 无思考内容; thinking_budget=10 → 短思考

---

## T3 — Thinking Token 元数据标记

**目标**: 在 JIT mega-kernel 的 generate 循环中实时标记哪些 token 是思考 token。

> **设计**: thinking token 追踪编译进 JIT mega-kernel。状态机在 JIT 代码内部运行，标记结果写入 output_buffer 的 metadata 区域。

**JIT 内部状态机**:
```
Normal → (遇到 <thinking> 起始 token) → Thinking → (遇到 </thinking>) → Normal
```

每个 token 的 `is_thinking` 标记用于:
1. T4 的 KV 跳过决策（JIT 内部）
2. thinking_budget 计数（JIT 内部）
3. 输出 metadata 供 Client 读取

**依赖**: T2
**验证**: 验证 output metadata 中 thinking_token_count == 实际 `<thinking>` 标签内的 token 数

---

## T4 — Thinking KV Cache 跳过

**目标**: 思考 token 不写入 KV cache。下一轮对话时，KV cache 中不包含思考过程。

> **设计**: KV 跳过逻辑编译进 JIT mega-kernel。JIT 代码在 thinking 状态下跳过 KV cache write 操作，但仍然执行当前 step 的 attention 计算（使用临时 buffer）。

**JIT 内部逻辑**:
```
每个 token 生成后:
  if is_thinking:
      执行前向传播 → 计算 attention → 生成 next token
      但跳过 KV cache write（K/V 不持久化）
      当前 step attention 使用临时 KV buffer
  else:
      正常流程：执行前向 + 写入 KV cache
```

**多轮对话**:
```
Turn 1: [prompt] + [thinking: 500 tokens] + [answer: 50 tokens]
KV cache 保留: [prompt] + [answer: 50 tokens] (跳过 500 thinking tokens)
Turn 2: KV cache hit → prefill 只需处理 Turn 2 的 prompt
```

**依赖**: T3 (需要 is_thinking 标记)
**验证**:
1. 单轮: thinking 模型生成后 KV cache 中不含 thinking 部分
2. 多轮: 第二轮 prefill 不包含第一轮的 thinking tokens
3. 正确性: 跳过 thinking KV 后输出质量不下降

---

## S1 — EESD 接线 (现有自推测架构接入)

**目标**: 将推测解码编译为 JIT mega-kernel 的 variant。

> **设计**: 推测解码通过 VariantRegistry 管理（详见 `06-RUNTIME.md §13`）。Draft variant 使用浅层前向，Verify variant 使用全量模型。两者都编译为独立的 JIT mega-kernel，dispatch-time 选择。

**Variant 编译**:
```
Draft variant: embed → 前 L/3 层 → draft lm_head → K 个候选 token
Verify variant: embed → 全 N 层 → lm_head → 一次验证所有候选
```

**执行流程** (dispatch-time 选择 variant):
1. `build_batch()` 检查 spec_phase → derive VariantKey
2. Draft mega-kernel 生成 K 个候选
3. Verify mega-kernel 一次性验证所有候选
4. 接受的 token commit 到 KV cache

**依赖**: 无
**验证**: 启用 EESD 后生成结果与标准解码一致，throughput 提升

---

## S2 — EAGLE Draft Head

**目标**: 使用训练好的 EAGLE draft head 替代浅层变体做 draft。

**EAGLE 原理**:
- 在目标模型最后一层 hidden state 之上添加一个轻量 draft head (1-2 层 transformer)
- Draft head 接收 `hidden_state + embedding` 作为输入
- 输出下一个 token 的 logits，比浅层变体更准确
- EAGLE-2: 动态推测树，根据置信度决定分支宽度

> **设计**: EAGLE head 编译为独立的 JIT mega-kernel variant。Draft head 的权重 pack 进独立的 weight_blob。dispatch-time 选择 EAGLE variant。

**依赖**: S1 (EESD 接线)
**验证**: EAGLE vs 标准解码输出一致，throughput 提升 2-3x

---

## S3 — MTP (Multi-Token Prediction)

**目标**: 模型一次前向传播预测多个未来 token (Qwen3/DeepSeek V3 MTP head)。

**MTP 原理**:
- 某些模型在训练时使用 MTP loss，同时预测 next-1/next-2/.../next-K token
- 推理时可利用 MTP head 一次前向生成 K 个 token 的 logits
- 相当于内置的 draft model，不需要额外权重

> **设计**: MTP head 编译为 JIT mega-kernel 的扩展。MTP projections 的权重 pack 进 weight_blob，偏移 bake 进 kernel 代码。一次 mega-kernel 调用输出 K 个候选 token。

**依赖**: S1 (EESD 接线)
**验证**: MTP 模型 (DeepSeek V3) 加速效果

---

## 优先级 & 时间线

| 阶段 | 任务 | 依赖 | 复杂度 | 说明 |
|------|------|------|--------|------|
| **Ⅰ** | T1 Multinomial 采样 | 无 | 低 | JIT codegen 采样 |
| **Ⅰ** | T2 Thinking Budget | 无 | 低 | API 扩展 + 栈参数 |
| **Ⅱ** | T3 Thinking Token 标记 | T2 | 中 | JIT 内部状态机 |
| **Ⅱ** | S1 EESD 接线 | 无 | 高 | Variant 编译 + dispatch |
| **Ⅲ** | T4 Thinking KV 跳过 | T3 | 高 | JIT 内部 KV skip |
| **Ⅲ** | S2 EAGLE | S1 | 高 | Draft head variant |
| **Ⅳ** | S3 MTP | S1 | 中 | MTP head variant |

**里程碑**:
- Ⅰ 完成 → 温度采样真正工作 + thinking 可控
- Ⅱ 完成 → 思考 token 可追踪 + EESD 推测解码可用
- Ⅲ 完成 → 思考不占 KV + EAGLE 加速 2-3x
- Ⅳ 完成 → MTP 原生加速 (DeepSeek V3 / Qwen3)
