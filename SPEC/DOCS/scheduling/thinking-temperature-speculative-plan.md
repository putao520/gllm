# Thinking + Temperature + Speculative 开发清单

> 三大功能的端到端实现计划。依赖图 + 影响文件 + 验证方法。

## 现状

| 功能 | 架构 | 接线 | 问题 |
|------|------|------|------|
| Temperature 采样 | ✅ 缩放有 | ❌ 永远 argmax | 无 multinomial 随机采样 |
| Thinking 模式 | ⚠️ 后处理 | ❌ 无前向控制 | 无 budget、无 KV 跳过 |
| EESD 自推测 | ✅ 完整 | ❌ step() 未调用 | 孤岛模块 |
| EAGLE | ❌ 不存在 | ❌ | 需从零实现 |
| MTP | ❌ 不存在 | ❌ | 需从零实现 |
| Thinking KV 跳过 | ❌ 不存在 | ❌ | 思考 token 不应占 KV 空间 |

## 依赖图

```
T1 Multinomial 采样 ──────────────────────────────────────┐
                                                           │
T2 Thinking Budget API ──→ T3 Thinking Token 标记 ──→ T4 Thinking KV 跳过
                                                      │
S1 EESD 接线 ──→ S2 EAGLE Draft Head ──→ S3 MTP 多 Token 预测
```

---

## T1 — Multinomial 采样 (温度真正生效)

**目标**: temperature > 0 时按概率分布随机采样，temperature = 0 时 greedy argmax。

**文件**:
| 文件 | 变更 |
|------|------|
| `src/compat/cpu_backend.rs` | `sample_from_tensor()`: argmax → multinomial 分支 |

**实现**:
```rust
if temperature == 0.0 {
    // Greedy: argmax (现有逻辑)
    probs.iter().enumerate().max_by(...)
} else {
    // Stochastic: multinomial sampling
    let mut rng = thread_rng();
    let dist = WeightedIndex::new(&probs).unwrap();
    indices[dist.sample(&mut rng)] as u32
}
```

**依赖**: 无
**验证**: 同 prompt + temperature=0.7 跑两次，输出应不同

---

## T2 — Thinking Budget API

**目标**: `GenerationBuilder.thinking_budget(max_tokens)` 控制思考 token 上限。

**文件**:
| 文件 | 变更 |
|------|------|
| `src/generation.rs` | `GenerationBuilder` 新增 `thinking_budget: Option<usize>` |
| `src/client.rs` | `execute_generation()` 传递 thinking_budget 到 executor |
| `src/engine/executor.rs` | `generate()` / `step()` 接受 thinking_budget 参数 |

**API**:
```rust
client.generate("Solve: 2+3*4")
    .thinking_budget(1024)  // 最多 1024 个思考 token
    .max_tokens(100)         // 最多 100 个输出 token
    .generate()
```

**依赖**: 无
**验证**: thinking_budget=0 → 无思考内容; thinking_budget=10 → 短思考

---

## T3 — Thinking Token 元数据标记

**目标**: 在生成循环中实时标记哪些 token 是思考 token，而非事后文本解析。

**核心设计**: 状态机追踪 `<thinking>` / `</thinking>` 标签的 token 序列。

**文件**:
| 文件 | 变更 |
|------|------|
| `src/engine/executor.rs` | 新增 `ThinkingTracker` 状态机 |
| `src/generation.rs` | `GenerationResponse` 新增 `thinking_token_count: usize` |

**状态机**:
```
Normal → (遇到 <thinking> 起始 token) → Thinking → (遇到 </thinking>) → Normal
```

每个 token 标记 `is_thinking: bool`，用于:
1. T4 的 KV 跳过决策
2. thinking_budget 计数
3. 流式输出时区分思考/回答

**依赖**: T2
**验证**: 验证 thinking_token_count == 实际 `<thinking>` 标签内的 token 数

---

## T4 — Thinking KV Cache 跳过

**目标**: 思考 token 不写入 KV cache。下一轮对话时，KV cache 中不包含思考过程。

**核心原理**: 思考是模型的内部推理过程，对后续对话无信息价值。跳过思考 token 的 KV 可以:
- 节省 KV cache 内存 (思考 token 可能占 50%+ 的生成量)
- 加速后续 prompt 的 prefill (更短的 past KV 序列)
- 避免思考内容"污染"后续注意力分布

**设计**:
```
生成阶段:
  Normal token → 写入 KV cache (正常流程)
  Thinking token → 执行前向传播但不写入 KV cache
                   (K/V 投影结果用于当前 step 的注意力计算，但不持久化)

多轮对话:
  Turn 1: [prompt] + [thinking: 500 tokens] + [answer: 50 tokens]
  KV cache 保留: [prompt] + [answer: 50 tokens] (跳过 500 thinking tokens)
  Turn 2: KV cache hit → prefill 只需处理 Turn 2 的 prompt
```

**文件**:
| 文件 | 变更 |
|------|------|
| `src/compat/cpu_backend.rs` | KV 写入时检查 `is_thinking` 标记 |
| `src/kv_cache/mod.rs` | `KvPageHeader` 或单独的 `thinking_mask: Vec<bool>` |
| `src/graph/executor.rs` | `run_with_kv_cache` 接受 thinking mask |
| `src/scheduler/paged_scheduler.rs` | 页分配时跳过 thinking 页 |

**实现核心** (cpu_backend.rs KV 写入):
```rust
fn update_kv_cache(layer: usize, pos: usize, k: &[f32], v: &[f32], is_thinking: bool) {
    if is_thinking {
        return; // 不持久化 thinking token 的 KV
    }
    // 正常 KV 写入...
}
```

**注意**: 当前 step 的注意力计算仍然需要 thinking token 的 KV (因为模型需要 attend 到自己的思考过程)。
解决方案: 使用临时 KV buffer 存储 thinking token，在 thinking 结束后丢弃。

**依赖**: T3 (需要 is_thinking 标记)
**验证**:
1. 单轮: thinking 模型生成后 KV cache 中不含 thinking 部分
2. 多轮: 第二轮 prefill 不包含第一轮的 thinking tokens
3. 正确性: 跳过 thinking KV 后输出质量不下降

---

## S1 — EESD 接线 (现有自推测架构接入)

**目标**: 将已有的 `speculative/` 模块接入 `step()` 生成循环。

**文件**:
| 文件 | 变更 |
|------|------|
| `src/engine/executor.rs` | `step()` 新增推测解码分支 |
| `src/compat/decoder_forward.rs` | 支持浅层前向 (仅前 L/3 层) |

**step() 改造**:
```rust
fn step(&mut self, ...) -> Result<StepOutput> {
    if self.spec_state.should_speculate() {
        // Draft: 浅层变体生成 K 个候选 token
        let drafts = self.draft_forward(K)?;
        // Verify: 全量模型一次性验证所有候选
        let accepted = self.verify_batch(&drafts)?;
        // Commit: 接受的 token 提交 KV cache
        self.commit_accepted(&accepted)?;
        Ok(StepOutput::Speculative { accepted })
    } else {
        // Standard: 标准单 token 解码
        self.standard_step()
    }
}
```

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

**文件**:
| 文件 | 变更 |
|------|------|
| `src/speculative/eagle.rs` | **新文件**: EAGLE draft head 前向传播 |
| `src/speculative/mod.rs` | 新增 `pub mod eagle;` |
| `src/speculative/engine.rs` | `SpecDecodingMode` 新增 `Eagle` 变体 |
| `src/weight_loader.rs` | 加载 EAGLE head 权重 (`model.eagle_head.*`) |
| `src/arch/templates/*.yaml` | 支持 EAGLE 权重的模板 |

**EAGLE head 结构**:
```rust
pub struct EagleHead {
    /// 融合层: hidden_state + embedding → draft hidden
    pub fc: LinearWeight,           // [hidden_size * 2, hidden_size]
    /// 1-2 层轻量 transformer
    pub layers: Vec<DraftLayer>,
    /// 共享 lm_head (复用主模型的 lm_head)
    pub share_lm_head: bool,
}

pub fn eagle_draft(
    hidden_state: &[f32],  // 主模型最后一层输出
    prev_embedding: &[f32], // 前一个 token 的 embedding
    head: &EagleHead,
) -> Vec<f32> { // logits
    let input = concat(hidden_state, prev_embedding);
    let h = fc_forward(&input, &head.fc);
    // 轻量 transformer 层
    for layer in &head.layers {
        h = layer.forward(&h);
    }
    // lm_head
    matmul(&h, &lm_head_weight)
}
```

**EAGLE-2 动态树**:
```rust
/// 根据 draft logits 置信度动态决定树宽度
fn build_eagle_tree(draft_logits: &[f32], confidence_threshold: f32) -> SpecTree {
    let probs = softmax(draft_logits);
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    if max_prob > confidence_threshold {
        // 高置信: 单分支深探
        SpecTree::chain(top_k(probs, 1))
    } else {
        // 低置信: 多分支宽探
        SpecTree::fan_out(top_k(probs, 3))
    }
}
```

**依赖**: S1 (EESD 接线)
**验证**: EAGLE vs 标准解码输出一致，throughput 提升 2-3x

---

## S3 — MTP (Multi-Token Prediction)

**目标**: 模型一次前向传播预测多个未来 token (Qwen3/DeepSeek V3 MTP head)。

**MTP 原理**:
- 某些模型 (如 DeepSeek V3) 在训练时使用 MTP loss，同时预测 next-1/next-2/.../next-K token
- 推理时可利用 MTP head 一次前向生成 K 个 token 的 logits
- 相当于内置的 draft model，不需要额外权重
- 验证阶段: 标准前向重新验证 K 个候选

**文件**:
| 文件 | 变更 |
|------|------|
| `src/speculative/mtp.rs` | **新文件**: MTP head 前向 + 多 token 提取 |
| `src/speculative/mod.rs` | 新增 `pub mod mtp;` |
| `src/speculative/engine.rs` | `SpecDecodingMode` 新增 `MTP` 变体 |
| `src/model_config.rs` | 解析 `num_mtp_heads` / `mtp_depth` 配置 |

**MTP Head 结构** (DeepSeek V3 style):
```rust
pub struct MtpHead {
    /// MTP 预测深度 (通常 2-4)
    pub depth: usize,
    /// 每层 MTP 投影: hidden → next-k logits
    pub projections: Vec<LinearWeight>,  // depth 个 [hidden, vocab]
}

pub fn mtp_draft(
    hidden_state: &[f32],  // 最后一层 hidden
    head: &MtpHead,
) -> Vec<Vec<f32>> { // [depth][vocab_size] — K 个 token 的 logits
    (0..head.depth)
        .map(|k| matmul(hidden_state, &head.projections[k]))
        .collect()
}
```

**依赖**: S1 (EESD 接线)
**验证**: MTP 模型 (DeepSeek V3) 加速效果

---

## 优先级 & 时间线

| 阶段 | 任务 | 依赖 | 复杂度 | 说明 |
|------|------|------|--------|------|
| **Ⅰ** | T1 Multinomial 采样 | 无 | 低 | 最基础，立即修 |
| **Ⅰ** | T2 Thinking Budget | 无 | 低 | API 扩展 |
| **Ⅱ** | T3 Thinking Token 标记 | T2 | 中 | 状态机 |
| **Ⅱ** | S1 EESD 接线 | 无 | 高 | 核心推测循环 |
| **Ⅲ** | T4 Thinking KV 跳过 | T3 | 高 | 临时 KV + 丢弃 |
| **Ⅲ** | S2 EAGLE | S1 | 高 | draft head + 动态树 |
| **Ⅳ** | S3 MTP | S1 | 中 | 多 token 预测 head |

**里程碑**:
- Ⅰ 完成 → 温度采样真正工作 + thinking 可控
- Ⅱ 完成 → 思考 token 可追踪 + EESD 推测解码可用
- Ⅲ 完成 → 思考不占 KV + EAGLE 加速 2-3x
- Ⅳ 完成 → MTP 原生加速 (DeepSeek V3 / Qwen3)
