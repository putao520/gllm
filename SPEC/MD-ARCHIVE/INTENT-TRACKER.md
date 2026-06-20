# Signal-Aware Intent Tracker — 自定义分类模型集成协议

> **SSOT**: 本文档定义 gllm 集成 SignalIntentTracker 自定义分类模型的唯一真源。
>
> **关联需求**: `SPEC/01-REQUIREMENTS.md §15` REQ-INTENT-001..003, `SPEC/SUPPORTED_MODELS.md §4` REQ-MODEL-006
>
> **关联上层 API**: `SPEC/04-API-DESIGN.md §3.10` Intent Recall SDK
>
> **关联 JIT 管线**: `SPEC/01-JIT-PIPELINE.md`, `SPEC/04-OPERATORS.md`, `SPEC/03-GRAPH-IR.md`
>
> **遵守铁律**: ARCH-FULL-JIT、ARCH-CPU-GPU-UNIFIED、NO_SILENT_FALLBACK、NO_ISLAND_MODULE、DRY

---

## §1 动机

### 1.1 为什么需要自定义分类模型

gllm 已支持标准 HuggingFace classifier 模型（BERT + Linear head）。但实际业务中存在**非标准分类架构**：

- 自定义 attention 机制（非标准 transformer 层）
- 多信号融合（文本 embedding + 结构化信号）
- 多任务输出（task_type + difficulty 同时预测）
- 固定 KV cache + 循环覆盖（状态追踪，非序列生成）

SignalIntentTracker 就是这类模型：3.65M 参数，接收预编码 768 维 embedding 序列，输出 3 类任务类型 + 4 类难度。gllm 应能以统一的 JIT 管线高效执行此类模型。

### 1.2 与现有 Intent Recall SDK 的关系

| 维度 | Intent Recall SDK | Intent Tracker (本文档) |
|------|-------------------|------------------------|
| 用途 | 从 LLM 中间层抽取 hidden | 独立分类模型推理 |
| 模型 | 复用已有 LLM 权重 | 独立的 3.65M 模型 |
| 输入 | 原始文本 | 预编码 embedding 序列 |
| 输出 | 中间层 embedding 向量 | 分类 logits (task + difficulty) |
| 关系 | 可作为 Tracker 的上游 | 消费 encoder 输出 |

**协同工作流**: Intent Recall SDK 提供 encoder，Intent Tracker 提供 classifier。两者可组合为端到端管线。

---

## §2 模型架构

### 2.1 信号流

```
预编码 embeddings (B, T, 768)
  + role_embedding (B, T, 768)
  → info_weight = MLP(e_t) → (B, T) 连续值 0-1
  → Q = W_q(e), K = W_k(e), V = W_v(e) * info_weight
  → Multi-head Attention (4×192, causal, recency bias)
  → Dual-path context (gate * last + (1-gate) * aggregated)
  → classifier_input = [e_last, context, scalar_features, signal_emb]
  → task_logits (B, 3), difficulty_logits (B, 4)
```

### 2.2 参数清单

| 模块 | 权重名 | 形状 | 参数量 |
|------|--------|------|--------|
| InfoWeightEstimator | `info_estimator.net.*` | 768→512→128→1 | ~460K |
| Role Embedding | `role_emb.weight` | (2, 768) | 1.5K |
| W_q | `W_q.weight/bias` | (768, 768) | ~590K |
| W_k | `W_k.weight/bias` | (768, 768) | ~590K |
| W_v | `W_v.weight/bias` | (768, 768) | ~590K |
| Per-head LayerNorm | `per_head_norm.*` | (192,) | 0.8K |
| Context LayerNorm | `context_norm.*` | (768,) | 1.5K |
| Recency scale | `recency_scale` | scalar | 1 |
| Context gate | `context_gate` | scalar | 1 |
| Signal Encoder | `signal_encoder.*` | 11→128→128→64 | ~17K |
| Task Classifier | `task_classifier.*` | 1602→384→192→3 | ~680K |
| Difficulty Classifier | `difficulty_classifier.*` | 1602→384→192→4 | ~680K |
| **Total** | | | **~3.65M** |

### 2.3 自定义算子

以下算子超出标准 transformer 操作，需要注册到 gllm JIT 管线：

| 算子 | 描述 | 对应 OpKind |
|------|------|-------------|
| InfoWeightModulation | V = W_v(e) * sigmoid(MLP(e)) | 复合：MatMul + ElementWise(Sigmoid) + Mul |
| RecencyBias | scores -= sigmoid(scale) * age_matrix | 复合：ElementWise(Sigmoid) + BroadcastSub |
| DualPathContext | gate * context_last + (1-gate) * weighted_mean | 复合：ElementWise(Sigmoid) + Reduce(Mean) + LinearCombination |
| CausalAttention | 标准 causal scaled dot-product (4-head) | 复用 Attention OpKind + reshape |

**关键**: 无需新增原子算子。所有操作由已有算子组合实现，gllm JIT 融合管线负责优化。

---

## §3 集成设计

### 3.1 ModelArchKey 注册

```rust
// src/graph/model_arch.rs
ModelArchKey {
    family: "signal-intent-tracker",
    variant: "v9",
    backend: Backend::Cpu, // 或 Gpu
}
```

### 3.2 GraphType

```rust
// 新增 GraphType 变体
enum GraphType {
    // ... 已有变体 ...
    SignalIntentTracker {
        num_heads: usize,        // 4
        head_dim: usize,         // 192
        num_tasks: usize,        // 3
        num_difficulties: usize, // 4
        signal_dim: usize,       // 11
        signal_hidden_dim: usize,// 64
        max_seq_len: usize,      // 32
    },
}
```

### 3.3 图构建器

```rust
fn build_signal_intent_tracker_graph(
    graph: &mut CompilerGraph,
    weights: &TensorMap,
    config: &SignalIntentTrackerConfig,
) -> GraphType {
    // 1. 输入节点: embeddings (B, T, 768), roles (B, T), signals (B, 11),
    //    seq_lens (B,), context_turns (B,)

    // 2. Role embedding: embeddings = embeddings + role_emb[roles]

    // 3. InfoWeight: info_w = sigmoid(MLP(embeddings)) → (B, T)

    // 4. Q/K/V projections + info_weight modulation on V

    // 5. Multi-head reshape: (B, T, 768) → (B, 4, T, 192)

    // 6. Causal scaled dot-product attention + recency bias

    // 7. Per-head LayerNorm + concat

    // 8. Dual-path context aggregation

    // 9. Signal encoding + feature concatenation

    // 10. Task + difficulty classifier heads
}
```

### 3.4 输入格式

与标准 classifier 不同，输入不是 token id 而是 **预编码 embedding**：

```rust
struct TrackerInput {
    embeddings: Tensor,    // (B, T, 768) bf16/f32
    roles: Tensor,         // (B, T) i64, 0=user 1=assistant
    signals: Tensor,       // (B, 11) bf16/f32, 10 bool + context_dependence
    seq_lens: Tensor,      // (B,) i64
    context_turns: Tensor, // (B,) bf16/f32
}
```

### 3.5 权重加载

从 PyTorch safetensors 加载，权重名映射：

```rust
// PyTorch 名 → gllm tensor 名
"info_estimator.net.0.weight" → "info_net_fc0_weight"
"W_q.weight"                  → "w_q_weight"
"task_classifier.0.weight"    → "task_fc0_weight"
// ...
```

---

## §4 API 设计

### 4.1 公共 API

```rust
impl Client {
    /// 加载 SignalIntentTracker 模型
    fn new_intent_tracker(model_path: &str) -> Result<IntentTracker>;

    /// 单次推理：预编码 embedding → 分类结果
    fn classify_turn(
        &self,
        embeddings: &[Vec<f32>],  // T 个 768 维向量
        roles: &[u8],             // T 个角色标记
        signals: &[f32; 11],      // 行为信号
        seq_len: usize,
        context_turns: f32,
    ) -> Result<Classification>;

    /// 批量推理
    fn classify_turns_batch(
        &self,
        batch: &[TrackerInput],
    ) -> Result<Vec<Classification>>;
}

struct Classification {
    task_type: TaskType,        // arch_refactor / code_deploy / debugging
    task_confidence: f32,
    difficulty: u8,             // 0-3
    difficulty_confidence: f32,
    info_weights: Vec<f32>,     // 每个 turn 的信息量权重
}
```

### 4.2 端到端管线（encoder + tracker）

```rust
impl Client {
    /// 原始文本 → 编码 → 分类（完整管线）
    fn classify_conversation_turn(
        &self,
        encoder: &Client,       // encoder 模型（ModernBERT）
        tracker: &IntentTracker, // tracker 模型
        turn_texts: &[&str],
        roles: &[u8],
        signals: &[f32; 11],
    ) -> Result<Classification> {
        // 1. encoder.encode(turn_texts) → embeddings
        // 2. tracker.classify_turn(embeddings, roles, signals, ...)
    }
}
```

---

## §5 JIT 编译策略

### 5.1 融合机会

| 融合组 | 算子链 | 融合模式 |
|--------|--------|----------|
| InfoWeight | Linear→LN→ReLU→Drop→Linear→LN→ReLU→Drop→Linear→Sigmoid | LoopFusion |
| QKV Proj | Q_proj + K_proj + V_proj | QkvSharedInput |
| Attention | Q@K/scale + recency_bias + softmax @ V | FusedAttention (扩展: recency bias 作为 Epilogue) |
| DualPath | gate*squeeze + (1-gate)*reduce + LN | LoopFusion |
| Classifier | concat → Linear→LN→ReLU→Drop→Linear→LN→ReLU→Drop→Linear | LoopFusion + NormIntoGemm |

### 5.2 性能预估

- 3.65M 参数，bf16 推理
- CPU (AVX-512): < 0.5ms/batch
- GPU (CUDA): < 0.1ms/batch
- 主要开销在 encoder 侧（311M ModernBERT），tracker 侧可忽略

---

## §6 REQ 清单

| REQ ID | 描述 | 优先级 |
|--------|------|--------|
| REQ-SIT-001 | `ModelArchKey` 注册 + `GraphType::SignalIntentTracker` 定义 | P0 |
| REQ-SIT-002 | `build_signal_intent_tracker_graph()` 图构建器 | P0 |
| REQ-SIT-003 | 权重加载（safetensors 名映射） | P0 |
| REQ-SIT-004 | `new_intent_tracker()` API | P0 |
| REQ-SIT-005 | `classify_turn()` 单次推理 | P0 |
| REQ-SIT-006 | `classify_turns_batch()` 批量推理 | P1 |
| REQ-SIT-007 | 端到端管线（encoder + tracker） | P1 |
| REQ-SIT-008 | GPU codegen (PTX) 支持 | P2 |
| REQ-SIT-009 | 量化支持 (bf16/fp16/int8) | P2 |
