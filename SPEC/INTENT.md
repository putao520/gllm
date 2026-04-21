# Intent Recall SDK — 中间层 Hidden 抽取协议

> **SSOT**: 本文档定义 gllm Intent Recall SDK 的技术协议, 提供**截断前向至 anchor 层后抽取 pooled hidden** 用于轻量意图分类 / RAG query 理解等下游任务。
>
> **需求 SSOT**: `SPEC/01-REQUIREMENTS.md §15` REQ-INTENT-001..003
>
> **API 定义 SSOT**: `SPEC/04-API-DESIGN.md §3.10`
>
> **遵守铁律**: ARCH-FULL-JIT、NO_SILENT_FALLBACK、NO_ISLAND_MODULE、DRY

---

## §1 动机

### 1.1 为什么需要中间层 encode

完整 forward 跑完所有 Transformer 层成本高。许多下游任务 (意图分类 / 粗召回 / 关键词匹配) 只需要"**前半层 hidden 足够抽取语义**"的证据 — 研究已反复表明 BERT/GPT 系列模型在 ~50-70% 深度处 hidden 空间几何对分类任务最有用。

Intent Recall SDK 将这一观察落地为一个**与 HR 共享代码路径的语义包装 API**:

- `Client::encode_to_layer(text, anchor, pool)` — HR 暴露的底层截断 encode
- `Client::encode_intent(text, anchor, pool)` — Intent SDK 语义包装, 内部 delegate 给 `encode_to_layer`

两者**完全等价**, 仅 API 命名不同。分离目的: 在 RAG / intent router 代码中读 `encode_intent` 比 `encode_to_layer` 更直观。

### 1.2 与 HR 的正交关系

| 维度 | Head Routing | Intent Recall |
|------|--------------|---------------|
| 用途描述 | 通用中间层 encode + pool | 专用意图识别 / query 召回 |
| API 命名语义 | `encode_to_layer` | `encode_intent` |
| 实现 | 原始 JIT 前向截断 | delegate 到 `encode_to_layer` (零代码复制) |
| 返回类型 | `Vec<f32>` | `IntentEncoding { embedding, actual_layer, pool }` |
| 错误集 | `HeadRoutingError` | `IntentError` (可 `From<HeadRoutingError>`) |

---

## §2 架构

### 2.1 数据流

```
 ┌─ Client::encode_intent(text, anchor, pool) ─┐
 │                                              │
 │  1. 解析 anchor → actual_layer               │
 │     (LayerAnchor::resolve)                   │
 │                                              │
 │  2. embedding = Client::encode_to_layer(...)  │
 │     ↓ (详见 HEAD-ROUTING.md §5)              │
 │     ├─ 构造 MidLayerEncodeCallback            │
 │     ├─ FusedGraphExecutor::run_with_callbacks │
 │     │    在 anchor_layer 的 post_node         │
 │     │    ExitEarly { logits: hidden_as_f32 }  │
 │     └─ pool.apply(hidden, seq_len, hidden_size)│
 │                                              │
 │  3. 返回 IntentEncoding {                    │
 │       embedding, actual_layer, pool          │
 │     }                                        │
 └──────────────────────────────────────────────┘
```

### 2.2 关键不变量

1. **DRY 铁律**: `encode_intent` 不得**复制** `encode_to_layer` 实现逻辑。一行 `delegate`, 然后包装为 `IntentEncoding`。
2. **JIT 合规**: 前向仍走 JIT 管线; 中间层 exit 仅通过 `CallbackAction::ExitEarly` 实现, 不绕开 JIT codegen。
3. **确定性**: `encode_intent("text", anchor, pool).embedding` 与 `encode_to_layer("text", anchor, pool)` 输出**严格按位相等** (除接口外无运算差异)。

---

## §3 API

### 3.1 方法签名

```rust
use gllm::{Client, IntentEncoding, LayerAnchor, PoolMode};

impl Client {
    pub fn encode_intent(
        &self,
        text: &str,
        anchor: LayerAnchor,
        pool: PoolMode,
    ) -> Result<IntentEncoding, ClientError>;
}
```

### 3.2 返回类型

```rust
pub struct IntentEncoding {
    pub embedding: Vec<f32>,   // 长度 = model hidden_size
    pub actual_layer: usize,   // anchor.resolve(num_layers) 的结果
    pub pool: PoolMode,        // 使用的 pool 模式 (原样透传)
}

impl IntentEncoding {
    pub fn dim(&self) -> usize;         // = embedding.len()
    pub fn l2_norm(&self) -> f32;       // L2 范数, 调试/健康检查
}
```

### 3.3 错误类型

```rust
pub enum IntentError {
    InvalidLayerAnchor(String),   // Anchor 越界 / NaN
    EncodeFailed(String),         // 下游 executor/backend 错误
    NoModelLoaded,
}

impl From<HeadRoutingError> for IntentError { /* 透传 */ }
```

注意: `Client::encode_intent` 当前通过 `encode_to_layer` 实现, 返回的是 `Result<IntentEncoding, ClientError>`。`IntentError` 保留给未来需要与 HR 错误分离的场景 (例如独立 intent router 库)。

---

## §4 用例

```rust
let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

// 对意图分类: 取 50% 层 + mean pool
let intent = client.encode_intent(
    "What is the weather in Paris?",
    LayerAnchor::Relative(0.5),
    PoolMode::MeanPool,
)?;

assert_eq!(intent.dim(), 576); // SmolLM2-135M hidden_size
println!("actual_layer = {}", intent.actual_layer);

// 用作 RAG query 召回向量, 对 knowledge base 做 cosine search:
let query_vec = intent.embedding;
let top_docs = kb.nearest(query_vec, k=5);
```

---

## §5 Integration Trace (NO_ISLAND_MODULE 合规证据)

| 步骤 | 文件:函数 |
|------|----------|
| 用户 API | `src/client.rs::Client::encode_intent` |
| Delegate | `Client::encode_intent` → `Client::encode_to_layer` |
| 底层实现 | `Client::encode_to_layer` → `Executor::encode_at_layer_for_prompt` |
| MidLayer Callback | `src/engine/callbacks/mid_layer_encode.rs::MidLayerEncodeCallback::post_node` |
| Executor 接入 | `Executor::encode_at_layer_for_prompt` 构造 `CallbackChain`, 设置 `callback_chain_ptr` |
| Backend CPU 路径 | `src/compat/cpu_backend.rs::encode_at_layer_forward_gpu_pure` → `run_with_optional_callbacks` → `FusedGraphExecutor::run_with_callbacks` |
| GPU 实现 | `Unimplemented` (macro 统一声明), 待 ARCH-CPU-GPU-UNIFIED 落地 |
| 公共导出 | `src/lib.rs` (`pub use intent::{IntentEncoding, IntentError}`) |
| E2E 测试 | `tests/test_e2e_intent.rs` |

审计命令:
```bash
grep -rn "encode_intent\|MidLayerEncodeCallback" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
# 期待至少有 client.rs / engine/executor.rs / engine/callbacks/*.rs 的真实调用链命中
```

---

## §6 E2E 验收 (REQ-INTENT-001..003)

### REQ-INTENT-001: Basic shape + finite

```
输入: SmolLM2-135M-Instruct, "Hello world", LayerAnchor::Relative(0.5), MeanPool
验收:
  (a) intent.dim() == model hidden_size
  (b) intent.embedding 所有元素 finite
  (c) intent.l2_norm() > 0
  (d) intent.pool == MeanPool (原样透传)
  (e) intent.actual_layer ∈ [0, num_layers)
```

### REQ-INTENT-002: PoolMode 影响结果

```
输入: 多 token 文本, 同一 anchor, 三种 PoolMode (MeanPool / LastToken / ClsToken)
验收:
  (a) 三个 embedding 维度相同 (= hidden_size)
  (b) MeanPool 与 LastToken 向量差异显著 (L1 delta > 1e-3)
  (c) ClsToken 与 LastToken 向量差异显著 (L1 delta > 1e-3)
```

### REQ-INTENT-003: `encode_intent` 与 `encode_to_layer` 等价

```
输入: 同一 text, 同一 anchor, 同一 PoolMode
验收:
  (a) encode_intent(...).embedding 与 encode_to_layer(...) 逐元素近似相等 (|Δ| < 1e-5)
  (b) DRY 铁律: client.rs 中 encode_intent 直接 delegate 到 encode_to_layer, 无重复实现
```

---

## §7 与 SG / Guardrail 的组合

同一 Client 加挂 SG + Guardrail 后再调 `encode_intent`:

- SG 在 pre_node 注入知识残差 — 影响中间 hidden
- Guardrail 在 post_node 评估 hidden 做安全检测
- Intent 在 anchor 层的 post_node 抽取 hidden 然后 ExitEarly

执行顺序 (按 priority): SG (90) → MidLayerEncode (55) → Guardrail (40)。

注意: 若 Guardrail 在 anchor_layer 前触发 veto, MidLayerEncodeCallback 永远不会执行, `encode_intent` 返回空 embedding — 这是**预期行为** (veto 优先于 encode)。

---

## §8 实现映射

| SPEC 条目 | 代码位置 |
|-----------|----------|
| `IntentEncoding` / `IntentError` | `src/intent.rs` |
| `Client::encode_intent` | `src/client.rs` |
| `MidLayerEncodeCallback` (`LayerCallback::post_node` 实现) | `src/engine/callbacks/mid_layer_encode.rs` |
| `Executor::encode_at_layer_for_prompt` | `src/engine/executor.rs` |
| `Backend::encode_at_layer_forward_gpu_pure` (CPU + GPU Unimplemented) | `src/compat/cpu_backend.rs`, `src/compat/gpu_backend_macro.rs` |
| `FusedGraphExecutor::run_with_callbacks` | `src/graph/executor.rs` |
| 公共导出 | `src/lib.rs` |
| E2E 测试 | `tests/test_e2e_intent.rs` |

---

## §9 变更历史

SPEC 文件不维护变更历史 (CLAUDE.md §6 铁律 4: SPEC 防腐化)。版本历史由 git 管理。
