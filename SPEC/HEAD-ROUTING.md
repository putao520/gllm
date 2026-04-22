# Head Routing SDK — 同一 LLM 多头 API

> **执行模型**: Hook/Callback 在 mega-kernel 架构下通过 JIT 内嵌条件 JMP 实现（详见 `08-EXECUTOR.md` §4.1.5）。无 hook 注册时不生成跳转代码。Hook 通信通过共享内存，不经过 Rust 函数调用。


> **📌 SSOT**: 本文档定义 gllm 的 Head Routing SDK 技术协议,允许同一加载后的 LLM 在运行时通过 Client API 切换输出头形态 (text generation / binary classify / multiway classify / mid-layer encode),**不重新加载模型权重**。

> **需求 SSOT**: `SPEC/01-REQUIREMENTS.md §13` REQ-HR-001..005
>
> **API 定义 SSOT**: `SPEC/04-API-DESIGN.md §3.8`

---

## §1 动机与定位

### 1.1 为什么需要 API 级头切换

`ModelKind` (`Chat` / `Embedding` / `Classifier` / `Reranker`) 描述**加载时的主 head 意图**——决定模型 YAML 架构模板、权重加载范围、默认 forward pass 路径。但实际应用中,同一份 generator LLM (如 SmolLM2-135M-Instruct) 的 `lm_head` 投影可以同时服务:

1. **开放式生成** (`generate`) — 自回归采样 token 序列
2. **二元判定** (`classify_binary`) — 读取 lm_head 对 positive/negative token 的 logit,softmax 归一化
3. **多类分类** (`classify_multiway`) — 读取 lm_head 对 N 个候选 label token 的 logit,softmax 归一化
4. **中间层 embedding** (`encode_to_layer`) — 截断前向到 anchor 层,pool 隐藏状态作为通用句向量

重新加载模型为每种形态切换 `ModelKind` 会触发 weight reload / JIT recompile,代价高昂且破坏 `BackendContext` 单例。Head Routing SDK 通过**零权重重载**的 API 切换,实现语义复用。

### 1.2 与 ModelKind 的正交关系

| 维度 | ModelKind | Head Routing |
|------|-----------|--------------|
| 作用时机 | 模型加载期 (`Client::new_*` / Builder) | 运行时每次 API 调用 |
| 改变 | 架构 YAML 选型、权重形状、JIT 编译产物 | 后处理逻辑 (读取哪些 logit、是否提前退出) |
| 粒度 | 整个 Client 的默认 head | 单次 API 调用的 head 形态 |
| 生效开销 | JIT 重编译 + 权重重装 | 仅后处理,前向图零开销变化 |

`Client::new_chat(...)` 加载后,generator 默认主 head 为自回归采样;通过 Head Routing API (`classify_binary` / `classify_multiway` / `encode_to_layer`) 可按需切换输出形态。已有 `Client::classify(...)` (encoder classifier head) 则对应 `ModelKind::Classifier` 的专用路径,不被 Head Routing 覆盖。

### 1.3 与其他 SDK 的关系

- **Semantic Gatekeeper** (`SPEC/SEMANTIC-GATEKEEPER.md`): HR 选择输出头,SG 注入知识残差到检测层 hidden。正交,互不干涉。
- **Generation Hooks** (`SPEC/04-API-DESIGN.md §3.1`): Hooks 作用于采样后 token veto,HR 在 lm_head logits 阶段介入,不重叠。
- **Embed/Rerank/Classify 专用 ModelKind**: 这些 API 绑定加载时的专门模型 (BAAI/bge-m3 embedder 等)。HR 绑定 **generator** 模型,让 generator 兼职前三者的简化形态。

---

## §2 架构总览

### 2.1 数据流

```
 ┌─ Client::classify_binary / classify_multiway / encode_to_layer ─┐
 │                                                                  │
 │  1. prompt → executor.encode_prompt() → tokens: Vec<u32>         │
 │                                                                  │
 │  2. Mega-Kernel 路径 (ARCH-RUST-IS-CODEGEN):                    │
 │     output_mode_selector 参数驱动 JMP table:                     │
 │     ├── 0: generate (argmax → store → check → loop)             │
 │     ├── 1: classify_binary (WriteLogits → output buffer)        │
 │     ├── 2: classify_multiway (WriteLogits → output buffer)      │
 │     └── 3: encode_to_layer (EarlyExit → pool → output buffer)   │
 │                                                                  │
 │     单一 mega-kernel CALL, 零重编译 (§2.2 铁律)                   │
 │                                                                  │
 │  3. Rust 后处理:                                                  │
 │     binary   — softmax over output buffer logits → P(positive)   │
 │     multiway — softmax over output buffer logits → Vec<f32>      │
 │     encode   — Vec<f32> 直接返回 (pool 已在 JIT 内完成)          │
 └──────────────────────────────────────────────────────────────────┘
```

> **Mega-Kernel 架构**: Head Routing 的 runtime switching 通过 MegaKernelFn ABI
> 参数 `output_mode_selector` (u32) 驱动 JMP table 实现。所有 output modes 在加载时
> 编译进单一 JIT 函数，运行时切换仅改变一个 ABI 参数值。详见
> `../gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §1.5.5`。

### 2.2 关键不变量 (铁律)

1. **零权重重载**: Head Routing API 调用前后 `Arc::strong_count(&client.state)` 变化量 ≤ 预期并发请求数,底层 `BackendContext` Arc 指针恒定 (REQ-HR-004)。
2. **零 JIT 重编译**: 所有 HR 调用复用加载期编译好的 `FusedGraphExecutor`,不触发 `compile_model_graphs()` (CLAUDE.md §5 REQ-JIT-CACHE-001 兼容)。
3. **ARCH-FULL-JIT 合规**: 前向图仍走 JIT 管线 (Scalar→SymExec→IR→ISA Lowering),HR 仅读取 JIT 产物最后一层的 hidden state,不绕开 JIT。
4. **NO_SILENT_FALLBACK**: token 找不到 → `HeadRoutingError::TokenNotFound(String)`; mid-layer 不支持 → `HeadRoutingError::MidLayerNotSupported`; 禁止默默返回 0 或 argmax。
5. **NO_ISLAND_MODULE**: HR 模块的公共函数必须在真实 `Client` API 路径被调用;`tests/test_e2e_head_routing.rs` 集成测试验证真实调用链 (REQ-HR-005)。

---

## §3 四个 Head 的数学定义 + ABI 契约

### 3.1 Generate Head (现有,不修改)

```
tokens → forward → sample(logits, temperature, top_k, top_p) → next_token → loop
```

由 `Client::generate()` 走 `executor.generate_with_sampling()` 路径;不在 HR 范围内。

### 3.2 Binary Classify Head

**数学定义**:

```
let tokens      = tokenizer.encode(prompt)
let H           = forward(tokens)                          # [seq_len, hidden_size]
let h_last      = H[-1]                                     # [hidden_size]
let logit_pos   = dot(h_last, embed_tokens[positive_token_id])
let logit_neg   = dot(h_last, embed_tokens[negative_token_id])
let scaled_pos  = logit_pos / temperature
let scaled_neg  = logit_neg / temperature
let P(positive) = softmax([scaled_pos, scaled_neg])[0]
```

**ABI 契约**:

- 输入: `prompt: &str`, `positive_token: &str`, `negative_token: &str`, `temperature: f32`
- 输出: `P(positive) ∈ [0.0, 1.0]` (f32)
- 错误: `TokenNotFound("positive")` / `TokenNotFound("negative")` / `InvalidConfig("temperature must be > 0")`
- 不变式: `P(positive) + P(negative) ≈ 1.0` (数值误差 < 1e-6)

### 3.3 Multiway Classify Head

**数学定义**:

```
let tokens    = tokenizer.encode(prompt)
let H         = forward(tokens)
let h_last    = H[-1]
let logits[i] = dot(h_last, embed_tokens[label_token_ids[i]]) / temperature   (i = 0..N)
let probs     = softmax(logits)                             # [N]
```

**ABI 契约**:

- 输入: `prompt: &str`, `labels: &[&str]`, `temperature: f32`
- 输出: `Vec<f32>` 长度 N
- 错误: `EmptyLabels` / `TokenNotFound(label)` / `InvalidConfig(...)`
- 不变式: `sum(probs) ≈ 1.0` (数值误差 < 1e-6); `probs[i] ∈ [0.0, 1.0]`

### 3.4 Encode to Layer Head

**数学定义** (假设 `FusedGraphExecutor` 支持 mid-layer exit):

```
let tokens     = tokenizer.encode(text)
let anchor_idx = LayerAnchor::resolve(num_layers)
let H_anchor   = forward(tokens, exit_at_layer=anchor_idx)  # [seq_len, hidden_size]
let emb        = PoolMode::apply(H_anchor, seq_len, hidden_size)
```

**ABI 契约**:

- 输入: `text: &str`, `anchor: LayerAnchor`, `pool: PoolMode`
- 输出: `Vec<f32>` 长度 `hidden_size`
- 错误: `MidLayerNotSupported` (当前状态) / `InvalidLayerAnchor(f32)` (越界 Relative)
- 不变式: `emb.len() == hidden_size`; `L2_norm(emb) > 0`

**当前状态**: `FusedGraphExecutor` 的单次前向路径 (`executor.run()` / `run_with_kv_cache_with_config`) 未暴露 `CallbackChain`,无法在中间层截断。`encode_to_layer` 以 `MidLayerNotSupported` 显式拒绝,禁止 stub 或 scalar fallback。落地依赖新增 `FusedGraphExecutor::run_with_early_exit(anchor_layer)` 或把 SG 的 callback chain 机制扩展到单次前向路径。

---

## §4 Binary / Multiway Token-Logit 提取协议

### 4.1 Token ID 解析

```rust
fn resolve_token_id(tokenizer: &TokenizerHandle, text: &str) -> Result<u32, HeadRoutingError> {
    let ids = tokenizer.encode(text, false)?;
    // 非空、长度为 1 的单 token 文本 (常见场景: "yes", "no", "sports", "politics")
    match ids.as_slice() {
        []       => Err(HeadRoutingError::TokenNotFound(format!("{text} tokenized to empty"))),
        [id]     => Ok(*id),
        multiple => Err(HeadRoutingError::TokenNotFound(format!(
            "{text} tokenized to {} tokens {:?}, expected single token",
            multiple.len(), multiple
        ))),
    }
}
```

**设计约束**:
- 要求 token 文本必须单 token 化。多 token 标签需用户预先拆分或选单 token 同义词。
- 如 `"yes"` / `"no"` / `"true"` / `"false"` 在大多数 tokenizer 中是单 token (BPE subword)。
- 失败即返回 `TokenNotFound`,不尝试 "取第一个"(NO_SILENT_FALLBACK)。

### 4.2 Logit 计算 (tied embedding)

SmolLM2 / Qwen3 / Llama 等 decoder generator 的 `lm_head` 权重**与 `embed_tokens.weight` 绑定** (tied)。因此:

```
logit_t = h_last · embed_tokens[t]     (t ∈ vocab)
```

实现上 `score_tokens_forward_gpu_pure` backend 方法接收目标 token id 列表,对每个 id 计算上述点积,避免投影到完整 vocab。

**非 tied 模型**: 未来支持独立 `lm_head` 时需区分 (`lm_head.weight[t]` vs `embed_tokens.weight[t]`);当前 SmolLM2 / Qwen3 / Gemma 等主流 generator 均 tied,暂不处理。

### 4.3 Softmax 温度

```
softmax_T(x)[i] = exp(x[i] / T) / sum(exp(x[j] / T))
```

`T > 0` 强制。`T = 1.0` 保持原始 logit 分布;`T < 1` 锐化 (更接近 argmax);`T > 1` 平滑。`T ≤ 0` → `InvalidConfig`。

---

## §5 Mid-layer Encode 协议

### 5.1 LayerAnchor 语义

```rust
pub enum LayerAnchor {
    /// 相对深度: 0.0 = 第 0 层, 1.0 = 最后一层 (num_layers - 1).
    Relative(f32),
    /// 绝对层索引 (0-based).
    Absolute(usize),
}

impl LayerAnchor {
    pub fn resolve(self, num_layers: usize) -> Result<usize, HeadRoutingError> {
        if num_layers == 0 {
            return Err(HeadRoutingError::InvalidLayerAnchor(-1.0));
        }
        match self {
            LayerAnchor::Relative(r) if (0.0..=1.0).contains(&r) => {
                Ok(((r * (num_layers - 1) as f32).round() as usize).min(num_layers - 1))
            }
            LayerAnchor::Relative(r) => Err(HeadRoutingError::InvalidLayerAnchor(r)),
            LayerAnchor::Absolute(i) if i < num_layers => Ok(i),
            LayerAnchor::Absolute(i) => Err(HeadRoutingError::InvalidLayerAnchor(i as f32)),
        }
    }
}
```

### 5.2 PoolMode 语义

```rust
pub enum PoolMode {
    /// 对 seq 维度平均: H[seq, hidden] → mean(H, axis=0) → [hidden]
    MeanPool,
    /// 取最后一个 token: H[-1]
    LastToken,
    /// 取第一个 token (CLS 位置,encoder 模型惯用): H[0]
    ClsToken,
}
```

### 5.3 当前限制

FusedGraphExecutor 单次前向不支持 mid-layer exit。`encode_to_layer` 统一返回:

```rust
Err(HeadRoutingError::MidLayerNotSupported)
```

错误消息包含固定子串 `"MidLayerNotSupported"` 以便上层程序匹配 (REQ-HR-003 验收标准 #2)。

### 5.4 实现路径 (Mega-Kernel EarlyExit)

`encode_to_layer` 通过 mega-kernel 的 `EarlyExit` op 实现:

1. 编译时: `decoder_model()` 在 `intent_anchor_layer` 对应的层循环位置插入 `EarlyExit` op
2. VM lowering: `EarlyExit` 编译为 CMP layer_counter + JE .early_exit_path
3. early_exit_path: 从当前 hidden state 做 pool (MeanPool/LastToken/ClsToken) → 写入 output buffer → BreakLoop
4. 运行时: `output_mode_selector=3` + `anchor_layer` 参数传入 mega-kernel → JIT 内完成截断和 pool

**依赖**: `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §1.5.5` OutputModeDispatch JMP table + `EarlyExit` op 的 x86 lowering。

---

## §6 LayerAnchor 枚举与 SG 共享语义

`LayerAnchor::Relative(f32)` 与 `SemanticGatekeeperConfig::detection_depths: Vec<f32>` 语义完全一致:

- 范围 `[0.0, 1.0]`
- `resolve(num_layers)` 用相同舍入规则: `round(r × (num_layers - 1))`
- 越界 / NaN → `InvalidLayerAnchor` / `SemanticGatekeeperError::InvalidDepth`

Head Routing 未来可以复用 SG 的 `resolve_detection_layers` 基础设施,避免两份实现漂移。当前 HR 独立实现 `LayerAnchor::resolve`,保留独立测试覆盖。

---

## §7 E2E 验收 (REQ-HR-001..005)

### REQ-HR-001: Binary Classify

```
输入: SmolLM2-135M-Instruct, "Is water wet? Answer yes or no:", {positive="yes", negative="no", T=1.0}
验收:
  (a) 返回 f32 ∈ [0.0, 1.0]
  (b) P(yes) > 0.5 (真实 LLM 行为,水 = 湿)
  (c) P(yes) + P(no) 在浮点精度内等于 1.0
```

### REQ-HR-002: Multiway Classify

```
输入: SmolLM2-135M-Instruct, prompt + ["sports", "politics", "technology"]
验收:
  (a) 返回 Vec<f32> 长度 3
  (b) sum ≈ 1.0 (|sum - 1.0| < 1e-5)
  (c) 所有 probs[i] ∈ [0.0, 1.0]
```

### REQ-HR-003: Encode to Layer

```
输入: SmolLM2-135M-Instruct, text, LayerAnchor::Relative(0.5), PoolMode::MeanPool
验收 (当前阶段, FusedGraphExecutor mid-layer exit 未落地):
  (a) 返回 Err(HeadRoutingError::MidLayerNotSupported)
  (b) 错误消息包含子串 "MidLayerNotSupported"
  (c) 不 panic、不 stub、不 silent fallback
```

### REQ-HR-004: 模型不重载

```
输入: 同一 client 依次调用 generate / classify_binary / classify_multiway
验收:
  (a) 调用前后 `Arc::as_ptr(&client.state_handle().load().unwrap().backend)` 地址恒定
  (b) 总加载耗时 = 首次 Client::new_chat 耗时 (第二/三次 API 调用耗时 < 首次的 5%)
```

### REQ-HR-005: NO_ISLAND_MODULE 接入

```
静态验证:
  grep -rn "classify_binary\|classify_multiway\|encode_to_layer" src/ --include="*.rs" \
    | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
  → 返回 src/client.rs 真实实现 + src/lib.rs 公共导出 (非空、非 test-only)

运行时验证:
  TEST-HR-004 中调用 classify_binary 能真实进入 `src/head_routing.rs` 的 config
  构造 + token 解析 + backend.score_tokens_forward_gpu_pure 路径
```

---

## §8 与 Semantic Gatekeeper 的正交性

| 关注点 | Semantic Gatekeeper | Head Routing |
|--------|---------------------|--------------|
| 作用阶段 | forward 中间层 `pre_node` callback 注入残差 | forward **完成后**读取最后一层 hidden |
| 修改对象 | `hidden_state[-1]` 原位修改 (InjectHidden) | 不修改 forward 结果,只读 |
| 依赖 | CallbackChain + FusedAttentionLayer q_tap | `FusedGraphExecutor::run` 完整前向产物 |
| 触发时机 | 每个 decode step 在检测层 | 每次 HR API 调用一次 (单次前向) |

HR 与 SG 可**同时启用**: Client 可以先 `register_semantic_gatekeeper(...)` 再调用 `classify_binary(...)`。SG 在前向中间层修改 hidden,HR 读到的就是 SG 注入后的最终 hidden,`P(positive)` 自动受 SG 残差影响。这是**设计意图**,不是 bug——允许知识注入驱动分类决策。

---

## §9 实现映射

| SPEC 条目 | 代码位置 |
|-----------|----------|
| `LayerAnchor` / `PoolMode` / `ClassifyBinaryConfig` / `ClassifyMultiwayConfig` / `HeadRoutingError` | `src/head_routing.rs` |
| `Client::classify_binary` / `classify_multiway` / `encode_to_layer` | `src/client.rs` |
| `Backend::score_tokens_forward_gpu_pure` (CPU 实现) | `src/compat/cpu_backend.rs` |
| `BackendExecutor::score_tokens` pass-through | `src/backend/mod.rs` |
| `Executor::score_tokens_for_prompt` | `src/engine/executor.rs` |
| 公共导出 | `src/lib.rs` |
| E2E 测试 | `tests/test_e2e_head_routing.rs` |

---

## §10 变更历史

SPEC 文件不维护变更历史(CLAUDE.md §6 铁律 4: SPEC 防腐化)。版本历史由 git 管理。本 SPEC 当前描述的是**唯一正确设计**,如需更新设计,整段重写对应章节。
