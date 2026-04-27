# Semantic Gatekeeper 技术协议 (ARCH-SEMANTIC-GATEKEEPER)

> **执行模型**: Hook/Callback 在 mega-kernel 架构下通过 JIT 内嵌条件 JMP 实现（详见 `08-EXECUTOR.md` §4.1.5）。无 hook 注册时不生成跳转代码。Hook 通信通过共享内存，不经过 Rust 函数调用。


> **SSOT**: 本文档是 gllm 隐藏状态驱动的免训练知识调度与注入架构的唯一真源。
>
> **关联需求**: `SPEC/01-REQUIREMENTS.md §12` REQ-SG-001..008
>
> **关联上层 API**: `SPEC/04-API-DESIGN.md §7` 用户接口、`§8` 内部实现架构
>
> **关联运行时接入**: `SPEC/05-OPTIMIZATIONS.md §2.9` Callback 集成、`SPEC/08-EXECUTOR.md §4.2` FusedAttentionLayer Q-tap
>
> **遵守铁律**: ARCH-FULL-JIT、ARCH-CPU-GPU-UNIFIED、NO_SILENT_FALLBACK、NO_ISLAND_MODULE

---

## 1. 动机与定位

### 1.1 问题域

Transformer 语言模型在结构化知识依赖任务（代码补全、架构规划、领域推理）中存在三类固有缺陷：

1. **RAG（检索增强生成）** 将知识置入 Prompt，模型可通过注意力权重选择性忽略；长文档侵占上下文窗口；检索-生成割裂，无法感知动态需求。
2. **约束解码（Logits 钳制）** 强制合法输出集合，剥夺模型对别名、新符号的泛化能力（如 `import pandas as pd` 后 `pd.read_csv` 别名场景失效）。
3. **工具调用（Function Calling）** 显式中断生成流程，依赖模型自身决策调用时机，延迟高、粒度粗。

### 1.2 核心论断

> **结构化知识的注入应是模型推理过程中"认知流"的自然延伸，既非外挂 Prompt，也非生硬输出过滤，而是在隐藏状态流中的软偏置注入。**

两个支撑事实：

- Transformer 自注意力 Query 向量蕴含当前位置对上下文的**信息检索意图**。通过与外部结构化层级键比对，模型可在不修改参数的前提下**主动表达所需知识的层级**。
- AST 提供语法骨架，LSP 赋予语义灵魂。Tree-sitter 实时捕捉语法状态；LSP 提供跨文件精确类型/符号/架构关系。两者协同构成完整的认知触发与知识供给链路。

### 1.3 定位

Semantic Gatekeeper（以下简称 **SG**）是 gllm 的**运行时可插拔 Callback 模块**，在 `FusedGraphExecutor::run_with_kv_cache_with_callbacks()` 的中间检测层 `pre_node` 钩子内运行，对模型参数、算子图结构、权重布局**完全零修改**。

---

## 2. 架构总览

### 2.1 数据流

```
┌────────────────────────── 模型加载阶段（一次性） ─────────────────────────┐
│                                                                              │
│  SemanticGatekeeperConfig                                                    │
│    ├─ level_descriptors: [String; 3]  (默认 L1/L2/L3 文本，可覆写)           │
│    ├─ detection_depths: Vec<f32>      (相对深度列表，如 [0.5, 0.75, 0.9])    │
│    ├─ gate_threshold: f32             (门控 τ)                               │
│    ├─ stability_threshold: f32        (锚点复用阈值)                         │
│    ├─ alpha: f32                      (注入强度基础值)                       │
│    ├─ knowledge_provider: Arc<dyn KnowledgeProvider>                         │
│    └─ ast_sentinel: Option<Arc<dyn AstSentinel>>                             │
│                                                                              │
│  预计算 Level Keys (§3):                                                     │
│    for each detection_depth d:                                               │
│      layer_L = floor(d × num_layers)                                         │
│      for each level_desc text in level_descriptors:                          │
│        tokens   = tokenizer(text)                                            │
│        embed    = execute_small_graph(EmbedLookupOnlyGraph, tokens)          │
│        K_layer  = execute_small_graph(                                       │
│                     KProjOnlyGraph { layer: layer_L },                       │
│                     embed)                                                   │
│        mean_K   = mean_pool(K_layer, axis=seq)                               │
│      LevelKeysAtLayer[layer_L] = [mean_K_L1, mean_K_L2, mean_K_L3]           │
│                                                                              │
│  结果：HashMap<layer_idx, [Vec<f32>; 3]> 常驻（总计 <3 KB × detection count）│
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── 推理阶段（每 decode step） ─────────────────────┐
│                                                                              │
│  FusedGraphExecutor::run_with_kv_cache_with_callbacks():                     │
│  for (node_idx, node) in graph.nodes:                                        │
│    ┌─ is_detection_layer(node_idx)?                                          │
│    │    ├─ NO  → 正常执行，不触发 SG                                         │
│    │    └─ YES → SemanticGatekeeperCallback.pre_node(ctx, node_idx)          │
│    │              ├─ 1. 稳定性追踪 (§5):                                    │
│    │              │    sim = cosine(ctx.hidden_state[-1], anchor_hidden)    │
│    │              │    if sim > stability_threshold && !node_changed:       │
│    │              │      → 复用 v_knowledge, 跳过步骤 2-6                    │
│    │              │                                                         │
│    │              ├─ 2. 从 Q-tap 读 Q (§4):                                 │
│    │              │    执行 FusedAttentionLayer (Q-tap variant) 将           │
│    │              │    q_proj(hidden)[-1] 写入 gatekeeper_ring_buffer       │
│    │              │    Q = gatekeeper_ring_buffer.read()                    │
│    │              │                                                         │
│    │              ├─ 3. 层级路由:                                           │
│    │              │    K_L1, K_L2, K_L3 = LevelKeysAtLayer[node.layer_idx]  │
│    │              │    scores = [cosine(Q, K_Lx) for x in 1..=3]            │
│    │              │    (best_idx, best_score) = argmax(scores)               │
│    │              │    if best_score < gate_threshold: return Continue       │
│    │              │                                                         │
│    │              ├─ 4. 知识检索:                                           │
│    │              │    level = SemanticLevel::from_idx(best_idx)             │
│    │              │    ast_ctx = ast_sentinel?.current_context(..)?          │
│    │              │    entry = knowledge_provider.retrieve(                  │
│    │              │             Q, level, &RetrieveContext { ast_ctx, .. })  │
│    │              │    if entry.is_none(): return Continue                   │
│    │              │                                                         │
│    │              ├─ 5. 文本编码:                                           │
│    │              │    tokens = tokenizer(entry.text)                        │
│    │              │    v_raw  = execute_small_graph(                         │
│    │              │               EmbedLookupOnlyGraph, tokens)              │
│    │              │    v_knowledge = mean_pool(v_raw, axis=seq)              │
│    │              │    (v_knowledge 与 hidden_state 最后 token 维度一致)     │
│    │              │                                                         │
│    │              ├─ 6. 残差相加注入:                                       │
│    │              │    effective_alpha = alpha × entry.confidence            │
│    │              │    new_hidden_last = hidden_state[-1]                    │
│    │              │                   + effective_alpha × v_knowledge        │
│    │              │    data = hidden_state.clone_with_last_replaced(..)      │
│    │              │    return InjectHidden { data }                          │
│    │              │                                                         │
│    │              └─ 7. 更新 active_state:                                  │
│    │                    anchor_hidden  = new_hidden_last                    │
│    │                    v_knowledge    = v_knowledge                         │
│    │                    level          = level                               │
│    │                    key_hash       = hash(entry.text)                    │
│    │                    ast_node_kind  = ast_ctx?.node_kind                  │
│    │                                                                          │
│    └─ 正常 execute_fused_op(node, hidden_state, ...)                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 五大组件

| 组件 | 职责 | 位置 |
|------|------|------|
| **LevelKeysCache** | 模型加载时预计算的层级键向量表 `HashMap<layer_idx, [Vec<f32>; 3]>` | `src/semantic_gatekeeper/level_keys.rs` |
| **SmallGraphExecutor** | 为 Level Keys 预计算 / runtime 文本编码构造并执行小 CompilerGraph（EmbedLookupOnly / KProjOnly） | `src/semantic_gatekeeper/small_graph.rs` |
| **SemanticGatekeeperCallback** | 实现 `LayerCallback` trait，在 `pre_node` 执行路由/检索/注入 | `src/semantic_gatekeeper/callback.rs` |
| **AstSentinel trait** | 用户实现：根据生成的 token 序列返回当前语法上下文 | 用户实现（lib 提供 trait） |
| **KnowledgeProvider trait** | 用户实现：根据 Query 向量 + 层级 + AST 上下文检索知识文本 | 用户实现（lib 提供 trait） |

---

## 3. Level Keys 预计算协议

### 3.1 数学定义

对每个检测层深度 `d ∈ detection_depths`（如 `0.5, 0.75, 0.9`）：

```
layer_L := floor(d × num_hidden_layers)

对每个层级描述文本 desc_x（x ∈ {L1, L2, L3}）：
  tokens_x   := tokenizer.encode(desc_x)               形状 (T_x,)
  embed_x    := embed_layer(tokens_x)                  形状 (T_x, hidden_size)
  k_proj_out := k_proj_at_layer(layer_L, embed_x)      形状 (T_x, kv_dim)
  K_Lx       := mean_over_axis0(k_proj_out)            形状 (kv_dim,)

LevelKeysAtLayer[layer_L] := [K_L1, K_L2, K_L3]
```

其中 `kv_dim = num_kv_heads × head_dim`（GQA 时小于 `hidden_size`）。

### 3.2 默认层级描述文本

| 层级 | 默认描述 | 语义 |
|------|----------|------|
| L1 | `"struct fields and method signatures"` | 符号签名、类型成员 |
| L2 | `"validation rules and invariants"` | 接口约束、业务规则 |
| L3 | `"module dependencies and design patterns"` | 架构分层、模块职责 |

用户可通过 `SemanticGatekeeperConfig.level_descriptors: [String; 3]` 覆写。

### 3.3 执行机制（ARCH-FULL-JIT 合规）

预计算阶段构造两个小 `CompilerGraph`（全 JIT 管线编译，无手写前向）：

**EmbedLookupOnlyGraph**（挑战 3 决策 B）：

```
节点: Gather(embed_table_weight)
输入: token_ids (Symbolic seq_len)
输出: embed (seq_len × hidden_size)
```

**KProjOnlyGraph@layer_L**（挑战 2 决策 B）：

```
节点: RmsNorm(input_layernorm_weight_@L, eps) → Gemm(k_proj_weight_@L)
输入: hidden (Symbolic seq_len × hidden_size)
输出: k_out (seq_len × kv_dim)
```

两个图在模型加载期通过 `FusedGraphExecutor::new(graph).compile()` 编译一次，其 `CompiledLayer` 与主模型 `FusedGraphExecutor` 并列常驻 `SemanticGatekeeperState`。

> **ARCH-CPU-GPU-UNIFIED 合规**: 两个小图与主推理图共享同一 `CompilerGraph` IR，Phase 3 codegen 按 `DeviceProfile` 生成 CPU/PTX/HIP/MSL 原生代码。禁止后端分叉实现。

### 3.4 SymDim 策略

- `seq_len` 维度 = `SymDim::Symbolic("sg_desc_seq")`（描述文本 token 数，运行时 binding）
- `hidden_size` / `kv_dim` 维度 = `SymDim::Concrete(value)`（模型加载后已知）

### 3.5 缓存格式

```rust
pub struct LevelKeysCache {
    /// layer_idx → [K_L1, K_L2, K_L3]
    keys: HashMap<usize, [Vec<f32>; 3]>,
    /// 预计算的检测层集合（与 config.detection_depths 一一映射）
    detection_layers: Vec<usize>,
}
```

**一致性不变量**：
- `keys.keys()` ⊆ `detection_layers` 集合
- 每个 `Vec<f32>.len() == kv_dim`
- 所有向量 finite（不含 NaN/Inf）
- 每个向量 L2 范数 > 0（非全零）

---

## 4. Q 截获协议：FusedAttentionLayer Q-Tap

### 4.1 决策依据（挑战 1 选项 B）

- 选项 A（通用中间量钩子）破坏融合完整性
- 选项 C（拆子算子）违反 ARCH-CPU-GPU-UNIFIED
- **选项 B（Q-tap epilogue）** 复用已有 Epilogue 白嫖网络（`SPEC/05-OPTIMIZATIONS.md §5`），在 `q_proj` 后尾段插入 STG 指令写入共享 ring buffer，不拆融合、不增 API、不增后端分叉

### 4.2 Q-Tap 变体定义

`FusedAttentionLayer` 扩展一个可选配置字段：

```rust
pub struct FusedAttentionLayerConfig {
    // ... 现有字段（head_dim/num_heads/num_kv_heads/rope_config/...）...
    /// 若 Some，编译期在 q_proj 后尾段插入 Q-tap STG。
    pub q_tap: Option<QTapConfig>,
}

pub struct QTapConfig {
    /// 写出目标：gatekeeper_ring_buffer 的设备可见指针
    pub sink_ptr: u64,
    /// 要写出的 token 位置（默认仅最后一个 token）
    pub tap_position: QTapPosition,
    /// 写出 dtype（默认 = k_proj 输出 dtype）
    pub dtype: DType,
}

pub enum QTapPosition {
    /// 仅最后一个 token（decode step 场景）
    LastToken,
    /// 全序列（prefill 场景，ring buffer 需要 seq_len × q_dim 容量）
    AllTokens,
}
```

### 4.3 CompilerGraph 扩展

`OpKind::GemmBias`（或等价的 `q_proj` GEMM 节点）在存在 `q_tap` 时新增 `tap_sink_ptr` 属性，JIT codegen 生成"主 GEMM 结果 → 寄存器 → 残差流 / 下游 RoPE"路径的同时，同一尾段追加 STG 指令写 `tap_sink_ptr`。

> **详述**: `SPEC/08-EXECUTOR.md §4.2` FusedAttentionLayer 扩展描述。

### 4.4 Ring Buffer 布局

```
GatekeeperRingBuffer (per-session):
  ├─ header: {
  │    write_cursor: AtomicU64,
  │    buffer_bytes: u64,
  │    q_dim: u32,
  │    step_index: AtomicU64,
  │  }
  └─ data: [u8; N × q_dim × dtype_size]
           (N = 双缓冲至少 2，防止 pre_node 读到未完成写入)
```

**同步协议**：
- JIT 的 STG 指令使用 `release` 内存序完成写入后 bump `step_index`
- `SemanticGatekeeperCallback.pre_node` 读取时 `acquire` 内存序载入 `step_index` 确认 step 匹配
- 当前 decode step 与 ring buffer 记录的 step_index 不一致 → 返回 `Err(StaleQTap)`

### 4.5 仅检测层携带 Q-tap

`FusedGraph` 构建期，`is_detection_layer(layer_idx)` 的 FusedAttentionLayer 节点才携带 `q_tap: Some(...)`。其他层 `q_tap: None`，完全零额外指令开销。

> **NO_ISLAND_MODULE 合规**: `q_tap` 字段必须在主推理路径（非测试代码）被 `FusedGraph` 构建函数真实写入，且 `SemanticGatekeeperCallback.pre_node` 必须真实读取 ring buffer。集成测试验证 Q 向量与 CPU 参考实现数值一致。

---

## 5. 稳定性追踪状态机

### 5.1 ActiveState 定义

```rust
pub struct ActiveState {
    /// 当前活跃的知识层级（None 表示未命中）
    pub level: Option<SemanticLevel>,
    /// 当前知识条目的文本 hash（避免重复检索同一条目）
    pub key_hash: Option<u64>,
    /// 注入时的 hidden_state 最后 token 向量（锚点）
    pub anchor_hidden: Option<Vec<f32>>,
    /// 当前注入的知识向量（complexity 追踪）
    pub v_knowledge: Option<Vec<f32>>,
    /// 注入时的 AST 节点 kind（AST 变化强制刷新）
    pub ast_node_kind: Option<String>,
    /// 最近一次更新的 decode step 号
    pub last_step: u64,
}
```

### 5.2 状态转移

每个 `pre_node` 触发时：

```
输入: ctx.hidden_state (当前), ast_ctx (当前), step (当前)

Decision =
  if active.anchor_hidden is None or active.ast_node_kind != ast_ctx.node_kind:
    → FullCompute  (走完 §2.1 步骤 2-7)
  elif cosine(ctx.hidden_state[-1], active.anchor_hidden) > stability_threshold:
    → ReuseCache   (复用 active.v_knowledge，跳过步骤 2-6，更新 last_step)
  else:
    → FullCompute  (走完 §2.1 步骤 2-7)
```

### 5.3 刷新触发器

**强制刷新**（清空 ActiveState）：

1. AST 哨兵报告节点 kind 变更（如 `member_expression` → `call_expression`）
2. Decode step 跨越请求边界（new request id）
3. 用户显式调用 `Client::reset_gatekeeper_state()`

### 5.4 开销估计

论文声称 95% 以上步骤复用 `v_knowledge`，实际效率取决于：

- `stability_threshold` 偏高（0.98） → 更保守，更多 FullCompute
- 偏低（0.90） → 更激进，可能错过应注入的新语义

默认 `0.95`，可用户调节。

---

## 6. KnowledgeProvider Trait 契约

### 6.1 Trait 定义

```rust
pub trait KnowledgeProvider: Send + Sync {
    /// 根据 Query 向量、层级、上下文检索知识条目。
    ///
    /// 返回 None 表示无匹配（SG 不注入，返回 Continue）。
    fn retrieve(
        &self,
        query: &[f32],
        level: SemanticLevel,
        ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry>;
}

pub struct KnowledgeEntry {
    /// 将被 tokenizer + embed 层编码的文本
    pub text: String,
    /// 置信度，用于动态调节 α_effective = α × confidence
    /// 取值范围 [0.0, 1.0]
    pub confidence: f32,
}

pub struct RetrieveContext<'a> {
    /// 最近生成的 token 序列（解码成字符串由 Provider 负责）
    pub generated_tokens: &'a [u32],
    /// AST 哨兵返回的语法上下文（可能为 None）
    pub ast: Option<AstContext<'a>>,
    /// 当前 decode step 号
    pub step: u64,
    /// 请求 ID
    pub request_id: RequestId,
}
```

### 6.2 AstSentinel Trait 定义

```rust
pub trait AstSentinel: Send + Sync {
    /// 根据最近生成的 token 序列返回当前语法上下文。
    ///
    /// 返回 None 表示无法解析（SG 仍可基于 Query 向量路由）。
    fn current_context<'a>(
        &self,
        generated_tokens: &'a [u32],
        tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>>;
}

pub struct AstContext<'a> {
    /// Tree-sitter 节点 kind（如 "member_expression" / "call_expression"）
    pub node_kind: &'a str,
    /// 光标所在的文本范围（用于 LSP 符号检索）
    pub cursor_line: u32,
    pub cursor_column: u32,
    /// 光标前已输入的字符串（用于补全匹配）
    pub prefix: &'a str,
}
```

### 6.3 实现者职责

**gllm 不实现任何具体的 KnowledgeProvider / AstSentinel**。开发者实现时可参考论文 §3.2 部署形态：

- **完全本地模式**：LSP 客户端 + LSH 索引 + 向量查询，均在进程内
- **云推理 + 本地知识库模式**：KnowledgeProvider 通过 HTTP/gRPC 回调到用户本地服务

---

## 7. CallbackChain 集成（挑战 4 决策 B）

### 7.1 利用现有 LayerContext

`SPEC/05-OPTIMIZATIONS.md §1.2` 定义的 `LayerContext` 已包含 `hidden_state: &'a mut [f32]`，SG Callback 直接在其上执行残差相加：

```rust
impl LayerCallback for SemanticGatekeeperCallback {
    fn pre_node(&mut self, ctx: &LayerContext, node_idx: usize) -> CallbackAction {
        if !self.is_detection_layer(node_idx) {
            return CallbackAction::Continue;
        }

        // 1. 稳定性追踪 (§5)
        let state = self.active_state.read();
        let last_token_hidden = &ctx.hidden_state[ctx.hidden_state.len() - self.hidden_size..];
        let action = match &state.anchor_hidden {
            Some(anchor) if self.cosine(last_token_hidden, anchor) > self.stability_threshold
                && self.ast_unchanged() =>
            {
                Action::ReuseCache(state.v_knowledge.clone().unwrap())
            }
            _ => Action::FullCompute,
        };
        drop(state);

        let v_knowledge = match action {
            Action::ReuseCache(v) => v,
            Action::FullCompute => {
                // 2. 读 Q-tap (§4)
                let q = match self.q_tap_sink.read_last_q(ctx.layer_idx, ctx.step) {
                    Ok(q) => q,
                    Err(_) => return CallbackAction::Continue, // 陈旧 → 跳过
                };

                // 3. 层级路由
                let keys = &self.level_keys_cache.keys[&ctx.layer_idx];
                let scores = [self.cosine(&q, &keys[0]),
                              self.cosine(&q, &keys[1]),
                              self.cosine(&q, &keys[2])];
                let (best_idx, best_score) = argmax(&scores);
                if best_score < self.gate_threshold {
                    return CallbackAction::Continue;
                }

                // 4. 知识检索
                let level = SemanticLevel::from_idx(best_idx);
                let ast_ctx = self.ast_sentinel.as_ref()
                    .and_then(|s| s.current_context(&ctx.generated_tokens, &self.tokenizer));
                let entry = match self.knowledge_provider.retrieve(
                    &q, level, &RetrieveContext { /* ... */ }
                ) {
                    Some(e) => e,
                    None => return CallbackAction::Continue,
                };

                // 5. 文本编码
                let tokens = self.tokenizer.encode(&entry.text);
                let embed = self.small_graph_exec.run_embed_lookup(&tokens)?;
                let v_k = mean_pool(&embed, self.hidden_size);

                // 更新 active_state
                let mut state = self.active_state.write();
                state.level = Some(level);
                state.key_hash = Some(hash(&entry.text));
                state.anchor_hidden = Some(last_token_hidden.to_vec());
                state.v_knowledge = Some(v_k.clone());
                state.ast_node_kind = ast_ctx.map(|c| c.node_kind.to_string());
                state.last_step = ctx.step;

                v_k * entry.confidence  // 置信度调节
            }
        };

        // 6. 残差相加 → 现有 InjectHidden
        let effective_alpha = self.alpha;  // 如果 ReuseCache 则 confidence 已编码在 v_knowledge
        let mut new_hidden = ctx.hidden_state.to_vec();
        let last_start = new_hidden.len() - self.hidden_size;
        for i in 0..self.hidden_size {
            new_hidden[last_start + i] += effective_alpha * v_knowledge[i];
        }
        CallbackAction::InjectHidden { data: new_hidden }
    }

    fn post_node(&mut self, _ctx: &LayerContext, _node_idx: usize, _output: &[f32]) {}
}
```

### 7.2 不修改 CallbackAction

复用现有 `CallbackAction::InjectHidden { data }`。零 API 扩展。

### 7.3 CallbackChain 优先级

注册到 `SPEC/05-OPTIMIZATIONS.md §8` 执行优先级表，优先级 **90**（替代原 "Knowledge Inject"），理由：

- 高于 RAG Inject（80），因 SG 是精细化层级注入，RAG 是粗粒度 Prompt 融合，两者并存时 SG 先执行
- 低于 Prefetch（100），因 Prefetch 是内存调度，与 SG 语义正交

### 7.4 Mega-Kernel 路径集成 (ARCH-RUST-IS-CODEGEN 合规)

> **铁律**: mega-kernel 路径中 Rust 不参与推理循环。SG 的路由/检索/注入
> 全部编译为 JIT 机器码，通过 `hook_ctx_ptr` 共享内存通信。

#### 7.4.1 架构差异

| 路径 | SG 实现 | 通信机制 |
|------|---------|---------|
| **batch_forward_gpu_pure** (非 mega-kernel) | Rust `LayerCallback::pre_node()` | Rust `CallbackChain::dispatch_pre_node()` |
| **mega-kernel generate_single_sequence** | JIT 内嵌 `SgDetect` + `SgInject` OpKind | `hook_ctx_ptr` 共享内存 |

非 mega-kernel 路径中，executor 的 `run_batch_forward()` 将 `SemanticGatekeeperCallbackShim` 注册到 `CallbackChain`，JIT 层循环通过 FFI 回调到 Rust 的 `dispatch_pre_node()`。

Mega-kernel 路径中，JIT 编译期根据 `MegaKernelBusinessConfig.semantic_gatekeeper` 配置，在 CompilerGraph 中插入 `SgDetect`/`SgInject` OpKind，这些 OpKind 被 JIT codegen 编译为直接操作共享内存的机器码 — 无 Rust 回调。

#### 7.4.2 共享内存布局 (hook_ctx_ptr)

`hook_ctx_ptr`（ABI arg 15）指向以下结构体，由 Rust 侧在每次 `generate_single_sequence` 调用前填充：

```rust
#[repr(C)]
pub struct SgSharedMemory {
    /// SG 控制标志
    /// bit 0: sg_enabled (1=激活, 0=不注入, SgDetect/SgInject 编译为 NOP)
    /// bit 1-31: reserved
    pub control: u32,
    /// 当前检测层的检测偏移 (由 Rust 侧 KnowledgeProvider 写入)
    pub knowledge_offset: u32,
    /// 知识向量维度 (= hidden_size)
    pub knowledge_dim: u32,
    /// 知识置信度 (IEEE 754 f32 位模式, 0.0 表示无注入)
    pub confidence: u32,
    /// 检测层提取的 hidden state (由 JIT SgDetect 写入, Rust 侧读)
    pub detect_hidden: [f32; 0], // 动态长度 = hidden_size
    /// 知识残差向量 (由 Rust 侧 KnowledgeProvider 写入, JIT SgInject 读取)
    pub knowledge_vector: [f32; 0], // 动态长度 = hidden_size
}
```

#### 7.4.3 Mega-Kernel SG 数据流

```
每次 generate_single_sequence 调用:

1. Rust 侧 (generate 前准备):
   a. Client 调用 register_semantic_gatekeeper() → 设置 sg_config
   b. Executor.generate_with_sampling() 准备 SgSharedMemory:
      - sg_enabled = 1
      - knowledge_dim = hidden_size
      - 分配 detect_hidden + knowledge_vector 缓冲

2. Mega-Kernel JIT 层循环 (每层):
   a. [if SgDetect layer] SgDetect OpKind:
      - 从当前 activation 提取最后 token hidden → 写入 detect_hidden
      - release 内存序确保 Rust 可见
   b. 层循环末尾检查:
      - CMP sg_shared->control, 0 → JE .skip_sg (零开销快速路径)

3. Mega-Kernel generate 循环 (每 decode step):
   a. 每次 generate loop 迭代开始:
      - Rust 通过 hook_ctx_ptr 读 detect_hidden
      - KnowledgeProvider.retrieve() 生成 knowledge_vector
      - 写入 knowledge_vector + confidence 到 SgSharedMemory
   b. 下一层循环迭代的 SgInject 读取 knowledge_vector:
      - hidden[-1] += alpha * confidence * knowledge_vector
      - acq 内存序确保读到最新 knowledge_vector

4. Rust 侧 (generate 后清理):
   - SgSharedMemory 随 scratchpad 生命周期结束
```

#### 7.4.4 NO_ISLAND_MODULE 验证

Mega-kernel 路径的 NO_ISLAND_MODULE 验证通过 `pre_node_detection_layer_count` 原子计数器实现：

1. **非 mega-kernel 路径**: `SemanticGatekeeperCallback.pre_node()` 中的 `AtomicUsize` 递增
2. **Mega-kernel 路径**: SgDetect OpKind 在 JIT 编译期静态存在（由 CompilerGraph 拓扑决定），运行时不需要动态计数器

E2E 测试通过以下方式验证 mega-kernel 路径:
- 验证 `hook_ctx_ptr` 非空（SG 配置传递到 mega-kernel ABI）
- 验证 JIT 编译的 CompilerGraph 包含 `SgDetect`/`SgInject` OpKind
- 验证 generate 输出在有 SG 时产生可测量差异

---

## 8. E2E 行为差异验证协议

### 8.1 NO_ISLAND_MODULE 合规

SG 是公开 SDK API，零非测试调用方是可接受的。但必须有 E2E 测试证明：

1. 注册 SG + 调用 `generate()` 真实走过 `SemanticGatekeeperCallback.pre_node`（至少一次命中）
2. **行为差异**：相同 query 在「注册了返回特定知识的 Provider」vs「未注册」下生成的 token 有可测量差异，且差异方向符合知识内容

### 8.2 E2E 测试用例（关联 `SPEC/01-REQUIREMENTS.md §12`）

| TEST ID | 场景 | 验收 |
|---------|------|------|
| TEST-SG-001 | Level Keys 预计算 | 加载 SmolLM2-135M + SG config，验证 `LevelKeysCache` 对每个 detection layer 填充 3 个非全零向量 |
| TEST-SG-002 | Q-tap 读写 | 注册 SG（无 Provider），调用 `generate`，验证 ring buffer 读出的 Q 与 CPU 参考 `q_proj(hidden)[-1]` 数值一致（L2 误差 < 1e-4） |
| TEST-SG-003 | 层级路由门控 | Mock Provider 始终返回 None；调用生成，验证 SG pre_node 被触发 ≥1 次，hidden_state 未被修改（cosine 相似度 = 1.0） |
| TEST-SG-004 | 残差注入行为差异 | Provider 返回固定文本 `"Paris"`；询问 `"Capital of France is"`；对比无 SG 与有 SG 输出 token 分布，验证 `"Paris"` logit 明显提升 |
| TEST-SG-005 | 稳定性追踪命中率 | 连续 decode 20 步相同语义上下文；验证 `active_state` 复用次数 ≥ 15（>75%） |
| TEST-SG-006 | AST 节点变更强制刷新 | 注册 Mock AstSentinel，第 10 步切换 node_kind；验证 state 刷新、FullCompute 重新触发 |
| TEST-SG-007 | KnowledgeProvider 置信度 | Provider 返回 confidence=0.0 的 entry；验证 `effective_alpha = 0`，hidden_state 未修改 |
| TEST-SG-008 | 多检测层 | detection_depths=[0.5, 0.75]；验证两个检测层都能独立触发注入 |

### 8.3 禁止的测试反模式

- ❌ 只验证"API 可调用"而不验证行为差异（违反 NO_ISLAND_MODULE 铁律）
- ❌ 用 mock Callback 替代真实 SemanticGatekeeperCallback
- ❌ 跳过 Q-tap 真实读取，用 Rust 手写 `q_proj` 替代

---

## 9. 部署形态

### 9.1 完全本地模式

所有组件驻留开发者本地机器：
- gllm 推理引擎（Rust binary）
- SG Callback（同进程）
- KnowledgeProvider 实现（同进程 LSP/LSH）

零网络延迟。适合离线、高隐私场景。

### 9.2 云端推理 + 本地知识库模式（推荐）

```
用户侧:
  LSP 扫描器 + LSH 索引 + 向量化（CPU） + 本地 HTTP/gRPC 服务

云端:
  gllm 推理引擎
  SemanticGatekeeperCallback 注册 HttpKnowledgeProvider
  → HttpKnowledgeProvider.retrieve() 发起回调到用户本地服务
```

**优势**：
- **零隐私风险**：代码与符号签名完全本地，云端无数据落地
- **零 GPU 显存压力**：向量库在用户机器
- **延迟容忍**：任务级代码生成本来就耗时分钟级，100-500ms 网络往返无显著影响
- **成本极低**：云端仅标准推理实例

### 9.3 SG Callback 对部署模式无感

`KnowledgeProvider` trait 是抽象接口。Provider 实现者自由选择本地 / 远程。SG 内核路由、路由、注入逻辑不变。

---

## 10. 与现有方法的对比

| 维度 | RAG | Constrained Decoding | Function Calling | **Semantic Gatekeeper** |
|------|-----|----------------------|------------------|--------------------------|
| 知识注入位置 | Prompt（输入端） | Logits（输出端） | 显式工具调用 | **Hidden States（推理中）** |
| 模型是否可忽略 | 是 | 否（强制） | 否 | **否（软偏置）** |
| 泛化能力 | 依赖检索 | 刚性无泛化 | 依赖工具描述 | **保留别名/新符号泛化** |
| 训练需求 | 无 | 无 | 可能微调 | **无** |
| 检索时机 | 生成前单次 | 每 Token | 模型决定 | **按需，隐藏状态主动触发** |
| 知识层级区分 | 无 | 无 | 工具级 | **L1/L2/L3 自适应** |
| 延迟开销 | 检索延迟 | 钳制计算 | API 调用 | **稳定性追踪下 <0.1%** |

---

## 11. 交叉引用

| 主题 | 位置 |
|------|------|
| 用户 API 定义 | `SPEC/04-API-DESIGN.md §7` |
| 内部实现架构 | `SPEC/04-API-DESIGN.md §8` |
| FusedAttentionLayer Q-tap 扩展 | `SPEC/08-EXECUTOR.md §4.2` |
| CallbackChain + InjectHidden | `SPEC/05-OPTIMIZATIONS.md §1 §2.9 §7 §8` |
| Epilogue 白嫖网络（Q-tap 基础设施） | `SPEC/05-OPTIMIZATIONS.md §5` |
| FusedGraphExecutor 执行模型 | `SPEC/03-GRAPH-IR.md §4` |
| SymDim 穿透协议 | `SPEC/DOCS/scheduling/symdim-threading-protocol.md` |
| JIT 编译缓存（小图编译一次常驻） | `SPEC/DOCS/scheduling/jit-cache-protocol.md` |
| 核心铁律 | `SPEC/00-PHILOSOPHY.md` |
| 需求清单 | `SPEC/01-REQUIREMENTS.md §12` REQ-SG-001..008 |
