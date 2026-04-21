# 执行引擎 (Executor)

> **SSOT 声明**: 本文档是 gllm 执行引擎结构、全 JIT 执行模型、FusedGraphExecutor 编排协议、Session 管理、模型热切换的唯一真源。
>
> **架构变更 (ARCH-FULL-JIT)**: 全 JIT 执行模型。所有模型（Decoder/Encoder/Reranker）统一通过 `FusedGraphExecutor` 的 REGISTER-VM JIT 管线执行。手写前向传播路径（bert_forward.rs / decoder_forward.rs）已废弃并物理删除。

## 1. Executor 结构

Executor 是推理引擎的核心，持有完整推理所需的全部组件。

```rust
pub struct Executor<B: Backend<E> + 'static, E: Element = f32> {
    // ── 核心运行时 ──────────────────────────────────────────
    pub backend: B,
    pub scheduler: PagedScheduler,
    pub batcher: ContinuousBatcher,
    pub observer: BasicObserver,
    pub policy: PolicyVariant,
    pub requests: HashMap<RequestId, RequestData>,

    // ── 模型数据 ──────────────────────────────────────────
    pub manifest: Arc<ModelManifest>,
    pub weights: WeightsHandle<B, E>,
    pub add_special_tokens: bool,
    pub model_config: ModelConfig,
    pub forward_config: GeneratorForwardConfig,
    pub kv_cache_config: KvCacheConfig,
    pub tokenizer: TokenizerHandle,

    // ── KV Cache ──────────────────────────────────────────
    pub kv_cache: Option<KvCacheDoubleBuffer>,
    pub kv_cache_slot: KvCacheSlot,
    pub memory_manager: GlobalMemoryManager,

    // ── 唯一执行路径 ─────────────────────────────────────
    pub graph_executor: FusedGraphExecutor,             // 全 JIT 图执行器（非 Option）

    // ── 注意力拓扑 ──────────────────────────────────────
    pub topology: AttentionTopology,

    // ── 优化子系统 ────────────────────────────────────────
    pub moe: Option<MoeSubsystem>,
    pub residual_bus: ResidualBus,
    pub profile_accumulator: ProfileAccumulator,
    pub hooks: Arc<RwLock<Vec<Box<dyn GenerationHook>>>>,
    pub epilogue: EpilogueSubsystem,
    pub last_epilogue_summary: Option<EpilogueBatchSummary>,
    pub page_headers: HashMap<PageId, KvPageHeader>,

    // ── 按需激活的优化子系统 ──────────────────────────────
    pub spec_state: Option<SpecDecodingState>,
    pub early_exit: Option<EarlyExitController>,
    pub rag: Option<LateFusionRag>,
    pub intent_config: Option<IntentConfig>,
    pub seq_histogram: SeqHistogram,
    pub golden_buckets: GoldenBucketRegistry,
}
```

**关键变更**: `graph_executor` 从 `Option<FusedGraphExecutor>` 改为 `FusedGraphExecutor`（非可选）。所有模型加载时必须成功编译 FusedGraph，否则返回 `ExecutorError::Compilation`。

### 1.1 MoeSubsystem

```rust
pub struct MoeSubsystem {
    pub route_config: ExpertRouteConfig,
    pub thermal_manager: ExpertThermalManager,
    pub weight_prefetcher: ExpertWeightPrefetcher,
    pub hardware_dispatcher: MoeHardwareDispatcher,
    pub hot_patch_manager: HotPatchManager,
    pub load_balancer: ExpertLoadBalancer,
}
```

### 1.2 GeneratorForwardConfig

```rust
pub struct GeneratorForwardConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rope_scale: f64,
    pub rope_interleaved: bool,
    pub rope_precompute: bool,
    pub position_encoding: PositionEncoding,
    pub arch_family: ArchFamily,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub rerank_yes_token_id: Option<u32>,
    pub rerank_no_token_id: Option<u32>,
    pub moe_config: Option<MoEConfig>,
    pub dtype: DType,
    pub paged_kv_page_table: Option<Vec<u32>>,
    pub paged_kv_page_size: usize,
}
```

**变更**: 删除 `graph_executor_ptr: *mut FusedGraphExecutor`。不再需要裸指针传递——`graph_executor` 是 Executor 的固定成员。

### 1.2.1 Effective KV max_seq_len (ARCH-KV-EFFECTIVE-MAXSEQ)

`GeneratorForwardConfig::max_seq_len()` 与 `KvCacheConfig::max_seq_len()` 返回的是 KV cache 的**有效**长度上界，必须等于 JIT 编译时 baked 进 attention kernel 的 `compile_seq_len`（由 `SYMDIM_MAX_SEQ_LEN` 统一定义，当前值 `2048`）。

**单一真源**:

```rust
#[inline]
pub fn effective_kv_max_seq_len(geometry_max_seq_len: usize) -> usize {
    geometry_max_seq_len.min(SYMDIM_MAX_SEQ_LEN)
}
```

**为什么必须 cap**:
- 现代 LLM 的 `max_position_embeddings` 动辄 10 万级（Gemma 4 E2B: 131 072; Kimi-K2: 200 000+），未 cap 时会在加载阶段分配数 GB 全零 KV cache（Gemma 4 E2B: 15 effective layers × 1 KV head × 131 072 × 256 head_dim × 4 B × 2 = 3.84 GB memset）。
- JIT 发射的 `emit_multi_head_attention` 把 `compile_seq_len * kv_dim * 4` 作为 `V_ptr` 的编译时常量偏移（见 `gllm-kernels/src/compiler/graph.rs`）。若 KV cache stride 小于该常量 → JIT 越界读；若大于 → 浪费内存但 JIT 读不到超出 stride 的位置。
- `merge_kv_cache_for_decode` / `perform_kv_cache_write` 的 `layer_byte_offset = effective_kv_layer × num_kv_heads × cache_max_seq × head_dim × elem_bytes` 必须使用同一 `cache_max_seq`，否则各层 K/V 写入位置漂移。

**禁止**:
- 直接使用 `geometry.max_seq_len`（未 cap 的原始 `max_position_embeddings`）作为 KV cache 维度
- 在 `KvCacheBuffer::new` / `merge_kv_cache_for_decode` / `perform_kv_cache_write` 中分别计算 cap（违反 SSOT）

### 1.2.2 SharedKvRef Effective Layer Remapping (ARCH-KV-EFFECTIVE-LAYER)

Gemma 4 E2B / E4B 的后 `num_kv_shared_layers` 层共享前一个同类型（sliding / global）非共享层的 KV cache（见 §3 SharedKvRef）。因此 KV cache buffer 只分配 `effective_kv_layers = num_layers - num_kv_shared_layers` 个物理层槽。

所有直接寻址 KV buffer 的调用点必须将原始 `layer_index` 映射到 `effective_kv_layer`：

```rust
pub fn effective_kv_layer(&self, layer: usize) -> usize {
    let shared_start = self.num_layers.saturating_sub(self.num_kv_shared_layers);
    if layer < shared_start {
        return layer.min(self.effective_kv_layers().saturating_sub(1));
    }
    // Shared tail: find nearest preceding same-type non-shared layer.
    let this_type = self.attention_pattern.get(layer).copied().unwrap_or(0);
    for j in (0..shared_start).rev() {
        if self.attention_pattern.get(j).copied().unwrap_or(0) == this_type {
            return j;
        }
    }
    self.effective_kv_layers().saturating_sub(1)  // clamp
}
```

**调用点** (必须使用 `effective_kv_layer`，不是原始 layer index):
- `FusedGraphExecutor::merge_kv_cache_for_decode`
- `FusedGraphExecutor::perform_kv_cache_write`
- `KvCacheBuffer::layer_kv_offset`

**越界保护**: 所有计算 `layer_byte_offset` 的位置必须附 `debug_assert!(effective_kv_layer < effective_kv_layers())`，防止 SharedKvRef 元数据错误导致 buffer 越界写。

### 1.3 ExecutorError

```rust
pub enum ExecutorError {
    Backend(BackendError),
    Config(ModelConfigError),
    Loader(LoaderError),
    Tokenizer(TokenizerError),
    KvCache(KvCacheError),
    MemoryManager(MemoryManagerError),
    Scheduler(String),
    Compilation(String),           // JIT 编译失败
    GraphExpansion(String),        // FusedGraph → CompilerGraph 展开失败
    EmptyPrompt,
    EmptySample,
    RequestNotFound { request_id: RequestId },
}
```

**变更**: 删除 `OnnxPlan`（ONNX 走同一管线）。新增 `GraphExpansion`（图展开错误）。

## 2. Backend 统一接口

四后端统一：CPU / CUDA / HIP / Metal。

> Backend 提供**内存管理**和**同步原语**。前向传播**不再是 Backend 的方法**——所有计算由 `FusedGraphExecutor` 驱动，Backend 只负责 buffer 分配和 device 同步。

### 2.1 后端检测

```rust
// 优先级: CUDA > ROCm > Metal > CPU
// 失败时返回 Err，禁止 panic
let backend = detect_backend()?;
```

### 2.2 Backend Trait 核心能力

| 能力 | 方法 | 说明 |
|------|------|------|
| 内存分配 | `alloc(n, dtype)` | 64 字节对齐 |
| 数据上传 | `upload_weights(src)` | CPU→GPU 或零拷贝 |
| 数据下载 | `download_f32(src, dst)` | 仅在最终输出时使用 |
| KV Cache | `alloc_kv_cache(batch, seq)` | 分页分配 |
| 采样 | `sample(logits, temp, top_k, top_p)` | 硬件原生 |
| 同步 | `sync()` | 等待异步操作完成 |

**废弃方法**: `decoder_forward()`、`encoder_forward()`、`embedding_forward_gpu_pure()` 全部删除。Backend 不再包含前向传播逻辑。

### 2.3 废弃的手写前向传播文件

| 文件 | 处理 | 理由 |
|------|------|------|
| `compat/bert_forward.rs` | **物理删除** | FusedGraphExecutor 统一处理 Encoder |
| `compat/decoder_forward.rs` | **物理删除** | FusedGraphExecutor 统一处理 Decoder |
| `compat/cpu_backend.rs` 中的 forward 分发 | **简化** | 只保留内存管理 + 采样 |

## 3. Executor::step() 生命周期

```
step()
 ├── 1. schedule()        — 调度器决策
 ├── 2. build_batch()     — 组装混合批次
 ├── 3. forward()         — FusedGraphExecutor 执行（唯一路径）
 │    └── graph_executor.run_with_kv_cache_and_callbacks()
 │         ├── 逐节点 JIT kernel launch
 │         ├── KV Cache 自动更新
 │         └── Callback chain（§9-§18 优化模块）
 ├── 4. sample()          — 从 logits 采样（仅 Generator）
 ├── 5. update()          — 更新请求状态
 └── 6. epilogue_ingest() — Epilogue 遥测收集 + 决策
```

**变更**: `forward()` 不再根据 `graph_executor_ptr` 分支。唯一路径 = `graph_executor.run_*()`。

## 4. 全 JIT 前向传播 (ARCH-FULL-JIT)

### 4.0 设计原则

| 铁律 | 说明 |
|------|------|
| **ARCH-FULL-JIT** | 所有算子走 REGISTER-VM JIT 管线。零手写前向传播代码。 |
| **ARCH-NO-EMPTY-GRAPH** | `build_node_graph()` 禁止返回空 CompilerGraph（output_numel=0）。每个 FusedNode 必须产生有效的 JIT 内核。 |
| **ARCH-SHAPE-ELIM** | Shape/Reshape 等元数据操作在编译时常量折叠或作为零成本视图处理，不产生运行时代码但保持正确的形状传播。 |
| **ARCH-GATHER-JIT** | Gather（embedding lookup）必须编译为 JIT 索引加载内核，不允许 CPU 手写循环。 |

### 4.1 FusedGraphExecutor 全图编排

```
模型加载
 ├── YAML 模板 → OnnxGraph → GraphOptimizer → FusedGraph
 │   或 ONNX protobuf → OnnxGraph → GraphOptimizer → FusedGraph
 │   或 GGUF metadata → arch resolve → YAML → FusedGraph
 ├── FusedGraph.bind_weights(weights)   ← 权重零拷贝绑定
 └── FusedGraphExecutor::compile()      ← 全图 JIT 编译
      └── for each FusedNode:
          ├── build_node_graph(node) → CompilerGraph  (output_numel > 0 不变量)
          ├── InferenceCompiler::compile_graph(graph) → CompiledLayer
          └── CompiledNode { compiled, shapes, ... }

推理执行
 └── graph_executor.run_with_kv_cache_and_callbacks()
      └── for each CompiledNode (拓扑序):
          ├── callback: pre_node (Gate Skip / RAG Inject / ...)
          ├── load_activation(tensors) → input buffer
          ├── pack_weight_blob(tensors) → weight buffer
          ├── alloc output_buf (output_numel × dtype.size_bytes)  ← 保证 > 0
          ├── compiled.execute(input, weights, kv_cache, output, scratchpad)
          ├── store outputs → tensor map
          └── callback: post_node (Epilogue / Telemetry / ...)
```

### 4.2 build_node_graph 全算子覆盖

**每种 FusedOp 必须产生有效的 CompilerGraph。禁止返回空图。**

| FusedOp | CompilerGraph 展开 | output_numel |
|---------|-------------------|--------------|
| `FusedAttentionLayer(cfg)` | RmsNorm → Gemm(Wq) → Gemm(Wk) → Gemm(Wv) → RoPE(Q) → RoPE(K) → Attention(Q,K,V) → Gemm(Wo) | seq_len × hidden_size |
| `FusedAttentionLayer(cfg) with q_tap` | 同上 + `q_proj` GEMM 后尾段 STG 指令写 GatekeeperRingBuffer（见 §4.2.1） | 同上（tap 仅副作用写，不改主输出） |
| `FusedFfnLayer(cfg)` | RmsNorm → Gemm(gate) → Silu → Gemm(up) → Mul → Gemm(down) | seq_len × hidden_size |
| `FlashAttention(cfg)` | Reshape(Q) → Reshape(K) → Reshape(V) → MatMul(Q,K^T) → Softmax → MatMul(attn,V) → Reshape(out) | seq_len × num_heads × head_dim |
| `GQA(cfg)` | 同 FlashAttention + scale + causal mask | seq_len × num_heads × head_dim |
| `SwiGLU(cfg)` | Gemm(gate) → Silu → Gemm(up) → Mul → Gemm(down) | seq_len × hidden_size |
| `RoPE(cfg)` | RoPE(Q, positions) | seq_len × head_dim |
| `FusedQkvRope(cfg)` | Gemm(Wq) → RoPE → Gemm(Wk) → RoPE → Gemm(Wv) | seq_len × (q_dim + 2 × kv_dim) |
| `FusedRMSLinear(cfg)` | RmsNorm → Gemm | seq_len × output_dim |
| `MoERouting(cfg)` | Gemm(gate) → Softmax → TopK → [expert branches] → Add | seq_len × hidden_size |
| `PerLayerEmbed(cfg)` | Slice(ple_weight) → Linear → Add | seq_len × hidden_size |
| **`Atomic("Gather")`** | **`OpKind::Gather { table_rows, embed_dim }`** | **seq_len × embed_dim** |
| **`Atomic("Slice")`** | **`OpKind::SliceView { axis, start, end }`** | **根据 slice 参数推导** |
| **`Atomic("Shape")`** | **编译时常量折叠 → `OpKind::Reshape`** | **输入 numel（恒等映射）** |
| `Atomic("MatMul")` | Gemm(m, n, k) | m × n |
| `Atomic("Add")` | Add | max(input_a, input_b) 逐元素广播 |
| `Atomic("Mul")` | Mul | 同上 |
| `Atomic("LayerNorm")` | LayerNorm(hidden, eps) | seq_len × hidden |
| `Atomic("RmsNorm")` | RmsNorm(hidden, eps) | seq_len × hidden |
| `Atomic("Softmax")` | Softmax | 同输入 |
| `Atomic("MeanPool")` | MeanPool(seq_len, hidden) | 1 × hidden |
| `Atomic("L2Normalize")` | L2Normalize(hidden) | 同输入 |

#### 4.2.1 FusedAttentionLayer Q-Tap 扩展 (ARCH-SG-QTAP)

> **关联**: `SPEC/SEMANTIC-GATEKEEPER.md §4`（Q 截获协议）

`FusedAttentionLayerConfig` 扩展可选字段 `q_tap: Option<QTapConfig>`。仅 Semantic Gatekeeper 注册时的"检测层"FusedAttentionLayer 节点携带 `Some(...)`，其他层保持 `None`，零额外指令开销。

```rust
pub struct QTapConfig {
    /// Gatekeeper ring buffer 的设备可见指针
    pub sink_ptr: u64,
    /// 要写出的 token 位置
    pub tap_position: QTapPosition,
    /// 写出 dtype（默认 = q_proj 输出 dtype）
    pub dtype: DType,
}

pub enum QTapPosition {
    LastToken,    // 仅最后一个 token（decode step）
    AllTokens,    // 全序列（prefill）
}
```

**JIT codegen 行为**（ARCH-CPU-GPU-UNIFIED 合规，全后端统一）：

- 编译期 `q_tap: Some(cfg)` → Phase 3 codegen 在 `q_proj` GEMM 的尾段追加 STG 指令写 `cfg.sink_ptr`
- 主计算路径（`q_proj` → RoPE → Attention）不变，tap 仅是副作用写
- Ring buffer 双缓冲 + atomic `step_index`（`release` 写 / `acquire` 读），防止 `pre_node` 回调读到陈旧值
- 编译期 `q_tap: None` → codegen 不生成 tap 指令（零开销）

**不允许的做法**：
- ❌ 拆分 FusedAttentionLayer 为子节点以暴露 Q（违反 ARCH-CPU-GPU-UNIFIED "禁止子算子级 GraphType"）
- ❌ 引入 Q-tap 专用的新 `GraphType` 变体（同上）
- ❌ 为 Q-tap 单独分配 backend-specific 的辅助 buffer（违反 CPU/GPU 统一路径）

### 4.3 Gather JIT 内核 (ARCH-GATHER-JIT)

Embedding lookup 必须通过 JIT 编译为索引加载内核。

```
输入: token_ids (seq_len,)  +  embed_table (vocab_size × embed_dim)
输出: embeddings (seq_len × embed_dim)

JIT 逻辑 (Phase 3 codegen):
  for i in 0..seq_len:
    row = token_ids[i]
    memcpy(output[i * embed_dim .. (i+1) * embed_dim],
           embed_table[row * embed_dim .. (row+1) * embed_dim])
```

**CompilerGraph OpKind**:

```rust
OpKind::Gather {
    table_rows: usize,    // vocab_size
    embed_dim: usize,     // embedding dimension
    index_dim: SymDim,    // seq_len (Symbolic)
}
```

**VM 管线处理**:
- Phase 0: `scalar_gather()` 参考实现 → SymExec → OpTrace
- Phase 1: SemanticDAG → `OpClass::Injective`（逐元素索引查找）
- Phase 2: Fusion → Standalone（Gather 不参与融合）
- Phase 3: lower → `emit_loop { VecLoad(index) → VecLoad(table[index]) → VecStore(output) }`

### 4.4 Shape/Slice 编译时处理 (ARCH-SHAPE-ELIM)

| 操作 | 编译时行为 | 运行时行为 |
|------|-----------|-----------|
| Shape | 常量折叠为具体值。CompilerGraph 中不产生 op。output_numel = 输入 numel（恒等传递） | 零运行时开销 |
| Slice | 转换为 `SliceView` 指针偏移。不拷贝数据。output_numel = slice 范围大小 | 零拷贝视图 |
| Reshape | 转换为 `Reshape` 元数据 op。JIT codegen 生成 NOP（已授权 NO_SILENT_FALLBACK 例外） | 零拷贝视图 |

```rust
OpKind::SliceView {
    axis: usize,
    start: usize,
    end: usize,
}
```

**output_numel 推导规则**: `input_shape[axis]` 替换为 `end - start`，其他维度不变。

### 4.5 三种模型类型的统一执行

| 模型类型 | FusedGraph 结构 | 特殊处理 |
|---------|----------------|---------|
| **Decoder (Generator)** | [Gather(embed)] → N × [FusedQkvRope → GQA → SwiGLU] → [FusedRMSLinear(lm_head)] | KV Cache 逐层更新 |
| **Encoder (Embedding)** | [Gather(embed)] → N × [FlashAttention → SwiGLU] → [MeanPool → L2Norm] | 无 KV Cache，双向注意力 |
| **Reranker** | [Gather(embed)] → N × [FlashAttention → SwiGLU] → [Slice(CLS) → Linear] | 无 KV Cache，CLS 提取 |

**三种类型共享同一执行路径**: `graph_executor.run_with_kv_cache_and_callbacks()`。
- Encoder/Reranker: `kv_cache_k = null`, `kv_cache_v = null`（无 KV Cache）。
- Decoder: KV Cache 指针非空，逐层更新。

### 4.6 output_numel 推导协议

**不变量**: 对任何 FusedNode，`build_node_graph()` 返回的 `output_numel` 必须 > 0。

推导规则：
1. 从 FusedNode 的输入形状和配置参数推导输出形状
2. 动态维度使用 `SymDim::Symbolic` 的 `max_value` 上界分配 buffer
3. 运行时实际使用的 numel 可以 < 分配的 buffer 大小（过量分配是安全的，不足分配导致 SIGSEGV）

```rust
// 推导优先级
match &node.op {
    FusedOp::FlashAttention(c) => max_seq_len * c.num_heads * c.head_dim,
    FusedOp::SwiGLU(c) => max_seq_len * c.hidden_size,
    FusedOp::Atomic(a) if a.op_type == "Gather" => {
        let embed_dim = a.attributes.get("embed_dim").unwrap();
        max_seq_len * embed_dim  // ← 保证 > 0
    }
    FusedOp::Atomic(a) if a.op_type == "Shape" => {
        // Shape 恒等传递：output = input
        input_numel  // ← 保证 > 0（来自前驱节点）
    }
    // ... 其他 FusedOp 类推
}
```

### 4.7 ABI 契约

CompiledLayer 的执行函数签名必须与 VM codegen 的 prologue 对齐。

```rust
pub type CompiledLayerFn = unsafe extern "C" fn(
    input_ptr: *const u8,     // rdi: 激活输入
    weight_ptr: *const u8,    // rsi: 权重
    kv_cache_ptr: *mut u8,    // rdx: KV Cache（Encoder 传 null）
    positions_ptr: *const u32,// rcx: 位置数组
    aux_ptr: *const u8,       // r8: 辅助指针（KV-V half / telemetry）
    batch_size: usize,        // r9: 批大小
    // ── 栈参数 ──
    seq_len: usize,           // [rsp+0]: 序列长度
    output_ptr: *mut u8,      // [rsp+8]: 输出缓冲区
    scratchpad_ptr: *mut u8,  // [rsp+16]: 临时缓冲区
    telemetry_ptr: *mut u8,   // [rsp+24]: 遥测缓冲区
);
```

**调用方保证**：
- `output_ptr` 指向至少 `output_numel × dtype.size_bytes()` 字节的可写内存
- `scratchpad_ptr` 指向至少 `compiled.scratchpad_bytes` 字节的可写内存
- Encoder 模型传 `kv_cache_ptr = null`, `positions_ptr = null`

## 5. Callback 构建流程

`build_step_callbacks()` 根据配置注册优化模块回调。每个 Callback 在 `run_with_kv_cache_and_callbacks()` 的逐节点循环中被调用。CallbackAction、LayerCallback trait 和 CallbackChain 完整定义见 `SPEC/05-OPTIMIZATIONS.md` §1。

## 6. Session 多轮 KV Cache 复用

Session API 支持跨轮次 KV Cache 确定性复用。`SessionKvCache` 底层数据结构和隔离约束定义见 `SPEC/06-RUNTIME.md` §4.3。

```rust
let session_id = executor.register_session();
loop {
    executor.claim_session_prefix(session_id, &new_tokens)?;
    // ... forward ...
    executor.finalize_session_tokens(session_id, new_tokens.len());
}
```

`finalized_position` 单调递增约束：禁止 claim 超过已确认边界。

## 7. 模型热切换

Stop-the-World 流程，基于 `Arc<RwLock<Option<Executor>>>`。

```rust
impl Client {
    pub fn swap_model(&self, new_model_id: &str) -> Result<()> {
        let mut guard = self.executor.write().unwrap();
        *guard = None;
        let new_executor = Executor::from_loader(/* ... */)?;
        *guard = Some(new_executor);
        Ok(())
    }
}
```

## 8. GPU 数据通路优化

### 8.1 KV Scatter Kernel

`OpKind::KvScatterWrite` + PTX/HIP/MSL codegen。grid = (num_kv_heads, seq_len)，block = (head_dim)。

### 8.2 权重常驻缓存

`GpuWeightCache` 持有 per-layer GPU buffer。首次 forward htod 上传 + 缓存 device_ptr，后续 DtoD。

### 8.3 Metal KV 直写

`metal_write_kv_direct()` 通过 shared memory 指针直写。

### 8.4 PagedKvView 三后端统一

Paged attention 支持 CUDA / HIP / Metal 三后端统一接口。

## 9. 废弃清单 (ARCH-FULL-JIT 迁移)

| 废弃项 | 替代 | 理由 |
|--------|------|------|
| `compat/bert_forward.rs` | `FusedGraphExecutor` | Encoder 模型走统一 JIT 路径 |
| `compat/decoder_forward.rs` | `FusedGraphExecutor` | Decoder 模型走统一 JIT 路径 |
| `graph_executor_ptr: *mut` | `graph_executor: FusedGraphExecutor` 固定成员 | 消除裸指针，消除 Option 分支 |
| `onnx_generator_plan` | ONNX → FusedGraph → JIT | ONNX 模型走同一管线 |
| `OnnxPlan` error variant | `GraphExpansion` | 统一错误类型 |
| `build_node_graph` 返回空图 | 所有 FusedOp 产生有效 CompilerGraph | ARCH-NO-EMPTY-GRAPH |
| Backend 的 `decoder_forward` 方法 | `graph_executor.run_*()` | Backend 只管内存 |
