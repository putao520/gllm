# 31-EXECUTOR-DECOMPOSITION — Executor 分解架构

**REQ 覆盖**: REQ-DECOMP-001~008

---

## §0 动机与约束

### §0.1 问题

Executor 累积 53 字段横跨 8 个概念子系统（调度、KV Cache、JIT/遥测、MoE、SG/Callback、推测解码/RAG/路由、权重/内存、模型/IO）。`step()` 交织 20 个阶段，每阶段访问 3-8 个子系统。存在裸指针绕过 Rust 所有权（`unsafe impl Send/Sync`），`Option<T>` 字段产生指数级潜在状态组合。

### §0.2 根因

Executor 累积 53 字段横跨 8 个概念子系统（调度、KV Cache、JIT/遥测、MoE、SG/Callback、推测解码/RAG/路由、权重/内存、模型/IO）。每次 `step()` 调用交织 20 个阶段，每阶段访问 3-8 个子系统，形成极其复杂的数据流图。

### §0.3 为什么先拆分再做 SPEC/20/21

1. SPEC/20 §0.1 明文："batch_size=1 是 batch_size=N 的特例"——意味着 step() 应被 generate_batch 统一，不是再膨胀
2. 当前 generate_batch 和 step() 并存 = 两条永不合并的执行路径，违反 SPEC/20 统一前提
3. SPEC/21 (权重分页) 必然落入内存域——先拆分能让 WP-001~010 单点改动在 KvCoordinator 内，否则又往 step() 塞 §21 代码
4. 不拆分：每个新 SPEC 往 step() 塞大量逻辑 → 线性恶化；先拆分：新 SPEC 落在子系统内 → 边际成本归零

### §0.4 约束

- **REQ-DECOMP-001**: 零行为变更。纯结构重构。
- **REQ-DECOMP-002**: 拆分后满足 CLAUDE.md 红线：文件 ≤2000 行，函数 ≤50 行，嵌套 ≤3，圈复杂度 ≤10。
- **REQ-DECOMP-003**: 所有现有测试不经修改即通过。
- **REQ-DECOMP-004**: `generate_batch()` 是架构目标——每个 coordinator 方法应是自包含单元。
- **REQ-DECOMP-005**: SPEC/20 BCI 自然统一在 `DispatchCoordinator.build_batch()` 中。
- **REQ-DECOMP-006**: 消除 `unsafe impl Send/Sync`，用 `Arc` 替换裸指针。

---

## §1 目标架构

### §1.1 Executor 瘦编排器 (7 字段)

```rust
pub struct Executor<B: Backend<E> + 'static, E: Element = f32> {
    backend: B,
    dispatch: DispatchCoordinator,
    kv: KvCoordinator,
    compute: ComputeCoordinator,
    inference: InferenceCoordinator,
    model: ModelContextHolder,
    observer: ObservabilityCoordinator,
}
```

从 53 字段缩减到 7（backend + 6 coordinator）。每个 coordinator 拥有其子系统字段并暴露方法。

### §1.2 step() 目标

```rust
pub fn step(&mut self) -> ExecutorResult<()> {
    self.kv.drain_swap_completions();
    self.kv.check_memory_pressure()?;

    let system_state = self.observer.capture_system_state(
        &self.backend, &self.dispatch)?;
    let decision = self.dispatch.decide_strategy(&system_state);

    let batch = self.dispatch.build_batch(&decision);
    if batch.is_empty() { return Ok(()); }
    let spec_advice = self.inference.should_speculate(batch.decode_count());
    let (batch_input, request_indices) =
        self.dispatch.prepare_inputs(&batch, &self.compute, &self.inference, &self.model)?;

    let moe_plan = self.inference.dispatch_moe(
        &self.model.geometry, &self.backend, self.compute.mega_kernel())?;
    self.inference.prefetch_experts(&moe_plan, &mut self.dispatch)?;
    self.compute.record_and_evolve(&batch_input);

    let callbacks = self.build_callback_chain()?;
    let (logits_list, sparsity, telemetry) =
        self.compute.run_forward(&batch_input, &mut self.backend, &self.model, callbacks)?;
    self.compute.advance_turboquant(&self.model.geometry, &self.kv.config());

    self.observer.update_forward_metrics(...)?;
    self.compute.push_telemetry(&telemetry, self.inference.moe_thermal(), &self.model.geometry);
    let epilogue = self.compute.epilogue_decide(&telemetry);
    self.inference.process_gate_skip(&epilogue, &request_indices);
    self.kv.optimize_tiers(&self.compute.layer_headers(&telemetry), ...);

    let (results, spec_extra) = self.process_results(
        &logits_list, &telemetry, &request_indices, &epilogue, spec_advice)?;

    self.kv.advance_cache(total_tokens, spec_extra);
    self.dispatch.finalize_batch(&results);
    self.kv.optimize_kv_cache(...);
    Ok(())
}
```

---

## §2 六个 Coordinator 定义

### §2.1 DispatchCoordinator (调度 + 分发)

**字段** (6):
```rust
pub struct DispatchCoordinator {
    scheduler: PagedScheduler,
    batcher: ContinuousBatcher,
    chunked_prefill_scheduler: ChunkedPrefillScheduler,
    requests: HashMap<RequestId, RequestData>,
    memory_manager: GlobalMemoryManager,
    policy: PolicyVariant,
}
```

**方法**:

| 方法 | 职责 |
|------|------|
| `decide_strategy(&self, state: &SystemState) -> Decision` | 策略仲裁 |
| `build_batch(&mut self, decision: &Decision) -> InterleavedBatch` | 交织调度 |
| `plan_prefill(&mut self, batch: &InterleavedBatch, geometry: &ModelGeometry) -> Result<()>` | 自适应 chunk + 页分配 |
| `prepare_inputs(&mut self, batch: &InterleavedBatch, compute: &ComputeCoordinator, inference: &InferenceCoordinator, model: &ModelContextHolder) -> Result<(BatchInput, Vec<RequestId>)>` | 序列输入构建 |
| `finalize_batch(&mut self, results: &[BatchResult])` | batcher 更新 |
| `has_pending_work(&self) -> bool` | 查询 |

**SPEC/20 BCI 统一**: `build_batch()` 已处理任意 decode + prefill token 计数的 `InterleavedBatch`。SPEC/20 完整实现后，`step()` 退化为 `generate_batch(N=1)` 的薄壳。

---

### §2.2 KvCoordinator (KV Cache + 内存层级)

**字段** (6):
```rust
pub struct KvCoordinator {
    kv_cache: Option<KvCacheDoubleBuffer>,
    kv_cache_slot: KvCacheSlot,
    kv_cache_config: KvCacheConfig,
    paged_kv_pool: Option<PagedKvPool>,
    kv_optimizer: KvOptimizer,
    majority_kv_tier: Option<String>,
}
```

**方法**:

| 方法 | 职责 |
|------|------|
| `check_memory_pressure(&mut self) -> Result<()>` | SwiftKV 蒸馏 + 三级换入换出 |
| `drain_swap_completions(&mut self)` | HGAL state sync |
| `optimize_tiers(&mut self, headers: &[Vec<KvPageHeader>], telemetry: &TelemetryAggregator, geometry: &ModelGeometry)` | 跨层 KV 优化 |
| `advance_cache(&mut self, total_tokens: usize, spec_extra: usize)` | KV cache slot 推进 |
| `optimize_kv_cache(&mut self, decode_pages: &[u32], geometry: &ModelGeometry, telemetry: &TelemetryAggregator) -> usize` | 重量化 decode pages |
| `active_handle(&self) -> Result<KvCacheHandle>` | 当前 KV handle |

**SPEC/21 落位**: 权重分页 (WP-001~010) 全部在 KvCoordinator 内完成，零侵入 step()。

---

### §2.3 ComputeCoordinator (JIT + 遥测 + Mega-Kernel)

**字段** (9):
```rust
pub struct ComputeCoordinator {
    mega_kernel: Option<MegaKernelExecutor>,
    jit_director: Option<JitDirector>,
    telemetry_aggregator: TelemetryAggregator,
    epilogue_subsystem: EpilogueSubsystem,
    sub_batch_dispatcher: SubBatchDispatcher,
    golden_buckets: GoldenBucketRegistry,
    seq_histogram: SeqHistogram,
    ragged_compaction: RaggedCompaction,
    turboquant: TurboQuantRuntime,
}
```

**方法**:

| 方法 | 职责 |
|------|------|
| `classify_shapes(&self, requests: &HashMap<RequestId, RequestData>, ...) -> HashMap<RequestId, SubBatchShape>` | Dead ratio, delta_rho, 形状分类 |
| `dispatch_sub_batches(&self, manifest: &mut BatchManifest, shape_map: &...) -> Option<DispatchPlan>` | Sub-batch 分发 |
| `record_and_evolve(&mut self, sequences: &[SequenceInput])` | 直方图 + golden bucket 演化 |
| `evaluate_compact(&self, manifest: &BatchManifest, config: &ChunkedPrefillConfig) -> CompactDecision` | Compact 决策 |
| `run_forward(&mut self, batch_input: &BatchInput, backend: &mut B, model: &ModelContextHolder, callbacks: Vec<Box<dyn LayerCallback + Send>>) -> Result<(Vec<LogitsHandle>, f32, Vec<SequenceTelemetry>)>` | 构建回调链 + 调用后端前向 |
| `advance_turboquant(&mut self, geometry: &ModelGeometry, kv_config: &KvCacheConfig)` | Per-channel scales, correction factors |
| `push_telemetry(&mut self, batch_tel: &[SequenceTelemetry], moe_thermal: Option<&ExpertThermalManager>, geometry: &ModelGeometry)` | JIT Director 遥测注入 + consensus 事件 |
| `epilogue_decide(&mut self, batch_tel: &[SequenceTelemetry]) -> EpilogueBatchSummary` | Epilogue 决策 |
| `layer_headers(&self, batch_tel: &[SequenceTelemetry], geometry: &ModelGeometry) -> Vec<Vec<KvPageHeader>>` | 遥测→页头映射 |

---

### §2.4 InferenceCoordinator (MoE + 推测解码 + RAG + 路由)

**字段** (12):
```rust
pub struct InferenceCoordinator {
    // MoE
    moe_thermal: Option<ExpertThermalManager>,
    moe_fault_handler: Option<ExpertFaultHandler>,
    moe_dispatcher: Option<MoeHardwareDispatcher>,
    moe_prefetcher: Option<ExpertWeightPrefetcher>,
    prefetch_pipeline: Option<PrefetchPipeline>,
    hot_patch_manager: Option<HotPatchManager>,
    expert_code_regions: HashMap<(usize, usize), (usize, usize)>,
    expert_saved_bytes: HashMap<(usize, usize), Vec<u8>>,
    // 推测/RAG/路由
    spec_decoding: SpecDecodingState,
    rag_system: Option<LateFusionRag>,
    residual_bus: ResidualBus,
    gate_skip_flags: HashMap<u64, bool>,
}
```

**方法**:

| 方法 | 职责 |
|------|------|
| `should_speculate(&self, decode_count: usize) -> SpecScheduleAdvice` | 推测解码决策 |
| `dispatch_moe(&mut self, geometry: &ModelGeometry, backend: &B, mega: Option<&MegaKernelExecutor>) -> Option<MoeDispatchPlan>` | 硬件分发 + 热度评估 + NOP patching |
| `prefetch_experts(&mut self, plan: &MoeDispatchPlan, dispatch: &mut DispatchCoordinator) -> Result<()>` | 专家权重预取 + tier 迁移 |
| `advance_prefetch_pipeline(&mut self, plan: Option<&MoeDispatchPlan>, kv_block_size: usize)` | Pipeline 推进 |
| `process_gate_skip(&mut self, summary: &EpilogueBatchSummary, request_indices: &[RequestId])` | Gate skip + bypass + prefetch advice |
| `process_spec_verification(&mut self, ...) -> usize` | Draft→verify 管线 |
| `moe_thermal(&self) -> Option<&ExpertThermalManager>` | 访问器 |

---

### §2.5 ModelContextHolder (模型数据 + SG Callbacks + 权重分页)

**字段** (16):
```rust
pub struct ModelContextHolder<B: Backend<E> + 'static, E: Element> {
    // 模型数据
    manifest: Arc<ModelManifest>,
    weights: WeightsHandle<B, E>,
    geometry: Arc<ModelGeometry>,
    model_config: ModelConfig,
    forward_config: GeneratorForwardConfig,
    tokenizer: TokenizerHandle,
    topology: AttentionTopology,
    add_special_tokens: bool,
    // 系统
    system_topology: SystemTopology,
    profile_accumulator: ProfileAccumulator,
    hooks: Arc<RwLock<Vec<Box<dyn GenerationHook>>>>,
    // SG/Callback (Arc 化, 消除 unsafe)
    sg_callback_shim: Option<Arc<SemanticGatekeeperCallbackShim>>,
    sg_ring_buffer: Option<Arc<GatekeeperRingBuffer>>,
    sg_shared_memory: Option<Arc<Mutex<SgSharedMemory>>>,
    callback_table: Arc<Mutex<MegaKernelCallbackTable>>,
    // 权重分页 + 三级换入换出
    weight_page_table: HashMap<usize, Vec<PhysicalId>>,
    weight_pages_registered: bool,
    three_tier_swap: Option<Arc<Mutex<ThreeTierSwapCoordinator>>>,
}
```

**关键变更**: `sg_callback_ctx_ptr: *const u8` 被消除。`Box::leak` 的 SgCallbackCtx 改为 `Arc<SgCallbackCtx>`。`callback_chain_ptr: *mut CallbackChain` 在 GeneratorForwardConfig 中改为 `Arc<Mutex<CallbackChain>>`。

**方法**:

| 方法 | 职责 |
|------|------|
| `register_sg_callback(&mut self, shim: SemanticGatekeeperCallbackShim)` | SG 注册 + Arc ctx |
| `register_weight_pages(&mut self, geometry: &ModelGeometry, scheduler: &mut PagedScheduler)` | WP-002/007 |
| `set_rag_system(&mut self, rag: LateFusionRag)` | RAG 设置 |
| `sample_from_logits(&self, logits: &LogitsHandle, config: &SamplingConfig) -> Result<u32>` | Token 采样 |

---

### §2.6 ObservabilityCoordinator (观测)

**字段** (1):
```rust
pub struct ObservabilityCoordinator {
    observer: BasicObserver,
}
```

**方法**:

| 方法 | 职责 |
|------|------|
| `capture_system_state(&mut self, backend: &B, dispatch: &DispatchCoordinator) -> SystemState` | 内存压力 + 碎片 + 调度指标 |
| `update_forward_metrics(&mut self, ...) -> Result<()>` | Logits 熵 + swap IO + MoE fault + 权重指标 |
| `last_state(&self) -> SystemState` | 访问器 |

---

## §3 ExecutorBuilder 模式

### §3.1 四阶段构造

当前 `from_loader()` 拆分为:

| 阶段 | 方法 | 职责 |
|------|------|------|
| 1 | `load_model()` | 解析模型配置, 创建 geometry/weights/tokenizer |
| 2 | `detect_system()` | SystemTopology 检测, LatencyProfiler probe |
| 3 | `compile_kernel()` | 构建 CompilerGraph, 编译 mega-kernel, 上传 GPU 产物 |
| 4 | `build_coordinators()` | 从模型 + 系统状态初始化 6 个 coordinator |
| 5 | `build()` | 组装 Executor |

每个阶段独立可测试。阶段 3 (最昂贵) 可缓存。

---

## §4 unsafe 消除 (REQ-DECOMP-006)

### §4.1 当前问题

```rust
// GeneratorForwardConfig (executor.rs)
pub callback_chain_ptr: *mut crate::graph::layer_callback::CallbackChain,
unsafe impl Send for GeneratorForwardConfig {}
unsafe impl Sync for GeneratorForwardConfig {}

// ModelContextHolder (coordinator/model_context.rs)
pub sg_callback_ctx_ptr: *const u8,

// Executor (executor.rs)
unsafe impl<B: Backend<E> + Send + 'static, E: Element + Send> Send for Executor<B, E> {}
unsafe impl<B: Backend<E> + Sync + 'static, E: Element + Sync> Sync for Executor<B, E> {}
```

存在原因: JIT FFI 回调需要 C ABI 裸指针（fn_ptr + ctx_ptr），无法接受 Rust Arc/Mutex。

### §4.2 解决方案: Safe Wrapper 类型

核心思路: FFI 边界必须有裸指针，但**主结构体不持有裸指针**。改用 safe wrapper 类型管理生命周期，只在 FFI 调用瞬间取裸指针。

#### §4.2.1 CallbackChainHandle — 替代 callback_chain_ptr

```rust
// coordinator/callback_slot.rs (新文件)
use std::sync::atomic::{AtomicPtr, Ordering};

/// Safe wrapper for CallbackChain FFI pointer.
/// Replaces `*mut CallbackChain` in GeneratorForwardConfig.
/// Thread-safe via AtomicPtr with Acquire/Release ordering.
pub struct CallbackChainHandle {
    inner: AtomicPtr<crate::graph::layer_callback::CallbackChain>,
}

impl CallbackChainHandle {
    pub fn new() -> Self {
        Self { inner: AtomicPtr::new(std::ptr::null_mut()) }
    }

    /// Temporarily borrow the chain as raw pointer for FFI call.
    /// Returns None if no chain is set.
    pub fn as_ffi_ptr(&self) -> *mut crate::graph::layer_callback::CallbackChain {
        self.inner.load(Ordering::Acquire)
    }

    /// Set the chain pointer. Only call when Executor mutex is held.
    pub fn set(&self, ptr: *mut crate::graph::layer_callback::CallbackChain) {
        self.inner.store(ptr, Ordering::Release);
    }

    /// Clear the pointer after FFI call returns.
    pub fn clear(&self) {
        self.inner.store(std::ptr::null_mut(), Ordering::Release);
    }
}

// Auto-derived: AtomicPtr is Send + Sync, no unsafe impl needed.
unsafe impl Send for CallbackChainHandle {}
unsafe impl Sync for CallbackChainHandle {}
```

**替换**: `GeneratorForwardConfig.callback_chain_ptr: *mut CallbackChain` → `callback_chain: CallbackChainHandle`

**FFI 调用模式变化**:
```rust
// 旧 (unsafe):
self.model_ctx.forward_config.callback_chain_ptr = &mut chain as *mut _;
result = ffi_call(..., self.model_ctx.forward_config.callback_chain_ptr);
self.model_ctx.forward_config.callback_chain_ptr = std::ptr::null_mut();

// 新 (safe wrapper):
self.model_ctx.forward_config.callback_chain.set(&mut chain as *mut _);
result = ffi_call(..., self.model_ctx.forward_config.callback_chain.as_ffi_ptr());
self.model_ctx.forward_config.callback_chain.clear();
```

**注意**: `unsafe impl Send/Sync` 仍保留在 CallbackChainHandle 上（AtomicPtr 本身需要），但 GeneratorForwardConfig **不再需要** unsafe impl——因为 CallbackChainHandle 的 unsafe impl 已封装在内部。

#### §4.2.2 SgCallbackHandle — 替代 sg_callback_ctx_ptr

```rust
// coordinator/sg_callback_handle.rs (新文件)
use std::pin::Pin;
use std::sync::Arc;

/// Safe wrapper for leaked SgCallbackCtx FFI pointer.
/// Replaces `*const u8` in ModelContextHolder.
/// Uses Pin<Box<>> for stable heap address guaranteed across moves.
pub struct SgCallbackHandle {
    inner: Option<Pin<Box<SgCallbackCtx>>>,
}

impl SgCallbackHandle {
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Register a new callback context. Returns the raw pointer for FFI callback table.
    pub fn register(&mut self, ctx: SgCallbackCtx) -> *const u8 {
        let pinned = Box::pin(ctx);
        let ptr = (&*pinned) as *const SgCallbackCtx as *const u8;
        self.inner = Some(pinned);
        ptr
    }

    /// Reclaim and drop the callback context. Returns the provider_state for Arc cleanup.
    pub fn reclaim(&mut self) -> Option<*const u8> {
        let pinned = self.inner.take()?;
        let ctx = Pin::into_inner(pinned);
        let provider_ptr = ctx.provider_state;
        // precomputed_knowledge is a leaked Box<[f32]>, reclaim it
        if !ctx.precomputed_knowledge.is_null() {
            let len = ctx.hidden_size as usize;
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    ctx.precomputed_knowledge as *mut f32, len
                );
                let _ = Box::from_raw(slice as *mut [f32]);
            }
        }
        // ctx dropped here, provider_state returned for Arc::from_raw
        Some(provider_ptr)
    }

    /// Check if a callback is registered.
    pub fn is_registered(&self) -> bool {
        self.inner.is_some()
    }
}
```

**替换**: `ModelContextHolder.sg_callback_ctx_ptr: *const u8` → `sg_callback_handle: SgCallbackHandle`

**注册模式变化**:
```rust
// 旧 (unsafe leak):
let cb_ctx = Box::new(SgCallbackCtx { ... });
let cb_ctx_ptr = Box::into_raw(cb_ctx) as *const u8;
unsafe { self.model_ctx.callback_table.register(slot, fn_ptr, cb_ctx_ptr); }
self.model_ctx.sg_callback_ctx_ptr = cb_ctx_ptr;

// 新 (safe Pin):
let cb_ctx = SgCallbackCtx { ... };
let cb_ctx_ptr = self.model_ctx.sg_callback_handle.register(cb_ctx);
unsafe { self.model_ctx.callback_table.register(slot, fn_ptr, cb_ctx_ptr); }
// sg_callback_handle owns the memory, no separate tracking needed
```

**清理模式变化**:
```rust
// 旧 (unsafe reclaim):
self.model_ctx.callback_table.clear(slot);
if !self.model_ctx.sg_callback_ctx_ptr.is_null() {
    unsafe { let ctx = Box::from_raw(self.model_ctx.sg_callback_ctx_ptr as *mut SgCallbackCtx); }
    self.model_ctx.sg_callback_ctx_ptr = std::ptr::null();
}

// 新 (safe reclaim):
self.model_ctx.callback_table.clear(slot);
if let Some(provider_ptr) = self.model_ctx.sg_callback_handle.reclaim() {
    if !provider_ptr.is_null() {
        let _ = Arc::from_raw(provider_ptr as *const Mutex<SemanticGatekeeperCallback>);
    }
}
```

**unsafe 边界**: `SgCallbackHandle::reclaim()` 内部仍有 unsafe（`Box::from_raw` 回收 precomputed_knowledge），但这是封装好的单一 unsafe 点，而非散布在 executor.rs 各处。

#### §4.2.3 GeneratorForwardConfig unsafe impl 消除

```rust
// 旧:
pub struct GeneratorForwardConfig {
    pub callback_chain_ptr: *mut CallbackChain,  // 需要 unsafe impl Send/Sync
}
unsafe impl Send for GeneratorForwardConfig {}
unsafe impl Sync for GeneratorForwardConfig {}

// 新:
pub struct GeneratorForwardConfig {
    pub callback_chain: CallbackChainHandle,  // 内部封装 AtomicPtr，自动 Send+Sync
}
// 无需 unsafe impl — CallbackChainHandle 已声明 Send+Sync
```

#### §4.2.4 Executor unsafe impl 消除

```rust
// 旧:
unsafe impl<B: Backend<E> + Send + 'static, E: Element + Send> Send for Executor<B, E> {}
unsafe impl<B: Backend<E> + Sync + 'static, E: Element + Sync> Sync for Executor<B, E> {}

// 新: 所有字段自动 Send+Sync:
pub struct Executor<B, E> {
    pub(crate) backend: B,                    // B: Send + Sync (bound)
    pub(crate) dispatch: DispatchCoordinator, // 所有字段 Send+Sync
    pub(crate) kv: KvCoordinator,             // 所有字段 Send+Sync
    pub(crate) compute: ComputeCoordinator,   // MegaKernelExecutor 内 fn ptr — 需要审查
    pub(crate) inference: InferenceCoordinator, // 所有字段 Send+Sync
    pub(crate) model_ctx: ModelContextHolder,  // CallbackChainHandle(Send+Sync) + SgCallbackHandle
    pub(crate) observability: ObservabilityCoordinator, // Send+Sync
}
// 删除两个 unsafe impl 块
```

**ComputeCoordinator.mega_kernel**: `MegaKernelExecutor` 内含 JIT 编译的函数指针（`fn(*const u8) -> i32` 等）。这些是 `extern "C"` 函数指针，Rust 认为它们不自动实现 Send/Sync。需要为 `MegaKernelExecutor` 添加 `unsafe impl Send/Sync`（这是合理的：JIT 编译产物在编译后不可变，线程安全）。

### §4.3 实施步骤

| 步骤 | 文件 | 变更 | 验证 |
|-----|------|------|------|
| 1 | `coordinator/callback_slot.rs` (新) | CallbackChainHandle 定义 | `cargo test callback_slot` |
| 2 | `coordinator/sg_callback_handle.rs` (新) | SgCallbackHandle 定义 | `cargo test sg_callback_handle` |
| 3 | `coordinator/model_context.rs` | `sg_callback_ctx_ptr` → `sg_callback_handle: SgCallbackHandle` | `cargo check` |
| 4 | `executor.rs` | `callback_chain_ptr` → `callback_chain: CallbackChainHandle` | `cargo check` |
| 5 | `executor.rs` | 删除 GeneratorForwardConfig 的 unsafe impl Send/Sync | `cargo check` |
| 6 | `executor_builder.rs` | 初始化新字段 | `cargo check` |
| 7 | `executor.rs` + `executor_api.rs` | 更新所有 set/get 点 | `cargo check` |
| 8 | `executor.rs` | 删除 Executor 的 unsafe impl Send/Sync | `cargo check` 通过即证明 |
| 9 | `mega_kernel.rs` | MegaKernelExecutor 添加 unsafe impl Send/Sync（合理的 JIT 产物） | `cargo check` |
| 10 | — | 全量测试 | `cargo test --lib` + E2E SG |

### §4.4 风险

1. **AtomicPtr Acquire/Release**: JIT 在另一线程读取时需要 Release 写、Acquire 读（已采纳）。若后续 JIT 改为非原子 load，需 `compiler_fence` 配合
2. **Pin<Box<SgCallbackCtx>>**: 必须用 Pin 保证 `as_ptr()` 返回的地址在 owner move 时不变。Box 本身的 heap 地址不会随 owner move 变化，但 Pin 是显式契约
3. **FFI 读点定位**: Step 4 改字段名后 `compat/` 中的读取点会有编译错误，正好定位所有需要更新的位置

---

## §5 文件布局

```
src/engine/
    mod.rs                    — Re-exports
    executor.rs               — 瘦编排器
    executor_builder.rs       — ExecutorBuilder
    coordinator/
        mod.rs                — 模块根
        dispatch.rs           — DispatchCoordinator
        kv.rs                 — KvCoordinator
        compute.rs            — ComputeCoordinator
        inference.rs          — InferenceCoordinator
        model_context.rs      — ModelContextHolder
        observability.rs      — ObservabilityCoordinator
    batch_executor.rs         — (不变)
    mega_kernel.rs            — (不变)
    callbacks/                — (不变)
```

---

## §6 REQ 定义

| REQ | 描述 | 验证方法 |
|-----|------|---------|
| REQ-DECOMP-001 | 零行为变更 | 所有现有测试不经修改即通过 |
| REQ-DECOMP-002 | CLAUDE.md 合规 | 文件行数 + 函数行数审计 |
| REQ-DECOMP-003 | 测试保留 | 现有测试无修改 |
| REQ-DECOMP-004 | 架构目标 | generate_batch() 模式复制 |
| REQ-DECOMP-005 | BCI 统一 | DispatchCoordinator.build_batch() |
| REQ-DECOMP-006 | unsafe 消除 | `grep "unsafe impl" src/engine/` = 0 |
| REQ-DECOMP-007 | Coordinator 独立 | 每个 coordinator 可独立单元测试 |
| REQ-DECOMP-008 | Executor 瘦编排器 | Executor ≤7 字段 |

