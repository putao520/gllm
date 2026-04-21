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
| 数据上传 | `upload_weights(src)` | 编译时: 权重 htod 上传到 GPU (一次性); CPU 路径: 零拷贝 |
| 数据下载 | `download_output(src, dst)` | GPU 路径: kernel 完成后 dtoh 传回 output tokens; CPU 路径: 不需要 |
| KV Cache | `alloc_kv_cache(batch, seq)` | 分页分配 |
| 同步 | `sync()` | 等待异步操作完成 |

**废弃方法**: `decoder_forward()`、`encoder_forward()`、`embedding_forward_gpu_pure()` 全部删除。Backend 不再包含前向传播逻辑。

### 2.3 废弃的手写前向传播文件

| 文件 | 处理 | 理由 |
|------|------|------|
| `compat/bert_forward.rs` | **物理删除** | FusedGraphExecutor 统一处理 Encoder |
| `compat/decoder_forward.rs` | **物理删除** | FusedGraphExecutor 统一处理 Decoder |
| `compat/cpu_backend.rs` 中的 forward 分发 | **简化** | 只保留内存管理 |

## 3. Executor::step() 生命周期

```
step()
 ├── 1. schedule()        — 调度器决策
 ├── 2. build_batch()     — 组装混合批次
 ├── 3. forward()         — 一次 CALL 进入 mega-kernel（唯一路径）
 │    └── compiled_model.entry_point(token_ids, weights, kv_cache, ...)
 │         └── [JIT 机器码内部: 所有层 + KV cache + sample + token→text + generate 循环]
 ├── 4. 返回结果           — output_buffer 中已有完整文本字符串（JIT 内部完成 token→text）
 ├── 5. update()          — 更新请求状态
 └── 6. epilogue_ingest() — Epilogue 遥测收集 + 决策
```

**核心**: `forward()` 是一次函数调用，不是循环。整个前向传播由 JIT mega-kernel 完成。

## 4. 全 JIT 前向传播 (ARCH-FULL-JIT)

### 4.0 设计原则

> **核心定位**: Rust 是**代码生成器**。模型加载时 Rust 通过 JIT 管线生成机器码。加载完成后，推理请求**直接跳入 JIT 机器码执行**，Rust 不参与任何计算、编排、数据搬运、文本解码。
>
> **Rust 的唯一职责**: 生成代码。之后的一切由 JIT 机器码完成。不参与推理、不参与计算、不参与数据搬运、不参与文本解码。
>
> **铁律 ARCH-RUST-IS-CODEGEN**: Rust = 代码生成器 + Hook ABI 工具库。推理阶段 Rust **什么都不做**。所有功能（forward、采样、generate 循环、KV cache 管理、stop condition、token→text 解码、Guardrail、Semantic Gatekeeper、Head Routing、Intent Recall、CoT Reasoner、Early Exit、MoE Routing）全部由 JIT 编译为机器码。Rust 在推理时只调用一次 JIT 入口函数，不参与任何推理/计算相关的操作。

| 铁律 | 说明 |
|------|------|
| **ARCH-FULL-JIT** | 所有算子走 REGISTER-VM JIT 管线。零手写前向传播代码。 |
| **ARCH-MEGA-KERNEL** | 整个前向传播编译为**一个** JIT mega-kernel。不存在"逐节点调用"、"逐层循环"。热路径 = 一次 JMP 指令。 |
| **ARCH-NO-EMPTY-GRAPH** | 每个 FusedNode 必须产生有效的 CompilerGraph（output_numel > 0）。 |
| **ARCH-SHAPE-ELIM** | Shape/Reshape 等元数据操作在编译时常量折叠。运行时零开销。 |
| **ARCH-GATHER-JIT** | Gather（embedding lookup）编译为 JIT 索引加载内核。 |
| **ARCH-HOTPATH-ZERO-OVERHEAD** | 热路径中零 Rust 参与。不存在 HashMap、String、Vec、for 循环、函数指针调用。所有编排逻辑在 JIT 生成时已烘焙进机器码。 |

### 4.1 架构模型：Rust = 代码生成器，热路径 = 纯 JIT

#### 4.1.1 两阶段模型

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 编译（Rust 活跃）— 模型加载时执行一次               │
│                                                               │
│  Rust 职责:                                                   │
│  ├── 解析模型 (YAML/ONNX/GGUF) → FusedGraph                 │
│  ├── 分析图拓扑，确定层间数据流                              │
│  ├── 为整个前向传播 + 采样生成一个 CompilerGraph              │
│  ├── JIT codegen → 一段连续机器码（完整前向传播 + 采样）     │
│  ├── 将所有权重 pack 到连续内存区域（weight blob）            │
│  ├── 分配所有中间 buffer（activation、scratchpad、KV stride）│
│  ├── 采样参数通过栈传入（每次请求可不同）                     │
│  └── 若有 hook 注册，生成对应层 JMP 跳转代码                 │
│     若无 hook 注册，不生成任何跳转代码（零开销）              │
│                                                               │
│  产出: 一个可执行的 CompiledKernel，包含：                    │
│  ├── code: &[u8]          — JIT 机器码                       │
│  ├── weight_blob: &[u8]   — 全部权重连续打包                 │
│  ├── scratchpad: &mut [u8] — 全部中间 buffer 预分配          │
│  └── 所有偏移 = JIT 机器码内的立即数，Rust 不持有            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: 推理（Rust 不参与）— 请求级别执行                   │
│                                                               │
│  Rust 职责: 一次 CALL。之后一切由 JIT 完成。                 │
│                                                               │
│  热路径: Rust 调用 kernel_entry，传入初始 token_ids           │
│  之后 JIT 机器码内部完成整个 generate 循环:                  │
│                                                               │
│  JIT 内部循环 (每个 token):                                   │
│  ├── Embedding lookup (token_ids → embeddings)               │
│  ├── for layer 0..N:                                          │
│  │   ├── RmsNorm → QKV GEMM → RoPE → Attn → Wo → Residual  │
│  │   ├── RmsNorm → Gate+SiLU+Up+Down → Residual              │
│  │   └── [Hook JMP] — 仅在有 hook 时存在                     │
│  ├── lm_head GEMM → logits                                   │
│  ├── Sample → next_token_id                                  │
│  ├── 写 next_token_id 到 output_buffer[index]                │
│  ├── 更新 KV cache position                                  │
│  ├── 检查 stop condition (EOS / max_tokens)                  │
│  └── 未停止 → JMP 回 embedding lookup (JIT 内部循环)         │
│                                                               │
│  停止条件满足时:                                              │
│  ├── RET                                                     │
│  └── output_buffer 中已包含完整 UTF-8 文本字符串              │
│      (token→text 解码也在 JIT 内部完成)                       │
│                                                               │
│  所有数据 (token_id, logits, text) 全部在 JIT 侧内存/显存中。│
│  Rust 不读取、不处理、不持有任何中间数据。                    │
│  只有 hook 需要调用外部函数时才 JMP 出 JIT。                  │
└─────────────────────────────────────────────────────────────┘
```

**热路径中 Rust 做了什么**: 一次 CALL。返回后 output_buffer 中已有完整 UTF-8 文本字符串。
**热路径中 Rust 不做什么**: 一切。forward、采样、generate 循环、KV cache 更新、stop condition、token→text 解码 — 全部在 JIT 机器码内部完成。
**采样**: temperature / top_k / top_p 通过栈参数传入（每次请求可不同）。结果直接存 JIT 侧 output buffer。
**数据流**: token_ids → [JIT 全程处理: forward + sample + decode + 循环] → output_buffer (UTF-8 文本字符串)。Rust 调用方直接读 output_buffer 即可。

#### 4.1.2 Mega-Kernel 编译过程

```
FusedGraphExecutor::compile()
 ├── 输入: FusedGraph (已优化，节点已融合)
 │
 ├── 1. 全图拓扑分析
 │   ├── 确定节点执行顺序（拓扑排序）
 │   ├── 计算每个节点的 activation 来源（哪个前驱的输出）
 │   ├── 计算残差连接（哪两个前驱的输出需要相加）
 │   └── 确定 KV cache 层偏移表
 │
 ├── 2. 全图 CompilerGraph 构建
 │   ├── 将所有 FusedNode 展开为子图 (RmsNorm, Gemm, RoPE, Attention, ...)
 │   ├── 按拓扑序拼接为单个大图
 │   ├── 层间数据流: activation_ping/pong 双缓冲交替作为层间传递
 │   ├── 残差连接: 保存 input 指针，在层末尾加法
 │   └── KV cache: 每层 attention 节点绑定对应层的 kv_cache 偏移
 │
 ├── 3. 单次 JIT 编译
 │   ├── Phase 0: Scalar 参考实现 (语义定义)
 │   ├── Phase 1: SymExec → SemanticDAG (算子分类 + 融合决策)
 │   ├── Phase 2: FusionEngine (层内融合 + 层间融合)
 │   │   ├── FusedAttentionLayer: RmsNorm+QKV+RoPE+Attn+Wo 融合
 │   │   ├── FusedFfnLayer: RmsNorm+Gate+SiLU+Up+Down 融合
 │   │   ├── ResidualAdd: 嵌入到前驱层的 Epilogue (TileLevelFusion)
 │   │   └── lm_head: 独立 Gemm
 │   └── Phase 3: ISA Lowering (DeviceProfile → 机器码)
 │       └── 生成一段连续的机器码，包含所有层的全部计算
 │
 ├── 4. 权重打包
 │   ├── 所有层权重按层序 pack 到一个连续 weight_blob
 │   ├── Embed weight + N × (Attention weights + FFN weights) + lm_head weight
 │   └── 每层权重偏移在编译时确定，bake 进 JIT 代码
 │
 ├── 5. Buffer 预分配
 │   ├── activation_buf: [hidden_size × max_seq_len × 2]  (双缓冲: ping/pong)
 │   ├── scratchpad: [max(所有层 scratchpad 需求)]
 │   └── sampling_tmp: [vocab_size] (采样临时空间: sort/cumsum)
 │
 ├── 6. 采样代码生成
 │   ├── greedy (temperature=0): argmax (VMAXPS + VPCMPEQD)
 │   ├── temperature scaling: JIT 内部除法 + softmax renorm
 │   ├── top_k: partial sort + mask + sample
 │   ├── top_p: sort + cumsum + mask + sample
 │   ├── stop condition: 检查 EOS + max_tokens 计数器
 │   ├── generate 循环: JMP 回 embedding lookup
 │   └── 参数从栈读取 (运行时传入，每次请求可不同)
 │
 ├── 7. Hook 跳转代码生成（仅当有 hook 注册时）
 │   ├── 无 hook → 不生成任何跳转代码（零指令零开销）
 │   ├── Guardrail veto → conditional JMP 跳过剩余层
 │   ├── Early exit → conditional RET 直接返回 token_id
 │   └── SG q-tap → STG 副作用写 (不改变控制流)
 │
 └── 产出: CompiledKernel
      ├── code: Vec<u8>              — JIT 机器码 (forward + sampling + hooks)
      ├── weight_blob: Vec<u8>       — 全部权重
      ├── activation_ping: Vec<u8>   — 双缓冲 A
      ├── activation_pong: Vec<u8>   — 双缓冲 B
      ├── scratchpad: Vec<u8>        — 临时空间 (含采样临时)
      └── rope_cache: Vec<u8>        — cos/sin 预计算表
```

#### 4.1.3 Mega-Kernel ABI（唯一的函数入口）

```rust
/// Mega-kernel 入口: 输入 token_ids → 输出 generated token 序列。
/// Rust 设置参数，CALL。返回后 output_buffer 中已有完整 UTF-8 文本。
/// 每次请求可以传不同的 temperature/top_k/top_p/max_tokens。
pub type MegaKernelFn = unsafe extern "C" fn(
    // ── 寄存器参数 (System V AMD64 ABI) ──
    input_ids_ptr: *const u32,   // rdi: prompt token ID 数组
    weight_blob_ptr: *const u8,  // rsi: 全部权重连续打包
    kv_cache_ptr: *mut u8,       // rdx: KV Cache 起始
    positions_ptr: *const u32,   // rcx: 位置数组
    aux_ptr: *const u8,          // r8: KV-V half 指针
    batch_size: usize,           // r9: 批大小
    // ── 栈参数 ──
    prompt_len: usize,           // [rsp+0]: prompt 长度
    scratchpad_ptr: *mut u8,     // [rsp+8]: 临时缓冲区
    output_tokens_ptr: *mut u32, // [rsp+16]: 输出 token 序列缓冲区
    // ── 采样参数 (每次请求可不同) ──
    temperature: f32,            // [rsp+24]: 温度 (0.0 = greedy)
    top_k: u32,                  // [rsp+28]: top-k (0 = disabled)
    top_p: f32,                  // [rsp+32]: top-p (1.0 = disabled)
    max_new_tokens: u32,         // [rsp+36]: 最大生成 token 数
    eos_token_id: u32,           // [rsp+40]: EOS token ID
    hook_ctx_ptr: *mut u8,       // [rsp+48]: hook 上下文 (无 hook 时 null)
    telemetry_ptr: *mut u8,      // [rsp+56]: 遥测缓冲区
) -> u32;  // rax: 实际生成的 token 数
```

**Rust 调用侧（热路径全部代码）**:

```rust
// 热路径全部代码：设置参数，CALL，读取结果。
let num_generated: u32 = unsafe {
    (kernel.entry_point)(
        prompt_tokens.as_ptr(),       // rdi: prompt
        kernel.weight_blob.as_ptr(),  // rsi: 权重
        kv_cache_k,                   // rdx: KV cache
        positions.as_ptr(),           // rcx: positions
        kv_cache_v as *const u8,      // r8: V half
        1,                            // r9: batch
        prompt_tokens.len(),          // [rsp+0]: prompt_len
        kernel.scratchpad.as_mut_ptr(), // [rsp+8]
        output_buf.as_mut_ptr(),      // [rsp+16]: output tokens
        0.7f32,                       // [rsp+24]: temperature
        50u32,                        // [rsp+28]: top_k
        0.9f32,                       // [rsp+32]: top_p
        100u32,                       // [rsp+36]: max_new_tokens
        eos_token_id,                 // [rsp+40]: EOS
        std::ptr::null_mut(),         // [rsp+48]: hook_ctx
        std::ptr::null_mut(),         // [rsp+56]: telemetry
    )
};
// output_buf 中已是完整 UTF-8 文本字符串。Rust 直接返回给调用方。
```

**没有**: logits 读取、采样逻辑、HashMap、String、Vec 分配。采样参数通过栈传入，每次请求可不同。

#### 4.1.4 JIT 内部数据流（机器码内部，Rust 不可见）

```
JIT 机器码内部执行流程（编译时烘焙，运行时不可变）:

  ; ── Embedding lookup ──
  for i in 0..seq_len:
      row = input_ids[i]
      memcpy(activation_ping[i*D..], weight_blob[embed_offset + row*D..])

  ; ── Layer loop (JIT 内部循环，非 Rust 循环) ──
  for layer in 0..num_layers:
      ; 保存残差指针
      residual_ptr = activation_ping

      ; RmsNorm
      rms_norm(activation_ping → activation_pong, weight_blob[norm_offset[layer]])

      ; QKV GEMM
      gemm(activation_pong → scratchpad_q, weight_blob[q_offset[layer]])
      gemm(activation_pong → scratchpad_k, weight_blob[k_offset[layer]])
      gemm(activation_pong → scratchpad_v, weight_blob[v_offset[layer]])

      ; RoPE
      rope(scratchpad_q, positions, rope_cache)
      rope(scratchpad_k, positions, rope_cache)

      ; Attention (含 KV cache merge，decode step)
      merge_kv(scratchpad_k, scratchpad_v, kv_cache[kv_offset[layer]])
      attention(scratchpad_q, kv_cache[kv_offset[layer]] → activation_ping)

      ; Wo GEMM
      gemm(activation_ping → activation_pong, weight_blob[o_offset[layer]])

      ; Residual Add
      add(activation_pong, residual_ptr → activation_ping)

      ; ── FFN ──
      ; 保存残差指针
      residual_ptr = activation_ping

      ; RmsNorm
      rms_norm(activation_ping → activation_pong, weight_blob[ffn_norm_offset[layer]])

      ; Gate GEMM + SiLU
      gemm(activation_pong → scratchpad_gate, weight_blob[gate_offset[layer]])
      silu(scratchpad_gate)

      ; Up GEMM
      gemm(activation_pong → scratchpad_up, weight_blob[up_offset[layer]])

      ; Gate * Up
      mul(scratchpad_gate, scratchpad_up → scratchpad_down)

      ; Down GEMM
      gemm(scratchpad_down → activation_pong, weight_blob[down_offset[layer]])

      ; Residual Add
      add(activation_pong, residual_ptr → activation_ping)

  ; ── lm_head ──
  rms_norm(activation_ping → activation_pong, weight_blob[final_norm_offset])
  gemm(activation_pong → logits_in_scratchpad, weight_blob[lm_head_offset])

  ; ── Sampling (JIT 内部，参数从栈读取) ──
  ; temperature = [rsp+24], top_k = [rsp+28], top_p = [rsp+32]
  ; temperature > 0: divide logits by temperature
  ; top_k > 0: keep only top_k logits, set rest to -inf
  ; top_p > 0: sort + cumsum + mask
  ; sample: random or argmax (temperature=0)
  next_token_id = sample(logits_in_scratchpad, vocab_size, temperature, top_k, top_p)

  ; 写 output_tokens[num_generated] = next_token_id
  ; vocab_lookup: next_token_id → UTF-8 bytes → append to output_text_buf
  ; num_generated++
  ; if next_token_id == eos_token_id || num_generated >= max_new_tokens:
  ;     RET (output_text_buf 已有完整 UTF-8 文本)
  ; else:
  ;     update positions, JMP back to embedding lookup
```

**所有偏移量（norm_offset、q_offset、kv_offset 等）在编译时烘焙为立即数**。
**activation_ping/pong 双缓冲**: 层间交替使用，零额外拷贝。
**KV cache merge**: decode step 时，JIT 代码从 kv_cache 读取历史 K/V，与当前 K/V 拼接后计算 attention。

#### 4.1.5 Hook 集成（条件跳转，无 hook = 零代码）

Hook（Guardrail、SG、HR、Intent、Early Exit）通过 **JIT 代码内嵌的条件跳转** 集成。
**没有 hook 注册时，不生成任何跳转代码。机器码中不存在 hook 相关的指令。**

```
编译时:
  ├── 检查 hook 注册表
  │
  ├── 无 hook 注册:
  │   └── 不生成任何跳转/分支/trap 代码。层循环是连续的直线代码。
  │
  └── 有 hook 注册:
      ├── Guardrail veto: 在 target layer 后生成 JMP (读 hook_ctx 状态字)
      │   └── hook_ctx 状态字由外部写入 (CPU: 共享内存; GPU: global mem)
      ├── SG q-tap: 在 q_proj GEMM 后生成 STG (副作用写，不改控制流)
      ├── HR head routing: 在对应层后生成分支 JMP
      ├── Early exit: 在指定层后生成 conditional RET (直接返回 token_id)
      └── hook_ctx_ptr 参数非 null 时有效

运行时:
  ├── 无 hook: 机器码是纯直线代码，无分支，最大 ILP
  ├── 有 hook: 条件 JMP 读 hook_ctx 状态字
  │   ├── veto/skip → JMP 跳过后续计算
  │   ├── exit → RET token_id
  │   └── continue → 零额外开销 (CMP+JE 被 branch predictor 吞掉)
  └── hook_ctx 写入方: 外部进程/线程/GPU kernel (不经过 Rust)
```

**关键**:
- **无 hook = 零代码**: 机器码中不存在任何 hook 指令。不是"生成了但跳过"，是**根本没生成**。
- **Hook 通信 = 共享内存**: hook_ctx 通过共享内存写入，JIT 机器码直接读取。不经过 Rust 函数调用、不经过函数指针。
- **GPU 路径**: hook trigger 通过 GPU global memory 写入，JIT kernel 读同一个地址。CPU callable hook 通过 GPU→CPU interrupt 或共享内存轮询。

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

| 模型类型 | Mega-Kernel 结构 | 特殊处理 |
|---------|-----------------|---------|
| **Decoder (Generator)** | Embed → N × [Attn+Residual + FFN+Residual] → lm_head | KV Cache 逐层更新（JIT 内部） |
| **Encoder (Embedding)** | Embed → N × [BiAttn + FFN] → MeanPool → L2Norm | 无 KV Cache |
| **Reranker** | Embed → N × [BiAttn + FFN] → Slice(CLS) → Linear | 无 KV Cache |

**三种类型共享同一 Mega-Kernel ABI**。差异在编译时烘焙进 JIT 机器码，不在运行时分支。

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

Mega-kernel 入口签名见 §4.1.3。调用方保证：

- `weight_blob_ptr` 指向编译时 pack 的全部权重（连续内存）
- `kv_cache_ptr` 指向 KV Cache 起始（Decoder），或 null（Encoder/Reranker）
- `scratchpad_ptr` 指向至少 `max(所有层 scratchpad 需求, vocab_size × dtype.size_bytes())` 字节的可写内存（采样在 scratchpad 内完成）
- `scratchpad_ptr` 指向至少 `max(所有层 scratchpad 需求)` 字节的可写内存
- `seq_len` 是运行时值，通过 `[rsp+0]` 传入（SymDim::Symbolic 的运行时绑定）

**废弃**: 单节点 ABI `CompiledLayerFn`。Mega-kernel 是唯一入口。

## 5. Callback 构建流程

`build_step_callbacks()` 根据配置注册优化模块回调。Callback 在编译时烘焙进 mega-kernel 机器码（见 §4.1.5），不在运行时通过 Rust 函数调用。CallbackAction、LayerCallback trait 和 CallbackChain 完整定义见 `SPEC/05-OPTIMIZATIONS.md` §1。

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

> **铁律**: GPU 推理遵循与 CPU 完全相同的 ARCH-RUST-IS-CODEGEN 铁律。编译时 Rust 一次性完成 JIT codegen + 权重 htod + buffer 分配。推理时一次 GPU kernel launch。详见 `02-ARCHITECTURE.md §8.10`。

### 8.1 KV Scatter（编译进 mega-kernel）

`OpKind::KvScatterWrite` + PTX/HIP/MSL codegen。KV scatter 逻辑编译进 mega-kernel，kernel 内部 per-thread-block 散写，不经过 Rust。

### 8.2 权重管理（编译时一次性 htod）

编译时：weight_blob pack → htod → GPU 常驻。所有权重偏移 bake 进 GPU kernel 立即数。推理时 kernel 直接从 GPU 常驻权重读取，Rust 不发起任何 htod/dtoh 操作。

### 8.3 Metal KV 直写（编译时 bake 指针）

Metal unified memory：编译时获取 shared memory 指针 bake 进 kernel。运行时 kernel 直接读写，零额外传输。

### 8.4 PagedKvView 三后端统一

编译时：page_table 构建 + htod 上传 + bake 进 kernel。运行时 kernel 内部 page lookup → gather K/V → attention。CUDA / HIP / Metal 三后端统一。

## 9. 废弃清单 (ARCH-FULL-JIT 迁移)

| 废弃项 | 替代 | 理由 |
|--------|------|------|
| `compat/bert_forward.rs` | Mega-kernel JIT | Encoder 模型走统一 JIT 路径 |
| `compat/decoder_forward.rs` | Mega-kernel JIT | Decoder 模型走统一 JIT 路径 |
| `graph_executor_ptr: *mut` | `graph_executor: FusedGraphExecutor` 固定成员 | 消除裸指针 |
| `onnx_generator_plan` | ONNX → FusedGraph → Mega-kernel JIT | ONNX 模型走同一管线 |
| `OnnxPlan` error variant | `GraphExpansion` | 统一错误类型 |
| Backend 的 `decoder_forward` 方法 | Mega-kernel `CALL` | Backend 只管内存 |
| **逐节点 Rust 循环 (`for node in nodes`)** | Mega-kernel 单次 `CALL` | ARCH-MEGA-KERNEL: 热路径零 Rust 参与 |
| **`CompiledNode` 数组 + 逐节点 `execute()`** | 单个 `CompiledModel` + 一次 `CALL` | ARCH-MEGA-KERNEL |
| **`CompiledLayerFn` 单节点 ABI** | `MegaKernelFn` 全图 ABI | ARCH-MEGA-KERNEL |
| **`tensors: HashMap<String, Vec<u8>>`** | 权重偏移 bake 进 JIT 机器码立即数 | ARCH-MEGA-KERNEL: Rust 不持有任何权重偏移数据结构 |
| **热路径 String 操作** | 编译时 bake 进机器码，运行时无 String | ARCH-MEGA-KERNEL |
| **热路径 Vec 分配 (`vec!`/`to_vec`/`clone`)** | 所有 buffer 编译时分配，运行时零分配 | ARCH-MEGA-KERNEL |
| **热路径 weight pack (`pack_weight_blob`)** | 编译时一次性 pack 全部权重，偏移 bake 进机器码立即数 | ARCH-MEGA-KERNEL |
| **Callback Rust 函数调用 (per-node)** | JIT 机器码内嵌条件跳转 | ARCH-MEGA-KERNEL: callback 烘焙进机器码 |

## 10. 热路径零开销执行协议 (ARCH-MEGA-KERNEL + ARCH-HOTPATH-ZERO-OVERHEAD)

> **SSOT**: 本节定义推理热路径中 Rust 的角色边界。
>
> **核心铁律 ARCH-RUST-IS-CODEGEN**: Rust = 代码生成器。推理阶段 Rust **什么都不做**。所有功能全部 JIT。Rust 只调用一次 JIT 入口函数。

### 10.1 热路径开销预算

**SmolLM2-135M 参考模型** (24 层, hidden=576, BF16→F32, CPU JIT):

| 指标 | 旧架构 (Rust 编排) | Mega-Kernel 目标 | 测量方法 |
|------|-------------------|-----------------|---------|
| 热路径 Rust 指令数 | ~5000 (循环+HashMap+clone) | **~20** (设置参数 + CALL) | `perf stat -e instructions` |
| 热路径 Rust malloc | ~100 | **0** | 代码审计 |
| 热路径 HashMap 操作 | ~100 | **0** | 代码审计 |
| 热路径 String 操作 | ~50 | **0** | 代码审计 |
| 热路径函数调用 | ~50 (逐节点 execute) | **1** (mega-kernel CALL) | 代码审计 |
| Decode ms/token (release) | ~220ms | **< 10ms** | `examples/bench_step.rs` |

### 10.2 编译时完整性验证

`compile()` 完成后，以下条件必须全部成立:

```rust
let model: &CompiledModel = &executor.graph_executor.compiled_model;
assert!(!model.code.is_empty(), "JIT 机器码必须非空");
assert!(!model.weight_blob.is_empty(), "权重必须已打包");
assert!(!model.activation_ping.is_empty(), "activation 双缓冲 A 必须已分配");
assert!(!model.activation_pong.is_empty(), "activation 双缓冲 B 必须已分配");
assert!(!model.scratchpad.is_empty(), "scratchpad 必须已分配 (含采样临时空间)");
```

### 10.3 热路径审计命令

```bash
# 热路径 = graph_executor 的 forward 方法。应仅包含:
# 1. 从 CompiledModel 提取 .as_ptr()/.as_mut_ptr()
# 2. 一次 unsafe 函数调用

# 1. 热路径中禁止任何堆分配
grep -n "vec!\[" src/graph/executor.rs | grep -v "compile\|new\|test\|//"
grep -n "to_vec\|\.clone()\|Vec::new\|Vec::with_capacity" src/graph/executor.rs | grep -v "compile\|test\|//"

# 2. 热路径中禁止 pack_weight_blob 调用
grep -n "pack_weight_blob" src/graph/executor.rs | grep -v "fn pack_weight_blob\|compile\|//"

# 3. 热路径中禁止 HashMap
grep -n "tensors.get\|tensors.insert\|tensors.remove\|HashMap" src/graph/executor.rs | grep -v "compile\|test\|fn seed\|//"

# 4. 热路径中禁止循环 (for/while)
grep -n "for .*\n.*in\|while " src/graph/executor.rs | grep -v "compile\|test\|//"

# 5. 热路径中禁止 String 操作
grep -n "\.clone()\|\.to_string()\|format!\|String" src/graph/executor.rs | grep -v "compile\|test\|//"
```

### 10.4 合规的特殊路径

| 操作 | 实现位置 | Rust 是否参与 | 说明 |
|------|---------|-------------|------|
| Embedding lookup | JIT 机器码 | ❌ | Gather 内嵌 |
| KV cache merge | JIT 机器码 | ❌ | JIT 内部处理 |
| GQA head expansion | JIT 机器码 | ❌ | broadcast 内嵌 |
| Residual add | JIT 机器码 | ❌ | Epilogue 融合 |
| RoPE cos/sin | JIT 机器码 | ❌ | rope_cache 表 |
| Logits GEMM | JIT 机器码 | ❌ | lm_head 内嵌 |
| 采样 (argmax/top_k/top_p) | JIT 机器码 | ❌ | 参数从栈传入 |
| Generate 循环 | JIT 机器码 | ❌ | JMP 回 embed |
| Stop condition | JIT 机器码 | ❌ | EOS + max_tokens |
| Hook 条件跳转 | JIT 机器码 | ❌ | 仅在有 hook 时生成 |
| Token→UTF-8 解码 | JIT 机器码 | ❌ | vocab lookup 内嵌 |

**Rust 在推理阶段不执行任何操作。**

