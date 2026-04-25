use super::backend_trait::{self, Backend};
use super::memory::get_system_memory_pressure;
use super::Element;
use crate::engine::executor::{
    AttentionTopology, BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheConfig,
    KvCacheHandle, LogitsHandle, SamplingConfig,
};
use crate::scheduler::types::{PageId, PageState, StorageKey};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// ARCH-FULL-JIT: FusedGraphExecutor 接入辅助函数
// ---------------------------------------------------------------------------

/// 从 GeneratorForwardConfig 中安全获取 FusedGraphExecutor 可变引用。
/// ARCH-HOTPATH-ZERO-OVERHEAD: 需要 &mut 以复用预分配的 output_buf/scratchpad。
fn get_graph_executor(config: &GeneratorForwardConfig) -> Result<&mut crate::graph::executor::FusedGraphExecutor, BE> {
    if config.graph_executor_ptr.is_null() {
        return Err(BE::Other(
            "ARCH-FULL-JIT: graph_executor_ptr is null — model was not JIT-compiled".into(),
        ));
    }
    // SAFETY: graph_executor_ptr 生命周期由 Executor 管理，在 forward 调用期间始终有效。
    // Executor 保证单线程访问 (Mutex 或单线程 client)。
    Ok(unsafe { &mut *config.graph_executor_ptr })
}

/// 为 encoder 模型（embedding/rerank/classify）准备 FusedGraphExecutor 输入。
///
/// 将 tokens (u32) 转换为图需要的输入格式。
/// 所有模型特定参数（position_offset 等）从 config 的 ModelManifest 获取。
fn prepare_encoder_inputs(tokens: &[u32], config: &GeneratorForwardConfig) -> HashMap<String, Vec<u8>> {
    let seq_len = tokens.len();
    let mut inputs = HashMap::new();

    // input_ids: u32 token IDs，以 u32 LE 字节传递。
    // JIT Gather 的 ScalarLoad (vmovss → vmovd) 将 u32 位模式直接加载到 GPR，
    // IntMulStride 做整数乘法得到行偏移。
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|&t| t.to_le_bytes()).collect();
    inputs.insert("input_ids".to_string(), token_bytes);

    // position_ids: 整数位置，从模型配置读取 position_offset（RoBERTa=2, BERT=0, GPT=0）
    let position_offset = config.geometry.position_offset.unwrap_or(0);
    let position_ids: Vec<u8> = (0..seq_len).flat_map(|i| ((i + position_offset) as u32).to_le_bytes()).collect();
    inputs.insert("position_ids".to_string(), position_ids);

    // token_type_ids: 整数类型 ID（单段=0，双段=0/1 交替）
    // 当前: 单段输入全 0（通用默认值，非硬编码特定模型）
    let token_type_ids: Vec<u8> = (0..seq_len).flat_map(|_| 0u32.to_le_bytes()).collect();
    inputs.insert("token_type_ids".to_string(), token_type_ids);

    inputs
}

/// 从 FusedGraphExecutor 输出中提取最终 hidden state (f32 切片)。
fn extract_final_hidden(
    outputs: &HashMap<String, Vec<u8>>,
    executor: &crate::graph::executor::FusedGraphExecutor,
) -> Result<Vec<f32>, BE> {
    // 尝试按图的输出名找
    for output_name in &executor.graph().outputs {
        if let Some(data) = outputs.get(output_name) {
            if data.len() >= 4 {
                return Ok(bytes_to_f32(data));
            }
        }
    }
    // 回退: 取任何非空输出
    for (name, data) in outputs {
        if data.len() >= 4 {
            log::debug!("[ARCH-FULL-JIT] using fallback output '{name}' ({} bytes)", data.len());
            return Ok(bytes_to_f32(data));
        }
    }
    Err(BE::Other("FusedGraphExecutor produced no output tensors".into()))
}

/// Mean pool over sequence dimension: [seq_len, hidden_size] → [hidden_size]
fn mean_pool_hidden(hidden: &[f32], seq_len: usize, hidden_size: usize) -> Vec<f32> {
    if seq_len == 0 || hidden_size == 0 || hidden.len() < seq_len * hidden_size {
        return hidden.to_vec();
    }
    let mut pooled = vec![0.0f32; hidden_size];
    let scale = 1.0 / seq_len as f32;
    for row in 0..seq_len {
        let offset = row * hidden_size;
        for col in 0..hidden_size {
            pooled[col] += hidden[offset + col] * scale;
        }
    }
    pooled
}

/// 从 token 数组构建 shape bindings（seq_len = tokens.len()）
fn shape_bindings_from_tokens(tokens: &[u32]) -> HashMap<String, usize> {
    let mut bindings = HashMap::new();
    bindings.insert("seq_len".to_string(), tokens.len());
    bindings
}

/// Dispatch to `FusedGraphExecutor::run` or `run_with_callbacks` based on
/// whether `config.callback_chain_ptr` is populated. The pointer is owned by
/// the caller (Executor) for the duration of the forward — safe as long as
/// the Executor holds the CallbackChain alive across the call.
fn run_with_optional_callbacks(
    executor: &crate::graph::executor::FusedGraphExecutor,
    inputs: &HashMap<String, Vec<u8>>,
    bindings: &HashMap<String, usize>,
    config: &GeneratorForwardConfig,
) -> Result<HashMap<String, Vec<u8>>, BE> {
    if config.callback_chain_ptr.is_null() {
        executor
            .run(inputs, bindings)
            .map_err(|e| BE::Other(format!("FusedGraphExecutor run failed: {e}")))
    } else {
        // SAFETY: Executor guarantees the CallbackChain lives until
        // `callback_chain_ptr` is cleared after the backend call returns.
        let chain = unsafe { &mut *config.callback_chain_ptr };
        executor
            .run_with_callbacks(inputs, bindings, chain, config)
            .map_err(|e| {
                BE::Other(format!("FusedGraphExecutor run_with_callbacks failed: {e}"))
            })
    }
}

/// 从 seq_len 直接构建 shape bindings
fn shape_bindings_from_seq(seq_len: usize) -> HashMap<String, usize> {
    let mut bindings = HashMap::new();
    bindings.insert("seq_len".to_string(), seq_len);
    bindings
}

/// f32 切片 → 字节 Vec
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let byte_len = data.len() * std::mem::size_of::<f32>();
    let ptr = data.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
}

/// 字节 Vec → f32 Vec
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let count = data.len() / std::mem::size_of::<f32>();
    let ptr = data.as_ptr() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
}

/// Return the output tensor name of the leading `Gather(embed_tokens, input_ids)`
/// node in a decoder graph, used as the injection point for
/// ARCH-MULTIMODAL-FUSION. Returns `None` when no such leading Gather exists
/// (the graph is not a standard decoder embedding graph).
fn first_gather_output(executor: &crate::graph::executor::FusedGraphExecutor) -> Option<String> {
    let first = executor.graph().nodes.first()?;
    if first.op.name() != "Gather" {
        return None;
    }
    first.outputs.first().cloned()
}

// ---------------------------------------------------------------------------
// KV Cache storage
// ---------------------------------------------------------------------------

/// Actual KV cache buffer holding Key and Value tensors for all layers.
///
/// Layout per tensor (K or V):
///   flat Vec<f32> of shape [num_layers, num_kv_heads, max_seq_len, head_dim]
///   Total elements = num_layers * num_kv_heads * max_seq_len * head_dim
#[derive(Debug, Clone)]
pub(crate) struct KvCacheBuffer {
    /// Key tensor: [num_layers * num_kv_heads * max_seq_len * head_dim] as raw bytes
    pub k: Vec<u8>,
    /// Value tensor: same shape as K
    pub v: Vec<u8>,
    /// Number of layers
    pub num_layers: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Page size (tokens per page)
    pub page_size: usize,
    /// Current sequence length written (tokens stored so far)
    pub seq_len: usize,
    /// Bytes per element (4 for F32, 2 for F16/BF16)
    pub elem_bytes: usize,
    /// Cache dtype for F16/BF16 distinction (ARCH-DTYPE-FULLCHAIN-ORCH)
    #[allow(dead_code)]
    pub cache_dtype: gllm_kernels::types::DType,

    /// SharedKvRef: 共享层的 KV donor 映射。
    /// `kv_donor_map[layer_i] = Some(donor_layer)` 表示 layer_i 共享 donor_layer 的 KV。
    /// Gemma 4 E2B 后 20 层、E4B 后 18 层使用此机制。
    /// 共享层在前向传播时从 donor 层偏移读取 KV，不分配独立存储。
    pub kv_donor_map: Vec<Option<usize>>,

    /// Thinking KV 跳过: 持久化 seq_len (不含 thinking tokens)。
    /// 思考 token 使用临时位置写入 KV（当前 step 的注意力需要），
    /// 但 persistent_seq_len 不递增，下一个非思考 token 覆盖该位置。
    ///
    /// 效果: 多轮对话的 KV cache 中不含思考内容。
    pub persistent_seq_len: usize,
    /// 当前是否处于思考阶段 (由 ThinkingTracker 驱动)
    pub in_thinking: bool,
}

impl KvCacheBuffer {
    fn new(config: &KvCacheConfig) -> Self {
        let elem_bytes = config.dtype_size().max(1);
        let cache_dtype = match elem_bytes {
            2 => gllm_kernels::types::DType::F16,
            _ => gllm_kernels::types::DType::F32,
        };
        let num_layers = config.num_layers();

        // SharedKvRef: 计算 donor 映射
        let (effective_layers, kv_donor_map) = Self::build_kv_donor_map(
            num_layers,
            config.num_kv_shared_layers(),
            config.attention_pattern(),
        );

        let total_bytes = effective_layers * config.num_heads() * config.max_seq_len() * config.head_dim() * elem_bytes;
        Self {
            k: vec![0u8; total_bytes],
            v: vec![0u8; total_bytes],
            num_layers,
            num_kv_heads: config.num_heads(),
            max_seq_len: config.max_seq_len(),
            head_dim: config.head_dim(),
            page_size: config.page_size,
            seq_len: 0,
            elem_bytes,
            cache_dtype,
            kv_donor_map,
            persistent_seq_len: 0,
            in_thinking: false,
        }
    }

    /// 构建 KV donor 映射。
    ///
    /// 返回 (effective_layers, donor_map):
    /// - effective_layers: 实际需要 KV 存储的层数 (= num_layers - num_kv_shared)
    /// - donor_map[i] = Some(donor) 表示层 i 共享 donor 层的 KV
    ///
    /// 共享规则: 后 N 层共享同类型 (sliding/global) 的最近非共享层。
    fn build_kv_donor_map(
        num_layers: usize,
        num_kv_shared: usize,
        attention_pattern: &[u8],
    ) -> (usize, Vec<Option<usize>>) {
        if num_kv_shared == 0 || num_layers == 0 {
            return (num_layers, vec![None; num_layers]);
        }

        let shared_start = num_layers.saturating_sub(num_kv_shared);
        let mut donor_map = vec![None; num_layers];

        for layer_i in shared_start..num_layers {
            // 找到同类型的最近非共享层
            let layer_type = attention_pattern.get(layer_i).copied().unwrap_or(0);
            let mut donor = None;
            for j in (0..shared_start).rev() {
                let j_type = attention_pattern.get(j).copied().unwrap_or(0);
                if j_type == layer_type {
                    donor = Some(j);
                    break;
                }
            }
            donor_map[layer_i] = donor;
        }

        let effective = num_layers - num_kv_shared;
        (effective, donor_map)
    }

    /// 获取层 i 的 KV 存储偏移 (字节)。
    /// 共享层返回 donor 层的偏移。
    pub fn layer_kv_offset(&self, layer_i: usize) -> usize {
        let effective_layer = match self.kv_donor_map.get(layer_i).copied().flatten() {
            Some(donor) => donor,
            None => layer_i,
        };
        effective_layer * self.num_kv_heads * self.max_seq_len * self.head_dim * self.elem_bytes
    }

    /// 层 i 是否是共享层 (不计算自己的 KV)
    pub fn is_shared_kv_layer(&self, layer_i: usize) -> bool {
        self.kv_donor_map.get(layer_i).copied().flatten().is_some()
    }

    /// Thinking KV: 标记进入/退出思考阶段。
    ///
    /// 思考阶段中：
    /// - `seq_len` 正常递增（当前 step 的注意力需要看到 thinking tokens）
    /// - `persistent_seq_len` 不递增
    ///
    /// 退出思考时：
    /// - `seq_len` 回退到 `persistent_seq_len`（丢弃 thinking KV 位置）
    /// - 后续非思考 token 从 persistent_seq_len 继续写入
    pub fn set_thinking(&mut self, thinking: bool) {
        if self.in_thinking && !thinking {
            // 退出思考: 回退 seq_len，丢弃临时 thinking KV
            self.seq_len = self.persistent_seq_len;
        }
        self.in_thinking = thinking;
    }

    /// 非思考 token 写入后，同步 persistent_seq_len。
    pub fn commit_position(&mut self) {
        if !self.in_thinking {
            self.persistent_seq_len = self.seq_len;
        }
    }

    /// Number of pages currently in use (rounded up).
    fn active_pages(&self) -> usize {
        if self.page_size == 0 {
            return 0;
        }
        self.seq_len.div_ceil(self.page_size)
    }

    /// Total number of pages this buffer can hold.
    fn total_pages(&self) -> usize {
        if self.page_size == 0 {
            return 0;
        }
        self.max_seq_len.div_ceil(self.page_size)
    }
}

/// Shared KV cache store, indexed by handle ID.
type KvStore = Arc<Mutex<HashMap<u64, KvCacheBuffer>>>;

/// Swap storage: holds page data that has been evicted from the KV cache.
/// Key = StorageKey, Value = (K_data, V_data) for one page across all layers/heads.
type SwapStore = Arc<Mutex<HashMap<StorageKey, SwapPageData>>>;

/// Data for a single swapped-out page.
#[derive(Debug, Clone)]
struct SwapPageData {
    k: Vec<u8>,
    v: Vec<u8>,
}

// ---------------------------------------------------------------------------
// CpuBackend<E>
// ---------------------------------------------------------------------------

/// CPU backend with real KV cache allocation and swap support.
#[derive(Debug, Clone)]
pub struct CpuBackend<E: Element = f32> {
    kv_store: KvStore,
    swap_store: SwapStore,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Element> CpuBackend<E> {
    pub fn new() -> Self {
        Self {
            kv_store: Arc::new(Mutex::new(HashMap::new())),
            swap_store: Arc::new(Mutex::new(HashMap::new())),
            _marker: std::marker::PhantomData,
        }
    }

    /// Access the KV store (for decoder forward pass).
    pub(crate) fn kv_store(&self) -> &KvStore {
        &self.kv_store
    }
}

impl<E: Element> Default for CpuBackend<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Element> Backend<E> for CpuBackend<E> {
    type Tensor = Vec<E>;


    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> Result<KvCacheHandle, BE> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let buffer = KvCacheBuffer::new(config);
        let total_bytes = buffer.k.len() + buffer.v.len();

        // Check memory pressure before allocating
        let pressure = get_system_memory_pressure().map_err(|e| {
            BE::Cpu(format!("failed to read system memory pressure: {e}"))
        })?;
        if pressure > 0.95 {
            return Err(BE::Cpu(format!(
                "cannot allocate KV cache ({} bytes): memory pressure {:.1}%",
                total_bytes,
                pressure * 100.0
            )));
        }

        let mut store = self.kv_store.lock().map_err(|e| {
            BE::Cpu(format!("KV store lock poisoned: {e}"))
        })?;
        store.insert(id, buffer);

        Ok(KvCacheHandle(id))
    }

    fn batch_forward_gpu_pure(
        &self,
        input: &BatchInput,
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        kv_caches: &mut [KvCacheHandle],
        config: &GeneratorForwardConfig,
    ) -> Result<(Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>), BE> {
        // ARCH-FULL-JIT: decoder forward 通过 FusedGraphExecutor JIT 路径执行
        let executor = get_graph_executor(config)?;

        // Decoder batch forward: 逐序列执行 (continuous batching 由调度器处理)
        let mut logits_handles = Vec::with_capacity(input.sequences.len());
        let mut total_compute_ms = 0.0f32;
        let mut telemetry = Vec::with_capacity(input.sequences.len());

        for (seq_idx, seq) in input.sequences.iter().enumerate() {
            let tokens = &seq.tokens;
            let seq_len = tokens.len();
            let position = seq.position;

            let mut inputs = HashMap::new();
            // input_ids: u32 LE 字节（与 JIT Gather 的 ScalarLoad+IntMulStride 一致）
            let token_bytes: Vec<u8> = tokens.iter().flat_map(|&t| t.to_le_bytes()).collect();
            inputs.insert("input_ids".to_string(), token_bytes);

            let position_ids: Vec<u8> = (position..position + seq_len)
                .flat_map(|p| (p as u32).to_le_bytes()).collect();
            inputs.insert("position_ids".to_string(), position_ids);

            // ARCH-MULTIMODAL-FUSION (SPEC/02-ARCHITECTURE.md):
            // When a multimodal request is dispatched, the Executor attaches a
            // pre-computed fused hidden state on the prefill step. Seed the
            // first Gather output so the graph executor's `is_node_computed`
            // check skips `Gather(embed_tokens, input_ids)` and continues with
            // the caller-provided text+media fusion. Strictly a prefill-only
            // bypass: decode steps always re-run Gather on input_ids because
            // newly generated tokens are always text.
            if let Some(fused) = seq.fused_hidden.as_ref() {
                let expected = seq_len * config.hidden_size();
                if fused.len() != expected {
                    return Err(BE::Other(format!(
                        "ARCH-MULTIMODAL-FUSION: fused_hidden length {} != seq_len * hidden_size ({} * {})",
                        fused.len(),
                        seq_len,
                        config.hidden_size(),
                    )));
                }
                let gather_output = first_gather_output(executor).ok_or_else(|| {
                    BE::Other(
                        "ARCH-MULTIMODAL-FUSION: decoder graph has no leading Gather(embed_tokens) node — \
                         cannot inject fused embedding".into(),
                    )
                })?;
                inputs.insert(gather_output, f32_to_bytes(fused));
            }

            let kv_handle = &mut kv_caches[seq_idx];
            let store = self.kv_store.lock().map_err(|e| BE::Cpu(format!("KV lock: {e}")))?;
            let has_kv = store.contains_key(&kv_handle.0);
            drop(store);

            let start = std::time::Instant::now();
            let total_seq = position + seq_len;
            let result = if has_kv {
                let (k_ptr, v_ptr, donor_map) = {
                    let store = self.kv_store.lock().map_err(|e| BE::Cpu(format!("KV lock: {e}")))?;
                    let buf = store.get(&kv_handle.0).ok_or_else(|| {
                        BE::Cpu("KV cache buffer not found".into())
                    })?;
                    let k_ptr = buf.k.as_ptr() as *mut f32;
                    let v_ptr = buf.v.as_ptr() as *mut f32;
                    // SharedKvRef (T43): consumer 层通过 donor_map 解析到 donor 层物理 slot.
                    let donor_map = buf.kv_donor_map.clone();
                    (k_ptr, v_ptr, donor_map)
                };

                let positions_u32: Vec<u32> = (position..position + seq_len).map(|p| p as u32).collect();
                executor.run_with_kv_cache_with_config(
                    &inputs, k_ptr, v_ptr,
                    0, total_seq, seq_len, positions_u32.as_ptr(), config,
                    Some(&donor_map),
                ).map_err(|e| BE::Other(format!("FusedGraphExecutor decoder run failed: {e}")))
            } else {
                let bindings = shape_bindings_from_seq(seq_len);
                executor.run(&inputs, &bindings).map_err(|e| {
                    BE::Other(format!("FusedGraphExecutor decoder run failed: {e}"))
                })
            };
            let elapsed = start.elapsed().as_secs_f32() * 1000.0;
            total_compute_ms += elapsed;

            let outputs = result?;
            let full_logits = extract_final_hidden(&outputs, executor)?;
            // SPEC 01-REQUIREMENTS REQ-KV-005 + autoregressive decoding 语义:
            // prefill 阶段 logits 是 [seq_len, vocab] 多行矩阵,只有最后一个位置的
            // logits 对"下一个 token 预测"有意义。flat argmax 会跨 seq 选到 vocab 外
            // 的位置 (token id > vocab_size) 导致生成乱码。decode 阶段 seq_len=1,
            // 天然就是 single-row logits, 不需裁剪。
            let vocab = config.vocab_size();
            let logits_data = if vocab > 0 && full_logits.len() >= vocab && full_logits.len() % vocab == 0 {
                let rows = full_logits.len() / vocab;
                if rows > 1 {
                    full_logits[(rows - 1) * vocab..].to_vec()
                } else {
                    full_logits
                }
            } else {
                full_logits
            };
            logits_handles.push(LogitsHandle { data: logits_data });
            telemetry.push(crate::scheduler::SequenceTelemetry::new());
        }

        Ok((logits_handles, total_compute_ms, telemetry))
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsHandle,
        _topology: &AttentionTopology,
        vocab_size: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        let data = &logits.data;
        if data.is_empty() {
            return Err(BE::Cpu("empty logits in sample_from_tensor".into()));
        }

        // 支持多序列 batch：若 vocab_size > 0 且 data.len() 是其整数倍，按行分别采样。
        // 否则退化为单序列行。
        let (rows, row_len) = if vocab_size > 0 && data.len() % vocab_size == 0 {
            (data.len() / vocab_size, vocab_size)
        } else {
            (1, data.len())
        };

        let mut tokens = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * row_len;
            let row = &data[start..start + row_len];
            tokens.push(super::sampling::sample_logits_row(row, sampling)?);
        }
        Ok(tokens)
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        // ARCH-FULL-JIT: 通过 FusedGraphExecutor JIT 路径执行
        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = executor.run(&inputs, &bindings).map_err(|e| {
            BE::Other(format!("FusedGraphExecutor embedding run failed: {e}"))
        })?;
        // 提取最终 hidden state → mean pool over seq dim → embedding 向量
        let hidden = extract_final_hidden(&outputs, executor)?;
        let hidden_size = config.hidden_size();
        let seq_len = tokens.len();
        Ok(mean_pool_hidden(&hidden, seq_len, hidden_size))
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        // ARCH-FULL-JIT: 通过 FusedGraphExecutor JIT 路径执行
        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = executor.run(&inputs, &bindings).map_err(|e| {
            BE::Other(format!("FusedGraphExecutor rerank run failed: {e}"))
        })?;
        let hidden = extract_final_hidden(&outputs, executor)?;
        if hidden.is_empty() {
            return Err(BE::Other("rerank forward produced empty output tensor".into()));
        }

        use crate::manifest::ArchFamily;
        let is_decoder = config.arch_family == ArchFamily::Decoder;
        // ARCH-RERANK-DECODER (SPEC 01-REQUIREMENTS REQ-TEST-004):
        // Decoder-based rerankers (Qwen3-Reranker) 生成 "yes"/"no" token,
        // 分数 = logits[yes_id] - logits[no_id]。YAML 不包含 lm_head (共享
        // embed_tokens), 这里 post-graph 做一次 last-token × embed.T 的 matmul
        // 只计算 yes/no 两个 vocab 位置, 避开完整 vocab 投影。
        if is_decoder {
            let yes_id = config.rerank_yes_token_id.map(|v| v as usize);
            let no_id = config.rerank_no_token_id.map(|v| v as usize);
            let hidden_size = config.hidden_size();
            let seq_len = tokens.len();
            if seq_len > 0 && hidden.len() >= seq_len * hidden_size {
                let last_start = (seq_len - 1) * hidden_size;
                let last_hidden = &hidden[last_start..last_start + hidden_size];
                // 取 embed_tokens 权重 (tied lm_head)
                const EMBED_ALIASES: &[&str] = &[
                    "model.embed_tokens.weight",
                    "embed_tokens.weight",
                    "transformer.wte.weight",
                    "embeddings.word_embeddings.weight",
                    "token_embd.weight",
                ];
                let mut embed_data: Option<(&[E], Vec<usize>)> = None;
                for alias in EMBED_ALIASES {
                    if let Some(t) = weights.get_tensor(alias) {
                        let shape = weights.tensor_shape(alias).map(|s| s.to_vec()).unwrap_or_default();
                        if shape.len() == 2 {
                            embed_data = Some((t.as_ref(), shape));
                            break;
                        }
                    }
                }
                if let (Some((data_e, shape)), Some(yes), Some(no)) = (embed_data, yes_id, no_id) {
                    // CPU 后端固定 f32
                    if std::mem::size_of::<E>() != std::mem::size_of::<f32>() {
                        return Err(BE::Other("rerank decoder path 仅支持 f32 backend".into()));
                    }
                    let data: &[f32] = unsafe {
                        std::slice::from_raw_parts(data_e.as_ptr() as *const f32, data_e.len())
                    };
                    let vocab = shape[0];
                    let hidden_dim = shape[1];
                    if hidden_dim != hidden_size {
                        return Err(BE::Other(format!(
                            "rerank embed shape mismatch: expected hidden={hidden_size}, got {hidden_dim}")));
                    }
                    let row_dot = |row_idx: usize| -> f32 {
                        let row = &data[row_idx * hidden_dim..(row_idx + 1) * hidden_dim];
                        let mut s = 0.0f64;
                        for i in 0..hidden_dim {
                            s += last_hidden[i] as f64 * row[i] as f64;
                        }
                        s as f32
                    };
                    if yes >= vocab || no >= vocab {
                        return Err(BE::Other(format!(
                            "rerank yes/no token id out of vocab: yes={yes} no={no} vocab={vocab}")));
                    }
                    let logit_yes = row_dot(yes);
                    let logit_no = row_dot(no);
                    // softmax(yes, no) 的 log-ratio 提供 [0,1] 风格分数。
                    // 直接 yes - no 是 log-odds;此处返回 softmax(yes) over {yes, no}。
                    let maxv = logit_yes.max(logit_no);
                    let ey = (logit_yes - maxv).exp();
                    let en = (logit_no - maxv).exp();
                    let score = ey / (ey + en);
                    return Ok(vec![score]);
                }
                // Fallback:无 embed 权重或 token id,返回 last token hidden[0]
                return Ok(vec![last_hidden.get(0).copied().unwrap_or(0.0)]);
            }
        }

        // Encoder reranker (xlmr-reranker 等) 已把 classifier 编织进图,
        // 输出 rerank_logit 的 shape 是 [batch, seq_len, 1] (每 token 一个 logit,
        // Linear 对 token 独立作用)。CLS token (position 0) 的 logit 是最终相关性分数。
        Ok(vec![hidden[0]])
    }

    fn classify_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        // ARCH-FULL-JIT: 通过 FusedGraphExecutor JIT 路径执行
        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = executor.run(&inputs, &bindings).map_err(|e| {
            BE::Other(format!("FusedGraphExecutor classify run failed: {e}"))
        })?;
        let hidden = extract_final_hidden(&outputs, executor)?;
        let hidden_size = config.hidden_size();
        // 分类: [CLS] token 表示
        Ok(hidden[..hidden_size].to_vec())
    }

    /// Head Routing SDK (REQ-HR-001 / REQ-HR-002) — 对 decoder generator 的
    /// 最后一个 token hidden state 与指定 vocab token 的 embed 行做点积,返回
    /// 每个 target_token_ids 对应的原始 logit。依赖 tied embedding
    /// (lm_head.weight == embed_tokens.weight),与 `rerank_forward_gpu_pure`
    /// decoder 分支同源实现。
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §4.2
    /// - SPEC/04-API-DESIGN.md §3.8
    fn score_tokens_forward_gpu_pure(
        &self,
        tokens: &[u32],
        target_token_ids: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        if tokens.is_empty() {
            return Err(BE::Other(
                "score_tokens_forward_gpu_pure: empty tokens".into(),
            ));
        }
        if target_token_ids.is_empty() {
            return Ok(Vec::new());
        }

        use crate::manifest::ArchFamily;
        let is_decoder = config.arch_family == ArchFamily::Decoder;
        if !is_decoder {
            return Err(BE::Unimplemented(
                "score_tokens_forward_gpu_pure currently requires decoder generator model",
            ));
        }

        // ARCH-FULL-JIT: 走 FusedGraphExecutor 单次前向。Guardrail callbacks
        // 可通过 `config.callback_chain_ptr` 在 forward 中途注入 post_node
        // 回调 (GuardrailProbeCallback 计算 safety 分数 / veto)。
        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = run_with_optional_callbacks(executor, &inputs, &bindings, config)?;

        // SPEC/GUARDRAIL.md §5.1 step 3-4: empty outputs + active callback chain
        // = Guardrail HaltAndVeto. Return empty logits so Client surfaces the
        // veto via `GuardrailAttachment::is_vetoed()` rather than computing a
        // classifier score from truncated hidden state.
        if outputs.is_empty() && !config.callback_chain_ptr.is_null() {
            return Ok(Vec::new());
        }

        let hidden = extract_final_hidden(&outputs, executor)?;
        if hidden.is_empty() {
            return Err(BE::Other(
                "score_tokens forward produced empty output tensor".into(),
            ));
        }

        let hidden_size = config.hidden_size();
        let seq_len = tokens.len();
        if hidden.len() < seq_len * hidden_size {
            return Err(BE::Other(format!(
                "score_tokens: hidden buffer too small, expected >= {}*{}={}, got {}",
                seq_len,
                hidden_size,
                seq_len * hidden_size,
                hidden.len()
            )));
        }
        let last_start = (seq_len - 1) * hidden_size;
        let last_hidden = &hidden[last_start..last_start + hidden_size];

        // Tied embedding: lm_head 权重与 embed_tokens 共享,直接读 embed_tokens.
        const EMBED_ALIASES: &[&str] = &[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "token_embd.weight",
        ];
        let mut embed_data: Option<(&[E], Vec<usize>)> = None;
        for alias in EMBED_ALIASES {
            if let Some(t) = weights.get_tensor(alias) {
                let shape = weights
                    .tensor_shape(alias)
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                if shape.len() == 2 {
                    embed_data = Some((t.as_ref(), shape));
                    break;
                }
            }
        }
        let (data_e, shape) = embed_data.ok_or_else(|| {
            BE::Other(
                "score_tokens: embed_tokens.weight not found in weight store \
                 (tried model.embed_tokens.weight / embed_tokens.weight / \
                 transformer.wte.weight / embeddings.word_embeddings.weight / \
                 token_embd.weight)"
                    .into(),
            )
        })?;

        // CPU backend: Element size 必须与 f32 相同
        if std::mem::size_of::<E>() != std::mem::size_of::<f32>() {
            return Err(BE::Other(
                "score_tokens: CPU backend path only supports f32 element size".into(),
            ));
        }
        // SAFETY: data_e 是 CPU backend 的 Element 存储,E 在 CPU 固定为 f32。
        // 上面的 size_of 断言已验证。
        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_e.as_ptr() as *const f32, data_e.len())
        };
        let vocab = shape[0];
        let hidden_dim = shape[1];
        if hidden_dim != hidden_size {
            return Err(BE::Other(format!(
                "score_tokens embed shape mismatch: expected hidden={}, got {}",
                hidden_size, hidden_dim
            )));
        }

        // 为每个 target token id 计算点积 logit_t = h_last · embed[t]
        let mut logits = Vec::with_capacity(target_token_ids.len());
        for &tid in target_token_ids {
            let t = tid as usize;
            if t >= vocab {
                return Err(BE::Other(format!(
                    "score_tokens: target token id {} out of vocab {}",
                    t, vocab
                )));
            }
            let row = &data[t * hidden_dim..(t + 1) * hidden_dim];
            // f64 accumulator 避免大模型 hidden_size 的浮点误差累积
            let mut acc: f64 = 0.0;
            for i in 0..hidden_dim {
                acc += last_hidden[i] as f64 * row[i] as f64;
            }
            logits.push(acc as f32);
        }
        Ok(logits)
    }

    /// HR `encode_to_layer` / Intent `encode_intent` — run the generator
    /// forward with a `MidLayerEncodeCallback` attached to
    /// `config.callback_chain_ptr`, returning the captured `[seq_len, hidden_size]`
    /// hidden state as flat f32.
    fn encode_at_layer_forward_gpu_pure(
        &self,
        tokens: &[u32],
        anchor_layer: usize,
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        if tokens.is_empty() {
            return Err(BE::Other(
                "encode_at_layer_forward_gpu_pure: empty tokens".into(),
            ));
        }
        use crate::manifest::ArchFamily;
        if config.arch_family != ArchFamily::Decoder {
            return Err(BE::Unimplemented(
                "encode_at_layer_forward_gpu_pure currently requires decoder generator model",
            ));
        }
        let num_layers = config.num_layers();
        if anchor_layer >= num_layers {
            return Err(BE::Other(format!(
                "encode_at_layer: anchor_layer {} out of [0, {})",
                anchor_layer, num_layers
            )));
        }
        if config.callback_chain_ptr.is_null() {
            return Err(BE::Other(
                "encode_at_layer: callback_chain_ptr must be populated (MidLayerEncodeCallback)".into(),
            ));
        }

        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = run_with_optional_callbacks(executor, &inputs, &bindings, config)?;

        // MidLayerEncodeCallback writes hidden state under the `"logits"`
        // key on ExitEarly (see FusedGraphExecutor post-node handler).
        let bytes = outputs
            .get("logits")
            .or_else(|| {
                // Fallback: walk graph outputs if the callback did not trigger
                // (e.g. anchor layer not reached because layer_idx mapping is
                // stale). This is an error, not a fallback path.
                None
            })
            .ok_or_else(|| {
                BE::Other(format!(
                    "encode_at_layer: MidLayerEncodeCallback did not fire at anchor_layer={} — \
                     graph produced no `logits` key; layer_idx mapping may be stale",
                    anchor_layer
                ))
            })?;

        Ok(bytes_to_f32(bytes))
    }

    /// Guardrail SDK — run generator forward with `GuardrailProbeCallback`
    /// attached. Returns the final hidden state on normal completion, or an
    /// empty vector when the guardrail raised ExitEarly (veto path).
    fn apply_guardrail_probe(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        if tokens.is_empty() {
            return Err(BE::Other("apply_guardrail_probe: empty tokens".into()));
        }
        use crate::manifest::ArchFamily;
        if config.arch_family != ArchFamily::Decoder {
            return Err(BE::Unimplemented(
                "apply_guardrail_probe currently requires decoder generator model",
            ));
        }
        if config.callback_chain_ptr.is_null() {
            return Err(BE::Other(
                "apply_guardrail_probe: callback_chain_ptr must be populated".into(),
            ));
        }

        let executor = get_graph_executor(config)?;
        let inputs = prepare_encoder_inputs(tokens, config);
        let bindings = shape_bindings_from_tokens(tokens);
        let outputs = run_with_optional_callbacks(executor, &inputs, &bindings, config)?;

        // SPEC/GUARDRAIL.md §5.1 step 3-4: Guardrail HaltAndVeto emits empty
        // `ExitEarly`, `FusedGraphExecutor` returns empty outputs. Caller reads
        // `GuardrailAttachment::is_vetoed()` for the reason string.
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        let hidden = extract_final_hidden(&outputs, executor)?;
        Ok(hidden)
    }

    fn get_memory_pressure(&self) -> Result<f32, BE> {
        get_system_memory_pressure()
    }

    fn swap_out_pages(
        &self,
        handle: &mut KvCacheHandle,
        mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        if mappings.is_empty() {
            return Ok(());
        }

        let store = self.kv_store.lock().map_err(|e| {
            BE::Cpu(format!("KV store lock poisoned: {e}"))
        })?;
        let buffer = store.get(&handle.0).ok_or_else(|| {
            BE::Cpu(format!("KV cache handle {} not found", handle.0))
        })?;

        let mut swap = self.swap_store.lock().map_err(|e| {
            BE::Cpu(format!("Swap store lock poisoned: {e}"))
        })?;

        let page_size = buffer.page_size;
        let num_layers = buffer.num_layers;
        let num_kv_heads = buffer.num_kv_heads;
        let head_dim = buffer.head_dim;
        let max_seq_len = buffer.max_seq_len;

        let elem_bytes = buffer.elem_bytes;
        let page_bytes = num_layers * num_kv_heads * page_size * head_dim * elem_bytes;

        for &(page_id, storage_key) in mappings {
            let token_start = page_id * page_size;
            if token_start >= max_seq_len {
                return Err(BE::Cpu(format!(
                    "swap_out: page {} starts at token {} beyond max_seq_len {}",
                    page_id, token_start, max_seq_len
                )));
            }

            let mut k_data = vec![0u8; page_bytes];
            let mut v_data = vec![0u8; page_bytes];

            // Copy page data from KV buffer (byte-level)
            // Buffer layout: [num_layers][num_kv_heads][max_seq_len][head_dim] * elem_bytes
            let mut dst_offset = 0;
            for layer in 0..num_layers {
                for head in 0..num_kv_heads {
                    let base = ((layer * num_kv_heads + head) * max_seq_len + token_start) * head_dim * elem_bytes;
                    let len = page_size.min(max_seq_len - token_start) * head_dim * elem_bytes;
                    k_data[dst_offset..dst_offset + len]
                        .copy_from_slice(&buffer.k[base..base + len]);
                    v_data[dst_offset..dst_offset + len]
                        .copy_from_slice(&buffer.v[base..base + len]);
                    dst_offset += page_size * head_dim * elem_bytes;
                }
            }

            swap.insert(storage_key, SwapPageData { k: k_data, v: v_data });
        }

        Ok(())
    }

    fn swap_in_pages(
        &self,
        handle: &mut KvCacheHandle,
        mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        if mappings.is_empty() {
            return Ok(());
        }

        let mut store = self.kv_store.lock().map_err(|e| {
            BE::Cpu(format!("KV store lock poisoned: {e}"))
        })?;
        let buffer = store.get_mut(&handle.0).ok_or_else(|| {
            BE::Cpu(format!("KV cache handle {} not found", handle.0))
        })?;

        let mut swap = self.swap_store.lock().map_err(|e| {
            BE::Cpu(format!("Swap store lock poisoned: {e}"))
        })?;

        let page_size = buffer.page_size;
        let num_layers = buffer.num_layers;
        let num_kv_heads = buffer.num_kv_heads;
        let head_dim = buffer.head_dim;
        let max_seq_len = buffer.max_seq_len;

        for &(page_id, storage_key) in mappings {
            let page_data = swap.remove(&storage_key).ok_or_else(|| {
                BE::Cpu(format!(
                    "swap_in: storage key {} not found in swap store", storage_key
                ))
            })?;

            let token_start = page_id * page_size;
            if token_start >= max_seq_len {
                return Err(BE::Cpu(format!(
                    "swap_in: page {} starts at token {} beyond max_seq_len {}",
                    page_id, token_start, max_seq_len
                )));
            }

            // Copy page data back into KV buffer (byte-level)
            let elem_bytes = buffer.elem_bytes;
            let mut src_offset = 0;
            for layer in 0..num_layers {
                for head in 0..num_kv_heads {
                    let base = ((layer * num_kv_heads + head) * max_seq_len + token_start) * head_dim * elem_bytes;
                    let len = page_size.min(max_seq_len - token_start) * head_dim * elem_bytes;
                    buffer.k[base..base + len]
                        .copy_from_slice(&page_data.k[src_offset..src_offset + len]);
                    buffer.v[base..base + len]
                        .copy_from_slice(&page_data.v[src_offset..src_offset + len]);
                    src_offset += page_size * head_dim * elem_bytes;
                }
            }
        }

        Ok(())
    }

    fn get_page_states(
        &self,
        handle: &KvCacheHandle,
    ) -> Result<Vec<(PageId, PageState)>, BE> {
        let store = self.kv_store.lock().map_err(|e| {
            BE::Cpu(format!("KV store lock poisoned: {e}"))
        })?;
        let buffer = store.get(&handle.0).ok_or_else(|| {
            BE::Cpu(format!("KV cache handle {} not found", handle.0))
        })?;

        let active_pages = buffer.active_pages();
        let total_pages = buffer.total_pages();

        let mut states = Vec::with_capacity(total_pages);
        for page_id in 0..total_pages {
            let state = if page_id < active_pages {
                PageState::Active
            } else {
                PageState::Free
            };
            states.push((page_id, state));
        }
        Ok(states)
    }


    fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BE> {
        Ok(data.to_vec())
    }

    fn upload_weights_f32_owned(&self, data: Vec<f32>) -> Result<Self::Tensor, BE> {
        // Safety: Self::Tensor = Vec<E>, and this is only called when E == f32.
        // Transmute the Vec<f32> to Vec<E> without copying.
        let tensor: Vec<E> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(data);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut E, v.len(), v.capacity())
        };
        Ok(tensor)
    }

    fn quantized_matmul(
        &self,
        weight_blocks: &[u8],
        input: &[E],
        output: &mut [E],
        quant_type: backend_trait::QuantType,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BE> {
        use gllm_kernels::quant::QuantType::*;
        use gllm_kernels::Kernels;
        let kern = gllm_kernels::cpu_kernels::CpuKernels::<E>::new();
        match quant_type {
            // K-Quant family
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K => {
                kern.kquant_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // Classic GGML family
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => {
                kern.classic_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // IQ family
            IQ1S | IQ1M | IQ2XXS | IQ2XS | IQ2S | IQ3XXS | IQ3S | IQ4NL | IQ4XS => {
                kern.iq_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // External formats not supported through this unified path
            AWQ4 | GPTQ4 | Squeeze => {
                return Err(BE::Unimplemented("quantized_matmul: external formats (AWQ4/GPTQ4/Squeeze) require dedicated API"));
            }
            // MXFP4 (OCP Microscaling FP4): dedicated matmul lowering not yet wired
            // (T-MXFP4-LOWER). Use `dequantize()` first, then dispatch to standard f32 matmul.
            Mxfp4 { .. } => {
                return Err(BE::Unimplemented("quantized_matmul: MXFP4 path requires dedicated matmul lowering (T-MXFP4-LOWER)"));
            }
        }
        Ok(())
    }

    fn dequantize(
        &self,
        block_data: &[u8],
        output: &mut [f32],
        quant_type: backend_trait::QuantType,
    ) -> Result<(), BE> {
        use gllm_kernels::quant::QuantType::*;
        use gllm_kernels::Kernels;
        let kern = gllm_kernels::cpu_kernels::CpuKernels::<E>::new();

        // Each dequant_* kernel decodes a single block. We must loop over
        // all blocks in the tensor, advancing the byte and output pointers.
        let blk_elems = quant_type.block_size();
        let blk_bytes = quant_type.block_bytes();

        macro_rules! decode_all_blocks {
            ($method:ident) => {
                for (blk_in, blk_out) in block_data
                    .chunks_exact(blk_bytes)
                    .zip(output.chunks_exact_mut(blk_elems))
                {
                    kern.$method(blk_in, blk_out);
                }
            };
        }

        match quant_type {
            // K-Quant family
            Q2K => decode_all_blocks!(dequant_q2_k),
            Q3K => decode_all_blocks!(dequant_q3_k),
            Q4K => decode_all_blocks!(dequant_q4_k),
            Q5K => decode_all_blocks!(dequant_q5_k),
            Q6K => decode_all_blocks!(dequant_q6_k),
            Q8K => decode_all_blocks!(dequant_q8_k),
            // Classic GGML family
            Q4_0 => decode_all_blocks!(dequant_q4_0),
            Q4_1 => decode_all_blocks!(dequant_q4_1),
            Q5_0 => decode_all_blocks!(dequant_q5_0),
            Q5_1 => decode_all_blocks!(dequant_q5_1),
            Q8_0 => decode_all_blocks!(dequant_q8_0),
            Q8_1 => decode_all_blocks!(dequant_q8_1),
            // IQ family
            IQ1S => decode_all_blocks!(dequant_iq1_s),
            IQ1M => decode_all_blocks!(dequant_iq1_m),
            IQ2XXS => decode_all_blocks!(dequant_iq2_xxs),
            IQ2XS => decode_all_blocks!(dequant_iq2_xs),
            IQ2S => decode_all_blocks!(dequant_iq2_s),
            IQ3XXS => decode_all_blocks!(dequant_iq3_xxs),
            IQ3S => decode_all_blocks!(dequant_iq3_s),
            IQ4NL => decode_all_blocks!(dequant_iq4_nl),
            IQ4XS => decode_all_blocks!(dequant_iq4_xs),
            // External formats not supported through this unified path
            AWQ4 | GPTQ4 | Squeeze => {
                return Err(BE::Unimplemented("dequantize: external formats (AWQ4/GPTQ4/Squeeze) require dedicated API"));
            }
            // MXFP4 (OCP Microscaling FP4): block layout = [scale_byte (e8m0), qs[block_size/2]
            // (packed e2m1)]. Standard `block_size = 32` ⇒ 17 bytes per block. Split scales out
            // and dispatch to the dedicated AVX2/scalar mxfp4 routine.
            Mxfp4 { block_size } => {
                let bytes_per_block = quant_type.block_bytes();
                debug_assert_eq!(bytes_per_block, 1 + block_size / 2);
                let num_blocks = block_data.len() / bytes_per_block;
                if block_data.len() != num_blocks * bytes_per_block {
                    return Err(BE::Unimplemented(
                        "dequantize: MXFP4 input length not a multiple of block_bytes",
                    ));
                }
                if output.len() != num_blocks * block_size {
                    return Err(BE::Unimplemented(
                        "dequantize: MXFP4 output length does not match num_blocks × block_size",
                    ));
                }
                // GGUF MXFP4 layout interleaves scale + qs per block:
                //   block i = block_data[i * bytes_per_block .. (i+1) * bytes_per_block]
                //   scale  = block[0], qs = block[1..]
                // The standalone routine expects two contiguous arrays — repack here.
                let mut scales = Vec::with_capacity(num_blocks);
                let mut packed = Vec::with_capacity(num_blocks * (block_size / 2));
                for blk in 0..num_blocks {
                    let base = blk * bytes_per_block;
                    scales.push(block_data[base]);
                    packed.extend_from_slice(&block_data[base + 1..base + bytes_per_block]);
                }
                gllm_kernels::quant_mxfp4::dequant_mxfp4(&packed, &scales, output, block_size);
            }
        }
        Ok(())
    }
}
