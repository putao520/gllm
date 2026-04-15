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
        topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        kv_caches: &mut [KvCacheHandle],
        config: &GeneratorForwardConfig,
    ) -> Result<(Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>), BE> {
        let _ = (input, topology, weights, kv_caches, config);
        Err(BE::Other("ARCH-FULL-JIT: hand-written decoder_forward deleted, use FusedGraphExecutor".into()))
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsHandle,
        _topology: &AttentionTopology,
        _vocab_size: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        let data = &logits.data;
        if data.is_empty() {
            return Err(BE::Cpu("empty logits in sample_from_tensor".into()));
        }

        // ── Greedy path: temperature == 0 → argmax on raw logits ──
        if sampling.temperature <= 0.0 {
            let best = data.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            return Ok(vec![best]);
        }

        // ── Stochastic path: temperature > 0 → top-k/top-p + multinomial ──
        let scaled: Vec<f32> = data.iter().map(|&x| x / sampling.temperature).collect();

        // Top-k filtering
        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        let effective_k = if sampling.top_k > 0 && sampling.top_k < scaled.len() {
            sampling.top_k
        } else {
            scaled.len()
        };
        indices.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(effective_k);

        // Softmax
        let max_val = scaled[indices[0]];
        let mut probs: Vec<f32> = indices.iter().map(|&i| (scaled[i] - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return Err(BE::Cpu(
                "softmax produced zero-sum probabilities: all logits are -inf after scaling".into(),
            ));
        }

        // Top-p (nucleus) filtering
        if sampling.top_p < 1.0 && sampling.top_p > 0.0 {
            let mut sorted_pairs: Vec<(usize, f32)> = indices.iter().copied()
                .zip(probs.iter().copied())
                .collect();
            sorted_pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut cutoff = sorted_pairs.len();
            for (i, &(_, prob)) in sorted_pairs.iter().enumerate() {
                cumulative += prob;
                if cumulative >= sampling.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            sorted_pairs.truncate(cutoff);

            let new_sum: f32 = sorted_pairs.iter().map(|(_, p)| p).sum();
            indices = sorted_pairs.iter().map(|(i, _)| *i).collect();
            probs = sorted_pairs.iter().map(|(_, p)| p / new_sum).collect();
        }

        // Multinomial sampling: 按概率分布随机抽样
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // 使用 logits 内容 + 系统时间作为 seed 生成伪随机数
        let mut hasher = DefaultHasher::new();
        data.len().hash(&mut hasher);
        // 浮点位模式作为熵源
        for &v in data.iter().take(8) {
            v.to_bits().hash(&mut hasher);
        }
        // 高精度时间戳作为不确定性源
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        nanos.hash(&mut hasher);
        let seed = hasher.finish();

        // xorshift64 快速 PRNG
        let mut rng_state = seed | 1; // 确保非零
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let rand_f32 = (rng_state & 0xFFFFFF) as f32 / 0xFFFFFF as f32; // [0, 1]

        // 累积分布采样
        let mut cumulative = 0.0f32;
        let mut selected = indices[0] as u32;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if rand_f32 <= cumulative {
                selected = indices[i] as u32;
                break;
            }
        }

        Ok(vec![selected])
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        let _ = (tokens, weights, config);
        Err(BE::Other("ARCH-FULL-JIT: hand-written embedding forward deleted, use FusedGraphExecutor".into()))
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        let _ = (tokens, weights, config);
        Err(BE::Other("ARCH-FULL-JIT: hand-written rerank forward deleted, use FusedGraphExecutor".into()))
    }

    fn classify_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        let _ = (tokens, weights, config);
        Err(BE::Other("ARCH-FULL-JIT: hand-written classify forward deleted, use FusedGraphExecutor".into()))
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
        }
        Ok(())
    }
}
