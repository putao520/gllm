use super::backend_trait::{self, Backend};
use super::memory::get_system_memory_pressure;
use super::{Element, PoolingMode};
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
}

impl KvCacheBuffer {
    fn new(config: &KvCacheConfig) -> Self {
        let elem_bytes = config.dtype_size.max(1);
        let total_bytes = config.num_layers * config.num_heads * config.max_seq_len * config.head_dim * elem_bytes;
        Self {
            k: vec![0u8; total_bytes],
            v: vec![0u8; total_bytes],
            num_layers: config.num_layers,
            num_kv_heads: config.num_heads,
            max_seq_len: config.max_seq_len,
            head_dim: config.head_dim,
            page_size: config.page_size,
            seq_len: 0,
            elem_bytes,
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
    ) -> Result<(Vec<LogitsHandle>, f32), BE> {
        if config.kernel_strategy != crate::scheduler::jit_types::KernelStrategy::AccuracyFirst {
            log::info!("cpu_backend: executing with {:?} strategy", config.kernel_strategy);
        }
        use crate::engine::executor::AttentionMaskType;
        match topology.mask_type {
            AttentionMaskType::Causal => {
                super::decoder_forward::decoder_forward(self, input, weights, kv_caches, config)
            }
            AttentionMaskType::Bidirectional => {
                // BERT-style encoder: handled by embedding/rerank paths, not batch_forward
                Err(BE::Other("batch_forward_gpu_pure does not support bidirectional attention; use embedding_forward_gpu_pure".into()))
            }
        }
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

        // Apply temperature scaling
        let temperature = if sampling.temperature <= 0.0 { 1e-8 } else { sampling.temperature };
        let scaled: Vec<f32> = data.iter().map(|&x| x / temperature).collect();

        // Top-k filtering: keep only top_k largest logits
        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        let effective_k = if sampling.top_k > 0 && sampling.top_k < scaled.len() {
            sampling.top_k
        } else {
            scaled.len()
        };

        // Partial sort to find top-k
        indices.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(effective_k);

        // Softmax over selected tokens
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
            // Sort by descending probability
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

            // Re-normalize
            let new_sum: f32 = sorted_pairs.iter().map(|(_, p)| p).sum();
            indices = sorted_pairs.iter().map(|(i, _)| *i).collect();
            probs = sorted_pairs.iter().map(|(_, p)| p / new_sum).collect();
        }

        // Greedy selection (deterministic): pick the highest probability token.
        // For stochastic sampling we'd use a RNG, but deterministic is safer
        // for accuracy-first design and test reproducibility.
        let best = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| indices[i] as u32)
            .unwrap_or(0);

        Ok(vec![best])
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        match config.arch_family {
            crate::manifest::ArchFamily::Decoder => {
                // Decoder-based embedding model (Qwen3-Embedding, etc.)
                super::decoder_forward::decoder_embedding_forward(self, tokens, weights, config)
            }
            crate::manifest::ArchFamily::Encoder => {
                // BERT-style embedding model (e5-small, XLM-R, etc.)
                super::bert_forward::bert_encoder_forward(self, tokens, weights, config, PoolingMode::MeanPool)
            }
        }
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        match config.arch_family {
            crate::manifest::ArchFamily::Decoder => {
                // Decoder-based reranker (Qwen3-Reranker, etc.)
                super::decoder_forward::decoder_rerank_forward(self, tokens, weights, config)
            }
            crate::manifest::ArchFamily::Encoder => {
                // BERT-style reranker (bge-reranker, etc.)
                super::bert_forward::bert_encoder_forward(self, tokens, weights, config, PoolingMode::ClsClassifier)
            }
        }
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
        let kern = gllm_kernels::backend::CpuKernels::<E>::new();
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
        let kern = gllm_kernels::backend::CpuKernels::<E>::new();

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
