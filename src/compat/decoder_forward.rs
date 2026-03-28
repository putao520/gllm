
use super::backend_trait;
use super::cpu_backend::CpuBackend;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::weight_helpers::get_f32_data;
use super::Element;
use crate::engine::executor::{
    BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheHandle, LogitsHandle,
};

// Legacy CPU incremental decode logic (MoeDecodeCachedJit, DecodeCachedJit, etc.)
// has been purged in accordance with ARCH-CPU-GPU-UNIFIED and REQ-JIT-CACHE-001.


// ---------------------------------------------------------------------------
// Full decoder forward pass
// ---------------------------------------------------------------------------

/// Full decoder forward pass for a single sequence.
///
/// Pipeline:
/// 1. Token embedding lookup
/// 2. For each layer: JIT-compiled decoder layer + KV cache update
/// 3. Final RMSNorm + lm_head projection → logits
///
/// Returns logits for the last token position only (for generation).
pub(crate) fn decoder_forward<E: Element>(
    backend: &CpuBackend<E>,
    input: &BatchInput,
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
) -> Result<(Vec<LogitsHandle>, f32), BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder forward only supports f32 element type".into()));
    }

    let hidden = config.hidden_size;
    let vocab_size = config.vocab_size;

    let mut results = Vec::with_capacity(input.sequences.len());
    let total_sparsity = 0.0f32;
    let sparsity_layers = 0u32;

    for (seq_idx, seq) in input.sequences.iter().enumerate() {
        let tokens = &seq.tokens;
        let position = seq.position;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(BE::Other("empty sequence in decoder forward".into()));
        }

        // (a) Token embedding lookup
        let embed_data = get_f32_data(
            weights, backend,
            &crate::weight_names::decoder_embed_aliases(),
        )?;

        let embed_vocab = embed_data.len() / hidden;
        let mut hidden_state = vec![0.0f32; seq_len * hidden];
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= embed_vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
                )));
            }
            hidden_state[s * hidden..(s + 1) * hidden]
                .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
        }


        // (c) Determine if this is an incremental decode step (position > 0 with KV cache)
        let has_kv_cache = seq_idx < kv_caches.len();
        let cached_seq_len = if has_kv_cache {
            let mut store = backend.kv_store().lock().map_err(|e| {
                BE::Cpu(format!("KV store lock poisoned: {e}"))
            })?;
            // Reset KV cache on new prefill (position == 0) to avoid stale data
            if position == 0 {
                if let Some(buf) = store.get_mut(&kv_caches[seq_idx].0) {
                    buf.seq_len = 0;
                }
            }
            store.get(&kv_caches[seq_idx].0)
                .ok_or_else(|| BE::Cpu(format!("KV cache entry missing for handle {:?}", kv_caches[seq_idx].0)))?
                .seq_len
        } else {
            0
        };
        let is_incremental = has_kv_cache && cached_seq_len > 0 && position > 0;

        // ── Graph executor path (YAML→JIT, ONLY valid path) ──
        if config.graph_executor_ptr.is_null() {
            return Err(BE::Other(
                "Decoder forward requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). \
                Legacy operator-level JIT has been removed. Please ensure YAML graph template exists for this architecture."
                .into()
            ));
        }

        let ge = unsafe { &mut *config.graph_executor_ptr };
        if ge.graph().nodes.is_empty() {
            return Err(BE::Other("GraphExecutor has empty nodes. Stub architecture templates are not runnable.".into()));
        }

        let mut inputs = std::collections::HashMap::new();
        let hs_bytes: Vec<u8> = hidden_state
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        inputs.insert("hidden_state".to_string(), hs_bytes);

        // Get KV cache pointers for this sequence
        let (kv_cache_k_ptr, kv_cache_v_ptr) = if has_kv_cache && seq_idx < kv_caches.len() {
            let store = backend.kv_store().lock().map_err(|e| {
                BE::Cpu(format!("KV store lock poisoned: {e}"))
            })?;
            if let Some(buf) = store.get(&kv_caches[seq_idx].0) {
                let k_ptr = buf.k.as_ptr() as *mut f32;
                let v_ptr = buf.v.as_ptr() as *mut f32;
                (k_ptr, v_ptr)
            } else {
                (std::ptr::null_mut(), std::ptr::null_mut())
            }
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };

        let total_seq = cached_seq_len + seq_len;
        let positions: Vec<u32> = (0..seq_len).map(|i| (position + i) as u32).collect();

        let output = ge.run_with_kv_cache(
            &inputs,
            kv_cache_k_ptr,
            kv_cache_v_ptr,
            0, // layer 0; graph executor handles all layers internally
            total_seq,
            positions.as_ptr(),
        ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

        // Extract logits from graph output
        if let Some(logits_bytes) = output.get("logits").or_else(|| output.values().next()) {
            let logits: Vec<f32> = logits_bytes
                .chunks_exact(4)
                .map(|c| {
                    let arr: [u8; 4] = c.try_into().map_err(|_| BE::Other("invalid f32 bytes in logits output".into()))?;
                    Ok(f32::from_le_bytes(arr))
                })
                .collect::<Result<Vec<f32>, BE>>()?;
            // Return last-token logits
            let last_start = if logits.len() >= vocab_size {
                logits.len() - vocab_size
            } else {
                0
            };
            results.push(LogitsHandle { data: logits[last_start..].to_vec() });
        } else {
            return Err(BE::Other("GraphExecutor produced no recognizable output".into()));
        }
    }

    let avg_sparsity = if sparsity_layers > 0 {
        total_sparsity / sparsity_layers as f32
    } else {
        0.0
    };
    Ok((results, avg_sparsity))
}

// ---------------------------------------------------------------------------
// Decoder-based Embedding Forward (for Qwen3-Embedding, etc.)
// ---------------------------------------------------------------------------

/// Decoder-based embedding forward pass (for models like Qwen3-Embedding that
/// use decoder architecture with RoPE instead of BERT-style absolute position embeddings).
///
/// Flow:
/// 1. Token embedding lookup (embed_tokens.weight)
/// 2. Per-layer JIT decoder execution (no KV cache)
/// 3. Final RMSNorm
/// 4. Mean pooling → output vector
pub(crate) fn decoder_embedding_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder embedding forward only supports f32".into()));
    }

    let hidden = config.hidden_size;

    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder embedding".into()));
    }

    // (a) Token embedding lookup
    let embed_data = get_f32_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    Err(BE::Other("CPU embedding forward requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). Legacy operator-level JIT has been removed.".into()))
}

/// Decoder-based reranker forward pass (for models like Qwen3-Reranker that
/// use decoder architecture with a score/classifier head).
///
/// Flow:
/// 1. Token embedding lookup (embed_tokens.weight)
/// 2. Per-layer JIT decoder execution (no KV cache)
/// 3. Final RMSNorm
/// 4. Last token hidden state → score head → sigmoid → relevance score
pub(crate) fn decoder_rerank_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder rerank forward only supports f32".into()));
    }

    let hidden = config.hidden_size;

    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder rerank".into()));
    }

    // (a) Token embedding lookup
    let embed_data = get_f32_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    Err(BE::Other("CPU rerank forward requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). Legacy operator-level JIT has been removed.".into()))
}

