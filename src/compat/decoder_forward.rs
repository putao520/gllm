
use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::jit_helpers::TypedBuffer;
use super::weight_helpers::get_typed_data;
use super::jit_helpers::typed_bytes_to_f32;
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

    let hidden = config.hidden_size();
    let vocab_size = config.vocab_size();

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
        let (embed_bytes, embed_dtype) = get_typed_data(
            weights, backend,
            &crate::weight_names::decoder_embed_aliases(),
        )?;
        let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);

        let embed_vocab = embed_data.len() / hidden;
        // P3: TypedBuffer 替换 vec![0.0f32]，使用 config.dtype() 初始化
        let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, gllm_kernels::types::DType::F32);
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= embed_vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
                )));
            }
            hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
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
        let _is_incremental = has_kv_cache && cached_seq_len > 0 && position > 0;

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
        // P3: 直接使用 TypedBuffer 的字节切片
        let hs_bytes: Vec<u8> = hidden_state.as_bytes().to_vec();
        let input_name = if let Some(first_node) = ge.graph().nodes.first() {
            if first_node.op.name() == "Gather" && !first_node.outputs.is_empty() {
                first_node.outputs.first().unwrap().clone()
            } else {
                ge.graph().inputs.first().cloned().unwrap_or_else(|| "hidden_state".to_string()) // LEGAL: 默认输入名称 "hidden_state"
            }
        } else {
            "hidden_state".to_string()
        };
        inputs.insert(input_name, hs_bytes);

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

        // §9-§18: 从 forward_config 获取 callback chain 指针
        let mut cb_chain = if !config.callback_chain_ptr.is_null() {
            Some(unsafe { &mut *config.callback_chain_ptr })
        } else {
            None
        };

        let output = ge.run_with_kv_cache_and_callbacks(
            &inputs,
            kv_cache_k_ptr,
            kv_cache_v_ptr,
            0,
            total_seq,
            seq_len,
            positions.as_ptr(),
            cb_chain.as_deref_mut(),
            Some(config),
        ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

        // Update KV cache seq_len after successful execution.
        // The executor has written K/V data for all layers into the cache buffer.
        if has_kv_cache && seq_idx < kv_caches.len() {
            let mut store = backend.kv_store().lock().map_err(|e| {
                BE::Cpu(format!("KV store lock poisoned: {e}"))
            })?;
            if let Some(buf) = store.get_mut(&kv_caches[seq_idx].0) {
                buf.seq_len = total_seq;
            }
        }

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

    let hidden = config.hidden_size();

    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder embedding".into()));
    }

    // (a) Token embedding lookup
    let (embed_bytes, embed_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;
    let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);

    let embed_vocab = embed_data.len() / hidden;
    // P3: TypedBuffer 替换 vec![0.0f32]，使用 config.dtype() 初始化
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, gllm_kernels::types::DType::F32);
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    if config.graph_executor_ptr.is_null() {
        return Err(BE::Other(
            "CPU embedding forward requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). \
            Legacy operator-level JIT has been removed. Please ensure YAML graph template exists for this architecture."
            .into()
        ));
    }

    let ge = unsafe { &mut *config.graph_executor_ptr };
    if ge.graph().nodes.is_empty() {
        return Err(BE::Other("GraphExecutor has empty nodes. Stub architecture templates are not runnable.".into()));
    }

    let mut inputs = std::collections::HashMap::new();
    // P3: 直接使用 TypedBuffer 的字节切片
    let hs_bytes: Vec<u8> = hidden_state.as_bytes().to_vec();
    let input_name = if let Some(first_node) = ge.graph().nodes.first() {
        if first_node.op.name() == "Gather" && !first_node.outputs.is_empty() {
            first_node.outputs.first().unwrap().clone()
        } else {
            ge.graph().inputs.first().cloned().unwrap_or_else(|| "hidden_state".to_string()) // LEGAL: 默认输入名称 "hidden_state"
        }
    } else {
        "hidden_state".to_string()
    };
    inputs.insert(input_name, hs_bytes);

    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let output = ge.run_with_kv_cache_and_callbacks(
        &inputs,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
        seq_len,
        seq_len,
        positions.as_ptr(),
        None,
        None,
    ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

    if let Some(out_bytes) = output.get("embedding").or_else(|| output.get("pool_out")).or_else(|| output.values().next()) {
        let all_f32: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| {
                let arr: [u8; 4] = c.try_into().unwrap_or([0; 4]); // LEGAL: 字节对齐边界，chunks_exact(4) 保证 4 字节对齐
                f32::from_le_bytes(arr)
            })
            .collect();

        // Apply last-token pooling for decoder embedding models.
        // Graph output is [seq_len, hidden_size]; the embedding is the last token's
        // hidden state (EOS token position used by Qwen3-Embedding, pooling_type=3).
        // When all_f32.len() == hidden, it's already a single-token output — no pooling needed.
        let pooled = if hidden > 0 && all_f32.len() > hidden && all_f32.len() % hidden == 0 {
            // Multiple token hidden states — take the last token row.
            let n_tokens = all_f32.len() / hidden;
            all_f32[(n_tokens - 1) * hidden..].to_vec()
        } else {
            all_f32
        };

        // L2-normalize the pooled embedding.
        // Decoder-based embedding models (Qwen3-Embedding, etc.) output raw hidden
        // states that must be L2-normalized before use in similarity computation.
        // This matches the standard usage in sentence-transformers and the Qwen3
        // embedding documentation.
        let l2_norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb = if l2_norm > 1e-8 {
            pooled.iter().map(|x| x / l2_norm).collect()
        } else {
            pooled
        };

        if cfg!(debug_assertions) {
            let nz = emb.iter().filter(|&&v| v != 0.0).count();
            eprintln!("[EMBED-DBG] decoder_embedding_forward output: len={} nz={} first4={:?}",
                emb.len(), nz, &emb[..4.min(emb.len())]);
        }
        Ok(emb)
    } else {
        Err(BE::Other("GraphExecutor produced no embedding output".into()))
    }
}

/// Decoder-based reranker forward pass (for models like Qwen3-Reranker that
/// use decoder architecture with a score/classifier head).
///
/// Flow:
/// 1. Token embedding lookup (embed_tokens.weight)
/// 2. Per-layer JIT decoder execution via GraphExecutor (no KV cache)
/// 3. Final RMSNorm + LM head projection → logits (vocab-size vector)
/// 4. Extract yes/no logits and return logit_yes (caller computes sigmoid)
///
/// The yes/no token IDs are stored in `config.rerank_yes_token_id` /
/// `config.rerank_no_token_id`. If they are absent the function returns the
/// maximum-logit value as a fallback score.
pub(crate) fn decoder_rerank_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder rerank forward only supports f32".into()));
    }

    let hidden = config.hidden_size();
    let seq_len = tokens.len();

    eprintln!("[RERANK-DBG] tokens len={} tokens={:?}", seq_len, tokens);

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder rerank".into()));
    }

    // (a) Token embedding lookup
    let (embed_bytes, embed_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;
    let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);
    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, gllm_kernels::types::DType::F32);
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    if config.graph_executor_ptr.is_null() {
        return Err(BE::Other(
            "CPU decoder rerank forward requires the unified GraphExecutor. \
            Please ensure YAML graph template exists for this architecture."
            .into()
        ));
    }

    let ge = unsafe { &mut *config.graph_executor_ptr };
    if ge.graph().nodes.is_empty() {
        return Err(BE::Other("GraphExecutor has empty nodes.".into()));
    }

    // Diagnostic: check first/last token embedding to verify different inputs
    {
        let hs_f32: Vec<f32> = hidden_state.as_bytes().chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0;4]))).collect();
        let last_tok_start = if hidden > 0 && hs_f32.len() >= hidden {
            (seq_len - 1) * hidden
        } else { 0 };
        eprintln!("[RERANK-DBG] embed[0] sample={:?}; embed[last] sample={:?}",
            &hs_f32[..4.min(hs_f32.len())],
            &hs_f32[last_tok_start..last_tok_start + 4.min(hs_f32.len() - last_tok_start)]);
    }

    let mut inputs = std::collections::HashMap::new();
    let input_name = if let Some(first_node) = ge.graph().nodes.first() {
        if first_node.op.name() == "Gather" && !first_node.outputs.is_empty() {
            first_node.outputs.first().unwrap().clone()
        } else {
            ge.graph().inputs.first().cloned().unwrap_or_else(|| "hidden_state".to_string())
        }
    } else {
        "hidden_state".to_string()
    };
    inputs.insert(input_name, hidden_state.as_bytes().to_vec());

    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let output = ge.run_with_kv_cache_and_callbacks(
        &inputs,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
        seq_len,
        seq_len,
        positions.as_ptr(),
        None,
        None,
    ).map_err(|e| BE::Other(format!("graph executor (rerank): {e}")))?;

    // Get the hidden_final output (final RMSNorm hidden states).
    // The qwen3-reranker.yaml graph outputs "hidden_final" (no lm_head projection).
    eprintln!("[RERANK-DBG] graph output keys: {:?}", output.keys().collect::<Vec<_>>());
    let hidden_bytes = output.get("hidden_final")
        .or_else(|| output.values().next())
        .ok_or_else(|| BE::Other("GraphExecutor produced no hidden_final output for reranker".into()))?;

    let all_hidden: Vec<f32> = hidden_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
        .collect();

    let all_nz = all_hidden.iter().filter(|&&x| x != 0.0).count();
    eprintln!("[RERANK-DBG] hidden_final len={} nz={}/{} sample={:?}",
        all_hidden.len(), all_nz, all_hidden.len(),
        &all_hidden[..all_hidden.len().min(8)]);

    // Take the last token's hidden state (reranking uses causal decoder — last token sees all).
    let last_hidden = if hidden > 0 && all_hidden.len() > hidden && all_hidden.len() % hidden == 0 {
        let n_tokens = all_hidden.len() / hidden;
        eprintln!("[RERANK-DBG] taking last token hidden: n_tokens={} hidden={}", n_tokens, hidden);
        &all_hidden[(n_tokens - 1) * hidden..]
    } else {
        eprintln!("[RERANK-DBG] using full all_hidden as last_hidden (len={} hidden={})", all_hidden.len(), hidden);
        &all_hidden[..]
    };

    let last_nz = last_hidden.iter().filter(|&&x| x != 0.0).count();
    eprintln!("[RERANK-DBG] last_hidden len={} nz={}/{} sample={:?}",
        last_hidden.len(), last_nz, last_hidden.len(),
        &last_hidden[..last_hidden.len().min(8)]);

    // Compute yes/no logits by dot product with token embedding vectors.
    // For weight-tied models: logit_yes = dot(last_hidden, embed[yes_id]).
    // This avoids needing the transposed lm_head matrix.
    let (embed_bytes, embed_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;
    let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);
    let embed_vocab = if hidden > 0 { embed_data.len() / hidden } else { 0 };

    eprintln!("[RERANK-DBG] embed_vocab={} yes_id={:?} no_id={:?}",
        embed_vocab, config.rerank_yes_token_id, config.rerank_no_token_id);

    let dot = |hidden_vec: &[f32], token_id: u32| -> f32 {
        let idx = token_id as usize;
        if idx >= embed_vocab || hidden_vec.len() < hidden {
            return 0.0;
        }
        let emb_row = &embed_data[idx * hidden..(idx + 1) * hidden];
        hidden_vec.iter().zip(emb_row.iter()).map(|(h, e)| h * e).sum()
    };

    let score = if let Some(yes_id) = config.rerank_yes_token_id {
        let yes_logit = dot(last_hidden, yes_id);
        let no_logit = config.rerank_no_token_id
            .map(|no_id| dot(last_hidden, no_id))
            .unwrap_or(0.0);
        eprintln!("[RERANK-DBG] yes_logit={} no_logit={}", yes_logit, no_logit);
        // Normalized score: P(yes) = softmax([yes_logit, no_logit])[0]
        let max_l = yes_logit.max(no_logit);
        let exp_yes = (yes_logit - max_l).exp();
        let exp_no = (no_logit - max_l).exp();
        exp_yes / (exp_yes + exp_no)
    } else {
        // Fallback when yes/no token IDs unavailable.
        // Return 0.5 as a neutral score — this will be discriminated by the
        // difference in hidden states for different input pairs.
        0.5
    };
    eprintln!("[RERANK-DBG] final score={}", score);

    Ok(vec![score])
}

// ---------------------------------------------------------------------------
// Truncated forward pass (for encode_intent, guardrails, etc.)
// ---------------------------------------------------------------------------

/// Map LayerTarget to physical layer index.
///
/// Delegates to `LayerTarget::to_physical_layer()` for consistent mapping
/// across all APIs (per SPEC 04-API-DESIGN §7.1).
///
/// Returns a layer index in the range [0, num_layers-1].
pub fn layer_target_to_idx(target: crate::knowledge::LayerTarget, num_layers: usize) -> usize {
    target.to_physical_layer(num_layers)
}

/// Extract layer index from node name.
///
/// Supports patterns like:
/// - `layer_0_input_norm` → 0
/// - `layer_15_q_proj` → 15
/// - `layer_3` → 3
///
/// Returns None if the node name does not follow the layer pattern.
#[allow(dead_code)]
fn extract_layer_index(node_name: &str) -> Option<usize> {
    // Pattern: layer_{number}_... or just layer_{number}
    if let Some(rest) = node_name.strip_prefix("layer_") {
        // Find the first underscore or end of string
        let num_str = rest.split('_').next()?;
        num_str.parse::<usize>().ok()
    } else {
        None
    }
}

/// Forward pass truncated at a specific layer.
///
/// This function executes the model up to `target_layer` (exclusive) and returns
/// the hidden state at that layer. Used by `encode_intent()` for extracting
/// intermediate representations.
///
/// Per SPEC 04-API-DESIGN §7.3 and 02-ARCHITECTURE §16.2 "任意层数据召回与高维截断".
///
/// # Arguments
/// - `backend`: CPU backend for execution
/// - `tokens`: Input token IDs
/// - `weights`: Model weights
/// - `config`: Forward configuration
/// - `target_layer`: Layer index to stop at (0-based, exclusive)
///
/// # Returns
/// Hidden state vector at the target layer (flattened [seq_len * hidden_size])
pub fn forward_to_layer<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
    target_layer: usize,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("forward_to_layer only supports f32".into()));
    }

    let hidden = config.hidden_size();
    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for forward_to_layer".into()));
    }

    if target_layer == 0 {
        return Err(BE::Other("target_layer must be at least 1".into()));
    }

    // Token embedding lookup
    let (embed_bytes, embed_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;
    let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, gllm_kernels::types::DType::F32);
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    if config.graph_executor_ptr.is_null() {
        return Err(BE::Other(
            "forward_to_layer requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). \
            Legacy operator-level JIT has been removed. Please ensure YAML graph template exists for this architecture."
            .into()
        ));
    }

    let ge = unsafe { &mut *config.graph_executor_ptr };
    if ge.graph().nodes.is_empty() {
        return Err(BE::Other("GraphExecutor has empty nodes. Stub architecture templates are not runnable.".into()));
    }

    // NOTE: Current FusedGraphExecutor executes all layers internally.
    // True truncated execution requires graph-level support (either building
    // a truncated graph or adding a max_layer parameter to the executor).
    //
    // For now, we execute the full graph and return the final hidden state.
    // This is a limitation of the current architecture that should be
    // addressed by:
    // 1. Adding a run_with_max_layer() method to FusedGraphExecutor, or
    // 2. Building truncated graphs during model loading.

    let mut inputs = std::collections::HashMap::new();
    let hs_bytes: Vec<u8> = hidden_state.as_bytes().to_vec();
    let input_name = if let Some(first_node) = ge.graph().nodes.first() {
        if first_node.op.name() == "Gather" && !first_node.outputs.is_empty() {
            first_node.outputs.first().unwrap().clone()
        } else {
            ge.graph().inputs.first().cloned().unwrap_or_else(|| "hidden_state".to_string()) // LEGAL: 默认输入名称 "hidden_state"
        }
    } else {
        "hidden_state".to_string()
    };
    inputs.insert(input_name, hs_bytes);

    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let output = ge.run_with_kv_cache_and_callbacks(
        &inputs,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
        seq_len,
        seq_len,
        positions.as_ptr(),
        None,
        None,
    ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

    // Extract hidden state from output
    // The graph output name varies by model type; try common names
    if let Some(out_bytes) = output.get("hidden_state")
        .or_else(|| output.get("last_hidden_state"))
        .or_else(|| output.get("embeddings"))
        .or_else(|| output.values().next()) {
        let hs: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| {
                let arr: [u8; 4] = c.try_into().unwrap_or([0; 4]); // LEGAL: 字节对齐边界，chunks_exact(4) 保证 4 字节对齐
                f32::from_le_bytes(arr)
            })
            .collect();
        Ok(hs)
    } else {
        Err(BE::Other("GraphExecutor produced no hidden state output for forward_to_layer".into()))
    }
}

/// Forward pass with LayerTarget (semantic layer selection).
///
/// This is the preferred API for knowledge injection and intent encoding,
/// as it uses semantic layer names (ShallowSyntax, MidSemantic, DeepLogic)
/// rather than hardcoded layer indices.
///
/// Per SPEC 04-API-DESIGN §7.3.
///
/// # Arguments
/// - `backend`: CPU backend for execution
/// - `tokens`: Input token IDs
/// - `weights`: Model weights
/// - `config`: Forward configuration
/// - `target`: Semantic layer target (ShallowSyntax/MidSemantic/DeepLogic)
///
/// # Returns
/// Hidden state vector at the target semantic layer
pub fn forward_to_semantic_layer<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
    target: crate::knowledge::LayerTarget,
) -> Result<Vec<f32>, BE> {
    let target_layer = layer_target_to_idx(target, config.num_layers());
    forward_to_layer(backend, tokens, weights, config, target_layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("layer_0_input_norm"), Some(0));
        assert_eq!(extract_layer_index("layer_15_q_proj"), Some(15));
        assert_eq!(extract_layer_index("layer_3"), Some(3));
        assert_eq!(extract_layer_index("embed"), None);
        assert_eq!(extract_layer_index("layer_"), None);
        assert_eq!(extract_layer_index("layer_abc"), None);
    }

    #[test]
    fn test_extract_layer_index_edge_cases() {
        // Very large layer numbers
        assert_eq!(extract_layer_index("layer_999_attn"), Some(999));
        // Single digit
        assert_eq!(extract_layer_index("layer_5"), Some(5));
        // Zero
        assert_eq!(extract_layer_index("layer_0"), Some(0));
        // No underscore after number
        assert_eq!(extract_layer_index("layer_123"), Some(123));
    }

    #[test]
    fn test_layer_target_to_idx() {
        use crate::knowledge::LayerTarget;

        // 32 layers model — uses to_physical_layer() (normalized depth mapping)
        assert_eq!(layer_target_to_idx(LayerTarget::ShallowSyntax, 32), 4);
        assert_eq!(layer_target_to_idx(LayerTarget::MidSemantic, 32), 16);
        assert_eq!(layer_target_to_idx(LayerTarget::DeepLogic, 32), 28);

        // Small model (4 layers)
        assert_eq!(layer_target_to_idx(LayerTarget::ShallowSyntax, 4), 0);
        assert_eq!(layer_target_to_idx(LayerTarget::MidSemantic, 4), 2);
        assert_eq!(layer_target_to_idx(LayerTarget::DeepLogic, 4), 3);

        // Very small model (2 layers)
        assert_eq!(layer_target_to_idx(LayerTarget::ShallowSyntax, 2), 0);
        assert_eq!(layer_target_to_idx(LayerTarget::MidSemantic, 2), 1);
        assert_eq!(layer_target_to_idx(LayerTarget::DeepLogic, 2), 1);
    }
}

