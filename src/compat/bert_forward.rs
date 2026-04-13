use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::jit_helpers::TypedBuffer;
use super::weight_helpers::{get_bias_data_typed, get_typed_data};
use super::jit_helpers::typed_bytes_to_f32;
use super::{Element, PoolingMode};
use crate::engine::executor::{BackendError as BE, GeneratorForwardConfig};

// ---------------------------------------------------------------------------
// Unified GraphExecutor BERT forward pass
// ---------------------------------------------------------------------------

/// Full BERT encoder forward pass.
pub(crate) fn bert_encoder_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
    pooling: PoolingMode,
) -> Result<Vec<f32>, BE> {
    #[allow(unused_imports)]
    use gllm_kernels::Kernels;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let _kern = gllm_kernels::cpu_kernels::CpuKernels::<f32>::new();

    let seq_len = tokens.len();
    let hidden = config.hidden_size();
    let eps = config.norm_eps();

    // Step (a): Token embedding lookup
    // All formats store embeddings as [vocab, hidden] in row-major after dequant.
    let (word_emb_bytes, word_emb_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")),
    )?;
    let word_emb = typed_bytes_to_f32(&word_emb_bytes, word_emb_dtype);
    // P3: TypedBuffer 替换 vec![0.0f32]，使用 config.dtype() 初始化
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, gllm_kernels::types::DType::F32);
    {
        let vocab = word_emb.len() / hidden;
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for word_embeddings (vocab {})", tok, vocab
                )));
            }
            hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
        }
    }

    // Step (b): Add position embeddings (positions 0..seq_len)
    let (pos_emb_bytes, pos_emb_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")),
    )?;
    let pos_emb = typed_bytes_to_f32(&pos_emb_bytes, pos_emb_dtype);
    {
        let max_pos = pos_emb.len() / hidden;
        for s in 0..seq_len {
            if s >= max_pos {
                return Err(BE::Other(format!(
                    "position {} out of range for position_embeddings (max {})", s, max_pos
                )));
            }
        }
        let mut added = vec![0.0f32; seq_len * hidden];
        _kern.vec_add(hidden_state.as_f32(), &pos_emb[..seq_len * hidden], &mut added);
        hidden_state.as_f32_mut().copy_from_slice(&added);
    }

    // Step (c): Add token_type embeddings (all type 0)
    let (tt_emb_bytes, tt_emb_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    let tt_emb = typed_bytes_to_f32(&tt_emb_bytes, tt_emb_dtype);
    // All formats: [num_types, hidden] row-major. Type 0 is the first row.
    if tt_emb.len() >= hidden {
        let tt_broadcast: Vec<f32> = tt_emb[..hidden].iter().cloned().cycle().take(seq_len * hidden).collect();
        let mut added = vec![0.0f32; seq_len * hidden];
        _kern.vec_add(hidden_state.as_f32(), &tt_broadcast, &mut added);
        hidden_state.as_f32_mut().copy_from_slice(&added);
    }

    // Step (d): Embedding LayerNorm
    let (emb_ln_w_bytes, emb_ln_w_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")),
    )?;
    let emb_ln_w = typed_bytes_to_f32(&emb_ln_w_bytes, emb_ln_w_dtype);
    let (emb_ln_b_bytes, emb_ln_b_dtype) = get_bias_data_typed(
        weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")),
        hidden,
    );
    let emb_ln_b = typed_bytes_to_f32(&emb_ln_b_bytes, emb_ln_b_dtype);
    {
        let mut ln_out = vec![0.0f32; seq_len * hidden];
        for s in 0..seq_len {
            _kern.layer_norm(
                &hidden_state.as_f32()[s * hidden..(s + 1) * hidden],
                &emb_ln_w,
                &emb_ln_b,
                &mut ln_out[s * hidden..(s + 1) * hidden],
                eps,
            );
        }
        hidden_state.as_f32_mut().copy_from_slice(&ln_out);
    }

    // ── Graph executor path (YAML→JIT, ONLY valid path) ──
    if config.graph_executor_ptr.is_null() {
        return Err(BE::Other(
            "BERT encoder requires the unified GraphExecutor (ARCH-CPU-GPU-UNIFIED). \
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
    // Seed embedding sub-graph outputs so is_node_computed() skips them.
    // xlmr.yaml embedding chain:
    //   embed_tokens(Gather) → embed_tok
    //   embed_pos(Gather) → embed_pos_out
    //   embed_type(Gather) → embed_type_out
    //   embed_add_pos(Add) → embed_sum_1
    //   embed_add_type(Add) → embed_sum
    //   embed_norm(LayerNorm) → hidden_0
    // bert_forward already executed all of these in Rust; the JIT-compiled
    // transformer layers start from "hidden_0". Each embedding node is skipped
    // because its output tensor is already present in the inputs map.
    inputs.insert("hidden_0".to_string(), hs_bytes.clone());
    inputs.insert("embed_tok".to_string(), hs_bytes.clone());
    inputs.insert("embed_pos_out".to_string(), hs_bytes.clone());
    inputs.insert("embed_type_out".to_string(), vec![0u8; hidden]);
    inputs.insert("embed_sum_1".to_string(), hs_bytes.clone());
    inputs.insert("embed_sum".to_string(), hs_bytes.clone());

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

    // Step (f): Output extraction based on pooling mode
    // GraphExecutor returns "hidden_0" which is [seq_len, hidden] in row-major f32.
    // Apply pooling to produce a single hidden-dim vector.
    let hidden_bytes = output.get("hidden_0")
        .or_else(|| output.get("pool_out"))
        .or_else(|| output.get("embedding"))
        .or_else(|| output.values().next())
        .ok_or_else(|| BE::Other("GraphExecutor produced no output".into()))?;
    let hidden_f32: Vec<f32> = hidden_bytes
        .chunks_exact(4)
        .map(|c| {
            let arr: [u8; 4] = c.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(arr)
        })
        .collect();

    match pooling {
        PoolingMode::MeanPool => {
            // Mean across seq_len dimension: result shape [hidden]
            if hidden_f32.len() != seq_len * hidden {
                return Err(BE::Other(format!(
                    "GraphExecutor output size mismatch: got {} f32 elements, expected {} ({}*{})",
                    hidden_f32.len(), seq_len * hidden, seq_len, hidden
                )));
            }
            let mut pooled = vec![0.0f32; hidden];
            for s in 0..seq_len {
                for h in 0..hidden {
                    pooled[h] += hidden_f32[s * hidden + h] / seq_len as f32;
                }
            }
            Ok(pooled)
        }
        PoolingMode::ClsClassifier => {
            // CLS token: first row [0..hidden]
            if hidden_f32.len() < hidden {
                return Err(BE::Other("GraphExecutor output too small for CLS".into()));
            }
            // For classifier models, hidden_0 is the encoder output;
            // a separate classifier head would be applied elsewhere.
            // Here we return the CLS embedding as a single-vector "score proxy".
            let cls = hidden_f32[..hidden].to_vec();
            if cls.len() == 1 {
                // Single logit → sigmoid → probability
                let logit = cls[0];
                let score = 1.0 / (1.0 + (-logit).exp());
                Ok(vec![score])
            } else {
                Ok(cls)
            }
        }
    }
}

/// BERT/XLM-R encoder-based classifier forward pass.
///
/// Flow:
/// 1. Full encoder forward (same as `bert_encoder_forward`)
/// 2. Extract CLS token hidden state
/// 3. Apply pooler dense + tanh (if weights exist)
/// 4. Apply classifier head (dense → logits)
/// 5. Return raw logits (caller applies softmax/argmax)
pub(crate) fn bert_classifier_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    // Step 1: Run encoder and get CLS token hidden state
    let cls_hidden = bert_encoder_forward(backend, tokens, weights, config, PoolingMode::ClsClassifier)?;
    let hidden = config.hidden_size();

    let _kern = gllm_kernels::cpu_kernels::CpuKernels::<f32>::new();

    // Step 2: Apply pooler dense + tanh (optional — some models don't have a pooler)
    let pooled = match (
        get_typed_data(weights, backend, &crate::weight_names::pooler_aliases("weight")),
        get_bias_data_typed(weights, &crate::weight_names::pooler_aliases("bias"), hidden),
    ) {
        (Ok((pw_bytes, pw_dtype)), (pb_bytes, pb_dtype)) => {
            let pw = typed_bytes_to_f32(&pw_bytes, pw_dtype);
            let pb = typed_bytes_to_f32(&pb_bytes, pb_dtype);
            // pooler dense: out = tanh(cls @ W^T + b)
            // W shape: [hidden, hidden], cls shape: [hidden]
            let out_dim = if !pw.is_empty() && hidden > 0 { pw.len() / hidden } else { hidden };
            let mut pooled_out = vec![0.0f32; out_dim];
            for o in 0..out_dim {
                let mut sum = pb.get(o).copied().unwrap_or(0.0);
                for i in 0..hidden {
                    sum += cls_hidden.get(i).copied().unwrap_or(0.0) * pw.get(o * hidden + i).copied().unwrap_or(0.0);
                }
                pooled_out[o] = sum.tanh();
            }
            pooled_out
        }
        _ => cls_hidden,
    };

    // Step 3: Apply classifier head (dense → logits)
    // Try classifier.weight first, then classifier.dense.weight
    let (cw_bytes, cw_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::classifier_aliases("weight"),
    ).map_err(|_| BE::Other(
        "classifier head weight not found. Model may not have a classification head. \
         Expected: classifier.weight, classifier.dense.weight, or pre_classifier.weight".into()
    ))?;
    let cw = typed_bytes_to_f32(&cw_bytes, cw_dtype);
    let (cb_bytes, cb_dtype) = get_bias_data_typed(
        weights,
        &crate::weight_names::classifier_aliases("bias"),
        0,
    );
    let cb = typed_bytes_to_f32(&cb_bytes, cb_dtype);

    let pooled_dim = pooled.len();
    let num_labels = if pooled_dim > 0 { cw.len() / pooled_dim } else { 0 };
    if num_labels == 0 {
        return Err(BE::Other(format!(
            "classifier weight dimension mismatch: weight has {} elements, pooled dim is {}",
            cw.len(), pooled_dim
        )));
    }

    // logits = pooled @ classifier_W^T + classifier_b
    let mut logits = vec![0.0f32; num_labels];
    for l in 0..num_labels {
        let mut sum = cb.get(l).copied().unwrap_or(0.0);
        for i in 0..pooled_dim {
            sum += pooled[i] * cw.get(l * pooled_dim + i).copied().unwrap_or(0.0);
        }
        logits[l] = sum;
    }

    Ok(logits)
}
