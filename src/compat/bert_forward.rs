use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::weight_helpers::{get_f32_data, get_bias_data};
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

    let _kern = gllm_kernels::backend::CpuKernels::<f32>::new();

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

    // Step (a): Token embedding lookup
    // All formats store embeddings as [vocab, hidden] in row-major after dequant.
    let word_emb = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")),
    )?;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    {
        let vocab = word_emb.len() / hidden;
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for word_embeddings (vocab {})", tok, vocab
                )));
            }
            hidden_state[s * hidden..(s + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
        }
    }

    // Step (b): Add position embeddings (positions 0..seq_len)
    let pos_emb = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")),
    )?;
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
        _kern.vec_add(&hidden_state, &pos_emb[..seq_len * hidden], &mut added);
        hidden_state.copy_from_slice(&added);
    }

    // Step (c): Add token_type embeddings (all type 0)
    let tt_emb = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    // All formats: [num_types, hidden] row-major. Type 0 is the first row.
    if tt_emb.len() >= hidden {
        let tt_broadcast: Vec<f32> = tt_emb[..hidden].iter().cloned().cycle().take(seq_len * hidden).collect();
        let mut added = vec![0.0f32; seq_len * hidden];
        _kern.vec_add(&hidden_state, &tt_broadcast, &mut added);
        hidden_state.copy_from_slice(&added);
    }

    // Step (d): Embedding LayerNorm
    let emb_ln_w = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")),
    )?;
    let emb_ln_b = get_bias_data(
        weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")),
        hidden,
    );
    {
        let mut ln_out = vec![0.0f32; seq_len * hidden];
        for s in 0..seq_len {
            _kern.layer_norm(
                &hidden_state[s * hidden..(s + 1) * hidden],
                &emb_ln_w,
                &emb_ln_b,
                &mut ln_out[s * hidden..(s + 1) * hidden],
                eps,
            );
        }
        hidden_state.copy_from_slice(&ln_out);
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
    let hs_bytes: Vec<u8> = hidden_state
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    inputs.insert("hidden_state".to_string(), hs_bytes);

    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let output = ge.run_with_kv_cache(
        &inputs,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
        seq_len,
        positions.as_ptr(),
    ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

    // Step (f): Output extraction based on pooling mode
    match pooling {
        PoolingMode::MeanPool => {
            if let Some(out_bytes) = output.get("pool_out").or_else(|| output.get("embedding")).or_else(|| output.values().next()) {
                let pooled: Vec<f32> = out_bytes
                    .chunks_exact(4)
                    .map(|c| {
                        let arr: [u8; 4] = c.try_into().unwrap_or([0; 4]);
                        f32::from_le_bytes(arr)
                    })
                    .collect();
                Ok(pooled)
            } else {
                Err(BE::Other("GraphExecutor produced no pool_out output".into()))
            }
        }
        PoolingMode::ClsClassifier => {
            if let Some(out_bytes) = output.get("score").or_else(|| output.values().next()) {
                if out_bytes.len() < 4 {
                    return Err(BE::Other("Invalid score bytes length from GraphExecutor".into()));
                }
                let arr: [u8; 4] = out_bytes[0..4].try_into().unwrap_or([0; 4]);
                let logit = f32::from_le_bytes(arr);
                let score = 1.0 / (1.0 + (-logit).exp());
                Ok(vec![score])
            } else {
                Err(BE::Other("GraphExecutor produced no score output".into()))
            }
        }
    }
}
