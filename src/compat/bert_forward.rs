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
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, config.dtype());
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
    inputs.insert("hidden_state".to_string(), hs_bytes);

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
    match pooling {
        PoolingMode::MeanPool => {
            if let Some(out_bytes) = output.get("pool_out").or_else(|| output.get("embedding")).or_else(|| output.values().next()) {
                let pooled: Vec<f32> = out_bytes
                    .chunks_exact(4)
                    .map(|c| {
                        let arr: [u8; 4] = c.try_into().unwrap_or([0; 4]); // LEGAL: 字节对齐边界，chunks_exact(4) 保证 4 字节对齐
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
                let arr: [u8; 4] = out_bytes[0..4].try_into().unwrap_or([0; 4]); // LEGAL: 字节对齐边界，前面已检查 len >= 4
                let logit = f32::from_le_bytes(arr);
                let score = 1.0 / (1.0 + (-logit).exp());
                Ok(vec![score])
            } else {
                Err(BE::Other("GraphExecutor produced no score output".into()))
            }
        }
    }
}
