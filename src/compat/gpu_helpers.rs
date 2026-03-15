//! Generic GPU helper functions that eliminate CUDA/HIP/Metal triplication.
//!
//! All functions are parameterized over `B: Backend<E>` so a single implementation
//! serves all three GPU backends. The logic is byte-for-byte identical to the
//! original per-backend versions — only the type parameter differs.

use super::backend_trait::{self, Backend};

use super::Element;
use crate::engine::executor::BackendError as BE;

/// Get tensor data as f32, trying quantized dequant first, then f32 fallback.
///
/// Replaces `get_f32_data_cuda` / `get_f32_data_hip` / `get_f32_data_metal`.
pub(super) fn get_f32_data_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    _backend: &B,
    aliases: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    // Try quantized path first
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut out = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes)
                .zip(out.chunks_exact_mut(blk_elems))
            {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4NL => kern.dequant_iq4_nl(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4XS => kern.dequant_iq4_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1S => kern.dequant_iq1_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1M => kern.dequant_iq1_m(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XXS => kern.dequant_iq2_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XS => kern.dequant_iq2_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2S => kern.dequant_iq2_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3XXS => kern.dequant_iq3_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3S => kern.dequant_iq3_s(blk_in, blk_out),
                    _ => {
                        return Err(BE::Other(format!(
                            "Unsupported quantization type {:?} for weight {:?}",
                            qt.quant_type, name.as_ref()
                        )));
                    }
                }
            }
            return Ok(out);
        }
    }

    // Fall back to f32 tensor
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = tensor.as_ref();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const f32,
                    std::mem::size_of_val(e_slice) / 4,
                )
            };
            return Ok(slice.to_vec());
        }
    }

    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Get bias data (zeros if not found).
///
/// Replaces `get_bias_data_cuda` / `get_bias_data_hip` / `get_bias_data_metal`.
pub(super) fn get_bias_data_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    aliases: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = tensor.as_ref();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const f32,
                    std::mem::size_of_val(e_slice) / 4,
                )
            };
            return slice.to_vec();
        }
    }
    vec![0.0f32; size]
}

/// Check if weights need transposition (SafeTensors stores [out, in]).
///
/// Replaces `needs_weight_transpose_cuda` / `_hip` / `_metal`.
pub(super) fn needs_weight_transpose_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
) -> bool {
    crate::weight_names::has_any_embedding_weight(|n| weights.get_tensor(n).is_some())
}

/// CPU-side token embedding lookup for GPU backends.
///
/// Replaces `embed_tokens_cpu` / `embed_tokens_cpu_hip` / `embed_tokens_cpu_metal`.
pub(super) fn embed_tokens_gpu<E: Element, B: Backend<E>>(
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

    // Word embeddings
    let word_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")),
    )?;

    let seq_len = tokens.len();
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
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

    // Position embeddings
    let pos_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")),
    )?;
    let max_pos = pos_emb.len() / hidden;
    for s in 0..seq_len {
        if s >= max_pos {
            return Err(BE::Other(format!(
                "position {} out of range for position_embeddings (max {})", s, max_pos
            )));
        }
        for i in 0..hidden {
            hidden_state[s * hidden + i] += pos_emb[s * hidden + i];
        }
    }

    // Token type embeddings (type 0)
    let tt_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            for i in 0..hidden {
                hidden_state[s * hidden + i] += tt_emb[i];
            }
        }
    }

    // Embedding LayerNorm
    let emb_ln_w = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")),
    )?;
    let emb_ln_b = get_bias_data_gpu(
        weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")),
        hidden,
    );
    {
        let kern = gllm_kernels::backend::CpuKernels::<f32>::new();
        use gllm_kernels::Kernels;
        let mut normed = vec![0.0f32; hidden];
        for s in 0..seq_len {
            let row = &hidden_state[s * hidden..(s + 1) * hidden];
            kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
            hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
        }
    }

    Ok(hidden_state)
}

/// Load BERT encoder layer weights into a HashMap for JIT execution.
///
/// Replaces `load_bert_layer_weights_cuda` / `_hip` / `_metal`.
pub(super) fn load_bert_layer_weights_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_gpu(weights, backend, $aliases)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            m.insert($graph_name.to_string(), get_bias_data_gpu(weights, $aliases, $size));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    if transpose {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = crate::compat::weight_helpers::transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}
