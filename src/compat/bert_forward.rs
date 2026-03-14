use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::weight_helpers::{get_f32_data, get_bias_data, needs_weight_transpose, transpose_f32};
use super::{Element, PoolingMode};
use crate::engine::executor::{BackendError as BE, GeneratorForwardConfig};

// ---------------------------------------------------------------------------
// JIT compilation for BERT encoder layers
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single BERT encoder layer (post-norm).
///
/// JIT compiler will auto-fuse:
/// - GemmBias + Gelu → EpilogueInjection
/// - Q/K/V GemmBias sharing input → ComputeRoot (shared pack_a)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_bert_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;

    // ── Graph inputs ──
    let input = g.add_tensor("input", vec![s, h], dt);

    // Attention weights + biases
    let w_q = g.add_tensor("w_q", vec![h, h], dt);
    let b_q = g.add_tensor("b_q", vec![h], dt);
    let w_k = g.add_tensor("w_k", vec![h, h], dt);
    let b_k = g.add_tensor("b_k", vec![h], dt);
    let w_v = g.add_tensor("w_v", vec![h, h], dt);
    let b_v = g.add_tensor("b_v", vec![h], dt);
    let w_o = g.add_tensor("w_o", vec![h, h], dt);
    let b_o = g.add_tensor("b_o", vec![h], dt);

    // LayerNorm 1 weights
    let ln1_w = g.add_tensor("ln1_w", vec![h], dt);
    let ln1_b = g.add_tensor("ln1_b", vec![h], dt);

    // FFN weights + biases
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let b_up = g.add_tensor("b_up", vec![inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);
    let b_down = g.add_tensor("b_down", vec![h], dt);

    // LayerNorm 2 weights
    let ln2_w = g.add_tensor("ln2_w", vec![h], dt);
    let ln2_b = g.add_tensor("ln2_b", vec![h], dt);

    g.inputs = vec![
        input, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        ln1_w, ln1_b, w_up, b_up, w_down, b_down, ln2_w, ln2_b,
    ];

    // ── Self-Attention ──

    // Q = input * W_q + b_q  [s, h] × [h, h] → [s, h]
    let q_out = g.add_tensor("q", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_q, b_q],
        vec![q_out],
        "gemm_q",
    );

    // K = input * W_k + b_k
    let k_out = g.add_tensor("k", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_k, b_k],
        vec![k_out],
        "gemm_k",
    );

    // V = input * W_v + b_v
    let v_out = g.add_tensor("v", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_v, b_v],
        vec![v_out],
        "gemm_v",
    );

    // Multi-head attention: Q[s,h], K[s,h], V[s,h] → attn_out[s,h]
    let attn_out = g.add_tensor("attn_out", vec![s, h], dt);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, head_dim },
        vec![q_out, k_out, v_out],
        vec![attn_out],
        "mha",
    );

    // Output projection + bias
    let o_out = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![attn_out, w_o, b_o],
        vec![o_out],
        "gemm_o",
    );

    // Residual₁: input + o_out
    let resid1 = g.add_tensor("residual1", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_out],
        vec![resid1],
        "residual_1",
    );

    // Post-attention LayerNorm
    let normed1 = g.add_tensor("normed1", vec![s, h], dt);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![resid1, ln1_w, ln1_b],
        vec![normed1],
        "layer_norm_1",
    );

    // ── FFN ──

    // Up projection + GELU: GemmBias + Gelu → EpilogueInjection candidate
    let up_out = g.add_tensor("ffn_up", vec![s, inter], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: inter, k: h },
        vec![normed1, w_up, b_up],
        vec![up_out],
        "gemm_ffn_up",
    );

    let act_out = g.add_tensor("ffn_act", vec![s, inter], dt);
    g.add_op(
        OpKind::Gelu,
        vec![up_out],
        vec![act_out],
        "gelu",
    );

    // Down projection
    let down_out = g.add_tensor("ffn_down", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: inter },
        vec![act_out, w_down, b_down],
        vec![down_out],
        "gemm_ffn_down",
    );

    // Residual₂: normed1 + down_out
    let resid2 = g.add_tensor("residual2", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![normed1, down_out],
        vec![resid2],
        "residual_2",
    );

    // Post-FFN LayerNorm
    let output = g.add_tensor("output", vec![s, h], dt);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![resid2, ln2_w, ln2_b],
        vec![output],
        "layer_norm_2",
    );

    g.outputs = vec![output];
    g
}

/// Execute a JIT-compiled BERT encoder layer.
///
/// Packs all weight tensors into a contiguous buffer matching the graph's
/// input tensor order, then calls the compiled layer function.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_bert_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    q_w: &[f32], q_b: &[f32],
    k_w: &[f32], k_b: &[f32],
    v_w: &[f32], v_b: &[f32],
    out_w: &[f32], out_b: &[f32],
    ln1_w: &[f32], ln1_b: &[f32],
    ffn_up_w: &[f32], ffn_up_b: &[f32],
    ffn_down_w: &[f32], ffn_down_b: &[f32],
    ln2_w: &[f32], ln2_b: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    // Pack weights contiguously in graph input order:
    // [w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln1_w, ln1_b,
    //  w_up, b_up, w_down, b_down, ln2_w, ln2_b]
    let weight_slices: &[&[f32]] = &[
        q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
        ln1_w, ln1_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b,
        ln2_w, ln2_b,
    ];
    let total_weight_bytes: usize = weight_slices.iter().map(|s| s.len() * 4).sum();
    let mut weights_buf = vec![0u8; total_weight_bytes];
    let mut offset = 0;
    for slice in weight_slices.iter() {
        let bytes = slice.len() * 4;
        weights_buf[offset..offset + bytes].copy_from_slice(
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) }
        );
        offset += bytes;
    }

    let scratchpad_bytes = compiled.scratchpad_bytes;
    let mut scratchpad = vec![0u8; scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr() as *const u8,
            std::ptr::null_mut(), // no KV cache for BERT
            std::ptr::null(),     // no positions
            std::ptr::null(),     // no seq_lens
            1,                    // batch_size = 1
            seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

/// Build a CompilerGraph for mean pooling: average seq_len rows into one.
pub(crate) fn build_mean_pool_graph(
    seq_len: usize,
    hidden: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let output = g.add_tensor("output", vec![hidden], dt);

    g.add_op(
        OpKind::MeanPool { seq_len, hidden },
        vec![input],
        vec![output],
        "mean_pool",
    );

    g.inputs = vec![input];
    g.outputs = vec![output];
    g
}

/// Full BERT encoder forward pass.
pub(crate) fn bert_encoder_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
    pooling: PoolingMode,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::Kernels;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let kern = gllm_kernels::backend::CpuKernels::<f32>::new();
    let transpose_weights = needs_weight_transpose(weights);

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;

    // Build BERT layer graph and attempt JIT compilation.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let jit_layer: Option<gllm_kernels::compiler::CompiledLayer> = {
        let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        match compiler.compile_graph(&graph) {
            Ok(compiled) => {
                Some(compiled)
            }
            Err(e) => {
                return Err(BE::Other(format!("BERT JIT compilation failed: {e}")));
            }
        }
    };

    // Step (a): Token embedding lookup
    // All formats store embeddings as [vocab, hidden] in row-major after dequant.
    // (GGUF shape [ne0=hidden, ne1=vocab] means vocab rows of hidden elements.)
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
    // All formats: [max_pos, hidden] row-major after dequant.
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
            let row = &mut hidden_state[s * hidden..(s + 1) * hidden];
            for i in 0..hidden {
                row[i] += pos_emb[s * hidden + i];
            }
        }
    }

    // Step (c): Add token_type embeddings (all type 0)
    let tt_emb = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    // All formats: [num_types, hidden] row-major. Type 0 is the first row.
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            let row = &mut hidden_state[s * hidden..(s + 1) * hidden];
            for i in 0..hidden {
                row[i] += tt_emb[i]; // first row = type 0
            }
        }
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
        let mut normed = vec![0.0f32; hidden];
        for s in 0..seq_len {
            let row = &hidden_state[s * hidden..(s + 1) * hidden];
            kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
            hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
        }
    }

    // Step (e): Encoder layers
    #[allow(unused_mut, unused_variables)]
    let mut buf_out = vec![0.0f32; seq_len * hidden];
    #[allow(unused_mut, unused_variables)]
    let mut buf_inter = vec![0.0f32; seq_len * inter];
    #[allow(unused_mut, unused_variables)]
    let mut normed = vec![0.0f32; hidden];

    for layer in 0..num_layers {
        // ── JIT fast path ──
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if let Some(ref compiled) = jit_layer {
            let q_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")))?;
            let q_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
            let k_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")))?;
            let k_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
            let v_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")))?;
            let v_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
            let out_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")))?;
            let out_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
            let ln1_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")))?;
            let ln1_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
            let ffn_up_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")))?;
            let ffn_up_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
            let ffn_down_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")))?;
            let ffn_down_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
            let ln2_w = get_f32_data(weights, backend, &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")))?;
            let ln2_b = get_bias_data(weights, &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);


            // SafeTensors stores [out, in]; our GEMM expects [in, out] (B in C=A@B)
            let (q_w, k_w, v_w, out_w, ffn_up_w, ffn_down_w) = if transpose_weights {
                (
                    transpose_f32(&q_w, hidden, hidden),
                    transpose_f32(&k_w, hidden, hidden),
                    transpose_f32(&v_w, hidden, hidden),
                    transpose_f32(&out_w, hidden, hidden),
                    transpose_f32(&ffn_up_w, inter, hidden),
                    transpose_f32(&ffn_down_w, hidden, inter),
                )
            } else {
                (q_w, k_w, v_w, out_w, ffn_up_w, ffn_down_w)
            };

            let mut layer_out = vec![0.0f32; seq_len * hidden];
            execute_jit_bert_layer(
                compiled,
                &hidden_state,
                &q_w, &q_b, &k_w, &k_b, &v_w, &v_b,
                &out_w, &out_b, &ln1_w, &ln1_b,
                &ffn_up_w, &ffn_up_b, &ffn_down_w, &ffn_down_b,
                &ln2_w, &ln2_b,
                seq_len,
                &mut layer_out,
            );

            hidden_state.copy_from_slice(&layer_out);
            continue;
        }

        // No fallback — JIT is mandatory
        return Err(BE::Other("BERT encoder requires JIT compilation".to_string()));
    }
    // Step (f): Output pooling — depends on mode
    match pooling {
        PoolingMode::MeanPool => {
            // Mean pooling over all tokens (JIT-compiled SIMD, no fallback)
            let mut pooled = vec![0.0f32; hidden];
            {
                let pool_graph = build_mean_pool_graph(seq_len, hidden);
                let mut pool_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                let compiled = pool_compiler.compile_graph(&pool_graph)
                    .map_err(|e| BE::Other(format!("MeanPool JIT compilation failed: {e}")))?;
                let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
                unsafe {
                    compiled.execute(
                        hidden_state.as_ptr() as *const u8,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                        1,
                        seq_len,
                        pooled.as_mut_ptr() as *mut u8,
                        scratchpad.as_mut_ptr(),
                    );
                }
            }

            Ok(pooled)
        }
        PoolingMode::ClsClassifier => {
            // [CLS] token extraction + classifier head (for reranker models)
            let cls = hidden_state[..hidden].to_vec();

            // Load classifier head weights
            let dense_w = get_f32_data(weights, backend,
                &crate::weight_names::classifier_aliases("dense.weight"))?;
            let dense_b = get_bias_data(weights,
                &crate::weight_names::classifier_aliases("dense.bias"), hidden);
            let out_proj_w = get_f32_data(weights, backend,
                &crate::weight_names::classifier_aliases("out_proj.weight"))?;
            let out_proj_b = get_bias_data(weights,
                &crate::weight_names::classifier_aliases("out_proj.bias"), 1);

            // dense: x = tanh(W @ cls + b)
            // dense_w is [hidden, hidden] but may be transposed
            let mut x = vec![0.0f32; hidden];
            if transpose_weights {
                // SafeTensors: stored as [out, in], need to compute W^T @ cls
                for i in 0..hidden {
                    let mut sum = dense_b[i];
                    for j in 0..hidden {
                        sum += dense_w[i * hidden + j] * cls[j];
                    }
                    x[i] = sum.tanh();
                }
            } else {
                for i in 0..hidden {
                    let mut sum = dense_b[i];
                    for j in 0..hidden {
                        sum += dense_w[j * hidden + i] * cls[j];
                    }
                    x[i] = sum.tanh();
                }
            }

            // out_proj: logit = W @ x + b
            // out_proj_w shape: [num_labels, hidden] (typically [1, hidden])
            let num_labels = out_proj_w.len() / hidden;
            let mut logit = 0.0f32;
            if num_labels >= 1 {
                for j in 0..hidden {
                    logit += out_proj_w[j] * x[j];
                }
                logit += out_proj_b[0];
            }

            // sigmoid for probability
            let score = 1.0 / (1.0 + (-logit).exp());
            Ok(vec![score])
        }
    }
}
