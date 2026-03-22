use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::jit_helpers::pack_weights;
use super::types::BertLayerWeights;
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
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;           // GEMM weight dtype
    let ft = dtype;           // activation / norm / bias dtype
    let s = seq_len;
    let h = hidden;

    // ── Graph inputs ──
    let input = g.add_tensor_concrete("input", &[s, h], ft);

    // Attention weights (GEMM dtype) + biases (F32)
    let w_q = g.add_tensor_concrete("w_q", &[h, h], dt);
    let b_q = g.add_tensor_concrete("b_q", &[h], ft);
    let w_k = g.add_tensor_concrete("w_k", &[h, h], dt);
    let b_k = g.add_tensor_concrete("b_k", &[h], ft);
    let w_v = g.add_tensor_concrete("w_v", &[h, h], dt);
    let b_v = g.add_tensor_concrete("b_v", &[h], ft);
    let w_o = g.add_tensor_concrete("w_o", &[h, h], dt);
    let b_o = g.add_tensor_concrete("b_o", &[h], ft);

    // LayerNorm 1 weights (F32)
    let ln1_w = g.add_tensor_concrete("ln1_w", &[h], ft);
    let ln1_b = g.add_tensor_concrete("ln1_b", &[h], ft);

    // FFN weights (GEMM dtype) + biases (F32)
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let b_up = g.add_tensor_concrete("b_up", &[inter], ft);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
    let b_down = g.add_tensor_concrete("b_down", &[h], ft);

    // LayerNorm 2 weights (F32)
    let ln2_w = g.add_tensor_concrete("ln2_w", &[h], ft);
    let ln2_b = g.add_tensor_concrete("ln2_b", &[h], ft);

    g.inputs = vec![
        input, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        ln1_w, ln1_b, w_up, b_up, w_down, b_down, ln2_w, ln2_b,
    ];

    // ── Self-Attention ──

    // Q = input * W_q + b_q  [s, h] × [h, h] → [s, h]
    let q_out = g.add_tensor_concrete("q", &[s, h], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h, dtype: dt },
        vec![input, w_q, b_q],
        vec![q_out],
        "gemm_q",
    );

    // K = input * W_k + b_k
    let k_out = g.add_tensor_concrete("k", &[s, h], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h, dtype: dt },
        vec![input, w_k, b_k],
        vec![k_out],
        "gemm_k",
    );

    // V = input * W_v + b_v
    let v_out = g.add_tensor_concrete("v", &[s, h], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h, dtype: dt },
        vec![input, w_v, b_v],
        vec![v_out],
        "gemm_v",
    );

    // Multi-head attention: Q[s,h], K[s,h], V[s,h] → attn_out[s,h]
    let attn_out = g.add_tensor_concrete("attn_out", &[s, h], ft);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, num_kv_heads: num_heads, head_dim },
        vec![q_out, k_out, v_out],
        vec![attn_out],
        "mha",
    );

    // Output projection + bias
    let o_out = g.add_tensor_concrete("o_proj", &[s, h], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h, dtype: dt },
        vec![attn_out, w_o, b_o],
        vec![o_out],
        "gemm_o",
    );

    // Residual₁: input + o_out
    let resid1 = g.add_tensor_concrete("residual1", &[s, h], ft);
    g.add_op(
        OpKind::Residual,
        vec![input, o_out],
        vec![resid1],
        "residual_1",
    );

    // Post-attention LayerNorm
    let normed1 = g.add_tensor_concrete("normed1", &[s, h], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![resid1, ln1_w, ln1_b],
        vec![normed1],
        "layer_norm_1",
    );

    // ── FFN ──

    // Up projection + GELU: GemmBias + Gelu → EpilogueInjection candidate
    let up_out = g.add_tensor_concrete("ffn_up", &[s, inter], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: inter, k: h, dtype: dt },
        vec![normed1, w_up, b_up],
        vec![up_out],
        "gemm_ffn_up",
    );

    let act_out = g.add_tensor_concrete("ffn_act", &[s, inter], ft);
    g.add_op(
        OpKind::Gelu,
        vec![up_out],
        vec![act_out],
        "gelu",
    );

    // Down projection
    let down_out = g.add_tensor_concrete("ffn_down", &[s, h], ft);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: inter, dtype: dt },
        vec![act_out, w_down, b_down],
        vec![down_out],
        "gemm_ffn_down",
    );

    // Residual₂: normed1 + down_out
    let resid2 = g.add_tensor_concrete("residual2", &[s, h], ft);
    g.add_op(
        OpKind::Residual,
        vec![normed1, down_out],
        vec![resid2],
        "residual_2",
    );

    // Post-FFN LayerNorm
    let output = g.add_tensor_concrete("output", &[s, h], ft);
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
/// ARCH-DTYPE-ADAPTIVE: GEMM weights packed at model dtype, bias/norm at F32.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_bert_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    weights: &BertLayerWeights,
    seq_len: usize,
    output: &mut [f32],
    dtype: gllm_kernels::types::DType,
) {
    use crate::compat::jit_helpers::pack_weights_multi;
    let ft = dtype;
    let weights_buf = pack_weights_multi(&[
        (weights.q_w, dtype), (weights.q_b, ft), (weights.k_w, dtype), (weights.k_b, ft),
        (weights.v_w, dtype), (weights.v_b, ft), (weights.out_w, dtype), (weights.out_b, ft),
        (weights.ln1_w, ft), (weights.ln1_b, ft), (weights.ffn_up_w, dtype), (weights.ffn_up_b, ft),
        (weights.ffn_down_w, dtype), (weights.ffn_down_b, ft), (weights.ln2_w, ft), (weights.ln2_b, ft),
    ]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
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
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let ft = dtype; // activation dtype

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], ft);
    let output = g.add_tensor_concrete("output", &[hidden], ft);

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
    #[allow(unused_imports)]
    use gllm_kernels::Kernels;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let _kern = gllm_kernels::backend::CpuKernels::<f32>::new();
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
    let jit_layer: Option<std::sync::Arc<gllm_kernels::compiler::CompiledLayer>> = {
        use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
        use crate::compat::jit_helpers::{computation_dtype_from_config, kernels_dtype_to_compat};
        let dt = computation_dtype_from_config(config);
        let key = JitCacheKey {
            arch: ModelArchKey {
                arch_name: "bert".to_string(),
                hidden_size: hidden,
                num_heads,
                num_kv_heads: num_heads,
                head_dim,
                dtype: kernels_dtype_to_compat(dt),
            },
            graph: GraphType::BertLayer { inter_size: inter },
        };
        let compiled = global_jit_cache()
            .get_or_compile(key, || {
                let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps, dt);
                let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                compiler.compile_graph(&graph).map_err(|e| format!("BERT JIT compilation failed: {e}"))
            })
            .map_err(|e| BE::Other(e))?;
        Some(compiled)
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
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let added = super::jit_helpers::jit_add(
                super::jit_helpers::f32_as_bytes(&hidden_state),
                super::jit_helpers::f32_as_bytes(&pos_emb[..seq_len * hidden]),
                super::jit_helpers::computation_dtype_from_config(config),
            )
                .map_err(|e| BE::Other(format!("pos embed add JIT failed: {e}")))?;
            hidden_state.copy_from_slice(super::jit_helpers::bytes_as_f32(&added));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return Err(BE::Other("pos embed add JIT requires x86_64 or aarch64".to_string())); }
    }

    // Step (c): Add token_type embeddings (all type 0)
    let tt_emb = get_f32_data(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    // All formats: [num_types, hidden] row-major. Type 0 is the first row.
    if tt_emb.len() >= hidden {
        let tt_broadcast: Vec<f32> = tt_emb[..hidden].iter().cloned().cycle().take(seq_len * hidden).collect();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let added = super::jit_helpers::jit_add(
                super::jit_helpers::f32_as_bytes(&hidden_state),
                super::jit_helpers::f32_as_bytes(&tt_broadcast),
                super::jit_helpers::computation_dtype_from_config(config),
            )
                .map_err(|e| BE::Other(format!("token type embed add JIT failed: {e}")))?;
            hidden_state.copy_from_slice(super::jit_helpers::bytes_as_f32(&added));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return Err(BE::Other("token type embed add JIT requires x86_64 or aarch64".to_string())); }
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
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let result = super::jit_helpers::jit_layer_norm(
                super::jit_helpers::f32_as_bytes(&hidden_state),
                super::jit_helpers::f32_as_bytes(&emb_ln_w),
                super::jit_helpers::f32_as_bytes(&emb_ln_b),
                seq_len, hidden,
                super::jit_helpers::computation_dtype_from_config(config),
            )
                .map_err(|e| BE::Other(format!("embedding LayerNorm JIT failed: {e}")))?;
            hidden_state.copy_from_slice(super::jit_helpers::bytes_as_f32(&result));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        compile_error!("embedding LayerNorm requires JIT support (x86_64 or aarch64)");
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
            let bert_weights = BertLayerWeights {
                q_w: &q_w, q_b: &q_b, k_w: &k_w, k_b: &k_b,
                v_w: &v_w, v_b: &v_b, out_w: &out_w, out_b: &out_b,
                ln1_w: &ln1_w, ln1_b: &ln1_b,
                ffn_up_w: &ffn_up_w, ffn_up_b: &ffn_up_b,
                ffn_down_w: &ffn_down_w, ffn_down_b: &ffn_down_b,
                ln2_w: &ln2_w, ln2_b: &ln2_b,
            };
            execute_jit_bert_layer(
                compiled,
                &hidden_state,
                &bert_weights,
                seq_len,
                &mut layer_out,
                super::jit_helpers::computation_dtype_from_config(config),
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
                use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
                use crate::compat::jit_helpers::{computation_dtype_from_config, kernels_dtype_to_compat};
                let dt = computation_dtype_from_config(config);
                let key = JitCacheKey {
                    arch: ModelArchKey {
                        arch_name: "bert".to_string(),
                        hidden_size: hidden,
                        num_heads: 0,
                        num_kv_heads: 0,
                        head_dim: 0,
                        dtype: kernels_dtype_to_compat(dt),
                    },
                    graph: GraphType::BertMeanPool,
                };
                let compiled = global_jit_cache()
                    .get_or_compile(key, || {
                        let pool_graph = build_mean_pool_graph(seq_len, hidden, dt);
                        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                        compiler.compile_graph(&pool_graph).map_err(|e| format!("MeanPool JIT compilation failed: {e}"))
                    })
                    .map_err(|e| BE::Other(e))?;
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

            // Try two-layer classifier first (dense + out_proj), e.g. bge-reranker
            let dense_aliases = crate::weight_names::classifier_aliases("dense.weight");
            let has_dense = dense_aliases.iter().any(|n| weights.get_tensor(n).is_some() || weights.get_quantized(n).is_some());

            let logit = if has_dense {
                // Two-layer: dense → tanh → out_proj
                let dense_w = get_f32_data(weights, backend, &dense_aliases)?;
                let dense_b = get_bias_data(weights,
                    &crate::weight_names::classifier_aliases("dense.bias"), hidden);
                let out_proj_w = get_f32_data(weights, backend,
                    &crate::weight_names::classifier_aliases("out_proj.weight"))?;
                let out_proj_b = get_bias_data(weights,
                    &crate::weight_names::classifier_aliases("out_proj.bias"), 1);

                // dense: x = tanh(W @ cls + b) — GEMM via JIT, tanh via iterator
                let dense_w_t = if transpose_weights {
                    dense_w.clone()
                } else {
                    super::weight_helpers::transpose_f32(&dense_w, hidden, hidden)
                };
                let x = {
                    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
                    use crate::compat::jit_helpers::{computation_dtype_from_config, kernels_dtype_to_compat};
                    use gllm_kernels::compiler::{CompilerGraph, OpKind};
                    let dt = computation_dtype_from_config(config);
                    let key = JitCacheKey {
                        arch: ModelArchKey {
                            arch_name: "bert_dense_gemm".to_string(),
                            hidden_size: hidden,
                            num_heads: 0, num_kv_heads: 0, head_dim: 0,
                            dtype: kernels_dtype_to_compat(dt),
                        },
                        graph: GraphType::BertLayer { inter_size: hidden },
                    };
                    let compiled = global_jit_cache().get_or_compile(key, || {
                        let mut g = CompilerGraph::new();
                        let ft = dt;
                        let x_in = g.add_tensor_concrete("x", &[1, hidden], ft);
                        let w_in = g.add_tensor_concrete("w", &[hidden, hidden], dt);
                        let b_in = g.add_tensor_concrete("b", &[hidden], ft);
                        g.inputs = vec![x_in, w_in, b_in];
                        let out = g.add_tensor_concrete("out", &[1, hidden], ft);
                        g.add_op(OpKind::GemmBias { m: 1, n: hidden, k: hidden, dtype: dt }, vec![x_in, w_in, b_in], vec![out], "dense_gemm");
                        g.outputs = vec![out];
                        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                        compiler.compile_graph(&g).map_err(|e| format!("bert dense JIT failed: {e}"))
                    }).map_err(|e| BE::Other(e))?;
                    // ARCH-DTYPE-FULLCHAIN-ORCH: weight and bias both at model dtype
                    let w_eb = dt.size_bytes();
                    let w_bytes = dense_w_t.len() * w_eb;
                    let b_bytes = dense_b.len() * w_eb;
                    let mut wbuf = vec![0u8; w_bytes + b_bytes];
                    // Pack weight at model dtype
                    match dt {
                        gllm_kernels::types::DType::F32 => unsafe {
                            std::ptr::copy_nonoverlapping(dense_w_t.as_ptr() as *const u8, wbuf.as_mut_ptr(), w_bytes);
                        },
                        gllm_kernels::types::DType::F16 => {
                            for (i, &val) in dense_w_t.iter().enumerate() {
                                let h = half::f16::from_f32(val);
                                let hb = h.to_le_bytes();
                                wbuf[i * 2] = hb[0];
                                wbuf[i * 2 + 1] = hb[1];
                            }
                        },
                        gllm_kernels::types::DType::BF16 => {
                            for (i, &val) in dense_w_t.iter().enumerate() {
                                let h = half::bf16::from_f32(val);
                                let hb = h.to_le_bytes();
                                wbuf[i * 2] = hb[0];
                                wbuf[i * 2 + 1] = hb[1];
                            }
                        },
                    }
                    // Pack bias at model dtype
                    let bias_converted = super::weight_helpers::f32_to_typed_bytes(&dense_b, dt);
                    wbuf[w_bytes..w_bytes + b_bytes].copy_from_slice(&bias_converted);
                    let mut gemm_out = vec![0.0f32; hidden];
                    let mut scratch = vec![0u8; compiled.scratchpad_bytes];
                    unsafe {
                        compiled.execute(
                            cls.as_ptr() as *const u8, wbuf.as_ptr(),
                            std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                            1, 1, gemm_out.as_mut_ptr() as *mut u8, scratch.as_mut_ptr(),
                        );
                    }
                    // tanh activation (element-wise, compiler auto-vectorizes)
                    gemm_out.iter().map(|v| v.tanh()).collect::<Vec<f32>>()
                };

                // out_proj: logit = dot(out_proj_w, x) + b
                let logit = out_proj_b[0] + out_proj_w.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>();
                logit
            } else {
                // Single-layer: classifier.weight [num_labels, hidden] @ cls + bias
                // e.g. cross-encoder/ms-marco-MiniLM (BertForSequenceClassification)
                let w = get_f32_data(weights, backend,
                    &crate::weight_names::classifier_aliases("weight"))?;
                let b = get_bias_data(weights,
                    &crate::weight_names::classifier_aliases("bias"), 1);

                let logit = b[0] + w.iter().zip(cls.iter()).map(|(a, b)| a * b).sum::<f32>();
                logit
            };

            // sigmoid for probability
            let score = 1.0 / (1.0 + (-logit).exp());
            Ok(vec![score])
        }
    }
}
