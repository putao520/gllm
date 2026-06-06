/// Canonical weight name for a per-layer tensor.
#[inline]
fn cn_layer(layer: usize, suffix: &str) -> String {
    format!("L{}.{suffix}", layer)
}

/// Canonical bias name for a per-layer tensor.
#[inline]
fn cn_layer_bias(layer: usize, suffix: &str) -> String {
    format!("L{}.{suffix}.bias", layer)
}

/// Canonical name for an MoE expert weight.
#[inline]
fn cn_expert(layer: usize, expert: usize, proj: &str) -> String {
    format!("L{}.expert.{}.{}", layer, expert, proj)
}

/// Canonical name for a shared expert weight.
#[inline]
fn cn_shared(layer: usize, proj: &str) -> String {
    format!("L{}.shared_expert.{}", layer, proj)
}

/// Helper: get shape from canonical-keyed weight_shapes, or return error.
fn get_shape(
    weight_shapes: &HashMap<String, Vec<usize>>,
    canonical: &str,
) -> Result<Vec<usize>, GraphBuildError> {
    weight_shapes.get(canonical)
        .cloned()
        .ok_or_else(|| GraphBuildError::MissingTensor(canonical.to_string()))
}

/// Build a CompilerGraph from architecture features + weight shapes.
///
/// All tensor names are canonical (e.g., `embed`, `L0.q_proj`).
/// `weight_shapes` must be keyed by canonical names (executor converts).
pub fn build_compiler_graph(
    features: &ArchitectureFeatures,
    config: &ResolvedConfig,
    weight_shapes: &HashMap<String, Vec<usize>>,
    weight_dtypes: &HashMap<String, DType>,
    weight_quant_types: &HashMap<String, gllm_kernels::quant::QuantType>,
    business_config: &MegaKernelBusinessConfig,
    max_seq_len: usize,
) -> Result<CompilerGraph, GraphBuildError> {
    let mut g = CompilerGraph::new();

    let s = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(max_seq_len),
    };
    // Activation dtype (always F32 — compute happens in F32).
    let dt = DType::F32;
    // Weight dtype: currently weight_blob stores F32 for all formats (pack_weights_from_graph
    // converts BF16→F32), so tensor dtype is always F32. Once raw-pack is implemented,
    // switch to: weight_dtypes.get(name).copied().unwrap_or(dt)
    let _ = weight_dtypes;
    let tdt = |_: &str| -> DType { dt };

    // Helper: generate Gemm or QuantGemm based on weight quantization type.
    let add_gemm_or_quant = |g: &mut CompilerGraph,
                              weight_name: &str,
                              m: SymDim, n: usize, k: usize,
                              inputs: Vec<TensorId>,
                              outputs: Vec<TensorId>,
                              label: &str| {
        if let Some(&qt) = weight_quant_types.get(weight_name) {
            g.add_op(OpKind::QuantGemm { m, n, k, quant_type: qt }, inputs, outputs, label);
        } else {
            g.add_op(OpKind::Gemm { m, n, k, dtype: dt, trans_b: true },
                inputs, outputs, label);
        }
    };

    let is_encoder = features.family == Family::Encoder;
    let eps = config.norm_eps;

    // ── Derive dimensions from weight shapes (canonical names) ──
    let embed_shape = get_shape(weight_shapes, "embed")?;
    let vocab_size = embed_shape[0];
    let hidden = embed_shape[1];

    let head_dim = config.head_dim;
    let (q_dim, k_dim) = if features.is_mla {
        // MLA: no standard k_proj/v_proj. Q dimension from q_b_proj (query expand).
        let mla_d_c = features.mla_latent_dim;
        let qd = if let Some(qb_shape) = weight_shapes.get(&cn_layer(0, "q_b_proj")) {
            qb_shape[0]
        } else {
            config.num_attention_heads * head_dim
        };
        (qd, mla_d_c)
    } else if let Some(qkv_shape) = weight_shapes.get(&cn_layer(0, "qkv_proj")) {
        // Fused QKV: derive from model geometry, not fused_n/3 (wrong for GQA)
        let qd = config.num_attention_heads * head_dim;
        let kd = config.num_key_value_heads * head_dim;
        let expected_fused = qd + 2 * kd;
        if qkv_shape[0] != expected_fused {
            return Err(GraphBuildError::InvalidDimension(
                format!("fused QKV output dim {} != expected {} (q={}+k={}+v={})",
                    qkv_shape[0], expected_fused, qd, kd, kd)
            ));
        }
        (qd, kd)
    } else {
        let q_shape = weight_shapes.get(&cn_layer(0, "q_proj"))
            .ok_or_else(|| GraphBuildError::MissingTensor(format!("{} or {}", cn_layer(0, "q_proj"), cn_layer(0, "qkv_proj"))))?;
        let k_shape = weight_shapes.get(&cn_layer(0, "k_proj"))
            .ok_or_else(|| GraphBuildError::MissingTensor(format!("{} or {}", cn_layer(0, "k_proj"), cn_layer(0, "qkv_proj"))))?;
        (q_shape[0], k_shape[0])
    };

    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;

    let intermediate_size = {
        let from_gate = weight_shapes.get(&cn_layer(0, "gate_proj")).map(|s| s[0]);
        let from_up = weight_shapes.get(&cn_layer(0, "up_proj")).map(|s| s[0]);
        let from_config = config.intermediate_size.unwrap_or(4 * hidden);
        // Fused gate_up_proj: gate_proj dim = 2 * actual intermediate_size
        match (from_gate, from_up) {
            (Some(g), None) if g == 2 * from_config => from_config,
            (Some(g), Some(_)) | (Some(g), None) => g,
            (None, Some(u)) => u,
            (None, None) => from_config,
        }
    };

    let use_rms = features.norm_type == NormType::RmsNorm;

    let mut tensor_map: HashMap<String, gllm_kernels::compiler::graph::TensorId> = HashMap::new();

    // ── SharedKvRef: layer execution guard (SPEC 03 §1.3.1, REQ-UGS-002 item 4) ──
    // Consumer layers skip k_proj/k_norm/v_proj/v_norm/rope_k via GprCondAction.
    // Non-SharedKvRef models (num_kv_shared_layers=0) → Always (zero overhead).
    let kv_guard = if config.num_kv_shared_layers > 0 {
        gllm_kernels::compiler::graph::LayerCondition::LayerIdxLt(
            features.num_layers - config.num_kv_shared_layers,
        )
    } else {
        gllm_kernels::compiler::graph::LayerCondition::Always
    };

    // ── Embedding: token_ids → Gather → hidden_0 (shared by decoder & encoder) ──
    let token_ids = g.add_tensor("token_ids", vec![s.clone()], dt);
    let embed_w = g.add_tensor_concrete("embed", &[vocab_size, hidden], tdt("embed"));
    let embedding = g.add_tensor("embedding", vec![s.clone(), SymDim::Concrete(hidden)], dt);

    // When embed weight is quantized, use QuantGather (JIT dequant per token).
    // Otherwise use standard Gather (zero-copy F32/BF16 row copy).
    let embed_quant_type = weight_quant_types.get("embed").copied();
    if let Some(qt) = embed_quant_type {
        g.add_op(
            OpKind::QuantGather {
                quant_type: qt,
                vocab_size,
                hidden_dim: hidden,
                index_dim: s.clone(),
            },
            vec![token_ids, embed_w],
            vec![embedding],
            "embed_gather",
        );
    } else {
        g.add_op(
            OpKind::Gather {
                table_rows: vocab_size, embed_dim: hidden,
                index_dim: s.clone(), indices_kind: Default::default(),
            },
            vec![token_ids, embed_w],
            vec![embedding],
            "embed_gather",
        );
    }
    tensor_map.insert("hidden_0".to_string(), embedding);

    // ── Encoder embedding: position + token_type + LayerNorm (BERT/XLM-R) ──
    // BERT encoder: hidden = LayerNorm(word_emb + pos_emb + type_emb, ln_w, ln_b)
    // Only encoder models with absolute position embeddings need this block.
    let has_position_embed = weight_shapes.contains_key("position_embed");
    let has_token_type_embed = weight_shapes.contains_key("token_type_embed");
    let has_embed_norm = weight_shapes.contains_key("embed_norm");
    let mut hidden_0 = embedding;
    if is_encoder && (has_position_embed || has_token_type_embed || has_embed_norm) {
        use gllm_kernels::compiler::graph::GatherIndicesKind;

        // Position embedding: Gather(position_emb_w, Arange{0..S})
        if has_position_embed {
            let pos_shape = get_shape(weight_shapes, "position_embed")?;
            let pos_w = g.add_tensor_concrete("position_embed", &pos_shape, tdt("position_embed"));
            let pos_emb = g.add_tensor("pos_emb", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(
                OpKind::Gather {
                    table_rows: pos_shape[0], embed_dim: hidden,
                    index_dim: s.clone(), indices_kind: GatherIndicesKind::Arange,
                },
                vec![token_ids, pos_w], vec![pos_emb], "position_embed",
            );
            let combined = g.add_tensor("embed_plus_pos", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![hidden_0, pos_emb], vec![combined], "embed_add_pos");
            hidden_0 = combined;
        }

        // Token type embedding: Gather(type_emb_w, Zeros) → broadcast row 0
        if has_token_type_embed {
            let tte_shape = get_shape(weight_shapes, "token_type_embed")?;
            let tte_w = g.add_tensor_concrete("token_type_embed", &tte_shape, tdt("token_type_embed"));
            let tte_emb = g.add_tensor("type_emb", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(
                OpKind::Gather {
                    table_rows: tte_shape[0], embed_dim: hidden,
                    index_dim: s.clone(), indices_kind: GatherIndicesKind::Zeros,
                },
                vec![token_ids, tte_w], vec![tte_emb], "token_type_embed",
            );
            let combined = g.add_tensor("embed_plus_type", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![hidden_0, tte_emb], vec![combined], "embed_add_type");
            hidden_0 = combined;
        }

        // Embedding LayerNorm: LayerNorm(combined, embed_ln_w, embed_ln_b)
        if has_embed_norm {
            let eln_shape = get_shape(weight_shapes, "embed_norm")?;
            let eln_w = g.add_tensor_concrete("embed_norm", &eln_shape, tdt("embed_norm"));
            let eln_bias_cn = "embed_norm.bias";
            // Embedding LayerNorm uses the same model-specific eps
            let embed_ln_eps = config.norm_eps;
            let has_eln_bias = weight_shapes.contains_key(eln_bias_cn);
            if has_eln_bias {
                let eln_bias_shape = weight_shapes.get(eln_bias_cn).cloned().unwrap_or_else(|| eln_shape.clone());
                let eln_b = g.add_tensor_concrete(eln_bias_cn, &eln_bias_shape, tdt(eln_bias_cn));
                let normed = g.add_tensor("embed_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::LayerNorm { eps: embed_ln_eps },
                    vec![hidden_0, eln_w, eln_b], vec![normed], "embed_norm",
                );
                hidden_0 = normed;
            } else {
                let normed = g.add_tensor("embed_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::RmsNorm { eps: embed_ln_eps },
                    vec![hidden_0, eln_w], vec![normed], "embed_norm",
                );
                hidden_0 = normed;
            }
        }

        tensor_map.insert("hidden_0".to_string(), hidden_0);
    }

    // ── AltUp (Alternating Updates) pre-layer setup (Gemma 4 E2B/E4B) ──
    // See SPEC/DOCS/architecture/gemma4-altup.md for full data flow.
    // AltUp extends the residual stream from [S,H] to [S,P*H] (P=altup_num_inputs).
    // Pre-layer init and PLE precompute are handled by the executor at runtime.
    // Here we register global AltUp weights and prepare fat activation buffer variables.
    let has_altup = features.has_per_layer_embedding && features.altup_num_inputs > 0;
    let altup_p = features.altup_num_inputs;

    // Global AltUp weights (model-level, NOT strided per layer).
    // These are optional — only present when AltUp weights exist in weight_shapes.
    let mut altup_router_norm_w: Option<TensorId> = None;
    let mut altup_modality_router_w: Option<TensorId> = None;
    let mut altup_prediction_coefs_w: Option<TensorId> = None;
    let mut altup_correction_coefs_w: Option<TensorId> = None;

    if has_altup {
        // Router RMSNorm weight [H]
        if let Ok(shape) = get_shape(weight_shapes, "altup_router_norm") {
            let w = g.add_tensor_concrete("altup_router_norm", &shape, tdt("altup_router_norm"));
            altup_router_norm_w = Some(w);
        }
        // Modality router weight [P, H]
        if let Ok(shape) = get_shape(weight_shapes, "altup_modality_router") {
            let w = g.add_tensor_concrete("altup_modality_router", &shape, tdt("altup_modality_router"));
            altup_modality_router_w = Some(w);
        }
        // Prediction coefficients [P², P]
        if let Ok(shape) = get_shape(weight_shapes, "altup_prediction_coefs") {
            let w = g.add_tensor_concrete("altup_prediction_coefs", &shape, tdt("altup_prediction_coefs"));
            altup_prediction_coefs_w = Some(w);
        }
        // Correction coefficients [P, P]
        if let Ok(shape) = get_shape(weight_shapes, "altup_correction_coefs") {
            let w = g.add_tensor_concrete("altup_correction_coefs", &shape, tdt("altup_correction_coefs"));
            altup_correction_coefs_w = Some(w);
        }
    }

    // Capture layer config for layer_loop_config setup after g.inputs is finalized
    let mut _lc_weight_tids: Vec<TensorId> = Vec::new();
    let mut _lc_weight_stride: usize = 0;
    let mut _lc_layer_input: Option<TensorId> = None;
    let mut _lc_layer_output: Option<TensorId> = None;

    // AltUp fat buffer output (set inside layer template closure, read in post-layer)
    // Set inside layer template closure, read in post-layer block below.
    #[allow(unused_assignments)]
    let mut altup_fat_output_tid: Option<TensorId> = None;

    // Helper: compute physical bytes for a weight tensor, accounting for quantization.
    // For quantized formats, bytes = N * (K / block_size) * block_bytes.
    // For unquantized (F32/BF16/F16), bytes = product(shape) * elem_size.
    let weight_physical_bytes = |canonical: &str, shape: &[usize]| -> usize {
        let numel: usize = shape.iter().product();
        if numel == 0 { return 0; }
        if let Some(&qt) = weight_quant_types.get(canonical) {
            // Quantized: [N, K] → N rows × (K/block_size blocks × block_bytes each)
            if shape.len() == 2 {
                let n = shape[0];
                let k = shape[1];
                let bs = qt.block_size();
                let bb = qt.block_bytes();
                if k.is_multiple_of(bs) && bs > 0 { n * (k / bs) * bb } else { numel * 4 }
            } else if shape.len() == 1 {
                // 1D quantized (rare): numel / block_size * block_bytes
                let bs = qt.block_size();
                let bb = qt.block_bytes();
                if numel.is_multiple_of(bs) && bs > 0 { (numel / bs) * bb } else { numel * 4 }
            } else {
                numel * 4
            }
        } else {
            numel * dt.size_bytes()
        }
    };

    // ── Layer template (single copy, JIT loops at runtime via layer_loop_config) ──
    //
    // ARCH-LAYER-LOOP: Instead of emitting N separate op sets (L0_*, L1_*, ..., L27_*),
    // emit ONE template with "layer." prefix. The JIT layer loop runs it N times,
    // stepping weight_ptr by weight_stride each iteration.
    //
    // Benefits: VmInstr count drops from N×K to K, RegAllocator from O(N²) to O(1).
    //
    // For heterogeneous models (e.g. Gemma 4 E2B with 4 layer types), emit 4 templates
    // with prefixes "layer_sliding_small.", "layer_full_small.", "layer_sliding_large.",
    // "layer_full_large." and assemble a HeteroLayerLoopConfig instead.

    // Hetero layer state — populated by either branch below
    let mut used_hetero = false;
    let mut hetero_strides_out: [usize; 4] = [0; 4];
    let mut hetero_layer_outputs_out: [Option<TensorId>; 4] = [None, None, None, None];
    let mut hetero_weight_tids_out: [Vec<TensorId>; 4] = [vec![], vec![], vec![], vec![]];
    let mut hetero_small_seg_stride: usize = 0;
    let mut hetero_large_seg_stride: usize = 0;

    if features.is_hetero_layer {
        // ====================================================================
        // HETEROGENEOUS PATH: 4 layer templates for Gemma 4 E2B/E4B
        // ====================================================================
        //
        // Layer types:
        //   sliding_small: sliding attention + small FFN  (layers 0,1,2,3,5,6,7,8,...)
        //   full_small:    full attention + small FFN     (layers 4,9,14)
        //   sliding_large: sliding attention + large FFN  (layers 15,16,17,18,...)
        //   full_large:    full attention + large FFN     (layers 19,24,29,34)
        //
        // Reference layers (first occurrence of each type, used for weight names):
        //   sliding_small → L0,  full_small → L4
        //   sliding_large → L15, full_large → L19
        let ref_layers = [0usize, 4, 15, 19]; // [ss, fs, sl, fl]
        let prefixes = ["layer_sliding_small", "layer_full_small", "layer_sliding_large", "layer_full_large"];
        let head_dims = [features.sliding_head_dim, features.full_head_dim, features.sliding_head_dim, features.full_head_dim];
        let intermediates = [features.small_intermediate, features.small_intermediate, features.large_intermediate, features.large_intermediate];
        // RoPE: sliding layers use base theta, full layers use global theta
        let rope_thetas = [config.rope_theta, config.global_rope_theta, config.rope_theta, config.global_rope_theta];
        let rope_partials = [config.rope_partial_ratio, config.rope_partial_ratio_global, config.rope_partial_ratio, config.rope_partial_ratio_global];

        let original_hidden = *tensor_map.get("hidden_0")
            .ok_or_else(|| GraphBuildError::InvalidDimension("no hidden_0".into()))?;

        // Per-template results
        let mut hetero_weight_tids: [Vec<TensorId>; 4] = [vec![], vec![], vec![], vec![]];
        let mut hetero_strides: [usize; 4] = [0; 4];
        let mut hetero_layer_outputs: [Option<TensorId>; 4] = [None, None, None, None];

        // AltUp fat buffer tensors (shared across all 4 templates)
        let hetero_fat_input: Option<TensorId> = if has_altup {
            let fat_in = g.add_tensor("altup.fat_in",
                vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
            Some(fat_in)
        } else {
            None
        };

        // Build each of the 4 templates
        for type_idx in 0..4 {
            let prefix = prefixes[type_idx];
            let ref_layer = ref_layers[type_idx];
            let this_head_dim = head_dims[type_idx];
            let this_intermediate = intermediates[type_idx];
            let this_rope_theta = rope_thetas[type_idx];
            let this_rope_partial = rope_partials[type_idx];

            // Derive attention dimensions from head_dim
            let this_q_dim = num_heads * this_head_dim;

            let mut wtids: Vec<(TensorId, usize)> = Vec::new();
            let mut byte_cursor: usize = 0;

            // Helper: prefixed tensor name
            let ptname = |suffix: &str| -> String { format!("{}.{}", prefix, suffix) };

            // ── AltUp predict (only for first template, others share fat buffer) ──
            let hidden_tid: TensorId;
            let mut predictions_tid: Option<TensorId> = None;
            let mut corr_coefs_tid: Option<TensorId> = None;
            let mut laurel_up_w: Option<TensorId> = None;
            let mut laurel_down_w: Option<TensorId> = None;
            let mut laurel_norm_w: Option<TensorId> = None;

            if has_altup {
                let fat_in = hetero_fat_input.unwrap();

                // Active path extraction (same for all types — operates on hidden, not head_dim)
                let active_in = g.add_tensor(&ptname("altup_active_in"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::ColumnSlice {
                    seq_len: s.clone(), input_inner: altup_p * hidden,
                    start: 0, slice_dim: hidden,
                }, vec![fat_in], vec![active_in], &ptname("altup_active_slice"));

                let router_normed = g.add_tensor(&ptname("router_normed"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                if let Some(rn_w) = altup_router_norm_w {
                    g.add_op(OpKind::RmsNorm { eps },
                        vec![active_in, rn_w], vec![router_normed], &ptname("altup_router_norm"));
                } else {
                    g.add_op(OpKind::RmsNorm { eps },
                        vec![active_in], vec![router_normed], &ptname("altup_router_norm"));
                }

                if let Some(router_w) = altup_modality_router_w {
                    let modalities_raw = g.add_tensor(&ptname("modalities_raw"),
                        vec![s.clone(), SymDim::Concrete(altup_p)], dt);
                    g.add_op(OpKind::Gemm {
                        m: s.clone(), n: altup_p, k: hidden, dtype: dt, trans_b: true,
                    }, vec![router_normed, router_w], vec![modalities_raw], &ptname("altup_router_gemm"));

                    let modalities = g.add_tensor(&ptname("modalities"),
                        vec![s.clone(), SymDim::Concrete(altup_p)], dt);
                    g.add_op(OpKind::Tanh,
                        vec![modalities_raw], vec![modalities], &ptname("altup_tanh"));

                    if let Some(pred_w) = altup_prediction_coefs_w {
                        let pred_coefs_out = g.add_tensor(&ptname("pred_coefs"),
                            vec![s.clone(), SymDim::Concrete(altup_p * altup_p)], dt);
                        g.add_op(OpKind::Gemm {
                            m: s.clone(), n: altup_p * altup_p, k: altup_p, dtype: dt, trans_b: true,
                        }, vec![modalities, pred_w], vec![pred_coefs_out], &ptname("altup_pred_coefs"));

                        let preds = g.add_tensor(&ptname("predictions"),
                            vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
                        g.add_op(OpKind::AltUpPredict {
                            seq_len: s.clone(), num_preds: altup_p, hidden,
                        }, vec![fat_in, pred_coefs_out], vec![preds], &ptname("altup_predict"));
                        predictions_tid = Some(preds);
                    }

                    if let Some(corr_w) = altup_correction_coefs_w {
                        let corr_coefs = g.add_tensor(&ptname("corr_coefs"),
                            vec![s.clone(), SymDim::Concrete(altup_p * altup_p)], dt);
                        g.add_op(OpKind::Gemm {
                            m: s.clone(), n: altup_p * altup_p, k: altup_p, dtype: dt, trans_b: true,
                        }, vec![modalities, corr_w], vec![corr_coefs], &ptname("altup_corr_coefs"));
                        corr_coefs_tid = Some(corr_coefs);
                    }
                }

                let active = g.add_tensor(&ptname("altup_active"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                if let Some(preds) = predictions_tid {
                    g.add_op(OpKind::ColumnSlice {
                        seq_len: s.clone(), input_inner: altup_p * hidden,
                        start: 0, slice_dim: hidden,
                    }, vec![preds], vec![active], &ptname("altup_active_extract"));
                }
                hidden_tid = active;
            } else {
                hidden_tid = original_hidden;
            }

            // ── Input norm ──
            let in_norm_cn = cn_layer(ref_layer, "input_norm");
            let norm_shape = get_shape(weight_shapes, &in_norm_cn)?;
            let norm_w_tid = g.add_tensor_concrete(&in_norm_cn, &norm_shape, tdt(&in_norm_cn));
            wtids.push((norm_w_tid, byte_cursor));
            byte_cursor += weight_physical_bytes(&in_norm_cn, &norm_shape);

            let normed: TensorId;
            if is_encoder {
                normed = hidden_tid;
            } else {
                let n = g.add_tensor(&ptname("normed"), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::RmsNorm { eps }, vec![hidden_tid, norm_w_tid], vec![n], &ptname("input_norm"));
                normed = n;
            }

            // ── AltUp PLE gate weights (per-layer, strided) ──
            if has_altup {
                let _hpl = features.hidden_size_per_layer_input;
                let gate_cn = cn_layer(ref_layer, "per_layer_input_gate");
                if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                    let gate_w = g.add_tensor_concrete(&gate_cn, gate_shape, tdt(&gate_cn));
                    wtids.push((gate_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&gate_cn, gate_shape);
                }
                let ple_proj_cn = cn_layer(ref_layer, "per_layer_proj");
                if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                    let ple_w = g.add_tensor_concrete(&ple_proj_cn, ple_shape, tdt(&ple_proj_cn));
                    wtids.push((ple_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&ple_proj_cn, ple_shape);
                }
                let ple_norm_cn = cn_layer(ref_layer, "post_per_layer_input_norm");
                if let Some(norm_shape) = weight_shapes.get(&ple_norm_cn) {
                    let norm_w = g.add_tensor_concrete(&ple_norm_cn, norm_shape, tdt(&ple_norm_cn));
                    wtids.push((norm_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&ple_norm_cn, norm_shape);
                }
                let laurel_up_cn = cn_layer(ref_layer, "laurel_up");
                if let Some(up_shape) = weight_shapes.get(&laurel_up_cn) {
                    let up_w = g.add_tensor_concrete(&laurel_up_cn, up_shape, tdt(&laurel_up_cn));
                    laurel_up_w = Some(up_w);
                    wtids.push((up_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&laurel_up_cn, up_shape);
                }
                let laurel_down_cn = cn_layer(ref_layer, "laurel_down");
                if let Some(down_shape) = weight_shapes.get(&laurel_down_cn) {
                    let down_w = g.add_tensor_concrete(&laurel_down_cn, down_shape, tdt(&laurel_down_cn));
                    laurel_down_w = Some(down_w);
                    wtids.push((down_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&laurel_down_cn, down_shape);
                }
                let laurel_norm_cn = cn_layer(ref_layer, "laurel_norm");
                if let Some(ln_shape) = weight_shapes.get(&laurel_norm_cn) {
                    let ln_w = g.add_tensor_concrete(&laurel_norm_cn, ln_shape, tdt(&laurel_norm_cn));
                    laurel_norm_w = Some(ln_w);
                    wtids.push((ln_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&laurel_norm_cn, ln_shape);
                }
            }

            // ── Standard Attention Path ── (hetero models always use standard, not MLA)
            // Q projection
            let q_cn = cn_layer(ref_layer, "q_proj");
            let q_shape = weight_shapes.get(&q_cn)
                .ok_or_else(|| GraphBuildError::MissingTensor(q_cn.clone()))?;
            let q_n = q_shape[0];
            let q_k = q_shape[1];
            let q_w = g.add_tensor_concrete(&q_cn, &[q_n, q_k], tdt(&q_cn));
            wtids.push((q_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&q_cn, &[q_n, q_k]);

            let q_out = g.add_tensor(&ptname("q"), vec![s.clone(), SymDim::Concrete(q_n)], dt);
            add_gemm_or_quant(&mut g, &q_cn, s.clone(), q_n, q_k,
                vec![normed, q_w], vec![q_out], &ptname("q_proj"));

            // K projection (guarded by kv_guard for SharedKvRef)
            let k_cn = cn_layer(ref_layer, "k_proj");
            let k_shape = weight_shapes.get(&k_cn)
                .ok_or_else(|| GraphBuildError::MissingTensor(k_cn.clone()))?;
            let k_n = k_shape[0];
            let k_k = k_shape[1];
            let k_w = g.add_tensor_concrete(&k_cn, &[k_n, k_k], tdt(&k_cn));
            wtids.push((k_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&k_cn, &[k_n, k_k]);

            let k_out = g.add_tensor(&ptname("k"), vec![s.clone(), SymDim::Concrete(k_n)], dt);
            if let Some(&qt) = weight_quant_types.get(&k_cn) {
                g.add_op_guarded(OpKind::QuantGemm { m: s.clone(), n: k_n, k: k_k, quant_type: qt },
                    vec![normed, k_w], vec![k_out], &ptname("k_proj"), kv_guard);
            } else {
                g.add_op_guarded(OpKind::Gemm { m: s.clone(), n: k_n, k: k_k, dtype: dt, trans_b: true },
                    vec![normed, k_w], vec![k_out], &ptname("k_proj"), kv_guard);
            }

            // V projection (guarded by kv_guard)
            let v_cn = cn_layer(ref_layer, "v_proj");
            let v_shape = weight_shapes.get(&v_cn)
                .ok_or_else(|| GraphBuildError::MissingTensor(v_cn.clone()))?;
            let v_n = v_shape[0];
            let v_k = v_shape[1];
            let v_w = g.add_tensor_concrete(&v_cn, &[v_n, v_k], tdt(&v_cn));
            wtids.push((v_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&v_cn, &[v_n, v_k]);

            let mut v_out = g.add_tensor(&ptname("v"), vec![s.clone(), SymDim::Concrete(v_n)], dt);
            if let Some(&qt) = weight_quant_types.get(&v_cn) {
                g.add_op_guarded(OpKind::QuantGemm { m: s.clone(), n: v_n, k: v_k, quant_type: qt },
                    vec![normed, v_w], vec![v_out], &ptname("v_proj"), kv_guard);
            } else {
                g.add_op_guarded(OpKind::Gemm { m: s.clone(), n: v_n, k: v_k, dtype: dt, trans_b: true },
                    vec![normed, v_w], vec![v_out], &ptname("v_proj"), kv_guard);
            }

            // ValueNorm (optional)
            if features.has_value_norm {
                let v_normed = g.add_tensor(&ptname("v_normed"), vec![s.clone(), SymDim::Concrete(v_n)], dt);
                g.add_op_guarded(OpKind::ValueNorm { eps },
                    vec![v_out], vec![v_normed], &ptname("v_norm"), kv_guard);
                v_out = v_normed;
            }

            let mut q_for_attn = q_out;
            let mut k_for_attn = k_out;

            // QkNorm (no learned weight, operates per-head)
            if features.has_qk_norm && !features.has_head_rms_norm {
                let qk_norm_eps = 1e-6f32;
                let q_normed = g.add_tensor(&ptname("qk_normed_q"), vec![s.clone(), SymDim::Concrete(q_n)], dt);
                g.add_op(OpKind::QkNorm { head_dim: this_head_dim, eps: qk_norm_eps },
                    vec![q_for_attn], vec![q_normed], &ptname("qk_norm_q"));
                q_for_attn = q_normed;

                let k_normed = g.add_tensor(&ptname("qk_normed_k"), vec![s.clone(), SymDim::Concrete(k_n)], dt);
                g.add_op_guarded(OpKind::QkNorm { head_dim: this_head_dim, eps: qk_norm_eps },
                    vec![k_for_attn], vec![k_normed], &ptname("qk_norm_k"), kv_guard);
                k_for_attn = k_normed;
            }

            // RoPE (using template-specific theta/partial — no DualRoPE needed)
            if features.has_rope {
                let rope_q = g.add_tensor(&ptname("q_rope"), vec![s.clone(), SymDim::Concrete(q_n)], dt);
                g.add_op(
                    OpKind::RoPE {
                        num_heads, head_dim: this_head_dim,
                        theta: this_rope_theta,
                        partial: this_rope_partial,
                        rope_scaling: config.rope_scaling,
                    },
                    vec![q_for_attn], vec![rope_q], &ptname("rope_q"));
                q_for_attn = rope_q;

                let rope_k = g.add_tensor(&ptname("k_rope"), vec![s.clone(), SymDim::Concrete(k_n)], dt);
                g.add_op_guarded(
                    OpKind::RoPE {
                        num_heads: num_kv_heads, head_dim: this_head_dim,
                        theta: this_rope_theta,
                        partial: this_rope_partial,
                        rope_scaling: config.rope_scaling,
                    },
                    vec![k_for_attn], vec![rope_k], &ptname("rope_k"), kv_guard);
                k_for_attn = rope_k;
            }

            // Attention
            let causal = !is_encoder;
            let attn = g.add_tensor(&ptname("attn"), vec![s.clone(), SymDim::Concrete(q_n)], dt);
            g.add_op(
                OpKind::MultiHeadAttention {
                    seq_len: s.clone(), num_heads, num_kv_heads, head_dim: this_head_dim, causal,
                    attention_sinks: features.attention_sinks,
                },
                vec![q_for_attn, k_for_attn, v_out],
                vec![attn],
                &ptname("mha"),
            );

            // O projection
            let o_cn = cn_layer(ref_layer, "o_proj");
            let (o_n, o_k_dim) = weight_shapes.get(&o_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((hidden, this_q_dim));
            let o_w = g.add_tensor_concrete(&o_cn, &[o_n, o_k_dim], tdt(&o_cn));
            wtids.push((o_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&o_cn, &[o_n, o_k_dim]);

            let o_out = g.add_tensor(&ptname("o"), vec![s.clone(), SymDim::Concrete(o_n)], dt);
            add_gemm_or_quant(&mut g, &o_cn, s.clone(), o_n, o_k_dim,
                vec![attn, o_w], vec![o_out], &ptname("o_proj"));

            // LAuReL
            let mut laurel_out_tid: Option<TensorId> = None;
            if let (Some(up_w), Some(down_w), Some(ln_w)) = (laurel_up_w, laurel_down_w, laurel_norm_w) {
                let laurel_rank = weight_shapes.get(&cn_layer(ref_layer, "laurel_up"))
                    .map(|s| s[0]).unwrap_or(64);
                let laurel_hidden = g.add_tensor(&ptname("laurel_hidden"),
                    vec![s.clone(), SymDim::Concrete(laurel_rank)], dt);
                add_gemm_or_quant(&mut g, &cn_layer(ref_layer, "laurel_up"),
                    s.clone(), laurel_rank, hidden,
                    vec![normed, up_w], vec![laurel_hidden], &ptname("laurel_up_proj"));

                let laurel_activated = g.add_tensor(&ptname("laurel_activated"),
                    vec![s.clone(), SymDim::Concrete(laurel_rank)], dt);
                g.add_op(OpKind::Gelu, vec![laurel_hidden], vec![laurel_activated], &ptname("laurel_gelu"));

                let laurel_proj = g.add_tensor(&ptname("laurel_proj"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                add_gemm_or_quant(&mut g, &cn_layer(ref_layer, "laurel_down"),
                    s.clone(), hidden, laurel_rank,
                    vec![laurel_activated, down_w], vec![laurel_proj], &ptname("laurel_down_proj"));

                let laurel_normed = g.add_tensor(&ptname("laurel_normed"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::RmsNorm { eps },
                    vec![laurel_proj, ln_w], vec![laurel_normed], &ptname("laurel_norm"));
                laurel_out_tid = Some(laurel_normed);
            }

            // Post-attention sandwich norm (Gemma 4: norm after attention, before residual)
            let attn_sandwich_cn = cn_layer(ref_layer, "post_attention_norm");
            let attn_sandwich_o_out = if let Some(sandwich_shape) = weight_shapes.get(&attn_sandwich_cn) {
                let sandwich_w = g.add_tensor_concrete(&attn_sandwich_cn, sandwich_shape, tdt(&attn_sandwich_cn));
                wtids.push((sandwich_w, byte_cursor));
                byte_cursor += weight_physical_bytes(&attn_sandwich_cn, sandwich_shape);
                let normed = g.add_tensor(&ptname("post_attn_sandwich"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::RmsNorm { eps }, vec![o_out, sandwich_w], vec![normed], &ptname("post_attn_sandwich_norm"));
                normed
            } else {
                o_out
            };

            // Residual
            let resid = g.add_tensor(&ptname("attn_resid"), vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(laurel_tid) = laurel_out_tid {
                let three_sum = g.add_tensor(&ptname("three_sum"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![hidden_tid, attn_sandwich_o_out], vec![three_sum], &ptname("attn_laurel_add12"));
                let three_full = g.add_tensor(&ptname("three_full"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![three_sum, laurel_tid], vec![three_full], &ptname("attn_laurel_add3"));
                g.add_op(OpKind::ScaleConst { value: (1.0 / 2.0_f32.sqrt()) },
                    vec![three_full], vec![resid], &ptname("attn_laurel_scale"));
            } else {
                g.add_op(OpKind::Add, vec![hidden_tid, attn_sandwich_o_out], vec![resid], &ptname("attn_resid"));
            }

            // Post-attention norm
            let post_cn = cn_layer(ref_layer, "post_attn_norm");
            let post_norm_shape = get_shape(weight_shapes, &post_cn)?;
            let post_norm_w = g.add_tensor_concrete(&post_cn, &post_norm_shape, tdt(&post_cn));
            wtids.push((post_norm_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&post_cn, &post_norm_shape);

            let post_normed = g.add_tensor(&ptname("post_normed"), vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::RmsNorm { eps }, vec![resid, post_norm_w], vec![post_normed], &ptname("post_norm"));

            // FFN (SwiGLU — all Gemma 4 variants use SwiGLU)
            let ffn_resid_src = resid;
            let gate_cn = cn_layer(ref_layer, "gate_proj");
            let gate_n = weight_shapes.get(&gate_cn).map(|s| s[0]).unwrap_or(this_intermediate);
            let gate_w = g.add_tensor_concrete(&gate_cn, &[gate_n, hidden], tdt(&gate_cn));
            wtids.push((gate_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&gate_cn, &[gate_n, hidden]);

            let gate_o = g.add_tensor(&ptname("gate"), vec![s.clone(), SymDim::Concrete(gate_n)], dt);
            add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_n, hidden,
                vec![post_normed, gate_w], vec![gate_o], &ptname("gate_proj"));

            let up_cn = cn_layer(ref_layer, "up_proj");
            let up_n = weight_shapes.get(&up_cn).map(|s| s[0]).unwrap_or(this_intermediate);
            let up_w = g.add_tensor_concrete(&up_cn, &[up_n, hidden], tdt(&up_cn));
            wtids.push((up_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&up_cn, &[up_n, hidden]);

            let up_o = g.add_tensor(&ptname("up"), vec![s.clone(), SymDim::Concrete(up_n)], dt);
            add_gemm_or_quant(&mut g, &up_cn, s.clone(), up_n, hidden,
                vec![post_normed, up_w], vec![up_o], &ptname("up_proj"));

            let ffn_act_out = g.add_tensor(&ptname("ffn_act"), vec![s.clone(), SymDim::Concrete(this_intermediate)], dt);
            if matches!(features.ffn_type, FfnType::GeGLU) {
                // GeGLU: GELU(gate) * up — fused BinaryElementwise trace
                g.add_op(OpKind::GeGlu, vec![gate_o, up_o], vec![ffn_act_out], &ptname("geglu"));
            } else {
                // SwiGLU: SiLU(gate) * up
                g.add_op(OpKind::SwiGlu, vec![gate_o, up_o], vec![ffn_act_out], &ptname("swiglu"));
            }

            let down_cn = cn_layer(ref_layer, "down_proj");
            let down_k = weight_shapes.get(&down_cn).map(|s| s[1]).unwrap_or(this_intermediate);
            let down_w = g.add_tensor_concrete(&down_cn, &[hidden, down_k], tdt(&down_cn));
            wtids.push((down_w, byte_cursor));
            byte_cursor += weight_physical_bytes(&down_cn, &[hidden, down_k]);

            let down_out = g.add_tensor(&ptname("down"), vec![s.clone(), SymDim::Concrete(hidden)], dt);
            add_gemm_or_quant(&mut g, &down_cn, s.clone(), hidden, down_k,
                vec![ffn_act_out, down_w], vec![down_out], &ptname("down_proj"));

            // Post-FFN sandwich norm (Gemma 4: norm after FFN, before residual)
            let ffn_sandwich_cn = cn_layer(ref_layer, "post_ffw_norm");
            let ffn_sandwich_out = if let Some(sandwich_shape) = weight_shapes.get(&ffn_sandwich_cn) {
                let sandwich_w = g.add_tensor_concrete(&ffn_sandwich_cn, sandwich_shape, tdt(&ffn_sandwich_cn));
                wtids.push((sandwich_w, byte_cursor));
                byte_cursor += weight_physical_bytes(&ffn_sandwich_cn, sandwich_shape);
                let normed = g.add_tensor(&ptname("post_ffn_sandwich"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::RmsNorm { eps }, vec![down_out, sandwich_w], vec![normed], &ptname("post_ffn_sandwich_norm"));
                normed
            } else {
                down_out
            };

            let ffn_resid = g.add_tensor(&ptname("ffn_resid"), vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![ffn_resid_src, ffn_sandwich_out], vec![ffn_resid], &ptname("ffn_resid"));

            // AltUp correct + PLE gate + inject
            let layer_output: TensorId;
            let fat_output_for_post: Option<TensorId>;

            if has_altup {
                let corrected = g.add_tensor(&ptname("corrected"),
                    vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
                if let (Some(preds), Some(coefs)) = (predictions_tid, corr_coefs_tid) {
                    g.add_op(OpKind::AltUpCorrect {
                        seq_len: s.clone(), num_preds: altup_p, hidden,
                    }, vec![preds, coefs, ffn_resid], vec![corrected], &ptname("altup_correct"));
                } else {
                    return Err(GraphBuildError::InvalidDimension(
                        "AltUp enabled but missing prediction/correction coefficients".into()));
                }

                let hpl = features.hidden_size_per_layer_input;
                let corrected_active = g.add_tensor(&ptname("corrected_active"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::ColumnSlice {
                    seq_len: s.clone(), input_inner: altup_p * hidden,
                    start: 0, slice_dim: hidden,
                }, vec![corrected], vec![corrected_active], &ptname("corrected_active_slice"));

                let gate_cn = cn_layer(ref_layer, "per_layer_input_gate");
                let gate_logits = g.add_tensor(&ptname("ple_gate_logits"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                    add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_shape[0], gate_shape[1],
                        vec![corrected_active], vec![gate_logits], &ptname("ple_gate_gemm"));
                }

                let gate = g.add_tensor(&ptname("ple_gate"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                g.add_op(OpKind::Gelu, vec![gate_logits], vec![gate], &ptname("ple_gelu"));

                let ple_input = g.add_tensor(&ptname("ple_input"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                let gated_ple = g.add_tensor(&ptname("gated_ple"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                g.add_op(OpKind::Mul, vec![gate, ple_input], vec![gated_ple], &ptname("ple_gated_mul"));

                let ple_proj_cn = cn_layer(ref_layer, "per_layer_proj");
                let projected = g.add_tensor(&ptname("ple_projected"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                    add_gemm_or_quant(&mut g, &ple_proj_cn, s.clone(), ple_shape[0], ple_shape[1],
                        vec![gated_ple], vec![projected], &ptname("ple_proj_gemm"));
                }

                let mut normalized = g.add_tensor(&ptname("ple_normalized"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                let ple_norm_cn = cn_layer(ref_layer, "post_per_layer_input_norm");
                if weight_shapes.contains_key(&ple_norm_cn) {
                    let norm_shape = get_shape(weight_shapes, &ple_norm_cn)?;
                    let norm_w = g.add_tensor_concrete(&ple_norm_cn, &norm_shape, tdt(&ple_norm_cn));
                    g.add_op(OpKind::RmsNorm { eps },
                        vec![projected, norm_w], vec![normalized], &ptname("ple_norm"));
                } else {
                    normalized = projected;
                }

                let fat_out = g.add_tensor(&ptname("fat_out"),
                    vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
                g.add_op(OpKind::AltUpInject {
                    seq_len: s.clone(), num_preds: altup_p, hidden,
                }, vec![corrected, normalized], vec![fat_out], &ptname("altup_inject"));

                layer_output = ffn_resid;
                fat_output_for_post = Some(fat_out);
            } else if false && features.has_per_layer_embedding {
                // PLE-only path (no AltUp): Gemma 4 E2B/E4B GGUF has PLE weights but no AltUp weights.
                // Data flow: cur → inp_gate GEMM → GELU → mul(ple_input) → proj GEMM → RMSNorm → add residual
                let hpl = features.hidden_size_per_layer_input;

                // GEMM(ffn_resid, inp_gate) → gate_logits [S, hpl]
                let gate_cn = cn_layer(ref_layer, "per_layer_input_gate");
                let gate_logits = g.add_tensor(&ptname("ple_gate_logits"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                    let gate_w = g.add_tensor_concrete(&gate_cn, gate_shape, tdt(&gate_cn));
                    wtids.push((gate_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&gate_cn, gate_shape);
                    add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_shape[0], gate_shape[1],
                        vec![ffn_resid], vec![gate_logits], &ptname("ple_gate_gemm"));
                }

                // GELU(gate_logits) → gate [S, hpl]
                let gate = g.add_tensor(&ptname("ple_gate"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                g.add_op(OpKind::Gelu, vec![gate_logits], vec![gate], &ptname("ple_gelu"));

                // gate × ple_input → gated_ple [S, hpl]
                let ple_input = g.add_tensor(&ptname("ple_input"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                let gated_ple = g.add_tensor(&ptname("gated_ple"),
                    vec![s.clone(), SymDim::Concrete(hpl)], dt);
                g.add_op(OpKind::Mul, vec![gate, ple_input], vec![gated_ple], &ptname("ple_gated_mul"));

                // GEMM(gated_ple, per_layer_proj) → projected [S, H]
                let ple_proj_cn = cn_layer(ref_layer, "per_layer_proj");
                let projected = g.add_tensor(&ptname("ple_projected"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                    let ple_w = g.add_tensor_concrete(&ple_proj_cn, ple_shape, tdt(&ple_proj_cn));
                    wtids.push((ple_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&ple_proj_cn, ple_shape);
                    add_gemm_or_quant(&mut g, &ple_proj_cn, s.clone(), ple_shape[0], ple_shape[1],
                        vec![gated_ple], vec![projected], &ptname("ple_proj_gemm"));
                }

                // RMSNorm(projected, post_per_layer_input_norm) → normalized [S, H]
                let mut normalized = g.add_tensor(&ptname("ple_normalized"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                let ple_norm_cn = cn_layer(ref_layer, "post_per_layer_input_norm");
                if weight_shapes.contains_key(&ple_norm_cn) {
                    let norm_shape = get_shape(weight_shapes, &ple_norm_cn)?;
                    let norm_w = g.add_tensor_concrete(&ple_norm_cn, &norm_shape, tdt(&ple_norm_cn));
                    wtids.push((norm_w, byte_cursor));
                    byte_cursor += weight_physical_bytes(&ple_norm_cn, &norm_shape);
                    g.add_op(OpKind::RmsNorm { eps },
                        vec![projected, norm_w], vec![normalized], &ptname("ple_norm"));
                } else {
                    normalized = projected;
                }

                // residual: ffn_resid + normalized
                let ple_resid = g.add_tensor(&ptname("ple_resid"),
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![ffn_resid, normalized], vec![ple_resid], &ptname("ple_resid"));

                layer_output = ple_resid;
                fat_output_for_post = None;
            } else {
                layer_output = ffn_resid;
                fat_output_for_post = None;
            }

            // layer_output_scale: per-layer scalar multiply (Gemma 4 E2B/E4B)
            // cur = cur * out_scale  (shape=[1], per-layer scalar)
            // NOTE: disabled for now — OpKind::Mul doesn't support [S,H] × [1] broadcasting.
            // Will be enabled once a ScalarMul OpKind is added.
            // let out_scale_cn = cn_layer(ref_layer, "layer_output_scale");
            // if let Some(scale_shape) = weight_shapes.get(&out_scale_cn) {
            //     ... TODO ...
            // }

            // Store results for this template type
            hetero_weight_tids[type_idx] = wtids.iter().map(|(t, _)| *t).collect();
            hetero_strides[type_idx] = byte_cursor;
            hetero_layer_outputs[type_idx] = Some(layer_output);

            // For the last template (full_large), set up layer_output/fat_output for post-layer
            if type_idx == 3 {
                _lc_weight_tids = wtids.iter().map(|(t, _)| *t).collect();
                _lc_weight_stride = byte_cursor;
                _lc_layer_input = Some(if has_altup { hetero_fat_input.unwrap() } else { original_hidden });
                _lc_layer_output = Some(layer_output);
                altup_fat_output_tid = fat_output_for_post;
                tensor_map.insert("hidden_0".to_string(), layer_output);
            }
        }

        // Store hetero results for HeteroLayerLoopConfig assembly below
        hetero_strides_out = hetero_strides;
        hetero_layer_outputs_out = hetero_layer_outputs;
        hetero_weight_tids_out = hetero_weight_tids;
        hetero_small_seg_stride = features.sliding_per_segment * hetero_strides[0] + hetero_strides[1];
        hetero_large_seg_stride = features.sliding_per_segment * hetero_strides[2] + hetero_strides[3];
        used_hetero = true;

    } else {
    {
        let original_hidden = *tensor_map.get("hidden_0")
            .ok_or_else(|| GraphBuildError::InvalidDimension("no hidden_0".into()))?;

        // ── Track per-layer weight layout for WeightLayout + stride calculation ──
        let mut layer_weight_tids: Vec<(TensorId, usize)> = Vec::new(); // (tid, bytes)
        let mut layer_weight_byte_cursor: usize = 0;

        // ── AltUp predict (before attention path) ──
        // When AltUp is active, extract predictions[0] as the "active" path
        // that feeds into input_norm → attention → FFN.
        // The fat buffer [S, P*H] carries all P parallel predictions.
        let fat_input_tid: Option<TensorId>;
        let mut predictions_tid: Option<TensorId> = None;
        let mut corr_coefs_tid: Option<TensorId> = None;
        let hidden_tid: TensorId; // the active path [S, H]
        let mut laurel_up_w: Option<TensorId> = None;
        let mut laurel_down_w: Option<TensorId> = None;
        let mut laurel_norm_w: Option<TensorId> = None;

        if has_altup {
            // Fat activation buffer: [S, P*H] — carries P parallel prediction paths.
            // Initialized by executor (AltUp init from hidden_0 + projections).
            let fat_in = g.add_tensor("altup.fat_in",
                vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
            fat_input_tid = Some(fat_in);

            // Router computation: RMSNorm(active) → GEMM(router) → Tanh → GEMM(coefs)
            // active = fat_in[0] (ColumnSlice the first H columns)
            let active_in = g.add_tensor("layer.altup_active_in",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::ColumnSlice {
                seq_len: s.clone(), input_inner: altup_p * hidden,
                start: 0, slice_dim: hidden,
            }, vec![fat_in], vec![active_in], "layer.altup_active_slice");

            // RMSNorm(active, router_norm) → router_normed [S, H]
            let router_normed = g.add_tensor("layer.router_normed",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(rn_w) = altup_router_norm_w {
                g.add_op(OpKind::RmsNorm { eps },
                    vec![active_in, rn_w], vec![router_normed], "layer.altup_router_norm");
            } else {
                // Fallback: use standard RmsNorm weight from layer template
                // (router shares input_norm weight in some configs)
                g.add_op(OpKind::RmsNorm { eps },
                    vec![active_in], vec![router_normed], "layer.altup_router_norm");
            }

            // GEMM(router_normed, modality_router) → modalities_raw [S, P]
            if let Some(router_w) = altup_modality_router_w {
                let modalities_raw = g.add_tensor("layer.modalities_raw",
                    vec![s.clone(), SymDim::Concrete(altup_p)], dt);
                g.add_op(OpKind::Gemm {
                    m: s.clone(), n: altup_p, k: hidden, dtype: dt, trans_b: true,
                }, vec![router_normed, router_w], vec![modalities_raw], "layer.altup_router_gemm");

                // Tanh(modalities_raw) → modalities [S, P]
                let modalities = g.add_tensor("layer.modalities",
                    vec![s.clone(), SymDim::Concrete(altup_p)], dt);
                g.add_op(OpKind::Tanh,
                    vec![modalities_raw], vec![modalities], "layer.altup_tanh");

                // GEMM(modalities, prediction_coefs) → prediction_coefs_out [S, P²]
                if let Some(pred_w) = altup_prediction_coefs_w {
                    let pred_coefs_out = g.add_tensor("layer.pred_coefs",
                        vec![s.clone(), SymDim::Concrete(altup_p * altup_p)], dt);
                    g.add_op(OpKind::Gemm {
                        m: s.clone(), n: altup_p * altup_p, k: altup_p, dtype: dt, trans_b: true,
                    }, vec![modalities, pred_w], vec![pred_coefs_out], "layer.altup_pred_coefs");

                    // AltUpPredict: fat_buffer + pred_coefs → predictions [S, P*H]
                    let preds = g.add_tensor("layer.predictions",
                        vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
                    g.add_op(OpKind::AltUpPredict {
                        seq_len: s.clone(), num_preds: altup_p, hidden,
                    }, vec![fat_in, pred_coefs_out], vec![preds], "layer.altup_predict");
                    predictions_tid = Some(preds);
                }

                // GEMM(modalities, correction_coefs) → correction_coefs [S, P]
                // (stored for AltUpCorrect after FFN)
                if let Some(corr_w) = altup_correction_coefs_w {
                    let corr_coefs = g.add_tensor("layer.corr_coefs",
                        vec![s.clone(), SymDim::Concrete(altup_p)], dt);
                    g.add_op(OpKind::Gemm {
                        m: s.clone(), n: altup_p, k: altup_p, dtype: dt, trans_b: true,
                    }, vec![modalities, corr_w], vec![corr_coefs], "layer.altup_corr_coefs");
                    corr_coefs_tid = Some(corr_coefs);
                }
            }

            // Extract predictions[0] as active path → ColumnSlice [S, H]
            if let Some(preds) = predictions_tid {
                let active = g.add_tensor("layer.altup_active",
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::ColumnSlice {
                    seq_len: s.clone(), input_inner: altup_p * hidden,
                    start: 0, slice_dim: hidden,
                }, vec![preds], vec![active], "layer.altup_active_extract");
                hidden_tid = active;
            } else {
                // No prediction coefs — fall back to direct active path from fat buffer
                hidden_tid = active_in;
            }
        } else {
            fat_input_tid = None;
            hidden_tid = original_hidden;
        }

        // ── Input norm ──
        // Pre-norm (decoder): norm BEFORE attention. Post-norm (encoder/BERT): norm AFTER attn residual.
        let in_norm_cn = cn_layer(0, "input_norm");
        let norm_shape = get_shape(weight_shapes, &in_norm_cn)?;
        let norm_w_tid = g.add_tensor_concrete(&in_norm_cn, &norm_shape, tdt(&in_norm_cn));
        layer_weight_tids.push((norm_w_tid, layer_weight_byte_cursor));
        layer_weight_byte_cursor += weight_physical_bytes(&in_norm_cn, &norm_shape);

        let input_norm_bias_tid: Option<TensorId> = if !use_rms {
            let bias_cn = cn_layer_bias(0, "input_norm");
            let bias_tid = g.add_tensor_concrete(&bias_cn, &[hidden], tdt(&bias_cn));
            layer_weight_tids.push((bias_tid, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&bias_cn, &[hidden]);
            Some(bias_tid)
        } else {
            None
        };

        let normed: TensorId;
        if is_encoder {
            // Post-norm: attention uses raw hidden; InputNorm applied after attn residual
            normed = hidden_tid;
        } else {
            // Pre-norm: InputNorm applied before attention
            let n = g.add_tensor("layer.normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(bias) = input_norm_bias_tid {
                g.add_op(OpKind::LayerNorm { eps }, vec![hidden_tid, norm_w_tid, bias], vec![n], "layer.input_norm");
            } else {
                g.add_op(OpKind::RmsNorm { eps }, vec![hidden_tid, norm_w_tid], vec![n], "layer.input_norm");
            }
            normed = n;
        }

        // ── AltUp per-layer PLE gate weights (registered in layer stride) ──
        // These weights are strided per-layer (like q_proj, k_proj, etc.).
        if has_altup {
            let _hpl = features.hidden_size_per_layer_input;

            // per_layer_input_gate: [hpl, H] — GELU gate projection H→hpl
            let gate_cn = cn_layer(0, "per_layer_input_gate");
            if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                let gate_w = g.add_tensor_concrete(&gate_cn, gate_shape, tdt(&gate_cn));
                layer_weight_tids.push((gate_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&gate_cn, gate_shape);
            }

            // per_layer_projection: [H, hpl] — PLE projection hpl→H
            let ple_proj_cn = cn_layer(0, "per_layer_proj");
            if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                let ple_w = g.add_tensor_concrete(&ple_proj_cn, ple_shape, tdt(&ple_proj_cn));
                layer_weight_tids.push((ple_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&ple_proj_cn, ple_shape);
            }

            // post_per_layer_input_norm: [H] — PLE post-norm
            let ple_norm_cn = cn_layer(0, "post_per_layer_input_norm");
            if let Some(norm_shape) = weight_shapes.get(&ple_norm_cn) {
                let norm_w = g.add_tensor_concrete(&ple_norm_cn, norm_shape, tdt(&ple_norm_cn));
                layer_weight_tids.push((norm_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&ple_norm_cn, norm_shape);
            }

            // LAuReL weights (Learned Augmented Residual Layer, rank=64)
            // See SPEC/DOCS/architecture/gemma4-altup.md §6 for data flow.
            let _laurel_rank = 64;
            let laurel_up_cn = cn_layer(0, "laurel_up");
            if let Some(up_shape) = weight_shapes.get(&laurel_up_cn) {
                let up_w = g.add_tensor_concrete(&laurel_up_cn, up_shape, tdt(&laurel_up_cn));
                laurel_up_w = Some(up_w);
                layer_weight_tids.push((up_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&laurel_up_cn, up_shape);
            }
            let laurel_down_cn = cn_layer(0, "laurel_down");
            if let Some(down_shape) = weight_shapes.get(&laurel_down_cn) {
                let down_w = g.add_tensor_concrete(&laurel_down_cn, down_shape, tdt(&laurel_down_cn));
                laurel_down_w = Some(down_w);
                layer_weight_tids.push((down_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&laurel_down_cn, down_shape);
            }
            let laurel_norm_cn = cn_layer(0, "laurel_norm");
            if let Some(ln_shape) = weight_shapes.get(&laurel_norm_cn) {
                let ln_w = g.add_tensor_concrete(&laurel_norm_cn, ln_shape, tdt(&laurel_norm_cn));
                laurel_norm_w = Some(ln_w);
                layer_weight_tids.push((ln_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&laurel_norm_cn, ln_shape);
            }
        }

        // ── Attention subgraph: MLA vs standard ──
        let attn_out = if features.is_mla {
            // ── MLA Absorbed Attention Path ──
            let mla_d_c = features.mla_latent_dim;
            let mla_d_rope = features.mla_rope_dim;

            // 1. Q projection: q_a_proj (compress to q_lora_rank) → q_b_proj (expand to n_h×d)
            let qa_cn = cn_layer(0, "q_a_proj");
            let (qa_n, qa_k) = weight_shapes.get(&qa_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((1536, hidden));
            let qa_w = g.add_tensor_concrete(&qa_cn, &[qa_n, qa_k], tdt(&qa_cn));
            layer_weight_tids.push((qa_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&qa_cn, &[qa_n, qa_k]);

            let c_q = g.add_tensor("layer.c_q", vec![s.clone(), SymDim::Concrete(qa_n)], dt);
            add_gemm_or_quant(&mut g, &qa_cn, s.clone(), qa_n, qa_k,
                vec![normed, qa_w], vec![c_q], "layer.q_a_proj");

            let qb_cn = cn_layer(0, "q_b_proj");
            let (qb_n, qb_k) = weight_shapes.get(&qb_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((q_dim, qa_n));
            let qb_w = g.add_tensor_concrete(&qb_cn, &[qb_n, qb_k], tdt(&qb_cn));
            layer_weight_tids.push((qb_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&qb_cn, &[qb_n, qb_k]);

            let q_raw = g.add_tensor("layer.q", vec![s.clone(), SymDim::Concrete(qb_n)], dt);
            add_gemm_or_quant(&mut g, &qb_cn, s.clone(), qb_n, qb_k,
                vec![c_q, qb_w], vec![q_raw], "layer.q_b_proj");

            // 2. KV compression: kv_b_proj → c_KV [M, d_c]
            let dkv_cn = cn_layer(0, "kv_b_proj");
            let (dkv_n, dkv_k) = weight_shapes.get(&dkv_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((mla_d_c, hidden));
            let dkv_w = g.add_tensor_concrete(&dkv_cn, &[dkv_n, dkv_k], tdt(&dkv_cn));
            layer_weight_tids.push((dkv_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&dkv_cn, &[dkv_n, dkv_k]);

            let c_kv = g.add_tensor("layer.c_kv", vec![s.clone(), SymDim::Concrete(dkv_n)], dt);
            g.add_op(OpKind::MlaKvCompress { m: s.clone(), d_c: dkv_n, hidden: dkv_k },
                vec![normed, dkv_w], vec![c_kv], "layer.kv_compress");

            // 3. RoPE key: k_pe_proj → k_pe [M, d_rope]
            let kr_cn = cn_layer(0, "k_pe_proj");
            let (kr_n, kr_k) = weight_shapes.get(&kr_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((mla_d_rope, hidden));
            let kr_w = g.add_tensor_concrete(&kr_cn, &[kr_n, kr_k], tdt(&kr_cn));
            layer_weight_tids.push((kr_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&kr_cn, &[kr_n, kr_k]);

            let k_pe = g.add_tensor("layer.k_pe", vec![s.clone(), SymDim::Concrete(kr_n)], dt);
            add_gemm_or_quant(&mut g, &kr_cn, s.clone(), kr_n, kr_k,
                vec![normed, kr_w], vec![k_pe], "layer.k_pe_proj");

            // 4. Q absorption: W_UK (k_b_proj) — Q · W_UK^T → Q_absorbed
            let uk_cn = cn_layer(0, "k_b_proj");
            let (uk_n, uk_k) = weight_shapes.get(&uk_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((mla_d_c, qb_n));
            let uk_w = g.add_tensor_concrete(&uk_cn, &[uk_n, uk_k], tdt(&uk_cn));
            layer_weight_tids.push((uk_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&uk_cn, &[uk_n, uk_k]);

            let q_absorbed = g.add_tensor("layer.q_absorbed",
                vec![s.clone(), SymDim::Concrete(uk_n * num_heads)], dt);
            g.add_op(OpKind::MlaQAbsorb { seq_len: s.clone(), num_heads, head_dim, d_c: uk_n },
                vec![q_raw, uk_w], vec![q_absorbed], "layer.q_absorb");

            // 5. V restore weight: W_UV (v_b_proj) — passed to MlaAttention for internal V restore
            let uv_cn = cn_layer(0, "v_b_proj");
            let (uv_n, uv_k) = weight_shapes.get(&uv_cn)
                .map(|s| (s[0], s[1]))
                .unwrap_or((head_dim * num_heads, mla_d_c));
            let uv_w = g.add_tensor_concrete(&uv_cn, &[uv_n, uv_k], tdt(&uv_cn));
            layer_weight_tids.push((uv_w, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&uv_cn, &[uv_n, uv_k]);

            // 6. MLA Attention: Absorbed (default) or Un-absorbed (short prefill)
            let causal = !is_encoder;
            if features.mla_use_unabsorbed {
                // ── MLA Un-absorbed Path (REQ-MLA-004) ──
                // Restore K/V from compressed c_KV, then standard FlashAttention.
                // K = c_KV · W_UK  [M, n_h × d]
                // V = c_KV · W_UV  [M, n_h × d]
                let k_restored = g.add_tensor("layer.k_restored",
                    vec![s.clone(), SymDim::Concrete(num_heads * head_dim)], dt);
                add_gemm_or_quant(&mut g, &uk_cn, s.clone(), num_heads * head_dim, mla_d_c,
                    vec![c_kv, uk_w], vec![k_restored], "layer.k_restore");

                let v_restored = g.add_tensor("layer.v_restored",
                    vec![s.clone(), SymDim::Concrete(num_heads * head_dim)], dt);
                add_gemm_or_quant(&mut g, &uv_cn, s.clone(), num_heads * head_dim, mla_d_c,
                    vec![c_kv, uv_w], vec![v_restored], "layer.v_restore");

                // RoPE on Q and K (un-absorbed uses standard RoPE, not MlaRopeMerge)
                let theta = config.rope_theta;
                let use_dual_mla = config.global_rope_theta > 0.0;
                let rope_q = g.add_tensor("layer.q_rope",
                    vec![s.clone(), SymDim::Concrete(qb_n)], dt);
                if use_dual_mla {
                    g.add_op(
                        OpKind::DualRoPE {
                            num_heads, head_dim,
                            sliding_theta: theta, sliding_partial: config.rope_partial_ratio,
                            global_theta: config.global_rope_theta, global_partial: config.rope_partial_ratio_global,
                            rope_scaling: config.rope_scaling,
                            layer_offset: 1, layer_divisor: 6, layer_remainder: 0,
                        },
                        vec![q_raw], vec![rope_q], "layer.rope_q_mla_unabs");
                } else {
                    g.add_op(
                        OpKind::RoPE { num_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: config.rope_scaling },
                        vec![q_raw], vec![rope_q], "layer.rope_q_mla_unabs");
                }

                let rope_k = g.add_tensor("layer.k_rope",
                    vec![s.clone(), SymDim::Concrete(num_heads * head_dim)], dt);
                if use_dual_mla {
                    g.add_op(
                        OpKind::DualRoPE {
                            num_heads, head_dim,
                            sliding_theta: theta, sliding_partial: config.rope_partial_ratio,
                            global_theta: config.global_rope_theta, global_partial: config.rope_partial_ratio_global,
                            rope_scaling: config.rope_scaling,
                            layer_offset: 1, layer_divisor: 6, layer_remainder: 0,
                        },
                        vec![k_restored], vec![rope_k], "layer.rope_k_mla_unabs");
                } else {
                    g.add_op(
                        OpKind::RoPE { num_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: config.rope_scaling },
                        vec![k_restored], vec![rope_k], "layer.rope_k_mla_unabs");
                }

                // Standard MHA (not MLA attention)
                let attn = g.add_tensor("layer.attn",
                    vec![s.clone(), SymDim::Concrete(num_heads * head_dim)], dt);
                g.add_op(
                    OpKind::MultiHeadAttention {
                        seq_len: s.clone(), num_heads, num_kv_heads: num_heads, head_dim, causal,
                        attention_sinks: false,
                    },
                    vec![rope_q, rope_k, v_restored],
                    vec![attn],
                    "layer.mha_mla_unabs",
                );
                attn
            } else {
                // ── MLA Absorbed Path (default, bandwidth-optimized) ──
                let attn = g.add_tensor("layer.attn",
                    vec![s.clone(), SymDim::Concrete(num_heads * head_dim)], dt);
                g.add_op(
                    OpKind::MlaAttention {
                        seq_len: s.clone(), num_heads, head_dim,
                        d_c: mla_d_c, d_rope: mla_d_rope, causal,
                    },
                    vec![q_absorbed, c_kv, k_pe, uv_w],
                    vec![attn],
                    "layer.mla_attn",
                );
                attn
            }
        } else {
            // ── Standard Attention Path ──
            let fused_qkv_shape = weight_shapes.get(&cn_layer(0, "qkv_proj")).cloned();
            let (q_out, fused_qkv_out, q_n) = if let Some(ref fqkv_shape) = fused_qkv_shape {
                let fused_n = fqkv_shape[0];
                let fused_k = fqkv_shape[1];
                let fqkv_cn = cn_layer(0, "qkv_proj");
                let fqkv_w = g.add_tensor_concrete(&fqkv_cn, &[fused_n, fused_k], tdt(&fqkv_cn));
                layer_weight_tids.push((fqkv_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&fqkv_cn, &[fused_n, fused_k]);

                let fused_out = g.add_tensor("layer.qkv", vec![s.clone(), SymDim::Concrete(fused_n)], dt);
                add_gemm_or_quant(&mut g, &fqkv_cn, s.clone(), fused_n, fused_k,
                    vec![normed, fqkv_w], vec![fused_out], "layer.qkv_proj");
                let q_slice = g.add_tensor("layer.q", vec![s.clone(), SymDim::Concrete(q_dim)], dt);
                g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: 0, slice_dim: q_dim },
                    vec![fused_out], vec![q_slice], "layer.q_slice");
                (q_slice, Some((fused_out, fused_n)), q_dim)
            } else {
                let q_s = weight_shapes.get(&cn_layer(0, "q_proj"));
                let q_n = q_s.map(|s| s[0]).unwrap_or(q_dim);
                let q_k = q_s.map(|s| s[1]).unwrap_or(hidden);
                let q_cn = cn_layer(0, "q_proj");
                let q_w = g.add_tensor_concrete(&q_cn, &[q_n, q_k], tdt(&q_cn));
                layer_weight_tids.push((q_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&q_cn, &[q_n, q_k]);

                let q_out = g.add_tensor("layer.q", vec![s.clone(), SymDim::Concrete(q_n)], dt);
                let q_bias_cn = cn_layer_bias(0, "q_proj");
                if let Some(bias_shape) = weight_shapes.get(&q_bias_cn) {
                    let q_bias = g.add_tensor_concrete(&q_bias_cn, bias_shape, tdt(&q_bias_cn));
                    layer_weight_tids.push((q_bias, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&q_bias_cn, bias_shape);
                    g.add_op(OpKind::GemmBias { m: s.clone(), n: q_n, k: q_k, dtype: dt, trans_b: true },
                        vec![normed, q_w, q_bias], vec![q_out], "layer.q_proj");
                } else {
                    add_gemm_or_quant(&mut g, &q_cn, s.clone(), q_n, q_k,
                        vec![normed, q_w], vec![q_out], "layer.q_proj");
                }
                (q_out, None, q_n)
            };

            // ── K and V projections ──
            let (k_for_attn_final, mut v_out) = if let Some((ref fused_out_tid, fused_n)) = fused_qkv_out {
                let k_slice = g.add_tensor("layer.k", vec![s.clone(), SymDim::Concrete(k_dim)], dt);
                g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: q_dim, slice_dim: k_dim },
                    vec![*fused_out_tid], vec![k_slice], "layer.k_slice");
                let v_slice = g.add_tensor("layer.v", vec![s.clone(), SymDim::Concrete(k_dim)], dt);
                g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: q_dim + k_dim, slice_dim: k_dim },
                    vec![*fused_out_tid], vec![v_slice], "layer.v_slice");
                (k_slice, v_slice)
            } else {
                let k_s = weight_shapes.get(&cn_layer(0, "k_proj"));
                let k_n = k_s.map(|s| s[0]).unwrap_or(k_dim);
                let k_k = k_s.map(|s| s[1]).unwrap_or(hidden);
                let k_cn = cn_layer(0, "k_proj");
                let k_w = g.add_tensor_concrete(&k_cn, &[k_n, k_k], tdt(&k_cn));
                layer_weight_tids.push((k_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&k_cn, &[k_n, k_k]);

                let k_out = g.add_tensor("layer.k", vec![s.clone(), SymDim::Concrete(k_n)], dt);
                let k_bias_cn = cn_layer_bias(0, "k_proj");
                if let Some(bias_shape) = weight_shapes.get(&k_bias_cn) {
                    let k_bias = g.add_tensor_concrete(&k_bias_cn, bias_shape, tdt(&k_bias_cn));
                    layer_weight_tids.push((k_bias, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&k_bias_cn, bias_shape);
                    g.add_op_guarded(OpKind::GemmBias { m: s.clone(), n: k_n, k: k_k, dtype: dt, trans_b: true },
                        vec![normed, k_w, k_bias], vec![k_out], "layer.k_proj", kv_guard);
                } else if let Some(&qt) = weight_quant_types.get(&k_cn) {
                    g.add_op_guarded(OpKind::QuantGemm { m: s.clone(), n: k_n, k: k_k, quant_type: qt }, vec![normed, k_w], vec![k_out], "layer.k_proj", kv_guard);
                } else {
                    g.add_op_guarded(OpKind::Gemm { m: s.clone(), n: k_n, k: k_k, dtype: dt, trans_b: true }, vec![normed, k_w], vec![k_out], "layer.k_proj", kv_guard);
                }

                let v_s = weight_shapes.get(&cn_layer(0, "v_proj"));
                let v_n = v_s.map(|s| s[0]).unwrap_or(k_dim);
                let v_k = v_s.map(|s| s[1]).unwrap_or(hidden);
                let v_cn = cn_layer(0, "v_proj");
                let v_w = g.add_tensor_concrete(&v_cn, &[v_n, v_k], tdt(&v_cn));
                layer_weight_tids.push((v_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&v_cn, &[v_n, v_k]);

                let v_out = g.add_tensor("layer.v", vec![s.clone(), SymDim::Concrete(v_n)], dt);
                let v_bias_cn = cn_layer_bias(0, "v_proj");
                if let Some(bias_shape) = weight_shapes.get(&v_bias_cn) {
                    let v_bias = g.add_tensor_concrete(&v_bias_cn, bias_shape, tdt(&v_bias_cn));
                    layer_weight_tids.push((v_bias, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&v_bias_cn, bias_shape);
                    g.add_op_guarded(OpKind::GemmBias { m: s.clone(), n: v_n, k: v_k, dtype: dt, trans_b: true },
                        vec![normed, v_w, v_bias], vec![v_out], "layer.v_proj", kv_guard);
                } else if let Some(&qt) = weight_quant_types.get(&v_cn) {
                    g.add_op_guarded(OpKind::QuantGemm { m: s.clone(), n: v_n, k: v_k, quant_type: qt }, vec![normed, v_w], vec![v_out], "layer.v_proj", kv_guard);
                } else {
                    g.add_op_guarded(OpKind::Gemm { m: s.clone(), n: v_n, k: v_k, dtype: dt, trans_b: true }, vec![normed, v_w], vec![v_out], "layer.v_proj", kv_guard);
                }

                (k_out, v_out)
            };

            // ── ValueNorm (optional, Gemma 4) ──
            // V 投影后的无学习参数 RMSNorm。Consumer 层跳过（因为不计算 V）。
            if features.has_value_norm {
                let v_n_for_norm = weight_shapes.get(&cn_layer(0, "v_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let v_normed = g.add_tensor("layer.v_normed", vec![s.clone(), SymDim::Concrete(v_n_for_norm)], dt);
                g.add_op_guarded(OpKind::ValueNorm { eps },
                    vec![v_out], vec![v_normed], "layer.v_norm", kv_guard);
                v_out = v_normed;
            }

            // ── HeadRmsNorm (optional) ──
            let mut q_for_attn = q_out;
            let mut k_for_attn = k_for_attn_final;
            if features.has_head_rms_norm {
                let head_rms_eps = 1e-6f32;
                let q_norm_cn = cn_layer(0, "q_norm");
                let q_norm_w = g.add_tensor_concrete(&q_norm_cn, &[head_dim], tdt(&q_norm_cn));
                layer_weight_tids.push((q_norm_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&q_norm_cn, &[head_dim]);

                let q_normed = g.add_tensor("layer.q_normed", vec![s.clone(), SymDim::Concrete(q_n)], dt);
                g.add_op(OpKind::HeadRmsNorm { head_dim, eps: head_rms_eps },
                    vec![q_for_attn, q_norm_w], vec![q_normed], "layer.q_norm");
                q_for_attn = q_normed;

                let k_norm_cn = cn_layer(0, "k_norm");
                let k_norm_w = g.add_tensor_concrete(&k_norm_cn, &[head_dim], tdt(&k_norm_cn));
                layer_weight_tids.push((k_norm_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&k_norm_cn, &[head_dim]);

                let k_n_for_norm = weight_shapes.get(&cn_layer(0, "k_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let k_normed = g.add_tensor("layer.k_normed", vec![s.clone(), SymDim::Concrete(k_n_for_norm)], dt);
                g.add_op_guarded(OpKind::HeadRmsNorm { head_dim, eps: head_rms_eps },
                    vec![k_for_attn, k_norm_w], vec![k_normed], "layer.k_norm", kv_guard);
                k_for_attn = k_normed;
            }

            // ── QkNorm (Gemma 4, optional) ──
            // Q/K L2 normalization + √head_dim rescale, no learned weight.
            // Mutually exclusive with HeadRmsNorm (Qwen3 has weight, Gemma 4 does not).
            if features.has_qk_norm && !features.has_head_rms_norm {
                let qk_norm_eps = 1e-6f32;
                let q_normed = g.add_tensor("layer.qk_normed_q", vec![s.clone(), SymDim::Concrete(q_n)], dt);
                g.add_op(OpKind::QkNorm { head_dim, eps: qk_norm_eps },
                    vec![q_for_attn], vec![q_normed], "layer.qk_norm_q");
                q_for_attn = q_normed;

                let k_n_for_qk_norm = weight_shapes.get(&cn_layer(0, "k_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let k_normed = g.add_tensor("layer.qk_normed_k", vec![s.clone(), SymDim::Concrete(k_n_for_qk_norm)], dt);
                g.add_op_guarded(OpKind::QkNorm { head_dim, eps: qk_norm_eps },
                    vec![k_for_attn], vec![k_normed], "layer.qk_norm_k", kv_guard);
                k_for_attn = k_normed;
            }

            // ── RoPE (optional) ──
            if features.has_rope {
                let theta = config.rope_theta;
                // DualRoPE: when global_rope_theta > 0, use DualRoPE with runtime layer selection.
                // Layer condition derived from attention_pattern: (layer_idx + 1) % 6 == 0 → global.
                let use_dual = config.global_rope_theta > 0.0;
                let rope_q = g.add_tensor("layer.q_rope", vec![s.clone(), SymDim::Concrete(q_n)], dt);
                if use_dual {
                    g.add_op(
                        OpKind::DualRoPE {
                            num_heads, head_dim,
                            sliding_theta: theta,
                            sliding_partial: config.rope_partial_ratio,
                            global_theta: config.global_rope_theta,
                            global_partial: config.rope_partial_ratio_global,
                            rope_scaling: config.rope_scaling,
                            layer_offset: 1, layer_divisor: 6, layer_remainder: 0,
                        },
                        vec![q_for_attn], vec![rope_q], "layer.rope_q");
                } else {
                    g.add_op(
                        OpKind::RoPE { num_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: config.rope_scaling },
                        vec![q_for_attn], vec![rope_q], "layer.rope_q");
                }
                q_for_attn = rope_q;

                let k_n_for_rope = weight_shapes.get(&cn_layer(0, "k_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let rope_k = g.add_tensor("layer.k_rope", vec![s.clone(), SymDim::Concrete(k_n_for_rope)], dt);
                if use_dual {
                    g.add_op_guarded(
                        OpKind::DualRoPE {
                            num_heads: num_kv_heads, head_dim,
                            sliding_theta: theta,
                            sliding_partial: config.rope_partial_ratio,
                            global_theta: config.global_rope_theta,
                            global_partial: config.rope_partial_ratio_global,
                            rope_scaling: config.rope_scaling,
                            layer_offset: 1, layer_divisor: 6, layer_remainder: 0,
                        },
                        vec![k_for_attn], vec![rope_k], "layer.rope_k", kv_guard);
                } else {
                    g.add_op_guarded(
                        OpKind::RoPE { num_heads: num_kv_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: config.rope_scaling },
                        vec![k_for_attn], vec![rope_k], "layer.rope_k", kv_guard);
                }
                k_for_attn = rope_k;
            }

            // ── Attention ──
            let causal = !is_encoder;
            let attn = g.add_tensor("layer.attn", vec![s.clone(), SymDim::Concrete(q_n)], dt);
            g.add_op(
                OpKind::MultiHeadAttention {
                    seq_len: s.clone(), num_heads, num_kv_heads, head_dim, causal,
                    attention_sinks: features.attention_sinks,
                },
                vec![q_for_attn, k_for_attn, v_out],
                vec![attn],
                "layer.mha",
            );
            attn
        };

        // ── O projection ──
        let o_cn = cn_layer(0, "o_proj");
        let (o_n, o_k_dim) = weight_shapes.get(&o_cn)
            .map(|s| (s[0], s[1]))
            .unwrap_or((hidden, hidden));
        let o_w = g.add_tensor_concrete(&o_cn, &[o_n, o_k_dim], tdt(&o_cn));
        layer_weight_tids.push((o_w, layer_weight_byte_cursor));
        layer_weight_byte_cursor += weight_physical_bytes(&o_cn, &[o_n, o_k_dim]);

        let o_out = g.add_tensor("layer.o", vec![s.clone(), SymDim::Concrete(o_n)], dt);
        let o_bias_cn = cn_layer_bias(0, "o_proj");
        if let Some(bias_shape) = weight_shapes.get(&o_bias_cn) {
            let o_bias = g.add_tensor_concrete(&o_bias_cn, bias_shape, tdt(&o_bias_cn));
            layer_weight_tids.push((o_bias, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&o_bias_cn, bias_shape);
            g.add_op(OpKind::GemmBias { m: s.clone(), n: o_n, k: o_k_dim, dtype: dt, trans_b: true },
                vec![attn_out, o_w, o_bias], vec![o_out], "layer.o_proj");
        } else {
            add_gemm_or_quant(&mut g, &o_cn, s.clone(), o_n, o_k_dim,
                vec![attn_out, o_w], vec![o_out], "layer.o_proj");
        }

        // ── LAuReL (Learned Augmented Residual Layer) ──
        // See SPEC/DOCS/architecture/gemma4-altup.md §6.
        // laurel = RMSNorm(GEMM(GELU(GEMM(normed, laurel_up)), laurel_down), laurel_norm)
        let mut laurel_out_tid: Option<TensorId> = None;
        if let (Some(up_w), Some(down_w), Some(ln_w)) = (laurel_up_w, laurel_down_w, laurel_norm_w) {
            let laurel_rank = weight_shapes.get(&cn_layer(0, "laurel_up"))
                .map(|s| s[0]).unwrap_or(64);

            // GEMM(normed, laurel_up) → [S, laurel_rank]
            let laurel_hidden = g.add_tensor("layer.laurel_hidden",
                vec![s.clone(), SymDim::Concrete(laurel_rank)], dt);
            add_gemm_or_quant(&mut g, &cn_layer(0, "laurel_up"),
                s.clone(), laurel_rank, hidden,
                vec![normed, up_w], vec![laurel_hidden], "layer.laurel_up_proj");

            // GELU(laurel_hidden)
            let laurel_activated = g.add_tensor("layer.laurel_activated",
                vec![s.clone(), SymDim::Concrete(laurel_rank)], dt);
            g.add_op(OpKind::Gelu, vec![laurel_hidden], vec![laurel_activated], "layer.laurel_gelu");

            // GEMM(laurel_activated, laurel_down) → [S, H]
            let laurel_proj = g.add_tensor("layer.laurel_proj",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            add_gemm_or_quant(&mut g, &cn_layer(0, "laurel_down"),
                s.clone(), hidden, laurel_rank,
                vec![laurel_activated, down_w], vec![laurel_proj], "layer.laurel_down_proj");

            // RMSNorm(laurel_proj, laurel_norm) → [S, H]
            let laurel_normed = g.add_tensor("layer.laurel_normed",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::RmsNorm { eps },
                vec![laurel_proj, ln_w], vec![laurel_normed], "layer.laurel_norm");
            laurel_out_tid = Some(laurel_normed);
        }

        // ── Residual ──
        // With LAuReL: (active + attn + laurel) / √2
        // Without:     active + attn
        let resid = g.add_tensor("layer.attn_resid", vec![s.clone(), SymDim::Concrete(hidden)], dt);
        if let Some(laurel_tid) = laurel_out_tid {
            // 3-way residual + √2 scaling
            let three_sum = g.add_tensor("layer.three_sum",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![hidden_tid, o_out], vec![three_sum], "layer.attn_laurel_add12");
            let three_full = g.add_tensor("layer.three_full",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![three_sum, laurel_tid], vec![three_full], "layer.attn_laurel_add3");
            // Divide by √2: 1/√2 ≈ 0.70710678
            g.add_op(OpKind::ScaleConst { value: (1.0 / 2.0_f32.sqrt()) },
                vec![three_full], vec![resid], "layer.attn_laurel_scale");
        } else {
            g.add_op(OpKind::Add, vec![hidden_tid, o_out], vec![resid], "layer.attn_resid");
        }

        // ── Post-attention norm / FFN input norm ──
        // Pre-norm (decoder): PostAttnNorm applied to resid → FFN input
        // Post-norm (encoder): InputNorm applied to resid → FFN input; PostAttnNorm deferred after FFN
        let post_cn = cn_layer(0, "post_attn_norm");
        let post_norm_shape = get_shape(weight_shapes, &post_cn)?;
        let post_norm_w = g.add_tensor_concrete(&post_cn, &post_norm_shape, tdt(&post_cn));
        layer_weight_tids.push((post_norm_w, layer_weight_byte_cursor));
        layer_weight_byte_cursor += weight_physical_bytes(&post_cn, &post_norm_shape);

        let post_norm_bias_tid: Option<TensorId> = if !use_rms {
            let bias_cn = cn_layer_bias(0, "post_attn_norm");
            let bias_tid = g.add_tensor_concrete(&bias_cn, &[hidden], tdt(&bias_cn));
            layer_weight_tids.push((bias_tid, layer_weight_byte_cursor));
            layer_weight_byte_cursor += weight_physical_bytes(&bias_cn, &[hidden]);
            Some(bias_tid)
        } else {
            None
        };

        let post_normed: TensorId;
        if is_encoder {
            // Post-norm: apply InputNorm (attention.output.LayerNorm) after attn residual → FFN input
            let n = g.add_tensor("layer.normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(bias) = input_norm_bias_tid {
                g.add_op(OpKind::LayerNorm { eps }, vec![resid, norm_w_tid, bias], vec![n], "layer.input_norm");
            } else {
                g.add_op(OpKind::RmsNorm { eps }, vec![resid, norm_w_tid], vec![n], "layer.input_norm");
            }
            post_normed = n;
        } else {
            // Pre-norm: apply PostAttnNorm to resid → FFN input
            let pn = g.add_tensor("layer.post_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(bias) = post_norm_bias_tid {
                g.add_op(OpKind::LayerNorm { eps }, vec![resid, post_norm_w, bias], vec![pn], "layer.post_norm");
            } else {
                g.add_op(OpKind::RmsNorm { eps }, vec![resid, post_norm_w], vec![pn], "layer.post_norm");
            }
            post_normed = pn;
        }

        // ── FFN ──
        // Pre-norm residual: resid + ffn_out. Post-norm residual: post_normed + ffn_out.
        let ffn_resid_src = if is_encoder { post_normed } else { resid };
        let ffn_resid_tid: Option<TensorId>;
        match &features.ffn_type {
            FfnType::SwiGLU | FfnType::GeGLU => {
                let gate_cn = cn_layer(0, "gate_proj");
                let up_cn = cn_layer(0, "up_proj");
                let has_fused_gate_up = weight_shapes.get(&up_cn).is_none()
                    && weight_shapes.get(&gate_cn).map(|s| s[0]) == Some(2 * intermediate_size);

                let (gate_out, up_out) = if has_fused_gate_up {
                    let fused_n = 2 * intermediate_size;
                    let gate_w = g.add_tensor_concrete(&gate_cn, &[fused_n, hidden], tdt(&gate_cn));
                    layer_weight_tids.push((gate_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&gate_cn, &[fused_n, hidden]);

                    let fused_out = g.add_tensor("layer.gate_up", vec![s.clone(), SymDim::Concrete(fused_n)], dt);
                    add_gemm_or_quant(&mut g, &gate_cn, s.clone(), fused_n, hidden,
                        vec![post_normed, gate_w], vec![fused_out], "layer.gate_up_proj");

                    let gate_slice = g.add_tensor("layer.gate", vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                    g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: 0, slice_dim: intermediate_size },
                        vec![fused_out], vec![gate_slice], "layer.gate_slice");

                    let up_slice = g.add_tensor("layer.up", vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                    g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: intermediate_size, slice_dim: intermediate_size },
                        vec![fused_out], vec![up_slice], "layer.up_slice");
                    (gate_slice, up_slice)
                } else {
                    let gate_n = weight_shapes.get(&gate_cn).map(|s| s[0]).unwrap_or(intermediate_size);
                    let gate_w = g.add_tensor_concrete(&gate_cn, &[gate_n, hidden], tdt(&gate_cn));
                    layer_weight_tids.push((gate_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&gate_cn, &[gate_n, hidden]);

                    let gate_o = g.add_tensor("layer.gate", vec![s.clone(), SymDim::Concrete(gate_n)], dt);
                    add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_n, hidden,
                        vec![post_normed, gate_w], vec![gate_o], "layer.gate_proj");

                    let up_n = weight_shapes.get(&up_cn).map(|s| s[0]).unwrap_or(intermediate_size);
                    let up_w = g.add_tensor_concrete(&up_cn, &[up_n, hidden], tdt(&up_cn));
                    layer_weight_tids.push((up_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&up_cn, &[up_n, hidden]);

                    let up_o = g.add_tensor("layer.up", vec![s.clone(), SymDim::Concrete(up_n)], dt);
                    add_gemm_or_quant(&mut g, &up_cn, s.clone(), up_n, hidden,
                        vec![post_normed, up_w], vec![up_o], "layer.up_proj");
                    (gate_o, up_o)
                };

                let ffn_act_out = g.add_tensor("layer.ffn_act", vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                if matches!(features.ffn_type, FfnType::GeGLU) {
                    // GeGLU: GELU(gate) * up — fused BinaryElementwise trace
                    g.add_op(OpKind::GeGlu, vec![gate_out, up_out], vec![ffn_act_out], "layer.geglu");
                } else {
                    // SwiGLU: SiLU(gate) * up
                    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![ffn_act_out], "layer.swiglu");
                }

                let down_cn = cn_layer(0, "down_proj");
                let down_k = weight_shapes.get(&down_cn).map(|s| s[1]).unwrap_or(intermediate_size);
                let down_w = g.add_tensor_concrete(&down_cn, &[hidden, down_k], tdt(&down_cn));
                layer_weight_tids.push((down_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&down_cn, &[hidden, down_k]);

                let down_out = g.add_tensor("layer.down", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                add_gemm_or_quant(&mut g, &down_cn, s.clone(), hidden, down_k,
                    vec![ffn_act_out, down_w], vec![down_out], "layer.down_proj");

                let ffn_resid = g.add_tensor("layer.ffn_resid", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![ffn_resid_src, down_out], vec![ffn_resid], "layer.ffn_resid");
                ffn_resid_tid = Some(ffn_resid);
            }
            FfnType::Standard => {
                let up_cn = cn_layer(0, "up_proj");
                let up_n = weight_shapes.get(&up_cn).map(|s| s[0]).unwrap_or(intermediate_size);
                let up_w = g.add_tensor_concrete(&up_cn, &[up_n, hidden], tdt(&up_cn));
                layer_weight_tids.push((up_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&up_cn, &[up_n, hidden]);

                let up_out = g.add_tensor("layer.up", vec![s.clone(), SymDim::Concrete(up_n)], dt);
                let up_bias_cn = cn_layer_bias(0, "up_proj");
                if let Some(bias_shape) = weight_shapes.get(&up_bias_cn) {
                    let up_bias = g.add_tensor_concrete(&up_bias_cn, bias_shape, tdt(&up_bias_cn));
                    layer_weight_tids.push((up_bias, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&up_bias_cn, bias_shape);
                    g.add_op(OpKind::GemmBias { m: s.clone(), n: up_n, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, up_w, up_bias], vec![up_out], "layer.up_proj");
                } else {
                    add_gemm_or_quant(&mut g, &up_cn, s.clone(), up_n, hidden,
                        vec![post_normed, up_w], vec![up_out], "layer.up_proj");
                }

                let act_out = g.add_tensor("layer.act", vec![s.clone(), SymDim::Concrete(up_n)], dt);
                g.add_op(OpKind::Gelu, vec![up_out], vec![act_out], "layer.gelu");

                let down_cn = cn_layer(0, "down_proj");
                let down_w = g.add_tensor_concrete(&down_cn, &[hidden, up_n], tdt(&down_cn));
                layer_weight_tids.push((down_w, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&down_cn, &[hidden, up_n]);

                let down_out = g.add_tensor("layer.down", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                let down_bias_cn = cn_layer_bias(0, "down_proj");
                if let Some(bias_shape) = weight_shapes.get(&down_bias_cn) {
                    let down_bias = g.add_tensor_concrete(&down_bias_cn, bias_shape, tdt(&down_bias_cn));
                    layer_weight_tids.push((down_bias, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&down_bias_cn, bias_shape);
                    g.add_op(OpKind::GemmBias { m: s.clone(), n: hidden, k: up_n, dtype: dt, trans_b: true },
                        vec![act_out, down_w, down_bias], vec![down_out], "layer.down_proj");
                } else {
                    add_gemm_or_quant(&mut g, &down_cn, s.clone(), hidden, up_n,
                        vec![act_out, down_w], vec![down_out], "layer.down_proj");
                }

                let ffn_resid = g.add_tensor("layer.ffn_resid", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![ffn_resid_src, down_out], vec![ffn_resid], "layer.ffn_resid");
                ffn_resid_tid = Some(ffn_resid);
            }
            FfnType::MoE => {
                let num_experts = features.num_experts;
                let top_k = features.moe_top_k;
                if num_experts == 0 {
                    return Err(GraphBuildError::MissingTensor("layer 0 MoE router weight".into()));
                }

                let router_cn = cn_layer(0, "moe_gate");
                let (router_n, router_k) = weight_shapes.get(&router_cn)
                    .map(|s| (s[0], s[1]))
                    .unwrap_or((hidden, num_experts));
                let inter = weight_shapes.get(&cn_expert(0, 0, "gate_proj"))
                    .map(|s| s[0])
                    .unwrap_or(intermediate_size);

                let w_router = g.add_tensor_concrete(&router_cn, &[router_n, router_k], tdt(&router_cn));
                layer_weight_tids.push((w_router, layer_weight_byte_cursor));
                layer_weight_byte_cursor += weight_physical_bytes(&router_cn, &[router_n, router_k]);

                let gate_probs = g.add_tensor("layer.gate_probs", vec![s.clone(), SymDim::Concrete(num_experts)], dt);
                g.add_op(OpKind::MoEGate { seq_len: max_seq_len, num_experts, hidden, top_k },
                    vec![post_normed, w_router], vec![gate_probs], "layer.moe_gate");

                let topk_idx = g.add_tensor("layer.topk_idx", vec![s.clone(), SymDim::Concrete(top_k)], DType::F32);
                let topk_w = g.add_tensor("layer.topk_w", vec![s.clone(), SymDim::Concrete(top_k)], DType::F32);
                g.add_op(OpKind::TopK { seq_len: max_seq_len, num_experts, top_k },
                    vec![gate_probs], vec![topk_idx, topk_w], "layer.topk");

                let mut current_acc = g.add_tensor("layer.expert_acc", vec![s.clone(), SymDim::Concrete(hidden)], dt);

                for e in 0..num_experts {
                    let gate_exp_cn = cn_expert(0, e, "gate_proj");
                    let w_gate_e = g.add_tensor_concrete(&gate_exp_cn, &[hidden, inter], tdt(&gate_exp_cn));
                    layer_weight_tids.push((w_gate_e, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&gate_exp_cn, &[hidden, inter]);

                    let gate_out = g.add_tensor(&format!("layer.exp{}.gate", e), vec![s.clone(), SymDim::Concrete(inter)], dt);
                    g.add_op(OpKind::Gemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, w_gate_e], vec![gate_out], &format!("layer.exp{}.gate_gemm", e));

                    let mask_out = g.add_tensor(&format!("layer.exp{}.mask", e), vec![s.clone(), SymDim::Concrete(inter)], dt);
                    g.add_op(OpKind::GateMask { hidden: inter }, vec![gate_out], vec![mask_out], &format!("layer.exp{}.gate_mask", e));

                    let up_exp_cn = cn_expert(0, e, "up_proj");
                    let w_up_e = g.add_tensor_concrete(&up_exp_cn, &[hidden, inter], tdt(&up_exp_cn));
                    layer_weight_tids.push((w_up_e, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&up_exp_cn, &[hidden, inter]);

                    let up_out = g.add_tensor(&format!("layer.exp{}.up", e), vec![s.clone(), SymDim::Concrete(inter)], dt);
                    g.add_op(OpKind::MaskedGemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, w_up_e, mask_out], vec![up_out], &format!("layer.exp{}.up_gemm", e));

                    let swiglu_out = g.add_tensor(&format!("layer.exp{}.swiglu", e), vec![s.clone(), SymDim::Concrete(inter)], dt);
                    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], &format!("layer.exp{}.swiglu", e));

                    let down_exp_cn = cn_expert(0, e, "down_proj");
                    let w_down_e = g.add_tensor_concrete(&down_exp_cn, &[inter, hidden], tdt(&down_exp_cn));
                    layer_weight_tids.push((w_down_e, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&down_exp_cn, &[inter, hidden]);

                    let down_out = g.add_tensor(&format!("layer.exp{}.down", e), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                    g.add_op(OpKind::Gemm { m: s.clone(), n: hidden, k: inter, dtype: dt, trans_b: true },
                        vec![swiglu_out, w_down_e], vec![down_out], &format!("layer.exp{}.down_gemm", e));

                    let next_acc = g.add_tensor(&format!("layer.exp{}.acc", e), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                    g.add_op(OpKind::MoEConditionalAdd { seq_len: s.clone(), hidden, num_experts, expert_idx: e },
                        vec![current_acc, down_out, gate_probs], vec![next_acc], &format!("layer.exp{}.cond_add", e));
                    current_acc = next_acc;
                }

                if features.has_shared_experts {
                    let se_gate_cn = cn_shared(0, "gate_proj");
                    let se_gate_w = g.add_tensor_concrete(&se_gate_cn, &[hidden, inter], tdt(&se_gate_cn));
                    layer_weight_tids.push((se_gate_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&se_gate_cn, &[hidden, inter]);

                    let se_gate = g.add_tensor("layer.shared_gate", vec![s.clone(), SymDim::Concrete(inter)], dt);

                    let se_up_cn = cn_shared(0, "up_proj");
                    let se_up_w = g.add_tensor_concrete(&se_up_cn, &[hidden, inter], tdt(&se_up_cn));
                    layer_weight_tids.push((se_up_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&se_up_cn, &[hidden, inter]);

                    let se_up = g.add_tensor("layer.shared_up", vec![s.clone(), SymDim::Concrete(inter)], dt);
                    let se_swiglu = g.add_tensor("layer.shared_swiglu", vec![s.clone(), SymDim::Concrete(inter)], dt);

                    let se_down_cn = cn_shared(0, "down_proj");
                    let se_down_w = g.add_tensor_concrete(&se_down_cn, &[inter, hidden], tdt(&se_down_cn));
                    layer_weight_tids.push((se_down_w, layer_weight_byte_cursor));
                    layer_weight_byte_cursor += weight_physical_bytes(&se_down_cn, &[inter, hidden]);

                    let se_down = g.add_tensor("layer.shared_down", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                    let se_out = g.add_tensor("layer.shared_out", vec![s.clone(), SymDim::Concrete(hidden)], dt);

                    add_gemm_or_quant(&mut g, &se_gate_cn, s.clone(), inter, hidden,
                        vec![post_normed, se_gate_w], vec![se_gate], "layer.shared_gate_gemm");
                    add_gemm_or_quant(&mut g, &se_up_cn, s.clone(), inter, hidden,
                        vec![post_normed, se_up_w], vec![se_up], "layer.shared_up_gemm");
                    g.add_op(OpKind::SwiGlu, vec![se_gate, se_up], vec![se_swiglu], "layer.shared_swiglu");
                    add_gemm_or_quant(&mut g, &se_down_cn, s.clone(), hidden, inter,
                        vec![se_swiglu, se_down_w], vec![se_down], "layer.shared_down_gemm");
                    g.add_op(OpKind::Add, vec![current_acc, se_down], vec![se_out], "layer.shared_add");
                    current_acc = se_out;
                }

                let ffn_resid = g.add_tensor("layer.ffn_resid", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![ffn_resid_src, current_acc], vec![ffn_resid], "layer.moe_resid");
                ffn_resid_tid = Some(ffn_resid);
            }
        }

        // ── AltUp correct + PLE gate + inject (after FFN) ──
        let gated = ffn_resid_tid
            .ok_or_else(|| GraphBuildError::InvalidDimension("no ffn_resid from layer template".into()))?;

        let layer_output: TensorId;
        let fat_output_tid: Option<TensorId>;

        if has_altup {
            // AltUpCorrect: predictions + correction_coefs + gated → corrected [S, P*H]
            let corrected = g.add_tensor("layer.corrected",
                vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
            if let (Some(preds), Some(coefs)) = (predictions_tid, corr_coefs_tid) {
                g.add_op(OpKind::AltUpCorrect {
                    seq_len: s.clone(), num_preds: altup_p, hidden,
                }, vec![preds, coefs, gated], vec![corrected], "layer.altup_correct");
            } else {
                return Err(GraphBuildError::InvalidDimension(
                    "AltUp enabled but missing prediction/correction coefficients".into()));
            }

            // PLE gate: GELU(gemm(corrected[0], gate_w)) → gate [S, hpl]
            let hpl = features.hidden_size_per_layer_input;
            let corrected_active = g.add_tensor("layer.corrected_active",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::ColumnSlice {
                seq_len: s.clone(), input_inner: altup_p * hidden,
                start: 0, slice_dim: hidden,
            }, vec![corrected], vec![corrected_active], "layer.corrected_active_slice");

            let gate_cn = cn_layer(0, "per_layer_input_gate");
            let gate_logits = g.add_tensor("layer.ple_gate_logits",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                let gate_n = gate_shape[0];
                let gate_k = gate_shape[1];
                add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_n, gate_k,
                    vec![corrected_active], vec![gate_logits], "layer.ple_gate_gemm");
            }

            let gate = g.add_tensor("layer.ple_gate",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            g.add_op(OpKind::Gelu, vec![gate_logits], vec![gate], "layer.ple_gelu");

            // gate × ple_input → gated_ple [S, hpl]
            // ple_input is a per-layer tensor stepped by the layer loop (runtime).
            let ple_input = g.add_tensor("layer.ple_input",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            let gated_ple = g.add_tensor("layer.gated_ple",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            g.add_op(OpKind::Mul, vec![gate, ple_input], vec![gated_ple], "layer.ple_gated_mul");

            // GEMM(gated_ple, per_layer_proj_w) → projected [S, H]
            let ple_proj_cn = cn_layer(0, "per_layer_proj");
            let projected = g.add_tensor("layer.ple_projected",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                let proj_n = ple_shape[0];
                let proj_k = ple_shape[1];
                add_gemm_or_quant(&mut g, &ple_proj_cn, s.clone(), proj_n, proj_k,
                    vec![gated_ple], vec![projected], "layer.ple_proj_gemm");
            }

            // RMSNorm(projected, post_per_layer_input_norm) → normalized [S, H]
            let mut normalized = g.add_tensor("layer.ple_normalized",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            let ple_norm_cn = cn_layer(0, "post_per_layer_input_norm");
            if weight_shapes.contains_key(&ple_norm_cn) {
                let norm_shape = get_shape(weight_shapes, &ple_norm_cn)?;
                let norm_w = g.add_tensor_concrete(&ple_norm_cn, &norm_shape, tdt(&ple_norm_cn));
                g.add_op(OpKind::RmsNorm { eps },
                    vec![projected, norm_w], vec![normalized], "layer.ple_norm");
            } else {
                normalized = projected;
            }

            // AltUpInject: corrected[1:] += normalized → fat_output [S, P*H]
            let fat_out = g.add_tensor("altup.fat_out",
                vec![s.clone(), SymDim::Concrete(altup_p * hidden)], dt);
            g.add_op(OpKind::AltUpInject {
                seq_len: s.clone(), num_preds: altup_p, hidden,
            }, vec![corrected, normalized], vec![fat_out], "layer.altup_inject");

            layer_output = fat_out;
            fat_output_tid = Some(fat_out);

            // For post-layer code, store the gated (active path) as hidden_0
            tensor_map.insert("hidden_0".to_string(), gated);
        } else if features.has_per_layer_embedding {
            // PLE-only path (no AltUp): Gemma 4 E2B/E4B GGUF has PLE weights but no AltUp weights.
            // Data flow: cur → inp_gate GEMM → GELU → mul(ple_input) → proj GEMM → RMSNorm → add residual
            let hpl = features.hidden_size_per_layer_input;

            let gate_cn = cn_layer(0, "per_layer_input_gate");
            let gate_logits = g.add_tensor("layer.ple_gate_logits",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            if let Some(gate_shape) = weight_shapes.get(&gate_cn) {
                let gate_n = gate_shape[0];
                let gate_k = gate_shape[1];
                add_gemm_or_quant(&mut g, &gate_cn, s.clone(), gate_n, gate_k,
                    vec![gated], vec![gate_logits], "layer.ple_gate_gemm");
            }

            let gate = g.add_tensor("layer.ple_gate",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            g.add_op(OpKind::Gelu, vec![gate_logits], vec![gate], "layer.ple_gelu");

            let ple_input = g.add_tensor("layer.ple_input",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            let gated_ple = g.add_tensor("layer.gated_ple",
                vec![s.clone(), SymDim::Concrete(hpl)], dt);
            g.add_op(OpKind::Mul, vec![gate, ple_input], vec![gated_ple], "layer.ple_gated_mul");

            let ple_proj_cn = cn_layer(0, "per_layer_proj");
            let projected = g.add_tensor("layer.ple_projected",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            if let Some(ple_shape) = weight_shapes.get(&ple_proj_cn) {
                let proj_n = ple_shape[0];
                let proj_k = ple_shape[1];
                add_gemm_or_quant(&mut g, &ple_proj_cn, s.clone(), proj_n, proj_k,
                    vec![gated_ple], vec![projected], "layer.ple_proj_gemm");
            }

            let mut normalized = g.add_tensor("layer.ple_normalized",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            let ple_norm_cn = cn_layer(0, "post_per_layer_input_norm");
            if weight_shapes.contains_key(&ple_norm_cn) {
                let norm_shape = get_shape(weight_shapes, &ple_norm_cn)?;
                let norm_w = g.add_tensor_concrete(&ple_norm_cn, &norm_shape, tdt(&ple_norm_cn));
                g.add_op(OpKind::RmsNorm { eps },
                    vec![projected, norm_w], vec![normalized], "layer.ple_norm");
            } else {
                normalized = projected;
            }

            let ple_resid = g.add_tensor("layer.ple_resid",
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add, vec![gated, normalized], vec![ple_resid], "layer.ple_resid");
            layer_output = ple_resid;
            fat_output_tid = None;
            tensor_map.insert("hidden_0".to_string(), ple_resid);
        } else {
            if is_encoder {
                // Post-norm: apply PostAttnNorm (output.LayerNorm) after FFN residual
                let post_ffn_normed = g.add_tensor("layer.post_ffn_normed",
                    vec![s.clone(), SymDim::Concrete(hidden)], dt);
                if let Some(bias) = post_norm_bias_tid {
                    g.add_op(OpKind::LayerNorm { eps }, vec![gated, post_norm_w, bias],
                        vec![post_ffn_normed], "layer.post_norm");
                } else {
                    g.add_op(OpKind::RmsNorm { eps }, vec![gated, post_norm_w],
                        vec![post_ffn_normed], "layer.post_norm");
                }
                layer_output = post_ffn_normed;
                tensor_map.insert("hidden_0".to_string(), post_ffn_normed);
            } else {
                layer_output = gated;
                tensor_map.insert("hidden_0".to_string(), gated);
            }
            fat_output_tid = None;
        }

        // layer_output_scale: per-layer scalar multiply (Gemma 4 E2B/E4B)
        // NOTE: disabled for now — OpKind::Mul doesn't support [S,H] × [1] broadcasting.
        // Will be enabled once a ScalarMul OpKind is added.
        // let out_scale_cn = cn_layer(0, "layer_output_scale");
        // if let Some(scale_shape) = weight_shapes.get(&out_scale_cn) {
        //     ... TODO ...
        // }

        // Save layer config for later layer_loop_config setup
        _lc_weight_tids = layer_weight_tids.iter().map(|(t, _)| *t).collect();
        _lc_weight_stride = layer_weight_byte_cursor;
        if let Some(fat_in) = fat_input_tid {
            _lc_layer_input = Some(fat_in);
        } else {
            _lc_layer_input = Some(original_hidden);
        }
        _lc_layer_output = Some(layer_output);
        altup_fat_output_tid = fat_output_tid;
    }
    } // close else block

    let final_hidden_raw = tensor_map.get("hidden_0")
        .copied()
        .ok_or_else(|| GraphBuildError::InvalidDimension("no hidden_0 after layer loop".into()))?;

    // ── AltUp unembed + mean (post-layer, Gemma 4 E2B/E4B) ──
    // Unembed non-active paths through altup_unembed_projections.{i},
    // then mean-pool all P paths → [S, H] for final_norm.
    let final_hidden: TensorId;
    if has_altup {
        let fat_final = altup_fat_output_tid
            .ok_or_else(|| GraphBuildError::InvalidDimension(
                "AltUp enabled but no fat buffer output from layer template".into()))?;

        // Path 0 (active) — no unembed needed
        let path0 = g.add_tensor("altup.path0",
            vec![s.clone(), SymDim::Concrete(hidden)], dt);
        g.add_op(OpKind::ColumnSlice {
            seq_len: s.clone(), input_inner: altup_p * hidden,
            start: 0, slice_dim: hidden,
        }, vec![fat_final], vec![path0], "altup.path0_slice");

        // Unembed paths 1..P-1 and accumulate into mean
        let mut path_sum = path0;
        for i in 1..altup_p {
            let path_i = g.add_tensor(&format!("altup.path{}", i),
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::ColumnSlice {
                seq_len: s.clone(), input_inner: altup_p * hidden,
                start: i * hidden, slice_dim: hidden,
            }, vec![fat_final], vec![path_i], &format!("altup.path{}_slice", i));

            // Unembed projection (optional — only if weight exists)
            let unembed_cn = format!("altup_unembed_projection.{}", i);
            let unembedded_i = if let Some(unembed_shape) = weight_shapes.get(&unembed_cn) {
                let un_n = unembed_shape[0];
                let un_k = unembed_shape[1];
                let unembed_w = g.add_tensor_concrete(&unembed_cn, &[un_n, un_k], tdt(&unembed_cn));
                let unemb = g.add_tensor(&format!("altup.unembedded{}", i),
                    vec![s.clone(), SymDim::Concrete(un_n)], dt);
                g.add_op(OpKind::Gemm {
                    m: s.clone(), n: un_n, k: un_k, dtype: dt, trans_b: true,
                }, vec![path_i, unembed_w], vec![unemb], &format!("altup.unembed{}", i));
                unemb
            } else {
                path_i
            };

            // Accumulate: path_sum += unembedded_i
            let new_sum = g.add_tensor(&format!("altup.path_sum{}", i),
                vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::Add,
                vec![path_sum, unembedded_i], vec![new_sum], &format!("altup.accumulate{}", i));
            path_sum = new_sum;
        }

        // Mean: sum of P paths. 1/P scaling folded into final_norm (weight absorbs it).
        final_hidden = path_sum;
    } else {
        final_hidden = final_hidden_raw;
    }

    // ── Encoder post-layer: MeanPool → Classifier head ──
    if is_encoder {
        // MeanPool: average over seq dimension → [hidden]
        let pooled = g.add_tensor("pooled", vec![SymDim::Concrete(hidden)], dt);
        g.add_op(OpKind::MeanPool { seq_len: 0, hidden, cls_mode: false }, vec![final_hidden], vec![pooled], "meanpool");

        // Classifier head (if present): Dense → tanh → OutProj → output
        if features.has_classifier {
            // Detect classifier dense weight by canonical name patterns
            let cls_dense_cn = weight_shapes.keys()
                .find(|k| k.contains("classifier") && k.contains("dense") && !k.contains("bias"))
                .cloned().unwrap_or("classifier.dense.weight".to_string());
            let cls_dense_shape = weight_shapes.get(&cls_dense_cn);
            let (cls_n, cls_k) = match cls_dense_shape {
                Some(s) if s.len() >= 2 => (s[0], s[1]),
                Some(s) if s.len() == 1 => (s[0], hidden),
                _ => (hidden, hidden),
            };
            let cls_dense_w = g.add_tensor_concrete(&cls_dense_cn, &[cls_n, cls_k], tdt(&cls_dense_cn));
            let cls_dense_out = g.add_tensor("cls_dense_out", vec![SymDim::Concrete(cls_n)], dt);
            add_gemm_or_quant(&mut g, &cls_dense_cn, SymDim::Concrete(1), cls_n, cls_k,
                vec![pooled, cls_dense_w], vec![cls_dense_out], "cls_dense");

            // tanh activation
            let cls_act = g.add_tensor("cls_act", vec![SymDim::Concrete(cls_n)], dt);
            g.add_op(OpKind::Tanh, vec![cls_dense_out], vec![cls_act], "cls_tanh");

            // OutProj: [cls_n] → [num_labels] (typically 1 for rerankers)
            let cls_out_cn = weight_shapes.keys()
                .find(|k| (k.contains("classifier") || k.contains("score")) && !k.contains("dense") && !k.contains("bias"))
                .cloned().unwrap_or("classifier.out_proj.weight".to_string());
            let cls_out_shape = weight_shapes.get(&cls_out_cn);
            let (num_labels, cls_out_k) = match cls_out_shape {
                Some(s) if s.len() >= 2 => (s[0], s[1]),
                Some(s) if s.len() == 1 => (s[0], cls_n),
                _ => (1, cls_n),
            };
            let cls_out_w = g.add_tensor_concrete(&cls_out_cn, &[num_labels, cls_out_k], tdt(&cls_out_cn));
            let cls_result = g.add_tensor("cls_result", vec![SymDim::Concrete(num_labels)], dt);
            add_gemm_or_quant(&mut g, &cls_out_cn, SymDim::Concrete(1), num_labels, cls_out_k,
                vec![cls_act, cls_out_w], vec![cls_result], "cls_out_proj");

            // Check for classifier bias — use GemmBias if present
            let cls_out_bias_cn = cls_out_cn.clone().replace(".weight", ".bias");
            let has_bias = weight_shapes.contains_key(&cls_out_bias_cn);
            if has_bias {
                // Rebuild with GemmBias: undo the Gemm, redo as GemmBias
                let bias_tid = g.add_tensor_concrete(&cls_out_bias_cn, &[num_labels], tdt(&cls_out_bias_cn));
                // Overwrite the last Gemm with GemmBias (reuse same output tensor)
                let biased = g.add_tensor("cls_result_biased", vec![SymDim::Concrete(num_labels)], dt);
                g.add_op(OpKind::GemmBias { m: SymDim::Concrete(1), n: num_labels, k: cls_out_k, dtype: dt, trans_b: true },
                    vec![cls_act, cls_out_w, bias_tid], vec![biased], "cls_out_proj_biased");
                g.outputs = vec![biased];
            } else {
                g.outputs = vec![cls_result];
            }
        } else {
            // No classifier: output the pooled hidden state
            g.outputs = vec![pooled];
        }
    }

    // ── Decoder post-layer ──
    if !is_encoder {
        let final_norm_w = g.add_tensor_concrete("final_norm", &[hidden], tdt("final_norm"));
        let final_normed = g.add_tensor("final_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
        if use_rms {
            g.add_op(OpKind::RmsNorm { eps }, vec![final_hidden, final_norm_w], vec![final_normed], "final_norm");
        } else {
            let bias_tid = g.add_tensor_concrete("final_norm.bias", &[hidden], tdt("final_norm.bias"));
            g.add_op(OpKind::LayerNorm { eps }, vec![final_hidden, final_norm_w, bias_tid], vec![final_normed], "final_norm");
        }

        use gllm_kernels::compiler::mega_kernel_abi::OutputMode;
        let is_embed_or_rerank = business_config.output_modes.iter().any(|m| matches!(m, OutputMode::EncodeToLayer { .. }));

        if is_embed_or_rerank {
            // Decoder used as embedding/reranker: MeanPool → output hidden state
            let pooled = g.add_tensor("pooled", vec![SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::MeanPool { seq_len: 0, hidden, cls_mode: false }, vec![final_normed], vec![pooled], "meanpool");
            g.outputs = vec![pooled];
        } else {
            // Generator: lm_head → [LogitSoftcap] → Argmax → generate loop
            let lm_head_w = g.add_tensor_concrete("lm_head", &[vocab_size, hidden], tdt("lm_head"));
            let mut logits = g.add_tensor("logits", vec![s.clone(), SymDim::Concrete(vocab_size)], dt);
            add_gemm_or_quant(&mut g, "lm_head", s.clone(), vocab_size, hidden,
                vec![final_normed, lm_head_w],
                vec![logits],
                "lm_head",
            );

            // ── Logit softcapping (Gemma 4: cap=30.0, Grok: cap=24.0) ──
            // cap * tanh(logits / cap) before argmax — prevents extreme logit values.
            if let Some(cap) = config.final_logit_softcapping {
                let softcapped = g.add_tensor("logits_softcapped", vec![s.clone(), SymDim::Concrete(vocab_size)], dt);
                g.add_op(OpKind::LogitSoftcap { cap }, vec![logits], vec![softcapped], "logit_softcap");
                logits = softcapped;
            }

            // ── MTP (Multi-Token Prediction) projection nodes (REQ-MTP-003) ──
            // Each MTP depth projects the final hidden state through a per-depth
            // weight matrix to produce candidate token logits:
            //   mtp_logits_d = final_normed @ W_mtp[d]^T  → [seq_len, vocab_size]
            //   candidate_d = argmax(mtp_logits_d)
            //
            // MTP weights are canonical-named "mtp_proj.{d}" (global) or
            // "L{N}.mtp_proj.{d}" (per-layer), where d is the depth index.
            // Only added when mtp_config is present and MTP weights exist in weight_shapes.
            let mtp_depth = business_config.mtp_config.as_ref()
                .map(|c| c.depth)
                .unwrap_or(0);

            for d in 0..mtp_depth {
                let mtp_global = format!("mtp_proj.{}", d);
                let mtp_layered = format!("L{}.mtp_proj.{}", features.num_layers, d);

                // Resolve canonical name: prefer global variant, fall back to per-layer.
                let actual_cn = if weight_shapes.contains_key(&mtp_global) {
                    mtp_global
                } else if weight_shapes.contains_key(&mtp_layered) {
                    mtp_layered
                } else {
                    // Weight not found for this depth — skip
                    continue;
                };

                let mtp_shape = weight_shapes.get(&actual_cn)
                    .expect("MTP weight shape must exist after contains_key check");
                let mtp_n = if mtp_shape.len() >= 2 { mtp_shape[0] } else { vocab_size };
                let mtp_k = if mtp_shape.len() >= 2 { mtp_shape[1] } else { hidden };

                let mtp_w = g.add_tensor_concrete(&actual_cn, &[mtp_n, mtp_k], tdt(&actual_cn));
                let mtp_logits = g.add_tensor(
                    &format!("mtp_logits_{}", d),
                    vec![s.clone(), SymDim::Concrete(mtp_n)],
                    dt,
                );
                add_gemm_or_quant(&mut g, &actual_cn, s.clone(), mtp_n, mtp_k,
                    vec![final_normed, mtp_w],
                    vec![mtp_logits],
                    &format!("mtp_proj_{}", d),
                );

                let mtp_token = g.add_tensor(
                    &format!("mtp_token_{}", d),
                    vec![SymDim::Concrete(1)],
                    dt,
                );
                g.add_op(OpKind::Argmax { vocab_size: mtp_n },
                    vec![mtp_logits], vec![mtp_token], &format!("mtp_argmax_{}", d));
                g.add_op(OpKind::StoreToken,
                    vec![mtp_token], vec![], &format!("mtp_store_{}", d));
            }

            for mode in &business_config.output_modes {
                if let OutputMode::Generate { .. } = mode {
                    let token_id = g.add_tensor("token_id", vec![SymDim::Concrete(1)], dt);
                    g.add_op(OpKind::Argmax { vocab_size }, vec![logits], vec![token_id], "argmax");
                    g.add_op(OpKind::StoreToken, vec![token_id], vec![], "store_token");
                    g.add_op(OpKind::CheckStopCondition, vec![token_id], vec![], "check_stop");
                }
            }
        }
    }

    // Set graph inputs: all tensors without a producer (external inputs).
    // First input = activation (first non-weight tensor), rest = weights.
    // weight_layout() uses inputs[1..] for offset calculation.
    //
    // CRITICAL LAYOUT: Layer weights must come LAST so that the layer loop
    // stride doesn't overlap with global weights (final_norm, lm_head).
    // Layout: activations → global weights (embed, final_norm, lm_head) → layer template (L0.*)
    {
        let mut external: Vec<_> = g.tensors.iter()
            .filter(|t| t.producer.is_none())
            .collect();
        external.sort_by_key(|t| {
            let name = t.name.to_ascii_lowercase();
            if name.contains("token") || name == "input" || name == "hidden_0" {
                (0u8, 0u8)
            } else if name.starts_with("l0.") || name.starts_with("l4.")
                || name.starts_with("l15.") || name.starts_with("l19.")
                || name.starts_with("layer.")
                || name.starts_with("layer_sliding_small.") || name.starts_with("layer_full_small.")
                || name.starts_with("layer_sliding_large.") || name.starts_with("layer_full_large.")
            {
                // Layer template weights LAST. Within the layer group, sort by
                // type index to maintain correct template offset ordering:
                // L0 (type 0) < L4 (type 1) < L15 (type 2) < L19 (type 3).
                let type_idx = if name.starts_with("l0.") || name.starts_with("layer_sliding_small.") {
                    0u8
                } else if name.starts_with("l4.") || name.starts_with("layer_full_small.") {
                    1u8
                } else if name.starts_with("l15.") || name.starts_with("layer_sliding_large.") {
                    2u8
                } else {
                    3u8 // l19.* / layer_full_large.*
                };
                (2u8, type_idx)
            } else {
                (1u8, 0u8) // global weights (embed, final_norm, lm_head, etc.) in the middle
            }
        });
        g.inputs = external.iter().map(|t| t.id).collect();
    }

    // Set layer_loop_config or hetero_layer_loop_config
    if used_hetero {
        // ── Heterogeneous: 4 template strides ──
        // Helper: find input indices for a set of weight tensor IDs
        let find_indices = |tids: &[TensorId]| -> Vec<usize> {
            g.inputs.iter().enumerate()
                .filter_map(|(idx, &input_tid)| {
                    if tids.iter().any(|t| t == &input_tid) { Some(idx) } else { None }
                })
                .collect()
        };

        let sliding_small_indices = find_indices(&hetero_weight_tids_out[0]);
        let full_small_indices = find_indices(&hetero_weight_tids_out[1]);
        let sliding_large_indices = find_indices(&hetero_weight_tids_out[2]);
        let full_large_indices = find_indices(&hetero_weight_tids_out[3]);

        // Collect all hetero layer weight indices for global_weight_bytes computation
        let all_hetero_indices: Vec<usize> = sliding_small_indices.iter()
            .chain(&full_small_indices).chain(&sliding_large_indices).chain(&full_large_indices)
            .copied().collect();

        // [HETERO-DEBUG] Print sorted inputs and hetero indices
        eprintln!("[HETERO-DEBUG] === Sorted g.inputs ===");
        for (idx, &tid) in g.inputs.iter().enumerate() {
            let tensor = g.tensor(tid).unwrap();
            let numel: usize = tensor.shape.iter().map(|d| d.as_concrete().unwrap_or(1)).product();
            let is_hetero = all_hetero_indices.contains(&idx);
            let qt_info = weight_quant_types.get(&tensor.name)
                .map(|qt| format!(" quant={:?}", qt))
                .unwrap_or_default();
            eprintln!("[HETERO-DEBUG] inputs[{}] = {} (numel={}, bytes={}{}) {}",
                idx, tensor.name, numel,
                if numel > 0 {
                    if let Some(&qt) = weight_quant_types.get(&tensor.name) {
                        let bs = qt.block_size(); let bb = qt.block_bytes();
                        if numel.is_multiple_of(bs) && bs > 0 { (numel/bs)*bb } else { numel * dt.size_bytes() }
                    } else { numel * dt.size_bytes() }
                } else { 0 },
                qt_info,
                if is_hetero { "<-- HETERO" } else if idx == 0 { "<-- ACTIVATION" } else { "<-- GLOBAL" }
            );
        }
        eprintln!("[HETERO-DEBUG] all_hetero_indices = {:?}", all_hetero_indices);

        let mut global_weight_bytes = 0usize;
        for (idx, &tid) in g.inputs.iter().enumerate() {
            if idx == 0 { continue; }
            if all_hetero_indices.contains(&idx) { break; }
            let tensor = g.tensor(tid).unwrap();
            let numel: usize = tensor.shape.iter().map(|d| d.as_concrete().unwrap_or(1)).product();
            if numel > 0 {
                let name = &tensor.name;
                if let Some(&qt) = weight_quant_types.get(name) {
                    let bs = qt.block_size();
                    let bb = qt.block_bytes();
                    if numel.is_multiple_of(bs) && bs > 0 {
                        global_weight_bytes += (numel / bs) * bb;
                    } else {
                        global_weight_bytes += numel * dt.size_bytes();
                    }
                } else {
                    global_weight_bytes += numel * dt.size_bytes();
                }
            }
        }
        eprintln!("[HETERO-DEBUG] global_weight_bytes (layer_blob_base_offset) = {}", global_weight_bytes);

        g.hetero_layer_loop_config = Some(gllm_kernels::compiler::graph::HeteroLayerLoopConfig {
            num_segments: features.num_segments,
            sliding_per_segment: features.sliding_per_segment,
            sliding_small_stride: hetero_strides_out[0],
            full_small_stride: hetero_strides_out[1],
            sliding_large_stride: hetero_strides_out[2],
            full_large_stride: hetero_strides_out[3],
            small_segment_stride: hetero_small_seg_stride,
            large_segment_stride: hetero_large_seg_stride,
            large_ffn_start_segment: features.large_ffn_start_segment,
            layer_blob_base_offset: global_weight_bytes,
            sliding_small_weight_input_indices: sliding_small_indices,
            full_small_weight_input_indices: full_small_indices,
            sliding_large_weight_input_indices: sliding_large_indices,
            full_large_weight_input_indices: full_large_indices,
            activation_aliases: {
                let input_tid = _lc_layer_input.unwrap();
                let mut aliases: Vec<(TensorId, TensorId)> = Vec::new();
                for out_opt in &hetero_layer_outputs_out {
                    if let Some(out_tid) = out_opt {
                        aliases.push((input_tid, *out_tid));
                    }
                }
                // Fallback: ensure at least the type-3 alias is present
                if aliases.is_empty() {
                    aliases.push((input_tid, _lc_layer_output.unwrap()));
                }
                eprintln!("[HETERO-DEBUG] activation_aliases: input_tid={}, output_tids={:?}, count={}",
                    input_tid.0,
                    aliases.iter().map(|(_, o)| o.0).collect::<Vec<_>>(),
                    aliases.len());
                aliases
            },
        });
    } else if !_lc_weight_tids.is_empty() {
        // ── Homogeneous: single-template layer loop ──
        let layer_weight_input_indices: Vec<usize> = g.inputs.iter().enumerate()
            .filter_map(|(idx, &tid)| {
                if _lc_weight_tids.iter().any(|lt| lt == &tid) { Some(idx) } else { None }
            })
            .collect();

        let mut global_weight_bytes = 0usize;
        for (idx, &tid) in g.inputs.iter().enumerate() {
            if idx == 0 { continue; }
            if layer_weight_input_indices.contains(&idx) {
                break;
            }
            let tensor = g.tensor(tid).unwrap();
            let numel: usize = tensor.shape.iter().map(|d| d.as_concrete().unwrap_or(1)).product();
            if numel > 0 {
                let name = &tensor.name;
                if let Some(&qt) = weight_quant_types.get(name) {
                    let bs = qt.block_size();
                    let bb = qt.block_bytes();
                    if numel.is_multiple_of(bs) && bs > 0 {
                        global_weight_bytes += (numel / bs) * bb;
                    } else {
                        global_weight_bytes += numel * dt.size_bytes();
                    }
                } else {
                    global_weight_bytes += numel * dt.size_bytes();
                }
            }
        }

        let _ple_stride = if features.altup_num_inputs > 0 && features.hidden_size_per_layer_input > 0 {
            g.max_seq_len * features.hidden_size_per_layer_input * 4
        } else {
            0
        };
        g.layer_loop_config = Some(gllm_kernels::compiler::graph::LayerLoopConfig {
            num_layers: features.num_layers,
            weight_stride: _lc_weight_stride,
            layer_blob_base_offset: global_weight_bytes,
            layer_weight_input_indices,
            activation_alias: Some((_lc_layer_input.unwrap(), _lc_layer_output.unwrap())),
            per_layer_input_stride: _ple_stride,
        });
    }

    // Register physical byte sizes for all quantized weight tensors.
    let quant_sizes: Vec<(gllm_kernels::compiler::graph::TensorId, usize)> = g.ops.iter()
        .filter_map(|op| {
            match op.kind {
                OpKind::QuantGemm { n, k, quant_type, .. } => {
                    op.inputs.get(1).map(|&weight_tid| {
                        let bs = quant_type.block_size();
                        let bb = quant_type.block_bytes();
                        if k % bs == 0 && bs > 0 {
                            (weight_tid, n * (k / bs) * bb)
                        } else {
                            (weight_tid, 0)
                        }
                    })
                }
                OpKind::QuantGather { quant_type, vocab_size, hidden_dim, .. } => {
                    op.inputs.get(1).map(|&weight_tid| {
                        let bs = quant_type.block_size();
                        let bb = quant_type.block_bytes();
                        if hidden_dim % bs == 0 && bs > 0 {
                            (weight_tid, vocab_size * (hidden_dim / bs) * bb)
                        } else {
                            (weight_tid, 0)
                        }
                    })
                }
                _ => None,
            }
        })
        .collect();
    for (tid, bytes) in quant_sizes {
        if bytes > 0 {
            g.set_quant_weight_bytes(tid, bytes);
        }
    }

    g.max_seq_len = max_seq_len;
    g.embedding_scale = if features.has_embedding_scale {
        Some((hidden as f32).sqrt())
    } else {
        None
    };
    Ok(g)
}

