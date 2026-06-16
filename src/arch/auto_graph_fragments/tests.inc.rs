#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::compiler::mega_kernel_abi::BusinessConfig;
    use gllm_kernels::compiler::RopeScaling;

    fn make_role_index(
        entries: Vec<(TensorRole, Option<usize>, &str)>,
    ) -> HashMap<(TensorRole, Option<usize>), String> {
        entries
            .into_iter()
            .map(|(r, l, n)| ((r, l), n.to_string()))
            .collect()
    }

    fn make_weight_shapes(entries: Vec<(&str, Vec<usize>)>) -> HashMap<String, Vec<usize>> {
        entries
            .into_iter()
            .map(|(n, s)| (n.to_string(), s))
            .collect()
    }

    fn make_config(
        num_layers: usize,
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> ResolvedConfig {
        ResolvedConfig {
            num_hidden_layers: num_layers,
            hidden_size: hidden,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            intermediate_size: Some(hidden * 4),
            vocab_size: 100,
            rope_theta: 10000.0,
            dtype: "f32".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn auto_decoder_with_swiglu() {
        let config = make_config(2, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(0),
                "model.layers.0.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(0),
                "model.layers.0.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "model.layers.0.mlp.down_proj.weight",
            ),
            // Layer 1
            (
                TensorRole::InputNorm,
                Some(1),
                "model.layers.1.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(1),
                "model.layers.1.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(1),
                "model.layers.1.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(1),
                "model.layers.1.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(1),
                "model.layers.1.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(1),
                "model.layers.1.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(1),
                "model.layers.1.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(1),
                "model.layers.1.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(1),
                "model.layers.1.mlp.down_proj.weight",
            ),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None, None);
        assert_eq!(features.family, Family::Decoder);
        assert_eq!(features.num_layers, 2);
        assert!(features.has_rope);
        assert!(!features.has_head_rms_norm);
        assert_eq!(features.ffn_type, FfnType::SwiGLU);

        // Canonical-keyed weight_shapes (as executor would provide)
        // Single-template model: only L0 per-layer weights needed
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        // Single-template: 1 × 15 layer ops + 1 kv_write + 6 global ops = 22
        // layer: input_norm + q + k + v + rope_q + rope_k + kv_write + mha + o + resid + post_norm
        //        + gate + up + swiglu + down + ffn_resid = 15 + kv_write = 16
        // global: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        assert_eq!(
            graph.ops.len(),
            22,
            "expected 22 ops, got {}: {:?}",
            graph.ops.len(),
            graph
                .ops
                .iter()
                .map(|o| o.label.clone())
                .collect::<Vec<_>>()
        );

        // Verify canonical tensor names are used
        let tensor_names: Vec<&str> = graph.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            tensor_names.iter().any(|n| *n == "embed"),
            "embedding tensor should use canonical name 'embed', got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.iter().any(|n| *n == "L0.input_norm"),
            "input_norm tensor should use canonical name 'L0.input_norm', got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.iter().any(|n| *n == "L0.q_proj"),
            "q_proj tensor should use canonical name 'L0.q_proj', got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.iter().any(|n| *n == "L0.gate_proj"),
            "ffn gate tensor should use canonical name 'L0.gate_proj', got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.iter().any(|n| *n == "final_norm"),
            "final_norm tensor should use canonical name 'final_norm', got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.iter().any(|n| *n == "lm_head"),
            "lm_head tensor should use canonical name 'lm_head', got: {:?}",
            tensor_names
        );

        // Verify MHA dims
        let mha = graph
            .ops
            .iter()
            .find(|op| matches!(op.op_v2, Op::MultiHeadAttention(..)))
            .unwrap();
        if let Op::MultiHeadAttention(spec) = &mha.op_v2 {
            assert_eq!(spec.geometry.num_q_heads, 4);
            assert_eq!(spec.geometry.num_kv_heads, 2);
            assert_eq!(spec.geometry.head_dim, 16);
        }

        // Verify layer_loop_config is set correctly
        let llc = graph
            .layer_loop_config
            .as_ref()
            .expect("layer_loop_config should be set");
        assert_eq!(llc.num_layers, 2);
        assert!(llc.weight_stride > 0, "weight_stride should be non-zero");
        assert!(
            llc.activation_alias.is_some(),
            "activation_alias should be set"
        );
    }

    #[test]
    fn auto_encoder_with_layer_norm() {
        let config = make_config(2, 32, 2, 2, 16);

        let ri = make_role_index(vec![
            (
                TensorRole::Embedding,
                None,
                "roberta.embeddings.word_embeddings.weight",
            ),
            (
                TensorRole::EmbedNorm,
                None,
                "roberta.embeddings.LayerNorm.weight",
            ),
            // Layer 0
            (
                TensorRole::InputNorm,
                Some(0),
                "roberta.encoder.layer.0.attention.output.LayerNorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "roberta.encoder.layer.0.attention.self.query.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "roberta.encoder.layer.0.attention.self.key.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "roberta.encoder.layer.0.attention.self.value.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "roberta.encoder.layer.0.attention.output.dense.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "roberta.encoder.layer.0.output.LayerNorm.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(0),
                "roberta.encoder.layer.0.intermediate.dense.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "roberta.encoder.layer.0.output.dense.weight",
            ),
            // Layer 1
            (
                TensorRole::InputNorm,
                Some(1),
                "roberta.encoder.layer.1.attention.output.LayerNorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(1),
                "roberta.encoder.layer.1.attention.self.query.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(1),
                "roberta.encoder.layer.1.attention.self.key.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(1),
                "roberta.encoder.layer.1.attention.self.value.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(1),
                "roberta.encoder.layer.1.attention.output.dense.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(1),
                "roberta.encoder.layer.1.output.LayerNorm.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(1),
                "roberta.encoder.layer.1.intermediate.dense.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(1),
                "roberta.encoder.layer.1.output.dense.weight",
            ),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("roberta.embeddings.word_embeddings.weight", vec![50, 32]),
            (
                "roberta.encoder.layer.0.attention.self.query.weight",
                vec![32, 32],
            ),
            (
                "roberta.encoder.layer.0.attention.self.key.weight",
                vec![32, 32],
            ),
            (
                "roberta.encoder.layer.0.attention.self.value.weight",
                vec![32, 32],
            ),
            (
                "roberta.encoder.layer.0.intermediate.dense.weight",
                vec![64, 32],
            ),
            // BERT LayerNorm has bias — presence triggers LayerNorm detection
            (
                "roberta.encoder.layer.0.attention.output.LayerNorm.bias",
                vec![32],
            ),
            ("roberta.encoder.layer.0.output.LayerNorm.bias", vec![32]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None, None);
        assert_eq!(features.family, Family::Encoder);
        assert_eq!(features.num_layers, 2);
        assert!(!features.has_rope);
        assert_eq!(features.norm_type, NormType::LayerNorm); // Bias detected → LayerNorm
        assert_eq!(features.ffn_type, FfnType::Standard);

        // Canonical-keyed weight_shapes (single-template: only L0)
        let ws = make_weight_shapes(vec![
            ("embed", vec![50, 32]),
            ("L0.input_norm", vec![32]),
            ("L0.input_norm.bias", vec![32]),
            ("L0.q_proj", vec![32, 32]),
            ("L0.k_proj", vec![32, 32]),
            ("L0.v_proj", vec![32, 32]),
            ("L0.o_proj", vec![32, 32]),
            ("L0.post_attn_norm", vec![32]),
            ("L0.post_attn_norm.bias", vec![32]),
            ("L0.up_proj", vec![64, 32]),
            ("L0.down_proj", vec![32, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        // Single-template: embed_gather(1) + 1 × 12 layer ops + meanpool(1) = 14
        assert_eq!(
            graph.ops.len(),
            14,
            "encoder should have 14 ops (embed_gather + 1 template × 12 + meanpool), got {}",
            graph.ops.len()
        );

        // Verify LayerNorm (2 in template: input_norm + post_norm)
        let ln_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::LayerNorm(..)))
            .count();
        assert_eq!(ln_count, 2, "1 template × 2 norms = 2 LayerNorm ops");

        // Verify Gelu
        assert!(graph.ops.iter().any(|op| matches!(op.op_v2, Op::Gelu)));
    }

    #[test]
    fn auto_qwen3_with_head_rms_norm() {
        let config = make_config(2, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::AttentionQNorm,
                Some(0),
                "model.layers.0.self_attn.q_norm.weight",
            ),
            (
                TensorRole::AttentionKNorm,
                Some(0),
                "model.layers.0.self_attn.k_norm.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(0),
                "model.layers.0.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(0),
                "model.layers.0.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "model.layers.0.mlp.down_proj.weight",
            ),
            // Layer 1
            (
                TensorRole::InputNorm,
                Some(1),
                "model.layers.1.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(1),
                "model.layers.1.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(1),
                "model.layers.1.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(1),
                "model.layers.1.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(1),
                "model.layers.1.self_attn.o_proj.weight",
            ),
            (
                TensorRole::AttentionQNorm,
                Some(1),
                "model.layers.1.self_attn.q_norm.weight",
            ),
            (
                TensorRole::AttentionKNorm,
                Some(1),
                "model.layers.1.self_attn.k_norm.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(1),
                "model.layers.1.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(1),
                "model.layers.1.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(1),
                "model.layers.1.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(1),
                "model.layers.1.mlp.down_proj.weight",
            ),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None, None);
        assert!(
            features.has_head_rms_norm,
            "Qwen3 should have head_rms_norm"
        );

        // Canonical-keyed weight_shapes (single-template: only L0)
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.q_norm", vec![16]),
            ("L0.k_norm", vec![16]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        // Single-template: 1 × 17 layer ops + 1 kv_write + 6 global ops = 24
        // layer: input_norm + q + k + v + q_norm + k_norm + rope_q + rope_k + kv_write + mha + o + resid + post_norm
        //        + gate + up + swiglu + down + ffn_resid = 17 + kv_write = 18
        // global: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        let expected = 18 + 6;
        assert_eq!(
            graph.ops.len(),
            expected,
            "expected {} ops, got {}: {:?}",
            expected,
            graph.ops.len(),
            graph
                .ops
                .iter()
                .map(|o| o.label.clone())
                .collect::<Vec<_>>()
        );

        // Verify HeadRmsNorm ops (2 in template: q_norm + k_norm)
        let hrn_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::HeadRmsNorm { .. }))
            .count();
        assert_eq!(hrn_count, 2, "1 template × 2 (q+k) = 2 HeadRmsNorm ops");
    }

    #[test]
    fn auto_moe_decoder() {
        let config = make_config(1, 64, 4, 2, 16);
        let num_experts = 4;
        let top_k = 2;
        let inter = 128;

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::MoEGate,
                Some(0),
                "model.layers.0.mlp.gate.weight",
            ),
        ]);

        let mut ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate.weight", vec![64, num_experts]),
        ]);
        ws_ext.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
            vec![inter, 64],
        );

        let features = analyze_architecture(&ri, &ws_ext, None, None);
        assert_eq!(features.family, Family::Decoder);
        assert_eq!(features.num_layers, 1);
        assert!(features.is_moe);
        assert_eq!(features.ffn_type, FfnType::MoE);
        assert_eq!(features.num_experts, num_experts);
        assert_eq!(features.moe_top_k, top_k);

        // Canonical-keyed weight_shapes (single-template: L0 per-layer weights)
        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.moe_gate", vec![64, num_experts]),
        ]);
        // Expert weights
        for e in 0..num_experts {
            ws.insert(cn_expert(0, e, "gate_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "up_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "down_proj"), vec![64, inter]);
        }
        ws.insert("final_norm".to_string(), vec![64]);
        ws.insert("lm_head".to_string(), vec![100, 64]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("MoE graph build should succeed");

        // Per-layer ops:
        //   input_norm + q + k + v + rope_q + rope_k + mha + o + resid + post_norm = 10
        //   MoE: moe_gate + topk = 2
        //   Per expert (4): gate_gemm + gate_mask + up_gemm + swiglu + down_gemm + cond_add = 6×4 = 24
        //   moe_resid = 1
        //   kv_write = 1
        //   Total per-layer = 10 + 2 + 24 + 1 + 1 = 38
        // Global: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        let expected = 38 + 6;
        assert_eq!(
            graph.ops.len(),
            expected,
            "expected {} ops, got {}: {:?}",
            expected,
            graph.ops.len(),
            graph
                .ops
                .iter()
                .map(|o| o.label.clone())
                .collect::<Vec<_>>()
        );

        // Verify MoE-specific ops
        let moe_gate_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::MoEGate { .. }))
            .count();
        assert_eq!(moe_gate_count, 1, "should have 1 MoEGate op");

        let topk_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::TopK { .. }))
            .count();
        assert_eq!(topk_count, 1, "should have 1 TopK op");

        let cond_add_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::MoEConditionalAdd { .. }))
            .count();
        assert_eq!(
            cond_add_count, num_experts,
            "should have {} MoEConditionalAdd ops",
            num_experts
        );

        let swiglu_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::SwiGlu))
            .count();
        assert_eq!(
            swiglu_count, num_experts,
            "should have {} SwiGlu ops (one per expert)",
            num_experts
        );
    }

    #[test]
    fn auto_gemma4_qknorm_value_norm_embedding_scale() {
        let _config = make_config(1, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(0),
                "model.layers.0.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(0),
                "model.layers.0.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "model.layers.0.mlp.down_proj.weight",
            ),
        ]);

        let ws = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        // Without hints → no Gemma-4 features (hints drive config, not arch_name)
        let features_no_hints = analyze_architecture(&ri, &ws, None, None);
        assert!(!features_no_hints.has_qk_norm, "no hints → no qk_norm");
        assert!(
            !features_no_hints.has_value_norm,
            "no hints → no value_norm"
        );
        assert!(
            !features_no_hints.has_embedding_scale,
            "no hints → no embedding_scale"
        );

        // With ArchHints → Gemma-4 features enabled (REQ-MC-EXT-001..003)
        let gemma4_hints = ArchHints {
            qk_norm: Some(true),
            value_norm: Some(true),
            embedding_scale_factor: Some(8.0), // sqrt(hidden_size=64) = 8.0
            hidden_act: Some(HiddenAct::GeluNew),
            mla_use_unabsorbed: None,
        };
        let features = analyze_architecture(&ri, &ws, Some("gemma4"), Some(&gemma4_hints));
        assert!(features.has_qk_norm, "gemma4 hints should enable qk_norm");
        assert!(
            features.has_value_norm,
            "gemma4 hints should enable value_norm"
        );
        assert!(
            features.has_embedding_scale,
            "gemma4 hints should enable embedding_scale"
        );
        assert!(
            !features.has_head_rms_norm,
            "gemma4 should NOT have head_rms_norm (no weight tensors)"
        );

        // Qwen3 with q_norm weights → HeadRmsNorm, not QkNorm
        let mut ri_qwen3 = ri.clone();
        ri_qwen3.insert(
            (TensorRole::AttentionQNorm, Some(0)),
            "model.layers.0.self_attn.q_norm.weight".to_string(),
        );
        ri_qwen3.insert(
            (TensorRole::AttentionKNorm, Some(0)),
            "model.layers.0.self_attn.k_norm.weight".to_string(),
        );
        let features_qwen3 = analyze_architecture(&ri_qwen3, &ws, Some("qwen3"), None);
        assert!(
            features_qwen3.has_head_rms_norm,
            "qwen3 should have head_rms_norm"
        );
        assert!(!features_qwen3.has_qk_norm, "qwen3 should NOT have qk_norm");
        assert!(
            !features_qwen3.has_value_norm,
            "qwen3 should NOT have value_norm"
        );
    }

    /// T43: SharedKvRef — GprCondAction 条件分支验证
    ///
    /// 单模板层循环 (NO_LAYER_EXPAND) 下，consumer 层的 k_proj/v_proj/k_norm/rope_k
    /// 通过 `LayerCondition` guard 标记，运行时由 GprCondAction + CmpGeU 条件跳过。
    /// 非 SharedKvRef 模型 (num_kv_shared_layers=0) → guard=Always，零开销。
    #[test]
    fn auto_shared_kv_ref_guards_k_v_ops() {
        use gllm_kernels::compiler::graph::LayerCondition;

        let num_layers = 4;
        let num_shared = 2;
        let mut config = make_config(num_layers, 64, 4, 2, 16);
        config.num_kv_shared_layers = num_shared;

        // Template reads shapes from layer 0 (donor layer). Layer 0 needs all weights.
        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![64]);
        ws.insert(cn_layer(0, "q_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "o_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![64]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "down_proj"), vec![64, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with SharedKvRef");

        // ── 1. Verify NO_LAYER_EXPAND: single template, "layer." prefix labels ──
        let op_labels: Vec<&str> = graph.ops.iter().map(|o| o.label.as_str()).collect();
        assert!(
            op_labels.iter().any(|l| *l == "layer.k_proj"),
            "single template should have layer.k_proj"
        );
        assert!(
            op_labels.iter().any(|l| *l == "layer.v_proj"),
            "single template should have layer.v_proj"
        );
        // No per-layer expansion (L0_k_proj, L1_k_proj etc.)
        for i in 0..num_layers {
            let per_layer_k = format!("L{}_k_proj", i);
            assert!(
                !op_labels.iter().any(|l| *l == per_layer_k),
                "NO_LAYER_EXPAND: should NOT have per-layer {per_layer_k}"
            );
        }

        // ── 2. Guard verification: guarded ops ──
        let expected_guard = LayerCondition::LayerIdxLt(num_layers - num_shared);
        for label in &["layer.k_proj", "layer.v_proj", "layer.rope_k"] {
            let op = graph
                .ops
                .iter()
                .find(|o| o.label == *label)
                .unwrap_or_else(|| panic!("op '{}' should exist", label));
            assert_eq!(
                op.guard, expected_guard,
                "op '{}' should have guard {:?}, got {:?}",
                label, expected_guard, op.guard
            );
        }

        // ── 3. Guard verification: always-executed ops ──
        let always = LayerCondition::Always;
        for label in &[
            "layer.q_proj",
            "layer.o_proj",
            "layer.rope_q",
            "layer.mha",
            "layer.gate_proj",
            "layer.input_norm",
        ] {
            let op = graph
                .ops
                .iter()
                .find(|o| o.label == *label)
                .unwrap_or_else(|| panic!("op '{}' should exist", label));
            assert_eq!(
                op.guard, always,
                "op '{}' should be Always, got {:?}",
                label, op.guard
            );
        }
    }

    /// T43: 非 SharedKvRef 模型 (num_kv_shared_layers=0) → 全部 ops guard=Always，零开销。
    #[test]
    fn auto_no_shared_kv_ref_all_always() {
        use gllm_kernels::compiler::graph::LayerCondition;

        let num_layers = 4;
        let config = make_config(num_layers, 64, 4, 2, 16);
        assert_eq!(config.num_kv_shared_layers, 0, "default should be 0");

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![64]);
        ws.insert(cn_layer(0, "q_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "o_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![64]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "down_proj"), vec![64, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed without SharedKvRef");

        // All ops must have guard=Always (zero overhead)
        let always = LayerCondition::Always;
        for op in &graph.ops {
            assert_eq!(
                op.guard, always,
                "non-SharedKvRef: op '{}' should be Always, got {:?}",
                op.label, op.guard
            );
        }
    }

    /// T43: SharedKvRef with HeadRmsNorm (Qwen3-style) — k_norm also guarded.
    #[test]
    fn auto_shared_kv_ref_with_head_rms_norm() {
        use gllm_kernels::compiler::graph::LayerCondition;

        let num_layers = 4;
        let num_shared = 2;
        let mut config = make_config(num_layers, 64, 4, 2, 16);
        config.num_kv_shared_layers = num_shared;

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![64]);
        ws.insert(cn_layer(0, "q_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "q_norm"), vec![16]);
        ws.insert(cn_layer(0, "k_norm"), vec![16]);
        ws.insert(cn_layer(0, "o_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![64]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "down_proj"), vec![64, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: true,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with SharedKvRef + HeadRmsNorm");

        // k_norm should be guarded (consumer layers skip it)
        let expected_guard = LayerCondition::LayerIdxLt(num_layers - num_shared);
        let k_norm = graph
            .ops
            .iter()
            .find(|o| o.label == "layer.k_norm")
            .expect("layer.k_norm should exist");
        assert_eq!(
            k_norm.guard, expected_guard,
            "k_norm should be guarded with {:?}",
            expected_guard
        );

        // q_norm should be Always (every layer computes Q)
        let always = LayerCondition::Always;
        let q_norm = graph
            .ops
            .iter()
            .find(|o| o.label == "layer.q_norm")
            .expect("layer.q_norm should exist");
        assert_eq!(
            q_norm.guard, always,
            "q_norm should be Always, got {:?}",
            q_norm.guard
        );
    }

    /// ValueNorm (Gemma 4): V 投影后的无学习参数 RMSNorm，应添加 v_norm op 并用 kv_guard 保护
    #[test]
    fn auto_value_norm_guarded() {
        use gllm_kernels::compiler::graph::LayerCondition;

        let num_layers = 4;
        let num_shared = 2;
        let mut config = make_config(num_layers, 64, 4, 2, 16);
        config.num_kv_shared_layers = num_shared;

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![64]);
        ws.insert(cn_layer(0, "q_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "q_norm"), vec![16]);
        ws.insert(cn_layer(0, "k_norm"), vec![16]);
        ws.insert(cn_layer(0, "o_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![64]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "down_proj"), vec![64, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: true,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: true,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with ValueNorm");

        // v_norm should exist and be guarded (consumer layers skip it)
        let expected_guard = LayerCondition::LayerIdxLt(num_layers - num_shared);
        let v_norm = graph
            .ops
            .iter()
            .find(|o| o.label == "layer.v_norm")
            .expect("layer.v_norm should exist when has_value_norm=true");
        assert_eq!(
            v_norm.guard, expected_guard,
            "v_norm should be guarded with {:?}",
            expected_guard
        );

        // Verify the op kind is ValueNorm
        match &v_norm.op_v2 {
            Op::ValueNorm(spec) => {}
            other => panic!("v_norm should be OpKind::ValueNorm, got {:?}", other),
        }

        // q_norm and k_norm should still be present
        assert!(
            graph.ops.iter().any(|o| o.label == "layer.q_norm"),
            "q_norm should exist"
        );
        assert!(
            graph.ops.iter().any(|o| o.label == "layer.k_norm"),
            "k_norm should exist"
        );
    }

    /// ValueNorm without SharedKvRef: v_norm should exist but be Always
    #[test]
    fn auto_value_norm_no_shared_kv() {
        use gllm_kernels::compiler::graph::LayerCondition;

        let num_layers = 4;
        let config = make_config(num_layers, 64, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![64]);
        ws.insert(cn_layer(0, "q_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, 64]);
        ws.insert(cn_layer(0, "q_norm"), vec![16]);
        ws.insert(cn_layer(0, "k_norm"), vec![16]);
        ws.insert(cn_layer(0, "o_proj"), vec![64, 64]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![64]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, 64]);
        ws.insert(cn_layer(0, "down_proj"), vec![64, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: true,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: true,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with ValueNorm (no SharedKvRef)");

        // v_norm should be Always (no SharedKvRef → no guard needed)
        let always = LayerCondition::Always;
        let v_norm = graph
            .ops
            .iter()
            .find(|o| o.label == "layer.v_norm")
            .expect("layer.v_norm should exist when has_value_norm=true");
        assert_eq!(
            v_norm.guard, always,
            "v_norm should be Always without SharedKvRef, got {:?}",
            v_norm.guard
        );
    }

    /// AltUp placeholder: PerLayerEmbed is deprecated, replaced by AltUp.
    /// This test verifies the graph builds successfully with has_per_layer_embedding=true
    /// and that no deprecated PerLayerEmbed ops exist.
    /// Full AltUp graph construction will be tested once the AltUp build phase is implemented.
    /// See SPEC/DOCS/architecture/gemma4-altup.md for the complete design.
    #[test]
    fn auto_per_layer_embed_builds_without_deprecated_ops() {
        let num_layers = 4;
        let hidden = 64;
        let dim_per_layer = 16;
        let config = make_config(num_layers, hidden, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: true,
            hidden_size_per_layer_input: dim_per_layer,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with has_per_layer_embedding=true");
    }

    /// Verify that has_per_layer_embedding=false produces no AltUp ops.
    #[test]
    fn auto_no_altup_without_ple() {
        let num_layers = 4;
        let hidden = 64;
        let config = make_config(num_layers, hidden, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed without PLE");

        // Verify no AltUp ops exist when has_per_layer_embedding=false
        assert!(
            !graph.ops.iter().any(|o| matches!(
                &o.op_v2,
                Op::AltUpPredict { .. } | Op::AltUpCorrect { .. } | Op::AltUpInject { .. }
            )),
            "AltUp ops should NOT exist when has_per_layer_embedding=false"
        );
    }

    /// Verify that quantized weight types produce OpKind::QuantGemm instead of OpKind::Gemm.
    #[test]
    fn auto_quantized_weights_produce_quant_gemm() {
        let config = make_config(1, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::FfnGate,
                Some(0),
                "model.layers.0.mlp.gate_proj.weight",
            ),
            (
                TensorRole::FfnUp,
                Some(0),
                "model.layers.0.mlp.up_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "model.layers.0.mlp.down_proj.weight",
            ),
        ]);

        // External-name shapes (for analyze_architecture)
        let ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None, None);

        // Canonical-name shapes (for build_compiler_graph, as executor provides)
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        // Test with Q4_0 quantized weights for all projection layers
        let quant_types_q4_0: HashMap<String, gllm_kernels::quant::QuantType> = [
            ("L0.q_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.k_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.v_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.o_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.gate_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.up_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.down_proj", gllm_kernels::quant::QuantType::Q4_0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let graph_q4_0 = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types_q4_0,
            &BusinessConfig::default(),
            512,
        )
        .unwrap();

        // Verify QuantGemm ops are generated with correct quant_type
        let quant_gemms: Vec<_> = graph_q4_0
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::QuantGemm { .. }))
            .collect();
        assert!(
            quant_gemms.len() >= 4,
            "should have QuantGemm ops, got {} total ops",
            graph_q4_0.ops.len()
        );

        // Verify each QuantGemm has Q4_0
        for op in &quant_gemms {
            if let Op::QuantGemm(spec) = &op.op_v2 {
                assert_eq!(
                    spec.quant_type,
                    gllm_kernels::quant::QuantType::Q4_0,
                    "QuantGemm should have Q4_0, got {:?}",
                    spec.quant_type
                );
            }
        }

        // Verify no regular Gemm ops for quantized weights
        let regular_gemms: Vec<_> = graph_q4_0
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::Gemm(..)))
            .collect();
        // lm_head may still be Gemm if not in quant_types
        assert!(
            regular_gemms.len() <= 1,
            "lm_head should be the only non-quantized Gemm"
        );

        // Test with Q8_0 — verify different QuantType is passed through
        let quant_types_q8_0: HashMap<String, gllm_kernels::quant::QuantType> = [
            ("L0.q_proj", gllm_kernels::quant::QuantType::Q8_0),
            ("L0.k_proj", gllm_kernels::quant::QuantType::Q8_0),
            ("L0.v_proj", gllm_kernels::quant::QuantType::Q8_0),
            ("L0.o_proj", gllm_kernels::quant::QuantType::Q8_0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let graph_q8_0 = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types_q8_0,
            &BusinessConfig::default(),
            512,
        )
        .unwrap();

        let q8_gemms: Vec<_> = graph_q8_0
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::QuantGemm { .. }))
            .collect();
        for op in &q8_gemms {
            if let Op::QuantGemm(spec) = &op.op_v2 {
                assert_eq!(spec.quant_type, gllm_kernels::quant::QuantType::Q8_0);
            }
        }

        // Test with Q4_K — verify K-Quant type passes through
        let quant_types_q4k: HashMap<String, gllm_kernels::quant::QuantType> =
            [("L0.q_proj", gllm_kernels::quant::QuantType::Q4K)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();

        let graph_q4k = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types_q4k,
            &BusinessConfig::default(),
            512,
        )
        .unwrap();

        let q4k_gemms: Vec<_> = graph_q4k
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op_v2,
                    Op::QuantGemm(spec) if spec.quant_type == gllm_kernels::quant::QuantType::Q4K
                )
            })
            .collect();
        assert_eq!(q4k_gemms.len(), 1, "should have exactly 1 Q4K QuantGemm");
    }

    /// REQ-MTP-003: MTP graph nodes are added when mtp_config is present and MTP weights exist.
    #[test]
    fn auto_mtp_projection_nodes_added_with_global_weights() {
        let config = make_config(1, 64, 4, 2, 16);
        let mtp_depth = 2;

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
            // MTP projection weights (global variant)
            ("mtp_proj.0", vec![100, 64]),
            ("mtp_proj.1", vec![100, 64]),
        ]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let mut business_config = BusinessConfig::default();
        business_config.output_modes = vec![
            gllm_kernels::compiler::mega_kernel_abi::OutputMode::Generate {
                max_new_tokens: 16,
                eos_token_id: 2,
            },
        ];

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &business_config,
            2048,
        )
        .expect("MTP graph build should succeed");

        // Verify MTP ops are present: for each depth, we expect:
        //   mtp_proj_{d} (Gemm) + mtp_argmax_{d} (Argmax) + mtp_store_{d} (StoreToken) = 3 per depth
        // Total = 2 * 3 = 6 MTP ops
        let mtp_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| op.label.starts_with("mtp_"))
            .collect();
        assert_eq!(
            mtp_ops.len(),
            6,
            "expected 6 MTP ops (3 per depth × 2 depths), got {}: {:?}",
            mtp_ops.len(),
            mtp_ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>()
        );

        // Verify MTP Gemm ops
        for d in 0..mtp_depth {
            let label = format!("mtp_proj_{}", d);
            let gemm = graph.ops.iter().find(|op| op.label == label);
            assert!(gemm.is_some(), "missing MTP Gemm op: {}", label);
            let gemm_op = gemm.unwrap();
            match &gemm_op.op_v2 {
                Op::Gemm(spec) => {
                    let n = &spec.n;
                    let k = &spec.k;
                    assert_eq!(*n, 100, "MTP depth { } Gemm n should be vocab_size", d);
                    assert_eq!(*k, 64, "MTP depth {} Gemm k should be hidden_size", d);
                }
                other => panic!("expected Gemm for MTP depth {}, got {:?}", d, other),
            }
        }

        // Verify MTP Argmax ops
        for d in 0..mtp_depth {
            let label = format!("mtp_argmax_{}", d);
            let argmax = graph.ops.iter().find(|op| op.label == label);
            assert!(argmax.is_some(), "missing MTP Argmax op: {}", label);
        }

        // Verify MTP StoreToken ops
        for d in 0..mtp_depth {
            let label = format!("mtp_store_{}", d);
            let store = graph.ops.iter().find(|op| op.label == label);
            assert!(store.is_some(), "missing MTP StoreToken op: {}", label);
        }

        // Verify MTP weight tensors are in the graph
        for d in 0..mtp_depth {
            let tensor_name = format!("mtp_proj.{}", d);
            let found = graph.tensors.iter().any(|t| t.name == tensor_name);
            assert!(
                found,
                "MTP weight tensor '{}' should be in graph",
                tensor_name
            );
        }

        // Verify MTP output tensors
        for d in 0..mtp_depth {
            let logits_name = format!("mtp_logits_{}", d);
            let found = graph.tensors.iter().any(|t| t.name == logits_name);
            assert!(
                found,
                "MTP logits tensor '{}' should be in graph",
                logits_name
            );

            let token_name = format!("mtp_token_{}", d);
            let found = graph.tensors.iter().any(|t| t.name == token_name);
            assert!(
                found,
                "MTP token tensor '{}' should be in graph",
                token_name
            );
        }
    }

    /// REQ-MTP-003: MTP nodes are NOT added when mtp_config is absent.
    #[test]
    fn auto_mtp_no_nodes_when_config_absent() {
        let config = make_config(1, 64, 4, 2, 16);

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
            // Note: MTP depth is now topology-driven from weight_shapes (SPEC/39).
            // No mtp_proj weights → no MTP ops generated.
        ]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        let mtp_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| op.label.starts_with("mtp_"))
            .collect();
        assert!(
            mtp_ops.is_empty(),
            "no MTP ops expected when mtp_config is absent, got: {:?}",
            mtp_ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>()
        );
    }

    /// REQ-MTP-003: MTP nodes use per-layer canonical names when global variant missing.
    #[test]
    fn auto_mtp_with_layered_weight_names() {
        let config = make_config(2, 64, 4, 2, 16);
        let mtp_depth = 1;

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
            // Per-layer MTP weight (layered variant: L{num_layers}.mtp_proj.{d})
            ("L2.mtp_proj.0", vec![100, 64]),
        ]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 2,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let mut business_config = BusinessConfig::default();
        business_config.output_modes = vec![
            gllm_kernels::compiler::mega_kernel_abi::OutputMode::Generate {
                max_new_tokens: 16,
                eos_token_id: 2,
            },
        ];

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &business_config,
            2048,
        )
        .expect("MTP layered graph build should succeed");

        // Verify the layered MTP op is present
        let mtp_gemms: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| op.label == "mtp_proj_0")
            .collect();
        assert_eq!(
            mtp_gemms.len(),
            1,
            "should have 1 MTP Gemm op with layered weight"
        );

        // Verify the layered weight tensor is used
        let layered_tensor = graph.tensors.iter().find(|t| t.name == "L2.mtp_proj.0");
        assert!(
            layered_tensor.is_some(),
            "layered MTP weight tensor 'L2.mtp_proj.0' should be in graph"
        );
    }

    /// GPT-OSS feature detection: MoE + attention_bias + YaRN RoPE scaling.
    #[test]
    fn auto_gptoss_moe_with_yarn_rope() {
        let num_experts = 4;
        let _top_k = 2;
        let hidden = 64;
        let inter = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let mut config = make_config(1, hidden, num_heads, num_kv_heads, head_dim);
        config.rope_scaling = Some(gllm_kernels::compiler::graph::RopeScaling::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 8192,
        });

        // Feature detection with attention bias (GPT-OSS has q_proj.bias)
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_layernorm.weight",
            ),
            (
                TensorRole::AttentionQuery,
                Some(0),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                TensorRole::AttentionKey,
                Some(0),
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                TensorRole::AttentionValue,
                Some(0),
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                TensorRole::AttentionOutput,
                Some(0),
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                TensorRole::PostAttnNorm,
                Some(0),
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                TensorRole::MoEGate,
                Some(0),
                "model.layers.0.mlp.gate.weight",
            ),
        ]);

        let mut ws_ext: HashMap<String, Vec<usize>> = HashMap::new();
        ws_ext.insert("model.embed_tokens.weight".into(), vec![100, hidden]);
        ws_ext.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![hidden, hidden],
        );
        ws_ext.insert("model.layers.0.self_attn.q_proj.bias".into(), vec![hidden]);
        ws_ext.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![num_kv_heads * head_dim, hidden],
        );
        ws_ext.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![num_kv_heads * head_dim, hidden],
        );
        ws_ext.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![hidden, hidden],
        );
        ws_ext.insert(
            "model.layers.0.mlp.gate.weight".into(),
            vec![hidden, num_experts],
        );

        let features = analyze_architecture(&ri, &ws_ext, None, None);
        assert_eq!(features.family, Family::Decoder);
        assert!(features.is_moe, "GPT-OSS should detect MoE");
        assert!(
            features.has_attention_bias,
            "GPT-OSS should detect attention bias"
        );
        assert_eq!(features.ffn_type, FfnType::MoE);
        assert_eq!(features.num_experts, num_experts);

        // Build graph with YaRN scaling
        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("L0.input_norm", vec![hidden]),
            ("L0.q_proj", vec![hidden, hidden]),
            ("L0.k_proj", vec![num_kv_heads * head_dim, hidden]),
            ("L0.v_proj", vec![num_kv_heads * head_dim, hidden]),
            ("L0.o_proj", vec![hidden, hidden]),
            ("L0.post_attn_norm", vec![hidden]),
            ("L0.moe_gate", vec![hidden, num_experts]),
        ]);
        for e in 0..num_experts {
            ws.insert(cn_expert(0, e, "gate_proj"), vec![inter, hidden]);
            ws.insert(cn_expert(0, e, "up_proj"), vec![inter, hidden]);
            ws.insert(cn_expert(0, e, "down_proj"), vec![hidden, inter]);
        }
        ws.insert("final_norm".into(), vec![hidden]);
        ws.insert("lm_head".into(), vec![100, hidden]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            8192,
        )
        .expect("GPT-OSS graph build should succeed");

        // Verify YaRN scaling propagated to all RoPE ops
        let yarn_rope_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op_v2,
                    Op::RoPE(spec) if matches!(&spec.rope_scaling, Some(RopeScaling::Yarn { .. }))
                )
            })
            .collect();
        assert_eq!(
            yarn_rope_ops.len(),
            2,
            "expected 2 YaRN RoPE ops (Q+K), got {}: {:?}",
            yarn_rope_ops.len(),
            yarn_rope_ops.iter().map(|o| &o.label).collect::<Vec<_>>()
        );

        // Verify YaRN parameters
        for op in &yarn_rope_ops {
            if let Op::RoPE(spec) = &op.op_v2 {
                if let RopeScaling::Yarn {
                    factor,
                    beta_fast,
                    beta_slow,
                    original_max_position,
                } = spec.rope_scaling.as_ref().unwrap()
                {
                    assert_eq!(*factor, 32.0);
                    assert_eq!(*beta_fast, 32.0);
                    assert_eq!(*beta_slow, 1.0);
                    assert_eq!(*original_max_position, 8192);
                }
            }
        }

        // Verify MoE ops present
        assert!(
            graph
                .ops
                .iter()
                .any(|op| matches!(op.op_v2, Op::MoEGate { .. })),
            "MoEGate op missing"
        );
        assert!(
            graph
                .ops
                .iter()
                .any(|op| matches!(op.op_v2, Op::TopK { .. })),
            "TopK op missing"
        );
    }

    // ── Pure enum/struct unit tests ──

    #[test]
    fn family_equality() {
        assert_eq!(Family::Decoder, Family::Decoder);
        assert_ne!(Family::Decoder, Family::Encoder);
    }

    #[test]
    fn family_clone() {
        let f = Family::Decoder;
        assert_eq!(f.clone(), Family::Decoder);
    }

    #[test]
    fn norm_type_copy() {
        let n = NormType::RmsNorm;
        let n2 = n;
        assert_eq!(n2, NormType::RmsNorm);
    }

    #[test]
    fn ffn_type_variants() {
        assert_eq!(FfnType::SwiGLU, FfnType::SwiGLU);
        assert_ne!(FfnType::SwiGLU, FfnType::GeGLU);
        assert_ne!(FfnType::Standard, FfnType::MoE);
    }

    #[test]
    fn ffn_type_clone() {
        let f = FfnType::MoE;
        assert_eq!(f.clone(), FfnType::MoE);
    }

    #[test]
    fn graph_build_error_display() {
        let e = GraphBuildError::MissingTensor("q_proj".into());
        assert!(e.to_string().contains("q_proj"));
        let e = GraphBuildError::InvalidDimension("hidden=0".into());
        assert!(e.to_string().contains("hidden=0"));
        let e = GraphBuildError::UnsupportedArchitecture("foo".into());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn architecture_features_minimal() {
        let features = ArchitectureFeatures {
            family: Family::Encoder,
            num_layers: 2,
            has_rope: false,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::LayerNorm,
            ffn_type: FfnType::Standard,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: true,

            is_post_norm: true,
            causal: false,
            has_absolute_position_embed: true,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        assert_eq!(features.family, Family::Encoder);
        assert_eq!(features.num_layers, 2);
        assert!(!features.has_rope);
        assert_eq!(features.norm_type, NormType::LayerNorm);
        assert_eq!(features.ffn_type, FfnType::Standard);
        assert!(features.has_classifier);
        assert!(!features.is_moe);
        assert!(!features.is_mla);
    }

    #[test]
    fn architecture_features_mla() {
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 60,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::MoE,
            is_moe: true,
            has_shared_experts: true,
            num_experts: 256,
            moe_top_k: 8,
            is_mla: true,
            mla_latent_dim: 512,
            mla_rope_dim: 64,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        assert!(features.is_mla);
        assert_eq!(features.mla_latent_dim, 512);
        assert_eq!(features.mla_rope_dim, 64);
        assert!(features.is_moe);
        assert!(features.has_shared_experts);
        assert_eq!(features.num_experts, 256);
    }

    // ── analyze_architecture edge cases ──

    #[test]
    fn analyze_empty_role_index() {
        let ri: HashMap<(TensorRole, Option<usize>), String> = HashMap::new();
        let ws: HashMap<String, Vec<usize>> = HashMap::new();
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.family, Family::Encoder); // no OutputHead → Encoder
        assert_eq!(features.num_layers, 0);
        assert!(!features.has_rope);
        assert_eq!(features.ffn_type, FfnType::Standard);
    }

    #[test]
    fn analyze_encoder_no_output_head() {
        // Encoder: only Embedding, no OutputHead/FinalNorm
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::ClassifierOutProj, None, "classifier.weight"),
        ]);
        let ws = make_weight_shapes(vec![("embed.weight", vec![100, 32])]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.family, Family::Encoder);
        assert!(!features.has_rope);
        assert!(features.has_classifier);
    }

    // ── NormType tests ──

    #[test]
    fn norm_type_equality() {
        assert_eq!(NormType::RmsNorm, NormType::RmsNorm);
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_ne!(NormType::RmsNorm, NormType::LayerNorm);
    }

    #[test]
    fn norm_type_clone_matches() {
        assert_eq!(NormType::RmsNorm.clone(), NormType::RmsNorm);
        assert_eq!(NormType::LayerNorm.clone(), NormType::LayerNorm);
    }

    // ── GraphBuildError content tests ──

    #[test]
    fn graph_build_error_missing_tensor_message() {
        let e = GraphBuildError::MissingTensor("embed.weight".into());
        let msg = e.to_string();
        assert!(
            msg.contains("embed.weight"),
            "message should contain tensor name"
        );
        assert!(
            msg.to_lowercase().contains("missing"),
            "message should describe missing"
        );
    }

    #[test]
    fn graph_build_error_invalid_dimension_message() {
        let e = GraphBuildError::InvalidDimension("hidden=0".into());
        let msg = e.to_string();
        assert!(
            msg.contains("hidden=0"),
            "message should contain dimension info"
        );
        assert!(
            msg.to_lowercase().contains("invalid"),
            "message should describe invalid"
        );
    }

    #[test]
    fn graph_build_error_unsupported_arch_message() {
        let e = GraphBuildError::UnsupportedArchitecture("mamba".into());
        let msg = e.to_string();
        assert!(msg.contains("mamba"), "message should contain arch name");
        assert!(
            msg.to_lowercase().contains("unsupported"),
            "message should describe unsupported"
        );
    }

    // ── Family Debug formatting ──

    #[test]
    fn family_debug_format() {
        let d = format!("{:?}", Family::Decoder);
        let e = format!("{:?}", Family::Encoder);
        assert!(d.contains("Decoder"), "Debug should contain variant name");
        assert!(e.contains("Encoder"), "Debug should contain variant name");
    }

    // ── analyze_architecture: FinalNorm-only decoder ──

    #[test]
    fn analyze_decoder_with_final_norm_only() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::AttentionKey, Some(0), "L0.k_proj"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.family,
            Family::Decoder,
            "FinalNorm alone → Decoder"
        );
        assert!(
            features.has_rope,
            "Decoder with AttentionKey should have RoPE"
        );
    }

    // ── analyze_architecture: attention_sinks detection ──

    #[test]
    fn analyze_attention_sinks() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::AttentionSinks,
                Some(0),
                "model.layers.0.sinks.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(
            features.attention_sinks,
            "AttentionSinks role should be detected"
        );
    }

    // ── analyze_architecture: num_layers from max layer_idx ──

    #[test]
    fn analyze_num_layers_from_max_index() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::InputNorm, Some(0), "L0.norm"),
            (TensorRole::InputNorm, Some(3), "L3.norm"),
            (TensorRole::InputNorm, Some(7), "L7.norm"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.num_layers, 8, "max layer_idx=7 → 8 layers");
    }

    // ── analyze_architecture: MoE with shared experts ──

    #[test]
    fn analyze_moe_with_shared_experts() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::MoEGate,
                Some(0),
                "model.layers.0.mlp.gate.weight",
            ),
            (
                TensorRole::MoESharedExpert,
                Some(0),
                "model.layers.0.shared_expert.gate_proj.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![("model.layers.0.mlp.gate.weight", vec![64, 8])]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(features.is_moe);
        assert!(features.has_shared_experts);
        assert_eq!(features.num_experts, 8);
        assert_eq!(features.moe_top_k, 2, "default top_k = 2 when experts > 0");
    }

    // ── analyze_architecture: MoE without shared experts ──

    #[test]
    fn analyze_moe_without_shared_experts() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::MoEGate,
                Some(0),
                "model.layers.0.mlp.gate.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![("model.layers.0.mlp.gate.weight", vec![64, 4])]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(features.is_moe);
        assert!(!features.has_shared_experts);
        assert_eq!(features.num_experts, 4);
    }

    // ── analyze_architecture: MLA detection ──

    #[test]
    fn analyze_mla_detection() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::MlaKvCompress,
                Some(0),
                "model.layers.0.kv_b_proj.weight",
            ),
            (
                TensorRole::MlaKeyAbsorb,
                Some(0),
                "model.layers.0.k_b_proj.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(features.is_mla, "MlaKvCompress + MlaKeyAbsorb → is_mla");
    }

    // ── analyze_architecture: MLA latent and rope dims from weight shapes ──

    // ── analyze_architecture: attention_bias via .bias tensors ──

    #[test]
    fn analyze_attention_bias_detection() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![
            ("model.layers.0.self_attn.q_proj.bias", vec![64]),
            ("model.layers.0.self_attn.k_proj.bias", vec![32]),
        ]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(
            features.has_attention_bias,
            "q_proj.bias/k_proj.bias → attention_bias"
        );
    }

    #[test]
    fn analyze_no_attention_bias_without_bias_tensors() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![(
            "model.layers.0.self_attn.q_proj.weight",
            vec![64, 64],
        )]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(
            !features.has_attention_bias,
            "no q/k/v bias → no attention_bias"
        );
    }

    // ── analyze_architecture: FFN type Standard (no gate, up+down only) ──

    #[test]
    fn analyze_ffn_standard_no_gate() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::FfnUp, Some(0), "model.layers.0.up.weight"),
            (TensorRole::FfnDown, Some(0), "model.layers.0.down.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.ffn_type,
            FfnType::Standard,
            "up+down without gate → Standard"
        );
    }

    // ── analyze_architecture: FFN type SwiGLU from gate+down only (fused gate_up) ──

    #[test]
    fn analyze_ffn_swiglu_gate_down_only() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::FfnGate,
                Some(0),
                "model.layers.0.gate_proj.weight",
            ),
            (
                TensorRole::FfnDown,
                Some(0),
                "model.layers.0.down_proj.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.ffn_type,
            FfnType::SwiGLU,
            "gate+down → SwiGLU (fused gate_up)"
        );
    }

    // ── analyze_architecture: norm_type RmsNorm (no bias) ──

    #[test]
    fn analyze_rms_norm_no_bias() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_norm.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![("model.layers.0.input_norm.weight", vec![64])]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.norm_type, NormType::RmsNorm, "no .bias → RmsNorm");
    }

    // ── analyze_architecture: norm_type LayerNorm (has bias) ──

    #[test]
    fn analyze_layer_norm_with_bias() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (
                TensorRole::InputNorm,
                Some(0),
                "model.layers.0.input_norm.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![
            ("model.layers.0.input_norm.weight", vec![64]),
            ("model.layers.0.input_norm.bias", vec![64]),
        ]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.norm_type,
            NormType::LayerNorm,
            ".bias detected → LayerNorm"
        );
    }

    // ── analyze_architecture: encoder default LayerNorm ──

    #[test]
    fn analyze_encoder_default_layer_norm() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::EmbedNorm, None, "embeddings.LayerNorm.weight"),
            (TensorRole::ClassifierDense, None, "classifier.dense.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.family, Family::Encoder);
        assert!(features.is_post_norm, "EmbedNorm → is_post_norm");
        assert_eq!(
            features.norm_type,
            NormType::LayerNorm,
            "post-norm default → LayerNorm"
        );
    }

    // ── analyze_architecture: moe_top_k = 0 when no experts ──

    #[test]
    fn analyze_moe_top_k_zero_when_no_experts() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.moe_top_k, 0, "no MoE → top_k = 0");
    }

    // ── build_compiler_graph: missing embed weight → error ──

    #[test]
    fn build_graph_missing_embed_weight() {
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        let ws = make_weight_shapes(vec![]);
        let result = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        );
        assert!(result.is_err(), "missing embed → should error");
        let err = result.unwrap_err();
        match err {
            GraphBuildError::MissingTensor(name) => {
                assert_eq!(name, "embed", "should report missing embed tensor");
            }
            other => panic!("expected MissingTensor, got {:?}", other),
        }
    }

    // ── build_compiler_graph: missing post_attn_norm weight → error ──

    #[test]
    fn build_graph_missing_post_attn_norm() {
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            // No L0.post_attn_norm
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
        ]);
        let result = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        );
        assert!(result.is_err(), "missing post_attn_norm → should error");
    }

    // ── build_compiler_graph: MoE with zero experts → error ──

    #[test]
    fn build_graph_moe_zero_experts_error() {
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::MoE,
            is_moe: true,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
        ]);
        let result = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        );
        assert!(result.is_err(), "MoE with 0 experts → should error");
    }

    // ── build_compiler_graph: encoder without classifier → pooled output ──

    #[test]
    fn build_graph_encoder_without_classifier_outputs_pooled() {
        let config = make_config(1, 32, 2, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Encoder,
            num_layers: 1,
            has_rope: false,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::LayerNorm,
            ffn_type: FfnType::Standard,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: true,
            causal: false,
            has_absolute_position_embed: true,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        let ws = make_weight_shapes(vec![
            ("embed", vec![50, 32]),
            ("L0.input_norm", vec![32]),
            ("L0.input_norm.bias", vec![32]),
            ("L0.q_proj", vec![32, 32]),
            ("L0.k_proj", vec![32, 32]),
            ("L0.v_proj", vec![32, 32]),
            ("L0.o_proj", vec![32, 32]),
            ("L0.post_attn_norm", vec![32]),
            ("L0.post_attn_norm.bias", vec![32]),
            ("L0.up_proj", vec![64, 32]),
            ("L0.down_proj", vec![32, 64]),
        ]);
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("encoder without classifier should build");

        // Output should be pooled tensor, not classifier result
        assert_eq!(
            graph.outputs.len(),
            1,
            "encoder without classifier should have 1 output"
        );
        let output_tid = graph.outputs[0];
        let output_tensor = graph
            .tensor(output_tid)
            .expect("output tensor should exist");
        assert_eq!(
            output_tensor.name, "pooled",
            "output should be 'pooled' tensor"
        );
    }

    // ── build_compiler_graph: fused QKV path ──

    // ── build_compiler_graph: QuantGather for quantized embedding ──

    #[test]
    fn build_graph_quant_gather_for_quantized_embed() {
        let config = make_config(1, 64, 4, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let quant_types: HashMap<String, gllm_kernels::quant::QuantType> =
            [("embed", gllm_kernels::quant::QuantType::Q4_0)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types,
            &BusinessConfig::default(),
            2048,
        )
        .expect("quantized embed graph should build");

        let embed_op = graph
            .ops
            .iter()
            .find(|op| op.label == "embed_gather")
            .expect("should have embed_gather op");
        assert!(
            matches!(embed_op.op_v2, Op::QuantGather { .. }),
            "quantized embed → QuantGather, got {:?}",
            embed_op.op_v2
        );
    }

    // ── build_compiler_graph: Standard FFN with Gelu activation ──

    #[test]
    fn build_graph_standard_ffn_has_gelu() {
        let config = make_config(1, 32, 2, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Standard,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![50, 32]),
            ("L0.input_norm", vec![32]),
            ("L0.q_proj", vec![32, 32]),
            ("L0.k_proj", vec![32, 32]),
            ("L0.v_proj", vec![32, 32]),
            ("L0.o_proj", vec![32, 32]),
            ("L0.post_attn_norm", vec![32]),
            ("L0.up_proj", vec![64, 32]),
            ("L0.down_proj", vec![32, 64]),
            ("final_norm", vec![32]),
            ("lm_head", vec![50, 32]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("standard FFN graph should build");

        assert!(
            graph.ops.iter().any(|op| matches!(op.op_v2, Op::Gelu)),
            "standard FFN should have Gelu activation"
        );
        assert!(
            !graph.ops.iter().any(|op| matches!(op.op_v2, Op::SwiGlu)),
            "standard FFN should NOT have SwiGlu"
        );
    }

    // ── build_compiler_graph: graph.max_seq_len is set ──

    #[test]
    fn build_graph_max_seq_len_set() {
        let config = make_config(1, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::InputNorm, Some(0), "L0.norm"),
            (TensorRole::AttentionQuery, Some(0), "L0.q"),
            (TensorRole::AttentionKey, Some(0), "L0.k"),
            (TensorRole::AttentionValue, Some(0), "L0.v"),
            (TensorRole::AttentionOutput, Some(0), "L0.o"),
            (TensorRole::PostAttnNorm, Some(0), "L0.post_norm"),
            (TensorRole::FfnGate, Some(0), "L0.gate"),
            (TensorRole::FfnUp, Some(0), "L0.up"),
            (TensorRole::FfnDown, Some(0), "L0.down"),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("embed.weight", vec![100, 64]),
            ("L0.q", vec![64, 64]),
            ("L0.k", vec![32, 64]),
            ("L0.v", vec![32, 64]),
            ("L0.o", vec![64, 64]),
            ("L0.gate", vec![256, 64]),
            ("L0.up", vec![256, 64]),
            ("L0.down", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None, None);

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let max_seq = 4096;
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            max_seq,
        )
        .expect("graph build should succeed");

        assert_eq!(graph.max_seq_len, max_seq);
    }

    // ── build_compiler_graph: layer_loop_config present for multi-layer ──

    #[test]
    fn build_graph_layer_loop_config_multi_layer() {
        let config = make_config(3, 64, 4, 2, 16);

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 3,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("multi-layer graph should build");

        let llc = graph
            .layer_loop_config
            .as_ref()
            .expect("layer_loop_config should be set");
        assert_eq!(llc.num_layers, 3);
        assert!(llc.weight_stride > 0, "weight_stride should be positive");
        assert!(
            !llc.layer_weight_input_indices.is_empty(),
            "should have layer weight indices"
        );
    }

    // ── build_compiler_graph: MoE with shared experts → shared ops present ──

    #[test]
    fn build_graph_moe_with_shared_experts_ops() {
        let config = make_config(1, 64, 4, 2, 16);
        let num_experts = 2;
        let inter = 128;

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::MoE,
            is_moe: true,
            has_shared_experts: true,
            num_experts,
            moe_top_k: 2,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.moe_gate", vec![64, num_experts]),
        ]);
        for e in 0..num_experts {
            ws.insert(cn_expert(0, e, "gate_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "up_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "down_proj"), vec![64, inter]);
        }
        // Shared expert weights
        ws.insert(cn_shared(0, "gate_proj"), vec![inter, 64]);
        ws.insert(cn_shared(0, "up_proj"), vec![inter, 64]);
        ws.insert(cn_shared(0, "down_proj"), vec![64, inter]);
        ws.insert("final_norm".to_string(), vec![64]);
        ws.insert("lm_head".to_string(), vec![100, 64]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("MoE shared expert graph should build");

        let shared_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| op.label.contains("shared"))
            .collect();
        assert!(
            shared_ops.len() >= 3,
            "should have shared expert ops (gate, up, down, swiglu, add), got {}",
            shared_ops.len()
        );

        let shared_swiglu = graph
            .ops
            .iter()
            .filter(|op| op.label == "layer.shared_swiglu")
            .count();
        assert_eq!(shared_swiglu, 1, "should have 1 shared expert SwiGlu");
    }

    // ── build_compiler_graph: graph inputs are ordered correctly ──

    #[test]
    fn build_graph_input_ordering_activations_before_weights() {
        let config = make_config(1, 64, 4, 2, 16);

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        // First input should be token_ids (activation)
        let first_input = graph
            .tensor(graph.inputs[0])
            .expect("first input should exist");
        assert_eq!(
            first_input.name, "token_ids",
            "first input should be token_ids"
        );

        // Verify non-trivial number of inputs
        assert!(
            graph.inputs.len() > 2,
            "graph should have multiple inputs (token_ids + weights)"
        );
    }

    // ── build_compiler_graph: fused gate_up_proj (Phi4 style) ──

    #[test]
    fn build_graph_fused_gate_up_proj() {
        let config = make_config(1, 64, 4, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        // gate_proj has 2×intermediate rows, no up_proj → fused gate_up
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![512, 64]), // 2 × 256 = fused
            // No up_proj
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("fused gate_up graph should build");

        // Should have ColumnSlice ops for gate and up slices
        let gate_slices: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| op.label.contains("gate_slice") || op.label.contains("up_slice"))
            .collect();
        assert!(
            gate_slices.len() >= 2,
            "fused gate_up → gate_slice + up_slice, got {}",
            gate_slices.len()
        );
    }

    // ── analyze_architecture: no MoE gate → num_experts = 0 ──

    #[test]
    fn analyze_no_moe_gate_zero_experts() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.num_experts, 0);
        assert_eq!(features.moe_top_k, 0);
        assert!(!features.is_moe);
    }

    // ── ArchitectureFeatures: all boolean flags default false ──

    #[test]
    fn architecture_features_all_bools_false_by_default() {
        let features = ArchitectureFeatures {
            family: Family::Encoder,
            num_layers: 0,
            has_rope: false,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::LayerNorm,
            ffn_type: FfnType::Standard,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: true,
            causal: false,
            has_absolute_position_embed: true,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        // Verify every boolean is false
        assert!(!features.has_rope);
        assert!(!features.has_head_rms_norm);
        assert!(!features.has_attention_bias);
        assert!(!features.attention_sinks);
        assert!(!features.has_qk_norm);
        assert!(!features.has_value_norm);
        assert!(!features.has_embedding_scale);
        assert!(!features.is_moe);
        assert!(!features.has_shared_experts);
        assert!(!features.is_mla);
        assert!(!features.mla_use_unabsorbed);
        assert!(!features.has_classifier);
    }

    // ── FfnType Debug formatting ──

    #[test]
    fn ffn_type_debug_format() {
        let s = format!("{:?}", FfnType::SwiGLU);
        assert!(s.contains("SwiGLU"), "Debug should contain variant name");
        let s = format!("{:?}", FfnType::MoE);
        assert!(s.contains("MoE"), "Debug should contain variant name");
    }

    // ── ArchitectureFeatures Debug output ──

    #[test]
    fn architecture_features_debug_format() {
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        let debug = format!("{:?}", features);
        assert!(debug.contains("Decoder"), "Debug should contain family");
        assert!(debug.contains("RmsNorm"), "Debug should contain norm_type");
        assert!(debug.contains("SwiGLU"), "Debug should contain ffn_type");
    }

    // ── analyze_architecture: combined MoE + vision (e.g., vision MoE model) ──

    #[test]
    fn analyze_moe_and_vision_combined() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::MoEGate, Some(0), "L0.moe_gate"),
            (TensorRole::PatchEmbed, None, "vision.patch_embed.weight"),
        ]);
        let ws = make_weight_shapes(vec![("L0.moe_gate", vec![64, 8])]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(features.is_moe);
        assert_eq!(features.num_experts, 8);
    }

    // ── Canonical name helper tests ──

    #[test]
    fn cn_layer_formats_layer_suffix() {
        assert_eq!(cn_layer(0, "q_proj"), "L0.q_proj");
        assert_eq!(cn_layer(3, "input_norm"), "L3.input_norm");
        assert_eq!(cn_layer(99, "o_proj"), "L99.o_proj");
    }

    #[test]
    fn cn_layer_bias_formats_bias_suffix() {
        assert_eq!(cn_layer_bias(0, "input_norm"), "L0.input_norm.bias");
        assert_eq!(cn_layer_bias(7, "post_attn_norm"), "L7.post_attn_norm.bias");
    }

    #[test]
    fn cn_expert_formats_expert_weight() {
        assert_eq!(cn_expert(0, 0, "gate_proj"), "L0.expert.0.gate_proj");
        assert_eq!(cn_expert(2, 5, "down_proj"), "L2.expert.5.down_proj");
    }

    #[test]
    fn cn_shared_formats_shared_expert_weight() {
        assert_eq!(cn_shared(0, "gate_proj"), "L0.shared_expert.gate_proj");
        assert_eq!(cn_shared(3, "up_proj"), "L3.shared_expert.up_proj");
    }

    // ── GraphBuildError Debug trait ──

    #[test]
    fn graph_build_error_debug_format() {
        let e = GraphBuildError::MissingTensor("q_proj".into());
        let debug = format!("{:?}", e);
        assert!(
            debug.contains("MissingTensor"),
            "Debug should contain variant name"
        );

        let e = GraphBuildError::InvalidDimension("hidden=0".into());
        let debug = format!("{:?}", e);
        assert!(
            debug.contains("InvalidDimension"),
            "Debug should contain variant name"
        );

        let e = GraphBuildError::UnsupportedArchitecture("mamba".into());
        let debug = format!("{:?}", e);
        assert!(
            debug.contains("UnsupportedArchitecture"),
            "Debug should contain variant name"
        );
    }

    // ── FfnType exhaustive variant inequality ──

    #[test]
    fn ffn_type_all_variants_distinct() {
        let variants = [
            FfnType::SwiGLU,
            FfnType::GeGLU,
            FfnType::Standard,
            FfnType::MoE,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "FfnType variants at {} and {} should differ", i, j);
                }
            }
        }
    }

    // ── NormType Debug formatting ──

    #[test]
    fn norm_type_debug_format() {
        let rms = format!("{:?}", NormType::RmsNorm);
        let ln = format!("{:?}", NormType::LayerNorm);
        assert!(rms.contains("RmsNorm"), "Debug should contain RmsNorm");
        assert!(ln.contains("LayerNorm"), "Debug should contain LayerNorm");
    }

    // ── analyze_architecture: non-gemma4 arch_name does not enable gemma4 features ──

    #[test]
    fn analyze_non_gemma4_arch_no_qknorm() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        for arch in &["qwen3", "llama4", "deepseek", "phi4", "mistral3"] {
            let features = analyze_architecture(&ri, &ws, Some(arch), None);
            assert!(!features.has_qk_norm, "{} should not have qk_norm", arch);
            assert!(
                !features.has_value_norm,
                "{} should not have value_norm",
                arch
            );
            assert!(
                !features.has_embedding_scale,
                "{} should not have embedding_scale",
                arch
            );
        }
    }

    // ── analyze_architecture: moe_top_k defaults to 2 for various expert counts ──

    #[test]
    fn analyze_moe_top_k_defaults_to_2_for_various_expert_counts() {
        for num_experts in [4, 8, 16, 64, 256] {
            let ri = make_role_index(vec![
                (TensorRole::Embedding, None, "embed.weight"),
                (TensorRole::OutputHead, None, "lm_head.weight"),
                (TensorRole::FinalNorm, None, "norm.weight"),
                (TensorRole::MoEGate, Some(0), "L0.moe_gate"),
            ]);
            let ws = make_weight_shapes(vec![("L0.moe_gate", vec![64, num_experts])]);
            let features = analyze_architecture(&ri, &ws, None, None);
            assert_eq!(features.num_experts, num_experts);
            assert_eq!(
                features.moe_top_k, 2,
                "default top_k should be 2 for {} experts",
                num_experts
            );
        }
    }

    // ── build_compiler_graph: fused QKV dimension mismatch → error ──

    #[test]
    fn build_graph_fused_qkv_dimension_mismatch_error() {
        let config = make_config(1, 64, 4, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        // q_dim=64, k_dim=32 → expected fused=64+2*32=128, but providing 100
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.qkv_proj", vec![100, 64]), // wrong: should be 128
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let result = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        );
        assert!(result.is_err(), "fused QKV dimension mismatch should error");
        match result.unwrap_err() {
            GraphBuildError::InvalidDimension(msg) => {
                assert!(
                    msg.contains("fused QKV"),
                    "error should mention fused QKV, got: {}",
                    msg
                );
            }
            other => panic!("expected InvalidDimension, got {:?}", other),
        }
    }

    // ── analyze_architecture: MLA latent dim from weight shape ──

    #[test]
    fn analyze_mla_latent_dim_from_weight_shape() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::MlaKvCompress,
                Some(0),
                "model.layers.0.kv_b_proj.weight",
            ),
            (
                TensorRole::MlaKeyAbsorb,
                Some(0),
                "model.layers.0.k_b_proj.weight",
            ),
            (
                TensorRole::MlaRopeKey,
                Some(0),
                "model.layers.0.k_pe_proj.weight",
            ),
        ]);
        // analyze_architecture takes shape[1] for latent/rope dim
        let ws = make_weight_shapes(vec![
            ("model.layers.0.kv_b_proj.weight", vec![7168, 512]),
            ("model.layers.0.k_pe_proj.weight", vec![7168, 64]),
        ]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert!(features.is_mla);
        assert_eq!(features.mla_latent_dim, 512, "MLA latent dim from shape[1]");
        assert_eq!(features.mla_rope_dim, 64, "MLA rope dim from shape[1]");
    }

    // ── build_compiler_graph: graph has embed_gather as first op ──

    #[test]
    fn build_graph_embed_gather_is_first_op() {
        let config = make_config(1, 64, 4, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        assert!(!graph.ops.is_empty(), "graph should have ops");
        assert_eq!(
            graph.ops[0].label, "embed_gather",
            "first op should be embed_gather"
        );
    }

    // ── analyze_architecture: PositionEmbedding role does not affect family ──

    #[test]
    fn analyze_position_embedding_does_not_change_family() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::PositionEmbedding, None, "position_embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.family,
            Family::Decoder,
            "OutputHead present → Decoder regardless of PositionEmbedding"
        );
    }

    // ── build_compiler_graph: decoder graph has final_norm op ──

    #[test]
    fn build_graph_decoder_has_final_norm_op() {
        let config = make_config(1, 64, 4, 2, 16);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed");

        let final_norm_op = graph.ops.iter().find(|op| op.label == "final_norm");
        assert!(
            final_norm_op.is_some(),
            "decoder graph should have final_norm op"
        );
        assert!(
            matches!(final_norm_op.unwrap().op_v2, Op::RmsNorm(..)),
            "final_norm should be RmsNorm when norm_type is RmsNorm"
        );
    }

    // ── analyze_architecture: multiple layers with gaps in layer indices ──

    #[test]
    fn analyze_num_layers_with_sparse_layer_indices() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::InputNorm, Some(0), "L0.norm"),
            (TensorRole::InputNorm, Some(5), "L5.norm"),
            (TensorRole::InputNorm, Some(2), "L2.norm"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.num_layers, 6,
            "max layer_idx=5 → 6 layers (0-indexed)"
        );
    }

    // ── get_shape helper: returns correct shape for existing tensor ──

    #[test]
    fn get_shape_returns_shape_for_existing_tensor() {
        let ws = make_weight_shapes(vec![("embed", vec![100, 64]), ("L0.q_proj", vec![64, 32])]);
        let shape = get_shape(&ws, "embed").expect("should find embed");
        assert_eq!(shape, vec![100, 64]);

        let shape = get_shape(&ws, "L0.q_proj").expect("should find L0.q_proj");
        assert_eq!(shape, vec![64, 32]);
    }

    // ── get_shape helper: returns error for missing tensor ──

    #[test]
    fn get_shape_returns_error_for_missing_tensor() {
        let ws = make_weight_shapes(vec![("embed", vec![100, 64])]);
        let result = get_shape(&ws, "nonexistent");
        assert!(result.is_err(), "missing tensor should error");
        match result.unwrap_err() {
            GraphBuildError::MissingTensor(name) => {
                assert_eq!(name, "nonexistent");
            }
            other => panic!("expected MissingTensor, got {:?}", other),
        }
    }

    // ── ResolvedConfig default values ──

    #[test]
    fn resolved_config_default_has_zero_layers() {
        let config = ResolvedConfig::default();
        assert_eq!(config.num_hidden_layers, 0);
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.num_attention_heads, 0);
        assert_eq!(config.num_key_value_heads, 0);
        assert_eq!(config.head_dim, 0);
        assert_eq!(config.vocab_size, 0);
        assert_eq!(config.rope_theta, 0.0);
    }

    // ── GraphBuildError: MissingTensor equality via message content ──

    #[test]
    fn graph_build_error_different_missing_tensors() {
        let e1 = GraphBuildError::MissingTensor("embed".into());
        let e2 = GraphBuildError::MissingTensor("lm_head".into());
        assert_ne!(
            e1.to_string(),
            e2.to_string(),
            "different tensor names should produce different messages"
        );
    }

    // ── GraphBuildError: all variants produce non-empty messages ──

    #[test]
    fn graph_build_error_all_variants_non_empty() {
        let cases = vec![
            GraphBuildError::MissingTensor("x".into()),
            GraphBuildError::InvalidDimension("d=0".into()),
            GraphBuildError::UnsupportedArchitecture("mamba".into()),
        ];
        for err in &cases {
            let msg = err.to_string();
            assert!(
                !msg.is_empty(),
                "error message should not be empty for {:?}",
                err
            );
        }
    }

    // ── Family: Decoder and Encoder are the only two variants ──

    #[test]
    fn family_all_variants_covered() {
        // Verify both variants exist and are distinct
        let decoder = Family::Decoder;
        let encoder = Family::Encoder;
        assert_ne!(decoder, encoder);
        // Verify Debug format contains the variant name
        assert!(format!("{:?}", decoder).contains("Decoder"));
        assert!(format!("{:?}", encoder).contains("Encoder"));
    }

    // ── NormType: only two variants exist and are both Copy ──

    #[test]
    fn norm_type_copy_semantic() {
        let original = NormType::LayerNorm;
        let copied = original; // Copy, not Clone
        assert_eq!(original, copied);
        // Modify after copy should not affect original (Copy is bitwise)
        let _ = NormType::RmsNorm;
        assert_eq!(original, NormType::LayerNorm);
    }

    // ── FfnType: GeGLU variant exists and is distinct from SwiGLU ──

    #[test]
    fn ffn_type_geglu_is_distinct() {
        assert_ne!(
            FfnType::GeGLU,
            FfnType::SwiGLU,
            "GeGLU and SwiGLU should be distinct variants"
        );
        assert_ne!(
            FfnType::GeGLU,
            FfnType::Standard,
            "GeGLU and Standard should be distinct"
        );
        assert_ne!(
            FfnType::GeGLU,
            FfnType::MoE,
            "GeGLU and MoE should be distinct"
        );
        let debug = format!("{:?}", FfnType::GeGLU);
        assert!(
            debug.contains("GeGLU"),
            "Debug should contain GeGLU, got: {}",
            debug
        );
    }

    // ── make_config helper: produces consistent config ──

    #[test]
    fn make_config_helper_sets_dimensions_correctly() {
        let config = make_config(4, 128, 8, 4, 16);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 16);
        assert_eq!(config.intermediate_size, Some(128 * 4));
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.rope_theta, 10000.0);
    }

    // ── analyze_architecture: single layer detection ──

    #[test]
    fn analyze_single_layer_from_layer_zero() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::InputNorm, Some(0), "L0.norm"),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(features.num_layers, 1, "only layer 0 → num_layers = 1");
    }

    // ── analyze_architecture: MlaKvCompress alone is not enough for MLA ──

    #[test]
    fn analyze_mla_requires_both_roles() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (
                TensorRole::MlaKvCompress,
                Some(0),
                "model.layers.0.kv_b_proj.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        // MlaKvCompress alone triggers is_mla because of the any() check
        assert!(
            features.is_mla,
            "MlaKvCompress alone triggers MLA detection"
        );
        assert_eq!(
            features.mla_latent_dim, 0,
            "no weight shape → latent_dim = 0"
        );
    }

    // ── analyze_architecture: encoder with ClassifierOutProj only ──

    #[test]
    fn analyze_encoder_with_classifier_out_proj() {
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (
                TensorRole::ClassifierOutProj,
                None,
                "classifier.out_proj.weight",
            ),
        ]);
        let ws = make_weight_shapes(vec![]);
        let features = analyze_architecture(&ri, &ws, None, None);
        assert_eq!(
            features.family,
            Family::Encoder,
            "no OutputHead/FinalNorm → Encoder"
        );
        assert!(
            features.has_classifier,
            "ClassifierOutProj → has_classifier"
        );
    }

    // ── cn_layer: boundary layer index zero ──

    #[test]
    fn cn_layer_zero_index() {
        assert_eq!(cn_layer(0, "input_norm"), "L0.input_norm");
        assert_eq!(cn_layer(0, "post_attn_norm"), "L0.post_attn_norm");
    }

    // ── cn_expert: zero indices ──

    #[test]
    fn cn_expert_zero_indices() {
        assert_eq!(cn_expert(0, 0, "gate_proj"), "L0.expert.0.gate_proj");
        assert_eq!(cn_expert(0, 0, "down_proj"), "L0.expert.0.down_proj");
        assert_eq!(cn_expert(0, 0, "up_proj"), "L0.expert.0.up_proj");
    }

    // ── cn_shared: consistent format ──

    #[test]
    fn cn_shared_consistency_with_cn_expert() {
        // shared expert names should be distinct from expert names
        let shared = cn_shared(0, "gate_proj");
        let expert = cn_expert(0, 0, "gate_proj");
        assert_ne!(shared, expert, "shared and expert[0] names should differ");
        assert!(
            shared.contains("shared_expert"),
            "shared name should contain 'shared_expert'"
        );
        assert!(
            expert.contains("expert.0"),
            "expert name should contain 'expert.0'"
        );
    }

    // ── build_compiler_graph: missing q_proj weight → error ──

    #[test]
    fn build_graph_missing_q_proj_weight_error() {
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };
        // Provide embed and norm but no q_proj
        let ws = make_weight_shapes(vec![("embed", vec![100, 64]), ("L0.input_norm", vec![64])]);
        let result = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        );
        assert!(result.is_err(), "missing q_proj → should error");
        match result.unwrap_err() {
            GraphBuildError::MissingTensor(name) => {
                assert!(
                    name.contains("q_proj"),
                    "error should mention q_proj, got: {}",
                    name
                );
            }
            other => panic!("expected MissingTensor, got {:?}", other),
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // 13 NEW TESTS — fragment matching, arch detection edge cases,
    //   quant config handling, special float values, empty/boundary inputs
    // ────────────────────────────────────────────────────────────────────────

    // ── 2. Arch detection edge case: gemma4 arch_name is case-sensitive ──

    #[test]
    fn analyze_gemma4_arch_name_case_sensitive() {
        // Arrange
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);

        // Act: no hints → gemma4 features are all false regardless of arch_name
        let features_upper = analyze_architecture(&ri, &ws, Some("GEMMA4"), None);
        let features_mixed = analyze_architecture(&ri, &ws, Some("Gemma4"), None);
        let features_exact = analyze_architecture(&ri, &ws, Some("gemma4"), None);
        let features_no_arch = analyze_architecture(&ri, &ws, None, None);

        // Assert: without ArchHints, all config-driven features default to false
        assert!(!features_upper.has_qk_norm, "no hints → no qk_norm");
        assert!(!features_mixed.has_value_norm, "no hints → no value_norm");
        assert!(
            !features_exact.has_embedding_scale,
            "no hints → no embedding_scale"
        );
        assert!(!features_no_arch.has_qk_norm, "no hints → no qk_norm");

        // Assert: with ArchHints, features are enabled regardless of arch_name case
        let hints = ArchHints {
            qk_norm: Some(true),
            value_norm: Some(true),
            embedding_scale_factor: Some(8.0),
            ..Default::default()
        };
        let features_with_hints = analyze_architecture(&ri, &ws, Some("GEMMA4"), Some(&hints));
        assert!(features_with_hints.has_qk_norm, "hints → qk_norm enabled");
        assert!(
            features_with_hints.has_value_norm,
            "hints → value_norm enabled"
        );
        assert!(
            features_with_hints.has_embedding_scale,
            "hints → embedding_scale enabled"
        );
    }

    // ── 3. Quant config handling: partial quantization (only q_proj + k_proj) ──

    #[test]
    fn build_graph_partial_quantization_mixes_gemm_and_quant_gemm() {
        // Arrange
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        // Only q_proj and k_proj are quantized
        let quant_types: HashMap<String, gllm_kernels::quant::QuantType> = [
            ("L0.q_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.k_proj", gllm_kernels::quant::QuantType::Q4_0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types,
            &BusinessConfig::default(),
            2048,
        )
        .expect("partial quantization graph should build");

        // Assert: 2 QuantGemm + rest are regular Gemm
        let quant_gemm_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::QuantGemm { .. }))
            .count();
        assert_eq!(
            quant_gemm_count, 2,
            "only q_proj and k_proj should be QuantGemm"
        );

        // v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head → regular Gemm
        let regular_gemm_count = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::Gemm(..)))
            .count();
        assert!(
            regular_gemm_count >= 5,
            "unquantized projections should remain regular Gemm"
        );
    }

    // ── 4. Special float values: rope_theta = 0.0 in config ──

    #[test]
    fn build_graph_zero_rope_theta_uses_config_value() {
        // Arrange: config with rope_theta = 0.0
        let mut config = make_config(1, 64, 4, 2, 16);
        config.rope_theta = 0.0;

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("zero rope_theta graph should build");

        // Assert: RoPE ops should still be generated with theta=0.0
        let rope_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::RoPE { .. }))
            .collect();
        assert_eq!(
            rope_ops.len(),
            2,
            "decoder should have 2 RoPE ops (Q+K) even with theta=0.0"
        );
        for op in &rope_ops {
            if let Op::RoPE(spec) = &op.op_v2 {
                assert_eq!(spec.theta, 0.0, "RoPE theta should be 0.0 as configured");
            }
        }
    }

    // ── 5. Boundary input: single-layer encoder with minimal tensors ──

    #[test]
    fn analyze_single_layer_encoder_minimal_tensors() {
        // Arrange: absolute minimum for encoder (Embedding + EmbedNorm + ClassifierDense)
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "word_embeddings.weight"),
            (TensorRole::EmbedNorm, None, "embeddings.LayerNorm.weight"),
            (TensorRole::ClassifierDense, None, "classifier.dense.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);

        // Act
        let features = analyze_architecture(&ri, &ws, None, None);

        // Assert
        assert_eq!(features.family, Family::Encoder);
        assert_eq!(features.num_layers, 0, "no per-layer tensors → 0 layers");
        assert!(!features.has_rope, "post-norm should not have RoPE");
        assert!(features.is_post_norm, "EmbedNorm → is_post_norm");
        assert!(features.has_classifier);
        assert_eq!(
            features.norm_type,
            NormType::LayerNorm,
            "post-norm default → LayerNorm"
        );
        assert_eq!(
            features.ffn_type,
            FfnType::Standard,
            "no FFN roles → Standard"
        );
    }

    // ── 6. Fragment matching: GeGLU detected from gate+up+down ──

    #[test]
    fn analyze_ffn_geglu_from_gate_up_down() {
        // Arrange: gate+up+down but arch_name = "phi4" to trigger GeGLU
        // Note: analyze_architecture returns SwiGLU for gate+up+down by default,
        // so we verify that gate+up+down detection works correctly.
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::FfnGate, Some(0), "L0.gate.weight"),
            (TensorRole::FfnUp, Some(0), "L0.up.weight"),
            (TensorRole::FfnDown, Some(0), "L0.down.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);

        // Act
        let features = analyze_architecture(&ri, &ws, None, None);

        // Assert: gate+up+down → SwiGLU (default activation detection)
        assert_eq!(features.ffn_type, FfnType::SwiGLU, "gate+up+down → SwiGLU");
        assert!(!features.is_moe, "no MoEGate → not MoE");
    }

    // ── 7. Fragment matching: missing gate_proj but has up+down → Standard FFN in graph ──

    #[test]
    fn build_graph_standard_ffn_with_only_up_down_no_gate() {
        // Arrange
        let config = make_config(1, 32, 2, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Standard,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![50, 32]),
            ("L0.input_norm", vec![32]),
            ("L0.q_proj", vec![32, 32]),
            ("L0.k_proj", vec![32, 32]),
            ("L0.v_proj", vec![32, 32]),
            ("L0.o_proj", vec![32, 32]),
            ("L0.post_attn_norm", vec![32]),
            ("L0.up_proj", vec![64, 32]),
            ("L0.down_proj", vec![32, 64]),
            ("final_norm", vec![32]),
            ("lm_head", vec![50, 32]),
        ]);

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("standard FFN with up+down should build");

        // Assert: Gelu present, no SwiGLU
        assert!(
            graph.ops.iter().any(|op| matches!(op.op_v2, Op::Gelu)),
            "Standard FFN should have Gelu"
        );
        assert!(
            !graph.ops.iter().any(|op| matches!(op.op_v2, Op::SwiGlu)),
            "Standard FFN should not have SwiGlu"
        );

        // Assert: up GEMM and down GEMM present
        let has_up = graph.ops.iter().any(|op| op.label.contains("up_proj"));
        let has_down = graph.ops.iter().any(|op| op.label.contains("down_proj"));
        assert!(has_up, "should have up_proj GEMM");
        assert!(has_down, "should have down_proj GEMM");
    }

    // ── 8. Empty input: weight_shapes is empty → analyze returns defaults ──

    #[test]
    fn analyze_empty_weight_shapes_returns_defaults() {
        // Arrange: only role_index, no weight_shapes at all
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
        ]);
        let ws: HashMap<String, Vec<usize>> = HashMap::new();

        // Act
        let features = analyze_architecture(&ri, &ws, None, None);

        // Assert: sensible defaults
        assert_eq!(features.family, Family::Decoder);
        assert_eq!(features.num_layers, 0);
        assert!(
            !features.has_attention_bias,
            "no weight shapes → cannot detect bias → false"
        );
        assert_eq!(
            features.norm_type,
            NormType::RmsNorm,
            "no bias tensors → RmsNorm default for decoder"
        );
        assert_eq!(features.num_experts, 0, "no MoEGate → 0 experts");
    }

    // ── 9. Boundary: layer index 0 only → num_layers = 1 ──

    #[test]
    fn analyze_single_layer_index_zero_consistency() {
        // Arrange: single layer at index 0 with all core roles
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::InputNorm, Some(0), "L0.norm.weight"),
            (TensorRole::AttentionQuery, Some(0), "L0.q.weight"),
            (TensorRole::FfnGate, Some(0), "L0.gate.weight"),
            (TensorRole::FfnUp, Some(0), "L0.up.weight"),
            (TensorRole::FfnDown, Some(0), "L0.down.weight"),
        ]);
        let ws = make_weight_shapes(vec![]);

        // Act
        let features = analyze_architecture(&ri, &ws, None, None);

        // Assert
        assert_eq!(features.num_layers, 1, "only layer 0 → 1 layer");
        assert_eq!(features.ffn_type, FfnType::SwiGLU, "gate+down → SwiGLU");
    }

    // ── 10. Quant config: quantized embed + quantized projections together ──

    #[test]
    fn build_graph_quant_embed_and_quant_projections() {
        // Arrange
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        // Quantize both embed and projection weights
        let quant_types: HashMap<String, gllm_kernels::quant::QuantType> = [
            ("embed", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.q_proj", gllm_kernels::quant::QuantType::Q4_0),
            ("L0.k_proj", gllm_kernels::quant::QuantType::Q4_0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &quant_types,
            &BusinessConfig::default(),
            2048,
        )
        .expect("quant embed + quant projections graph should build");

        // Assert: embed_gather should be QuantGather
        let embed_op = graph
            .ops
            .iter()
            .find(|op| op.label == "embed_gather")
            .expect("should have embed_gather op");
        assert!(
            matches!(embed_op.op_v2, Op::QuantGather { .. }),
            "quantized embed → QuantGather"
        );

        // Assert: q_proj and k_proj should be QuantGemm, v_proj should be regular Gemm
        let quant_gemm_labels: Vec<&str> = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::QuantGemm { .. }))
            .map(|op| op.label.as_str())
            .collect();
        assert!(
            quant_gemm_labels.iter().any(|l| l.contains("q_proj")),
            "q_proj should be QuantGemm"
        );
        assert!(
            quant_gemm_labels.iter().any(|l| l.contains("k_proj")),
            "k_proj should be QuantGemm"
        );
    }

    // ── 11. Arch detection: multiple TensorRoles at same layer → correct feature union ──

    #[test]
    fn analyze_multiple_roles_same_layer_union_features() {
        // Arrange: layer 0 with MoE + attention sinks + bias + vision + audio all at once
        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "embed.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "norm.weight"),
            (TensorRole::AttentionSinks, Some(0), "L0.sinks.weight"),
            (TensorRole::MoEGate, Some(0), "L0.moe_gate.weight"),
            (TensorRole::DepthwiseConv, Some(0), "L0.dw_conv.weight"),
            (TensorRole::PatchEmbed, None, "vision.patch_embed.weight"),
        ]);
        let ws = make_weight_shapes(vec![
            ("L0.moe_gate.weight", vec![64, 4]),
            ("model.layers.0.self_attn.q_proj.bias", vec![64]),
        ]);

        // Act
        let features = analyze_architecture(&ri, &ws, None, None);

        // Assert: all features active simultaneously
        assert!(
            features.attention_sinks,
            "AttentionSinks at layer 0 → attention_sinks"
        );
        assert!(features.is_moe, "MoEGate at layer 0 → is_moe");
        assert!(features.has_attention_bias, "q_proj.bias → attention_bias");
        assert_eq!(features.num_experts, 4);
    }

    // ── 12. Fragment matching: decoder with Generate output mode has argmax + store_token ──

    #[test]
    fn build_graph_decoder_generate_mode_has_decode_ops() {
        // Arrange: standard 1-layer decoder with Generate output mode
        let config = make_config(1, 64, 4, 2, 16);
        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let mut business_config = BusinessConfig::default();
        business_config.output_modes = vec![
            gllm_kernels::compiler::mega_kernel_abi::OutputMode::Generate {
                max_new_tokens: 16,
                eos_token_id: 2,
            },
        ];

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &business_config,
            2048,
        )
        .expect("decoder graph with Generate mode should build");

        // Assert: should have argmax, store_token, and check_stop ops
        assert!(
            graph.ops.iter().any(|op| op.label == "argmax"),
            "decoder Generate mode should have argmax op"
        );
        assert!(
            graph.ops.iter().any(|op| op.label == "store_token"),
            "decoder Generate mode should have store_token op"
        );
        assert!(
            graph.ops.iter().any(|op| op.label == "check_stop"),
            "decoder Generate mode should have check_stop op"
        );

        // Assert: token_id tensor should exist
        let token_id_tensor = graph.tensors.iter().find(|t| t.name == "token_id");
        assert!(token_id_tensor.is_some(), "should have token_id tensor");
    }

    // ── 13. Special float values: rope_theta with very large value ──

    #[test]
    fn build_graph_large_rope_theta_propagates_to_rope_ops() {
        // Arrange: config with very large rope_theta (e.g., Gemma-style 1M)
        let mut config = make_config(1, 64, 4, 2, 16);
        config.rope_theta = 1_000_000.0;

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers: 1,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.input_norm", vec![64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.post_attn_norm", vec![64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        // Act
        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &HashMap::new(),
            &HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("large rope_theta graph should build");

        // Assert: both RoPE ops should carry the large theta
        let rope_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_v2, Op::RoPE { .. }))
            .collect();
        assert_eq!(rope_ops.len(), 2, "should have 2 RoPE ops");
        for op in &rope_ops {
            if let Op::RoPE(spec) = &op.op_v2 {
                assert_eq!(
                    spec.theta, 1_000_000.0,
                    "RoPE theta should be 1M as configured"
                );
            }
        }
    }

    /// QkNorm: when has_qk_norm=true and has_head_rms_norm=false,
    /// QkNorm ops should be emitted for Q and K after projection.
    #[test]
    fn auto_qk_norm_adds_ops_for_gemma4() {
        let num_layers = 4;
        let hidden = 64;
        let head_dim = 16;
        let config = make_config(num_layers, hidden, 4, 2, head_dim);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: true,
            has_value_norm: true,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: true,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with QkNorm");

        // Verify QkNorm ops exist for both Q and K
        let qk_norm_ops: Vec<_> = graph
            .ops
            .iter()
            .filter(|o| matches!(&o.op_v2, Op::QkNorm { .. }))
            .collect();
        assert_eq!(qk_norm_ops.len(), 2, "should have 2 QkNorm ops (Q and K)");

        // Verify Q QkNorm label
        let q_qk_norm = qk_norm_ops
            .iter()
            .find(|o| o.label == "layer.qk_norm_q")
            .expect("Q QkNorm op should exist");
        if let Op::QkNorm { head_dim: hd, .. } = &q_qk_norm.op_v2 {
            assert_eq!(*hd, head_dim, "QkNorm head_dim should match config");
        }

        // Verify K QkNorm label
        let _k_qk_norm = qk_norm_ops
            .iter()
            .find(|o| o.label == "layer.qk_norm_k")
            .expect("K QkNorm op should exist");

        // K QkNorm guard: Always when no SharedKvRef, LayerIdxLt when SharedKvRef active.
        // This test has num_kv_shared_layers=0, so guard is Always.

        // Verify QkNorm output tensors exist
        assert!(
            graph.tensors.iter().any(|t| t.name == "layer.qk_normed_q"),
            "layer.qk_normed_q tensor should exist"
        );
        assert!(
            graph.tensors.iter().any(|t| t.name == "layer.qk_normed_k"),
            "layer.qk_normed_k tensor should exist"
        );

        // Verify no HeadRmsNorm ops (mutually exclusive)
        assert!(
            !graph
                .ops
                .iter()
                .any(|o| matches!(&o.op_v2, Op::HeadRmsNorm { .. })),
            "HeadRmsNorm should NOT exist when QkNorm is active"
        );
    }

    /// QkNorm: when has_qk_norm=false, no QkNorm ops should be emitted.
    #[test]
    fn auto_no_qk_norm_without_feature() {
        let num_layers = 4;
        let hidden = 64;
        let config = make_config(num_layers, hidden, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed without QkNorm");

        // Verify no QkNorm ops exist
        assert!(
            !graph
                .ops
                .iter()
                .any(|o| matches!(&o.op_v2, Op::QkNorm { .. })),
            "QkNorm ops should NOT exist when has_qk_norm=false"
        );

        // Verify no qk_normed tensors exist
        assert!(
            !graph.tensors.iter().any(|t| t.name == "layer.qk_normed_q"),
            "layer.qk_normed_q tensor should NOT exist without QkNorm"
        );
        assert!(
            !graph.tensors.iter().any(|t| t.name == "layer.qk_normed_k"),
            "layer.qk_normed_k tensor should NOT exist without QkNorm"
        );
    }

    /// embedding_scale: Gemma 4 multiplies embeddings by sqrt(hidden_size).
    /// When has_embedding_scale=true, CompilerGraph.embedding_scale should be Some(sqrt(hidden)).
    #[test]
    fn auto_embedding_scale_set_for_gemma4() {
        let num_layers = 4;
        let hidden = 256;
        let config = make_config(num_layers, hidden, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: true,
            has_value_norm: true,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: true,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed with embedding_scale");

        let scale = graph
            .embedding_scale
            .expect("embedding_scale should be set for Gemma 4");
        let expected = (hidden as f32).sqrt();
        assert!(
            (scale - expected).abs() < 1e-3,
            "embedding_scale should be sqrt(hidden_size), got {scale}, expected {expected}"
        );
    }

    /// embedding_scale: when has_embedding_scale=false, it should be None.
    #[test]
    fn auto_no_embedding_scale_without_feature() {
        let num_layers = 4;
        let hidden = 256;
        let config = make_config(num_layers, hidden, 4, 2, 16);

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, hidden]),
            ("final_norm", vec![hidden]),
            ("lm_head", vec![100, hidden]),
        ]);
        ws.insert(cn_layer(0, "input_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "q_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "k_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "v_proj"), vec![32, hidden]);
        ws.insert(cn_layer(0, "o_proj"), vec![hidden, hidden]);
        ws.insert(cn_layer(0, "post_attn_norm"), vec![hidden]);
        ws.insert(cn_layer(0, "gate_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "up_proj"), vec![256, hidden]);
        ws.insert(cn_layer(0, "down_proj"), vec![hidden, 256]);

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,

            has_classifier: false,

            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        };

        let graph = build_compiler_graph(
            &features,
            &config,
            &ws,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &BusinessConfig::default(),
            2048,
        )
        .expect("graph build should succeed without embedding_scale");

        assert!(
            graph.embedding_scale.is_none(),
            "embedding_scale should be None when has_embedding_scale=false"
        );
    }
}
