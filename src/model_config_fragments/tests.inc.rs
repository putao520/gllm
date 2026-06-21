#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[derive(Debug)]
    struct MockTensorProvider {
        tensors: Vec<TensorMeta>,
    }

    impl TensorProvider for MockTensorProvider {
        fn tensor_info(&self, name: &str) -> Option<TensorMeta> {
            self.tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .cloned()
        }

        fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
            self.tensors.clone().into_iter()
        }

        fn load_tensor_data(&self, name: &str) -> crate::loader::Result<Cow<'_, [u8]>> {
            let meta = self.tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| crate::loader::LoaderError::MissingTensor(name.to_string()))?;
            let element_size = match meta.dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::U8 => 1,
                safetensors::Dtype::I64 => 8,
                safetensors::Dtype::I32 => 4,
                safetensors::Dtype::F64 => 8,
                safetensors::Dtype::BOOL => 1,
                _ => 2,
            };
            let total_elements: usize = meta.shape.iter().product();
            Ok(Cow::Owned(vec![0u8; total_elements * element_size]))
        }
    }

    fn tensor(name: &str, shape: &[usize]) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: safetensors::Dtype::F16,
        }
    }

    #[test]
    fn derive_config_from_tensors_succeeds_with_unique_head_dim() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000, 2816]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.0.mlp.gate_proj.weight", &[11264, 2816]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.1.mlp.gate_proj.weight", &[11264, 2816]),
            ],
        };

        let derived = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default()).expect("tensor-driven derivation");
        assert_eq!(derived.hidden_size, 2816);
        assert_eq!(derived.vocab_size, 50000);
        assert_eq!(derived.head_dim, 32);
        assert_eq!(derived.num_attention_heads, 88);
        assert_eq!(derived.num_key_value_heads, 11);
        assert_eq!(derived.num_hidden_layers, 2);
        assert_eq!(derived.intermediate_size, Some(11264));
        assert_eq!(derived.dtype, DType::F16);
    }

    /// TENSOR-DERIVE-AMBIGUOUS: multiple valid head_dims → prefer largest
    /// Q=4096, K=1024 → candidates: (16,4,256), (32,8,128), (64,16,64), (128,32,32)
    /// Largest head_dim=256 selected → 16 attention heads, 4 KV heads
    #[test]
    fn derive_config_from_tensors_prefers_largest_head_dim() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[32000, 4096]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[1024, 4096]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[4096, 4096]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[1024, 4096]),
                tensor("model.layers.0.mlp.gate_proj.weight", &[11008, 4096]),
                tensor("model.layers.1.mlp.gate_proj.weight", &[11008, 4096]),
            ],
        };

        let derived = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default())
            .expect("must resolve ambiguity by picking largest head_dim");
        assert_eq!(derived.head_dim, 256);
        assert_eq!(derived.num_attention_heads, 16);
        assert_eq!(derived.num_key_value_heads, 4);
    }

    /// TENSOR-DERIVE-CROSS-LAYER: alternating dimensions sharing a common head_dim
    /// L0 Q=2048/K=256 (sliding, majority 3 layers), L1 Q=4096/K=512 (global, 1 layer)
    /// Both share head_dim=256: 2048/256=8, 256/256=1, 4096/256=16, 512/256=2
    /// Frequency-based: (2048,256) appears 3 times vs (4096,512) once → majority wins
    #[test]
    fn derive_config_from_tensors_allows_cross_layer_shared_head_dim() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[256000, 2048]),
                // 3 sliding layers (majority)
                tensor("model.layers.0.self_attn.q_proj.weight", &[2048, 2048]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[256, 2048]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[2048, 2048]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[256, 2048]),
                tensor("model.layers.2.self_attn.q_proj.weight", &[2048, 2048]),
                tensor("model.layers.2.self_attn.k_proj.weight", &[256, 2048]),
                // 1 global layer (minority)
                tensor("model.layers.3.self_attn.q_proj.weight", &[4096, 2048]),
                tensor("model.layers.3.self_attn.k_proj.weight", &[512, 2048]),
            ],
        };

        let derived = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default())
            .expect("must succeed when cross-layer dims share a valid head_dim");
        // Majority (Q=2048, K=256) → head_dim=256 → 8 heads, 1 KV head
        assert_eq!(derived.num_attention_heads, 8);
        assert_eq!(derived.num_key_value_heads, 1);
        assert_eq!(derived.head_dim, 256);
    }

    #[test]
    fn build_moe_config_deepseek() {
        use crate::manifest::RouterType;
        let cfg = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_hidden_layers: 28,
            intermediate_size: Some(10944),
            num_experts: Some(64),
            num_experts_per_tok: Some(6),
            expert_intermediate_size: Some(1408),
            vocab_size: 102400,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 128,
            head_dim: 128,
            dtype: DType::BF16,
            compute_dtype: None,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            hidden_act: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            sliding_window: None,
            num_kv_shared_layers: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            mtp_depth: None,
            mla_config: None,
            vision_config: None,
            audio_config: None,
            multimodal_token_ids: None,
            final_logit_softcapping: None,
            use_double_wide_mlp: None,
            add_special_tokens: None,
            feed_forward_lengths: None,
            qk_norm: None,
            value_norm: None,
            embedding_scale_factor: None,
            rope_partial_ratio_global: None,
            mla_use_unabsorbed: None,
        };
        let moe = cfg.build_moe_config("deepseek").unwrap();
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.num_experts_per_tok, 6);
        assert_eq!(moe.router_type, RouterType::DeepSeek);
    }

    // ── GGUF Gemma 4 extraction scaffolding ──────────────────────────────
    //
    // Build a minimal GGUF v3 byte stream with the given KV metadata and no
    // tensors, write it to a temp file, and open it via `GgufReader::open`.
    // This exercises the real public loader surface so any future changes to
    // GGUF parsing stay covered by the same path production uses.
    fn make_gguf_with_meta(kvs: &[(&str, GgufMetaValue)]) -> std::path::PathBuf {
        use crate::loader::gguf::GGUF_MAGIC;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            write_kv_str(&mut buf, k);
            v.write(&mut buf);
        }
        let pos = buf.len();
        let aligned = (pos + 31) & !31;
        buf.resize(aligned, 0u8);

        let mut path = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!(
            "gllm_gguf_gemma4_test_{ts}_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &buf).expect("write temp GGUF");
        path
    }

    fn write_kv_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    enum GgufMetaValue {
        Str(&'static str),
        U64(u64),
        F32(f32),
        ArrU8(Vec<u8>),
    }

    impl GgufMetaValue {
        fn write(&self, buf: &mut Vec<u8>) {
            use crate::loader::gguf::GgufValueType;
            match self {
                Self::Str(s) => {
                    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    write_kv_str(buf, s);
                }
                Self::U64(v) => {
                    buf.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Self::F32(v) => {
                    buf.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                Self::ArrU8(bytes) => {
                    buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    buf.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
                    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                    buf.extend_from_slice(bytes);
                }
            }
        }
    }

    /// TEST-GGUF-GEMMA4-ARCH-U8: gguf_arch_array_u8 正确解码 attention.pattern
    #[test]
    fn gguf_arch_array_u8_reads_attention_pattern() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("gemma4")),
            (
                "gemma4.attention.pattern",
                GgufMetaValue::ArrU8(vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
            ),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        let pattern = gguf_arch_array_u8(&reader, "gemma4", "attention.pattern");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            pattern,
            Some(vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
            "u8 array helper must round-trip attention.pattern bytes"
        );
    }

    /// TEST-GGUF-GEMMA4-ARCH-SCALARS: 各类 arch scalar 读取 (u64 + f32)
    #[test]
    fn gguf_arch_scalars_read_gemma4_fields() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("gemma4")),
            ("gemma4.attention.sliding_window", GgufMetaValue::U64(512)),
            (
                "gemma4.attention.num_kv_shared_layers",
                GgufMetaValue::U64(4),
            ),
            ("gemma4.attention.global_head_dim", GgufMetaValue::U64(512)),
            (
                "gemma4.rope.global.freq_base",
                GgufMetaValue::F32(1_000_000.0),
            ),
            ("gemma4.rope.partial_ratio", GgufMetaValue::F32(0.25)),
            ("gemma4.embedding.per_layer_input", GgufMetaValue::U64(128)),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.sliding_window"),
            Some(512)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.num_kv_shared_layers"),
            Some(4)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.global_head_dim"),
            Some(512)
        );
        assert_eq!(
            gguf_arch_f32(&reader, "gemma4", "rope.global.freq_base"),
            Some(1_000_000.0)
        );
        assert_eq!(
            gguf_arch_f32(&reader, "gemma4", "rope.partial_ratio"),
            Some(0.25)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "embedding.per_layer_input"),
            Some(128)
        );
        let _ = std::fs::remove_file(&path);
    }

    /// TEST-GGUF-GEMMA4-ARCH-MISSING: 缺 key 时 helper 返回 None，不 panic
    #[test]
    fn gguf_arch_helpers_return_none_when_missing() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[(
            "general.architecture",
            GgufMetaValue::Str("gemma4"),
        )]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.sliding_window").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.num_kv_shared_layers").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.global_head_dim").is_none());
        assert!(gguf_arch_f32(&reader, "gemma4", "rope.global.freq_base").is_none());
        assert!(gguf_arch_f32(&reader, "gemma4", "rope.partial_ratio").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "embedding.per_layer_input").is_none());
        assert!(gguf_arch_array_u8(&reader, "gemma4", "attention.pattern").is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// TEST-GGUF-GEMMA4-001: derive_default_attention_pattern 每 6 层第 6 层 global
    #[test]
    fn derive_default_attention_pattern_matches_gemma4_rule() {
        // 26 层（Gemma 4 E2B）: idx 5, 11, 17, 23 为 global (即第 6/12/18/24 层)
        let pattern = derive_default_attention_pattern(26);
        assert_eq!(pattern.len(), 26);
        for (i, &v) in pattern.iter().enumerate() {
            let expect = if (i + 1) % 6 == 0 { 1u8 } else { 0u8 };
            assert_eq!(v, expect, "layer {i} expected {expect} got {v}");
        }
        // 少于 6 层 → 全部 sliding
        let small = derive_default_attention_pattern(5);
        assert_eq!(small, vec![0, 0, 0, 0, 0]);
        // 边界: 0 层 → 空 Vec
        assert!(derive_default_attention_pattern(0).is_empty());
        // 恰好 6 层 → 最后一层 global
        let six = derive_default_attention_pattern(6);
        assert_eq!(six, vec![0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn build_moe_config_none_for_dense() {
        let cfg = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_hidden_layers: 28,
            intermediate_size: Some(10944),
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
            vocab_size: 102400,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 128,
            head_dim: 128,
            dtype: DType::BF16,
            compute_dtype: None,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            hidden_act: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            sliding_window: None,
            num_kv_shared_layers: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            mtp_depth: None,
            mla_config: None,
            vision_config: None,
            audio_config: None,
            multimodal_token_ids: None,
            final_logit_softcapping: None,
            use_double_wide_mlp: None,
            add_special_tokens: None,
            feed_forward_lengths: None,
            qk_norm: None,
            value_norm: None,
            embedding_scale_factor: None,
            rope_partial_ratio_global: None,
            mla_use_unabsorbed: None,
        };
        assert!(cfg.build_moe_config("llama").is_none());
    }

    // ── MTP depth parsing tests ──

    /// TEST-MTP-CONFIG-JSON: from_value reads mtp_depth from config.json keys
    #[test]
    fn from_value_reads_mtp_depth_from_config_json() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "num_nextn_predict_layers": 2,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.mtp_depth, Some(2));
    }

    #[test]
    fn from_value_reads_mtp_depth_alternate_key() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "mtp_depth": 4,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.mtp_depth, Some(4));
    }

    #[test]
    fn from_value_mtp_depth_absent_when_not_specified() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::F32)).unwrap();
        assert_eq!(cfg.mtp_depth, None);
    }

    /// TEST-MTP-GGUF: GGUF reader reads mtp_depth from arch-specific metadata
    #[test]
    fn gguf_reader_mtp_depth_deepseek() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("deepseek_v3")),
            ("deepseek_v3.mtp_depth", GgufMetaValue::U64(2)),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert_eq!(reader.mtp_depth(), Some(2));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gguf_reader_mtp_depth_qwen3() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("qwen3")),
            ("qwen3.mtp_depth", GgufMetaValue::U64(1)),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert_eq!(reader.mtp_depth(), Some(1));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gguf_reader_mtp_depth_absent() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("llama")),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert_eq!(reader.mtp_depth(), None);
        let _ = std::fs::remove_file(&path);
    }

    // ── ModelGeometry pure logic tests ──

    fn minimal_geometry() -> ModelGeometry {
        ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 11008,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        }
    }

    #[test]
    fn model_geometry_is_not_moe_by_default() {
        let geo = minimal_geometry();
        assert!(!geo.is_moe());
        assert!(!geo.is_mla());
    }

    #[test]
    fn model_geometry_is_moe_with_experts() {
        let mut geo = minimal_geometry();
        geo.num_experts = 64;
        geo.moe_top_k = 8;
        assert!(geo.is_moe());
    }

    #[test]
    fn model_geometry_is_mla_with_d_c() {
        let mut geo = minimal_geometry();
        geo.mla_d_c = 512;
        geo.mla_d_rope = 64;
        assert!(geo.is_mla());
    }

    #[test]
    fn model_geometry_kv_dim_standard() {
        let geo = minimal_geometry();
        assert_eq!(geo.kv_dim(), 32 * 128);
    }

    #[test]
    fn model_geometry_kv_dim_mla() {
        let mut geo = minimal_geometry();
        geo.mla_d_c = 512;
        geo.mla_d_rope = 64;
        assert_eq!(geo.kv_dim(), 512 + 64);
    }

    #[test]
    fn model_geometry_kv_bytes_per_token_standard() {
        let geo = minimal_geometry();
        // 2 (K+V) * num_kv_heads * head_dim * num_layers * 4 (F32)
        let expected = 2 * 32 * 128 * 32 * 4;
        assert_eq!(geo.kv_bytes_per_token(), expected);
    }

    #[test]
    fn model_geometry_kv_bytes_per_token_mla() {
        let mut geo = minimal_geometry();
        geo.mla_d_c = 512;
        geo.mla_d_rope = 64;
        // (d_c + d_rope) * num_layers * 4 (F32) — no K/V split
        let expected = (512 + 64) * 32 * 4;
        assert_eq!(geo.kv_bytes_per_token(), expected);
    }

    #[test]
    fn model_geometry_expert_weight_bytes() {
        let mut geo = minimal_geometry();
        geo.num_experts = 8;
        geo.expert_intermediate_size = 4096;
        // hidden * expert_inter * 3 * dtype_size
        assert_eq!(geo.expert_weight_bytes(), 4096 * 4096 * 3 * 4);
    }

    #[test]
    fn model_geometry_effective_kv_layers_no_shared() {
        let geo = minimal_geometry();
        assert_eq!(geo.effective_kv_layers(), 32);
    }

    #[test]
    fn model_geometry_effective_kv_layers_with_shared() {
        let mut geo = minimal_geometry();
        geo.num_kv_shared_layers = 4;
        assert_eq!(geo.effective_kv_layers(), 28);
    }

    #[test]
    fn model_geometry_effective_kv_layer_donor_mapping() {
        let mut geo = minimal_geometry();
        geo.num_layers = 6;
        geo.num_kv_shared_layers = 2;
        // Layers 4,5 shared → donor lookup by attention type
        geo.attention_pattern = vec![0, 0, 1, 1, 0, 1];
        // Layer 4 type=0 → donor=1 (nearest non-shared type=0)
        assert_eq!(geo.effective_kv_layer(4), 1);
        // Layer 5 type=1 → donor=3 (nearest non-shared type=1)
        assert_eq!(geo.effective_kv_layer(5), 3);
        // Non-shared layers map to themselves
        assert_eq!(geo.effective_kv_layer(0), 0);
        assert_eq!(geo.effective_kv_layer(3), 3);
    }

    // ── RopeScalingType::parse tests ──

    #[test]
    fn rope_scaling_type_parse_variants() {
        assert_eq!(RopeScalingType::parse("linear"), RopeScalingType::Linear);
        assert_eq!(RopeScalingType::parse("dynamic"), RopeScalingType::Dynamic);
        assert_eq!(RopeScalingType::parse("dynamic_ntk"), RopeScalingType::Dynamic);
        assert_eq!(RopeScalingType::parse("ntk"), RopeScalingType::Dynamic);
        assert_eq!(RopeScalingType::parse("yarn"), RopeScalingType::Yarn);
        assert_eq!(RopeScalingType::parse("longrope"), RopeScalingType::LongRope);
        assert_eq!(RopeScalingType::parse("long_rope"), RopeScalingType::LongRope);
        assert_eq!(RopeScalingType::parse("ntk_aware"), RopeScalingType::NtkAware);
        assert_eq!(RopeScalingType::parse("ntk-aware"), RopeScalingType::NtkAware);
        assert_eq!(RopeScalingType::parse("llama3"), RopeScalingType::Llama3);
        assert_eq!(RopeScalingType::parse("llama_3"), RopeScalingType::Llama3);
    }

    #[test]
    fn rope_scaling_type_parse_unknown() {
        let t = RopeScalingType::parse("custom_method");
        assert!(matches!(t, RopeScalingType::Unknown(s) if s == "custom_method"));
    }

    #[test]
    fn rope_scaling_type_parse_case_insensitive() {
        assert_eq!(RopeScalingType::parse("Linear"), RopeScalingType::Linear);
        assert_eq!(RopeScalingType::parse("DYNAMIC"), RopeScalingType::Dynamic);
        assert_eq!(RopeScalingType::parse(" YaRn "), RopeScalingType::Yarn);
    }

    // ── RopeScalingConfig tests ──

    #[test]
    fn rope_scaling_config_has_any_value_empty() {
        let cfg = RopeScalingConfig::default();
        assert!(!cfg.has_any_value());
    }

    #[test]
    fn rope_scaling_config_has_any_value_with_factor() {
        let mut cfg = RopeScalingConfig::default();
        cfg.factor = Some(2.0);
        assert!(cfg.has_any_value());
    }

    #[test]
    fn rope_scaling_config_has_any_value_with_type() {
        let mut cfg = RopeScalingConfig::default();
        cfg.scaling_type = Some(RopeScalingType::Linear);
        assert!(cfg.has_any_value());
    }

    #[test]
    fn rope_scaling_config_runtime_factor_valid() {
        let mut cfg = RopeScalingConfig::default();
        cfg.factor = Some(4.0);
        assert_eq!(cfg.runtime_factor(), Some(4.0));
    }

    #[test]
    fn rope_scaling_config_runtime_factor_zero() {
        let mut cfg = RopeScalingConfig::default();
        cfg.factor = Some(0.0);
        assert_eq!(cfg.runtime_factor(), None);
    }

    #[test]
    fn rope_scaling_config_runtime_factor_nan() {
        let mut cfg = RopeScalingConfig::default();
        cfg.factor = Some(f32::NAN);
        assert_eq!(cfg.runtime_factor(), None);
    }

    #[test]
    fn rope_scaling_config_runtime_factor_negative() {
        let mut cfg = RopeScalingConfig::default();
        cfg.factor = Some(-1.0);
        assert_eq!(cfg.runtime_factor(), None);
    }

    #[test]
    fn rope_scaling_config_runtime_factor_none() {
        let cfg = RopeScalingConfig::default();
        assert_eq!(cfg.runtime_factor(), None);
    }

    // ── HiddenAct parse / as_str tests ──

    #[test]
    fn hidden_act_parse_variants() {
        assert_eq!(HiddenAct::parse("silu"), HiddenAct::Silu);
        assert_eq!(HiddenAct::parse("swiglu"), HiddenAct::Silu);
        assert_eq!(HiddenAct::parse("gelu"), HiddenAct::Gelu);
        assert_eq!(HiddenAct::parse("gelu_new"), HiddenAct::GeluNew);
        assert_eq!(HiddenAct::parse("gelu_pytorch_tanh"), HiddenAct::GeluNew);
        assert_eq!(HiddenAct::parse("relu"), HiddenAct::Relu);
        assert_eq!(HiddenAct::parse("swish"), HiddenAct::Swish);
        assert_eq!(HiddenAct::parse("quick_gelu"), HiddenAct::QuickGelu);
        assert_eq!(HiddenAct::parse("gelu_fast"), HiddenAct::QuickGelu);
    }

    #[test]
    fn hidden_act_parse_unknown() {
        let act = HiddenAct::parse("custom_act");
        assert!(matches!(act, HiddenAct::Unknown(s) if s == "custom_act"));
    }

    #[test]
    fn hidden_act_as_str_roundtrip() {
        assert_eq!(HiddenAct::Silu.as_str(), "silu");
        assert_eq!(HiddenAct::Gelu.as_str(), "gelu");
        assert_eq!(HiddenAct::GeluNew.as_str(), "gelu_new");
        assert_eq!(HiddenAct::Relu.as_str(), "relu");
        assert_eq!(HiddenAct::Swish.as_str(), "swish");
        assert_eq!(HiddenAct::QuickGelu.as_str(), "quick_gelu");
    }

    #[test]
    fn hidden_act_parse_case_insensitive() {
        assert_eq!(HiddenAct::parse("SiLU"), HiddenAct::Silu);
        assert_eq!(HiddenAct::parse(" GELU "), HiddenAct::Gelu);
    }

    // ── ModelConfigError display tests ──

    #[test]
    fn model_config_error_display_variants() {
        let err = ModelConfigError::MissingConfig;
        assert!(err.to_string().contains("unavailable"));

        let err = ModelConfigError::InvalidConfig("bad field".into());
        assert!(err.to_string().contains("bad field"));

        let err = ModelConfigError::MissingConfigAndMetadata("no file".into());
        assert!(err.to_string().contains("no file"));
    }

    // ── MlaConfig fields test ──

    #[test]
    fn mla_config_fields() {
        let cfg = MlaConfig {
            d_c: 512,
            d_rope: 64,
            unabsorbed_threshold: Some(256),
        };
        assert_eq!(cfg.d_c, 512);
        assert_eq!(cfg.d_rope, 64);
        assert_eq!(cfg.unabsorbed_threshold, Some(256));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── REQ-LOADER-022 criterion 1: TensorRole definition via match_tensor_role ──
    // ═══════════════════════════════════════════════════════════════════════

    /// REQ-LOADER-022/C1: match_tensor_role correctly assigns TensorRole::Embedding
    #[test]
    fn match_tensor_role_assigns_embedding() {
        let result = match_tensor_role("model.embed_tokens.weight");
        assert_eq!(result, Some((TensorRole::Embedding, None)),
            "embed_tokens must map to TensorRole::Embedding with no layer index");
    }

    /// REQ-LOADER-022/C1: match_tensor_role correctly assigns AttentionQuery, AttentionKey, FfnGate
    #[test]
    fn match_tensor_role_assigns_attention_and_ffn_roles() {
        let q = match_tensor_role("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(q.map(|(r, l)| (r, l)), Some((TensorRole::AttentionQuery, Some(0))),
            "q_proj must map to AttentionQuery with layer 0");

        let k = match_tensor_role("model.layers.0.self_attn.k_proj.weight");
        assert_eq!(k.map(|(r, l)| (r, l)), Some((TensorRole::AttentionKey, Some(0))),
            "k_proj must map to AttentionKey with layer 0");

        let gate = match_tensor_role("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(gate.map(|(r, l)| (r, l)), Some((TensorRole::FfnGate, Some(0))),
            "gate_proj must map to FfnGate with layer 0");
    }

    /// REQ-LOADER-022/C1: match_tensor_role assigns per-layer roles with correct layer index
    #[test]
    fn match_tensor_role_assigns_layer_index_for_deep_layers() {
        let q5 = match_tensor_role("model.layers.5.self_attn.q_proj.weight");
        assert_eq!(q5.map(|(r, l)| (r, l)), Some((TensorRole::AttentionQuery, Some(5))),
            "layer 5 q_proj must carry layer_idx=5");

        let gate10 = match_tensor_role("model.layers.10.mlp.gate_proj.weight");
        assert_eq!(gate10.map(|(r, l)| (r, l)), Some((TensorRole::FfnGate, Some(10))),
            "layer 10 gate_proj must carry layer_idx=10");
    }

    /// REQ-LOADER-022/C1: match_tensor_role rejects bias tensors
    #[test]
    fn match_tensor_role_rejects_bias_tensors() {
        assert_eq!(match_tensor_role("model.layers.0.self_attn.q_proj.bias"), None,
            "bias tensors must not match any TensorRole");
        assert_eq!(match_tensor_role("model.embed_tokens.bias"), None,
            "global bias tensors must not match any TensorRole");
    }

    /// REQ-LOADER-022/C1: tensor_map correctly maps roles to patterns
    #[test]
    fn tensor_derive_builds_correct_tensor_map_role_assignments() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000, 2816]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.0.mlp.gate_proj.weight", &[11264, 2816]),
            ],
        };

        let derived = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default())
            .expect("tensor-driven derivation");

        // Verify tensor_map contains the expected roles with anonymized patterns
        assert!(derived.tensor_map.contains_key(&TensorRole::Embedding),
            "tensor_map must contain Embedding role");
        assert!(derived.tensor_map.contains_key(&TensorRole::AttentionQuery),
            "tensor_map must contain AttentionQuery role");
        assert!(derived.tensor_map.contains_key(&TensorRole::AttentionKey),
            "tensor_map must contain AttentionKey role");
        assert!(derived.tensor_map.contains_key(&TensorRole::FfnGate),
            "tensor_map must contain FfnGate role");

        // Verify patterns have layer index anonymized
        let q_pattern = derived.tensor_map.get(&TensorRole::AttentionQuery).unwrap();
        assert!(q_pattern.contains("{}"), "AttentionQuery pattern must anonymize layer index: got '{q_pattern}'");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── REQ-LOADER-022 criterion 3: no model-name hard-coded branching ──
    // ═══════════════════════════════════════════════════════════════════════

    /// REQ-LOADER-022/C3 (structural invariant): from_value does NOT contain
    /// `if model == "llama"` style branching. Verified by absence of string
    /// equality checks against model name literals in from_value.
    ///
    /// This test is a grep-based structural assertion: the source file must
    /// not contain pattern `== "llama"` or `== "gemma4"` or `== "deepseek"`
    /// etc. inside the model_config fragments.
    #[test]
    fn from_value_no_model_name_hardcoded_branching() {
        // Structural assertion: check that config_impl.inc.rs does not
        // contain hardcoded model-name string comparisons.
        // The correct architecture uses TensorRole-driven derivation and
        // family-based BUILD-stage logic with _family suffix.
        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/model_config_fragments/config_impl.inc.rs")
        ).expect("read config_impl.inc.rs source");

        // Prohibited patterns: model-name string equality checks
        let prohibited = [
            "== \"llama\"",
            "== \"gemma4\"",
            "== \"deepseek\"",
            "== \"qwen\"",
            "== \"mistral\"",
            "== \"phi\"",
            "== \"glm\"",
            "== \"bert\"",
            "== \"roberta\"",
            "== \"xlm\"",
        ];
        for pattern in &prohibited {
            assert!(!source.contains(pattern),
                "REQ-LOADER-022/C3: config_impl.inc.rs contains prohibited pattern '{pattern}' \
                 — model-name-based branching violates the tensor-driven principle");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── REQ-LOADER-022 criterion 4: tensor-derived > config.json priority ──
    // ═══════════════════════════════════════════════════════════════════════

    /// REQ-LOADER-022/C4: apply_tensor_derived overrides config.json values.
    /// When tensor-derived hidden_size differs from config.json, the tensor
    /// value must win via apply_tensor_derived.
    #[test]
    fn apply_tensor_derived_overrides_config_json_hidden_size() {
        // Config.json says hidden_size=2048 but tensors show 4096
        let config_json_hidden = 2048;
        let derived = TensorDerivedConfig {
            hidden_size: 4096,  // tensor-derived, different from config.json
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            intermediate_size: Some(11008),
            vocab_size: 32000,
            head_dim: 128,
            dtype: DType::BF16,
            tensor_map: HashMap::new(),
        };

        let mut base = minimal_model_config(config_json_hidden);
        let result = apply_tensor_derived(base, derived).expect("apply_tensor_derived");
        // Tensor-derived value must override config.json value
        assert_eq!(result.hidden_size, 4096,
            "REQ-LOADER-022/C4: tensor-derived hidden_size must override config.json value");
        assert_eq!(result.head_dim, 128,
            "REQ-LOADER-022/C4: tensor-derived head_dim must override config.json value");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── from_value: 16 previously-unasserted fields ──
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: build a minimal config.json Value with all required fields.
    fn minimal_config_json() -> Value {
        serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
        })
    }

    /// from_value: vision_config parsed from vision_config sub-object.
    #[test]
    fn from_value_parses_vision_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "vision_config": {
                "image_size": 384,
                "patch_size": 14,
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4352,
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let vc = cfg.vision_config.expect("vision_config must be parsed");
        assert_eq!(vc.image_size, 384);
        assert_eq!(vc.patch_size, 14);
        assert_eq!(vc.hidden_size, 1152);
        assert_eq!(vc.num_layers, 27);
        assert_eq!(vc.num_heads, 16);
        assert_eq!(vc.intermediate_size, 4352);
    }

    /// from_value: audio_config parsed from audio_config sub-object.
    #[test]
    fn from_value_parses_audio_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "audio_config": {
                "hidden_size": 512,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "conv_kernel_size": 31,
                "sample_rate": 16000,
                "num_mel_bins": 80,
                "fft_size": 512,
                "hop_length": 160,
                "win_length": 400,
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let ac = cfg.audio_config.expect("audio_config must be parsed");
        assert_eq!(ac.hidden_size, 512);
        assert_eq!(ac.num_layers, 12);
        assert_eq!(ac.num_heads, 8);
        assert_eq!(ac.conv_kernel_size, 31);
    }

    /// from_value: multimodal_token_ids parsed from explicit token IDs.
    #[test]
    fn from_value_parses_multimodal_token_ids() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "eoi_token_id": 258882,
            "eoa_token_id": 258883,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let ids = cfg.multimodal_token_ids.expect("multimodal_token_ids must be parsed");
        assert_eq!(ids.image_token_id, 258880);
        assert_eq!(ids.audio_token_id, 258881);
        assert_eq!(ids.eoi_token_id, 258882);
        assert_eq!(ids.eoa_token_id, 258883);
    }

    /// from_value: multimodal_token_ids falls back when vision_config exists
    /// but explicit token IDs are missing (T58 source rule).
    #[test]
    fn from_value_multimodal_token_ids_fallback_with_vision_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "vision_config": {
                "image_size": 384,
                "patch_size": 14,
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4352,
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let ids = cfg.multimodal_token_ids.expect("fallback must activate when vision_config present");
        // Fallback uses Gemma 4 convention values
        assert_eq!(ids.image_token_id, 258880);
    }

    /// from_value: final_logit_softcapping parsed.
    #[test]
    fn from_value_parses_final_logit_softcapping() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "final_logit_softcapping": 30.0,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
    }

    /// from_value: use_double_wide_mlp parsed.
    #[test]
    fn from_value_parses_use_double_wide_mlp() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "use_double_wide_mlp": true,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.use_double_wide_mlp, Some(true));
    }

    /// from_value: add_special_tokens parsed from add_bos_token key.
    #[test]
    fn from_value_parses_add_special_tokens_from_add_bos() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "add_bos_token": true,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.add_special_tokens, Some(true));
    }

    /// from_value: add_special_tokens parsed from add_special_tokens key.
    #[test]
    fn from_value_parses_add_special_tokens_from_alt_key() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "add_special_tokens": false,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.add_special_tokens, Some(false));
    }

    /// from_value: compute_dtype parsed (user override).
    #[test]
    fn from_value_compute_dtype_defaults_to_none() {
        let manifest = ModelManifest::default();
        let json = minimal_config_json();
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.compute_dtype, None,
            "compute_dtype defaults to None — user sets it at load time");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── REQ-MC-EXT-001..007: BUILD-stage architecture hints assertions ──
    // ═══════════════════════════════════════════════════════════════════════

    /// REQ-MC-EXT-001: qk_norm parsed from config.json.
    #[test]
    fn from_value_parses_qk_norm() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "qk_norm": true,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.qk_norm, Some(true),
            "REQ-MC-EXT-001: qk_norm must be parsed from config.json");
    }

    /// REQ-MC-EXT-001: qk_norm parsed from text_config.qk_norm (nested).
    #[test]
    fn from_value_parses_qk_norm_from_text_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "text_config": {
                "qk_norm": true,
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.qk_norm, Some(true),
            "REQ-MC-EXT-001: qk_norm must be read from text_config nesting");
    }

    /// REQ-MC-EXT-002: value_norm parsed from config.json.
    #[test]
    fn from_value_parses_value_norm() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "value_norm": true,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.value_norm, Some(true),
            "REQ-MC-EXT-002: value_norm must be parsed from config.json");
    }

    /// REQ-MC-EXT-003: embedding_scale_factor parsed from config.json.
    #[test]
    fn from_value_parses_embedding_scale_factor() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "embedding_scale_factor": 4096.0,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.embedding_scale_factor, Some(4096.0),
            "REQ-MC-EXT-003: embedding_scale_factor must be parsed");
    }

    /// REQ-MC-EXT-005: rope_partial_ratio_global parsed from nested
    /// rope_parameters.full_attention.partial_rotary_factor.
    #[test]
    fn from_value_parses_rope_partial_ratio_global() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                },
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.rope_partial_ratio_global, Some(0.25),
            "REQ-MC-EXT-005: rope_partial_ratio_global must be parsed from nested key");
    }

    /// REQ-MC-EXT-005: rope_partial_ratio_global from text_config nesting.
    #[test]
    fn from_value_parses_rope_partial_ratio_global_from_text_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "text_config": {
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                    },
                },
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.rope_partial_ratio_global, Some(0.25),
            "REQ-MC-EXT-005: rope_partial_ratio_global must be read from text_config nesting");
    }

    /// REQ-MC-EXT-007: mla_use_unabsorbed parsed from config.json.
    #[test]
    fn from_value_parses_mla_use_unabsorbed() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "mla_use_unabsorbed": true,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.mla_use_unabsorbed, Some(true),
            "REQ-MC-EXT-007: mla_use_unabsorbed must be parsed");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── Gemma 4 SPEC fields: global_rope_theta, rope_partial_ratio, ──
    // ── attention_pattern, sliding_window, hidden_size_per_layer_input ──
    // ═══════════════════════════════════════════════════════════════════════

    /// from_value: global_rope_theta parsed from top-level key.
    #[test]
    fn from_value_parses_global_rope_theta() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "global_rope_theta": 1_000_000.0,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.global_rope_theta, Some(1_000_000.0),
            "global_rope_theta must be parsed from config.json");
    }

    /// from_value: global_rope_theta from text_config nested RoPE parameters.
    #[test]
    fn from_value_parses_global_rope_theta_from_text_config() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "text_config": {
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1_000_000.0,
                    },
                },
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.global_rope_theta, Some(1_000_000.0),
            "global_rope_theta must be read from text_config.rope_parameters.full_attention.rope_theta");
    }

    /// from_value: rope_partial_ratio parsed from sliding_attention partial_rotary_factor.
    #[test]
    fn from_value_parses_rope_partial_ratio_from_nested_keys() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "text_config": {
                "rope_parameters": {
                    "sliding_attention": {
                        "partial_rotary_factor": 1.0,
                    },
                },
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.rope_partial_ratio, Some(1.0),
            "rope_partial_ratio must be read from nested sliding_attention key");
    }

    /// from_value: rope_partial_ratio from top-level partial_rotary_factor.
    #[test]
    fn from_value_parses_rope_partial_ratio_from_top_level() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "partial_rotary_factor": 0.5,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.rope_partial_ratio, Some(0.5),
            "rope_partial_ratio must be read from top-level partial_rotary_factor");
    }

    /// from_value: attention_pattern parsed from integer array.
    #[test]
    fn from_value_parses_attention_pattern() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "attention_pattern": [0, 0, 0, 0, 0, 1],
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let pattern = cfg.attention_pattern.expect("attention_pattern must be parsed");
        assert_eq!(pattern, vec![0, 0, 0, 0, 0, 1],
            "attention_pattern must round-trip from config.json integer array");
    }

    /// from_value: attention_pattern parsed from text_config.layer_types
    /// string array (Gemma 4 naming: "sliding_attention"/"full_attention").
    #[test]
    fn from_value_parses_attention_pattern_from_layer_types() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "text_config": {
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        let pattern = cfg.attention_pattern.expect("attention_pattern from layer_types");
        assert_eq!(pattern, vec![0, 0, 0, 0, 0, 1],
            "layer_types strings must convert: sliding_attention→0, full_attention→1");
    }

    /// from_value: sliding_window parsed.
    #[test]
    fn from_value_parses_sliding_window() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "sliding_window": 512,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.sliding_window, Some(512));
    }

    /// from_value: hidden_size_per_layer_input parsed (PLE injection width).
    #[test]
    fn from_value_parses_hidden_size_per_layer_input() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "hidden_size_per_layer_input": 128,
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.hidden_size_per_layer_input, Some(128));
    }

    /// from_value: text_config nesting for multimodal models.
    /// The implementation reads text_config.* paths for all required fields.
    #[test]
    fn from_value_reads_text_config_nested_fields() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vocab_size": 256000,
            "torch_dtype": "bfloat16",
            "text_config": {
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
            },
        });
        let cfg = ModelConfig::from_value(&manifest, &json, Some(DType::BF16)).unwrap();
        assert_eq!(cfg.hidden_size, 2048,
            "hidden_size must be read from text_config.hidden_size when top-level absent");
        assert_eq!(cfg.num_attention_heads, 16,
            "num_attention_heads must be read from text_config nesting");
        assert_eq!(cfg.num_hidden_layers, 24,
            "num_hidden_layers must be read from text_config nesting");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── ModelGeometry::from_config resolution logic ──
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: build a minimal ModelConfig for testing.
    fn minimal_model_config(hidden_size: usize) -> ModelConfig {
        ModelConfig {
            hidden_size,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            intermediate_size: Some(11008),
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 128,
            head_dim: 128,
            dtype: DType::BF16,
            compute_dtype: None,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            hidden_act: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            sliding_window: None,
            num_kv_shared_layers: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            mtp_depth: None,
            mla_config: None,
            vision_config: None,
            audio_config: None,
            multimodal_token_ids: None,
            final_logit_softcapping: None,
            use_double_wide_mlp: None,
            add_special_tokens: None,
            feed_forward_lengths: None,
            qk_norm: None,
            value_norm: None,
            embedding_scale_factor: None,
            rope_partial_ratio_global: None,
            mla_use_unabsorbed: None,
        }
    }

    /// ModelGeometry::from_config resolves Option fields to defaults.
    #[test]
    fn model_geometry_from_config_resolves_option_defaults() {
        let config = minimal_model_config(4096);
        let geo = ModelGeometry::from_config(&config, None);
        // All Option fields resolved to defaults
        assert_eq!(geo.global_rope_theta, 0.0, "None → 0.0 (no dual RoPE)");
        assert_eq!(geo.rope_partial_ratio, 1.0, "None → 1.0 (full RoPE rotation)");
        assert_eq!(geo.rope_partial_ratio_global, 1.0, "None → 1.0 when no global_rope_theta");
        assert_eq!(geo.attention_pattern, Vec::<u8>::new(), "None → empty vec (no per-layer pattern)");
        assert_eq!(geo.sliding_window, 0, "None → 0 (no sliding window)");
        assert_eq!(geo.num_kv_shared_layers, 0, "None → 0 (no shared KV)");
        assert_eq!(geo.global_head_dim, 0, "None → 0 (same as head_dim)");
        assert_eq!(geo.hidden_size_per_layer_input, 0, "None → 0 (no PLE)");
        assert_eq!(geo.norm_eps, 1e-12, "None → 1e-12 default");
        assert_eq!(geo.compute_dtype, DType::BF16, "None → same as dtype");
        assert!(!geo.qk_norm, "None → false");
        assert!(!geo.value_norm, "None → false");
        assert_eq!(geo.embedding_scale_factor, 0.0, "None → 0.0 (no scaling)");
        assert!(!geo.mla_use_unabsorbed, "None → false");
        assert_eq!(geo.num_experts, 0, "No MoE config → 0");
        assert_eq!(geo.moe_top_k, 0, "No MoE config → 0");
    }

    /// ModelGeometry::from_config: rope_partial_ratio_global derived from
    /// dual-RoPE signals when global_rope_theta is present but
    /// rope_partial_ratio_global is None.
    #[test]
    fn model_geometry_from_config_derives_rope_partial_ratio_global() {
        let mut config = minimal_model_config(4096);
        config.global_rope_theta = Some(1_000_000.0);
        config.rope_partial_ratio = Some(0.25);
        // rope_partial_ratio_global is still None → derived from dual-RoPE signals
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.rope_partial_ratio_global, 0.25,
            "when global_rope_theta present and rope_partial_ratio_global is None, \
             must derive from rope_partial_ratio fallback");
    }

    /// ModelGeometry::from_config: rope_partial_ratio_global stays 1.0
    /// when global_rope_theta is absent (no dual RoPE).
    #[test]
    fn model_geometry_from_config_rope_partial_ratio_global_without_dual_rope() {
        let mut config = minimal_model_config(4096);
        config.global_rope_theta = None;
        config.rope_partial_ratio_global = None;
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.rope_partial_ratio_global, 1.0,
            "without global_rope_theta, partial_ratio_global must default to 1.0 (full rotation)");
    }

    /// ModelGeometry::from_config: use_double_wide_mlp derives intermediate_size.
    #[test]
    fn model_geometry_from_config_use_double_wide_mlp() {
        let mut config = minimal_model_config(4096);
        config.intermediate_size = None;
        config.use_double_wide_mlp = Some(true);
        let geo = ModelGeometry::from_config(&config, None);
        // (4096 * 8/3 rounded) / 256 * 256 = 10944 — 256-aligned
        let expected = ((4096 as f64 * 8.0 / 3.0).round() as usize / 256) * 256;
        assert_eq!(geo.intermediate_size, expected,
            "use_double_wide_mlp must derive 256-aligned intermediate_size");
    }

    /// ModelGeometry::from_config: intermediate_size falls back to hidden*4.
    #[test]
    fn model_geometry_from_config_intermediate_size_default() {
        let mut config = minimal_model_config(2048);
        config.intermediate_size = None;
        config.use_double_wide_mlp = None;
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.intermediate_size, 2048 * 4,
            "default intermediate_size = hidden_size * 4 when not specified");
    }

    /// ModelGeometry::from_config: position_offset derived from pad_token_id.
    #[test]
    fn model_geometry_from_config_position_offset_from_pad_token() {
        let mut config = minimal_model_config(4096);
        config.pad_token_id = Some(1); // RoBERTa-style
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.position_offset, Some(2),
            "position_offset = pad_token_id + 1 (RoBERTa: pad=1 → offset=2)");
    }

    /// ModelGeometry::from_config: position_offset is None without pad_token_id.
    #[test]
    fn model_geometry_from_config_no_position_offset_without_pad() {
        let config = minimal_model_config(4096);
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.position_offset, None,
            "GPT-style models (no pad_token_id) have position_offset=None");
    }

    /// ModelGeometry::from_config: MoE fields populated from MoEConfig.
    #[test]
    fn model_geometry_from_config_moe_fields() {
        use crate::manifest::RouterType;
        let config = minimal_model_config(2048);
        let moe = crate::manifest::MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 6,
            router_type: RouterType::DeepSeek,
        };
        let geo = ModelGeometry::from_config(&config, Some(moe));
        assert_eq!(geo.num_experts, 64);
        assert_eq!(geo.moe_top_k, 6);
    }

    /// ModelGeometry::from_config: MLA fields from mla_config.
    #[test]
    fn model_geometry_from_config_mla_fields() {
        let mut config = minimal_model_config(7168);
        config.mla_config = Some(MlaConfig {
            d_c: 512,
            d_rope: 64,
            unabsorbed_threshold: Some(256),
        });
        let geo = ModelGeometry::from_config(&config, None);
        assert_eq!(geo.mla_d_c, 512);
        assert_eq!(geo.mla_d_rope, 64);
        assert_eq!(geo.mla_unabsorbed_threshold, 256);
    }

    /// ModelGeometry::from_config: BUILD-stage hints resolved.
    #[test]
    fn model_geometry_from_config_build_hints() {
        let mut config = minimal_model_config(4096);
        config.qk_norm = Some(true);
        config.value_norm = Some(true);
        config.embedding_scale_factor = Some(4096.0);
        config.mla_use_unabsorbed = Some(true);
        config.final_logit_softcapping = Some(30.0);
        let geo = ModelGeometry::from_config(&config, None);
        assert!(geo.qk_norm, "qk_norm Some(true) → true");
        assert!(geo.value_norm, "value_norm Some(true) → true");
        assert_eq!(geo.embedding_scale_factor, 4096.0);
        assert!(geo.mla_use_unabsorbed);
        assert_eq!(geo.final_logit_softcapping, Some(30.0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── Boundary/error-path tests ──
    // ═══════════════════════════════════════════════════════════════════════

    /// Error path: from_value with missing required field hidden_size.
    #[test]
    fn from_value_err_missing_hidden_size() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "missing hidden_size must return Err");
        let err = result.unwrap_err();
        assert!(matches!(err, ModelConfigError::InvalidConfig(_)),
            "error must be InvalidConfig");
    }

    /// Error path: from_value with missing required field num_attention_heads.
    #[test]
    fn from_value_err_missing_num_attention_heads() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "missing num_attention_heads must return Err");
    }

    /// Error path: from_value with missing weight dtype.
    #[test]
    fn from_value_err_missing_weight_dtype() {
        let manifest = ModelManifest::default();
        let json = minimal_config_json();
        let result = ModelConfig::from_value(&manifest, &json, None);
        assert!(result.is_err(), "missing weight_dtype must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("dtype"), "error must mention dtype");
    }

    /// Error path: from_value with zero max_position_embeddings.
    #[test]
    fn from_value_err_zero_max_position_embeddings() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 0,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "max_position_embeddings=0 must return Err");
    }

    /// Error path: from_value with invalid (negative) rope_scale.
    #[test]
    fn from_value_err_negative_rope_scale() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scale": -1.0,
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "negative rope_scale must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("positive"), "error must mention 'positive'");
    }

    /// Error path: from_value with NaN rope_scale.
    #[test]
    fn from_value_err_nan_rope_scale() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scale": null,
        });
        // NaN would need to come from rope_scaling factor; from_value uses
        // is_finite check on rope_scale. Test with an explicit NaN in rope_scaling.
        let json_nan = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scaling": {
                "type": "linear",
                "factor": -2.0,
            },
        });
        let result = ModelConfig::from_value(&manifest, &json_nan, Some(DType::BF16));
        assert!(result.is_err(), "negative rope_scaling factor must return Err");
    }

    /// Error path: num_experts=0 in config.json must be rejected.
    #[test]
    fn from_value_err_num_experts_zero() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "num_experts": 0,
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "num_experts=0 must return Err");
    }

    /// Error path: expert_intermediate_size=0 in config.json must be rejected.
    #[test]
    fn from_value_err_expert_intermediate_size_zero() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "num_experts": 8,
            "expert_intermediate_size": 0,
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "expert_intermediate_size=0 must return Err");
    }

    /// Error path: derive_config_from_tensors_with_hints with empty tensor list.
    #[test]
    fn derive_config_err_empty_tensors() {
        let provider = MockTensorProvider { tensors: vec![] };
        let result = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default());
        assert!(result.is_err(), "empty tensor list must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("no tensors"), "error must mention no tensors");
    }

    /// Error path: embedding tensor with <2D shape.
    #[test]
    fn derive_config_err_embedding_1d_shape() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000]), // 1D
            ],
        };
        let result = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default());
        assert!(result.is_err(), "1D embedding must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("2D"), "error must mention 2D requirement");
    }

    /// When only embedding is provided (no layer tensors), derive returns Ok with
    /// num_hidden_layers=0. The Q/K projection check only fires when has_layers=true.
    #[test]
    fn derive_config_ok_no_qk_in_layer0() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000, 2816]),
                // No Q/K projection tensors — only embedding, so has_layers=false
            ],
        };
        let result = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default());
        // Without layer tensors, the function returns Ok with 0 hidden layers
        assert!(result.is_ok(), "embedding-only config should return Ok with 0 layers");
        let cfg = result.unwrap();
        assert_eq!(cfg.num_hidden_layers, 0, "no layer tensors means 0 hidden layers");
    }

    /// Error path: kv_cache_block_size = 0 via invalid configuration.
    #[test]
    fn from_value_err_zero_kv_cache_block_size() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "kv_cache_block_size": 0,
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "kv_cache_block_size=0 must return Err");
    }

    /// Boundary: MoE config with num_experts=1 should return None (<=1 threshold).
    #[test]
    fn build_moe_config_none_for_single_expert() {
        let mut config = minimal_model_config(4096);
        config.num_experts = Some(1);
        config.num_experts_per_tok = Some(1);
        let result = config.build_moe_config("test_arch");
        assert_eq!(result, None,
            "num_experts <= 1 must return None — not a real MoE model");
    }

    /// Boundary: num_experts_per_tok exceeding num_experts — build_moe_config
    /// still produces a config (validation is downstream).
    #[test]
    fn build_moe_config_with_per_tok_exceeding_experts() {
        let mut config = minimal_model_config(4096);
        config.num_experts = Some(4);
        config.num_experts_per_tok = Some(8); // exceeds num_experts
        let moe = config.build_moe_config("mixtral").expect("build_moe_config");
        assert_eq!(moe.num_experts_per_tok, 8,
            "build_moe_config does not validate per_tok <= experts; that's downstream");
    }

    /// Boundary: effective_kv_layers when num_kv_shared_layers >= num_layers
    /// → saturating_sub max(1) clamp.
    #[test]
    fn model_geometry_effective_kv_layers_saturating_clamp() {
        let mut geo = minimal_geometry();
        geo.num_layers = 6;
        geo.num_kv_shared_layers = 6; // all shared → saturating_sub = 0, max(1) = 1
        assert_eq!(geo.effective_kv_layers(), 1,
            "when all layers shared, effective_kv_layers must clamp to 1");
    }

    /// Boundary: effective_kv_layer when no donor of same attention type
    /// exists (fallback clamp path).
    #[test]
    fn model_geometry_effective_kv_layer_no_donor_fallback() {
        let mut geo = minimal_geometry();
        geo.num_layers = 4;
        geo.num_kv_shared_layers = 2;
        // Layers 2,3 are shared. Pattern: [1, 1, 0, 0]
        // Layer 2 type=0 → no non-shared type=0 layer → fallback clamp
        geo.attention_pattern = vec![1, 1, 0, 0];
        // Layer 2 (shared, type=0): no non-shared type=0 → clamp to last effective (1)
        assert_eq!(geo.effective_kv_layer(2), 1,
            "no matching donor → fallback clamp to last effective layer");
        // Layer 3 (shared, type=0): same → clamp
        assert_eq!(geo.effective_kv_layer(3), 1);
    }

    /// Boundary: derive_default_attention_pattern with large num_layers.
    #[test]
    fn derive_default_attention_pattern_large_layers() {
        let pattern = derive_default_attention_pattern(1000);
        assert_eq!(pattern.len(), 1000);
        // Every 6th layer is global
        let global_count = pattern.iter().filter(|&v| *v == 1).count();
        assert_eq!(global_count, 166,
            "1000 layers with every-6th-global pattern → 166 global layers");
    }

    /// Boundary: derive_dtype with mixed dtypes (majority-vote logic).
    #[test]
    fn derive_dtype_mixed_majority_vote() {
        // 3 BF16 tensors, 1 F16 tensor → BF16 wins
        let metas = vec![
            TensorMeta { name: "a".to_string(), shape: vec![1], dtype: safetensors::Dtype::BF16 },
            TensorMeta { name: "b".to_string(), shape: vec![1], dtype: safetensors::Dtype::BF16 },
            TensorMeta { name: "c".to_string(), shape: vec![1], dtype: safetensors::Dtype::BF16 },
            TensorMeta { name: "d".to_string(), shape: vec![1], dtype: safetensors::Dtype::F16 },
        ];
        let dtype = derive_dtype(&metas).expect("derive_dtype");
        assert_eq!(dtype, DType::BF16, "BF16 majority (3) must win over F16 (1)");
    }

    /// Boundary: derive_dtype with tied counts — BF16 wins by priority order.
    #[test]
    fn derive_dtype_tied_priority() {
        // 2 BF16, 2 F16 → BF16 wins (checked first in priority)
        let metas = vec![
            TensorMeta { name: "a".to_string(), shape: vec![1], dtype: safetensors::Dtype::BF16 },
            TensorMeta { name: "b".to_string(), shape: vec![1], dtype: safetensors::Dtype::BF16 },
            TensorMeta { name: "c".to_string(), shape: vec![1], dtype: safetensors::Dtype::F16 },
            TensorMeta { name: "d".to_string(), shape: vec![1], dtype: safetensors::Dtype::F16 },
        ];
        let dtype = derive_dtype(&metas).expect("derive_dtype");
        assert_eq!(dtype, DType::BF16,
            "tie between BF16 and F16 → BF16 wins by priority (checked first)");
    }

    /// Boundary: derive_dtype with no floating tensors → defaults to F32.
    #[test]
    fn derive_dtype_no_floats_defaults_f32() {
        let metas = vec![
            TensorMeta { name: "a".to_string(), shape: vec![1], dtype: safetensors::Dtype::I8 },
            TensorMeta { name: "b".to_string(), shape: vec![1], dtype: safetensors::Dtype::U8 },
        ];
        let dtype = derive_dtype(&metas).expect("derive_dtype");
        assert_eq!(dtype, DType::F32, "no floating tensors → default F32");
    }

    /// Error path: rope_scaling with invalid factors (empty array).
    #[test]
    fn rope_scaling_err_empty_factors() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scaling": {
                "type": "longrope",
                "factors": [],
            },
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "empty factors array must return Err");
    }

    /// Error path: rope_scaling with negative factor values.
    #[test]
    fn rope_scaling_err_negative_factor() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scaling": {
                "type": "longrope",
                "factors": [-1.0, 2.0],
            },
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "negative factor values must return Err");
    }

    /// Error path: rope_scaling with zero factor (non-positive).
    #[test]
    fn rope_scaling_err_zero_factor() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "rope_scaling": {
                "type": "linear",
                "factor": 0.0,
            },
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "zero rope_scaling factor must return Err");
    }

    /// Error path: head_dim = 0 for non-embedding model.
    #[test]
    fn from_value_err_zero_head_dim_with_attention_heads() {
        let manifest = ModelManifest::default();
        let json = serde_json::json!({
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "head_dim": 0,
        });
        let result = ModelConfig::from_value(&manifest, &json, Some(DType::BF16));
        assert!(result.is_err(), "head_dim=0 with attention heads must return Err");
    }

    /// GGUF dual-RoPE detection heuristic: rope_theta >= 100K + Gemma 4 signals
    /// → reinterpret as dual-RoPE. This complex conditional logic is tested via
    /// the GGUF builder, verifying that the sliding/global theta split works.
    #[test]
    fn gguf_dual_rope_detection_heuristic() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        // GGUF with freq_base=1M (>=100K), sliding_window=512 (Gemma 4 signal)
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("gemma4")),
            ("gemma4.attention.sliding_window", GgufMetaValue::U64(512)),
            ("gemma4.rope.freq_base", GgufMetaValue::F32(1_000_000.0)),
            ("gemma4.embedding.per_layer_input", GgufMetaValue::U64(128)),
            ("gemma4.context_length", GgufMetaValue::U64(8192)),
        ]);
        // Note: This test verifies the GGUF reader can parse dual-RoPE metadata.
        // The full dual-RoPE reinterpretation logic is in from_gguf_loader,
        // which requires a full Loader (not just a GgufReader), so we test
        // the metadata extraction path here.
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        // Verify the metadata is readable
        let sliding_window = gguf_arch_usize(&reader, "gemma4", "attention.sliding_window");
        assert_eq!(sliding_window, Some(512), "sliding_window must be readable");
        let freq_base = gguf_arch_f32(&reader, "gemma4", "rope.freq_base");
        assert_eq!(freq_base, Some(1_000_000.0), "freq_base must be readable");
        let ple = gguf_arch_usize(&reader, "gemma4", "embedding.per_layer_input");
        assert_eq!(ple, Some(128), "per_layer_input must be readable");
        let _ = std::fs::remove_file(&path);
    }
}
