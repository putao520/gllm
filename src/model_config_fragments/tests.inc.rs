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
            unabsorbed_threshold: 256,
        };
        assert_eq!(cfg.d_c, 512);
        assert_eq!(cfg.d_rope, 64);
        assert_eq!(cfg.unabsorbed_threshold, 256);
    }
}
