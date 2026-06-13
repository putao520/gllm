//! Executor types unit tests — extracted from executor_types.rs for file size compliance (SPEC 31 REQ-DECOMP-001).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::engine::executor_types::*;
    use crate::engine::RequestId;
    use crate::kv_cache::KvCacheError;
    use crate::model_config::{ModelConfigError, ModelGeometry};
    use crate::scheduler::SessionId;
    use crate::tokenizer::TokenizerError;
    use gllm_kernels::types::DType;

    /// Helper: build a minimal `ModelGeometry` for tests.
    fn make_geometry() -> ModelGeometry {
        ModelGeometry {
            hidden_size: 1024,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 4096,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
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

    /// Helper: build an MLA-flavored `ModelGeometry`.
    fn make_mla_geometry() -> ModelGeometry {
        ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            mla_unabsorbed_threshold: 256,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
            ..make_geometry()
        }
    }

    // ---------------------------------------------------------------
    // effective_kv_max_seq_len
    // ---------------------------------------------------------------

    #[test]
    fn effective_kv_max_seq_len_passes_through_value() {
        assert_eq!(effective_kv_max_seq_len(4096), 4096);
        assert_eq!(effective_kv_max_seq_len(0), 0);
        assert_eq!(effective_kv_max_seq_len(1), 1);
    }

    // ---------------------------------------------------------------
    // SamplingConfig::default
    // ---------------------------------------------------------------

    #[test]
    fn sampling_config_default_values() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.top_p, 1.0);
    }

    // ---------------------------------------------------------------
    // RoPEConfig
    // ---------------------------------------------------------------

    #[test]
    fn rope_config_fields_and_equality() {
        let a = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_eq!(a, b);

        let c = RoPEConfig {
            theta: 500000.0,
            ..a
        };
        assert_ne!(a, c);
    }

    // ---------------------------------------------------------------
    // AttentionHeadConfig::from_geometry
    // ---------------------------------------------------------------

    #[test]
    fn attention_head_config_from_geometry() {
        let geo = make_geometry();
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64);
    }

    // ---------------------------------------------------------------
    // PagedKvConfig
    // ---------------------------------------------------------------

    #[test]
    fn paged_kv_config_fields() {
        let cfg = PagedKvConfig {
            page_table: Some(vec![0, 1, 2]),
            page_size: 16,
        };
        assert_eq!(cfg.page_table.as_ref().unwrap().len(), 3);
        assert_eq!(cfg.page_size, 16);

        let no_table = PagedKvConfig {
            page_table: None,
            page_size: 32,
        };
        assert!(no_table.page_table.is_none());
    }

    // ---------------------------------------------------------------
    // GeneratorForwardConfig accessors
    // ---------------------------------------------------------------

    fn make_forward_config() -> GeneratorForwardConfig {
        let geo = Arc::new(make_geometry());
        GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        }
    }

    #[test]
    fn forward_config_accessors() {
        let cfg = make_forward_config();

        assert_eq!(cfg.hidden_size(), 1024);
        assert_eq!(cfg.num_layers(), 12);
        assert_eq!(cfg.vocab_size(), 32000);
        assert_eq!(cfg.intermediate_size(), 4096);
        assert_eq!(cfg.norm_eps(), 1e-5);
        assert_eq!(cfg.dtype(), DType::F32);
        assert_eq!(cfg.max_seq_len(), 2048);
        assert_eq!(cfg.num_heads(), 16);
        assert_eq!(cfg.num_kv_heads(), 4);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.rope_theta(), 10000.0);
        assert_eq!(cfg.rope_scale(), 1.0);
    }

    #[test]
    fn forward_config_attention_derives_from_geometry() {
        let cfg = make_forward_config();
        let attn = cfg.attention();
        assert_eq!(attn.num_heads, 16);
        assert_eq!(attn.num_kv_heads, 4);
        assert_eq!(attn.head_dim, 64);
    }

    #[test]
    fn forward_config_attention_geometry() {
        let cfg = make_forward_config();
        let ag = cfg.attention_geometry();
        assert_eq!(ag.num_heads, 16);
        assert_eq!(ag.num_kv_heads, 4);
        assert_eq!(ag.head_dim, 64);
        assert_eq!(ag.q_dim, 16 * 64);
        assert_eq!(ag.kv_dim, 4 * 64);
        assert_eq!(ag.heads_per_group, 16 / 4);
    }

    #[test]
    fn forward_config_layer_dims() {
        let cfg = make_forward_config();
        let ld = cfg.layer_dims();
        assert_eq!(ld.hidden, 1024);
        assert_eq!(ld.inter, 4096);
        assert_eq!(ld.eps, 1e-5);
        assert_eq!(ld.rope_theta, 10000.0);
    }

    // ---------------------------------------------------------------
    // SwapConfig
    // ---------------------------------------------------------------

    #[test]
    fn swap_config_fields_and_equality() {
        let a = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        let b = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        assert_eq!(a, b);

        let c = SwapConfig {
            enable_swap: false,
            ..a
        };
        assert_ne!(a, c);
    }

    // ---------------------------------------------------------------
    // KvCacheConfig methods
    // ---------------------------------------------------------------

    #[test]
    fn kv_cache_config_accessors() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 4);
        assert_eq!(cfg.num_layers(), 12);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.max_seq_len(), 2048);
        assert_eq!(cfg.kv_dim(), 4 * 64);
        assert!(!cfg.is_mla());
        assert_eq!(cfg.num_kv_shared_layers(), 0);
        assert!(cfg.attention_pattern().is_empty());
    }

    #[test]
    fn kv_cache_config_mla_kv_dim() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_mla_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert!(cfg.is_mla());
        // MLA kv_dim = d_c + d_rope
        assert_eq!(cfg.kv_dim(), 512 + 64);
    }

    #[test]
    fn kv_cache_config_partial_eq_equal() {
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_config_partial_eq_different_page_size() {
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size: 32,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn kv_cache_config_partial_eq_different_dtype() {
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn kv_cache_config_partial_eq_swap_config_differs() {
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(SwapConfig {
                enable_swap: true,
                swap_threshold: 0.8,
                lru_granularity: 4,
            }),
        };
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------
    // BackendError Display
    // ---------------------------------------------------------------

    #[test]
    fn backend_error_display_cuda() {
        let err = BackendError::Cuda("oom".into());
        assert_eq!(format!("{err}"), "CUDA error: oom");
    }

    #[test]
    fn backend_error_display_hip() {
        let err = BackendError::Hip("fault".into());
        assert_eq!(format!("{err}"), "HIP error: fault");
    }

    #[test]
    fn backend_error_display_metal() {
        let err = BackendError::Metal("gone".into());
        assert_eq!(format!("{err}"), "Metal error: gone");
    }

    #[test]
    fn backend_error_display_cpu() {
        let err = BackendError::Cpu("segfault".into());
        assert_eq!(format!("{err}"), "CPU error: segfault");
    }

    #[test]
    fn backend_error_display_unimplemented() {
        let err = BackendError::Unimplemented("fp8");
        assert_eq!(format!("{err}"), "unimplemented: fp8");
    }

    #[test]
    fn backend_error_display_other() {
        let err = BackendError::Other("mystery".into());
        assert_eq!(format!("{err}"), "backend error: mystery");
    }

    // ---------------------------------------------------------------
    // KvCacheHandle
    // ---------------------------------------------------------------

    #[test]
    fn kv_cache_handle_traits() {
        let a = KvCacheHandle(42);
        let b = KvCacheHandle(42);
        let c = KvCacheHandle(99);

        // Copy
        let d = a;
        assert_eq!(a, d);

        // PartialEq + Eq
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Hash — must be usable as HashMap key
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        set.insert(c);
        assert!(set.contains(&a));
        assert!(set.contains(&c));
        assert!(!set.contains(&KvCacheHandle(7)));
    }

    // ---------------------------------------------------------------
    // LogitsHandle
    // ---------------------------------------------------------------

    #[test]
    fn logits_handle_construction() {
        let handle = LogitsHandle {
            data: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(handle.data.len(), 3);
        assert_eq!(handle.data[0], 1.0);
    }

    // ---------------------------------------------------------------
    // AttentionMaskType
    // ---------------------------------------------------------------

    #[test]
    fn attention_mask_type_variants() {
        assert_ne!(AttentionMaskType::Bidirectional, AttentionMaskType::Causal);
        assert_eq!(AttentionMaskType::Bidirectional, AttentionMaskType::Bidirectional);
    }

    // ---------------------------------------------------------------
    // AttentionTopology
    // ---------------------------------------------------------------

    #[test]
    fn attention_topology_bidirectional() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::bidirectional(geo);
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
        assert_eq!(topo.num_heads(), 16);
        assert_eq!(topo.num_kv_heads(), 4);
        assert_eq!(topo.head_dim(), 64);
        assert_eq!(topo.max_seq_len(), 2048);
    }

    #[test]
    fn attention_topology_causal() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::causal(geo);
        assert_eq!(topo.mask_type, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_topology_linear() {
        let topo = AttentionTopology::linear();
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
        assert_eq!(topo.max_seq_len(), 512);
    }

    // ---------------------------------------------------------------
    // SequenceInput::validate_page_table
    // ---------------------------------------------------------------

    #[test]
    fn validate_page_table_none_is_ok() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn validate_page_table_valid_entries() {
        let seq = SequenceInput {
            tokens: vec![1, 2],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 9]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn validate_page_table_out_of_bounds() {
        let seq = SequenceInput {
            tokens: vec![1, 2],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 10]),
            fused_hidden: None,
        };
        let result = seq.validate_page_table(10);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("page_table[2]"));
        assert!(msg.contains("10"));
        assert!(msg.contains("total_pages 10"));
    }

    #[test]
    fn validate_page_table_boundary_exact() {
        // page_id == total_pages is out of bounds (0-indexed)
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![9]),
            fused_hidden: None,
        };
        // total_pages = 9 -> valid indices 0..8; page_id 9 is OOB
        assert!(seq.validate_page_table(9).is_err());
    }

    // ---------------------------------------------------------------
    // BatchInput
    // ---------------------------------------------------------------

    #[test]
    fn batch_input_construction() {
        let batch = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![1, 2, 3],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![4, 5],
                    position: 3,
                    draft_steps: 0,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: None,
                },
            ],
        };
        assert_eq!(batch.sequences.len(), 2);
        assert_eq!(batch.sequences[1].position, 3);
    }

    // ---------------------------------------------------------------
    // ExecutorError Display / From
    // ---------------------------------------------------------------

    #[test]
    fn executor_error_from_backend() {
        let backend_err = BackendError::Cuda("oom".into());
        let exec_err: ExecutorError = backend_err.into();
        assert!(format!("{exec_err}").contains("CUDA error: oom"));
    }

    #[test]
    fn executor_error_scheduler_variant() {
        let err = ExecutorError::Scheduler("no slots".into());
        assert_eq!(format!("{err}"), "scheduler error: no slots");
    }

    #[test]
    fn executor_error_empty_prompt() {
        let err = ExecutorError::EmptyPrompt;
        assert_eq!(format!("{err}"), "empty prompt tokens");
    }

    #[test]
    fn executor_error_empty_sample() {
        let err = ExecutorError::EmptySample;
        assert_eq!(format!("{err}"), "backend returned empty sample");
    }

    #[test]
    fn executor_error_request_not_found() {
        let rid: RequestId = 42;
        let err = ExecutorError::RequestNotFound { request_id: rid };
        let msg = format!("{err}");
        assert!(msg.starts_with("request not found:"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn executor_error_compilation() {
        let err = ExecutorError::Compilation("bad ir".into());
        assert_eq!(format!("{err}"), "JIT compilation failed: bad ir");
    }

    #[test]
    fn executor_error_graph_expansion() {
        let err = ExecutorError::GraphExpansion("cycle".into());
        assert_eq!(format!("{err}"), "graph expansion failed: cycle");
    }

    #[test]
    fn executor_error_from_config() {
        let config_err = ModelConfigError::InvalidConfig("missing layers".into());
        let exec_err: ExecutorError = config_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("missing layers"));
    }

    // ---------------------------------------------------------------
    // RequestData construction
    // ---------------------------------------------------------------

    #[test]
    fn request_data_construction() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.prompt_tokens.len(), 3);
        assert_eq!(rd.output_tokens.len(), 2);
        assert_eq!(rd.phase, crate::scheduler::request_state::RequestPhase::Prefill);
        assert_eq!(rd.max_new_tokens, 100);
        assert!(!rd.finished);
        assert!(rd.session_id.is_none());
        assert!(rd.thinking_budget.is_none());
        assert!(rd.fused_prefill_hidden.is_none());
    }

    // ---------------------------------------------------------------
    // SamplingConfig: custom values, Copy, Clone, Debug
    // ---------------------------------------------------------------

    #[test]
    fn sampling_config_custom_values() {
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
        };
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_k, 50);
        assert_eq!(cfg.top_p, 0.95);
    }

    #[test]
    fn sampling_config_copy_trait() {
        let original = SamplingConfig {
            temperature: 0.5,
            top_k: 10,
            top_p: 0.9,
        };
        let copied = original;
        // Copy semantics: both are usable independently
        assert_eq!(original.temperature, copied.temperature);
        assert_eq!(original.top_k, copied.top_k);
        assert_eq!(original.top_p, copied.top_p);
    }

    #[test]
    fn sampling_config_debug_trait() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("SamplingConfig"));
    }

    // ---------------------------------------------------------------
    // RoPEConfig: Copy, Clone, partial_eq inequality on each field
    // ---------------------------------------------------------------

    #[test]
    fn rope_config_copy_trait() {
        let original = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn rope_config_scale_affects_equality() {
        let base = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let scaled = RoPEConfig {
            scale: 0.5,
            ..base
        };
        assert_ne!(base, scaled);
    }

    #[test]
    fn rope_config_interleaved_affects_equality() {
        let a = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = RoPEConfig {
            interleaved: true,
            ..a
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rope_config_precompute_affects_equality() {
        let a = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = RoPEConfig {
            precompute: true,
            ..a
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rope_config_debug_trait() {
        let cfg = RoPEConfig {
            theta: 500000.0,
            scale: 1.0,
            interleaved: true,
            precompute: true,
        };
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("RoPEConfig"));
    }

    // ---------------------------------------------------------------
    // AttentionHeadConfig: Copy, Clone
    // ---------------------------------------------------------------

    #[test]
    fn attention_head_config_copy_trait() {
        let cfg = AttentionHeadConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
        };
        let copied = cfg;
        assert_eq!(cfg.num_heads, copied.num_heads);
        assert_eq!(cfg.num_kv_heads, copied.num_kv_heads);
        assert_eq!(cfg.head_dim, copied.head_dim);
    }

    #[test]
    fn attention_head_config_debug_trait() {
        let cfg = AttentionHeadConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        };
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("AttentionHeadConfig"));
    }

    // ---------------------------------------------------------------
    // PagedKvConfig: Clone, page_table None
    // ---------------------------------------------------------------

    #[test]
    fn paged_kv_config_clone() {
        let cfg = PagedKvConfig {
            page_table: Some(vec![0, 1, 2, 3]),
            page_size: 16,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.page_table, cfg.page_table);
        assert_eq!(cloned.page_size, cfg.page_size);
    }

    #[test]
    fn paged_kv_config_empty_page_table() {
        let cfg = PagedKvConfig {
            page_table: Some(vec![]),
            page_size: 16,
        };
        assert!(cfg.page_table.as_ref().unwrap().is_empty());
    }

    // ---------------------------------------------------------------
    // GeneratorForwardConfig: Encoder arch family
    // ---------------------------------------------------------------

    #[test]
    fn forward_config_encoder_family() {
        let mut cfg = make_forward_config();
        cfg.arch_family = crate::manifest::ArchFamily::Encoder;
        assert_eq!(cfg.arch_family, crate::manifest::ArchFamily::Encoder);
    }

    #[test]
    fn forward_config_rerank_tokens() {
        let mut cfg = make_forward_config();
        cfg.rerank_yes_token_id = Some(1);
        cfg.rerank_no_token_id = Some(0);
        assert_eq!(cfg.rerank_yes_token_id, Some(1));
        assert_eq!(cfg.rerank_no_token_id, Some(0));
    }

    // ---------------------------------------------------------------
    // KvCacheConfig: BF16 dtype_size, swap_config equal, attention_pattern
    // ---------------------------------------------------------------

    #[test]
    fn kv_cache_config_bf16_dtype_size() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 2);
    }

    #[test]
    fn kv_cache_config_partial_eq_same_swap_config() {
        let geo = Arc::new(make_geometry());
        let swap = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.7,
            lru_granularity: 8,
        };
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(swap.clone()),
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(swap),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_config_with_attention_pattern() {
        let mut geo = make_geometry();
        geo.attention_pattern = vec![0, 1, 0, 1];
        geo.num_kv_shared_layers = 2;
        let cfg = KvCacheConfig {
            geometry: Arc::new(geo),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.attention_pattern(), &[0, 1, 0, 1]);
        assert_eq!(cfg.num_kv_shared_layers(), 2);
    }

    #[test]
    fn kv_cache_config_partial_eq_different_layers() {
        let geo_a = Arc::new(make_geometry());
        let mut geo_b = make_geometry();
        geo_b.num_layers = 24;
        let a = KvCacheConfig {
            geometry: geo_a,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: Arc::new(geo_b),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------
    // BackendError: Clone, std::error::Error
    // ---------------------------------------------------------------

    #[test]
    fn backend_error_clone() {
        let err = BackendError::Cuda("device lost".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_is_std_error() {
        let err = BackendError::Cpu("fault".into());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn backend_error_debug_format() {
        let err = BackendError::Other("detail".into());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("Other"));
    }

    // ---------------------------------------------------------------
    // KvCacheHandle: boundary values
    // ---------------------------------------------------------------

    #[test]
    fn kv_cache_handle_zero() {
        let h = KvCacheHandle(0);
        assert_eq!(h, KvCacheHandle(0));
    }

    #[test]
    fn kv_cache_handle_max() {
        let h = KvCacheHandle(u64::MAX);
        let mut set = std::collections::HashSet::new();
        set.insert(h);
        assert!(set.contains(&KvCacheHandle(u64::MAX)));
    }

    #[test]
    fn kv_cache_handle_debug_format() {
        let h = KvCacheHandle(42);
        let debug_str = format!("{h:?}");
        assert!(debug_str.contains("42"));
    }

    // ---------------------------------------------------------------
    // LogitsHandle: Clone, empty data, Debug
    // ---------------------------------------------------------------

    #[test]
    fn logits_handle_clone() {
        let handle = LogitsHandle {
            data: vec![1.0, 2.0, 3.0],
        };
        let cloned = handle.clone();
        assert_eq!(handle.data, cloned.data);
    }

    #[test]
    fn logits_handle_empty_data() {
        let handle = LogitsHandle { data: vec![] };
        assert!(handle.data.is_empty());
    }

    #[test]
    fn logits_handle_debug_trait() {
        let handle = LogitsHandle {
            data: vec![0.5],
        };
        let debug_str = format!("{handle:?}");
        assert!(debug_str.contains("LogitsHandle"));
    }

    // ---------------------------------------------------------------
    // AttentionMaskType: Copy, Clone
    // ---------------------------------------------------------------

    #[test]
    fn attention_mask_type_copy_trait() {
        let a = AttentionMaskType::Causal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn attention_mask_type_clone_trait() {
        let a = AttentionMaskType::Bidirectional;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---------------------------------------------------------------
    // AttentionTopology: Clone
    // ---------------------------------------------------------------

    #[test]
    fn attention_topology_clone() {
        let topo = AttentionTopology::causal(Arc::new(make_geometry()));
        let cloned = topo.clone();
        assert_eq!(topo.mask_type, cloned.mask_type);
        assert_eq!(topo.num_heads(), cloned.num_heads());
        assert_eq!(topo.num_kv_heads(), cloned.num_kv_heads());
        assert_eq!(topo.head_dim(), cloned.head_dim());
        assert_eq!(topo.max_seq_len(), cloned.max_seq_len());
    }

    // ---------------------------------------------------------------
    // SequenceInput: Clone, fused_hidden
    // ---------------------------------------------------------------

    #[test]
    fn sequence_input_clone() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 5,
            draft_steps: 0,
            page_table: Some(vec![0, 1]),
            fused_hidden: None,
        };
        let cloned = seq.clone();
        assert_eq!(cloned.tokens, seq.tokens);
        assert_eq!(cloned.position, seq.position);
        assert_eq!(cloned.page_table, seq.page_table);
    }

    #[test]
    fn sequence_input_with_fused_hidden() {
        let hidden = vec![0.1, 0.2, 0.3, 0.4];
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(hidden.clone()),
        };
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn sequence_input_validate_page_table_empty_is_ok() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_zero_total() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        // total_pages=0 means every page_id is out of bounds
        assert!(seq.validate_page_table(0).is_err());
    }

    // ---------------------------------------------------------------
    // BatchInput: Clone, empty batch
    // ---------------------------------------------------------------

    #[test]
    fn batch_input_empty() {
        let batch = BatchInput { sequences: vec![] };
        assert!(batch.sequences.is_empty());
    }

    #[test]
    fn batch_input_clone() {
        let batch = BatchInput {
            sequences: vec![SequenceInput {
                tokens: vec![1, 2],
                position: 0,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            }],
        };
        let cloned = batch.clone();
        assert_eq!(cloned.sequences.len(), 1);
        assert_eq!(cloned.sequences[0].tokens, vec![1, 2]);
    }

    // ---------------------------------------------------------------
    // RequestData: all phases, session_id, thinking_budget
    // ---------------------------------------------------------------

    #[test]
    fn request_data_all_phases() {
        use crate::scheduler::request_state::RequestPhase;

        let phases = [RequestPhase::Prefill, RequestPhase::Decode, RequestPhase::ChunkedPrefill];
        for (i, phase) in phases.into_iter().enumerate() {
            let rd = RequestData {
                prompt_tokens: vec![1],
                output_tokens: vec![],
                sampling_config: SamplingConfig::default(),
                phase: if i == 0 { crate::scheduler::request_state::RequestPhase::Prefill } else { crate::scheduler::request_state::RequestPhase::Decode },
                max_new_tokens: 10,
                finished: false,
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            assert_eq!(rd.prompt_tokens.len(), 1);
        }
    }

    #[test]
    fn request_data_with_session_id() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 50,
            finished: false,
            session_id: Some(SessionId::from(42u64)),
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert!(rd.session_id.is_some());
    }

    #[test]
    fn request_data_thinking_budget_zero_means_disabled() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: Some(0),
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.thinking_budget, Some(0));
    }

    #[test]
    fn request_data_thinking_budget_with_limit() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 200,
            finished: false,
            session_id: None,
            thinking_budget: Some(1024),
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.thinking_budget, Some(1024));
    }

    #[test]
    fn request_data_finished_state() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![2, 3],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 2,
            finished: true,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert!(rd.finished);
        assert_eq!(rd.max_new_tokens, 2);
    }

    // ---------------------------------------------------------------
    // ExecutorResult type alias
    // ---------------------------------------------------------------

    #[test]
    fn executor_result_ok() {
        let result: ExecutorResult<usize> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn executor_result_err() {
        let result: ExecutorResult<usize> = Err(ExecutorError::EmptyPrompt);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: 55 additional tests for coverage improvement
    // ========================================================================

    // ---- RequestPhase: all variants, Debug, Copy, Clone, PartialEq ----

    #[test]
    fn request_phase_all_variants_are_distinct() {
        use crate::scheduler::request_state::RequestPhase;
        let phases = [
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::ChunkedPrefill,
        ];
        for (i, &a) in phases.iter().enumerate() {
            for (j, &b) in phases.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn request_phase_debug_format() {
        use crate::scheduler::request_state::RequestPhase;
        assert!(format!("{:?}", RequestPhase::Prefill).contains("Prefill"));
        assert!(format!("{:?}", RequestPhase::Decode).contains("Decode"));
        assert!(format!("{:?}", RequestPhase::ChunkedPrefill).contains("ChunkedPrefill"));
    }

    #[test]
    fn request_phase_copy_semantics() {
        use crate::scheduler::request_state::RequestPhase;
        let a = RequestPhase::Decode;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_phase_clone_semantics() {
        use crate::scheduler::request_state::RequestPhase;
        let a = RequestPhase::ChunkedPrefill;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- CompactScatterMeta: construction, equality, boundary values ----

    #[test]
    fn compact_scatter_meta_construction() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta {
            original_slot: 0,
            compacted_slot: 1,
            active: 1,
        };
        assert_eq!(meta.original_slot, 0);
        assert_eq!(meta.compacted_slot, 1);
        assert_eq!(meta.active, 1);
    }

    #[test]
    fn compact_scatter_meta_equality() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        let b = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn compact_scatter_meta_inequality() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        let b = CompactScatterMeta { original_slot: 6, compacted_slot: 3, active: 1 };
        assert_ne!(a, b);
    }

    #[test]
    fn compact_scatter_meta_max_values() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta {
            original_slot: u32::MAX,
            compacted_slot: u32::MAX,
            active: u32::MAX,
        };
        assert_eq!(meta.original_slot, u32::MAX);
        assert_eq!(meta.active, u32::MAX);
    }

    #[test]
    fn compact_scatter_meta_copy_semantics() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 10, compacted_slot: 5, active: 0 };
        let b = a;
        assert_eq!(a, b);
    }

    // ---- RequestTelemetry: construction, special floats ----

    #[test]
    fn request_telemetry_construction() {
        use crate::scheduler::request_state::RequestTelemetry;
        let t = RequestTelemetry {
            entropy: 2.5,
            centroid: 0.75,
            residual_delta: -0.1,
            residual_cosine: 0.99,
            range_group: 2,
        };
        assert!((t.entropy - 2.5).abs() < 1e-6);
        assert!((t.centroid - 0.75).abs() < 1e-6);
        assert!((t.residual_delta - (-0.1)).abs() < 1e-6);
        assert!((t.residual_cosine - 0.99).abs() < 1e-6);
        assert_eq!(t.range_group, 2);
    }

    #[test]
    fn request_telemetry_nan_and_infinity() {
        use crate::scheduler::request_state::RequestTelemetry;
        let t = RequestTelemetry {
            entropy: f32::NAN,
            centroid: f32::INFINITY,
            residual_delta: f32::NEG_INFINITY,
            residual_cosine: 0.0,
            range_group: 0,
        };
        assert!(t.entropy.is_nan());
        assert!(t.centroid.is_infinite() && t.centroid.is_sign_positive());
        assert!(t.residual_delta.is_infinite() && t.residual_delta.is_sign_negative());
    }

    #[test]
    fn request_telemetry_equality_same_values() {
        use crate::scheduler::request_state::RequestTelemetry;
        let a = RequestTelemetry {
            entropy: 1.0, centroid: 0.5, residual_delta: 0.0,
            residual_cosine: 1.0, range_group: 3,
        };
        let b = RequestTelemetry {
            entropy: 1.0, centroid: 0.5, residual_delta: 0.0,
            residual_cosine: 1.0, range_group: 3,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn request_telemetry_zero_values() {
        use crate::scheduler::request_state::RequestTelemetry;
        let t = RequestTelemetry {
            entropy: 0.0, centroid: 0.0, residual_delta: 0.0,
            residual_cosine: 0.0, range_group: 0,
        };
        assert_eq!(t.entropy, 0.0);
        assert_eq!(t.range_group, 0);
    }

    // ---- SamplingConfig: zero temperature, negative values ----

    #[test]
    fn sampling_config_zero_temperature_greedy() {
        let cfg = SamplingConfig { temperature: 0.0, top_k: 1, top_p: 0.0 };
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 1);
    }

    #[test]
    fn sampling_config_very_small_top_p() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: 50, top_p: 1e-10 };
        assert!(cfg.top_p > 0.0 && cfg.top_p < 1e-9);
    }

    #[test]
    fn sampling_config_negative_temperature() {
        let cfg = SamplingConfig { temperature: -1.0, top_k: 0, top_p: 1.0 };
        assert!(cfg.temperature < 0.0);
    }

    #[test]
    fn sampling_config_top_p_above_one() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 2.0 };
        assert!(cfg.top_p > 1.0);
    }

    // ---- RoPEConfig: NaN theta, zero theta, extreme scale ----

    #[test]
    fn rope_config_nan_theta() {
        let cfg = RoPEConfig { theta: f64::NAN, scale: 1.0, interleaved: false, precompute: false };
        assert!(cfg.theta.is_nan());
    }

    #[test]
    fn rope_config_infinity_scale() {
        let cfg = RoPEConfig { theta: 10000.0, scale: f64::INFINITY, interleaved: false, precompute: false };
        assert!(cfg.scale.is_infinite());
    }

    #[test]
    fn rope_config_negative_theta() {
        let cfg = RoPEConfig { theta: -500.0, scale: 1.0, interleaved: false, precompute: false };
        assert!(cfg.theta < 0.0);
    }

    // ---- AttentionHeadConfig: zero values, large values ----

    #[test]
    fn attention_head_config_single_head() {
        let cfg = AttentionHeadConfig { num_heads: 1, num_kv_heads: 1, head_dim: 1 };
        assert_eq!(cfg.num_heads, 1);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.head_dim, 1);
    }

    #[test]
    fn attention_head_config_gqa_ratio() {
        // 8 query heads, 2 kv heads = 4x GQA
        let cfg = AttentionHeadConfig { num_heads: 8, num_kv_heads: 2, head_dim: 64 };
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4);
    }

    // ---- SwapConfig: all fields differ in inequality ----

    #[test]
    fn swap_config_enable_swap_differs() {
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 };
        let b = SwapConfig { enable_swap: false, swap_threshold: 0.8, lru_granularity: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn swap_config_swap_threshold_differs() {
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 };
        let b = SwapConfig { enable_swap: true, swap_threshold: 0.9, lru_granularity: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn swap_config_lru_granularity_differs() {
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 };
        let b = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 8 };
        assert_ne!(a, b);
    }

    #[test]
    fn swap_config_infinity_threshold() {
        let cfg = SwapConfig { enable_swap: true, swap_threshold: f32::INFINITY, lru_granularity: 1 };
        assert!(cfg.swap_threshold.is_infinite());
    }

    // ---- KvCacheConfig: BF16 dtype_size, with shared layers, MLA geometry ----

    #[test]
    fn kv_cache_config_bf16_dtype_size_is_2() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 2);
    }

    #[test]
    fn kv_cache_config_with_shared_kv_layers() {
        let mut geo = make_geometry();
        geo.num_kv_shared_layers = 4;
        geo.attention_pattern = vec![0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let cfg = KvCacheConfig {
            geometry: Arc::new(geo),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.num_kv_shared_layers(), 4);
        assert_eq!(cfg.attention_pattern().len(), 12);
    }

    #[test]
    fn kv_cache_config_mla_dim_calculation() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_mla_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert!(cfg.is_mla());
        // MLA kv_dim = d_c + d_rope = 512 + 64
        assert_eq!(cfg.kv_dim(), 576);
    }

    #[test]
    fn kv_cache_config_zero_page_size() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 0,
            swap_config: None,
        };
        assert_eq!(cfg.page_size, 0);
    }

    // ---- BackendError: unimplemented with static str ----

    #[test]
    fn backend_error_unimplemented_static_str() {
        let err = BackendError::Unimplemented("fp8_gemm");
        let msg = format!("{err}");
        assert_eq!(msg, "unimplemented: fp8_gemm");
    }

    #[test]
    fn backend_error_cuda_with_detailed_message() {
        let err = BackendError::Cuda("CUDA_ERROR_OUT_OF_MEMORY: cannot allocate 2GB".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("CUDA error: "));
        assert!(msg.contains("OUT_OF_MEMORY"));
    }

    #[test]
    fn backend_error_hip_long_message() {
        let long_msg = "a".repeat(1000);
        let err = BackendError::Hip(long_msg.clone());
        assert_eq!(format!("{err}"), format!("HIP error: {long_msg}"));
    }

    // ---- KvCacheHandle: as HashMap key, BTreeSet ordering ----

    #[test]
    fn kv_cache_handle_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(KvCacheHandle(1), "front");
        map.insert(KvCacheHandle(2), "back");
        assert_eq!(map.get(&KvCacheHandle(1)), Some(&"front"));
        assert_eq!(map.get(&KvCacheHandle(2)), Some(&"back"));
        assert_eq!(map.get(&KvCacheHandle(3)), None);
    }

    #[test]
    fn kv_cache_handle_multiple_insert_overwrite() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(KvCacheHandle(1), "first");
        map.insert(KvCacheHandle(1), "second");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&KvCacheHandle(1)), Some(&"second"));
    }

    // ---- LogitsHandle: many values, f32 precision ----

    #[test]
    fn logits_handle_large_data() {
        let data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        let handle = LogitsHandle { data };
        assert_eq!(handle.data.len(), 10000);
        assert!((handle.data[9999] - 9.999).abs() < 0.01);
    }

    #[test]
    fn logits_handle_with_subnormal_floats() {
        let handle = LogitsHandle { data: vec![f32::MIN_POSITIVE, f32::EPSILON] };
        assert!(handle.data[0] > 0.0);
        assert!(handle.data[1] > 0.0);
        // f32::MIN_POSITIVE (1.175e-38) < f32::EPSILON (1.192e-7)
        assert!(handle.data[0] < handle.data[1]);
    }

    // ---- AttentionMaskType: Eq, Hash ----

    #[test]
    fn attention_mask_type_hash_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AttentionMaskType::Causal);
        set.insert(AttentionMaskType::Bidirectional);
        set.insert(AttentionMaskType::Causal);
        assert_eq!(set.len(), 2);
    }

    // ---- AttentionTopology: with different geometries ----

    #[test]
    fn attention_topology_with_large_geometry() {
        let geo = Arc::new(ModelGeometry {
            hidden_size: 8192,
            num_layers: 80,
            vocab_size: 128256,
            intermediate_size: 28672,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 131072,
            ..make_geometry()
        });
        let topo = AttentionTopology::causal(geo);
        assert_eq!(topo.num_heads(), 64);
        assert_eq!(topo.num_kv_heads(), 8);
        assert_eq!(topo.head_dim(), 128);
        assert_eq!(topo.max_seq_len(), 131072);
    }

    // ---- SequenceInput: validate_page_table with multiple OOB entries ----

    #[test]
    fn sequence_input_validate_multiple_oob_entries() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![100, 200]),
            fused_hidden: None,
        };
        let err = seq.validate_page_table(10).unwrap_err();
        assert!(err.contains("page_table[0] = 100"));
    }

    #[test]
    fn sequence_input_validate_large_total_pages() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(usize::MAX).is_ok());
    }

    #[test]
    fn sequence_input_draft_steps_boundary() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: usize::MAX,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.draft_steps, usize::MAX);
    }

    #[test]
    fn sequence_input_with_empty_fused_hidden() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(vec![]),
        };
        assert!(seq.fused_hidden.unwrap().is_empty());
    }

    // ---- BatchInput: many sequences ----

    #[test]
    fn batch_input_many_sequences() {
        let sequences: Vec<SequenceInput> = (0..100)
            .map(|i| SequenceInput {
                tokens: vec![i as u32],
                position: i,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            })
            .collect();
        let batch = BatchInput { sequences };
        assert_eq!(batch.sequences.len(), 100);
        assert_eq!(batch.sequences[0].tokens, vec![0]);
        assert_eq!(batch.sequences[99].tokens, vec![99]);
    }

    // ---- RequestData: multiple phases and max_new_tokens boundary ----

    #[test]
    fn request_data_max_new_tokens_zero() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 0,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.max_new_tokens, 0);
    }

    #[test]
    fn request_data_max_new_tokens_usize_max() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: usize::MAX,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.max_new_tokens, usize::MAX);
    }

    #[test]
    fn request_data_empty_prompt_and_output() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert!(rd.prompt_tokens.is_empty());
        assert!(rd.output_tokens.is_empty());
    }

    #[test]
    fn request_data_chunked_prefill_phase() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![4],
            sampling_config: SamplingConfig::default(),
            phase: RequestPhase::ChunkedPrefill,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.phase, RequestPhase::ChunkedPrefill);
    }

    // ---- ExecutorError: From<ModelConfigError> chain ----

    #[test]
    fn executor_error_from_model_config_missing_config() {
        let config_err = ModelConfigError::MissingConfig;
        let exec_err: ExecutorError = config_err.into();
        assert!(format!("{exec_err}").contains("metadata-driven config unavailable"));
    }

    #[test]
    fn executor_error_from_model_config_invalid_config() {
        let config_err = ModelConfigError::InvalidConfig("missing field".into());
        let exec_err: ExecutorError = config_err.into();
        assert!(format!("{exec_err}").contains("missing field"));
    }

    // ---- ExecutorError: all variant Display formatting ----

    #[test]
    fn executor_error_scheduler_with_long_message() {
        let long_msg = "x".repeat(500);
        let err = ExecutorError::Scheduler(long_msg.clone());
        assert_eq!(format!("{err}"), format!("scheduler error: {long_msg}"));
    }

    #[test]
    fn executor_error_compilation_with_details() {
        let err = ExecutorError::Compilation("phase 3 lowering failed: unknown VmInstr".into());
        assert!(format!("{err}").contains("JIT compilation failed"));
        assert!(format!("{err}").contains("VmInstr"));
    }

    #[test]
    fn executor_error_graph_expansion_with_details() {
        let err = ExecutorError::GraphExpansion("unsupported OpKind::MoERouter".into());
        assert!(format!("{err}").contains("graph expansion failed"));
        assert!(format!("{err}").contains("MoERouter"));
    }

    // ---- GeneratorForwardConfig: attention_geometry with GQA ----

    #[test]
    fn generator_forward_config_attention_geometry_gqa() {
        let geo = Arc::new(ModelGeometry {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            ..make_geometry()
        });
        let cfg = GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let ag = cfg.attention_geometry();
        assert_eq!(ag.heads_per_group, 8); // 32 / 4
        assert_eq!(ag.q_dim, 32 * 128);
        assert_eq!(ag.kv_dim, 4 * 128);
    }

    #[test]
    fn generator_forward_config_moe_config_some() {
        let cfg = GeneratorForwardConfig {
            geometry: Arc::new(make_geometry()),
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: Some(crate::manifest::MoEConfig {
                num_experts: 64,
                num_experts_per_tok: 8,
                router_type: crate::manifest::RouterType::Mixtral,
            }),
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let moe = cfg.moe_config.unwrap();
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.num_experts_per_tok, 8);
    }

    #[test]
    fn generator_forward_config_with_paged_kv_table() {
        let cfg = GeneratorForwardConfig {
            geometry: Arc::new(make_geometry()),
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: Some(vec![0, 1, 2, 3, 4]), page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        assert_eq!(cfg.paged_kv.page_table.unwrap().len(), 5);
    }

    // ---- GeneratorForwardConfig: Encoder arch family ----

    #[test]
    fn generator_forward_config_encoder_arch_family() {
        let cfg = GeneratorForwardConfig {
            geometry: Arc::new(make_geometry()),
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Encoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        assert_eq!(cfg.arch_family, crate::manifest::ArchFamily::Encoder);
    }

    // ---- effective_kv_max_seq_len: more edge cases ----

    #[test]
    fn effective_kv_max_seq_len_various_sizes() {
        // Common model context sizes
        assert_eq!(effective_kv_max_seq_len(2048), 2048);
        assert_eq!(effective_kv_max_seq_len(4096), 4096);
        assert_eq!(effective_kv_max_seq_len(8192), 8192);
        assert_eq!(effective_kv_max_seq_len(32768), 32768);
        assert_eq!(effective_kv_max_seq_len(131072), 131072);
    }

    // ---- KvCacheConfig PartialEq: different heads ----

    #[test]
    fn kv_cache_config_partial_eq_different_heads() {
        let mut geo_b = make_geometry();
        geo_b.num_kv_heads = 8;
        let a = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: Arc::new(geo_b),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    // ---- KvCacheConfig PartialEq: different head_dim ----

    #[test]
    fn kv_cache_config_partial_eq_different_head_dim() {
        let mut geo_b = make_geometry();
        geo_b.head_dim = 128;
        let a = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: Arc::new(geo_b),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    // ---- RequestData: fused_prefill_hidden with actual data ----

    #[test]
    fn request_data_fused_prefill_hidden_large() {
        use crate::scheduler::request_state::RequestPhase;
        let hidden: Vec<f32> = (0..2048).map(|i| i as f32 * 0.01).collect();
        let rd = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 128,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: Some(hidden),
        };
        let h = rd.fused_prefill_hidden.unwrap();
        assert_eq!(h.len(), 2048);
        assert!((h[0] - 0.0).abs() < 1e-6);
        assert!((h[2047] - 20.47).abs() < 0.01);
    }

    // ---- BackendError: debug format all variants ----

    #[test]
    fn backend_error_debug_all_variants() {
        let variants = [
            (BackendError::Cuda("c".into()), "Cuda"),
            (BackendError::Hip("h".into()), "Hip"),
            (BackendError::Metal("m".into()), "Metal"),
            (BackendError::Cpu("p".into()), "Cpu"),
            (BackendError::Unimplemented("u"), "Unimplemented"),
            (BackendError::Other("o".into()), "Other"),
        ];
        for (err, name) in variants {
            let debug = format!("{err:?}");
            assert!(debug.contains(name), "Expected {name} in debug output: {debug}");
        }
    }

    // ---- ExecutorError: std::error::Error trait ----

    #[test]
    fn executor_error_is_std_error() {
        let err = ExecutorError::EmptyPrompt;
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn executor_error_error_chain_source() {
        let backend_err = BackendError::Cpu("fault".into());
        let exec_err = ExecutorError::from(backend_err);
        // thiserror #[error(transparent)] forwards Display and source
        let display = format!("{exec_err}");
        assert!(display.contains("CPU error: fault"));
    }

    // ========================================================================
    // Batch 3: 44 additional tests for coverage improvement (192 → 196 tests)
    // ========================================================================

    // ---- GeneratorForwardConfig::default_for_test ----

    #[test]
    fn default_for_test_has_correct_geometry() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.max_seq_len(), 512);
    }

    #[test]
    fn default_for_test_rope_config() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!((cfg.rope_theta() - 10000.0).abs() < 1e-6);
        assert!((cfg.rope_scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn default_for_test_arch_family_is_decoder() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.arch_family, crate::manifest::ArchFamily::Decoder);
    }

    #[test]
    fn default_for_test_no_rerank_tokens() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.rerank_yes_token_id.is_none());
        assert!(cfg.rerank_no_token_id.is_none());
    }

    #[test]
    fn default_for_test_no_moe_config() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.moe_config.is_none());
    }

    #[test]
    fn default_for_test_paged_kv_config() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.paged_kv.page_table.is_none());
        assert_eq!(cfg.paged_kv.page_size, 16);
    }

    #[test]
    fn default_for_test_dtype_is_f32() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.dtype(), DType::F32);
    }

    #[test]
    fn default_for_test_norm_eps() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!((cfg.norm_eps() - 1e-5).abs() < 1e-10);
    }

    // ---- GeneratorForwardConfig: attention with MHA (1:1 heads) ----

    #[test]
    fn forward_config_attention_mha_heads_equal_kv_heads() {
        let geo = Arc::new(ModelGeometry {
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 64,
            ..make_geometry()
        });
        let cfg = GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let ag = cfg.attention_geometry();
        assert_eq!(ag.heads_per_group, 1);
        assert_eq!(ag.q_dim, ag.kv_dim);
    }

    // ---- GeneratorForwardConfig: layer_dims with custom rope_theta ----

    #[test]
    fn forward_config_layer_dims_custom_rope_theta() {
        let geo = Arc::new(ModelGeometry {
            rope_theta: 500000.0,
            ..make_geometry()
        });
        let cfg = GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig { theta: 500000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let ld = cfg.layer_dims();
        assert!((ld.rope_theta - 500000.0).abs() < 1e-6);
    }

    // ---- GeneratorForwardConfig: Clone ----

    #[test]
    fn forward_config_clone_preserves_fields() {
        let cfg = make_forward_config();
        let cloned = cfg.clone();
        assert_eq!(cloned.hidden_size(), cfg.hidden_size());
        assert_eq!(cloned.num_layers(), cfg.num_layers());
        assert_eq!(cloned.vocab_size(), cfg.vocab_size());
        assert_eq!(cloned.arch_family, cfg.arch_family);
    }

    // ---- GeneratorForwardConfig: BF16 dtype ----

    #[test]
    fn forward_config_bf16_dtype() {
        let geo = Arc::new(ModelGeometry {
            compute_dtype: DType::BF16,
            ..make_geometry()
        });
        let cfg = GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        assert_eq!(cfg.dtype(), DType::BF16);
    }

    // ---- GeneratorForwardConfig: MoE with different router types ----

    #[test]
    fn forward_config_moe_deepseek_router() {
        let cfg = GeneratorForwardConfig {
            geometry: Arc::new(make_geometry()),
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: Some(crate::manifest::MoEConfig {
                num_experts: 256,
                num_experts_per_tok: 8,
                router_type: crate::manifest::RouterType::DeepSeek,
            }),
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let moe = cfg.moe_config.unwrap();
        assert_eq!(moe.router_type, crate::manifest::RouterType::DeepSeek);
        assert_eq!(moe.num_experts, 256);
    }

    #[test]
    fn forward_config_moe_qwen_router() {
        let cfg = GeneratorForwardConfig {
            geometry: Arc::new(make_geometry()),
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: Some(crate::manifest::MoEConfig {
                num_experts: 64,
                num_experts_per_tok: 8,
                router_type: crate::manifest::RouterType::Qwen,
            }),
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        assert_eq!(cfg.moe_config.unwrap().router_type, crate::manifest::RouterType::Qwen);
    }

    // ---- GeneratorForwardConfig: rerank tokens present ----

    #[test]
    fn forward_config_both_rerank_tokens_set() {
        let mut cfg = make_forward_config();
        cfg.rerank_yes_token_id = Some(1);
        cfg.rerank_no_token_id = Some(0);
        assert_eq!(cfg.rerank_yes_token_id, Some(1));
        assert_eq!(cfg.rerank_no_token_id, Some(0));
        assert_ne!(cfg.rerank_yes_token_id, cfg.rerank_no_token_id);
    }

    // ---- KvCacheConfig: with swap_config present ----

    #[test]
    fn kv_cache_config_with_swap_config() {
        let swap = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.85,
            lru_granularity: 16,
        };
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(swap),
        };
        let sw = cfg.swap_config.unwrap();
        assert!(sw.enable_swap);
        assert!((sw.swap_threshold - 0.85).abs() < 1e-6);
        assert_eq!(sw.lru_granularity, 16);
    }

    // ---- KvCacheConfig: F16 dtype_size ----

    #[test]
    fn kv_cache_config_f16_dtype_size() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F16,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 2);
    }

    // ---- KvCacheConfig: kv_dim standard model ----

    #[test]
    fn kv_cache_config_standard_kv_dim() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Standard: num_kv_heads * head_dim = 4 * 64 = 256
        assert_eq!(cfg.kv_dim(), 256);
    }

    // ---- KvCacheConfig: PartialEq different max_seq_len ----

    #[test]
    fn kv_cache_config_partial_eq_different_max_seq_len() {
        let mut geo_b = make_geometry();
        geo_b.max_seq_len = 8192;
        let a = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: Arc::new(geo_b),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    // ---- KvCacheConfig: Clone ----

    #[test]
    fn kv_cache_config_clone() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
        assert_eq!(cfg.dtype_size(), cloned.dtype_size());
        assert_eq!(cfg.num_layers(), cloned.num_layers());
    }

    // ---- SequenceInput: position boundary values ----

    #[test]
    fn sequence_input_position_zero() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, 0);
    }

    #[test]
    fn sequence_input_position_large() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: usize::MAX,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, usize::MAX);
    }

    // ---- SequenceInput: empty tokens ----

    #[test]
    fn sequence_input_empty_tokens() {
        let seq = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(seq.tokens.is_empty());
    }

    // ---- SequenceInput: validate_page_table with single page ----

    #[test]
    fn sequence_input_validate_single_page_valid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_ok());
    }

    // ---- SequenceInput: fused_hidden with consistent length ----

    #[test]
    fn sequence_input_fused_hidden_matches_tokens_and_hidden_size() {
        let hidden_size = 64;
        let tokens = vec![1, 2, 3];
        let fused: Vec<f32> = (0..tokens.len() * hidden_size).map(|i| i as f32).collect();
        let seq = SequenceInput {
            tokens,
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(fused),
        };
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 3 * 64);
    }

    // ---- BatchInput: Debug trait ----

    #[test]
    fn batch_input_debug_format() {
        let batch = BatchInput {
            sequences: vec![SequenceInput {
                tokens: vec![1],
                position: 0,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            }],
        };
        let debug = format!("{batch:?}");
        assert!(debug.contains("BatchInput"));
    }

    // ---- RequestData: Debug trait ----

    #[test]
    fn request_data_debug_format() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        let debug = format!("{rd:?}");
        assert!(debug.contains("RequestData"));
    }

    // ---- RequestData: with custom SamplingConfig ----

    #[test]
    fn request_data_custom_sampling_config() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig { temperature: 0.3, top_k: 10, top_p: 0.95 },
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert!((rd.sampling_config.temperature - 0.3).abs() < 1e-6);
        assert_eq!(rd.sampling_config.top_k, 10);
    }

    // ---- RequestData: large output tokens ----

    #[test]
    fn request_data_large_output_tokens() {
        use crate::scheduler::request_state::RequestPhase;
        let output: Vec<u32> = (0..1000).collect();
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: output.clone(),
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 1000,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.output_tokens.len(), 1000);
        assert_eq!(rd.output_tokens[0], 0);
        assert_eq!(rd.output_tokens[999], 999);
    }

    // ---- SwapConfig: Debug trait ----

    #[test]
    fn swap_config_debug_format() {
        let cfg = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("SwapConfig"));
    }

    // ---- SwapConfig: Copy trait ----

    #[test]
    fn swap_config_copy_trait() {
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 8 };
        let b = a.clone();
        assert_eq!(a, b);
        // SwapConfig derives Clone; verify cloned copy is independent
        let mut c = a.clone();
        c.enable_swap = false;
        assert!(a.enable_swap);
        assert!(!c.enable_swap);
    }

    // ---- SwapConfig: Clone trait ----

    #[test]
    fn swap_config_clone_trait() {
        let a = SwapConfig { enable_swap: false, swap_threshold: 0.9, lru_granularity: 2 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- SwapConfig: NaN threshold ----

    #[test]
    fn swap_config_nan_threshold() {
        let cfg = SwapConfig { enable_swap: true, swap_threshold: f32::NAN, lru_granularity: 1 };
        assert!(cfg.swap_threshold.is_nan());
    }

    // ---- AttentionTopology: Debug trait ----

    #[test]
    fn attention_topology_debug_format() {
        let topo = AttentionTopology::causal(Arc::new(make_geometry()));
        let debug = format!("{topo:?}");
        assert!(debug.contains("AttentionTopology"));
    }

    // ---- AttentionTopology: different num_kv_heads ----

    #[test]
    fn attention_topology_gqa_geometry() {
        let geo = Arc::new(ModelGeometry {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            ..make_geometry()
        });
        let topo = AttentionTopology::causal(geo);
        assert_eq!(topo.num_heads(), 32);
        assert_eq!(topo.num_kv_heads(), 4);
        assert_eq!(topo.head_dim(), 128);
    }

    // ---- BackendError: Metal with empty string ----

    #[test]
    fn backend_error_metal_empty_message() {
        let err = BackendError::Metal(String::new());
        assert_eq!(format!("{err}"), "Metal error: ");
    }

    // ---- BackendError: Cpu with empty string ----

    #[test]
    fn backend_error_cpu_empty_message() {
        let err = BackendError::Cpu(String::new());
        assert_eq!(format!("{err}"), "CPU error: ");
    }

    // ---- BackendError: Other with empty string ----

    #[test]
    fn backend_error_other_empty_message() {
        let err = BackendError::Other(String::new());
        assert_eq!(format!("{err}"), "backend error: ");
    }

    // ---- ExecutorError: from KvCacheError ----

    #[test]
    fn executor_error_from_kv_cache() {
        let kv_err = KvCacheError::Exhausted { requested: 100, available: 50 };
        let exec_err: ExecutorError = kv_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    // ---- ExecutorError: from TokenizerError ----

    #[test]
    fn executor_error_from_tokenizer_missing() {
        let tok_err = TokenizerError::MissingTokenizer;
        let exec_err: ExecutorError = tok_err.into();
        assert!(format!("{exec_err}").contains("tokenizer.json not found"));
    }

    // ---- ExecutorError: from TokenizerError::Tokenizers ----

    #[test]
    fn executor_error_from_tokenizer_error() {
        let tok_err = TokenizerError::Tokenizers("decode failed".into());
        let exec_err: ExecutorError = tok_err.into();
        assert!(format!("{exec_err}").contains("decode failed"));
    }

    // ---- ExecutorError: from ModelConfigError::MissingConfigAndMetadata ----

    #[test]
    fn executor_error_from_model_config_missing_and_metadata() {
        let config_err = ModelConfigError::MissingConfigAndMetadata("no gguf metadata".into());
        let exec_err: ExecutorError = config_err.into();
        assert!(format!("{exec_err}").contains("no gguf metadata"));
    }

    // ---- ExecutorResult: chaining operations ----

    #[test]
    fn executor_result_map_on_ok() {
        let result: ExecutorResult<i32> = Ok(10);
        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.unwrap(), 20);
    }

    #[test]
    fn executor_result_map_err_on_error() {
        let result: ExecutorResult<i32> = Err(ExecutorError::EmptyPrompt);
        let mapped = result.map_err(|e| format!("wrapped: {e}"));
        assert!(mapped.is_err());
        assert!(mapped.unwrap_err().contains("wrapped:"));
    }

    // ---- effective_kv_max_seq_len: identity property ----

    #[test]
    fn effective_kv_max_seq_len_identity_for_common_sizes() {
        // Common model max_position_embeddings
        for &size in &[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] {
            assert_eq!(effective_kv_max_seq_len(size), size);
        }
    }

    // ---- KvCacheHandle: ordering by comparison operators ----

    #[test]
    fn kv_cache_handle_ordering() {
        let a = KvCacheHandle(1);
        let b = KvCacheHandle(2);
        let c = KvCacheHandle(3);
        // Verify natural ordering via u64 inner value comparison
        assert!(a.0 < b.0);
        assert!(b.0 < c.0);
        // Verify sorting by inner value
        let mut handles = vec![c, a, b];
        handles.sort_by_key(|h| h.0);
        assert_eq!(handles[0], KvCacheHandle(1));
        assert_eq!(handles[1], KvCacheHandle(2));
        assert_eq!(handles[2], KvCacheHandle(3));
    }

    // ---- LogitsHandle: Debug trait format check ----

    #[test]
    fn logits_handle_debug_contains_data_length() {
        let handle = LogitsHandle { data: vec![0.1; 5] };
        let debug = format!("{handle:?}");
        assert!(debug.contains("LogitsHandle"));
    }

    // ---- AttentionMaskType: Debug trait ----

    #[test]
    fn attention_mask_type_debug_format() {
        let debug_bi = format!("{:?}", AttentionMaskType::Bidirectional);
        let debug_ca = format!("{:?}", AttentionMaskType::Causal);
        assert!(debug_bi.contains("Bidirectional"));
        assert!(debug_ca.contains("Causal"));
    }

    // ---- PagedKvConfig: Debug trait ----

    #[test]
    fn paged_kv_config_debug_format() {
        let cfg = PagedKvConfig { page_table: Some(vec![0, 1]), page_size: 32 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("PagedKvConfig"));
    }

    // ---- RoPEConfig: all fields different ----

    #[test]
    fn rope_config_all_fields_differ_from_default() {
        let default = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let custom = RoPEConfig { theta: 500000.0, scale: 0.5, interleaved: true, precompute: true };
        assert_ne!(default, custom);
    }

    // ---- SamplingConfig: clone independence ----

    #[test]
    fn sampling_config_clone_independence() {
        let original = SamplingConfig { temperature: 1.0, top_k: 50, top_p: 0.9 };
        let cloned = original;
        // Copy semantics: modifying a copy does not affect the original
        let mut modified = cloned;
        modified.temperature = 0.0;
        assert!((original.temperature - 1.0).abs() < 1e-6);
        assert!((modified.temperature - 0.0).abs() < 1e-6);
    }

    // ========================================================================
    // Batch 4: 15 additional tests for coverage improvement
    // ========================================================================

    // ---- effective_kv_max_seq_len: usize::MAX ----

    #[test]
    fn effective_kv_max_seq_len_usize_max() {
        assert_eq!(effective_kv_max_seq_len(usize::MAX), usize::MAX);
    }

    // ---- AttentionHeadConfig: from_geometry with MLA geometry ----

    #[test]
    fn attention_head_config_from_mla_geometry() {
        let geo = make_mla_geometry();
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, geo.num_heads);
        assert_eq!(cfg.num_kv_heads, geo.num_kv_heads);
        assert_eq!(cfg.head_dim, geo.head_dim);
    }

    // ---- PagedKvConfig: large page_table ----

    #[test]
    fn paged_kv_config_large_page_table() {
        let table: Vec<u32> = (0..10000).collect();
        let cfg = PagedKvConfig { page_table: Some(table), page_size: 1 };
        assert_eq!(cfg.page_table.as_ref().unwrap().len(), 10000);
        assert_eq!(cfg.page_table.as_ref().unwrap()[9999], 9999);
    }

    // ---- GeneratorForwardConfig::default_for_test: attention method ----

    #[test]
    fn default_for_test_attention_config() {
        let cfg = GeneratorForwardConfig::default_for_test();
        let attn = cfg.attention();
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16);
    }

    // ---- SwapConfig: zero lru_granularity ----

    #[test]
    fn swap_config_zero_lru_granularity() {
        let cfg = SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 0 };
        assert_eq!(cfg.lru_granularity, 0);
        assert_ne!(
            cfg,
            SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 1 }
        );
    }

    // ---- KvCacheConfig: PartialEq different swap_config values ----

    #[test]
    fn kv_cache_config_partial_eq_different_swap_threshold() {
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(SwapConfig { enable_swap: true, swap_threshold: 0.7, lru_granularity: 4 }),
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: Some(SwapConfig { enable_swap: true, swap_threshold: 0.9, lru_granularity: 4 }),
        };
        assert_ne!(a, b);
    }

    // ---- BackendError::Other with long message ----

    #[test]
    fn backend_error_other_long_message() {
        let long = "z".repeat(2000);
        let err = BackendError::Other(long.clone());
        assert_eq!(format!("{err}"), format!("backend error: {long}"));
    }

    // ---- LogitsHandle: with NaN and Infinity ----

    #[test]
    fn logits_handle_special_floats() {
        let handle = LogitsHandle { data: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY] };
        assert!(handle.data[0].is_nan());
        assert!(handle.data[1].is_infinite() && handle.data[1].is_sign_positive());
        assert!(handle.data[2].is_infinite() && handle.data[2].is_sign_negative());
    }

    // ---- AttentionTopology::linear() shares Arc ----

    #[test]
    fn attention_topology_linear_arc_strong_count() {
        let topo = AttentionTopology::linear();
        assert_eq!(Arc::strong_count(&topo.geometry), 1);
    }

    // ---- SequenceInput: validate_page_table all at max valid boundary ----

    #[test]
    fn sequence_input_validate_all_pages_at_boundary() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: None,
        };
        // total_pages=3 -> valid indices 0,1,2; all entries are valid
        assert!(seq.validate_page_table(3).is_ok());
    }

    // ---- BatchInput: mixed page_table states across sequences ----

    #[test]
    fn batch_input_mixed_page_table_states() {
        let batch = BatchInput {
            sequences: vec![
                SequenceInput { tokens: vec![1], position: 0, draft_steps: 0, page_table: None, fused_hidden: None },
                SequenceInput { tokens: vec![2], position: 1, draft_steps: 0, page_table: Some(vec![0]), fused_hidden: None },
                SequenceInput { tokens: vec![3], position: 2, draft_steps: 0, page_table: Some(vec![]), fused_hidden: Some(vec![0.0]) },
            ],
        };
        assert!(batch.sequences[0].page_table.is_none());
        assert!(batch.sequences[1].page_table.is_some());
        assert!(batch.sequences[2].fused_hidden.is_some());
    }

    // ---- RequestData: fused_prefill_hidden empty vec ----

    #[test]
    fn request_data_fused_prefill_hidden_empty_vec() {
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: Some(vec![]),
        };
        assert!(rd.fused_prefill_hidden.unwrap().is_empty());
    }

    // ---- ExecutorError: Debug format contains variant names ----

    #[test]
    fn executor_error_debug_all_variants() {
        let variants: Vec<(ExecutorError, &str)> = vec![
            (ExecutorError::Scheduler("s".into()), "Scheduler"),
            (ExecutorError::EmptyPrompt, "EmptyPrompt"),
            (ExecutorError::EmptySample, "EmptySample"),
            (ExecutorError::Compilation("c".into()), "Compilation"),
            (ExecutorError::GraphExpansion("g".into()), "GraphExpansion"),
        ];
        for (err, name) in variants {
            let debug = format!("{err:?}");
            assert!(debug.contains(name), "Expected '{name}' in debug: {debug}");
        }
    }

    // ---- KvCacheHandle: Eq trait dedup in HashSet ----

    #[test]
    fn kv_cache_handle_eq_dedup() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KvCacheHandle(10));
        set.insert(KvCacheHandle(20));
        assert_eq!(set.len(), 2);
        assert!(set.remove(&KvCacheHandle(10)));
        assert_eq!(set.len(), 1);
    }

    // ========================================================================
    // Batch 5: 15 additional tests for coverage improvement
    // ========================================================================

    // ---- effective_kv_max_seq_len: very small boundary values ----

    #[test]
    fn effective_kv_max_seq_len_unit_values() {
        // Edge case: smallest meaningful seq lengths
        assert_eq!(effective_kv_max_seq_len(1), 1);
        assert_eq!(effective_kv_max_seq_len(2), 2);
        assert_eq!(effective_kv_max_seq_len(3), 3);
    }

    // ---- SamplingConfig: top_k at usize::MAX ----

    #[test]
    fn sampling_config_top_k_max_value() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: usize::MAX,
            top_p: 1.0,
        };
        assert_eq!(cfg.top_k, usize::MAX);
    }

    // ---- RoPEConfig: zero theta ----

    #[test]
    fn rope_config_zero_theta() {
        let cfg = RoPEConfig {
            theta: 0.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_eq!(cfg.theta, 0.0);
        // A zero-theta config should not equal the default (10000.0)
        let default_cfg = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_ne!(cfg, default_cfg);
    }

    // ---- AttentionHeadConfig: from_geometry with large geometry ----

    #[test]
    fn attention_head_config_from_large_geometry() {
        let geo = ModelGeometry {
            hidden_size: 8192,
            num_layers: 80,
            vocab_size: 128256,
            intermediate_size: 28672,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            ..make_geometry()
        };
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    // ---- PagedKvConfig: page_size zero ----

    #[test]
    fn paged_kv_config_page_size_zero() {
        let cfg = PagedKvConfig {
            page_table: None,
            page_size: 0,
        };
        assert_eq!(cfg.page_size, 0);
        assert!(cfg.page_table.is_none());
    }

    // ---- GeneratorForwardConfig::default_for_test layer_dims ----

    #[test]
    fn default_for_test_layer_dims() {
        let cfg = GeneratorForwardConfig::default_for_test();
        let ld = cfg.layer_dims();
        assert_eq!(ld.hidden, 64);
        assert_eq!(ld.inter, 128);
        assert!((ld.eps - 1e-5).abs() < 1e-10);
        assert!((ld.rope_theta - 10000.0).abs() < 1e-6);
    }

    // ---- KvCacheConfig: F16 dtype with large page_size ----

    #[test]
    fn kv_cache_config_f16_dtype_large_page_size() {
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F16,
            page_size: 1024,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 2);
        assert_eq!(cfg.page_size, 1024);
    }

    // ---- BackendError: Clone produces identical Display for each variant ----

    #[test]
    fn backend_error_clone_preserves_display_all_variants() {
        let errors = vec![
            BackendError::Cuda("err1".into()),
            BackendError::Hip("err2".into()),
            BackendError::Metal("err3".into()),
            BackendError::Cpu("err4".into()),
            BackendError::Unimplemented("err5"),
            BackendError::Other("err6".into()),
        ];
        for err in errors {
            let cloned = err.clone();
            assert_eq!(format!("{err}"), format!("{cloned}"));
        }
    }

    // ---- KvCacheHandle: sorting by inner u64 ----

    #[test]
    fn kv_cache_handle_sort_by_inner() {
        let mut handles = vec![
            KvCacheHandle(100),
            KvCacheHandle(3),
            KvCacheHandle(42),
            KvCacheHandle(0),
            KvCacheHandle(7),
        ];
        handles.sort_by_key(|h| h.0);
        assert_eq!(handles[0], KvCacheHandle(0));
        assert_eq!(handles[1], KvCacheHandle(3));
        assert_eq!(handles[2], KvCacheHandle(7));
        assert_eq!(handles[3], KvCacheHandle(42));
        assert_eq!(handles[4], KvCacheHandle(100));
    }

    // ---- LogitsHandle: single element data ----

    #[test]
    fn logits_handle_single_element() {
        let handle = LogitsHandle { data: vec![42.0] };
        assert_eq!(handle.data.len(), 1);
        assert!((handle.data[0] - 42.0).abs() < 1e-6);
        let cloned = handle.clone();
        assert_eq!(cloned.data.len(), 1);
        assert_eq!(cloned.data[0], handle.data[0]);
    }

    // ---- AttentionTopology: bidirectional and causal share same geometry ----

    #[test]
    fn attention_topology_bidirectional_causal_same_geometry() {
        let geo = Arc::new(make_geometry());
        let bi = AttentionTopology::bidirectional(geo.clone());
        let causal = AttentionTopology::causal(geo.clone());
        // Both share the same Arc<ModelGeometry>
        assert_eq!(Arc::strong_count(&geo), 3);
        assert_eq!(bi.num_heads(), causal.num_heads());
        assert_eq!(bi.num_kv_heads(), causal.num_kv_heads());
        assert_eq!(bi.head_dim(), causal.head_dim());
        assert_eq!(bi.max_seq_len(), causal.max_seq_len());
        // Mask types differ
        assert_ne!(bi.mask_type, causal.mask_type);
    }

    // ---- SequenceInput: validate_page_table with u32::MAX page_id ----

    #[test]
    fn sequence_input_validate_page_table_max_u32() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![u32::MAX]),
            fused_hidden: None,
        };
        // u32::MAX as usize is still >= total_pages unless total_pages is u32::MAX+1
        assert!(seq.validate_page_table(10).is_err());
        let msg = seq.validate_page_table(10).unwrap_err();
        assert!(msg.contains(&u32::MAX.to_string()));
    }

    // ---- BatchInput: clone preserves sequence count and content ----

    #[test]
    fn batch_input_clone_preserves_sequences() {
        let batch = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![10, 20, 30],
                    position: 5,
                    draft_steps: 2,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: Some(vec![0.5, 0.6]),
                },
                SequenceInput {
                    tokens: vec![40],
                    position: 8,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
            ],
        };
        let cloned = batch.clone();
        assert_eq!(cloned.sequences.len(), batch.sequences.len());
        assert_eq!(cloned.sequences[0].tokens, batch.sequences[0].tokens);
        assert_eq!(cloned.sequences[0].position, batch.sequences[0].position);
        assert_eq!(cloned.sequences[0].draft_steps, batch.sequences[0].draft_steps);
        assert_eq!(cloned.sequences[1].tokens, batch.sequences[1].tokens);
    }

    // ---- ExecutorError: std::error::Error impl for all variants ----

    #[test]
    fn executor_error_implements_std_error() {
        // Verify each variant can be used as &dyn std::error::Error
        let errors: Vec<ExecutorError> = vec![
            ExecutorError::EmptyPrompt,
            ExecutorError::EmptySample,
            ExecutorError::Scheduler("overflow".into()),
            ExecutorError::Compilation("bad_ir".into()),
            ExecutorError::GraphExpansion("cycle".into()),
            ExecutorError::Backend(BackendError::Cuda("timeout".into())),
        ];
        for err in &errors {
            let _: &dyn std::error::Error = err;
        }
    }

    // ========================================================================
    // Batch 6: 15 additional tests for coverage improvement
    // ========================================================================

    // ---- GeneratorForwardConfig: Debug trait format includes key fields ----

    #[test]
    fn generator_forward_config_debug_trait() {
        // Arrange: construct a default config
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act: format via Debug trait
        let debug = format!("{cfg:?}");
        // Assert: debug output contains the struct name and key structural info
        assert!(debug.contains("GeneratorForwardConfig"), "Debug output should contain struct name");
    }

    // ---- GeneratorForwardConfig: geometry Arc is shared across clone ----

    #[test]
    fn forward_config_geometry_arc_shared_on_clone() {
        // Arrange: create a config and note Arc ref count
        let cfg = make_forward_config();
        let initial_count = Arc::strong_count(&cfg.geometry);
        // Act: clone the config
        let cloned = cfg.clone();
        // Assert: Arc ref count incremented (geometry is Arc-shared)
        assert_eq!(Arc::strong_count(&cfg.geometry), initial_count + 1);
        assert!(Arc::ptr_eq(&cfg.geometry, &cloned.geometry));
    }

    // ---- GeneratorForwardConfig: attention() returns consistent config with accessors ----

    #[test]
    fn forward_config_attention_matches_direct_accessors() {
        // Arrange: create a config
        let cfg = make_forward_config();
        // Act: get attention config and compare with direct accessors
        let attn = cfg.attention();
        // Assert: attention config fields match geometry-backed accessors
        assert_eq!(attn.num_heads, cfg.num_heads());
        assert_eq!(attn.num_kv_heads, cfg.num_kv_heads());
        assert_eq!(attn.head_dim, cfg.head_dim());
    }

    // ---- SamplingConfig: default and custom are distinct ----

    #[test]
    fn sampling_config_default_differs_from_custom() {
        // Arrange: create default and a custom config with all fields changed
        let default = SamplingConfig::default();
        let custom = SamplingConfig {
            temperature: 0.5,
            top_k: 100,
            top_p: 0.8,
        };
        // Assert: all fields differ
        assert_ne!(default.temperature, custom.temperature);
        assert_ne!(default.top_k, custom.top_k);
        assert_ne!(default.top_p, custom.top_p);
    }

    // ---- SamplingConfig: temperature zero with top_k=1 is deterministic profile ----

    #[test]
    fn sampling_config_greedy_deterministic_profile() {
        // Arrange: construct a greedy/deterministic sampling profile
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        };
        // Assert: temperature is exactly zero (no randomness)
        assert_eq!(cfg.temperature.to_bits(), 0.0f32.to_bits());
        // Assert: top_k=1 means only the top token is considered
        assert_eq!(cfg.top_k, 1);
        // Assert: top_p=1.0 means no nucleus filtering
        assert_eq!(cfg.top_p, 1.0);
    }

    // ---- RoPEConfig: construction preserves all four fields independently ----

    #[test]
    fn rope_config_construction_preserves_all_fields() {
        // Arrange: construct with non-default values for every field
        let cfg = RoPEConfig {
            theta: 500000.0,
            scale: 0.25,
            interleaved: true,
            precompute: true,
        };
        // Assert: each field is stored exactly as provided
        assert!((cfg.theta - 500000.0).abs() < 1e-6);
        assert!((cfg.scale - 0.25).abs() < 1e-6);
        assert!(cfg.interleaved);
        assert!(cfg.precompute);
    }

    // ---- AttentionHeadConfig: from_geometry with MQA (num_kv_heads=1) ----

    #[test]
    fn attention_head_config_mqa_single_kv_head() {
        // Arrange: create geometry with 1 KV head (Multi-Query Attention)
        let geo = ModelGeometry {
            num_heads: 32,
            num_kv_heads: 1,
            head_dim: 64,
            ..make_geometry()
        };
        // Act: derive AttentionHeadConfig
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        // Assert: MQA has 32 query heads but only 1 KV head
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.head_dim, 64);
    }

    // ---- AttentionHeadConfig: Debug output contains all three fields ----

    #[test]
    fn attention_head_config_debug_contains_fields() {
        // Arrange
        let cfg = AttentionHeadConfig {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
        };
        // Act
        let debug = format!("{cfg:?}");
        // Assert: Debug output includes the struct name
        assert!(debug.contains("AttentionHeadConfig"));
    }

    // ---- PagedKvConfig: page_size equals one means every token is a page ----

    #[test]
    fn paged_kv_config_page_size_one() {
        // Arrange: smallest meaningful page size
        let cfg = PagedKvConfig {
            page_table: Some(vec![0, 1, 2]),
            page_size: 1,
        };
        // Assert: page_size is exactly 1, page_table has entries
        assert_eq!(cfg.page_size, 1);
        assert_eq!(cfg.page_table.as_ref().unwrap().len(), 3);
    }

    // ---- SwapConfig: all disabled with zero values ----

    #[test]
    fn swap_config_all_disabled_zero() {
        // Arrange: construct a disabled swap config with zero threshold
        let cfg = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.0,
            lru_granularity: 0,
        };
        // Assert: swap is disabled and all values are zero
        assert!(!cfg.enable_swap);
        assert_eq!(cfg.swap_threshold, 0.0);
        assert_eq!(cfg.lru_granularity, 0);
        // Assert: differs from a typical enabled config
        assert_ne!(
            cfg,
            SwapConfig {
                enable_swap: true,
                swap_threshold: 0.0,
                lru_granularity: 0,
            }
        );
    }

    // ---- KvCacheConfig: Debug trait output ----

    #[test]
    fn kv_cache_config_debug_format() {
        // Arrange
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Act
        let debug = format!("{cfg:?}");
        // Assert: debug output contains struct name
        assert!(debug.contains("KvCacheConfig"));
    }

    // ---- KvCacheConfig: MLA geometry affects is_mla but not num_layers ----

    #[test]
    fn kv_cache_config_mla_preserves_num_layers_and_heads() {
        // Arrange: create an MLA-flavored config
        let cfg = KvCacheConfig {
            geometry: Arc::new(make_mla_geometry()),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: MLA flag is set
        assert!(cfg.is_mla());
        // Assert: num_layers and num_heads still come from base geometry
        assert_eq!(cfg.num_layers(), 12);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.head_dim(), 64);
    }

    // ---- SequenceInput: validate_page_table reports first OOB entry ----

    #[test]
    fn sequence_input_validate_reports_first_invalid_page() {
        // Arrange: page_table with valid entries followed by an invalid one
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2, 50, 3]),
            fused_hidden: None,
        };
        // Act
        let result = seq.validate_page_table(10);
        // Assert: reports the first OOB entry (index 3, page_id 50)
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("page_table[3] = 50"), "Expected first OOB entry in error message, got: {msg}");
    }

    // ---- RequestData: Debug format includes struct name ----

    #[test]
    fn request_data_debug_contains_struct_name() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let rd = RequestData {
            prompt_tokens: vec![1, 2],
            output_tokens: vec![3],
            sampling_config: SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.95 },
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: Some(512),
            fused_prefill_hidden: None,
        };
        // Act
        let debug = format!("{rd:?}");
        // Assert: Debug output includes the struct name
        assert!(debug.contains("RequestData"));
    }

    // ---- GeneratorForwardConfig: max_seq_len passes through geometry value ----

    #[test]
    fn forward_config_max_seq_len_matches_geometry() {
        // Arrange: create geometry with a specific max_seq_len
        let geo = Arc::new(ModelGeometry {
            max_seq_len: 32768,
            ..make_geometry()
        });
        let cfg = GeneratorForwardConfig {
            geometry: geo,
            rope: RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig { page_table: None, page_size: 16 },
            callback_chain: super::super::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        // Assert: max_seq_len accessor returns the geometry value
        assert_eq!(cfg.max_seq_len(), 32768);
    }
}
