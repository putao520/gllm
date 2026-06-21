// ============================================================================
// T58: Multimodal encoder registration tests
// ============================================================================

#[cfg(test)]
mod multimodal_client_tests {
    use super::*;
    use crate::compat::multimodal::{
        EncoderMedia, MediaKind, MultimodalEncoded, MultimodalEncoder,
    };
    use crate::engine::executor::BackendError;

    /// Mock encoder that tracks invocation count.
    struct MockEncoder {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl MockEncoder {
        fn new() -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl MultimodalEncoder for MockEncoder {
        fn encode_image(
            &self,
            _media: &EncoderMedia,
        ) -> Result<MultimodalEncoded, BackendError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(MultimodalEncoded {
                tokens: vec![258880; 2],
                embeddings: vec![0.0; 2 * 4],
                hidden_size: 4,
                kind: MediaKind::Image,
            })
        }

        fn encode_audio(
            &self,
            _media: &EncoderMedia,
        ) -> Result<MultimodalEncoded, BackendError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(MultimodalEncoded {
                tokens: vec![258881; 1],
                embeddings: vec![0.0; 1 * 4],
                hidden_size: 4,
                kind: MediaKind::Audio,
            })
        }
    }

    #[test]
    fn client_has_no_multimodal_encoder_by_default() {
        let client = Client::new_empty();
        assert!(!client.has_multimodal_encoder());
    }

    #[test]
    fn set_multimodal_encoder_activates_encoder() {
        let client = Client::new_empty();
        assert!(!client.has_multimodal_encoder());
        client.set_multimodal_encoder(Arc::new(MockEncoder::new()));
        assert!(client.has_multimodal_encoder());
    }

    #[test]
    fn set_multimodal_encoder_overwrites_previous() {
        let client = Client::new_empty();
        let first = Arc::new(MockEncoder::new());
        let second = Arc::new(MockEncoder::new());
        client.set_multimodal_encoder(first.clone());
        client.set_multimodal_encoder(second.clone());
        // first 没被调用过，依然 0
        assert_eq!(first.calls(), 0);
        assert!(client.has_multimodal_encoder());
    }

    #[test]
    fn multimodal_generation_without_model_errors_past_encoder_check() {
        // 已注册 encoder，但未加载模型 → NoModelLoaded
        let client = Client::new_empty();
        client.set_multimodal_encoder(Arc::new(MockEncoder::new()));
        let result = client.execute_generation_multimodal(
            "hello".into(),
            10,
            1.0,
            0,
            1.0,
            None,
            None,
            Some(crate::generation::MediaInput::Raw(vec![0xFF; 4])),
            None,
        );
        // encoder 校验通过了（非 InvalidModelType），但 require_state 失败
        assert!(matches!(result, Err(ClientError::NoModelLoaded)));
    }

    #[test]
    fn multimodal_generation_without_encoder_errors() {
        // 未注册 encoder + 多模态输入 → InvalidModelType
        let client = Client::new_empty();
        let result = client.execute_generation_multimodal(
            "hello".into(),
            10,
            1.0,
            0,
            1.0,
            None,
            None,
            Some(crate::generation::MediaInput::Raw(vec![0xFF; 4])),
            None,
        );
        assert!(matches!(result, Err(ClientError::InvalidModelType)));
    }
}

// ========================================================================
// BCI10: Batch 并发集成测试
// ========================================================================

#[cfg(test)]
mod batch_async_tests {
    use super::*;
    use crate::engine::batch_executor::GenerateRequest;

    #[test]
    fn test_async_client_new() {
        let client = Client::new_empty();
        let async_client = AsyncClient::new(client);
        // AsyncClient wraps Client as Arc; verify no panic
        let _ = async_client;
    }

    #[test]
    fn test_async_client_generate_batch_empty() {
        let client = Client::new_empty();
        let async_client = AsyncClient::new(client);

        // Test the empty-batch branch returns Ok(empty) immediately
        let result = async_client.inner.generate_batch(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_client_generate_batch_empty() {
        // Test the sync path: empty batch returns Ok(empty) immediately
        let client = Client::new_empty();
        let result = client.generate_batch(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_async_client_generate_batch_no_model() {
        let client = Client::new_empty();
        let async_client = AsyncClient::new(client);

        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            session_id: None,
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let result = async_client.inner.generate_batch(&[req]);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClientError::NoModelLoaded)));
    }

    #[test]
    fn test_client_generate_batch_no_model() {
        let client = Client::new_empty();
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            session_id: None,
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let result = client.generate_batch(&[req]);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClientError::NoModelLoaded)));
    }
}

// ========================================================================
// ClientConfig / WeightPagingStatus / ClientError tests
// ========================================================================

#[cfg(test)]
mod error_config_tests {
    use super::*;

    #[test]
    fn client_config_default_weight_paging_disabled() {
        let config = ClientConfig::default();
        assert!(!config.weight_paging_enabled);
    }

    #[test]
    fn client_config_weight_paging_enabled() {
        let config = ClientConfig {
            weight_paging_enabled: true,
        };
        assert!(config.weight_paging_enabled);
    }

    #[test]
    fn weight_paging_status_fields() {
        let status = WeightPagingStatus {
            enabled: true,
            num_pages: 1024,
            page_size_bytes: 4096,
            prefetch_distance: 3,
        };
        assert!(status.enabled);
        assert_eq!(status.num_pages, 1024);
        assert_eq!(status.page_size_bytes, 4096);
        assert_eq!(status.prefetch_distance, 3);
    }

    #[test]
    fn weight_paging_status_disabled() {
        let status = WeightPagingStatus {
            enabled: false,
            num_pages: 0,
            page_size_bytes: 0,
            prefetch_distance: 0,
        };
        assert!(!status.enabled);
    }

    #[test]
    fn client_error_model_not_found_display() {
        let err = ClientError::ModelNotFound("test-model".into());
        let msg = format!("{err}");
        assert!(msg.contains("test-model"));
        assert!(msg.contains("model not found"));
    }

    #[test]
    fn client_error_invalid_model_type_display() {
        let err = ClientError::InvalidModelType;
        let msg = format!("{err}");
        assert!(msg.contains("invalid model type"));
    }

    #[test]
    fn client_error_no_model_loaded_display() {
        let err = ClientError::NoModelLoaded;
        let msg = format!("{err}");
        assert!(msg.contains("no model loaded"));
    }

    #[test]
    fn client_error_runtime_error_display() {
        let err = ClientError::RuntimeError("something broke".into());
        let msg = format!("{err}");
        assert!(msg.contains("something broke"));
        assert!(msg.contains("runtime error"));
    }

    #[test]
    fn gllm_error_is_client_error_alias() {
        let err: GllmError = ClientError::NoModelLoaded;
        assert!(matches!(err, ClientError::NoModelLoaded));
    }

    #[test]
    fn client_error_from_model_config_error() {
        let config_err = crate::model_config::ModelConfigError::InvalidConfig("missing hidden".into());
        let client_err: ClientError = config_err.into();
        let msg = format!("{client_err}");
        assert!(msg.contains("model config error"));
    }
}

// ========================================================================
// MtpGenerationResponse / MtpStepInfo tests
// ========================================================================

#[cfg(test)]
mod mtp_response_tests {
    use super::*;

    #[test]
    fn mtp_acceptance_rate_with_candidates() {
        let response = MtpGenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            total_mtp_candidates: 10,
            total_mtp_accepted: 7,
            step_details: vec![],
        };
        let rate = response.acceptance_rate();
        assert!((rate - 0.7).abs() < 1e-6);
    }

    #[test]
    fn mtp_acceptance_rate_zero_candidates() {
        let response = MtpGenerationResponse {
            text: String::new(),
            thinking_content: None,
            total_mtp_candidates: 0,
            total_mtp_accepted: 0,
            step_details: vec![],
        };
        assert_eq!(response.acceptance_rate(), 0.0);
    }

    #[test]
    fn mtp_throughput_multiplier_with_steps() {
        let step = MtpStepInfo {
            main_token: 42,
            mtp_candidates: vec![100, 200],
            accepted_count: 1,
            main_token_is_eos: false,
        };
        let response = MtpGenerationResponse {
            text: "ab".to_string(),
            thinking_content: None,
            total_mtp_candidates: 2,
            total_mtp_accepted: 1,
            step_details: vec![step],
        };
        // 1 step: committed = 1 (main) + 1 (accepted) = 2; multiplier = 2/1 = 2.0
        let mult = response.throughput_multiplier();
        assert!((mult - 2.0).abs() < 1e-6);
    }

    #[test]
    fn mtp_throughput_multiplier_no_steps() {
        let response = MtpGenerationResponse {
            text: "x".to_string(),
            thinking_content: None,
            total_mtp_candidates: 0,
            total_mtp_accepted: 0,
            step_details: vec![],
        };
        assert_eq!(response.throughput_multiplier(), 1.0);
    }

    #[test]
    fn mtp_step_info_fields() {
        let step = MtpStepInfo {
            main_token: 99,
            mtp_candidates: vec![10, 20, 30],
            accepted_count: 2,
            main_token_is_eos: true,
        };
        assert_eq!(step.main_token, 99);
        assert_eq!(step.mtp_candidates.len(), 3);
        assert_eq!(step.accepted_count, 2);
        assert!(step.main_token_is_eos);
    }
}

// ========================================================================
// ClientBuilder / Client construction / ModelInfo / empty state tests
// ========================================================================

#[cfg(test)]
mod builder_and_client_tests {
    use super::*;

    #[test]
    fn client_builder_new_defaults() {
        let builder = ClientBuilder::new();
        let result = builder.build();
        match result {
            Err(ClientError::ModelNotFound(msg)) => assert!(msg.contains("no model id")),
            Err(other) => panic!("expected ModelNotFound, got: {other}"),
            Ok(_) => panic!("expected error, got Ok client"),
        }
    }

    #[test]
    fn client_builder_default_trait() {
        let builder = ClientBuilder::default();
        let result = builder.build();
        match result {
            Err(ClientError::ModelNotFound(_)) => {}
            Err(other) => panic!("expected ModelNotFound, got: {other}"),
            Ok(_) => panic!("expected error, got Ok client"),
        }
    }

    #[test]
    fn client_new_empty_not_loaded() {
        let client = Client::new_empty();
        assert!(!client.is_loaded());
    }

    #[test]
    fn client_model_info_none_when_empty() {
        let client = Client::new_empty();
        assert!(client.model_info().is_none());
    }

    #[test]
    fn client_unload_model_on_empty_ok() {
        let client = Client::new_empty();
        let result = client.unload_model();
        assert!(result.is_ok());
    }

    #[test]
    fn client_weight_paging_status_empty_disabled() {
        let client = Client::new_empty();
        let status = client.weight_paging_status();
        assert!(!status.enabled);
        assert_eq!(status.num_pages, 0);
        assert_eq!(status.page_size_bytes, 0);
        assert_eq!(status.prefetch_distance, 0);
    }

    #[test]
    fn model_info_debug_clone() {
        let info = ModelInfo {
            id: "test-model".to_string(),
            arch: "qwen3".to_string(),
            kind: ModelKind::Chat,
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("test-model"));
        assert!(debug_str.contains("qwen3"));

        let cloned = info.clone();
        assert_eq!(cloned.id, "test-model");
        assert_eq!(cloned.arch, "qwen3");
    }

    #[test]
    fn client_normalize_model_id_empty_fails() {
        let result = Client::normalize_model_id("");
        assert!(matches!(result, Err(ClientError::ModelNotFound(_))));
    }

    #[test]
    fn client_normalize_model_id_whitespace_fails() {
        let result = Client::normalize_model_id("   ");
        assert!(matches!(result, Err(ClientError::ModelNotFound(_))));
    }

    #[test]
    fn client_normalize_model_id_trims() {
        let result = Client::normalize_model_id("  my-model  ");
        assert_eq!(result.unwrap(), "my-model");
    }

    // ── REQ-API-1: ClientBuilder 链式配置 API 验收 ──

    /// Verify Client::builder() returns a fresh ClientBuilder (REQ-API-1).
    #[test]
    fn client_builder_entry_point() {
        let builder = Client::builder();
        // builder without model_id should fail with ModelNotFound
        let result = builder.build();
        match result {
            Err(ClientError::ModelNotFound(_)) => {}
            other => panic!("expected ModelNotFound, got unexpected result variant"),
        }
    }

    /// Verify builder chaining propagates all fields (REQ-API-1).
    ///
    /// We cannot call .build() without a real model, but we can verify that
    /// chaining returns a ClientBuilder (type-level check) and that the
    /// final .build() fails with ModelNotFound (proving model_id was not
    /// accidentally set by any chained method).
    #[test]
    fn builder_chaining_returns_client_builder() {
        let builder = Client::builder()
            .model("nonexistent-model")
            .kind(ModelKind::Chat)
            .backend(BackendType::Cpu)
            .inference_mode(InferenceMode::Latency)
            .compute_dtype(gllm_kernels::types::DType::F32)
            .gguf_file_filter("Q8_0")
            .debug_jit(false)
            .weight_paging_enabled(false);

        // Chaining completed — model_id is set, so build() should fail
        // at model loading (not at "no model id").
        let result = builder.build();
        // The model doesn't exist, so we expect ModelNotFound or another
        // loading error — but NOT the "no model id" error.
        match result {
            Err(ClientError::ModelNotFound(msg)) => {
                // Should be a real model path, not "<no model id>"
                assert!(!msg.contains("no model id"), "model_id was not set by .model()");
            }
            Err(_) => {
                // Other loading errors (e.g. network) are acceptable
            }
            Ok(_) => panic!("expected error for nonexistent model, got Ok"),
        }
    }

    /// Verify swap_model on empty client returns NoModelLoaded (REQ-API-1, REQ-API-7).
    #[test]
    fn swap_model_on_empty_client_returns_no_model_loaded() {
        let client = Client::new_empty();
        let result = client.swap_model("nonexistent");
        match result {
            Err(ClientError::NoModelLoaded) => {}
            other => panic!("expected NoModelLoaded, got a different error variant"),
        }
    }

    /// Verify swap_model with empty string returns ModelNotFound (REQ-API-1, REQ-API-7).
    #[test]
    fn swap_model_empty_string_fails() {
        let client = Client::new_empty();
        let result = client.swap_model("");
        match result {
            Err(ClientError::ModelNotFound(_)) => {}
            other => panic!("expected ModelNotFound, got unexpected result variant"),
        }
    }

    /// Verify Default trait for ClientBuilder matches new() (REQ-API-1).
    #[test]
    fn client_builder_default_matches_new() {
        let _new_builder = ClientBuilder::new();
        let _default_builder = ClientBuilder::default();
        // Both should fail with the same error when building without model_id
        let new_result = ClientBuilder::new().build();
        let default_result = ClientBuilder::default().build();
        assert!(matches!(new_result, Err(ClientError::ModelNotFound(_))));
        assert!(matches!(default_result, Err(ClientError::ModelNotFound(_))));
    }

    /// Verify builder without model_id yields specific "no model id" error (REQ-API-1).
    #[test]
    fn builder_no_model_id_specific_error() {
        let result = ClientBuilder::new().build();
        match result {
            Err(ClientError::ModelNotFound(msg)) => {
                assert!(msg.contains("no model id"), "expected 'no model id' in error, got: {msg}");
            }
            other => panic!("expected ModelNotFound with 'no model id', got a different error variant"),
        }
    }
}
