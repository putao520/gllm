//! Weight loader facade (re-exports loader).

pub use crate::loader::{
    ChecksumPolicy, Loader, LoaderConfig, LoaderError, ParallelPolicy, TensorInfo, UploadedTensor,
    WeightsHandle,
};

// ---------------------------------------------------------------------------
// SharedKvRef (§P1.1): helpers so loaders can tolerate missing K/V weights
// on shared-consumer layers (Gemma 4 E2B / E4B).
// ---------------------------------------------------------------------------

/// Returns `true` when layer `layer_i` is a KV-sharing *consumer* and the
/// checkpoint therefore does NOT contain its `self_attn.k_proj.weight` /
/// `self_attn.v_proj.weight`.
///
/// Non-shared layers still require the K/V projections and the loader must
/// error if they are absent.
#[inline]
pub fn layer_allows_missing_kv_weights(
    layer_i: usize,
    num_hidden_layers: usize,
    num_kv_shared_layers: usize,
) -> bool {
    if num_kv_shared_layers == 0 || layer_i >= num_hidden_layers {
        return false;
    }
    layer_i + num_kv_shared_layers >= num_hidden_layers
}

/// True when `role` describes a K/V projection weight (the weights that shared
/// consumer layers are allowed to omit).
#[inline]
pub fn is_kv_projection_role(role: crate::manifest::TensorRole) -> bool {
    matches!(
        role,
        crate::manifest::TensorRole::AttentionKey | crate::manifest::TensorRole::AttentionValue
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TensorRole;
    use std::collections::HashSet;

    // ── layer_allows_missing_kv_weights ──────────────────────────────────

    #[test]
    fn consumer_layers_allow_missing_kv_weights_gemma4_e2b() {
        // Gemma 4 E2B: 32 layers, last 20 share KV.
        let layers = 32;
        let shared = 20;
        for l in 0..12 {
            assert!(!layer_allows_missing_kv_weights(l, layers, shared));
        }
        for l in 12..32 {
            assert!(layer_allows_missing_kv_weights(l, layers, shared));
        }
    }

    #[test]
    fn consumer_layers_allow_missing_kv_weights_gemma4_e4b() {
        // Gemma 4 E4B: 34 layers, last 18 share KV.
        let layers = 34;
        let shared = 18;
        for l in 0..16 {
            assert!(!layer_allows_missing_kv_weights(l, layers, shared));
        }
        for l in 16..34 {
            assert!(layer_allows_missing_kv_weights(l, layers, shared));
        }
    }

    #[test]
    fn no_shared_layers_disables_relaxation() {
        for l in 0..32 {
            assert!(!layer_allows_missing_kv_weights(l, 32, 0));
        }
    }

    #[test]
    fn layer_equals_num_hidden_layers_is_out_of_range() {
        assert!(!layer_allows_missing_kv_weights(32, 32, 20));
    }

    #[test]
    fn layer_exceeds_num_hidden_layers_returns_false() {
        assert!(!layer_allows_missing_kv_weights(100, 32, 20));
        assert!(!layer_allows_missing_kv_weights(usize::MAX, 32, 20));
    }

    #[test]
    fn zero_hidden_layers_disables_relaxation() {
        assert!(!layer_allows_missing_kv_weights(0, 0, 10));
        assert!(!layer_allows_missing_kv_weights(5, 0, 10));
    }

    #[test]
    fn boundary_layer_is_exact_threshold() {
        // layer_i + 20 >= 32 => layer_i >= 12
        assert!(!layer_allows_missing_kv_weights(11, 32, 20));
        assert!(layer_allows_missing_kv_weights(12, 32, 20));
    }

    #[test]
    fn all_layers_shared() {
        assert!(layer_allows_missing_kv_weights(0, 8, 8));
        assert!(layer_allows_missing_kv_weights(7, 8, 8));
    }

    #[test]
    fn single_layer_single_shared() {
        assert!(layer_allows_missing_kv_weights(0, 1, 1));
    }

    #[test]
    fn single_layer_zero_shared() {
        assert!(!layer_allows_missing_kv_weights(0, 1, 0));
    }

    #[test]
    fn large_values_without_overflow() {
        // Use values large enough to test boundary logic but where
        // layer_i + num_kv_shared_layers does not overflow usize.
        let big = 1_000_000;
        assert!(layer_allows_missing_kv_weights(500_000, big, big));
        assert!(layer_allows_missing_kv_weights(0, big, big));
        assert!(!layer_allows_missing_kv_weights(0, big, big - 1));
    }

    #[test]
    fn layer_zero_with_large_shared_count() {
        // layer 0 + large shared should be true if shared >= num_hidden_layers
        assert!(layer_allows_missing_kv_weights(0, 100, 100));
        assert!(!layer_allows_missing_kv_weights(0, 100, 99));
    }

    // ── is_kv_projection_role ────────────────────────────────────────────

    #[test]
    fn kv_projection_role_classification() {
        assert!(is_kv_projection_role(TensorRole::AttentionKey));
        assert!(is_kv_projection_role(TensorRole::AttentionValue));
        assert!(!is_kv_projection_role(TensorRole::AttentionQuery));
        assert!(!is_kv_projection_role(TensorRole::Embedding));
    }

    #[test]
    fn non_kv_roles_all_return_false() {
        let non_kv_roles = [
            TensorRole::Embedding,
            TensorRole::OutputHead,
            TensorRole::FinalNorm,
            TensorRole::AttentionQuery,
            TensorRole::AttentionFusedQkv,
            TensorRole::AttentionOutput,
            TensorRole::InputNorm,
            TensorRole::PostAttnNorm,
            TensorRole::FfnGate,
            TensorRole::FfnUp,
            TensorRole::FfnDown,
            TensorRole::MoEGate,
            TensorRole::Rope,
        ];
        for role in non_kv_roles {
            assert!(!is_kv_projection_role(role), "expected false for {role:?}");
        }
    }

    #[test]
    fn tensor_role_all_non_kv_variants_exhaustive() {
        let non_kv: Vec<TensorRole> = vec![
            TensorRole::Embedding,
            TensorRole::OutputHead,
            TensorRole::FinalNorm,
            TensorRole::ClassifierDense,
            TensorRole::ClassifierOutProj,
            TensorRole::PatchEmbed,
            TensorRole::PositionEmbedding,
            TensorRole::AttentionQuery,
            TensorRole::AttentionFusedQkv,
            TensorRole::AttentionOutput,
            TensorRole::AttentionQNorm,
            TensorRole::AttentionKNorm,
            TensorRole::AttentionSinks,
            TensorRole::InputNorm,
            TensorRole::PostAttnNorm,
            TensorRole::LayerNorm,
            TensorRole::FfnGate,
            TensorRole::FfnUp,
            TensorRole::FfnDown,
            TensorRole::MoEGate,
            TensorRole::MoESharedExpert,
            TensorRole::MoEExpert,
            TensorRole::DepthwiseConv,
            TensorRole::MlaQCompress,
            TensorRole::MlaQExpand,
            TensorRole::MlaKvCompress,
            TensorRole::MlaKeyAbsorb,
            TensorRole::MlaValueAbsorb,
            TensorRole::MlaRopeKey,
            TensorRole::MtpProjection,
            TensorRole::Rope,
        ];
        for role in non_kv {
            assert!(!is_kv_projection_role(role), "{role:?} should not be a KV role");
        }
    }

    #[test]
    fn tensor_role_kv_variants_hashable() {
        let mut set = HashSet::new();
        set.insert(TensorRole::AttentionKey);
        set.insert(TensorRole::AttentionValue);
        set.insert(TensorRole::AttentionKey);
        assert_eq!(set.len(), 2);
    }

    // ── ChecksumPolicy ───────────────────────────────────────────────────

    #[test]
    fn checksum_policy_variants_equality() {
        assert_eq!(ChecksumPolicy::Ignore, ChecksumPolicy::Ignore);
        assert_eq!(ChecksumPolicy::Verify, ChecksumPolicy::Verify);
        assert_ne!(ChecksumPolicy::Ignore, ChecksumPolicy::Verify);
    }

    #[test]
    fn checksum_policy_default_is_ignore() {
        assert_eq!(ChecksumPolicy::default(), ChecksumPolicy::Ignore);
    }

    #[test]
    fn checksum_policy_copy() {
        let a = ChecksumPolicy::Verify;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn checksum_policy_debug() {
        assert!(format!("{:?}", ChecksumPolicy::Ignore).contains("Ignore"));
        assert!(format!("{:?}", ChecksumPolicy::Verify).contains("Verify"));
        assert!(format!("{:?}", ChecksumPolicy::Default).contains("Default"));
    }

    #[test]
    fn checksum_policy_all_variants_distinct() {
        let variants = [ChecksumPolicy::Ignore, ChecksumPolicy::Verify, ChecksumPolicy::Default];
        let mut set = HashSet::new();
        for v in &variants {
            set.insert(*v);
        }
        assert_eq!(set.len(), 3);
    }

    // ── ParallelPolicy ───────────────────────────────────────────────────

    #[test]
    fn parallel_policy_default_enabled() {
        let p = ParallelPolicy::default();
        assert!(p.enabled);
    }

    #[test]
    fn parallel_policy_fields() {
        let p = ParallelPolicy { enabled: false };
        assert!(!p.enabled);
    }

    #[test]
    fn parallel_policy_copy() {
        let a = ParallelPolicy { enabled: true };
        let b = a;
        assert_eq!(a.enabled, b.enabled);
    }

    #[test]
    fn parallel_policy_debug() {
        let p = ParallelPolicy { enabled: false };
        let debug = format!("{p:?}");
        assert!(debug.contains("ParallelPolicy"));
    }

    #[test]
    fn parallel_policy_clone() {
        let a = ParallelPolicy { enabled: true };
        let b = a.clone();
        assert_eq!(a.enabled, b.enabled);
    }

    // ── UploadedTensor ───────────────────────────────────────────────────

    #[test]
    fn uploaded_tensor_construction() {
        let t = UploadedTensor {
            name: "weight.0".to_string(),
            shape: vec![256, 512],
        };
        assert_eq!(t.name, "weight.0");
        assert_eq!(t.shape, vec![256, 512]);
    }

    #[test]
    fn uploaded_tensor_empty_shape() {
        let t = UploadedTensor {
            name: "scalar".to_string(),
            shape: vec![],
        };
        assert!(t.shape.is_empty());
    }

    #[test]
    fn uploaded_tensor_empty_name() {
        let t = UploadedTensor {
            name: String::new(),
            shape: vec![1],
        };
        assert!(t.name.is_empty());
    }

    #[test]
    fn uploaded_tensor_clone() {
        let t = UploadedTensor {
            name: "test".to_string(),
            shape: vec![3, 4],
        };
        let cloned = t.clone();
        assert_eq!(cloned.name, t.name);
        assert_eq!(cloned.shape, t.shape);
    }

    #[test]
    fn uploaded_tensor_debug() {
        let t = UploadedTensor {
            name: "w".to_string(),
            shape: vec![1],
        };
        let debug = format!("{t:?}");
        assert!(debug.contains("UploadedTensor"));
    }

    // ── LoaderConfig ─────────────────────────────────────────────────────

    #[test]
    fn loader_config_default_checksum_policy() {
        let cfg = LoaderConfig::default();
        assert_eq!(cfg.checksum_policy, ChecksumPolicy::Ignore);
    }

    #[test]
    fn loader_config_default_fallback_enabled() {
        let cfg = LoaderConfig::default();
        assert!(cfg.enable_fallback);
    }

    #[test]
    fn loader_config_debug() {
        let cfg = LoaderConfig::default();
        let debug = format!("{cfg:?}");
        assert!(debug.contains("LoaderConfig"));
    }

    #[test]
    fn loader_config_clone() {
        let cfg = LoaderConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.enable_fallback, cfg.enable_fallback);
        assert_eq!(cloned.checksum_policy, cfg.checksum_policy);
    }

    // ── LoaderError ──────────────────────────────────────────────────────

    #[test]
    fn loader_error_io_variant() {
        let err = LoaderError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file"));
        let msg = format!("{err}");
        assert!(msg.contains("IO error"));
    }

    #[test]
    fn loader_error_network_variant() {
        let err = LoaderError::Network("timeout".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Network error"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn loader_error_missing_weights() {
        let err = LoaderError::MissingWeights;
        let msg = format!("{err}");
        assert!(msg.contains("Missing weights"));
    }

    #[test]
    fn loader_error_missing_tensor() {
        let err = LoaderError::MissingTensor("k_proj.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("k_proj.weight"));
    }

    #[test]
    fn loader_error_gguf_variant() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("GGUF error"));
    }

    #[test]
    fn loader_error_backend_variant() {
        let err = LoaderError::Backend("OOM".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Backend error"));
    }

    #[test]
    fn loader_error_cache_variant() {
        let err = LoaderError::Cache("corrupted".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Cache error"));
        assert!(msg.contains("corrupted"));
    }

    #[test]
    fn loader_error_onnx_variant() {
        let err = LoaderError::Onnx("parse failure".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("ONNX error"));
    }

    #[test]
    fn loader_error_arch_detection_variant() {
        let err = LoaderError::ArchDetection("unknown arch".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Architecture detection failed"));
    }

    #[test]
    fn loader_error_debug() {
        let err = LoaderError::MissingWeights;
        let debug = format!("{err:?}");
        assert!(debug.contains("MissingWeights"));
    }

    #[test]
    fn loader_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let loader_err: LoaderError = io_err.into();
        let msg = format!("{loader_err}");
        assert!(msg.contains("IO error"));
    }

    // ── LoaderError: untested variants ─────────────────────────────────

    #[test]
    fn loader_error_duplicate_tensor_contains_name() {
        let err = LoaderError::DuplicateTensor("layers.0.q_proj.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Duplicate tensor"), "expected 'Duplicate tensor' in: {msg}");
        assert!(msg.contains("layers.0.q_proj.weight"), "expected tensor name in: {msg}");
    }

    #[test]
    fn loader_error_unsupported_dtype_shows_type() {
        let err = LoaderError::UnsupportedDtype(safetensors::Dtype::BOOL);
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported dtype"), "expected 'Unsupported dtype' in: {msg}");
        assert!(msg.contains("BOOL"), "expected dtype variant name in: {msg}");
    }

    #[test]
    fn loader_error_authentication_error_contains_hint() {
        let err = LoaderError::AuthenticationError {
            hint: "set HF_TOKEN".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Authentication error"), "expected 'Authentication error' in: {msg}");
        assert!(msg.contains("set HF_TOKEN"), "expected hint text in: {msg}");
    }

    #[test]
    fn loader_error_pytorch_variant_message() {
        let err = LoaderError::Pytorch("pickle parse failed".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("PyTorch error"), "expected 'PyTorch error' in: {msg}");
        assert!(msg.contains("pickle parse failed"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_gllm_variant_message() {
        let err = LoaderError::Gllm("invalid header magic".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("GLLM error"), "expected 'GLLM error' in: {msg}");
        assert!(msg.contains("invalid header magic"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_hfhub_variant_message() {
        let err = LoaderError::HfHub("rate limited".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("HfHub error"), "expected 'HfHub error' in: {msg}");
        assert!(msg.contains("rate limited"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_invalid_quantization_message() {
        let err = LoaderError::InvalidQuantization("missing scales".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invalid quantization"), "expected 'Invalid quantization' in: {msg}");
        assert!(msg.contains("missing scales"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_format_not_found_shows_format() {
        let err = LoaderError::FormatNotFound(crate::loader::WeightFormat::Gguf);
        let msg = format!("{err}");
        assert!(msg.contains("Format not found"), "expected 'Format not found' in: {msg}");
        assert!(msg.contains("Gguf"), "expected format name in: {msg}");
    }

    #[test]
    fn loader_error_multiple_weight_formats_message() {
        let formats = vec![
            crate::loader::WeightFormat::SafeTensors,
            crate::loader::WeightFormat::Gguf,
        ];
        let err = LoaderError::MultipleWeightFormats(formats);
        let msg = format!("{err}");
        assert!(msg.contains("Multiple weight formats"), "expected 'Multiple weight formats' in: {msg}");
    }

    // ── LoaderConfig: untested fields ──────────────────────────────────

    #[test]
    fn loader_config_default_source_is_huggingface() {
        let cfg = LoaderConfig::default();
        assert_eq!(cfg.source, crate::loader::ModelSource::HuggingFace);
    }

    #[test]
    fn loader_config_default_gguf_file_filter_is_none() {
        let cfg = LoaderConfig::default();
        assert!(cfg.gguf_file_filter.is_none());
    }

    #[test]
    fn loader_config_default_hf_token_path_is_none() {
        let cfg = LoaderConfig::default();
        assert!(cfg.hf_token_path.is_none());
    }

    // ── layer_allows_missing_kv_weights: overflow safety ───────────────

    #[test]
    fn layer_allows_missing_kv_overflow_safe() {
        // layer_i=usize::MAX with num_kv_shared_layers=0 must not panic
        assert!(!layer_allows_missing_kv_weights(usize::MAX, usize::MAX, 0));
        // small layer with huge hidden_layers and shared=0 also safe
        assert!(!layer_allows_missing_kv_weights(0, usize::MAX, 0));
    }

    #[test]
    fn layer_allows_missing_kv_shared_exceeds_hidden_layers() {
        // num_kv_shared_layers > num_hidden_layers: layer 0 qualifies
        assert!(layer_allows_missing_kv_weights(0, 4, 10));
        assert!(layer_allows_missing_kv_weights(3, 4, 10));
    }

    // ── LoaderError: from_json conversion ──────────────────────────────

    #[test]
    fn loader_error_from_json_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid");
        let loader_err: LoaderError = json_err.unwrap_err().into();
        let msg = format!("{loader_err}");
        assert!(msg.contains("JSON error"), "expected 'JSON error' in: {msg}");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NEW TESTS (13)
    // ═══════════════════════════════════════════════════════════════════════

    // ── TEST-WL-01: LoaderError::SafeTensors variant via From conversion ──

    // @trace TEST-WL-01 [req:REQ-LOADER] [level:unit]
    #[test]
    fn loader_error_from_safetensor_error() {
        // Arrange: create a SafeTensorError by attempting invalid metadata parse.
        let st_result = safetensors::SafeTensors::deserialize(&[0u8; 8]);
        let st_err = st_result.unwrap_err();
        // Act: convert via From.
        let loader_err: LoaderError = st_err.into();
        // Assert: message contains "SafeTensors error".
        let msg = format!("{loader_err}");
        assert!(
            msg.contains("SafeTensors error"),
            "expected 'SafeTensors error' in: {msg}"
        );
    }

    // ── TEST-WL-02: LoaderError::UnsupportedWeightExtension variant ──────

    // @trace TEST-WL-02 [req:REQ-LOADER] [level:unit]
    #[test]
    fn loader_error_unsupported_weight_extension_message() {
        // Arrange
        let err = LoaderError::UnsupportedWeightExtension(".binx".to_string());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(
            msg.contains("Unsupported weight extension"),
            "expected 'Unsupported weight extension' in: {msg}"
        );
        assert!(
            msg.contains(".binx"),
            "expected extension in: {msg}"
        );
    }

    // ── TEST-WL-03: ModelSource ModelScope variant debug and equality ────

    // @trace TEST-WL-03 [req:REQ-LOADER] [level:unit]
    #[test]
    fn model_source_model_scope_debug_and_equality() {
        // Arrange
        let hf = crate::loader::ModelSource::HuggingFace;
        let ms = crate::loader::ModelSource::ModelScope;
        // Act & Assert: distinct variants
        assert_ne!(hf, ms);
        // Self-equality
        assert_eq!(ms, ms);
        // Debug format contains variant name
        let debug = format!("{ms:?}");
        assert!(debug.contains("ModelScope"), "expected 'ModelScope' in: {debug}");
    }

    // ── TEST-WL-04: ModelSource clone and copy ───────────────────────────

    // @trace TEST-WL-04 [req:REQ-LOADER] [level:unit]
    #[test]
    fn model_source_clone_and_copy() {
        // Arrange
        let original = crate::loader::ModelSource::ModelScope;
        // Act
        let cloned = original.clone();
        let copied = original;
        // Assert
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    // ── TEST-WL-05: WeightFormat all variants are distinct ───────────────

    // @trace TEST-WL-05 [req:REQ-LOADER] [level:unit]
    #[test]
    fn weight_format_all_variants_distinct() {
        // Arrange
        use crate::loader::WeightFormat;
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        // Act: collect into HashSet to verify uniqueness
        let mut set = HashSet::new();
        for v in &variants {
            set.insert(*v);
        }
        // Assert: 5 distinct variants
        assert_eq!(set.len(), 5, "WeightFormat should have 5 distinct variants");
    }

    // ── TEST-WL-06: WeightFormat clone and debug ─────────────────────────

    // @trace TEST-WL-06 [req:REQ-LOADER] [level:unit]
    #[test]
    fn weight_format_clone_and_debug() {
        // Arrange
        use crate::loader::WeightFormat;
        let original = WeightFormat::Gllm;
        // Act
        let cloned = original.clone();
        let debug = format!("{cloned:?}");
        // Assert
        assert_eq!(original, cloned);
        assert!(debug.contains("Gllm"), "expected 'Gllm' in debug: {debug}");
    }

    // ── TEST-WL-07: TensorRole::to_canonical_name with layer index ───────

    // @trace TEST-WL-07 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_role_to_canonical_name_with_layer() {
        // Arrange
        let role = TensorRole::AttentionQuery;
        // Act
        let name = role.to_canonical_name(Some(3));
        // Assert: layer-prefixed format "L3.q_proj"
        assert_eq!(name, "L3.q_proj");
    }

    // ── TEST-WL-08: TensorRole::to_canonical_name without layer index ────

    // @trace TEST-WL-08 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_role_to_canonical_name_without_layer() {
        // Arrange
        let role = TensorRole::Embedding;
        // Act
        let name = role.to_canonical_name(None);
        // Assert: bare canonical name
        assert_eq!(name, "embed");
    }

    // ── TEST-WL-09: TensorRole::to_canonical_name all MLA variants ───────

    // @trace TEST-WL-09 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_role_mla_variants_canonical_names() {
        // Arrange & Act & Assert: verify each MLA role maps to its canonical name
        assert_eq!(TensorRole::MlaQCompress.to_canonical_name(None), "q_a_proj");
        assert_eq!(TensorRole::MlaQExpand.to_canonical_name(None), "q_b_proj");
        assert_eq!(TensorRole::MlaKvCompress.to_canonical_name(None), "kv_b_proj");
        assert_eq!(TensorRole::MlaKeyAbsorb.to_canonical_name(None), "k_b_proj");
        assert_eq!(TensorRole::MlaValueAbsorb.to_canonical_name(None), "v_b_proj");
        assert_eq!(TensorRole::MlaRopeKey.to_canonical_name(None), "k_pe_proj");
    }

    // ── TEST-WL-10: LoaderConfig with custom non-default values ──────────

    // @trace TEST-WL-10 [req:REQ-LOADER] [level:unit]
    #[test]
    fn loader_config_custom_values() {
        // Arrange: construct a non-default LoaderConfig
        let cfg = LoaderConfig {
            cache_dir: std::path::PathBuf::from("/tmp/custom_cache"),
            source: crate::loader::ModelSource::ModelScope,
            hf_token_path: Some(std::path::PathBuf::from("/etc/hf_token")),
            enable_fallback: false,
            checksum_policy: ChecksumPolicy::Verify,
            gguf_file_filter: Some("Q8".to_string()),
            tensor_skip_config: crate::loader::TensorSkipConfig::default(),
            extra_suffix_patterns: Vec::new(),
        };
        // Assert: all custom values preserved
        assert_eq!(cfg.cache_dir, std::path::PathBuf::from("/tmp/custom_cache"));
        assert_eq!(cfg.source, crate::loader::ModelSource::ModelScope);
        assert_eq!(cfg.hf_token_path.as_deref(), Some(std::path::Path::new("/etc/hf_token")));
        assert!(!cfg.enable_fallback);
        assert_eq!(cfg.checksum_policy, ChecksumPolicy::Verify);
        assert_eq!(cfg.gguf_file_filter.as_deref(), Some("Q8"));
    }

    // ── TEST-WL-11: LoaderError from GgufError conversion ────────────────

    // @trace TEST-WL-11 [req:REQ-LOADER] [level:unit]
    #[test]
    fn loader_error_from_gguf_error() {
        // Arrange: create a GgufError variant
        let gguf_err = crate::loader::gguf::GgufError::InvalidMagic(0xDEADBEEF);
        // Act: convert via From
        let loader_err: LoaderError = gguf_err.into();
        // Assert: wraps into Gguf variant
        let msg = format!("{loader_err}");
        assert!(
            msg.contains("GGUF error"),
            "expected 'GGUF error' in: {msg}"
        );
        assert!(
            msg.contains("deadbeef"),
            "expected hex magic in: {msg}"
        );
    }

    // ── TEST-WL-12: LoaderError from GllmError conversion ────────────────

    // @trace TEST-WL-12 [req:REQ-LOADER] [level:unit]
    #[test]
    fn loader_error_from_gllm_error() {
        // Arrange: create a GllmError variant
        let gllm_err = crate::loader::gllm::GllmError::InvalidMagic(0x12345678);
        // Act: convert via From
        let loader_err: LoaderError = gllm_err.into();
        // Assert: wraps into Gllm variant
        let msg = format!("{loader_err}");
        assert!(
            msg.contains("GLLM error"),
            "expected 'GLLM error' in: {msg}"
        );
    }

    // ── TEST-WL-13: layer_allows_missing_kv_weights exact boundary arithmetic ─

    // @trace TEST-WL-13 [req:REQ-LOADER] [level:unit]
    #[test]
    fn layer_allows_missing_kv_exact_arithmetic_no_wrap() {
        // Arrange: layer in the shared range where layer + shared >= total
        let total = 100;
        let shared = 10;
        let layer = 95; // 95 + 10 = 105 >= 100 => true
        // Act
        let result = layer_allows_missing_kv_weights(layer, total, shared);
        // Assert
        assert!(result, "layer + shared should exceed total");
        // Verify same layer does not qualify when shared is 0
        assert!(!layer_allows_missing_kv_weights(layer, total, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NEW TESTS (10) — GgmlDType, GgufValue, GgufArray, TensorRole canonical
    // ═══════════════════════════════════════════════════════════════════════

    // ── TEST-WL-14: GgmlDType as_str round-trips for representative dtypes ─

    // @trace TEST-WL-14 [req:REQ-LOADER] [level:unit]
    #[test]
    fn ggml_dtype_as_str_covers_key_variants() {
        use crate::loader::gguf::GgmlDType;
        // Arrange: pick one dtype from each family (native, quantized, IQ, special)
        let pairs: &[(GgmlDType, &str)] = &[
            (GgmlDType::F32, "F32"),
            (GgmlDType::BF16, "BF16"),
            (GgmlDType::Q4_0, "Q4_0"),
            (GgmlDType::Q8_K, "Q8_K"),
            (GgmlDType::IQ3_XXS, "IQ3_XXS"),
            (GgmlDType::AWQ4, "AWQ4"),
            (GgmlDType::GPTQ4, "GPTQ4"),
            (GgmlDType::NVFP4, "NVFP4"),
            (GgmlDType::SQUEEZE, "SQUEEZE"),
            (GgmlDType::MXFP4, "MXFP4"),
        ];
        // Act & Assert
        for (dtype, expected) in pairs {
            assert_eq!(dtype.as_str(), *expected, "as_str mismatch for {dtype:?}");
        }
    }

    // ── TEST-WL-15: GgmlDType is_quantized distinguishes native from quantized ─

    // @trace TEST-WL-15 [req:REQ-LOADER] [level:unit]
    #[test]
    fn ggml_dtype_is_quantized_classification() {
        use crate::loader::gguf::GgmlDType;
        // Arrange: native (non-quantized) dtypes
        let native = [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16,
                      GgmlDType::F64, GgmlDType::I8, GgmlDType::I16,
                      GgmlDType::I32, GgmlDType::I64];
        // Act & Assert: none of the native types are quantized
        for dt in &native {
            assert!(!dt.is_quantized(), "{dt:?} should not be quantized");
        }
        // Quantized types return true
        let quantized = [GgmlDType::Q4_0, GgmlDType::Q4_K, GgmlDType::AWQ4,
                         GgmlDType::NVFP4, GgmlDType::SQUEEZE, GgmlDType::IQ1_S];
        for dt in &quantized {
            assert!(dt.is_quantized(), "{dt:?} should be quantized");
        }
    }

    // ── TEST-WL-16: GgmlDType TryFrom<u32> valid and invalid values ────────

    // @trace TEST-WL-16 [req:REQ-LOADER] [level:unit]
    #[test]
    fn ggml_dtype_try_from_roundtrip_valid_and_invalid() {
        use crate::loader::gguf::GgmlDType;
        // Arrange & Act & Assert: valid mappings from u32 to GgmlDType
        assert!(matches!(GgmlDType::try_from(0u32), Ok(GgmlDType::F32)));
        assert!(matches!(GgmlDType::try_from(53u32), Ok(GgmlDType::NVFP4)));
        assert!(matches!(GgmlDType::try_from(30u32), Ok(GgmlDType::BF16)));
        // Verify discriminant round-trip for a few variants
        assert_eq!(GgmlDType::F32 as u32, 0);
        assert_eq!(GgmlDType::NVFP4 as u32, 53);
        // Invalid value produces error
        let err = GgmlDType::try_from(99u32);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("99"), "expected invalid dtype value in error: {msg}");
    }

    // ── TEST-WL-17: GgmlDType all() returns non-empty list with expected length ─

    // @trace TEST-WL-17 [req:REQ-LOADER] [level:unit]
    #[test]
    fn ggml_dtype_all_returns_every_variant() {
        use crate::loader::gguf::GgmlDType;
        // Act
        let all = GgmlDType::all();
        // Assert: non-empty, each variant is unique
        assert!(!all.is_empty());
        let mut set = HashSet::new();
        for &dt in all {
            assert!(set.insert(dt), "duplicate variant {dt:?} in GgmlDType::all()");
        }
    }

    // ── TEST-WL-18: GgmlDType block_size and block_bytes consistency ───────

    // @trace TEST-WL-18 [req:REQ-LOADER] [level:unit]
    #[test]
    fn ggml_dtype_block_size_bytes_positive_for_all() {
        use crate::loader::gguf::GgmlDType;
        // Arrange & Act & Assert: every variant has positive block_size and block_bytes
        for &dt in GgmlDType::all() {
            let bs = dt.block_size();
            let bb = dt.block_bytes();
            assert!(bs > 0, "{dt:?} block_size must be > 0, got {bs}");
            assert!(bb > 0, "{dt:?} block_bytes must be > 0, got {bb}");
        }
    }

    // ── TEST-WL-19: GgufValue accessor methods return correct types ────────

    // @trace TEST-WL-19 [req:REQ-LOADER] [level:unit]
    #[test]
    fn gguf_value_accessor_type_discrimination() {
        use crate::loader::gguf::{GgufArray, GgufValue};
        use std::sync::Arc;
        // Arrange
        let int_val = GgufValue::Uint64(42);
        let float_val = GgufValue::Float32(3.14);
        let bool_val = GgufValue::Bool(true);
        let str_val = GgufValue::String(Arc::from("hello"));
        let arr_val = GgufValue::Array(GgufArray {
            item_type: crate::loader::gguf::GgufValueType::Uint32,
            items: vec![],
        });
        // Act & Assert: each accessor returns Some only for its type
        assert_eq!(int_val.as_u64(), Some(42));
        assert!(int_val.as_f32().is_none());
        assert_eq!(float_val.as_f32(), Some(3.14f32));
        assert!(float_val.as_u64().is_none());
        assert_eq!(bool_val.as_bool(), Some(true));
        assert!(bool_val.as_str().is_none());
        assert_eq!(str_val.as_str(), Some("hello"));
        assert!(str_val.as_u64().is_none());
        assert!(arr_val.as_array().is_some());
        assert!(arr_val.as_array().unwrap().is_empty());
    }

    // ── TEST-WL-20: GgufValue signed integer as_u64 handles negative ──────

    // @trace TEST-WL-20 [req:REQ-LOADER] [level:unit]
    #[test]
    fn gguf_value_signed_int_as_u64_negative_returns_none() {
        use crate::loader::gguf::GgufValue;
        // Arrange
        let neg = GgufValue::Int8(-1);
        let pos = GgufValue::Int8(10);
        // Act & Assert
        assert!(neg.as_u64().is_none(), "negative Int8 should yield None");
        assert_eq!(pos.as_u64(), Some(10));
    }

    // ── TEST-WL-21: TensorRole to_canonical_name non-MLA roles with layer ──

    // @trace TEST-WL-21 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_role_canonical_name_all_non_mla_with_layer() {
        // Arrange & Act & Assert: verify canonical name for representative roles
        let pairs: &[(TensorRole, &str)] = &[
            (TensorRole::OutputHead, "L0.lm_head"),
            (TensorRole::FinalNorm, "L5.final_norm"),
            (TensorRole::AttentionFusedQkv, "L2.qkv_proj"),
            (TensorRole::AttentionOutput, "L7.o_proj"),
            (TensorRole::AttentionQNorm, "L1.q_norm"),
            (TensorRole::AttentionKNorm, "L1.k_norm"),
            (TensorRole::AttentionSinks, "L3.attn_sinks"),
            (TensorRole::InputNorm, "L4.input_norm"),
            (TensorRole::PostAttnNorm, "L4.post_attn_norm"),
            (TensorRole::LayerNorm, "L4.input_norm"),
            (TensorRole::FfnGate, "L6.gate_proj"),
            (TensorRole::FfnUp, "L6.up_proj"),
            (TensorRole::FfnDown, "L6.down_proj"),
            (TensorRole::MoEGate, "L2.moe_gate"),
            (TensorRole::MoESharedExpert, "L2.shared_expert"),
            (TensorRole::MoEExpert, "L2.expert"),
            (TensorRole::MtpProjection, "L0.mtp_proj"),
            (TensorRole::Rope, "L0.rope"),
            (TensorRole::DepthwiseConv, "L3.depthwise_conv"),
        ];
        for (role, expected) in pairs {
            let layer_idx = expected.split('.').next().unwrap().strip_prefix('L').unwrap().parse::<usize>().unwrap();
            assert_eq!(role.to_canonical_name(Some(layer_idx)), *expected,
                       "mismatch for {role:?} with layer {layer_idx}");
        }
    }

    // ── TEST-WL-22: TensorRole clone/copy/PartialEq round-trip ────────────

    // @trace TEST-WL-22 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_role_clone_copy_equality() {
        // Arrange
        let original = TensorRole::MoEExpert;
        // Act
        let cloned = original.clone();
        let copied = original; // Copy
        // Assert: all three are equal
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
        assert_eq!(cloned, copied);
    }

    // ── TEST-WL-23: tensor_nbytes for simple and quantized shapes ──────────

    // @trace TEST-WL-23 [req:REQ-LOADER] [level:unit]
    #[test]
    fn tensor_nbytes_f32_and_q4_0_shapes() {
        use crate::loader::gguf::{GgmlDType, tensor_nbytes};
        // Arrange: F32 [4] = 4 * 4 = 16 bytes
        let f32_bytes = tensor_nbytes(GgmlDType::F32, &[4]).unwrap();
        assert_eq!(f32_bytes, 16);
        // Q4_0: block_size=32, block_bytes=18, [32] => 18 bytes
        let q4_0_bytes = tensor_nbytes(GgmlDType::Q4_0, &[32]).unwrap();
        assert_eq!(q4_0_bytes, 18);
        // Empty shape => 0 bytes
        let empty = tensor_nbytes(GgmlDType::F32, &[]).unwrap();
        assert_eq!(empty, 0);
    }
}
