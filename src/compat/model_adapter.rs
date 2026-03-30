//! Model adapter implementations for gllm.
//!
//! This module provides concrete implementations of model-specific adapters
//! that bridge between gllm's model loading layer and gllm-kernels' JIT compiler.
//!
//! # Architecture
//!
//! - **gllm-kernels**: Defines `ModelAdapter` trait with configuration fusion hints
//! - **gllm**: Implements concrete adapters that use gllm's `ModelArchitecture` enum
//!   and `ModelManifest` to select the right kernel adapter
//!
//! # CPU/GPU Unification
//!
//! All adapters work with both CPU and GPU backends. The backend-specific
//! behavior is handled by the JIT compiler in gllm-kernels.

use std::sync::Arc;

use gllm_kernels::compiler::{
    ModelAdapter as KernelModelAdapter, FusionHints, FfnActivation,
    LlamaAdapter, QwenAdapter, GemmaAdapter, MistralAdapter, PhiAdapter,
    AdapterWeightLayout,
};
use gllm_kernels::types::{ModelConfig, DType};

use crate::manifest::{ModelArchitecture, ModelManifest};

/// Errors from model adapter operations.
#[derive(Debug, Clone)]
pub enum AdapterError {
    /// Unsupported model architecture
    UnsupportedArchitecture { arch: ModelArchitecture },
    /// Invalid configuration
    InvalidConfig { reason: String },
    /// Missing required field
    MissingField { field: String },
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedArchitecture { arch } => {
                write!(f, "unsupported model architecture: {:?}", arch)
            }
            Self::InvalidConfig { reason } => {
                write!(f, "invalid configuration: {}", reason)
            }
            Self::MissingField { field } => {
                write!(f, "missing required field: {}", field)
            }
        }
    }
}

impl std::error::Error for AdapterError {}

/// gllm's model adapter — bridges ModelManifest to kernel ModelAdapter.
///
/// This wraps the kernel's adapter trait and adds gllm-specific logic
/// for architecture detection and configuration loading.
#[derive(Debug, Clone)]
pub struct GllmModelAdapter {
    inner: Arc<dyn KernelModelAdapter + Send + Sync>,
    arch: ModelArchitecture,
}

impl GllmModelAdapter {
    /// Create a new adapter from a model manifest.
    ///
    /// This selects the appropriate kernel adapter based on the model's
    /// architecture enum from the manifest.
    pub fn from_manifest(manifest: &ModelManifest) -> Result<Self, AdapterError> {
        let inner: Arc<dyn KernelModelAdapter + Send + Sync> = match manifest.arch {
            ModelArchitecture::Llama4 | ModelArchitecture::SmolLM2 | ModelArchitecture::InternLM3 => {
                Arc::new(LlamaAdapter)
            }
            ModelArchitecture::Qwen2_5 | ModelArchitecture::Qwen3 => Arc::new(QwenAdapter),
            ModelArchitecture::Qwen3MoE => Arc::new(QwenAdapter), // Uses same base adapter
            ModelArchitecture::Gemma2 => Arc::new(GemmaAdapter),
            ModelArchitecture::Mistral3 | ModelArchitecture::Ministral => {
                // Mistral may have sliding window — could be configured from manifest
                Arc::new(MistralAdapter { sliding_window: manifest.max_context_override })
            }
            ModelArchitecture::Phi4 => Arc::new(PhiAdapter::default()),
            ModelArchitecture::GLM4 | ModelArchitecture::GLM5 => {
                // GLM uses Llama-like architecture with some differences
                // For now, use LlamaAdapter as base
                Arc::new(LlamaAdapter)
            }
            ModelArchitecture::DeepSeek => {
                // DeepSeek V2/V3 uses MoE + Llama-like base
                Arc::new(LlamaAdapter)
            }
            ModelArchitecture::XlmR | ModelArchitecture::XlmRNext => {
                // XLM-R is encoder-only, different architecture
                return Err(AdapterError::UnsupportedArchitecture { arch: manifest.arch });
            }
        };

        Ok(Self {
            inner,
            arch: manifest.arch,
        })
    }

    /// Create adapter directly from ModelArchitecture enum.
    pub fn from_arch(arch: ModelArchitecture) -> Result<Self, AdapterError> {
        let manifest = ModelManifest {
            arch,
            ..Default::default()
        };
        Self::from_manifest(&manifest)
    }

    /// Get the wrapped kernel adapter.
    pub fn kernel_adapter(&self) -> &(dyn KernelModelAdapter + Send + Sync) {
        self.inner.as_ref()
    }

    /// Get the model architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.arch
    }

    /// Adapt a model manifest to a ModelConfig.
    ///
    /// This extracts configuration from the manifest (or its defaults)
    /// and uses the kernel adapter to create a proper ModelConfig.
    pub fn adapt_manifest(
        &self,
        manifest: &ModelManifest,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Result<ModelConfig, AdapterError> {
        let mut config = self.inner.adapt_config(
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            dtype,
        );

        // Apply manifest overrides
        if let Some(rope_base) = manifest.rope_base_override {
            config.rope_theta = rope_base as f64;
        }
        if let Some(max_ctx) = manifest.max_context_override {
            config.max_seq_len = max_ctx;
        }

        // Validate the resulting config
        self.inner
            .validate_config(&config)
            .map_err(|e| AdapterError::InvalidConfig { reason: e })?;

        Ok(config)
    }

    /// Get fusion hints for this adapter.
    pub fn fusion_hints(&self) -> FusionHints {
        self.inner.fusion_hints()
    }

    /// Get weight layout for this adapter.
    pub fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        self.inner.weight_layout(config)
    }

    /// Check if this model uses MoE (Mixture of Experts).
    pub fn is_moe(&self) -> bool {
        matches!(
            self.arch,
            ModelArchitecture::Qwen3MoE | ModelArchitecture::DeepSeek
        )
    }

    /// Check if this model uses GQA (Grouped Query Attention).
    pub fn is_gqa(&self) -> bool {
        self.inner.fusion_hints().use_gqa
    }

    /// Get the FFN activation type.
    pub fn ffn_activation(&self) -> FfnActivation {
        self.inner.fusion_hints().ffn_activation
    }
}

/// Adapter registry — maps ModelArchitecture to GllmModelAdapter.
///
/// This provides a centralized way to create adapters for supported architectures.
pub struct AdapterRegistry;

impl AdapterRegistry {
    /// Get an adapter for the given architecture.
    pub fn get(arch: ModelArchitecture) -> Result<GllmModelAdapter, AdapterError> {
        GllmModelAdapter::from_arch(arch)
    }

    /// Get an adapter from a model manifest.
    pub fn from_manifest(manifest: &ModelManifest) -> Result<GllmModelAdapter, AdapterError> {
        GllmModelAdapter::from_manifest(manifest)
    }

    /// Check if an architecture is supported.
    pub fn is_supported(arch: ModelArchitecture) -> bool {
        Self::get(arch).is_ok()
    }

    /// List all supported architectures.
    pub fn supported_architectures() -> &'static [ModelArchitecture] {
        &[
            ModelArchitecture::Llama4,
            ModelArchitecture::SmolLM2,
            ModelArchitecture::InternLM3,
            ModelArchitecture::Qwen2_5,
            ModelArchitecture::Qwen3,
            ModelArchitecture::Qwen3MoE,
            ModelArchitecture::Gemma2,
            ModelArchitecture::Mistral3,
            ModelArchitecture::Ministral,
            ModelArchitecture::Phi4,
            ModelArchitecture::GLM4,
            ModelArchitecture::GLM5,
            ModelArchitecture::DeepSeek,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_registry_llama() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Llama4).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Llama4);
        assert!(!adapter.is_moe());
        assert!(adapter.is_gqa());
        assert_eq!(adapter.ffn_activation(), FfnActivation::SwiGlu);
    }

    #[test]
    fn test_adapter_registry_qwen() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Qwen3).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Qwen3);
        assert!(!adapter.is_moe());
        assert!(adapter.is_gqa());
    }

    #[test]
    fn test_adapter_registry_mistral() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Mistral3).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Mistral3);
        assert!(!adapter.is_moe());
        assert!(adapter.is_gqa());
        assert_eq!(adapter.ffn_activation(), FfnActivation::SwiGlu);
    }

    #[test]
    fn test_adapter_registry_phi() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Phi4).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Phi4);
        assert!(!adapter.is_moe());
        assert!(!adapter.is_gqa()); // Phi uses MHA
        assert_eq!(adapter.ffn_activation(), FfnActivation::Gelu);
    }

    #[test]
    fn test_adapter_registry_gemma() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Gemma2).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Gemma2);
        assert!(!adapter.is_moe());
        assert!(adapter.is_gqa());
        assert_eq!(adapter.ffn_activation(), FfnActivation::GeGlu);
    }

    #[test]
    fn test_adapter_is_moe() {
        let qwen_moe = AdapterRegistry::get(ModelArchitecture::Qwen3MoE).unwrap();
        assert!(qwen_moe.is_moe());

        let deepseek = AdapterRegistry::get(ModelArchitecture::DeepSeek).unwrap();
        assert!(deepseek.is_moe());

        let llama = AdapterRegistry::get(ModelArchitecture::Llama4).unwrap();
        assert!(!llama.is_moe());
    }

    #[test]
    fn test_adapter_from_manifest() {
        let manifest = ModelManifest {
            arch: ModelArchitecture::Llama4,
            rope_base_override: Some(50000.0),
            max_context_override: Some(8192),
            ..Default::default()
        };

        let adapter = GllmModelAdapter::from_manifest(&manifest).unwrap();
        assert_eq!(adapter.architecture(), ModelArchitecture::Llama4);

        let config = adapter.adapt_manifest(
            &manifest,
            4096, // hidden_size
            32,   // num_layers
            32,   // num_heads
            8,    // num_kv_heads
            128,  // head_dim
            11008, // intermediate_size
            32000, // vocab_size
            4096,  // max_seq_len
            DType::F32,
        ).unwrap();

        // Manifest overrides should be applied
        assert_eq!(config.rope_theta, 50000.0);
        assert_eq!(config.max_seq_len, 8192);
    }

    #[test]
    fn test_adapter_fusion_hints() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Llama4).unwrap();
        let hints = adapter.fusion_hints();

        assert!(hints.use_flash_attention);
        assert!(hints.use_gqa);
        assert_eq!(hints.ffn_activation, FfnActivation::SwiGlu);
        assert!(hints.sliding_window.is_none());
    }

    #[test]
    fn test_adapter_unsupported_arch() {
        let result = AdapterRegistry::get(ModelArchitecture::XlmR);
        assert!(result.is_err());

        match result.unwrap_err() {
            AdapterError::UnsupportedArchitecture { arch } => {
                assert_eq!(arch, ModelArchitecture::XlmR);
            }
            _ => panic!("expected UnsupportedArchitecture error"),
        }
    }

    #[test]
    fn test_adapter_validate_config_invalid() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Llama4).unwrap();
        let manifest = ModelManifest {
            arch: ModelArchitecture::Llama4,
            ..Default::default()
        };

        // Invalid: num_kv_heads > num_heads
        let result = adapter.adapt_manifest(
            &manifest,
            4096, 32, 8,  // num_heads < num_kv_heads
            32, 128, 11008, 32000, 4096, DType::F32,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_adapter_weight_layout() {
        let adapter = AdapterRegistry::get(ModelArchitecture::Llama4).unwrap();
        let config = ModelConfig::llama_7b();
        let layout = adapter.weight_layout(&config);

        assert_eq!(layout.dtype, DType::F32);
        assert!(!layout.shapes.is_empty());
        assert_eq!(layout.shapes[0].0, "embed_tokens");
    }
}
