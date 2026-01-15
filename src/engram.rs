//! Engram conditional memory module for gllm.
//!
//! Provides O(1) hash-based lookup for static knowledge retrieval.
//! Reference: DeepSeek "Conditional Memory via Scalable Lookup" (https://arxiv.org/abs/2601.07372)
//!
//! ## Integration
//!
//! Engram output is fused with attention output in the decoder layer:
//! ```text
//! output = attention_output + engram_scale * engram_output
//! ```
//!
//! ## Requirements
//!
//! Models must ship with pre-trained Engram embedding weights (e.g., DeepSeek-V4+).
//! The embedding table is memory-mapped from disk for zero-copy access.

use crate::model_config::ModelConfig;
use crate::types::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use gllm_kernels::{
    Engram as KernelEngram, EngramConfig as KernelEngramConfig,
    EngramHashConfig, EngramLookupConfig, fuse_engram_attention_simd,
};
use std::path::Path;
use std::sync::Arc;

/// Engram module wrapper for Burn integration.
///
/// Provides tensor-based interface to the gllm-kernels Engram implementation.
pub struct EngramModule {
    inner: KernelEngram,
    scale: f32,
}

impl EngramModule {
    /// Create a new Engram module from configuration.
    ///
    /// # Arguments
    /// * `config` - Model configuration with Engram settings
    ///
    /// # Returns
    /// * `None` if Engram is not enabled in config
    /// * `Some(EngramModule)` if Engram is enabled (but without weights loaded)
    pub fn from_config(config: &ModelConfig) -> Option<Self> {
        let (ngram_size, num_buckets, embedding_dim, scale) = config.engram_config()?;

        let hash_config = EngramHashConfig {
            ngram_size,
            num_buckets,
            ..Default::default()
        };

        let lookup_config = EngramLookupConfig {
            embedding_dim,
            num_buckets,
            ..Default::default()
        };

        let kernel_config = KernelEngramConfig {
            hash: hash_config,
            lookup: lookup_config,
            fuse_with_attention: true,
            output_scale: scale,
        };

        // Create with empty data - will be loaded later
        let num_elements = num_buckets * embedding_dim;
        let data = vec![0.0f32; num_elements];

        match KernelEngram::from_bytes(data, kernel_config) {
            Ok(inner) => Some(Self { inner, scale }),
            Err(_) => None,
        }
    }

    /// Load Engram module from embedding file.
    ///
    /// The file must contain `num_buckets * embedding_dim * 4` bytes
    /// (f32 embeddings in row-major order).
    pub fn from_file<P: AsRef<Path>>(path: P, config: &ModelConfig) -> Result<Self> {
        let (ngram_size, num_buckets, embedding_dim, scale) = config.engram_config()
            .ok_or_else(|| Error::InvalidConfig("Engram not enabled in config".to_string()))?;

        let hash_config = EngramHashConfig {
            ngram_size,
            num_buckets,
            ..Default::default()
        };

        let lookup_config = EngramLookupConfig {
            embedding_dim,
            num_buckets,
            ..Default::default()
        };

        let kernel_config = KernelEngramConfig {
            hash: hash_config,
            lookup: lookup_config,
            fuse_with_attention: true,
            output_scale: scale,
        };

        let inner = KernelEngram::from_file(path, kernel_config)
            .map_err(|e| Error::LoadError(format!("Failed to load Engram: {}", e)))?;

        Ok(Self { inner, scale })
    }

    /// Load Engram module from raw embedding data.
    ///
    /// # Arguments
    /// * `data` - Flattened embedding table [num_buckets * embedding_dim]
    /// * `config` - Model configuration with Engram settings
    pub fn from_embeddings(data: Vec<f32>, config: &ModelConfig) -> Result<Self> {
        let (ngram_size, num_buckets, embedding_dim, scale) = config.engram_config()
            .ok_or_else(|| Error::InvalidConfig("Engram not enabled in config".to_string()))?;

        let expected_len = num_buckets * embedding_dim;
        if data.len() < expected_len {
            return Err(Error::LoadError(format!(
                "Engram data too small: {} < {}",
                data.len(), expected_len
            )));
        }

        let hash_config = EngramHashConfig {
            ngram_size,
            num_buckets,
            ..Default::default()
        };

        let lookup_config = EngramLookupConfig {
            embedding_dim,
            num_buckets,
            ..Default::default()
        };

        let kernel_config = KernelEngramConfig {
            hash: hash_config,
            lookup: lookup_config,
            fuse_with_attention: true,
            output_scale: scale,
        };

        let inner = KernelEngram::from_bytes(data, kernel_config)
            .map_err(|e| Error::LoadError(format!("Failed to create Engram: {}", e)))?;

        Ok(Self { inner, scale })
    }

    /// Forward pass: lookup embeddings for token sequence.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// * Embeddings tensor [batch, num_ngrams, embedding_dim]
    pub fn forward<B: Backend>(&self, tokens: &[u32], device: &B::Device) -> Tensor<B, 2> {
        let embeddings = self.inner.forward_scaled(tokens);

        if embeddings.is_empty() {
            let embedding_dim = self.inner.embedding_dim();
            return Tensor::zeros([0, embedding_dim], device);
        }

        let num_ngrams = embeddings.len();
        let embedding_dim = embeddings[0].len();

        // Flatten embeddings to contiguous array
        let data: Vec<f32> = embeddings.into_iter().flatten().collect();

        Tensor::from_data(
            TensorData::new(data, [num_ngrams, embedding_dim]),
            device,
        )
    }

    /// Forward pass into pre-allocated buffer.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs
    /// * `output` - Pre-allocated buffer [num_ngrams * embedding_dim]
    pub fn forward_into(&self, tokens: &[u32], output: &mut [f32]) {
        self.inner.forward_into(tokens, output);
    }

    /// Fuse Engram output with attention output.
    ///
    /// Computes: attention_out += scale * engram_out
    ///
    /// # Arguments
    /// * `attention_out` - Attention output tensor (mutable)
    /// * `engram_out` - Engram lookup output
    ///
    /// # Note
    /// Uses SIMD acceleration when available (AVX2 on x86_64).
    pub fn fuse_with_attention<B: Backend>(
        &self,
        attention_out: &mut Tensor<B, 3>,
        tokens: &[u32],
    ) {
        let [batch_size, seq_len, hidden_size] = attention_out.dims();

        // Compute number of N-grams
        let num_ngrams = self.inner.num_ngrams(tokens.len());
        if num_ngrams == 0 {
            return; // No N-grams to fuse
        }

        // Get attention data
        let attn_data: Vec<f32> = attention_out.clone().into_data().into_vec()
            .expect("attention tensor to f32");

        // Get Engram embeddings
        let engram_embeddings = self.inner.forward_scaled(tokens);
        if engram_embeddings.is_empty() {
            return;
        }

        // Flatten for fusion
        let mut attn_slice: Vec<f32> = attn_data;
        let engram_flat: Vec<f32> = engram_embeddings.into_iter().flatten().collect();

        // The Engram output corresponds to N-gram positions (seq_len - ngram_size + 1)
        // We need to align it with the attention output
        let ngram_size = self.inner.ngram_size();
        let start_pos = ngram_size - 1; // First N-gram starts at position ngram_size-1

        // Fuse at corresponding positions
        let embedding_dim = self.inner.embedding_dim();
        for (i, engram_emb) in engram_flat.chunks(embedding_dim).enumerate() {
            let pos = start_pos + i;
            if pos >= seq_len {
                break;
            }

            // Calculate offset in attention tensor
            for b in 0..batch_size {
                let offset = b * seq_len * hidden_size + pos * hidden_size;
                let attn_slice_part = &mut attn_slice[offset..offset + hidden_size.min(embedding_dim)];
                let engram_part = &engram_emb[..hidden_size.min(embedding_dim)];

                // Use SIMD-accelerated fusion
                fuse_engram_attention_simd(attn_slice_part, engram_part, self.scale);
            }
        }

        // Reconstruct tensor
        let device = attention_out.device();
        *attention_out = Tensor::from_data(
            TensorData::new(attn_slice, [batch_size, seq_len, hidden_size]),
            &device,
        );
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    /// Get the N-gram size.
    pub fn ngram_size(&self) -> usize {
        self.inner.ngram_size()
    }

    /// Get the output scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Compute number of N-grams for a given sequence length.
    pub fn num_ngrams(&self, seq_len: usize) -> usize {
        self.inner.num_ngrams(seq_len)
    }
}

/// Shared Engram module for use across decoder layers.
pub type SharedEngram = Arc<EngramModule>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    fn create_test_config() -> ModelConfig {
        let mut config = ModelConfig::default();
        config.hidden_size = 16;
        config.engram_enabled = Some(true);
        config.engram_ngram_size = Some(2);
        config.engram_num_buckets = Some(100);
        config.engram_embedding_dim = Some(16);
        config.engram_scale = Some(0.5);
        config
    }

    #[test]
    fn test_engram_from_config() {
        let config = create_test_config();
        let engram = EngramModule::from_config(&config);
        assert!(engram.is_some());

        let engram = engram.unwrap();
        assert_eq!(engram.embedding_dim(), 16);
        assert_eq!(engram.ngram_size(), 2);
        assert_eq!(engram.scale(), 0.5);
    }

    #[test]
    fn test_engram_disabled() {
        let config = ModelConfig::default();
        let engram = EngramModule::from_config(&config);
        assert!(engram.is_none());
    }

    #[test]
    fn test_engram_forward() {
        let config = create_test_config();

        // Create embedding data
        let num_buckets = 100;
        let embedding_dim = 16;
        let data: Vec<f32> = (0..num_buckets * embedding_dim)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        let engram = EngramModule::from_embeddings(data, &config).unwrap();

        let tokens = [1u32, 2, 3, 4, 5];
        let device = <NdArray<f32> as Backend>::Device::default();
        let output = engram.forward::<NdArray<f32>>(&tokens, &device);

        // With ngram_size=2 and 5 tokens, should get 4 embeddings
        let dims = output.dims();
        assert_eq!(dims[0], 4); // num_ngrams
        assert_eq!(dims[1], 16); // embedding_dim
    }

    #[test]
    fn test_engram_num_ngrams() {
        let config = create_test_config();
        let engram = EngramModule::from_config(&config).unwrap();

        assert_eq!(engram.num_ngrams(5), 4); // 5 - 2 + 1 = 4
        assert_eq!(engram.num_ngrams(2), 1); // 2 - 2 + 1 = 1
        assert_eq!(engram.num_ngrams(1), 0); // Too short
    }
}
