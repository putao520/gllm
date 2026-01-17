//! Engram conditional memory module for gllm.
//!
//! Provides O(1) hash-based lookup for static knowledge retrieval.

use crate::model_config::ModelConfig;
use crate::types::{Error, Result};
use gllm_kernels::{
    Engram as KernelEngram, EngramConfig as KernelEngramConfig, EngramHashConfig,
    EngramLookupConfig, fuse_engram_attention_simd,
};
use std::path::Path;
use std::sync::Arc;

/// Engram module wrapper for gllm-kernels.
pub struct EngramModule {
    inner: KernelEngram,
    scale: f32,
}

impl EngramModule {
    /// Create a new Engram module from configuration.
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

        let num_elements = num_buckets * embedding_dim;
        let data = vec![0.0f32; num_elements];

        KernelEngram::from_bytes(data, kernel_config)
            .ok()
            .map(|inner| Self { inner, scale })
    }

    /// Load Engram module from embedding file.
    pub fn from_file<P: AsRef<Path>>(path: P, config: &ModelConfig) -> Result<Self> {
        let (ngram_size, num_buckets, embedding_dim, scale) = config
            .engram_config()
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
            .map_err(|e| Error::LoadError(format!("Failed to load Engram: {e}")))?;

        Ok(Self { inner, scale })
    }

    /// Load Engram module from raw embedding data.
    pub fn from_embeddings(data: Vec<f32>, config: &ModelConfig) -> Result<Self> {
        let (ngram_size, num_buckets, embedding_dim, scale) = config
            .engram_config()
            .ok_or_else(|| Error::InvalidConfig("Engram not enabled in config".to_string()))?;

        let expected_len = num_buckets * embedding_dim;
        if data.len() != expected_len {
            return Err(Error::LoadError(format!(
                "Engram data length mismatch: expected {expected_len}, got {}",
                data.len()
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
            .map_err(|e| Error::LoadError(format!("Failed to create Engram: {e}")))?;

        Ok(Self { inner, scale })
    }

    /// Forward pass: lookup embeddings for token sequence.
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.inner.forward_scaled(tokens)
    }

    /// Forward pass into pre-allocated buffer.
    pub fn forward_into(&self, tokens: &[u32], output: &mut [f32]) {
        self.inner.forward_into(tokens, output);
    }

    /// Fuse Engram output with attention output.
    pub fn fuse_with_attention(
        &self,
        attention_out: &mut [f32],
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        tokens: &[u32],
    ) {
        let engram_embeddings = self.inner.forward_scaled(tokens);
        if engram_embeddings.is_empty() {
            return;
        }

        let embedding_dim = engram_embeddings[0].len();
        let ngram_size = self.inner.ngram_size();
        let start_pos = ngram_size.saturating_sub(1);

        for (i, engram_emb) in engram_embeddings.iter().enumerate() {
            let pos = start_pos + i;
            if pos >= seq_len {
                break;
            }
            for b in 0..batch_size {
                let offset = (b * seq_len + pos) * hidden_size;
                let attn_slice = &mut attention_out[offset..offset + hidden_size];
                let engram_slice = &engram_emb[..hidden_size.min(embedding_dim)];
                fuse_engram_attention_simd(attn_slice, engram_slice, self.scale);
            }
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    pub fn ngram_size(&self) -> usize {
        self.inner.ngram_size()
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
}

pub type SharedEngram = Arc<EngramModule>;
