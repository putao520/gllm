//! KV Cache for autoregressive generation.
//!
//! Stores key-value pairs for each layer to avoid recomputation during generation.

use crate::types::{Error, Result};

/// KV Cache for transformer layers.
#[derive(Clone)]
pub struct KVCache {
    /// Key cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
    v_cache: Vec<Vec<f32>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_len: usize,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layer_size = max_seq_len * num_kv_heads * head_dim;
        let k_cache = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let v_cache = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();

        Self {
            k_cache,
            v_cache,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
        }
    }

    /// Update cache for a layer with new k, v values.
    /// k, v shape: [batch=1, seq_len, num_kv_heads * head_dim]
    pub fn update(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.num_layers {
            return Err(Error::InvalidConfig(format!(
                "Layer {} out of bounds (max {})",
                layer, self.num_layers
            )));
        }

        let seq_tokens = k.len() / (self.num_kv_heads * self.head_dim);
        if self.current_len + seq_tokens > self.max_seq_len {
            return Err(Error::InvalidConfig(format!(
                "Cache overflow: {} + {} > {}",
                self.current_len, seq_tokens, self.max_seq_len
            )));
        }

        let offset = self.current_len * self.num_kv_heads * self.head_dim;
        self.k_cache[layer][offset..offset + k.len()].copy_from_slice(k);
        self.v_cache[layer][offset..offset + v.len()].copy_from_slice(v);

        // Only update current_len on layer 0 to avoid double counting
        if layer == 0 {
            self.current_len += seq_tokens;
        }

        Ok(())
    }

    /// Get cached keys for a layer.
    pub fn layer_k(&self, layer: usize) -> Result<&[f32]> {
        if layer >= self.num_layers {
            return Err(Error::InvalidConfig(format!(
                "Layer {} out of bounds",
                layer
            )));
        }
        let len = self.current_len * self.num_kv_heads * self.head_dim;
        Ok(&self.k_cache[layer][..len])
    }

    /// Get cached values for a layer.
    pub fn layer_v(&self, layer: usize) -> Result<&[f32]> {
        if layer >= self.num_layers {
            return Err(Error::InvalidConfig(format!(
                "Layer {} out of bounds",
                layer
            )));
        }
        let len = self.current_len * self.num_kv_heads * self.head_dim;
        Ok(&self.v_cache[layer][..len])
    }

    /// Current sequence length in cache.
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// Reset cache to empty.
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Get max sequence length.
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }
}
