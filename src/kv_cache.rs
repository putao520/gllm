//! KV Cache for autoregressive generation.
//!
//! Stores key-value pairs for each layer to avoid recomputation during generation.

use crate::types::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCompressionStrategy {
    None,
    SlidingWindow { window_size: usize },
}

impl Default for KvCompressionStrategy {
    fn default() -> Self {
        Self::None
    }
}

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
    compression: KvCompressionStrategy,
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
            compression: KvCompressionStrategy::None,
        }
    }

    pub fn new_with_compression(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        compression: KvCompressionStrategy,
    ) -> Result<Self> {
        let capacity = match compression {
            KvCompressionStrategy::None => max_seq_len,
            KvCompressionStrategy::SlidingWindow { window_size } => {
                if window_size == 0 {
                    return Err(Error::InvalidConfig(
                        "Sliding window size must be greater than 0".into(),
                    ));
                }
                window_size.min(max_seq_len)
            }
        };
        let layer_size = capacity * num_kv_heads * head_dim;
        let k_cache = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let v_cache = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();

        Ok(Self {
            k_cache,
            v_cache,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len: capacity,
            current_len: 0,
            compression,
        })
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

        if k.len() != v.len() {
            return Err(Error::InferenceError(
                "KV cache update length mismatch".into(),
            ));
        }

        let per_token = self.num_kv_heads * self.head_dim;
        if per_token == 0 || k.len() % per_token != 0 {
            return Err(Error::InferenceError(
                "KV cache update has invalid token alignment".into(),
            ));
        }
        let seq_tokens = k.len() / per_token;

        let base_len = if layer == 0 {
            self.current_len
        } else {
            self.current_len.saturating_sub(seq_tokens)
        };

        match self.compression {
            KvCompressionStrategy::None => {
                if base_len + seq_tokens > self.max_seq_len {
                    return Err(Error::InvalidConfig(format!(
                        "Cache overflow: {} + {} > {}",
                        base_len, seq_tokens, self.max_seq_len
                    )));
                }
                let offset = base_len * per_token;
                self.k_cache[layer][offset..offset + k.len()].copy_from_slice(k);
                self.v_cache[layer][offset..offset + v.len()].copy_from_slice(v);
            }
            KvCompressionStrategy::SlidingWindow { .. } => {
                self.update_sliding_window(layer, k, v, seq_tokens, per_token, base_len)?;
            }
        }

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
        let len = self.cached_len() * self.num_kv_heads * self.head_dim;
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
        let len = self.cached_len() * self.num_kv_heads * self.head_dim;
        Ok(&self.v_cache[layer][..len])
    }

    /// Current sequence length in cache.
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    pub(crate) fn cached_len(&self) -> usize {
        self.current_len.min(self.max_seq_len)
    }

    /// Reset cache to empty.
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Get max sequence length.
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }

    pub(crate) fn load_from_snapshot(
        &mut self,
        total_len: usize,
        k_layers: &[Vec<f32>],
        v_layers: &[Vec<f32>],
    ) -> Result<()> {
        if k_layers.len() != self.num_layers || v_layers.len() != self.num_layers {
            return Err(Error::InvalidConfig(
                "Prompt cache layer count mismatch".into(),
            ));
        }
        let per_token = self.num_kv_heads * self.head_dim;
        if per_token == 0 {
            return Err(Error::InvalidConfig(
                "KV cache has invalid head dimensions".into(),
            ));
        }
        let cached_len = match k_layers.first() {
            Some(layer) => layer.len() / per_token,
            None => 0,
        };
        if total_len < cached_len {
            return Err(Error::InvalidConfig(
                "Prompt cache length exceeds total length".into(),
            ));
        }
        if cached_len > self.max_seq_len {
            return Err(Error::InvalidConfig(
                "Prompt cache exceeds KV cache capacity".into(),
            ));
        }
        for (layer_idx, (k_layer, v_layer)) in k_layers.iter().zip(v_layers.iter()).enumerate() {
            if k_layer.len() != v_layer.len() {
                return Err(Error::InvalidConfig(
                    "Prompt cache KV length mismatch".into(),
                ));
            }
            if k_layer.len() != cached_len * per_token {
                return Err(Error::InvalidConfig(
                    "Prompt cache layer size mismatch".into(),
                ));
            }
            self.k_cache[layer_idx][..k_layer.len()].copy_from_slice(k_layer);
            self.v_cache[layer_idx][..v_layer.len()].copy_from_slice(v_layer);
        }
        self.current_len = total_len;
        Ok(())
    }

    fn update_sliding_window(
        &mut self,
        layer: usize,
        k: &[f32],
        v: &[f32],
        seq_tokens: usize,
        per_token: usize,
        base_len: usize,
    ) -> Result<()> {
        let capacity = self.max_seq_len;
        if capacity == 0 {
            return Err(Error::InvalidConfig(
                "KV cache capacity must be greater than 0".into(),
            ));
        }

        if seq_tokens >= capacity {
            let start = (seq_tokens - capacity) * per_token;
            let k_slice = &k[start..];
            let v_slice = &v[start..];
            self.k_cache[layer][..k_slice.len()].copy_from_slice(k_slice);
            self.v_cache[layer][..v_slice.len()].copy_from_slice(v_slice);
            return Ok(());
        }

        let existing_len = base_len.min(capacity);
        let incoming_len = seq_tokens;
        let overflow = (existing_len + incoming_len).saturating_sub(capacity);
        let retained_len = existing_len.saturating_sub(overflow);

        if overflow > 0 {
            let src_start = overflow * per_token;
            let src_end = existing_len * per_token;
            self.k_cache[layer].copy_within(src_start..src_end, 0);
            self.v_cache[layer].copy_within(src_start..src_end, 0);
        }

        let dst_start = retained_len * per_token;
        let dst_end = dst_start + k.len();
        self.k_cache[layer][dst_start..dst_end].copy_from_slice(k);
        self.v_cache[layer][dst_start..dst_end].copy_from_slice(v);
        Ok(())
    }
}
