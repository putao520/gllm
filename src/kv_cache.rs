use crate::types::{Error, Result};

/// Simple KV cache storing keys/values in [batch, heads, seq, head_dim] layout.
#[derive(Clone)]
pub struct KVCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    num_layers: usize,
    max_len: usize,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    current_len: usize,
}

impl KVCache {
    pub fn preallocate(
        num_layers: usize,
        max_len: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layer_len = batch_size
            .saturating_mul(num_heads)
            .saturating_mul(max_len)
            .saturating_mul(head_dim);
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            keys.push(vec![0.0; layer_len]);
            values.push(vec![0.0; layer_len]);
        }

        Self {
            keys,
            values,
            num_layers,
            max_len,
            batch_size,
            num_heads,
            head_dim,
            current_len: 0,
        }
    }

    pub fn update(&mut self, layer: usize, new_k: &[f32], new_v: &[f32]) -> Result<usize> {
        if layer >= self.num_layers {
            return Err(Error::InferenceError(
                "KV cache layer index out of range".into(),
            ));
        }
        if self.max_len == 0 || self.num_heads == 0 || self.head_dim == 0 {
            return Ok(self.current_len);
        }
        if new_k.len() != new_v.len() {
            return Err(Error::InferenceError(
                "KV cache update expects matching K/V lengths".into(),
            ));
        }

        let stride = self
            .batch_size
            .saturating_mul(self.num_heads)
            .saturating_mul(self.head_dim);
        if stride == 0 || new_k.len() % stride != 0 {
            return Err(Error::InferenceError(
                "KV cache update length does not match shape".into(),
            ));
        }

        let seq_len = new_k.len() / stride;
        let start = self.current_len;
        let end = start + seq_len;
        if end > self.max_len {
            return Err(Error::InferenceError(format!(
                "KV cache exceeded max_len ({} > {})",
                end, self.max_len
            )));
        }

        let key_buf = &mut self.keys[layer];
        let val_buf = &mut self.values[layer];
        let per_head = seq_len * self.head_dim;
        let max_stride = self.max_len * self.head_dim;

        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                let src_base = (b * self.num_heads + h) * per_head;
                let dst_base = (b * self.num_heads + h) * max_stride + start * self.head_dim;
                let src_slice = &new_k[src_base..src_base + per_head];
                let dst_slice = &mut key_buf[dst_base..dst_base + per_head];
                dst_slice.copy_from_slice(src_slice);

                let src_slice = &new_v[src_base..src_base + per_head];
                let dst_slice = &mut val_buf[dst_base..dst_base + per_head];
                dst_slice.copy_from_slice(src_slice);
            }
        }

        if layer + 1 == self.num_layers {
            self.current_len = end;
        }

        Ok(end)
    }

    pub fn layer_k(&self, layer: usize) -> Result<&[f32]> {
        self.keys.get(layer).map(|v| v.as_slice()).ok_or_else(|| {
            Error::InferenceError("KV cache layer index out of range".into())
        })
    }

    pub fn layer_v(&self, layer: usize) -> Result<&[f32]> {
        self.values.get(layer).map(|v| v.as_slice()).ok_or_else(|| {
            Error::InferenceError("KV cache layer index out of range".into())
        })
    }

    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}
