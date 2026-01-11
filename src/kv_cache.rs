use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub struct KVCache<B: Backend> {
    keys: Vec<Tensor<B, 4>>,
    values: Vec<Tensor<B, 4>>,
    num_layers: usize,
    max_len: usize,
    current_len: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn preallocate(
        num_layers: usize,
        max_len: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            keys.push(Tensor::zeros([batch_size, num_heads, max_len, head_dim], device));
            values.push(Tensor::zeros([batch_size, num_heads, max_len, head_dim], device));
        }

        Self {
            keys,
            values,
            num_layers,
            max_len,
            current_len: 0,
        }
    }

    pub fn update(
        &mut self,
        layer: usize,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if layer >= self.num_layers {
            return (new_k, new_v);
        }

        let [batch_size, num_heads, seq_len, head_dim] = new_k.dims();
        if self.max_len == 0 {
            return (new_k, new_v);
        }

        let start = self.current_len;
        let end = start + seq_len;
        assert!(
            end <= self.max_len,
            "KV cache exceeded max_len ({} > {})",
            end,
            self.max_len
        );

        let updated_k = self.keys[layer].clone().slice_assign(
            [0..batch_size, 0..num_heads, start..end, 0..head_dim],
            new_k,
        );
        let updated_v = self.values[layer].clone().slice_assign(
            [0..batch_size, 0..num_heads, start..end, 0..head_dim],
            new_v,
        );

        self.keys[layer] = updated_k;
        self.values[layer] = updated_v;

        if layer + 1 == self.num_layers {
            self.current_len = end;
        }

        let cached_k = self.keys[layer]
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..end, 0..head_dim]);
        let cached_v = self.values[layer]
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..end, 0..head_dim]);

        (cached_k, cached_v)
    }

    pub fn seq_len(&self) -> usize {
        self.current_len
    }
}
