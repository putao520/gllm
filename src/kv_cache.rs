use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub struct KVCache<B: Backend> {
    keys: Vec<Option<Tensor<B, 4>>>,
    values: Vec<Option<Tensor<B, 4>>>,
    num_layers: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new(num_layers: usize) -> Self {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        keys.resize_with(num_layers, || None);
        values.resize_with(num_layers, || None);

        Self {
            keys,
            values,
            num_layers,
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

        let updated_k = match self.keys[layer].take() {
            Some(prev) => Tensor::cat(vec![prev, new_k], 2),
            None => new_k,
        };
        let updated_v = match self.values[layer].take() {
            Some(prev) => Tensor::cat(vec![prev, new_v], 2),
            None => new_v,
        };

        self.keys[layer] = Some(updated_k.clone());
        self.values[layer] = Some(updated_v.clone());

        (updated_k, updated_v)
    }

    pub fn seq_len(&self) -> usize {
        self.keys
            .iter()
            .filter_map(|tensor| {
                tensor.as_ref().map(|kv| {
                    let [_batch, _heads, seq_len, _dim] = kv.dims();
                    seq_len
                })
            })
            .next()
            .unwrap_or(0)
    }
}
