//! Simplified paged attention KV cache (burn-free).

use crate::distributed::{SequenceConfig, SequenceFactory, SequenceHandle, SequenceKV};

#[derive(Clone, Debug)]
struct LayerCache {
    keys: Vec<Vec<f32>>,  // per sequence
    values: Vec<Vec<f32>>,
    seq_lens: Vec<usize>,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_lens: Vec::new(),
        }
    }

    fn allocate(&mut self) -> usize {
        let id = self.seq_lens.len();
        self.keys.push(Vec::new());
        self.values.push(Vec::new());
        self.seq_lens.push(0);
        id
    }
}

#[derive(Clone, Debug)]
pub struct PagedKVCache {
    layers: Vec<LayerCache>,
    num_heads: usize,
    head_dim: usize,
}

impl PagedKVCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerCache::new());
        }
        Self {
            layers,
            num_heads,
            head_dim,
        }
    }

    pub fn allocate_sequence(&mut self) -> usize {
        let mut seq_id = None;
        for layer in &mut self.layers {
            let id = layer.allocate();
            seq_id = Some(id);
        }
        seq_id.unwrap_or(0)
    }

    pub fn append(
        &mut self,
        layer: usize,
        seq_id: usize,
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<(), &'static str> {
        let layer = self.layers.get_mut(layer).ok_or("invalid layer")?;
        let keys = layer.keys.get_mut(seq_id).ok_or("invalid sequence")?;
        let values = layer.values.get_mut(seq_id).ok_or("invalid sequence")?;
        keys.extend_from_slice(k);
        values.extend_from_slice(v);
        layer.seq_lens[seq_id] += seq_len;
        Ok(())
    }

    pub fn get_kv(&self, layer: usize, seq_id: usize) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        let layer = self.layers.get(layer).ok_or("invalid layer")?;
        let keys = layer.keys.get(seq_id).ok_or("invalid sequence")?;
        let values = layer.values.get(seq_id).ok_or("invalid sequence")?;
        Ok((keys.clone(), values.clone()))
    }

    pub fn seq_len(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        let layer = self.layers.get(layer).ok_or("invalid layer")?;
        layer.seq_lens.get(seq_id).copied().ok_or("invalid sequence")
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

pub struct IsolatedPagedKVCache {
    factory: SequenceFactory<()>,
}

impl IsolatedPagedKVCache {
    pub fn new(config: SequenceConfig) -> Self {
        Self {
            factory: SequenceFactory::new(config, ()),
        }
    }

    pub fn create_sequence(&self) -> (SequenceHandle, SequenceKV<()>) {
        self.factory.create()
    }
}
