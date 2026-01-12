//! Per-sequence isolated KV storage.
//!
//! Each sequence has its own isolated KV storage, eliminating
//! any shared mutable state between concurrent requests.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::shard_manager::{ShardConfig, ShardManager, ShardLocation};

/// Configuration for sequence KV storage.
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Shard configuration
    pub shard: ShardConfig,
    /// Number of layers
    pub num_layers: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Block size for paged allocation
    pub block_size: usize,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            shard: ShardConfig::default(),
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
        }
    }
}

/// Handle to a sequence for safe access.
///
/// This handle is NOT Send or Sync, forcing single-threaded access
/// to each sequence while allowing different sequences to be
/// processed in parallel on different threads.
pub struct SequenceHandle {
    /// Sequence ID
    id: usize,
    /// Prevent Send/Sync
    _marker: PhantomData<*const ()>,
}

impl SequenceHandle {
    /// Create a new sequence handle.
    fn new(id: usize) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }

    /// Get the sequence ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

// Explicitly NOT implementing Send/Sync for SequenceHandle
// This forces each sequence to be accessed from a single thread

/// Per-layer KV storage for a sequence.
#[derive(Debug)]
pub struct LayerKV<B: Backend> {
    /// Keys for this layer [accumulated_len, num_heads, head_dim]
    pub keys: Option<Tensor<B, 3>>,
    /// Values for this layer [accumulated_len, num_heads, head_dim]
    pub values: Option<Tensor<B, 3>>,
    /// Current sequence length for this layer
    pub seq_len: usize,
}

impl<B: Backend> LayerKV<B> {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            seq_len: 0,
        }
    }

    /// Append new KV to this layer.
    pub fn append(&mut self, k: Tensor<B, 3>, v: Tensor<B, 3>) {
        let new_len = k.dims()[0];

        self.keys = Some(match self.keys.take() {
            Some(existing) => Tensor::cat(vec![existing, k], 0),
            None => k,
        });

        self.values = Some(match self.values.take() {
            Some(existing) => Tensor::cat(vec![existing, v], 0),
            None => v,
        });

        self.seq_len += new_len;
    }

    /// Get the current KV tensors.
    pub fn get(&self) -> Option<(Tensor<B, 3>, Tensor<B, 3>)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }
}

impl<B: Backend> Default for LayerKV<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-sequence KV storage.
///
/// Each sequence has completely isolated storage, ensuring
/// no data races between concurrent requests.
pub struct SequenceKV<B: Backend> {
    /// Sequence ID
    id: usize,
    /// KV storage per layer
    layers: Vec<LayerKV<B>>,
    /// Shard manager for distributed storage
    shard_manager: ShardManager,
    /// Configuration
    config: SequenceConfig,
    /// Device for tensor allocation
    device: B::Device,
}

impl<B: Backend> SequenceKV<B> {
    /// Create a new sequence KV storage.
    pub fn new(id: usize, config: SequenceConfig, device: B::Device) -> Self {
        let layers = (0..config.num_layers).map(|_| LayerKV::new()).collect();
        let shard_manager = ShardManager::new(config.shard.clone());

        Self {
            id,
            layers,
            shard_manager,
            config,
            device,
        }
    }

    /// Get the sequence ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Append KV for a specific layer.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k` - Key tensor [new_len, num_heads, head_dim]
    /// * `v` - Value tensor [new_len, num_heads, head_dim]
    pub fn append(&mut self, layer: usize, k: Tensor<B, 3>, v: Tensor<B, 3>) {
        if layer >= self.layers.len() {
            return;
        }

        let new_len = k.dims()[0];

        // Update shard allocation (only for layer 0 to avoid double-counting)
        if layer == 0 {
            self.shard_manager.allocate(new_len);
        }

        self.layers[layer].append(k, v);
    }

    /// Get KV for a specific layer.
    pub fn get_kv(&self, layer: usize) -> Option<(Tensor<B, 3>, Tensor<B, 3>)> {
        self.layers.get(layer).and_then(|l| l.get())
    }

    /// Get the current sequence length.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Get the shard locations for this sequence.
    pub fn shards(&self) -> &[ShardLocation] {
        self.shard_manager.shards()
    }

    /// Get the device.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get the configuration.
    pub fn config(&self) -> &SequenceConfig {
        &self.config
    }

    /// Check if the sequence has capacity for more tokens.
    pub fn has_capacity(&self, additional: usize) -> bool {
        self.shard_manager.has_capacity(additional)
    }

    /// Reset the sequence (clear all KV).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.keys = None;
            layer.values = None;
            layer.seq_len = 0;
        }
        self.shard_manager.reset();
    }
}

/// Factory for creating isolated sequences.
pub struct SequenceFactory<B: Backend> {
    /// Next sequence ID
    next_id: AtomicUsize,
    /// Configuration template
    config: SequenceConfig,
    /// Device
    device: B::Device,
}

impl<B: Backend> SequenceFactory<B> {
    /// Create a new sequence factory.
    pub fn new(config: SequenceConfig, device: B::Device) -> Self {
        Self {
            next_id: AtomicUsize::new(0),
            config,
            device,
        }
    }

    /// Create a new isolated sequence.
    pub fn create(&self) -> (SequenceHandle, SequenceKV<B>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = SequenceHandle::new(id);
        let kv = SequenceKV::new(id, self.config.clone(), self.device.clone());
        (handle, kv)
    }

    /// Get the number of sequences created.
    pub fn num_created(&self) -> usize {
        self.next_id.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sequence_kv_basic() {
        let device = <TestBackend as Backend>::Device::default();
        let config = SequenceConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            ..Default::default()
        };

        let mut seq = SequenceKV::<TestBackend>::new(0, config, device.clone());

        // Append to layer 0
        let k = Tensor::zeros([10, 4, 8], &device);
        let v = Tensor::zeros([10, 4, 8], &device);
        seq.append(0, k, v);

        assert_eq!(seq.seq_len(), 10);

        // Get KV
        let (k, v) = seq.get_kv(0).unwrap();
        assert_eq!(k.dims(), [10, 4, 8]);
        assert_eq!(v.dims(), [10, 4, 8]);
    }

    #[test]
    fn test_sequence_kv_multiple_appends() {
        let device = <TestBackend as Backend>::Device::default();
        let config = SequenceConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            ..Default::default()
        };

        let mut seq = SequenceKV::<TestBackend>::new(0, config, device.clone());

        // Multiple appends
        for _ in 0..5 {
            let k = Tensor::zeros([8, 2, 4], &device);
            let v = Tensor::zeros([8, 2, 4], &device);
            seq.append(0, k, v);
        }

        assert_eq!(seq.seq_len(), 40);

        let (k, _) = seq.get_kv(0).unwrap();
        assert_eq!(k.dims(), [40, 2, 4]);
    }

    #[test]
    fn test_sequence_factory() {
        let device = <TestBackend as Backend>::Device::default();
        let config = SequenceConfig::default();

        let factory = SequenceFactory::<TestBackend>::new(config, device);

        let (handle1, _kv1) = factory.create();
        let (handle2, _kv2) = factory.create();

        assert_eq!(handle1.id(), 0);
        assert_eq!(handle2.id(), 1);
        assert_eq!(factory.num_created(), 2);
    }

    #[test]
    fn test_sequence_isolation() {
        let device = <TestBackend as Backend>::Device::default();
        let config = SequenceConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            ..Default::default()
        };

        let factory = SequenceFactory::<TestBackend>::new(config, device.clone());

        let (_, mut seq1) = factory.create();
        let (_, mut seq2) = factory.create();

        // Modify seq1
        let k = Tensor::ones([5, 2, 4], &device);
        let v = Tensor::ones([5, 2, 4], &device);
        seq1.append(0, k, v);

        // seq2 should be unaffected
        assert_eq!(seq1.seq_len(), 5);
        assert_eq!(seq2.seq_len(), 0);
    }
}
