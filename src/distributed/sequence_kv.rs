//! Per-sequence isolated KV storage.
//!
//! Each sequence has its own isolated KV storage, eliminating
//! any shared mutable state between concurrent requests.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Per-layer KV storage for a sequence.
#[derive(Debug, Default)]
pub struct LayerKV {
    /// Keys for this layer [accumulated_len, num_heads, head_dim]
    pub keys: Option<Vec<f32>>,
    /// Values for this layer [accumulated_len, num_heads, head_dim]
    pub values: Option<Vec<f32>>,
    /// Current sequence length for this layer
    pub seq_len: usize,
}

impl LayerKV {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            seq_len: 0,
        }
    }

    /// Append new KV to this layer.
    pub fn append(&mut self, k: Vec<f32>, v: Vec<f32>, new_len: usize) {
        self.keys = Some(match self.keys.take() {
            Some(mut existing) => {
                existing.extend_from_slice(&k);
                existing
            }
            None => k,
        });

        self.values = Some(match self.values.take() {
            Some(mut existing) => {
                existing.extend_from_slice(&v);
                existing
            }
            None => v,
        });

        self.seq_len += new_len;
    }

    /// Get the current KV buffers.
    pub fn get(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }
}

/// Per-sequence KV storage.
///
/// Each sequence has completely isolated storage, ensuring
/// no data races between concurrent requests.
pub struct SequenceKV<D> {
    /// Sequence ID
    id: usize,
    /// KV storage per layer
    layers: Vec<LayerKV>,
    /// Shard manager for distributed storage
    shard_manager: ShardManager,
    /// Configuration
    config: SequenceConfig,
    /// Device marker
    device: D,
}

impl<D: Clone> SequenceKV<D> {
    /// Create a new sequence KV storage.
    pub fn new(id: usize, config: SequenceConfig, device: D) -> Self {
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
    pub fn append(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>, new_len: usize) {
        if layer >= self.layers.len() {
            return;
        }

        if layer == 0 {
            self.shard_manager.allocate(new_len);
        }

        self.layers[layer].append(k, v, new_len);
    }

    /// Get KV for a specific layer.
    pub fn get_kv(&self, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
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
    pub fn device(&self) -> &D {
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
pub struct SequenceFactory<D> {
    /// Next sequence ID
    next_id: AtomicUsize,
    /// Configuration template
    config: SequenceConfig,
    /// Device marker
    device: D,
}

impl<D: Clone> SequenceFactory<D> {
    /// Create a new sequence factory.
    pub fn new(config: SequenceConfig, device: D) -> Self {
        Self {
            next_id: AtomicUsize::new(0),
            config,
            device,
        }
    }

    /// Create a new sequence and handle.
    pub fn create(&self) -> (SequenceHandle, SequenceKV<D>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = SequenceHandle::new(id);
        let sequence = SequenceKV::new(id, self.config.clone(), self.device.clone());
        (handle, sequence)
    }
}
