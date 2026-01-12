//! Shard management for distributed KV cache.
//!
//! This module handles partitioning of KV cache across multiple devices,
//! tracking which tokens are stored on which device.

use std::ops::Range;

/// Configuration for shard management.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of tokens per shard
    pub shard_size: usize,
    /// Number of available devices
    pub num_devices: usize,
    /// Maximum total tokens supported
    pub max_tokens: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            shard_size: 100_000, // 100K tokens per shard
            num_devices: 1,
            max_tokens: 2_000_000, // 2M tokens
        }
    }
}

impl ShardConfig {
    /// Create config for single GPU.
    pub fn single_gpu(max_tokens: usize) -> Self {
        Self {
            shard_size: max_tokens,
            num_devices: 1,
            max_tokens,
        }
    }

    /// Create config for multi-GPU setup.
    pub fn multi_gpu(num_devices: usize, max_tokens: usize) -> Self {
        let shard_size = (max_tokens / num_devices).max(10_000);
        Self {
            shard_size,
            num_devices,
            max_tokens,
        }
    }

    /// Calculate number of shards needed for given token count.
    pub fn num_shards(&self, num_tokens: usize) -> usize {
        (num_tokens + self.shard_size - 1) / self.shard_size
    }
}

/// Location of a shard in the distributed system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardLocation {
    /// Token range covered by this shard
    pub token_start: usize,
    pub token_end: usize,
    /// Device ID where this shard is stored
    pub device_id: usize,
    /// Shard index within the device
    pub local_shard_id: usize,
}

impl ShardLocation {
    /// Check if this shard contains the given token.
    pub fn contains(&self, token_idx: usize) -> bool {
        token_idx >= self.token_start && token_idx < self.token_end
    }

    /// Get the token range as a Range.
    pub fn range(&self) -> Range<usize> {
        self.token_start..self.token_end
    }

    /// Get the number of tokens in this shard.
    pub fn len(&self) -> usize {
        self.token_end - self.token_start
    }

    /// Check if shard is empty.
    pub fn is_empty(&self) -> bool {
        self.token_start >= self.token_end
    }
}

/// Manages shard allocation and lookup.
#[derive(Debug)]
pub struct ShardManager {
    /// Configuration
    config: ShardConfig,
    /// Allocated shards
    shards: Vec<ShardLocation>,
    /// Current total tokens
    current_tokens: usize,
}

impl ShardManager {
    /// Create a new shard manager.
    pub fn new(config: ShardConfig) -> Self {
        Self {
            config,
            shards: Vec::new(),
            current_tokens: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }

    /// Allocate shards for new tokens.
    ///
    /// Returns the locations of newly allocated shards.
    pub fn allocate(&mut self, num_new_tokens: usize) -> Vec<ShardLocation> {
        let mut new_shards = Vec::new();
        let mut remaining = num_new_tokens;
        let mut token_pos = self.current_tokens;

        while remaining > 0 {
            // Find or create shard for this position
            let device_id = (token_pos / self.config.shard_size) % self.config.num_devices;
            let local_shard_id = token_pos / (self.config.shard_size * self.config.num_devices);

            // Calculate shard boundaries
            let shard_start = (token_pos / self.config.shard_size) * self.config.shard_size;
            let shard_end = (shard_start + self.config.shard_size).min(self.config.max_tokens);

            // How many tokens fit in this shard?
            let space_in_shard = shard_end - token_pos;
            let tokens_to_add = remaining.min(space_in_shard);

            // Check if we need a new shard or extend existing
            let needs_new_shard = self.shards.is_empty()
                || self.shards.last().map(|s| s.token_end).unwrap_or(0) <= token_pos;

            if needs_new_shard {
                let location = ShardLocation {
                    token_start: token_pos,
                    token_end: token_pos + tokens_to_add,
                    device_id,
                    local_shard_id,
                };
                new_shards.push(location);
                self.shards.push(location);
            } else if let Some(last) = self.shards.last_mut() {
                // Extend the last shard
                last.token_end = token_pos + tokens_to_add;
                new_shards.push(*last);
            }

            token_pos += tokens_to_add;
            remaining -= tokens_to_add;
        }

        self.current_tokens = token_pos;
        new_shards
    }

    /// Find the shard containing a specific token.
    pub fn find_shard(&self, token_idx: usize) -> Option<&ShardLocation> {
        // Binary search for efficiency
        let idx = self.shards.partition_point(|s| s.token_end <= token_idx);
        self.shards.get(idx).filter(|s| s.contains(token_idx))
    }

    /// Get all shards in order.
    pub fn shards(&self) -> &[ShardLocation] {
        &self.shards
    }

    /// Get shards for a token range.
    pub fn shards_for_range(&self, range: Range<usize>) -> Vec<&ShardLocation> {
        self.shards
            .iter()
            .filter(|s| s.token_start < range.end && s.token_end > range.start)
            .collect()
    }

    /// Get the current number of tokens.
    pub fn current_tokens(&self) -> usize {
        self.current_tokens
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Check if capacity is available for more tokens.
    pub fn has_capacity(&self, additional: usize) -> bool {
        self.current_tokens + additional <= self.config.max_tokens
    }

    /// Reset the shard manager.
    pub fn reset(&mut self) {
        self.shards.clear();
        self.current_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_config_defaults() {
        let config = ShardConfig::default();
        assert_eq!(config.shard_size, 100_000);
        assert_eq!(config.num_devices, 1);
        assert_eq!(config.max_tokens, 2_000_000);
    }

    #[test]
    fn test_shard_manager_allocate() {
        let config = ShardConfig {
            shard_size: 100,
            num_devices: 2,
            max_tokens: 1000,
        };
        let mut manager = ShardManager::new(config);

        // Allocate first batch
        let shards = manager.allocate(50);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].token_start, 0);
        assert_eq!(shards[0].token_end, 50);
        assert_eq!(shards[0].device_id, 0);

        // Allocate second batch (extends first shard)
        let shards = manager.allocate(50);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].token_end, 100);

        // Allocate third batch (new shard on device 1)
        let shards = manager.allocate(50);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].token_start, 100);
        assert_eq!(shards[0].device_id, 1);
    }

    #[test]
    fn test_shard_manager_find() {
        let config = ShardConfig {
            shard_size: 100,
            num_devices: 1,
            max_tokens: 1000,
        };
        let mut manager = ShardManager::new(config);

        manager.allocate(250);

        let shard = manager.find_shard(50);
        assert!(shard.is_some());
        assert_eq!(shard.unwrap().token_start, 0);

        let shard = manager.find_shard(150);
        assert!(shard.is_some());
        assert_eq!(shard.unwrap().token_start, 100);

        let shard = manager.find_shard(500);
        assert!(shard.is_none());
    }

    #[test]
    fn test_multi_device_distribution() {
        let config = ShardConfig {
            shard_size: 100,
            num_devices: 4,
            max_tokens: 1000,
        };
        let mut manager = ShardManager::new(config);

        // Allocate enough to span multiple devices
        manager.allocate(400);

        assert_eq!(manager.num_shards(), 4);

        // Check device distribution
        let devices: Vec<_> = manager.shards().iter().map(|s| s.device_id).collect();
        assert_eq!(devices, vec![0, 1, 2, 3]);
    }
}
