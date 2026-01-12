//! Coordinator for distributed KV cache operations.
//!
//! The coordinator handles routing of KV operations across devices
//! and aggregation of results for attention computation.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::shard_manager::ShardLocation;

/// Configuration for the coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Number of devices
    pub num_devices: usize,
    /// Enable strict ordering for determinism
    pub strict_ordering: bool,
    /// Timeout for cross-device operations (ms)
    pub timeout_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            strict_ordering: true,
            timeout_ms: 30_000,
        }
    }
}

/// Statistics for coordinator operations.
#[derive(Debug, Default)]
pub struct CoordinatorStats {
    /// Total KV appends
    pub appends: AtomicUsize,
    /// Total attention computations
    pub attentions: AtomicUsize,
    /// Cross-device transfers
    pub transfers: AtomicUsize,
    /// Active sequences
    pub active_sequences: AtomicUsize,
}

impl CoordinatorStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_append(&self) {
        self.appends.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_attention(&self) {
        self.attentions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_transfer(&self) {
        self.transfers.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_sequences(&self) {
        self.active_sequences.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_sequences(&self) {
        self.active_sequences.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CoordinatorStatsSnapshot {
        CoordinatorStatsSnapshot {
            appends: self.appends.load(Ordering::Relaxed),
            attentions: self.attentions.load(Ordering::Relaxed),
            transfers: self.transfers.load(Ordering::Relaxed),
            active_sequences: self.active_sequences.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of coordinator statistics.
#[derive(Debug, Clone)]
pub struct CoordinatorStatsSnapshot {
    pub appends: usize,
    pub attentions: usize,
    pub transfers: usize,
    pub active_sequences: usize,
}

/// Coordinator for distributed KV cache operations.
///
/// In a full implementation, this would handle:
/// 1. Routing KV append operations to the correct device
/// 2. Coordinating attention computation across shards
/// 3. Managing cross-device data transfers
/// 4. Ensuring deterministic ordering
///
/// Current implementation is a placeholder for single-device operation.
pub struct Coordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Statistics
    stats: CoordinatorStats,
}

impl Coordinator {
    /// Create a new coordinator.
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            stats: CoordinatorStats::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Get the statistics.
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }

    /// Determine target device for a new KV append.
    pub fn route_append(&self, current_tokens: usize, shard_size: usize) -> usize {
        if self.config.num_devices == 1 {
            return 0;
        }

        // Round-robin based on shard
        let shard_id = current_tokens / shard_size;
        shard_id % self.config.num_devices
    }

    /// Plan attention computation across shards.
    ///
    /// Returns an ordered list of shards to process.
    pub fn plan_attention<'a>(&self, shards: &'a [ShardLocation]) -> Vec<&'a ShardLocation> {
        if self.config.strict_ordering {
            // Strict ordering: process in token order
            let mut ordered: Vec<_> = shards.iter().collect();
            ordered.sort_by_key(|s| s.token_start);
            ordered
        } else {
            // Relaxed ordering: can process in any order (potential parallelism)
            shards.iter().collect()
        }
    }

    /// Check if a shard requires cross-device transfer.
    pub fn needs_transfer(&self, _shard: &ShardLocation, _local_device: usize) -> bool {
        // In current single-device implementation, no transfers needed
        // In multi-device: return shard.device_id != local_device
        false
    }

    /// Record a new sequence.
    pub fn register_sequence(&self) -> usize {
        self.stats.increment_sequences();
        // In full implementation: allocate sequence ID, set up routing
        0
    }

    /// Release a sequence.
    pub fn release_sequence(&self, _seq_id: usize) {
        self.stats.decrement_sequences();
        // In full implementation: clean up routing, free resources
    }
}

/// Aggregation strategy for multi-shard attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Sequential aggregation (deterministic, slower)
    Sequential,
    /// Parallel aggregation (faster, may have numerical differences)
    Parallel,
    /// Hierarchical aggregation (balanced)
    Hierarchical,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::Sequential
    }
}

/// Builder for attention aggregation plan.
pub struct AggregationPlan {
    /// Shards to process
    shards: Vec<ShardLocation>,
    /// Aggregation strategy
    strategy: AggregationStrategy,
    /// Strict ordering
    strict_order: bool,
}

impl AggregationPlan {
    /// Create a new aggregation plan.
    pub fn new(shards: Vec<ShardLocation>) -> Self {
        Self {
            shards,
            strategy: AggregationStrategy::default(),
            strict_order: true,
        }
    }

    /// Set the aggregation strategy.
    pub fn with_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set strict ordering.
    pub fn with_strict_order(mut self, strict: bool) -> Self {
        self.strict_order = strict;
        self
    }

    /// Get the ordered shards.
    pub fn ordered_shards(&self) -> Vec<&ShardLocation> {
        let mut shards: Vec<_> = self.shards.iter().collect();
        if self.strict_order {
            shards.sort_by_key(|s| s.token_start);
        }
        shards
    }

    /// Get the strategy.
    pub fn strategy(&self) -> AggregationStrategy {
        self.strategy
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_routing() {
        let config = CoordinatorConfig {
            num_devices: 4,
            ..Default::default()
        };
        let coord = Coordinator::new(config);

        // Test routing with shard_size = 100
        assert_eq!(coord.route_append(0, 100), 0);
        assert_eq!(coord.route_append(100, 100), 1);
        assert_eq!(coord.route_append(200, 100), 2);
        assert_eq!(coord.route_append(400, 100), 0); // Wraps around
    }

    #[test]
    fn test_coordinator_stats() {
        let coord = Coordinator::new(CoordinatorConfig::default());

        coord.stats.record_append();
        coord.stats.record_append();
        coord.stats.record_attention();

        let snapshot = coord.stats.snapshot();
        assert_eq!(snapshot.appends, 2);
        assert_eq!(snapshot.attentions, 1);
    }

    #[test]
    fn test_aggregation_plan() {
        let shards = vec![
            ShardLocation {
                token_start: 200,
                token_end: 300,
                device_id: 1,
                local_shard_id: 1,
            },
            ShardLocation {
                token_start: 0,
                token_end: 100,
                device_id: 0,
                local_shard_id: 0,
            },
            ShardLocation {
                token_start: 100,
                token_end: 200,
                device_id: 0,
                local_shard_id: 1,
            },
        ];

        let plan = AggregationPlan::new(shards);
        let ordered = plan.ordered_shards();

        // Should be sorted by token_start
        assert_eq!(ordered[0].token_start, 0);
        assert_eq!(ordered[1].token_start, 100);
        assert_eq!(ordered[2].token_start, 200);
    }
}
