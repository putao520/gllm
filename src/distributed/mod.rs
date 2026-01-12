//! Distributed KV cache and multi-GPU support for ultra-long contexts.
//!
//! This module provides infrastructure for distributing KV cache across
//! multiple GPUs, enabling 2M+ token contexts that exceed single-GPU memory.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Distributed PagedKVCache                          │
//! │                                                                      │
//! │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
//! │  │    GPU 0      │  │    GPU 1      │  │    GPU 2      │  ...      │
//! │  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │           │
//! │  │  │ Shard 0 │  │  │  │ Shard 1 │  │  │  │ Shard 2 │  │           │
//! │  │  │ 0-700K  │  │  │  │700K-1.4M│  │  │  │1.4M-2M  │  │           │
//! │  │  │ tokens  │  │  │  │ tokens  │  │  │  │ tokens  │  │           │
//! │  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │           │
//! │  └───────────────┘  └───────────────┘  └───────────────┘           │
//! │         │                  │                  │                     │
//! │         └──────────────────┼──────────────────┘                     │
//! │                            │                                        │
//! │                    ┌───────▼───────┐                                │
//! │                    │  Coordinator  │                                │
//! │                    │  - Routing    │                                │
//! │                    │  - Aggregation│                                │
//! │                    └───────────────┘                                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Components
//!
//! - [`ShardManager`]: Manages partitioning of KV cache across devices
//! - [`SequenceKV`]: Per-sequence isolated KV storage
//! - [`Coordinator`]: Handles routing and result aggregation
//!
//! # Design Principles
//!
//! 1. **Sequence isolation**: Each sequence has independent storage
//! 2. **No shared mutable state**: Concurrency through isolation
//! 3. **Deterministic aggregation**: Strict ordering for reproducibility

mod shard_manager;
mod sequence_kv;
mod coordinator;

pub use shard_manager::{ShardConfig, ShardManager, ShardLocation};
pub use sequence_kv::{SequenceKV, SequenceHandle, SequenceConfig, SequenceFactory};
pub use coordinator::{Coordinator, CoordinatorConfig};
