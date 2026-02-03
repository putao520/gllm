//! Scheduler building blocks (HGAL).
//!
//! This module hosts the gang-aware, LIRS-inspired scheduling logic that
//! complements the engine layer. It is intentionally independent from the
//! backend so it can be unit-tested without GPU involvement.

pub mod hgal;
pub mod types;

pub use hgal::{HGALConfig, HGALScheduler};
pub use types::{GroupState, PageMetadata, SequenceGroup};
