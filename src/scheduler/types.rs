use std::time::Instant;

use gllm_kernels::kernel_types::{PageId, PageState, RequestId};

/// A group of pages that belong to the same request/sequence.
/// Gang scheduling evicts whole groups to avoid intra-sequence fragmentation.
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    pub id: RequestId,
    pub pages: Vec<PageId>,
    pub state: GroupState,
    pub access_count: usize,
    pub last_access: Instant,
    /// Pinned groups are immune to eviction (e.g., during prefill).
    pub is_pinned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    Running,
    Swapped,
    Paused,
}

/// Per-page metadata needed by HGAL (Hybrid Gang-Aware LIRS).
#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub page_id: PageId,
    pub sequence_id: Option<RequestId>,
    /// Inter-Reference Recency (IRR) value for LIRS-style scoring.
    pub recency: usize,
    pub access_count: usize,
    pub last_access: Instant,
    pub swap_in_time: Option<Instant>,
    /// Whether this page currently belongs to the LIR working set.
    pub is_lir: bool,
    /// Current state mirrored from backend for eviction decisions.
    pub state: PageState,
    /// Warm protection expiry; None means not under warm-up protection.
    pub warm_until: Option<Instant>,
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        }
    }
}
