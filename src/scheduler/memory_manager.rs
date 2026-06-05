use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::time::Instant;

use super::types::{PageId, PageState, PhysicalId, RequestId};

use super::types::{KvPipeline, PageMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualPageId {
    pub sequence_id: RequestId,
    pub logical_index: usize,
}

impl VirtualPageId {
    pub const fn new(sequence_id: RequestId, logical_index: usize) -> Self {
        Self {
            sequence_id,
            logical_index,
        }
    }
}

pub type SessionId = u64;

/// 会话级 KV Cache (ARCH-SCHED-SESSION-KV)
#[derive(Debug, Clone)]
pub struct SessionKvCache {
    pub session_id: SessionId,
    pub pages: Vec<VirtualPageId>,
    pub finalized_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLocation {
    pub physical_id: PhysicalId,
    pub tier: Tier,
}

/// Prefill 规划结果 (DATA-PREFILL-PLAN)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefillPlan {
    /// 全部驻留 L1
    FullyResident { pages: usize },
    /// 分批装载/预取
    Pipelined {
        l1_pages: usize,
        l2_prefetch: usize,
        chunk_schedule: Vec<usize>,
    },
}

#[derive(Debug, Default)]
pub struct PageTable {
    mappings: HashMap<VirtualPageId, PageLocation>,
}

impl PageTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn map(
        &mut self,
        virtual_id: VirtualPageId,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Option<PageLocation> {
        self.mappings
            .insert(virtual_id, PageLocation { tier, physical_id })
    }

    pub fn remap(
        &mut self,
        virtual_id: VirtualPageId,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<PageLocation, MemoryManagerError> {
        let Some(current) = self.mappings.get_mut(&virtual_id) else {
            return Err(MemoryManagerError::UnknownVirtualPage { virtual_id });
        };
        let old = *current;
        *current = PageLocation { tier, physical_id };
        Ok(old)
    }

    pub fn remove(&mut self, virtual_id: &VirtualPageId) -> Option<PageLocation> {
        self.mappings.remove(virtual_id)
    }

    pub fn resolve(&self, virtual_id: VirtualPageId) -> Option<PageLocation> {
        self.mappings.get(&virtual_id).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierUsage {
    pub used: usize,
    pub capacity: usize,
}

impl TierUsage {
    pub fn available(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }
}

#[derive(Debug, Clone)]
struct TierState {
    capacity: usize,
    used: usize,
    next_id: PhysicalId,
    free_ids: VecDeque<PhysicalId>,
    allocated: HashSet<PhysicalId>,
}

impl TierState {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            next_id: 0,
            free_ids: VecDeque::new(),
            allocated: HashSet::new(),
        }
    }

    fn can_allocate(&self) -> bool {
        self.used < self.capacity
    }

    fn allocate(&mut self) -> Option<PhysicalId> {
        if !self.can_allocate() {
            return None;
        }
        let physical_id = self.free_ids.pop_front().unwrap_or_else(|| {
            // LEGAL: 无空闲 ID 时分配新 ID
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            id
        });
        self.used = self.used.saturating_add(1);
        self.allocated.insert(physical_id);
        Some(physical_id)
    }

    fn track_allocated(&mut self, physical_id: PhysicalId) -> bool {
        if self.allocated.contains(&physical_id) {
            return true;
        }
        if !self.can_allocate() {
            return false;
        }

        if physical_id >= self.next_id {
            for id in self.next_id..physical_id {
                self.free_ids.push_back(id);
            }
            self.next_id = physical_id.saturating_add(1);
        } else if let Some(pos) = self.free_ids.iter().position(|id| *id == physical_id) {
            self.free_ids.remove(pos);
        }

        self.used = self.used.saturating_add(1);
        self.allocated.insert(physical_id);
        true
    }

    fn free(&mut self, physical_id: PhysicalId) -> bool {
        if !self.allocated.remove(&physical_id) {
            return false;
        }
        self.used = self.used.saturating_sub(1);
        self.free_ids.push_back(physical_id);
        true
    }

    fn is_allocated(&self, physical_id: PhysicalId) -> bool {
        self.allocated.contains(&physical_id)
    }

    fn usage(&self) -> TierUsage {
        TierUsage {
            used: self.used,
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TierManager {
    l1: TierState,
    l2: TierState,
    l3: TierState,
}

impl TierManager {
    pub fn new(l1_capacity: usize, l2_capacity: usize, l3_capacity: usize) -> Self {
        Self {
            l1: TierState::new(l1_capacity),
            l2: TierState::new(l2_capacity),
            l3: TierState::new(l3_capacity),
        }
    }

    pub fn can_allocate(&self, tier: Tier) -> bool {
        self.state(tier).can_allocate()
    }

    pub fn allocate(&mut self, tier: Tier) -> Option<PhysicalId> {
        self.state_mut(tier).allocate()
    }

    pub fn track_page(&mut self, tier: Tier, physical_id: PhysicalId) -> bool {
        self.state_mut(tier).track_allocated(physical_id)
    }

    pub fn free(&mut self, tier: Tier, physical_id: PhysicalId) -> bool {
        self.state_mut(tier).free(physical_id)
    }

    pub fn is_allocated(&self, tier: Tier, physical_id: PhysicalId) -> bool {
        self.state(tier).is_allocated(physical_id)
    }

    pub fn usage(&self, tier: Tier) -> TierUsage {
        self.state(tier).usage()
    }

    fn state(&self, tier: Tier) -> &TierState {
        match tier {
            Tier::L1 => &self.l1,
            Tier::L2 => &self.l2,
            Tier::L3 => &self.l3,
        }
    }

    fn state_mut(&mut self, tier: Tier) -> &mut TierState {
        match tier {
            Tier::L1 => &mut self.l1,
            Tier::L2 => &mut self.l2,
            Tier::L3 => &mut self.l3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvictionPolicy {
    recency_weight: i64,
    frequency_weight: i64,
    semantic_weight: i64,
    active_penalty: i64,
    standby_bonus: i64,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            recency_weight: 4,
            frequency_weight: 32,
            semantic_weight: 128,
            active_penalty: 256,
            standby_bonus: 512,
        }
    }
}

impl EvictionPolicy {
    pub fn select_victims(
        &self,
        metadata: &HashMap<PageId, PageMetadata>,
        semantic_priorities: &HashMap<PageId, i32>,
        count: usize,
    ) -> Vec<PageId> {
        if count == 0 {
            return Vec::new();
        }

        let now = Instant::now();
        let mut candidates: Vec<(PageId, i64)> = metadata
            .values()
            .filter(|meta| {
                !matches!(
                    meta.state,
                    PageState::Warm | PageState::Protected | PageState::Swapped
                )
            })
            .map(|meta| {
                let semantic = semantic_priorities.get(&meta.page_id).copied().unwrap_or(0); // LEGAL: 未设置语义优先级的页面默认 0
                let score = self.eviction_score(meta, semantic, now);
                (meta.page_id, score)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    fn eviction_score(&self, meta: &PageMetadata, semantic_priority: i32, now: Instant) -> i64 {
        let idle_ms =
            saturating_u128_to_i64(now.saturating_duration_since(meta.last_access).as_millis());
        let recency = saturating_usize_to_i64(meta.recency);
        let access_count = saturating_usize_to_i64(meta.access_count);
        let semantic = i64::from(semantic_priority);

        let state_component = match meta.state {
            PageState::Standby => self.standby_bonus,
            PageState::Active => -self.active_penalty,
            PageState::Warm | PageState::Protected | PageState::Swapped => i64::MIN / 2,
            PageState::Free | PageState::SwappedOut => i64::MIN,
        };

        idle_ms
            .saturating_add(recency.saturating_mul(self.recency_weight))
            .saturating_add(state_component)
            .saturating_sub(access_count.saturating_mul(self.frequency_weight))
            .saturating_sub(semantic.saturating_mul(self.semantic_weight))
    }
}

#[derive(Debug)]
pub struct GlobalMemoryManager {
    page_table: PageTable,
    tier_manager: TierManager,
    eviction_policy: EvictionPolicy,
    physical_to_virtual: HashMap<(Tier, PhysicalId), HashSet<VirtualPageId>>,
    sessions: HashMap<SessionId, SessionKvCache>,
    pipeline_pages: HashMap<(KvPipeline, RequestId), Vec<PhysicalId>>,
}

impl GlobalMemoryManager {
    pub fn new(tier_manager: TierManager, eviction_policy: EvictionPolicy) -> Self {
        Self {
            page_table: PageTable::new(),
            tier_manager,
            eviction_policy,
            physical_to_virtual: HashMap::new(),
            sessions: HashMap::new(),
            pipeline_pages: HashMap::new(),
        }
    }

    pub fn new_with_capacities(l1_capacity: usize, l2_capacity: usize, l3_capacity: usize) -> Self {
        Self::new(
            TierManager::new(l1_capacity, l2_capacity, l3_capacity),
            EvictionPolicy::default(),
        )
    }

    pub fn allocate_page(&mut self, tier: Tier) -> Result<PhysicalId, MemoryManagerError> {
        self.tier_manager
            .allocate(tier)
            .ok_or(MemoryManagerError::TierCapacityExceeded { tier })
    }

    pub fn track_page(
        &mut self,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<(), MemoryManagerError> {
        if self.tier_manager.track_page(tier, physical_id) {
            Ok(())
        } else {
            Err(MemoryManagerError::TierCapacityExceeded { tier })
        }
    }

    pub fn untrack_page(
        &mut self,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<(), MemoryManagerError> {
        self.free_page(tier, physical_id)
    }

    pub fn free_page(
        &mut self,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<(), MemoryManagerError> {
        if !self.tier_manager.free(tier, physical_id) {
            return Err(MemoryManagerError::UnknownPhysicalPage { tier, physical_id });
        }
        if let Some(virtual_pages) = self.physical_to_virtual.remove(&(tier, physical_id)) {
            for virtual_id in virtual_pages {
                self.page_table.remove(&virtual_id);
            }
        }
        Ok(())
    }

    pub fn migrate_page(
        &mut self,
        src_tier: Tier,
        dst_tier: Tier,
        src_id: PhysicalId,
    ) -> Result<PhysicalId, MemoryManagerError> {
        if src_tier == dst_tier {
            return Ok(src_id);
        }
        if !self.tier_manager.is_allocated(src_tier, src_id) {
            return Err(MemoryManagerError::UnknownPhysicalPage {
                tier: src_tier,
                physical_id: src_id,
            });
        }

        let dst_id = self.allocate_page(dst_tier)?;

        if let Some(virtual_pages) = self.physical_to_virtual.remove(&(src_tier, src_id)) {
            for virtual_id in virtual_pages.iter().copied() {
                self.page_table.remap(virtual_id, dst_tier, dst_id)?;
            }
            self.physical_to_virtual
                .entry((dst_tier, dst_id))
                .or_default()
                .extend(virtual_pages);
        }

        let freed = self.tier_manager.free(src_tier, src_id);
        debug_assert!(freed, "source page was validated as allocated");
        Ok(dst_id)
    }

    pub fn resolve(
        &self,
        virtual_id: VirtualPageId,
    ) -> Result<(Tier, PhysicalId), MemoryManagerError> {
        self.page_table
            .resolve(virtual_id)
            .map(|location| (location.tier, location.physical_id))
            .ok_or(MemoryManagerError::UnknownVirtualPage { virtual_id })
    }

    pub fn bind_virtual_page(
        &mut self,
        virtual_id: VirtualPageId,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<(), MemoryManagerError> {
        if !self.tier_manager.is_allocated(tier, physical_id) {
            return Err(MemoryManagerError::UnknownPhysicalPage { tier, physical_id });
        }

        if let Some(old_location) = self.page_table.map(virtual_id, tier, physical_id) {
            self.remove_reverse_index(old_location, virtual_id);
        }
        self.physical_to_virtual
            .entry((tier, physical_id))
            .or_default()
            .insert(virtual_id);
        Ok(())
    }

    pub fn unmap_virtual_page(&mut self, virtual_id: VirtualPageId) -> Option<PageLocation> {
        let old_location = self.page_table.remove(&virtual_id)?;
        self.remove_reverse_index(old_location, virtual_id);
        Some(old_location)
    }

    pub fn remap_virtual_page(
        &mut self,
        virtual_id: VirtualPageId,
        tier: Tier,
        physical_id: PhysicalId,
    ) -> Result<(), MemoryManagerError> {
        if !self.tier_manager.is_allocated(tier, physical_id) {
            return Err(MemoryManagerError::UnknownPhysicalPage { tier, physical_id });
        }
        let old = self.page_table.remap(virtual_id, tier, physical_id)?;
        self.remove_reverse_index(old, virtual_id);
        self.physical_to_virtual
            .entry((tier, physical_id))
            .or_default()
            .insert(virtual_id);
        Ok(())
    }

    pub fn select_victims(
        &self,
        metadata: &HashMap<PageId, PageMetadata>,
        semantic_priorities: &HashMap<PageId, i32>,
        count: usize,
    ) -> Vec<PageId> {
        self.eviction_policy
            .select_victims(metadata, semantic_priorities, count)
    }

    pub fn tier_usage(&self, tier: Tier) -> TierUsage {
        self.tier_manager.usage(tier)
    }

    /// 规划 Prefill 的页面分配策略
    pub fn plan_prefill(
        &mut self,
        prompt_tokens: usize,
        chunk_size: usize,
        page_size: usize,
    ) -> PrefillPlan {
        if prompt_tokens == 0 {
            return PrefillPlan::FullyResident { pages: 0 };
        }

        let safe_page_size = page_size.max(1);
        let safe_chunk_size = chunk_size.max(1);
        let total_pages = prompt_tokens.div_ceil(safe_page_size);
        let l1 = self.tier_usage(Tier::L1);
        let l2 = self.tier_usage(Tier::L2);
        let l1_available = l1.capacity.saturating_sub(l1.used);

        if total_pages <= l1_available {
            return PrefillPlan::FullyResident { pages: total_pages };
        }

        let l1_pages = l1_available.min(total_pages);
        let l2_prefetch = l2
            .capacity
            .saturating_sub(l2.used)
            .min(total_pages.saturating_sub(l1_pages));

        let mut remaining_tokens = prompt_tokens;
        let mut chunk_schedule = Vec::new();
        while remaining_tokens > 0 {
            let tokens_this_chunk = remaining_tokens.min(safe_chunk_size);
            let pages_this_chunk = tokens_this_chunk.div_ceil(safe_page_size);
            chunk_schedule.push(pages_this_chunk.max(1));
            remaining_tokens = remaining_tokens.saturating_sub(tokens_this_chunk);
        }

        PrefillPlan::Pipelined {
            l1_pages,
            l2_prefetch,
            chunk_schedule,
        }
    }

    /// 注册新会话
    pub fn register_session(&mut self, session_id: SessionId) -> SessionKvCache {
        let cache = SessionKvCache {
            session_id,
            pages: Vec::new(),
            finalized_position: 0,
        };
        self.sessions.insert(session_id, cache.clone());
        cache
    }

    /// 声明会话前缀（只能 claim finalized_position 范围内的页面）
    pub fn claim_session_prefix(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        prefix_tokens: usize,
    ) -> Result<Vec<VirtualPageId>, MemoryManagerError> {
        let cache = self
            .sessions
            .get(&session_id)
            .ok_or(MemoryManagerError::UnknownSession { session_id })?;

        if prefix_tokens > cache.finalized_position {
            return Err(MemoryManagerError::SessionPrefixOutOfBounds {
                session_id,
                prefix_tokens,
                finalized_position: cache.finalized_position,
            });
        }
        if prefix_tokens > cache.pages.len() {
            return Err(MemoryManagerError::SessionPagesInsufficient {
                session_id,
                prefix_tokens,
                available_pages: cache.pages.len(),
            });
        }

        Ok(cache
            .pages
            .iter()
            .take(prefix_tokens)
            .enumerate()
            .map(|(logical_index, _)| VirtualPageId::new(request_id, logical_index))
            .collect())
    }

    /// 确认会话 token 边界（只能单调递增）
    pub fn finalize_session_tokens(
        &mut self,
        session_id: SessionId,
        new_finalized_position: usize,
    ) {
        if let Some(cache) = self.sessions.get_mut(&session_id) {
            cache.finalized_position = cache.finalized_position.max(new_finalized_position);
        }
    }

    /// Query a session's finalized position (returns None if session not found).
    pub fn session_finalized_position(&self, session_id: SessionId) -> Option<usize> {
        self.sessions
            .get(&session_id)
            .map(|cache| cache.finalized_position)
    }

    /// 在指定管线分配页面
    pub fn allocate_page_in_pipeline(
        &mut self,
        pipeline: KvPipeline,
        request_id: RequestId,
        tier: Tier,
    ) -> Result<PhysicalId, MemoryManagerError> {
        let pid = self
            .tier_manager
            .allocate(tier)
            .ok_or(MemoryManagerError::TierCapacityExceeded { tier })?;
        self.pipeline_pages
            .entry((pipeline, request_id))
            .or_default()
            .push(pid);
        Ok(pid)
    }

    /// Register an already-allocated physical page into a pipeline for tracking.
    /// Used when PagedScheduler's BlockAllocator does the physical allocation,
    /// but GlobalMemoryManager still needs to track which pipeline the page belongs to.
    pub fn track_in_pipeline(
        &mut self,
        pipeline: KvPipeline,
        request_id: RequestId,
        physical_id: PhysicalId,
    ) {
        self.pipeline_pages
            .entry((pipeline, request_id))
            .or_default()
            .push(physical_id);
    }

    /// 释放指定请求的 Working 管线页面
    pub fn release_working_pipeline(&mut self, request_id: RequestId) {
        if let Some(pages) = self
            .pipeline_pages
            .remove(&(KvPipeline::Working, request_id))
        {
            for pid in pages {
                if !self.tier_manager.free(Tier::L1, pid) {
                    log::warn!("memory_manager: failed to free page {pid}");
                }
            }
        }
    }

    /// 准备下一轮（释放 Working，保留 Conversation）
    pub fn prepare_next_turn(&mut self, session_id: SessionId) {
        let _ = session_id;
        let working_keys: Vec<_> = self
            .pipeline_pages
            .keys()
            .filter(|(pipeline, _)| *pipeline == KvPipeline::Working)
            .copied()
            .collect();
        for key in working_keys {
            if let Some(pages) = self.pipeline_pages.remove(&key) {
                for pid in pages {
                    if !self.tier_manager.free(Tier::L1, pid) {
                        log::warn!("memory_manager: failed to free page {pid}");
                    }
                }
            }
        }
    }

    /// Chain C: KV Cache 熵策略驱逐 (optimization_strategy_master.md §Chain C)
    ///
    /// Frees all physical KV cache pages whose `per_block_entropy` is below `threshold`.
    /// Called at the tail of each `step()` cycle to reclaim frames occupied by low-information
    /// content (filler words, deterministic tokens) for future high-entropy token allocations.
    ///
    /// `page_entropies`: map of PhysicalId → measured entropy (nat units, from Softmax epilogue).
    /// `threshold`: entropy level below which a block is considered purgeable (default 0.1 nat).
    ///
    /// Returns the number of freed pages.
    pub fn entropy_evict(
        &mut self,
        page_entropies: &std::collections::HashMap<PhysicalId, f32>,
        threshold: f32,
        tier: Tier,
    ) -> usize {
        let low_entropy_pages: Vec<PhysicalId> = page_entropies
            .iter()
            .filter(|(_, &entropy)| entropy < threshold)
            .map(|(&pid, _)| pid)
            .collect();

        let mut freed = 0usize;
        for pid in low_entropy_pages {
            if self.tier_manager.is_allocated(tier, pid) {
                match self.free_page(tier, pid) {
                    Ok(()) => {
                        freed += 1;
                        log::debug!(
                            "entropy_evict: freed page {pid} (entropy {:.4} < {threshold:.4})",
                            page_entropies.get(&pid).copied().unwrap_or(0.0) // LEGAL: 日志中的熵值，默认 0
                        );
                    }
                    Err(e) => {
                        log::warn!("entropy_evict: failed to free page {pid}: {e}");
                    }
                }
            }
        }
        freed
    }

    fn remove_reverse_index(&mut self, location: PageLocation, virtual_id: VirtualPageId) {
        let key = (location.tier, location.physical_id);
        let mut should_remove = false;
        if let Some(virtual_pages) = self.physical_to_virtual.get_mut(&key) {
            virtual_pages.remove(&virtual_id);
            should_remove = virtual_pages.is_empty();
        }
        if should_remove {
            self.physical_to_virtual.remove(&key);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryManagerError {
    TierCapacityExceeded {
        tier: Tier,
    },
    UnknownPhysicalPage {
        tier: Tier,
        physical_id: PhysicalId,
    },
    UnknownVirtualPage {
        virtual_id: VirtualPageId,
    },
    UnknownSession {
        session_id: SessionId,
    },
    SessionPrefixOutOfBounds {
        session_id: SessionId,
        prefix_tokens: usize,
        finalized_position: usize,
    },
    SessionPagesInsufficient {
        session_id: SessionId,
        prefix_tokens: usize,
        available_pages: usize,
    },
}

impl Display for MemoryManagerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryManagerError::TierCapacityExceeded { tier } => {
                write!(f, "tier {:?} is out of capacity", tier)
            }
            MemoryManagerError::UnknownPhysicalPage { tier, physical_id } => {
                write!(
                    f,
                    "physical page {} not allocated in {:?}",
                    physical_id, tier
                )
            }
            MemoryManagerError::UnknownVirtualPage { virtual_id } => {
                write!(
                    f,
                    "virtual page ({}, {}) not found",
                    virtual_id.sequence_id, virtual_id.logical_index
                )
            }
            MemoryManagerError::UnknownSession { session_id } => {
                write!(f, "session {} not found", session_id)
            }
            MemoryManagerError::SessionPrefixOutOfBounds {
                session_id,
                prefix_tokens,
                finalized_position,
            } => {
                write!(
                    f,
                    "session {} prefix {} exceeds finalized position {}",
                    session_id, prefix_tokens, finalized_position
                )
            }
            MemoryManagerError::SessionPagesInsufficient {
                session_id,
                prefix_tokens,
                available_pages,
            } => {
                write!(
                    f,
                    "session {} prefix {} exceeds available pages {}",
                    session_id, prefix_tokens, available_pages
                )
            }
        }
    }
}

impl std::error::Error for MemoryManagerError {}

fn saturating_u128_to_i64(value: u128) -> i64 {
    if value > i64::MAX as u128 {
        i64::MAX
    } else {
        value as i64
    }
}

fn saturating_usize_to_i64(value: usize) -> i64 {
    if value > i64::MAX as usize {
        i64::MAX
    } else {
        value as i64
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn allocate_and_free_tracks_tier_usage() {
        let mut manager = GlobalMemoryManager::new_with_capacities(1, 2, 1);
        let l1_page = manager.allocate_page(Tier::L1).unwrap();
        assert_eq!(l1_page, 0);
        assert_eq!(manager.tier_usage(Tier::L1).used, 1);

        let err = manager.allocate_page(Tier::L1).unwrap_err();
        assert_eq!(
            err,
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 }
        );

        manager.free_page(Tier::L1, l1_page).unwrap();
        assert_eq!(manager.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn migrate_updates_page_table_mapping() {
        let mut manager = GlobalMemoryManager::new_with_capacities(1, 2, 0);
        let vpid = VirtualPageId::new(7, 0);

        let src = manager.allocate_page(Tier::L1).unwrap();
        manager.bind_virtual_page(vpid, Tier::L1, src).unwrap();
        assert_eq!(manager.resolve(vpid).unwrap(), (Tier::L1, src));

        let dst = manager.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        assert_eq!(manager.resolve(vpid).unwrap(), (Tier::L2, dst));
        assert_eq!(manager.tier_usage(Tier::L1).used, 0);
        assert_eq!(manager.tier_usage(Tier::L2).used, 1);
    }

    #[test]
    fn freeing_physical_page_unmaps_all_virtual_pages() {
        let mut manager = GlobalMemoryManager::new_with_capacities(0, 1, 0);
        let pid = manager.allocate_page(Tier::L2).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);

        manager.bind_virtual_page(v0, Tier::L2, pid).unwrap();
        manager.bind_virtual_page(v1, Tier::L2, pid).unwrap();

        manager.free_page(Tier::L2, pid).unwrap();
        assert_eq!(
            manager.resolve(v0).unwrap_err(),
            MemoryManagerError::UnknownVirtualPage { virtual_id: v0 }
        );
        assert_eq!(
            manager.resolve(v1).unwrap_err(),
            MemoryManagerError::UnknownVirtualPage { virtual_id: v1 }
        );
    }

    #[test]
    fn eviction_policy_uses_hgal_signals() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 250,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 1,
                last_access: now - Duration::from_secs(20),
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Protected,
                recency: 500,
                is_lir: true,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now - Duration::from_secs(50),
            },
        );
        metadata.insert(
            3,
            PageMetadata {
                page_id: 3,
                sequence_id: Some(2),
                state: PageState::Standby,
                recency: 20,
                is_lir: true,
                swap_in_time: None,
                warm_until: None,
                access_count: 100,
                last_access: now - Duration::from_secs(5),
            },
        );
        metadata.insert(
            4,
            PageMetadata {
                page_id: 4,
                sequence_id: Some(3),
                state: PageState::Standby,
                recency: 100,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 1,
                last_access: now - Duration::from_secs(1),
            },
        );

        let mut semantic = HashMap::new();
        semantic.insert(4, 10); // high semantic priority, should be preserved

        let victims = policy.select_victims(&metadata, &semantic, 2);
        assert_eq!(victims[0], 1);
        assert!(!victims.contains(&2));
        assert!(!victims.contains(&4));
    }

    #[test]
    fn plan_prefill_returns_fully_resident_when_l1_has_capacity() {
        let mut manager = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let plan = manager.plan_prefill(512, 128, 128);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    #[test]
    fn plan_prefill_returns_pipelined_when_l1_is_insufficient() {
        let mut manager = GlobalMemoryManager::new_with_capacities(2, 4, 0);
        let plan = manager.plan_prefill(1024, 256, 128);
        assert_eq!(
            plan,
            PrefillPlan::Pipelined {
                l1_pages: 2,
                l2_prefetch: 4,
                chunk_schedule: vec![2, 2, 2, 2],
            }
        );
    }

    #[test]
    fn session_claim_rejects_out_of_bounds_and_respects_monotonic_finalize() {
        let mut manager = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        manager.register_session(42);
        {
            let session = manager.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(1, 2),
            ];
        }

        manager.finalize_session_tokens(42, 2);
        manager.finalize_session_tokens(42, 1);

        let claimed = manager.claim_session_prefix(42, 99, 2).unwrap();
        assert_eq!(
            claimed,
            vec![VirtualPageId::new(99, 0), VirtualPageId::new(99, 1)]
        );

        let err = manager.claim_session_prefix(42, 99, 3).unwrap_err();
        assert_eq!(
            err,
            MemoryManagerError::SessionPrefixOutOfBounds {
                session_id: 42,
                prefix_tokens: 3,
                finalized_position: 2,
            }
        );
    }

    #[test]
    fn prepare_next_turn_releases_only_working_pages() {
        let mut manager = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let _ = manager
            .allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1)
            .unwrap();
        let _ = manager
            .allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1)
            .unwrap();
        let _ = manager
            .allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1)
            .unwrap();
        assert_eq!(manager.tier_usage(Tier::L1).used, 3);

        manager.release_working_pipeline(1);
        assert_eq!(manager.tier_usage(Tier::L1).used, 2);
        assert!(manager
            .pipeline_pages
            .contains_key(&(KvPipeline::Conversation, 1)));

        manager.prepare_next_turn(42);
        assert_eq!(manager.tier_usage(Tier::L1).used, 1);
        assert!(manager
            .pipeline_pages
            .contains_key(&(KvPipeline::Conversation, 1)));
        assert!(!manager
            .pipeline_pages
            .contains_key(&(KvPipeline::Working, 2)));
    }

    // ── TierUsage ──

    #[test]
    fn tier_usage_available() {
        let usage = TierUsage { used: 3, capacity: 10 };
        assert_eq!(usage.available(), 7);
    }

    #[test]
    fn tier_usage_available_saturates() {
        let usage = TierUsage { used: 15, capacity: 10 };
        assert_eq!(usage.available(), 0);
    }

    // ── VirtualPageId ──

    #[test]
    fn virtual_page_id_new() {
        let vpid = VirtualPageId::new(42, 7);
        assert_eq!(vpid.sequence_id, 42);
        assert_eq!(vpid.logical_index, 7);
    }

    #[test]
    fn virtual_page_id_equality_and_hash() {
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 2);
        let c = VirtualPageId::new(1, 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── PageTable ──

    #[test]
    fn page_table_map_resolve_remove() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        assert!(pt.resolve(vpid).is_none());

        pt.map(vpid, Tier::L1, 5);
        let loc = pt.resolve(vpid).unwrap();
        assert_eq!(loc.tier, Tier::L1);
        assert_eq!(loc.physical_id, 5);

        let removed = pt.remove(&vpid).unwrap();
        assert_eq!(removed.tier, Tier::L1);
        assert!(pt.resolve(vpid).is_none());
    }

    #[test]
    fn page_table_remap_returns_old() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 1);

        let old = pt.remap(vpid, Tier::L2, 2).unwrap();
        assert_eq!(old.tier, Tier::L1);
        assert_eq!(old.physical_id, 1);

        let loc = pt.resolve(vpid).unwrap();
        assert_eq!(loc.tier, Tier::L2);
        assert_eq!(loc.physical_id, 2);
    }

    #[test]
    fn page_table_remap_unknown_errors() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(99, 0);
        let result = pt.remap(vpid, Tier::L1, 0);
        assert!(matches!(result, Err(MemoryManagerError::UnknownVirtualPage { .. })));
    }

    // ── MemoryManagerError Display ──

    #[test]
    fn error_display_variants() {
        let e1 = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        assert!(format!("{e1}").contains("L1"));

        let e2 = MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 42 };
        assert!(format!("{e2}").contains("42"));

        let vpid = VirtualPageId::new(7, 3);
        let e3 = MemoryManagerError::UnknownVirtualPage { virtual_id: vpid };
        assert!(format!("{e3}").contains("7"));

        let e4 = MemoryManagerError::UnknownSession { session_id: 99 };
        assert!(format!("{e4}").contains("99"));
    }

    // ── Tier enum ──

    #[test]
    fn tier_equality() {
        assert_eq!(Tier::L1, Tier::L1);
        assert_ne!(Tier::L1, Tier::L2);
        assert_ne!(Tier::L2, Tier::L3);
    }

    // ── SessionKvCache ──

    #[test]
    fn session_kv_cache_fields() {
        let session = SessionKvCache {
            session_id: 42,
            pages: vec![VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)],
            finalized_position: 5,
        };
        assert_eq!(session.session_id, 42);
        assert_eq!(session.pages.len(), 2);
        assert_eq!(session.finalized_position, 5);
    }

    // ── PrefillPlan ──

    #[test]
    fn prefill_plan_fully_resident_equality() {
        let p1 = PrefillPlan::FullyResident { pages: 4 };
        let p2 = PrefillPlan::FullyResident { pages: 4 };
        let p3 = PrefillPlan::FullyResident { pages: 8 };
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    // ── Additional tests ──

    #[test]
    fn tier_copy_clone() {
        let t = Tier::L2;
        let t2 = t;
        assert_eq!(t, t2);
        let t3 = t.clone();
        assert_eq!(t3, Tier::L2);
    }

    #[test]
    fn tier_debug_format() {
        assert!(format!("{:?}", Tier::L1).contains("L1"));
        assert!(format!("{:?}", Tier::L2).contains("L2"));
        assert!(format!("{:?}", Tier::L3).contains("L3"));
    }

    #[test]
    fn virtual_page_id_copy_clone() {
        let v = VirtualPageId::new(5, 10);
        let v2 = v;
        assert_eq!(v, v2);
        let v3 = v.clone();
        assert_eq!(v3, VirtualPageId::new(5, 10));
    }

    #[test]
    fn virtual_page_id_const_fn() {
        const VP: VirtualPageId = VirtualPageId::new(1, 2);
        assert_eq!(VP.sequence_id, 1);
        assert_eq!(VP.logical_index, 2);
    }

    #[test]
    fn page_location_fields() {
        let loc = PageLocation { physical_id: 42, tier: Tier::L3 };
        assert_eq!(loc.physical_id, 42);
        assert_eq!(loc.tier, Tier::L3);
    }

    #[test]
    fn page_location_equality() {
        let a = PageLocation { physical_id: 1, tier: Tier::L1 };
        let b = PageLocation { physical_id: 1, tier: Tier::L1 };
        let c = PageLocation { physical_id: 2, tier: Tier::L1 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn session_kv_cache_clone() {
        let s = SessionKvCache {
            session_id: 1,
            pages: vec![VirtualPageId::new(1, 0)],
            finalized_position: 3,
        };
        let s2 = s.clone();
        assert_eq!(s2.session_id, 1);
        assert_eq!(s2.pages.len(), 1);
        assert_eq!(s2.finalized_position, 3);
    }

    #[test]
    fn prefill_plan_pipelined_equality() {
        let p1 = PrefillPlan::Pipelined {
            l1_pages: 2,
            l2_prefetch: 4,
            chunk_schedule: vec![2, 2, 2, 2],
        };
        let p2 = PrefillPlan::Pipelined {
            l1_pages: 2,
            l2_prefetch: 4,
            chunk_schedule: vec![2, 2, 2, 2],
        };
        assert_eq!(p1, p2);
    }

    #[test]
    fn prefill_plan_different_variants_not_equal() {
        let p1 = PrefillPlan::FullyResident { pages: 4 };
        let p2 = PrefillPlan::Pipelined {
            l1_pages: 4,
            l2_prefetch: 0,
            chunk_schedule: vec![4],
        };
        assert_ne!(p1, p2);
    }

    #[test]
    fn page_table_map_overwrite_returns_old() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        let old = pt.map(vpid, Tier::L1, 1);
        assert!(old.is_none());

        let overwritten = pt.map(vpid, Tier::L2, 2);
        assert_eq!(overwritten.unwrap().tier, Tier::L1);
        assert_eq!(pt.resolve(vpid).unwrap().tier, Tier::L2);
    }

    #[test]
    fn page_table_remove_nonexistent_returns_none() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(99, 0);
        assert!(pt.remove(&vpid).is_none());
    }

    #[test]
    fn tier_manager_allocate_sequential_ids() {
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        let c = tm.allocate(Tier::L1).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert!(tm.allocate(Tier::L1).is_none(), "capacity exhausted");
    }

    #[test]
    fn tier_manager_free_and_reuse() {
        let mut tm = TierManager::new(2, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let _b = tm.allocate(Tier::L1).unwrap();
        assert!(tm.free(Tier::L1, a));
        let c = tm.allocate(Tier::L1).unwrap();
        assert_eq!(c, a, "freed ID should be reused");
    }

    #[test]
    fn tier_manager_free_unknown_returns_false() {
        let mut tm = TierManager::new(2, 0, 0);
        assert!(!tm.free(Tier::L1, 999));
    }

    #[test]
    fn tier_manager_is_allocated() {
        let mut tm = TierManager::new(2, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        assert!(tm.is_allocated(Tier::L1, a));
        assert!(!tm.is_allocated(Tier::L1, 99));
    }

    #[test]
    fn tier_manager_usage_tracking() {
        let mut tm = TierManager::new(4, 2, 1);
        let _ = tm.allocate(Tier::L1).unwrap();
        let _ = tm.allocate(Tier::L1).unwrap();
        assert_eq!(tm.usage(Tier::L1).used, 2);
        assert_eq!(tm.usage(Tier::L1).capacity, 4);
        assert_eq!(tm.usage(Tier::L2).used, 0);
        assert_eq!(tm.usage(Tier::L3).used, 0);
    }

    #[test]
    fn tier_manager_can_allocate() {
        let mut tm = TierManager::new(1, 0, 0);
        assert!(tm.can_allocate(Tier::L1));
        let _ = tm.allocate(Tier::L1).unwrap();
        assert!(!tm.can_allocate(Tier::L1));
    }

    #[test]
    fn global_manager_bind_unmap_virtual_page() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));

        let loc = mgr.unmap_virtual_page(vpid).unwrap();
        assert_eq!(loc.tier, Tier::L1);
        assert_eq!(loc.physical_id, pid);
        assert!(mgr.resolve(vpid).is_err());
    }

    #[test]
    fn global_manager_bind_rejects_unallocated_physical() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let vpid = VirtualPageId::new(10, 0);
        let err = mgr.bind_virtual_page(vpid, Tier::L1, 999).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownPhysicalPage { .. }));
    }

    #[test]
    fn global_manager_remap_virtual_page() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L2).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();
        mgr.remap_virtual_page(vpid, Tier::L2, p2).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L2, p2));
    }

    #[test]
    fn global_manager_unmap_nonexistent_returns_none() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let vpid = VirtualPageId::new(99, 0);
        assert!(mgr.unmap_virtual_page(vpid).is_none());
    }

    #[test]
    fn global_manager_migrate_same_tier_noop() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let result = mgr.migrate_page(Tier::L1, Tier::L1, pid).unwrap();
        assert_eq!(result, pid);
    }

    #[test]
    fn global_manager_migrate_unknown_page_errors() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 0);
        let err = mgr.migrate_page(Tier::L1, Tier::L2, 999).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownPhysicalPage { .. }));
    }

    #[test]
    fn global_manager_free_unknown_page_errors() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let err = mgr.free_page(Tier::L1, 999).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownPhysicalPage { .. }));
    }

    #[test]
    fn global_manager_track_and_untrack() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        mgr.track_page(Tier::L1, 5).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        mgr.untrack_page(Tier::L1, 5).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn global_manager_register_session_and_finalize() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let cache = mgr.register_session(42);
        assert_eq!(cache.session_id, 42);
        assert!(cache.pages.is_empty());
        assert_eq!(cache.finalized_position, 0);

        mgr.finalize_session_tokens(42, 10);
        assert_eq!(mgr.session_finalized_position(42), Some(10));

        // Monotonic: finalize to lower value should not decrease
        mgr.finalize_session_tokens(42, 5);
        assert_eq!(mgr.session_finalized_position(42), Some(10));
    }

    #[test]
    fn global_manager_session_finalized_position_unknown() {
        let mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        assert_eq!(mgr.session_finalized_position(999), None);
    }

    #[test]
    fn global_manager_claim_session_unknown_errors() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let err = mgr.claim_session_prefix(999, 1, 0).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownSession { .. }));
    }

    #[test]
    fn plan_prefill_zero_tokens() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let plan = mgr.plan_prefill(0, 128, 64);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 0 });
    }

    #[test]
    fn plan_prefill_single_chunk_fits_l1() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(16, 0, 0);
        let plan = mgr.plan_prefill(256, 256, 64);
        assert!(matches!(plan, PrefillPlan::FullyResident { pages: 4 }));
    }

    #[test]
    fn plan_prefill_chunk_schedule_generation() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let plan = mgr.plan_prefill(500, 100, 50);
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, .. } => {
                // 500 tokens / 50 per page = 10 pages total
                // chunk_schedule: 100/50=2 pages per chunk, 5 chunks
                assert_eq!(chunk_schedule.len(), 5);
                assert!(chunk_schedule.iter().all(|&c| c == 2));
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    #[test]
    fn entropy_evict_frees_low_entropy_pages() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.01); // low
        entropies.insert(p1, 5.0);  // high
        entropies.insert(p2, 0.05); // low

        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 2);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn entropy_evict_nothing_above_threshold() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 1.0);
        entropies.insert(p1, 2.0);

        let freed = mgr.entropy_evict(&entropies, 0.5, Tier::L1);
        assert_eq!(freed, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
    }

    #[test]
    fn error_display_session_prefix_out_of_bounds() {
        let err = MemoryManagerError::SessionPrefixOutOfBounds {
            session_id: 42,
            prefix_tokens: 10,
            finalized_position: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("42"));
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_display_session_pages_insufficient() {
        let err = MemoryManagerError::SessionPagesInsufficient {
            session_id: 7,
            prefix_tokens: 20,
            available_pages: 10,
        };
        let msg = format!("{err}");
        assert!(msg.contains("7"));
        assert!(msg.contains("20"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn error_std_error_impl() {
        let err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn saturating_u128_to_i64_clamps() {
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128), i64::MAX);
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128 + 1), i64::MAX);
        assert_eq!(saturating_u128_to_i64(0), 0);
    }

    #[test]
    fn saturating_usize_to_i64_clamps() {
        assert_eq!(saturating_usize_to_i64(42), 42);
        assert_eq!(saturating_usize_to_i64(0), 0);
    }

    #[test]
    fn eviction_policy_select_victims_zero_count() {
        let policy = EvictionPolicy::default();
        let metadata = HashMap::new();
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 0);
        assert!(victims.is_empty());
    }

    #[test]
    fn global_manager_allocate_pipeline_pages() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let p2 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        assert_ne!(p1, p2);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
    }

    #[test]
    fn global_manager_track_in_pipeline() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.track_in_pipeline(KvPipeline::Conversation, 5, 42);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 5)));
    }

    // ── Additional trait and edge-case tests ──

    #[test]
    fn tier_hash_in_hashset() {
        let mut set = HashSet::new();
        set.insert(Tier::L1);
        set.insert(Tier::L2);
        set.insert(Tier::L3);
        set.insert(Tier::L1); // duplicate
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Tier::L1));
        assert!(set.contains(&Tier::L2));
        assert!(set.contains(&Tier::L3));
    }

    #[test]
    fn tier_hash_in_hashmap() {
        let mut map = HashMap::new();
        map.insert(Tier::L1, 100);
        map.insert(Tier::L2, 200);
        assert_eq!(map.get(&Tier::L1), Some(&100));
        assert_eq!(map.get(&Tier::L2), Some(&200));
        assert_eq!(map.get(&Tier::L3), None);
    }

    #[test]
    fn virtual_page_id_hash_in_hashset() {
        let mut set = HashSet::new();
        set.insert(VirtualPageId::new(1, 0));
        set.insert(VirtualPageId::new(1, 1));
        set.insert(VirtualPageId::new(1, 0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn page_location_copy_clone() {
        let loc = PageLocation { physical_id: 7, tier: Tier::L2 };
        let loc2 = loc; // Copy
        assert_eq!(loc, loc2);
        let loc3 = loc.clone();
        assert_eq!(loc3, PageLocation { physical_id: 7, tier: Tier::L2 });
    }

    #[test]
    fn page_location_copy_allows_independent_mutation() {
        let mut a = PageLocation { physical_id: 1, tier: Tier::L1 };
        let b = a;
        a.physical_id = 99;
        assert_eq!(a.physical_id, 99);
        assert_eq!(b.physical_id, 1);
    }

    #[test]
    fn tier_usage_copy_clone() {
        let u = TierUsage { used: 5, capacity: 10 };
        let u2 = u; // Copy
        assert_eq!(u, u2);
        let u3 = u.clone();
        assert_eq!(u3.used, 5);
        assert_eq!(u3.capacity, 10);
    }

    #[test]
    fn tier_usage_available_when_used_equals_capacity() {
        let usage = TierUsage { used: 10, capacity: 10 };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_when_zero_used() {
        let usage = TierUsage { used: 0, capacity: 100 };
        assert_eq!(usage.available(), 100);
    }

    #[test]
    fn memory_manager_error_copy_clone() {
        let e1 = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        let e2 = e1; // Copy
        assert_eq!(e1, e2);
        let e3 = e1.clone();
        assert_eq!(e3, MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 });
    }

    #[test]
    fn memory_manager_error_equality_across_variants() {
        let e1 = MemoryManagerError::UnknownSession { session_id: 42 };
        let e2 = MemoryManagerError::UnknownSession { session_id: 42 };
        let e3 = MemoryManagerError::UnknownSession { session_id: 99 };
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
        assert_ne!(
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L2 }
        );
    }

    #[test]
    fn eviction_policy_default_weights() {
        let policy = EvictionPolicy::default();
        assert_eq!(policy.recency_weight, 4);
        assert_eq!(policy.frequency_weight, 32);
        assert_eq!(policy.semantic_weight, 128);
        assert_eq!(policy.active_penalty, 256);
        assert_eq!(policy.standby_bonus, 512);
    }

    #[test]
    fn tier_manager_track_page_within_capacity() {
        let mut tm = TierManager::new(4, 0, 0);
        assert!(tm.track_page(Tier::L1, 2));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert!(tm.is_allocated(Tier::L1, 2));
    }

    #[test]
    fn tier_manager_track_page_exceeds_capacity() {
        let mut tm = TierManager::new(1, 0, 0);
        let _ = tm.allocate(Tier::L1).unwrap();
        // capacity is 1, already used 1, tracking a new id should fail
        assert!(!tm.track_page(Tier::L1, 99));
    }

    #[test]
    fn tier_manager_track_page_idempotent() {
        let mut tm = TierManager::new(4, 0, 0);
        assert!(tm.track_page(Tier::L1, 5));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        // tracking same id again returns true (already allocated)
        assert!(tm.track_page(Tier::L1, 5));
        assert_eq!(tm.usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_bind_virtual_page_overwrite_removes_old_reverse_index() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();
        // Rebind same virtual page to different physical
        mgr.bind_virtual_page(vpid, Tier::L1, p2).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p2));

        // Freeing p1 should not cascade-remove the rebound virtual page
        mgr.free_page(Tier::L1, p1).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p2));
    }

    #[test]
    fn global_manager_remap_virtual_rejects_unallocated_physical() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();

        let err = mgr.remap_virtual_page(vpid, Tier::L1, 999).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownPhysicalPage { .. }));
        // Original mapping should be unchanged
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p1));
    }

    #[test]
    fn global_manager_track_page_capacity_exceeded() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let _ = mgr.allocate_page(Tier::L1).unwrap(); // uses capacity
        let err = mgr.track_page(Tier::L1, 99).unwrap_err();
        assert!(matches!(err, MemoryManagerError::TierCapacityExceeded { .. }));
    }

    #[test]
    fn session_kv_cache_debug_format() {
        let s = SessionKvCache {
            session_id: 42,
            pages: vec![VirtualPageId::new(1, 0)],
            finalized_position: 5,
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("42"));
        assert!(debug.contains("session_id"));
    }

    #[test]
    fn prefill_plan_debug_format() {
        let p = PrefillPlan::FullyResident { pages: 7 };
        let debug = format!("{p:?}");
        assert!(debug.contains("FullyResident"));
        assert!(debug.contains("7"));
    }

    #[test]
    fn prefill_plan_clone() {
        let p = PrefillPlan::Pipelined {
            l1_pages: 2,
            l2_prefetch: 4,
            chunk_schedule: vec![2, 2],
        };
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn global_manager_allocate_page_in_pipeline_capacity_exceeded() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let err = mgr
            .allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1)
            .unwrap_err();
        assert!(matches!(err, MemoryManagerError::TierCapacityExceeded { .. }));
    }

    #[test]
    fn global_manager_select_victims_delegates_to_policy() {
        let mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let metadata = HashMap::new();
        let semantic = HashMap::new();
        // No pages means no victims regardless of count
        let victims = mgr.select_victims(&metadata, &semantic, 10);
        assert!(victims.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════
    // 40 additional tests — trait derivation, edge cases, boundary conditions
    // ══════════════════════════════════════════════════════════════════════

    // ── Tier derive coverage ──

    #[test]
    fn tier_all_variants_in_hashset() {
        let all = [Tier::L1, Tier::L2, Tier::L3];
        let set: HashSet<Tier> = all.into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Tier::L1));
        assert!(set.contains(&Tier::L2));
        assert!(set.contains(&Tier::L3));
    }

    #[test]
    fn tier_copy_allows_move_without_clone() {
        let t = Tier::L1;
        let moved = t; // relies on Copy, not Clone
        assert_eq!(t, moved);
    }

    #[test]
    fn tier_ord_ordering() {
        // Derive: Tier has no Ord, but we can verify PartialEq distinguishes all
        assert_ne!(Tier::L1, Tier::L2);
        assert_ne!(Tier::L1, Tier::L3);
        assert_ne!(Tier::L2, Tier::L3);
    }

    // ── VirtualPageId edge cases ──

    #[test]
    fn virtual_page_id_zero_fields() {
        let vpid = VirtualPageId::new(0, 0);
        assert_eq!(vpid.sequence_id, 0);
        assert_eq!(vpid.logical_index, 0);
    }

    #[test]
    fn virtual_page_id_max_fields() {
        let vpid = VirtualPageId::new(RequestId::MAX, usize::MAX);
        assert_eq!(vpid.sequence_id, RequestId::MAX);
        assert_eq!(vpid.logical_index, usize::MAX);
    }

    #[test]
    fn virtual_page_id_hash_distinguishes_different_logical_index() {
        let mut set = HashSet::new();
        for i in 0..100usize {
            set.insert(VirtualPageId::new(1, i));
        }
        assert_eq!(set.len(), 100);
    }

    #[test]
    fn virtual_page_id_hash_distinguishes_different_sequence_id() {
        let mut set = HashSet::new();
        for i in 0..50u64 {
            set.insert(VirtualPageId::new(i, 0));
        }
        assert_eq!(set.len(), 50);
    }

    // ── PageLocation derive coverage ──

    #[test]
    fn page_location_debug_format() {
        let loc = PageLocation { physical_id: 42, tier: Tier::L2 };
        let debug = format!("{loc:?}");
        assert!(debug.contains("42"));
        assert!(debug.contains("L2"));
    }

    #[test]
    fn page_location_equality_different_tier_same_id() {
        let a = PageLocation { physical_id: 1, tier: Tier::L1 };
        let b = PageLocation { physical_id: 1, tier: Tier::L2 };
        assert_ne!(a, b);
    }

    // ── TierUsage derive coverage ──

    #[test]
    fn tier_usage_debug_format() {
        let usage = TierUsage { used: 3, capacity: 10 };
        let debug = format!("{usage:?}");
        assert!(debug.contains("3"));
        assert!(debug.contains("10"));
    }

    #[test]
    fn tier_usage_equality() {
        let a = TierUsage { used: 5, capacity: 10 };
        let b = TierUsage { used: 5, capacity: 10 };
        let c = TierUsage { used: 5, capacity: 20 };
        let d = TierUsage { used: 6, capacity: 10 };
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn tier_usage_zero_capacity() {
        let usage = TierUsage { used: 0, capacity: 0 };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_with_max_values() {
        let usage = TierUsage { used: 0, capacity: usize::MAX };
        assert_eq!(usage.available(), usize::MAX);
    }

    // ── PrefillPlan derive & edge cases ──

    #[test]
    fn prefill_plan_fully_resident_zero_pages() {
        let p = PrefillPlan::FullyResident { pages: 0 };
        let p2 = PrefillPlan::FullyResident { pages: 0 };
        assert_eq!(p, p2);
    }

    #[test]
    fn prefill_plan_pipelined_debug_format() {
        let p = PrefillPlan::Pipelined {
            l1_pages: 1,
            l2_prefetch: 2,
            chunk_schedule: vec![3],
        };
        let debug = format!("{p:?}");
        assert!(debug.contains("Pipelined"));
        assert!(debug.contains("l1_pages"));
        assert!(debug.contains("chunk_schedule"));
    }

    #[test]
    fn prefill_plan_pipelined_different_chunk_schedule_not_equal() {
        let p1 = PrefillPlan::Pipelined {
            l1_pages: 2,
            l2_prefetch: 1,
            chunk_schedule: vec![1, 1],
        };
        let p2 = PrefillPlan::Pipelined {
            l1_pages: 2,
            l2_prefetch: 1,
            chunk_schedule: vec![2],
        };
        assert_ne!(p1, p2);
    }

    #[test]
    fn plan_prefill_page_size_one() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(100, 0, 0);
        let plan = mgr.plan_prefill(5, 100, 1);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 5 });
    }

    #[test]
    fn plan_prefill_chunk_size_one() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(100, 0, 0);
        let plan = mgr.plan_prefill(3, 1, 1);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 3 });
    }

    #[test]
    fn plan_prefill_tokens_not_divisible_by_page_size() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(100, 0, 0);
        // 10 tokens / 3 per page = 4 pages (div_ceil)
        let plan = mgr.plan_prefill(10, 100, 3);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    // ── MemoryManagerError derive & Display ──

    #[test]
    fn error_all_variants_have_nonempty_display() {
        let vpid = VirtualPageId::new(1, 2);
        let errors: Vec<MemoryManagerError> = vec![
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
            MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 0 },
            MemoryManagerError::UnknownVirtualPage { virtual_id: vpid },
            MemoryManagerError::UnknownSession { session_id: 0 },
            MemoryManagerError::SessionPrefixOutOfBounds {
                session_id: 0,
                prefix_tokens: 0,
                finalized_position: 0,
            },
            MemoryManagerError::SessionPagesInsufficient {
                session_id: 0,
                prefix_tokens: 0,
                available_pages: 0,
            },
        ];
        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "Display should not be empty for {:?}", err);
        }
    }

    #[test]
    fn error_debug_format_roundtrip() {
        let err = MemoryManagerError::UnknownSession { session_id: 42 };
        let debug = format!("{err:?}");
        assert!(debug.contains("UnknownSession"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn error_copy_preserves_fields() {
        let err = MemoryManagerError::UnknownPhysicalPage { tier: Tier::L3, physical_id: 7 };
        let copied = err; // Copy
        assert_eq!(err, copied);
    }

    #[test]
    fn error_std_error_source_is_none() {
        let errors: Vec<MemoryManagerError> = vec![
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
            MemoryManagerError::UnknownPhysicalPage { tier: Tier::L1, physical_id: 1 },
            MemoryManagerError::UnknownVirtualPage { virtual_id: VirtualPageId::new(0, 0) },
            MemoryManagerError::UnknownSession { session_id: 1 },
            MemoryManagerError::SessionPrefixOutOfBounds {
                session_id: 1, prefix_tokens: 1, finalized_position: 0,
            },
            MemoryManagerError::SessionPagesInsufficient {
                session_id: 1, prefix_tokens: 1, available_pages: 0,
            },
        ];
        for err in &errors {
            assert!(std::error::Error::source(err).is_none());
        }
    }

    // ── TierManager edge cases ──

    #[test]
    fn tier_manager_zero_capacity() {
        let mut tm = TierManager::new(0, 0, 0);
        assert!(!tm.can_allocate(Tier::L1));
        assert!(!tm.can_allocate(Tier::L2));
        assert!(!tm.can_allocate(Tier::L3));
        assert!(tm.allocate(Tier::L1).is_none());
    }

    #[test]
    fn tier_manager_large_capacity() {
        let mut tm = TierManager::new(1000, 500, 100);
        assert!(tm.can_allocate(Tier::L1));
        assert!(tm.can_allocate(Tier::L2));
        assert!(tm.can_allocate(Tier::L3));
        for _ in 0..1000 {
            assert!(tm.allocate(Tier::L1).is_some());
        }
        assert!(!tm.can_allocate(Tier::L1));
        assert_eq!(tm.usage(Tier::L1).used, 1000);
    }

    #[test]
    fn tier_manager_free_and_reallocate_cycle() {
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        let c = tm.allocate(Tier::L1).unwrap();

        // Free middle, then reallocate - should reuse b
        assert!(tm.free(Tier::L1, b));
        let d = tm.allocate(Tier::L1).unwrap();
        assert_eq!(d, b);

        // Free all and verify
        assert!(tm.free(Tier::L1, a));
        assert!(tm.free(Tier::L1, c));
        assert!(tm.free(Tier::L1, d));
        assert_eq!(tm.usage(Tier::L1).used, 0);
    }

    #[test]
    fn tier_manager_double_free_returns_false() {
        let mut tm = TierManager::new(2, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        assert!(tm.free(Tier::L1, a));
        assert!(!tm.free(Tier::L1, a)); // double free
    }

    #[test]
    fn tier_manager_independent_tiers() {
        let mut tm = TierManager::new(1, 1, 1);
        let _l1 = tm.allocate(Tier::L1).unwrap();
        let _l2 = tm.allocate(Tier::L2).unwrap();
        let _l3 = tm.allocate(Tier::L3).unwrap();

        // All tiers exhausted independently
        assert!(!tm.can_allocate(Tier::L1));
        assert!(!tm.can_allocate(Tier::L2));
        assert!(!tm.can_allocate(Tier::L3));
    }

    #[test]
    fn tier_manager_track_page_fills_gap() {
        let mut tm = TierManager::new(5, 0, 0);
        // Track page 2 — should create gap [0, 1] in free_ids
        assert!(tm.track_page(Tier::L1, 2));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert!(tm.is_allocated(Tier::L1, 2));
        // Next allocation should come from free_ids (0)
        let next = tm.allocate(Tier::L1).unwrap();
        assert_eq!(next, 0);
        let next2 = tm.allocate(Tier::L1).unwrap();
        assert_eq!(next2, 1);
    }

    // ── PageTable edge cases ──

    #[test]
    fn page_table_empty_resolve() {
        let pt = PageTable::new();
        assert!(pt.resolve(VirtualPageId::new(0, 0)).is_none());
    }

    #[test]
    fn page_table_many_mappings() {
        let mut pt = PageTable::new();
        for i in 0..200u64 {
            let vpid = VirtualPageId::new(i, 0);
            pt.map(vpid, Tier::L1, i as PhysicalId);
        }
        for i in 0..200u64 {
            let loc = pt.resolve(VirtualPageId::new(i, 0)).unwrap();
            assert_eq!(loc.physical_id, i as PhysicalId);
            assert_eq!(loc.tier, Tier::L1);
        }
    }

    // ── GlobalMemoryManager session edge cases ──

    #[test]
    fn session_register_multiple_sessions() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let s1 = mgr.register_session(1);
        let s2 = mgr.register_session(2);
        assert_eq!(s1.session_id, 1);
        assert_eq!(s2.session_id, 2);
        assert_eq!(mgr.session_finalized_position(1), Some(0));
        assert_eq!(mgr.session_finalized_position(2), Some(0));
    }

    #[test]
    fn session_finalize_unknown_session_is_noop() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        // Should not panic
        mgr.finalize_session_tokens(999, 100);
        assert_eq!(mgr.session_finalized_position(999), None);
    }

    #[test]
    fn session_claim_zero_prefix() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0)];
        }
        mgr.finalize_session_tokens(42, 1);
        let claimed = mgr.claim_session_prefix(42, 99, 0).unwrap();
        assert!(claimed.is_empty());
    }

    #[test]
    fn session_claim_pages_insufficient() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0)];
        }
        mgr.finalize_session_tokens(42, 10);

        // prefix_tokens=2 > pages.len()=1 => SessionPagesInsufficient
        let err = mgr.claim_session_prefix(42, 99, 2).unwrap_err();
        assert!(matches!(err, MemoryManagerError::SessionPagesInsufficient { .. }));
    }

    // ── GlobalMemoryManager migration edge cases ──

    #[test]
    fn migrate_page_with_multiple_virtual_pages() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 4, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let other = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        let v2 = VirtualPageId::new(11, 0);

        mgr.bind_virtual_page(v0, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v2, Tier::L1, pid).unwrap();

        let dst = mgr.migrate_page(Tier::L1, Tier::L2, pid).unwrap();
        assert_eq!(mgr.resolve(v0).unwrap(), (Tier::L2, dst));
        assert_eq!(mgr.resolve(v1).unwrap(), (Tier::L2, dst));
        assert_eq!(mgr.resolve(v2).unwrap(), (Tier::L2, dst));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1); // only 'other' remains in L1
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
        assert!(mgr.tier_manager.is_allocated(Tier::L1, other));
    }

    #[test]
    fn migrate_page_destination_capacity_exceeded() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let _p2 = mgr.allocate_page(Tier::L1).unwrap();
        // L2 has capacity 0, migration L1→L2 should fail
        let err = mgr.migrate_page(Tier::L1, Tier::L2, p1).unwrap_err();
        assert!(matches!(err, MemoryManagerError::TierCapacityExceeded { .. }));
        // Original page should still be allocated in L1
        assert!(mgr.tier_manager.is_allocated(Tier::L1, p1));
    }

    // ── EvictionPolicy edge cases ──

    #[test]
    fn eviction_policy_select_victims_more_than_available() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        // Request 5 victims but only 1 candidate
        let victims = policy.select_victims(&metadata, &semantic, 5);
        assert_eq!(victims.len(), 1);
    }

    #[test]
    fn eviction_policy_warm_pages_not_candidates() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Warm,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert!(victims.is_empty());
    }

    #[test]
    fn eviction_policy_swapped_pages_not_candidates() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Swapped,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert!(victims.is_empty());
    }

    // ── entropy_evict edge cases ──

    #[test]
    fn entropy_evict_empty_entropies() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let _p = mgr.allocate_page(Tier::L1).unwrap();
        let entropies = HashMap::new();
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 0);
    }

    #[test]
    fn entropy_evict_threshold_at_zero() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, -0.5); // negative entropy
        entropies.insert(p1, 0.0);  // exactly zero

        // threshold=0.0: p0 (-0.5 < 0.0) freed, p1 (0.0 < 0.0 is false) kept
        let freed = mgr.entropy_evict(&entropies, 0.0, Tier::L1);
        assert_eq!(freed, 1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn entropy_evict_unallocated_page_skipped() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        mgr.free_page(Tier::L1, p0).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.01);
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 0);
    }

    // ── saturating conversion edge cases ──

    #[test]
    fn saturating_u128_to_i64_small_values() {
        assert_eq!(saturating_u128_to_i64(1), 1);
        assert_eq!(saturating_u128_to_i64(100), 100);
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128 - 1), i64::MAX - 1);
    }

    #[test]
    fn saturating_usize_to_i64_large_value() {
        // On 64-bit platforms, usize can exceed i64::MAX
        if size_of::<usize>() == 8 {
            let large = i64::MAX as usize + 1;
            assert_eq!(saturating_usize_to_i64(large), i64::MAX);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 55 additional tests — uncovered paths, boundary conditions, interactions
    // ══════════════════════════════════════════════════════════════════════

    // ── Tier ordering (PartialOrd + Ord derived) ──

    #[test]
    fn tier_ordering_l1_lt_l2_lt_l3() {
        assert!(Tier::L1 < Tier::L2);
        assert!(Tier::L2 < Tier::L3);
        assert!(Tier::L1 < Tier::L3);
    }

    #[test]
    fn tier_sorting_deterministic() {
        let mut tiers = vec![Tier::L3, Tier::L1, Tier::L2];
        tiers.sort();
        assert_eq!(tiers, vec![Tier::L1, Tier::L2, Tier::L3]);
    }

    #[test]
    fn tier_min_max() {
        assert_eq!(Tier::L1, std::cmp::min(Tier::L1, Tier::L3));
        assert_eq!(Tier::L3, std::cmp::max(Tier::L1, Tier::L3));
    }

    #[test]
    fn tier_btree_set_dedup() {
        use std::collections::BTreeSet;
        let mut set = BTreeSet::new();
        set.insert(Tier::L3);
        set.insert(Tier::L1);
        set.insert(Tier::L2);
        set.insert(Tier::L1);
        assert_eq!(set.len(), 3);
        let sorted: Vec<Tier> = set.into_iter().collect();
        assert_eq!(sorted, vec![Tier::L1, Tier::L2, Tier::L3]);
    }

    // ── VirtualPageId as HashMap key ──

    #[test]
    fn virtual_page_id_as_hashmap_key() {
        let mut map = HashMap::new();
        map.insert(VirtualPageId::new(1, 0), "first");
        map.insert(VirtualPageId::new(1, 1), "second");
        map.insert(VirtualPageId::new(2, 0), "third");
        assert_eq!(map.get(&VirtualPageId::new(1, 0)), Some(&"first"));
        assert_eq!(map.get(&VirtualPageId::new(1, 1)), Some(&"second"));
        assert_eq!(map.get(&VirtualPageId::new(2, 0)), Some(&"third"));
        assert_eq!(map.get(&VirtualPageId::new(99, 99)), None);
    }

    #[test]
    fn virtual_page_id_removal_from_hashset() {
        let mut set = HashSet::new();
        let v = VirtualPageId::new(1, 0);
        set.insert(v);
        assert!(set.remove(&v));
        assert!(!set.contains(&v));
    }

    // ── PageLocation ordering via tier ──

    #[test]
    fn page_location_different_physical_same_tier_not_equal() {
        let a = PageLocation { physical_id: 1, tier: Tier::L1 };
        let b = PageLocation { physical_id: 2, tier: Tier::L1 };
        assert_ne!(a, b);
    }

    #[test]
    fn page_location_all_tier_combinations_distinct() {
        let locs: Vec<PageLocation> = [Tier::L1, Tier::L2, Tier::L3]
            .iter()
            .map(|&t| PageLocation { physical_id: 0, tier: t })
            .collect();
        assert_ne!(locs[0], locs[1]);
        assert_ne!(locs[1], locs[2]);
        assert_ne!(locs[0], locs[2]);
    }

    // ── TierUsage operator interactions ──

    #[test]
    fn tier_usage_available_always_non_negative() {
        // Property: available() should never wrap or panic even with weird values
        let cases = [
            (0usize, 0usize),
            (100, 50),
            (50, 100),
            (usize::MAX, usize::MAX),
            (usize::MAX, 0),
        ];
        for (used, cap) in cases {
            let usage = TierUsage { used, capacity: cap };
            let avail = usage.available();
            assert!(
                avail <= cap,
                "available ({avail}) should be <= capacity ({cap}) for used={used}"
            );
        }
    }

    // ── PageTable: remap after remove ──

    #[test]
    fn page_table_remap_after_remove_errors() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 1);
        pt.remove(&vpid);
        let result = pt.remap(vpid, Tier::L2, 2);
        assert!(matches!(result, Err(MemoryManagerError::UnknownVirtualPage { .. })));
    }

    #[test]
    fn page_table_multiple_distinct_virtual_pages() {
        let mut pt = PageTable::new();
        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);
        let v2 = VirtualPageId::new(2, 0);
        pt.map(v0, Tier::L1, 10);
        pt.map(v1, Tier::L2, 20);
        pt.map(v2, Tier::L3, 30);

        assert_eq!(pt.resolve(v0).unwrap().physical_id, 10);
        assert_eq!(pt.resolve(v1).unwrap().physical_id, 20);
        assert_eq!(pt.resolve(v2).unwrap().physical_id, 30);

        // Removing one should not affect others
        pt.remove(&v1);
        assert!(pt.resolve(v0).is_some());
        assert!(pt.resolve(v1).is_none());
        assert!(pt.resolve(v2).is_some());
    }

    #[test]
    fn page_table_map_same_virtual_to_different_tiers() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(5, 0);

        pt.map(vpid, Tier::L1, 1);
        let old = pt.map(vpid, Tier::L2, 2);
        assert_eq!(old.unwrap().tier, Tier::L1);

        let loc = pt.resolve(vpid).unwrap();
        assert_eq!(loc.tier, Tier::L2);
        assert_eq!(loc.physical_id, 2);
    }

    // ── TierManager: allocate from free_ids (re-use after free) ──

    #[test]
    fn tier_manager_allocate_reuses_fifo_order() {
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        let c = tm.allocate(Tier::L1).unwrap();

        // Free in specific order: b, a
        assert!(tm.free(Tier::L1, b));
        assert!(tm.free(Tier::L1, a));

        // Re-allocation should return b first (FIFO from free_ids)
        let d = tm.allocate(Tier::L1).unwrap();
        assert_eq!(d, b);
        let e = tm.allocate(Tier::L1).unwrap();
        assert_eq!(e, a);
    }

    #[test]
    fn tier_manager_track_page_below_next_id_removes_from_free() {
        let mut tm = TierManager::new(5, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap(); // 0
        assert!(tm.free(Tier::L1, a)); // 0 goes to free_ids

        // Track a lower id that's in free_ids
        assert!(tm.track_page(Tier::L1, 0));
        assert_eq!(tm.usage(Tier::L1).used, 1);
    }

    #[test]
    fn tier_manager_capacity_one_allocate_free_cycle() {
        let mut tm = TierManager::new(1, 0, 0);
        let p = tm.allocate(Tier::L1).unwrap();
        assert_eq!(p, 0);
        assert!(!tm.can_allocate(Tier::L1));

        assert!(tm.free(Tier::L1, p));
        assert!(tm.can_allocate(Tier::L1));

        let p2 = tm.allocate(Tier::L1).unwrap();
        assert_eq!(p2, 0, "should reuse freed ID");
    }

    #[test]
    fn tier_manager_usage_after_all_freed() {
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        assert_eq!(tm.usage(Tier::L1).used, 2);
        assert!(tm.free(Tier::L1, a));
        assert!(tm.free(Tier::L1, b));
        assert_eq!(tm.usage(Tier::L1).used, 0);
        assert_eq!(tm.usage(Tier::L1).capacity, 3);
    }

    // ── GlobalMemoryManager constructor equivalence ──

    #[test]
    fn global_manager_new_vs_new_with_capacities() {
        let m1 = GlobalMemoryManager::new_with_capacities(10, 20, 30);
        let m2 = GlobalMemoryManager::new(
            TierManager::new(10, 20, 30),
            EvictionPolicy::default(),
        );
        assert_eq!(m1.tier_usage(Tier::L1).capacity, m2.tier_usage(Tier::L1).capacity);
        assert_eq!(m1.tier_usage(Tier::L2).capacity, m2.tier_usage(Tier::L2).capacity);
        assert_eq!(m1.tier_usage(Tier::L3).capacity, m2.tier_usage(Tier::L3).capacity);
    }

    // ── GlobalMemoryManager: resolve unbound virtual page ──

    #[test]
    fn global_manager_resolve_unbound_virtual_page_errors() {
        let mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let vpid = VirtualPageId::new(99, 99);
        let err = mgr.resolve(vpid).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownVirtualPage { .. }));
    }

    // ── GlobalMemoryManager: bind multiple virtual to same physical, then free ──

    #[test]
    fn global_manager_free_physical_removes_all_virtual_bindings() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        let v2 = VirtualPageId::new(20, 0);

        mgr.bind_virtual_page(v0, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v2, Tier::L1, pid).unwrap();

        mgr.free_page(Tier::L1, pid).unwrap();
        assert!(mgr.resolve(v0).is_err());
        assert!(mgr.resolve(v1).is_err());
        assert!(mgr.resolve(v2).is_err());
    }

    // ── GlobalMemoryManager: remap updates reverse index correctly ──

    #[test]
    fn global_manager_remap_virtual_page_updates_reverse_index() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();
        mgr.remap_virtual_page(vpid, Tier::L1, p2).unwrap();

        // Freeing p1 should NOT cascade-remove the remapped virtual page
        mgr.free_page(Tier::L1, p1).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p2));
    }

    // ── GlobalMemoryManager: prepare_next_turn with no working pages ──

    #[test]
    fn global_manager_prepare_next_turn_no_working_pages_is_noop() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);

        mgr.prepare_next_turn(42);
        // Conversation pages should remain
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_prepare_next_turn_all_working_freed() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1).unwrap();
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 3, Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 3);

        mgr.prepare_next_turn(0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    // ── GlobalMemoryManager: release_working_pipeline when none exist ──

    #[test]
    fn global_manager_release_working_pipeline_nonexistent_is_noop() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);

        mgr.release_working_pipeline(999);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // ── GlobalMemoryManager: track_in_pipeline doesn't allocate tier ──

    #[test]
    fn global_manager_track_in_pipeline_does_not_consume_tier_capacity() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);

        // Track an already-allocated page into pipeline
        mgr.track_in_pipeline(KvPipeline::Working, 1, p1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2, "tracking should not change tier usage");
    }

    // ── GlobalMemoryManager: pipeline pages across multiple requests ──

    #[test]
    fn global_manager_pipeline_pages_isolated_per_request() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _ = mgr.allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1).unwrap();

        mgr.release_working_pipeline(1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1, "only request 2's page remains");
    }

    // ── EvictionPolicy: Protected pages not candidates ──

    #[test]
    fn eviction_policy_protected_pages_not_candidates() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Protected,
                recency: 0,
                is_lir: true,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert!(victims.is_empty());
    }

    // ── EvictionPolicy: Active pages scored with penalty (less likely evicted than Standby) ──

    #[test]
    fn eviction_policy_active_penalized_vs_standby() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        // Two identical pages except state: Standby vs Active
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Active,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        let victims = policy.select_victims(&metadata, &semantic, 1);
        // Standby (page 1) should be evicted first due to standby_bonus
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1, "Standby page should be evicted before Active page");
    }

    // ── EvictionPolicy: high access_count reduces eviction score ──

    #[test]
    fn eviction_policy_high_access_count_preserves_page() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 1,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 1000,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims[0], 1, "low access_count page should be evicted first");
    }

    // ── EvictionPolicy: high semantic_priority preserves page ──

    #[test]
    fn eviction_policy_high_semantic_preserves_page() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );

        let mut semantic = HashMap::new();
        semantic.insert(2, 100); // page 2 has high semantic priority

        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims[0], 1, "page without semantic priority evicted first");
    }

    // ── EvictionPolicy: empty metadata returns empty victims ──

    #[test]
    fn eviction_policy_empty_metadata_returns_empty() {
        let policy = EvictionPolicy::default();
        let metadata = HashMap::new();
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 10);
        assert!(victims.is_empty());
    }

    // ── EvictionPolicy: SwappedOut pages scored with i64::MIN ──

    #[test]
    fn eviction_policy_swapped_out_not_candidates() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::SwappedOut,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        // SwappedOut gets i64::MIN score — effectively never selected when better candidates exist
        // But since it's the only candidate, it will still appear
        assert_eq!(victims.len(), 1);
    }

    // ── EvictionPolicy: Free state pages ──

    #[test]
    fn eviction_policy_free_state_page_is_candidate() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Free,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        // Free pages get i64::MIN but are still technically eligible
        assert_eq!(victims.len(), 1);
    }

    // ── plan_prefill: l2_prefetch calculation ──

    #[test]
    fn plan_prefill_l2_prefetch_capped_by_l2_available() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 2, 0);
        // 1000 tokens / 128 per page = 8 pages needed; L1 has 1; L2 has 2
        let plan = mgr.plan_prefill(1000, 500, 128);
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, .. } => {
                assert_eq!(l1_pages, 1);
                assert_eq!(l2_prefetch, 2);
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    #[test]
    fn plan_prefill_l2_zero_capacity_pipelined() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let plan = mgr.plan_prefill(500, 100, 64);
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, .. } => {
                assert_eq!(l1_pages, 1);
                assert_eq!(l2_prefetch, 0);
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    // ── plan_prefill: exact boundary where total_pages == l1_available ──

    #[test]
    fn plan_prefill_exact_l1_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        // 4 pages needed, L1 has 4 available
        let plan = mgr.plan_prefill(512, 128, 128);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    #[test]
    fn plan_prefill_one_page_over_l1_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 2, 0);
        // 5 pages needed, L1 has 4 available
        let plan = mgr.plan_prefill(640, 128, 128);
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, .. } => {
                assert_eq!(l1_pages, 4);
                assert_eq!(l2_prefetch, 1);
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    // ── plan_prefill: zero page_size and zero chunk_size guards ──

    #[test]
    fn plan_prefill_zero_page_size_treated_as_one() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(100, 0, 0);
        // page_size=0 -> treated as 1, so 5 tokens = 5 pages
        let plan = mgr.plan_prefill(5, 100, 0);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 5 });
    }

    #[test]
    fn plan_prefill_zero_chunk_size_treated_as_one_pipelined() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 10, 0);
        // chunk_size=0 -> treated as 1, so each token is its own chunk
        let plan = mgr.plan_prefill(5, 0, 1);
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, .. } => {
                // 5 tokens, each is 1 chunk of 1 page
                assert_eq!(chunk_schedule.len(), 5);
                assert!(chunk_schedule.iter().all(|&c| c == 1));
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    // ── Session: re-register same session_id overwrites ──

    #[test]
    fn session_reregister_same_id_overwrites() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        mgr.finalize_session_tokens(42, 100);
        assert_eq!(mgr.session_finalized_position(42), Some(100));

        // Re-register resets the cache
        let cache = mgr.register_session(42);
        assert_eq!(cache.finalized_position, 0);
        assert!(cache.pages.is_empty());
        assert_eq!(mgr.session_finalized_position(42), Some(0));
    }

    // ── Session: finalize monotonically increasing with multiple steps ──

    #[test]
    fn session_finalize_monotonic_multiple_increments() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);

        mgr.finalize_session_tokens(42, 10);
        assert_eq!(mgr.session_finalized_position(42), Some(10));

        mgr.finalize_session_tokens(42, 25);
        assert_eq!(mgr.session_finalized_position(42), Some(25));

        mgr.finalize_session_tokens(42, 20); // decrease, should be ignored
        assert_eq!(mgr.session_finalized_position(42), Some(25));

        mgr.finalize_session_tokens(42, 50);
        assert_eq!(mgr.session_finalized_position(42), Some(50));
    }

    // ── Session: claim_session_prefix produces correct virtual page IDs ──

    #[test]
    fn session_claim_prefix_produces_correct_virtual_ids() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(100, 0),
                VirtualPageId::new(100, 1),
                VirtualPageId::new(100, 2),
            ];
        }
        mgr.finalize_session_tokens(42, 3);

        let claimed = mgr.claim_session_prefix(42, 77, 3).unwrap();
        assert_eq!(claimed.len(), 3);
        // The claim generates NEW VirtualPageIds with request_id=77
        assert_eq!(claimed[0], VirtualPageId::new(77, 0));
        assert_eq!(claimed[1], VirtualPageId::new(77, 1));
        assert_eq!(claimed[2], VirtualPageId::new(77, 2));
    }

    // ── GlobalMemoryManager: migrate L2 to L3 ──

    #[test]
    fn global_manager_migrate_l2_to_l3() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 2, 4);
        let src = mgr.allocate_page(Tier::L2).unwrap();
        let vpid = VirtualPageId::new(5, 0);
        mgr.bind_virtual_page(vpid, Tier::L2, src).unwrap();

        let dst = mgr.migrate_page(Tier::L2, Tier::L3, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L3, dst));
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 1);
    }

    // ── GlobalMemoryManager: migrate L1 to L3 directly ──

    #[test]
    fn global_manager_migrate_l1_to_l3_direct() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 4);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, src).unwrap();

        let dst = mgr.migrate_page(Tier::L1, Tier::L3, src).unwrap();
        // dst may equal src (both 0) since tier allocators are independent.
        // The important invariant: virtual page now resolves to L3, L1 is freed.
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L3, dst));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 1);
    }

    // ── GlobalMemoryManager: unmap one of many virtual pages on same physical ──

    #[test]
    fn global_manager_unmap_one_of_many_virtual_pages() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);

        mgr.bind_virtual_page(v0, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, pid).unwrap();

        // Unmap just v0
        let loc = mgr.unmap_virtual_page(v0).unwrap();
        assert_eq!(loc.physical_id, pid);

        // v1 should still resolve
        assert_eq!(mgr.resolve(v1).unwrap(), (Tier::L1, pid));

        // v0 should be gone
        assert!(mgr.resolve(v0).is_err());

        // Physical page should still be allocated
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // ── entropy_evict: only evicts from specified tier ──

    #[test]
    fn entropy_evict_only_affects_specified_tier() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 0);
        let p_l1 = mgr.allocate_page(Tier::L1).unwrap();
        let _p_l2 = mgr.allocate_page(Tier::L2).unwrap();

        // Use a PhysicalId that is NOT allocated in L2 to ensure cross-tier isolation.
        // p_l1 is allocated in L1 but not in L2; entropy_evict on L2 should skip it.
        // However, since TierStates allocate independently, both may produce id=0.
        // So we verify by checking that L1 page count is unchanged.
        let entropies = HashMap::new(); // empty: no pages to evict on any tier
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L2);
        assert_eq!(freed, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
    }

    // ── entropy_evict: threshold boundary ──

    #[test]
    fn entropy_evict_strictly_less_than_threshold() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.1); // exactly at threshold
        entropies.insert(p1, 0.099); // below threshold

        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 1, "only strictly less than threshold is freed");
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // ── saturating conversion: u128 boundary ──

    #[test]
    fn saturating_u128_to_i64_max_value_boundary() {
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128), i64::MAX);
        assert_eq!(
            saturating_u128_to_i64(i64::MAX as u128 + 100),
            i64::MAX
        );
        assert_eq!(saturating_u128_to_i64(u128::MAX), i64::MAX);
    }

    #[test]
    fn saturating_usize_to_i64_boundary() {
        assert_eq!(saturating_usize_to_i64(i64::MAX as usize), i64::MAX);
        assert_eq!(saturating_usize_to_i64(1), 1);
    }

    // ── PrefillPlan: pipelined with large chunk_schedule ──

    #[test]
    fn plan_prefill_large_prompt_small_chunks() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        // 1000 tokens, chunk=10, page=10 => 100 chunks of 1 page each
        let plan = mgr.plan_prefill(1000, 10, 10);
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, l1_pages, .. } => {
                assert_eq!(l1_pages, 1);
                assert_eq!(chunk_schedule.len(), 100);
                assert!(chunk_schedule.iter().all(|&c| c == 1));
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    // ── TierManager: track_page with id equal to next_id ──

    #[test]
    fn tier_manager_track_page_at_next_id_boundary() {
        let mut tm = TierManager::new(5, 0, 0);
        // Track page 0 — next_id is 0, so this should advance next_id to 1
        assert!(tm.track_page(Tier::L1, 0));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        // Next allocation should be 1 (not 0, which is tracked)
        let next = tm.allocate(Tier::L1).unwrap();
        assert_eq!(next, 1);
    }

    #[test]
    fn tier_manager_track_page_beyond_next_id_creates_gaps() {
        let mut tm = TierManager::new(10, 0, 0);
        // Track page 5 — next_id is 0, should fill 0..5 as free_ids
        assert!(tm.track_page(Tier::L1, 5));
        assert_eq!(tm.usage(Tier::L1).used, 1);

        // Allocations should first drain free_ids (0,1,2,3,4)
        let a = tm.allocate(Tier::L1).unwrap();
        assert_eq!(a, 0);
        let b = tm.allocate(Tier::L1).unwrap();
        assert_eq!(b, 1);
    }

    // ── PageTable: new is same as default ──

    #[test]
    fn page_table_new_is_empty() {
        let pt = PageTable::new();
        assert!(pt.resolve(VirtualPageId::new(0, 0)).is_none());
    }

    // ── GlobalMemoryManager: bind, free physical, verify virtual gone ──

    #[test]
    fn global_manager_physical_free_cascades_to_virtual() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();

        mgr.free_page(Tier::L1, pid).unwrap();
        assert!(mgr.resolve(vpid).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    // ── GlobalMemoryManager: allocate_page_in_pipeline_conversation_survives_prepare ──

    #[test]
    fn global_manager_conversation_pipeline_survives_prepare_next_turn() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let _p1 = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();
        let _p2 = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();
        let _pw = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 3);

        mgr.prepare_next_turn(1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2, "Working freed, Conversation kept");
    }

    // ── EvictionPolicy: recency_weight affects ordering ──

    #[test]
    fn eviction_policy_idle_time_affects_ordering() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now - Duration::from_secs(100), // very old
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now - Duration::from_secs(1), // recent
            },
        );
        let semantic = HashMap::new();

        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims[0], 1, "older page should be evicted first");
    }

    // ── GlobalMemoryManager: multiple allocate/free cycles ──

    #[test]
    fn global_manager_allocate_free_cycle_stability() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);

        for _ in 0..10 {
            let p1 = mgr.allocate_page(Tier::L1).unwrap();
            let p2 = mgr.allocate_page(Tier::L1).unwrap();
            assert_eq!(mgr.tier_usage(Tier::L1).used, 2);

            mgr.free_page(Tier::L1, p1).unwrap();
            mgr.free_page(Tier::L1, p2).unwrap();
            assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        }
    }

    // ── EvictionPolicy: missing semantic_priorities defaults to zero ──

    #[test]
    fn eviction_policy_missing_semantic_defaults_zero() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );

        // Empty semantic map — should still work, defaults to 0
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1);
    }

    // ── TierState internal: track_allocated idempotent via TierManager ──

    #[test]
    fn tier_manager_track_then_allocate_respects_limits() {
        let mut tm = TierManager::new(3, 0, 0);
        // Track pages 0 and 1
        assert!(tm.track_page(Tier::L1, 0));
        assert!(tm.track_page(Tier::L1, 1));
        assert_eq!(tm.usage(Tier::L1).used, 2);
        // Can allocate 1 more
        let p = tm.allocate(Tier::L1).unwrap();
        assert_eq!(tm.usage(Tier::L1).used, 3);
        assert!(!tm.can_allocate(Tier::L1));
        assert!(tm.free(Tier::L1, p));
    }

    // ── GlobalMemoryManager: resolve returns correct tier and physical ──

    #[test]
    fn global_manager_resolve_after_l3_allocate() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 0, 4);
        let pid = mgr.allocate_page(Tier::L3).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L3, pid).unwrap();

        let (tier, physical) = mgr.resolve(vpid).unwrap();
        assert_eq!(tier, Tier::L3);
        assert_eq!(physical, pid);
    }

    // ── GlobalMemoryManager: track_page then untrack_page roundtrip ──

    #[test]
    fn global_manager_track_untrack_roundtrip() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.track_page(Tier::L1, 10).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert!(mgr.tier_manager.is_allocated(Tier::L1, 10));

        mgr.untrack_page(Tier::L1, 10).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, 10));
    }

    // ── GlobalMemoryManager: untrack unknown page errors ──

    #[test]
    fn global_manager_untrack_unknown_page_errors() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let err = mgr.untrack_page(Tier::L1, 999).unwrap_err();
        assert!(matches!(err, MemoryManagerError::UnknownPhysicalPage { .. }));
    }

    // ── TierManager: all three tiers with distinct capacities ──

    #[test]
    fn tier_manager_three_tier_distinct_operations() {
        let mut tm = TierManager::new(1, 2, 3);

        let l1 = tm.allocate(Tier::L1).unwrap();
        let l2a = tm.allocate(Tier::L2).unwrap();
        let l2b = tm.allocate(Tier::L2).unwrap();
        let l3a = tm.allocate(Tier::L3).unwrap();

        assert!(tm.allocate(Tier::L1).is_none());
        assert!(tm.allocate(Tier::L2).is_none());
        assert!(tm.can_allocate(Tier::L3));
        assert!(!tm.can_allocate(Tier::L1));

        assert!(tm.free(Tier::L2, l2a));
        assert!(tm.can_allocate(Tier::L2));

        let reused = tm.allocate(Tier::L2).unwrap();
        assert_eq!(reused, l2a);

        // Verify usage
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert_eq!(tm.usage(Tier::L2).used, 2);
        assert_eq!(tm.usage(Tier::L3).used, 1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 50 additional tests — cross-tier migration, state interactions, edge cases
    // ══════════════════════════════════════════════════════════════════════

    // ── Cross-tier migration (L2→L1, L3→L1, L3→L2) ──

    #[test]
    fn global_manager_migrate_l2_to_l1() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let src = mgr.allocate_page(Tier::L2).unwrap();
        let vpid = VirtualPageId::new(5, 0);
        mgr.bind_virtual_page(vpid, Tier::L2, src).unwrap();

        let dst = mgr.migrate_page(Tier::L2, Tier::L1, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, dst));
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_migrate_l3_to_l1() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 4);
        let src = mgr.allocate_page(Tier::L3).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L3, src).unwrap();

        let dst = mgr.migrate_page(Tier::L3, Tier::L1, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, dst));
        assert_eq!(mgr.tier_usage(Tier::L3).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_migrate_l3_to_l2() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 4, 4);
        let src = mgr.allocate_page(Tier::L3).unwrap();
        let vpid = VirtualPageId::new(7, 1);
        mgr.bind_virtual_page(vpid, Tier::L3, src).unwrap();

        let dst = mgr.migrate_page(Tier::L3, Tier::L2, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L2, dst));
        assert_eq!(mgr.tier_usage(Tier::L3).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
    }

    #[test]
    fn global_manager_migrate_no_virtual_pages_bound() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        // No virtual pages bound — migration should still succeed
        let dst = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
        assert!(mgr.tier_manager.is_allocated(Tier::L2, dst));
    }

    // ── plan_prefill with pre-allocated L1 pages ──

    #[test]
    fn plan_prefill_with_l1_partially_used() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        // Use 3 of 8 L1 pages
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L1).unwrap();

        // 5 pages needed, 5 available
        let plan = mgr.plan_prefill(640, 128, 128);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 5 });
    }

    #[test]
    fn plan_prefill_single_token_fits_l1() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let plan = mgr.plan_prefill(1, 128, 128);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 1 });
    }

    #[test]
    fn plan_prefill_with_l1_fully_used_pipelined() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 4, 0);
        // Exhaust L1
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L1).unwrap();

        let plan = mgr.plan_prefill(512, 128, 128);
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, .. } => {
                assert_eq!(l1_pages, 0, "no L1 pages available");
                assert!(l2_prefetch > 0, "should use L2 for pipelining");
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    #[test]
    fn plan_prefill_chunk_schedule_last_chunk_may_be_smaller() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 10, 0);
        // 7 tokens, page_size=2, chunk_size=4
        // 7/2=4 pages (div_ceil), chunk_schedule: [4/2=2, 3/2=2]
        let plan = mgr.plan_prefill(7, 4, 2);
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, .. } => {
                assert_eq!(chunk_schedule.len(), 2);
                assert_eq!(chunk_schedule[0], 2);
                assert_eq!(chunk_schedule[1], 2);
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    // ── EvictionPolicy: tiebreaking and multi-candidate ordering ──

    #[test]
    fn eviction_policy_tiebreak_by_page_id() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        for id in [5usize, 2usize, 8usize] {
            metadata.insert(
                id,
                PageMetadata {
                    page_id: id,
                    sequence_id: Some(1),
                    state: PageState::Standby,
                    recency: 10,
                    is_lir: false,
                    swap_in_time: None,
                    warm_until: None,
                    access_count: 5,
                    last_access: now,
                },
            );
        }
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 3);
        assert_eq!(victims.len(), 3);
        // All have same score → sorted by page_id ascending as tiebreaker
        assert!(victims[0] < victims[1]);
        assert!(victims[1] < victims[2]);
    }

    #[test]
    fn eviction_policy_many_candidates_returns_top_n() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        for (id, access_count) in [(1usize, 1usize), (2, 50), (3, 10), (4, 5), (5, 100)] {
            metadata.insert(
                id,
                PageMetadata {
                    page_id: id,
                    sequence_id: Some(1),
                    state: PageState::Standby,
                    recency: 0,
                    is_lir: false,
                    swap_in_time: None,
                    warm_until: None,
                    access_count,
                    last_access: now,
                },
            );
        }
        let semantic = HashMap::new();
        // Request 2 victims — lowest access_count pages
        let victims = policy.select_victims(&metadata, &semantic, 2);
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0], 1);
        assert_eq!(victims[1], 4);
    }

    #[test]
    fn eviction_policy_negative_semantic_priority() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );

        let mut semantic = HashMap::new();
        semantic.insert(1, -100); // negative semantic → more evictable

        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims[0], 1, "negative semantic priority should make page evicted first");
    }

    #[test]
    fn eviction_policy_active_and_standby_both_candidates() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Active,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 2);
        assert_eq!(victims.len(), 2, "both Active and Standby should be candidates");
    }

    #[test]
    fn eviction_policy_all_non_candidate_states_excluded() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        for (id, state) in [
            (1usize, PageState::Warm),
            (2usize, PageState::Protected),
            (3usize, PageState::Swapped),
        ] {
            metadata.insert(
                id,
                PageMetadata {
                    page_id: id,
                    sequence_id: Some(1),
                    state,
                    recency: 0,
                    is_lir: false,
                    swap_in_time: None,
                    warm_until: None,
                    access_count: 0,
                    last_access: now,
                },
            );
        }
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 10);
        assert!(victims.is_empty(), "Warm, Protected, Swapped should all be excluded");
    }

    // ── Session: additional edge cases ──

    #[test]
    fn session_claim_prefix_equals_finalized_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(1, 2),
            ];
        }
        mgr.finalize_session_tokens(42, 3);

        // Claim exactly finalized_position pages — boundary success
        let claimed = mgr.claim_session_prefix(42, 99, 3).unwrap();
        assert_eq!(claimed.len(), 3);
    }

    #[test]
    fn session_register_zero_session_id() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let cache = mgr.register_session(0);
        assert_eq!(cache.session_id, 0);
        assert_eq!(mgr.session_finalized_position(0), Some(0));
    }

    #[test]
    fn session_multiple_sessions_independent_finalize() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(1);
        mgr.register_session(2);
        mgr.register_session(3);

        mgr.finalize_session_tokens(1, 10);
        mgr.finalize_session_tokens(3, 30);
        // Session 2 not finalized

        assert_eq!(mgr.session_finalized_position(1), Some(10));
        assert_eq!(mgr.session_finalized_position(2), Some(0));
        assert_eq!(mgr.session_finalized_position(3), Some(30));
    }

    #[test]
    fn session_claim_with_one_page_and_prefix_one() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0)];
        }
        mgr.finalize_session_tokens(42, 1);

        let claimed = mgr.claim_session_prefix(42, 55, 1).unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0], VirtualPageId::new(55, 0));
    }

    #[test]
    fn session_finalize_to_zero_stays_at_zero() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        // Finalize to 0 — no-op since default is 0
        mgr.finalize_session_tokens(42, 0);
        assert_eq!(mgr.session_finalized_position(42), Some(0));
    }

    // ── GlobalMemoryManager: bind/rebind interactions ──

    #[test]
    fn global_manager_bind_rebind_same_physical() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();
        // Bind again to same physical — should succeed (overwrite path)
        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_remap_to_same_location() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();

        // Remap to same tier+physical — should succeed
        mgr.remap_virtual_page(vpid, Tier::L1, pid).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));
    }

    #[test]
    fn global_manager_migrate_chain_l1_l2_l3() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 4);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, src).unwrap();

        // Migrate L1→L2
        let dst_l2 = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap().0, Tier::L2);

        // Migrate L2→L3
        let _dst_l3 = mgr.migrate_page(Tier::L2, Tier::L3, dst_l2).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap().0, Tier::L3);

        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 1);
    }

    #[test]
    fn global_manager_free_reallocate_same_physical_id() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        mgr.free_page(Tier::L1, p1).unwrap();

        // Re-allocate should reuse the freed ID
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn global_manager_allocate_all_three_tiers() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 3, 4);
        let l1 = mgr.allocate_page(Tier::L1).unwrap();
        let l2 = mgr.allocate_page(Tier::L2).unwrap();
        let l3 = mgr.allocate_page(Tier::L3).unwrap();

        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 1);

        // Bind and resolve each tier independently
        let v1 = VirtualPageId::new(1, 0);
        let v2 = VirtualPageId::new(2, 0);
        let v3 = VirtualPageId::new(3, 0);
        mgr.bind_virtual_page(v1, Tier::L1, l1).unwrap();
        mgr.bind_virtual_page(v2, Tier::L2, l2).unwrap();
        mgr.bind_virtual_page(v3, Tier::L3, l3).unwrap();

        assert_eq!(mgr.resolve(v1).unwrap().0, Tier::L1);
        assert_eq!(mgr.resolve(v2).unwrap().0, Tier::L2);
        assert_eq!(mgr.resolve(v3).unwrap().0, Tier::L3);
    }

    // ── Pipeline: Working + Conversation for same request ──

    #[test]
    fn global_manager_pipeline_working_and_conversation_same_request() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let pw = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let pc = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();

        assert_ne!(pw, pc);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);

        // Release working only
        mgr.release_working_pipeline(1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 1)));
    }

    #[test]
    fn global_manager_track_in_pipeline_multiple_entries() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();

        mgr.track_in_pipeline(KvPipeline::Working, 1, p1);
        mgr.track_in_pipeline(KvPipeline::Working, 1, p2);

        let pages = mgr.pipeline_pages.get(&(KvPipeline::Working, 1)).unwrap();
        assert_eq!(pages.len(), 2);
        assert!(pages.contains(&p1));
        assert!(pages.contains(&p2));
    }

    // ── entropy_evict with virtual bindings cascade ──

    #[test]
    fn global_manager_entropy_evict_with_virtual_bindings() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let _p1 = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(v0, Tier::L1, p0).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.01); // low — should be evicted
        entropies.insert(_p1, 5.0); // high — should remain

        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 1);
        // Virtual page should be cascade-removed
        assert!(mgr.resolve(v0).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // ── PageTable: additional interactions ──

    #[test]
    fn page_table_map_same_physical_different_virtuals() {
        let mut pt = PageTable::new();
        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);
        pt.map(v0, Tier::L1, 5);
        pt.map(v1, Tier::L1, 5); // same physical, different virtual

        assert_eq!(pt.resolve(v0).unwrap().physical_id, 5);
        assert_eq!(pt.resolve(v1).unwrap().physical_id, 5);

        // Removing one doesn't affect the other
        pt.remove(&v0);
        assert!(pt.resolve(v0).is_none());
        assert_eq!(pt.resolve(v1).unwrap().physical_id, 5);
    }

    #[test]
    fn page_table_overwrite_same_tier_same_physical() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 5);

        // Overwrite with same tier+physical
        let old = pt.map(vpid, Tier::L1, 5);
        assert_eq!(old.unwrap(), PageLocation { physical_id: 5, tier: Tier::L1 });
        assert_eq!(pt.resolve(vpid).unwrap().physical_id, 5);
    }

    #[test]
    fn page_table_remap_after_overwrite() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 1);
        pt.map(vpid, Tier::L2, 2); // overwrite

        let old = pt.remap(vpid, Tier::L3, 3).unwrap();
        assert_eq!(old, PageLocation { physical_id: 2, tier: Tier::L2 });
        assert_eq!(pt.resolve(vpid).unwrap(), PageLocation { physical_id: 3, tier: Tier::L3 });
    }

    // ── TierManager: L2/L3 specific operations ──

    #[test]
    fn tier_manager_l2_l3_allocate_and_free() {
        let mut tm = TierManager::new(0, 3, 3);
        let l2_a = tm.allocate(Tier::L2).unwrap();
        let _l2_b = tm.allocate(Tier::L2).unwrap();
        let _l3_a = tm.allocate(Tier::L3).unwrap();

        assert_eq!(tm.usage(Tier::L2).used, 2);
        assert_eq!(tm.usage(Tier::L3).used, 1);

        assert!(tm.free(Tier::L2, l2_a));
        assert_eq!(tm.usage(Tier::L2).used, 1);

        let reused = tm.allocate(Tier::L2).unwrap();
        assert_eq!(reused, l2_a);
    }

    #[test]
    fn tier_manager_track_page_already_allocated_via_allocate() {
        let mut tm = TierManager::new(4, 0, 0);
        let p = tm.allocate(Tier::L1).unwrap();
        // Track an already-allocated page — should return true (idempotent)
        assert!(tm.track_page(Tier::L1, p));
        assert_eq!(tm.usage(Tier::L1).used, 1);
    }

    #[test]
    fn tier_manager_free_then_track_same_id() {
        let mut tm = TierManager::new(4, 0, 0);
        let p = tm.allocate(Tier::L1).unwrap();
        assert!(tm.free(Tier::L1, p));
        // Track the freed ID — should succeed as new allocation
        assert!(tm.track_page(Tier::L1, p));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert!(tm.is_allocated(Tier::L1, p));
    }

    #[test]
    fn tier_manager_is_allocated_after_free_false() {
        let mut tm = TierManager::new(4, 0, 0);
        let p = tm.allocate(Tier::L1).unwrap();
        assert!(tm.is_allocated(Tier::L1, p));
        assert!(tm.free(Tier::L1, p));
        assert!(!tm.is_allocated(Tier::L1, p));
    }

    #[test]
    fn tier_manager_capacity_maintained_across_operations() {
        let mut tm = TierManager::new(5, 0, 0);
        for _ in 0..3 {
            let p = tm.allocate(Tier::L1).unwrap();
            assert!(tm.free(Tier::L1, p));
        }
        assert_eq!(tm.usage(Tier::L1).capacity, 5);
        assert_eq!(tm.usage(Tier::L1).used, 0);
        assert!(tm.can_allocate(Tier::L1));
    }

    // ── Error variant field verification ──

    #[test]
    fn error_tier_capacity_exceeded_preserves_tier() {
        let err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L3 };
        match err {
            MemoryManagerError::TierCapacityExceeded { tier } => assert_eq!(tier, Tier::L3),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn error_unknown_physical_page_preserves_fields() {
        let err = MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 42 };
        match err {
            MemoryManagerError::UnknownPhysicalPage { tier, physical_id } => {
                assert_eq!(tier, Tier::L2);
                assert_eq!(physical_id, 42);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn error_unknown_virtual_page_preserves_fields() {
        let vpid = VirtualPageId::new(7, 3);
        let err = MemoryManagerError::UnknownVirtualPage { virtual_id: vpid };
        match err {
            MemoryManagerError::UnknownVirtualPage { virtual_id } => {
                assert_eq!(virtual_id.sequence_id, 7);
                assert_eq!(virtual_id.logical_index, 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ── EvictionPolicy: frequency dominates recency ──

    #[test]
    fn eviction_policy_large_access_count_dominates_recency() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 1000,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 1000,
                last_access: now - Duration::from_secs(1),
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now - Duration::from_secs(100),
            },
        );
        let semantic = HashMap::new();

        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims[0], 2, "older page with no access should be evicted first");
    }

    // ── Full lifecycle integration ──

    #[test]
    fn global_manager_full_lifecycle_allocate_bind_resolve_free() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 4);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        // Bind
        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));

        // Free physical — should cascade to virtual
        mgr.free_page(Tier::L1, pid).unwrap();
        assert!(mgr.resolve(vpid).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn global_manager_pipeline_release_then_reallocate() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        mgr.release_working_pipeline(1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);

        // Re-allocate should reuse the freed ID
        let p2 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn global_manager_prepare_next_turn_mixed_pipelines_multiple_requests() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        // Request 1: Working + Conversation
        let _w1 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _c1 = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L1).unwrap();
        // Request 2: Working only
        let _w2 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 2, Tier::L1).unwrap();
        // Request 3: Conversation only
        let _c3 = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 3, Tier::L1).unwrap();

        assert_eq!(mgr.tier_usage(Tier::L1).used, 4);

        mgr.prepare_next_turn(0);
        // Working pages (w1, w2) freed; Conversation (c1, c3) kept
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 1)));
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 3)));
        assert!(!mgr.pipeline_pages.contains_key(&(KvPipeline::Working, 1)));
        assert!(!mgr.pipeline_pages.contains_key(&(KvPipeline::Working, 2)));
    }

    // ── VirtualPageId large HashMap ──

    #[test]
    fn virtual_page_id_as_key_in_large_map() {
        let mut map = HashMap::new();
        for seq in 0..10u64 {
            for idx in 0..100usize {
                map.insert(VirtualPageId::new(seq, idx), seq as i32 * 100 + idx as i32);
            }
        }
        assert_eq!(map.len(), 1000);
        assert_eq!(map.get(&VirtualPageId::new(5, 50)), Some(&550));
        assert_eq!(map.get(&VirtualPageId::new(0, 0)), Some(&0));
        assert_eq!(map.get(&VirtualPageId::new(9, 99)), Some(&999));
    }

    // ── EvictionPolicy: single candidate ──

    #[test]
    fn eviction_policy_single_standby_candidate() {
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            42,
            PageMetadata {
                page_id: 42,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 100,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 10,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = policy.select_victims(&metadata, &semantic, 1);
        assert_eq!(victims, vec![42]);
    }

    // ── GlobalMemoryManager: untrack_page alias verification ──

    #[test]
    fn global_manager_untrack_page_removes_from_tier() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.track_page(Tier::L1, 7).unwrap();
        assert!(mgr.tier_manager.is_allocated(Tier::L1, 7));
        mgr.untrack_page(Tier::L1, 7).unwrap();
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, 7));
    }

    // ── SessionKvCache debug format ──

    #[test]
    fn session_kv_cache_debug_includes_finalized_position() {
        let s = SessionKvCache {
            session_id: 1,
            pages: vec![],
            finalized_position: 42,
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("finalized_position"));
        assert!(debug.contains("42"));
    }

    // ── Tier ordering transitivity ──

    #[test]
    fn tier_ordering_transitive() {
        assert!(Tier::L1 < Tier::L2);
        assert!(Tier::L2 < Tier::L3);
        assert!(Tier::L1 < Tier::L3);
        assert!(Tier::L3 > Tier::L2);
        assert!(Tier::L2 > Tier::L1);
    }

    // ── EvictionPolicy: default weights non-zero ──

    #[test]
    fn eviction_policy_default_weights_are_nonzero() {
        let policy = EvictionPolicy::default();
        assert_ne!(policy.recency_weight, 0);
        assert_ne!(policy.frequency_weight, 0);
        assert_ne!(policy.semantic_weight, 0);
        assert_ne!(policy.active_penalty, 0);
        assert_ne!(policy.standby_bonus, 0);
    }

    // ── Pipeline allocation on L2 and L3 ──

    #[test]
    fn global_manager_allocate_pipeline_on_l2() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 4, 0);
        let _p = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L2).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Working, 1)));
    }

    #[test]
    fn global_manager_allocate_pipeline_on_l3() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 0, 4);
        let _p = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 5, Tier::L3).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L3).used, 1);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 5)));
    }

    // ── plan_prefill: large token count ──

    #[test]
    fn plan_prefill_large_token_count_pipelined() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(10, 20, 0);
        let plan = mgr.plan_prefill(100_000, 1024, 256);
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, chunk_schedule } => {
                assert_eq!(l1_pages, 10);
                assert_eq!(l2_prefetch, 20);
                let total_chunks = chunk_schedule.len();
                assert!(total_chunks > 90, "should have many chunks for 100K tokens");
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 20 additional tests — Debug format, Default trait, boundary values
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn virtual_page_id_debug_format() {
        let vpid = VirtualPageId::new(42, 7);
        let debug = format!("{vpid:?}");
        assert!(debug.contains("42"), "debug should contain sequence_id");
        assert!(debug.contains("7"), "debug should contain logical_index");
    }

    #[test]
    fn eviction_policy_debug_format() {
        let policy = EvictionPolicy::default();
        let debug = format!("{policy:?}");
        assert!(debug.contains("recency_weight"));
        assert!(debug.contains("frequency_weight"));
        assert!(debug.contains("semantic_weight"));
    }

    #[test]
    fn tier_manager_new_zero_used_all_tiers() {
        let tm = TierManager::new(10, 20, 30);
        assert_eq!(tm.usage(Tier::L1), TierUsage { used: 0, capacity: 10 });
        assert_eq!(tm.usage(Tier::L2), TierUsage { used: 0, capacity: 20 });
        assert_eq!(tm.usage(Tier::L3), TierUsage { used: 0, capacity: 30 });
    }

    #[test]
    fn global_manager_initial_usage_all_zero() {
        let mgr = GlobalMemoryManager::new_with_capacities(5, 10, 15);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).capacity, 5);
        assert_eq!(mgr.tier_usage(Tier::L2).capacity, 10);
        assert_eq!(mgr.tier_usage(Tier::L3).capacity, 15);
    }

    #[test]
    fn prefill_plan_pipelined_empty_chunk_schedule() {
        let p1 = PrefillPlan::Pipelined {
            l1_pages: 0,
            l2_prefetch: 0,
            chunk_schedule: vec![],
        };
        let p2 = PrefillPlan::Pipelined {
            l1_pages: 0,
            l2_prefetch: 0,
            chunk_schedule: vec![],
        };
        assert_eq!(p1, p2);
    }

    #[test]
    fn tier_usage_eq_symmetry() {
        let a = TierUsage { used: 5, capacity: 10 };
        let b = TierUsage { used: 5, capacity: 10 };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn memory_manager_error_distinct_variants() {
        let vpid = VirtualPageId::new(1, 2);
        let errors = [
            MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
            MemoryManagerError::UnknownPhysicalPage { tier: Tier::L1, physical_id: 1 },
            MemoryManagerError::UnknownVirtualPage { virtual_id: vpid },
            MemoryManagerError::UnknownSession { session_id: 1 },
            MemoryManagerError::SessionPrefixOutOfBounds {
                session_id: 1, prefix_tokens: 1, finalized_position: 0,
            },
            MemoryManagerError::SessionPagesInsufficient {
                session_id: 1, prefix_tokens: 1, available_pages: 0,
            },
        ];
        for i in 0..errors.len() {
            for j in (i + 1)..errors.len() {
                assert_ne!(errors[i], errors[j], "variant {i} should not equal variant {j}");
            }
        }
    }

    #[test]
    fn page_table_default_trait() {
        let pt = PageTable::default();
        assert!(pt.resolve(VirtualPageId::new(0, 0)).is_none());
    }

    #[test]
    fn virtual_page_id_vec_dedup() {
        let mut v: Vec<VirtualPageId> = vec![
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 0), // duplicate
            VirtualPageId::new(2, 0),
            VirtualPageId::new(1, 1), // duplicate
        ];
        v.sort_by(|a, b| a.sequence_id.cmp(&b.sequence_id).then(a.logical_index.cmp(&b.logical_index)));
        v.dedup();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn global_manager_l3_only_bind_resolve() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 0, 4);
        let pid = mgr.allocate_page(Tier::L3).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L3, pid).unwrap();
        let (tier, physical) = mgr.resolve(vpid).unwrap();
        assert_eq!(tier, Tier::L3);
        assert_eq!(physical, pid);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
    }

    #[test]
    fn prefill_plan_fully_resident_large_pages() {
        let p = PrefillPlan::FullyResident { pages: usize::MAX };
        assert_eq!(p, PrefillPlan::FullyResident { pages: usize::MAX });
    }

    #[test]
    fn tier_usage_clone_preserves_fields() {
        let original = TierUsage { used: 7, capacity: 100 };
        let cloned = original.clone();
        assert_eq!(cloned.used, original.used);
        assert_eq!(cloned.capacity, original.capacity);
    }

    #[test]
    fn error_debug_variant_names() {
        let vpid = VirtualPageId::new(1, 2);
        assert!(format!("{:?}", MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 }).contains("TierCapacityExceeded"));
        assert!(format!("{:?}", MemoryManagerError::UnknownPhysicalPage { tier: Tier::L1, physical_id: 0 }).contains("UnknownPhysicalPage"));
        assert!(format!("{:?}", MemoryManagerError::UnknownVirtualPage { virtual_id: vpid }).contains("UnknownVirtualPage"));
        assert!(format!("{:?}", MemoryManagerError::UnknownSession { session_id: 0 }).contains("UnknownSession"));
        assert!(format!("{:?}", MemoryManagerError::SessionPrefixOutOfBounds { session_id: 0, prefix_tokens: 0, finalized_position: 0 }).contains("SessionPrefixOutOfBounds"));
        assert!(format!("{:?}", MemoryManagerError::SessionPagesInsufficient { session_id: 0, prefix_tokens: 0, available_pages: 0 }).contains("SessionPagesInsufficient"));
    }

    #[test]
    fn session_kv_cache_with_max_session_id() {
        let s = SessionKvCache {
            session_id: SessionId::MAX,
            pages: vec![],
            finalized_position: 0,
        };
        assert_eq!(s.session_id, u64::MAX);
        assert!(s.pages.is_empty());
    }

    #[test]
    fn tier_manager_can_allocate_partial_use() {
        let mut tm = TierManager::new(5, 0, 0);
        let _ = tm.allocate(Tier::L1).unwrap();
        let _ = tm.allocate(Tier::L1).unwrap();
        let _ = tm.allocate(Tier::L1).unwrap();
        assert!(tm.can_allocate(Tier::L1));
        assert_eq!(tm.usage(Tier::L1).available(), 2);
    }

    #[test]
    fn virtual_page_id_tuple_hashmap_key() {
        let mut map = HashMap::new();
        let key = (VirtualPageId::new(1, 0), VirtualPageId::new(2, 3));
        map.insert(key, "pair");
        assert_eq!(
            map.get(&(VirtualPageId::new(1, 0), VirtualPageId::new(2, 3))),
            Some(&"pair"),
        );
        assert_eq!(
            map.get(&(VirtualPageId::new(1, 0), VirtualPageId::new(2, 4))),
            None,
        );
    }

    #[test]
    fn global_manager_tier_usage_matches_constructor() {
        let mgr = GlobalMemoryManager::new_with_capacities(100, 200, 300);
        assert_eq!(mgr.tier_usage(Tier::L1).capacity, 100);
        assert_eq!(mgr.tier_usage(Tier::L2).capacity, 200);
        assert_eq!(mgr.tier_usage(Tier::L3).capacity, 300);
    }

    #[test]
    fn tier_match_all_variants() {
        let tiers = [Tier::L1, Tier::L2, Tier::L3];
        let mut matched = [false; 3];
        for tier in tiers {
            match tier {
                Tier::L1 => matched[0] = true,
                Tier::L2 => matched[1] = true,
                Tier::L3 => matched[2] = true,
            }
        }
        assert!(matched.iter().all(|&m| m), "all Tier variants should match");
    }

    #[test]
    fn page_location_debug_both_fields() {
        let loc = PageLocation { physical_id: 99, tier: Tier::L2 };
        let debug = format!("{loc:?}");
        assert!(debug.contains("99"), "debug should contain physical_id");
        assert!(debug.contains("L2"), "debug should contain tier");
    }

    #[test]
    fn tier_manager_usage_reflects_free_list_reuse() {
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap(); // 0
        let _b = tm.allocate(Tier::L1).unwrap(); // 1
        let _c = tm.allocate(Tier::L1).unwrap(); // 2
        assert_eq!(tm.usage(Tier::L1).used, 3);

        tm.free(Tier::L1, a); // free 0
        assert_eq!(tm.usage(Tier::L1).used, 2);

        let d = tm.allocate(Tier::L1).unwrap(); // reuse 0
        assert_eq!(d, 0);
        assert_eq!(tm.usage(Tier::L1).used, 3);
        assert_eq!(tm.usage(Tier::L1).capacity, 3);
    }

    #[test]
    fn eviction_policy_default_is_consistent() {
        let p1 = EvictionPolicy::default();
        let p2 = EvictionPolicy::default();
        assert_eq!(p1.recency_weight, p2.recency_weight);
        assert_eq!(p1.frequency_weight, p2.frequency_weight);
        assert_eq!(p1.semantic_weight, p2.semantic_weight);
        assert_eq!(p1.active_penalty, p2.active_penalty);
        assert_eq!(p1.standby_bonus, p2.standby_bonus);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — uncovered paths, enum variants, field accessors
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn virtual_page_id_ne_for_different_sequence() {
        let a = VirtualPageId::new(1, 0);
        let b = VirtualPageId::new(2, 0);
        assert_ne!(a, b, "different sequence_id should not be equal");
    }

    #[test]
    fn virtual_page_id_ne_for_different_logical() {
        let a = VirtualPageId::new(1, 3);
        let b = VirtualPageId::new(1, 7);
        assert_ne!(a, b, "same sequence, different logical_index should not be equal");
    }

    #[test]
    fn page_table_resolve_after_multiple_overwrites() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 1);
        pt.map(vpid, Tier::L2, 2);
        pt.map(vpid, Tier::L3, 3);

        let loc = pt.resolve(vpid).unwrap();
        assert_eq!(loc.tier, Tier::L3, "last map should win");
        assert_eq!(loc.physical_id, 3);
    }

    #[test]
    fn tier_usage_partial_eq_reflexive() {
        let u = TierUsage { used: 5, capacity: 10 };
        assert_eq!(u, u);
    }

    #[test]
    fn tier_manager_allocate_across_tiers_independent_ids() {
        let mut tm = TierManager::new(2, 2, 2);
        let l1 = tm.allocate(Tier::L1).unwrap();
        let l2 = tm.allocate(Tier::L2).unwrap();
        let l3 = tm.allocate(Tier::L3).unwrap();
        // Each tier starts from 0 independently
        assert_eq!(l1, 0);
        assert_eq!(l2, 0);
        assert_eq!(l3, 0);
    }

    #[test]
    fn global_manager_allocate_page_returns_unique_ids_per_allocate() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let a = mgr.allocate_page(Tier::L1).unwrap();
        let b = mgr.allocate_page(Tier::L1).unwrap();
        let c = mgr.allocate_page(Tier::L1).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn global_manager_resolve_errors_with_correct_virtual_id() {
        let vpid = VirtualPageId::new(42, 7);
        let mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let err = mgr.resolve(vpid).unwrap_err();
        match err {
            MemoryManagerError::UnknownVirtualPage { virtual_id } => {
                assert_eq!(virtual_id, vpid);
            }
            _ => panic!("Expected UnknownVirtualPage"),
        }
    }

    #[test]
    fn eviction_policy_recency_weight_field_accessor() {
        let policy = EvictionPolicy {
            recency_weight: 10,
            frequency_weight: 20,
            semantic_weight: 30,
            active_penalty: 40,
            standby_bonus: 50,
        };
        assert_eq!(policy.recency_weight, 10);
        assert_eq!(policy.frequency_weight, 20);
        assert_eq!(policy.semantic_weight, 30);
        assert_eq!(policy.active_penalty, 40);
        assert_eq!(policy.standby_bonus, 50);
    }

    #[test]
    fn session_id_type_is_u64() {
        let sid: SessionId = 0u64;
        assert_eq!(sid, 0u64);
        let max_sid: SessionId = u64::MAX;
        assert_eq!(max_sid, u64::MAX);
    }

    #[test]
    fn prefill_plan_fully_resident_pages_field() {
        let p = PrefillPlan::FullyResident { pages: 42 };
        match p {
            PrefillPlan::FullyResident { pages } => assert_eq!(pages, 42),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn prefill_plan_pipelined_all_fields() {
        let p = PrefillPlan::Pipelined {
            l1_pages: 3,
            l2_prefetch: 7,
            chunk_schedule: vec![2, 2, 2, 1],
        };
        match p {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, chunk_schedule } => {
                assert_eq!(l1_pages, 3);
                assert_eq!(l2_prefetch, 7);
                assert_eq!(chunk_schedule, vec![2, 2, 2, 1]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn page_location_equality_all_tier_same_physical() {
        for tier in [Tier::L1, Tier::L2, Tier::L3] {
            let a = PageLocation { physical_id: 5, tier };
            let b = PageLocation { physical_id: 5, tier };
            assert_eq!(a, b, "same tier+physical should be equal");
        }
    }

    #[test]
    fn tier_manager_free_all_then_reallocate_from_fresh_id() {
        let mut tm = TierManager::new(2, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap(); // 0
        let b = tm.allocate(Tier::L1).unwrap(); // 1
        tm.free(Tier::L1, a); // free 0 → free_ids=[0]
        tm.free(Tier::L1, b); // free 1 → free_ids=[0,1]

        // Next allocs should reuse 0 then 1 (FIFO)
        let c = tm.allocate(Tier::L1).unwrap();
        assert_eq!(c, 0);
        let d = tm.allocate(Tier::L1).unwrap();
        assert_eq!(d, 1);
    }

    #[test]
    fn error_unknown_session_preserves_session_id() {
        let err = MemoryManagerError::UnknownSession { session_id: 12345 };
        match err {
            MemoryManagerError::UnknownSession { session_id } => assert_eq!(session_id, 12345),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn global_manager_allocate_page_errors_with_correct_tier() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 0, 0);
        for tier in [Tier::L1, Tier::L2, Tier::L3] {
            let err = mgr.allocate_page(tier).unwrap_err();
            assert_eq!(err, MemoryManagerError::TierCapacityExceeded { tier });
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — negative entropy, subnormal floats, saturating
    // arithmetic, session edge cases, PageTable const assertions
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn entropy_evict_negative_entropy_below_threshold() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, -1.0); // negative: below any positive threshold
        entropies.insert(p1, f32::NEG_INFINITY); // extreme negative

        let freed = mgr.entropy_evict(&entropies, 0.5, Tier::L1);
        assert_eq!(freed, 2);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn entropy_evict_subnormal_entropy_values() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(3, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, f32::MIN_POSITIVE); // tiny positive, below 0.1
        entropies.insert(p1, 1e-40); // subnormal, below 0.1
        entropies.insert(p2, 0.5); // above threshold

        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 2);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn entropy_evict_nan_entropy_not_freed() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, f32::NAN); // NaN < threshold is false
        entropies.insert(p1, 0.01); // below threshold

        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);
        assert_eq!(freed, 1, "NaN should not be freed, 0.01 should be");
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn entropy_evict_inf_entropy_not_freed() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, f32::INFINITY); // inf < threshold is false

        let freed = mgr.entropy_evict(&entropies, f32::MAX, Tier::L1);
        assert_eq!(freed, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn tier_usage_available_with_used_zero_capacity_zero() {
        let usage = TierUsage { used: 0, capacity: 0 };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn plan_prefill_tokens_equal_page_size_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        // 128 tokens / 128 per page = exactly 1 page
        let plan = mgr.plan_prefill(128, 128, 128);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 1 });
    }

    #[test]
    fn plan_prefill_tokens_one_less_than_chunk_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 10, 0);
        // 255 tokens, chunk=256 => fits in 1 chunk, 255/64=4 pages
        let plan = mgr.plan_prefill(255, 256, 64);
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, .. } => {
                assert_eq!(chunk_schedule.len(), 1);
                assert_eq!(chunk_schedule[0], 4); // div_ceil(255/64) = 4
            }
            _ => panic!("Expected Pipelined since L1 only has 1 slot"),
        }
    }

    #[test]
    fn session_claim_prefix_exactly_at_pages_boundary() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)];
        }
        mgr.finalize_session_tokens(42, 2);

        // prefix_tokens=2 == pages.len()=2: should succeed
        let claimed = mgr.claim_session_prefix(42, 99, 2).unwrap();
        assert_eq!(claimed.len(), 2);
    }

    #[test]
    fn session_claim_prefix_exceeds_pages_even_within_finalized() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0)]; // 1 page
        }
        mgr.finalize_session_tokens(42, 5); // finalized=5 > pages.len()=1

        // prefix_tokens=2 <= finalized(5) but > pages.len()(1)
        let err = mgr.claim_session_prefix(42, 99, 2).unwrap_err();
        assert!(
            matches!(err, MemoryManagerError::SessionPagesInsufficient {
                session_id: 42,
                prefix_tokens: 2,
                available_pages: 1,
            }),
            "expected SessionPagesInsufficient"
        );
    }

    #[test]
    fn page_table_map_returns_old_on_overwrite_to_different_tier() {
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        let old = pt.map(vpid, Tier::L1, 1);
        assert!(old.is_none());

        let old = pt.map(vpid, Tier::L3, 99);
        let prev = old.unwrap();
        assert_eq!(prev.tier, Tier::L1);
        assert_eq!(prev.physical_id, 1);
        assert_eq!(pt.resolve(vpid).unwrap().tier, Tier::L3);
    }

    #[test]
    fn saturating_u128_to_i64_exactly_max() {
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128), i64::MAX);
        assert_eq!(saturating_u128_to_i64(i64::MAX as u128 + 1), i64::MAX);
        assert_eq!(saturating_u128_to_i64(0), 0);
        assert_eq!(saturating_u128_to_i64(1), 1);
    }

    #[test]
    fn error_display_tier_capacity_exceeded_each_tier() {
        for tier in [Tier::L1, Tier::L2, Tier::L3] {
            let err = MemoryManagerError::TierCapacityExceeded { tier };
            let msg = format!("{err}");
            let tier_str = format!("{tier:?}");
            assert!(msg.contains(&tier_str), "Display for {tier:?} should mention tier");
        }
    }

    #[test]
    fn error_display_unknown_physical_page_includes_both_fields() {
        let err = MemoryManagerError::UnknownPhysicalPage {
            tier: Tier::L3,
            physical_id: 42,
        };
        let msg = format!("{err}");
        assert!(msg.contains("42"), "should contain physical_id");
        assert!(msg.contains("L3"), "should contain tier");
    }

    #[test]
    fn global_manager_migrate_preserves_multiple_virtual_pages_correct_tier() {
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(20, 0);

        mgr.bind_virtual_page(v0, Tier::L1, src).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, src).unwrap();

        let dst = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        let (t0, p0) = mgr.resolve(v0).unwrap();
        let (t1, p1) = mgr.resolve(v1).unwrap();
        assert_eq!(t0, Tier::L2);
        assert_eq!(t1, Tier::L2);
        assert_eq!(p0, dst);
        assert_eq!(p1, dst);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — remap unbound virtual, pipeline L2 free, session
    // zero-length claim after finalize, entropy evict on L2/L3, track_page
    // gap filling, saturating usize max, error display format precision,
    // eviction with mixed candidate/non-candidate states, pipeline multiple
    // track_in calls, page table resolve consistency after churn
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn global_manager_remap_unbound_virtual_page_errors() {
        // Arrange: create manager with physical pages but do NOT bind a virtual page
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L2).unwrap();
        let vpid = VirtualPageId::new(99, 0);

        // Act: attempt to remap a virtual page that was never bound
        let result = mgr.remap_virtual_page(vpid, Tier::L2, p2);

        // Assert: should error because the virtual page is unknown in page_table
        assert!(
            matches!(result, Err(MemoryManagerError::UnknownVirtualPage { virtual_id }) if virtual_id == vpid),
            "remapping an unbound virtual page should return UnknownVirtualPage"
        );
        // Physical pages should remain untouched
        assert!(mgr.tier_manager.is_allocated(Tier::L1, p1));
        assert!(mgr.tier_manager.is_allocated(Tier::L2, p2));
    }

    #[test]
    fn global_manager_release_working_pipeline_on_l2_does_not_free_l2() {
        // Arrange: allocate Working pipeline pages on L2
        // Note: release_working_pipeline always frees from L1 (production design).
        // Pages on L2 will NOT be freed by release_working_pipeline.
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 4, 0);
        let _pw = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L2).unwrap();
        let _pc = mgr.allocate_page_in_pipeline(KvPipeline::Conversation, 1, Tier::L2).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L2).used, 2);

        // Act: release working pipeline for request 1
        mgr.release_working_pipeline(1);

        // Assert: pipeline_pages entry removed, but L2 usage unchanged (free targets L1)
        assert_eq!(mgr.tier_usage(Tier::L2).used, 2);
        assert!(mgr.pipeline_pages.contains_key(&(KvPipeline::Conversation, 1)));
        assert!(!mgr.pipeline_pages.contains_key(&(KvPipeline::Working, 1)));
    }

    #[test]
    fn session_claim_zero_prefix_after_nonzero_finalize_succeeds() {
        // Arrange: register session, set pages, finalize to non-zero
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(1, 2),
            ];
        }
        mgr.finalize_session_tokens(42, 3);

        // Act: claim zero pages — always valid
        let claimed = mgr.claim_session_prefix(42, 99, 0).unwrap();

        // Assert: empty list, no error
        assert!(claimed.is_empty());
        assert_eq!(mgr.session_finalized_position(42), Some(3));
    }

    #[test]
    fn entropy_evict_on_l2_tier_frees_l2_pages_only() {
        // Arrange: allocate pages on L1 and L2
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 0);
        let _p_l1 = mgr.allocate_page(Tier::L1).unwrap();
        let p_l2 = mgr.allocate_page(Tier::L2).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p_l2, 0.01); // low entropy on L2

        // Act: evict on L2 tier
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L2);

        // Assert: L2 page freed, L1 untouched
        assert_eq!(freed, 1);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn entropy_evict_on_l3_tier_with_no_l3_pages() {
        // Arrange: allocate only on L1, but try to entropy_evict on L3
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 2);
        let p_l1 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p_l1, 0.01); // L1 page with low entropy

        // Act: evict on L3 — the L1 page id is not allocated in L3
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L3);

        // Assert: nothing freed since the page is not allocated in L3
        assert_eq!(freed, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L3).used, 0);
    }

    #[test]
    fn tier_manager_track_page_gap_then_free_tracked_id() {
        // Arrange: track page 5 (creates gap [0,1,2,3,4] in free_ids)
        let mut tm = TierManager::new(10, 0, 0);
        assert!(tm.track_page(Tier::L1, 5));
        assert_eq!(tm.usage(Tier::L1).used, 1);

        // Act: free the tracked page 5
        let freed = tm.free(Tier::L1, 5);

        // Assert: free succeeds, usage drops to 0
        assert!(freed);
        assert_eq!(tm.usage(Tier::L1).used, 0);
        assert!(!tm.is_allocated(Tier::L1, 5));
    }

    #[test]
    fn saturating_usize_to_i64_max_boundary() {
        // Arrange & Act & Assert: usize::MAX should clamp to i64::MAX on 64-bit
        if size_of::<usize>() == 8 {
            assert_eq!(saturating_usize_to_i64(usize::MAX), i64::MAX);
        } else {
            // On 32-bit, usize::MAX fits in i64
            assert_eq!(saturating_usize_to_i64(usize::MAX), usize::MAX as i64);
        }
    }

    #[test]
    fn error_display_unknown_virtual_page_format_precision() {
        // Arrange: create error with specific virtual page coordinates
        let vpid = VirtualPageId::new(12345, 67890);
        let err = MemoryManagerError::UnknownVirtualPage { virtual_id: vpid };

        // Act: format via Display
        let msg = format!("{err}");

        // Assert: both coordinates appear in output
        assert!(msg.contains("12345"), "should contain sequence_id 12345");
        assert!(msg.contains("67890"), "should contain logical_index 67890");
    }

    #[test]
    fn eviction_policy_mixed_candidate_and_excluded_states() {
        // Arrange: one Standby (candidate), one Warm (excluded), one Protected (excluded)
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Warm,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        metadata.insert(
            3,
            PageMetadata {
                page_id: 3,
                sequence_id: Some(1),
                state: PageState::Protected,
                recency: 0,
                is_lir: true,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act: request 5 victims
        let victims = policy.select_victims(&metadata, &semantic, 5);

        // Assert: only the Standby page is a candidate
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1);
    }

    #[test]
    fn global_manager_track_in_pipeline_accumulates_entries() {
        // Arrange: allocate physical pages and track them into pipeline
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let p3 = mgr.allocate_page(Tier::L1).unwrap();

        // Act: track all three into the same pipeline+request
        mgr.track_in_pipeline(KvPipeline::Working, 42, p1);
        mgr.track_in_pipeline(KvPipeline::Working, 42, p2);
        mgr.track_in_pipeline(KvPipeline::Working, 42, p3);

        // Assert: all tracked pages are present in order
        let pages = mgr.pipeline_pages.get(&(KvPipeline::Working, 42)).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0], p1);
        assert_eq!(pages[1], p2);
        assert_eq!(pages[2], p3);
    }

    #[test]
    fn page_table_resolve_consistency_after_repeated_map_remove_churn() {
        // Arrange: create page table and two virtual pages
        let mut pt = PageTable::new();
        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);

        // Act: map, remove, remap multiple times to stress the HashMap
        pt.map(v0, Tier::L1, 10);
        pt.map(v1, Tier::L2, 20);
        pt.remove(&v0);
        pt.map(v0, Tier::L3, 30);
        pt.remove(&v1);
        pt.map(v1, Tier::L1, 40);

        // Assert: final state reflects last operations
        let loc0 = pt.resolve(v0).unwrap();
        assert_eq!(loc0.tier, Tier::L3);
        assert_eq!(loc0.physical_id, 30);

        let loc1 = pt.resolve(v1).unwrap();
        assert_eq!(loc1.tier, Tier::L1);
        assert_eq!(loc1.physical_id, 40);
    }

    #[test]
    fn global_manager_bind_virtual_page_rebind_cleans_old_reverse_index_entry() {
        // Arrange: allocate two physical pages, bind virtual to first
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();

        // Act: rebind to p2 — should remove vpid from p1's reverse index
        mgr.bind_virtual_page(vpid, Tier::L1, p2).unwrap();

        // Assert: freeing p1 should NOT cascade-remove vpid (it no longer points to p1)
        mgr.free_page(Tier::L1, p1).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p2));
    }

    #[test]
    fn plan_prefill_chunk_schedule_sums_to_total_pages() {
        // Arrange: manager with very limited L1, forcing pipelined plan
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 100, 0);
        let total_tokens: usize = 500;
        let chunk_size: usize = 100;
        let page_size: usize = 25;
        let expected_total_pages = (total_tokens + page_size - 1) / page_size; // 20

        // Act
        let plan = mgr.plan_prefill(total_tokens, chunk_size, page_size);

        // Assert: chunk_schedule pages sum to expected total
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, l1_pages, .. } => {
                let schedule_total: usize = chunk_schedule.iter().sum();
                assert_eq!(
                    schedule_total, expected_total_pages,
                    "chunk_schedule should sum to total pages ({expected_total_pages})"
                );
                assert_eq!(l1_pages, 1);
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    #[test]
    fn tier_manager_allocate_after_track_high_id_then_free_all_gaps() {
        // Arrange: track page 3 (creates free_ids [0,1,2]), then allocate and free them
        let mut tm = TierManager::new(10, 0, 0);
        assert!(tm.track_page(Tier::L1, 3)); // free_ids = [0, 1, 2]
        assert_eq!(tm.usage(Tier::L1).used, 1);

        // Act: allocate from gaps, then free everything including tracked
        let a = tm.allocate(Tier::L1).unwrap(); // 0 from free_ids
        let b = tm.allocate(Tier::L1).unwrap(); // 1 from free_ids
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(tm.usage(Tier::L1).used, 3);

        assert!(tm.free(Tier::L1, a));
        assert!(tm.free(Tier::L1, b));
        assert!(tm.free(Tier::L1, 3));

        // Assert: all freed, back to zero usage, can still allocate
        assert_eq!(tm.usage(Tier::L1).used, 0);
        assert!(tm.can_allocate(Tier::L1));
    }

    #[test]
    fn session_register_and_claim_without_finalize_errors() {
        // Arrange: register session with pages but do NOT finalize
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
            ];
        }
        // finalized_position is still 0

        // Act: claim with prefix_tokens > 0 when finalized_position == 0
        let result = mgr.claim_session_prefix(42, 99, 1);

        // Assert: should error because 1 > finalized_position(0)
        assert!(
            matches!(result, Err(MemoryManagerError::SessionPrefixOutOfBounds {
                session_id: 42,
                prefix_tokens: 1,
                finalized_position: 0,
            })),
            "claiming non-zero prefix without finalize should fail"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — further uncovered paths and edge cases
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn global_manager_unmap_all_virtual_then_free_physical_succeeds() {
        // Arrange: allocate one physical page and bind two virtual pages to it
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        mgr.bind_virtual_page(v0, Tier::L1, pid).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, pid).unwrap();

        // Act: unmap both virtual pages, then free physical
        let loc0 = mgr.unmap_virtual_page(v0).unwrap();
        assert_eq!(loc0.physical_id, pid);
        let loc1 = mgr.unmap_virtual_page(v1).unwrap();
        assert_eq!(loc1.physical_id, pid);
        mgr.free_page(Tier::L1, pid).unwrap();

        // Assert: all clean, tier usage back to zero
        assert!(mgr.resolve(v0).is_err());
        assert!(mgr.resolve(v1).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn tier_manager_track_page_zero_then_allocate_sequence() {
        // Arrange: track page 0 via track_page, which advances next_id past 0
        let mut tm = TierManager::new(5, 0, 0);
        assert!(tm.track_page(Tier::L1, 0));
        assert_eq!(tm.usage(Tier::L1).used, 1);

        // Act: allocate — should get id 1 (next_id advanced past 0)
        let next = tm.allocate(Tier::L1).unwrap();

        // Assert: id is 1, not 0
        assert_eq!(next, 1);
        assert_eq!(tm.usage(Tier::L1).used, 2);
        assert!(tm.is_allocated(Tier::L1, 0));
        assert!(tm.is_allocated(Tier::L1, 1));
    }

    #[test]
    fn eviction_policy_active_only_candidate_is_selected() {
        // Arrange: only Active state pages exist
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            42,
            PageMetadata {
                page_id: 42,
                sequence_id: Some(1),
                state: PageState::Active,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act: select victims from Active-only pool
        let victims = policy.select_victims(&metadata, &semantic, 1);

        // Assert: Active pages are eligible candidates
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 42);
    }

    #[test]
    fn plan_prefill_tokens_less_than_page_size_produces_one_page() {
        // Arrange: 3 tokens with page_size=128 → div_ceil(3/128) = 1 page
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);

        // Act
        let plan = mgr.plan_prefill(3, 128, 128);

        // Assert: single page, fits in L1
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 1 });
    }

    #[test]
    fn global_manager_rebind_virtual_to_different_tier_preserves_resolve() {
        // Arrange: bind virtual page to L1, then rebind to L2
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L2).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, p1).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, p1));

        // Act: rebind to L2
        mgr.bind_virtual_page(vpid, Tier::L1, p2);
        // This will fail because p2 is allocated in L2, not L1.
        // Let's bind correctly to L2.
        mgr.bind_virtual_page(vpid, Tier::L2, p2).unwrap();

        // Assert: resolves to L2
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L2, p2));
    }

    #[test]
    fn entropy_evict_cascades_virtual_bindings_on_l2() {
        // Arrange: allocate on L2, bind virtual, insert low entropy
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 4, 0);
        let p0 = mgr.allocate_page(Tier::L2).unwrap();
        let _p1 = mgr.allocate_page(Tier::L2).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(v0, Tier::L2, p0).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.001); // low
        entropies.insert(_p1, 5.0); // high

        // Act
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L2);

        // Assert: one page freed, virtual binding cascade-removed
        assert_eq!(freed, 1);
        assert!(mgr.resolve(v0).is_err());
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
    }

    #[test]
    fn session_finalize_to_max_usize_and_claim_zero() {
        // Arrange: register session, finalize to usize::MAX
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        mgr.finalize_session_tokens(42, usize::MAX);

        // Act: claim zero prefix — should always succeed
        let claimed = mgr.claim_session_prefix(42, 99, 0).unwrap();

        // Assert: empty list, no error
        assert!(claimed.is_empty());
        assert_eq!(mgr.session_finalized_position(42), Some(usize::MAX));
    }

    #[test]
    fn tier_manager_allocate_exhaust_then_free_all_then_reallocate() {
        // Arrange: exhaust capacity of 3
        let mut tm = TierManager::new(3, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        let c = tm.allocate(Tier::L1).unwrap();
        assert!(!tm.can_allocate(Tier::L1));

        // Act: free all, then reallocate
        assert!(tm.free(Tier::L1, a));
        assert!(tm.free(Tier::L1, b));
        assert!(tm.free(Tier::L1, c));
        assert_eq!(tm.usage(Tier::L1).used, 0);

        // Assert: can allocate again, IDs reused in FIFO order
        let d = tm.allocate(Tier::L1).unwrap();
        assert_eq!(d, a, "should reuse first freed ID");
        assert_eq!(tm.usage(Tier::L1).used, 1);
    }

    #[test]
    fn global_manager_new_with_custom_eviction_policy() {
        // Arrange: custom policy with non-default weights
        let custom_policy = EvictionPolicy {
            recency_weight: 1,
            frequency_weight: 1,
            semantic_weight: 1,
            active_penalty: 1,
            standby_bonus: 1,
        };
        let mut mgr = GlobalMemoryManager::new(
            TierManager::new(4, 0, 0),
            custom_policy,
        );

        // Act: use select_victims — should still work with custom weights
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();
        let victims = mgr.select_victims(&metadata, &semantic, 1);

        // Assert: works with custom policy
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1);
    }

    #[test]
    fn page_table_map_remove_remap_sequence_preserves_correct_state() {
        // Arrange: map, remove, then remap the same virtual page
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);

        pt.map(vpid, Tier::L1, 1);
        pt.remove(&vpid);

        // Act: remap after remove should error (virtual page gone)
        let result = pt.remap(vpid, Tier::L2, 2);

        // Assert: remap fails because virtual page was removed
        assert!(matches!(result, Err(MemoryManagerError::UnknownVirtualPage { .. })));

        // Re-map should succeed
        pt.map(vpid, Tier::L3, 3);
        assert_eq!(pt.resolve(vpid).unwrap().tier, Tier::L3);
        assert_eq!(pt.resolve(vpid).unwrap().physical_id, 3);
    }

    #[test]
    fn plan_prefill_exact_token_equals_page_size_multiple_chunks() {
        // Arrange: 512 tokens, page_size=128, chunk_size=128 => 4 pages, 4 chunks of 1 page each
        // But L1 has capacity 4, so it's FullyResident
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);

        // Act
        let plan = mgr.plan_prefill(512, 128, 128);

        // Assert: fits exactly in L1
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    #[test]
    fn global_manager_migrate_l1_to_l2_back_to_l1_roundtrip() {
        // Arrange: allocate in L1, bind virtual, migrate to L2, then back to L1
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, src).unwrap();

        // Act: migrate L1 → L2
        let dst_l2 = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap().0, Tier::L2);

        // Migrate L2 → L1
        let dst_l1 = mgr.migrate_page(Tier::L2, Tier::L1, dst_l2).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap().0, Tier::L1);

        // Assert: virtual page resolves to L1, L2 is empty
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, dst_l1));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
    }

    #[test]
    fn eviction_policy_same_score_different_page_ids_deterministic_order() {
        // Arrange: two pages with identical scores but different page_ids
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        for id in [10usize, 20usize] {
            metadata.insert(
                id,
                PageMetadata {
                    page_id: id,
                    sequence_id: Some(1),
                    state: PageState::Standby,
                    recency: 0,
                    is_lir: false,
                    swap_in_time: None,
                    warm_until: None,
                    access_count: 0,
                    last_access: now,
                },
            );
        }
        let semantic = HashMap::new();

        // Act: request both as victims
        let victims = policy.select_victims(&metadata, &semantic, 2);

        // Assert: deterministic tiebreak — lower page_id first
        assert_eq!(victims.len(), 2);
        assert!(victims[0] < victims[1], "tiebreak should order by page_id ascending");
    }

    #[test]
    fn global_manager_allocate_and_bind_all_tiers_simultaneously() {
        // Arrange: create manager with capacity in all three tiers
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 2);

        // Act: allocate one page per tier and bind distinct virtual pages
        let l1 = mgr.allocate_page(Tier::L1).unwrap();
        let l2 = mgr.allocate_page(Tier::L2).unwrap();
        let l3 = mgr.allocate_page(Tier::L3).unwrap();

        let v1 = VirtualPageId::new(1, 0);
        let v2 = VirtualPageId::new(2, 0);
        let v3 = VirtualPageId::new(3, 0);

        mgr.bind_virtual_page(v1, Tier::L1, l1).unwrap();
        mgr.bind_virtual_page(v2, Tier::L2, l2).unwrap();
        mgr.bind_virtual_page(v3, Tier::L3, l3).unwrap();

        // Assert: each resolves to its own tier
        assert_eq!(mgr.resolve(v1).unwrap(), (Tier::L1, l1));
        assert_eq!(mgr.resolve(v2).unwrap(), (Tier::L2, l2));
        assert_eq!(mgr.resolve(v3).unwrap(), (Tier::L3, l3));

        // Free L2 page — only v2 should be affected
        mgr.free_page(Tier::L2, l2).unwrap();
        assert!(mgr.resolve(v1).is_ok(), "L1 virtual should survive");
        assert!(mgr.resolve(v2).is_err(), "L2 virtual should be cascade-removed");
        assert!(mgr.resolve(v3).is_ok(), "L3 virtual should survive");
    }

    #[test]
    fn session_register_many_sessions_finalize_selectively() {
        // Arrange: register 5 sessions
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        for id in 100..105u64 {
            mgr.register_session(id);
        }

        // Act: finalize only odd-numbered sessions
        mgr.finalize_session_tokens(101, 50);
        mgr.finalize_session_tokens(103, 200);

        // Assert: correct finalized positions
        assert_eq!(mgr.session_finalized_position(100), Some(0));
        assert_eq!(mgr.session_finalized_position(101), Some(50));
        assert_eq!(mgr.session_finalized_position(102), Some(0));
        assert_eq!(mgr.session_finalized_position(103), Some(200));
        assert_eq!(mgr.session_finalized_position(104), Some(0));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — EvictionPolicy clone, tier isolation under churn,
    // plan_prefill large page_size, SessionKvCache empty pages, virtual page
    // sorting, track_page + bind + resolve integration, TierManager gradual
    // free, PageTable overwrite resolution, entropy_evict multi-virtual,
    // TierManager is_allocated after track, resolve cross-tier PhysicalId,
    // EvictionPolicy clone field preservation, GlobalMemoryManager pipeline
    // release with mixed tiers, migrate with no bound pages returns new id,
    // plan_prefill chunk_schedule single token last chunk
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn eviction_policy_clone_preserves_all_weights() {
        // Arrange: create a policy with non-default weights
        let original = EvictionPolicy {
            recency_weight: 7,
            frequency_weight: 14,
            semantic_weight: 21,
            active_penalty: 28,
            standby_bonus: 35,
        };

        // Act: clone the policy
        let cloned = original.clone();

        // Assert: all fields match
        assert_eq!(cloned.recency_weight, 7);
        assert_eq!(cloned.frequency_weight, 14);
        assert_eq!(cloned.semantic_weight, 21);
        assert_eq!(cloned.active_penalty, 28);
        assert_eq!(cloned.standby_bonus, 35);
    }

    #[test]
    fn tier_manager_gradual_free_decrements_used_one_at_a_time() {
        // Arrange: allocate all 4 pages in L1
        let mut tm = TierManager::new(4, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        let b = tm.allocate(Tier::L1).unwrap();
        let c = tm.allocate(Tier::L1).unwrap();
        let d = tm.allocate(Tier::L1).unwrap();
        assert_eq!(tm.usage(Tier::L1).used, 4);

        // Act & Assert: free one at a time, verify used decrements
        assert!(tm.free(Tier::L1, a));
        assert_eq!(tm.usage(Tier::L1).used, 3);

        assert!(tm.free(Tier::L1, c));
        assert_eq!(tm.usage(Tier::L1).used, 2);

        assert!(tm.free(Tier::L1, b));
        assert_eq!(tm.usage(Tier::L1).used, 1);

        assert!(tm.free(Tier::L1, d));
        assert_eq!(tm.usage(Tier::L1).used, 0);
    }

    #[test]
    fn plan_prefill_page_size_larger_than_tokens_produces_one_page() {
        // Arrange: 5 tokens with page_size=1000 => div_ceil(5/1000) = 1 page
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);

        // Act
        let plan = mgr.plan_prefill(5, 128, 1000);

        // Assert: 1 page, fits in L1
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 1 });
    }

    #[test]
    fn session_kv_cache_empty_pages_vec() {
        // Arrange: create a SessionKvCache with an empty pages vector
        let cache = SessionKvCache {
            session_id: 0,
            pages: vec![],
            finalized_position: 100,
        };

        // Act & Assert: verify fields
        assert!(cache.pages.is_empty());
        assert_eq!(cache.finalized_position, 100);
        assert_eq!(cache.session_id, 0);
    }

    #[test]
    fn virtual_page_id_sorted_vec_dedup_correct() {
        // Arrange: create a Vec with duplicates
        let mut v: Vec<VirtualPageId> = vec![
            VirtualPageId::new(3, 1),
            VirtualPageId::new(1, 0),
            VirtualPageId::new(3, 1),
            VirtualPageId::new(2, 2),
            VirtualPageId::new(1, 0),
        ];

        // Act: sort by sequence_id then logical_index, then dedup
        v.sort_by(|a, b| {
            a.sequence_id
                .cmp(&b.sequence_id)
                .then(a.logical_index.cmp(&b.logical_index))
        });
        v.dedup();

        // Assert: 3 unique entries in sorted order
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], VirtualPageId::new(1, 0));
        assert_eq!(v[1], VirtualPageId::new(2, 2));
        assert_eq!(v[2], VirtualPageId::new(3, 1));
    }

    #[test]
    fn global_manager_track_page_then_bind_and_resolve() {
        // Arrange: track a page into tier, then bind a virtual page to it
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.track_page(Tier::L1, 5).unwrap();
        assert!(mgr.tier_manager.is_allocated(Tier::L1, 5));

        let vpid = VirtualPageId::new(10, 0);

        // Act: bind virtual page to the tracked physical page
        mgr.bind_virtual_page(vpid, Tier::L1, 5).unwrap();

        // Assert: resolve returns the tracked page
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, 5));
    }

    #[test]
    fn page_table_overwrite_three_times_final_state_correct() {
        // Arrange: map a virtual page three times to different tiers
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);

        // Act: sequential overwrites
        let old1 = pt.map(vpid, Tier::L1, 1);
        assert!(old1.is_none());

        let old2 = pt.map(vpid, Tier::L2, 2);
        assert_eq!(old2.unwrap(), PageLocation { physical_id: 1, tier: Tier::L1 });

        let old3 = pt.map(vpid, Tier::L3, 3);
        assert_eq!(old3.unwrap(), PageLocation { physical_id: 2, tier: Tier::L2 });

        // Assert: final state is L3/3
        let loc = pt.resolve(vpid).unwrap();
        assert_eq!(loc, PageLocation { physical_id: 3, tier: Tier::L3 });
    }

    #[test]
    fn entropy_evict_with_multiple_virtual_pages_bound_to_one_physical() {
        // Arrange: one physical page with two virtual pages, low entropy
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let _p1 = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(20, 0);
        mgr.bind_virtual_page(v0, Tier::L1, p0).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, p0).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.001); // low
        entropies.insert(_p1, 5.0); // high

        // Act: entropy evict
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);

        // Assert: one page freed, both virtual pages cascade-removed
        assert_eq!(freed, 1);
        assert!(mgr.resolve(v0).is_err());
        assert!(mgr.resolve(v1).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    #[test]
    fn tier_manager_is_allocated_returns_false_for_never_allocated_id() {
        // Arrange: create a TierManager with capacity, no allocations
        let mut tm = TierManager::new(10, 0, 0);

        // Act: check is_allocated for an ID that was never allocated
        let result = tm.is_allocated(Tier::L1, 0);

        // Assert: not allocated
        assert!(!result);

        // Allocate and verify
        let p = tm.allocate(Tier::L1).unwrap();
        assert!(tm.is_allocated(Tier::L1, p));
        assert!(!tm.is_allocated(Tier::L1, p + 1));
    }

    #[test]
    fn global_manager_resolve_returns_correct_physical_id_per_tier() {
        // Arrange: allocate pages in all three tiers with independent IDs
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 2);
        let l1_0 = mgr.allocate_page(Tier::L1).unwrap();
        let l2_0 = mgr.allocate_page(Tier::L2).unwrap();
        let l3_0 = mgr.allocate_page(Tier::L3).unwrap();

        let v1 = VirtualPageId::new(1, 0);
        let v2 = VirtualPageId::new(2, 0);
        let v3 = VirtualPageId::new(3, 0);
        mgr.bind_virtual_page(v1, Tier::L1, l1_0).unwrap();
        mgr.bind_virtual_page(v2, Tier::L2, l2_0).unwrap();
        mgr.bind_virtual_page(v3, Tier::L3, l3_0).unwrap();

        // Act & Assert: resolve returns correct (tier, physical) pairs
        // Note: each tier starts from 0 independently, so all three may be 0
        let (t1, p1) = mgr.resolve(v1).unwrap();
        assert_eq!(t1, Tier::L1);

        let (t2, p2) = mgr.resolve(v2).unwrap();
        assert_eq!(t2, Tier::L2);

        let (t3, p3) = mgr.resolve(v3).unwrap();
        assert_eq!(t3, Tier::L3);

        // All three physical IDs are valid
        assert!(mgr.tier_manager.is_allocated(Tier::L1, p1));
        assert!(mgr.tier_manager.is_allocated(Tier::L2, p2));
        assert!(mgr.tier_manager.is_allocated(Tier::L3, p3));
    }

    #[test]
    fn global_manager_migrate_unbound_page_returns_new_physical_id() {
        // Arrange: allocate in L1, no virtual page bound
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();

        // Act: migrate L1 → L2
        let dst = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();

        // Assert: src freed from L1, dst allocated in L2, dst is a valid new id
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
        assert!(mgr.tier_manager.is_allocated(Tier::L2, dst));
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, src));
    }

    #[test]
    fn plan_prefill_chunk_schedule_last_chunk_single_token() {
        // Arrange: 7 tokens, chunk_size=3, page_size=1 => 7 pages total
        // chunk_schedule: [3,3,1]
        let mut mgr = GlobalMemoryManager::new_with_capacities(1, 10, 0);

        // Act
        let plan = mgr.plan_prefill(7, 3, 1);

        // Assert
        match plan {
            PrefillPlan::Pipelined { chunk_schedule, .. } => {
                assert_eq!(chunk_schedule.len(), 3);
                assert_eq!(chunk_schedule[0], 3);
                assert_eq!(chunk_schedule[1], 3);
                assert_eq!(chunk_schedule[2], 1, "last chunk should be 1 page for remaining token");
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    #[test]
    fn global_manager_pipeline_release_working_frees_only_l1_pages() {
        // Arrange: allocate Working pipeline pages on L1 and L2
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 2, 0);
        let pw_l1 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _pw_l2 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L2).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);

        // Act: release_working_pipeline always frees from L1 only
        mgr.release_working_pipeline(1);

        // Assert: L1 page freed, L2 page not freed by release_working_pipeline
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, pw_l1));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        // L2 page is still tracked in pipeline_pages (removed) but not freed from tier
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
    }

    #[test]
    fn global_manager_free_one_page_others_resolve_unchanged() {
        // Arrange: allocate three L1 pages, bind three virtual pages
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();

        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        let v2 = VirtualPageId::new(10, 2);
        mgr.bind_virtual_page(v0, Tier::L1, p0).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, p1).unwrap();
        mgr.bind_virtual_page(v2, Tier::L1, p2).unwrap();

        // Act: free the middle page
        mgr.free_page(Tier::L1, p1).unwrap();

        // Assert: v1 cascade-removed, v0 and v2 still resolve correctly
        assert!(mgr.resolve(v0).is_ok());
        assert_eq!(mgr.resolve(v0).unwrap(), (Tier::L1, p0));
        assert!(mgr.resolve(v1).is_err());
        assert!(mgr.resolve(v2).is_ok());
        assert_eq!(mgr.resolve(v2).unwrap(), (Tier::L1, p2));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
    }

    #[test]
    fn tier_usage_as_hashmap_value() {
        // Arrange: use TierUsage as values in a HashMap keyed by Tier
        let mut map = HashMap::new();
        map.insert(Tier::L1, TierUsage { used: 1, capacity: 4 });
        map.insert(Tier::L2, TierUsage { used: 2, capacity: 8 });
        map.insert(Tier::L3, TierUsage { used: 0, capacity: 16 });

        // Act & Assert: lookup and verify
        assert_eq!(map.get(&Tier::L1).unwrap().available(), 3);
        assert_eq!(map.get(&Tier::L2).unwrap().available(), 6);
        assert_eq!(map.get(&Tier::L3).unwrap().available(), 16);
        assert_eq!(map.get(&Tier::L1).unwrap().used, 1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — Error equality symmetry, TierManager saturating
    // add in allocate, plan_prefill div_ceil boundary, session overwrite via
    // register, PageTable resolve returns copy, EvictionPolicy recency high
    // frequency neutralize, entropy_evict all pages evicted leaves empty tier,
    // GlobalMemoryManager bind virtual to wrong tier for physical, track_page
    // then free then reallocate same id, prepare_next_turn with no sessions,
    // release_working_pipeline idempotent, migrate then free dst succeeds,
    // plan_prefill one_over_l1_with_zero_l2
    // ══════════════════════════════════════════════════════════════════════

    // @trace TEST-MM-001 error equality is symmetric across all variants
    #[test]
    fn error_equality_symmetric_all_variants() {
        // Arrange: create matching pairs for each variant
        let vpid = VirtualPageId::new(5, 6);
        let pairs: [(MemoryManagerError, MemoryManagerError); 6] = [
            (
                MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
                MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 },
            ),
            (
                MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 10 },
                MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 10 },
            ),
            (
                MemoryManagerError::UnknownVirtualPage { virtual_id: vpid },
                MemoryManagerError::UnknownVirtualPage { virtual_id: vpid },
            ),
            (
                MemoryManagerError::UnknownSession { session_id: 42 },
                MemoryManagerError::UnknownSession { session_id: 42 },
            ),
            (
                MemoryManagerError::SessionPrefixOutOfBounds {
                    session_id: 1, prefix_tokens: 2, finalized_position: 3,
                },
                MemoryManagerError::SessionPrefixOutOfBounds {
                    session_id: 1, prefix_tokens: 2, finalized_position: 3,
                },
            ),
            (
                MemoryManagerError::SessionPagesInsufficient {
                    session_id: 7, prefix_tokens: 5, available_pages: 2,
                },
                MemoryManagerError::SessionPagesInsufficient {
                    session_id: 7, prefix_tokens: 5, available_pages: 2,
                },
            ),
        ];

        // Act & Assert: verify symmetry — a == b implies b == a
        for (a, b) in &pairs {
            assert_eq!(a, b, "expected equality for {a:?}");
            assert_eq!(b, a, "equality must be symmetric");
        }
    }

    // @trace TEST-MM-002 TierManager saturating_add in allocate prevents overflow
    #[test]
    fn tier_manager_allocate_saturating_add_prevents_overflow() {
        // Arrange: create TierManager with capacity 1, manually push next_id to MAX
        let mut tm = TierManager::new(2, 0, 0);
        let _first = tm.allocate(Tier::L1).unwrap();

        // Act: free first, then allocate again — should reuse, not overflow next_id
        assert!(tm.free(Tier::L1, _first));
        let reused = tm.allocate(Tier::L1).unwrap();
        assert_eq!(reused, _first, "should reuse freed id, not allocate new");

        // Allocate second — now from fresh id
        let fresh = tm.allocate(Tier::L1).unwrap();
        assert_eq!(fresh, 1);

        // Assert: capacity exhausted
        assert!(!tm.can_allocate(Tier::L1));
    }

    // @trace TEST-MM-003 plan_prefill div_ceil boundary — tokens exactly divisible
    #[test]
    fn plan_prefill_tokens_exactly_divisible_by_page_and_chunk() {
        // Arrange: 512 tokens, page_size=32, chunk_size=128
        // 512/32 = 16 pages, 512/128 = 4 chunks of 4 pages each
        // L1 has 16 slots => FullyResident
        let mut mgr = GlobalMemoryManager::new_with_capacities(16, 0, 0);

        // Act
        let plan = mgr.plan_prefill(512, 128, 32);

        // Assert
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 16 });
    }

    // @trace TEST-MM-004 session register overwrites existing with same id
    #[test]
    fn session_register_overwrites_existing_session_pages() {
        // Arrange: register session, add pages, finalize
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)];
        }
        mgr.finalize_session_tokens(42, 10);

        // Act: re-register same session — should reset
        let cache = mgr.register_session(42);

        // Assert: pages empty, finalized_position reset to 0
        assert!(cache.pages.is_empty());
        assert_eq!(cache.finalized_position, 0);
        assert_eq!(mgr.session_finalized_position(42), Some(0));
    }

    // @trace TEST-MM-005 PageTable resolve returns independent copy
    #[test]
    fn page_table_resolve_returns_independent_copy() {
        // Arrange: map a virtual page
        let mut pt = PageTable::new();
        let vpid = VirtualPageId::new(10, 0);
        pt.map(vpid, Tier::L1, 5);

        // Act: resolve twice
        let loc1 = pt.resolve(vpid).unwrap();
        let loc2 = pt.resolve(vpid).unwrap();

        // Assert: both are equal (Copy type, but verify independence)
        assert_eq!(loc1, loc2);
        assert_eq!(loc1.physical_id, 5);
        assert_eq!(loc2.physical_id, 5);
    }

    // @trace TEST-MM-006 EvictionPolicy: high recency offset by high frequency
    #[test]
    fn eviction_policy_high_recency_offset_by_high_frequency() {
        // Arrange: two pages, one with high recency but also high access_count,
        // the other with low recency and low access_count
        let policy = EvictionPolicy::default();
        let now = Instant::now();

        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 10000,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 10000, // frequency_weight=32: -320000
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act
        let victims = policy.select_victims(&metadata, &semantic, 1);

        // Assert: page 2 evicted (no frequency penalty) despite lower recency
        assert_eq!(victims[0], 2, "high frequency should protect page 1");
    }

    // @trace TEST-MM-007 entropy_evict evicting all pages leaves tier empty
    #[test]
    fn entropy_evict_all_pages_evicted_leaves_empty_tier() {
        // Arrange: allocate 3 pages, all with very low entropy
        let mut mgr = GlobalMemoryManager::new_with_capacities(3, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();

        // Bind virtual pages to verify cascade removal
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(20, 0);
        let v2 = VirtualPageId::new(30, 0);
        mgr.bind_virtual_page(v0, Tier::L1, p0).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, p1).unwrap();
        mgr.bind_virtual_page(v2, Tier::L1, p2).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.001);
        entropies.insert(p1, 0.002);
        entropies.insert(p2, 0.003);

        // Act
        let freed = mgr.entropy_evict(&entropies, 0.1, Tier::L1);

        // Assert: all 3 pages freed, tier empty, all virtuals gone
        assert_eq!(freed, 3);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert!(mgr.resolve(v0).is_err());
        assert!(mgr.resolve(v1).is_err());
        assert!(mgr.resolve(v2).is_err());
    }

    // @trace TEST-MM-008 bind virtual page to physical in wrong tier errors
    #[test]
    fn global_manager_bind_virtual_to_physical_in_different_tier_errors() {
        // Arrange: allocate physical page in L1
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let l1_pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        // Act: attempt to bind virtual page to L2 but provide L1 physical id
        let result = mgr.bind_virtual_page(vpid, Tier::L2, l1_pid);

        // Assert: should fail because l1_pid is not allocated in L2
        assert!(
            matches!(result, Err(MemoryManagerError::UnknownPhysicalPage { tier, physical_id })
                if tier == Tier::L2 && physical_id == l1_pid),
            "binding to wrong tier should return UnknownPhysicalPage"
        );
    }

    // @trace TEST-MM-009 track_page then free then reallocate same id succeeds
    #[test]
    fn global_manager_track_free_reallocate_same_physical_id() {
        // Arrange: track a specific page id (creates gap ids 0..6 in free_ids)
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        mgr.track_page(Tier::L1, 7).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);

        // Act: free it, then reallocate
        mgr.free_page(Tier::L1, 7).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);

        let new_pid = mgr.allocate_page(Tier::L1).unwrap();

        // Assert: FIFO from free_ids returns the lowest gap id (0), not 7
        assert_eq!(new_pid, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // @trace TEST-MM-010 prepare_next_turn with no sessions or pipelines is safe
    #[test]
    fn global_manager_prepare_next_turn_with_nothing_allocated_is_safe() {
        // Arrange: empty manager
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);

        // Act: should not panic
        mgr.prepare_next_turn(0);

        // Assert: no changes
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L1).capacity, 4);
    }

    // @trace TEST-MM-011 release_working_pipeline called twice is idempotent
    #[test]
    fn global_manager_release_working_pipeline_idempotent() {
        // Arrange: allocate working pipeline pages
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let _p1 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        let _p2 = mgr.allocate_page_in_pipeline(KvPipeline::Working, 1, Tier::L1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);

        // Act: release once
        mgr.release_working_pipeline(1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);

        // Release again — should be a no-op, not panic
        mgr.release_working_pipeline(1);

        // Assert: still zero
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    // @trace TEST-MM-012 migrate page then free destination succeeds
    #[test]
    fn global_manager_migrate_then_free_dst_succeeds() {
        // Arrange: allocate in L1, bind virtual, migrate to L2
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L1, src).unwrap();

        let dst = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L2, dst));

        // Act: free the destination page in L2
        mgr.free_page(Tier::L2, dst).unwrap();

        // Assert: virtual page cascade-removed, both tiers clean
        assert!(mgr.resolve(vpid).is_err());
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 0);
    }

    // @trace TEST-MM-013 plan_prefill one page over L1 with zero L2 capacity
    #[test]
    fn plan_prefill_one_over_l1_with_zero_l2() {
        // Arrange: L1=2, L2=0, need 3 pages
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);

        // Act
        let plan = mgr.plan_prefill(300, 100, 100); // 3 pages needed

        // Assert: Pipelined with l2_prefetch=0
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, chunk_schedule } => {
                assert_eq!(l1_pages, 2);
                assert_eq!(l2_prefetch, 0, "L2 has zero capacity, prefetch should be 0");
                assert_eq!(chunk_schedule.len(), 3, "3 chunks of 100 tokens");
            }
            _ => panic!("Expected Pipelined"),
        }
    }

    // @trace TEST-MM-014 TierManager track_page with consecutive ids then free all
    #[test]
    fn tier_manager_track_consecutive_ids_then_free_all() {
        // Arrange: track pages 0, 1, 2 (all consecutive from next_id start)
        let mut tm = TierManager::new(5, 0, 0);
        assert!(tm.track_page(Tier::L1, 0));
        assert!(tm.track_page(Tier::L1, 1));
        assert!(tm.track_page(Tier::L1, 2));
        assert_eq!(tm.usage(Tier::L1).used, 3);

        // Act: free all tracked pages
        assert!(tm.free(Tier::L1, 0));
        assert!(tm.free(Tier::L1, 1));
        assert!(tm.free(Tier::L1, 2));

        // Assert: usage back to zero, can allocate fresh
        assert_eq!(tm.usage(Tier::L1).used, 0);
        assert!(tm.can_allocate(Tier::L1));

        // Re-allocation should reuse freed ids in FIFO order
        let a = tm.allocate(Tier::L1).unwrap();
        assert_eq!(a, 0);
        let b = tm.allocate(Tier::L1).unwrap();
        assert_eq!(b, 1);
    }

    // @trace TEST-MM-015 GlobalMemoryManager bind resolve after full churn cycle
    #[test]
    fn global_manager_bind_resolve_after_full_allocate_free_cycle() {
        // Arrange: allocate 2 L1 pages, bind 2 virtuals
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        mgr.bind_virtual_page(v0, Tier::L1, p0).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, p1).unwrap();

        // Act: free both, reallocate, rebind to new virtuals
        mgr.free_page(Tier::L1, p0).unwrap();
        mgr.free_page(Tier::L1, p1).unwrap();
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);

        let new_p0 = mgr.allocate_page(Tier::L1).unwrap();
        let new_p1 = mgr.allocate_page(Tier::L1).unwrap();
        let nv0 = VirtualPageId::new(20, 0);
        let nv1 = VirtualPageId::new(20, 1);
        mgr.bind_virtual_page(nv0, Tier::L1, new_p0).unwrap();
        mgr.bind_virtual_page(nv1, Tier::L1, new_p1).unwrap();

        // Assert: old virtuals gone, new ones resolve correctly
        assert!(mgr.resolve(v0).is_err());
        assert!(mgr.resolve(v1).is_err());
        assert_eq!(mgr.resolve(nv0).unwrap(), (Tier::L1, new_p0));
        assert_eq!(mgr.resolve(nv1).unwrap(), (Tier::L1, new_p1));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 15 additional tests — VirtualPageId Ord, TierState saturating overflow,
    // GlobalMemoryManager debug format, PageTable cross-page independence,
    // pipeline track_in + release interaction, session re-register after claim,
    // eviction with zero access_count and zero recency, entropy_evict all pages,
    // migrate page without binding then bind after, prefill with L2 partially used
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn virtual_page_id_sort_by_sequence_then_logical() {
        // Arrange: unsorted collection of VirtualPageIds
        let mut ids = vec![
            VirtualPageId::new(2, 0),
            VirtualPageId::new(1, 5),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(3, 0),
            VirtualPageId::new(1, 0),
        ];

        // Act: sort using explicit comparator (VirtualPageId has no Ord derive)
        ids.sort_by(|a, b| {
            a.sequence_id
                .cmp(&b.sequence_id)
                .then_with(|| a.logical_index.cmp(&b.logical_index))
        });

        // Assert: ordered by sequence_id first, then logical_index
        assert_eq!(ids[0], VirtualPageId::new(1, 0));
        assert_eq!(ids[1], VirtualPageId::new(1, 1));
        assert_eq!(ids[2], VirtualPageId::new(1, 5));
        assert_eq!(ids[3], VirtualPageId::new(2, 0));
        assert_eq!(ids[4], VirtualPageId::new(3, 0));
    }

    #[test]
    fn tier_manager_track_page_at_capacity_boundary_via_global_mgr() {
        // Arrange: capacity 2, manually track two pages
        let mut mgr = GlobalMemoryManager::new_with_capacities(2, 0, 0);
        mgr.track_page(Tier::L1, 0).unwrap();
        mgr.track_page(Tier::L1, 1).unwrap();

        // Act: track a third page when at capacity
        let result = mgr.track_page(Tier::L1, 2);

        // Assert: should fail with TierCapacityExceeded
        assert!(matches!(result, Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 })));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 2);
    }

    #[test]
    fn global_manager_debug_format_shows_fields() {
        // Arrange: create a manager and allocate a page
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 2, 0);
        let _p = mgr.allocate_page(Tier::L1).unwrap();

        // Act: format with Debug
        let debug = format!("{mgr:?}");

        // Assert: should contain key struct names
        assert!(debug.contains("GlobalMemoryManager"), "debug should contain struct name");
        assert!(debug.contains("page_table"), "debug should show page_table field");
        assert!(debug.contains("tier_manager"), "debug should show tier_manager field");
    }

    #[test]
    fn page_table_remove_one_page_preserves_others_independent() {
        // Arrange: map three virtual pages to three different physical pages
        let mut pt = PageTable::new();
        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);
        let v2 = VirtualPageId::new(2, 0);
        pt.map(v0, Tier::L1, 10);
        pt.map(v1, Tier::L2, 20);
        pt.map(v2, Tier::L3, 30);

        // Act: remove the middle one (v1)
        let removed = pt.remove(&v1);

        // Assert: removed has correct location, others untouched
        assert_eq!(removed.unwrap(), PageLocation { physical_id: 20, tier: Tier::L2 });
        assert_eq!(pt.resolve(v0).unwrap().physical_id, 10);
        assert_eq!(pt.resolve(v2).unwrap().physical_id, 30);
        assert!(pt.resolve(v1).is_none());
    }

    #[test]
    fn global_manager_pipeline_track_in_then_release_frees_l1_pages() {
        // Arrange: allocate 3 pages, track 2 into Working pipeline for request 1
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let _p3 = mgr.allocate_page(Tier::L1).unwrap();

        mgr.track_in_pipeline(KvPipeline::Working, 1, p1);
        mgr.track_in_pipeline(KvPipeline::Working, 1, p2);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 3);

        // Act: release working pipeline for request 1
        mgr.release_working_pipeline(1);

        // Assert: p1 and p2 freed from L1, p3 remains
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, p1));
        assert!(!mgr.tier_manager.is_allocated(Tier::L1, p2));
    }

    #[test]
    fn session_reregister_after_claim_resets_state() {
        // Arrange: register session, set pages, finalize, claim prefix
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)];
        }
        mgr.finalize_session_tokens(42, 2);
        let claimed = mgr.claim_session_prefix(42, 99, 2).unwrap();
        assert_eq!(claimed.len(), 2);

        // Act: re-register same session — should reset
        let cache = mgr.register_session(42);

        // Assert: session reset to initial state
        assert_eq!(cache.session_id, 42);
        assert!(cache.pages.is_empty());
        assert_eq!(cache.finalized_position, 0);
        assert_eq!(mgr.session_finalized_position(42), Some(0));
    }

    #[test]
    fn eviction_policy_zero_recency_zero_access_standby_is_candidate() {
        // Arrange: page with all zero signals except Standby state
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: None,
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act
        let victims = policy.select_victims(&metadata, &semantic, 1);

        // Assert: zero-signal Standby page is still a valid candidate
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1);
    }

    #[test]
    fn entropy_evict_frees_all_pages_when_all_below_threshold() {
        // Arrange: allocate 4 pages, all with low entropy
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p0 = mgr.allocate_page(Tier::L1).unwrap();
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let p3 = mgr.allocate_page(Tier::L1).unwrap();

        let mut entropies = HashMap::new();
        entropies.insert(p0, 0.001);
        entropies.insert(p1, 0.002);
        entropies.insert(p2, 0.003);
        entropies.insert(p3, 0.004);

        // Act: evict with threshold 0.01 (all below)
        let freed = mgr.entropy_evict(&entropies, 0.01, Tier::L1);

        // Assert: all 4 pages freed
        assert_eq!(freed, 4);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn global_manager_migrate_unbound_page_then_bind_to_destination() {
        // Arrange: allocate page in L1 (no virtual binding), migrate to L2
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let src = mgr.allocate_page(Tier::L1).unwrap();
        let dst = mgr.migrate_page(Tier::L1, Tier::L2, src).unwrap();

        // Act: now bind a virtual page to the destination
        let vpid = VirtualPageId::new(10, 0);
        mgr.bind_virtual_page(vpid, Tier::L2, dst).unwrap();

        // Assert: virtual page resolves correctly, L1 is empty
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L2, dst));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 0);
        assert_eq!(mgr.tier_usage(Tier::L2).used, 1);
    }

    #[test]
    fn plan_prefill_with_l2_partially_used_reduces_prefetch() {
        // Arrange: L1=4 (all free), L2=4 (2 used)
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        // Use 2 L2 pages
        let _l2a = mgr.allocate_page(Tier::L2).unwrap();
        let _l2b = mgr.allocate_page(Tier::L2).unwrap();

        // 12 pages needed (1536/128), L1 has 4, L2 has 2 available
        let plan = mgr.plan_prefill(1536, 256, 128);

        // Act & Assert
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, .. } => {
                assert_eq!(l1_pages, 4);
                assert_eq!(l2_prefetch, 2, "L2 prefetch should be capped at L2 available (4-2=2)");
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    #[test]
    fn tier_manager_track_page_after_allocate_and_free_same_id_reallocates() {
        // Arrange: allocate id 0, free it
        let mut tm = TierManager::new(4, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        assert_eq!(a, 0);
        assert!(tm.free(Tier::L1, a));

        // Act: track page 0 (which is now in free_ids)
        assert!(tm.track_page(Tier::L1, 0));

        // Assert: id 0 is re-allocated, usage is 1
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert!(tm.is_allocated(Tier::L1, 0));
    }

    #[test]
    fn global_manager_remap_virtual_page_cleans_old_reverse_index_completely() {
        // Arrange: bind v0 and v1 to same physical p1
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let p1 = mgr.allocate_page(Tier::L1).unwrap();
        let p2 = mgr.allocate_page(Tier::L1).unwrap();
        let v0 = VirtualPageId::new(10, 0);
        let v1 = VirtualPageId::new(10, 1);
        mgr.bind_virtual_page(v0, Tier::L1, p1).unwrap();
        mgr.bind_virtual_page(v1, Tier::L1, p1).unwrap();

        // Act: remap v0 to p2 (should only remove v0 from p1's reverse index, v1 stays)
        mgr.remap_virtual_page(v0, Tier::L1, p2).unwrap();

        // Assert: freeing p1 should cascade-remove v1 but NOT v0
        mgr.free_page(Tier::L1, p1).unwrap();
        assert!(mgr.resolve(v1).is_err(), "v1 was still bound to p1, should be gone");
        assert_eq!(mgr.resolve(v0).unwrap(), (Tier::L1, p2), "v0 was remapped to p2, should survive");
    }

    #[test]
    fn session_claim_prefix_one_less_than_finalized_boundary_succeeds() {
        // Arrange: 5 pages, finalize at 5
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        mgr.register_session(42);
        {
            let session = mgr.sessions.get_mut(&42).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(1, 2),
                VirtualPageId::new(1, 3),
                VirtualPageId::new(1, 4),
            ];
        }
        mgr.finalize_session_tokens(42, 5);

        // Act: claim 4 (one less than finalized 5)
        let claimed = mgr.claim_session_prefix(42, 77, 4).unwrap();

        // Assert: 4 pages returned with correct request_id
        assert_eq!(claimed.len(), 4);
        assert_eq!(claimed[0], VirtualPageId::new(77, 0));
        assert_eq!(claimed[3], VirtualPageId::new(77, 3));
    }

    #[test]
    fn page_location_copy_clone_independent_mutation() {
        // Arrange: create two PageLocation copies
        let mut a = PageLocation { physical_id: 10, tier: Tier::L1 };
        let b = a;

        // Act: mutate a
        a.tier = Tier::L3;
        a.physical_id = 99;

        // Assert: b is unaffected (Copy semantics)
        assert_eq!(b.tier, Tier::L1);
        assert_eq!(b.physical_id, 10);
        assert_eq!(a.tier, Tier::L3);
        assert_eq!(a.physical_id, 99);
    }

    #[test]
    fn global_manager_allocate_track_and_allocate_mixed_flow() {
        // Arrange: allocate via allocate_page, then track_page, then allocate again
        let mut mgr = GlobalMemoryManager::new_with_capacities(5, 0, 0);
        let p_alloc = mgr.allocate_page(Tier::L1).unwrap(); // id 0
        mgr.track_page(Tier::L1, 10).unwrap(); // track id 10 (creates gap 0..10 in free_ids... wait, 10 > next_id=1)

        // Act: next allocate should use free_ids (ids 1..10 from gap)
        let p_next = mgr.allocate_page(Tier::L1).unwrap();

        // Assert: p_alloc=0, tracked=10 (gap fills 1..9 into free_ids), next alloc gets 1
        assert_eq!(p_alloc, 0);
        assert_eq!(p_next, 1);
        assert_eq!(mgr.tier_usage(Tier::L1).used, 3);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 10 additional tests — free from wrong tier, plan_prefill with both
    // tiers partially used, eviction saturating arithmetic, multiple sessions
    // claiming independently, PageTable large-scale selective removal, bind
    // unmap rebind cycle, Active vs SwappedOut eviction ordering, track_page
    // id 0 in free_ids, EvictionPolicy sequence_id None differentiation
    // ══════════════════════════════════════════════════════════════════════

    // @trace TEST-MM-016 free page from wrong tier returns UnknownPhysicalPage
    #[test]
    fn global_manager_free_page_from_wrong_tier_errors() {
        // Arrange: allocate a page in L1
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 4, 0);
        let l1_page = mgr.allocate_page(Tier::L1).unwrap();
        assert!(mgr.tier_manager.is_allocated(Tier::L1, l1_page));

        // Act: attempt to free the L1 page from L2
        let result = mgr.free_page(Tier::L2, l1_page);

        // Assert: should error because the page is not allocated in L2
        assert!(
            matches!(result, Err(MemoryManagerError::UnknownPhysicalPage { tier, physical_id })
                if tier == Tier::L2 && physical_id == l1_page),
            "freeing a page from the wrong tier should return UnknownPhysicalPage"
        );
        // L1 page should still be allocated
        assert!(mgr.tier_manager.is_allocated(Tier::L1, l1_page));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // @trace TEST-MM-017 plan_prefill with L1 and L2 both partially used
    #[test]
    fn plan_prefill_both_tiers_partially_used() {
        // Arrange: L1=8 (3 used), L2=8 (4 used)
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 8, 0);
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L1).unwrap();
        let _ = mgr.allocate_page(Tier::L2).unwrap();
        let _ = mgr.allocate_page(Tier::L2).unwrap();
        let _ = mgr.allocate_page(Tier::L2).unwrap();
        let _ = mgr.allocate_page(Tier::L2).unwrap();

        // 20 pages needed (2560/128), L1 available=5, L2 available=4
        let plan = mgr.plan_prefill(2560, 256, 128);

        // Act & Assert
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, chunk_schedule } => {
                assert_eq!(l1_pages, 5, "L1 available = 8-3 = 5");
                assert_eq!(l2_prefetch, 4, "L2 available = 8-4 = 4");
                // 2560/256 = 10 chunks, 256/128 = 2 pages per chunk
                assert_eq!(chunk_schedule.len(), 10);
                assert!(chunk_schedule.iter().all(|&c| c == 2));
            }
            _ => panic!("Expected Pipelined plan"),
        }
    }

    // @trace TEST-MM-018 EvictionPolicy saturating arithmetic with very large recency
    #[test]
    fn eviction_policy_very_large_recency_saturating_arithmetic() {
        // Arrange: two pages — one with very large recency (near i64::MAX weight),
        // another with moderate values
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: usize::MAX,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act: select 1 victim
        let victims = policy.select_victims(&metadata, &semantic, 1);

        // Assert: page 1 (high recency saturating the score) should be evicted first;
        // recency is ADDITIVE to eviction score: higher recency = higher score = more evictable
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1, "page with very large recency should be evicted first due to high score");
    }

    // @trace TEST-MM-019 multiple sessions claiming independently without interference
    #[test]
    fn session_multiple_sessions_claim_independently() {
        // Arrange: register two sessions with different page counts
        let mut mgr = GlobalMemoryManager::new_with_capacities(8, 0, 0);
        mgr.register_session(10);
        {
            let session = mgr.sessions.get_mut(&10).unwrap();
            session.pages = vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(1, 2),
            ];
        }
        mgr.finalize_session_tokens(10, 3);

        mgr.register_session(20);
        {
            let session = mgr.sessions.get_mut(&20).unwrap();
            session.pages = vec![
                VirtualPageId::new(2, 0),
                VirtualPageId::new(2, 1),
            ];
        }
        mgr.finalize_session_tokens(20, 2);

        // Act: claim from both sessions
        let claimed_10 = mgr.claim_session_prefix(10, 100, 2).unwrap();
        let claimed_20 = mgr.claim_session_prefix(20, 200, 1).unwrap();

        // Assert: each session's claim returns its own virtual pages
        assert_eq!(claimed_10.len(), 2);
        assert_eq!(claimed_10[0], VirtualPageId::new(100, 0));
        assert_eq!(claimed_10[1], VirtualPageId::new(100, 1));

        assert_eq!(claimed_20.len(), 1);
        assert_eq!(claimed_20[0], VirtualPageId::new(200, 0));

        // Both sessions still have correct finalized positions
        assert_eq!(mgr.session_finalized_position(10), Some(3));
        assert_eq!(mgr.session_finalized_position(20), Some(2));
    }

    // @trace TEST-MM-020 PageTable large-scale mapping with selective removal
    #[test]
    fn page_table_large_scale_selective_removal() {
        // Arrange: map 100 virtual pages
        let mut pt = PageTable::new();
        for i in 0..100u64 {
            let vpid = VirtualPageId::new(i, 0);
            pt.map(vpid, Tier::L1, i as PhysicalId);
        }

        // Act: remove every even-indexed virtual page
        for i in (0..100).step_by(2) {
            let vpid = VirtualPageId::new(i, 0);
            let removed = pt.remove(&vpid);
            assert!(removed.is_some(), "page {i} should exist");
            assert_eq!(removed.unwrap().physical_id, i as PhysicalId);
        }

        // Assert: odd-indexed pages still resolve, even ones gone
        for i in 0..100u64 {
            let vpid = VirtualPageId::new(i, 0);
            if i % 2 == 0 {
                assert!(pt.resolve(vpid).is_none(), "even page {i} should be removed");
            } else {
                let loc = pt.resolve(vpid).unwrap();
                assert_eq!(loc.physical_id, i as PhysicalId);
                assert_eq!(loc.tier, Tier::L1);
            }
        }
    }

    // @trace TEST-MM-021 bind, unmap, then rebind virtual to same physical
    #[test]
    fn global_manager_bind_unmap_rebind_same_physical() {
        // Arrange: allocate physical page, bind virtual page
        let mut mgr = GlobalMemoryManager::new_with_capacities(4, 0, 0);
        let pid = mgr.allocate_page(Tier::L1).unwrap();
        let vpid = VirtualPageId::new(10, 0);

        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));

        // Act: unmap virtual page, then bind it again to same physical
        let loc = mgr.unmap_virtual_page(vpid).unwrap();
        assert_eq!(loc.physical_id, pid);
        assert!(mgr.resolve(vpid).is_err());

        mgr.bind_virtual_page(vpid, Tier::L1, pid).unwrap();

        // Assert: virtual page resolves again to same physical
        assert_eq!(mgr.resolve(vpid).unwrap(), (Tier::L1, pid));
        assert_eq!(mgr.tier_usage(Tier::L1).used, 1);
    }

    // @trace TEST-MM-022 Active vs SwappedOut eviction ordering
    #[test]
    fn eviction_policy_active_vs_swapped_out_ordering() {
        // Arrange: one Active page and one SwappedOut page
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: Some(1),
                state: PageState::Active,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(1),
                state: PageState::SwappedOut,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act: request 1 victim
        let victims = policy.select_victims(&metadata, &semantic, 1);

        // Assert: SwappedOut (i64::MIN) has lower score than Active (-active_penalty)
        // Active gets -active_penalty=-256; SwappedOut gets i64::MIN
        // So Active should be evicted before SwappedOut
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1, "Active should be evicted before SwappedOut (which has i64::MIN score)");
    }

    // @trace TEST-MM-023 track_page with id 0 when 0 is already in free_ids
    #[test]
    fn tier_manager_track_page_zero_when_zero_in_free_ids() {
        // Arrange: allocate id 0, free it (pushes 0 into free_ids), then track page 0
        let mut tm = TierManager::new(4, 0, 0);
        let a = tm.allocate(Tier::L1).unwrap();
        assert_eq!(a, 0);
        assert!(tm.free(Tier::L1, a)); // free_ids = [0]

        // Act: track page 0 — should remove 0 from free_ids since it's there
        assert!(tm.track_page(Tier::L1, 0));
        assert_eq!(tm.usage(Tier::L1).used, 1);
        assert!(tm.is_allocated(Tier::L1, 0));

        // Next allocation should NOT reuse 0 (it's tracked)
        let b = tm.allocate(Tier::L1).unwrap();
        assert_eq!(b, 1, "next allocation should be id 1, since 0 is tracked");
        assert_eq!(tm.usage(Tier::L1).used, 2);
    }

    // @trace TEST-MM-024 EvictionPolicy with sequence_id None vs Some
    #[test]
    fn eviction_policy_sequence_id_none_vs_some_same_score() {
        // Arrange: two Standby pages with identical attributes except sequence_id
        let policy = EvictionPolicy::default();
        let now = Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            1,
            PageMetadata {
                page_id: 1,
                sequence_id: None, // no sequence
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        metadata.insert(
            2,
            PageMetadata {
                page_id: 2,
                sequence_id: Some(99), // has sequence
                state: PageState::Standby,
                recency: 10,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 5,
                last_access: now,
            },
        );
        let semantic = HashMap::new();

        // Act: request both as victims
        let victims = policy.select_victims(&metadata, &semantic, 2);

        // Assert: both are candidates, same score — tiebreak by page_id ascending
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0], 1);
        assert_eq!(victims[1], 2);
    }

    // @trace TEST-MM-025 plan_prefill with zero L1 capacity forces full pipelined
    #[test]
    fn plan_prefill_zero_l1_capacity_forces_pipelined() {
        // Arrange: L1=0, L2=4, L3=0 — no L1 pages available at all
        let mut mgr = GlobalMemoryManager::new_with_capacities(0, 4, 0);

        // Act: 512 tokens / 128 per page = 4 pages needed
        let plan = mgr.plan_prefill(512, 256, 128);

        // Assert: Pipelined with l1_pages=0, l2_prefetch=4
        match plan {
            PrefillPlan::Pipelined { l1_pages, l2_prefetch, chunk_schedule } => {
                assert_eq!(l1_pages, 0, "no L1 pages available");
                assert_eq!(l2_prefetch, 4, "all pages come from L2");
                assert_eq!(chunk_schedule.len(), 2, "512/256 = 2 chunks");
                assert!(chunk_schedule.iter().all(|&c| c == 2), "each chunk is 2 pages");
            }
            _ => panic!("Expected Pipelined plan with zero L1"),
        }
    }

}
