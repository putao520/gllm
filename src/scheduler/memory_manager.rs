use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::time::Instant;

use gllm_kernels::kernel_types::{PageId, PageState, PhysicalId, RequestId};

use super::types::{KvPipeline, PageMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
                let semantic = semantic_priorities.get(&meta.page_id).copied().unwrap_or(0);
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

    /// 释放指定请求的 Working 管线页面
    pub fn release_working_pipeline(&mut self, request_id: RequestId) {
        if let Some(pages) = self
            .pipeline_pages
            .remove(&(KvPipeline::Working, request_id))
        {
            for pid in pages {
                let _ = self.tier_manager.free(Tier::L1, pid);
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
                    let _ = self.tier_manager.free(Tier::L1, pid);
                }
            }
        }
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
}
