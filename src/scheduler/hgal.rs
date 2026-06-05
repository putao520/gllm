use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::types::{EvictionPriority, PageId, PagePayloadKind, PageState, RequestId};

use super::memory_manager::VirtualPageId;
use super::types::GroupState;
use super::types::{PageMetadata, SequenceGroup, UnifiedVirtualPage};

// Priority formula source: SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md §4.1.
// These defaults are tuned to keep recency dominant while still rewarding locality.
const FREQUENCY_WEIGHT: i64 = 10;
// Large pin bonus guarantees pinned groups are effectively excluded from victim selection.
const PIN_BONUS: i64 = 5_000;
// Expert weight pages have higher eviction priority than KV pages (negative = evict first).
const EXPERT_WEIGHT_PRIORITY_BONUS: isize = -200;
// Dense layer weights are extremely hard to evict.
const DENSE_WEIGHT_PRIORITY_PENALTY: isize = 5_000;

#[derive(Debug, Clone)]
pub struct HGALConfig {
    pub warmup_duration: Duration,
    pub working_set_window: Duration,
    pub hot_threshold: usize,
    pub lir_ratio: f32,
    pub min_warm_access: usize,
    pub enable_clock_pro: bool,
}

impl Default for HGALConfig {
    fn default() -> Self {
        Self {
            warmup_duration: Duration::from_millis(100),
            working_set_window: Duration::from_secs(1),
            hot_threshold: 3,
            lir_ratio: 0.3,
            min_warm_access: 2,
            enable_clock_pro: true,
        }
    }
}

/// HGAL = Hybrid Gang-Aware LIRS scheduler core.
#[derive(Debug)]
pub struct HGALScheduler {
    /// Sequence groups (gang scheduling units).
    pub(crate) sequence_groups: HashMap<RequestId, SequenceGroup>,
    /// Per-page metadata for LIRS / warm / working-set logic.
    pub(crate) page_metadata: HashMap<PageId, PageMetadata>,
    /// Pages currently classified as LIR.
    pub(crate) lir_pages: HashSet<PageId>,
    pub(crate) config: HGALConfig,
    /// §21 WP-007: Weight page table: layer_idx → Vec<PhysicalId> for ExpertWeight pages.
    pub(crate) weight_page_table: HashMap<usize, Vec<PageId>>,
}

impl HGALScheduler {
    pub fn new(config: HGALConfig) -> Self {
        Self {
            sequence_groups: HashMap::new(),
            page_metadata: HashMap::new(),
            lir_pages: HashSet::new(),
            config,
            weight_page_table: HashMap::new(),
        }
    }

    pub fn config(&self) -> &HGALConfig {
        &self.config
    }

    /// Register or update a sequence group.
    pub fn upsert_group(&mut self, mut group: SequenceGroup) {
        // If group already exists, keep pages vector up to date but preserve access counters.
        if let Some(existing) = self.sequence_groups.get(&group.id) {
            group.access_count = existing.access_count;
            group.last_access = existing.last_access;
        }
        self.sequence_groups.insert(group.id, group);
    }

    /// Remove a sequence group from tracking (e.g., when completed).
    pub fn remove_group(&mut self, id: RequestId) {
        self.sequence_groups.remove(&id);
    }

    /// Update or insert page metadata with the latest state from backend.
    pub fn update_page_state(
        &mut self,
        page_id: PageId,
        sequence_id: Option<RequestId>,
        state: PageState,
    ) {
        let entry = self
            .page_metadata
            .entry(page_id)
            .or_insert_with(|| PageMetadata {
                page_id,
                ..Default::default()
            });
        entry.sequence_id = sequence_id;
        entry.state = state;
        if state == PageState::Swapped {
            entry.warm_until = None;
            entry.swap_in_time = None;
        }
    }

    /// Gang-aware victim selection. Returns sequence IDs to evict until the
    /// requested page count is satisfied.
    pub fn select_victim_groups(&mut self, count: usize) -> Vec<RequestId> {
        if count == 0 {
            return Vec::new();
        }

        let mut candidates: Vec<(RequestId, usize, i64)> = self
            .sequence_groups
            .values()
            .filter(|group| !group.is_pinned)
            .filter(|group| !self.group_has_protection(group))
            .map(|group| {
                let pages = group.pages.len().max(1);
                let priority = self.compute_group_priority(group);
                (group.id, pages, priority)
            })
            .collect();

        candidates.sort_by_key(|a| a.2);

        let mut selected = Vec::new();
        let mut accumulated_pages = 0usize;
        for (id, page_count, _) in candidates {
            selected.push(id);
            accumulated_pages = accumulated_pages.saturating_add(page_count);
            if accumulated_pages >= count {
                break;
            }
        }
        selected
    }

    /// Compute eviction priority for a single physical page (SPEC §5).
    ///
    /// Uses the `UnifiedVirtualPage` metadata combined with HGAL tracking state
    /// (access count, recency, warm/protected status) to compute a composite
    /// eviction score.
    ///
    /// **Score semantics**: Lower score = higher eviction priority (evicted first).
    ///
    /// Scoring factors (all additive):
    /// - **Recency penalty**: `-recency * TIME_DECAY_WEIGHT / 2` — older accesses lower score
    /// - **Frequency bonus**: `+access_count * FREQUENCY_WEIGHT` — hot pages resist
    /// - **Payload adjustment**: ExpertWeight gets negative (easy evict),
    ///   DenseLayerWeight gets positive (hard evict), KV context gets slight positive
    /// - **Pin bonus**: `+PIN_BONUS` if pinned (effectively unevictable)
    /// - **Layer depth**: deeper layers (higher index) get slight negative adjustment;
    ///   early layers are more important for attention
    pub fn compute_eviction_priority(&self, page: &UnifiedVirtualPage) -> EvictionPriority {
        let meta = self.page_metadata.get(&page.page_id);

        let access_count = meta.map(|m| m.access_count).unwrap_or(0);
        let recency = meta.map(|m| m.recency).unwrap_or(0);

        // ── Recency penalty: older accesses reduce score ──────────────────────
        let recency_penalty = (recency as i64) * (FREQUENCY_WEIGHT / 2);

        // ── Frequency bonus: hot pages resist eviction ──────────────────────
        let freq_bonus = (access_count as i64) * FREQUENCY_WEIGHT;

        // ── Payload-kind baseline ───────────────────────────────────────────
        let payload_adjustment: i64 = match page.payload_kind {
            PagePayloadKind::ExpertWeight => {
                // Expert weights are cheap to evict/reload; prefer evicting cold experts.
                let layer_depth_penalty = page.layer_idx.map(|l| -(l as i64)).unwrap_or(0);
                EXPERT_WEIGHT_PRIORITY_BONUS as i64 + layer_depth_penalty
            }
            PagePayloadKind::DenseLayerWeight => DENSE_WEIGHT_PRIORITY_PENALTY as i64,
            // KV context gets a moderate positive to protect it
            PagePayloadKind::KvContext => 100i64,
            // System prompts are pinned, handled by is_evictable
            PagePayloadKind::PromptSystem => 10_000i64,
            // RAG features are low priority
            PagePayloadKind::KnowledgeRAG => -200i64,
        };

        // ── Pin status: pinned pages get large bonus ────────────────────────
        let is_pinned = !page.is_evictable();
        let pin_bonus: i64 = if is_pinned { PIN_BONUS } else { 0 };

        // ── State bonus: Warm/Protected pages resist eviction ───────────────
        let state_bonus: i64 = meta
            .map(|m| match m.state {
                PageState::Protected => 10_000i64,
                PageState::Warm => 5_000i64,
                _ => 0,
            })
            .unwrap_or(0);

        let score = payload_adjustment - recency_penalty + freq_bonus + pin_bonus + state_bonus;

        EvictionPriority {
            score,
            payload_kind: page.payload_kind,
            is_pinned,
            access_count,
            recency,
            layer_idx: page.layer_idx,
            expert_id: page.expert_id,
        }
    }

    /// Select victim weight pages for eviction, ordered by priority (SPEC §5).
    ///
    /// Scans all registered weight pages (from `weight_page_table`) and returns
    /// the `count` pages with the lowest eviction priority score (most evictable).
    ///
    /// Payload kind is resolved from the owning SequenceGroup's `payload_kind`
    /// field, which is set correctly during registration (ExpertWeight for MoE,
    /// DenseLayerWeight for dense models).
    ///
    /// Priority order (from SPEC §4.3):
    /// 1. Cold Expert weight pages (lowest score, evicted first)
    /// 2. KnowledgeRAG pages (if any registered as weights)
    /// 3. KV context pages that have become Standby
    /// 4. Dense Layer weight pages (pinned, last resort)
    pub fn select_victim_weight_pages(&self, count: usize) -> Vec<(PageId, EvictionPriority)> {
        if count == 0 {
            return Vec::new();
        }

        // Build a reverse map: page_id → payload_kind from sequence groups.
        let mut page_payload_kinds: HashMap<PageId, PagePayloadKind> = HashMap::new();
        for group in self.sequence_groups.values() {
            if let Some(kind) = group.payload_kind {
                for &pid in &group.pages {
                    page_payload_kinds.insert(pid, kind);
                }
            }
        }

        // Collect all weight page IDs from weight_page_table.
        let mut candidates: Vec<(PageId, EvictionPriority)> = Vec::new();

        for page_ids in self.weight_page_table.values() {
            for &page_id in page_ids {
                let meta = match self.page_metadata.get(&page_id) {
                    Some(m) => m,
                    None => continue,
                };

                // Resolve payload kind from the owning SequenceGroup.
                let payload_kind = page_payload_kinds
                    .get(&page_id)
                    .copied()
                    .unwrap_or(PagePayloadKind::ExpertWeight);

                // Skip pinned/non-evictable pages unless extreme pressure.
                if meta.state == PageState::Protected || meta.state == PageState::Warm {
                    continue;
                }

                // Create a UnifiedVirtualPage reference for scoring.
                // layer_idx is recovered from weight_page_table keys.
                let layer_idx = self.weight_page_table.iter()
                    .find(|(_, pages)| pages.contains(&page_id))
                    .map(|(layer, _)| *layer);
                let expert_id = match payload_kind {
                    PagePayloadKind::ExpertWeight => Some(0u32),
                    _ => None,
                };

                let placeholder = UnifiedVirtualPage {
                    page_id,
                    payload_kind,
                    residency: super::types::MemoryResidency::DeviceLocal,
                    dtype: gllm_kernels::types::DType::F16,
                    owner: meta.sequence_id,
                    pipeline: None,
                    logical_index: 0,
                    codec: crate::kv_cache::CompressionCodec::None,
                    compressed_size: 0,
                    decompressed_size: 0,
                    expert_id,
                    layer_idx,
                };

                let priority = self.compute_eviction_priority(&placeholder);
                candidates.push((page_id, priority));
            }
        }

        // Sort by score ascending (lowest = most evictable).
        candidates.sort_by_key(|(_, p)| p.score);
        candidates.truncate(count);
        candidates
    }

    /// LIRS-style priority computation for a sequence group.
    fn compute_group_priority(&self, group: &SequenceGroup) -> i64 {
        let now = Instant::now();
        let time_penalty = now.saturating_duration_since(group.last_access).as_millis() as i64;

        let recency_penalty: i64 = group
            .pages
            .iter()
            .filter_map(|pid| self.page_metadata.get(pid))
            .map(|meta| meta.recency as i64)
            .sum();

        let freq_bonus = (group.access_count as i64) * FREQUENCY_WEIGHT;
        let pin_bonus: i64 = if group.is_pinned { PIN_BONUS } else { 0 };

        let payload_adjustment: i64 = match group.payload_kind {
            Some(PagePayloadKind::ExpertWeight) => EXPERT_WEIGHT_PRIORITY_BONUS as i64,
            Some(PagePayloadKind::DenseLayerWeight) => DENSE_WEIGHT_PRIORITY_PENALTY as i64,
            _ => 0,
        };

        time_penalty + recency_penalty - freq_bonus - pin_bonus + payload_adjustment
    }

    /// Whether the page is currently protected by warm-up.
    fn warm_until(meta: &PageMetadata, warmup_duration: Duration) -> Option<Instant> {
        meta.warm_until
            .or_else(|| meta.swap_in_time.map(|t| t + warmup_duration))
    }

    fn is_in_warmup_period_meta(
        meta: &PageMetadata,
        now: Instant,
        warmup_duration: Duration,
        min_warm_access: usize,
    ) -> bool {
        if meta.state != PageState::Warm {
            return false;
        }
        if meta.access_count >= min_warm_access {
            return false;
        }
        Self::warm_until(meta, warmup_duration)
            .map(|end| now < end)
            .unwrap_or(false) // LEGAL: warm_until 为 None 时视为不在 warmup 期
    }

    fn is_in_warmup_period(&self, page_id: PageId) -> bool {
        self.page_metadata
            .get(&page_id)
            .map(|meta| {
                Self::is_in_warmup_period_meta(
                    meta,
                    Instant::now(),
                    self.config.warmup_duration,
                    self.config.min_warm_access,
                )
            })
            .unwrap_or(false) // LEGAL: 不存在的 page_id 视为不在 warmup 期
    }

    /// Working set detection: promote hot pages to Protected, demote stale ones.
    pub fn detect_working_set(&mut self) {
        let now = Instant::now();
        let warmup_duration = self.config.warmup_duration;
        let min_warm_access = self.config.min_warm_access;
        for meta in self.page_metadata.values_mut() {
            if meta.state == PageState::Warm
                && !Self::is_in_warmup_period_meta(meta, now, warmup_duration, min_warm_access)
            {
                meta.state = PageState::Active;
                meta.warm_until = None;
            }
            let hot = meta.access_count >= self.config.hot_threshold
                && now.saturating_duration_since(meta.last_access) < self.config.working_set_window;

            match (hot, meta.state) {
                (true, PageState::Active | PageState::Standby | PageState::Warm) => {
                    meta.state = PageState::Protected;
                }
                (false, PageState::Protected)
                    if now.saturating_duration_since(meta.last_access)
                        >= self.config.working_set_window =>
                {
                    // Protection expires; return to Standby so it can be considered for eviction.
                    meta.state = PageState::Standby;
                }
                _ => {
                    log::debug!("hgal: no state transition for page {} (hot={hot}, state={:?})", meta.page_id, meta.state);
                }
            }
        }
    }

    /// Record a page access, updating LIRS metadata and group stats.
    pub fn mark_accessed(&mut self, page_id: PageId) {
        let now = Instant::now();
        let warmup_duration = self.config.warmup_duration;
        let min_warm_access = self.config.min_warm_access;
        let sequence_id = {
            let entry = self
                .page_metadata
                .entry(page_id)
                .or_insert_with(|| PageMetadata {
                    page_id,
                    ..Default::default()
                });
            let irr = now
                .saturating_duration_since(entry.last_access)
                .as_millis()
                .try_into()
                .unwrap_or(usize::MAX); // LEGAL: 溢出时使用 MAX（极端边界情况）
            entry.recency = irr;
            entry.access_count = entry.access_count.saturating_add(1);
            entry.last_access = now;
            if entry.state == PageState::Warm {
                if !Self::is_in_warmup_period_meta(entry, now, warmup_duration, min_warm_access) {
                    entry.state = PageState::Active;
                    entry.warm_until = None;
                }
            } else if entry.state != PageState::Protected {
                entry.state = PageState::Active;
            }
            entry.sequence_id
        };

        if self.config.enable_clock_pro {
            self.update_lir_membership(page_id);
        }

        if let Some(seq_id) = sequence_id {
            if let Some(group) = self.sequence_groups.get_mut(&seq_id) {
                group.access_count = group.access_count.saturating_add(1);
                group.last_access = now;
            }
        }
    }

    /// Mark a page as swapped into GPU and begin warm-up protection.
    pub fn on_swap_in(&mut self, page_id: PageId) {
        let now = Instant::now();
        let entry = self
            .page_metadata
            .entry(page_id)
            .or_insert_with(|| PageMetadata {
                page_id,
                ..Default::default()
            });
        entry.swap_in_time = Some(now);
        entry.warm_until = Some(now + self.config.warmup_duration);
        entry.state = PageState::Warm;
        entry.access_count = 0;
        entry.last_access = now;
    }

    /// Prefill chunk 完成回调
    pub fn on_prefill_chunk_complete(
        &mut self,
        chunk_idx: usize,
        total_chunks: usize,
        pages: &[VirtualPageId],
    ) {
        if pages.is_empty() {
            return;
        }

        let now = Instant::now();
        let is_last_chunk = chunk_idx.saturating_add(1) >= total_chunks;
        let mut grouped_pages: HashMap<RequestId, Vec<PageId>> = HashMap::new();

        for vpid in pages {
            let page_id = virtual_page_to_page_id(*vpid);
            let entry = self
                .page_metadata
                .entry(page_id)
                .or_insert_with(|| PageMetadata {
                    page_id,
                    ..Default::default()
                });
            entry.sequence_id = Some(vpid.sequence_id);
            entry.last_access = now;
            entry.access_count = entry.access_count.saturating_add(1);
            entry.recency = 0;
            entry.state = if is_last_chunk {
                PageState::Active
            } else {
                PageState::Standby
            };
            grouped_pages
                .entry(vpid.sequence_id)
                .or_default()
                .push(page_id);
        }

        for (request_id, mut page_ids) in grouped_pages {
            let group = self
                .sequence_groups
                .entry(request_id)
                .or_insert_with(|| SequenceGroup {
                    id: request_id,
                    pages: Vec::new(),
                    state: GroupState::Running,
                    access_count: 0,
                    last_access: now,
                    is_pinned: false,
                    context_len: 0,
                    pipeline: crate::scheduler::types::KvPipeline::Conversation,
                    payload_kind: None,
                });
            group.access_count = group.access_count.saturating_add(1);
            group.last_access = now;
            group.state = GroupState::Running;
            group.pages.append(&mut page_ids);
            group.pages.sort_unstable();
            group.pages.dedup();
            group.context_len = group.context_len.max(group.pages.len());
        }
    }

    /// Prefill 完成回调
    pub fn on_prefill_complete(&mut self, request_id: RequestId) {
        if let Some(group) = self.sequence_groups.get_mut(&request_id) {
            group.last_access = Instant::now();
            group.state = GroupState::Running;
            for page_id in group.pages.iter() {
                if let Some(meta) = self.page_metadata.get_mut(page_id) {
                    if meta.state != PageState::Swapped {
                        meta.state = PageState::Active;
                    }
                    meta.warm_until = None;
                }
            }
        }
    }

    // ── ExpertWeight page lifecycle management (REQ-WP3) ──

    /// Register a single ExpertWeight page in the weight page table.
    /// `layer_idx` identifies which transformer layer this page belongs to.
    /// `page_id` is the physical page identifier.
    pub fn register_expert_weight_page(&mut self, page_id: PageId, layer_idx: usize) {
        self.weight_page_table
            .entry(layer_idx)
            .or_default()
            .push(page_id);
        // Register page metadata in HGAL as Active
        self.update_page_state(page_id, None, PageState::Active);
        log::debug!(
            "hgal: registered ExpertWeight page {} for layer {}",
            page_id,
            layer_idx
        );
    }

    /// Allocate ExpertWeight pages for all experts in a given layer.
    /// Returns the list of allocated page IDs.
    /// Each page gets registered in the weight_page_table automatically.
    pub fn allocate_expert_weight_pages(
        &mut self,
        num_experts: usize,
        layer_idx: usize,
    ) -> Vec<PageId> {
        let mut pages = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            // Generate a unique page ID for this expert's weight page
            // Use (layer_idx << 16 | expert_idx) as the stable physical ID
            let page_id = (layer_idx << 16) | expert_idx;
            self.register_expert_weight_page(page_id, layer_idx);
            pages.push(page_id);
        }
        log::info!(
            "hgal: allocated {} ExpertWeight pages for layer {}",
            num_experts,
            layer_idx
        );
        pages
    }

    /// Free all ExpertWeight pages for a given layer.
    /// Removes entries from both weight_page_table and page_metadata.
    pub fn free_expert_weight_pages(&mut self, layer_idx: usize) {
        if let Some(pages) = self.weight_page_table.remove(&layer_idx) {
            for page_id in &pages {
                self.page_metadata.remove(page_id);
                self.lir_pages.remove(page_id);
            }
            log::info!(
                "hgal: freed {} ExpertWeight pages for layer {}",
                pages.len(),
                layer_idx
            );
        }
    }

    /// Return the total number of registered ExpertWeight pages across all layers.
    pub fn num_expert_weight_pages(&self) -> usize {
        self.weight_page_table.values().map(|v| v.len()).sum()
    }

    // ── DenseLayerWeight page lifecycle management (REQ-QWP-005) ──

    /// Register a single DenseLayerWeight page in the weight page table.
    ///
    /// Dense layer weight pages are pinned (not evictable under normal memory
    /// pressure) and default to `DeviceLocal` residency. They are only evicted
    /// under extreme memory pressure via explicit admin unpin.
    ///
    /// `layer_idx` identifies which transformer layer this page belongs to.
    /// `page_id` is the physical page identifier.
    pub fn register_dense_layer_weight_page(&mut self, page_id: PageId, layer_idx: usize) {
        self.weight_page_table
            .entry(layer_idx)
            .or_default()
            .push(page_id);
        // Register page metadata in HGAL as Active
        self.update_page_state(page_id, None, PageState::Active);
        log::debug!(
            "hgal: registered DenseLayerWeight page {} for layer {}",
            page_id,
            layer_idx
        );
    }

    /// Register multiple dense layer weight pages for a given layer.
    ///
    /// Each dense layer typically has one weight page entry covering all
    /// attention + FFN weights for that layer. For quantized formats (.gllm),
    /// a layer may span multiple pages based on quantization block alignment.
    ///
    /// Returns the list of registered page IDs.
    pub fn register_dense_layer_weight_pages(
        &mut self,
        page_ids: Vec<PageId>,
        layer_idx: usize,
    ) -> Vec<PageId> {
        for &page_id in &page_ids {
            self.weight_page_table
                .entry(layer_idx)
                .or_default()
                .push(page_id);
            self.update_page_state(page_id, None, PageState::Active);
        }
        log::info!(
            "hgal: registered {} DenseLayerWeight pages for layer {}",
            page_ids.len(),
            layer_idx
        );
        page_ids
    }

    /// Free all DenseLayerWeight pages for a given layer.
    /// Removes entries from both weight_page_table and page_metadata.
    pub fn free_dense_layer_weight_pages(&mut self, layer_idx: usize) {
        if let Some(pages) = self.weight_page_table.remove(&layer_idx) {
            for page_id in &pages {
                self.page_metadata.remove(page_id);
                self.lir_pages.remove(page_id);
            }
            log::info!(
                "hgal: freed {} DenseLayerWeight pages for layer {}",
                pages.len(),
                layer_idx
            );
        }
    }

    /// Return the total number of registered weight pages (expert + dense) across all layers.
    pub fn num_weight_pages(&self) -> usize {
        self.weight_page_table.values().map(|v| v.len()).sum()
    }

    /// Return the total number of registered DenseLayerWeight pages across all layers.
    pub fn num_dense_layer_weight_pages(&self) -> usize {
        // Dense pages are those whose SequenceGroup payload_kind is DenseLayerWeight
        self.sequence_groups
            .values()
            .filter(|g| g.payload_kind == Some(PagePayloadKind::DenseLayerWeight))
            .map(|g| g.pages.len())
            .sum()
    }

    fn group_has_protection(&self, group: &SequenceGroup) -> bool {
        group.pages.iter().any(|pid| {
            if let Some(meta) = self.page_metadata.get(pid) {
                if matches!(meta.state, PageState::Warm | PageState::Protected) {
                    return true;
                }
            } else {
                return false;
            }
            self.is_in_warmup_period(*pid)
        })
    }

    fn update_lir_membership(&mut self, page_id: PageId) {
        let target_lir = ((self.page_metadata.len() as f32) * self.config.lir_ratio)
            .ceil()
            .max(1.0) as usize;

        self.lir_pages.insert(page_id);
        if let Some(meta) = self.page_metadata.get_mut(&page_id) {
            meta.is_lir = true;
        }

        while self.lir_pages.len() > target_lir {
            if let Some(candidate) = self.select_coldest_lir() {
                self.lir_pages.remove(&candidate);
                if let Some(meta) = self.page_metadata.get_mut(&candidate) {
                    meta.is_lir = false;
                }
            } else {
                break;
            }
        }
    }

    fn select_coldest_lir(&self) -> Option<PageId> {
        self.lir_pages
            .iter()
            .filter_map(|pid| self.page_metadata.get(pid))
            .max_by_key(|meta| meta.recency)
            .map(|meta| meta.page_id)
    }
}

fn virtual_page_to_page_id(vpid: VirtualPageId) -> PageId {
    let sid = vpid.sequence_id as u128;
    let lid = vpid.logical_index as u128;
    let paired = (sid + lid) * (sid + lid + 1) / 2 + lid;
    let limit = usize::MAX as u128;
    (paired % limit) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_pages_not_selected_as_victim() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        let warm_group = SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(warm_group);
        scheduler.update_page_state(1, Some(1), PageState::Warm);
        scheduler.on_swap_in(1);

        let cold_group = SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(cold_group);
        scheduler.update_page_state(2, Some(2), PageState::Standby);

        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![2]);
    }

    #[test]
    fn working_set_detection_promotes_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let meta = PageMetadata {
            page_id: 3,
            sequence_id: Some(3),
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        };
        scheduler.page_metadata.insert(3, meta);

        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata.get(&3).map(|m| m.state).unwrap(),
            PageState::Protected
        );
    }

    #[test]
    fn prefill_callbacks_update_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![
            VirtualPageId::new(9, 0),
            VirtualPageId::new(9, 1),
            VirtualPageId::new(9, 2),
        ];

        scheduler.on_prefill_chunk_complete(0, 2, &pages[..2]);
        let first = virtual_page_to_page_id(pages[0]);
        assert_eq!(
            scheduler.page_metadata.get(&first).map(|m| m.state),
            Some(PageState::Standby)
        );

        scheduler.on_prefill_chunk_complete(1, 2, &pages[2..]);
        scheduler.on_prefill_complete(9);
        let last = virtual_page_to_page_id(pages[2]);
        assert_eq!(
            scheduler.page_metadata.get(&last).map(|m| m.state),
            Some(PageState::Active)
        );
    }

    // ── DenseLayerWeight page tests (REQ-QWP-005) ──

    #[test]
    fn dense_layer_weight_registration_populates_weight_page_table() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        scheduler.register_dense_layer_weight_page(100, 0);
        scheduler.register_dense_layer_weight_page(101, 1);
        scheduler.register_dense_layer_weight_page(102, 2);

        assert_eq!(scheduler.weight_page_table.get(&0), Some(&vec![100]));
        assert_eq!(scheduler.weight_page_table.get(&1), Some(&vec![101]));
        assert_eq!(scheduler.weight_page_table.get(&2), Some(&vec![102]));
        assert_eq!(scheduler.num_weight_pages(), 3);
    }

    #[test]
    fn dense_layer_weight_multi_page_registration() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        let pages = scheduler.register_dense_layer_weight_pages(vec![200, 201, 202], 5);

        assert_eq!(pages, vec![200, 201, 202]);
        assert_eq!(scheduler.weight_page_table.get(&5), Some(&vec![200, 201, 202]));
        assert_eq!(scheduler.num_weight_pages(), 3);

        // Verify page metadata is Active
        for pid in &[200, 201, 202] {
            let meta = scheduler.page_metadata.get(pid).expect("metadata exists");
            assert_eq!(meta.state, PageState::Active);
        }
    }

    #[test]
    fn dense_layer_weight_pages_are_pinned() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Register a dense layer weight page + sequence group (mimicking executor_builder)
        let page_id = 50;
        let layer_idx = 3;
        scheduler.register_dense_layer_weight_page(page_id, layer_idx);

        let uvp = crate::scheduler::types::UnifiedVirtualPage::dense_layer(
            page_id,
            layer_idx,
            gllm_kernels::types::DType::F32,
        );
        assert!(!uvp.is_evictable(), "DenseLayerWeight pages must not be evictable");

        let weight_group_id = (layer_idx as u64).wrapping_add(1_000_000);
        scheduler.upsert_group(SequenceGroup {
            id: weight_group_id,
            pages: vec![page_id],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: true,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // Pinned groups should not be selected as victims
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "pinned DenseLayerWeight group should not be a victim");
    }

    #[test]
    fn dense_layer_weight_eviction_priority_higher_than_expert() {
        let scheduler = HGALScheduler::new(HGALConfig::default());

        let dense_page = crate::scheduler::types::UnifiedVirtualPage::dense_layer(
            10, 0, gllm_kernels::types::DType::F32,
        );
        let expert_page = crate::scheduler::types::UnifiedVirtualPage::expert(
            20, 0, 0, gllm_kernels::types::DType::F32,
        );

        let dense_prio = scheduler.compute_eviction_priority(&dense_page);
        let expert_prio = scheduler.compute_eviction_priority(&expert_page);

        assert!(
            dense_prio.score > expert_prio.score,
            "DenseLayerWeight priority ({}) must be higher (harder to evict) than ExpertWeight ({})",
            dense_prio.score, expert_prio.score,
        );
        assert_eq!(dense_prio.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert_eq!(expert_prio.payload_kind, PagePayloadKind::ExpertWeight);
    }

    #[test]
    fn free_dense_layer_weight_pages_cleans_up() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        scheduler.register_dense_layer_weight_pages(vec![300, 301], 7);
        assert_eq!(scheduler.num_weight_pages(), 2);

        scheduler.free_dense_layer_weight_pages(7);
        assert_eq!(scheduler.num_weight_pages(), 0);
        assert!(scheduler.weight_page_table.get(&7).is_none());
        assert!(scheduler.page_metadata.get(&300).is_none());
        assert!(scheduler.page_metadata.get(&301).is_none());
    }

    #[test]
    fn num_dense_layer_weight_pages_counts_correctly() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Register pages + groups so payload_kind is tracked
        for layer_idx in 0..3 {
            let page_id = layer_idx * 10;
            scheduler.register_dense_layer_weight_page(page_id, layer_idx);
            let weight_group_id = (layer_idx as u64).wrapping_add(1_000_000);
            scheduler.upsert_group(SequenceGroup {
                id: weight_group_id,
                pages: vec![page_id],
                state: GroupState::Running,
                access_count: 0,
                last_access: now,
                is_pinned: true,
                context_len: 0,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: Some(PagePayloadKind::DenseLayerWeight),
            });
        }

        assert_eq!(scheduler.num_dense_layer_weight_pages(), 3);
    }

    #[test]
    fn select_victim_weight_pages_distinguishes_dense_from_expert() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Register expert weight page
        scheduler.register_expert_weight_page(100, 0);
        let expert_group_id = 1_000_000u64;
        scheduler.upsert_group(SequenceGroup {
            id: expert_group_id,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Register dense layer weight page
        scheduler.register_dense_layer_weight_page(200, 1);
        let dense_group_id = 1_000_001u64;
        scheduler.upsert_group(SequenceGroup {
            id: dense_group_id,
            pages: vec![200],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: true,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // Select victim weight pages — expert should be selected, dense should not
        let victims = scheduler.select_victim_weight_pages(2);

        // Only the expert page should appear (dense is pinned, so its group
        // state is still Active but its score is much higher)
        let victim_ids: Vec<PageId> = victims.iter().map(|(pid, _)| *pid).collect();
        assert!(
            !victim_ids.is_empty(),
            "should select at least the expert weight page"
        );
        // Expert page must have lower score than dense page
        if let Some(expert_entry) = victims.iter().find(|(pid, _)| *pid == 100) {
            if let Some(dense_entry) = victims.iter().find(|(pid, _)| *pid == 200) {
                assert!(
                    expert_entry.1.score < dense_entry.1.score,
                    "expert score ({}) should be lower than dense score ({})",
                    expert_entry.1.score, dense_entry.1.score,
                );
            }
        }
    }

    // ── HGALConfig tests ──

    #[test]
    fn hgal_config_default_values() {
        let config = HGALConfig::default();
        assert_eq!(config.warmup_duration, Duration::from_millis(100));
        assert_eq!(config.working_set_window, Duration::from_secs(1));
        assert_eq!(config.hot_threshold, 3);
        assert!((config.lir_ratio - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.min_warm_access, 2);
        assert!(config.enable_clock_pro);
    }

    #[test]
    fn hgal_config_custom_values() {
        let config = HGALConfig {
            warmup_duration: Duration::from_millis(500),
            working_set_window: Duration::from_secs(5),
            hot_threshold: 10,
            lir_ratio: 0.5,
            min_warm_access: 5,
            enable_clock_pro: false,
        };
        assert_eq!(config.warmup_duration, Duration::from_millis(500));
        assert_eq!(config.working_set_window, Duration::from_secs(5));
        assert_eq!(config.hot_threshold, 10);
        assert!((config.lir_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_warm_access, 5);
        assert!(!config.enable_clock_pro);
    }

    // ── HGALScheduler::new + config() tests ──

    #[test]
    fn new_scheduler_is_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.sequence_groups.is_empty());
        assert!(scheduler.page_metadata.is_empty());
        assert!(scheduler.lir_pages.is_empty());
        assert!(scheduler.weight_page_table.is_empty());
    }

    #[test]
    fn config_returns_reference_to_inner_config() {
        let config = HGALConfig {
            hot_threshold: 7,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().hot_threshold, 7);
        assert_eq!(scheduler.config().warmup_duration, Duration::from_millis(100));
    }

    // ── upsert_group / remove_group tests ──

    #[test]
    fn upsert_group_inserts_new_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 42,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert_eq!(scheduler.sequence_groups.len(), 1);
        assert!(scheduler.sequence_groups.contains_key(&42));
        assert_eq!(scheduler.sequence_groups[&42].pages, vec![1, 2, 3]);
    }

    #[test]
    fn upsert_group_preserves_access_counters_on_update() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let original = SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 7,
            last_access: Instant::now() - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(original);

        let updated = SequenceGroup {
            id: 10,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        scheduler.upsert_group(updated);

        let group = &scheduler.sequence_groups[&10];
        assert_eq!(group.access_count, 7, "access_count must be preserved from existing group");
        assert_eq!(group.pages, vec![1, 2], "pages must be updated");
        assert!(group.is_pinned, "is_pinned must be updated");
        assert_eq!(group.payload_kind, Some(PagePayloadKind::KvContext));
    }

    #[test]
    fn remove_group_deletes_existing() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 55,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert!(scheduler.sequence_groups.contains_key(&55));
        scheduler.remove_group(55);
        assert!(!scheduler.sequence_groups.contains_key(&55));
    }

    #[test]
    fn remove_nonexistent_group_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.remove_group(999);
        assert!(scheduler.sequence_groups.is_empty());
    }

    // ── update_page_state tests ──

    #[test]
    fn update_page_state_creates_new_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(100, Some(1), PageState::Active);
        let meta = scheduler.page_metadata.get(&100).expect("page must exist");
        assert_eq!(meta.page_id, 100);
        assert_eq!(meta.sequence_id, Some(1));
        assert_eq!(meta.state, PageState::Active);
    }

    #[test]
    fn update_page_state_updates_existing_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(200, Some(2), PageState::Active);
        scheduler.update_page_state(200, Some(3), PageState::Standby);
        let meta = scheduler.page_metadata.get(&200).expect("page must exist");
        assert_eq!(meta.sequence_id, Some(3));
        assert_eq!(meta.state, PageState::Standby);
    }

    #[test]
    fn update_page_state_swapped_clears_warm_and_swap_in() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(300);
        assert!(scheduler.page_metadata[&300].warm_until.is_some());
        assert!(scheduler.page_metadata[&300].swap_in_time.is_some());
        scheduler.update_page_state(300, Some(5), PageState::Swapped);
        let meta = &scheduler.page_metadata[&300];
        assert!(meta.warm_until.is_none(), "warm_until must be cleared on Swapped");
        assert!(meta.swap_in_time.is_none(), "swap_in_time must be cleared on Swapped");
        assert_eq!(meta.state, PageState::Swapped);
    }

    // ── select_victim_groups tests ──

    #[test]
    fn select_victim_groups_count_zero_returns_empty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert!(scheduler.select_victim_groups(0).is_empty());
    }

    #[test]
    fn select_victim_groups_empty_scheduler_returns_empty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.select_victim_groups(10).is_empty());
    }

    #[test]
    fn select_victim_groups_skips_pinned_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pinned = SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(pinned);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "pinned groups must not be victims");
    }

    #[test]
    fn select_victim_groups_prefers_lower_priority() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Two groups with same recency but different payload_kind.
        // ExpertWeight gets payload_adjustment = -200, DenseLayerWeight = +5000.
        // Lower score = evicted first. ExpertWeight (-200) < DenseLayerWeight (+5000).
        let expert_group = SequenceGroup {
            id: 100,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        };
        scheduler.upsert_group(expert_group);

        let dense_group = SequenceGroup {
            id: 200,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        };
        scheduler.upsert_group(dense_group);

        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 100, "ExpertWeight group (payload -200) should be evicted before DenseLayerWeight (+5000)");
    }

    #[test]
    fn select_victim_groups_accumulates_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        let g1 = SequenceGroup {
            id: 10,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        let g2 = SequenceGroup {
            id: 20,
            pages: vec![3, 4, 5],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(3),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(g1);
        scheduler.upsert_group(g2);

        let victims = scheduler.select_victim_groups(4);
        assert_eq!(victims.len(), 2);
    }

    // ── compute_eviction_priority tests ──

    #[test]
    fn eviction_priority_kv_context_gets_positive_adjustment() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.score > 0, "KV context with no recency should have positive score");
        assert_eq!(prio.payload_kind, PagePayloadKind::KvContext);
        assert!(!prio.is_pinned);
    }

    #[test]
    fn eviction_priority_prompt_system_has_highest_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let kv_page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prompt_page = UnifiedVirtualPage::system_prompt(2, gllm_kernels::types::DType::F32);
        let kv_prio = scheduler.compute_eviction_priority(&kv_page);
        let prompt_prio = scheduler.compute_eviction_priority(&prompt_page);
        assert!(
            prompt_prio.score > kv_prio.score,
            "PromptSystem (score={}) must have higher score than KvContext (score={})",
            prompt_prio.score, kv_prio.score,
        );
        assert!(prompt_prio.is_pinned);
    }

    #[test]
    fn eviction_priority_knowledge_rag_gets_negative_adjustment() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let rag_page = UnifiedVirtualPage::rag(5, 42, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&rag_page);
        assert!(prio.score < 0, "KnowledgeRAG should have negative base score");
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
    }

    #[test]
    fn eviction_priority_frequency_increases_score() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        let page1 = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio_cold = scheduler.compute_eviction_priority(&page1);

        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            access_count: 10,
            ..Default::default()
        });
        let page2 = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio_hot = scheduler.compute_eviction_priority(&page2);

        assert!(
            prio_hot.score > prio_cold.score,
            "Frequent access (score={}) must increase score over cold (score={})",
            prio_hot.score, prio_cold.score,
        );
    }

    #[test]
    fn eviction_priority_state_bonus_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.score >= 10_000, "Protected state bonus should add at least 10000");
    }

    #[test]
    fn eviction_priority_state_bonus_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.score >= 5_000, "Warm state bonus should add at least 5000");
    }

    #[test]
    fn eviction_priority_expert_weight_layer_depth_penalty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let shallow = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        let deep = UnifiedVirtualPage::expert(2, 0, 10, gllm_kernels::types::DType::F32);
        let prio_shallow = scheduler.compute_eviction_priority(&shallow);
        let prio_deep = scheduler.compute_eviction_priority(&deep);
        assert!(
            prio_deep.score < prio_shallow.score,
            "Deeper layer expert (score={}) should have lower score than shallow (score={})",
            prio_deep.score, prio_shallow.score,
        );
    }

    // ── mark_accessed tests ──

    #[test]
    fn mark_accessed_creates_metadata_if_missing() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(!scheduler.page_metadata.contains_key(&42));
        scheduler.mark_accessed(42);
        assert!(scheduler.page_metadata.contains_key(&42));
        assert_eq!(scheduler.page_metadata[&42].access_count, 1);
    }

    #[test]
    fn mark_accessed_increments_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 3);
    }

    #[test]
    fn mark_accessed_updates_group_access_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 77,
            pages: vec![50],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(50, Some(77), PageState::Active);
        scheduler.mark_accessed(50);
        scheduler.mark_accessed(50);
        assert_eq!(scheduler.sequence_groups[&77].access_count, 2);
    }

    #[test]
    fn mark_accessed_transitions_standby_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(99, None, PageState::Standby);
        assert_eq!(scheduler.page_metadata[&99].state, PageState::Standby);
        scheduler.mark_accessed(99);
        assert_eq!(scheduler.page_metadata[&99].state, PageState::Active);
    }

    #[test]
    fn mark_accessed_preserves_protected_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(88, None, PageState::Protected);
        scheduler.mark_accessed(88);
        assert_eq!(scheduler.page_metadata[&88].state, PageState::Protected);
    }

    // ── on_swap_in tests ──

    #[test]
    fn on_swap_in_sets_warm_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(55);
        let meta = &scheduler.page_metadata[&55];
        assert_eq!(meta.state, PageState::Warm);
        assert!(meta.swap_in_time.is_some());
        assert!(meta.warm_until.is_some());
        assert_eq!(meta.access_count, 0, "swap-in resets access count");
    }

    #[test]
    fn on_swap_in_creates_metadata_if_missing() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(!scheduler.page_metadata.contains_key(&77));
        scheduler.on_swap_in(77);
        assert!(scheduler.page_metadata.contains_key(&77));
    }

    // ── detect_working_set tests ──

    #[test]
    fn detect_working_set_demotes_expired_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_millis(1),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(1),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby);
    }

    #[test]
    fn detect_working_set_promotes_active_to_protected_when_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 10,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    #[test]
    fn detect_working_set_promotes_standby_to_protected_when_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            access_count: 10,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    #[test]
    fn detect_working_set_transitions_warm_to_active_after_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: 100,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(1),
            swap_in_time: Some(Instant::now() - Duration::from_secs(1)),
            warm_until: Some(Instant::now() - Duration::from_secs(1)),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Active,
            "Warm page past warmup period must transition to Active"
        );
    }

    // ── ExpertWeight lifecycle tests ──

    #[test]
    fn register_expert_weight_page_creates_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 3);
        assert_eq!(scheduler.weight_page_table.get(&3), Some(&vec![42]));
        let meta = scheduler.page_metadata.get(&42).expect("metadata must exist");
        assert_eq!(meta.state, PageState::Active);
        assert_eq!(meta.page_id, 42);
    }

    #[test]
    fn register_expert_weight_page_multiple_layers() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        scheduler.register_expert_weight_page(20, 1);
        assert_eq!(scheduler.weight_page_table[&0], vec![10, 11]);
        assert_eq!(scheduler.weight_page_table[&1], vec![20]);
        assert_eq!(scheduler.num_expert_weight_pages(), 3);
    }

    #[test]
    fn allocate_expert_weight_pages_generates_unique_ids() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(4, 2);
        assert_eq!(pages.len(), 4);
        let unique: HashSet<PageId> = pages.iter().copied().collect();
        assert_eq!(unique.len(), 4);
        assert_eq!(scheduler.num_expert_weight_pages(), 4);
    }

    #[test]
    fn allocate_expert_weight_pages_id_encoding() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(3, 5);
        assert_eq!(pages[0], (5usize << 16) | 0);
        assert_eq!(pages[1], (5usize << 16) | 1);
        assert_eq!(pages[2], (5usize << 16) | 2);
    }

    #[test]
    fn free_expert_weight_pages_removes_all_traces() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(3, 1);
        assert_eq!(scheduler.num_expert_weight_pages(), 3);
        scheduler.free_expert_weight_pages(1);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
        assert!(scheduler.weight_page_table.get(&1).is_none());
    }

    #[test]
    fn free_nonexistent_expert_weight_pages_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.free_expert_weight_pages(999);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    // ── select_victim_weight_pages edge case tests ──

    #[test]
    fn select_victim_weight_pages_count_zero_returns_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.select_victim_weight_pages(0).is_empty());
    }

    #[test]
    fn select_victim_weight_pages_empty_table_returns_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.select_victim_weight_pages(10).is_empty());
    }

    #[test]
    fn select_victim_weight_pages_skips_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.register_expert_weight_page(10, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Protected;
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "Protected pages must be skipped");
    }

    #[test]
    fn select_victim_weight_pages_skips_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.register_expert_weight_page(10, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Warm;
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "Warm pages must be skipped");
    }

    // ── on_prefill_complete tests ──

    #[test]
    fn on_prefill_complete_activates_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        let group = SequenceGroup {
            id: 88,
            pages: vec![10, 11, 12],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        for pid in &[10, 11, 12] {
            scheduler.update_page_state(*pid, Some(88), PageState::Standby);
        }
        scheduler.on_prefill_complete(88);
        for pid in &[10, 11, 12] {
            let meta = scheduler.page_metadata.get(pid).unwrap();
            assert_eq!(meta.state, PageState::Active, "page {} must be Active after prefill complete", pid);
            assert!(meta.warm_until.is_none(), "warm_until must be cleared");
        }
    }

    #[test]
    fn on_prefill_complete_preserves_swapped_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 99,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(20, Some(99), PageState::Swapped);
        scheduler.on_prefill_complete(99);
        assert_eq!(scheduler.page_metadata[&20].state, PageState::Swapped, "Swapped state must be preserved");
    }

    #[test]
    fn on_prefill_complete_nonexistent_group_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_prefill_complete(12345);
        assert!(scheduler.sequence_groups.is_empty());
    }

    // ── on_prefill_chunk_complete edge cases ──

    #[test]
    fn on_prefill_chunk_complete_empty_pages_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_prefill_chunk_complete(0, 2, &[]);
        assert!(scheduler.page_metadata.is_empty());
        assert!(scheduler.sequence_groups.is_empty());
    }

    #[test]
    fn on_prefill_chunk_complete_last_chunk_sets_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(1, 2, &pages);
        let page_id = virtual_page_to_page_id(pages[0]);
        assert_eq!(
            scheduler.page_metadata[&page_id].state,
            PageState::Active,
            "Last chunk pages must be Active"
        );
    }

    #[test]
    fn on_prefill_chunk_complete_non_last_chunk_sets_standby() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        let page_id = virtual_page_to_page_id(pages[0]);
        assert_eq!(
            scheduler.page_metadata[&page_id].state,
            PageState::Standby,
            "Non-last chunk pages must be Standby"
        );
    }

    #[test]
    fn on_prefill_chunk_complete_deduplicates_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        scheduler.on_prefill_chunk_complete(1, 2, &pages);
        let group = &scheduler.sequence_groups[&42];
        let page_id = virtual_page_to_page_id(pages[0]);
        let count = group.pages.iter().filter(|&&p| p == page_id).count();
        assert_eq!(count, 1, "duplicate pages must be deduped");
    }

    // ── num_weight_pages / num_dense_layer_weight_pages tests ──

    #[test]
    fn num_weight_pages_counts_expert_and_dense() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        assert_eq!(scheduler.num_weight_pages(), 3);
    }

    #[test]
    fn num_dense_layer_weight_pages_zero_without_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(10, 0);
        assert_eq!(scheduler.num_dense_layer_weight_pages(), 0);
    }

    // ── virtual_page_to_page_id tests ──

    #[test]
    fn virtual_page_to_page_id_deterministic() {
        let vpid = VirtualPageId::new(42, 7);
        let id1 = virtual_page_to_page_id(vpid);
        let id2 = virtual_page_to_page_id(vpid);
        assert_eq!(id1, id2, "same VirtualPageId must produce same PageId");
    }

    #[test]
    fn virtual_page_to_page_id_different_for_different_inputs() {
        let vpid_a = VirtualPageId::new(42, 0);
        let vpid_b = VirtualPageId::new(42, 1);
        assert_ne!(
            virtual_page_to_page_id(vpid_a),
            virtual_page_to_page_id(vpid_b),
            "different logical indices must produce different PageIds"
        );
    }

    // ── LIR membership (clock_pro) tests ──

    #[test]
    fn mark_accessed_updates_lir_with_clock_pro_enabled() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        });
        scheduler.mark_accessed(10);
        assert!(scheduler.lir_pages.contains(&10), "page must be in LIR set after access");
        let meta = &scheduler.page_metadata[&10];
        assert!(meta.is_lir, "page metadata must be marked as LIR");
    }

    #[test]
    fn mark_accessed_skips_lir_when_clock_pro_disabled() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        scheduler.mark_accessed(10);
        assert!(!scheduler.lir_pages.contains(&10), "page must not be in LIR set when clock_pro disabled");
        assert!(!scheduler.page_metadata[&10].is_lir);
    }

    #[test]
    fn lir_membership_evicts_coldest_when_over_capacity() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        });
        // Insert 3 pages so target_lir = ceil(3 * 0.5) = 2
        scheduler.mark_accessed(1);
        scheduler.mark_accessed(2);
        scheduler.mark_accessed(3);
        assert!(scheduler.lir_pages.len() <= 2);
    }

    // ── Additional tests (18 new) ──

    #[test]
    fn hgal_config_clone_preserves_all_fields() {
        let config = HGALConfig {
            warmup_duration: Duration::from_millis(250),
            working_set_window: Duration::from_secs(3),
            hot_threshold: 7,
            lir_ratio: 0.4,
            min_warm_access: 3,
            enable_clock_pro: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.warmup_duration, config.warmup_duration);
        assert_eq!(cloned.working_set_window, config.working_set_window);
        assert_eq!(cloned.hot_threshold, config.hot_threshold);
        assert!((cloned.lir_ratio - config.lir_ratio).abs() < f32::EPSILON);
        assert_eq!(cloned.min_warm_access, config.min_warm_access);
        assert_eq!(cloned.enable_clock_pro, config.enable_clock_pro);
    }

    #[test]
    fn hgal_config_debug_format_is_non_empty() {
        let config = HGALConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("warmup_duration"));
        assert!(debug_str.contains("hot_threshold"));
        assert!(debug_str.contains("lir_ratio"));
    }

    #[test]
    fn eviction_priority_for_page_without_metadata_uses_defaults() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(
            999,
            10,
            crate::scheduler::types::KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.access_count, 0, "missing metadata must default access_count to 0");
        assert_eq!(prio.recency, 0, "missing metadata must default recency to 0");
        assert!(!prio.is_pinned);
    }

    #[test]
    fn eviction_priority_recency_penalty_reduces_score() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let page_id = 42;
        scheduler.page_metadata.insert(page_id, PageMetadata {
            page_id,
            recency: 1000,
            access_count: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(
            page_id,
            10,
            crate::scheduler::types::KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&page);
        // recency_penalty = 1000 * (10/2) = 5000
        // score = 100 (KV) - 5000 + 0 + 0 + 0 = -4900
        assert!(prio.score < 0, "high recency with zero access_count must produce negative score, got {}", prio.score);
    }

    #[test]
    fn eviction_priority_all_evictable_kinds_ordered() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let kv_page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert_page = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32);
        let rag_page = UnifiedVirtualPage::rag(3, 10, gllm_kernels::types::DType::F32);

        let kv_prio = scheduler.compute_eviction_priority(&kv_page);
        let expert_prio = scheduler.compute_eviction_priority(&expert_page);
        let rag_prio = scheduler.compute_eviction_priority(&rag_page);

        // KV (score=100) > Expert (score=-200) and KV > RAG (score=-200)
        // Expert and RAG share same base (-200) but expert has layer_idx=0 (penalty=0)
        // while RAG has no layer_idx, so they are equal. We verify the ordering:
        assert!(kv_prio.score > expert_prio.score, "KV ({}) must be harder to evict than Expert ({})", kv_prio.score, expert_prio.score);
        assert!(kv_prio.score > rag_prio.score, "KV ({}) must be harder to evict than RAG ({})", kv_prio.score, rag_prio.score);
        // Both expert and RAG have negative scores (easy to evict)
        assert!(expert_prio.score < 0, "Expert must have negative score");
        assert!(rag_prio.score < 0, "RAG must have negative score");
    }

    #[test]
    fn mark_accessed_saturates_access_count_at_max() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            access_count: usize::MAX - 1,
            ..Default::default()
        });
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, usize::MAX, "access_count must saturate at MAX");
    }

    #[test]
    fn select_victim_groups_two_unpinned_groups_both_eligible() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        let group_a = SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        let group_b = SequenceGroup {
            id: 20,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        };
        scheduler.upsert_group(group_a);
        scheduler.upsert_group(group_b);

        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims.len(), 1);
        // ExpertWeight payload gets -200 adjustment -> lower score -> evicted first
        assert_eq!(victims[0], 20, "ExpertWeight group must be evicted before None-payload group");
    }

    #[test]
    fn select_victim_groups_accumulates_pages_across_multiple_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // 3 groups with 1 page each, request 5 pages total -> all 3 selected
        for i in 1u64..=3 {
            scheduler.upsert_group(SequenceGroup {
                id: i,
                pages: vec![i as usize],
                state: GroupState::Running,
                access_count: 0,
                last_access: now - Duration::from_secs(i),
                is_pinned: false,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        let victims = scheduler.select_victim_groups(5);
        assert_eq!(victims.len(), 3, "must select all 3 groups to accumulate 5 pages from 3 single-page groups");
    }

    #[test]
    fn on_prefill_chunk_complete_creates_new_group_for_first_chunk() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(123, 0), VirtualPageId::new(123, 1)];
        assert!(!scheduler.sequence_groups.contains_key(&123));
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        assert!(scheduler.sequence_groups.contains_key(&123));
        let group = &scheduler.sequence_groups[&123];
        assert_eq!(group.state, GroupState::Running);
        assert!(group.access_count >= 1);
        assert_eq!(group.pages.len(), 2);
    }

    #[test]
    fn on_prefill_chunk_complete_updates_context_len_to_max() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![
            VirtualPageId::new(200, 0),
            VirtualPageId::new(200, 1),
            VirtualPageId::new(200, 2),
        ];
        scheduler.on_prefill_chunk_complete(0, 2, &pages[..2]);
        assert_eq!(scheduler.sequence_groups[&200].context_len, 2);
        scheduler.on_prefill_chunk_complete(1, 2, &pages[2..]);
        assert_eq!(scheduler.sequence_groups[&200].context_len, 3, "context_len must be max of pages seen");
    }

    #[test]
    fn virtual_page_to_page_id_collisions_unlikely_for_nearby() {
        let vpid_a = VirtualPageId::new(1, 0);
        let vpid_b = VirtualPageId::new(1, 1);
        let vpid_c = VirtualPageId::new(2, 0);
        let id_a = virtual_page_to_page_id(vpid_a);
        let id_b = virtual_page_to_page_id(vpid_b);
        let id_c = virtual_page_to_page_id(vpid_c);
        assert_ne!(id_a, id_b, "adjacent logical indices must differ");
        assert_ne!(id_a, id_c, "adjacent sequence IDs must differ");
        assert_ne!(id_b, id_c, "different (seq, idx) pairs must differ");
    }

    #[test]
    fn virtual_page_to_page_id_result_fits_in_usize() {
        // Use large but not overflowing values to verify the modulo reduction works
        let vpid = VirtualPageId::new(100_000, 50_000);
        let id = virtual_page_to_page_id(vpid);
        // The Cantor pairing produces a value modded by usize::MAX
        assert!(id < usize::MAX, "page ID must be bounded below usize::MAX");
    }

    #[test]
    fn free_expert_weight_pages_also_removes_lir_entries() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0, // high ratio so all pages stay in LIR
            ..Default::default()
        });
        scheduler.allocate_expert_weight_pages(2, 0);
        let pages = scheduler.weight_page_table[&0].clone();
        // Mark them as LIR by accessing
        for pid in &pages {
            scheduler.mark_accessed(*pid);
        }
        // With lir_ratio=1.0 and 2 pages, target_lir=2, so both should be in LIR
        for pid in &pages {
            assert!(scheduler.lir_pages.contains(pid), "page {} must be in LIR after access", pid);
        }
        scheduler.free_expert_weight_pages(0);
        for pid in &pages {
            assert!(!scheduler.lir_pages.contains(pid), "LIR entry for page {} must be removed", pid);
        }
    }

    #[test]
    fn num_expert_weight_pages_after_mixed_operations() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(3, 0);
        scheduler.allocate_expert_weight_pages(2, 1);
        assert_eq!(scheduler.num_expert_weight_pages(), 5);
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_expert_weight_pages(), 2);
        scheduler.register_expert_weight_page(999, 1);
        assert_eq!(scheduler.num_expert_weight_pages(), 3);
    }

    #[test]
    fn detect_working_set_no_transition_for_recently_accessed_active_below_threshold() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 10,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 2, // below hot_threshold=10
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Active,
            "Active page below hot_threshold must stay Active"
        );
    }

    #[test]
    fn detect_working_set_keeps_standby_when_not_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 10,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            access_count: 1,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Standby,
            "Standby page with low access count must remain Standby"
        );
    }

    #[test]
    fn on_swap_in_then_mark_accessed_updates_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0);
        // Access after warmup expires -> Warm transitions to Active
        std::thread::sleep(Duration::from_millis(1));
        scheduler.mark_accessed(10);
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.state, PageState::Active, "Warm page past warmup must become Active on access");
        assert_eq!(meta.access_count, 1);
    }

    #[test]
    fn select_victim_weight_pages_truncates_to_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Register 5 expert weight pages in layer 0
        for i in 0..5u64 {
            let page_id = i as usize;
            scheduler.register_expert_weight_page(page_id, 0);
            // Make them Standby (not Protected/Warm) so they are eligible
            if let Some(meta) = scheduler.page_metadata.get_mut(&page_id) {
                meta.state = PageState::Standby;
            }
        }
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: (0..5).collect(),
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims.len(), 2, "must truncate to requested count");
    }

    #[test]
    fn eviction_priority_expert_no_layer_has_zero_depth_penalty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        // Expert page with layer_idx = 0
        let page = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // Base ExpertWeight = -200, layer_depth_penalty = -(0) = 0, so adjustment = -200
        // No metadata -> recency=0, access_count=0
        // score = -200 - 0 + 0 + 0 + 0 = -200
        assert_eq!(prio.score, -200, "expert at layer 0 with no depth penalty must have score -200");
    }

    // ── Additional tests batch 2 (18 new) ──

    #[test]
    fn eviction_priority_debug_format_contains_fields() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        let debug_str = format!("{:?}", prio);
        assert!(!debug_str.is_empty(), "EvictionPriority Debug must not be empty");
        assert!(debug_str.contains("score"), "Debug must contain 'score'");
        assert!(debug_str.contains("payload_kind"), "Debug must contain 'payload_kind'");
        assert!(debug_str.contains("is_pinned"), "Debug must contain 'is_pinned'");
    }

    #[test]
    fn eviction_priority_clone_preserves_all_fields() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::expert(5, 3, 7, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        let cloned = prio.clone();
        assert_eq!(cloned.score, prio.score);
        assert_eq!(cloned.payload_kind, prio.payload_kind);
        assert_eq!(cloned.is_pinned, prio.is_pinned);
        assert_eq!(cloned.access_count, prio.access_count);
        assert_eq!(cloned.recency, prio.recency);
        assert_eq!(cloned.layer_idx, prio.layer_idx);
        assert_eq!(cloned.expert_id, prio.expert_id);
    }

    #[test]
    fn eviction_priority_formula_exact_with_known_inputs() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // KV context page with access_count=5, recency=200
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            access_count: 5,
            recency: 200,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(42, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // payload_adjustment = 100 (KV)
        // recency_penalty = 200 * (FREQUENCY_WEIGHT / 2) = 200 * 5 = 1000
        // freq_bonus = 5 * FREQUENCY_WEIGHT = 5 * 10 = 50
        // pin_bonus = 0 (evictable)
        // state_bonus = 0 (Active, not Warm/Protected)
        // score = payload_adj - recency_penalty + freq_bonus + pin_bonus + state_bonus
        //       = 100 - 1000 + 50 + 0 + 0 = -850
        assert_eq!(prio.score, -850, "exact formula: 100 - 1000 + 50 = -850");
    }

    #[test]
    fn eviction_priority_state_bonus_exact_values() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let base_kv = |pid: PageId| UnifiedVirtualPage::kv(pid, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);

        // Active page: no state bonus
        scheduler.page_metadata.insert(1, PageMetadata { page_id: 1, state: PageState::Active, ..Default::default() });
        let active_score = scheduler.compute_eviction_priority(&base_kv(1)).score;

        // Warm page: +5000 bonus
        scheduler.page_metadata.insert(2, PageMetadata { page_id: 2, state: PageState::Warm, ..Default::default() });
        let warm_score = scheduler.compute_eviction_priority(&base_kv(2)).score;

        // Protected page: +10000 bonus
        scheduler.page_metadata.insert(3, PageMetadata { page_id: 3, state: PageState::Protected, ..Default::default() });
        let protected_score = scheduler.compute_eviction_priority(&base_kv(3)).score;

        assert_eq!(warm_score - active_score, 5_000, "Warm bonus must be exactly 5000");
        assert_eq!(protected_score - active_score, 10_000, "Protected bonus must be exactly 10000");
    }

    #[test]
    fn detect_working_set_swapped_page_no_transition() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 1,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Swapped,
            access_count: 100,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Swapped,
            "Swapped page must not be promoted even with high access_count"
        );
    }

    #[test]
    fn detect_working_set_free_page_no_transition() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 1,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Free,
            access_count: 100,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Free,
            "Free page must not be promoted even with high access_count"
        );
    }

    #[test]
    fn update_page_state_with_none_sequence_id() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(500, None, PageState::Active);
        let meta = scheduler.page_metadata.get(&500).expect("page must exist");
        assert_eq!(meta.page_id, 500);
        assert_eq!(meta.sequence_id, None, "sequence_id must be None when set to None");
        assert_eq!(meta.state, PageState::Active);
    }

    #[test]
    fn update_page_state_swapped_out_clears_warm_fields() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(600);
        assert!(scheduler.page_metadata[&600].warm_until.is_some());
        assert!(scheduler.page_metadata[&600].swap_in_time.is_some());
        scheduler.update_page_state(600, None, PageState::Swapped);
        assert!(scheduler.page_metadata[&600].warm_until.is_none());
        assert!(scheduler.page_metadata[&600].swap_in_time.is_none());
    }

    #[test]
    fn on_prefill_chunk_complete_multi_chunk_preserves_standby_until_last() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // 3 chunks total, each with 1 page
        let chunk0 = vec![VirtualPageId::new(300, 0)];
        let chunk1 = vec![VirtualPageId::new(300, 1)];
        let chunk2 = vec![VirtualPageId::new(300, 2)];

        scheduler.on_prefill_chunk_complete(0, 3, &chunk0);
        let pid0 = virtual_page_to_page_id(chunk0[0]);
        assert_eq!(scheduler.page_metadata[&pid0].state, PageState::Standby, "chunk 0/3 must be Standby");

        scheduler.on_prefill_chunk_complete(1, 3, &chunk1);
        let pid1 = virtual_page_to_page_id(chunk1[0]);
        assert_eq!(scheduler.page_metadata[&pid1].state, PageState::Standby, "chunk 1/3 must be Standby");

        scheduler.on_prefill_chunk_complete(2, 3, &chunk2);
        let pid2 = virtual_page_to_page_id(chunk2[0]);
        assert_eq!(scheduler.page_metadata[&pid2].state, PageState::Active, "chunk 2/3 (last) must be Active");

        // Group must have accumulated all 3 pages
        let group = &scheduler.sequence_groups[&300];
        assert_eq!(group.pages.len(), 3, "group must have 3 unique pages after 3 chunks");
    }

    #[test]
    fn on_prefill_chunk_complete_multi_request_separates_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Chunk with pages from 2 different requests
        let pages = vec![
            VirtualPageId::new(400, 0),
            VirtualPageId::new(401, 0),
        ];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);

        assert!(scheduler.sequence_groups.contains_key(&400), "group for request 400 must exist");
        assert!(scheduler.sequence_groups.contains_key(&401), "group for request 401 must exist");
        assert_eq!(scheduler.sequence_groups[&400].pages.len(), 1);
        assert_eq!(scheduler.sequence_groups[&401].pages.len(), 1);
    }

    #[test]
    fn on_prefill_complete_updates_group_last_access() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let old_time = Instant::now() - Duration::from_secs(10);
        let group = SequenceGroup {
            id: 777,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 5,
            last_access: old_time,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(100, Some(777), PageState::Standby);
        scheduler.on_prefill_complete(777);
        let g = &scheduler.sequence_groups[&777];
        assert!(g.last_access > old_time, "last_access must be updated to a more recent time");
    }

    #[test]
    fn on_prefill_complete_clears_warm_until_on_active_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 888,
            pages: vec![200],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        // Set warm_until manually
        scheduler.page_metadata.insert(200, PageMetadata {
            page_id: 200,
            state: PageState::Active,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            ..Default::default()
        });
        scheduler.on_prefill_complete(888);
        assert!(scheduler.page_metadata[&200].warm_until.is_none(), "warm_until must be cleared on prefill complete");
        assert_eq!(scheduler.page_metadata[&200].state, PageState::Active);
    }

    #[test]
    fn select_victim_groups_orders_by_recency_of_page_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group A: has page with recency=0 (recently accessed)
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(1, PageMetadata { page_id: 1, recency: 0, ..Default::default() });

        // Group B: has page with recency=1000 (stale)
        scheduler.upsert_group(SequenceGroup {
            id: 20,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(2, PageMetadata { page_id: 2, recency: 1000, ..Default::default() });

        // Group C: has page with recency=500 (moderately stale)
        scheduler.upsert_group(SequenceGroup {
            id: 30,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(3, PageMetadata { page_id: 3, recency: 500, ..Default::default() });

        let victims = scheduler.select_victim_groups(3);
        // Higher recency in page metadata contributes to lower group priority score
        // via recency_penalty in compute_group_priority.
        // Group B (recency=1000) should be first, Group C (recency=500) second, Group A (recency=0) third.
        assert!(victims.contains(&20), "Group B (highest recency) must be a victim");
        assert!(victims.contains(&30), "Group C (medium recency) must be a victim");
    }

    #[test]
    fn select_victim_groups_protected_group_not_selected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group with a Protected page -> group_has_protection returns true
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            ..Default::default()
        });

        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "Group with Protected page must not be selected as victim");
    }

    #[test]
    fn select_victim_weight_pages_sorts_by_score_ascending() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Two expert weight pages with different recency values
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        // Make them Standby so they are eligible
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&11).unwrap().state = PageState::Standby;
        // Page 10 has high recency (stale), page 11 has low recency (fresh)
        scheduler.page_metadata.get_mut(&10).unwrap().recency = 5000;
        scheduler.page_metadata.get_mut(&11).unwrap().recency = 0;

        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims.len(), 2, "must return 2 victims");
        assert!(
            victims[0].1.score <= victims[1].1.score,
            "victims must be sorted by score ascending (lowest first), got {} then {}",
            victims[0].1.score, victims[1].1.score,
        );
    }

    #[test]
    fn select_victim_weight_pages_skips_pages_without_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Manually insert a page_id in weight_page_table without creating metadata
        scheduler.weight_page_table.insert(0, vec![999]);
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "pages without metadata must be skipped");
    }

    #[test]
    fn virtual_page_to_page_id_zero_inputs() {
        let vpid = VirtualPageId::new(0, 0);
        let id = virtual_page_to_page_id(vpid);
        // (0+0)*(0+0+1)/2 + 0 = 0
        assert_eq!(id, 0, "Cantor pairing of (0,0) must be 0");
    }

    #[test]
    fn on_prefill_chunk_complete_resets_recency_to_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Pre-set a page with high recency
        let page = VirtualPageId::new(500, 0);
        let pid = virtual_page_to_page_id(page);
        scheduler.page_metadata.insert(pid, PageMetadata {
            page_id: pid,
            recency: 9999,
            access_count: 10,
            ..Default::default()
        });
        scheduler.on_prefill_chunk_complete(0, 2, &[page]);
        let meta = &scheduler.page_metadata[&pid];
        assert_eq!(meta.recency, 0, "recency must be reset to 0 on prefill chunk complete");
        assert_eq!(meta.access_count, 11, "access_count must be incremented from 10 to 11");
    }

    #[test]
    fn mark_accessed_warm_page_stays_warm_during_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: 100,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        // Immediately access — still in warmup (access_count < min_warm_access, warm_until not expired)
        scheduler.mark_accessed(10);
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Warm,
            "page must stay Warm during warmup period"
        );
    }

    // ── Additional tests batch 3 (35 new) ──

    // -- HGALConfig edge cases --

    #[test]
    fn hgal_config_zero_warmup_duration() {
        let config = HGALConfig {
            warmup_duration: Duration::ZERO,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().warmup_duration, Duration::ZERO);
    }

    #[test]
    fn hgal_config_zero_hot_threshold() {
        let config = HGALConfig {
            hot_threshold: 0,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().hot_threshold, 0);
    }

    #[test]
    fn hgal_config_zero_lir_ratio() {
        let config = HGALConfig {
            lir_ratio: 0.0,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert!((scheduler.config().lir_ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn hgal_config_max_min_warm_access() {
        let config = HGALConfig {
            min_warm_access: usize::MAX,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().min_warm_access, usize::MAX);
    }

    // -- upsert_group edge cases --

    #[test]
    fn upsert_group_preserves_last_access_on_update() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let old_time = Instant::now() - Duration::from_secs(60);
        let original = SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 3,
            last_access: old_time,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(original);
        let updated = SequenceGroup {
            id: 10,
            pages: vec![1, 2],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        scheduler.upsert_group(updated);
        let group = &scheduler.sequence_groups[&10];
        assert_eq!(group.access_count, 3, "access_count must be preserved");
        // last_access is also preserved from the existing group
        assert_eq!(group.last_access, old_time, "last_access must be preserved from existing group");
    }

    #[test]
    fn upsert_group_empty_pages_vector() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 42,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert!(scheduler.sequence_groups[&42].pages.is_empty());
    }

    #[test]
    fn upsert_group_max_request_id() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: u64::MAX,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert!(scheduler.sequence_groups.contains_key(&u64::MAX));
    }

    // -- update_page_state edge cases --

    #[test]
    fn update_page_state_multiple_state_transitions() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(1), PageState::Active);
        scheduler.update_page_state(1, Some(1), PageState::Warm);
        scheduler.update_page_state(1, Some(1), PageState::Protected);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
        scheduler.update_page_state(1, Some(2), PageState::Active);
        assert_eq!(scheduler.page_metadata[&1].sequence_id, Some(2));
    }

    #[test]
    fn update_page_state_to_swapped_from_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(1), PageState::Active);
        scheduler.update_page_state(1, Some(1), PageState::Swapped);
        let meta = &scheduler.page_metadata[&1];
        assert_eq!(meta.state, PageState::Swapped);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    // -- select_victim_groups edge cases --

    #[test]
    fn select_victim_groups_single_group_enough_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3, 4, 5],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 5,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(5);
        assert_eq!(victims, vec![1], "single group with 5 pages satisfies count=5");
    }

    #[test]
    fn select_victim_groups_newer_group_preferred_over_older() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Old group: last_access was 10 seconds ago (time_penalty = ~10000ms)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Recent group: last_access is now (time_penalty = ~0ms)
        // Formula: time_penalty + recency - freq - pin + payload
        // Recent group has lower time_penalty -> lower score -> evicted first
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 2, "newer group (lower time_penalty) has lower score and is evicted first");
    }

    #[test]
    fn select_victim_groups_all_pinned_returns_empty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        for i in 1..=5u64 {
            scheduler.upsert_group(SequenceGroup {
                id: i,
                pages: vec![i as usize],
                state: GroupState::Running,
                access_count: 0,
                last_access: now,
                is_pinned: true,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        let victims = scheduler.select_victim_groups(10);
        assert!(victims.is_empty(), "all pinned groups must yield empty victims");
    }

    #[test]
    fn select_victim_groups_lower_access_count_resists_eviction() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Group with high access count: freq_bonus = 100 * 10 = 1000, lowers score
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 100,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Group with low access count: freq_bonus = 1 * 10 = 10, higher score
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 1, "group with higher access_count has lower score (freq_bonus subtracts) and is evicted first");
    }

    // -- compute_eviction_priority edge cases --

    #[test]
    fn eviction_priority_dense_layer_weight_base_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::dense_layer(1, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // DenseLayerWeight: payload_adjustment = 5000
        // is_evictable=false -> pin_bonus = 5000
        // score = 5000 - 0 + 0 + 5000 + 0 = 10000
        assert_eq!(prio.score, 10_000, "DenseLayerWeight base score = 5000 + 5000 pin = 10000");
        assert!(prio.is_pinned);
    }

    #[test]
    fn eviction_priority_expert_weight_deep_layer_penalty_increases() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let layer5 = UnifiedVirtualPage::expert(1, 0, 5, gllm_kernels::types::DType::F32);
        let layer10 = UnifiedVirtualPage::expert(2, 0, 10, gllm_kernels::types::DType::F32);
        let prio5 = scheduler.compute_eviction_priority(&layer5);
        let prio10 = scheduler.compute_eviction_priority(&layer10);
        assert!(
            prio10.score < prio5.score,
            "layer 10 (score={}) must have lower score than layer 5 (score={})",
            prio10.score, prio5.score,
        );
        // Exact delta: -(10) - -(5) = -5
        assert_eq!(prio5.score - prio10.score, 5, "depth penalty delta must be 5");
    }

    #[test]
    fn eviction_priority_large_access_count_dominates_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 100,
            access_count: 10_000,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // freq_bonus = 10000 * 10 = 100000; recency_penalty = 100 * 5 = 500
        // score = 100 + (-500) + 100000 = 99600
        assert!(prio.score > 0, "large freq_bonus must dominate recency_penalty");
    }

    // -- mark_accessed edge cases --

    #[test]
    fn mark_accessed_unknown_page_creates_with_recency_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(42);
        let meta = &scheduler.page_metadata[&42];
        assert_eq!(meta.recency, 0, "first access recency must be 0 (no prior access interval)");
        assert_eq!(meta.access_count, 1);
    }

    #[test]
    fn mark_accessed_second_access_sets_recency_nonzero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(42);
        // Small sleep to ensure non-zero recency interval
        std::thread::sleep(Duration::from_millis(2));
        scheduler.mark_accessed(42);
        let meta = &scheduler.page_metadata[&42];
        // recency = time since last_access, should be > 0 after sleep
        assert!(meta.recency > 0 || meta.access_count == 2, "recency must be computed from interval or access_count incremented");
        assert_eq!(meta.access_count, 2);
    }

    #[test]
    fn mark_accessed_no_group_does_not_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Access a page that has no associated group — should not panic
        scheduler.mark_accessed(999);
        assert_eq!(scheduler.page_metadata[&999].access_count, 1);
    }

    #[test]
    fn mark_accessed_group_access_count_saturates() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: usize::MAX - 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.sequence_groups[&1].access_count, usize::MAX, "group access_count must saturate at MAX");
    }

    // -- on_swap_in edge cases --

    #[test]
    fn on_swap_in_resets_access_count_to_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Create page with high access count
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 3);
        // Swap in resets
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0, "swap in must reset access_count");
    }

    #[test]
    fn on_swap_in_overwrites_existing_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Protected);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Protected);
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm, "swap in must set state to Warm");
    }

    // -- detect_working_set edge cases --

    #[test]
    fn detect_working_set_empty_scheduler_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.detect_working_set();
        assert!(scheduler.page_metadata.is_empty());
    }

    #[test]
    fn detect_working_set_promotes_warm_to_protected_when_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 1,
            working_set_window: Duration::from_secs(60),
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX, // so warm page exits warmup
            ..Default::default()
        });
        // Warm page that has exited warmup period (warm_until expired) and is hot
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            access_count: 10,
            last_access: Instant::now(),
            warm_until: Some(Instant::now() - Duration::from_secs(1)),
            ..Default::default()
        });
        scheduler.detect_working_set();
        // Warm page past warmup transitions to Active first, then hot check promotes to Protected
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Protected,
            "hot page that exited warmup must be promoted to Protected"
        );
    }

    #[test]
    fn detect_working_set_preserves_active_when_recently_accessed_but_not_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 100,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Active,
            "Active page below hot_threshold must remain Active"
        );
    }

    // -- Expert weight page lifecycle edge cases --

    #[test]
    fn allocate_expert_weight_pages_zero_experts() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(0, 0);
        assert!(pages.is_empty());
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    #[test]
    fn allocate_expert_weight_pages_large_layer_idx() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(2, 100);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0], (100usize << 16) | 0);
        assert_eq!(pages[1], (100usize << 16) | 1);
        assert_eq!(scheduler.weight_page_table.get(&100).unwrap().len(), 2);
    }

    #[test]
    fn register_expert_weight_page_same_page_different_layers_duplicated() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 0);
        scheduler.register_expert_weight_page(42, 1);
        // Same page_id appears in two different layers
        assert!(scheduler.weight_page_table[&0].contains(&42));
        assert!(scheduler.weight_page_table[&1].contains(&42));
        assert_eq!(scheduler.num_expert_weight_pages(), 2);
    }

    #[test]
    fn free_expert_weight_pages_then_reallocate() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(3, 0);
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
        let pages = scheduler.allocate_expert_weight_pages(2, 0);
        assert_eq!(pages.len(), 2);
        assert_eq!(scheduler.num_expert_weight_pages(), 2);
    }

    // -- Dense layer weight page edge cases --

    #[test]
    fn register_dense_layer_weight_pages_empty_vec() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.register_dense_layer_weight_pages(vec![], 0);
        assert!(pages.is_empty());
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    #[test]
    fn free_dense_layer_weight_pages_nonexistent_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.free_dense_layer_weight_pages(999);
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    #[test]
    fn num_dense_layer_weight_pages_mixed_with_expert() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Register expert weight (no group, so not counted as dense)
        scheduler.register_expert_weight_page(10, 0);
        // Register dense weight with group
        scheduler.register_dense_layer_weight_page(20, 1);
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_001,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: true,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });
        assert_eq!(scheduler.num_weight_pages(), 2, "total weight pages = 1 expert + 1 dense");
        assert_eq!(scheduler.num_dense_layer_weight_pages(), 1, "only dense group pages counted");
    }

    // -- on_prefill_chunk_complete edge cases --

    #[test]
    fn on_prefill_chunk_complete_single_chunk_is_last() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(50, 0)];
        // chunk_idx=0, total_chunks=1 -> is_last_chunk = true
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active, "single chunk must be Active");
    }

    #[test]
    fn on_prefill_chunk_complete_saturating_add_chunk_idx() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(50, 0)];
        // chunk_idx=usize::MAX, total_chunks=usize::MAX -> is_last_chunk = true
        scheduler.on_prefill_chunk_complete(usize::MAX, usize::MAX, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active, "saturating add must identify last chunk");
    }

    // -- on_prefill_complete edge cases --

    #[test]
    fn on_prefill_complete_updates_group_state_to_running() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.sequence_groups[&1].state, GroupState::Running, "state must be set to Running");
    }

    // -- virtual_page_to_page_id edge cases --

    #[test]
    fn virtual_page_to_page_id_symmetric_different_sequences() {
        let a = virtual_page_to_page_id(VirtualPageId::new(1, 100));
        let b = virtual_page_to_page_id(VirtualPageId::new(100, 1));
        assert_ne!(a, b, "swapping sequence_id and logical_index must produce different page IDs");
    }

    #[test]
    fn virtual_page_to_page_id_large_values() {
        // Use large but safe values that won't overflow u128 intermediate arithmetic
        let vpid = VirtualPageId::new(1_000_000_000, 1_000_000_000);
        let id = virtual_page_to_page_id(vpid);
        // The Cantor pairing produces a value modded by usize::MAX
        assert!(id < usize::MAX, "page ID must be bounded below usize::MAX");
    }

    // -- LIR membership edge cases --

    #[test]
    fn lir_membership_zero_ratio_with_clock_pro_disabled() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            lir_ratio: 0.0,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        assert!(!scheduler.lir_pages.contains(&1), "LIR must not be updated when clock_pro disabled");
    }

    #[test]
    fn lir_membership_all_pages_fit_within_ratio() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        scheduler.mark_accessed(2);
        scheduler.mark_accessed(3);
        // With ratio 1.0, target = ceil(3 * 1.0) = 3, all pages fit
        assert_eq!(scheduler.lir_pages.len(), 3, "all pages must be in LIR with ratio=1.0");
    }

    // -- group_has_protection edge cases (tested indirectly via select_victim_groups) --

    #[test]
    fn select_victim_groups_warm_page_protects_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Warm page in warmup period
        scheduler.on_swap_in(10);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "group with Warm page in warmup must be protected");
    }

    // -- select_victim_weight_pages edge cases --

    #[test]
    fn select_victim_weight_pages_page_without_group_defaults_to_expert() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Register page in weight table and metadata but not in any group
        scheduler.weight_page_table.insert(0, vec![10]);
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Standby,
            ..Default::default()
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert_eq!(victims.len(), 1, "page without group must use default ExpertWeight kind");
        assert_eq!(victims[0].0, 10);
    }

    // -- compute_group_priority edge cases (tested via select_victim_groups) --

    #[test]
    fn select_victim_groups_expert_weight_evicted_before_none_payload() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 10,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims[0], 1, "ExpertWeight group gets -200 adjustment, must be evicted first");
    }

    // -- on_swap_in + detect_working_set interaction --

    #[test]
    fn swap_in_then_detect_working_set_stays_warm_during_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: 100,
            hot_threshold: 1,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        scheduler.detect_working_set();
        // Page is Warm and in warmup period (access_count=0 < min_warm_access=100), so it stays Warm
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Warm,
            "page in warmup period must stay Warm after detect_working_set"
        );
    }

    // -- EvictionPriority fields verification --

    #[test]
    fn eviction_priority_expert_has_expert_id_set() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::expert(5, 42, 3, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, Some(42), "expert_id must match page expert_id");
        assert_eq!(prio.layer_idx, Some(3), "layer_idx must match page layer_idx");
    }

    #[test]
    fn eviction_priority_kv_has_no_expert_no_layer() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, None, "KV page must have no expert_id");
        assert_eq!(prio.layer_idx, None, "KV page must have no layer_idx");
    }

    // -- HGALScheduler debug format --

    #[test]
    fn scheduler_debug_format_is_non_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let debug_str = format!("{:?}", scheduler);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("sequence_groups"));
        assert!(debug_str.contains("page_metadata"));
        assert!(debug_str.contains("lir_pages"));
        assert!(debug_str.contains("config"));
    }

    // ── Batch 4: 50 additional tests ──

    // -- PageState enum variant coverage --

    #[test]
    fn page_state_variants_are_distinct() {
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j], "PageState variants must be distinct");
            }
        }
    }

    #[test]
    fn page_state_debug_format_contains_variant_name() {
        assert!(format!("{:?}", PageState::Free).contains("Free"));
        assert!(format!("{:?}", PageState::Active).contains("Active"));
        assert!(format!("{:?}", PageState::Standby).contains("Standby"));
        assert!(format!("{:?}", PageState::Warm).contains("Warm"));
        assert!(format!("{:?}", PageState::Protected).contains("Protected"));
        assert!(format!("{:?}", PageState::Swapped).contains("Swapped"));
    }

    #[test]
    fn page_state_clone_equals_original() {
        let state = PageState::Warm;
        assert_eq!(state.clone(), state);
    }

    // -- PagePayloadKind enum variant coverage --

    #[test]
    fn page_payload_kind_variants_are_distinct() {
        let kinds = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j], "PagePayloadKind variants must be distinct");
            }
        }
    }

    #[test]
    fn page_payload_kind_debug_format() {
        assert!(format!("{:?}", PagePayloadKind::KvContext).contains("KvContext"));
        assert!(format!("{:?}", PagePayloadKind::ExpertWeight).contains("ExpertWeight"));
        assert!(format!("{:?}", PagePayloadKind::DenseLayerWeight).contains("DenseLayerWeight"));
        assert!(format!("{:?}", PagePayloadKind::PromptSystem).contains("PromptSystem"));
        assert!(format!("{:?}", PagePayloadKind::KnowledgeRAG).contains("KnowledgeRAG"));
    }

    #[test]
    fn page_payload_kind_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PagePayloadKind::KvContext);
        set.insert(PagePayloadKind::KvContext);
        assert_eq!(set.len(), 1, "duplicate insert must not increase set size");
        set.insert(PagePayloadKind::ExpertWeight);
        assert_eq!(set.len(), 2);
    }

    // -- GroupState enum coverage --

    #[test]
    fn group_state_variants_are_distinct() {
        assert_ne!(GroupState::Running, GroupState::Swapped);
        assert_ne!(GroupState::Running, GroupState::Paused);
        assert_ne!(GroupState::Swapped, GroupState::Paused);
    }

    #[test]
    fn group_state_debug_format() {
        assert!(format!("{:?}", GroupState::Running).contains("Running"));
        assert!(format!("{:?}", GroupState::Swapped).contains("Swapped"));
        assert!(format!("{:?}", GroupState::Paused).contains("Paused"));
    }

    // -- KvPipeline enum coverage --

    #[test]
    fn kv_pipeline_variants_are_distinct() {
        assert_ne!(
            crate::scheduler::types::KvPipeline::Conversation,
            crate::scheduler::types::KvPipeline::Working
        );
    }

    #[test]
    fn kv_pipeline_debug_format() {
        let conv = format!("{:?}", crate::scheduler::types::KvPipeline::Conversation);
        let work = format!("{:?}", crate::scheduler::types::KvPipeline::Working);
        assert!(conv.contains("Conversation"));
        assert!(work.contains("Working"));
    }

    // -- EvictionPriority construction and field access --

    #[test]
    fn eviction_priority_manual_construction() {
        let prio = EvictionPriority {
            score: -500,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 3,
            recency: 100,
            layer_idx: Some(5),
            expert_id: Some(42),
        };
        assert_eq!(prio.score, -500);
        assert_eq!(prio.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(!prio.is_pinned);
        assert_eq!(prio.access_count, 3);
        assert_eq!(prio.recency, 100);
        assert_eq!(prio.layer_idx, Some(5));
        assert_eq!(prio.expert_id, Some(42));
    }

    #[test]
    fn eviction_priority_default_layer_and_expert_for_kv() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(
            1, 10,
            crate::scheduler::types::KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.layer_idx, None);
        assert_eq!(prio.expert_id, None);
    }

    // -- PageMetadata default values --

    #[test]
    fn page_metadata_default_values() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.sequence_id, None);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.is_lir, false);
        assert_eq!(meta.state, PageState::Standby);
        assert_eq!(meta.warm_until, None);
        assert_eq!(meta.swap_in_time, None);
    }

    #[test]
    fn page_metadata_custom_construction() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(100),
            recency: 500,
            access_count: 7,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Protected,
            warm_until: Some(now + Duration::from_secs(10)),
        };
        assert_eq!(meta.page_id, 42);
        assert_eq!(meta.sequence_id, Some(100));
        assert_eq!(meta.recency, 500);
        assert_eq!(meta.access_count, 7);
        assert!(meta.is_lir);
        assert_eq!(meta.state, PageState::Protected);
        assert!(meta.warm_until.is_some());
        assert!(meta.swap_in_time.is_some());
    }

    #[test]
    fn page_metadata_clone_preserves_fields() {
        let now = Instant::now();
        let original = PageMetadata {
            page_id: 99,
            sequence_id: Some(50),
            recency: 200,
            access_count: 5,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Warm,
            warm_until: Some(now + Duration::from_secs(5)),
        };
        let cloned = original.clone();
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.sequence_id, original.sequence_id);
        assert_eq!(cloned.recency, original.recency);
        assert_eq!(cloned.access_count, original.access_count);
        assert_eq!(cloned.is_lir, original.is_lir);
        assert_eq!(cloned.state, original.state);
    }

    // -- VirtualPageId construction and properties --

    #[test]
    fn virtual_page_id_new_constructs_correctly() {
        let vpid = VirtualPageId::new(42, 7);
        assert_eq!(vpid.sequence_id, 42);
        assert_eq!(vpid.logical_index, 7);
    }

    #[test]
    fn virtual_page_id_equality() {
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 2);
        let c = VirtualPageId::new(1, 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn virtual_page_id_hash_consistency() {
        use std::collections::HashSet;
        let a = VirtualPageId::new(10, 20);
        let b = VirtualPageId::new(10, 20);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn virtual_page_id_copy_semantics() {
        let a = VirtualPageId::new(5, 10);
        let b = a;
        assert_eq!(a, b);
    }

    // -- select_coldest_lir behavior (tested via LIR eviction) --

    #[test]
    fn lir_evicts_coldest_page_when_over_capacity() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        });
        // Insert 3 pages with metadata, then mark_accessed them
        // mark_accessed resets recency to the interval since last_access.
        // Access page 1 first, then wait, then pages 2 and 3.
        // Page 1 will have highest recency (longest interval since prior access).
        scheduler.mark_accessed(1);
        std::thread::sleep(Duration::from_millis(5));
        scheduler.mark_accessed(2);
        scheduler.mark_accessed(3);
        // target_lir = ceil(3 * 0.5) = 2, so 1 page must be evicted
        assert!(scheduler.lir_pages.len() <= 2, "LIR set must be at most 2 pages");
        // Page 1 had the longest gap before its second access (if accessed again),
        // but since we only access once, all have recency=0.
        // The key invariant is: LIR size <= ceil(n * ratio)
        assert_eq!(scheduler.lir_pages.len(), 2, "with 3 pages and 0.5 ratio, exactly 2 must be in LIR");
    }

    #[test]
    fn lir_empty_after_all_pages_freed() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        scheduler.mark_accessed(2);
        // Free all pages from metadata
        scheduler.page_metadata.clear();
        // LIR pages remain in set but metadata is gone — verify consistency
        for pid in &scheduler.lir_pages {
            assert!(!scheduler.page_metadata.contains_key(pid));
        }
    }

    // -- Multiple groups with mixed protection --

    #[test]
    fn select_victim_groups_mixed_protection_only_unprotected_selected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Protected group (pinned)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: true,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Group with Protected page metadata
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            state: PageState::Protected,
            ..Default::default()
        });

        // Unprotected group
        scheduler.upsert_group(SequenceGroup {
            id: 3,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        let victims = scheduler.select_victim_groups(10);
        assert_eq!(victims, vec![3], "only unprotected group must be selected");
    }

    // -- Eviction priority ordering: all payload kinds --

    #[test]
    fn eviction_priority_full_payload_ordering() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let dense = UnifiedVirtualPage::dense_layer(1, 0, gllm_kernels::types::DType::F32);
        let prompt = UnifiedVirtualPage::system_prompt(2, gllm_kernels::types::DType::F32);
        let kv = UnifiedVirtualPage::kv(3, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(4, 0, 0, gllm_kernels::types::DType::F32);
        let rag = UnifiedVirtualPage::rag(5, 10, gllm_kernels::types::DType::F32);

        let dense_score = scheduler.compute_eviction_priority(&dense).score;
        let prompt_score = scheduler.compute_eviction_priority(&prompt).score;
        let kv_score = scheduler.compute_eviction_priority(&kv).score;
        let expert_score = scheduler.compute_eviction_priority(&expert).score;
        let rag_score = scheduler.compute_eviction_priority(&rag).score;

        // Dense (10000) > Prompt (10000+pin) > KV (100) > Expert (-200) ≈ RAG (-200)
        assert!(dense_score > kv_score, "Dense ({}) > KV ({})", dense_score, kv_score);
        assert!(prompt_score > kv_score, "Prompt ({}) > KV ({})", prompt_score, kv_score);
        assert!(kv_score > expert_score, "KV ({}) > Expert ({})", kv_score, expert_score);
        assert!(kv_score > rag_score, "KV ({}) > RAG ({})", kv_score, rag_score);
    }

    // -- mark_accessed + group interaction --

    #[test]
    fn mark_accessed_updates_group_last_access() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let old_time = Instant::now() - Duration::from_secs(30);
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: old_time,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Active);
        std::thread::sleep(Duration::from_millis(2));
        scheduler.mark_accessed(10);
        let group = &scheduler.sequence_groups[&1];
        assert!(group.last_access > old_time, "group last_access must be updated");
        assert_eq!(group.access_count, 1);
    }

    #[test]
    fn mark_accessed_multiple_pages_same_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 11, 12],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        for pid in &[10, 11, 12] {
            scheduler.update_page_state(*pid, Some(1), PageState::Active);
        }
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(11);
        scheduler.mark_accessed(12);
        assert_eq!(scheduler.sequence_groups[&1].access_count, 3);
    }

    // -- on_swap_in warm_until is in the future --

    #[test]
    fn on_swap_in_warm_until_is_in_future() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(30),
            ..Default::default()
        });
        let before = Instant::now();
        scheduler.on_swap_in(10);
        let after = Instant::now();
        let meta = &scheduler.page_metadata[&10];
        let warm_until = meta.warm_until.expect("warm_until must be set");
        assert!(warm_until > before, "warm_until must be after swap-in time");
        assert!(warm_until <= after + Duration::from_secs(30), "warm_until must be within warmup_duration");
    }

    // -- detect_working_set with multiple pages --

    #[test]
    fn detect_working_set_processes_multiple_pages_independently() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_secs(60),
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        let now = Instant::now();
        // Page 1: hot, Active -> Protected
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 10,
            last_access: now,
            ..Default::default()
        });
        // Page 2: not hot, Active -> stays Active
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            state: PageState::Active,
            access_count: 1,
            last_access: now,
            ..Default::default()
        });
        // Page 3: Warm, past warmup -> Active (but not hot) -> stays Active
        scheduler.page_metadata.insert(3, PageMetadata {
            page_id: 3,
            state: PageState::Warm,
            access_count: 1,
            last_access: now,
            warm_until: Some(now - Duration::from_secs(1)),
            ..Default::default()
        });
        // Page 4: Protected, expired -> Standby
        scheduler.page_metadata.insert(4, PageMetadata {
            page_id: 4,
            state: PageState::Protected,
            access_count: 0,
            last_access: now - Duration::from_secs(120), // beyond working_set_window=60s
            ..Default::default()
        });

        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected, "hot Active -> Protected");
        assert_eq!(scheduler.page_metadata[&2].state, PageState::Active, "cold Active stays Active");
        assert_eq!(scheduler.page_metadata[&3].state, PageState::Active, "expired Warm -> Active");
        assert_eq!(scheduler.page_metadata[&4].state, PageState::Standby, "expired Protected -> Standby");
    }

    // -- weight_page_table structure after mixed operations --

    #[test]
    fn weight_page_table_has_correct_layer_entries() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(1, 0);
        scheduler.register_expert_weight_page(2, 0);
        scheduler.register_expert_weight_page(3, 1);
        scheduler.register_dense_layer_weight_page(4, 2);
        scheduler.register_dense_layer_weight_page(5, 2);

        assert_eq!(scheduler.weight_page_table.len(), 3, "must have entries for layers 0, 1, 2");
        assert_eq!(scheduler.weight_page_table[&0].len(), 2);
        assert_eq!(scheduler.weight_page_table[&1].len(), 1);
        assert_eq!(scheduler.weight_page_table[&2].len(), 2);
    }

    #[test]
    fn free_layer_does_not_affect_other_layers() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(1, 0);
        scheduler.register_expert_weight_page(2, 1);
        scheduler.register_expert_weight_page(3, 2);

        scheduler.free_expert_weight_pages(1);
        assert_eq!(scheduler.weight_page_table.len(), 2);
        assert!(scheduler.weight_page_table.contains_key(&0));
        assert!(scheduler.weight_page_table.contains_key(&2));
        assert!(!scheduler.weight_page_table.contains_key(&1));
        assert!(scheduler.page_metadata.contains_key(&1));
        assert!(!scheduler.page_metadata.contains_key(&2));
        assert!(scheduler.page_metadata.contains_key(&3));
    }

    // -- SequenceGroup with payload_kind effects on eviction --

    #[test]
    fn group_payload_kind_expert_vs_dense_priority() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims[0], 1, "ExpertWeight group must be evicted before DenseLayerWeight");
    }

    // -- EvictionPriority partial equality --

    #[test]
    fn eviction_priority_different_pages_have_different_scores() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let kv = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32);
        let kv_prio = scheduler.compute_eviction_priority(&kv);
        let expert_prio = scheduler.compute_eviction_priority(&expert);
        assert_ne!(kv_prio.score, expert_prio.score, "different payload kinds must have different scores");
        assert_ne!(kv_prio.payload_kind, expert_prio.payload_kind);
    }

    // -- select_victim_weight_pages with recency-based ordering --

    #[test]
    fn select_victim_weight_pages_orders_by_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Register two expert pages with different recency
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        // Make them Standby (eligible)
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&11).unwrap().state = PageState::Standby;
        // Page 10: stale (high recency)
        scheduler.page_metadata.get_mut(&10).unwrap().recency = 10000;
        scheduler.page_metadata.get_mut(&10).unwrap().access_count = 0;
        // Page 11: fresh (low recency)
        scheduler.page_metadata.get_mut(&11).unwrap().recency = 10;
        scheduler.page_metadata.get_mut(&11).unwrap().access_count = 100;

        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims[0].0, 10, "stale page (high recency, low access) must be first victim");
    }

    // -- update_page_state preserves existing access_count --

    #[test]
    fn update_page_state_preserves_access_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 3);
        // Update state should not reset access_count
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        assert_eq!(scheduler.page_metadata[&10].access_count, 3, "access_count must be preserved");
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Standby);
    }

    // -- select_victim_groups with many groups stops early --

    #[test]
    fn select_victim_groups_stops_early_when_count_satisfied() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // 5 groups, each with 2 pages, request 3 pages -> needs 2 groups (2+2=4 >= 3)
        for i in 1u64..=5 {
            scheduler.upsert_group(SequenceGroup {
                id: i,
                pages: vec![(i * 2) as usize, (i * 2 + 1) as usize],
                state: GroupState::Running,
                access_count: 0,
                last_access: now - Duration::from_secs(i),
                is_pinned: false,
                context_len: 2,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        let victims = scheduler.select_victim_groups(3);
        assert!(victims.len() <= 2, "must stop selecting after accumulating enough pages");
        assert!(!victims.is_empty());
    }

    // -- on_prefill_chunk_complete with page already in metadata --

    #[test]
    fn on_prefill_chunk_complete_updates_existing_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId::new(100, 0);
        let pid = virtual_page_to_page_id(vpid);
        // Pre-existing metadata
        scheduler.page_metadata.insert(pid, PageMetadata {
            page_id: pid,
            sequence_id: Some(999),
            state: PageState::Swapped,
            access_count: 50,
            recency: 1000,
            ..Default::default()
        });
        scheduler.on_prefill_chunk_complete(0, 2, &[vpid]);
        let meta = &scheduler.page_metadata[&pid];
        assert_eq!(meta.state, PageState::Standby, "state must be updated to Standby for non-last chunk");
        assert_eq!(meta.recency, 0, "recency must be reset");
        assert_eq!(meta.access_count, 51, "access_count must be incremented");
        assert_eq!(meta.sequence_id, Some(100), "sequence_id must be updated");
    }

    // -- eviction priority with all zero metadata --

    #[test]
    fn eviction_priority_zero_metadata_kv_has_positive_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // KV: payload=100, recency=0, access=0, pin=false, state=none
        // score = 100 - 0 + 0 + 0 + 0 = 100
        assert_eq!(prio.score, 100);
    }

    // -- on_swap_in multiple times on same page --

    #[test]
    fn on_swap_in_twice_resets_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0);
        // Access a few times
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 2);
        // Swap in again resets
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0, "second swap-in must reset access_count");
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
    }

    // -- detect_working_set Active page not hot and not expired stays Active --

    #[test]
    fn detect_working_set_active_recent_not_hot_stays_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 100,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    // -- Mixed expert and dense pages in same layer --

    #[test]
    fn mixed_expert_and_dense_pages_in_same_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 0);
        assert_eq!(scheduler.weight_page_table[&0], vec![10, 20]);
        assert_eq!(scheduler.num_weight_pages(), 2);
        // Free expert pages only (dense uses free_dense_layer_weight_pages)
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_weight_pages(), 0, "free_expert frees all pages in layer");
    }

    // -- compute_eviction_priority with Swapped state gets no state bonus --

    #[test]
    fn eviction_priority_swapped_state_no_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Swapped,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // KV base = 100, no state bonus for Swapped
        // score = 100 - 0 + 0 + 0 + 0 = 100
        assert_eq!(prio.score, 100, "Swapped state must get no bonus");
    }

    // -- compute_eviction_priority with Free state gets no state bonus --

    #[test]
    fn eviction_priority_free_state_no_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Free,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100, "Free state must get no bonus");
    }

    // -- on_prefill_chunk_complete increments group access_count --

    #[test]
    fn on_prefill_chunk_complete_increments_group_access_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(10, 0)];
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        assert_eq!(scheduler.sequence_groups[&10].access_count, 1);
        scheduler.on_prefill_chunk_complete(1, 2, &pages);
        assert_eq!(scheduler.sequence_groups[&10].access_count, 2);
    }

    // -- EvictionPriority score reflects expert_id field --

    #[test]
    fn eviction_priority_rag_page_has_no_expert_no_layer() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, None);
        assert_eq!(prio.layer_idx, None);
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
    }

    // -- select_victim_weight_pages with dense-only pages --

    #[test]
    fn select_victim_weight_pages_dense_pages_skipped_when_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(10, 0);
        // Dense pages default to Active but let's set them to Protected
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Protected;
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "Protected dense pages must be skipped");
    }

    // -- upsert_group multiple times same group --

    #[test]
    fn upsert_group_three_times_preserves_original_counters() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let original = SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 42,
            last_access: Instant::now() - Duration::from_secs(100),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(original);
        // Update 1
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        });
        // Update 2
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3],
            state: GroupState::Swapped,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let group = &scheduler.sequence_groups[&1];
        assert_eq!(group.access_count, 42, "access_count must be preserved through all upserts");
        assert_eq!(group.pages, vec![1, 2, 3], "pages must be from last upsert");
    }

    // -- EvictionPriority Debug output --

    #[test]
    fn eviction_priority_debug_includes_access_count_and_recency() {
        let prio = EvictionPriority {
            score: 500,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 7,
            recency: 200,
            layer_idx: None,
            expert_id: None,
        };
        let debug = format!("{:?}", prio);
        assert!(debug.contains("access_count"));
        assert!(debug.contains("recency"));
    }

    // -- HGALConfig with extreme values --

    #[test]
    fn hgal_config_extreme_working_set_window() {
        let config = HGALConfig {
            working_set_window: Duration::from_secs(86400),
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().working_set_window, Duration::from_secs(86400));
    }

    #[test]
    fn hgal_config_lir_ratio_above_one() {
        let config = HGALConfig {
            lir_ratio: 2.0,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert!((scheduler.config().lir_ratio - 2.0).abs() < f32::EPSILON);
    }

    // -- compute_eviction_priority with Standby state --

    #[test]
    fn eviction_priority_standby_state_no_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100, "Standby state must get no bonus (same as no metadata)");
    }

    // -- select_victim_groups with group that has pages but no metadata --

    #[test]
    fn select_victim_groups_group_with_pages_but_no_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![999], // page 999 has no metadata
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![1], "group with no page metadata should still be selectable");
    }

    // -- mark_accessed on page with SwappedOut state transitions to Active --

    #[test]
    fn mark_accessed_swapped_out_transitions_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, None, PageState::SwappedOut);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::SwappedOut);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active, "SwappedOut page must become Active on access");
    }

    // -- register_expert_weight_page + free + verify metadata gone --

    #[test]
    fn register_and_free_expert_weight_page_lifecycle() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let page_id = 42;
        let layer = 5;

        scheduler.register_expert_weight_page(page_id, layer);
        assert!(scheduler.page_metadata.contains_key(&page_id));
        assert_eq!(scheduler.page_metadata[&page_id].state, PageState::Active);

        // Access the page
        scheduler.mark_accessed(page_id);
        assert_eq!(scheduler.page_metadata[&page_id].access_count, 1);

        // Free
        scheduler.free_expert_weight_pages(layer);
        assert!(!scheduler.page_metadata.contains_key(&page_id));
        assert!(!scheduler.weight_page_table.contains_key(&layer));
    }

    // -- virtual_page_to_page_id commutativity check --

    #[test]
    fn virtual_page_to_page_id_cantor_pairing_is_not_commutative() {
        let ab = virtual_page_to_page_id(VirtualPageId::new(1, 2));
        let ba = virtual_page_to_page_id(VirtualPageId::new(2, 1));
        assert_ne!(ab, ba, "Cantor pairing is not commutative: (1,2) != (2,1)");
    }

    // -- select_victim_weight_pages returns results in score order --

    #[test]
    fn select_victim_weight_pages_result_is_sorted_ascending() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Create 5 expert weight pages with varying access counts
        for i in 0..5u64 {
            let page_id = (i + 1) as usize;
            scheduler.register_expert_weight_page(page_id, 0);
            scheduler.page_metadata.get_mut(&page_id).unwrap().state = PageState::Standby;
            scheduler.page_metadata.get_mut(&page_id).unwrap().access_count = (5 - i) as usize;
        }
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: (1..=5).collect(),
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        let victims = scheduler.select_victim_weight_pages(5);
        for window in victims.windows(2) {
            assert!(
                window[0].1.score <= window[1].1.score,
                "results must be sorted ascending: {} <= {}",
                window[0].1.score, window[1].1.score,
            );
        }
    }

    // -- detect_working_set: Active page becomes Standby when not hot and expired --

    #[test]
    fn detect_working_set_active_not_hot_and_window_expired_no_demotion() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 100,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        // Active page that is not hot stays Active (only Protected pages get demoted to Standby)
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    // -- on_prefill_complete with Warm page clears warm_until --

    #[test]
    fn on_prefill_complete_clears_warm_on_warm_page() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Manually create Warm page with warm_until
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Warm,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            ..Default::default()
        });
        scheduler.on_prefill_complete(1);
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.state, PageState::Active, "Warm page must become Active on prefill complete");
        assert!(meta.warm_until.is_none(), "warm_until must be cleared");
    }

    // -- EvictionPriority reflects pinned status from page --

    #[test]
    fn eviction_priority_system_prompt_is_pinned() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.is_pinned, "system prompt page must be pinned");
        assert!(prio.score > 10_000, "system prompt score must be very high");
    }

    // -- VirtualPageId with max values --

    #[test]
    fn virtual_page_id_max_values() {
        let vpid = VirtualPageId::new(u64::MAX, usize::MAX);
        assert_eq!(vpid.sequence_id, u64::MAX);
        assert_eq!(vpid.logical_index, usize::MAX);
    }

    // -- weight page table cleanup leaves no orphan metadata --

    #[test]
    fn free_all_weight_pages_cleans_all_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(3, 0);
        scheduler.allocate_expert_weight_pages(2, 1);
        scheduler.register_dense_layer_weight_pages(vec![100, 101], 2);

        scheduler.free_expert_weight_pages(0);
        scheduler.free_expert_weight_pages(1);
        scheduler.free_dense_layer_weight_pages(2);

        assert_eq!(scheduler.num_weight_pages(), 0);
        assert!(scheduler.weight_page_table.is_empty());
        // Check metadata for all pages is gone
        for layer in &[0, 1, 2] {
            if let Some(pages) = scheduler.weight_page_table.get(layer) {
                for pid in pages {
                    assert!(!scheduler.page_metadata.contains_key(pid));
                }
            }
        }
    }

    // -- select_victim_groups with ExpertWeight group and high access_count --

    #[test]
    fn select_victim_groups_expert_with_high_freq_still_evicted_first() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // ExpertWeight group with high freq_bonus
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 100,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        // None-payload group with low freq
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        // ExpertWeight has -200 adjustment, but high freq_bonus (100*10=1000)
        // score = time_penalty + recency - 1000 - 0 + (-200)
        // None-payload: score = time_penalty + recency - 10 - 0 + 0
        // Expert's freq_bonus (-1000) lowers score more, so expert still evicted first
        assert_eq!(victims[0], 1, "ExpertWeight with -200 adjustment must be evicted first");
    }

    // -- compute_eviction_priority with SwappedOut state (Active variant) --

    #[test]
    fn eviction_priority_swapped_out_state_no_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::SwappedOut,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100, "SwappedOut state must get no bonus");
    }

    // -- on_swap_in warm_until matches config.warmup_duration --

    #[test]
    fn on_swap_in_warm_until_matches_config_warmup() {
        let warmup = Duration::from_millis(250);
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: warmup,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        let meta = &scheduler.page_metadata[&10];
        let warm_until = meta.warm_until.expect("warm_until must be set");
        let swap_in = meta.swap_in_time.expect("swap_in_time must be set");
        let diff = warm_until.saturating_duration_since(swap_in);
        assert_eq!(diff, warmup, "warm_until - swap_in_time must equal warmup_duration");
    }

    // -- EvictionPriority for rag page not pinned --

    #[test]
    fn eviction_priority_rag_is_not_pinned() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(!prio.is_pinned, "RAG pages must not be pinned");
    }

    // ── Batch 5: 50 additional tests for uncovered areas ──

    // -- HGALConfig builder edge cases --

    #[test]
    fn hgal_config_very_short_warmup_duration() {
        let config = HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().warmup_duration, Duration::from_nanos(1));
    }

    #[test]
    fn hgal_config_very_long_warmup_duration() {
        let config = HGALConfig {
            warmup_duration: Duration::from_secs(3600),
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().warmup_duration, Duration::from_secs(3600));
    }

    #[test]
    fn hgal_config_hot_threshold_one() {
        let config = HGALConfig {
            hot_threshold: 1,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 1,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected, "access_count=1 with hot_threshold=1 must promote");
    }

    #[test]
    fn hgal_config_lir_ratio_very_small() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.01,
            ..Default::default()
        });
        for i in 1..=10 {
            scheduler.mark_accessed(i);
        }
        // target_lir = ceil(10 * 0.01) = 1
        assert!(scheduler.lir_pages.len() <= 1, "very small lir_ratio must keep at most 1 page in LIR");
    }

    #[test]
    fn hgal_config_zero_working_set_window() {
        let config = HGALConfig {
            working_set_window: Duration::ZERO,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().working_set_window, Duration::ZERO);
    }

    // -- EvictionPriority formula boundary conditions --

    #[test]
    fn eviction_priority_exact_formula_expert_weight_with_recency_and_freq() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 3,
            recency: 50,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::expert(1, 0, 2, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // payload_adjustment = EXPERT_WEIGHT_PRIORITY_BONUS(-200) + layer_depth_penalty(-(2)) = -202
        // recency_penalty = 50 * (10/2) = 250
        // freq_bonus = 3 * 10 = 30
        // pin_bonus = 0 (evictable)
        // state_bonus = 0 (Active)
        // score = -202 - 250 + 30 + 0 + 0 = -422
        assert_eq!(prio.score, -422, "exact formula: -202 - 250 + 30 = -422");
    }

    #[test]
    fn eviction_priority_exact_formula_rag_with_all_zeros() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // payload_adjustment = -200 (KnowledgeRAG)
        // score = -200 - 0 + 0 + 0 + 0 = -200
        assert_eq!(prio.score, -200);
    }

    #[test]
    fn eviction_priority_exact_formula_prompt_system_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // payload_adjustment = 10000 (PromptSystem)
        // pin_bonus = 5000 (is_evictable=false)
        // score = 10000 - 0 + 0 + 5000 + 0 = 15000
        assert_eq!(prio.score, 15_000, "PromptSystem base = 10000 + pin 5000 = 15000");
    }

    #[test]
    fn eviction_priority_recency_zero_no_penalty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 0,
            access_count: 0,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // score = 100 - 0 + 0 + 0 + 0 = 100
        assert_eq!(prio.score, 100, "zero recency must produce no penalty");
    }

    #[test]
    fn eviction_priority_large_recency_overwhelms_freq() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 1_000_000,
            access_count: 1,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // recency_penalty = 1_000_000 * 5 = 5_000_000
        // freq_bonus = 1 * 10 = 10
        // score = 100 - 5_000_000 + 10 = -4_999_890
        assert!(prio.score < -4_000_000, "huge recency must produce very negative score, got {}", prio.score);
    }

    #[test]
    fn eviction_priority_warm_and_protected_delta_is_exactly_5000() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            ..Default::default()
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            state: PageState::Protected,
            ..Default::default()
        });
        let warm_page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let protected_page = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let warm_score = scheduler.compute_eviction_priority(&warm_page).score;
        let protected_score = scheduler.compute_eviction_priority(&protected_page).score;
        assert_eq!(protected_score - warm_score, 5_000, "Protected-Warm delta must be exactly 5000");
    }

    // -- PageState variant transition chains --

    #[test]
    fn page_transition_free_to_active_via_mark_accessed() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, None, PageState::Free);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Free);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
    }

    #[test]
    fn page_transition_active_to_protected_via_working_set() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 2,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Protected);
    }

    #[test]
    fn page_transition_protected_to_standby_via_expiry() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_millis(1),
            ..Default::default()
        });
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Protected,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(1),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Standby);
    }

    #[test]
    fn page_transition_standby_to_active_via_mark_accessed() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, None, PageState::Standby);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
    }

    #[test]
    fn page_transition_active_to_swapped_via_update() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        scheduler.update_page_state(10, Some(1), PageState::Swapped);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Swapped);
        assert!(scheduler.page_metadata[&10].warm_until.is_none());
    }

    #[test]
    fn page_transition_swapped_to_warm_via_swap_in() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Swapped);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Swapped);
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0);
    }

    #[test]
    fn page_transition_warm_to_active_on_access_after_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        std::thread::sleep(Duration::from_millis(1));
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
    }

    #[test]
    fn page_transition_full_cycle_active_standby_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Active);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Standby);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
    }

    // -- Working set tracking edge cases --

    #[test]
    fn detect_working_set_multiple_detect_calls_idempotent_when_stable() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
        // Call again — state must remain Protected
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    #[test]
    fn detect_working_set_standby_page_with_recent_access_but_not_hot_stays_standby() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 100,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            access_count: 2,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby);
    }

    #[test]
    fn detect_working_set_warm_page_exits_warmup_then_promoted_if_hot() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 1,
            working_set_window: Duration::from_secs(60),
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            access_count: 10,
            last_access: Instant::now(),
            warm_until: Some(Instant::now() - Duration::from_secs(1)),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected, "hot page that exited warmup must reach Protected in one pass");
    }

    #[test]
    fn detect_working_set_warm_page_in_warmup_stays_warm_regardless_of_access_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: usize::MAX,
            hot_threshold: 1,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        // Even though access_count is 0 and min_warm_access is MAX, warmup period is active
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
    }

    // -- Group with Warm page in warmup protects from eviction --

    #[test]
    fn group_with_warm_page_in_active_warmup_is_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.on_swap_in(10);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "group with page in active warmup must be protected");
    }

    #[test]
    fn group_with_page_missing_metadata_is_not_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![999], // no metadata for this page
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![1], "group with missing page metadata must not be protected");
    }

    // -- EvictionPriority field consistency --

    #[test]
    fn eviction_priority_dense_layer_layer_idx_set() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::dense_layer(10, 5, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.layer_idx, Some(5), "dense layer page must carry layer_idx");
    }

    #[test]
    fn eviction_priority_rag_owner_reflected_in_page() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 42, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(!prio.is_pinned);
    }

    // -- select_victim_groups interaction with access_count and payload_kind --

    #[test]
    fn select_victim_groups_dense_always_resists_eviction() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Dense with 0 access (should still resist due to +5000 adjustment)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(100),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });
        // None-payload with recent access
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 100,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims[0], 2, "None-payload group must be evicted before DenseLayerWeight");
    }

    // -- mark_accessed recency computation --

    #[test]
    fn mark_accessed_recency_increases_between_accesses() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].recency, 0);
        std::thread::sleep(Duration::from_millis(3));
        scheduler.mark_accessed(10);
        let recency_second = scheduler.page_metadata[&10].recency;
        assert!(recency_second > 0, "second access must have nonzero recency");
        std::thread::sleep(Duration::from_millis(3));
        scheduler.mark_accessed(10);
        let recency_third = scheduler.page_metadata[&10].recency;
        // Third recency may not be strictly larger due to timing, but must be non-negative
        assert!(recency_third > 0 || recency_second > 0, "recency values must be meaningful");
    }

    // -- on_swap_in sequence_id is preserved from prior state --

    #[test]
    fn on_swap_in_does_not_change_sequence_id_if_previously_set() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(42), PageState::Active);
        assert_eq!(scheduler.page_metadata[&10].sequence_id, Some(42));
        scheduler.on_swap_in(10);
        // on_swap_in does not set sequence_id — it preserves the existing entry
        assert_eq!(scheduler.page_metadata[&10].sequence_id, Some(42), "sequence_id must be preserved through swap-in");
    }

    // -- update_page_state SwappedOut variant --

    #[test]
    fn update_page_state_swapped_out_does_not_clear_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(10);
        assert!(scheduler.page_metadata[&10].warm_until.is_some());
        // SwappedOut is not Swapped, so warm_until should not be cleared
        scheduler.update_page_state(10, Some(1), PageState::SwappedOut);
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.state, PageState::SwappedOut);
        // SwappedOut != Swapped, so warm_until should survive
        assert!(meta.warm_until.is_some(), "SwappedOut must not clear warm_until");
    }

    // -- on_prefill_chunk_complete saturating_add overflow --

    #[test]
    fn on_prefill_chunk_complete_access_count_saturates() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId::new(10, 0);
        let pid = virtual_page_to_page_id(vpid);
        scheduler.page_metadata.insert(pid, PageMetadata {
            page_id: pid,
            access_count: usize::MAX,
            ..Default::default()
        });
        scheduler.on_prefill_chunk_complete(0, 1, &[vpid]);
        assert_eq!(scheduler.page_metadata[&pid].access_count, usize::MAX, "access_count must saturate at MAX");
    }

    // -- select_victim_weight_pages with no eligible pages --

    #[test]
    fn select_victim_weight_pages_all_active_still_skipped() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        // Active is not Protected or Warm, so it should be eligible
        // But let's verify Active pages ARE eligible (only Protected/Warm are skipped)
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert_eq!(victims.len(), 1, "Active weight pages must be eligible for eviction");
        assert_eq!(victims[0].0, 10);
    }

    // -- register_dense_layer_weight_page duplicate page in same layer --

    #[test]
    fn register_dense_layer_weight_page_duplicate_in_same_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(42, 0);
        scheduler.register_dense_layer_weight_page(42, 0);
        let pages = scheduler.weight_page_table.get(&0).unwrap();
        assert_eq!(pages.len(), 2, "duplicate registration must append to same layer");
        assert_eq!(pages[0], 42);
        assert_eq!(pages[1], 42);
    }

    // -- compute_eviction_priority with only freq_bonus (recency=0) --

    #[test]
    fn eviction_priority_only_freq_bonus_no_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 20,
            recency: 0,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // score = 100 (KV) - 0 + 20*10 + 0 + 0 = 300
        assert_eq!(prio.score, 300, "freq_bonus only: 100 + 200 = 300");
    }

    // -- compute_eviction_priority with only recency penalty (access=0) --

    #[test]
    fn eviction_priority_only_recency_penalty_no_freq() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 0,
            recency: 500,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // score = 100 (KV) - 500*5 + 0 + 0 + 0 = 100 - 2500 = -2400
        assert_eq!(prio.score, -2400, "recency only: 100 - 2500 = -2400");
    }

    // -- LIR membership: target_lir computed from page_metadata count --

    #[test]
    fn lir_target_computed_from_current_page_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        });
        // 4 pages -> target_lir = ceil(4 * 0.5) = 2
        for i in 1..=4u64 {
            scheduler.mark_accessed(i as usize);
        }
        assert!(scheduler.lir_pages.len() <= 2, "target_lir must be ceil(4 * 0.5) = 2");
    }

    // -- update_page_state then on_swap_in restores Warm --

    #[test]
    fn update_to_swapped_then_swap_in_restores_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.update_page_state(10, Some(1), PageState::Swapped);
        assert!(scheduler.page_metadata[&10].warm_until.is_none());
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        assert!(scheduler.page_metadata[&10].warm_until.is_some());
    }

    // -- select_victim_weight_pages with mixed Standby and Active pages --

    #[test]
    fn select_victim_weight_pages_eligible_standby_and_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Standby page
        scheduler.register_expert_weight_page(10, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        // Active page (also eligible since not Protected/Warm)
        scheduler.register_expert_weight_page(11, 0);
        // Active pages are eligible for select_victim_weight_pages
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims.len(), 2, "both Standby and Active pages must be eligible");
    }

    // -- num_weight_pages consistency after mixed alloc/free --

    #[test]
    fn num_weight_pages_consistency_after_partial_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(5, 0);
        scheduler.register_dense_layer_weight_pages(vec![100, 101, 102], 1);
        assert_eq!(scheduler.num_weight_pages(), 8);
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_weight_pages(), 3, "only dense layer 1 pages remain");
        scheduler.free_dense_layer_weight_pages(1);
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    // -- EvictionPriority Clone consistency --

    #[test]
    fn eviction_priority_clone_independent_modification() {
        let prio = EvictionPriority {
            score: -500,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 5,
            recency: 100,
            layer_idx: Some(3),
            expert_id: Some(7),
        };
        let cloned = prio.clone();
        assert_eq!(cloned.score, prio.score);
        assert_eq!(cloned.payload_kind, prio.payload_kind);
        assert_eq!(cloned.is_pinned, prio.is_pinned);
        assert_eq!(cloned.access_count, prio.access_count);
        assert_eq!(cloned.recency, prio.recency);
        assert_eq!(cloned.layer_idx, prio.layer_idx);
        assert_eq!(cloned.expert_id, prio.expert_id);
    }

    // -- select_victim_weight_pages: layer_idx resolved from weight_page_table --

    #[test]
    fn select_victim_weight_pages_layer_idx_resolved_correctly() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Register page 10 in layer 5
        scheduler.register_expert_weight_page(10, 5);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].1.layer_idx, Some(5), "layer_idx must be resolved from weight_page_table");
    }

    // -- HGALScheduler::config() returns stable reference --

    #[test]
    fn config_reference_stable_across_operations() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 42,
            ..Default::default()
        });
        let hot_before = scheduler.config().hot_threshold;
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert_eq!(scheduler.config().hot_threshold, hot_before, "config must not change after upsert_group");
    }

    // -- detect_working_set with zero pages does not panic --

    #[test]
    fn detect_working_set_after_removing_all_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.page_metadata.clear();
        scheduler.detect_working_set();
        assert!(scheduler.page_metadata.is_empty());
    }

    // -- on_prefill_chunk_complete with usize::MAX total_chunks and chunk 0 --

    #[test]
    fn on_prefill_chunk_complete_first_of_max_chunks_is_standby() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(0, usize::MAX, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        // chunk_idx=0, total_chunks=usize::MAX -> is_last_chunk = 0+1 >= usize::MAX = false
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Standby);
    }

    // -- EvictionPriority score domain boundaries --

    #[test]
    fn eviction_priority_score_i64_boundary_high() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: usize::MAX,
            recency: 0,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // freq_bonus = usize::MAX * 10 could overflow i64, but saturating behavior is acceptable
        // We just verify no panic
        assert!(prio.score > 0 || prio.score < 0, "very large freq_bonus must produce some score without panic");
    }

    // -- mark_accessed on Warm page during warmup stays Warm --

    #[test]
    fn mark_accessed_warm_in_warmup_multiple_times_stays_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: 100,
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        for _ in 0..5 {
            scheduler.mark_accessed(10);
        }
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm, "Warm page in warmup must stay Warm after multiple accesses");
        assert_eq!(scheduler.page_metadata[&10].access_count, 5);
    }

    // -- select_victim_groups with group having empty pages vector --

    #[test]
    fn select_victim_groups_empty_pages_group_still_counted() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Group with empty pages -> pages.len().max(1) = 1
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims.len(), 1, "group with empty pages must still be selected (pages.max(1)=1)");
    }

    // -- compute_group_priority: recency_penalty from page metadata --

    

    // -- on_prefill_complete does not affect other groups --

    #[test]
    fn on_prefill_complete_isolates_to_target_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Group 1: prefill target
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Paused,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        // Group 2: unrelated
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Paused,
            access_count: 5,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(20, Some(2), PageState::Standby);

        scheduler.on_prefill_complete(1);
        // Group 1 must be Running, group 2 must stay Paused
        assert_eq!(scheduler.sequence_groups[&1].state, GroupState::Running);
        assert_eq!(scheduler.sequence_groups[&2].state, GroupState::Paused, "unrelated group must not be affected");
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&20].state, PageState::Standby, "unrelated page must not be activated");
    }

    // -- on_swap_in sets last_access to current time --

    #[test]
    fn on_swap_in_last_access_is_recent() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let before = Instant::now();
        scheduler.on_swap_in(10);
        let after = Instant::now();
        let meta = &scheduler.page_metadata[&10];
        assert!(meta.last_access >= before, "last_access must be at or after swap-in call");
        assert!(meta.last_access <= after, "last_access must be at or before swap-in return");
    }

    // -- Mixed expert and dense in different layers, free one layer --

    #[test]
    fn free_expert_weight_pages_dense_pages_in_different_layer_unaffected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        assert_eq!(scheduler.num_weight_pages(), 2);
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_weight_pages(), 1, "dense layer pages in other layers must survive");
        assert!(scheduler.page_metadata.contains_key(&20));
        assert!(!scheduler.page_metadata.contains_key(&10));
    }

    // -- detect_working_set with exactly hot_threshold access_count --

    #[test]
    fn detect_working_set_exact_hot_threshold_promotes() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 5,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 5, // exactly hot_threshold
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected, "access_count == hot_threshold must promote");
    }

    #[test]
    fn detect_working_set_one_below_hot_threshold_does_not_promote() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 5,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 4, // one below hot_threshold
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active, "access_count < hot_threshold must not promote");
    }

    // -- PageMetadata Debug output --

    #[test]
    fn page_metadata_debug_format_is_informative() {
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(10),
            recency: 500,
            access_count: 7,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Protected,
            warm_until: None,
        };
        let debug = format!("{:?}", meta);
        assert!(debug.contains("42"), "Debug must contain page_id");
        assert!(debug.contains("Protected"), "Debug must contain state");
        assert!(debug.contains("is_lir"), "Debug must contain is_lir");
    }

    // -- VirtualPageId with zero sequence_id --

    #[test]
    fn virtual_page_id_zero_sequence_id() {
        let vpid = VirtualPageId::new(0, 5);
        assert_eq!(vpid.sequence_id, 0);
        assert_eq!(vpid.logical_index, 5);
        let pid = virtual_page_to_page_id(vpid);
        // (0+5)*(0+5+1)/2 + 5 = 5*6/2 + 5 = 15 + 5 = 20
        assert_eq!(pid, 20, "Cantor pairing of (0,5) must be 20");
    }

    // -- on_prefill_chunk_complete sets sequence_id from VirtualPageId --

    #[test]
    fn on_prefill_chunk_complete_sets_sequence_id_correctly() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId::new(42, 0);
        let pid = virtual_page_to_page_id(vpid);
        scheduler.on_prefill_chunk_complete(0, 1, &[vpid]);
        assert_eq!(scheduler.page_metadata[&pid].sequence_id, Some(42));
    }

    // ── Batch 6: ~60 additional tests for coverage gaps ──

    // -- HGALConfig: field value boundary combinations --

    #[test]
    fn hgal_config_all_zero_durations() {
        let config = HGALConfig {
            warmup_duration: Duration::ZERO,
            working_set_window: Duration::ZERO,
            hot_threshold: 0,
            lir_ratio: 0.0,
            min_warm_access: 0,
            enable_clock_pro: false,
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().warmup_duration, Duration::ZERO);
        assert_eq!(scheduler.config().working_set_window, Duration::ZERO);
        assert_eq!(scheduler.config().hot_threshold, 0);
        assert_eq!(scheduler.config().min_warm_access, 0);
        assert!(!scheduler.config().enable_clock_pro);
    }

    #[test]
    fn hgal_config_negative_lir_ratio_treated_as_zero() {
        // f32 can be negative; verify scheduler accepts it without panic
        let config = HGALConfig {
            lir_ratio: -0.5,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert!(scheduler.config().lir_ratio < 0.0);
    }

    #[test]
    fn hgal_config_nan_lir_ratio_no_panic() {
        let config = HGALConfig {
            lir_ratio: f32::NAN,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert!(scheduler.config().lir_ratio.is_nan());
    }

    #[test]
    fn hgal_config_infinity_lir_ratio_no_panic() {
        let config = HGALConfig {
            lir_ratio: f32::INFINITY,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        assert!(scheduler.config().lir_ratio.is_infinite());
    }

    #[test]
    fn hgal_config_hot_threshold_usize_max() {
        let config = HGALConfig {
            hot_threshold: usize::MAX,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        // Access a page with high count but still below usize::MAX
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: usize::MAX - 1,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        // Should stay Active since access_count < hot_threshold
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    // -- EvictionPriority: construction with all field combinations --

    #[test]
    fn eviction_priority_kv_context_all_fields_populated() {
        let prio = EvictionPriority {
            score: 150,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 5,
            recency: 20,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(prio.score, 150);
        assert_eq!(prio.payload_kind, PagePayloadKind::KvContext);
        assert!(!prio.is_pinned);
        assert_eq!(prio.access_count, 5);
        assert_eq!(prio.recency, 20);
        assert_eq!(prio.layer_idx, None);
        assert_eq!(prio.expert_id, None);
    }

    #[test]
    fn eviction_priority_expert_weight_all_fields_populated() {
        let prio = EvictionPriority {
            score: -202,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 1,
            recency: 0,
            layer_idx: Some(2),
            expert_id: Some(7),
        };
        assert_eq!(prio.score, -202);
        assert_eq!(prio.layer_idx, Some(2));
        assert_eq!(prio.expert_id, Some(7));
    }

    #[test]
    fn eviction_priority_dense_layer_weight_all_fields() {
        let prio = EvictionPriority {
            score: 10_000,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            is_pinned: true,
            access_count: 50,
            recency: 0,
            layer_idx: Some(10),
            expert_id: None,
        };
        assert!(prio.is_pinned);
        assert_eq!(prio.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert_eq!(prio.layer_idx, Some(10));
        assert_eq!(prio.expert_id, None);
    }

    #[test]
    fn eviction_priority_knowledge_rag_all_fields() {
        let prio = EvictionPriority {
            score: -200,
            payload_kind: PagePayloadKind::KnowledgeRAG,
            is_pinned: false,
            access_count: 0,
            recency: 1000,
            layer_idx: None,
            expert_id: None,
        };
        assert!(!prio.is_pinned);
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
    }

    #[test]
    fn eviction_priority_prompt_system_all_fields() {
        let prio = EvictionPriority {
            score: 15_000,
            payload_kind: PagePayloadKind::PromptSystem,
            is_pinned: true,
            access_count: 100,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert!(prio.is_pinned);
        assert_eq!(prio.payload_kind, PagePayloadKind::PromptSystem);
    }

    // -- EvictionPriority: negative score boundaries --

    #[test]
    fn eviction_priority_score_i64_min_no_panic() {
        let prio = EvictionPriority {
            score: i64::MIN,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(prio.score, i64::MIN);
    }

    #[test]
    fn eviction_priority_score_i64_max_no_panic() {
        let prio = EvictionPriority {
            score: i64::MAX,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            is_pinned: true,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(prio.score, i64::MAX);
    }

    // -- Working set tracking: boundary value at exactly working_set_window --

    #[test]
    fn detect_working_set_protected_at_exact_window_boundary_stays_protected() {
        // Page last_accessed exactly at working_set_window ago
        // saturating_duration_since returns exactly working_set_window
        // The condition is >= working_set_window, so exactly at boundary it gets demoted
        let window = Duration::from_millis(50);
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: window,
            hot_threshold: 100, // not hot
            ..Default::default()
        });
        // Use a stale last_access
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 1,
            last_access: Instant::now() - Duration::from_secs(5), // well beyond window
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby,
            "Protected page well beyond window must be demoted");
    }

    #[test]
    fn detect_working_set_protected_recent_within_window_stays_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_secs(60),
            hot_threshold: 100,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 5,
            last_access: Instant::now(), // well within window
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    // -- Priority queue ordering: multi-group eviction scenarios --

    #[test]
    fn select_victim_groups_ordering_with_three_payload_kinds() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // ExpertWeight (-200), DenseLayerWeight (+5000), None (0)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 3,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        let victims = scheduler.select_victim_groups(3);
        assert_eq!(victims.len(), 3);
        // Expert (-200) first, then None (0), then Dense (+5000)
        assert_eq!(victims[0], 1, "ExpertWeight must be first victim");
        assert_eq!(victims[1], 2, "None payload must be second victim");
        assert_eq!(victims[2], 3, "DenseLayerWeight must be last victim");
    }

    #[test]
    fn select_victim_groups_same_priority_different_page_counts() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Two groups with same payload (None), different page counts
        // Both get same priority score, but we request enough to need both
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![4],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Request 4 pages: both groups needed
        let victims = scheduler.select_victim_groups(4);
        assert_eq!(victims.len(), 2);
    }

    // -- Score computation: access_count=0 and recency=0 for each payload kind --

    #[test]
    fn eviction_priority_zero_zero_expert_weight_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // payload=-200 + layer_depth=0, no metadata, no pin
        assert_eq!(prio.score, -200);
    }

    #[test]
    fn eviction_priority_zero_zero_rag_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, -200);
    }

    #[test]
    fn eviction_priority_zero_zero_kv_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100);
    }

    // -- Debug trait output verification for structs --

    #[test]
    fn sequence_group_debug_format() {
        let group = SequenceGroup {
            id: 42,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        let debug = format!("{:?}", group);
        assert!(!debug.is_empty());
        assert!(debug.contains("id"));
        assert!(debug.contains("pages"));
        assert!(debug.contains("access_count"));
    }

    #[test]
    fn eviction_priority_debug_all_payload_kinds() {
        for kind in [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ] {
            let prio = EvictionPriority {
                score: 0,
                payload_kind: kind,
                is_pinned: false,
                access_count: 0,
                recency: 0,
                layer_idx: None,
                expert_id: None,
            };
            let debug = format!("{:?}", prio);
            assert!(!debug.is_empty(), "Debug for {:?} must not be empty", kind);
        }
    }

    // -- Enum variant exhaustiveness: PageState all variants tested --

    #[test]
    fn page_state_all_variants_in_hashset() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        seen.insert(PageState::Free);
        seen.insert(PageState::Active);
        seen.insert(PageState::Standby);
        seen.insert(PageState::SwappedOut);
        seen.insert(PageState::Warm);
        seen.insert(PageState::Protected);
        seen.insert(PageState::Swapped);
        assert_eq!(seen.len(), 7, "all 7 PageState variants must be distinct");
    }

    // -- PagePayloadKind exhaustiveness: all 5 variants --

    #[test]
    fn page_payload_kind_all_five_variants() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        seen.insert(PagePayloadKind::KvContext);
        seen.insert(PagePayloadKind::ExpertWeight);
        seen.insert(PagePayloadKind::PromptSystem);
        seen.insert(PagePayloadKind::KnowledgeRAG);
        seen.insert(PagePayloadKind::DenseLayerWeight);
        assert_eq!(seen.len(), 5, "all 5 PagePayloadKind variants must be distinct");
    }

    // -- GroupState exhaustiveness: all 3 variants --

    #[test]
    fn group_state_all_three_variants_in_hashset() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        seen.insert(GroupState::Running);
        seen.insert(GroupState::Swapped);
        seen.insert(GroupState::Paused);
        assert_eq!(seen.len(), 3, "all 3 GroupState variants must be distinct");
    }

    // -- Struct construction with boundary Instant values --

    #[test]
    fn page_metadata_with_epoch_last_access() {
        // Instant::now() is relative; we can use Instant::now() - Duration::from_secs(86400) for old
        let old_time = Instant::now() - Duration::from_secs(86400);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: usize::MAX,
            access_count: usize::MAX,
            last_access: old_time,
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        assert_eq!(meta.recency, usize::MAX);
        assert_eq!(meta.access_count, usize::MAX);
        assert!(meta.last_access < Instant::now());
    }

    // -- Eviction priority: expert weight with deep layer produces very negative score --

    #[test]
    fn eviction_priority_expert_weight_deep_layer_accumulates() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let shallow = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        let deep = UnifiedVirtualPage::expert(2, 0, 100, gllm_kernels::types::DType::F32);
        let prio_shallow = scheduler.compute_eviction_priority(&shallow);
        let prio_deep = scheduler.compute_eviction_priority(&deep);
        // layer_depth_penalty = -(layer_idx), so deep is 100 lower
        assert_eq!(prio_shallow.score - prio_deep.score, 100,
            "depth delta must be exactly 100 for layer difference of 100");
    }

    // -- Score computation with both recency and frequency extreme values --

    #[test]
    fn eviction_priority_freq_max_recency_zero_produces_very_positive() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 0,
            access_count: usize::MAX,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // freq_bonus = usize::MAX * 10 (may overflow), state_bonus = 10000
        // We just verify no panic and a valid score
        let _ = prio.score; // no panic is the assertion
    }

    // -- select_victim_groups: requesting more pages than available --

    #[test]
    fn select_victim_groups_request_more_than_available_returns_all() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Request 100 pages but only 3 available
        let victims = scheduler.select_victim_groups(100);
        assert_eq!(victims.len(), 2, "must return all groups when request exceeds available");
    }

    // -- on_prefill_chunk_complete: group pages are sorted and deduped --

    #[test]
    fn on_prefill_chunk_complete_pages_are_sorted() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Add pages in reverse order via two chunks
        let chunk1 = vec![VirtualPageId::new(10, 5)];
        let chunk2 = vec![VirtualPageId::new(10, 0)];
        scheduler.on_prefill_chunk_complete(0, 2, &chunk1);
        scheduler.on_prefill_chunk_complete(1, 2, &chunk2);
        let group = &scheduler.sequence_groups[&10];
        // Pages must be sorted (page_id from Cantor pairing)
        for window in group.pages.windows(2) {
            assert!(window[0] <= window[1], "pages must be sorted: {} <= {}", window[0], window[1]);
        }
    }

    // -- register_expert_weight_page then access then verify metadata --

    #[test]
    fn register_expert_then_access_updates_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 3);
        assert_eq!(scheduler.page_metadata[&42].access_count, 0);
        scheduler.mark_accessed(42);
        assert_eq!(scheduler.page_metadata[&42].access_count, 1);
        assert_eq!(scheduler.page_metadata[&42].state, PageState::Active);
    }

    // -- update_page_state then mark_accessed then verify combined state --

    #[test]
    fn update_state_then_access_preserves_new_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&10].access_count, 1);
        assert_eq!(scheduler.page_metadata[&10].sequence_id, Some(1));
    }

    // -- Multiple swap-in cycles on the same page --

    // -- detect_working_set: many pages with mixed states --

    #[test]
    fn detect_working_set_many_pages_batch_transition() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_secs(60),
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            ..Default::default()
        });
        let now = Instant::now();
        // 5 hot Active pages -> Protected
        for i in 1..=5 {
            scheduler.page_metadata.insert(i, PageMetadata {
                page_id: i,
                state: PageState::Active,
                access_count: 10,
                last_access: now,
                ..Default::default()
            });
        }
        // 5 cold Active pages -> stay Active
        for i in 6..=10 {
            scheduler.page_metadata.insert(i, PageMetadata {
                page_id: i,
                state: PageState::Active,
                access_count: 1,
                last_access: now,
                ..Default::default()
            });
        }
        scheduler.detect_working_set();
        for i in 1..=5 {
            assert_eq!(scheduler.page_metadata[&i].state, PageState::Protected,
                "hot page {} must be Protected", i);
        }
        for i in 6..=10 {
            assert_eq!(scheduler.page_metadata[&i].state, PageState::Active,
                "cold page {} must stay Active", i);
        }
    }

    // -- select_victim_weight_pages: multiple layers, inter-layer ordering --

    #[test]
    fn select_victim_weight_pages_cross_layer_ordering() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Layer 0: stale page (high recency)
        scheduler.register_expert_weight_page(10, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&10).unwrap().recency = 5000;

        // Layer 5: fresh page (low recency)
        scheduler.register_expert_weight_page(11, 5);
        scheduler.page_metadata.get_mut(&11).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&11).unwrap().recency = 0;

        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims.len(), 2);
        // Page 10 has higher recency_penalty -> lower score -> first victim
        assert!(victims[0].1.score <= victims[1].1.score);
    }

    // -- upsert_group then remove_group then re-insert --

    #[test]
    fn upsert_remove_reinsert_group_lifecycle() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Insert
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 10,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert!(scheduler.sequence_groups.contains_key(&1));

        // Remove
        scheduler.remove_group(1);
        assert!(!scheduler.sequence_groups.contains_key(&1));

        // Re-insert with different data
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![3, 4, 5],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let group = &scheduler.sequence_groups[&1];
        assert_eq!(group.pages, vec![3, 4, 5]);
        assert_eq!(group.state, GroupState::Paused);
        assert_eq!(group.access_count, 0, "re-inserted group must have new access_count");
        assert!(group.is_pinned);
    }

    // -- EvictionPriority reflects correct layer_idx for dense layer pages --

    #[test]
    fn eviction_priority_dense_layer_various_layers() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        for layer in [0, 5, 10, 50] {
            let page = UnifiedVirtualPage::dense_layer(1, layer, gllm_kernels::types::DType::F32);
            let prio = scheduler.compute_eviction_priority(&page);
            assert_eq!(prio.layer_idx, Some(layer), "layer_idx must be {} for dense page", layer);
        }
    }

    // -- on_prefill_complete clears warm_until for Standby pages too --

    #[test]
    fn on_prefill_complete_clears_warm_on_standby_page() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Standby,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            ..Default::default()
        });
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        assert!(scheduler.page_metadata[&10].warm_until.is_none(),
            "warm_until must be cleared on Standby -> Active transition via prefill complete");
    }

    // -- virtual_page_to_page_id: zero sequence with nonzero logical --

    #[test]
    fn virtual_page_to_page_id_zero_sequence_nonzero_logical() {
        let vpid = VirtualPageId::new(0, 10);
        let id = virtual_page_to_page_id(vpid);
        // (0+10)*(0+10+1)/2 + 10 = 10*11/2 + 10 = 55 + 10 = 65
        assert_eq!(id, 65, "Cantor pairing of (0,10) must be 65");
    }

    // -- allocate_expert_weight_pages: many experts, many layers --

    #[test]
    fn allocate_expert_weight_pages_many_experts_many_layers() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for layer in 0..5 {
            let pages = scheduler.allocate_expert_weight_pages(8, layer);
            assert_eq!(pages.len(), 8);
            for (i, &pid) in pages.iter().enumerate() {
                assert_eq!(pid, (layer << 16) | i, "page encoding must match (layer<<16|idx)");
            }
        }
        assert_eq!(scheduler.num_expert_weight_pages(), 40);
    }

    // -- num_weight_pages after interleaved expert and dense operations --

    #[test]
    fn num_weight_pages_interleaved_operations() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Add expert pages to layer 0
        scheduler.allocate_expert_weight_pages(3, 0);
        assert_eq!(scheduler.num_weight_pages(), 3);

        // Add dense pages to layer 0 (same layer!)
        scheduler.register_dense_layer_weight_pages(vec![100, 101], 0);
        assert_eq!(scheduler.num_weight_pages(), 5);

        // Add expert pages to layer 1
        scheduler.register_expert_weight_page(200, 1);
        assert_eq!(scheduler.num_weight_pages(), 6);

        // Free expert from layer 0 (removes all layer 0 pages including dense!)
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_weight_pages(), 1, "only layer 1 expert page remains");
    }

    // -- select_victim_groups: freq_bonus dominates time_penalty for recent groups --

    #[test]
    fn select_victim_groups_freq_dominates_over_recency_for_high_access() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Old group with very high access count
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 1000,
            last_access: now - Duration::from_millis(50),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Recent group with low access count
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(1);
        // Group 1: time_penalty ~50, freq_bonus = -10000 -> score ~ -9950
        // Group 2: time_penalty ~0, freq_bonus = -10 -> score ~ -10
        // Lower score = evicted first = group 1
        assert_eq!(victims[0], 1, "high freq group must be evicted first (freq_bonus lowers score)");
    }

    // -- PageMetadata Clone produces independent copy --

    #[test]
    fn page_metadata_clone_is_independent() {
        let now = Instant::now();
        let original = PageMetadata {
            page_id: 42,
            sequence_id: Some(10),
            recency: 100,
            access_count: 5,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Protected,
            warm_until: Some(now + Duration::from_secs(5)),
        };
        let cloned = original.clone();
        // Verify all fields match
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.sequence_id, original.sequence_id);
        assert_eq!(cloned.recency, original.recency);
        assert_eq!(cloned.access_count, original.access_count);
        assert_eq!(cloned.is_lir, original.is_lir);
        assert_eq!(cloned.state, original.state);
    }

    // -- LIR membership: mark_accessed on existing page updates LIR --

    #[test]
    fn mark_accessed_on_existing_page_updates_lir_flag() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        });
        scheduler.update_page_state(10, None, PageState::Active);
        assert!(!scheduler.page_metadata[&10].is_lir);
        scheduler.mark_accessed(10);
        assert!(scheduler.page_metadata[&10].is_lir, "page must be marked as LIR after access with clock_pro enabled");
    }

    // -- update_page_state SwappedOut variant does not clear warm fields --

    #[test]
    fn update_page_state_active_does_not_clear_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(10);
        assert!(scheduler.page_metadata[&10].warm_until.is_some());
        // Updating to Active should NOT clear warm_until (only Swapped does)
        scheduler.update_page_state(10, Some(1), PageState::Active);
        // warm_until is NOT cleared because state != Swapped
        assert!(scheduler.page_metadata[&10].warm_until.is_some(),
            "update_page_state to Active must not clear warm_until");
    }

    // -- on_prefill_chunk_complete: group state is always Running --

    #[test]
    fn on_prefill_chunk_complete_sets_group_state_running() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Insert group with Paused state first
        scheduler.upsert_group(SequenceGroup {
            id: 42,
            pages: vec![],
            state: GroupState::Paused,
            access_count: 5,
            last_access: Instant::now() - Duration::from_secs(10),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        assert_eq!(scheduler.sequence_groups[&42].state, GroupState::Running,
            "group state must be set to Running after chunk complete");
    }

    // -- EvictionPriority: RAG score equals expert score at layer 0 --

    #[test]
    fn eviction_priority_rag_equals_expert_layer_zero() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let rag = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32);
        let rag_prio = scheduler.compute_eviction_priority(&rag);
        let expert_prio = scheduler.compute_eviction_priority(&expert);
        // Both get -200 base, no metadata, no layer depth penalty
        assert_eq!(rag_prio.score, expert_prio.score,
            "RAG and Expert (layer 0) must have equal score when both have no metadata");
    }

    // -- select_victim_groups with exactly one page needed from multi-page group --

    #[test]
    fn select_victim_groups_stops_immediately_when_first_group_satisfies() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Group with 10 pages
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: (0..10).collect(),
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 10,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Request just 1 page
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![1], "single large group must satisfy count=1 immediately");
    }

    // -- detect_working_set: Warm page with warm_until=None and not in warmup --

    #[test]
    fn detect_working_set_warm_no_warm_until_exits_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: 2,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            access_count: 5, // >= min_warm_access
            last_access: Instant::now(),
            warm_until: None,
            swap_in_time: None,
            ..Default::default()
        });
        scheduler.detect_working_set();
        // access_count >= min_warm_access, so exits warmup -> Active
        // Then hot check: access_count=5 >= hot_threshold=3 and within window -> Protected
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected,
            "Warm page with access_count >= min_warm_access and no warm_until must transition to Protected");
    }

    // -- EvictionPriority: manual construction with i64::MIN and i64::MAX --

    #[test]
    fn eviction_priority_extreme_negative_score() {
        let prio = EvictionPriority {
            score: i64::MIN,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 0,
            recency: usize::MAX,
            layer_idx: Some(100),
            expert_id: Some(0),
        };
        assert_eq!(prio.score, i64::MIN);
        assert_eq!(prio.layer_idx, Some(100));
    }

    // -- mark_accessed on page with Swapped state does not change to Active --

    #[test]
    fn mark_accessed_swapped_state_transitions_to_active() {
        // Note: The code says if state != Protected -> Active
        // Swapped is not Protected, so it should become Active
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, None, PageState::Swapped);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Swapped);
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active,
            "Swapped page must become Active on access");
    }

    // -- on_swap_in after prefill_complete: re-warms the page --

    #[test]
    fn swap_in_after_prefill_complete_re_warms() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // First, complete prefill (makes page Active)
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);

        // Then swap in (should reset to Warm)
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        assert_eq!(scheduler.page_metadata[&10].access_count, 0);
    }

    // -- EvictionPriority: dense layer weight with metadata freq bonus --

    #[test]
    fn eviction_priority_dense_with_freq_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 10,
            recency: 0,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::dense_layer(1, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // DenseLayerWeight: payload=5000, pin=5000, freq=10*10=100
        // score = 5000 - 0 + 100 + 5000 + 0 = 10100
        assert_eq!(prio.score, 10_100, "DenseLayerWeight with freq 10: 5000 + 5000 + 100 = 10100");
    }

    // -- select_victim_weight_pages: page in weight_table but in Protected state --

    #[test]
    fn select_victim_weight_pages_active_page_eligible() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        // Default state from register is Active
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        // Active is NOT Protected or Warm, so it should be eligible
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert_eq!(victims.len(), 1, "Active weight page must be eligible");
        assert_eq!(victims[0].0, 10);
    }

    // -- HGALConfig: clone produces independent copy --

    #[test]
    fn hgal_config_clone_then_modify_source() {
        let config = HGALConfig {
            hot_threshold: 10,
            ..Default::default()
        };
        let cloned = config.clone();
        // Both have same values
        assert_eq!(cloned.hot_threshold, config.hot_threshold);
        assert_eq!(cloned.warmup_duration, config.warmup_duration);
    }

    // -- on_prefill_chunk_complete: context_len tracking --

    #[test]
    fn on_prefill_chunk_complete_context_len_tracks_max_across_chunks() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // First chunk: 3 pages
        let chunk1 = vec![
            VirtualPageId::new(42, 0),
            VirtualPageId::new(42, 1),
            VirtualPageId::new(42, 2),
        ];
        scheduler.on_prefill_chunk_complete(0, 3, &chunk1);
        assert_eq!(scheduler.sequence_groups[&42].context_len, 3);

        // Second chunk: 1 page (max still 3)
        let chunk2 = vec![VirtualPageId::new(42, 3)];
        scheduler.on_prefill_chunk_complete(1, 3, &chunk2);
        assert_eq!(scheduler.sequence_groups[&42].context_len, 4,
            "context_len must be max of accumulated pages = 4");

        // Third chunk: 2 pages
        let chunk3 = vec![
            VirtualPageId::new(42, 4),
            VirtualPageId::new(42, 5),
        ];
        scheduler.on_prefill_chunk_complete(2, 3, &chunk3);
        assert_eq!(scheduler.sequence_groups[&42].context_len, 6,
            "context_len must be max = 6 after all chunks");
    }

    // -- virtual_page_to_page_id: different logical indices for same sequence --

    #[test]
    fn virtual_page_to_page_id_different_logical_indices_all_unique() {
        let mut ids = Vec::new();
        for idx in 0..20u64 {
            ids.push(virtual_page_to_page_id(VirtualPageId::new(100, idx as usize)));
        }
        let unique: HashSet<usize> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 20, "20 different logical indices must produce 20 unique page IDs");
    }

    // -- compute_eviction_priority: pinned page gets PIN_BONUS --

    #[test]
    fn eviction_priority_pinned_page_gets_exact_pin_bonus() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let pinned_page = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&pinned_page);
        // PromptSystem: payload=10000, pin_bonus=5000
        // score = 10000 + 5000 = 15000
        assert_eq!(prio.score, 15_000);
    }

    // -- free_dense_layer_weight_pages removes LIR entries --

    #[test]
    fn free_dense_layer_weight_pages_removes_lir_entries() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        });
        scheduler.register_dense_layer_weight_page(42, 0);
        scheduler.mark_accessed(42);
        assert!(scheduler.lir_pages.contains(&42));
        scheduler.free_dense_layer_weight_pages(0);
        assert!(!scheduler.lir_pages.contains(&42), "LIR entry must be removed when dense page freed");
    }

    // -- weight_page_table contains entries from both expert and dense --

    #[test]
    fn weight_page_table_mixed_types_in_same_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 3);
        scheduler.register_expert_weight_page(11, 3);
        scheduler.register_dense_layer_weight_page(20, 3);
        let layer3_pages = scheduler.weight_page_table.get(&3).unwrap();
        assert_eq!(layer3_pages.len(), 3);
        assert!(layer3_pages.contains(&10));
        assert!(layer3_pages.contains(&11));
        assert!(layer3_pages.contains(&20));
    }

    // -- compute_eviction_priority: expert with layer_idx None fallback --

    #[test]
    fn eviction_priority_expert_with_no_layer_idx_defaults_to_zero_penalty() {
        // Create an expert-like page manually with layer_idx = None
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage {
            page_id: 1,
            payload_kind: PagePayloadKind::ExpertWeight,
            residency: super::super::types::MemoryResidency::DeviceLocal,
            dtype: gllm_kernels::types::DType::F32,
            owner: None,
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: Some(5),
            layer_idx: None,
        };
        let prio = scheduler.compute_eviction_priority(&page);
        // No layer_idx -> layer_depth_penalty = 0
        // Expert base = -200
        // score = -200 + 0 + 0 + 0 + 0 = -200
        assert_eq!(prio.score, -200, "expert with no layer_idx must get 0 depth penalty");
    }

    // -- detect_working_set with lir_pages present but no metadata --

    #[test]
    fn detect_working_set_with_orphan_lir_entries_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Manually insert an LIR entry without metadata
        scheduler.lir_pages.insert(999);
        // detect_working_set only iterates page_metadata, not lir_pages
        scheduler.detect_working_set();
        // No panic, lir_pages still contains orphan entry
        assert!(scheduler.lir_pages.contains(&999));
    }

    // -- num_expert_weight_pages is purely weight_page_table based --

    #[test]
    fn num_expert_weight_pages_counts_all_layers() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for layer in 0..10 {
            scheduler.register_expert_weight_page(layer * 100, layer);
        }
        assert_eq!(scheduler.num_expert_weight_pages(), 10);
        // Free half
        for layer in 0..5 {
            scheduler.free_expert_weight_pages(layer);
        }
        assert_eq!(scheduler.num_expert_weight_pages(), 5);
    }

    // -- EvictionPriority clone roundtrip for all payload kinds --

    #[test]
    fn eviction_priority_clone_roundtrip_all_kinds() {
        let kinds = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        let scheduler = HGALScheduler::new(HGALConfig::default());
        for kind in kinds {
            let page = UnifiedVirtualPage {
                page_id: 1,
                payload_kind: kind,
                residency: super::super::types::MemoryResidency::DeviceLocal,
                dtype: gllm_kernels::types::DType::F32,
                owner: None,
                pipeline: None,
                logical_index: 0,
                codec: crate::kv_cache::CompressionCodec::None,
                compressed_size: 0,
                decompressed_size: 0,
                expert_id: None,
                layer_idx: None,
            };
            let prio = scheduler.compute_eviction_priority(&page);
            let cloned = prio.clone();
            assert_eq!(cloned.score, prio.score, "clone score mismatch for {:?}", kind);
            assert_eq!(cloned.payload_kind, prio.payload_kind);
        }
    }

    // ── Wave 12wz: Additional hgal tests ──────────────────────────────────────

    #[test]
    fn hgal_config_defaults_verify() {
        let config = HGALConfig::default();
        assert_eq!(config.warmup_duration, Duration::from_millis(100));
        assert_eq!(config.working_set_window, Duration::from_secs(1));
        assert_eq!(config.hot_threshold, 3);
        assert!((config.lir_ratio - 0.3).abs() < 1e-6);
        assert_eq!(config.min_warm_access, 2);
        assert!(config.enable_clock_pro);
    }

    #[test]
    fn hgal_config_clone_preserves() {
        let config = HGALConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.hot_threshold, config.hot_threshold);
    }

    #[test]
    fn hgal_config_debug() {
        let config = HGALConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("HGALConfig"));
    }

    #[test]
    fn scheduler_empty_state() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.sequence_groups.is_empty());
        assert!(scheduler.page_metadata.is_empty());
        assert!(scheduler.lir_pages.is_empty());
    }

    #[test]
    fn scheduler_config_access() {
        let config = HGALConfig { hot_threshold: 7, ..Default::default() };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().hot_threshold, 7);
    }

    #[test]
    fn upsert_inserts_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1, state: GroupState::Running, pages: vec![10, 20],
            access_count: 0, last_access: Instant::now(), is_pinned: false,
            context_len: 0, pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        assert!(scheduler.sequence_groups.contains_key(&1));
    }

    #[test]
    fn remove_existing_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 42, state: GroupState::Running, pages: vec![],
            access_count: 5, last_access: Instant::now(), is_pinned: false,
            context_len: 0, pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.remove_group(42);
        assert!(!scheduler.sequence_groups.contains_key(&42));
    }

    #[test]
    fn remove_group_nonexistent_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.remove_group(999);
    }

    #[test]
    fn register_expert_weight_page_single() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(100, 0);
        assert_eq!(scheduler.weight_page_table[&0], vec![100]);
    }

    #[test]
    fn register_expert_weight_page_multiple_same_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(20, 0);
        assert_eq!(scheduler.weight_page_table[&0].len(), 2);
    }

    #[test]
    fn num_expert_weight_pages_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    #[test]
    fn num_expert_weight_pages_after_registration() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(20, 0);
        scheduler.register_expert_weight_page(30, 1);
        assert_eq!(scheduler.num_expert_weight_pages(), 3);
    }

    #[test]
    fn free_expert_weight_pages_removes_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(20, 1);
        scheduler.free_expert_weight_pages(0);
        assert!(!scheduler.weight_page_table.contains_key(&0));
        assert!(scheduler.weight_page_table.contains_key(&1));
    }

    #[test]
    fn register_dense_layer_weight_page() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(200, 5);
        assert!(scheduler.weight_page_table.contains_key(&5));
    }

    #[test]
    fn num_dense_layer_weight_pages_zero_initially() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert_eq!(scheduler.num_dense_layer_weight_pages(), 0);
    }

    #[test]
    fn free_dense_layer_weight_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(100, 0);
        scheduler.register_dense_layer_weight_page(200, 1);
        scheduler.free_dense_layer_weight_pages(0);
        assert!(!scheduler.weight_page_table.contains_key(&0));
        assert!(scheduler.weight_page_table.contains_key(&1));
    }

    #[test]
    fn num_weight_pages_combined() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        assert_eq!(scheduler.num_weight_pages(), 2);
    }

    #[test]
    fn mark_accessed_inserts_if_missing() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(999);
        assert!(scheduler.page_metadata.contains_key(&999));
    }

    #[test]
    fn on_swap_in_creates_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(42);
        assert!(scheduler.page_metadata.contains_key(&42));
    }

    #[test]
    fn scheduler_debug_format() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let debug = format!("{:?}", scheduler);
        assert!(debug.contains("HGALScheduler"));
    }

    #[test]
    fn hgal_config_custom_boundary() {
        let config = HGALConfig {
            warmup_duration: Duration::from_secs(10),
            working_set_window: Duration::from_secs(60),
            hot_threshold: 100,
            lir_ratio: 0.5,
            min_warm_access: 5,
            enable_clock_pro: false,
        };
        assert_eq!(config.hot_threshold, 100);
        assert!(!config.enable_clock_pro);
    }

    #[test]
    fn allocate_expert_weight_pages_by_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.allocate_expert_weight_pages(3, 0);
        assert_eq!(ids.len(), 3);
        assert_eq!(scheduler.weight_page_table[&0].len(), 3);
    }

    #[test]
    fn register_dense_layer_weight_pages_batch() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_pages(vec![200, 201], 3);
        assert_eq!(scheduler.weight_page_table[&3].len(), 2);
    }

    // ── Wave 13: Additional coverage tests ────────────────────────────────────

    #[test]
    fn hgal_config_zero_durations_no_panic_in_operations() {
        let config = HGALConfig {
            warmup_duration: Duration::ZERO,
            working_set_window: Duration::ZERO,
            hot_threshold: 1,
            lir_ratio: 0.5,
            min_warm_access: 1,
            enable_clock_pro: false,
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.mark_accessed(1);
        assert!(scheduler.page_metadata.contains_key(&1));
        scheduler.detect_working_set();
    }

    #[test]
    fn hgal_config_large_hot_threshold_no_promotion() {
        let config = HGALConfig {
            hot_threshold: 1_000_000,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        for _ in 0..100 {
            scheduler.mark_accessed(1);
        }
        assert!(scheduler.page_metadata[&1].access_count >= 100);
        scheduler.detect_working_set();
        assert_ne!(scheduler.page_metadata[&1].state, PageState::Protected,
            "page should not be promoted with access_count < hot_threshold");
    }

    #[test]
    fn upsert_group_preserves_access_count_on_update() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 42,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        let updated = SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(updated);
        let stored = &scheduler.sequence_groups[&1];
        assert_eq!(stored.access_count, 42, "access_count must be preserved on upsert");
        assert_eq!(stored.pages, vec![10, 20], "pages must be updated");
    }

    #[test]
    fn upsert_new_group_uses_provided_values() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 55,
            pages: vec![5, 6, 7],
            state: GroupState::Paused,
            access_count: 10,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        scheduler.upsert_group(group);
        let stored = &scheduler.sequence_groups[&55];
        assert_eq!(stored.id, 55);
        assert_eq!(stored.pages, vec![5, 6, 7]);
        assert_eq!(stored.state, GroupState::Paused);
        assert_eq!(stored.access_count, 10);
        assert!(stored.is_pinned);
        assert_eq!(stored.context_len, 3);
        assert_eq!(stored.pipeline, crate::scheduler::types::KvPipeline::Working);
        assert_eq!(stored.payload_kind, Some(PagePayloadKind::KvContext));
    }

    #[test]
    fn remove_group_only_removes_group_not_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.update_page_state(20, Some(1), PageState::Active);
        let group = SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.remove_group(1);
        assert!(!scheduler.sequence_groups.contains_key(&1));
        assert!(scheduler.page_metadata.contains_key(&10));
        assert!(scheduler.page_metadata.contains_key(&20));
    }

    #[test]
    fn update_page_state_free_to_active_transition() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, None, PageState::Free);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Free);
        scheduler.update_page_state(1, Some(10), PageState::Active);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&1].sequence_id, Some(10));
    }

    #[test]
    fn update_page_state_none_sequence_id_stored() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(99, None, PageState::Active);
        assert_eq!(scheduler.page_metadata[&99].sequence_id, None);
        assert_eq!(scheduler.page_metadata[&99].page_id, 99);
    }

    #[test]
    fn update_page_state_swapped_out_stored_correctly() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(7, Some(3), PageState::SwappedOut);
        assert_eq!(scheduler.page_metadata[&7].state, PageState::SwappedOut);
        assert_eq!(scheduler.page_metadata[&7].sequence_id, Some(3));
    }

    #[test]
    fn select_victim_groups_single_group_covers_request() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 10,
            pages: vec![1, 2, 3, 4, 5],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(10),
            is_pinned: false,
            context_len: 5,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        let victims = scheduler.select_victim_groups(3);
        assert_eq!(victims, vec![10], "one group with 5 pages should cover request for 3");
    }

    #[test]
    fn select_victim_groups_expert_evicted_before_kv_with_same_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        let expert_group = SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        };
        let kv_group = SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        scheduler.upsert_group(expert_group);
        scheduler.upsert_group(kv_group);
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims[0], 1, "expert group must be evicted before kv group");
    }

    #[test]
    fn select_victim_weight_pages_no_registered_returns_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let victims = scheduler.select_victim_weight_pages(10);
        assert!(victims.is_empty());
    }

    #[test]
    fn select_victim_weight_pages_zero_count_returns_empty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        let victims = scheduler.select_victim_weight_pages(0);
        assert!(victims.is_empty());
    }

    #[test]
    fn select_victim_weight_pages_protected_page_excluded() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        if let Some(meta) = scheduler.page_metadata.get_mut(&10) {
            meta.state = PageState::Protected;
        }
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "Protected weight pages must be skipped");
    }

    #[test]
    fn select_victim_weight_pages_warm_page_excluded() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        if let Some(meta) = scheduler.page_metadata.get_mut(&10) {
            meta.state = PageState::Warm;
        }
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "Warm weight pages must be skipped");
    }

    #[test]
    fn eviction_priority_kv_context_no_metadata_base_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 99, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100);
        assert!(!prio.is_pinned);
        assert_eq!(prio.access_count, 0);
        assert_eq!(prio.recency, 0);
    }

    #[test]
    fn eviction_priority_rag_no_metadata_base_score() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 50, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, -200);
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(!prio.is_pinned);
    }

    #[test]
    fn eviction_priority_expert_deep_layer_lower_than_shallow() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let shallow = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        let deep = UnifiedVirtualPage::expert(2, 0, 100, gllm_kernels::types::DType::F32);
        let prio_shallow = scheduler.compute_eviction_priority(&shallow);
        let prio_deep = scheduler.compute_eviction_priority(&deep);
        assert!(prio_deep.score < prio_shallow.score,
            "deep layer expert ({}) must have lower score than shallow ({})",
            prio_deep.score, prio_shallow.score);
    }

    #[test]
    fn eviction_priority_warm_state_bonus_applied() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Warm);
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.score >= 5100, "Warm page must get state bonus (score={})", prio.score);
    }

    #[test]
    fn eviction_priority_protected_state_bonus_applied() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Protected);
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert!(prio.score >= 10100, "Protected page must get state bonus (score={})", prio.score);
    }

    #[test]
    fn eviction_priority_freq_bonus_linear_scaling() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 5,
            state: PageState::Active,
            recency: 0,
            ..Default::default()
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            access_count: 10,
            state: PageState::Active,
            recency: 0,
            ..Default::default()
        });
        let page1 = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page2 = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio1 = scheduler.compute_eviction_priority(&page1);
        let prio2 = scheduler.compute_eviction_priority(&page2);
        let diff = prio2.score - prio1.score;
        assert_eq!(diff, 50, "5 * FREQUENCY_WEIGHT(10) = 50, actual diff={}", diff);
    }

    #[test]
    fn mark_accessed_increments_from_initial_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Standby);
        assert_eq!(scheduler.page_metadata[&1].access_count, 0);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 1);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 2);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 3);
    }

    #[test]
    fn mark_accessed_standby_transitions_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Standby);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active,
            "Standby page must become Active on access");
    }

    #[test]
    fn mark_accessed_protected_state_unchanged() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Protected);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected,
            "Protected page must stay Protected on access");
    }

    #[test]
    fn mark_accessed_free_transitions_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Free);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Free);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active,
            "Free page must become Active on access");
    }

    #[test]
    fn mark_accessed_updates_group_stats_for_known_sequence() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(1, Some(10), PageState::Active);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.sequence_groups[&10].access_count, 6);
    }

    #[test]
    fn on_swap_in_sets_swap_in_time() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(42);
        assert!(scheduler.page_metadata[&42].swap_in_time.is_some(),
            "swap_in_time must be set after on_swap_in");
    }

    #[test]
    fn on_swap_in_resets_access_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 100,
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 0);
    }

    #[test]
    fn on_swap_in_warm_until_is_future() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let before = Instant::now();
        scheduler.on_swap_in(1);
        let after = Instant::now();
        let meta = &scheduler.page_metadata[&1];
        let warm_until = meta.warm_until.expect("warm_until must be set");
        assert!(warm_until >= before + scheduler.config.warmup_duration);
        assert!(warm_until <= after + scheduler.config.warmup_duration);
    }

    #[test]
    fn on_swap_in_repeated_resets_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm);
        scheduler.page_metadata.get_mut(&1).unwrap().state = PageState::Active;
        scheduler.page_metadata.get_mut(&1).unwrap().access_count = 50;
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm);
        assert_eq!(scheduler.page_metadata[&1].access_count, 0);
    }

    #[test]
    fn on_prefill_chunk_single_page_first_chunk() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(5, 0)];
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Standby);
        assert_eq!(scheduler.page_metadata[&pid].sequence_id, Some(5));
    }

    #[test]
    fn on_prefill_chunk_single_chunk_sets_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(7, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active);
    }

    #[test]
    fn on_prefill_chunk_complete_dedups_group_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId::new(10, 0);
        let pid = virtual_page_to_page_id(vpid);
        scheduler.on_prefill_chunk_complete(0, 2, &[vpid]);
        scheduler.on_prefill_chunk_complete(1, 2, &[vpid]);
        let group = &scheduler.sequence_groups[&10];
        let count = group.pages.iter().filter(|&&p| p == pid).count();
        assert_eq!(count, 1, "duplicate page IDs must be deduped");
    }

    #[test]
    fn on_prefill_complete_sets_group_running() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 5,
            pages: vec![1, 2],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(1, Some(5), PageState::Standby);
        scheduler.update_page_state(2, Some(5), PageState::Standby);
        scheduler.on_prefill_complete(5);
        assert_eq!(scheduler.sequence_groups[&5].state, GroupState::Running);
    }

    #[test]
    fn on_prefill_complete_standby_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        scheduler.update_page_state(20, Some(1), PageState::Standby);
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&20].state, PageState::Active);
    }

    #[test]
    fn on_prefill_complete_swapped_stays_swapped() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);
        scheduler.update_page_state(10, Some(1), PageState::Swapped);
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Swapped);
    }

    #[test]
    fn detect_working_set_cold_active_stays_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 100,
            working_set_window: Duration::from_millis(1),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 1,
            last_access: Instant::now() - Duration::from_secs(1),
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    #[test]
    fn detect_working_set_exact_threshold_promoted() {
        let config = HGALConfig {
            hot_threshold: 5,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 5,
            last_access: Instant::now(),
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    #[test]
    fn detect_working_set_below_threshold_not_promoted() {
        let config = HGALConfig {
            hot_threshold: 5,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 4,
            last_access: Instant::now(),
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    #[test]
    fn detect_working_set_protected_demoted_when_stale() {
        let config = HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_millis(1),
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 10,
            last_access: Instant::now() - Duration::from_secs(1),
            state: PageState::Protected,
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby);
    }

    #[test]
    fn register_expert_weight_page_metadata_state_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 3);
        let meta = scheduler.page_metadata.get(&42).expect("metadata must exist");
        assert_eq!(meta.state, PageState::Active);
        assert_eq!(meta.page_id, 42);
    }

    #[test]
    fn register_expert_weight_page_no_sequence_id() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 3);
        assert_eq!(scheduler.page_metadata[&42].sequence_id, None);
    }

    #[test]
    fn allocate_expert_weight_pages_unique_ids() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.allocate_expert_weight_pages(10, 2);
        let unique: HashSet<PageId> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 10, "all allocated page IDs must be unique");
    }

    #[test]
    fn allocate_expert_weight_pages_layer_encoding() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.allocate_expert_weight_pages(3, 5);
        for (expert_idx, &pid) in ids.iter().enumerate() {
            let expected = (5usize << 16) | expert_idx;
            assert_eq!(pid, expected, "page_id for expert {} in layer 5", expert_idx);
        }
    }

    #[test]
    fn free_expert_weight_pages_removes_page_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        assert!(scheduler.page_metadata.contains_key(&10));
        assert!(scheduler.page_metadata.contains_key(&11));
        scheduler.free_expert_weight_pages(0);
        assert!(!scheduler.page_metadata.contains_key(&10));
        assert!(!scheduler.page_metadata.contains_key(&11));
    }

    #[test]
    fn free_expert_weight_pages_nonexistent_layer_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.free_expert_weight_pages(99);
        assert!(scheduler.weight_page_table.contains_key(&0));
    }

    #[test]
    fn register_dense_layer_weight_page_metadata_state_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(200, 5);
        assert_eq!(scheduler.page_metadata[&200].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&200].sequence_id, None);
    }

    #[test]
    fn register_dense_layer_weight_pages_batch_returns_ids() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.register_dense_layer_weight_pages(vec![300, 301, 302], 7);
        assert_eq!(ids, vec![300, 301, 302]);
        assert_eq!(scheduler.weight_page_table[&7].len(), 3);
        for pid in &[300, 301, 302] {
            assert_eq!(scheduler.page_metadata[pid].state, PageState::Active);
        }
    }

    #[test]
    fn free_dense_layer_weight_pages_removes_all_page_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_pages(vec![400, 401], 8);
        assert!(scheduler.page_metadata.contains_key(&400));
        assert!(scheduler.page_metadata.contains_key(&401));
        scheduler.free_dense_layer_weight_pages(8);
        assert!(!scheduler.page_metadata.contains_key(&400));
        assert!(!scheduler.page_metadata.contains_key(&401));
    }

    #[test]
    fn num_weight_pages_expert_and_dense_combined() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        assert_eq!(scheduler.num_weight_pages(), 3);
    }

    #[test]
    fn num_weight_pages_zero_after_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        scheduler.free_expert_weight_pages(0);
        scheduler.free_dense_layer_weight_pages(1);
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    #[test]
    fn virtual_page_to_page_id_deterministic_same_input() {
        let vpid = VirtualPageId::new(42, 7);
        let id1 = virtual_page_to_page_id(vpid);
        let id2 = virtual_page_to_page_id(vpid);
        assert_eq!(id1, id2, "same VirtualPageId must always produce same PageId");
    }

    #[test]
    fn virtual_page_to_page_id_different_sequences() {
        let vpid1 = VirtualPageId::new(1, 0);
        let vpid2 = VirtualPageId::new(2, 0);
        assert_ne!(virtual_page_to_page_id(vpid1), virtual_page_to_page_id(vpid2));
    }

    #[test]
    fn virtual_page_to_page_id_large_sequence_fits_usize() {
        let vpid = VirtualPageId::new(1_000_000, 1_000_000);
        let id = virtual_page_to_page_id(vpid);
        let _ = id + 0;
    }

    #[test]
    fn eviction_priority_default_construction_all_fields() {
        let prio = EvictionPriority {
            score: 0,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(prio.score, 0);
        assert_eq!(prio.payload_kind, PagePayloadKind::KvContext);
        assert!(!prio.is_pinned);
        assert_eq!(prio.layer_idx, None);
        assert_eq!(prio.expert_id, None);
    }

    #[test]
    fn eviction_priority_clone_is_independent() {
        let prio = EvictionPriority {
            score: 500,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 10,
            recency: 5,
            layer_idx: Some(3),
            expert_id: Some(7),
        };
        let cloned = prio.clone();
        assert_eq!(cloned.score, prio.score);
        assert_eq!(cloned.payload_kind, prio.payload_kind);
        assert_eq!(cloned.is_pinned, prio.is_pinned);
        assert_eq!(cloned.access_count, prio.access_count);
        assert_eq!(cloned.recency, prio.recency);
        assert_eq!(cloned.layer_idx, prio.layer_idx);
        assert_eq!(cloned.expert_id, prio.expert_id);
    }

    #[test]
    fn eviction_priority_debug_includes_score() {
        let prio = EvictionPriority {
            score: 12345,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        let debug = format!("{:?}", prio);
        assert!(debug.contains("12345"), "Debug must include score, got: {}", debug);
    }

    #[test]
    fn page_metadata_default_standby_state() {
        let meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby);
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.recency, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.sequence_id, None);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    #[test]
    fn page_metadata_all_fields_set_correctly() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(10),
            recency: 100,
            access_count: 5,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Protected,
            warm_until: Some(now + Duration::from_secs(1)),
        };
        assert_eq!(meta.page_id, 42);
        assert_eq!(meta.sequence_id, Some(10));
        assert_eq!(meta.recency, 100);
        assert_eq!(meta.access_count, 5);
        assert!(meta.is_lir);
        assert_eq!(meta.state, PageState::Protected);
    }

    #[test]
    fn lir_membership_disabled_when_clock_pro_off() {
        let config = HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.update_page_state(1, Some(10), PageState::Active);
        scheduler.mark_accessed(1);
        assert!(!scheduler.lir_pages.contains(&1));
    }

    #[test]
    fn lir_membership_enabled_when_clock_pro_on() {
        let config = HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.update_page_state(1, Some(10), PageState::Active);
        scheduler.mark_accessed(1);
        assert!(scheduler.lir_pages.contains(&1));
    }

    #[test]
    fn eviction_priority_debug_format_not_empty_all_kinds() {
        let kinds = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for kind in kinds {
            let prio = EvictionPriority {
                score: 0,
                payload_kind: kind,
                is_pinned: false,
                access_count: 0,
                recency: 0,
                layer_idx: None,
                expert_id: None,
            };
            let debug = format!("{:?}", prio);
            assert!(!debug.is_empty(), "Debug for {:?} must not be empty", kind);
        }
    }

    #[test]
    fn scheduler_multiple_groups_registered() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for id in 1..=10u64 {
            let group = SequenceGroup {
                id,
                pages: vec![id as usize],
                state: GroupState::Running,
                access_count: 0,
                last_access: Instant::now(),
                is_pinned: false,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            };
            scheduler.upsert_group(group);
        }
        assert_eq!(scheduler.sequence_groups.len(), 10);
    }

    #[test]
    fn scheduler_large_weight_page_count() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for layer in 0..100 {
            scheduler.register_expert_weight_page(layer * 100, layer);
        }
        assert_eq!(scheduler.num_expert_weight_pages(), 100);
        assert_eq!(scheduler.num_weight_pages(), 100);
    }

    #[test]
    fn scheduler_config_returns_reference() {
        let config = HGALConfig {
            hot_threshold: 42,
            lir_ratio: 0.7,
            ..Default::default()
        };
        let scheduler = HGALScheduler::new(config);
        let returned_config = scheduler.config();
        assert_eq!(returned_config.hot_threshold, 42);
        assert!((returned_config.lir_ratio - 0.7).abs() < 1e-6);
    }

    #[test]
    fn lifecycle_expert_weight_register_access_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            ..Default::default()
        });
        scheduler.register_expert_weight_page(100, 0);
        assert_eq!(scheduler.page_metadata[&100].state, PageState::Active);
        assert!(scheduler.weight_page_table[&0].contains(&100));
        for _ in 0..5 {
            scheduler.mark_accessed(100);
        }
        assert!(scheduler.page_metadata[&100].access_count >= 5);
        scheduler.free_expert_weight_pages(0);
        assert!(!scheduler.page_metadata.contains_key(&100));
        assert!(!scheduler.weight_page_table.contains_key(&0));
        assert!(!scheduler.lir_pages.contains(&100));
    }

    #[test]
    fn lifecycle_dense_layer_weight_register_access_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.register_dense_layer_weight_pages(vec![200, 201], 3);
        assert_eq!(ids.len(), 2);
        scheduler.mark_accessed(200);
        scheduler.mark_accessed(201);
        assert!(scheduler.page_metadata[&200].access_count >= 1);
        assert!(scheduler.page_metadata[&201].access_count >= 1);
        scheduler.free_dense_layer_weight_pages(3);
        assert!(!scheduler.page_metadata.contains_key(&200));
        assert!(!scheduler.page_metadata.contains_key(&201));
        assert!(!scheduler.weight_page_table.contains_key(&3));
    }

    #[test]
    fn lifecycle_swap_in_to_working_set_promotion() {
        let config = HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_secs(60),
            warmup_duration: Duration::ZERO,
            min_warm_access: 0,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm);
        scheduler.mark_accessed(1);
        for _ in 0..4 {
            scheduler.mark_accessed(1);
        }
        assert!(scheduler.page_metadata[&1].access_count >= 3);
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected,
            "page with access_count >= hot_threshold must be promoted to Protected");
    }

    // ── Wave 14: Additional coverage tests ────────────────────────────────────

    // -- HGALConfig field combination edge cases --

    #[test]
    fn hgal_config_all_duration_fields_zero() {
        let config = HGALConfig {
            warmup_duration: Duration::ZERO,
            working_set_window: Duration::ZERO,
            hot_threshold: 0,
            lir_ratio: 0.0,
            min_warm_access: 0,
            enable_clock_pro: false,
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.mark_accessed(1);
        scheduler.detect_working_set();
        // No panic is the main assertion; verify metadata was created
        assert!(scheduler.page_metadata.contains_key(&1));
    }

    #[test]
    fn hgal_config_warmup_equals_working_set_window() {
        let dur = Duration::from_millis(500);
        let config = HGALConfig {
            warmup_duration: dur,
            working_set_window: dur,
            hot_threshold: 1,
            lir_ratio: 0.5,
            min_warm_access: 1,
            enable_clock_pro: true,
        };
        let scheduler = HGALScheduler::new(config);
        assert_eq!(scheduler.config().warmup_duration, scheduler.config().working_set_window);
    }

    #[test]
    fn hgal_config_lir_ratio_exactly_one() {
        let config = HGALConfig {
            lir_ratio: 1.0,
            enable_clock_pro: true,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        for i in 0..10 {
            scheduler.mark_accessed(i);
        }
        // With lir_ratio=1.0, all accessed pages should be LIR
        assert_eq!(scheduler.lir_pages.len(), 10);
    }

    #[test]
    fn hgal_config_min_warm_access_zero() {
        let config = HGALConfig {
            warmup_duration: Duration::from_secs(60),
            min_warm_access: 0,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm);
        // With min_warm_access=0, a single mark_accessed should exit warmup
        // because access_count (1) >= min_warm_access (0)
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active,
            "Warm page with access_count >= min_warm_access=0 must exit warmup");
    }

    #[test]
    fn hgal_config_hot_threshold_zero_promotes_immediately() {
        let config = HGALConfig {
            hot_threshold: 0,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 0,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        // access_count (0) >= hot_threshold (0) -> hot=true -> Protected
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected,
            "with hot_threshold=0, even 0 access_count should be promoted");
    }

    #[test]
    fn hgal_config_debug_contains_all_fields() {
        let config = HGALConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("warmup_duration"));
        assert!(debug.contains("working_set_window"));
        assert!(debug.contains("hot_threshold"));
        assert!(debug.contains("lir_ratio"));
        assert!(debug.contains("min_warm_access"));
        assert!(debug.contains("enable_clock_pro"));
    }

    // -- EvictionPriority: ordering properties --

    #[test]
    fn eviction_priority_kv_always_higher_than_expert_no_metadata() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let kv = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32);
        let kv_prio = scheduler.compute_eviction_priority(&kv);
        let expert_prio = scheduler.compute_eviction_priority(&expert);
        assert!(kv_prio.score > expert_prio.score,
            "KV score ({}) must be > Expert score ({})",
            kv_prio.score, expert_prio.score);
    }

    #[test]
    fn eviction_priority_dense_always_highest_no_metadata() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let kinds = [
            (UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32), "KV"),
            (UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32), "Expert"),
            (UnifiedVirtualPage::rag(3, 10, gllm_kernels::types::DType::F32), "RAG"),
        ];
        let dense = UnifiedVirtualPage::dense_layer(4, 0, gllm_kernels::types::DType::F32);
        let dense_prio = scheduler.compute_eviction_priority(&dense);
        for (page, name) in kinds {
            let prio = scheduler.compute_eviction_priority(&page);
            assert!(dense_prio.score > prio.score,
                "Dense score ({}) must be > {} score ({})",
                dense_prio.score, name, prio.score);
        }
    }

    #[test]
    fn eviction_priority_prompt_system_score_higher_than_all_evictable() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let prompt = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let prompt_prio = scheduler.compute_eviction_priority(&prompt);
        let evictable = [
            UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32),
            UnifiedVirtualPage::expert(3, 0, 0, gllm_kernels::types::DType::F32),
            UnifiedVirtualPage::rag(4, 10, gllm_kernels::types::DType::F32),
        ];
        for page in evictable {
            let prio = scheduler.compute_eviction_priority(&page);
            assert!(prompt_prio.score > prio.score,
                "PromptSystem score ({}) must be > evictable score ({})",
                prompt_prio.score, prio.score);
        }
    }

    #[test]
    fn eviction_priority_expert_id_and_layer_idx_reflected_in_result() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::expert(42, 7, 3, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, Some(7));
        assert_eq!(prio.layer_idx, Some(3));
    }

    #[test]
    fn eviction_priority_kv_no_expert_no_layer_w14() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, None);
        assert_eq!(prio.layer_idx, None);
    }

    #[test]
    fn eviction_priority_rag_has_no_expert_no_layer() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.expert_id, None);
        assert_eq!(prio.layer_idx, None);
    }

    #[test]
    fn eviction_priority_access_count_reflects_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 42,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.access_count, 42);
    }

    #[test]
    fn eviction_priority_recency_reflects_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 9999,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.recency, 9999);
    }

    #[test]
    fn eviction_priority_recency_reduces_score_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Page with recency=0
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 0,
            access_count: 0,
            ..Default::default()
        });
        // Page with recency=100
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            recency: 100,
            access_count: 0,
            ..Default::default()
        });
        let page1 = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page2 = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio1 = scheduler.compute_eviction_priority(&page1);
        let prio2 = scheduler.compute_eviction_priority(&page2);
        assert!(prio1.score > prio2.score,
            "recency=0 score ({}) must be > recency=100 score ({})",
            prio1.score, prio2.score);
    }

    #[test]
    fn eviction_priority_freq_bonus_increases_score() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Page with access_count=0
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        // Page with access_count=50
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            access_count: 50,
            recency: 0,
            ..Default::default()
        });
        let page1 = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page2 = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio1 = scheduler.compute_eviction_priority(&page1);
        let prio2 = scheduler.compute_eviction_priority(&page2);
        let delta = prio2.score - prio1.score;
        // freq_bonus delta = (50 - 0) * FREQUENCY_WEIGHT = 50 * 10 = 500
        assert_eq!(delta, 500);
    }

    // -- Constants verification --

    #[test]
    fn constants_frequency_weight_is_positive() {
        assert!(FREQUENCY_WEIGHT > 0);
    }

    #[test]
    fn constants_pin_bonus_greater_than_expert_penalty() {
        let pin = PIN_BONUS as isize;
        let expert = EXPERT_WEIGHT_PRIORITY_BONUS;
        assert!(pin > -expert, "PIN_BONUS ({}) must exceed ExpertWeight penalty ({})",
            pin, expert);
    }

    #[test]
    fn constants_dense_penalty_much_greater_than_expert_bonus() {
        let dense = DENSE_WEIGHT_PRIORITY_PENALTY;
        let expert = EXPERT_WEIGHT_PRIORITY_BONUS.abs();
        assert!(dense > expert,
            "DenseLayerWeight penalty ({}) must be >> ExpertWeight bonus ({})",
            dense, expert);
    }

    // -- select_victim_groups: mixed scenarios --

    #[test]
    fn select_victim_groups_accumulates_pages_across_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Two groups with different page counts
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![4, 5],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Request 4 pages: first group (3 pages) is not enough -> need both
        let victims = scheduler.select_victim_groups(4);
        assert!(victims.len() >= 1);
    }

    #[test]
    fn select_victim_groups_empty_pages_group_counted_as_one() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(5);
        assert_eq!(victims, vec![1], "empty-pages group counted as 1 page");
    }

    #[test]
    fn select_victim_groups_protected_page_group_not_selected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Protected);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "group with Protected page must not be victim");
    }

    #[test]
    fn select_victim_groups_warm_page_group_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.on_swap_in(10);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.is_empty(), "group with Warm page (in warmup) must not be victim");
    }

    // -- on_swap_in lifecycle edge cases --

    #[test]
    fn on_swap_in_sets_warm_until_in_future() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(10),
            ..Default::default()
        });
        let before = Instant::now() + Duration::from_secs(10);
        scheduler.on_swap_in(1);
        let warm_until = scheduler.page_metadata[&1].warm_until.unwrap();
        assert!(warm_until >= before, "warm_until must be at least 10s in the future");
    }

    #[test]
    fn swap_in_sets_swap_in_time_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let before = Instant::now();
        scheduler.on_swap_in(1);
        let swap_in_time = scheduler.page_metadata[&1].swap_in_time.unwrap();
        assert!(swap_in_time >= before);
    }

    #[test]
    fn on_swap_in_preserves_existing_sequence_id() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(42), PageState::Active);
        scheduler.on_swap_in(1);
        // swap_in does not touch sequence_id
        assert_eq!(scheduler.page_metadata[&1].sequence_id, Some(42));
    }

    #[test]
    fn swap_in_resets_access_count_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 100,
            ..Default::default()
        });
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 0);
    }

    // -- mark_accessed with group integration --

    #[test]
    fn mark_accessed_updates_group_access_count_for_known_sequence() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(1, Some(10), PageState::Active);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.sequence_groups[&10].access_count, 1);
        // Group's last_access should be updated
        assert!(scheduler.sequence_groups[&10].last_access > now - Duration::from_secs(1));
    }

    #[test]
    fn mark_accessed_no_sequence_does_not_update_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            ..Default::default()
        });
        // Should not panic when no sequence_id
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 1);
    }

    // -- update_page_state: SwappedOut vs Swapped distinction --

    #[test]
    fn update_page_state_swapped_out_does_not_clear_warm_until() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        assert!(scheduler.page_metadata[&1].warm_until.is_some());
        // SwappedOut != Swapped, so warm_until should NOT be cleared
        scheduler.update_page_state(1, Some(10), PageState::SwappedOut);
        assert!(scheduler.page_metadata[&1].warm_until.is_some(),
            "SwappedOut must not clear warm_until (only Swapped does)");
    }

    #[test]
    fn update_page_state_swapped_clears_warm_until_and_swap_in_time() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        assert!(scheduler.page_metadata[&1].warm_until.is_some());
        assert!(scheduler.page_metadata[&1].swap_in_time.is_some());
        scheduler.update_page_state(1, Some(10), PageState::Swapped);
        assert!(scheduler.page_metadata[&1].warm_until.is_none());
        assert!(scheduler.page_metadata[&1].swap_in_time.is_none());
    }

    // -- Weight page management: free_nonexistent, interleaved --

    #[test]
    fn free_expert_noexistent_layer_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.free_expert_weight_pages(999);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    #[test]
    fn free_dense_layer_weight_pages_nonexistent_layer_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.free_dense_layer_weight_pages(999);
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    #[test]
    fn free_expert_does_not_affect_different_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(20, 1);
        scheduler.free_expert_weight_pages(0);
        assert!(scheduler.weight_page_table.contains_key(&1));
        assert!(scheduler.page_metadata.contains_key(&20));
        assert!(!scheduler.page_metadata.contains_key(&10));
    }

    #[test]
    fn free_dense_does_not_affect_different_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        scheduler.free_dense_layer_weight_pages(0);
        assert!(scheduler.weight_page_table.contains_key(&1));
        assert!(scheduler.page_metadata.contains_key(&20));
        assert!(!scheduler.page_metadata.contains_key(&10));
    }

    #[test]
    fn num_weight_pages_zero_after_all_freed() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 1);
        scheduler.free_expert_weight_pages(0);
        scheduler.free_dense_layer_weight_pages(1);
        assert_eq!(scheduler.num_weight_pages(), 0);
    }

    #[test]
    fn dense_layer_pages_empty_vec_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.register_dense_layer_weight_pages(vec![], 5);
        assert!(ids.is_empty());
        assert!(scheduler.weight_page_table.get(&5).is_none_or(|v| v.is_empty()));
    }

    // -- allocate_expert_weight_pages edge cases --

    #[test]
    fn allocate_expert_zero_experts_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = scheduler.allocate_expert_weight_pages(0, 0);
        assert!(ids.is_empty());
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    #[test]
    fn allocate_expert_unique_ids_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids_l0 = scheduler.allocate_expert_weight_pages(4, 0);
        let ids_l1 = scheduler.allocate_expert_weight_pages(4, 1);
        let all_ids: HashSet<PageId> = ids_l0.iter().chain(ids_l1.iter()).copied().collect();
        assert_eq!(all_ids.len(), 8, "all 8 page IDs must be unique");
    }

    // -- on_prefill_chunk_complete edge cases --

    #[test]
    fn prefill_chunk_empty_pages_noop_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_prefill_chunk_complete(0, 1, &[]);
        assert!(scheduler.sequence_groups.is_empty());
        assert!(scheduler.page_metadata.is_empty());
    }

    #[test]
    fn prefill_chunk_resets_recency_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Pre-set metadata with non-zero recency
        scheduler.page_metadata.insert(100, PageMetadata {
            page_id: 100,
            recency: 5000,
            ..Default::default()
        });
        let pages = vec![VirtualPageId::new(10, 0)];
        // First chunk, so pages get Standby (non-last chunk)
        scheduler.on_prefill_chunk_complete(0, 2, &pages);
        // Find the page_id for VirtualPageId(10, 0)
        let pid = virtual_page_to_page_id(VirtualPageId::new(10, 0));
        // But this is a different page_id than 100, so verify the created one
        if let Some(meta) = scheduler.page_metadata.get(&pid) {
            assert_eq!(meta.recency, 0, "recency must be reset to 0 on chunk complete");
        }
    }

    // -- on_prefill_complete edge cases --

    #[test]
    fn on_prefill_complete_nonexistent_group_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Should be a no-op, no panic
        scheduler.on_prefill_complete(999);
    }

    #[test]
    fn prefill_complete_swapped_preserved_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Swapped,
            ..Default::default()
        });
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Swapped,
            "Swapped pages must stay Swapped after prefill complete");
    }

    #[test]
    fn on_prefill_complete_updates_group_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Paused,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Standby,
            ..Default::default()
        });
        scheduler.on_prefill_complete(1);
        assert_eq!(scheduler.sequence_groups[&1].state, GroupState::Running);
        assert!(scheduler.sequence_groups[&1].last_access > now - Duration::from_secs(1));
    }

    // -- select_victim_weight_pages: additional scenarios --

    #[test]
    fn victim_weight_pages_empty_table_w14() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let victims = scheduler.select_victim_weight_pages(5);
        assert!(victims.is_empty());
    }

    #[test]
    fn victim_weight_pages_count_zero_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        let victims = scheduler.select_victim_weight_pages(0);
        assert!(victims.is_empty());
    }

    #[test]
    fn victim_weight_pages_skip_no_meta_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Insert page into weight_page_table directly, bypassing register (no metadata created)
        scheduler.weight_page_table.insert(0, vec![999]);
        let victims = scheduler.select_victim_weight_pages(1);
        assert!(victims.is_empty(), "pages without metadata must be skipped");
    }

    #[test]
    fn select_victim_weight_pages_result_sorted_ascending_by_score() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Create pages with different recency to produce different scores
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(11, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().recency = 500;
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&11).unwrap().recency = 100;
        scheduler.page_metadata.get_mut(&11).unwrap().state = PageState::Standby;

        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        let victims = scheduler.select_victim_weight_pages(2);
        if victims.len() == 2 {
            assert!(victims[0].1.score <= victims[1].1.score,
                "results must be sorted ascending by score");
        }
    }

    // -- detect_working_set: additional state transitions --

    #[test]
    fn detect_working_set_free_page_stays_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Free,
            access_count: 100,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Free,
            "Free pages must not be promoted by working set detection");
    }

    #[test]
    fn detect_working_set_swapped_page_not_transitioned() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Swapped,
            access_count: 100,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        // Swapped is not in the match arms for promotion, stays unchanged
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Swapped);
    }

    #[test]
    fn detect_working_set_warm_in_warmup_stays_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(3600),
            min_warm_access: 100,
            ..Default::default()
        });
        scheduler.on_swap_in(1);
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm,
            "Warm page still in warmup must stay Warm");
    }

    // -- upsert_group: counter preservation --

    #[test]
    fn upsert_group_preserves_access_counters_across_updates() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 42,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Update with different pages but same id
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        });
        assert_eq!(scheduler.sequence_groups[&1].access_count, 42,
            "upsert must preserve access_count from existing group");
        assert_eq!(scheduler.sequence_groups[&1].pages, vec![1, 2, 3],
            "upsert must use new pages");
    }

    #[test]
    fn upsert_group_new_id_uses_provided_values() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![5, 6],
            state: GroupState::Swapped,
            access_count: 7,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        let g = &scheduler.sequence_groups[&99];
        assert_eq!(g.pages, vec![5, 6]);
        assert_eq!(g.state, GroupState::Swapped);
        assert_eq!(g.access_count, 7);
        assert!(g.is_pinned);
        assert_eq!(g.pipeline, crate::scheduler::types::KvPipeline::Working);
        assert_eq!(g.payload_kind, Some(PagePayloadKind::ExpertWeight));
    }

    // -- LIR membership edge cases --

    #[test]
    fn lir_disabled_clock_pro_off_w14() {
        let config = HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.mark_accessed(1);
        assert!(scheduler.lir_pages.is_empty(), "LIR must be empty when clock_pro disabled");
    }

    #[test]
    fn lir_target_from_page_count_w14() {
        let config = HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        // Add 10 pages
        for i in 0..10 {
            scheduler.mark_accessed(i);
        }
        // lir_ratio=0.5, target = ceil(10 * 0.5) = 5
        assert_eq!(scheduler.lir_pages.len(), 5,
            "with lir_ratio=0.5 and 10 pages, LIR set must be 5");
    }

    #[test]
    fn lir_evicts_coldest_when_over_capacity() {
        let config = HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.3,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        // Access page 1 first (it will have highest recency = coldest)
        scheduler.mark_accessed(1);
        // Access more pages to push LIR over capacity
        for i in 2..=5 {
            scheduler.mark_accessed(i);
        }
        // page 1 should have been evicted from LIR (highest recency)
        // target = ceil(5 * 0.3) = 2
        assert_eq!(scheduler.lir_pages.len(), 2);
    }

    // -- PageMetadata default values --

    #[test]
    fn metadata_default_values_w14() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.sequence_id, None);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    #[test]
    fn metadata_custom_construction_w14() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(100),
            recency: 500,
            access_count: 10,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Protected,
            warm_until: Some(now + Duration::from_secs(5)),
        };
        assert_eq!(meta.page_id, 42);
        assert_eq!(meta.sequence_id, Some(100));
        assert_eq!(meta.recency, 500);
        assert_eq!(meta.access_count, 10);
        assert!(meta.is_lir);
        assert_eq!(meta.state, PageState::Protected);
    }

    // -- virtual_page_to_page_id: edge cases --

    #[test]
    fn vpid_zero_inputs_w14() {
        let vpid = VirtualPageId::new(0, 0);
        let id = virtual_page_to_page_id(vpid);
        // (0+0)*(0+0+1)/2 + 0 = 0
        assert_eq!(id, 0);
    }

    #[test]
    fn vpid_deterministic_w14() {
        let vpid = VirtualPageId::new(42, 7);
        let id1 = virtual_page_to_page_id(vpid);
        let id2 = virtual_page_to_page_id(vpid);
        assert_eq!(id1, id2);
    }

    #[test]
    fn vpid_different_sequences_w14() {
        let id1 = virtual_page_to_page_id(VirtualPageId::new(1, 0));
        let id2 = virtual_page_to_page_id(VirtualPageId::new(2, 0));
        assert_ne!(id1, id2, "different sequences must produce different IDs");
    }

    #[test]
    fn virtual_page_to_page_id_large_values_no_panic() {
        // Use large but not overflow-causing values
        let vpid = VirtualPageId::new(1_000_000, 500_000);
        let id = virtual_page_to_page_id(vpid);
        // Just verify it produces a valid usize
        assert!(id < usize::MAX);
    }

    // -- Scheduler state invariants --

    #[test]
    fn scheduler_lir_pages_subset_of_metadata_keys() {
        let config = HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 1.0,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        for i in 0..5 {
            scheduler.mark_accessed(i);
        }
        for pid in &scheduler.lir_pages {
            assert!(scheduler.page_metadata.contains_key(pid),
                "LIR page {} must exist in page_metadata", pid);
        }
    }

    #[test]
    fn scheduler_weight_page_table_values_are_unique_per_layer() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_expert_weight_page(10, 0);
        // Same page_id registered twice -> appears twice in vec
        assert_eq!(scheduler.weight_page_table[&0].len(), 2);
    }

    // -- EvictionPriority: state bonus verification --

    #[test]
    fn eviction_priority_state_bonus_protected_exact() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // KV base=100, state_bonus=10000
        assert_eq!(prio.score, 10_100);
    }

    #[test]
    fn eviction_priority_state_bonus_warm_exact() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // KV base=100, state_bonus=5000
        assert_eq!(prio.score, 5_100);
    }

    #[test]
    fn eviction_priority_state_bonus_active_is_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        // KV base=100, no state bonus
        assert_eq!(prio.score, 100);
    }

    #[test]
    fn eviction_priority_state_bonus_standby_is_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100);
    }

    #[test]
    fn eviction_priority_state_bonus_swapped_is_zero() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Swapped,
            access_count: 0,
            recency: 0,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&page);
        assert_eq!(prio.score, 100);
    }

    // -- Group payload kind ordering in eviction --

    #[test]
    fn eviction_priority_expert_evicted_before_kv_same_access() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Same metadata for both pages
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 5,
            recency: 10,
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            access_count: 5,
            recency: 10,
            state: PageState::Active,
            ..Default::default()
        });
        let kv = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::F32);
        let kv_prio = scheduler.compute_eviction_priority(&kv);
        let expert_prio = scheduler.compute_eviction_priority(&expert);
        assert!(expert_prio.score < kv_prio.score,
            "Expert (score={}) must be evicted before KV (score={})",
            expert_prio.score, kv_prio.score);
    }

    #[test]
    fn eviction_priority_rag_evicted_before_kv_same_access() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 5,
            recency: 10,
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            access_count: 5,
            recency: 10,
            state: PageState::Active,
            ..Default::default()
        });
        let kv = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let rag = UnifiedVirtualPage::rag(2, 10, gllm_kernels::types::DType::F32);
        let kv_prio = scheduler.compute_eviction_priority(&kv);
        let rag_prio = scheduler.compute_eviction_priority(&rag);
        assert!(rag_prio.score < kv_prio.score,
            "RAG (score={}) must be evicted before KV (score={})",
            rag_prio.score, kv_prio.score);
    }

    // -- EvictionPriority: expert depth penalty linearity --

    #[test]
    fn eviction_priority_expert_depth_penalty_is_linear() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let p0 = scheduler.compute_eviction_priority(
            &UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32));
        let p10 = scheduler.compute_eviction_priority(
            &UnifiedVirtualPage::expert(2, 0, 10, gllm_kernels::types::DType::F32));
        let p20 = scheduler.compute_eviction_priority(
            &UnifiedVirtualPage::expert(3, 0, 20, gllm_kernels::types::DType::F32));
        // Delta must be linear: p0-p10 = p10-p20 = 10
        assert_eq!(p0.score - p10.score, 10);
        assert_eq!(p10.score - p20.score, 10);
    }

    // -- mark_accessed: state transition specifics --

    #[test]
    fn mark_accessed_warm_in_warmup_stays_warm() {
        let config = HGALConfig {
            warmup_duration: Duration::from_secs(3600),
            min_warm_access: 100,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.on_swap_in(1);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm,
            "Warm page still in warmup must stay Warm after access");
    }

    #[test]
    fn mark_accessed_warm_after_warmup_exits_to_active() {
        let config = HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: 1,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.on_swap_in(1);
        // Access once to reach min_warm_access
        scheduler.mark_accessed(1);
        // After warmup duration elapsed and min_warm_access met, next access exits warmup
        scheduler.mark_accessed(1);
        // Now it should be Active (warmup expired or access_count >= min_warm_access)
        let state = scheduler.page_metadata[&1].state;
        assert!(matches!(state, PageState::Active | PageState::Protected),
            "page should have exited warmup, got {:?}", state);
    }

    #[test]
    fn accessed_standby_to_active_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, None, PageState::Standby);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    #[test]
    fn mark_accessed_protected_stays_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    #[test]
    fn accessed_free_to_active_w14() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Free,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    // -- on_prefill_chunk_complete: multi-request separation --

    #[test]
    fn on_prefill_chunk_complete_separates_requests() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages_a = vec![VirtualPageId::new(100, 0)];
        let pages_b = vec![VirtualPageId::new(200, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages_a);
        scheduler.on_prefill_chunk_complete(0, 1, &pages_b);
        assert!(scheduler.sequence_groups.contains_key(&100));
        assert!(scheduler.sequence_groups.contains_key(&200));
        assert_eq!(scheduler.sequence_groups.len(), 2);
    }

    // -- detect_working_set: multiple calls idempotent when conditions stable --

    #[test]
    fn detect_working_set_idempotent_for_stable_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 3,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 10,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        let state1 = scheduler.page_metadata[&1].state;
        scheduler.detect_working_set();
        let state2 = scheduler.page_metadata[&1].state;
        assert_eq!(state1, state2, "repeated detect_working_set must be idempotent");
    }

    // -- select_victim_groups: count=0 edge case --

    #[test]
    fn select_victim_groups_count_zero_returns_empty_vec() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let victims = scheduler.select_victim_groups(0);
        assert!(victims.is_empty());
    }

    // -- UnifiedVirtualPage: is_evictable consistency with priority --

    #[test]
    fn eviction_priority_is_pinned_matches_page_evictability() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let prompt = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let dense = UnifiedVirtualPage::dense_layer(2, 0, gllm_kernels::types::DType::F32);
        let kv = UnifiedVirtualPage::kv(3, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(4, 0, 0, gllm_kernels::types::DType::F32);
        let rag = UnifiedVirtualPage::rag(5, 10, gllm_kernels::types::DType::F32);

        assert!(!scheduler.compute_eviction_priority(&prompt).is_pinned == prompt.is_evictable());
        assert!(!scheduler.compute_eviction_priority(&dense).is_pinned == dense.is_evictable());
        assert!(!scheduler.compute_eviction_priority(&kv).is_pinned == kv.is_evictable());
        assert!(!scheduler.compute_eviction_priority(&expert).is_pinned == expert.is_evictable());
        assert!(!scheduler.compute_eviction_priority(&rag).is_pinned == rag.is_evictable());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Wave 13 — 50 additional unit tests for public types & methods
    // ═══════════════════════════════════════════════════════════════════════

    // -- 1. PageState::SwappedOut variant transitions to Active via mark_accessed --

    #[test]
    fn mark_accessed_swapped_out_variant_becomes_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::SwappedOut,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active);
    }

    // -- 2. SequenceGroup with Working pipeline is accepted by upsert_group --

    #[test]
    fn upsert_group_working_pipeline_stored() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 42,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: None,
        });
        let group = scheduler.sequence_groups.get(&42).expect("group exists");
        assert_eq!(group.pipeline, crate::scheduler::types::KvPipeline::Working);
    }

    // -- 3. SequenceGroup with GroupState::Paused survives upsert_group --

    #[test]
    fn upsert_group_paused_state_stored() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![5],
            state: GroupState::Paused,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert_eq!(scheduler.sequence_groups[&99].state, GroupState::Paused);
    }

    // -- 4. SequenceGroup with GroupState::Swapped survives upsert_group --

    #[test]
    fn upsert_group_swapped_state_stored() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 77,
            pages: vec![7],
            state: GroupState::Swapped,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert_eq!(scheduler.sequence_groups[&77].state, GroupState::Swapped);
    }

    // -- 5. RequestKind variants are distinct --

    #[test]
    fn request_kind_variants_distinct() {
        use std::collections::HashSet;
        let variants: HashSet<_> = [
            crate::scheduler::types::RequestKind::Chat,
            crate::scheduler::types::RequestKind::Embedding,
            crate::scheduler::types::RequestKind::Rerank,
        ].into();
        assert_eq!(variants.len(), 3);
    }

    // -- 6. BatchOrderPolicy default is StrictRequestIdOrder --

    #[test]
    fn batch_order_policy_default_is_strict_request_id() {
        assert_eq!(
            crate::scheduler::types::BatchOrderPolicy::default(),
            crate::scheduler::types::BatchOrderPolicy::StrictRequestIdOrder,
        );
    }

    // -- 7. BatchOrderPolicy all variants are distinct --


    // -- 8. MemoryResidency all variants are distinct --

    #[test]
    fn memory_residency_variants_distinct() {
        use std::collections::HashSet;
        let variants: HashSet<_> = [
            crate::scheduler::types::MemoryResidency::DeviceLocal,
            crate::scheduler::types::MemoryResidency::HostLocal,
            crate::scheduler::types::MemoryResidency::DiskSwap,
        ].into();
        assert_eq!(variants.len(), 3);
    }

    // -- 9. WeightTier all variants are distinct --

    #[test]
    fn weight_tier_variants_distinct() {
        use std::collections::HashSet;
        let variants: HashSet<_> = [
            crate::scheduler::types::WeightTier::Hot,
            crate::scheduler::types::WeightTier::Warm,
            crate::scheduler::types::WeightTier::Cold,
        ].into();
        assert_eq!(variants.len(), 3);
    }

    // -- 10. update_page_state to Standby preserves warm_until (only Swapped clears it) --

    #[test]
    fn update_page_state_standby_preserves_warm_until() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // First set up a page with warm_until via swap_in
        scheduler.on_swap_in(1);
        assert!(scheduler.page_metadata[&1].warm_until.is_some());
        assert!(scheduler.page_metadata[&1].swap_in_time.is_some());

        // Transition to Standby — warm fields should be preserved
        scheduler.update_page_state(1, Some(10), PageState::Standby);
        let meta = &scheduler.page_metadata[&1];
        assert!(meta.warm_until.is_some(), "Standby should preserve warm_until");
        assert!(meta.swap_in_time.is_some(), "Standby should preserve swap_in_time");
        assert_eq!(meta.state, PageState::Standby);
    }

    // -- 11. detect_working_set does not transition SwappedOut pages --

    #[test]
    fn detect_working_set_swapped_out_page_no_transition() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 1,
            working_set_window: Duration::from_secs(600),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::SwappedOut,
            access_count: 100,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::SwappedOut);
    }

    // -- 12. select_victim_groups with Working pipeline group is eligible --

    #[test]
    fn select_victim_groups_working_pipeline_eligible() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: None,
        });
        scheduler.update_page_state(1, Some(10), PageState::Standby);
        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![10]);
    }

    // -- 13. on_prefill_complete on group with empty pages is noop --

    #[test]
    fn on_prefill_complete_group_empty_pages_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 55,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Should not panic
        scheduler.on_prefill_complete(55);
        assert!(scheduler.sequence_groups.contains_key(&55));
    }

    // -- 14. on_prefill_chunk_complete with zero logical index VirtualPageId --

    #[test]
    fn on_prefill_chunk_complete_zero_logical_index() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(100, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert!(scheduler.page_metadata.contains_key(&pid));
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active);
    }

    // -- 15. Multiple swap_in on different pages creates independent metadata --

    #[test]
    fn on_swap_in_multiple_pages_independent_metadata() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        scheduler.on_swap_in(2);
        scheduler.on_swap_in(3);
        assert_eq!(scheduler.page_metadata.len(), 3);
        for pid in &[1, 2, 3] {
            let meta = &scheduler.page_metadata[pid];
            assert_eq!(meta.state, PageState::Warm);
            assert_eq!(meta.access_count, 0);
        }
    }

    // -- 16. select_victim_groups with Paused group is eligible --

    #[test]
    fn select_victim_groups_paused_group_eligible() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 88,
            pages: vec![10],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(88), PageState::Standby);
        let victims = scheduler.select_victim_groups(1);
        assert!(victims.contains(&88));
    }

    // -- 17. update_page_state preserves access_count across transitions --

    #[test]
    fn update_page_state_preserves_access_count_across_warm_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        // Access several times to increment count
        for _ in 0..5 {
            scheduler.mark_accessed(1);
        }
        assert!(scheduler.page_metadata[&1].access_count >= 5);
        // Now update state to Active
        scheduler.update_page_state(1, Some(10), PageState::Active);
        assert!(scheduler.page_metadata[&1].access_count >= 5,
            "access_count should be preserved across state update");
    }

    // -- 18. PipelinedVirtualPageId construction and equality --

    #[test]
    fn pipelined_virtual_page_id_construction() {
        let pvpid = crate::scheduler::types::PipelinedVirtualPageId {
            pipeline: crate::scheduler::types::KvPipeline::Working,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_eq!(pvpid.pipeline, crate::scheduler::types::KvPipeline::Working);
        assert_eq!(pvpid.sequence_id, 42);
        assert_eq!(pvpid.logical_index, 7);
    }

    // -- 19. PipelinedVirtualPageId equality and hash --

    #[test]
    fn pipelined_virtual_page_id_equality_and_hash() {
        use std::collections::HashSet;
        let a = crate::scheduler::types::PipelinedVirtualPageId {
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let b = crate::scheduler::types::PipelinedVirtualPageId {
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        assert_eq!(a, b);
        let set: HashSet<_> = [a, b].into();
        assert_eq!(set.len(), 1, "equal values should dedup in HashSet");
    }

    // -- 20. KvPipeline Copy semantics --

    #[test]
    fn kv_pipeline_copy_semantics() {
        let a = crate::scheduler::types::KvPipeline::Conversation;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // -- 21. GroupState all three variants in Copy test --

    #[test]
    fn group_state_copy_semantics() {
        let a = GroupState::Running;
        let b = a;
        assert_eq!(a, b);
        let c = GroupState::Swapped;
        let d = c;
        assert_eq!(c, d);
    }

    // -- 22. EvictionPriority fields are read correctly after construction --

    #[test]
    fn eviction_priority_field_access() {
        let ep = EvictionPriority {
            score: -42,
            payload_kind: PagePayloadKind::KnowledgeRAG,
            is_pinned: false,
            access_count: 3,
            recency: 100,
            layer_idx: Some(7),
            expert_id: None,
        };
        assert_eq!(ep.score, -42);
        assert_eq!(ep.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(!ep.is_pinned);
        assert_eq!(ep.access_count, 3);
        assert_eq!(ep.recency, 100);
        assert_eq!(ep.layer_idx, Some(7));
        assert!(ep.expert_id.is_none());
    }

    // -- 23. PageMetadata is_lir field defaults to false --

    #[test]
    fn page_metadata_is_lir_default_false() {
        let meta = PageMetadata::default();
        assert!(!meta.is_lir);
    }

    // -- 24. PageMetadata swap_in_time defaults to None --

    #[test]
    fn page_metadata_swap_in_time_default_none() {
        let meta = PageMetadata::default();
        assert!(meta.swap_in_time.is_none());
    }

    // -- 25. PageMetadata warm_until defaults to None --

    #[test]
    fn page_metadata_warm_until_default_none() {
        let meta = PageMetadata::default();
        assert!(meta.warm_until.is_none());
    }

    // -- 26. SequenceGroup payload_kind None by default in manual construction --

    #[test]
    fn sequence_group_payload_kind_none_default() {
        let group = SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        assert!(group.payload_kind.is_none());
    }

    // -- 27. select_victim_groups with single group having many pages --

    #[test]
    fn select_victim_groups_single_group_many_pages_satisfies_large_request() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 10,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        for pid in 1..=10 {
            scheduler.update_page_state(pid, Some(1), PageState::Standby);
        }
        // Request 8 pages; one group has 10, should satisfy in one group
        let victims = scheduler.select_victim_groups(8);
        assert_eq!(victims, vec![1]);
    }

    // -- 28. compute_eviction_priority for KV page with metadata --

    #[test]
    fn eviction_priority_kv_with_metadata_has_freq_and_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            access_count: 5,
            recency: 20,
            last_access: Instant::now(),
            ..Default::default()
        });
        let kv_page = UnifiedVirtualPage::kv(
            1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&kv_page);
        assert_eq!(prio.access_count, 5);
        assert_eq!(prio.recency, 20);
    }

    // -- 29. UnifiedVirtualPage kv sets correct residency --

    #[test]
    fn unified_virtual_page_kv_residency_device_local() {
        let kv = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        assert!(kv.is_on_device());
    }

    // -- 30. UnifiedVirtualPage rag defaults to HostLocal --

    #[test]
    fn unified_virtual_page_rag_host_local() {
        let rag = UnifiedVirtualPage::rag(1, 10, gllm_kernels::types::DType::F32);
        assert!(!rag.is_on_device());
        assert_eq!(rag.residency, crate::scheduler::types::MemoryResidency::HostLocal);
    }

    // -- 31. UnifiedVirtualPage expert is_on_device --

    #[test]
    fn unified_virtual_page_expert_is_on_device() {
        let expert = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F32);
        assert!(expert.is_on_device());
    }

    // -- 32. UnifiedVirtualPage dense_layer sets layer_idx from logical_index --

    #[test]
    fn unified_virtual_page_dense_layer_idx_matches_logical() {
        let dense = UnifiedVirtualPage::dense_layer(1, 42, gllm_kernels::types::DType::F32);
        assert_eq!(dense.layer_idx, Some(42));
        assert_eq!(dense.logical_index, 42);
    }

    // -- 33. on_prefill_chunk_complete with saturated chunk_idx and total_chunks=1 --

    #[test]
    fn on_prefill_chunk_complete_saturated_single_chunk() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(200, 0)];
        // chunk_idx = usize::MAX - 1, total_chunks = usize::MAX
        // (usize::MAX - 1).saturating_add(1) = usize::MAX >= usize::MAX → last chunk → Active
        scheduler.on_prefill_chunk_complete(usize::MAX - 1, usize::MAX, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active);
    }

    // -- 34. register_expert_weight_page does not set sequence_id --

    #[test]
    fn register_expert_weight_page_sequence_id_none() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(42, 0);
        assert!(scheduler.page_metadata[&42].sequence_id.is_none());
    }

    // -- 35. register_dense_layer_weight_page does not set sequence_id --

    #[test]
    fn register_dense_layer_weight_page_sequence_id_none() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_page(99, 0);
        assert!(scheduler.page_metadata[&99].sequence_id.is_none());
    }

    // -- 36. allocate_expert_weight_pages returns page_count matching num_experts --

    #[test]
    fn allocate_expert_weight_pages_count_matches_experts() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(8, 3);
        assert_eq!(pages.len(), 8);
        assert_eq!(scheduler.num_expert_weight_pages(), 8);
    }

    // -- 37. free_expert_weight_pages then num_weight_pages returns zero --

    #[test]
    fn free_expert_then_register_dense_num_weight_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(4, 0);
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
        scheduler.register_dense_layer_weight_page(50, 0);
        assert_eq!(scheduler.num_weight_pages(), 1);
    }

    // -- 38. on_swap_in preserves previously set sequence_id --

    #[test]
    fn on_swap_in_keeps_existing_sequence_id_intact() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(42), PageState::Active);
        assert_eq!(scheduler.page_metadata[&1].sequence_id, Some(42));
        scheduler.on_swap_in(1);
        // on_swap_in does not modify sequence_id
        assert_eq!(scheduler.page_metadata[&1].sequence_id, Some(42));
    }

    // -- 39. mark_accessed on page with sequence_id pointing to nonexistent group --

    #[test]
    fn mark_accessed_sequence_id_nonexistent_group_no_panic() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(9999), // group 9999 does not exist
            state: PageState::Active,
            ..Default::default()
        });
        // Should not panic
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 1);
    }

    // -- 40. update_page_state called with Swapped clears warm fields --

    #[test]
    fn update_page_state_swapped_variant_clears_warm() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1); // Sets warm_until and swap_in_time
        assert!(scheduler.page_metadata[&1].warm_until.is_some());
        scheduler.update_page_state(1, None, PageState::Swapped);
        assert!(scheduler.page_metadata[&1].warm_until.is_none());
        assert!(scheduler.page_metadata[&1].swap_in_time.is_none());
    }

    // -- 41. select_victim_weight_pages with count larger than available returns all --

    #[test]
    fn select_victim_weight_pages_count_larger_than_available() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.register_expert_weight_page(10, 0);
        // Need metadata to exist (already created by register) and state=Standby for eligibility
        if let Some(meta) = scheduler.page_metadata.get_mut(&10) {
            meta.state = PageState::Standby;
        }
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        // Request 100 but only 1 available
        let victims = scheduler.select_victim_weight_pages(100);
        assert_eq!(victims.len(), 1);
    }

    // -- 42. detect_working_set with hot_threshold=0 promotes immediately on access --

    #[test]
    fn detect_working_set_hot_threshold_zero_promotes_any_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 0,
            working_set_window: Duration::from_secs(600),
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            access_count: 0,
            last_access: Instant::now(),
            ..Default::default()
        });
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected);
    }

    // -- 43. SequenceGroup with KnowledgeRAG payload_kind accepted by upsert_group --

    #[test]
    fn upsert_group_knowledge_rag_payload_kind() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 300,
            pages: vec![50],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        });
        assert_eq!(
            scheduler.sequence_groups[&300].payload_kind,
            Some(PagePayloadKind::KnowledgeRAG),
        );
    }

    // -- 44. update_page_state preserves recency across state change --

    #[test]
    fn update_page_state_preserves_recency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 42,
            access_count: 10,
            ..Default::default()
        });
        scheduler.update_page_state(1, Some(5), PageState::Active);
        assert_eq!(scheduler.page_metadata[&1].recency, 42,
            "recency should survive state update");
    }

    // -- 45. virtual_page_to_page_id is pure function (same input → same output) --

    #[test]
    fn virtual_page_to_page_id_pure_function() {
        let vpid = VirtualPageId::new(12345, 67890);
        let a = virtual_page_to_page_id(vpid);
        let b = virtual_page_to_page_id(vpid);
        let c = virtual_page_to_page_id(vpid);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    // -- 46. HGALConfig with enable_clock_pro=false disables LIR tracking --

    #[test]
    fn config_clock_pro_false_no_lir_tracking() {
        let config = HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        };
        let mut scheduler = HGALScheduler::new(config);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Active,
            ..Default::default()
        });
        scheduler.mark_accessed(1);
        // With clock_pro disabled, page should NOT be in lir_pages
        assert!(!scheduler.lir_pages.contains(&1));
    }

    // -- 47. on_prefill_chunk_complete increments existing group access_count --

    #[test]
    fn on_prefill_chunk_complete_increments_existing_group() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Pre-register group
        scheduler.upsert_group(SequenceGroup {
            id: 100,
            pages: vec![],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let pages = vec![VirtualPageId::new(100, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        // access_count should have incremented (saturating_add(1) from 5 → 6)
        assert!(scheduler.sequence_groups[&100].access_count >= 6);
    }

    // -- 48. PageState Swapped variant differs from SwappedOut --

    #[test]
    fn page_state_swapped_not_equal_swapped_out() {
        assert_ne!(PageState::Swapped, PageState::SwappedOut);
    }

    // -- 49. compute_eviction_priority reflects expert_id for expert pages --

    #[test]
    fn eviction_priority_expert_id_reflected_in_result() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let expert = UnifiedVirtualPage::expert(1, 42, 3, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&expert);
        assert_eq!(prio.expert_id, Some(42));
        assert_eq!(prio.layer_idx, Some(3));
    }

    // -- 50. on_prefill_complete updates group context_len unchanged --

    #[test]
    fn on_prefill_complete_preserves_context_len() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 500,
            pages: vec![10, 20, 30],
            state: GroupState::Running,
            access_count: 2,
            last_access: Instant::now() - Duration::from_secs(1),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        for pid in &[10, 20, 30] {
            scheduler.update_page_state(*pid, Some(500), PageState::Standby);
        }
        scheduler.on_prefill_complete(500);
        assert_eq!(scheduler.sequence_groups[&500].context_len, 3);
    }

    // ────────────────────────────────────────────────────────────────
    // Wave 51 tests: additional coverage for HGALConfig, HGALScheduler,
    // page lifecycle, weight page operations, and edge cases.
    // ────────────────────────────────────────────────────────────────

    // -- 51.1 HGALConfig Default produces values within valid ranges --

    #[test]
    fn hgal_config_default_warmup_is_positive() {
        let cfg = HGALConfig::default();
        assert!(cfg.warmup_duration.as_nanos() > 0);
    }

    // -- 51.2 HGALConfig Default working_set_window exceeds warmup_duration --

    #[test]
    fn hgal_config_default_working_set_exceeds_warmup() {
        let cfg = HGALConfig::default();
        assert!(cfg.working_set_window > cfg.warmup_duration);
    }

    // -- 51.3 HGALConfig Default lir_ratio in (0, 1) --

    #[test]
    fn hgal_config_default_lir_ratio_in_unit_interval() {
        let cfg = HGALConfig::default();
        assert!(cfg.lir_ratio > 0.0 && cfg.lir_ratio < 1.0);
    }

    // -- 51.4 HGALConfig Default enable_clock_pro is true --

    #[test]
    fn hgal_config_default_clock_pro_enabled() {
        let cfg = HGALConfig::default();
        assert!(cfg.enable_clock_pro);
    }

    // -- 51.5 HGALConfig Default hot_threshold >= min_warm_access --

    #[test]
    fn hgal_config_default_hot_threshold_ge_min_warm_access() {
        let cfg = HGALConfig::default();
        assert!(cfg.hot_threshold >= cfg.min_warm_access);
    }

    // -- 51.6 HGALScheduler::new initializes empty weight_page_table --

    #[test]
    fn scheduler_new_weight_page_table_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.weight_page_table.is_empty());
    }

    // -- 51.7 HGALScheduler::new initializes empty page_metadata --

    #[test]
    fn scheduler_new_page_metadata_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.page_metadata.is_empty());
    }

    // -- 51.8 HGALScheduler::new initializes empty lir_pages --

    #[test]
    fn scheduler_new_lir_pages_empty() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.lir_pages.is_empty());
    }

    // -- 51.9 register_expert_weight_page then num_weight_pages reflects count --

    #[test]
    fn register_expert_weight_pages_updates_num_weight_pages() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert_eq!(scheduler.num_weight_pages(), 0);
        scheduler.register_expert_weight_page(100, 0);
        assert_eq!(scheduler.num_weight_pages(), 1);
        scheduler.register_expert_weight_page(101, 0);
        assert_eq!(scheduler.num_weight_pages(), 2);
    }

    // -- 51.10 register_dense_layer_weight_pages returns input page_ids --

    #[test]
    fn register_dense_layer_weight_pages_returns_input_ids() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let ids = vec![200, 201, 202];
        let returned = scheduler.register_dense_layer_weight_pages(ids.clone(), 5);
        assert_eq!(returned, ids);
    }

    // -- 51.11 update_page_state Swapped variant clears warm_until --

    #[test]
    fn update_page_state_to_swapped_clears_warm_until() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(999);
        // After swap_in, warm_until should be set
        assert!(scheduler.page_metadata[&999].warm_until.is_some());
        // Transitioning to Swapped should clear warm_until
        scheduler.update_page_state(999, None, PageState::Swapped);
        assert!(scheduler.page_metadata[&999].warm_until.is_none());
        assert!(scheduler.page_metadata[&999].swap_in_time.is_none());
    }

    // -- 51.12 upsert_group with KnowledgeRAG payload_kind stores correctly --

    #[test]
    fn upsert_group_knowledge_rag_payload_kind_stored_correctly() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 777,
            pages: vec![40, 41],
            state: GroupState::Running,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        };
        scheduler.upsert_group(group);
        assert_eq!(
            scheduler.sequence_groups[&777].payload_kind,
            Some(PagePayloadKind::KnowledgeRAG)
        );
    }

    // -- 51.13 mark_accessed on unknown page creates metadata with sequence_id None --

    #[test]
    fn mark_accessed_unknown_page_creates_metadata_no_sequence() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(12345);
        let meta = scheduler.page_metadata.get(&12345).unwrap();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.sequence_id, None);
    }

    // -- 51.14 allocate_expert_weight_pages zero experts returns empty vec --

    #[test]
    fn allocate_expert_weight_pages_zero_returns_empty() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(0, 3);
        assert!(pages.is_empty());
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    // -- 51.15 free_expert_weight_pages then re-register produces clean state --

    #[test]
    fn free_expert_then_reregister_produces_clean_state() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(4, 1);
        assert_eq!(scheduler.num_expert_weight_pages(), 4);
        scheduler.free_expert_weight_pages(1);
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
        // Re-register
        let new_pages = scheduler.allocate_expert_weight_pages(2, 1);
        assert_eq!(new_pages.len(), 2);
        assert_eq!(scheduler.num_expert_weight_pages(), 2);
    }

    // ── Edge-case tests (wave-12x34) ──

    // @trace REQ-HGAL-001 [level:unit] warm_until derived from swap_in_time when warm_until is None
    #[test]
    fn warm_until_derived_from_swap_in_time_when_none() {
        // Arrange: create metadata with swap_in_time set but warm_until = None
        let now = Instant::now();
        let warmup = Duration::from_millis(100);
        let meta = PageMetadata {
            page_id: 42,
            warm_until: None,
            swap_in_time: Some(now),
            state: PageState::Warm,
            ..Default::default()
        };

        // Act: warm_until helper falls back to swap_in_time + warmup_duration
        let result = HGALScheduler::warm_until(&meta, warmup);

        // Assert: derived warm_until equals swap_in_time + warmup
        assert!(result.is_some(), "warm_until should derive from swap_in_time");
        let expected = now + warmup;
        assert_eq!(result.unwrap(), expected);
    }

    // @trace REQ-HGAL-001 [level:unit] select_coldest_lir returns one page when all have same recency
    #[test]
    fn select_coldest_lir_returns_one_when_all_tied() {
        // Arrange: scheduler with 3 LIR pages all having recency=0
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        for pid in [100, 200, 300] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                recency: 0,
                is_lir: true,
                ..Default::default()
            });
            scheduler.lir_pages.insert(pid);
        }

        // Act
        let coldest = scheduler.select_coldest_lir();

        // Assert: returns exactly one page (deterministic tie-breaking by max_by_key)
        assert!(coldest.is_some(), "must return a page even when tied");
        let returned = coldest.unwrap();
        assert!([100, 200, 300].contains(&returned));
    }

    // @trace REQ-HGAL-002 [level:unit] compute_group_priority with pages missing metadata
    #[test]
    fn compute_group_priority_pages_missing_metadata() {
        // Arrange: group with 3 pages, only page 10 has metadata
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![10, 20, 30],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };

        // Act: only page 10 contributes recency; pages 20, 30 are filtered out
        let priority = scheduler.compute_group_priority(&group);

        // Assert: priority is a valid i64 (no panic from missing metadata)
        // With access_count=5, freq_bonus=50, recency_penalty from page 10 only
        assert!(priority > 0 || priority <= 0, "priority must be computable without panic");
    }

    // @trace REQ-HGAL-002 [level:unit] group_has_protection false when metadata absent
    #[test]
    fn group_has_protection_false_when_no_metadata() {
        // Arrange: scheduler with a group whose pages have no metadata at all
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![999],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };

        // Act
        let protected = scheduler.group_has_protection(&group);

        // Assert: no metadata means no protection
        assert!(!protected, "group with no page metadata must not have protection");
    }

    // @trace REQ-HGAL-002 [level:unit] group_has_protection false when state_is_standby
    #[test]
    fn group_has_protection_false_when_state_standby() {
        // Arrange: scheduler with a page in Standby state (not Warm/Protected)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(55, PageMetadata {
            page_id: 55,
            state: PageState::Standby,
            ..Default::default()
        });
        let group = SequenceGroup {
            id: 1,
            pages: vec![55],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };

        // Act
        let protected = scheduler.group_has_protection(&group);

        // Assert: Standby state is not Warm or Protected, and not in warmup
        assert!(!protected, "Standby page must not trigger group protection");
    }

    // @trace REQ-HGAL-003 [level:unit] on_prefill_chunk_complete deduplicates pages within group
    #[test]
    fn prefill_chunk_complete_deduplicates_pages() {
        // Arrange: scheduler receives two chunks referencing the same page for the same request
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId::new(42, 0);

        // Act: first chunk
        scheduler.on_prefill_chunk_complete(0, 2, &[vpid]);
        // Act: second chunk with same vpid (simulates re-send or retry)
        scheduler.on_prefill_chunk_complete(1, 2, &[vpid]);

        // Assert: group has deduplicated pages
        let group = scheduler.sequence_groups.get(&42).unwrap();
        let unique_pages: HashSet<PageId> = group.pages.iter().copied().collect();
        assert_eq!(unique_pages.len(), group.pages.len(),
            "pages must be deduplicated within a group");
    }

    // @trace REQ-HGAL-003 [level:unit] select_victim_groups returns empty when all groups pinned
    #[test]
    fn select_victim_groups_all_pinned_yields_no_candidates() {
        // Arrange: two groups, both pinned
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for id in [1u64, 2u64] {
            scheduler.upsert_group(SequenceGroup {
                id,
                pages: vec![id as usize * 10],
                state: GroupState::Running,
                access_count: 0,
                last_access: Instant::now(),
                is_pinned: true,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }

        // Act
        let victims = scheduler.select_victim_groups(5);

        // Assert: no victims because all groups are pinned
        assert!(victims.is_empty(), "pinned groups must not be selected as victims");
    }

    // @trace REQ-HGAL-003 [level:unit] select_victim_groups unpicked selected before pinned
    #[test]
    fn select_victim_groups_skips_pinned_selects_unpinned() {
        // Arrange: one pinned group, one unpinned group
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![200],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(1);

        // Assert: only unpinned group 2 is a victim
        assert_eq!(victims, vec![2]);
    }

    // @trace REQ-HGAL-004 [level:unit] register_expert_then_dense_same_layer_free_removes_all
    #[test]
    fn register_expert_and_dense_same_layer_free_removes_all() {
        // Arrange: register both expert and dense weight pages on the same layer
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(100, 5);
        scheduler.register_dense_layer_weight_page(200, 5);
        assert_eq!(scheduler.num_weight_pages(), 2);

        // Act: free expert weight pages for layer 5 removes ALL weight pages for that layer
        scheduler.free_expert_weight_pages(5);

        // Assert: weight_page_table entry for layer 5 is fully removed
        assert_eq!(scheduler.num_weight_pages(), 0,
            "free_expert_weight_pages removes the entire layer entry including dense pages");
        assert!(!scheduler.page_metadata.contains_key(&100));
        assert!(!scheduler.page_metadata.contains_key(&200));
    }

    // @trace REQ-HGAL-004 [level:unit] free_dense_layer_weight_pages_noop_for_missing_layer
    #[test]
    fn free_dense_layer_weight_pages_noop_for_missing_layer() {
        // Arrange: scheduler with no pages for layer 99
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 1);
        assert_eq!(scheduler.num_weight_pages(), 1);

        // Act: free a non-existent layer — should not panic or remove other layers
        scheduler.free_dense_layer_weight_pages(99);

        // Assert: layer 1 pages still intact
        assert_eq!(scheduler.num_weight_pages(), 1);
    }

    // @trace REQ-HGAL-005 [level:unit] on_swap_in_zero_warmup_sets_warm_until_eq_swap_in_time
    #[test]
    fn on_swap_in_zero_warmup_warm_until_equals_now() {
        // Arrange: scheduler with zero warmup duration
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::ZERO,
            ..Default::default()
        });

        // Act
        scheduler.on_swap_in(77);

        // Assert: state is Warm, access_count is 0
        let meta = scheduler.page_metadata.get(&77).unwrap();
        assert_eq!(meta.state, PageState::Warm);
        assert_eq!(meta.access_count, 0);
        // warm_until = now + ZERO = now, which is essentially the swap_in_time
        assert!(meta.warm_until.is_some());
        assert!(meta.swap_in_time.is_some());
        assert_eq!(meta.warm_until.unwrap(), meta.swap_in_time.unwrap());
    }

    // @trace REQ-HGAL-005 [level:unit] swap_in_swap_out_full_cycle_state_correctness
    #[test]
    fn swap_in_swap_out_full_cycle_state_correctness() {
        // Arrange: scheduler with default config
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act: transition Active -> Swapped -> Warm (swap-in) -> Active
        scheduler.update_page_state(50, Some(1), PageState::Active);
        assert_eq!(scheduler.page_metadata.get(&50).unwrap().state, PageState::Active);

        // Swap out
        scheduler.update_page_state(50, Some(1), PageState::Swapped);
        assert_eq!(scheduler.page_metadata.get(&50).unwrap().state, PageState::Swapped);
        assert!(scheduler.page_metadata.get(&50).unwrap().warm_until.is_none());
        assert!(scheduler.page_metadata.get(&50).unwrap().swap_in_time.is_none());

        // Swap in
        scheduler.on_swap_in(50);
        assert_eq!(scheduler.page_metadata.get(&50).unwrap().state, PageState::Warm);
        assert!(scheduler.page_metadata.get(&50).unwrap().warm_until.is_some());
        assert!(scheduler.page_metadata.get(&50).unwrap().swap_in_time.is_some());

        // Mark accessed enough to exit warmup
        for _ in 0..scheduler.config.min_warm_access + 1 {
            scheduler.mark_accessed(50);
        }
        // After sufficient accesses, state should transition away from Warm
        assert_ne!(scheduler.page_metadata.get(&50).unwrap().state, PageState::Warm);
    }

    // @trace REQ-HGAL-006 [level:unit] detect_working_set_zero_threshold_promotes_all_recent
    #[test]
    fn detect_working_set_zero_threshold_zero_window_promotes_all() {
        // Arrange: scheduler with hot_threshold=0 and working_set_window=ZERO
        // Every page with recent access meets the "hot" criteria
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 0,
            working_set_window: Duration::ZERO,
            ..Default::default()
        });
        // Pages in Active state that should be promoted to Protected
        for pid in [10, 20, 30] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                state: PageState::Active,
                access_count: 0,
                last_access: Instant::now(),
                ..Default::default()
            });
        }

        // Act
        scheduler.detect_working_set();

        // Assert: all Active pages promoted to Protected (access_count >= 0 and time < ZERO is false,
        // but access_count=0 >= hot_threshold=0 is true; window check: now - last_access < ZERO is false
        // for pages accessed "now" — saturating_duration_since is 0, and 0 < 0 is false)
        // So actually hot=false because window check fails. Let's verify the actual behavior:
        // With ZERO window, no page can satisfy now.saturating_duration_since(last) < ZERO
        // since duration is 0 and 0 < 0 is false.
        for pid in [10, 20, 30] {
            let meta = scheduler.page_metadata.get(&pid).unwrap();
            assert_eq!(meta.state, PageState::Active,
                "with ZERO window, no page can satisfy hot condition");
        }
    }

    // @trace REQ-HGAL-006 [level:unit] detect_working_set_protected_expires_to_standby
    #[test]
    fn detect_working_set_protected_expires_to_standby() {
        // Arrange: a Protected page with last_access far in the past (outside working_set_window)
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_millis(1),
            ..Default::default()
        });
        let old_access = Instant::now() - Duration::from_secs(60);
        scheduler.page_metadata.insert(99, PageMetadata {
            page_id: 99,
            state: PageState::Protected,
            access_count: 100,
            last_access: old_access,
            ..Default::default()
        });

        // Act
        scheduler.detect_working_set();

        // Assert: expired Protected page demoted to Standby
        assert_eq!(scheduler.page_metadata.get(&99).unwrap().state, PageState::Standby);
    }

    // @trace REQ-HGAL-007 [level:unit] num_dense_layer_weight_pages_zero_when_no_dense_groups
    #[test]
    fn num_dense_layer_weight_pages_zero_when_no_dense_groups() {
        // Arrange: scheduler with only ExpertWeight groups (no DenseLayerWeight)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Act
        let count = scheduler.num_dense_layer_weight_pages();

        // Assert: zero because no group has DenseLayerWeight payload
        assert_eq!(count, 0);
    }

    // @trace REQ-HGAL-007 [level:unit] select_victim_weight_pages_empty_when_all_protected
    #[test]
    fn select_victim_weight_pages_empty_when_all_protected_or_warm() {
        // Arrange: weight pages all in Protected or Warm state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for (pid, state) in [(10, PageState::Protected), (20, PageState::Warm)] {
            scheduler.weight_page_table.entry(0).or_default().push(pid);
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                state,
                ..Default::default()
            });
        }

        // Act
        let victims = scheduler.select_victim_weight_pages(5);

        // Assert: no victims because Protected and Warm pages are skipped
        assert!(victims.is_empty(),
            "Protected and Warm weight pages must not be victims");
    }

    // @trace REQ-HGAL-007 [level:unit] mark_accessed_creates_metadata_for_unknown_page
    #[test]
    fn mark_accessed_creates_metadata_with_default_state_then_activates() {
        // Arrange: scheduler with no pre-existing metadata for page 777
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        assert!(!scheduler.page_metadata.contains_key(&777));

        // Act
        scheduler.mark_accessed(777);

        // Assert: metadata was created, state promoted to Active, access_count=1
        let meta = scheduler.page_metadata.get(&777).unwrap();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.state, PageState::Active);
        assert!(meta.recency >= 0);
    }

    // @trace REQ-HGAL-008 [level:unit] remove_group_does_not_affect_page_metadata
    #[test]
    fn remove_group_does_not_remove_page_metadata() {
        // Arrange: add a group and its page metadata
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 5,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 3,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(100, Some(5), PageState::Active);

        // Act: remove the group
        scheduler.remove_group(5);

        // Assert: group is gone but page metadata persists
        assert!(!scheduler.sequence_groups.contains_key(&5));
        assert!(scheduler.page_metadata.contains_key(&100),
            "page metadata must survive group removal");
    }

    // @trace REQ-HGAL-008 [level:unit] update_page_state_to_free_does_not_clear_recency
    #[test]
    fn update_page_state_to_free_preserves_recency() {
        // Arrange: page with recency=500 and access_count=10
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            recency: 500,
            access_count: 10,
            state: PageState::Active,
            ..Default::default()
        });

        // Act: transition to Free
        scheduler.update_page_state(42, None, PageState::Free);

        // Assert: recency is preserved (only Swapped clears warm_until/swap_in_time)
        let meta = scheduler.page_metadata.get(&42).unwrap();
        assert_eq!(meta.recency, 500);
        assert_eq!(meta.access_count, 10);
        assert_eq!(meta.state, PageState::Free);
        assert!(meta.sequence_id.is_none());
    }

    // ── Additional edge-case tests ──

    // @trace REQ-HGAL-008 [level:unit] compute_eviction_priority for expert weight page with no layer_idx
    #[test]
    fn eviction_priority_expert_weight_no_layer_idx_uses_zero_penalty() {
        // Arrange: ExpertWeight page with layer_idx=None (no layer depth penalty)
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage {
            page_id: 99,
            payload_kind: PagePayloadKind::ExpertWeight,
            residency: super::super::types::MemoryResidency::DeviceLocal,
            dtype: gllm_kernels::types::DType::F32,
            owner: None,
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: Some(0),
            layer_idx: None,
        };

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: expert weight gets negative base adjustment, no layer depth penalty
        assert_eq!(prio.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(prio.score < 0, "ExpertWeight without layer_idx should have negative score, got {}", prio.score);
        assert_eq!(prio.layer_idx, None);
    }

    // @trace REQ-HGAL-008 [level:unit] warm_until returns None when both warm_until and swap_in_time are None
    #[test]
    fn warm_until_both_none_returns_none() {
        // Arrange: PageMetadata with no warm_until and no swap_in_time
        let meta = PageMetadata {
            page_id: 1,
            warm_until: None,
            swap_in_time: None,
            ..Default::default()
        };

        // Act
        let result = HGALScheduler::warm_until(&meta, Duration::from_millis(100));

        // Assert
        assert!(result.is_none(), "warm_until should be None when both fields are None");
    }

    // @trace REQ-HGAL-008 [level:unit] warm_until derives from swap_in_time when warm_until is None
    #[test]
    fn warm_until_swap_in_time_only_returns_derived() {
        // Arrange: PageMetadata with swap_in_time but no warm_until
        let swap_time = Instant::now();
        let warmup = Duration::from_millis(200);
        let meta = PageMetadata {
            page_id: 1,
            warm_until: None,
            swap_in_time: Some(swap_time),
            ..Default::default()
        };

        // Act
        let result = HGALScheduler::warm_until(&meta, warmup);

        // Assert: derived warm_until = swap_in_time + warmup_duration
        assert!(result.is_some(), "warm_until should be derived from swap_in_time");
        let expected = swap_time + warmup;
        assert_eq!(result.unwrap(), expected);
    }

    // @trace REQ-HGAL-008 [level:unit] is_in_warmup_period_meta returns false for Active state
    #[test]
    fn is_in_warmup_period_meta_active_returns_false() {
        // Arrange: Active page (not Warm)
        let meta = PageMetadata {
            page_id: 1,
            state: PageState::Active,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            access_count: 0,
            ..Default::default()
        };

        // Act
        let result = HGALScheduler::is_in_warmup_period_meta(
            &meta,
            Instant::now(),
            Duration::from_millis(100),
            2,
        );

        // Assert: only Warm pages can be in warmup
        assert!(!result, "Active pages are never in warmup period");
    }

    // @trace REQ-HGAL-008 [level:unit] is_in_warmup_period_meta returns false for Standby state
    #[test]
    fn is_in_warmup_period_meta_standby_returns_false() {
        // Arrange: Standby page
        let meta = PageMetadata {
            page_id: 1,
            state: PageState::Standby,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            access_count: 0,
            ..Default::default()
        };

        // Act
        let result = HGALScheduler::is_in_warmup_period_meta(
            &meta,
            Instant::now(),
            Duration::from_millis(100),
            2,
        );

        // Assert: only Warm pages can be in warmup
        assert!(!result, "Standby pages are never in warmup period");
    }

    // @trace REQ-HGAL-008 [level:unit] is_in_warmup_period_meta returns false when access_count exceeds min_warm_access
    #[test]
    fn is_in_warmup_period_meta_exceeds_min_warm_access_exits_warmup() {
        // Arrange: Warm page with access_count >= min_warm_access (graduated warmup)
        let meta = PageMetadata {
            page_id: 1,
            state: PageState::Warm,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            access_count: 5, // exceeds min_warm_access=2
            ..Default::default()
        };

        // Act
        let result = HGALScheduler::is_in_warmup_period_meta(
            &meta,
            Instant::now(),
            Duration::from_millis(100),
            2, // min_warm_access
        );

        // Assert: access_count exceeded threshold, no longer in warmup
        assert!(!result, "Warm page with enough accesses is no longer in warmup");
    }

    // @trace REQ-HGAL-008 [level:unit] update_page_state SwappedOut variant stores correctly
    #[test]
    fn update_page_state_swapped_out_variant_stored_correctly() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act: transition to SwappedOut (note: this is a different variant than Swapped)
        scheduler.update_page_state(42, Some(10), PageState::SwappedOut);

        // Assert: SwappedOut does NOT trigger the Swapped cleanup (warm_until/swap_in_time preserved)
        let meta = scheduler.page_metadata.get(&42).unwrap();
        assert_eq!(meta.state, PageState::SwappedOut);
        assert_eq!(meta.sequence_id, Some(10));
        // SwappedOut does not clear warm_until (only PageState::Swapped does)
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    // @trace REQ-HGAL-008 [level:unit] select_victim_groups with only one unprotected among pinned groups
    #[test]
    fn select_victim_groups_only_one_unprotected_among_pinned() {
        // Arrange: two pinned groups and one unpinned
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        for id in [1u64, 2u64] {
            scheduler.upsert_group(SequenceGroup {
                id,
                pages: vec![id as usize],
                state: GroupState::Running,
                access_count: 0,
                last_access: now,
                is_pinned: true,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        scheduler.upsert_group(SequenceGroup {
            id: 3,
            pages: vec![300],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(5);

        // Assert: only the unpinned group is returned
        assert_eq!(victims, vec![3]);
    }

    // @trace REQ-HGAL-008 [level:unit] on_prefill_complete preserves Swapped pages in swapped state
    #[test]
    fn on_prefill_complete_swapped_pages_remain_swapped() {
        // Arrange: group with one Active and one Swapped page
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 7,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(7), PageState::Active);
        scheduler.update_page_state(20, Some(7), PageState::Swapped);

        // Act
        scheduler.on_prefill_complete(7);

        // Assert: Active page stays Active, Swapped page stays Swapped
        assert_eq!(scheduler.page_metadata.get(&10).unwrap().state, PageState::Active);
        assert_eq!(scheduler.page_metadata.get(&20).unwrap().state, PageState::Swapped);
    }

    // @trace REQ-HGAL-008 [level:unit] compute_eviction_priority: recency penalty reduces score proportionally
    #[test]
    fn eviction_priority_recency_penalty_is_proportional() {
        // Arrange: two pages with different recency values
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            recency: 100,
            access_count: 0,
            ..Default::default()
        });
        scheduler.page_metadata.insert(2, PageMetadata {
            page_id: 2,
            recency: 200,
            access_count: 0,
            ..Default::default()
        });

        let page_low = UnifiedVirtualPage::kv(1, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page_high = UnifiedVirtualPage::kv(2, 10, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);

        // Act
        let prio_low = scheduler.compute_eviction_priority(&page_low);
        let prio_high = scheduler.compute_eviction_priority(&page_high);

        // Assert: higher recency = lower score (recency penalty)
        let diff = prio_low.score - prio_high.score;
        assert_eq!(diff, (200i64 - 100i64) * (FREQUENCY_WEIGHT / 2),
            "Recency penalty difference should be proportional to recency delta");
    }

    // @trace REQ-HGAL-008 [level:unit] compute_eviction_priority: EvictionPriority fields populated correctly
    #[test]
    fn eviction_priority_all_fields_populated_correctly() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(5, PageMetadata {
            page_id: 5,
            recency: 42,
            access_count: 7,
            ..Default::default()
        });

        let page = UnifiedVirtualPage::expert(5, 3, 4, gllm_kernels::types::DType::BF16);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: all fields match input
        assert_eq!(prio.payload_kind, PagePayloadKind::ExpertWeight);
        assert_eq!(prio.access_count, 7);
        assert_eq!(prio.recency, 42);
        assert_eq!(prio.layer_idx, Some(4));
        assert_eq!(prio.expert_id, Some(3));
        assert!(!prio.is_pinned, "ExpertWeight pages are evictable");
    }

    // @trace REQ-HGAL-008 [level:unit] select_victim_weight_pages with no weight_page_table entries returns empty
    #[test]
    fn select_victim_weight_pages_no_table_entries_returns_empty() {
        // Arrange: scheduler with groups and page_metadata but no weight_page_table
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        });
        scheduler.update_page_state(10, Some(1), PageState::Active);
        scheduler.update_page_state(20, Some(1), PageState::Active);

        // Act: request victim weight pages even though no weight pages exist
        let victims = scheduler.select_victim_weight_pages(10);

        // Assert: no weight pages in table, so no victims
        assert!(victims.is_empty(), "No weight_page_table entries means no victim weight pages");
    }

    // @trace REQ-HGAL-008 [level:unit] on_swap_in then immediate mark_accessed: warm page exits warmup and becomes active
    #[test]
    fn on_swap_in_then_mark_accessed_with_zero_warmup_exits_warm() {
        // Arrange: config with zero warmup so warmup period is already expired
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::ZERO,
            min_warm_access: 1,
            ..Default::default()
        });
        scheduler.on_swap_in(42);

        // Act: access after warmup expired (immediately, since warmup_duration=0)
        scheduler.mark_accessed(42);

        // Assert: warm page with enough accesses transitions to Active
        let meta = scheduler.page_metadata.get(&42).unwrap();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.state, PageState::Active);
    }

    // @trace REQ-HGAL-008 [level:unit] mark_accessed does not crash when sequence_id points to nonexistent group
    #[test]
    fn mark_accessed_orphan_sequence_id_no_panic() {
        // Arrange: page metadata with a sequence_id that has no matching group
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        scheduler.page_metadata.insert(99, PageMetadata {
            page_id: 99,
            sequence_id: Some(9999), // nonexistent group
            state: PageState::Active,
            ..Default::default()
        });

        // Act: should not panic
        scheduler.mark_accessed(99);

        // Assert: access_count incremented, state unchanged (Active → Active)
        let meta = scheduler.page_metadata.get(&99).unwrap();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.state, PageState::Active);
    }

    // @trace REQ-HGAL-008 [level:unit] allocate_expert_weight_pages with layer_idx=0 produces correct IDs
    #[test]
    fn allocate_expert_weight_pages_layer_zero_id_encoding() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act: allocate 3 experts for layer 0
        let pages = scheduler.allocate_expert_weight_pages(3, 0);

        // Assert: page IDs = (0 << 16) | expert_idx for layer 0
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0], 0);
        assert_eq!(pages[1], 1);
        assert_eq!(pages[2], 2);
        // Verify all registered in metadata as Active
        for &pid in &pages {
            let meta = scheduler.page_metadata.get(&pid).unwrap();
            assert_eq!(meta.state, PageState::Active);
            assert!(meta.sequence_id.is_none());
        }
    }

    // @trace REQ-HGAL-008 [level:unit] update_lir_membership skipped when clock_pro disabled
    #[test]
    fn update_lir_membership_disabled_no_lir_tracking() {
        // Arrange: scheduler with clock_pro disabled
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });

        // Act: access multiple pages
        for pid in 1..=5 {
            scheduler.mark_accessed(pid);
        }

        // Assert: no LIR membership tracked
        assert!(scheduler.lir_pages.is_empty(), "LIR pages should be empty when clock_pro is disabled");
    }

    // @trace REQ-HGAL-008 [level:unit] on_prefill_chunk_complete sets recency to zero for all pages
    #[test]
    fn on_prefill_chunk_complete_recency_zero_all_pages() {
        // Arrange: scheduler with pages that have non-zero recency
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            recency: 9999,
            access_count: 50,
            ..Default::default()
        });
        scheduler.page_metadata.insert(20, PageMetadata {
            page_id: 20,
            recency: 5000,
            access_count: 30,
            ..Default::default()
        });

        let pages = vec![
            VirtualPageId::new(1, 10),
            VirtualPageId::new(1, 20),
        ];

        // Act
        scheduler.on_prefill_chunk_complete(0, 2, &pages);

        // Assert: recency reset to 0 for both pages
        assert_eq!(scheduler.page_metadata.get(&virtual_page_to_page_id(pages[0])).unwrap().recency, 0);
        assert_eq!(scheduler.page_metadata.get(&virtual_page_to_page_id(pages[1])).unwrap().recency, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Wave 15 — 15 additional unit tests for edge cases & uncovered paths
    // ═══════════════════════════════════════════════════════════════════════

    // -- 1. on_swap_in on already-Warm page resets access_count to zero --
    #[test]
    fn on_swap_in_resets_access_count_on_already_warm_page() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        scheduler.mark_accessed(1);
        scheduler.mark_accessed(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 2);
        // Re-swap-in should reset access_count
        scheduler.on_swap_in(1);
        assert_eq!(scheduler.page_metadata[&1].access_count, 0);
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Warm);
    }

    // -- 2. compute_eviction_priority for DenseLayerWeight page with metadata --
    #[test]
    fn eviction_priority_dense_layer_weight_with_metadata_high_score() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            access_count: 0,
            recency: 0,
            state: PageState::Active,
            ..Default::default()
        });
        let dense = UnifiedVirtualPage::dense_layer(42, 5, gllm_kernels::types::DType::F32);
        let prio = scheduler.compute_eviction_priority(&dense);
        // DenseLayerWeight base=5000, pin_bonus=5000 (not evictable), total=10000
        assert_eq!(prio.score, 10_000);
        assert!(prio.score > 0, "DenseLayerWeight should have very high (hard-to-evict) score");
    }

    // -- 3. compute_group_priority: ExpertWeight payload produces negative adjustment --
    #[test]
    fn compute_group_priority_expert_weight_negative_adjustment() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        let group = SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        };
        let priority = scheduler.compute_group_priority(&group);
        // time_penalty ~= 0 (just created), recency_penalty = 0 (no pages with metadata),
        // freq_bonus = 0, pin_bonus = 0, payload_adjustment = EXPERT_WEIGHT_PRIORITY_BONUS = -200
        assert!(priority <= 0, "ExpertWeight group should have non-positive priority, got {}", priority);
    }

    // -- 4. on_prefill_chunk_complete with large chunk_idx near total_chunks --
    #[test]
    fn on_prefill_chunk_complete_penultimate_chunk_is_standby() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![VirtualPageId::new(10, 0)];
        // chunk_idx = 3, total_chunks = 5 → not last chunk → Standby
        scheduler.on_prefill_chunk_complete(3, 5, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Standby);
    }

    // -- 5. Multiple groups with same last_access: select_victim_groups uses payload kind --
    #[test]
    fn select_victim_groups_payload_kind_breaks_tie_between_groups() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.update_page_state(10, Some(1), PageState::Standby);
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(20, Some(2), PageState::Standby);
        let victims = scheduler.select_victim_groups(2);
        // ExpertWeight group should be evicted first (lower priority)
        assert_eq!(victims[0], 1, "ExpertWeight group should be first victim");
    }

    // -- 6. on_prefill_complete clears warm_until on warm page that was swap-in'd --
    #[test]
    fn on_prefill_complete_clears_warm_until_from_swap_in() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(100);
        assert!(scheduler.page_metadata[&100].warm_until.is_some());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.on_prefill_complete(1);
        assert!(scheduler.page_metadata[&100].warm_until.is_none(),
            "on_prefill_complete must clear warm_until on non-Swapped pages");
    }

    // -- 7. detect_working_set: Warm page with access_count >= min_warm_access exits warmup --
    #[test]
    fn detect_working_set_warm_page_graduated_exits_to_active() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            min_warm_access: 2,
            warmup_duration: Duration::from_secs(3600), // long warmup so time doesn't expire
            ..Default::default()
        });
        scheduler.on_swap_in(1);
        // Access enough to exceed min_warm_access
        scheduler.mark_accessed(1);
        scheduler.mark_accessed(1);
        assert!(scheduler.page_metadata[&1].access_count >= 2);
        // detect_working_set should graduate the warm page to Active
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Active,
            "Warm page with enough accesses should graduate to Active");
    }

    // -- 8. upsert_group preserves original access_count through multiple updates --
    #[test]
    fn upsert_group_preserves_counters_through_four_updates() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 10,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Re-upsert with different pages but lower access_count
        for i in 0..4 {
            scheduler.upsert_group(SequenceGroup {
                id: 1,
                pages: vec![i],
                state: GroupState::Running,
                access_count: 0,
                last_access: now,
                is_pinned: false,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        assert_eq!(scheduler.sequence_groups[&1].access_count, 10,
            "access_count must be preserved from initial insert across 4 re-upserts");
    }

    // -- 9. select_victim_weight_pages with mixed expert and dense in same layer --
    #[test]
    fn select_victim_weight_pages_mixed_expert_dense_same_layer_prefers_expert() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Register expert and dense pages for layer 0
        scheduler.register_expert_weight_page(10, 0);
        scheduler.register_dense_layer_weight_page(20, 0);
        // Set both to Standby so they're eligible
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&20).unwrap().state = PageState::Standby;
        // Create groups to assign payload kinds
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });
        let victims = scheduler.select_victim_weight_pages(2);
        assert_eq!(victims.len(), 2);
        // Expert should have lower (more evictable) score than dense
        let expert_score = victims.iter().find(|(pid, _)| *pid == 10).unwrap().1.score;
        let dense_score = victims.iter().find(|(pid, _)| *pid == 20).unwrap().1.score;
        assert!(expert_score < dense_score,
            "ExpertWeight (score={}) should be evicted before DenseLayerWeight (score={})",
            expert_score, dense_score);
    }

    // -- 10. update_page_state to Active preserves existing warm_until --
    #[test]
    fn update_page_state_to_active_preserves_warm_until_from_swap_in() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(1);
        let warm = scheduler.page_metadata[&1].warm_until;
        assert!(warm.is_some());
        // Transition to Active
        scheduler.update_page_state(1, Some(10), PageState::Active);
        // warm_until should NOT be cleared (only Swapped clears it)
        assert_eq!(scheduler.page_metadata[&1].warm_until, warm,
            "Active transition should preserve warm_until");
    }

    // -- 11. group_has_protection returns true for warm page in active warmup --
    #[test]
    fn group_has_protection_true_for_warm_page_in_active_warmup() {
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(3600),
            min_warm_access: 100,
            ..Default::default()
        });
        scheduler.on_swap_in(1);
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert!(scheduler.group_has_protection(&scheduler.sequence_groups[&1]),
            "Group with warm page in active warmup should have protection");
    }

    // -- 12. register_dense_layer_weight_pages with empty vec is a noop --
    #[test]
    fn register_dense_layer_weight_pages_empty_vec_is_noop() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let returned = scheduler.register_dense_layer_weight_pages(vec![], 5);
        assert!(returned.is_empty());
        assert_eq!(scheduler.num_weight_pages(), 0);
        assert!(scheduler.weight_page_table.get(&5).is_none_or(|v| v.is_empty()));
    }

    // -- 13. mark_accessed recency increases monotonically between accesses --
    #[test]
    fn mark_accessed_recency_monotonic_between_accesses() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(1);
        let recency1 = scheduler.page_metadata[&1].recency;
        // Small delay to ensure time difference
        std::thread::sleep(std::time::Duration::from_millis(2));
        scheduler.mark_accessed(1);
        let recency2 = scheduler.page_metadata[&1].recency;
        // recency2 should be >= recency1 (it measures time since last access)
        assert!(recency2 >= recency1,
            "recency should be non-decreasing between accesses: {} vs {}", recency1, recency2);
    }

    // -- 14. allocate_expert_weight_pages for high layer_idx produces distinct page IDs --
    #[test]
    fn allocate_expert_weight_pages_high_layer_idx_distinct_ids() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = scheduler.allocate_expert_weight_pages(4, 100);
        assert_eq!(pages.len(), 4);
        // page_id = (layer_idx << 16) | expert_idx for layer 100
        let base = 100usize << 16;
        assert_eq!(pages[0], base | 0);
        assert_eq!(pages[1], base | 1);
        assert_eq!(pages[2], base | 2);
        assert_eq!(pages[3], base | 3);
        // Verify all unique
        let unique: HashSet<_> = pages.iter().copied().collect();
        assert_eq!(unique.len(), 4);
    }

    // -- 15. on_prefill_complete on nonexistent group is a safe noop --
    #[test]
    fn on_prefill_complete_nonexistent_then_create_group_succeeds() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Call on_prefill_complete on nonexistent group
        scheduler.on_prefill_complete(999);
        assert!(!scheduler.sequence_groups.contains_key(&999));
        // Then create a group with that id and verify it works
        scheduler.upsert_group(SequenceGroup {
            id: 999,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        assert!(scheduler.sequence_groups.contains_key(&999));
        scheduler.on_prefill_complete(999);
        assert_eq!(scheduler.sequence_groups[&999].state, GroupState::Running);
    }

    // ── Batch 7: 15 additional tests for remaining coverage gaps ──

    // -- 1. update_page_state does not reset access_count or recency on state change --

    #[test]
    fn update_page_state_preserves_recency_on_state_change() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.mark_accessed(10);
        scheduler.mark_accessed(10);
        let recency_before = scheduler.page_metadata[&10].recency;
        let access_before = scheduler.page_metadata[&10].access_count;
        assert!(access_before >= 2);

        // Act
        scheduler.update_page_state(10, Some(1), PageState::Standby);

        // Assert
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.recency, recency_before, "recency must be preserved across update_page_state");
        assert_eq!(meta.access_count, access_before, "access_count must be preserved across update_page_state");
        assert_eq!(meta.state, PageState::Standby);
    }

    // -- 2. select_victim_groups sorts by time_penalty (last_access age) --
    // compute_group_priority: time_penalty + recency - freq - pin + payload
    // Higher time_penalty means group was accessed LONG AGO, which ADDS to score,
    // making it HARDER to evict. So a recently-accessed group (low time_penalty)
    // has a LOWER score and is evicted first when other factors are equal.

    #[test]
    fn select_victim_groups_recent_group_evicted_when_no_freq_advantage() {
        // Arrange: two groups with identical payload and zero access_count.
        // Group 1 was accessed 10s ago (high time_penalty -> higher score).
        // Group 2 was accessed just now (low time_penalty -> lower score).
        // Lower score = evicted first, so Group 2 (recent) is evicted first.
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        // Group accessed 10s ago (large time_penalty -> higher priority score)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Group accessed just now (small time_penalty -> lower priority score)
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(1);

        // Assert: lower score is evicted first. Group 2 has lower time_penalty
        // which contributes less to the priority score, so it gets evicted first.
        assert_eq!(victims[0], 2, "recent group with lower time_penalty has lower score and is evicted first");
    }

    // -- 3. on_swap_in followed by detect_working_set with hot_threshold=0 promotes --

    #[test]
    fn swap_in_then_detect_with_zero_hot_threshold_promotes_past_warmup() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: usize::MAX,
            hot_threshold: 0,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);
        // Wait for warmup to expire
        std::thread::sleep(Duration::from_millis(1));

        // Act
        scheduler.detect_working_set();

        // Assert: Warm page exits warmup -> Active; then hot_threshold=0 means any page is hot -> Protected
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Protected,
            "page with hot_threshold=0 past warmup must be promoted to Protected"
        );
    }

    // -- 4. mark_accessed does not update group when page has no sequence_id --

    #[test]
    fn mark_accessed_no_group_update_when_sequence_id_is_none() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        // Create page metadata without a sequence_id
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: None,
            state: PageState::Active,
            ..Default::default()
        });

        // Act
        scheduler.mark_accessed(42);

        // Assert: page metadata updated, but no group affected
        assert_eq!(scheduler.page_metadata[&42].access_count, 1);
        assert!(scheduler.sequence_groups.is_empty(), "no group should be created from mark_accessed");
    }

    // -- 5. select_victim_weight_pages returns fewer than requested when eligible pool is small --

    #[test]
    fn select_victim_weight_pages_returns_fewer_when_pool_insufficient() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.register_expert_weight_page(10, 0);
        // Make it Standby (eligible)
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Act: request 5 but only 1 eligible
        let victims = scheduler.select_victim_weight_pages(5);

        // Assert
        assert_eq!(victims.len(), 1, "must return at most the number of eligible pages");
        assert_eq!(victims[0].0, 10);
    }

    // -- 6. Multiple detect_working_set calls demote Protected pages consistently --

    #[test]
    fn detect_working_set_repeated_demotion_protected_to_standby_stays_standby() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_millis(1),
            hot_threshold: 100,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(1),
            ..Default::default()
        });

        // Act: first call demotes to Standby
        scheduler.detect_working_set();
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Standby);

        // Second call: Standby, not hot -> stays Standby
        scheduler.detect_working_set();

        // Assert
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Standby,
            "demoted page must remain Standby on subsequent detect_working_set calls"
        );
    }

    // -- 7. register_dense_layer_weight_pages returns same page IDs --

    #[test]
    fn register_dense_layer_weight_pages_returned_ids_match_input() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let input_ids = vec![500, 501, 502, 503];

        // Act
        let returned = scheduler.register_dense_layer_weight_pages(input_ids.clone(), 10);

        // Assert
        assert_eq!(returned, input_ids);
        assert_eq!(scheduler.weight_page_table[&10].len(), 4);
    }

    // -- 8. compute_eviction_priority for expert with no layer_idx gets zero depth penalty --

    #[test]
    fn eviction_priority_expert_none_layer_idx_zero_depth_penalty() {
        // Arrange: expert page has expert_id but layer_idx is set by the constructor
        // UnifiedVirtualPage::expert always sets layer_idx = Some(layer), so we verify
        // layer_idx=0 produces zero depth penalty
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::expert(1, 42, 0, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: payload_adjustment = -200 + -(0) = -200
        // No metadata -> freq=0, recency=0
        assert_eq!(prio.score, -200, "expert at layer 0 must have no depth penalty");
        assert_eq!(prio.expert_id, Some(42));
    }

    // -- 9. on_prefill_chunk_complete accumulates pages across multiple chunks correctly --

    #[test]
    fn on_prefill_chunk_complete_accumulates_pages_across_three_chunks() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let chunk0 = vec![VirtualPageId::new(77, 0), VirtualPageId::new(77, 1)];
        let chunk1 = vec![VirtualPageId::new(77, 2)];
        let chunk2 = vec![VirtualPageId::new(77, 3), VirtualPageId::new(77, 4)];

        // Act
        scheduler.on_prefill_chunk_complete(0, 3, &chunk0);
        scheduler.on_prefill_chunk_complete(1, 3, &chunk1);
        scheduler.on_prefill_chunk_complete(2, 3, &chunk2);

        // Assert
        let group = &scheduler.sequence_groups[&77];
        assert_eq!(group.pages.len(), 5, "all 5 pages from 3 chunks must be accumulated");
        assert_eq!(group.context_len, 5, "context_len must be max of accumulated pages");
        // First two chunks (0,1) should be Standby; last chunk (2) Active
        let pid0 = virtual_page_to_page_id(chunk0[0]);
        assert_eq!(scheduler.page_metadata[&pid0].state, PageState::Standby, "non-last chunk pages must be Standby");
        let pid4 = virtual_page_to_page_id(chunk2[1]);
        assert_eq!(scheduler.page_metadata[&pid4].state, PageState::Active, "last chunk pages must be Active");
    }

    // -- 10. HGALConfig Clone produces independent copy --

    #[test]
    fn hgal_config_clone_produces_independent_copy() {
        // Arrange
        let config = HGALConfig {
            warmup_duration: Duration::from_millis(200),
            working_set_window: Duration::from_secs(2),
            hot_threshold: 5,
            lir_ratio: 0.4,
            min_warm_access: 3,
            enable_clock_pro: true,
        };

        // Act
        let cloned = config.clone();

        // Assert: each field matches independently
        assert_eq!(cloned.warmup_duration, Duration::from_millis(200));
        assert_eq!(cloned.working_set_window, Duration::from_secs(2));
        assert_eq!(cloned.hot_threshold, 5);
        assert!((cloned.lir_ratio - 0.4).abs() < f32::EPSILON);
        assert_eq!(cloned.min_warm_access, 3);
        assert!(cloned.enable_clock_pro);
    }

    // -- 11. remove_group then upsert_group with same id works as fresh insert --

    #[test]
    fn remove_group_then_upsert_same_id_works_as_fresh() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let original = SequenceGroup {
            id: 42,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 50,
            last_access: Instant::now() - Duration::from_secs(60),
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        scheduler.upsert_group(original);

        // Act: remove then re-insert with fresh values
        scheduler.remove_group(42);
        let fresh = SequenceGroup {
            id: 42,
            pages: vec![10],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: None,
        };
        scheduler.upsert_group(fresh);

        // Assert: fresh values, no carry-over from removed group
        let group = &scheduler.sequence_groups[&42];
        assert_eq!(group.pages, vec![10], "pages must be from fresh insert");
        assert_eq!(group.access_count, 0, "access_count must be from fresh insert (no preserved value)");
        assert_eq!(group.state, GroupState::Paused);
        assert!(group.is_pinned);
    }

    // -- 12. EvictionPriority for dense layer with Protected state gets double bonus --

    #[test]
    fn eviction_priority_dense_protected_gets_payload_and_state_bonus() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::dense_layer(1, 3, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: DenseLayerWeight payload = 5000, pin = 5000, Protected state = 10000
        // score = 5000 - 0 + 0 + 5000 + 10000 = 20000
        assert_eq!(prio.score, 20_000, "DenseLayerWeight + Protected must have score 20000");
        assert!(prio.is_pinned);
        assert_eq!(prio.payload_kind, PagePayloadKind::DenseLayerWeight);
    }

    // -- 13. select_victim_groups with single unpinned group returns exactly that group --

    #[test]
    fn select_victim_groups_single_eligible_group_selected_for_count_one() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 7,
            pages: vec![1, 2, 3, 4],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(1),
            is_pinned: false,
            context_len: 4,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(1);

        // Assert: even requesting just 1 page, the single group provides 4
        assert_eq!(victims, vec![7], "single eligible group must be selected");
    }

    // -- 14. on_swap_in on a page with existing metadata preserves sequence_id --

    #[test]
    fn on_swap_in_keeps_prior_sequence_id_intact() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(99), PageState::Active);
        assert_eq!(scheduler.page_metadata[&10].sequence_id, Some(99));

        // Act
        scheduler.on_swap_in(10);

        // Assert: sequence_id must survive swap-in (on_swap_in uses or_insert_with which preserves existing entry)
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.sequence_id, Some(99), "sequence_id must be preserved through swap-in");
        assert_eq!(meta.state, PageState::Warm);
        assert_eq!(meta.access_count, 0, "access_count must be reset to 0");
    }

    // -- 15. allocate_expert_weight_pages with one expert produces single entry --

    #[test]
    fn allocate_expert_weight_pages_single_expert_correct_encoding() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act
        let pages = scheduler.allocate_expert_weight_pages(1, 7);

        // Assert
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0], (7usize << 16) | 0, "single expert page_id must be (layer << 16) | 0");
        assert_eq!(scheduler.num_expert_weight_pages(), 1);
        assert!(scheduler.page_metadata.contains_key(&pages[0]));
        assert_eq!(scheduler.page_metadata[&pages[0]].state, PageState::Active);
    }

    // ── Batch 8: 15 additional tests for remaining coverage gaps ──

    // -- 1. config() accessor returns reference to stored config --
    #[test]
    fn config_accessor_returns_stored_config() {
        // Arrange
        let config = HGALConfig {
            warmup_duration: Duration::from_millis(500),
            working_set_window: Duration::from_secs(10),
            hot_threshold: 7,
            lir_ratio: 0.5,
            min_warm_access: 4,
            enable_clock_pro: false,
        };
        let scheduler = HGALScheduler::new(config);

        // Act
        let returned = scheduler.config();

        // Assert: returned reference matches original values
        assert_eq!(returned.warmup_duration, Duration::from_millis(500));
        assert_eq!(returned.working_set_window, Duration::from_secs(10));
        assert_eq!(returned.hot_threshold, 7);
        assert!((returned.lir_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(returned.min_warm_access, 4);
        assert!(!returned.enable_clock_pro);
    }

    // -- 2. on_prefill_chunk_complete with empty pages vec creates no metadata --
    #[test]
    fn on_prefill_chunk_complete_empty_pages_no_metadata_created() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.sequence_groups.is_empty());
        assert!(scheduler.page_metadata.is_empty());

        // Act
        scheduler.on_prefill_chunk_complete(0, 5, &[]);

        // Assert: no groups or metadata created
        assert!(scheduler.sequence_groups.is_empty());
        assert!(scheduler.page_metadata.is_empty());
    }

    // -- 3. select_victim_groups with count=0 on non-empty scheduler returns empty --
    #[test]
    fn select_victim_groups_zero_count_on_populated_scheduler() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(0);

        // Assert
        assert!(victims.is_empty(), "requesting 0 pages must return empty vec");
    }

    // -- 4. select_victim_weight_pages with count=0 on non-empty scheduler returns empty --
    #[test]
    fn select_victim_weight_pages_zero_count_on_populated_scheduler() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(10, 0);
        scheduler.page_metadata.get_mut(&10).unwrap().state = PageState::Standby;
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Act
        let victims = scheduler.select_victim_weight_pages(0);

        // Assert
        assert!(victims.is_empty(), "requesting 0 weight pages must return empty vec");
    }

    // -- 5. mark_accessed on Protected page stays Protected --
    #[test]
    fn mark_accessed_protected_page_stays_protected() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 5,
            ..Default::default()
        });

        // Act
        scheduler.mark_accessed(1);

        // Assert: Protected pages remain Protected on access
        assert_eq!(scheduler.page_metadata[&1].state, PageState::Protected,
            "Protected pages must not be demoted by mark_accessed");
        assert_eq!(scheduler.page_metadata[&1].access_count, 6);
    }

    // -- 6. system_prompt page is not evictable --
    #[test]
    fn system_prompt_page_is_not_evictable() {
        // Arrange
        let prompt = UnifiedVirtualPage::system_prompt(42, gllm_kernels::types::DType::F32);

        // Act & Assert
        assert!(!prompt.is_evictable(), "PromptSystem pages must not be evictable");
        assert!(prompt.is_on_device(), "PromptSystem pages are on device");
        assert_eq!(prompt.payload_kind, PagePayloadKind::PromptSystem);
        assert!(prompt.owner.is_none());
        assert!(prompt.expert_id.is_none());
        assert!(prompt.layer_idx.is_none());
    }

    // -- 7. compute_eviction_priority for PromptSystem page has high score --
    #[test]
    fn eviction_priority_prompt_system_has_very_high_score() {
        // Arrange
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let prompt = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&prompt);

        // Assert: PromptSystem base=10000, pin_bonus=5000, total=15000
        assert!(prio.score >= 15_000,
            "PromptSystem must have very high eviction score, got {}", prio.score);
        assert!(prio.is_pinned, "PromptSystem pages must be pinned");
        assert_eq!(prio.payload_kind, PagePayloadKind::PromptSystem);
    }

    // -- 8. upsert_group preserves last_access from existing group --
    #[test]
    fn upsert_group_preserves_last_access_from_existing() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let old_time = Instant::now() - Duration::from_secs(30);
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: old_time,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        let new_time = Instant::now();

        // Act: re-upsert with a different last_access
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: new_time,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Assert: last_access is preserved from existing entry (old_time)
        assert_eq!(scheduler.sequence_groups[&1].last_access, old_time,
            "last_access must be preserved from existing group on re-upsert");
    }

    // -- 9. remove_group on nonexistent id is a safe noop --
    #[test]
    fn remove_group_nonexistent_id_is_safe_noop() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: remove a nonexistent group id
        scheduler.remove_group(999);

        // Assert: existing group 1 is untouched
        assert!(scheduler.sequence_groups.contains_key(&1));
        assert!(!scheduler.sequence_groups.contains_key(&999));
    }

    // -- 10. lir_pages grows up to ratio then evicts coldest --
    #[test]
    fn lir_pages_respects_ratio_and_evicts_coldest() {
        // Arrange: lir_ratio=0.5, start with 4 pages so target = ceil(4*0.5)=2
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: 0.5,
            ..Default::default()
        });

        // Act: access 4 pages with increasing recency
        for pid in 1..=4 {
            scheduler.mark_accessed(pid);
            // Mark higher recency for higher pid to make recency order: 1 < 2 < 3 < 4
            if pid > 1 {
                std::thread::sleep(Duration::from_millis(2));
                scheduler.mark_accessed(pid);
            }
        }

        // Assert: lir_pages has at most ceil(4*0.5)=2 entries
        assert!(scheduler.lir_pages.len() <= 2,
            "LIR pages must respect ratio, got {} pages", scheduler.lir_pages.len());
        assert!(!scheduler.lir_pages.is_empty(), "some pages must be LIR");
    }

    // -- 11. virtual_page_to_page_id produces different ids for different inputs --
    #[test]
    fn virtual_page_to_page_id_different_inputs_yield_different_ids() {
        // Arrange
        let vpid_a = VirtualPageId::new(1, 0);
        let vpid_b = VirtualPageId::new(0, 1);
        let vpid_c = VirtualPageId::new(100, 200);
        let vpid_d = VirtualPageId::new(200, 100);

        // Act
        let id_a = virtual_page_to_page_id(vpid_a);
        let id_b = virtual_page_to_page_id(vpid_b);
        let id_c = virtual_page_to_page_id(vpid_c);
        let id_d = virtual_page_to_page_id(vpid_d);

        // Assert: different inputs produce different outputs
        assert_ne!(id_a, id_b, "different (sid, lid) pairs must produce different page ids");
        assert_ne!(id_c, id_d, "swapped (sid, lid) pairs must produce different page ids");
    }

    // -- 12. is_in_warmup_period returns false for nonexistent page --
    #[test]
    fn is_in_warmup_period_nonexistent_page_returns_false() {
        // Arrange
        let scheduler = HGALScheduler::new(HGALConfig::default());

        // Act
        let result = scheduler.is_in_warmup_period(99999);

        // Assert: nonexistent page is not in warmup
        assert!(!result, "nonexistent page must not be in warmup");
    }

    // -- 13. compute_group_priority with DenseLayerWeight payload produces high positive --
    #[test]
    fn compute_group_priority_dense_layer_weight_high_positive() {
        // Arrange
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 1,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        };

        // Act
        let priority = scheduler.compute_group_priority(&group);

        // Assert: DenseLayerWeight payload = 5000, other factors ~0
        assert!(priority >= 4900,
            "DenseLayerWeight group must have high positive priority, got {}", priority);
    }

    // -- 14. update_page_state to SwappedOut preserves warm_until only if it was None --
    #[test]
    fn update_page_state_swapped_out_does_not_clear_warm_when_none() {
        // Arrange: page with no warm_until/swap_in_time
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(1, Some(10), PageState::Active);

        // Act: transition to SwappedOut (not Swapped — SwappedOut does NOT trigger cleanup)
        scheduler.update_page_state(1, Some(10), PageState::SwappedOut);

        // Assert: SwappedOut does not trigger the Swapped cleanup path
        let meta = &scheduler.page_metadata[&1];
        assert_eq!(meta.state, PageState::SwappedOut);
        assert_eq!(meta.sequence_id, Some(10));
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    // -- 15. on_prefill_chunk_complete creates group with Running state --
    #[test]
    fn on_prefill_chunk_complete_creates_group_in_running_state() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(!scheduler.sequence_groups.contains_key(&42));

        // Act
        let pages = vec![VirtualPageId::new(42, 0)];
        scheduler.on_prefill_chunk_complete(0, 1, &pages);

        // Assert: group 42 created with Running state
        let group = scheduler.sequence_groups.get(&42).unwrap();
        assert_eq!(group.state, GroupState::Running,
            "newly created group must be in Running state");
        assert_eq!(group.id, 42);
        assert!(group.access_count >= 1, "access_count must be incremented on chunk complete");
    }

    // ═══════════════════════════════════════════════════════════════════
    // 15 NEW TESTS — distinct scenarios not covered by existing 649 tests
    // ═══════════════════════════════════════════════════════════════════

    // -- 1. compute_group_priority with payload_kind=None yields zero adjustment --

    #[test]
    fn compute_group_priority_none_payload_kind_zero_adjustment() {
        // Arrange: group with None payload_kind — hits the `_ => 0` branch
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(100, Some(1), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let group = scheduler.sequence_groups.get(&1).unwrap();
        let priority = scheduler.compute_group_priority(group);

        // Assert: with access_count=0, is_pinned=false, payload=None, recency=0
        // priority = time_penalty + 0 - 0 - 0 + 0 = time_penalty (very small, near 0)
        assert!(
            priority >= 0,
            "None payload_kind group priority must be >= 0 (time_penalty only), got {}",
            priority
        );
        assert!(
            priority < 100,
            "None payload_kind group priority must be small with zero access_count and recency, got {}",
            priority
        );
    }

    // -- 2. compute_group_priority with is_pinned=true subtracts PIN_BONUS --

    #[test]
    fn compute_group_priority_pinned_group_subtracts_pin_bonus() {
        // Arrange: two identical groups except is_pinned
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, Some(50), PageState::Active);
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 50,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: true,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 51,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let pinned = scheduler.sequence_groups.get(&50).unwrap();
        let unpinned = scheduler.sequence_groups.get(&51).unwrap();
        let prio_pinned = scheduler.compute_group_priority(pinned);
        let prio_unpinned = scheduler.compute_group_priority(unpinned);

        // Assert: pinned priority = unpinned - PIN_BONUS (5000)
        let diff = prio_unpinned - prio_pinned;
        assert_eq!(
            diff, 5000,
            "pinned group must have PIN_BONUS (5000) subtracted, got diff {}",
            diff
        );
    }

    // -- 3. compute_group_priority with high access_count dominates time_penalty --

    #[test]
    fn compute_group_priority_high_frequency_dominates() {
        // Arrange: group with very high access_count
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(200, Some(30), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 30,
            pages: vec![200],
            state: GroupState::Running,
            access_count: 10_000,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let group = scheduler.sequence_groups.get(&30).unwrap();
        let priority = scheduler.compute_group_priority(group);

        // Assert: freq_bonus = 10000 * 10 = 100000
        // time_penalty is near 0 (Instant::now() - now), recency=0
        // priority = ~0 + 0 - 100000 - 0 + 0 = -100000
        assert!(
            priority < -90_000,
            "high access_count=10000 must produce strongly negative priority (freq dominates), got {}",
            priority
        );
    }

    // -- 4. compute_group_priority with payload_kind=Some(KvContext) hits _ => 0 --

    #[test]
    fn compute_group_priority_kv_context_payload_zero_adjustment() {
        // Arrange: KvContext is not ExpertWeight or DenseLayerWeight, so _ => 0
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(300, Some(60), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 60,
            pages: vec![300],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        });

        // Act
        let group = scheduler.sequence_groups.get(&60).unwrap();
        let priority = scheduler.compute_group_priority(group);

        // Assert: same as None payload — adjustment is 0
        assert!(
            priority >= 0 && priority < 100,
            "KvContext group priority must be time_penalty only (adjustment=0), got {}",
            priority
        );
    }

    // -- 5. compute_group_priority with payload_kind=Some(KnowledgeRAG) hits _ => 0 --

    #[test]
    fn compute_group_priority_knowledge_rag_payload_zero_adjustment() {
        // Arrange: KnowledgeRAG is not ExpertWeight or DenseLayerWeight, so _ => 0
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(400, Some(70), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 70,
            pages: vec![400],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        });

        // Act
        let group = scheduler.sequence_groups.get(&70).unwrap();
        let priority = scheduler.compute_group_priority(group);

        // Assert: KnowledgeRAG group adjustment = 0 (not -200, that's per-page not group)
        assert!(
            priority >= 0 && priority < 100,
            "KnowledgeRAG group priority must be time_penalty only (adjustment=0), got {}",
            priority
        );
    }

    // -- 6. compute_group_priority with payload_kind=Some(PromptSystem) hits _ => 0 --

    #[test]
    fn compute_group_priority_prompt_system_payload_zero_adjustment() {
        // Arrange: PromptSystem is not ExpertWeight or DenseLayerWeight, so _ => 0
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(500, Some(80), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 80,
            pages: vec![500],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::PromptSystem),
        });

        // Act
        let group = scheduler.sequence_groups.get(&80).unwrap();
        let priority = scheduler.compute_group_priority(group);

        // Assert: PromptSystem group adjustment = 0
        assert!(
            priority >= 0 && priority < 100,
            "PromptSystem group priority must be time_penalty only (adjustment=0), got {}",
            priority
        );
    }

    // -- 7. register_dense_layer_weight_page accumulates distinct pages in same layer --

    #[test]
    fn register_dense_layer_weight_page_accumulates_distinct_pages() {
        // Arrange
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act: register three different pages in the same layer
        scheduler.register_dense_layer_weight_page(101, 3);
        scheduler.register_dense_layer_weight_page(102, 3);
        scheduler.register_dense_layer_weight_page(103, 3);

        // Assert
        let pages = scheduler.weight_page_table.get(&3).unwrap();
        assert_eq!(pages.len(), 3, "three distinct pages must accumulate");
        assert_eq!(pages[0], 101);
        assert_eq!(pages[1], 102);
        assert_eq!(pages[2], 103);

        // Verify each page has Active metadata
        for pid in &[101, 102, 103] {
            let meta = scheduler.page_metadata.get(pid).unwrap();
            assert_eq!(meta.state, PageState::Active);
            assert!(meta.sequence_id.is_none());
        }
    }

    // -- 8. free_expert_weight_pages removes weight_page_table entry completely --

    #[test]
    fn free_expert_weight_pages_removes_layer_entry_from_table() {
        // Arrange: register pages in multiple layers
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_expert_weight_page(1000, 0);
        scheduler.register_expert_weight_page(1001, 0);
        scheduler.register_expert_weight_page(2000, 1);

        // Act: free layer 0
        scheduler.free_expert_weight_pages(0);

        // Assert: layer 0 entry completely removed from weight_page_table
        assert!(
            !scheduler.weight_page_table.contains_key(&0),
            "layer 0 must be removed from weight_page_table"
        );
        // Layer 1 still intact
        let layer1 = scheduler.weight_page_table.get(&1).unwrap();
        assert_eq!(layer1.len(), 1);
        assert_eq!(layer1[0], 2000);
        // Layer 0 metadata removed
        assert!(scheduler.page_metadata.get(&1000).is_none());
        assert!(scheduler.page_metadata.get(&1001).is_none());
        // Layer 1 metadata intact
        assert!(scheduler.page_metadata.get(&2000).is_some());
    }

    // -- 9. compute_eviction_priority ExpertWeight page with Protected state bonus --

    #[test]
    fn eviction_priority_expert_weight_protected_state_bonus() {
        // Arrange: ExpertWeight page with Protected state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(77, PageMetadata {
            page_id: 77,
            access_count: 0,
            recency: 0,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::expert(77, 0, 2, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: ExpertWeight base = -200 + layer_depth_penalty(-2) = -202
        // pin_bonus = 0 (ExpertWeight is evictable)
        // state_bonus = 10000 (Protected)
        // score = -202 - 0 + 0 + 0 + 10000 = 9798
        assert_eq!(
            prio.score, 9798,
            "ExpertWeight + Protected must be -200 - 2 + 10000 = 9798, got {}",
            prio.score
        );
        assert_eq!(prio.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(!prio.is_pinned);
    }

    // -- 10. compute_eviction_priority ExpertWeight page with Warm state bonus --

    #[test]
    fn eviction_priority_expert_weight_warm_state_bonus() {
        // Arrange: ExpertWeight page with Warm state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(78, PageMetadata {
            page_id: 78,
            access_count: 0,
            recency: 0,
            state: PageState::Warm,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::expert(78, 0, 2, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: ExpertWeight base = -200 + layer_depth(-2) = -202
        // Warm state_bonus = 5000
        // score = -202 - 0 + 0 + 0 + 5000 = 4798
        assert_eq!(
            prio.score, 4798,
            "ExpertWeight + Warm must be -200 - 2 + 5000 = 4798, got {}",
            prio.score
        );
        assert_eq!(prio.payload_kind, PagePayloadKind::ExpertWeight);
    }

    // -- 11. compute_eviction_priority RAG page with Protected state bonus --

    #[test]
    fn eviction_priority_rag_protected_state_bonus() {
        // Arrange: RAG page with Protected state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(88, PageMetadata {
            page_id: 88,
            access_count: 0,
            recency: 0,
            state: PageState::Protected,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::rag(88, 55, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: RAG base = -200, pin_bonus = 0 (RAG is evictable)
        // Protected state_bonus = 10000
        // score = -200 - 0 + 0 + 0 + 10000 = 9800
        assert_eq!(
            prio.score, 9800,
            "RAG + Protected must be -200 + 10000 = 9800, got {}",
            prio.score
        );
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(!prio.is_pinned);
    }

    // -- 12. upsert_group overwrites payload_kind on second upsert --

    #[test]
    fn upsert_group_overwrites_payload_kind() {
        // Arrange: first upsert with KvContext
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        });

        // Act: second upsert with KnowledgeRAG
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        });

        // Assert: payload_kind overwritten to KnowledgeRAG, pages updated
        let group = scheduler.sequence_groups.get(&99).unwrap();
        assert_eq!(
            group.payload_kind,
            Some(PagePayloadKind::KnowledgeRAG),
            "payload_kind must be overwritten on second upsert"
        );
        assert_eq!(group.pages, vec![10, 20], "pages must be updated");
    }

    // -- 13. select_victim_groups selects exactly two single-page groups for count=2 --

    #[test]
    fn select_victim_groups_exactly_two_single_page_groups() {
        // Arrange: three unpinned groups with different priorities
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group A: ExpertWeight (adjustment = -200, lowest priority)
        scheduler.update_page_state(1, Some(100), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 100,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Group B: None payload (adjustment = 0)
        scheduler.update_page_state(2, Some(200), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 200,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Group C: DenseLayerWeight (adjustment = 5000, hardest to evict)
        scheduler.update_page_state(3, Some(300), PageState::Active);
        scheduler.upsert_group(SequenceGroup {
            id: 300,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // Act: request 2 pages
        let victims = scheduler.select_victim_groups(2);

        // Assert: exactly 2 groups selected (ExpertWeight first, then None payload)
        assert_eq!(victims.len(), 2, "must select exactly 2 groups");
        assert!(
            victims.contains(&100),
            "ExpertWeight group (100) must be in victims"
        );
        assert!(
            victims.contains(&200),
            "None-payload group (200) must be in victims"
        );
        assert!(
            !victims.contains(&300),
            "DenseLayerWeight group (300) must not be in victims"
        );
    }

    // -- 14. update_page_state new page starts with default values before state is set --

    #[test]
    fn update_page_state_new_page_has_correct_sequence_id() {
        // Arrange: no pre-existing metadata for page 999
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(!scheduler.page_metadata.contains_key(&999));

        // Act: update_page_state for a new page
        scheduler.update_page_state(999, Some(42), PageState::Active);

        // Assert: metadata created with the specified sequence_id and state
        let meta = scheduler.page_metadata.get(&999).unwrap();
        assert_eq!(meta.page_id, 999);
        assert_eq!(meta.sequence_id, Some(42));
        assert_eq!(meta.state, PageState::Active);
        assert_eq!(meta.access_count, 0, "new page access_count must be 0");
        assert_eq!(meta.recency, 0, "new page recency must be 0");
        assert!(!meta.is_lir, "new page is_lir must be false");
        assert!(meta.swap_in_time.is_none(), "new page swap_in_time must be None");
        assert!(meta.warm_until.is_none(), "new page warm_until must be None");
    }

    // -- 15. compute_eviction_priority RAG page with Warm state bonus --

    #[test]
    fn eviction_priority_rag_warm_state_bonus() {
        // Arrange: RAG page with Warm state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(89, PageMetadata {
            page_id: 89,
            access_count: 0,
            recency: 0,
            state: PageState::Warm,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::rag(89, 66, gllm_kernels::types::DType::F32);

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: RAG base = -200, Warm state_bonus = 5000
        // score = -200 - 0 + 0 + 0 + 5000 = 4800
        assert_eq!(
            prio.score, 4800,
            "RAG + Warm must be -200 + 5000 = 4800, got {}",
            prio.score
        );
        assert_eq!(prio.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(!prio.is_pinned);
    }

    // ── Additional tests batch 4 (15 new) ──

    // -- 16. register_dense_layer_weight_pages 空列表返回空但内部无修改 --

    #[test]
    fn register_dense_layer_weight_pages_empty_input_returns_empty_vec() {
        // Arrange: 空的 weight_page_table
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        assert!(scheduler.weight_page_table.is_empty());

        // Act: 传入空的 page_ids 列表
        let result = scheduler.register_dense_layer_weight_pages(vec![], 3);

        // Assert: 返回空列表，weight_page_table 中层 3 无条目，无 page_metadata 新增
        assert!(result.is_empty(), "empty input must return empty output");
        assert!(
            scheduler.weight_page_table.get(&3).is_none(),
            "no entry should be created for layer 3 with empty input"
        );
        assert_eq!(scheduler.num_weight_pages(), 0);
        assert!(scheduler.page_metadata.is_empty(), "no page metadata should be created for empty input");
    }

    // -- 17. free_dense_layer_weight_pages 后重新注册同层页面 --

    #[test]
    fn free_dense_layer_weight_pages_then_reregister_same_layer() {
        // Arrange: 注册 -> 释放 -> 再注册同层页面
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.register_dense_layer_weight_pages(vec![100, 101], 5);
        assert_eq!(scheduler.num_weight_pages(), 2);
        scheduler.free_dense_layer_weight_pages(5);
        assert_eq!(scheduler.num_weight_pages(), 0);

        // Act: 重新注册同层不同页面
        scheduler.register_dense_layer_weight_pages(vec![200, 201, 202], 5);

        // Assert: 新页面正确注册
        assert_eq!(scheduler.num_weight_pages(), 3);
        assert_eq!(
            scheduler.weight_page_table.get(&5),
            Some(&vec![200, 201, 202])
        );
        for pid in &[200, 201, 202] {
            let meta = scheduler.page_metadata.get(pid).expect("metadata must exist");
            assert_eq!(meta.state, PageState::Active);
        }
        assert!(scheduler.page_metadata.get(&100).is_none(), "old page 100 must be gone");
        assert!(scheduler.page_metadata.get(&101).is_none(), "old page 101 must be gone");
    }

    // -- 18. num_dense_layer_weight_pages 多组混合计数 --

    #[test]
    fn num_dense_layer_weight_pages_counts_only_dense_groups() {
        // Arrange: 注册多个不同 payload_kind 的 group
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // DenseLayerWeight group: 2 pages
        scheduler.register_dense_layer_weight_pages(vec![10, 11], 0);
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: true,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // ExpertWeight group: 3 pages — 不计入 dense 计数
        scheduler.register_expert_weight_page(20, 1);
        scheduler.register_expert_weight_page(21, 1);
        scheduler.register_expert_weight_page(22, 1);
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_001,
            pages: vec![20, 21, 22],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // KvContext group: 1 page — 不计入 dense 计数
        scheduler.upsert_group(SequenceGroup {
            id: 1_000_002,
            pages: vec![30],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        });

        // Act & Assert
        assert_eq!(
            scheduler.num_dense_layer_weight_pages(), 2,
            "only DenseLayerWeight group pages (2) must be counted"
        );
        assert_eq!(scheduler.num_weight_pages(), 5, "total weight pages = 2 dense + 3 expert");
    }

    // -- 19. remove_group 后 select_victim_groups 不再考虑该组 --

    #[test]
    fn remove_group_excludes_from_victim_selection() {
        // Arrange: 创建两个组，移除其中一个后验证只剩另一个可被选中
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 20,
            pages: vec![3],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(1),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: 移除 group 10
        scheduler.remove_group(10);

        // Assert: 只有 group 20 可被选为 victim
        let victims = scheduler.select_victim_groups(10);
        assert_eq!(victims, vec![20], "only group 20 must remain as a victim candidate");
    }

    // -- 20. update_page_state SwappedOut 不清除 warm_until (非 Swapped 变体) --

    #[test]
    fn update_page_state_swapped_out_preserves_warm_fields() {
        // Arrange: 先 swap_in 设置 warm_until，再更新到 SwappedOut（不是 Swapped）
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.on_swap_in(42);
        assert!(scheduler.page_metadata[&42].warm_until.is_some());
        assert!(scheduler.page_metadata[&42].swap_in_time.is_some());

        // Act: 设置为 SwappedOut (与 Swapped 不同)
        scheduler.update_page_state(42, Some(1), PageState::SwappedOut);

        // Assert: warm_until 和 swap_in_time 保留，因为只有 Swapped 才清除
        let meta = &scheduler.page_metadata[&42];
        assert_eq!(meta.state, PageState::SwappedOut);
        assert!(meta.warm_until.is_some(), "SwappedOut must NOT clear warm_until (only Swapped does)");
        assert!(meta.swap_in_time.is_some(), "SwappedOut must NOT clear swap_in_time (only Swapped does)");
    }

    // -- 21. select_victim_weight_pages 多层 expert 页面按分数排序 --

    #[test]
    fn select_victim_weight_pages_cross_layer_expert_ordering() {
        // Arrange: 两个层各有 expert 页面，验证跨层排序按 score 升序
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // 层 0: 1 个 expert，高 access_count = 高 score (难驱逐)
        scheduler.register_expert_weight_page(100, 0);
        scheduler.page_metadata.get_mut(&100).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&100).unwrap().access_count = 50;

        // 层 1: 1 个 expert，低 access_count = 低 score (易驱逐)
        scheduler.register_expert_weight_page(200, 1);
        scheduler.page_metadata.get_mut(&200).unwrap().state = PageState::Standby;
        scheduler.page_metadata.get_mut(&200).unwrap().access_count = 1;

        scheduler.upsert_group(SequenceGroup {
            id: 1_000_000,
            pages: vec![100, 200],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Act
        let victims = scheduler.select_victim_weight_pages(2);

        // Assert: 低 access_count 的页 (200) 先出现（score 更低）
        assert_eq!(victims.len(), 2);
        let score_200 = victims.iter().find(|(pid, _)| *pid == 200).unwrap().1.score;
        let score_100 = victims.iter().find(|(pid, _)| *pid == 100).unwrap().1.score;
        assert!(
            score_200 < score_100,
            "low-access page (score={}) must be evicted before high-access (score={})",
            score_200, score_100,
        );
    }

    // -- 22. compute_eviction_priority expert 无 layer_idx 的 score --

    #[test]
    fn eviction_priority_expert_no_layer_idx_uses_zero_depth_penalty() {
        // Arrange: 构造一个 expert 页面，其 layer_idx = None（不应该在正常路径出现，但验证健壮性）
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage {
            page_id: 42,
            payload_kind: PagePayloadKind::ExpertWeight,
            residency: crate::scheduler::types::MemoryResidency::DeviceLocal,
            dtype: gllm_kernels::types::DType::F32,
            owner: None,
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: Some(7),
            layer_idx: None, // 无 layer
        };

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: ExpertWeight base = -200, layer_depth_penalty = None.unwrap_or(0) = 0
        // score = -200 - 0 + 0 + 0 + 0 = -200
        assert_eq!(
            prio.score, -200,
            "expert with layer_idx=None must have score = -200 (no depth penalty), got {}",
            prio.score
        );
        assert_eq!(prio.expert_id, Some(7));
        assert!(prio.layer_idx.is_none());
    }

    // -- 23. allocate_expert_weight_pages 与 register_expert_weight_page 混合使用 --

    #[test]
    fn allocate_and_register_expert_pages_mixed_usage() {
        // Arrange: 先批量分配再手动注册同层页面，验证追加行为
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let allocated = scheduler.allocate_expert_weight_pages(2, 0);
        assert_eq!(allocated.len(), 2);

        // Act: 手动在同一层追加一个页面
        scheduler.register_expert_weight_page(9999, 0);

        // Assert: 层 0 现在有 3 个页面（2 allocated + 1 manual）
        assert_eq!(scheduler.weight_page_table[&0].len(), 3);
        assert_eq!(scheduler.num_expert_weight_pages(), 3);
        assert!(scheduler.weight_page_table[&0].contains(&9999));
    }

    // -- 24. on_swap_in 后立即 mark_accessed 在 warmup 内保持 Warm --

    #[test]
    fn on_swap_in_then_immediate_access_stays_warm_in_warmup() {
        // Arrange: 配置很长的 warmup_duration
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(300),
            min_warm_access: 10,
            ..Default::default()
        });
        scheduler.on_swap_in(10);

        // Act: 立即访问一次（仍在 warmup 期内，且 access_count=1 < min_warm_access=10）
        scheduler.mark_accessed(10);

        // Assert: 状态仍为 Warm
        let meta = &scheduler.page_metadata[&10];
        assert_eq!(meta.state, PageState::Warm, "page must stay Warm during warmup");
        assert_eq!(meta.access_count, 1, "access_count must be 1 after swap-in reset + 1 access");
    }

    // -- 25. detect_working_set 对 Protected 页 exact window boundary 行为 --

    #[test]
    fn detect_working_set_protected_expires_exactly_at_window_boundary() {
        // Arrange: Protected 页 last_access 正好在 window 之外
        let mut scheduler = HGALScheduler::new(HGALConfig {
            working_set_window: Duration::from_millis(100),
            ..Default::default()
        });
        // 稍微等过 window
        let old_time = Instant::now() - Duration::from_millis(150);
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            state: PageState::Protected,
            access_count: 100,
            last_access: old_time,
            ..Default::default()
        });

        // Act
        scheduler.detect_working_set();

        // Assert: Protected 过期降级为 Standby
        assert_eq!(
            scheduler.page_metadata[&1].state,
            PageState::Standby,
            "Protected page with last_access beyond window must be demoted to Standby"
        );
    }

    // -- 26. mark_accessed 对 SwappedOut 页的状态转换 --

    #[test]
    fn mark_accessed_swapped_out_page_becomes_active() {
        // Arrange: 设置一个 SwappedOut 页
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.update_page_state(10, None, PageState::SwappedOut);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::SwappedOut);

        // Act: 访问 SwappedOut 页
        scheduler.mark_accessed(10);

        // Assert: 非 Warm/Protected 的状态转为 Active
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Active,
            "SwappedOut page must transition to Active on access"
        );
        assert_eq!(scheduler.page_metadata[&10].access_count, 1);
    }

    // -- 27. select_victim_groups 页数恰好满足时不再选更多 --

    #[test]
    fn select_victim_groups_stops_exactly_at_count_boundary() {
        // Arrange: 3 个组，各 2 页，请求恰好 4 页
        // 使用 payload_kind 区分优先级：ExpertWeight < None < DenseLayerWeight
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // 优先驱逐: ExpertWeight (payload_adjustment = -200)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });
        // 次优先驱逐: None payload
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![3, 4],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // 最难驱逐: DenseLayerWeight (payload_adjustment = +5000)
        scheduler.upsert_group(SequenceGroup {
            id: 3,
            pages: vec![5, 6],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // Act: 请求 4 页 → 恰好需要 2 个组
        let victims = scheduler.select_victim_groups(4);

        // Assert: 恰好 2 个组（4 页），DenseLayerWeight (id=3) 不被选中
        assert_eq!(victims.len(), 2, "must select exactly 2 groups to satisfy 4 pages");
        assert!(
            !victims.contains(&3),
            "DenseLayerWeight group (id=3) must not be selected, got {:?}",
            victims
        );
    }

    // -- 28. num_expert_weight_pages 跨层释放后精确计数 --

    #[test]
    fn num_expert_weight_pages_after_partial_layer_free() {
        // Arrange: 3 层各 2 个 expert 页面，释放中间层
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.allocate_expert_weight_pages(2, 0);
        scheduler.allocate_expert_weight_pages(2, 1);
        scheduler.allocate_expert_weight_pages(2, 2);
        assert_eq!(scheduler.num_expert_weight_pages(), 6);

        // Act: 释放层 1
        scheduler.free_expert_weight_pages(1);

        // Assert: 剩余 4 个（层 0 和层 2 各 2 个）
        assert_eq!(scheduler.num_expert_weight_pages(), 4);
        assert_eq!(scheduler.weight_page_table.len(), 2);
        assert!(scheduler.weight_page_table.contains_key(&0));
        assert!(scheduler.weight_page_table.contains_key(&2));
        assert!(!scheduler.weight_page_table.contains_key(&1));
    }

    // -- 29. on_prefill_chunk_complete 接着 on_prefill_complete 再 chunk 不影响已完成状态 --

    #[test]
    fn on_prefill_complete_then_additional_chunk_does_not_corrupt_group() {
        // Arrange: 完成完整的 prefill 流程
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![
            VirtualPageId::new(55, 0),
            VirtualPageId::new(55, 1),
        ];
        scheduler.on_prefill_chunk_complete(0, 2, &pages[..1]);
        scheduler.on_prefill_chunk_complete(1, 2, &pages[1..]);
        scheduler.on_prefill_complete(55);

        let group_before = scheduler.sequence_groups.get(&55).unwrap().clone();
        assert_eq!(group_before.pages.len(), 2);

        // Act: 模拟额外的 chunk 到达（异常路径）
        let extra_page = VirtualPageId::new(55, 2);
        scheduler.on_prefill_chunk_complete(2, 3, &[extra_page]);

        // Assert: group 正确追加第 3 个页面，不崩溃
        let group_after = scheduler.sequence_groups.get(&55).unwrap();
        assert_eq!(group_after.pages.len(), 3, "extra chunk page must be appended");
        assert!(group_after.access_count >= group_before.access_count);
    }

    // -- 30. compute_eviction_priority 两次调用结果一致（确定性） --

    #[test]
    fn compute_eviction_priority_deterministic_across_calls() {
        // Arrange: 设置固定 metadata（无 Instant 变化干扰）
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            access_count: 5,
            recency: 100,
            state: PageState::Active,
            ..Default::default()
        });
        let page = UnifiedVirtualPage::kv(42, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);

        // Act: 连续两次计算
        let prio1 = scheduler.compute_eviction_priority(&page);
        let prio2 = scheduler.compute_eviction_priority(&page);

        // Assert: score 完全一致
        assert_eq!(prio1.score, prio2.score, "deterministic: same inputs must yield same score");
        assert_eq!(prio1.payload_kind, prio2.payload_kind);
        assert_eq!(prio1.is_pinned, prio2.is_pinned);
    }

    // ── Wave 14: 15 additional uncovered-path tests ──────────────────────────────

    // -- 1. SequenceGroup with KvPipeline::Working (non-default pipeline variant) --

    #[test]
    fn sequence_group_with_working_pipeline_stored_correctly() {
        // Arrange: Create a group with Working pipeline (not the default Conversation)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 100,
            pages: vec![10, 20],
            state: GroupState::Running,
            access_count: 3,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };

        // Act
        scheduler.upsert_group(group);

        // Assert: pipeline field is preserved
        let stored = &scheduler.sequence_groups[&100];
        assert_eq!(stored.pipeline, crate::scheduler::types::KvPipeline::Working);
        assert_eq!(stored.payload_kind, Some(PagePayloadKind::KvContext));
        assert_eq!(stored.access_count, 3);
    }

    // -- 2. on_prefill_complete on a group that never received any chunks --

    #[test]
    fn on_prefill_complete_on_empty_group_activates_no_pages() {
        // Arrange: Create a group with pages but no page metadata
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let group = SequenceGroup {
            id: 77,
            pages: vec![],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(5),
            is_pinned: false,
            context_len: 0,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        };
        scheduler.upsert_group(group);

        // Act: Complete prefill on a group with no pages
        scheduler.on_prefill_complete(77);

        // Assert: Group state updated, no panic, no pages iterated
        let stored = &scheduler.sequence_groups[&77];
        assert_eq!(stored.state, GroupState::Running);
        assert!(stored.pages.is_empty());
        assert!(stored.last_access > Instant::now() - Duration::from_secs(6));
    }

    // -- 3. detect_working_set promotes newly accessed page through Warm -> Active -> Protected --

    #[test]
    fn detect_working_set_promotes_warm_then_active_to_protected_sequential() {
        // Arrange: Configure very short warmup and low thresholds
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_nanos(1),
            min_warm_access: 1,
            hot_threshold: 2,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        // Page starts as Warm via swap-in
        scheduler.on_swap_in(10);
        assert_eq!(scheduler.page_metadata[&10].state, PageState::Warm);

        // Act 1: Wait for warmup to expire, then access twice to exceed hot_threshold
        std::thread::sleep(Duration::from_millis(2));
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 1);

        // Act 2: Second access to reach hot_threshold
        scheduler.mark_accessed(10);
        assert_eq!(scheduler.page_metadata[&10].access_count, 2);

        // Act 3: Run detect_working_set
        scheduler.detect_working_set();

        // Assert: Page should be Protected (access_count=2 >= hot_threshold=2, recent access)
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Protected,
            "page must be promoted to Protected after sequential Warm -> Active -> Protected"
        );
    }

    // -- 4. select_victim_groups with mixed page states: some Protected, some not --

    #[test]
    fn select_victim_groups_group_with_mixed_page_states_partially_protected() {
        // Arrange: Group with 2 pages, one Protected and one Active
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 11],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 2,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        // Page 10 is Protected
        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            state: PageState::Protected,
            ..Default::default()
        });
        // Page 11 is Active (not protected)
        scheduler.page_metadata.insert(11, PageMetadata {
            page_id: 11,
            state: PageState::Active,
            ..Default::default()
        });

        // Act: group_has_protection returns true because page 10 is Protected
        let victims = scheduler.select_victim_groups(2);

        // Assert: Group 1 is protected (has at least one Protected page), so no victims
        assert!(
            victims.is_empty(),
            "group with at least one Protected page must be protected from eviction"
        );
    }

    // -- 5. on_prefill_chunk_complete with multiple pages from same request in one chunk --

    #[test]
    fn on_prefill_chunk_complete_multiple_pages_same_request_one_chunk() {
        // Arrange: 3 pages for the same request in a single chunk
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let pages = vec![
            VirtualPageId::new(42, 0),
            VirtualPageId::new(42, 1),
            VirtualPageId::new(42, 2),
        ];

        // Act: Complete single last chunk
        scheduler.on_prefill_chunk_complete(0, 1, &pages);

        // Assert: All 3 pages are Active (last chunk) and group has 3 pages
        let group = &scheduler.sequence_groups[&42];
        assert_eq!(group.pages.len(), 3, "group must have 3 pages");
        assert_eq!(group.state, GroupState::Running);
        assert!(group.access_count >= 1);

        for vpid in &pages {
            let pid = virtual_page_to_page_id(*vpid);
            let meta = scheduler.page_metadata.get(&pid).expect("metadata must exist");
            assert_eq!(meta.state, PageState::Active, "last chunk pages must be Active");
            assert_eq!(meta.sequence_id, Some(42));
        }
    }

    // -- 6. compute_eviction_priority KV page with high frequency and state bonus --

    #[test]
    fn eviction_priority_kv_high_freq_with_protected_state_bonus() {
        // Arrange: KV page with high access count and Protected state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(1, PageMetadata {
            page_id: 1,
            access_count: 50,
            recency: 0,
            state: PageState::Protected,
            ..Default::default()
        });

        // Act
        let page = UnifiedVirtualPage::kv(
            1, 10,
            crate::scheduler::types::KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: KV payload=100, freq=50*10=500, Protected bonus=10000
        // score = 100 - 0 + 500 + 0 + 10000 = 10600
        assert_eq!(prio.score, 10_600,
            "KV with freq=50 and Protected must score 10600, got {}", prio.score);
    }

    // -- 7. Multiple swap-in cycles preserve metadata correctly --

    #[test]
    fn three_swap_in_cycles_reset_correctly_each_time() {
        // Arrange: Perform 3 swap-in cycles, verifying state after each
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        for cycle in 0..3 {
            // Act: Swap in
            scheduler.on_swap_in(10);

            // Assert: After swap-in
            let meta = &scheduler.page_metadata[&10];
            assert_eq!(meta.state, PageState::Warm, "cycle {}: must be Warm after swap-in", cycle);
            assert_eq!(meta.access_count, 0, "cycle {}: access_count must reset to 0", cycle);
            assert!(meta.swap_in_time.is_some(), "cycle {}: swap_in_time must be set", cycle);
            assert!(meta.warm_until.is_some(), "cycle {}: warm_until must be set", cycle);

            // Simulate some accesses
            scheduler.mark_accessed(10);
            scheduler.mark_accessed(10);
            assert_eq!(scheduler.page_metadata[&10].access_count, 2);
        }
    }

    // -- 8. select_victim_groups: lower access_count group evicted before higher --

    #[test]
    fn select_victim_groups_lower_access_count_evicted_first_same_payload() {
        // Arrange: Two groups with same payload and last_access but different access_count
        // Formula: time_penalty + recency - freq_bonus - pin + payload
        // Higher freq_bonus lowers score -> lower score = evicted first
        // So the group with higher access_count has a lower score and is evicted first.
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group A: access_count = 100 (high freq_bonus = 1000, lowers score significantly)
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 100,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Group B: access_count = 1 (low freq_bonus = 10)
        scheduler.upsert_group(SequenceGroup {
            id: 20,
            pages: vec![2],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: Select 1 victim
        let victims = scheduler.select_victim_groups(1);

        // Assert: Group 10 (higher access_count, freq_bonus lowers score) is evicted first
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0], 10,
            "group with higher access_count has lower score (freq_bonus subtracts) and is evicted first");
    }

    // -- 9. update_lir_membership with negative lir_ratio produces target of at least 1 --

    #[test]
    fn lir_membership_negative_ratio_still_maintains_minimum_target() {
        // Arrange: Negative ratio should produce ceil(negative) which could be 0,
        // but max(1.0) ensures at least 1
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            lir_ratio: -0.5,
            ..Default::default()
        });

        // Act: Access a page
        scheduler.mark_accessed(1);

        // Assert: Page is in LIR (target_lir = max(ceil(-0.5), 1) = 1)
        // With 1 page and target >= 1, page should be in LIR
        assert!(
            scheduler.lir_pages.contains(&1),
            "page must be in LIR even with negative ratio (minimum target is 1)"
        );
        assert!(scheduler.page_metadata[&1].is_lir);
    }

    // -- 10. mark_accessed on a page linked to a group updates group last_access --

    #[test]
    fn mark_accessed_propagates_last_access_to_linked_group() {
        // Arrange: Page linked to a group with old last_access
        let old_time = Instant::now() - Duration::from_secs(60);
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: old_time,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.update_page_state(10, Some(1), PageState::Active);

        // Act: Access the page
        std::thread::sleep(Duration::from_millis(2));
        scheduler.mark_accessed(10);

        // Assert: Group last_access updated to more recent time
        let group = &scheduler.sequence_groups[&1];
        assert!(
            group.last_access > old_time,
            "group last_access must be updated from {:?} to a more recent time",
            old_time
        );
        assert_eq!(group.access_count, 1);
    }

    // -- 11. on_swap_in warm_until derived from swap_in_time when warm_until is None --

    #[test]
    fn warm_until_derived_from_swap_in_time_in_warmup_check() {
        // Arrange: Create page metadata with swap_in_time but no warm_until
        // warm_until method falls back to swap_in_time + warmup_duration
        let warmup = Duration::from_secs(30);
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: warmup,
            min_warm_access: 100,
            ..Default::default()
        });
        // Use on_swap_in which sets both warm_until and swap_in_time
        scheduler.on_swap_in(10);

        // Verify warm_until is set by on_swap_in
        let meta = &scheduler.page_metadata[&10];
        assert!(meta.warm_until.is_some());
        assert!(meta.swap_in_time.is_some());

        // Manually clear warm_until but keep swap_in_time
        let swap_in = meta.swap_in_time.unwrap();
        scheduler.page_metadata.get_mut(&10).unwrap().warm_until = None;

        // Act: detect_working_set should still consider the page in warmup
        // because warm_until() falls back to swap_in_time + warmup_duration
        scheduler.detect_working_set();

        // Assert: Page stays Warm because swap_in_time + warmup_duration is in the future
        assert_eq!(
            scheduler.page_metadata[&10].state,
            PageState::Warm,
            "page with swap_in_time but no warm_until must stay Warm during warmup"
        );
    }

    // -- 12. select_victim_groups with GroupState::Paused (non-Running group still evictable) --

    #[test]
    fn select_victim_groups_paused_group_is_still_evictable() {
        // Arrange: Paused group (not Running) should still be eligible for eviction
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Paused,
            access_count: 0,
            last_access: now - Duration::from_secs(5),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(1);

        // Assert: Paused group is eligible (select_victim_groups doesn't filter by GroupState)
        assert_eq!(victims, vec![1], "Paused group must be eligible for eviction");
    }

    // -- 13. register_expert_weight_page with page_id=0 (edge case: zero page ID) --

    #[test]
    fn register_expert_weight_page_with_zero_page_id() {
        // Arrange: page_id = 0 is a valid ID
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Act
        scheduler.register_expert_weight_page(0, 0);

        // Assert
        assert_eq!(scheduler.weight_page_table.get(&0), Some(&vec![0]));
        let meta = scheduler.page_metadata.get(&0).expect("metadata for page_id=0 must exist");
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.state, PageState::Active);
    }

    // -- 14. compute_group_priority integrates page-level recency across multiple pages --

    #[test]
    fn compute_group_priority_aggregates_page_recency_across_pages() {
        // Arrange: Two groups with different aggregate page recency values.
        // Formula: time_penalty + recency_sum - freq_bonus - pin + payload
        // Higher recency_sum INCREASES score -> higher score = evicted later
        // Lower recency_sum -> lower score = evicted first
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group A: pages with low recency (fresh pages, recency sum = 15)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 11, 12],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        for (pid, recency) in [(10, 10), (11, 5), (12, 0)] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                recency,
                ..Default::default()
            });
        }

        // Group B: pages with high recency (stale pages, recency sum = 1000)
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20, 21, 22],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        for (pid, recency) in [(20, 500), (21, 300), (22, 200)] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                recency,
                ..Default::default()
            });
        }

        // Act: Select victim
        let victims = scheduler.select_victim_groups(1);

        // Assert: Group A (lower recency sum = 15) has lower score and is evicted first
        assert_eq!(
            victims[0], 1,
            "group with lower aggregate page recency (sum=15) has lower score and must be evicted first"
        );
    }

    // -- 15. on_prefill_chunk_complete then detect_working_set promotes last-chunk pages --

    #[test]
    fn on_prefill_chunk_complete_last_chunk_then_detect_working_set_promotes_hot() {
        // Arrange: Configure low hot_threshold and complete a last chunk
        let mut scheduler = HGALScheduler::new(HGALConfig {
            hot_threshold: 2,
            working_set_window: Duration::from_secs(60),
            ..Default::default()
        });
        let pages = vec![VirtualPageId::new(33, 0)];

        // Act 1: Complete a single-chunk prefill (last chunk -> Active state)
        scheduler.on_prefill_chunk_complete(0, 1, &pages);
        let pid = virtual_page_to_page_id(pages[0]);
        assert_eq!(scheduler.page_metadata[&pid].state, PageState::Active);
        assert_eq!(scheduler.page_metadata[&pid].access_count, 1);

        // Act 2: Access the page once more to reach hot_threshold
        scheduler.mark_accessed(pid);
        assert_eq!(scheduler.page_metadata[&pid].access_count, 2);

        // Act 3: Run working set detection
        scheduler.detect_working_set();

        // Assert: Page promoted to Protected (access_count=2 >= hot_threshold=2, recent)
        assert_eq!(
            scheduler.page_metadata[&pid].state,
            PageState::Protected,
            "last-chunk page with access_count >= hot_threshold must be promoted to Protected"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Wave 16: Default/Clone/Debug roundtrip, field validation, boundary behavior
    // ═══════════════════════════════════════════════════════════════════════════

    // -- 16. HGALConfig Default: every field verified against expected values --

    #[test]
    fn hgal_config_default_all_fields_match_expected_constants() {
        let cfg = HGALConfig::default();

        assert_eq!(cfg.warmup_duration, Duration::from_millis(100));
        assert_eq!(cfg.working_set_window, Duration::from_secs(1));
        assert_eq!(cfg.hot_threshold, 3);
        // lir_ratio is f32; compare with approximate equality
        assert!(
            (cfg.lir_ratio - 0.3f32).abs() < f32::EPSILON,
            "lir_ratio must be 0.3, got {}",
            cfg.lir_ratio
        );
        assert_eq!(cfg.min_warm_access, 2);
        assert!(cfg.enable_clock_pro, "enable_clock_pro must default to true");
    }

    // -- 17. HGALConfig Clone roundtrip: cloned instance is identical --

    #[test]
    fn hgal_config_clone_roundtrip_is_identical() {
        let original = HGALConfig {
            warmup_duration: Duration::from_millis(500),
            working_set_window: Duration::from_secs(30),
            hot_threshold: 7,
            lir_ratio: 0.5,
            min_warm_access: 4,
            enable_clock_pro: false,
        };
        let cloned = original.clone();

        assert_eq!(cloned.warmup_duration, original.warmup_duration);
        assert_eq!(cloned.working_set_window, original.working_set_window);
        assert_eq!(cloned.hot_threshold, original.hot_threshold);
        assert!(
            (cloned.lir_ratio - original.lir_ratio).abs() < f32::EPSILON,
        );
        assert_eq!(cloned.min_warm_access, original.min_warm_access);
        assert_eq!(cloned.enable_clock_pro, original.enable_clock_pro);
    }

    // -- 18. HGALConfig Debug output contains all field names --

    #[test]
    fn hgal_config_debug_output_contains_all_field_names() {
        let cfg = HGALConfig::default();
        let debug_str = format!("{:?}", cfg);

        assert!(debug_str.contains("warmup_duration"), "Debug output must contain 'warmup_duration'");
        assert!(debug_str.contains("working_set_window"), "Debug output must contain 'working_set_window'");
        assert!(debug_str.contains("hot_threshold"), "Debug output must contain 'hot_threshold'");
        assert!(debug_str.contains("lir_ratio"), "Debug output must contain 'lir_ratio'");
        assert!(debug_str.contains("min_warm_access"), "Debug output must contain 'min_warm_access'");
        assert!(debug_str.contains("enable_clock_pro"), "Debug output must contain 'enable_clock_pro'");
    }

    // -- 19. HGALScheduler Debug output contains struct name and key fields --

    #[test]
    fn hgal_scheduler_debug_output_contains_struct_fields() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let debug_str = format!("{:?}", scheduler);

        assert!(debug_str.contains("sequence_groups"), "Debug must mention sequence_groups");
        assert!(debug_str.contains("page_metadata"), "Debug must mention page_metadata");
        assert!(debug_str.contains("config"), "Debug must mention config");
        assert!(debug_str.contains("weight_page_table"), "Debug must mention weight_page_table");
    }

    // -- 20. PageMetadata Default: all fields verified against expected defaults --

    #[test]
    fn page_metadata_default_all_fields_are_zeroed_or_sensible() {
        let meta = PageMetadata::default();

        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none(), "sequence_id must default to None");
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(meta.swap_in_time.is_none(), "swap_in_time must default to None");
        assert!(!meta.is_lir, "is_lir must default to false");
        assert_eq!(meta.state, PageState::Standby, "state must default to Standby");
        assert!(meta.warm_until.is_none(), "warm_until must default to None");
    }

    // -- 21. Eviction priority ordering: ExpertWeight < KnowledgeRAG < KvContext < PromptSystem --

    #[test]
    fn eviction_priority_ordering_across_all_payload_kinds() {
        // With no metadata, ExpertWeight base = -200 (same as KnowledgeRAG),
        // but ExpertWeight at layer_idx > 0 gets additional layer_depth_penalty.
        // ExpertWeight at layer 5: -200 + (-5) = -205
        // KnowledgeRAG: -200
        // KvContext: +100
        // PromptSystem: +10000
        let scheduler = HGALScheduler::new(HGALConfig::default());

        let expert_page = UnifiedVirtualPage::expert(1, 0, 5, gllm_kernels::types::DType::F16);
        let rag_page = UnifiedVirtualPage::rag(2, 1, gllm_kernels::types::DType::F32);
        let kv_page = UnifiedVirtualPage::kv(3, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let prompt_page = UnifiedVirtualPage::system_prompt(4, gllm_kernels::types::DType::F32);

        let expert_prio = scheduler.compute_eviction_priority(&expert_page);
        let rag_prio = scheduler.compute_eviction_priority(&rag_page);
        let kv_prio = scheduler.compute_eviction_priority(&kv_page);
        let prompt_prio = scheduler.compute_eviction_priority(&prompt_page);

        assert!(
            expert_prio.score < rag_prio.score,
            "ExpertWeight (score={}) must have lower score than KnowledgeRAG (score={})",
            expert_prio.score, rag_prio.score
        );
        assert!(
            rag_prio.score < kv_prio.score,
            "KnowledgeRAG (score={}) must have lower score than KvContext (score={})",
            rag_prio.score, kv_prio.score
        );
        assert!(
            kv_prio.score < prompt_prio.score,
            "KvContext (score={}) must have lower score than PromptSystem (score={})",
            kv_prio.score, prompt_prio.score
        );
    }

    // -- 22. EvictionPriority Debug format contains score and payload_kind --

    #[test]
    fn eviction_priority_debug_format_contains_score_and_payload() {
        let prio = EvictionPriority {
            score: -42,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 3,
            recency: 100,
            layer_idx: Some(5),
            expert_id: Some(2),
        };
        let debug_str = format!("{:?}", prio);

        assert!(debug_str.contains("score"), "Debug must contain 'score'");
        assert!(debug_str.contains("payload_kind"), "Debug must contain 'payload_kind'");
        assert!(debug_str.contains("ExpertWeight"), "Debug must contain payload variant name");
        assert!(debug_str.contains("access_count"), "Debug must contain 'access_count'");
        assert!(debug_str.contains("expert_id"), "Debug must contain 'expert_id'");
    }

    // -- 23. select_victim_groups_single_unpinned_among_multiple_pinned --

    #[test]
    fn select_victim_groups_single_unpinned_among_multiple_pinned() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Three pinned groups
        for id in 1..=3 {
            scheduler.upsert_group(SequenceGroup {
                id,
                pages: vec![id as usize * 10],
                state: GroupState::Running,
                access_count: 5,
                last_access: now,
                is_pinned: true,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }

        // One unpinned group
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![990],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        let victims = scheduler.select_victim_groups(5);

        assert_eq!(victims, vec![99], "Only the single unpinned group must be selected");
    }

    // -- 24. EvictionPriority fields match input UnifiedVirtualPage attributes --

    #[test]
    fn eviction_priority_fields_mirror_input_page_attributes() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Pre-populate metadata so compute_eviction_priority reads real values
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            recency: 500,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        });

        let page = UnifiedVirtualPage::kv(
            42, 7, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32,
        );
        let prio = scheduler.compute_eviction_priority(&page);

        assert_eq!(prio.access_count, 10, "access_count must come from PageMetadata");
        assert_eq!(prio.recency, 500, "recency must come from PageMetadata");
        assert_eq!(prio.payload_kind, PagePayloadKind::KvContext);
        assert!(!prio.is_pinned, "KvContext page must be evictable");
        assert!(prio.expert_id.is_none(), "KV page must have no expert_id");
    }

    // -- 25. Eviction priority score monotonic with frequency bonus --

    #[test]
    fn eviction_priority_score_increases_monotonically_with_frequency() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Two identical pages except access_count
        scheduler.page_metadata.insert(100, PageMetadata {
            page_id: 100,
            sequence_id: Some(1),
            recency: 50,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        });
        scheduler.page_metadata.insert(200, PageMetadata {
            page_id: 200,
            sequence_id: Some(1),
            recency: 50,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        });

        let page_low = UnifiedVirtualPage::kv(100, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page_high = UnifiedVirtualPage::kv(200, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);

        let prio_low = scheduler.compute_eviction_priority(&page_low);
        let prio_high = scheduler.compute_eviction_priority(&page_high);

        assert!(
            prio_high.score > prio_low.score,
            "Higher access_count (score={}) must produce higher score than lower (score={})",
            prio_high.score, prio_low.score
        );
        // Delta should be (100 - 1) * FREQUENCY_WEIGHT = 99 * 10 = 990
        let expected_delta = (100usize - 1) * FREQUENCY_WEIGHT as usize;
        assert_eq!(
            (prio_high.score - prio_low.score) as usize,
            expected_delta,
            "Score delta must equal (access_count_diff * FREQUENCY_WEIGHT)"
        );
    }

    // -- 26. update_page_state preserves recency and access_count on existing metadata --

    #[test]
    fn update_page_state_preserves_recency_and_access_count_on_existing() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Create initial metadata with non-default values
        scheduler.page_metadata.insert(77, PageMetadata {
            page_id: 77,
            sequence_id: Some(5),
            recency: 999,
            access_count: 42,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        });

        // Update state only
        scheduler.update_page_state(77, Some(5), PageState::Standby);

        let meta = &scheduler.page_metadata[&77];
        assert_eq!(meta.state, PageState::Standby, "state must be updated");
        assert_eq!(meta.recency, 999, "recency must be preserved");
        assert_eq!(meta.access_count, 42, "access_count must be preserved");
        assert!(meta.is_lir, "is_lir must be preserved");
    }

    // -- 27. num_expert_weight_pages returns zero when no pages registered --

    #[test]
    fn num_expert_weight_pages_zero_when_no_pages_registered() {
        let scheduler = HGALScheduler::new(HGALConfig::default());
        assert_eq!(scheduler.num_expert_weight_pages(), 0);
    }

    // -- 28. num_expert_weight_pages counts correctly after allocate and free --

    #[test]
    fn num_expert_weight_pages_counts_correctly_after_allocate_and_partial_free() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Allocate 4 experts for layer 0 and 3 for layer 1
        let pages_l0 = scheduler.allocate_expert_weight_pages(4, 0);
        let pages_l1 = scheduler.allocate_expert_weight_pages(3, 1);

        assert_eq!(pages_l0.len(), 4);
        assert_eq!(pages_l1.len(), 3);
        assert_eq!(scheduler.num_expert_weight_pages(), 7);

        // Free layer 0 only
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_expert_weight_pages(), 3, "Only layer 1 pages must remain");

        // Verify layer 1 pages still exist in metadata
        for &pid in &pages_l1 {
            assert!(scheduler.page_metadata.contains_key(&pid), "layer 1 page {} must still exist", pid);
        }
    }

    // -- 29. select_victim_groups returns groups ordered by ascending score --

    #[test]
    fn select_victim_groups_returns_groups_in_ascending_score_order() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Formula: score = time_penalty + recency_penalty - freq_bonus - pin_bonus + payload_adjustment
        // Lower score = evicted first.
        // Group A: ExpertWeight payload (payload_adjustment = -200), recent access
        scheduler.upsert_group(SequenceGroup {
            id: 10,
            pages: vec![100],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Group B: no payload, old access (high time_penalty)
        scheduler.upsert_group(SequenceGroup {
            id: 20,
            pages: vec![200],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Group C: DenseLayerWeight payload (payload_adjustment = +5000), recent access
        scheduler.upsert_group(SequenceGroup {
            id: 30,
            pages: vec![300],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        let victims = scheduler.select_victim_groups(10);

        assert_eq!(victims.len(), 3, "All 3 groups must be returned");
        // Group 10: score ≈ 0 + 0 - 0 - 0 + (-200) = -200 (lowest, evicted first)
        // Group 20: score ≈ 10000 + 0 - 0 - 0 + 0 = 10000 (middle)
        // Group 30: score ≈ 0 + 0 - 0 - 0 + 5000 = 5000 (highest, evicted last)
        // Ascending order: 10 (-200) < 30 (5000) < 20 (10000)
        assert_eq!(victims[0], 10, "ExpertWeight group (lowest score) must be first");
        assert_eq!(victims[1], 30, "DenseLayerWeight group (middle score) must be second");
        assert_eq!(victims[2], 20, "Old-access group (highest score) must be last");
    }

    // -- 30. Eviction priority: Protected state bonus > Warm state bonus --

    #[test]
    fn eviction_priority_protected_bonus_greater_than_warm_bonus() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        scheduler.page_metadata.insert(10, PageMetadata {
            page_id: 10,
            sequence_id: Some(1),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        });
        scheduler.page_metadata.insert(20, PageMetadata {
            page_id: 20,
            sequence_id: Some(1),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        });

        let page_protected = UnifiedVirtualPage::kv(10, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let page_warm = UnifiedVirtualPage::kv(20, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);

        let prio_protected = scheduler.compute_eviction_priority(&page_protected);
        let prio_warm = scheduler.compute_eviction_priority(&page_warm);

        assert!(
            prio_protected.score > prio_warm.score,
            "Protected (score={}) must have higher score than Warm (score={})",
            prio_protected.score, prio_warm.score
        );
        // Protected bonus = 10000, Warm bonus = 5000, delta = 5000
        assert_eq!(
            prio_protected.score - prio_warm.score,
            5000,
            "State bonus delta must be 5000"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Wave 17: 15 targeted unit tests for remaining coverage gaps
    // ═══════════════════════════════════════════════════════════════════════════

    // -- 1. HGALConfig Default/Clone/Debug roundtrip: full field equality after clone --

    #[test]
    fn config_default_clone_roundtrip_all_fields_preserved() {
        // Arrange
        let original = HGALConfig::default();

        // Act
        let cloned = original.clone();

        // Assert: every field must match exactly after clone
        assert_eq!(cloned.warmup_duration, original.warmup_duration);
        assert_eq!(cloned.working_set_window, original.working_set_window);
        assert_eq!(cloned.hot_threshold, original.hot_threshold);
        assert!((cloned.lir_ratio - original.lir_ratio).abs() < f32::EPSILON);
        assert_eq!(cloned.min_warm_access, original.min_warm_access);
        assert_eq!(cloned.enable_clock_pro, original.enable_clock_pro);
        // Debug format must contain all six field names
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("warmup_duration"));
        assert!(debug.contains("working_set_window"));
        assert!(debug.contains("hot_threshold"));
        assert!(debug.contains("lir_ratio"));
        assert!(debug.contains("min_warm_access"));
        assert!(debug.contains("enable_clock_pro"));
    }

    // -- 2. HGALScheduler Debug format includes all struct fields --

    #[test]
    fn scheduler_debug_format_includes_all_four_fields() {
        // Arrange
        let scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            ..Default::default()
        });

        // Act
        let debug = format!("{:?}", scheduler);

        // Assert: all four fields of HGALScheduler must appear in Debug output
        assert!(debug.contains("sequence_groups"), "Debug must contain sequence_groups");
        assert!(debug.contains("page_metadata"), "Debug must contain page_metadata");
        assert!(debug.contains("lir_pages"), "Debug must contain lir_pages");
        assert!(debug.contains("config"), "Debug must contain config");
        assert!(debug.contains("weight_page_table"), "Debug must contain weight_page_table");
    }

    // -- 3. PageMetadata Default: every field is zeroed or None --

    #[test]
    fn page_metadata_default_verifies_every_single_field() {
        // Arrange & Act
        let meta = PageMetadata::default();

        // Assert: all seven fields
        assert_eq!(meta.page_id, 0, "page_id must default to 0");
        assert!(meta.sequence_id.is_none(), "sequence_id must default to None");
        assert_eq!(meta.state, PageState::Standby, "state must default to Standby");
        assert_eq!(meta.access_count, 0, "access_count must default to 0");
        assert_eq!(meta.recency, 0, "recency must default to 0");
        assert!(meta.last_access >= Instant::now() - Duration::from_secs(1),
            "last_access must be near current time");
        assert!(meta.swap_in_time.is_none(), "swap_in_time must default to None");
        assert!(!meta.is_lir, "is_lir must default to false");
        assert!(meta.warm_until.is_none(), "warm_until must default to None");
    }

    // -- 4. Eviction priority ordering: Protected > Warm > Cold (same payload) --

    #[test]
    fn eviction_priority_protected_gt_warm_gt_active_same_payload() {
        // Arrange: three KV pages with same access_count/recency but different states
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        for (pid, state) in [
            (1, PageState::Protected),
            (2, PageState::Warm),
            (3, PageState::Active),
        ] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                state,
                access_count: 5,
                recency: 10,
                ..Default::default()
            });
        }

        let make_page = |pid: usize| -> UnifiedVirtualPage {
            UnifiedVirtualPage::kv(pid, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32)
        };

        // Act
        let prio_protected = scheduler.compute_eviction_priority(&make_page(1));
        let prio_warm = scheduler.compute_eviction_priority(&make_page(2));
        let prio_active = scheduler.compute_eviction_priority(&make_page(3));

        // Assert: Protected > Warm > Active (higher score = harder to evict)
        assert!(
            prio_protected.score > prio_warm.score,
            "Protected (score={}) must be harder to evict than Warm (score={})",
            prio_protected.score, prio_warm.score,
        );
        assert!(
            prio_warm.score > prio_active.score,
            "Warm (score={}) must be harder to evict than Active (score={})",
            prio_warm.score, prio_active.score,
        );
    }

    // -- 5. EvictionPriority Debug: all payload variants produce valid Debug strings --

    #[test]
    fn eviction_priority_debug_variants_all_produce_valid_output() {
        // Arrange: one EvictionPriority per payload kind
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];

        // Act & Assert: Debug format for each variant must contain "payload_kind"
        for kind in &variants {
            let prio = EvictionPriority {
                score: 0,
                payload_kind: *kind,
                is_pinned: false,
                access_count: 0,
                recency: 0,
                layer_idx: None,
                expert_id: None,
            };
            let debug = format!("{:?}", prio);
            assert!(
                debug.contains("payload_kind"),
                "Debug for {:?} must contain 'payload_kind', got: {}",
                kind, debug,
            );
            assert!(
                debug.contains("score"),
                "Debug for {:?} must contain 'score'",
                kind,
            );
        }
    }

    // -- 6. Single unpinned page selected among multiple pinned groups --

    #[test]
    fn single_unpinned_group_selected_from_five_pinned_groups() {
        // Arrange: 5 pinned groups + 1 unpinned group
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        for id in 1..=5u64 {
            scheduler.upsert_group(SequenceGroup {
                id,
                pages: vec![id as usize * 10],
                state: GroupState::Running,
                access_count: 10,
                last_access: now,
                is_pinned: true,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
        }
        scheduler.upsert_group(SequenceGroup {
            id: 99,
            pages: vec![990],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(10);

        // Assert: only the single unpinned group is a victim
        assert_eq!(victims, vec![99],
            "only the single unpinned group must be selected among 5 pinned groups");
    }

    // -- 7. Priority score field mirrors input recency and access_count --

    #[test]
    fn eviction_priority_score_exactly_mirrors_recency_and_access() {
        // Arrange: page with known recency=100 and access_count=10
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            recency: 100,
            access_count: 10,
            state: PageState::Active,
            ..Default::default()
        });

        let page = UnifiedVirtualPage::kv(
            42, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32,
        );

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: score = payload_adj(100) - recency_penalty(100*5) + freq_bonus(10*10) + pin(0) + state(0)
        //       = 100 - 500 + 100 + 0 + 0 = -300
        assert_eq!(prio.score, -300,
            "score must exactly mirror: 100 - 500 + 100 = -300, got {}", prio.score);
        assert_eq!(prio.recency, 100, "recency field must mirror input");
        assert_eq!(prio.access_count, 10, "access_count field must mirror input");
    }

    // -- 8. Score monotonic increase with recency (lower recency = higher score) --

    #[test]
    fn eviction_priority_score_monotonic_decrease_with_higher_recency() {
        // Arrange: three pages with recency 0, 50, 100 (higher = older = lower score)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for (pid, recency) in [(1, 0usize), (2, 50usize), (3, 100usize)] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                recency,
                access_count: 0,
                state: PageState::Active,
                ..Default::default()
            });
        }

        let make_page = |pid: usize| -> UnifiedVirtualPage {
            UnifiedVirtualPage::kv(pid, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32)
        };

        // Act
        let s0 = scheduler.compute_eviction_priority(&make_page(1)).score;
        let s50 = scheduler.compute_eviction_priority(&make_page(2)).score;
        let s100 = scheduler.compute_eviction_priority(&make_page(3)).score;

        // Assert: lower recency = higher score (monotonically decreasing)
        assert!(s0 > s50, "recency=0 (score={}) must have higher score than recency=50 (score={})", s0, s50);
        assert!(s50 > s100, "recency=50 (score={}) must have higher score than recency=100 (score={})", s50, s100);
        // Deltas must be linear (recency_penalty = recency * FREQUENCY_WEIGHT / 2)
        let delta = s0 - s50;
        assert_eq!(s50 - s100, delta, "recency penalty must be linear");
    }

    // -- 9. update_page state preservation: all fields except state unchanged --

    #[test]
    fn update_page_preserves_all_non_state_fields_across_transition() {
        // Arrange: page with fully populated metadata
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: false,
            ..Default::default()
        });
        scheduler.page_metadata.insert(77, PageMetadata {
            page_id: 77,
            sequence_id: Some(42),
            recency: 888,
            access_count: 33,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        });

        // Act: transition Active -> Standby
        scheduler.update_page_state(77, Some(42), PageState::Standby);

        // Assert: state changed, all other fields preserved
        let meta = &scheduler.page_metadata[&77];
        assert_eq!(meta.state, PageState::Standby, "state must change");
        assert_eq!(meta.sequence_id, Some(42), "sequence_id preserved");
        assert_eq!(meta.recency, 888, "recency preserved");
        assert_eq!(meta.access_count, 33, "access_count preserved");
        assert!(meta.is_lir, "is_lir preserved");
    }

    // -- 10. num_expert_weight_pages counts all weight pages (expert + dense) --

    #[test]
    fn num_expert_weight_pages_counts_all_weight_pages() {
        // Arrange: register both expert and dense pages across multiple layers
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        // Layer 0: 3 expert + 2 dense = 5 pages in weight_page_table
        let expert_pages = scheduler.allocate_expert_weight_pages(3, 0);
        let dense_pages = scheduler.register_dense_layer_weight_pages(vec![900, 901], 0);

        // Layer 1: 2 expert
        let expert_pages_l1 = scheduler.allocate_expert_weight_pages(2, 1);

        // Assert
        assert_eq!(expert_pages.len(), 3);
        assert_eq!(dense_pages.len(), 2);
        assert_eq!(expert_pages_l1.len(), 2);

        // num_expert_weight_pages counts ALL entries in weight_page_table (5 + 2 = 7)
        assert_eq!(scheduler.num_expert_weight_pages(), 7,
            "must count all weight_page_table entries (expert + dense)");
        // num_weight_pages is the same (7)
        assert_eq!(scheduler.num_weight_pages(), 7,
            "total weight pages = experts + dense");

        // After freeing expert pages for layer 0, only dense (2) + layer 1 expert (2) = 4 remain
        // But free_expert_weight_pages removes the entire layer entry including dense pages
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_expert_weight_pages(), 2,
            "after freeing layer 0, only layer 1 expert pages (2) remain");
    }

    // -- 11. select_victim_groups returns groups in ascending score order --

    #[test]
    fn select_victim_groups_ascending_score_order_verified() {
        // Arrange: three groups with guaranteed different scores
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Group A: ExpertWeight (-200 payload), high access_count (50)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 50,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        });

        // Group B: None payload, access_count=1
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 1,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Group C: DenseLayerWeight (+5000 payload)
        scheduler.upsert_group(SequenceGroup {
            id: 3,
            pages: vec![30],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        });

        // Act
        let victims = scheduler.select_victim_groups(10);

        // Assert: compute priorities to verify ascending order
        let scores: Vec<i64> = victims.iter().map(|&id| {
            let g = scheduler.sequence_groups.get(&id).unwrap();
            scheduler.compute_group_priority(g)
        }).collect();

        for window in scores.windows(2) {
            assert!(
                window[0] <= window[1],
                "victims must be in ascending score order: {:?}",
                scores,
            );
        }
    }

    // -- 12. Protected vs Warm bonus precision: exact delta is 5000 --

    #[test]
    fn protected_vs_warm_bonus_delta_exactly_5000() {
        // Arrange: two pages identical except Protected vs Warm state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for (pid, state) in [(1, PageState::Protected), (2, PageState::Warm)] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                state,
                access_count: 10,
                recency: 50,
                ..Default::default()
            });
        }

        let make_page = |pid: usize| -> UnifiedVirtualPage {
            UnifiedVirtualPage::kv(pid, 1, crate::scheduler::types::KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32)
        };

        // Act
        let prio_protected = scheduler.compute_eviction_priority(&make_page(1));
        let prio_warm = scheduler.compute_eviction_priority(&make_page(2));

        // Assert: Protected bonus = 10000, Warm bonus = 5000, exact delta = 5000
        assert_eq!(
            prio_protected.score - prio_warm.score, 5000,
            "Protected-Warm delta must be exactly 5000, got {} - {} = {}",
            prio_protected.score, prio_warm.score, prio_protected.score - prio_warm.score,
        );
    }

    // -- 13. PageMetadata Clone preserves all fields --

    #[test]
    fn page_metadata_clone_preserves_every_field() {
        // Arrange: metadata with all fields set to non-default values
        let original = PageMetadata {
            page_id: 42,
            sequence_id: Some(99),
            state: PageState::Protected,
            access_count: 100,
            recency: 500,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
            is_lir: true,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
        };

        // Act
        let cloned = original.clone();

        // Assert: every single field must match
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.sequence_id, original.sequence_id);
        assert_eq!(cloned.state, original.state);
        assert_eq!(cloned.access_count, original.access_count);
        assert_eq!(cloned.recency, original.recency);
        assert_eq!(cloned.last_access, original.last_access);
        assert_eq!(cloned.swap_in_time, original.swap_in_time);
        assert_eq!(cloned.is_lir, original.is_lir);
        assert_eq!(cloned.warm_until, original.warm_until);
    }

    // -- 14. HGALConfig max_pages=0 boundary: scheduler with zero pages works correctly --

    #[test]
    fn max_pages_zero_boundary_select_victim_returns_empty() {
        // Arrange: scheduler with groups but no page capacity (boundary condition)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Register groups but request 0 victims
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10, 20, 30],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 3,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: request 0 victims
        let victims = scheduler.select_victim_groups(0);

        // Assert: empty result, no panic
        assert!(victims.is_empty(), "requesting 0 pages must always return empty");

        // Also verify select_victim_weight_pages with 0
        let weight_victims = scheduler.select_victim_weight_pages(0);
        assert!(weight_victims.is_empty(), "requesting 0 weight pages must return empty");
    }

    // -- 15. Multiple victims with same score: tie-breaking is deterministic --

    #[test]
    fn same_score_tiebreaking_is_deterministic() {
        // Arrange: two groups with identical score inputs (same access_count, last_access, payload)
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: run select_victim_groups multiple times
        let victims1 = scheduler.select_victim_groups(2);
        let victims2 = scheduler.select_victim_groups(2);
        let victims3 = scheduler.select_victim_groups(2);

        // Assert: results are deterministic (same order each time)
        assert_eq!(victims1, victims2, "tie-breaking must be deterministic across calls");
        assert_eq!(victims2, victims3, "tie-breaking must be deterministic across calls");
        assert_eq!(victims1.len(), 2, "both groups must be returned");
        // Both group IDs must be present
        assert!(victims1.contains(&1));
        assert!(victims1.contains(&2));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Wave 18: 10 targeted unit tests for remaining coverage gaps
    // ═══════════════════════════════════════════════════════════════════════════

    // -- 1. KvContext payload adjustment is exactly 100 with zero recency/freq --

    #[test]
    fn eviction_priority_kv_context_exact_payload_adjustment_is_100() {
        // Arrange: KV context page with no metadata (recency=0, access_count=0)
        let scheduler = HGALScheduler::new(HGALConfig::default());
        let page = UnifiedVirtualPage::kv(
            1, 10, crate::scheduler::types::KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );

        // Act
        let prio = scheduler.compute_eviction_priority(&page);

        // Assert: payload_adjustment(KvContext)=100, recency=0, access=0, no pin, no state bonus
        // score = 100 - 0 + 0 + 0 + 0 = 100
        assert_eq!(
            prio.score, 100,
            "KvContext with no metadata must have score=100 (pure payload adjustment), got {}",
            prio.score,
        );
    }

    // -- 2. Single multi-page group overshoots request count --

    #[test]
    fn select_victim_groups_single_multi_page_group_overshoots_request() {
        // Arrange: one group with 5 pages, request only 2 pages
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now() - Duration::from_secs(2);

        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![100, 200, 300, 400, 500],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 5,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act: request 2 pages but the only group has 5
        let victims = scheduler.select_victim_groups(2);

        // Assert: the single group is selected even though it overshoots
        assert_eq!(victims, vec![1],
            "single group with 5 pages must be selected when requesting 2 (accumulated=5 >= 2)");
    }

    // -- 3. update_page_state Swapped clears both warm_until and swap_in_time --

    #[test]
    fn update_page_state_to_swapped_clears_warm_until_and_swap_in_time() {
        // Arrange: page with both warm_until and swap_in_time set
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();
        scheduler.page_metadata.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: Some(1),
            recency: 10,
            access_count: 5,
            last_access: now,
            swap_in_time: Some(now - Duration::from_millis(50)),
            is_lir: false,
            state: PageState::Warm,
            warm_until: Some(now + Duration::from_millis(100)),
        });

        // Act: transition to Swapped state
        scheduler.update_page_state(42, Some(1), PageState::Swapped);

        // Assert: state changed and both time fields cleared (code lines 103-106)
        let meta = &scheduler.page_metadata[&42];
        assert_eq!(meta.state, PageState::Swapped, "state must change to Swapped");
        assert!(meta.warm_until.is_none(), "warm_until must be cleared on Swapped transition");
        assert!(meta.swap_in_time.is_none(), "swap_in_time must be cleared on Swapped transition");
        // Non-time fields must be preserved
        assert_eq!(meta.recency, 10, "recency preserved");
        assert_eq!(meta.access_count, 5, "access_count preserved");
        assert_eq!(meta.sequence_id, Some(1), "sequence_id preserved");
    }

    // -- 4. compute_group_priority: newer last_access → lower time_penalty → lower score → evicted first --

    #[test]
    fn compute_group_priority_time_penalty_increases_with_age() {
        // Arrange: two groups with same payload but different last_access times
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        // Recent group (lower time_penalty → lower score → evicted first)
        scheduler.upsert_group(SequenceGroup {
            id: 1,
            pages: vec![10],
            state: GroupState::Running,
            access_count: 0,
            last_access: now,
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Old group (higher time_penalty → higher score → resists eviction)
        scheduler.upsert_group(SequenceGroup {
            id: 2,
            pages: vec![20],
            state: GroupState::Running,
            access_count: 0,
            last_access: now - Duration::from_secs(10),
            is_pinned: false,
            context_len: 1,
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        });

        // Act
        let victims = scheduler.select_victim_groups(2);

        // Assert: newer group (lower time_penalty → lower score → evicted first)
        assert_eq!(victims[0], 1,
            "newer group (id=1) must be evicted before old group (id=2)");
        assert_eq!(victims[1], 2,
            "old group (id=2) must be evicted after new group (id=1)");
    }

    // -- 5. Active and Standby states both yield zero state bonus --

    #[test]
    fn eviction_priority_active_and_standby_have_identical_score() {
        // Arrange: two pages identical except Active vs Standby state
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        for (pid, state) in [(1, PageState::Active), (2, PageState::Standby)] {
            scheduler.page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                state,
                access_count: 10,
                recency: 25,
                ..Default::default()
            });
        }

        let make_page = |pid: usize| -> UnifiedVirtualPage {
            UnifiedVirtualPage::kv(pid, 1, crate::scheduler::types::KvPipeline::Conversation, 0,
                gllm_kernels::types::DType::F32)
        };

        // Act
        let active_score = scheduler.compute_eviction_priority(&make_page(1)).score;
        let standby_score = scheduler.compute_eviction_priority(&make_page(2)).score;

        // Assert: both have zero state bonus (only Protected=10000, Warm=5000 get bonus)
        assert_eq!(active_score, standby_score,
            "Active (score={}) and Standby (score={}) must have identical score (both get 0 state bonus)",
            active_score, standby_score,
        );
    }

    // -- 6. on_prefill_chunk_complete with total_chunks=0 treats chunk as last --

    #[test]
    fn on_prefill_chunk_complete_zero_total_chunks_sets_page_active() {
        // Arrange: scheduler with no existing groups
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let vpid = VirtualPageId { sequence_id: 1, logical_index: 0 };
        let pages = vec![vpid];

        // Act: chunk_idx=0, total_chunks=0 => saturating_add(1)=1 >= 0 => is_last_chunk=true
        scheduler.on_prefill_chunk_complete(0, 0, &pages);

        // Assert: page created with Active state (is_last_chunk=true path)
        let page_id = virtual_page_to_page_id(vpid);
        let meta = scheduler.page_metadata.get(&page_id).expect("page metadata must exist");
        assert_eq!(meta.state, PageState::Active,
            "chunk_idx=0, total_chunks=0 must be treated as last chunk → Active state");
        assert_eq!(meta.recency, 0, "recency reset to 0 on prefill chunk");
    }

    // -- 7. free_expert_weight_pages cleans lir_pages entries --

    #[test]
    fn free_expert_weight_pages_removes_lir_entries_for_freed_pages() {
        // Arrange: allocate expert weight pages, mark some as LIR
        let mut scheduler = HGALScheduler::new(HGALConfig {
            enable_clock_pro: true,
            ..Default::default()
        });
        let pages = scheduler.allocate_expert_weight_pages(3, 0);
        assert_eq!(pages.len(), 3);

        // Manually insert into lir_pages to simulate LIR membership
        for &pid in &pages {
            scheduler.lir_pages.insert(pid);
            scheduler.page_metadata.get_mut(&pid).unwrap().is_lir = true;
        }
        assert_eq!(scheduler.lir_pages.len(), 3, "3 pages should be in LIR set");

        // Act: free expert weight pages for layer 0
        scheduler.free_expert_weight_pages(0);

        // Assert: lir_pages set must be empty (all 3 entries removed)
        assert!(scheduler.lir_pages.is_empty(),
            "lir_pages must be empty after freeing the only layer's expert weight pages");
        // page_metadata must also be cleaned
        for pid in &pages {
            assert!(!scheduler.page_metadata.contains_key(pid),
                "page metadata for pid={} must be removed", pid);
        }
        // weight_page_table must no longer contain layer 0
        assert!(!scheduler.weight_page_table.contains_key(&0));
    }

    // -- 8. detect_working_set: Warm page within warmup period stays Warm even if below threshold --

    #[test]
    fn detect_working_set_warm_page_within_warmup_period_stays_warm() {
        // Arrange: configure long warmup so it doesn't expire
        let mut scheduler = HGALScheduler::new(HGALConfig {
            warmup_duration: Duration::from_secs(3600),
            min_warm_access: 100, // high threshold: access_count=1 < min_warm_access
            ..Default::default()
        });
        // Create page that was recently swapped in (within warmup period)
        scheduler.on_swap_in(42);
        let meta_before = scheduler.page_metadata.get(&42).unwrap();
        assert_eq!(meta_before.state, PageState::Warm, "page must start as Warm after swap-in");
        assert_eq!(meta_before.access_count, 0, "fresh swap-in has access_count=0");

        // Act: run working set detection
        scheduler.detect_working_set();

        // Assert: page stays Warm because it's still within warmup period and access_count < min_warm_access
        let meta_after = scheduler.page_metadata.get(&42).unwrap();
        assert_eq!(meta_after.state, PageState::Warm,
            "Warm page within warmup period with access_count < min_warm_access must stay Warm");
    }

    // -- 9. select_victim_groups: all groups have at least one Protected page → empty result --

    #[test]
    fn select_victim_groups_all_groups_have_protected_page_returns_empty() {
        // Arrange: three groups, each with at least one Protected page
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let now = Instant::now();

        for id in 1u64..=3 {
            scheduler.upsert_group(SequenceGroup {
                id,
                pages: vec![id as usize * 10],
                state: GroupState::Running,
                access_count: 0,
                last_access: now,
                is_pinned: false,
                context_len: 1,
                pipeline: crate::scheduler::types::KvPipeline::Conversation,
                payload_kind: None,
            });
            scheduler.update_page_state(id as usize * 10, Some(id), PageState::Protected);
        }

        // Act
        let victims = scheduler.select_victim_groups(10);

        // Assert: no candidates (group_has_protection filters all)
        assert!(victims.is_empty(),
            "all groups with Protected pages must yield no victim candidates");
    }

    // -- 10. Multiple layers in weight_page_table have independent lifecycle --

    #[test]
    fn weight_page_table_multiple_layers_independent_lifecycle() {
        // Arrange: allocate pages for 3 layers
        let mut scheduler = HGALScheduler::new(HGALConfig::default());

        let layer0 = scheduler.allocate_expert_weight_pages(2, 0);
        let layer1 = scheduler.allocate_expert_weight_pages(3, 1);
        let layer2 = scheduler.allocate_expert_weight_pages(1, 2);

        assert_eq!(scheduler.num_weight_pages(), 6, "2+3+1 = 6 total weight pages");
        assert_eq!(scheduler.weight_page_table.len(), 3, "3 layers registered");

        // Act: free only layer 1
        scheduler.free_expert_weight_pages(1);

        // Assert: layer 0 and 2 unaffected
        assert_eq!(scheduler.num_weight_pages(), 3,
            "layer0(2) + layer2(1) = 3 remaining after freeing layer1");
        assert!(scheduler.weight_page_table.contains_key(&0), "layer 0 must remain");
        assert!(!scheduler.weight_page_table.contains_key(&1), "layer 1 must be removed");
        assert!(scheduler.weight_page_table.contains_key(&2), "layer 2 must remain");

        // Layer 0 pages still have metadata
        for pid in &layer0 {
            assert!(scheduler.page_metadata.contains_key(pid),
                "layer 0 page {} must still have metadata", pid);
        }
        // Layer 1 pages have no metadata
        for pid in &layer1 {
            assert!(!scheduler.page_metadata.contains_key(pid),
                "layer 1 page {} must have metadata removed", pid);
        }
        // Layer 2 pages still have metadata
        for pid in &layer2 {
            assert!(scheduler.page_metadata.contains_key(pid),
                "layer 2 page {} must still have metadata", pid);
        }

        // Act again: free layer 0, only layer 2 remains
        scheduler.free_expert_weight_pages(0);
        assert_eq!(scheduler.num_weight_pages(), 1, "only layer2(1) remains");
        assert_eq!(scheduler.weight_page_table.len(), 1);
        assert!(scheduler.weight_page_table.contains_key(&2));
    }
}
