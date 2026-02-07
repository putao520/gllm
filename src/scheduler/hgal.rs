use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use gllm_kernels::kernel_types::{PageId, PageState, RequestId};

#[cfg(test)]
use super::types::GroupState;
use super::types::{PageMetadata, SequenceGroup};

// Priority formula source: SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md §4.1.
// These defaults are tuned to keep recency dominant while still rewarding locality.
const FREQUENCY_WEIGHT: isize = 10;
// Large pin bonus guarantees pinned groups are effectively excluded from victim selection.
const PIN_BONUS: isize = 5_000;

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
}

impl HGALScheduler {
    pub fn new(config: HGALConfig) -> Self {
        Self {
            sequence_groups: HashMap::new(),
            page_metadata: HashMap::new(),
            lir_pages: HashSet::new(),
            config,
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
        let entry = self.page_metadata.entry(page_id).or_insert_with(|| {
            let mut meta = PageMetadata::default();
            meta.page_id = page_id;
            meta
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

        let mut candidates: Vec<(RequestId, usize, isize)> = self
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

        candidates.sort_by(|a, b| a.2.cmp(&b.2));

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

    /// LIRS-style priority computation for a sequence group.
    fn compute_group_priority(&self, group: &SequenceGroup) -> isize {
        let now = Instant::now();
        let time_penalty = now.saturating_duration_since(group.last_access).as_millis() as isize;

        let recency_penalty: isize = group
            .pages
            .iter()
            .filter_map(|pid| self.page_metadata.get(pid))
            .map(|meta| meta.recency as isize)
            .sum();

        let freq_bonus = (group.access_count as isize) * FREQUENCY_WEIGHT;
        let pin_bonus = if group.is_pinned { PIN_BONUS } else { 0 };

        time_penalty + recency_penalty - freq_bonus - pin_bonus
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
            .unwrap_or(false)
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
            .unwrap_or(false)
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
                _ => {}
            }
        }
    }

    /// Record a page access, updating LIRS metadata and group stats.
    pub fn mark_accessed(&mut self, page_id: PageId) {
        let now = Instant::now();
        let warmup_duration = self.config.warmup_duration;
        let min_warm_access = self.config.min_warm_access;
        let sequence_id = {
            let entry = self.page_metadata.entry(page_id).or_insert_with(|| {
                let mut meta = PageMetadata::default();
                meta.page_id = page_id;
                meta
            });
            let irr = now
                .saturating_duration_since(entry.last_access)
                .as_millis()
                .try_into()
                .unwrap_or(usize::MAX);
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
        let entry = self.page_metadata.entry(page_id).or_insert_with(|| {
            let mut meta = PageMetadata::default();
            meta.page_id = page_id;
            meta
        });
        entry.swap_in_time = Some(now);
        entry.warm_until = Some(now + self.config.warmup_duration);
        entry.state = PageState::Warm;
        entry.access_count = 0;
        entry.last_access = now;
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
        };
        scheduler.upsert_group(cold_group);
        scheduler.update_page_state(2, Some(2), PageState::Standby);

        let victims = scheduler.select_victim_groups(1);
        assert_eq!(victims, vec![2]);
    }

    #[test]
    fn working_set_detection_promotes_protected() {
        let mut scheduler = HGALScheduler::new(HGALConfig::default());
        let mut meta = PageMetadata::default();
        meta.page_id = 3;
        meta.sequence_id = Some(3);
        meta.state = PageState::Active;
        meta.access_count = 5;
        meta.last_access = Instant::now();
        scheduler.page_metadata.insert(3, meta);

        scheduler.detect_working_set();
        assert_eq!(
            scheduler.page_metadata.get(&3).map(|m| m.state).unwrap(),
            PageState::Protected
        );
    }
}
