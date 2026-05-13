use super::allocator::BlockAllocator;
use super::hgal::{HGALConfig, HGALScheduler};
use super::memory_manager::{
    EvictionPolicy, GlobalMemoryManager, PrefillPlan, Tier, TierManager,
    VirtualPageId,
};
use super::prefix_index::{KvPrefixIndex, PrefixMatch, TokenId};
use super::types::{GroupState, PageMetadata, SequenceGroup};
use super::vllm2024::{Scheduler2024Config, Scheduler2024State};
use super::types::{PageId, PageState, RequestId, StorageKey};
use crate::kv_cache::LayerDonorInfo;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct BlockTable {
    // Maps logical block index -> physical block ID
    pub blocks: Vec<PageId>,
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockTable {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }
}

pub struct SchedulerOutput {
    pub running: Vec<RequestId>,
    pub swapped_out: Vec<RequestId>,
    pub blocks_to_swap_out: HashMap<RequestId, Vec<PageId>>,
    pub blocks_to_free: Vec<PageId>, // Blocks that are just freed (e.g. finished seq)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchedulerError {
    #[error(
        "out of memory during {operation}: need {needed_blocks} blocks, only {free_blocks} free"
    )]
    OutOfMemory {
        operation: &'static str,
        needed_blocks: usize,
        free_blocks: usize,
    },
    #[error("missing sequence group {request_id} in {context}")]
    MissingGroup {
        request_id: RequestId,
        context: &'static str,
    },
    #[error("allocator invariant violated during {operation}")]
    AllocatorInvariant { operation: &'static str },
    #[error("storage key conversion overflow: {field}")]
    StorageKeyOverflow { field: &'static str },
    #[error(
        "shared-KV donor layer {donor_layer} has no page for request {request_id} \
         (consumer layer {consumer_layer})"
    )]
    MissingDonorPage {
        request_id: RequestId,
        consumer_layer: usize,
        donor_layer: usize,
    },
    #[error(
        "no donor candidate for consumer layer {layer} with attention bucket {bucket}"
    )]
    NoDonorForConsumer { layer: usize, bucket: u8 },
    #[error(
        "attention_pattern length {pattern_len} must equal num_layers {num_layers} \
         when num_kv_shared_layers > 0"
    )]
    AttentionPatternMismatch {
        pattern_len: usize,
        num_layers: usize,
    },
}

pub struct PagedScheduler {
    pub(crate) hgal: HGALScheduler,
    allocator: BlockAllocator,
    // Maps RequestId -> BlockTable
    pub(crate) block_tables: HashMap<RequestId, BlockTable>,
    // Maps RequestId -> stable storage keys ordered by logical block index.
    swapped_storage_keys: HashMap<RequestId, Vec<StorageKey>>,
    // Requests that have newly allocated physical pages and require backend swap-in.
    pending_swap_ins: HashMap<RequestId, Vec<(PageId, StorageKey)>>,
    block_size: usize,
    pub(crate) vllm_state: Option<Scheduler2024State>,
    /// KV cache prefix tree for prompt sharing across requests.
    pub(crate) prefix_index: KvPrefixIndex,
    /// OS-style three-tier KV memory manager (L1 GPU / L2 CPU / L3 NVMe).
    pub(crate) memory_manager: GlobalMemoryManager,
}

impl PagedScheduler {
    pub fn new(total_blocks: usize, block_size: usize, hgal_config: HGALConfig) -> Self {
        // Default: all blocks in L1 (GPU), L2/L3 empty (can be reconfigured via new_with_tiers)
        let memory_manager = GlobalMemoryManager::new_with_capacities(total_blocks, 0, 0);
        Self {
            hgal: HGALScheduler::new(hgal_config),
            allocator: BlockAllocator::new(block_size, total_blocks),
            block_tables: HashMap::new(),
            swapped_storage_keys: HashMap::new(),
            pending_swap_ins: HashMap::new(),
            block_size,
            vllm_state: None,
            prefix_index: KvPrefixIndex::new(),
            memory_manager,
        }
    }

    /// Create with explicit tier capacities (L1=GPU, L2=CPU, L3=NVMe).
    pub fn new_with_tiers(
        total_blocks: usize,
        block_size: usize,
        hgal_config: HGALConfig,
        l1_capacity: usize,
        l2_capacity: usize,
        l3_capacity: usize,
    ) -> Self {
        let memory_manager = GlobalMemoryManager::new(
            TierManager::new(l1_capacity, l2_capacity, l3_capacity),
            EvictionPolicy::default(),
        );
        Self {
            hgal: HGALScheduler::new(hgal_config),
            allocator: BlockAllocator::new(block_size, total_blocks),
            block_tables: HashMap::new(),
            swapped_storage_keys: HashMap::new(),
            pending_swap_ins: HashMap::new(),
            block_size,
            vllm_state: None,
            prefix_index: KvPrefixIndex::new(),
            memory_manager,
        }
    }

    pub fn storage_key(
        request_id: RequestId,
        logical_block_idx: usize,
    ) -> Result<StorageKey, SchedulerError> {
        let req = request_id;
        let block =
            u64::try_from(logical_block_idx).map_err(|_| SchedulerError::StorageKeyOverflow {
                field: "logical_block_idx",
            })?;
        if req > u32::MAX as u64 {
            return Err(SchedulerError::StorageKeyOverflow {
                field: "request_id",
            });
        }
        if block > u32::MAX as u64 {
            return Err(SchedulerError::StorageKeyOverflow {
                field: "logical_block_idx",
            });
        }
        Ok((req << 32) | block)
    }

    pub fn enable_vllm_2024(&mut self, config: Scheduler2024Config) {
        self.vllm_state = Some(Scheduler2024State::new(config));
    }

    pub fn page_size(&self) -> usize {
        self.block_size
    }

    /// Export the page table for a request as a flat u32 array.
    ///
    /// Each entry maps a logical block index to a physical page ID.
    /// The mega-kernel uses this for PagedAttention indirect addressing.
    /// Returns None if the request has no block table (not yet scheduled).
    ///
    /// Safety: validates all page IDs are within [0, total_pages).
    pub fn get_page_table(&self, request_id: RequestId) -> Option<Vec<u32>> {
        self.block_tables.get(&request_id).map(|bt| {
            let total_pages = self.allocator.get_total_blocks();
            bt.blocks.iter().map(|&page_id| {
                debug_assert!(page_id < total_pages, "page_id {} >= total_pages {}", page_id, total_pages);
                page_id as u32
            }).collect()
        })
    }

    /// Returns the total number of physical pages in the pool.
    /// Used for bounds validation of page table entries.
    pub fn total_pages(&self) -> usize {
        self.allocator.get_total_blocks()
    }

    pub fn add_sequence(&mut self, mut group: SequenceGroup) -> Result<(), SchedulerError> {
        // Calculate needed blocks for the context
        let needed_blocks = group.context_len.div_ceil(self.block_size);
        let free_blocks = self.allocator.get_num_free_blocks();

        // Ensure we have enough blocks
        if free_blocks < needed_blocks {
            return Err(SchedulerError::OutOfMemory {
                operation: "add_sequence",
                needed_blocks,
                free_blocks,
            });
        }

        let mut allocated = Vec::new();
        for logical_idx in 0..needed_blocks {
            let block = self
                .allocator
                .allocate()
                .ok_or(SchedulerError::AllocatorInvariant {
                    operation: "add_sequence",
                })?;
            allocated.push(block);
            self.hgal.mark_accessed(block);

            // Register in GMM page table (L1 tier)
            let virtual_id = VirtualPageId::new(group.id, logical_idx);
            if let Err(e) = self.memory_manager.track_page(Tier::L1, block) { log::warn!("GMM track_page failed: {}", e); }
            if let Err(e) = self.memory_manager.bind_virtual_page(virtual_id, Tier::L1, block) { log::warn!("GMM bind_virtual_page failed: {}", e); }
        }

        let mut block_table = BlockTable::new();
        block_table.blocks = allocated.clone();
        self.block_tables.insert(group.id, block_table);

        group.pages = allocated;
        self.hgal.upsert_group(group);

        Ok(())
    }

    /// Add a sequence with prefix reuse: checks the prefix tree first, reuses matched pages.
    /// Returns the number of tokens that were reused from the prefix cache.
    pub fn add_sequence_with_prefix_reuse(
        &mut self,
        mut group: SequenceGroup,
        tokens: &[TokenId],
    ) -> Result<usize, SchedulerError> {
        // Check prefix tree for longest matching prefix
        let prefix_match = self.prefix_index.find_longest_prefix(tokens);
        let reused_tokens = prefix_match.as_ref().map(|m| m.matched_tokens).unwrap_or(0); // LEGAL: 无匹配前缀时 reused_tokens=0
        let reused_blocks = reused_tokens.div_ceil(self.block_size.max(1));

        // Only allocate blocks for the non-reused portion
        let new_blocks_needed = group.context_len.saturating_sub(reused_tokens).div_ceil(self.block_size.max(1));
        let free_blocks = self.allocator.get_num_free_blocks();

        if free_blocks < new_blocks_needed {
            return Err(SchedulerError::OutOfMemory {
                operation: "add_sequence_with_prefix_reuse",
                needed_blocks: new_blocks_needed,
                free_blocks,
            });
        }

        let mut allocated = Vec::new();

        // Reuse prefix blocks as read-only virtual mappings (copy-on-write semantics)
        if let Some(ref prefix) = prefix_match {
            for (logical_idx, &vpid) in prefix.matched_pages.iter().enumerate().take(reused_blocks) {
                // Resolve the physical page from GMM
                if let Ok((tier, physical_id)) = self.memory_manager.resolve(vpid) {
                    let new_vpid = VirtualPageId::new(group.id, logical_idx);
                    if let Err(e) = self.memory_manager.bind_virtual_page(new_vpid, tier, physical_id) { log::warn!("GMM bind_virtual_page failed: {}", e); }
                    allocated.push(physical_id as PageId);
                    self.hgal.mark_accessed(physical_id as PageId);
                }
            }
        }

        // Allocate new blocks for the remaining tokens
        let already_allocated = allocated.len();
        for logical_idx in already_allocated..(already_allocated + new_blocks_needed) {
            let block = self
                .allocator
                .allocate()
                .ok_or(SchedulerError::AllocatorInvariant {
                    operation: "add_sequence_with_prefix_reuse",
                })?;
            allocated.push(block);
            self.hgal.mark_accessed(block);

            let virtual_id = VirtualPageId::new(group.id, logical_idx);
            if let Err(e) = self.memory_manager.track_page(Tier::L1, block) { log::warn!("GMM track_page failed: {}", e); }
            if let Err(e) = self.memory_manager.bind_virtual_page(virtual_id, Tier::L1, block) { log::warn!("GMM bind_virtual_page failed: {}", e); }
        }

        let mut block_table = BlockTable::new();
        block_table.blocks = allocated.clone();
        self.block_tables.insert(group.id, block_table);

        group.pages = allocated;
        self.hgal.upsert_group(group);

        Ok(reused_tokens)
    }

    /// Query the prefix tree for the longest matching prefix of `tokens`.
    /// Returns `Some(PrefixMatch)` if a shared prefix is found.
    pub fn find_prefix(&self, tokens: &[TokenId]) -> Option<PrefixMatch> {
        self.prefix_index.find_longest_prefix(tokens)
    }

    /// Insert a completed prefill's token sequence into the prefix tree,
    /// mapping each token position to its corresponding virtual page.
    pub fn insert_prefix(&mut self, request_id: RequestId, tokens: &[TokenId]) {
        let pages: Vec<VirtualPageId> = tokens
            .iter()
            .enumerate()
            .map(|(i, _)| VirtualPageId::new(request_id, i / self.block_size))
            .collect();
        self.prefix_index.insert(tokens, &pages);
    }

    pub fn allocate_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<PageId>, SchedulerError> {
        let is_swapped = self
            .hgal
            .sequence_groups
            .get(&request_id)
            .ok_or(SchedulerError::MissingGroup {
                request_id,
                context: "hgal.sequence_groups",
            })?
            .state
            == GroupState::Swapped;
        if is_swapped {
            self.restore_swapped_sequence(request_id)?;
        }

        let needs_alloc = {
            let block_table =
                self.block_tables
                    .get(&request_id)
                    .ok_or(SchedulerError::MissingGroup {
                        request_id,
                        context: "block_tables",
                    })?;
            let group = self.hgal.sequence_groups.get_mut(&request_id).ok_or(
                SchedulerError::MissingGroup {
                    request_id,
                    context: "hgal.sequence_groups",
                },
            )?;

            // Always increment context length
            group.context_len += 1;

            let current_len = group.context_len;
            let capacity = block_table.blocks.len() * self.block_size;
            current_len > capacity
        };

        if !needs_alloc {
            return Ok(None);
        }

        // Need new block
        let free_blocks = self.allocator.get_num_free_blocks();
        if free_blocks == 0 {
            return Err(SchedulerError::OutOfMemory {
                operation: "allocate_next_token",
                needed_blocks: 1,
                free_blocks,
            });
        }

        let block = self
            .allocator
            .allocate()
            .ok_or(SchedulerError::AllocatorInvariant {
                operation: "allocate_next_token",
            })?;

        self.block_tables
            .get_mut(&request_id)
            .ok_or(SchedulerError::MissingGroup {
                request_id,
                context: "block_tables",
            })?
            .blocks
            .push(block);
        let logical_idx = self.block_tables[&request_id].blocks.len() - 1;
        self.hgal
            .sequence_groups
            .get_mut(&request_id)
            .ok_or(SchedulerError::MissingGroup {
                request_id,
                context: "hgal.sequence_groups",
            })?
            .pages
            .push(block);
        self.hgal.mark_accessed(block);

        // Register new block in GMM
        let virtual_id = VirtualPageId::new(request_id, logical_idx);
        if let Err(e) = self.memory_manager.track_page(Tier::L1, block) { log::warn!("GMM track_page failed: {}", e); }
        if let Err(e) = self.memory_manager.bind_virtual_page(virtual_id, Tier::L1, block) { log::warn!("GMM bind_virtual_page failed: {}", e); }

        Ok(Some(block))
    }

    /// Select victim requests for eviction, but do not free blocks yet.
    pub fn select_victims(&mut self, needed_blocks: usize) -> Vec<(RequestId, Vec<PageId>)> {
        let victim_ids = self.hgal.select_victim_groups(needed_blocks);
        victim_ids
            .into_iter()
            .filter_map(|request_id| {
                self.block_tables
                    .get(&request_id)
                    .map(|table| (request_id, table.blocks.clone()))
            })
            .filter(|(_, pages)| !pages.is_empty())
            .collect()
    }

    /// Free selected victims after swap-out is complete.
    pub fn free_victims(&mut self, victims: &[RequestId]) -> Result<(), SchedulerError> {
        for &request_id in victims {
            let Some(block_table) = self.block_tables.get_mut(&request_id) else {
                continue;
            };
            let pages = block_table.blocks.clone();
            block_table.blocks.clear();

            let mut storage_keys = Vec::with_capacity(pages.len());
            for (logical_idx, &page_id) in pages.iter().enumerate() {
                let storage_key = Self::storage_key(request_id, logical_idx)?;
                storage_keys.push(storage_key);
                self.allocator.free(page_id);
                self.hgal
                    .update_page_state(page_id, Some(request_id), PageState::Swapped);
            }
            self.swapped_storage_keys.insert(request_id, storage_keys);

            if let Some(group) = self.hgal.sequence_groups.get_mut(&request_id) {
                group.state = GroupState::Swapped;
                group.pages.clear();
            }
        }
        Ok(())
    }

    fn restore_swapped_sequence(&mut self, request_id: RequestId) -> Result<(), SchedulerError> {
        let Some(storage_keys) = self.swapped_storage_keys.get(&request_id).cloned() else {
            return Ok(());
        };

        let free_blocks = self.allocator.get_num_free_blocks();
        let needed_blocks = storage_keys.len();
        if free_blocks < needed_blocks {
            return Err(SchedulerError::OutOfMemory {
                operation: "restore_swapped_sequence",
                needed_blocks,
                free_blocks,
            });
        }

        let mut restored_pages = Vec::with_capacity(storage_keys.len());
        let mut swap_in_mappings = Vec::with_capacity(storage_keys.len());
        for storage_key in storage_keys {
            let physical_id =
                self.allocator
                    .allocate()
                    .ok_or(SchedulerError::AllocatorInvariant {
                        operation: "restore_swapped_sequence",
                    })?;
            restored_pages.push(physical_id);
            swap_in_mappings.push((physical_id, storage_key));
        }

        if let Some(block_table) = self.block_tables.get_mut(&request_id) {
            block_table.blocks = restored_pages.clone();
        } else {
            let mut block_table = BlockTable::new();
            block_table.blocks = restored_pages.clone();
            self.block_tables.insert(request_id, block_table);
        }

        if let Some(group) = self.hgal.sequence_groups.get_mut(&request_id) {
            group.state = GroupState::Running;
            group.pages = restored_pages;
        }

        self.pending_swap_ins.insert(request_id, swap_in_mappings);
        self.swapped_storage_keys.remove(&request_id);
        Ok(())
    }

    pub fn take_pending_swap_in(
        &mut self,
        request_id: RequestId,
    ) -> Option<Vec<(PageId, StorageKey)>> {
        self.pending_swap_ins.remove(&request_id)
    }

    pub fn request_pages(&self, request_id: RequestId) -> Vec<(usize, PageId)> {
        self.block_tables
            .get(&request_id)
            .map(|table| table.blocks.iter().copied().enumerate().collect())
            .unwrap_or_default() // LEGAL: 不存在的 request 返回空 pages 列表
    }

    pub fn free_sequence(&mut self, request_id: RequestId) {
        if let Some(block_table) = self.block_tables.remove(&request_id) {
            for (logical_idx, block) in block_table.blocks.iter().enumerate() {
                self.allocator.free(*block);
                // Unmap from GMM page table
                let virtual_id = VirtualPageId::new(request_id, logical_idx);
                self.memory_manager.unmap_virtual_page(virtual_id);
                if let Err(e) = self.memory_manager.free_page(Tier::L1, *block ) {
                    log::warn!("free_page L1 block {} failed during sequence cleanup: {}", *block, e);
                }
            }
        }
        self.swapped_storage_keys.remove(&request_id);
        self.pending_swap_ins.remove(&request_id);
        self.hgal.remove_group(request_id);
    }

    pub fn on_swap_in(&mut self, _request_id: RequestId, page_indices: &[PageId]) {
        for &page_id in page_indices {
            self.hgal.on_swap_in(page_id);
        }
    }

    pub fn on_page_evicted(&mut self, request_id: RequestId, page_indices: &[PageId]) {
        for &page_id in page_indices {
            self.hgal
                .update_page_state(page_id, Some(request_id), PageState::Swapped);
        }
    }

    pub fn sync_page_states(&mut self, states: &[(PageId, PageState)]) {
        for &(page_id, state) in states {
            // sequence_id is unknown here without reverse mapping, but HGAL might handle None
            self.hgal.update_page_state(page_id, None, state);
        }
    }

    pub fn config(&self) -> &HGALConfig {
        self.hgal.config()
    }

    pub fn kv_fragmentation_ratio(&self) -> f32 {
        let mut allocated_tokens = 0usize;
        let mut used_tokens = 0usize;

        for (request_id, table) in &self.block_tables {
            let capacity = table.blocks.len().saturating_mul(self.block_size);
            allocated_tokens = allocated_tokens.saturating_add(capacity);

            let used = self
                .hgal
                .sequence_groups
                .get(request_id)
                .map(|group| group.context_len.min(capacity))
                .unwrap_or(0); // LEGAL: 不存在的 sequence group 返回 0 used tokens
            used_tokens = used_tokens.saturating_add(used);
        }

        if allocated_tokens == 0 {
            return 0.0;
        }

        ((allocated_tokens.saturating_sub(used_tokens)) as f32 / allocated_tokens as f32)
            .clamp(0.0, 1.0)
    }

    pub fn num_free_blocks(&self) -> usize {
        self.allocator.get_num_free_blocks()
    }

    pub fn num_total_blocks(&self) -> usize {
        self.allocator.get_total_blocks()
    }

    /// Plan prefill page allocation strategy via GlobalMemoryManager.
    /// Returns FullyResident if L1 has capacity, Pipelined otherwise.
    pub fn plan_prefill(&mut self, prompt_tokens: usize, chunk_size: usize) -> PrefillPlan {
        self.memory_manager.plan_prefill(prompt_tokens, chunk_size, self.block_size)
    }

    /// Migrate a physical page from one tier to another, updating the GMM page table.
    /// Returns the new physical page ID in the destination tier.
    pub fn migrate_to_tier(
        &mut self,
        request_id: RequestId,
        logical_idx: usize,
        dst_tier: Tier,
    ) -> Result<usize, SchedulerError> {
        let virtual_id = VirtualPageId::new(request_id, logical_idx);
        let (src_tier, src_id) = self.memory_manager.resolve(virtual_id)
            .map_err(|_| SchedulerError::MissingGroup { request_id, context: "gmm.resolve" })?;
        if src_tier == dst_tier {
            return Ok(src_id);
        }
        self.memory_manager.migrate_page(src_tier, dst_tier, src_id)
            .map_err(|_| SchedulerError::AllocatorInvariant { operation: "migrate_to_tier" })
    }

    /// Select eviction victims using GMM's HGAL-aware eviction policy.
    /// `metadata` and `semantic_priorities` come from the HGAL scheduler.
    pub fn select_victims_gmm(
        &self,
        metadata: &std::collections::HashMap<PageId, PageMetadata>,
        semantic_priorities: &std::collections::HashMap<PageId, i32>,
        count: usize,
    ) -> Vec<PageId> {
        self.memory_manager.select_victims(metadata, semantic_priorities, count)
    }

    /// Expose tier usage for observability.
    pub fn tier_usage(&self, tier: Tier) -> super::memory_manager::TierUsage {
        self.memory_manager.tier_usage(tier)
    }
}

// ===========================================================================
// SharedKvRef — layer-granular page allocation (§P1.1).
//
// Gemma 4 E2B (20 shared) / E4B (18 shared) trailing layers reuse a donor
// layer's KV storage instead of computing their own K/V.
//
// Unlike the request-level BlockTable above, `LayerPageTable` tracks
// `(request, layer)` → page_id so consumer layers can *reference* their
// donor's PageId without a new physical allocation.
//
// Invariants enforced here:
//   1. Shared-ref allocation never calls `BlockAllocator::allocate()`.
//   2. The donor's owned page is protected by `borrower_refcount`; the
//      physical block is not returned to the allocator while any consumer
//      holds a reference.
//   3. All donor lookup paths return a typed `Err` (no silent defaults).
// ===========================================================================

/// Composite key for the per-layer page table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerPageKey {
    pub request: RequestId,
    pub layer: usize,
}

impl LayerPageKey {
    #[inline]
    pub fn new(request: RequestId, layer: usize) -> Self {
        Self { request, layer }
    }
}

/// Allocation intent for a single `(request, layer)` page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAllocHint {
    /// Fresh page — pop from the allocator and mark as owned.
    Owned { attn_bucket: u8 },
    /// Shared-ref page — reuse `donor_layer`'s physical storage.
    SharedRef {
        attn_bucket: u8,
        donor_layer: usize,
    },
}

/// Resolve the donor layer for a shared-KV consumer layer.
///
/// Returns:
/// - `Ok(None)` when `layer_i` is **not** a shared-KV consumer (either
///   `num_kv_shared_layers == 0` or `layer_i < num_layers - num_kv_shared_layers`).
/// - `Ok(Some(donor))` — the donor layer index for this consumer.
/// - `Err(..)` when the attention pattern is malformed or no donor of the
///   matching bucket exists.
///
/// The donor is the **latest** non-shared layer (strictly less than
/// `num_layers - num_kv_shared_layers`) whose `attention_pattern[j]` equals
/// the consumer's bucket.
pub fn find_donor(
    layer_i: usize,
    num_layers: usize,
    num_kv_shared_layers: usize,
    attention_pattern: &[u8],
) -> Result<Option<usize>, SchedulerError> {
    if num_kv_shared_layers == 0 || layer_i >= num_layers {
        return Ok(None);
    }
    if layer_i + num_kv_shared_layers < num_layers {
        return Ok(None);
    }
    if attention_pattern.len() != num_layers {
        return Err(SchedulerError::AttentionPatternMismatch {
            pattern_len: attention_pattern.len(),
            num_layers,
        });
    }
    let first_consumer = num_layers - num_kv_shared_layers;
    let target = attention_pattern[layer_i];
    for candidate in (0..first_consumer).rev() {
        if attention_pattern[candidate] == target {
            return Ok(Some(candidate));
        }
    }
    Err(SchedulerError::NoDonorForConsumer {
        layer: layer_i,
        bucket: target,
    })
}

/// Layer-level page table: maps `(request, layer)` to a `PageId` and records
/// donor / borrower refcount. Instantiate one per sequence and pass it into
/// `PagedScheduler::allocate_layer_page` / `free_layer_page`.
#[derive(Debug, Default)]
pub struct LayerPageTable {
    entries: HashMap<LayerPageKey, LayerPageEntry>,
}

#[derive(Debug, Clone, Copy)]
struct LayerPageEntry {
    page_id: PageId,
    info: LayerDonorInfo,
}

impl LayerPageTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve the physical `PageId` for a layer, transparently following
    /// donor links.
    pub fn resolve(&self, key: LayerPageKey) -> Option<PageId> {
        self.entries.get(&key).map(|e| e.page_id)
    }

    /// Fetch the full `LayerDonorInfo` entry (diagnostics / tests).
    pub fn info(&self, key: LayerPageKey) -> Option<LayerDonorInfo> {
        self.entries.get(&key).map(|e| e.info)
    }

    /// Borrower ref-count for the *owned* entry under `key`. Returns 0 when
    /// the key does not resolve or when the entry is itself a reference.
    pub fn borrower_refcount(&self, key: LayerPageKey) -> u32 {
        match self.entries.get(&key) {
            Some(entry) if entry.info.donor_layer.is_none() => entry.info.borrower_refcount,
            _ => 0,
        }
    }

    fn insert_owned(&mut self, key: LayerPageKey, page_id: PageId, attn_bucket: u8) {
        self.entries.insert(
            key,
            LayerPageEntry {
                page_id,
                info: LayerDonorInfo::owned(key.layer as u16, attn_bucket),
            },
        );
    }

    fn insert_shared_ref(
        &mut self,
        key: LayerPageKey,
        donor_layer: usize,
        attn_bucket: u8,
    ) -> Result<PageId, SchedulerError> {
        let donor_key = LayerPageKey::new(key.request, donor_layer);
        let donor_page = match self.entries.get_mut(&donor_key) {
            Some(entry) if entry.info.donor_layer.is_none() => {
                entry.info.borrower_refcount = entry.info.borrower_refcount.saturating_add(1);
                entry.page_id
            }
            _ => {
                return Err(SchedulerError::MissingDonorPage {
                    request_id: key.request,
                    consumer_layer: key.layer,
                    donor_layer,
                });
            }
        };
        self.entries.insert(
            key,
            LayerPageEntry {
                page_id: donor_page,
                info: LayerDonorInfo::reference(
                    key.layer as u16,
                    attn_bucket,
                    donor_layer as u16,
                ),
            },
        );
        Ok(donor_page)
    }

    /// Remove a layer entry.
    ///
    /// - Shared-reference entries: decrement the donor's `borrower_refcount`
    ///   and return `None` (the physical block stays with its owner).
    /// - Owned entries: return `Some(page_id)` **only** when
    ///   `borrower_refcount == 0`. Otherwise re-insert the entry and return
    ///   `None` (caller must release consumers first).
    fn remove(&mut self, key: LayerPageKey) -> Option<PageId> {
        let entry = self.entries.remove(&key)?;
        match entry.info.donor_layer {
            Some(donor_layer) => {
                let donor_key = LayerPageKey::new(key.request, donor_layer as usize);
                if let Some(donor) = self.entries.get_mut(&donor_key) {
                    donor.info.borrower_refcount =
                        donor.info.borrower_refcount.saturating_sub(1);
                }
                None
            }
            None => {
                if entry.info.borrower_refcount > 0 {
                    // Would create a use-after-free; caller must release
                    // consumers first.
                    self.entries.insert(key, entry);
                    None
                } else {
                    Some(entry.page_id)
                }
            }
        }
    }
}

impl PagedScheduler {
    /// Allocate a page for `layer_i` using the given `hint`.
    ///
    /// - `Owned`: pops a fresh block from the allocator and registers it in
    ///   GMM (L1).
    /// - `SharedRef`: consumes zero physical blocks; records a reference to
    ///   the donor layer's page and increments the donor's borrower refcount.
    pub fn allocate_layer_page(
        &mut self,
        request_id: RequestId,
        layer_i: usize,
        hint: LayerAllocHint,
        layer_pages: &mut LayerPageTable,
    ) -> Result<PageId, SchedulerError> {
        let key = LayerPageKey::new(request_id, layer_i);
        match hint {
            LayerAllocHint::Owned { attn_bucket } => {
                let free_blocks = self.allocator.get_num_free_blocks();
                if free_blocks == 0 {
                    return Err(SchedulerError::OutOfMemory {
                        operation: "allocate_layer_page",
                        needed_blocks: 1,
                        free_blocks,
                    });
                }
                let page_id = self
                    .allocator
                    .allocate()
                    .ok_or(SchedulerError::AllocatorInvariant {
                        operation: "allocate_layer_page",
                    })?;
                self.hgal.mark_accessed(page_id);
                if let Err(e) = self.memory_manager.track_page(Tier::L1, page_id as usize) {
                    log::warn!("GMM track_page failed on layer-page alloc: {}", e);
                }
                layer_pages.insert_owned(key, page_id, attn_bucket);
                Ok(page_id)
            }
            LayerAllocHint::SharedRef {
                attn_bucket,
                donor_layer,
            } => {
                let page_id =
                    layer_pages.insert_shared_ref(key, donor_layer, attn_bucket)?;
                self.hgal.mark_accessed(page_id);
                Ok(page_id)
            }
        }
    }

    /// Release a per-layer page. Returns `true` when the physical block was
    /// returned to the allocator. Shared-ref releases only decrement donor
    /// refcounts and always return `false`.
    pub fn free_layer_page(
        &mut self,
        request_id: RequestId,
        layer_i: usize,
        layer_pages: &mut LayerPageTable,
    ) -> bool {
        let key = LayerPageKey::new(request_id, layer_i);
        match layer_pages.remove(key) {
            Some(page_id) => {
                self.allocator.free(page_id);
                if let Err(e) = self.memory_manager.free_page(Tier::L1, page_id as usize) {
                    log::warn!("GMM free_page failed on layer-page free: {}", e);
                }
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::types::GroupState;
    use std::time::Instant;

    fn make_group(id: RequestId, context_len: usize) -> SequenceGroup {
        SequenceGroup {
            id,
            pages: Vec::new(),
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len,
        }
    }

    #[test]
    fn select_victims_does_not_free_before_free_victims() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler
            .add_sequence(make_group(1, 4))
            .expect("add group 1");
        scheduler
            .add_sequence(make_group(2, 4))
            .expect("add group 2");

        let free_before = scheduler.num_free_blocks();
        let victims = scheduler.select_victims(1);
        assert!(!victims.is_empty());
        assert_eq!(scheduler.num_free_blocks(), free_before);

        let victim_ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
        scheduler
            .free_victims(&victim_ids)
            .expect("free victims should succeed");
        assert!(scheduler.num_free_blocks() > free_before);
    }

    #[test]
    fn swapped_sequence_restore_generates_pending_swap_in() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        let request_id = 42;
        scheduler
            .add_sequence(make_group(request_id, 8))
            .expect("add sequence");

        let victims = scheduler.select_victims(2);
        let victim_ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
        scheduler
            .free_victims(&victim_ids)
            .expect("free victims should succeed");

        let group = scheduler
            .hgal
            .sequence_groups
            .get(&request_id)
            .expect("group exists");
        assert_eq!(group.state, GroupState::Swapped);

        let _ = scheduler
            .allocate_next_token(request_id)
            .expect("restore should happen inside allocate_next_token");
        let mappings = scheduler
            .take_pending_swap_in(request_id)
            .expect("pending swap-in expected");
        assert_eq!(mappings.len(), 2);
    }

    /// GMM 集成：add_sequence 后 GMM 页表应有正确映射
    #[test]
    fn gmm_page_table_tracks_allocated_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");

        // 2 blocks for 8 tokens with block_size=4
        let usage = scheduler.tier_usage(Tier::L1);
        assert_eq!(usage.used, 2);

        // Virtual pages should resolve
        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);
        assert!(scheduler.memory_manager.resolve(v0).is_ok());
        assert!(scheduler.memory_manager.resolve(v1).is_ok());
    }

    /// GMM 集成：free_sequence 后 GMM 页表应清除映射
    #[test]
    fn gmm_page_table_cleared_on_free_sequence() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(2, 4)).expect("add");

        let usage_before = scheduler.tier_usage(Tier::L1);
        assert_eq!(usage_before.used, 1);

        scheduler.free_sequence(2);

        let usage_after = scheduler.tier_usage(Tier::L1);
        assert_eq!(usage_after.used, 0);

        let v0 = VirtualPageId::new(2, 0);
        assert!(scheduler.memory_manager.resolve(v0).is_err());
    }

    /// 前缀树集成：add_sequence_with_prefix_reuse 命中前缀时复用页面
    #[test]
    fn prefix_reuse_reduces_new_allocations() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());

        // 先添加一个请求并记录前缀
        let tokens: Vec<u32> = (0..8).collect();
        scheduler.add_sequence(make_group(1, 8)).expect("add first");
        scheduler.insert_prefix(1, &tokens);

        let free_before = scheduler.num_free_blocks();

        // 第二个请求共享前 8 个 token 的前缀
        let tokens2: Vec<u32> = (0..12).collect();
        let reused = scheduler
            .add_sequence_with_prefix_reuse(make_group(2, 12), &tokens2)
            .expect("add with prefix reuse");

        // 应该复用了 8 个 token
        assert_eq!(reused, 8);

        // 只需分配 1 个新 block（12-8=4 tokens = 1 block），而非 3 个
        let free_after = scheduler.num_free_blocks();
        assert_eq!(free_before - free_after, 1);
    }

    /// plan_prefill 代理：L1 充足时返回 FullyResident
    #[test]
    fn plan_prefill_delegates_to_gmm() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            16, 4, HGALConfig::default(), 16, 8, 0,
        );
        let plan = scheduler.plan_prefill(16, 16);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    /// allocate_next_token 新块注册到 GMM
    #[test]
    fn allocate_next_token_registers_in_gmm() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(3, 4)).expect("add");

        let usage_before = scheduler.tier_usage(Tier::L1);

        // 触发新块分配（第 5 个 token 需要第 2 个 block）
        for _ in 0..4 {
            scheduler.allocate_next_token(3).expect("alloc token");
        }

        let usage_after = scheduler.tier_usage(Tier::L1);
        assert!(usage_after.used > usage_before.used);
    }

    // -----------------------------------------------------------------------
    // SharedKvRef (§P1.1) — donor discovery + layer-page allocation.
    // -----------------------------------------------------------------------

    use super::{find_donor, LayerAllocHint, LayerPageKey, LayerPageTable};

    #[test]
    fn find_donor_returns_latest_same_bucket_non_shared() {
        // 32 layers, pattern with interleaved sliding(0) / global(1).
        let num_layers = 32;
        let num_kv_shared = 20;
        let mut pattern = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            pattern.push(if i % 6 == 5 { 1 } else { 0 });
        }
        let first_consumer = num_layers - num_kv_shared; // 12
        let consumer = 15;
        let donor = find_donor(consumer, num_layers, num_kv_shared, &pattern)
            .expect("lookup ok")
            .expect("donor present");
        assert!(donor < first_consumer, "donor {donor} must be non-shared");
        assert_eq!(
            pattern[donor], pattern[consumer],
            "donor bucket must match consumer bucket"
        );
    }

    #[test]
    fn find_donor_returns_none_for_non_consumer_layer() {
        let pattern = vec![0u8; 32];
        let donor = find_donor(5, 32, 20, &pattern).expect("ok");
        assert!(donor.is_none());
    }

    #[test]
    fn find_donor_returns_none_when_sharing_disabled() {
        let donor = find_donor(7, 12, 0, &[]).expect("ok");
        assert!(donor.is_none());
    }

    #[test]
    fn find_donor_errors_on_pattern_length_mismatch() {
        let err = find_donor(15, 32, 20, &[0u8; 10]).expect_err("must error");
        assert!(matches!(err, SchedulerError::AttentionPatternMismatch { .. }));
    }

    #[test]
    fn find_donor_errors_when_bucket_absent_in_non_shared_prefix() {
        let mut pattern = vec![0u8; 10];
        pattern.extend_from_slice(&[1, 1]); // only 2 consumers, all global, no global donor.
        let err = find_donor(10, 12, 2, &pattern).expect_err("must error");
        assert!(matches!(err, SchedulerError::NoDonorForConsumer { .. }));
    }

    #[test]
    fn shared_ref_page_reuses_donor_page_id_and_bumps_refcount() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let req = 7u64;
        let donor_layer = 3usize;
        let consumer = 9usize;
        let donor_page = scheduler
            .allocate_layer_page(
                req,
                donor_layer,
                LayerAllocHint::Owned { attn_bucket: 0 },
                &mut table,
            )
            .expect("alloc donor");
        let shared_page = scheduler
            .allocate_layer_page(
                req,
                consumer,
                LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer },
                &mut table,
            )
            .expect("alloc shared ref");
        assert_eq!(
            donor_page, shared_page,
            "consumer must inherit donor's PageId"
        );
        assert_eq!(
            table.borrower_refcount(LayerPageKey::new(req, donor_layer)),
            1
        );
        let info = table
            .info(LayerPageKey::new(req, consumer))
            .expect("info present");
        assert!(info.is_shared());
        assert_eq!(info.donor_layer, Some(donor_layer as u16));
    }

    #[test]
    fn shared_ref_page_does_not_consume_free_blocks() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let req = 11u64;
        let free_total = scheduler.num_free_blocks();
        scheduler
            .allocate_layer_page(req, 0, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect("owned");
        assert_eq!(scheduler.num_free_blocks(), free_total - 1);
        for shared_layer in 1..4 {
            scheduler
                .allocate_layer_page(
                    req,
                    shared_layer,
                    LayerAllocHint::SharedRef {
                        attn_bucket: 0,
                        donor_layer: 0,
                    },
                    &mut table,
                )
                .expect("shared");
        }
        assert_eq!(
            scheduler.num_free_blocks(),
            free_total - 1,
            "shared refs must not consume blocks"
        );
        assert_eq!(table.borrower_refcount(LayerPageKey::new(req, 0)), 3);
    }

    #[test]
    fn owned_donor_cannot_be_freed_while_consumers_alive() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let req = 13u64;
        scheduler
            .allocate_layer_page(req, 0, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect("owned");
        scheduler
            .allocate_layer_page(
                req,
                1,
                LayerAllocHint::SharedRef {
                    attn_bucket: 0,
                    donor_layer: 0,
                },
                &mut table,
            )
            .expect("shared");
        let free_before = scheduler.num_free_blocks();
        assert!(
            !scheduler.free_layer_page(req, 0, &mut table),
            "donor must not free while borrower alive"
        );
        assert_eq!(scheduler.num_free_blocks(), free_before);
        // Release consumer first — no physical free.
        assert!(!scheduler.free_layer_page(req, 1, &mut table));
        assert_eq!(table.borrower_refcount(LayerPageKey::new(req, 0)), 0);
        // Now donor can be reclaimed.
        assert!(scheduler.free_layer_page(req, 0, &mut table));
        assert_eq!(scheduler.num_free_blocks(), free_before + 1);
    }

    #[test]
    fn shared_ref_without_donor_errors() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let err = scheduler
            .allocate_layer_page(
                17,
                2,
                LayerAllocHint::SharedRef {
                    attn_bucket: 0,
                    donor_layer: 0,
                },
                &mut table,
            )
            .expect_err("must error on missing donor");
        assert!(matches!(err, SchedulerError::MissingDonorPage { .. }));
    }
}
