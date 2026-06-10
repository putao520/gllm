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

#[derive(Debug, Clone, PartialEq)]
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

    /// SwiftKV 蒸馏：在内存压力下合并相似 KV 页面以释放物理块。
    /// 返回蒸馏后可释放的页面数（0 = 无蒸馏收益）。
    pub fn swiftkv_distill(&mut self, request_id: RequestId) -> usize {
        let Some(ref mut vllm) = self.vllm_state else {
            return 0;
        };
        if !vllm.swift_kv.config.enabled {
            return 0;
        }
        let Some(blocks) = self.block_tables.get(&request_id) else {
            return 0;
        };
        let page_count = blocks.blocks.len();
        if page_count < 2 {
            return 0;
        }
        let summary = vllm.swift_kv.distill_pages(page_count);
        page_count.saturating_sub(summary.distilled_pages)
    }

    pub fn swiftkv_config_enabled(&self) -> bool {
        self.vllm_state
            .as_ref()
            .is_some_and(|v| v.swift_kv.config.enabled)
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

    /// Total blocks in the pool (ENT-PAGED-SCHEDULER total_blocks, REQ-KV-EXT-001).
    pub fn total_blocks(&self) -> usize {
        self.allocator.get_total_blocks()
    }

    /// Used blocks in the pool (ENT-PAGED-SCHEDULER used_blocks, REQ-KV-EXT-001).
    pub fn used_blocks(&self) -> usize {
        self.allocator.get_total_blocks() - self.allocator.get_num_free_blocks()
    }

    /// Number of block tables (one per active request, ENT-PAGED-SCHEDULER num_block_tables).
    pub fn num_block_tables(&self) -> usize {
        self.block_tables.len()
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

            // Register in GMM page table (L1 tier) + pipeline tracking
            let virtual_id = VirtualPageId::new(group.id, logical_idx);
            if let Err(e) = self.memory_manager.track_page(Tier::L1, block) { log::warn!("GMM track_page failed: {}", e); }
            if let Err(e) = self.memory_manager.bind_virtual_page(virtual_id, Tier::L1, block) { log::warn!("GMM bind_virtual_page failed: {}", e); }
            self.memory_manager.track_in_pipeline(group.pipeline, group.id, block);
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
            self.memory_manager.track_in_pipeline(group.pipeline, group.id, block);
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

        let pipeline = self.hgal.sequence_groups.get(&request_id)
            .map(|g| g.pipeline)
            .unwrap_or(crate::scheduler::types::KvPipeline::Conversation);

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

        let virtual_id = VirtualPageId::new(request_id, logical_idx);
        if let Err(e) = self.memory_manager.track_page(Tier::L1, block) { log::warn!("GMM track_page failed: {}", e); }
        if let Err(e) = self.memory_manager.bind_virtual_page(virtual_id, Tier::L1, block) { log::warn!("GMM bind_virtual_page failed: {}", e); }
        self.memory_manager.track_in_pipeline(pipeline, request_id, block);

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

        let pipeline = self.hgal.sequence_groups.get(&request_id)
            .map(|g| g.pipeline)
            .unwrap_or(crate::scheduler::types::KvPipeline::Conversation);

        let mut restored_pages = Vec::with_capacity(storage_keys.len());
        let mut swap_in_mappings = Vec::with_capacity(storage_keys.len());
        for storage_key in storage_keys {
            let physical_id =
                self.allocator
                    .allocate()
                    .ok_or(SchedulerError::AllocatorInvariant {
                        operation: "restore_swapped_sequence",
                    })?;
            self.memory_manager.track_in_pipeline(pipeline, request_id, physical_id);
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

    /// Rollback KV cache pages for rejected MTP candidates (REQ-MTP-004).
    ///
    /// After MTP verify, rejected tokens' KV pages must be freed. This trims
    /// the block table to keep only `keep_count` logical pages (main token +
    /// accepted candidates), and frees the rest.
    pub fn rollback_kv_pages(&mut self, request_id: RequestId, rejected_count: usize) {
        if rejected_count == 0 {
            return;
        }
        if let Some(block_table) = self.block_tables.get_mut(&request_id) {
            let total_blocks = block_table.blocks.len();
            let keep_count = total_blocks.saturating_sub(rejected_count);
            let blocks_to_free: Vec<PageId> = block_table.blocks.drain(keep_count..).collect();
            for (i, page_id) in blocks_to_free.iter().enumerate() {
                self.allocator.free(*page_id);
                let logical_idx = keep_count + i;
                let virtual_id = VirtualPageId::new(request_id, logical_idx);
                self.memory_manager.unmap_virtual_page(virtual_id);
                if let Err(e) = self.memory_manager.free_page(Tier::L1, *page_id) {
                    log::warn!("rollback: free_page L1 block {} failed: {}", page_id, e);
                }
            }
        }
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

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 流转 — SPEC 22-PAGE-COMPRESSION §7.5.4 REQ-COMP-016
    // ─────────────────────────────────────────────────────────────────────────

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
                let page_id = self.allocator.allocate().ok_or({
                    SchedulerError::AllocatorInvariant {
                        operation: "allocate_layer_page",
                    }
                })?;
                self.memory_manager.track_in_pipeline(
                    crate::scheduler::types::KvPipeline::Conversation,
                    request_id,
                    page_id,
                );
                self.hgal.mark_accessed(page_id);
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
                if let Err(e) = self.memory_manager.free_page(Tier::L1, page_id) {
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
    use crate::scheduler::vllm2024::SwiftKVConfig;
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
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
        }
    }

    // ── Original 16 tests (restored from commit 7661401) ────────────────────

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

    #[test]
    fn gmm_page_table_tracks_allocated_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");

        let usage = scheduler.tier_usage(Tier::L1);
        assert_eq!(usage.used, 2);

        let v0 = VirtualPageId::new(1, 0);
        let v1 = VirtualPageId::new(1, 1);
        assert!(scheduler.memory_manager.resolve(v0).is_ok());
        assert!(scheduler.memory_manager.resolve(v1).is_ok());
    }

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

    #[test]
    fn prefix_reuse_reduces_new_allocations() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());

        let tokens: Vec<u32> = (0..8).collect();
        scheduler.add_sequence(make_group(1, 8)).expect("add first");
        scheduler.insert_prefix(1, &tokens);

        let free_before = scheduler.num_free_blocks();

        let tokens2: Vec<u32> = (0..12).collect();
        let reused = scheduler
            .add_sequence_with_prefix_reuse(make_group(2, 12), &tokens2)
            .expect("add with prefix reuse");

        assert_eq!(reused, 8);

        let free_after = scheduler.num_free_blocks();
        assert_eq!(free_before - free_after, 1);
    }

    #[test]
    fn plan_prefill_delegates_to_gmm() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            16, 4, HGALConfig::default(), 16, 8, 0,
        );
        let plan = scheduler.plan_prefill(16, 16);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 4 });
    }

    #[test]
    fn allocate_next_token_registers_in_gmm() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(3, 4)).expect("add");

        let usage_before = scheduler.tier_usage(Tier::L1);

        for _ in 0..4 {
            scheduler.allocate_next_token(3).expect("alloc token");
        }

        let usage_after = scheduler.tier_usage(Tier::L1);
        assert!(usage_after.used > usage_before.used);
    }

    #[test]
    fn find_donor_returns_latest_same_bucket_non_shared() {
        let num_layers = 32;
        let num_kv_shared = 20;
        let mut pattern = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            pattern.push(if i % 6 == 5 { 1 } else { 0 });
        }
        let first_consumer = num_layers - num_kv_shared;
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
        pattern.extend_from_slice(&[1, 1]);
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
        assert_eq!(donor_page, shared_page, "consumer must inherit donor's PageId");
        assert_eq!(table.borrower_refcount(LayerPageKey::new(req, donor_layer)), 1);
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
        assert!(!scheduler.free_layer_page(req, 1, &mut table));
        assert_eq!(table.borrower_refcount(LayerPageKey::new(req, 0)), 0);
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

    // ===========================================================================
    // New tests — comprehensive coverage
    // ===========================================================================

    // ── BlockTable ───────────────────────────────────────────────────────────

    #[test]
    fn block_table_new_is_empty() {
        let bt = BlockTable::new();
        assert!(bt.blocks.is_empty());
    }

    #[test]
    fn block_table_default_equals_new() {
        let bt_new = BlockTable::new();
        let bt_default = BlockTable::default();
        assert_eq!(bt_new.blocks, bt_default.blocks);
    }

    #[test]
    fn block_table_stores_page_ids() {
        let mut bt = BlockTable::new();
        bt.blocks.push(0);
        bt.blocks.push(5);
        bt.blocks.push(3);
        assert_eq!(bt.blocks, vec![0, 5, 3]);
    }

    // ── storage_key ──────────────────────────────────────────────────────────

    #[test]
    fn storage_key_combines_request_and_block_index() {
        let key = PagedScheduler::storage_key(1, 2).expect("valid key");
        assert_eq!(key, (1u64 << 32) | 2u64);
    }

    #[test]
    fn storage_key_zero_values() {
        let key = PagedScheduler::storage_key(0, 0).expect("valid key");
        assert_eq!(key, 0);
    }

    #[test]
    fn storage_key_max_valid_values() {
        let key = PagedScheduler::storage_key(u32::MAX as u64, u32::MAX as usize).expect("valid");
        assert_eq!(key, (u32::MAX as u64) << 32 | (u32::MAX as u64));
    }

    #[test]
    fn storage_key_overflow_request_id() {
        let err = PagedScheduler::storage_key((u32::MAX as u64) + 1, 0).expect_err("overflow");
        assert!(matches!(err, SchedulerError::StorageKeyOverflow { field: "request_id" }));
    }

    #[test]
    fn storage_key_overflow_logical_block_idx() {
        let err = PagedScheduler::storage_key(0, (u32::MAX as usize) + 1).expect_err("overflow");
        assert!(matches!(err, SchedulerError::StorageKeyOverflow { field: "logical_block_idx" }));
    }

    // ── SchedulerError display ───────────────────────────────────────────────

    #[test]
    fn scheduler_error_out_of_memory_display() {
        let err = SchedulerError::OutOfMemory {
            operation: "test_op",
            needed_blocks: 10,
            free_blocks: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("test_op"));
        assert!(msg.contains("10"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn scheduler_error_missing_group_display() {
        let err = SchedulerError::MissingGroup {
            request_id: 42,
            context: "test_context",
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("test_context"));
    }

    #[test]
    fn scheduler_error_allocator_invariant_display() {
        let err = SchedulerError::AllocatorInvariant { operation: "my_op" };
        let msg = err.to_string();
        assert!(msg.contains("my_op"));
    }

    #[test]
    fn scheduler_error_missing_donor_page_display() {
        let err = SchedulerError::MissingDonorPage {
            request_id: 1,
            consumer_layer: 5,
            donor_layer: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("1"));
        assert!(msg.contains("5"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn scheduler_error_no_donor_for_consumer_display() {
        let err = SchedulerError::NoDonorForConsumer { layer: 10, bucket: 1 };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("1"));
    }

    #[test]
    fn scheduler_error_attention_pattern_mismatch_display() {
        let err = SchedulerError::AttentionPatternMismatch {
            pattern_len: 5,
            num_layers: 32,
        };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("32"));
    }

    // ── Empty scheduler basics ───────────────────────────────────────────────

    #[test]
    fn empty_scheduler_reports_correct_counts() {
        let scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        assert_eq!(scheduler.page_size(), 4);
        assert_eq!(scheduler.num_free_blocks(), 16);
        assert_eq!(scheduler.num_total_blocks(), 16);
        assert_eq!(scheduler.total_pages(), 16);
    }

    #[test]
    fn empty_scheduler_zero_fragmentation() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert_eq!(scheduler.kv_fragmentation_ratio(), 0.0);
    }

    #[test]
    fn empty_scheduler_no_pending_swap_in() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(scheduler.take_pending_swap_in(99).is_none());
    }

    #[test]
    fn empty_scheduler_request_pages_returns_empty() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(scheduler.request_pages(42).is_empty());
    }

    #[test]
    fn empty_scheduler_get_page_table_returns_none() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(scheduler.get_page_table(1).is_none());
    }

    #[test]
    fn empty_scheduler_config_accessible() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let config = scheduler.config();
        assert_eq!(config.hot_threshold, HGALConfig::default().hot_threshold);
    }

    #[test]
    fn empty_scheduler_select_victims_returns_empty() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let victims = scheduler.select_victims(1);
        assert!(victims.is_empty());
    }

    #[test]
    fn empty_scheduler_select_victims_gmm_returns_empty() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let meta = HashMap::new();
        let priorities = HashMap::new();
        let victims = scheduler.select_victims_gmm(&meta, &priorities, 5);
        assert!(victims.is_empty());
    }

    // ── add_sequence ─────────────────────────────────────────────────────────

    #[test]
    fn add_sequence_single_block() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 7);
        let pages = scheduler.request_pages(1);
        assert_eq!(pages.len(), 1);
    }

    #[test]
    fn add_sequence_multiple_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 12 tokens / 4 block_size = 3 blocks
        scheduler.add_sequence(make_group(1, 12)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 5);
        let pages = scheduler.request_pages(1);
        assert_eq!(pages.len(), 3);
    }

    #[test]
    fn add_sequence_rounds_up_to_next_block() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 5 tokens / 4 block_size = ceil = 2 blocks
        scheduler.add_sequence(make_group(1, 5)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 6);
        let pages = scheduler.request_pages(1);
        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn add_sequence_out_of_memory() {
        let mut scheduler = PagedScheduler::new(2, 4, HGALConfig::default());
        // Need 2 blocks for 8 tokens, have 2 — fits
        scheduler.add_sequence(make_group(1, 8)).expect("add first");
        // Need 1 block, have 0 — fails
        let err = scheduler.add_sequence(make_group(2, 4)).expect_err("should OOM");
        assert!(matches!(
            err,
            SchedulerError::OutOfMemory {
                operation: "add_sequence",
                needed_blocks: 1,
                free_blocks: 0,
            }
        ));
    }

    #[test]
    fn add_sequence_zero_context_len() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 0 tokens -> 0 blocks needed
        scheduler.add_sequence(make_group(1, 0)).expect("add zero");
        assert_eq!(scheduler.num_free_blocks(), 8);
        let pages = scheduler.request_pages(1);
        assert!(pages.is_empty());
    }

    #[test]
    fn add_two_sequences_independent_block_tables() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add 1");
        scheduler.add_sequence(make_group(2, 8)).expect("add 2");

        let pages_1 = scheduler.request_pages(1);
        let pages_2 = scheduler.request_pages(2);
        assert_eq!(pages_1.len(), 1);
        assert_eq!(pages_2.len(), 2);

        // Different physical page IDs
        assert_ne!(pages_1[0].1, pages_2[0].1);
    }

    // ── get_page_table ───────────────────────────────────────────────────────

    #[test]
    fn get_page_table_returns_u32_page_ids() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        let table = scheduler.get_page_table(1).expect("table exists");
        assert_eq!(table.len(), 2);
        for &pid in &table {
            assert!(pid < scheduler.total_pages() as u32);
        }
    }

    #[test]
    fn get_page_table_unknown_request_returns_none() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(scheduler.get_page_table(999).is_none());
    }

    // ── allocate_next_token ──────────────────────────────────────────────────

    #[test]
    fn allocate_next_token_within_capacity_returns_none() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // context_len=3 -> 1 block, capacity=4
        scheduler.add_sequence(make_group(1, 3)).expect("add");
        // context_len becomes 4, capacity is 4 -> no new block needed
        let result = scheduler.allocate_next_token(1).expect("alloc");
        assert_eq!(result, None);
    }

    #[test]
    fn allocate_next_token_crosses_block_boundary() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        // context_len=4, capacity=4*1=4, increment to 5 -> needs new block
        let new_page = scheduler.allocate_next_token(1).expect("alloc");
        assert!(new_page.is_some());
        let pages = scheduler.request_pages(1);
        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn allocate_next_token_missing_group() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let err = scheduler.allocate_next_token(999).expect_err("missing group");
        assert!(matches!(err, SchedulerError::MissingGroup { request_id: 999, .. }));
    }

    #[test]
    fn allocate_next_token_out_of_memory() {
        let mut scheduler = PagedScheduler::new(1, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        // All 1 block used, no free blocks for the next token
        let err = scheduler.allocate_next_token(1).expect_err("should OOM");
        assert!(matches!(
            err,
            SchedulerError::OutOfMemory {
                operation: "allocate_next_token",
                ..
            }
        ));
    }

    // ── free_sequence ────────────────────────────────────────────────────────

    #[test]
    fn free_sequence_releases_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // 2 blocks
        assert_eq!(scheduler.num_free_blocks(), 6);
        scheduler.free_sequence(1);
        assert_eq!(scheduler.num_free_blocks(), 8);
    }

    #[test]
    fn free_sequence_unknown_request_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.free_sequence(999);
        assert_eq!(scheduler.num_free_blocks(), 8);
    }

    #[test]
    fn free_sequence_cleans_up_all_state() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        scheduler.free_sequence(1);
        assert!(scheduler.request_pages(1).is_empty());
        assert!(scheduler.hgal.sequence_groups.get(&1).is_none());
        assert!(scheduler.take_pending_swap_in(1).is_none());
    }

    #[test]
    fn freed_blocks_can_be_reallocated() {
        let mut scheduler = PagedScheduler::new(2, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // uses 2/2 blocks
        scheduler.free_sequence(1);
        // Now 2 blocks free again
        scheduler.add_sequence(make_group(2, 8)).expect("re-add"); // reuses blocks
        assert_eq!(scheduler.num_free_blocks(), 0);
    }

    // ── rollback_kv_pages ────────────────────────────────────────────────────

    #[test]
    fn rollback_kv_pages_with_zero_rejected_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 12)).expect("add"); // 3 blocks
        let free_before = scheduler.num_free_blocks();
        scheduler.rollback_kv_pages(1, 0);
        assert_eq!(scheduler.num_free_blocks(), free_before);
    }

    #[test]
    fn rollback_kv_pages_frees_trailing_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 12)).expect("add"); // 3 blocks
        let free_before = scheduler.num_free_blocks();
        scheduler.rollback_kv_pages(1, 1);
        assert_eq!(scheduler.num_free_blocks(), free_before + 1);
        let pages = scheduler.request_pages(1);
        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn rollback_kv_pages_unknown_request_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.rollback_kv_pages(999, 2);
        assert_eq!(scheduler.num_free_blocks(), 8);
    }

    // ── kv_fragmentation_ratio ───────────────────────────────────────────────

    #[test]
    fn fragmentation_ratio_single_sequence_fully_used() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // context_len=4, 1 block, capacity=4, used=4 -> frag = 0
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        assert_eq!(scheduler.kv_fragmentation_ratio(), 0.0);
    }

    #[test]
    fn fragmentation_ratio_partial_usage() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // context_len=2, 1 block allocated (block_size=4), capacity=4, used=2 -> frag=0.5
        scheduler.add_sequence(make_group(1, 2)).expect("add");
        let frag = scheduler.kv_fragmentation_ratio();
        assert!((frag - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fragmentation_ratio_clamped_to_zero_one() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert_eq!(scheduler.kv_fragmentation_ratio(), 0.0);
    }

    // ── enable_vllm_2024 / swiftkv ──────────────────────────────────────────

    #[test]
    fn swiftkv_config_disabled_by_default() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(!scheduler.swiftkv_config_enabled());
    }

    #[test]
    fn enable_vllm_2024_with_swiftkv_enabled() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let config = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: Default::default(),
            swift_kv: SwiftKVConfig {
                enabled: true,
                window_size: 4,
                enable_across_kv: false,
                similarity_threshold: 0.9,
                precision_guard: 0.1,
            },
        };
        scheduler.enable_vllm_2024(config);
        assert!(scheduler.swiftkv_config_enabled());
    }

    #[test]
    fn swiftkv_distill_returns_zero_when_not_enabled() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        assert_eq!(scheduler.swiftkv_distill(1), 0);
    }

    #[test]
    fn swiftkv_distill_returns_zero_for_unknown_request() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let config = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: Default::default(),
            swift_kv: SwiftKVConfig {
                enabled: true,
                ..Default::default()
            },
        };
        scheduler.enable_vllm_2024(config);
        assert_eq!(scheduler.swiftkv_distill(999), 0);
    }

    #[test]
    fn swiftkv_distill_returns_zero_for_single_page_sequence() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 3)).expect("add"); // 1 block
        let config = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: Default::default(),
            swift_kv: SwiftKVConfig {
                enabled: true,
                ..Default::default()
            },
        };
        scheduler.enable_vllm_2024(config);
        // Single page (< 2) -> no distillation possible
        assert_eq!(scheduler.swiftkv_distill(1), 0);
    }

    // ── sync_page_states / on_swap_in / on_page_evicted ──────────────────────

    #[test]
    fn sync_page_states_does_not_panic() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        let pages = scheduler.request_pages(1);
        let page_id = pages[0].1;
        scheduler.sync_page_states(&[(page_id, PageState::Swapped)]);
        let meta = scheduler.hgal.page_metadata.get(&page_id).expect("meta exists");
        assert_eq!(meta.state, PageState::Swapped);
    }

    #[test]
    fn on_swap_in_updates_hgal_metadata() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        let pages = scheduler.request_pages(1);
        let page_id = pages[0].1;
        // Should not panic; HGAL records swap-in time
        scheduler.on_swap_in(1, &[page_id]);
    }

    #[test]
    fn on_page_evicted_updates_state() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        let pages = scheduler.request_pages(1);
        let page_id = pages[0].1;
        scheduler.on_page_evicted(1, &[page_id]);
        let meta = scheduler.hgal.page_metadata.get(&page_id).expect("meta");
        assert_eq!(meta.state, PageState::Swapped);
    }

    // ── find_prefix / insert_prefix ─────────────────────────────────────────

    #[test]
    fn find_prefix_empty_tree_returns_none() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        assert!(scheduler.find_prefix(&tokens).is_none());
    }

    #[test]
    fn insert_and_find_prefix_roundtrip() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        let tokens: Vec<TokenId> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        scheduler.insert_prefix(1, &tokens);
        let result = scheduler.find_prefix(&tokens);
        assert!(result.is_some());
        assert_eq!(result.unwrap().matched_tokens, 8);
    }

    #[test]
    fn find_prefix_partial_match() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        let tokens: Vec<TokenId> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        scheduler.insert_prefix(1, &tokens);
        // Query with a longer sequence sharing the same prefix
        let query: Vec<TokenId> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let result = scheduler.find_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().matched_tokens, 8);
    }

    #[test]
    fn find_prefix_no_match() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4];
        scheduler.insert_prefix(1, &tokens);
        let query: Vec<TokenId> = vec![5, 6, 7, 8];
        assert!(scheduler.find_prefix(&query).is_none());
    }

    // ── add_sequence_with_prefix_reuse no match ─────────────────────────────

    #[test]
    fn prefix_reuse_with_no_prefix_reuses_zero() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4];
        // No prefix in tree
        let reused = scheduler
            .add_sequence_with_prefix_reuse(make_group(1, 4), &tokens)
            .expect("add");
        assert_eq!(reused, 0);
        assert_eq!(scheduler.num_free_blocks(), 7); // 1 block allocated
    }

    // ── new_with_tiers ───────────────────────────────────────────────────────

    #[test]
    fn new_with_tiers_sets_capacity() {
        let scheduler = PagedScheduler::new_with_tiers(
            16, 4, HGALConfig::default(), 10, 4, 2,
        );
        let l1 = scheduler.tier_usage(Tier::L1);
        let l2 = scheduler.tier_usage(Tier::L2);
        let l3 = scheduler.tier_usage(Tier::L3);
        assert_eq!(l1.capacity, 10);
        assert_eq!(l2.capacity, 4);
        assert_eq!(l3.capacity, 2);
    }

    // ── migrate_to_tier ──────────────────────────────────────────────────────

    #[test]
    fn migrate_to_tier_same_tier_is_noop() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            8, 4, HGALConfig::default(), 8, 4, 0,
        );
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        // Migrate to same tier (L1 -> L1)
        let result = scheduler.migrate_to_tier(1, 0, Tier::L1).expect("migrate");
        // Should return the original physical page ID
        let pages = scheduler.request_pages(1);
        assert_eq!(result, pages[0].1);
    }

    #[test]
    fn migrate_to_tier_unknown_request_errors() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            8, 4, HGALConfig::default(), 8, 4, 0,
        );
        let err = scheduler.migrate_to_tier(999, 0, Tier::L2).expect_err("missing");
        assert!(matches!(err, SchedulerError::MissingGroup { .. }));
    }

    // ── LayerPageKey ─────────────────────────────────────────────────────────

    #[test]
    fn layer_page_key_new() {
        let key = LayerPageKey::new(42, 7);
        assert_eq!(key.request, 42);
        assert_eq!(key.layer, 7);
    }

    #[test]
    fn layer_page_key_equality() {
        let a = LayerPageKey::new(1, 2);
        let b = LayerPageKey::new(1, 2);
        let c = LayerPageKey::new(1, 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn layer_page_key_hash_consistent() {
        use std::collections::HashSet;
        let key = LayerPageKey::new(5, 10);
        let mut set = HashSet::new();
        set.insert(key);
        assert!(set.contains(&LayerPageKey::new(5, 10)));
        assert!(!set.contains(&LayerPageKey::new(5, 11)));
    }

    // ── LayerPageTable ───────────────────────────────────────────────────────

    #[test]
    fn layer_page_table_new_is_empty() {
        let table = LayerPageTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn layer_page_table_default_is_empty() {
        let table = LayerPageTable::default();
        assert!(table.is_empty());
    }

    #[test]
    fn layer_page_table_resolve_missing_returns_none() {
        let table = LayerPageTable::new();
        assert!(table.resolve(LayerPageKey::new(1, 0)).is_none());
    }

    #[test]
    fn layer_page_table_info_missing_returns_none() {
        let table = LayerPageTable::new();
        assert!(table.info(LayerPageKey::new(1, 0)).is_none());
    }

    #[test]
    fn layer_page_table_borrower_refcount_missing_returns_zero() {
        let table = LayerPageTable::new();
        assert_eq!(table.borrower_refcount(LayerPageKey::new(1, 0)), 0);
    }

    // ── LayerAllocHint variants ──────────────────────────────────────────────

    #[test]
    fn layer_alloc_hint_owned_equality() {
        let a = LayerAllocHint::Owned { attn_bucket: 0 };
        let b = LayerAllocHint::Owned { attn_bucket: 0 };
        let c = LayerAllocHint::Owned { attn_bucket: 1 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn layer_alloc_hint_shared_ref_equality() {
        let a = LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 3 };
        let b = LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 3 };
        let c = LayerAllocHint::SharedRef { attn_bucket: 1, donor_layer: 3 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn layer_alloc_hint_owned_vs_shared_not_equal() {
        let owned = LayerAllocHint::Owned { attn_bucket: 0 };
        let shared = LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 0 };
        assert_ne!(owned, shared);
    }

    // ── allocate_layer_page OOM ──────────────────────────────────────────────

    #[test]
    fn allocate_layer_page_owned_out_of_memory() {
        let mut scheduler = PagedScheduler::new(1, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        // Use up the single block
        scheduler
            .allocate_layer_page(1, 0, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect("first alloc");
        // Second allocation should fail
        let err = scheduler
            .allocate_layer_page(1, 1, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect_err("should OOM");
        assert!(matches!(
            err,
            SchedulerError::OutOfMemory {
                operation: "allocate_layer_page",
                ..
            }
        ));
    }

    // ── free_layer_page on non-existent entry ────────────────────────────────

    #[test]
    fn free_layer_page_nonexistent_returns_false() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        assert!(!scheduler.free_layer_page(99, 0, &mut table));
    }

    // ── find_donor edge cases ────────────────────────────────────────────────

    #[test]
    fn find_donor_layer_at_boundary_is_not_consumer() {
        let pattern = vec![0u8; 32];
        let result = find_donor(11, 32, 20, &pattern).expect("ok");
        assert!(result.is_none());
    }

    #[test]
    fn find_donor_first_consumer_layer() {
        let mut pattern = vec![0u8; 32];
        // Make layer 11 (last non-shared) have bucket 1
        pattern[11] = 1;
        pattern[12] = 1;
        let result = find_donor(12, 32, 20, &pattern).expect("ok").expect("donor");
        assert_eq!(result, 11);
    }

    #[test]
    fn find_donor_layer_beyond_num_layers_returns_none() {
        let pattern = vec![0u8; 10];
        let result = find_donor(20, 10, 0, &pattern).expect("ok");
        assert!(result.is_none());
    }

    // ── SchedulerOutput construction ────────────────────────────────────────

    #[test]
    fn scheduler_output_fields() {
        let output = SchedulerOutput {
            running: vec![1, 2],
            swapped_out: vec![3],
            blocks_to_swap_out: {
                let mut m = HashMap::new();
                m.insert(3, vec![10, 11]);
                m
            },
            blocks_to_free: vec![20, 21],
        };
        assert_eq!(output.running.len(), 2);
        assert_eq!(output.swapped_out.len(), 1);
        assert_eq!(output.blocks_to_swap_out[&3].len(), 2);
        assert_eq!(output.blocks_to_free.len(), 2);
    }

    // ── tier_usage empty scheduler ───────────────────────────────────────────

    #[test]
    fn tier_usage_l1_empty_scheduler() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let usage = scheduler.tier_usage(Tier::L1);
        assert_eq!(usage.used, 0);
        assert_eq!(usage.capacity, 8);
    }

    // ── Multiple sequences lifecycle ─────────────────────────────────────────

    #[test]
    fn full_lifecycle_add_generate_free() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());

        // Add request 1 with 4 tokens -> 1 block
        scheduler.add_sequence(make_group(1, 4)).expect("add 1");
        assert_eq!(scheduler.num_free_blocks(), 15);

        // Add request 2 with 8 tokens -> 2 blocks
        scheduler.add_sequence(make_group(2, 8)).expect("add 2");
        assert_eq!(scheduler.num_free_blocks(), 13);

        // Generate 5 more tokens for request 1 (crosses 1-block boundary)
        for _ in 0..5 {
            scheduler.allocate_next_token(1).expect("alloc token");
        }
        // Request 1 now has 9 tokens: ceil(9/4)=3 blocks
        assert_eq!(scheduler.request_pages(1).len(), 3);

        // Free request 2
        scheduler.free_sequence(2);
        // 2 blocks returned from request 2
        let expected_free = 16 - 3; // 3 blocks still used by request 1
        assert_eq!(scheduler.num_free_blocks(), expected_free);
    }

    // ── select_victims then free_victims restores blocks ─────────────────────

    #[test]
    fn select_and_free_victims_full_cycle() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add 1");
        scheduler.add_sequence(make_group(2, 4)).expect("add 2");
        scheduler.add_sequence(make_group(3, 4)).expect("add 3");

        let free_before = scheduler.num_free_blocks();
        let victims = scheduler.select_victims(2);
        assert_eq!(scheduler.num_free_blocks(), free_before);

        let victim_ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
        assert!(!victim_ids.is_empty());
        scheduler.free_victims(&victim_ids).expect("free");

        let free_after = scheduler.num_free_blocks();
        assert!(free_after > free_before);
    }

    // ── plan_prefill with different parameters ───────────────────────────────

    #[test]
    fn plan_prefill_large_prompt_exceeds_l1() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            4, 4, HGALConfig::default(), 4, 8, 0,
        );
        // 32 tokens / 4 block_size = 8 pages needed, L1 has 4 pages
        let plan = scheduler.plan_prefill(32, 32);
        assert_ne!(plan, PrefillPlan::FullyResident { pages: 8 });
    }

    // ── SwiftKVConfig default ────────────────────────────────────────────────

    #[test]
    fn swiftkv_config_default_disabled() {
        let config = SwiftKVConfig::default();
        assert!(!config.enabled);
    }

    // ── SchedulerError StorageKeyOverflow display ─────────────────────────────

    #[test]
    fn scheduler_error_storage_key_overflow_request_id_display() {
        let err = SchedulerError::StorageKeyOverflow {
            field: "request_id",
        };
        let msg = err.to_string();
        assert!(msg.contains("request_id"));
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn scheduler_error_storage_key_overflow_logical_block_idx_display() {
        let err = SchedulerError::StorageKeyOverflow {
            field: "logical_block_idx",
        };
        let msg = err.to_string();
        assert!(msg.contains("logical_block_idx"));
    }

    // ── SchedulerError PartialEq via thiserror derives ────────────────────────

    #[test]
    fn scheduler_error_equality_out_of_memory() {
        let a = SchedulerError::OutOfMemory {
            operation: "op",
            needed_blocks: 5,
            free_blocks: 2,
        };
        let b = SchedulerError::OutOfMemory {
            operation: "op",
            needed_blocks: 5,
            free_blocks: 2,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_error_equality_missing_group() {
        let a = SchedulerError::MissingGroup {
            request_id: 10,
            context: "ctx",
        };
        let b = SchedulerError::MissingGroup {
            request_id: 10,
            context: "ctx",
        };
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_error_inequality_different_operation() {
        let a = SchedulerError::AllocatorInvariant { operation: "alpha" };
        let b = SchedulerError::AllocatorInvariant { operation: "beta" };
        assert_ne!(a, b);
    }

    #[test]
    fn scheduler_error_inequality_different_variants() {
        let a = SchedulerError::StorageKeyOverflow { field: "x" };
        let b = SchedulerError::AllocatorInvariant { operation: "x" };
        assert_ne!(a, b);
    }

    // ── BlockTable Debug and Clone traits ─────────────────────────────────────

    #[test]
    fn block_table_debug_format() {
        let mut bt = BlockTable::new();
        bt.blocks.push(1);
        bt.blocks.push(2);
        let debug_str = format!("{:?}", bt);
        assert!(debug_str.contains("blocks"));
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
    }

    #[test]
    fn block_table_clone_is_independent() {
        let mut bt = BlockTable::new();
        bt.blocks.push(10);
        bt.blocks.push(20);
        let cloned = bt.clone();
        assert_eq!(cloned.blocks, bt.blocks);
        // Verify independence: drop original, cloned still valid
        drop(bt);
        assert_eq!(cloned.blocks, vec![10, 20]);
    }

    // ── LayerPageKey Copy and Clone traits ─────────────────────────────────────

    #[test]
    fn layer_page_key_copy_independence() {
        let original = LayerPageKey::new(7, 3);
        let copied = original;
        assert_eq!(original, copied);
        // Both are still usable after copy (Copy trait)
        assert_eq!(original.request, 7);
        assert_eq!(copied.layer, 3);
    }

    #[test]
    fn layer_page_key_debug_format() {
        let key = LayerPageKey::new(42, 5);
        let debug = format!("{:?}", key);
        assert!(debug.contains("42") || debug.contains("request"));
    }

    // ── LayerAllocHint Copy and Clone traits ───────────────────────────────────

    #[test]
    fn layer_alloc_hint_clone_owned() {
        let hint = LayerAllocHint::Owned { attn_bucket: 3 };
        let cloned = hint.clone();
        assert_eq!(hint, cloned);
    }

    #[test]
    fn layer_alloc_hint_copy_shared_ref() {
        let hint = LayerAllocHint::SharedRef {
            attn_bucket: 1,
            donor_layer: 4,
        };
        let copied = hint;
        // Both still valid (Copy trait)
        assert_eq!(
            hint,
            LayerAllocHint::SharedRef {
                attn_bucket: 1,
                donor_layer: 4,
            }
        );
        assert_eq!(
            copied,
            LayerAllocHint::SharedRef {
                attn_bucket: 1,
                donor_layer: 4,
            }
        );
    }

    #[test]
    fn layer_alloc_hint_debug_format() {
        let owned = LayerAllocHint::Owned { attn_bucket: 0 };
        let debug = format!("{:?}", owned);
        assert!(debug.contains("Owned") || debug.contains("attn_bucket"));

        let shared = LayerAllocHint::SharedRef {
            attn_bucket: 1,
            donor_layer: 2,
        };
        let debug_shared = format!("{:?}", shared);
        assert!(debug_shared.contains("SharedRef") || debug_shared.contains("donor_layer"));
    }

    // ── SchedulerOutput empty construction ─────────────────────────────────────

    #[test]
    fn scheduler_output_empty_fields() {
        let output = SchedulerOutput {
            running: vec![],
            swapped_out: vec![],
            blocks_to_swap_out: HashMap::new(),
            blocks_to_free: vec![],
        };
        assert!(output.running.is_empty());
        assert!(output.swapped_out.is_empty());
        assert!(output.blocks_to_swap_out.is_empty());
        assert!(output.blocks_to_free.is_empty());
    }

    // ── PagedScheduler::new defaults all blocks to L1 ─────────────────────────

    #[test]
    fn new_scheduler_all_blocks_in_l1() {
        let scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        let l1 = scheduler.tier_usage(Tier::L1);
        let l2 = scheduler.tier_usage(Tier::L2);
        let l3 = scheduler.tier_usage(Tier::L3);
        assert_eq!(l1.capacity, 16);
        assert_eq!(l2.capacity, 0);
        assert_eq!(l3.capacity, 0);
    }

    // ── rollback_kv_pages frees all blocks ─────────────────────────────────────

    #[test]
    fn rollback_kv_pages_frees_all_blocks_when_rejected_equals_total() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // 2 blocks
        assert_eq!(scheduler.num_free_blocks(), 6);
        scheduler.rollback_kv_pages(1, 2);
        assert_eq!(scheduler.num_free_blocks(), 8);
        assert!(scheduler.request_pages(1).is_empty());
    }

    #[test]
    fn rollback_kv_pages_saturates_when_rejected_exceeds_total() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add"); // 1 block
        let free_before = scheduler.num_free_blocks();
        // rejected_count = 5 but only 1 block exists; saturating_sub prevents underflow
        scheduler.rollback_kv_pages(1, 5);
        assert_eq!(scheduler.num_free_blocks(), free_before + 1);
        assert!(scheduler.request_pages(1).is_empty());
    }

    // ── LayerDonorInfo construction and is_shared ──────────────────────────────

    #[test]
    fn layer_donor_info_owned_is_not_shared() {
        let info = LayerDonorInfo::owned(3, 0);
        assert!(!info.is_shared());
        assert_eq!(info.layer, 3);
        assert_eq!(info.attn_bucket, 0);
        assert!(info.donor_layer.is_none());
        assert_eq!(info.borrower_refcount, 0);
    }

    #[test]
    fn layer_donor_info_reference_is_shared() {
        let info = LayerDonorInfo::reference(5, 1, 2);
        assert!(info.is_shared());
        assert_eq!(info.layer, 5);
        assert_eq!(info.attn_bucket, 1);
        assert_eq!(info.donor_layer, Some(2));
        assert_eq!(info.borrower_refcount, 0);
    }

    #[test]
    fn layer_donor_info_equality() {
        let a = LayerDonorInfo::owned(1, 0);
        let b = LayerDonorInfo::owned(1, 0);
        let c = LayerDonorInfo::reference(1, 0, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── storage_key distinct combinations ──────────────────────────────────────

    #[test]
    fn storage_key_different_requests_same_block_are_distinct() {
        let key_a = PagedScheduler::storage_key(1, 0).expect("ok");
        let key_b = PagedScheduler::storage_key(2, 0).expect("ok");
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn storage_key_same_request_different_blocks_are_distinct() {
        let key_a = PagedScheduler::storage_key(1, 0).expect("ok");
        let key_b = PagedScheduler::storage_key(1, 1).expect("ok");
        assert_ne!(key_a, key_b);
    }

    // ── kv_fragmentation_ratio with multiple sequences ─────────────────────────

    #[test]
    fn fragmentation_ratio_multiple_sequences() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        // Seq 1: context_len=4, 1 block -> 0 unused out of 4 capacity
        scheduler.add_sequence(make_group(1, 4)).expect("add 1");
        // Seq 2: context_len=2, 1 block -> 2 unused out of 4 capacity
        scheduler.add_sequence(make_group(2, 2)).expect("add 2");
        // Total: allocated=8 tokens, used=6 tokens, frag = 2/8 = 0.25
        let frag = scheduler.kv_fragmentation_ratio();
        assert!((frag - 0.25).abs() < 1e-6);
    }

    // ── page_size returns block_size ───────────────────────────────────────────

    #[test]
    fn page_size_reflects_block_size() {
        let scheduler = PagedScheduler::new(8, 16, HGALConfig::default());
        assert_eq!(scheduler.page_size(), 16);
    }

    // ── find_donor edge: layer_i == num_layers ─────────────────────────────────

    #[test]
    fn find_donor_layer_equals_num_layers_returns_none() {
        let pattern = vec![0u8; 10];
        let result = find_donor(10, 10, 3, &pattern).expect("ok");
        assert!(result.is_none());
    }

    // ── find_donor: boundary at first consumer layer ───────────────────────────

    #[test]
    fn find_donor_exact_first_consumer_layer() {
        // num_layers=10, num_kv_shared=3 -> first_consumer=7
        // Layer 7 is the first consumer, bucket matches layer 6
        let mut pattern = vec![0u8; 10];
        pattern[6] = 2;
        pattern[7] = 2;
        let result = find_donor(7, 10, 3, &pattern).expect("ok").expect("donor");
        assert_eq!(result, 6);
    }

    // ===========================================================================
    // Additional tests — 45+ new tests
    // ===========================================================================

    // ── SchedulerError: all variant Display messages ─────────────────────────

    #[test]
    fn scheduler_error_out_of_memory_contains_all_fields() {
        let err = SchedulerError::OutOfMemory {
            operation: "prefill",
            needed_blocks: 99,
            free_blocks: 0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("prefill"), "display must contain operation");
        assert!(msg.contains("99"), "display must contain needed_blocks");
        assert!(msg.contains("0"), "display must contain free_blocks");
    }

    #[test]
    fn scheduler_error_missing_group_display_contains_both_fields() {
        let err = SchedulerError::MissingGroup {
            request_id: 1234,
            context: "block_tables",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1234"));
        assert!(msg.contains("block_tables"));
    }

    #[test]
    fn scheduler_error_allocator_invariant_display_detailed() {
        let err = SchedulerError::AllocatorInvariant {
            operation: "restore_swapped",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("restore_swapped"));
        assert!(msg.contains("invariant"));
    }

    #[test]
    fn scheduler_error_storage_key_overflow_display_all() {
        let err = SchedulerError::StorageKeyOverflow {
            field: "logical_block_idx",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("logical_block_idx"));
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn scheduler_error_missing_donor_page_display_all_fields() {
        let err = SchedulerError::MissingDonorPage {
            request_id: 42,
            consumer_layer: 30,
            donor_layer: 10,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("30"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn scheduler_error_no_donor_for_consumer_display_fields() {
        let err = SchedulerError::NoDonorForConsumer {
            layer: 25,
            bucket: 3,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("25"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn scheduler_error_attention_pattern_mismatch_display_detailed() {
        let err = SchedulerError::AttentionPatternMismatch {
            pattern_len: 8,
            num_layers: 32,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("8"));
        assert!(msg.contains("32"));
        assert!(msg.contains("attention_pattern"));
    }

    // ── SchedulerError: PartialEq for all variants ──────────────────────────

    #[test]
    fn scheduler_error_eq_storage_key_overflow() {
        let a = SchedulerError::StorageKeyOverflow { field: "x" };
        let b = SchedulerError::StorageKeyOverflow { field: "x" };
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_error_neq_storage_key_overflow_different_field() {
        let a = SchedulerError::StorageKeyOverflow { field: "a" };
        let b = SchedulerError::StorageKeyOverflow { field: "b" };
        assert_ne!(a, b);
    }

    #[test]
    fn scheduler_error_eq_missing_donor_page() {
        let a = SchedulerError::MissingDonorPage {
            request_id: 1,
            consumer_layer: 2,
            donor_layer: 3,
        };
        let b = SchedulerError::MissingDonorPage {
            request_id: 1,
            consumer_layer: 2,
            donor_layer: 3,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_error_neq_missing_donor_page_different_consumer() {
        let a = SchedulerError::MissingDonorPage {
            request_id: 1,
            consumer_layer: 2,
            donor_layer: 3,
        };
        let b = SchedulerError::MissingDonorPage {
            request_id: 1,
            consumer_layer: 9,
            donor_layer: 3,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn scheduler_error_eq_no_donor_for_consumer() {
        let a = SchedulerError::NoDonorForConsumer { layer: 5, bucket: 1 };
        let b = SchedulerError::NoDonorForConsumer { layer: 5, bucket: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_error_eq_attention_pattern_mismatch() {
        let a = SchedulerError::AttentionPatternMismatch {
            pattern_len: 10,
            num_layers: 32,
        };
        let b = SchedulerError::AttentionPatternMismatch {
            pattern_len: 10,
            num_layers: 32,
        };
        assert_eq!(a, b);
    }

    // ── SchedulerError: Debug trait ─────────────────────────────────────────

    #[test]
    fn scheduler_error_debug_format_out_of_memory() {
        let err = SchedulerError::OutOfMemory {
            operation: "test",
            needed_blocks: 1,
            free_blocks: 0,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("OutOfMemory"));
    }

    #[test]
    fn scheduler_error_debug_format_all_variants() {
        let variants: Vec<String> = vec![
            format!("{:?}", SchedulerError::OutOfMemory { operation: "x", needed_blocks: 0, free_blocks: 0 }),
            format!("{:?}", SchedulerError::MissingGroup { request_id: 0, context: "c" }),
            format!("{:?}", SchedulerError::AllocatorInvariant { operation: "o" }),
            format!("{:?}", SchedulerError::StorageKeyOverflow { field: "f" }),
            format!("{:?}", SchedulerError::MissingDonorPage { request_id: 0, consumer_layer: 0, donor_layer: 0 }),
            format!("{:?}", SchedulerError::NoDonorForConsumer { layer: 0, bucket: 0 }),
            format!("{:?}", SchedulerError::AttentionPatternMismatch { pattern_len: 0, num_layers: 0 }),
        ];
        // Every variant produces a non-empty debug string
        for s in &variants {
            assert!(!s.is_empty());
        }
    }

    // ── SchedulerError: Clone trait ─────────────────────────────────────────

    #[test]
    fn scheduler_error_clone_preserves_equality() {
        let original = SchedulerError::OutOfMemory {
            operation: "clone_test",
            needed_blocks: 42,
            free_blocks: 7,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn scheduler_error_clone_missing_group() {
        let original = SchedulerError::MissingGroup {
            request_id: 99,
            context: "ctx",
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── storage_key edge cases ──────────────────────────────────────────────

    #[test]
    fn storage_key_request_one_block_zero() {
        let key = PagedScheduler::storage_key(1, 0).expect("ok");
        assert_eq!(key, 1u64 << 32);
    }

    #[test]
    fn storage_key_request_zero_block_one() {
        let key = PagedScheduler::storage_key(0, 1).expect("ok");
        assert_eq!(key, 1u64);
    }

    #[test]
    fn storage_key_u32_max_request_u32_max_block() {
        let key = PagedScheduler::storage_key(u32::MAX as u64, u32::MAX as usize).expect("ok");
        let expected = ((u32::MAX as u64) << 32) | (u32::MAX as u64);
        assert_eq!(key, expected);
    }

    #[test]
    fn storage_key_overflow_request_id_exactly_one_over() {
        let err = PagedScheduler::storage_key((u32::MAX as u64) + 1, 0).expect_err("overflow");
        assert!(matches!(err, SchedulerError::StorageKeyOverflow { field: "request_id" }));
    }

    #[test]
    fn storage_key_overflow_block_idx_exactly_one_over() {
        let block_idx = (u32::MAX as usize) + 1;
        let err = PagedScheduler::storage_key(0, block_idx).expect_err("overflow");
        assert!(matches!(err, SchedulerError::StorageKeyOverflow { field: "logical_block_idx" }));
    }

    // ── BlockTable edge cases ───────────────────────────────────────────────

    #[test]
    fn block_table_default_trait() {
        let bt = BlockTable::default();
        assert!(bt.blocks.is_empty());
    }

    #[test]
    fn block_table_with_large_page_ids() {
        let mut bt = BlockTable::new();
        bt.blocks.push(usize::MAX);
        bt.blocks.push(0);
        bt.blocks.push(1);
        assert_eq!(bt.blocks[0], usize::MAX);
        assert_eq!(bt.blocks[1], 0);
        assert_eq!(bt.blocks[2], 1);
    }

    #[test]
    fn block_table_clone_produces_equal_copy() {
        let mut bt = BlockTable::new();
        bt.blocks.extend_from_slice(&[10, 20, 30]);
        let cloned = bt.clone();
        assert_eq!(bt, cloned);
    }

    #[test]
    fn block_table_debug_output_contains_blocks_field() {
        let mut bt = BlockTable::new();
        bt.blocks.push(42);
        let debug = format!("{:?}", bt);
        assert!(debug.contains("blocks"));
        assert!(debug.contains("42"));
    }

    // ── LayerPageKey edge cases ─────────────────────────────────────────────

    #[test]
    fn layer_page_key_zero_fields() {
        let key = LayerPageKey::new(0, 0);
        assert_eq!(key.request, 0);
        assert_eq!(key.layer, 0);
    }

    #[test]
    fn layer_page_key_max_values() {
        let key = LayerPageKey::new(u64::MAX, usize::MAX);
        assert_eq!(key.request, u64::MAX);
        assert_eq!(key.layer, usize::MAX);
    }

    #[test]
    fn layer_page_key_copy_trait_works() {
        let a = LayerPageKey::new(100, 200);
        let b = a; // Copy
        assert_eq!(a.request, 100); // Still usable
        assert_eq!(b.layer, 200);
    }

    #[test]
    fn layer_page_key_hash_in_hashmap() {
        let mut map = HashMap::new();
        let key = LayerPageKey::new(5, 10);
        map.insert(key, 42u32);
        assert_eq!(map.get(&LayerPageKey::new(5, 10)), Some(&42));
        assert_eq!(map.get(&LayerPageKey::new(5, 11)), None);
        assert_eq!(map.get(&LayerPageKey::new(6, 10)), None);
    }

    // ── LayerAllocHint edge cases ───────────────────────────────────────────

    #[test]
    fn layer_alloc_hint_owned_zero_bucket() {
        let hint = LayerAllocHint::Owned { attn_bucket: 0 };
        assert_eq!(hint, LayerAllocHint::Owned { attn_bucket: 0 });
    }

    #[test]
    fn layer_alloc_hint_owned_max_bucket() {
        let hint = LayerAllocHint::Owned { attn_bucket: u8::MAX };
        assert_eq!(hint, LayerAllocHint::Owned { attn_bucket: 255 });
    }

    #[test]
    fn layer_alloc_hint_shared_ref_max_values() {
        let hint = LayerAllocHint::SharedRef {
            attn_bucket: u8::MAX,
            donor_layer: usize::MAX,
        };
        let other = LayerAllocHint::SharedRef {
            attn_bucket: u8::MAX,
            donor_layer: usize::MAX,
        };
        assert_eq!(hint, other);
    }

    #[test]
    fn layer_alloc_hint_shared_ref_different_donor_not_equal() {
        let a = LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 1 };
        let b = LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 2 };
        assert_ne!(a, b);
    }

    #[test]
    fn layer_alloc_hint_clone_and_copy() {
        let hint = LayerAllocHint::SharedRef { attn_bucket: 5, donor_layer: 10 };
        let cloned = hint.clone();
        assert_eq!(hint, cloned);
        let copied = hint; // Copy
        assert_eq!(hint, copied);
    }

    // ── LayerDonorInfo edge cases ───────────────────────────────────────────

    #[test]
    fn layer_donor_info_owned_zero_layer() {
        let info = LayerDonorInfo::owned(0, 0);
        assert_eq!(info.layer, 0);
        assert_eq!(info.attn_bucket, 0);
        assert!(info.donor_layer.is_none());
        assert_eq!(info.borrower_refcount, 0);
        assert!(!info.is_shared());
    }

    #[test]
    fn layer_donor_info_owned_max_values() {
        let info = LayerDonorInfo::owned(u16::MAX, u8::MAX);
        assert_eq!(info.layer, u16::MAX);
        assert_eq!(info.attn_bucket, u8::MAX);
        assert!(!info.is_shared());
    }

    #[test]
    fn layer_donor_info_reference_zero_values() {
        let info = LayerDonorInfo::reference(0, 0, 0);
        assert_eq!(info.layer, 0);
        assert_eq!(info.attn_bucket, 0);
        assert_eq!(info.donor_layer, Some(0));
        assert!(info.is_shared());
    }

    #[test]
    fn layer_donor_info_reference_max_values() {
        let info = LayerDonorInfo::reference(u16::MAX, u8::MAX, u16::MAX);
        assert!(info.is_shared());
        assert_eq!(info.donor_layer, Some(u16::MAX));
    }

    #[test]
    fn layer_donor_info_equality_owned_same() {
        let a = LayerDonorInfo::owned(10, 3);
        let b = LayerDonorInfo::owned(10, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn layer_donor_info_inequality_different_bucket() {
        let a = LayerDonorInfo::owned(10, 1);
        let b = LayerDonorInfo::owned(10, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn layer_donor_info_reference_equality() {
        let a = LayerDonorInfo::reference(5, 1, 3);
        let b = LayerDonorInfo::reference(5, 1, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn layer_donor_info_copy_trait() {
        let info = LayerDonorInfo::owned(7, 2);
        let copied = info; // Copy
        assert_eq!(info, copied);
        assert_eq!(info.layer, 7);
    }

    #[test]
    fn layer_donor_info_clone_trait() {
        let info = LayerDonorInfo::reference(3, 0, 1);
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    // ── LayerPageTable edge cases ───────────────────────────────────────────

    #[test]
    fn layer_page_table_borrower_refcount_for_shared_ref_entry_is_zero() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(1, 0);
        table.entries.insert(
            key,
            LayerPageEntry {
                page_id: 0,
                info: LayerDonorInfo::reference(0, 0, 5),
            },
        );
        // Shared-ref entries have borrower_refcount == 0 from the table's perspective
        assert_eq!(table.borrower_refcount(key), 0);
    }

    #[test]
    fn layer_page_table_insert_owned_and_resolve() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(1, 0);
        table.insert_owned(key, 42, 0);
        assert_eq!(table.resolve(key), Some(42));
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
    }

    #[test]
    fn layer_page_table_insert_shared_ref_returns_donor_page() {
        let mut table = LayerPageTable::new();
        let donor_key = LayerPageKey::new(1, 0);
        let consumer_key = LayerPageKey::new(1, 5);
        table.insert_owned(donor_key, 99, 0);
        let page = table.insert_shared_ref(consumer_key, 0, 0).expect("ref ok");
        assert_eq!(page, 99);
        assert_eq!(table.resolve(consumer_key), Some(99));
    }

    #[test]
    fn layer_page_table_remove_owned_returns_page_id() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(1, 0);
        table.insert_owned(key, 77, 0);
        let page = table.remove(key);
        assert_eq!(page, Some(77));
        assert!(table.is_empty());
    }

    #[test]
    fn layer_page_table_remove_shared_ref_decrements_donor_refcount() {
        let mut table = LayerPageTable::new();
        let donor_key = LayerPageKey::new(1, 0);
        let consumer_key = LayerPageKey::new(1, 5);
        table.insert_owned(donor_key, 10, 0);
        table.insert_shared_ref(consumer_key, 0, 0).expect("ref ok");
        assert_eq!(table.borrower_refcount(donor_key), 1);
        let result = table.remove(consumer_key);
        assert!(result.is_none()); // Shared ref removal returns None
        assert_eq!(table.borrower_refcount(donor_key), 0);
    }

    #[test]
    fn layer_page_table_remove_owned_with_borrowers_keeps_entry() {
        let mut table = LayerPageTable::new();
        let donor_key = LayerPageKey::new(1, 0);
        let consumer_key = LayerPageKey::new(1, 5);
        table.insert_owned(donor_key, 10, 0);
        table.insert_shared_ref(consumer_key, 0, 0).expect("ref ok");
        // Try to remove the donor while a borrower exists
        let result = table.remove(donor_key);
        assert!(result.is_none()); // Cannot free donor with active borrowers
        assert_eq!(table.len(), 2); // Both entries still present
    }

    #[test]
    fn layer_page_table_remove_missing_key_returns_none() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(99, 99);
        assert!(table.remove(key).is_none());
    }

    // ── find_donor: comprehensive edge cases ────────────────────────────────

    #[test]
    fn find_donor_returns_err_when_num_kv_shared_equals_num_layers() {
        // All layers are shared consumers -> no non-shared prefix exists
        let pattern = vec![0u8; 5];
        let result = find_donor(2, 5, 5, &pattern);
        // first_consumer = 5-5=0. Loop (0..0).rev() is empty -> Err(NoDonorForConsumer)
        assert!(matches!(result, Err(SchedulerError::NoDonorForConsumer { .. })));
    }

    #[test]
    fn find_donor_with_single_shared_layer() {
        // num_layers=3, num_kv_shared=1 -> first_consumer=2
        let pattern = vec![0u8; 3];
        let result = find_donor(2, 3, 1, &pattern).expect("ok").expect("donor");
        assert_eq!(result, 1); // Latest non-shared layer with bucket 0 is layer 1
    }

    #[test]
    fn find_donor_prefers_latest_donor() {
        // 10 layers, 3 shared -> first_consumer=7
        // Layers 0-6 are non-shared, layers 7-9 are consumers
        // Consumer layer 9, bucket=1, non-shared layers with bucket 1: layers 3, 5
        // Should return 5 (latest)
        let mut pattern = vec![0u8; 10];
        pattern[3] = 1;
        pattern[5] = 1;
        pattern[9] = 1;
        let result = find_donor(9, 10, 3, &pattern).expect("ok").expect("donor");
        assert_eq!(result, 5);
    }

    #[test]
    fn find_donor_layer_just_before_first_consumer_is_not_consumer() {
        // num_layers=10, num_kv_shared=3 -> first_consumer=7
        // layer 6 is NOT a consumer (6 + 3 = 9 < 10)
        let pattern = vec![0u8; 10];
        let result = find_donor(6, 10, 3, &pattern).expect("ok");
        assert!(result.is_none());
    }

    // ── PagedScheduler page_size ────────────────────────────────────────────

    #[test]
    fn page_size_various_block_sizes() {
        let s1 = PagedScheduler::new(8, 1, HGALConfig::default());
        assert_eq!(s1.page_size(), 1);
        let s2 = PagedScheduler::new(8, 16, HGALConfig::default());
        assert_eq!(s2.page_size(), 16);
        let s3 = PagedScheduler::new(8, 256, HGALConfig::default());
        assert_eq!(s3.page_size(), 256);
    }

    // ── kv_fragmentation_ratio edge cases ───────────────────────────────────

    #[test]
    fn fragmentation_ratio_after_full_free_is_zero() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        scheduler.free_sequence(1);
        // No sequences remain -> fragmentation 0
        assert_eq!(scheduler.kv_fragmentation_ratio(), 0.0);
    }

    #[test]
    fn fragmentation_ratio_single_token_one_block() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 1 token -> 1 block allocated, capacity=4, used=1 -> frag=3/4=0.75
        scheduler.add_sequence(make_group(1, 1)).expect("add");
        let frag = scheduler.kv_fragmentation_ratio();
        assert!((frag - 0.75).abs() < 1e-6);
    }

    #[test]
    fn fragmentation_ratio_exact_block_alignment() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 8 tokens -> 2 blocks, capacity=8, used=8 -> frag=0
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        assert_eq!(scheduler.kv_fragmentation_ratio(), 0.0);
    }

    // ── SchedulerOutput with real data ──────────────────────────────────────

    #[test]
    fn scheduler_output_swap_out_entries() {
        let mut blocks_map = HashMap::new();
        blocks_map.insert(1u64, vec![100, 101]);
        blocks_map.insert(2u64, vec![200]);
        let output = SchedulerOutput {
            running: vec![1, 2],
            swapped_out: vec![3],
            blocks_to_swap_out: blocks_map,
            blocks_to_free: vec![50],
        };
        assert_eq!(output.blocks_to_swap_out.len(), 2);
        assert_eq!(output.blocks_to_swap_out[&1], vec![100, 101]);
        assert_eq!(output.blocks_to_swap_out[&2], vec![200]);
    }

    // ── request_pages for non-existent and freed sequences ──────────────────

    #[test]
    fn request_pages_after_free_returns_empty() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        scheduler.free_sequence(1);
        assert!(scheduler.request_pages(1).is_empty());
    }

    #[test]
    fn request_pages_nonexistent_returns_empty() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert!(scheduler.request_pages(0).is_empty());
        assert!(scheduler.request_pages(u64::MAX).is_empty());
    }

    // ── allocate_next_token multiple tokens sequentially ────────────────────

    #[test]
    fn allocate_next_token_multiple_until_oom() {
        let mut scheduler = PagedScheduler::new(3, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add"); // 1 block, context_len=4, capacity=4
        // Token 5 -> context_len=5 > capacity=4, allocates block 2. capacity=8
        let r1 = scheduler.allocate_next_token(1).expect("alloc token 5");
        assert!(r1.is_some());
        // Tokens 6-8 -> context_len 6,7,8 <= capacity=8, no alloc needed
        for _ in 0..3 {
            let r = scheduler.allocate_next_token(1).expect("within capacity");
            assert!(r.is_none());
        }
        // Token 9 -> context_len=9 > capacity=8, allocates block 3. capacity=12, all blocks used
        let r2 = scheduler.allocate_next_token(1).expect("alloc block 3");
        assert!(r2.is_some());
        assert_eq!(scheduler.request_pages(1).len(), 3); // all 3 blocks used
        assert_eq!(scheduler.num_free_blocks(), 0);
        // Tokens 10-12 -> context_len 10,11,12 <= capacity=12, no alloc needed
        for _ in 0..3 {
            let r = scheduler.allocate_next_token(1).expect("within capacity 2");
            assert!(r.is_none());
        }
        // Token 13 -> context_len=13 > capacity=12, needs new block, 0 free -> OOM
        let err = scheduler.allocate_next_token(1).expect_err("OOM");
        assert!(matches!(err, SchedulerError::OutOfMemory { .. }));
    }

    // ── add_sequence context_len exactly at block boundary ──────────────────

    #[test]
    fn add_sequence_context_len_exact_multiple_of_block_size() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 8 tokens / block_size=4 = exactly 2 blocks
        scheduler.add_sequence(make_group(1, 8)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 6);
        assert_eq!(scheduler.request_pages(1).len(), 2);
    }

    #[test]
    fn add_sequence_context_len_one_over_block_boundary() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 9 tokens / block_size=4 = ceil = 3 blocks
        scheduler.add_sequence(make_group(1, 9)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 5);
        assert_eq!(scheduler.request_pages(1).len(), 3);
    }

    // ── rollback_kv_pages edge cases ────────────────────────────────────────

    #[test]
    fn rollback_kv_pages_single_block_from_two() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // 2 blocks
        assert_eq!(scheduler.num_free_blocks(), 6);
        scheduler.rollback_kv_pages(1, 1);
        assert_eq!(scheduler.num_free_blocks(), 7);
        assert_eq!(scheduler.request_pages(1).len(), 1);
    }

    // ── new_with_tiers capacity validation ──────────────────────────────────

    #[test]
    fn new_with_tiers_zero_tiers_beyond_l1() {
        let scheduler = PagedScheduler::new_with_tiers(
            8, 4, HGALConfig::default(), 8, 0, 0,
        );
        let l2 = scheduler.tier_usage(Tier::L2);
        let l3 = scheduler.tier_usage(Tier::L3);
        assert_eq!(l2.capacity, 0);
        assert_eq!(l3.capacity, 0);
    }

    #[test]
    fn new_with_tiers_all_tiers_have_capacity() {
        let scheduler = PagedScheduler::new_with_tiers(
            32, 4, HGALConfig::default(), 16, 8, 8,
        );
        assert_eq!(scheduler.tier_usage(Tier::L1).capacity, 16);
        assert_eq!(scheduler.tier_usage(Tier::L2).capacity, 8);
        assert_eq!(scheduler.tier_usage(Tier::L3).capacity, 8);
    }

    // ── migrate_to_tier same tier returns same physical id ──────────────────

    #[test]
    fn migrate_to_tier_unknown_virtual_page_errors() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            8, 4, HGALConfig::default(), 8, 4, 0,
        );
        // No sequence added, so virtual page (999,0) does not exist
        let err = scheduler.migrate_to_tier(999, 0, Tier::L1).expect_err("missing");
        assert!(matches!(err, SchedulerError::MissingGroup { .. }));
    }

    // ── on_page_evicted and on_swap_in no-op for unknown pages ──────────────

    #[test]
    fn on_swap_in_unknown_page_does_not_panic() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // Should not panic on unknown page IDs
        scheduler.on_swap_in(1, &[9999]);
    }

    #[test]
    fn on_page_evicted_unknown_page_does_not_panic() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.on_page_evicted(1, &[8888]);
    }

    // ── swiftkv distill lifecycle ───────────────────────────────────────────

    #[test]
    fn swiftkv_distill_enabled_with_multiple_pages_returns_nonzero() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        let config = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: Default::default(),
            swift_kv: SwiftKVConfig {
                enabled: true,
                window_size: 4,
                enable_across_kv: false,
                similarity_threshold: 0.9,
                precision_guard: 0.1,
            },
        };
        scheduler.enable_vllm_2024(config);
        scheduler.add_sequence(make_group(1, 16)).expect("add"); // 4 blocks
        let distilled = scheduler.swiftkv_distill(1);
        // distill_pages returns a result based on window_size
        // The exact value depends on SwiftKV internals, but it should not panic
        assert!(distilled <= 4);
    }

    // ── get_page_table validation after allocation ──────────────────────────

    #[test]
    fn get_page_table_all_ids_within_bounds() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 16)).expect("add"); // 4 blocks
        let table = scheduler.get_page_table(1).expect("table");
        assert_eq!(table.len(), 4);
        for &pid in &table {
            assert!(pid < 16, "page_id {} must be < total_pages 16", pid);
        }
    }

    // ── shared_ref multiple consumers increment refcount correctly ──────────

    #[test]
    fn shared_ref_multiple_consumers_refcount_tracks_all() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let req = 1u64;
        scheduler
            .allocate_layer_page(req, 0, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect("owned");
        // Add 3 consumers referencing the same donor
        for layer in 1..=3 {
            scheduler
                .allocate_layer_page(
                    req, layer,
                    LayerAllocHint::SharedRef { attn_bucket: 0, donor_layer: 0 },
                    &mut table,
                )
                .expect("shared");
        }
        assert_eq!(table.borrower_refcount(LayerPageKey::new(req, 0)), 3);
        assert_eq!(table.len(), 4);
    }

    // ── free_layer_page owned with no borrowers returns true ────────────────

    #[test]
    fn free_layer_page_owned_no_borrowers_frees_block() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        let mut table = LayerPageTable::new();
        let free_before = scheduler.num_free_blocks();
        scheduler
            .allocate_layer_page(1, 0, LayerAllocHint::Owned { attn_bucket: 0 }, &mut table)
            .expect("alloc");
        assert_eq!(scheduler.num_free_blocks(), free_before - 1);
        let freed = scheduler.free_layer_page(1, 0, &mut table);
        assert!(freed);
        assert_eq!(scheduler.num_free_blocks(), free_before);
        assert!(table.is_empty());
    }

    // ── LayerPageTable info returns correct donor info ──────────────────────

    #[test]
    fn layer_page_table_info_returns_owned_details() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(5, 3);
        table.insert_owned(key, 42, 2);
        let info = table.info(key).expect("info");
        assert!(!info.is_shared());
        assert_eq!(info.layer, 3);
        assert_eq!(info.attn_bucket, 2);
        assert!(info.donor_layer.is_none());
    }

    #[test]
    fn layer_page_table_info_returns_shared_details() {
        let mut table = LayerPageTable::new();
        let donor_key = LayerPageKey::new(5, 0);
        let consumer_key = LayerPageKey::new(5, 10);
        table.insert_owned(donor_key, 42, 1);
        // donor_layer=0 references the owned entry at layer 0
        table.insert_shared_ref(consumer_key, 0, 0).expect("ref");
        let info = table.info(consumer_key).expect("info");
        assert!(info.is_shared());
        assert_eq!(info.donor_layer, Some(0));
    }

    // ── LayerPageTable resolve after remove ─────────────────────────────────

    #[test]
    fn layer_page_table_resolve_after_remove_is_none() {
        let mut table = LayerPageTable::new();
        let key = LayerPageKey::new(1, 0);
        table.insert_owned(key, 10, 0);
        assert_eq!(table.resolve(key), Some(10));
        table.remove(key);
        assert!(table.resolve(key).is_none());
    }

    // ── free_victims on non-existent request is_noop ────────────────────────

    #[test]
    fn free_victims_nonexistent_request_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let free_before = scheduler.num_free_blocks();
        scheduler.free_victims(&[999]).expect("free");
        assert_eq!(scheduler.num_free_blocks(), free_before);
    }

    // ── prefix: insert empty tokens is_safe ─────────────────────────────────

    #[test]
    fn insert_prefix_empty_tokens_is_safe() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let tokens: Vec<TokenId> = vec![];
        scheduler.insert_prefix(1, &tokens);
        // Should not panic, prefix tree accepts empty input
    }

    #[test]
    fn find_prefix_empty_query_is_safe() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let tokens: Vec<TokenId> = vec![];
        let result = scheduler.find_prefix(&tokens);
        // Empty query should return None (no meaningful prefix)
        assert!(result.is_none());
    }

    // ── select_victims_gmm with populated metadata returns subset ───────────

    #[test]
    fn select_victims_gmm_returns_at_most_count() {
        let scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let meta = HashMap::new();
        let priorities = HashMap::new();
        let victims = scheduler.select_victims_gmm(&meta, &priorities, 5);
        // No pages tracked, so result should be empty
        assert!(victims.is_empty());
    }

    // ── plan_prefill small prompt fits_l1 ───────────────────────────────────

    #[test]
    fn plan_prefill_small_prompt_fully_resident() {
        let mut scheduler = PagedScheduler::new_with_tiers(
            16, 4, HGALConfig::default(), 16, 0, 0,
        );
        let plan = scheduler.plan_prefill(4, 4);
        assert_eq!(plan, PrefillPlan::FullyResident { pages: 1 });
    }

    // ── storage_key is_unique for all valid combinations ────────────────────

    #[test]
    fn storage_key_uniqueness_small_range() {
        let mut keys = std::collections::HashSet::new();
        for req in 0..10u64 {
            for block in 0..10usize {
                let key = PagedScheduler::storage_key(req, block).expect("ok");
                assert!(keys.insert(key), "duplicate key for req={}, block={}", req, block);
            }
        }
        assert_eq!(keys.len(), 100);
    }

    // ===========================================================================
    // 13 new tests — edge case coverage expansion
    // ===========================================================================

    // ── 1. add_sequence with block_size=1 allocates one block per token ──────

    #[test]
    fn add_sequence_block_size_one_allocates_per_token() {
        let mut scheduler = PagedScheduler::new(8, 1, HGALConfig::default());
        // 5 tokens / block_size=1 = 5 blocks needed
        scheduler.add_sequence(make_group(1, 5)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 3);
        assert_eq!(scheduler.request_pages(1).len(), 5);
    }

    // ── 2. add_sequence that exhausts the entire block pool ──────────────────

    #[test]
    fn add_sequence_exhausts_pool_exactly() {
        let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
        // 16 tokens / block_size=4 = 4 blocks = entire pool
        scheduler.add_sequence(make_group(1, 16)).expect("add");
        assert_eq!(scheduler.num_free_blocks(), 0);
        assert_eq!(scheduler.num_total_blocks(), 4);
    }

    // ── 3. allocate_next_token recycles freed blocks from another sequence ───

    #[test]
    fn allocate_next_token_uses_recycled_blocks() {
        let mut scheduler = PagedScheduler::new(2, 4, HGALConfig::default());
        // Seq 1: 4 tokens -> 1 block, Seq 2: 4 tokens -> 1 block, pool exhausted
        scheduler.add_sequence(make_group(1, 4)).expect("add 1");
        scheduler.add_sequence(make_group(2, 4)).expect("add 2");
        assert_eq!(scheduler.num_free_blocks(), 0);
        // Free seq 2, returning 1 block
        scheduler.free_sequence(2);
        assert_eq!(scheduler.num_free_blocks(), 1);
        // Seq 1 crosses boundary, reuses the freed block
        let new_page = scheduler.allocate_next_token(1).expect("recycle");
        assert!(new_page.is_some());
        assert_eq!(scheduler.num_free_blocks(), 0);
    }

    // ── 4. fragmentation_ratio updates after allocate_next_token ─────────────

    #[test]
    fn fragmentation_ratio_decreases_as_tokens_fill_blocks() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        // 1 token -> 1 block, capacity=4, used=1 -> frag=0.75
        scheduler.add_sequence(make_group(1, 1)).expect("add");
        assert!((scheduler.kv_fragmentation_ratio() - 0.75).abs() < 1e-6);
        // Generate 2 more tokens (no new block): context_len=3, capacity=4 -> frag=0.25
        scheduler.allocate_next_token(1).expect("token 2");
        scheduler.allocate_next_token(1).expect("token 3");
        assert!((scheduler.kv_fragmentation_ratio() - 0.25).abs() < 1e-6);
    }

    // ── 5. get_page_table reflects new blocks after allocate_next_token ──────

    #[test]
    fn get_page_table_grows_with_allocate_next_token() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add"); // 1 block
        let table_before = scheduler.get_page_table(1).expect("table");
        assert_eq!(table_before.len(), 1);
        // Cross block boundary
        scheduler.allocate_next_token(1).expect("alloc");
        let table_after = scheduler.get_page_table(1).expect("table");
        assert_eq!(table_after.len(), 2);
    }

    // ── 6. take_pending_swap_in returns None on second call ──────────────────

    #[test]
    fn take_pending_swap_in_second_call_returns_none() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // 2 blocks
        let victims = scheduler.select_victims(2);
        let ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
        scheduler.free_victims(&ids).expect("free");
        // Restore triggers pending swap-in
        let _ = scheduler.allocate_next_token(1).expect("restore");
        assert!(scheduler.take_pending_swap_in(1).is_some());
        assert!(scheduler.take_pending_swap_in(1).is_none());
    }

    // ── 7. sync_page_states with empty slice is a no-op ──────────────────────

    #[test]
    fn sync_page_states_empty_slice_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        scheduler.add_sequence(make_group(1, 4)).expect("add");
        let free_before = scheduler.num_free_blocks();
        scheduler.sync_page_states(&[]);
        assert_eq!(scheduler.num_free_blocks(), free_before);
    }

    // ── 8. on_swap_in with empty page list is a no-op ────────────────────────

    #[test]
    fn on_swap_in_empty_page_list_is_noop() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        let free_before = scheduler.num_free_blocks();
        scheduler.on_swap_in(1, &[]);
        assert_eq!(scheduler.num_free_blocks(), free_before);
    }

    // ── 9. prefix_reuse with partial overlap allocates only remainder ────────

    #[test]
    fn prefix_reuse_partial_overlap_allocates_remainder() {
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());
        let tokens_a: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6]; // 6 tokens
        scheduler.add_sequence(make_group(1, 6)).expect("add first");
        scheduler.insert_prefix(1, &tokens_a);
        // New request shares first 4 tokens, needs 8 total
        let tokens_b: Vec<TokenId> = vec![1, 2, 3, 4, 10, 11, 12, 13];
        let reused = scheduler
            .add_sequence_with_prefix_reuse(make_group(2, 8), &tokens_b)
            .expect("add");
        assert_eq!(reused, 4);
    }

    // ── 10. find_donor with distinct buckets per half ────────────────────────

    #[test]
    fn find_donor_selects_matching_bucket_from_non_shared_half() {
        let mut pattern = vec![0u8; 20];
        // Non-shared layers 0..10: only layer 7 has bucket 3
        pattern[7] = 3;
        // Consumer layer 15 has bucket 3
        pattern[15] = 3;
        let result = find_donor(15, 20, 10, &pattern).expect("ok").expect("donor");
        assert_eq!(result, 7);
    }

    // ── 11. num_total_blocks stays constant through alloc/free cycles ────────

    #[test]
    fn num_total_blocks_unchanged_by_alloc_and_free() {
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());
        assert_eq!(scheduler.num_total_blocks(), 8);
        scheduler.add_sequence(make_group(1, 8)).expect("add"); // 2 blocks
        assert_eq!(scheduler.num_total_blocks(), 8);
        scheduler.free_sequence(1);
        assert_eq!(scheduler.num_total_blocks(), 8);
        scheduler.add_sequence(make_group(2, 4)).expect("add"); // 1 block
        assert_eq!(scheduler.num_total_blocks(), 8);
    }

    // ── 12. add_sequence OOM when context needs more than entire pool ────────

    #[test]
    fn add_sequence_oom_when_context_exceeds_pool_capacity() {
        let mut scheduler = PagedScheduler::new(2, 4, HGALConfig::default());
        // 20 tokens / block_size=4 = 5 blocks needed, pool has only 2
        let err = scheduler.add_sequence(make_group(1, 20)).expect_err("should OOM");
        assert!(matches!(
            err,
            SchedulerError::OutOfMemory {
                operation: "add_sequence",
                needed_blocks: 5,
                free_blocks: 2,
            }
        ));
    }

    // ── 13. LayerPageTable resolve follows donor chain for correct page_id ───

    #[test]
    fn layer_page_table_resolve_follows_donor_to_physical_page() {
        let mut table = LayerPageTable::new();
        let donor_key = LayerPageKey::new(1, 0);
        let consumer_a = LayerPageKey::new(1, 5);
        let consumer_b = LayerPageKey::new(1, 6);
        // Owned entry at layer 0 with page_id=77
        table.insert_owned(donor_key, 77, 0);
        // Two consumers both reference donor layer 0
        let page_a = table.insert_shared_ref(consumer_a, 0, 0).expect("ref a");
        let page_b = table.insert_shared_ref(consumer_b, 0, 0).expect("ref b");
        // All resolve to the same physical page_id
        assert_eq!(page_a, 77);
        assert_eq!(page_b, 77);
        assert_eq!(table.resolve(consumer_a), Some(77));
        assert_eq!(table.resolve(consumer_b), Some(77));
        // Donor has refcount=2
        assert_eq!(table.borrower_refcount(donor_key), 2);
    }
}