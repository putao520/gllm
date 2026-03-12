use super::allocator::BlockAllocator;
use super::hgal::{HGALConfig, HGALScheduler};
use super::types::{GroupState, SequenceGroup};
use super::vllm2024::{Scheduler2024Config, Scheduler2024State};
use super::types::{PageId, PageState, RequestId, StorageKey};
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
}

impl PagedScheduler {
    pub fn new(total_blocks: usize, block_size: usize, hgal_config: HGALConfig) -> Self {
        Self {
            hgal: HGALScheduler::new(hgal_config),
            allocator: BlockAllocator::new(block_size, total_blocks),
            block_tables: HashMap::new(),
            swapped_storage_keys: HashMap::new(),
            pending_swap_ins: HashMap::new(),
            block_size,
            vllm_state: None,
        }
    }

    pub fn storage_key(
        request_id: RequestId,
        logical_block_idx: usize,
    ) -> Result<StorageKey, SchedulerError> {
        let req = u64::try_from(request_id).map_err(|_| SchedulerError::StorageKeyOverflow {
            field: "request_id",
        })?;
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
        for _ in 0..needed_blocks {
            let block = self
                .allocator
                .allocate()
                .ok_or(SchedulerError::AllocatorInvariant {
                    operation: "add_sequence",
                })?;
            allocated.push(block);
            self.hgal.mark_accessed(block);
        }

        let mut block_table = BlockTable::new();
        block_table.blocks = allocated.clone();
        self.block_tables.insert(group.id, block_table);

        group.pages = allocated;
        self.hgal.upsert_group(group);

        Ok(())
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
            .unwrap_or_default()
    }

    pub fn free_sequence(&mut self, request_id: RequestId) {
        if let Some(block_table) = self.block_tables.remove(&request_id) {
            for block in block_table.blocks {
                self.allocator.free(block);
                // Also update HGAL metadata if needed, though remove_group handles group logic
                // But HGAL page metadata might need clearing
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
                .unwrap_or(0);
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
}
