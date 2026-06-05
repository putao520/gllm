use std::collections::VecDeque;

/// Physical Block Allocator
/// Manages the pool of physical KV cache blocks on the device.
pub struct BlockAllocator {
    block_size: usize,
    total_blocks: usize,
    free_blocks: VecDeque<usize>,
}

impl BlockAllocator {
    pub fn new(block_size: usize, total_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::with_capacity(total_blocks);
        for i in 0..total_blocks {
            free_blocks.push_back(i);
        }

        Self {
            block_size,
            total_blocks,
            free_blocks,
        }
    }

    /// Allocate a single block
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop_front()
    }

    /// Free a single block
    pub fn free(&mut self, block_id: usize) {
        if block_id < self.total_blocks {
            self.free_blocks.push_back(block_id);
        }
    }

    /// Get number of free blocks
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get total number of blocks
    pub fn get_total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Get block size
    pub fn get_block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_basic() {
        let mut allocator = BlockAllocator::new(16, 10);
        assert_eq!(allocator.get_num_free_blocks(), 10);

        let b1 = allocator.allocate();
        assert_eq!(b1, Some(0));
        assert_eq!(allocator.get_num_free_blocks(), 9);

        allocator.free(0);
        assert_eq!(allocator.get_num_free_blocks(), 10);
    }

    #[test]
    fn test_allocator_exhaustion() {
        let mut allocator = BlockAllocator::new(16, 1);
        let b1 = allocator.allocate();
        assert_eq!(b1, Some(0));

        let b2 = allocator.allocate();
        assert_eq!(b2, None);
    }

    #[test]
    fn test_allocator_block_size_and_total() {
        let alloc = BlockAllocator::new(64, 100);
        assert_eq!(alloc.get_block_size(), 64);
        assert_eq!(alloc.get_total_blocks(), 100);
        assert_eq!(alloc.get_num_free_blocks(), 100);
    }

    #[test]
    fn test_allocator_fifo_ordering() {
        let mut alloc = BlockAllocator::new(16, 4);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), None);

        alloc.free(1);
        // Freed block goes to back
        assert_eq!(alloc.allocate(), Some(1));
    }

    #[test]
    fn test_allocator_free_out_of_range_ignored() {
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.free(99); // out of range, should be ignored
        assert_eq!(alloc.get_num_free_blocks(), 2);
    }

    #[test]
    fn test_allocator_zero_blocks() {
        let alloc = BlockAllocator::new(16, 0);
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert_eq!(alloc.get_total_blocks(), 0);
    }

    #[test]
    fn test_allocator_reuse_after_free() {
        let mut alloc = BlockAllocator::new(16, 2);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        alloc.free(b0);
        alloc.free(b1);
        assert_eq!(alloc.get_num_free_blocks(), 2);
        // Freed blocks are reused
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
    }

    #[test]
    fn allocate_all_then_free_all_roundtrip() {
        let mut alloc = BlockAllocator::new(32, 5);
        let mut allocated = Vec::new();
        while let Some(b) = alloc.allocate() {
            allocated.push(b);
        }
        assert_eq!(allocated.len(), 5);
        assert_eq!(alloc.get_num_free_blocks(), 0);
        for b in allocated {
            alloc.free(b);
        }
        assert_eq!(alloc.get_num_free_blocks(), 5);
    }

    #[test]
    fn free_same_block_twice_increments_count() {
        let mut alloc = BlockAllocator::new(16, 3);
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        assert_eq!(alloc.get_num_free_blocks(), 0);
        alloc.free(0);
        alloc.free(0); // duplicate free — count goes to 2
        assert_eq!(alloc.get_num_free_blocks(), 2);
    }

    #[test]
    fn block_size_preserved() {
        let alloc = BlockAllocator::new(128, 10);
        assert_eq!(alloc.get_block_size(), 128);
    }

    #[test]
    fn single_block_lifecycle() {
        let mut alloc = BlockAllocator::new(16, 1);
        let b = alloc.allocate();
        assert_eq!(b, Some(0));
        assert!(alloc.allocate().is_none());
        alloc.free(0);
        assert_eq!(alloc.allocate(), Some(0));
    }

    // --- 15 new tests ---

    #[test]
    fn sequential_allocation_returns_ascending_ids() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 5);

        // Act
        let ids: Vec<usize> = (0..5).map(|_| alloc.allocate().unwrap()).collect();

        // Assert
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn free_boundary_blocks_first_and_last() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 4);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        let b3 = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free first and last blocks
        alloc.free(b0);
        alloc.free(b3);

        // Assert — two blocks available
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(3));
    }

    #[test]
    fn multiple_alloc_free_cycles() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 3);

        // Act & Assert — cycle 1
        let a = alloc.allocate().unwrap();
        alloc.free(a);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act & Assert — cycle 2
        let b = alloc.allocate().unwrap();
        alloc.free(b);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act & Assert — cycle 3
        let c = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 2);
        alloc.free(c);
        assert_eq!(alloc.get_num_free_blocks(), 3);
    }

    #[test]
    fn zero_blocks_allocator_allocate_returns_none() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 0);

        // Act
        let result = alloc.allocate();

        // Assert
        assert_eq!(result, None);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn varying_block_sizes_do_not_affect_allocation() {
        // Arrange — two allocators with different block sizes but same block count
        let mut alloc_small = BlockAllocator::new(8, 3);
        let mut alloc_large = BlockAllocator::new(4096, 3);

        // Act
        let s = alloc_small.allocate();
        let l = alloc_large.allocate();

        // Assert — allocation behavior is identical regardless of block_size
        assert_eq!(s, Some(0));
        assert_eq!(l, Some(0));
        assert_eq!(alloc_small.get_num_free_blocks(), 2);
        assert_eq!(alloc_large.get_num_free_blocks(), 2);
    }

    #[test]
    fn utilization_tracking_allocated_equals_total_minus_free() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 10);
        assert_eq!(alloc.get_total_blocks() - alloc.get_num_free_blocks(), 0);

        // Act
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        let _ = alloc.allocate();

        // Assert
        let allocated = alloc.get_total_blocks() - alloc.get_num_free_blocks();
        assert_eq!(allocated, 3);
    }

    #[test]
    fn freeing_subset_preserves_rest() {
        // Arrange — allocate all 6, then we control what's free
        let mut alloc = BlockAllocator::new(16, 6);
        let blocks: Vec<usize> = (0..6).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free only middle two (indices 1 and 2)
        alloc.free(blocks[1]);
        alloc.free(blocks[2]);

        // Assert — exactly 2 free blocks
        assert_eq!(alloc.get_num_free_blocks(), 2);
        // Freed blocks are served FIFO from free list: 1 then 2
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
    }

    #[test]
    fn free_all_in_reverse_order_restores_full_capacity() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 4);
        let mut blocks = Vec::new();
        for _ in 0..4 {
            blocks.push(alloc.allocate().unwrap());
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free in reverse order
        for b in blocks.into_iter().rev() {
            alloc.free(b);
        }

        // Assert
        assert_eq!(alloc.get_num_free_blocks(), 4);
    }

    #[test]
    fn multiple_out_of_range_frees_are_all_ignored() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 3);
        let initial_free = alloc.get_num_free_blocks();

        // Act — free multiple out-of-range blocks
        alloc.free(10);
        alloc.free(100);
        alloc.free(usize::MAX);

        // Assert — count unchanged
        assert_eq!(alloc.get_num_free_blocks(), initial_free);
    }

    #[test]
    fn large_capacity_allocator_distributes_all_ids() {
        // Arrange
        let capacity = 1000;
        let mut alloc = BlockAllocator::new(16, capacity);

        // Act — allocate all blocks
        let mut ids = Vec::with_capacity(capacity);
        while let Some(b) = alloc.allocate() {
            ids.push(b);
        }

        // Assert — every ID from 0..capacity appears exactly once
        assert_eq!(ids.len(), capacity);
        let mut sorted = ids.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..capacity).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn interleaved_allocate_and_free() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 4);
        let mut active = Vec::new();

        // Act — interleaved allocate/free pattern
        active.push(alloc.allocate().unwrap()); // allocate 0
        active.push(alloc.allocate().unwrap()); // allocate 1
        let freed = active.pop().unwrap();      // pop 1
        alloc.free(freed);                      // free 1, free_list = [2,3,1]
        active.push(alloc.allocate().unwrap()); // gets 2 (front of free list)
        active.push(alloc.allocate().unwrap()); // gets 3
        active.push(alloc.allocate().unwrap()); // gets 1
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Assert — all 4 blocks active, all valid IDs
        assert_eq!(active.len(), 4);
        let mut sorted = active.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn no_blocks_freed_allocate_always_returns_none() {
        // Arrange — allocate everything, never free
        let mut alloc = BlockAllocator::new(16, 2);
        let _ = alloc.allocate();
        let _ = alloc.allocate();

        // Act & Assert — all further allocations fail
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn freed_blocks_reused_in_free_order() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 3);
        let _ = alloc.allocate().unwrap(); // 0
        let _ = alloc.allocate().unwrap(); // 1
        let _ = alloc.allocate().unwrap(); // 2

        // Act — free in specific order: 2, then 0
        alloc.free(2);
        alloc.free(0);

        // Assert — reused in free order (FIFO: 2 first, then 0)
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(0));
    }

    #[test]
    fn free_order_does_not_affect_final_capacity() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 5);
        let blocks: Vec<usize> = (0..5).map(|_| alloc.allocate().unwrap()).collect();

        // Act — free in scrambled order
        alloc.free(blocks[3]);
        alloc.free(blocks[0]);
        alloc.free(blocks[4]);
        alloc.free(blocks[1]);
        alloc.free(blocks[2]);

        // Assert — all 5 blocks available regardless of free order
        assert_eq!(alloc.get_num_free_blocks(), 5);
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }
        assert_eq!(reclaimed.len(), 5);
    }

    #[test]
    fn initial_state_all_free_equals_total() {
        // Arrange
        let total = 42;
        let alloc = BlockAllocator::new(64, total);

        // Assert — invariant: initial state has all blocks free
        assert_eq!(alloc.get_num_free_blocks(), alloc.get_total_blocks());
        assert_eq!(alloc.get_num_free_blocks(), total);
    }

    // --- 15 additional tests ---

    #[test]
    fn free_count_increments_linearly_on_consecutive_frees() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 4);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        let b3 = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act & Assert — each free increments count by exactly 1
        alloc.free(b2);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        alloc.free(b0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        alloc.free(b3);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        alloc.free(b1);
        assert_eq!(alloc.get_num_free_blocks(), 4);
    }

    #[test]
    fn all_allocated_ids_are_unique() {
        // Arrange
        let capacity = 50;
        let mut alloc = BlockAllocator::new(16, capacity);

        // Act — allocate all blocks
        let mut ids = Vec::with_capacity(capacity);
        while let Some(b) = alloc.allocate() {
            ids.push(b);
        }

        // Assert — no duplicate IDs in the allocated set
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(*id), "duplicate block ID: {}", id);
        }
        assert_eq!(seen.len(), capacity);
    }

    #[test]
    fn free_at_upper_boundary_is_accepted() {
        // Arrange — total_blocks=5, valid IDs are 0..=4
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block at the upper boundary (total_blocks - 1)
        alloc.free(4);

        // Assert — accepted and count incremented
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(4));
    }

    #[test]
    fn free_at_lower_boundary_zero_is_accepted() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block ID 0 (lower boundary)
        alloc.free(0);

        // Assert
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(0));
    }

    #[test]
    fn repeated_single_slot_allocate_free_cycles() {
        // Arrange — single-slot allocator
        let mut alloc = BlockAllocator::new(16, 1);

        // Act & Assert — 10 cycles of allocate then free
        for i in 0..10 {
            let b = alloc.allocate();
            assert_eq!(b, Some(0), "cycle {} alloc failed", i);
            assert_eq!(alloc.get_num_free_blocks(), 0, "cycle {} not exhausted", i);
            alloc.free(0);
            assert_eq!(alloc.get_num_free_blocks(), 1, "cycle {} not restored", i);
        }
    }

    #[test]
    fn recovery_after_exhaustion_via_single_free() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 2);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert!(alloc.allocate().is_none());

        // Act — free just one block
        alloc.free(b0);

        // Assert — one allocation succeeds, second fails again
        assert_eq!(alloc.allocate(), Some(b0));
        assert_eq!(alloc.allocate(), None);
        // b1 still held
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Cleanup — free b1 to avoid unused variable warning
        alloc.free(b1);
    }

    #[test]
    fn minimum_block_size_one_still_allocates() {
        // Arrange — block_size=1 (minimum meaningful value)
        let mut alloc = BlockAllocator::new(1, 5);

        // Act
        let b = alloc.allocate();

        // Assert — allocation works regardless of block_size value
        assert_eq!(b, Some(0));
        assert_eq!(alloc.get_block_size(), 1);
        assert_eq!(alloc.get_num_free_blocks(), 4);
    }

    #[test]
    fn large_block_size_does_not_affect_allocation_count() {
        // Arrange — very large block_size, small block count
        let mut alloc = BlockAllocator::new(usize::MAX / 2, 3);

        // Act — allocate all
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();
        let d = alloc.allocate();

        // Assert — only block count matters, not block_size
        assert_eq!(a, Some(0));
        assert_eq!(b, Some(1));
        assert_eq!(c, Some(2));
        assert_eq!(d, None);
        assert_eq!(alloc.get_block_size(), usize::MAX / 2);
    }

    #[test]
    fn three_full_exhaust_and_recover_cycles() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 3);

        // Act & Assert — 3 complete exhaust-then-recover cycles
        for cycle in 0..3 {
            let mut blocks = Vec::new();
            while let Some(b) = alloc.allocate() {
                blocks.push(b);
            }
            assert_eq!(blocks.len(), 3, "cycle {} did not allocate all", cycle);
            assert_eq!(alloc.get_num_free_blocks(), 0, "cycle {} not exhausted", cycle);

            for b in blocks {
                alloc.free(b);
            }
            assert_eq!(alloc.get_num_free_blocks(), 3, "cycle {} not recovered", cycle);
        }
    }

    #[test]
    fn partial_allocation_then_partial_free() {
        // Arrange — 8 total blocks
        let mut alloc = BlockAllocator::new(16, 8);

        // Act — allocate only 5 (IDs 0..4), leaving free list [5, 6, 7]
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        let b3 = alloc.allocate().unwrap();
        let b4 = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act — free 2 of the 5 allocated; freed blocks append to back of free list
        // free list becomes [5, 6, 7, 2, 4]
        alloc.free(b2);
        alloc.free(b4);
        assert_eq!(alloc.get_num_free_blocks(), 5);

        // Assert — unallocated blocks served first from front, then freed blocks
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.allocate(), Some(6));
        assert_eq!(alloc.allocate(), Some(7));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(4));
        // Now all 8 blocks are allocated
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Cleanup — free everything
        alloc.free(b0);
        alloc.free(b1);
        alloc.free(b3);
        alloc.free(5);
        alloc.free(6);
        alloc.free(7);
        alloc.free(2);
        alloc.free(4);
        assert_eq!(alloc.get_num_free_blocks(), 8);
    }

    #[test]
    fn minimal_config_block_size_one_total_one() {
        // Arrange — smallest valid configuration
        let mut alloc = BlockAllocator::new(1, 1);
        assert_eq!(alloc.get_block_size(), 1);
        assert_eq!(alloc.get_total_blocks(), 1);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Act
        let b = alloc.allocate();

        // Assert
        assert_eq!(b, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Recover
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);
    }

    #[test]
    fn allocate_after_free_returns_exact_freed_id() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        alloc.allocate(); // 3
        alloc.allocate(); // 4
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 3 specifically
        alloc.free(3);

        // Assert — next allocation returns exactly the freed ID
        let reclaimed = alloc.allocate();
        assert_eq!(reclaimed, Some(3));
    }

    #[test]
    fn two_allocators_have_independent_state() {
        // Arrange — two independent allocators
        let mut alloc_a = BlockAllocator::new(16, 3);
        let mut alloc_b = BlockAllocator::new(32, 5);

        // Act — different operations on each
        let a0 = alloc_a.allocate();
        let b0 = alloc_b.allocate();
        let b1 = alloc_b.allocate();

        // Assert — no cross-contamination
        assert_eq!(a0, Some(0));
        assert_eq!(alloc_a.get_num_free_blocks(), 2);
        assert_eq!(alloc_a.get_block_size(), 16);

        assert_eq!(b0, Some(0));
        assert_eq!(b1, Some(1));
        assert_eq!(alloc_b.get_num_free_blocks(), 3);
        assert_eq!(alloc_b.get_block_size(), 32);

        // Free from one does not affect the other
        alloc_a.free(0);
        assert_eq!(alloc_a.get_num_free_blocks(), 3);
        assert_eq!(alloc_b.get_num_free_blocks(), 3); // unchanged
    }

    #[test]
    fn free_at_exact_total_blocks_boundary_is_ignored() {
        // Arrange — total_blocks=5, valid IDs are 0..=4
        let mut alloc = BlockAllocator::new(16, 5);
        let initial = alloc.get_num_free_blocks();

        // Act — free(total_blocks) is out of range (valid max is total_blocks-1)
        alloc.free(5);
        alloc.free(5); // try twice for good measure

        // Assert — count unchanged because 5 >= total_blocks
        assert_eq!(alloc.get_num_free_blocks(), initial);
    }

    #[test]
    fn free_count_matches_allocated_after_each_operation() {
        // Arrange
        let total = 10;
        let mut alloc = BlockAllocator::new(16, total);
        let mut outstanding = 0usize;

        // Act & Assert — verify invariant after each operation
        for i in 0..7 {
            let b = alloc.allocate();
            assert!(b.is_some(), "allocation {} should succeed", i);
            outstanding += 1;
            assert_eq!(
                alloc.get_num_free_blocks(),
                total - outstanding,
                "free count wrong after allocating block {}",
                i
            );
        }

        // Free 3 blocks one at a time, verifying each step
        alloc.free(2);
        outstanding -= 1;
        assert_eq!(alloc.get_num_free_blocks(), total - outstanding);

        alloc.free(5);
        outstanding -= 1;
        assert_eq!(alloc.get_num_free_blocks(), total - outstanding);

        alloc.free(0);
        outstanding -= 1;
        assert_eq!(alloc.get_num_free_blocks(), total - outstanding);
    }

    // --- 15 more new tests ---

    #[test]
    fn free_unallocated_in_range_id_increments_count() {
        // Arrange — allocator with 5 blocks; never allocate block 3
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        // 3 and 4 still in free list
        alloc.allocate(); // 3
        alloc.allocate(); // 4
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 3 (which we DID allocate above, so this is valid)
        // Then free block 3 again (duplicate free — no guard against double-free)
        alloc.free(3);
        alloc.free(3);

        // Assert — count is 2 (double free accepted, block 3 appears twice)
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(3));
    }

    #[test]
    fn block_size_zero_allocator_still_allocates_by_count() {
        // Arrange — block_size=0 is unusual but struct doesn't prevent it
        let mut alloc = BlockAllocator::new(0, 3);
        assert_eq!(alloc.get_block_size(), 0);

        // Act
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();
        let d = alloc.allocate();

        // Assert — allocation is purely count-based, block_size is metadata
        assert_eq!(a, Some(0));
        assert_eq!(b, Some(1));
        assert_eq!(c, Some(2));
        assert_eq!(d, None);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn total_blocks_is_immutable_through_operations() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 7);
        let initial_total = alloc.get_total_blocks();

        // Act — perform many operations
        for _ in 0..initial_total {
            alloc.allocate();
        }
        assert_eq!(alloc.get_total_blocks(), initial_total);

        for i in 0..initial_total {
            alloc.free(i);
        }
        assert_eq!(alloc.get_total_blocks(), initial_total);

        // Allocate/free a few more times
        alloc.allocate();
        alloc.free(0);

        // Assert — total_blocks never changes
        assert_eq!(alloc.get_total_blocks(), initial_total);
    }

    #[test]
    fn free_list_fifo_after_interleaved_partial_frees() {
        // Arrange — 6 blocks, allocate all, free in non-contiguous pattern
        let mut alloc = BlockAllocator::new(16, 6);
        for _ in 0..6 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free blocks 5, 1, 3 (in that order)
        alloc.free(5);
        alloc.free(1);
        alloc.free(3);

        // Assert — FIFO: next allocations return 5, 1, 3 in that order
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn single_block_triple_duplicate_free_yields_triple_allocate() {
        // Arrange — single block
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free the same block 3 times
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);

        // Assert — count is 3 (block 0 appears 3 times in free list)
        assert_eq!(alloc.get_num_free_blocks(), 3);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn zero_blocks_allocator_free_boundary_zero_ignored() {
        // Arrange — 0 total blocks, valid range is empty
        let mut alloc = BlockAllocator::new(16, 0);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0, which is out of range (0 >= total_blocks=0)
        alloc.free(0);

        // Assert — no blocks added
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn rapid_ten_cycle_stress_with_correctness() {
        // Arrange
        let total = 20;
        let mut alloc = BlockAllocator::new(32, total);

        // Act & Assert — 10 full exhaust-recover cycles
        for cycle in 0..10 {
            let mut blocks = Vec::with_capacity(total);
            while let Some(b) = alloc.allocate() {
                blocks.push(b);
            }
            // Every cycle, all IDs 0..total must be allocated
            assert_eq!(blocks.len(), total, "cycle {} short allocation", cycle);
            let mut sorted = blocks.clone();
            sorted.sort();
            let expected: Vec<usize> = (0..total).collect();
            assert_eq!(sorted, expected, "cycle {} wrong IDs", cycle);

            // Free all
            for b in blocks {
                alloc.free(b);
            }
            assert_eq!(alloc.get_num_free_blocks(), total, "cycle {} not recovered", cycle);
        }
    }

    #[test]
    fn block_size_query_unaffected_by_state_changes() {
        // Arrange
        let mut alloc = BlockAllocator::new(256, 10);

        // Act — various state mutations
        alloc.allocate();
        alloc.allocate();
        alloc.free(0);

        // Assert — block_size is construction-time metadata, never changes
        assert_eq!(alloc.get_block_size(), 256);
    }

    #[test]
    fn allocate_from_virgin_allocator_returns_id_zero() {
        // Arrange
        let alloc = BlockAllocator::new(16, 5);

        // Act — no allocations yet, but we can check free count
        // The first allocation from a fresh allocator must be ID 0
        let mut alloc_mut = alloc;
        let first = alloc_mut.allocate();

        // Assert
        assert_eq!(first, Some(0));
    }

    #[test]
    fn free_list_order_after_non_sequential_free() {
        // Arrange — allocate 5 blocks (0..4), then free 2 and 4
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act
        alloc.free(2);
        alloc.free(4);

        // Assert — free list: [2, 4], so allocate returns 2 then 4
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn exhaust_free_subset_reexhaust_verify_ids() {
        // Arrange — 4 blocks, allocate all, free only block 1
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Act — re-allocate the freed block
        let reclaimed = alloc.allocate();
        assert_eq!(reclaimed, Some(1));

        // Now free blocks 0, 1, 3 (leaving 2 still allocated)
        alloc.free(0);
        alloc.free(1);
        alloc.free(3);

        // Assert — free list is [0, 1, 3], allocate in that order
        assert_eq!(alloc.get_num_free_blocks(), 3);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_just_below_boundary_accepted_just_above_ignored() {
        // Arrange — total_blocks=10, valid IDs 0..=9
        let mut alloc = BlockAllocator::new(16, 10);
        for _ in 0..10 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free ID 9 (valid, just below boundary)
        alloc.free(9);
        // Free ID 10 (invalid, exactly at boundary)
        alloc.free(10);

        // Assert — only the valid free took effect
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(9));
    }

    #[test]
    fn allocate_none_does_not_change_free_count() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.allocate();
        alloc.allocate();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — try to allocate when exhausted, multiple times
        let before = alloc.get_num_free_blocks();
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        let _ = alloc.allocate();

        // Assert — free count unchanged by failed allocations
        assert_eq!(alloc.get_num_free_blocks(), before);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn high_count_allocator_with_partial_free_pattern() {
        // Arrange — 100 blocks, allocate all, free every other block
        let total = 100;
        let mut alloc = BlockAllocator::new(16, total);
        let mut blocks: Vec<usize> = (0..total).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free even-indexed blocks (0, 2, 4, ..., 98)
        let freed: Vec<usize> = blocks.iter().filter(|&&b| b % 2 == 0).copied().collect();
        for &b in &freed {
            alloc.free(b);
        }

        // Assert — 50 blocks free
        assert_eq!(alloc.get_num_free_blocks(), 50);

        // Re-allocate all freed blocks
        let mut reclaimed = Vec::with_capacity(50);
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }
        assert_eq!(reclaimed.len(), 50);

        // All reclaimed blocks should be even numbers
        for b in &reclaimed {
            assert_eq!(b % 2, 0, "reclaimed block {} should be even", b);
        }
    }

    // --- 13 additional edge-case tests ---

    #[test]
    fn out_of_range_free_then_valid_free_only_valid_counts() {
        // Arrange — 4 blocks, allocate all
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free an out-of-range ID, then a valid ID
        alloc.free(99);
        assert_eq!(alloc.get_num_free_blocks(), 0, "out-of-range free should not increment");
        alloc.free(2);
        assert_eq!(alloc.get_num_free_blocks(), 1, "only valid free should increment");

        // Assert — only block 2 is available
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn exhaust_then_free_scattered_reallocate_all_unique() {
        // Arrange — 5 blocks, exhaust, free 1,3,4 in that order
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }
        alloc.free(1);
        alloc.free(3);
        alloc.free(4);

        // Act — re-allocate the 3 freed blocks
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }

        // Assert — exactly 3 blocks, all unique, in FIFO free order
        assert_eq!(reclaimed.len(), 3);
        assert_eq!(reclaimed, vec![1, 3, 4]);
    }

    #[test]
    fn free_never_allocated_in_range_block_increments_count() {
        // Arrange — fresh allocator with 4 blocks, never allocate anything
        let mut alloc = BlockAllocator::new(16, 4);
        // Block 2 was never allocated, but is in valid range (0..4)
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Act — "free" block 2 (which is still in the free list)
        alloc.free(2);

        // Assert — count incremented (duplicate entry), but block 2 still
        // exists at its original position AND at the back
        assert_eq!(alloc.get_num_free_blocks(), 5);
        // Original free list was [0,1,2,3]; after free(2) it is [0,1,2,3,2]
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(2)); // duplicate
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn partial_alloc_never_exhausted_preserves_remaining_order() {
        // Arrange — 6 blocks, allocate only 2 (IDs 0 and 1)
        let mut alloc = BlockAllocator::new(16, 6);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Act — allocate remaining without freeing
        let remaining: Vec<usize> = (0..4).map(|_| alloc.allocate().unwrap()).collect();

        // Assert — remaining blocks served in ascending order from original pool
        assert_eq!(remaining, vec![2, 3, 4, 5]);
    }

    #[test]
    fn allocate_half_free_all_reallocate_half_order() {
        // Arrange — 8 blocks
        let mut alloc = BlockAllocator::new(16, 8);
        let first_half: Vec<usize> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(first_half, vec![0, 1, 2, 3]);

        // Act — free all first half, then re-allocate 4 more
        for b in &first_half {
            alloc.free(*b);
        }
        // Free list: [4,5,6,7,0,1,2,3] — untouched first, then freed
        // But wait: 4,5,6,7 were never allocated, so free list is [4,5,6,7,0,1,2,3]
        assert_eq!(alloc.get_num_free_blocks(), 8);

        let second_half: Vec<usize> = (0..4).map(|_| alloc.allocate().unwrap()).collect();

        // Assert — front of free list served first (untouched 4,5,6,7)
        assert_eq!(second_half, vec![4, 5, 6, 7]);
    }

    #[test]
    fn free_then_immediate_reallocate_same_block_tight_loop() {
        // Arrange — single block, tight alloc-free cycle 20 times
        let mut alloc = BlockAllocator::new(16, 1);

        // Act & Assert
        for i in 0..20 {
            let b = alloc.allocate();
            assert_eq!(b, Some(0), "cycle {} allocate failed", i);
            assert_eq!(alloc.get_num_free_blocks(), 0);
            alloc.free(0);
            assert_eq!(alloc.get_num_free_blocks(), 1);
        }
    }

    #[test]
    fn interleaved_alloc_free_four_block_wave_pattern() {
        // Arrange — 4 blocks
        let mut alloc = BlockAllocator::new(16, 4);
        let mut held = Vec::new();

        // Wave 1: allocate 2
        held.push(alloc.allocate().unwrap()); // 0
        held.push(alloc.allocate().unwrap()); // 1

        // Wave 2: free 1, allocate 1
        let released = held.pop().unwrap(); // 1
        alloc.free(released);
        held.push(alloc.allocate().unwrap()); // 2

        // Wave 3: allocate 2 more (should get 3 then 1)
        let next1 = alloc.allocate().unwrap();
        let next2 = alloc.allocate().unwrap();
        assert_eq!(next1, 3);
        assert_eq!(next2, 1);

        // Assert — all 4 blocks allocated, none free
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn stats_invariant_free_plus_allocated_equals_total_throughout() {
        // Arrange — 7 blocks, track allocations
        let total = 7;
        let mut alloc = BlockAllocator::new(16, total);
        let mut allocated_count = 0usize;

        // Act — allocate 5
        let mut blocks = Vec::new();
        for _ in 0..5 {
            blocks.push(alloc.allocate().unwrap());
            allocated_count += 1;
        }

        // Assert invariant at partial capacity
        assert_eq!(alloc.get_num_free_blocks(), total - allocated_count);

        // Act — free 2 specific blocks
        alloc.free(blocks[1]);
        alloc.free(blocks[3]);
        allocated_count -= 2;
        assert_eq!(alloc.get_num_free_blocks(), total - allocated_count);

        // Act — allocate 1 more
        let _extra = alloc.allocate().unwrap();
        allocated_count += 1;
        assert_eq!(alloc.get_num_free_blocks(), total - allocated_count);

        // Final invariant
        assert_eq!(alloc.get_num_free_blocks() + allocated_count, total);
    }

    #[test]
    fn exhaust_partial_free_still_exhausted_until_all_returned() {
        // Arrange — 3 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 3);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        assert!(alloc.allocate().is_none());

        // Act & Assert — free one block, only one alloc succeeds
        alloc.free(b1);
        assert_eq!(alloc.allocate(), Some(b1));
        assert!(alloc.allocate().is_none()); // still exhausted

        // Free one more, only one succeeds
        alloc.free(b0);
        assert_eq!(alloc.allocate(), Some(b0));
        assert!(alloc.allocate().is_none()); // still exhausted

        // Free last one
        alloc.free(b2);
        assert_eq!(alloc.allocate(), Some(b2));
        assert!(alloc.allocate().is_none()); // b0 and b1 still held
    }

    #[test]
    fn allocate_all_free_odd_indices_reallocate_verify_only_odds() {
        // Arrange — 10 blocks, exhaust then free odd IDs
        let total = 10;
        let mut alloc = BlockAllocator::new(16, total);
        for _ in 0..total {
            alloc.allocate();
        }

        // Act — free odd blocks: 1, 3, 5, 7, 9
        for id in (1..total).step_by(2) {
            alloc.free(id);
        }
        assert_eq!(alloc.get_num_free_blocks(), 5);

        // Re-allocate all freed
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }

        // Assert — only odd IDs reclaimed, in the order freed
        assert_eq!(reclaimed.len(), 5);
        assert_eq!(reclaimed, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn valid_free_after_multiple_out_of_range_frees_counts_correctly() {
        // Arrange — 3 blocks, allocate all
        let mut alloc = BlockAllocator::new(16, 3);
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        let _ = alloc.allocate();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — several out-of-range frees followed by valid free
        alloc.free(3);   // out of range (valid: 0..2)
        alloc.free(100);
        alloc.free(usize::MAX);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        alloc.free(1);   // valid
        assert_eq!(alloc.get_num_free_blocks(), 1);

        alloc.free(0);   // valid
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — only valid frees produced available blocks
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn consecutive_different_block_alloc_free_fifo_after_each_free() {
        // Arrange — 4 blocks
        let mut alloc = BlockAllocator::new(16, 4);

        // Act — allocate A=0, free A, free list becomes [1,2,3,0]
        let a = alloc.allocate().unwrap();
        assert_eq!(a, 0);
        alloc.free(a);

        // Next alloc gets front of free list (1), not the just-freed 0
        let b = alloc.allocate().unwrap();
        assert_eq!(b, 1);

        // Next alloc gets 2
        let c = alloc.allocate().unwrap();
        assert_eq!(c, 2);

        // Free B and C, free list becomes [3,0,1,2]
        alloc.free(b);
        alloc.free(c);

        // Assert — FIFO: 3 (front of untouched), then 0, then 1, then 2
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn large_capacity_fragmentation_recovery_all_ids_unique() {
        // Arrange — 200 blocks, exhaust, free every 3rd block
        let total = 200;
        let mut alloc = BlockAllocator::new(16, total);
        for _ in 0..total {
            alloc.allocate();
        }

        // Act — free every 3rd block: 0, 3, 6, ..., 198
        let freed_count = (0..total).filter(|&i| i % 3 == 0).count();
        for i in (0..total).filter(|&i| i % 3 == 0) {
            alloc.free(i);
        }
        assert_eq!(alloc.get_num_free_blocks(), freed_count);

        // Re-allocate all freed blocks
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }

        // Assert — count matches, all unique, all multiples of 3
        assert_eq!(reclaimed.len(), freed_count);
        let mut seen = std::collections::HashSet::new();
        for b in &reclaimed {
            assert!(seen.insert(*b), "duplicate reclaimed block {}", b);
            assert_eq!(b % 3, 0, "block {} should be multiple of 3", b);
        }
    }

    // --- 13 new edge-case tests ---

    #[test]
    fn free_at_boundary_minus_one_accepted_boundary_plus_one_rejected() {
        // Arrange — total_blocks=5, valid range [0,4]
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 4 (boundary - 1, valid) and block 6 (boundary + 1, invalid)
        alloc.free(4);
        alloc.free(6);
        alloc.free(5); // exactly total_blocks, also invalid

        // Assert — only the valid free counted
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(4));
    }

    #[test]
    fn free_block_zero_after_only_allocating_high_ids() {
        // Arrange — 5 blocks, allocate first 4 (IDs 0-3)
        let mut alloc = BlockAllocator::new(16, 5);
        let _b0 = alloc.allocate().unwrap(); // 0
        let _b1 = alloc.allocate().unwrap(); // 1
        let _b2 = alloc.allocate().unwrap(); // 2
        let _b3 = alloc.allocate().unwrap(); // 3
        // Free list now: [4]

        // Act — allocate the remaining block (ID 4)
        let b4 = alloc.allocate().unwrap();
        assert_eq!(b4, 4);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Free only block 0
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Assert — only block 0 is available
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn allocation_statistics_accuracy_through_mixed_operations() {
        // Arrange
        let total = 12;
        let mut alloc = BlockAllocator::new(16, total);

        // Act — allocate 5
        let mut held: Vec<usize> = (0..5).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 7);
        assert_eq!(alloc.get_total_blocks(), total);
        assert_eq!(alloc.get_block_size(), 16);

        // Free 2 from the middle
        let mid1 = held.remove(2); // was index 2
        let mid2 = held.remove(0); // was index 0
        alloc.free(mid1);
        alloc.free(mid2);
        assert_eq!(alloc.get_num_free_blocks(), 9);

        // Allocate 4 more — should get untouched blocks first, then freed ones
        let extra: Vec<usize> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 5);

        // Assert — total invariant holds
        let allocated = total - alloc.get_num_free_blocks();
        assert_eq!(allocated, held.len() + extra.len());

        // All held + extra IDs are unique
        let mut all_ids = held.clone();
        all_ids.extend(&extra);
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), held.len() + extra.len());
    }

    #[test]
    fn free_id_one_when_only_zero_allocated_goes_above_total() {
        // Arrange — 1 block total, allocate only block 0
        let mut alloc = BlockAllocator::new(16, 1);
        let _b0 = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 1 (out of range, since total=1 means only ID 0 is valid)
        alloc.free(1);

        // Assert — out of range free ignored
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Free the actually allocated block
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(0));
    }

    #[test]
    fn free_list_content_after_three_wave_pattern() {
        // Arrange — 4 blocks, perform 3 waves of partial alloc/free
        let mut alloc = BlockAllocator::new(16, 4);

        // Wave 1: allocate 2, free 1
        let a = alloc.allocate().unwrap(); // 0
        let b = alloc.allocate().unwrap(); // 1
        alloc.free(a);
        // Free list: [2, 3, 0]

        // Wave 2: allocate 3
        let c = alloc.allocate().unwrap(); // 2
        let d = alloc.allocate().unwrap(); // 3
        let e = alloc.allocate().unwrap(); // 0
        assert!(alloc.allocate().is_none());
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Wave 3: free b, c, d (IDs 1, 2, 3)
        alloc.free(b); // 1
        alloc.free(c); // 2
        alloc.free(d); // 3

        // Assert — FIFO order: 1, 2, 3
        assert_eq!(alloc.get_num_free_blocks(), 3);
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // e (ID 0) is still held
        assert!(alloc.allocate().is_none());
        let _ = e;
    }

    #[test]
    fn two_allocators_same_config_independent_lifecycles() {
        // Arrange — two allocators with identical config
        let mut alloc1 = BlockAllocator::new(32, 3);
        let mut alloc2 = BlockAllocator::new(32, 3);

        // Act — diverge their states
        let a1 = alloc1.allocate().unwrap();
        let a2 = alloc2.allocate().unwrap();
        assert_eq!(a1, 0);
        assert_eq!(a2, 0);

        // Allocator 1: exhaust
        alloc1.allocate();
        alloc1.allocate();
        assert_eq!(alloc1.get_num_free_blocks(), 0);
        assert_eq!(alloc2.get_num_free_blocks(), 2);

        // Allocator 2: free and re-allocate
        alloc2.free(a2);
        assert_eq!(alloc2.get_num_free_blocks(), 3);
        let b2 = alloc2.allocate();
        assert_eq!(b2, Some(1)); // front of free list [1,2,0]

        // Assert — independent states
        assert_eq!(alloc1.get_num_free_blocks(), 0);
        assert_eq!(alloc2.get_num_free_blocks(), 2);
    }

    #[test]
    fn allocate_then_immediate_free_preserves_total_capacity() {
        // Arrange — 5 blocks
        let mut alloc = BlockAllocator::new(16, 5);
        let initial_free = alloc.get_num_free_blocks();
        assert_eq!(initial_free, 5);

        // Act — allocate and immediately free each block one by one
        for expected_id in 0..5 {
            let b = alloc.allocate();
            assert_eq!(b, Some(expected_id));
            alloc.free(b.unwrap());
            assert_eq!(alloc.get_num_free_blocks(), initial_free);
        }

        // Assert — all 5 still available in correct order (freed blocks appended to back)
        // After each immediate free, the freed block goes to back of the free list,
        // but the next unallocated block is at the front.
        // After the loop, free list has rotated through all blocks.
        let mut ids = Vec::new();
        while let Some(b) = alloc.allocate() {
            ids.push(b);
        }
        assert_eq!(ids.len(), 5);
        // All IDs present
        let mut sorted = ids;
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn out_of_range_free_on_partially_exhausted_allocator() {
        // Arrange — 5 blocks, allocate 3
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Act — free out-of-range IDs, then verify untouched blocks still available
        alloc.free(100);
        alloc.free(5);
        alloc.free(usize::MAX);

        // Assert — free count unchanged by out-of-range frees
        assert_eq!(alloc.get_num_free_blocks(), 2);
        // Untouched blocks still at front of free list
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn double_free_causes_over_count_but_allocate_drains_to_real_total() {
        // Arrange — 3 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — double-free block 1
        alloc.free(1);
        alloc.free(1); // duplicate
        assert_eq!(alloc.get_num_free_blocks(), 2); // over-count

        // Allocate until exhausted
        let first = alloc.allocate();
        let second = alloc.allocate();
        let third = alloc.allocate();

        // Assert — block 1 served twice, then None (real total exhausted)
        assert_eq!(first, Some(1));
        assert_eq!(second, Some(1));
        assert_eq!(third, None);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_all_blocks_in_random_permutation_restores_exact_capacity() {
        // Arrange — 7 blocks, exhaust
        let total = 7;
        let mut alloc = BlockAllocator::new(16, total);
        let blocks: Vec<usize> = (0..total).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free in a specific non-trivial permutation
        let order = [3, 6, 0, 5, 1, 4, 2];
        for &idx in &order {
            alloc.free(blocks[idx]);
        }

        // Assert — full capacity restored
        assert_eq!(alloc.get_num_free_blocks(), total);

        // All IDs recoverable, in FIFO free order
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }
        assert_eq!(reclaimed.len(), total);
        // Free order was 3,6,0,5,1,4,2 — so that's what FIFO returns
        assert_eq!(reclaimed, vec![3, 6, 0, 5, 1, 4, 2]);
    }

    #[test]
    fn multiple_duplicate_frees_create_multiple_entries_in_free_list() {
        // Arrange — 2 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 four times
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);

        // Assert — 4 entries in free list, all ID 0
        assert_eq!(alloc.get_num_free_blocks(), 4);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn zero_block_size_and_zero_total_blocks_combined() {
        // Arrange — both block_size and total_blocks are zero
        let mut alloc = BlockAllocator::new(0, 0);
        assert_eq!(alloc.get_block_size(), 0);
        assert_eq!(alloc.get_total_blocks(), 0);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — allocate and free on degenerate allocator
        let result = alloc.allocate();
        assert_eq!(result, None);

        // Free block 0 — should be ignored (0 >= total_blocks=0)
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Assert — still cannot allocate
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn complex_interleaved_pattern_final_state_consistency() {
        // Arrange — 6 blocks
        let total = 6;
        let mut alloc = BlockAllocator::new(16, total);

        // Phase 1: allocate all
        let mut all: Vec<usize> = (0..total).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Phase 2: free blocks 0, 2, 4 (even indices from all vec)
        alloc.free(all[0]); // ID 0
        alloc.free(all[2]); // ID 2
        alloc.free(all[4]); // ID 4
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Phase 3: allocate 2 back — gets 0, 2 (FIFO from free list)
        let r1 = alloc.allocate().unwrap();
        let r2 = alloc.allocate().unwrap();
        assert_eq!(r1, 0);
        assert_eq!(r2, 2);
        assert_eq!(alloc.get_num_free_blocks(), 1); // only block 4 left free

        // Phase 4: free blocks 1, 3, 5 (odd IDs)
        alloc.free(all[1]); // ID 1
        alloc.free(all[3]); // ID 3
        alloc.free(all[5]); // ID 5
        // Free list now: [4, 1, 3, 5]
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Phase 5: allocate all free blocks
        let final_batch: Vec<usize> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(final_batch, vec![4, 1, 3, 5]);

        // Assert — all blocks accounted for, none free
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Verify all 6 IDs are held: r1=0, r2=2, final_batch=[4,1,3,5]
        let mut held = vec![r1, r2];
        held.extend(&final_batch);
        held.sort();
        assert_eq!(held, vec![0, 1, 2, 3, 4, 5]);
    }

    // --- 13 additional edge-case tests (target: 94 total) ---

    #[test]
    fn free_block_id_equal_to_total_blocks_is_ignored() {
        // Arrange — total_blocks=3, valid IDs 0,1,2; ID 3 is out of range
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free(total_blocks) is the exact boundary, must be ignored
        alloc.free(3);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Free a valid block for contrast
        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(1));
    }

    #[test]
    fn block_size_query_returns_correct_value_after_full_exhaustion() {
        // Arrange — block_size=512
        let mut alloc = BlockAllocator::new(512, 4);

        // Act — exhaust all blocks
        for _ in 0..4 {
            alloc.allocate();
        }

        // Assert — block_size is metadata, unaffected by allocation state
        assert_eq!(alloc.get_block_size(), 512);
    }

    #[test]
    fn single_block_allocator_repeated_alloc_with_intervening_double_free() {
        // Arrange
        let mut alloc = BlockAllocator::new(16, 1);

        // Cycle 1: allocate, double-free, allocate
        let b = alloc.allocate();
        assert_eq!(b, Some(0));
        alloc.free(0);
        alloc.free(0); // double free -> 2 entries
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // First allocate drains one entry
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Second allocate drains the duplicate
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Assert — now truly exhausted
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn two_allocators_different_total_blocks_no_interference() {
        // Arrange — one small, one large allocator
        let mut small = BlockAllocator::new(16, 2);
        let mut large = BlockAllocator::new(16, 100);

        // Act — exhaust small
        small.allocate();
        small.allocate();
        assert!(small.allocate().is_none());

        // Large still works fine
        let first = large.allocate();
        assert_eq!(first, Some(0));
        assert_eq!(large.get_num_free_blocks(), 99);

        // Free in small, verify large unaffected
        small.free(0);
        assert_eq!(small.get_num_free_blocks(), 1);
        assert_eq!(large.get_num_free_blocks(), 99);
    }

    #[test]
    fn free_list_serves_original_blocks_before_freed_blocks() {
        // Arrange — 5 blocks, allocate only 2 (IDs 0 and 1)
        // Free list remaining: [2, 3, 4]
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act — free block 0 (appended to back of free list)
        // Free list: [2, 3, 4, 0]
        alloc.free(0);

        // Assert — untouched blocks served before the freed block
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(0)); // freed block last
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn free_with_usize_max_block_id_ignored() {
        // Arrange — small allocator
        let mut alloc = BlockAllocator::new(16, 3);
        let initial = alloc.get_num_free_blocks();

        // Act — free the maximum possible usize value
        alloc.free(usize::MAX);

        // Assert — out of range, count unchanged
        assert_eq!(alloc.get_num_free_blocks(), initial);
    }

    #[test]
    fn total_blocks_correct_for_various_sizes() {
        // Arrange & Assert — verify constructor stores total correctly
        assert_eq!(BlockAllocator::new(16, 0).get_total_blocks(), 0);
        assert_eq!(BlockAllocator::new(16, 1).get_total_blocks(), 1);
        assert_eq!(BlockAllocator::new(16, 50).get_total_blocks(), 50);
        assert_eq!(BlockAllocator::new(16, 500).get_total_blocks(), 500);
    }

    #[test]
    fn block_size_correct_for_various_sizes() {
        // Arrange & Assert — verify constructor stores block_size correctly
        assert_eq!(BlockAllocator::new(0, 10).get_block_size(), 0);
        assert_eq!(BlockAllocator::new(1, 10).get_block_size(), 1);
        assert_eq!(BlockAllocator::new(64, 10).get_block_size(), 64);
        assert_eq!(BlockAllocator::new(4096, 10).get_block_size(), 4096);
    }

    #[test]
    fn allocate_from_fresh_allocator_returns_zero_then_one_then_two() {
        // Arrange — verify the exact ID sequence from a fresh allocator
        let mut alloc = BlockAllocator::new(32, 5);

        // Act & Assert — IDs are strictly sequential from 0
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        // Verify free count decremented each time
        assert_eq!(alloc.get_num_free_blocks(), 2);
    }

    #[test]
    fn free_middle_block_then_allocate_returns_middle_id() {
        // Arrange — 5 blocks, allocate all
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free only block 2 (the middle)
        alloc.free(2);

        // Assert — exactly block 2 is returned
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn allocate_drains_free_list_until_empty_then_consistent_none() {
        // Arrange — 3 blocks
        let mut alloc = BlockAllocator::new(16, 3);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));

        // Act — verify 5 consecutive None returns from exhausted allocator
        for i in 0..5 {
            assert_eq!(alloc.allocate(), None, "iteration {} should be None", i);
        }

        // Assert — free count is still 0
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_after_partial_allocation_preserves_unallocated_block_order() {
        // Arrange — 6 blocks, allocate 3 (IDs 0, 1, 2), free block 1
        // Free list before free: [3, 4, 5]
        // Free list after free(1): [3, 4, 5, 1]
        let mut alloc = BlockAllocator::new(16, 6);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2

        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Act — allocate all 4 remaining
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();
        let d = alloc.allocate();

        // Assert — unallocated blocks first, then freed block
        assert_eq!(a, Some(3));
        assert_eq!(b, Some(4));
        assert_eq!(c, Some(5));
        assert_eq!(d, Some(1));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn full_lifecycle_allocate_all_free_all_reallocate_all_verify_ids() {
        // Arrange
        let total = 8;
        let mut alloc = BlockAllocator::new(16, total);

        // Phase 1: allocate all, verify IDs
        let phase1: Vec<usize> = (0..total).map(|_| alloc.allocate().unwrap()).collect();
        let mut sorted1 = phase1.clone();
        sorted1.sort();
        assert_eq!(sorted1, (0..total).collect::<Vec<_>>());

        // Phase 2: free all in order
        for b in &phase1 {
            alloc.free(*b);
        }
        assert_eq!(alloc.get_num_free_blocks(), total);

        // Phase 3: re-allocate all, verify same set of IDs
        let phase3: Vec<usize> = (0..total).map(|_| alloc.allocate().unwrap()).collect();
        let mut sorted3 = phase3.clone();
        sorted3.sort();
        assert_eq!(sorted3, (0..total).collect::<Vec<_>>());

        // Assert — exhausted again
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    // --- 13 new edge-case tests (target: 107 total) ---

    #[test]
    fn free_at_total_blocks_minus_one_accepted_total_blocks_plus_one_rejected() {
        // Arrange — 7 blocks, valid IDs 0..=6
        let mut alloc = BlockAllocator::new(16, 7);
        for _ in 0..7 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free highest valid ID and lowest invalid ID back to back
        alloc.free(6);  // valid: total_blocks - 1
        alloc.free(7);  // invalid: total_blocks

        // Assert — only the valid free counted
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(6));
    }

    #[test]
    fn single_block_double_free_triple_allocate_drains_to_none() {
        // Arrange — 1 block, allocate it, double-free
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate();
        alloc.free(0);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Act — allocate three times
        let first = alloc.allocate();
        let second = alloc.allocate();
        let third = alloc.allocate();

        // Assert — first two succeed (from double-free entries), third fails
        assert_eq!(first, Some(0));
        assert_eq!(second, Some(0));
        assert_eq!(third, None);
    }

    #[test]
    fn free_preserves_order_of_remaining_unallocated_blocks() {
        // Arrange — 5 blocks, allocate first 2 (IDs 0, 1)
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        // Free list: [2, 3, 4]

        // Act — free block 0
        alloc.free(0);
        // Free list: [2, 3, 4, 0]

        // Assert — unallocated blocks 2, 3, 4 served before freed block 0
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn triple_exhaust_cycle_with_scattered_frees_each_cycle() {
        // Arrange — 3 blocks
        let mut alloc = BlockAllocator::new(16, 3);

        for cycle in 0..3 {
            // Exhaust
            let mut blocks = Vec::new();
            while let Some(b) = alloc.allocate() {
                blocks.push(b);
            }
            assert_eq!(blocks.len(), 3, "cycle {} should allocate all 3", cycle);

            // Free in non-sequential order: middle, first, last
            alloc.free(blocks[1]);
            alloc.free(blocks[0]);
            alloc.free(blocks[2]);

            // Re-allocate and verify FIFO order matches free order
            let r1 = alloc.allocate();
            let r2 = alloc.allocate();
            let r3 = alloc.allocate();
            assert_eq!(r1, Some(blocks[1]), "cycle {} first reclaim", cycle);
            assert_eq!(r2, Some(blocks[0]), "cycle {} second reclaim", cycle);
            assert_eq!(r3, Some(blocks[2]), "cycle {} third reclaim", cycle);

            // Free all for next cycle
            alloc.free(blocks[1]);
            alloc.free(blocks[0]);
            alloc.free(blocks[2]);
        }
    }

    #[test]
    fn allocate_none_then_free_then_allocate_succeeds_recovery() {
        // Arrange — 2 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 2);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();

        // Attempt allocation when exhausted
        assert!(alloc.allocate().is_none());
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free one block
        alloc.free(b1);

        // Assert — recovery: exactly one allocation succeeds
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(b1));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Cleanup
        alloc.free(b0);
    }

    #[test]
    fn block_size_zero_with_multiple_blocks_allocate_deallocate_cycle() {
        // Arrange — block_size=0, 3 blocks
        let mut alloc = BlockAllocator::new(0, 3);
        assert_eq!(alloc.get_block_size(), 0);
        assert_eq!(alloc.get_total_blocks(), 3);

        // Act — full allocate-deallocate cycle
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();
        let d = alloc.allocate();

        assert_eq!(a, Some(0));
        assert_eq!(b, Some(1));
        assert_eq!(c, Some(2));
        assert_eq!(d, None);

        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Assert — block_size=0 does not affect allocation mechanics
        let reclaimed = alloc.allocate();
        assert_eq!(reclaimed, Some(1));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_order_determines_reallocation_order_fifo_guarantee() {
        // Arrange — 4 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free in exact sequence: 3, 0, 2, 1
        alloc.free(3);
        alloc.free(0);
        alloc.free(2);
        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Assert — allocations follow exact FIFO free order
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn partial_alloc_free_corner_block_verify_isolated_reuse() {
        // Arrange — 5 blocks, allocate only the corner blocks (first and last)
        let mut alloc = BlockAllocator::new(16, 5);
        let b0 = alloc.allocate().unwrap(); // 0
        // Skip 1, 2, 3
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        alloc.allocate(); // 3
        let b4 = alloc.allocate().unwrap(); // 4
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free only the corner blocks
        alloc.free(b0); // ID 0
        alloc.free(b4); // ID 4
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — only corner IDs are returned, in free order
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn out_of_range_free_with_usize_max_on_non_exhausted_allocator() {
        // Arrange — 3 blocks, only 1 allocated
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Act — free several out-of-range IDs including usize::MAX
        alloc.free(3);
        alloc.free(usize::MAX);
        alloc.free(1_000_000);

        // Assert — free count unchanged, untouched blocks still available
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn allocate_all_free_one_reallocate_one_still_exhausted_for_rest() {
        // Arrange — 4 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }

        // Act — free exactly block 2
        alloc.free(2);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        let reclaimed = alloc.allocate();
        assert_eq!(reclaimed, Some(2));
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Assert — allocator is fully exhausted again
        assert!(alloc.allocate().is_none());
        assert!(alloc.allocate().is_none());
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_and_reallocate_same_block_multiple_times_sequentially() {
        // Arrange — 1 block, allocate it first
        let mut alloc = BlockAllocator::new(16, 1);
        let _b = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act & Assert — 5 sequential free-then-reallocate cycles for the same block
        for i in 0..5 {
            alloc.free(0);
            assert_eq!(alloc.get_num_free_blocks(), 1, "cycle {} free failed", i);
            let b = alloc.allocate();
            assert_eq!(b, Some(0), "cycle {} alloc wrong ID", i);
            assert_eq!(alloc.get_num_free_blocks(), 0, "cycle {} not exhausted", i);
            assert!(alloc.allocate().is_none(), "cycle {} should be None", i);
        }
    }

    #[test]
    fn two_allocators_cross_free_has_no_effect() {
        // Arrange — allocator A with 3 blocks, allocator B with 3 blocks
        let mut alloc_a = BlockAllocator::new(16, 3);
        let mut alloc_b = BlockAllocator::new(16, 3);

        // Exhaust A
        let a0 = alloc_a.allocate().unwrap(); // 0
        let a1 = alloc_a.allocate().unwrap(); // 1
        let a2 = alloc_a.allocate().unwrap(); // 2
        assert_eq!(alloc_a.get_num_free_blocks(), 0);

        // B only has 1 allocated
        let _b0 = alloc_b.allocate().unwrap(); // 0
        assert_eq!(alloc_b.get_num_free_blocks(), 2);

        // Act — free A's block ID 1 into B (valid range for B, but wrong allocator)
        alloc_b.free(a1); // ID 1 is valid for B, so B's count increases
        assert_eq!(alloc_b.get_num_free_blocks(), 3); // B gains an entry

        // Free A's block ID 1 into A (correct)
        alloc_a.free(a1);
        assert_eq!(alloc_a.get_num_free_blocks(), 1);

        // Assert — A only has block 1 available
        assert_eq!(alloc_a.allocate(), Some(1));
        assert_eq!(alloc_a.get_num_free_blocks(), 0);

        // Cleanup
        alloc_a.free(a0);
        alloc_a.free(a2);
    }

    #[test]
    fn large_block_count_free_first_half_reallocate_preserves_order() {
        // Arrange — 50 blocks, exhaust all
        let total = 50;
        let mut alloc = BlockAllocator::new(16, total);
        for _ in 0..total {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free first half (IDs 0..25)
        for id in 0..25 {
            alloc.free(id);
        }
        assert_eq!(alloc.get_num_free_blocks(), 25);

        // Re-allocate all freed blocks
        let mut reclaimed = Vec::with_capacity(25);
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }
        assert_eq!(reclaimed.len(), 25);

        // Assert — FIFO order matches free order: 0, 1, 2, ..., 24
        let expected: Vec<usize> = (0..25).collect();
        assert_eq!(reclaimed, expected);

        // Assert — no more blocks available
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    // --- 13 new edge-case tests (target: 120 total) ---

    #[test]
    fn fresh_allocator_free_never_allocated_in_range_block_produces_extra_entry() {
        // Arrange — 3 blocks, never allocate; block 1 was never handed out
        let mut alloc = BlockAllocator::new(16, 3);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act — "free" block 1 which is still in the free list (duplicate injection)
        alloc.free(1);

        // Assert — free list now has 4 entries: [0, 1, 2, 1]
        assert_eq!(alloc.get_num_free_blocks(), 4);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(1)); // the duplicate
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn constructor_with_large_total_produces_contiguous_zero_based_ids() {
        // Arrange — 500 blocks
        let total = 500;
        let mut alloc = BlockAllocator::new(16, total);

        // Act — allocate first 10, verify they are 0..10
        let first_ten: Vec<usize> = (0..10).map(|_| alloc.allocate().unwrap()).collect();

        // Assert — IDs are contiguous starting from 0
        assert_eq!(first_ten, (0..10).collect::<Vec<_>>());
        assert_eq!(alloc.get_num_free_blocks(), total - 10);

        // Free count invariant
        assert_eq!(alloc.get_num_free_blocks() + 10, alloc.get_total_blocks());
    }

    #[test]
    fn block_size_usize_max_still_allows_allocation_by_count() {
        // Arrange — block_size at usize extreme
        let mut alloc = BlockAllocator::new(usize::MAX, 2);
        assert_eq!(alloc.get_block_size(), usize::MAX);

        // Act
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();

        // Assert — allocation purely count-driven, block_size is passive metadata
        assert_eq!(a, Some(0));
        assert_eq!(b, Some(1));
        assert_eq!(c, None);
    }

    #[test]
    fn free_descending_ids_reallocate_returns_descending_fifo_order() {
        // Arrange — 4 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free in descending order: 3, 2, 1, 0
        alloc.free(3);
        alloc.free(2);
        alloc.free(1);
        alloc.free(0);

        // Assert — FIFO returns blocks in the exact free order (3, 2, 1, 0)
        assert_eq!(alloc.get_num_free_blocks(), 4);
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn tight_single_block_reuse_five_cycles_verify_intermediate_states() {
        // Arrange — 1 block
        let mut alloc = BlockAllocator::new(16, 1);

        // Act & Assert — 5 cycles, verifying every intermediate state
        for i in 0..5 {
            let b = alloc.allocate();
            assert_eq!(b, Some(0), "cycle {}: allocate should return 0", i);
            assert_eq!(alloc.get_num_free_blocks(), 0, "cycle {}: should be exhausted", i);
            assert!(alloc.allocate().is_none(), "cycle {}: second alloc should fail", i);

            alloc.free(0);
            assert_eq!(alloc.get_num_free_blocks(), 1, "cycle {}: should have 1 free", i);
        }
    }

    #[test]
    fn exhaust_allocator_a_verify_allocator_b_still_fully_functional() {
        // Arrange — two independent allocators
        let mut alloc_a = BlockAllocator::new(16, 2);
        let mut alloc_b = BlockAllocator::new(16, 4);

        // Act — exhaust A completely
        let _a0 = alloc_a.allocate();
        let _a1 = alloc_a.allocate();
        assert_eq!(alloc_a.get_num_free_blocks(), 0);

        // Assert — B is completely unaffected
        assert_eq!(alloc_b.get_num_free_blocks(), 4);
        assert_eq!(alloc_b.get_total_blocks(), 4);
        assert_eq!(alloc_b.get_block_size(), 16);

        let b0 = alloc_b.allocate();
        assert_eq!(b0, Some(0));
        assert_eq!(alloc_b.get_num_free_blocks(), 3);

        // A still exhausted
        assert!(alloc_a.allocate().is_none());
    }

    #[test]
    fn free_block_zero_on_virgin_zero_block_allocator_ignored() {
        // Arrange — zero-block allocator, all IDs are out of range
        let mut alloc = BlockAllocator::new(16, 0);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 (out of range since 0 >= total_blocks=0)
        alloc.free(0);

        // Assert — still zero free blocks, allocation still fails
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn allocate_three_free_middle_then_end_verify_only_freed_ids_available() {
        // Arrange — 4 blocks, allocate all
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free only IDs 1 and 3 (middle and end)
        alloc.free(1);
        alloc.free(3);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — only the freed IDs are available, in FIFO order
        let r1 = alloc.allocate();
        let r2 = alloc.allocate();
        let r3 = alloc.allocate();

        assert_eq!(r1, Some(1));
        assert_eq!(r2, Some(3));
        assert_eq!(r3, None); // IDs 0 and 2 still held
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_same_block_repeatedly_between_allocates_drains_correctly() {
        // Arrange — 1 block
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — triple free block 0, then allocate twice, free once, allocate once
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        let first = alloc.allocate();
        let second = alloc.allocate();
        assert_eq!(first, Some(0));
        assert_eq!(second, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 1);

        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        let third = alloc.allocate();
        assert_eq!(third, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Assert — one entry remains
        let fourth = alloc.allocate();
        assert_eq!(fourth, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn get_total_blocks_and_get_block_size_independent_of_each_other() {
        // Arrange — verify that the two metadata fields don't influence each other
        let a1 = BlockAllocator::new(1, 100);
        let a2 = BlockAllocator::new(100, 1);
        let a3 = BlockAllocator::new(0, 0);
        let a4 = BlockAllocator::new(usize::MAX, 1); // large block_size, small count

        // Assert — each allocator stores its own parameters independently
        assert_eq!(a1.get_block_size(), 1);
        assert_eq!(a1.get_total_blocks(), 100);

        assert_eq!(a2.get_block_size(), 100);
        assert_eq!(a2.get_total_blocks(), 1);

        assert_eq!(a3.get_block_size(), 0);
        assert_eq!(a3.get_total_blocks(), 0);

        assert_eq!(a4.get_block_size(), usize::MAX);
        assert_eq!(a4.get_total_blocks(), 1);
    }

    #[test]
    fn partial_allocate_free_reallocate_verify_mixed_id_sources() {
        // Arrange — 5 blocks, allocate first 3 (IDs 0, 1, 2)
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        // Free list: [3, 4]

        // Act — free block 1 (from allocated set)
        alloc.free(1);
        // Free list: [3, 4, 1]

        // Allocate 3 blocks — should get untouched 3, 4 then freed 1
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();

        // Assert — unallocated pool drained before freed block
        assert_eq!(a, Some(3));
        assert_eq!(b, Some(4));
        assert_eq!(c, Some(1));

        // Now free block 1 again, and block 0
        alloc.free(1);
        alloc.free(0);
        // Free list: [1, 0]
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(0));
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn single_block_allocated_then_freed_then_freed_again_count_is_two() {
        // Arrange — 1 block, allocate then double-free
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free twice
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — both entries are block 0
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn allocator_with_total_two_full_cycle_verify_id_set_after_each_phase() {
        // Arrange — 2 blocks
        let mut alloc = BlockAllocator::new(16, 2);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Phase 1: allocate both
        let p1_a = alloc.allocate();
        let p1_b = alloc.allocate();
        assert_eq!(p1_a, Some(0));
        assert_eq!(p1_b, Some(1));
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Phase 2: free both in reverse
        alloc.free(1);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Phase 3: re-allocate both — FIFO order is [1, 0]
        let p3_a = alloc.allocate();
        let p3_b = alloc.allocate();
        assert_eq!(p3_a, Some(1)); // freed first
        assert_eq!(p3_b, Some(0)); // freed second
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Phase 4: free both again in forward order
        alloc.free(1);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Phase 5: final allocation — FIFO order is [1, 0] again
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    // --- 13 new tests (target: 133 total) ---

    #[test]
    fn free_during_partial_allocation_does_not_skip_unallocated_blocks() {
        // Arrange — 6 blocks, allocate 3 (IDs 0,1,2), free ID 1, then allocate 4 more
        // Free list after alloc 3: [3,4,5]
        // After free(1): [3,4,5,1]
        let mut alloc = BlockAllocator::new(16, 6);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2

        alloc.free(1);
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // Act — allocate 4 blocks
        let r1 = alloc.allocate();
        let r2 = alloc.allocate();
        let r3 = alloc.allocate();
        let r4 = alloc.allocate();

        // Assert — untouched blocks 3,4,5 served first, then freed block 1
        assert_eq!(r1, Some(3));
        assert_eq!(r2, Some(4));
        assert_eq!(r3, Some(5));
        assert_eq!(r4, Some(1));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn triple_free_of_same_block_yields_three_consecutive_allocations() {
        // Arrange — 2 blocks, exhaust, then triple-free block 1
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — triple free
        alloc.free(1);
        alloc.free(1);
        alloc.free(1);

        // Assert — three entries of block 1 in free list
        assert_eq!(alloc.get_num_free_blocks(), 3);
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn block_size_query_stable_across_ten_alloc_free_cycles() {
        // Arrange — block_size=256, verify it never changes across cycles
        let mut alloc = BlockAllocator::new(256, 3);

        for _ in 0..10 {
            // Act
            let a = alloc.allocate();
            let b = alloc.allocate();
            let c = alloc.allocate();

            // Assert — block_size invariant after allocation
            assert_eq!(alloc.get_block_size(), 256);
            assert!(a.is_some());
            assert!(b.is_some());
            assert!(c.is_some());

            alloc.free(a.unwrap());
            alloc.free(b.unwrap());
            alloc.free(c.unwrap());

            // Assert — block_size invariant after free
            assert_eq!(alloc.get_block_size(), 256);
        }
    }

    #[test]
    fn allocate_returns_none_after_exhaustion_then_free_restores_single_slot() {
        // Arrange — 3 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 3);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();

        // Confirm exhaustion with 4 attempts
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.allocate(), None);

        // Act — free just block b1
        alloc.free(b1);

        // Assert — exactly one allocation succeeds, returns b1
        assert_eq!(alloc.allocate(), Some(b1));
        assert_eq!(alloc.allocate(), None);

        // Cleanup
        alloc.free(b0);
        alloc.free(b2);
    }

    #[test]
    fn free_list_accumulates_in_exact_push_order_across_mixed_frees() {
        // Arrange — 5 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 5);
        for _ in 0..5 {
            alloc.allocate();
        }

        // Act — free in order: 4, 0, 2
        alloc.free(4);
        alloc.free(0);
        alloc.free(2);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Assert — allocations follow exact FIFO push order
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn two_independent_allocators_concurrent_lifecycle_no_cross_pollution() {
        // Arrange
        let mut alloc_x = BlockAllocator::new(32, 3);
        let mut alloc_y = BlockAllocator::new(64, 2);

        // Act — X: allocate 2, free 1
        let x0 = alloc_x.allocate().unwrap();
        let x1 = alloc_x.allocate().unwrap();
        alloc_x.free(x0);
        assert_eq!(alloc_x.get_num_free_blocks(), 2);

        // Y: allocate all, free all
        let y0 = alloc_y.allocate().unwrap();
        let y1 = alloc_y.allocate().unwrap();
        alloc_y.free(y0);
        alloc_y.free(y1);
        assert_eq!(alloc_y.get_num_free_blocks(), 2);

        // X re-allocate — should get untouched block 2 first, then freed block 0
        let x2 = alloc_x.allocate().unwrap();
        assert_eq!(x2, 2);

        // Assert — Y's state is completely independent
        assert_eq!(alloc_y.get_block_size(), 64);
        assert_eq!(alloc_y.get_total_blocks(), 2);
        let y_new = alloc_y.allocate().unwrap();
        assert_eq!(y_new, y0); // FIFO from Y's own free list

        // X still has block x0 available
        let x3 = alloc_x.allocate().unwrap();
        assert_eq!(x3, x0);

        // Cleanup
        alloc_x.free(x1);
        alloc_x.free(x2);
        alloc_x.free(x3);
        alloc_y.free(y_new);
        alloc_y.free(y1);
    }

    #[test]
    fn free_only_first_block_then_allocate_verifies_first_block_returned() {
        // Arrange — 3 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2

        // Act — free only block 0
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Assert — exactly block 0 is returned
        let result = alloc.allocate();
        assert_eq!(result, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn allocate_one_from_large_pool_verify_remaining_count_accurate() {
        // Arrange — 200 blocks
        let total = 200;
        let mut alloc = BlockAllocator::new(16, total);

        // Act — allocate exactly 1 block
        let b = alloc.allocate();
        assert_eq!(b, Some(0));

        // Assert — free count decremented by exactly 1
        assert_eq!(alloc.get_num_free_blocks(), total - 1);

        // total_blocks invariant
        assert_eq!(alloc.get_total_blocks(), total);

        // Next allocation returns ID 1
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.get_num_free_blocks(), total - 2);
    }

    #[test]
    fn exhaust_via_loop_then_free_every_third_block_verify_correct_count() {
        // Arrange — 9 blocks, exhaust
        let total = 9;
        let mut alloc = BlockAllocator::new(16, total);
        for _ in 0..total {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free every 3rd block: 0, 3, 6
        let expected_freed = 3;
        for id in (0..total).step_by(3) {
            alloc.free(id);
        }

        // Assert — exactly 3 blocks freed
        assert_eq!(alloc.get_num_free_blocks(), expected_freed);

        // Re-allocate and verify the exact IDs
        let mut reclaimed = Vec::new();
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }
        assert_eq!(reclaimed, vec![0, 3, 6]);
    }

    #[test]
    fn single_block_exhaust_then_free_non_existent_id_still_exhausted() {
        // Arrange — 1 block
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        assert!(alloc.allocate().is_none());

        // Act — free an out-of-range ID
        alloc.free(1); // 1 >= total_blocks=1, ignored

        // Assert — still exhausted
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Free the actual block
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(0));
    }

    #[test]
    fn allocate_from_two_separate_pools_verify_no_id_collision() {
        // Arrange — two independent allocators with overlapping ID spaces
        let mut pool_a = BlockAllocator::new(16, 3);
        let mut pool_b = BlockAllocator::new(16, 3);

        // Act — both start allocating from ID 0
        let a_ids: Vec<usize> = (0..3).map(|_| pool_a.allocate().unwrap()).collect();
        let b_ids: Vec<usize> = (0..3).map(|_| pool_b.allocate().unwrap()).collect();

        // Assert — each pool produced its own [0,1,2] sequence independently
        assert_eq!(a_ids, vec![0, 1, 2]);
        assert_eq!(b_ids, vec![0, 1, 2]);

        // Free all of pool A, none of pool B
        for id in &a_ids {
            pool_a.free(*id);
        }
        assert_eq!(pool_a.get_num_free_blocks(), 3);
        assert_eq!(pool_b.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_list_content_after_allocate_all_free_two_reallocate_two_free_rest() {
        // Arrange — 4 blocks
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free blocks 2 and 0
        alloc.free(2);
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Re-allocate — should get 2, 0 (FIFO)
        let r1 = alloc.allocate();
        let r2 = alloc.allocate();
        assert_eq!(r1, Some(2));
        assert_eq!(r2, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Free the remaining held blocks (1 and 3) plus the re-allocated ones
        alloc.free(1);
        alloc.free(3);
        alloc.free(2);
        alloc.free(0);

        // Assert — all 4 blocks back
        assert_eq!(alloc.get_num_free_blocks(), 4);

        // FIFO order: 1, 3, 2, 0
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_block_id_just_above_valid_range_boundary_ignored_below_accepted() {
        // Arrange — total_blocks=8, valid IDs 0..=7
        let mut alloc = BlockAllocator::new(16, 8);
        for _ in 0..8 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free ID 7 (valid, just below boundary)
        alloc.free(7);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Free ID 8 (invalid, exactly at boundary)
        alloc.free(8);
        assert_eq!(alloc.get_num_free_blocks(), 1); // unchanged

        // Free ID 9 (invalid, above boundary)
        alloc.free(9);
        assert_eq!(alloc.get_num_free_blocks(), 1); // unchanged

        // Free ID 6 (valid, below boundary)
        alloc.free(6);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — FIFO returns 7 then 6
        assert_eq!(alloc.allocate(), Some(7));
        assert_eq!(alloc.allocate(), Some(6));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    // --- 13 new edge-case tests (target: 146 total) ---

    #[test]
    fn allocate_exactly_half_free_all_reallocate_verify_fifo_order() {
        // Arrange — 10 blocks, allocate exactly 5 (IDs 0..4)
        let mut alloc = BlockAllocator::new(16, 10);
        let first_batch: Vec<usize> = (0..5).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(first_batch, vec![0, 1, 2, 3, 4]);
        assert_eq!(alloc.get_num_free_blocks(), 5);

        // Act — free all 5 in reverse order
        for &b in first_batch.iter().rev() {
            alloc.free(b);
        }
        // Free list: [5,6,7,8,9,4,3,2,1,0]
        assert_eq!(alloc.get_num_free_blocks(), 10);

        // Assert — untouched blocks served first, then freed blocks in reverse-free order
        let second_batch: Vec<usize> = (0..10).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(second_batch, vec![5, 6, 7, 8, 9, 4, 3, 2, 1, 0]);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_list_after_exhaust_free_odd_then_even_reallocate_verify_interleaved_order() {
        // Arrange — 6 blocks, exhaust all
        let mut alloc = BlockAllocator::new(16, 6);
        for _ in 0..6 {
            alloc.allocate();
        }

        // Act — free odds first (1,3,5), then evens (0,2,4)
        alloc.free(1);
        alloc.free(3);
        alloc.free(5);
        alloc.free(0);
        alloc.free(2);
        alloc.free(4);
        assert_eq!(alloc.get_num_free_blocks(), 6);

        // Assert — FIFO: 1,3,5,0,2,4
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn block_size_and_total_blocks_queries_invariant_under_rapid_mutations() {
        // Arrange — verify both metadata queries are stable across 20 rapid mutations
        let mut alloc = BlockAllocator::new(1024, 5);
        let expected_block_size = 1024;
        let expected_total = 5;

        // Act & Assert
        for _ in 0..10 {
            let b = alloc.allocate().unwrap();
            assert_eq!(alloc.get_block_size(), expected_block_size);
            assert_eq!(alloc.get_total_blocks(), expected_total);
            alloc.free(b);
            assert_eq!(alloc.get_block_size(), expected_block_size);
            assert_eq!(alloc.get_total_blocks(), expected_total);
        }
    }

    #[test]
    fn two_allocators_same_params_diverge_after_asymmetric_frees() {
        // Arrange — two allocators with identical configuration
        let mut alloc_a = BlockAllocator::new(64, 4);
        let mut alloc_b = BlockAllocator::new(64, 4);

        // Exhaust both
        let a_blocks: Vec<usize> = (0..4).map(|_| alloc_a.allocate().unwrap()).collect();
        let b_blocks: Vec<usize> = (0..4).map(|_| alloc_b.allocate().unwrap()).collect();
        assert_eq!(alloc_a.get_num_free_blocks(), 0);
        assert_eq!(alloc_b.get_num_free_blocks(), 0);

        // Act — A: free only block 0; B: free only block 3
        alloc_a.free(a_blocks[0]);
        alloc_b.free(b_blocks[3]);

        // Assert — different blocks available from each allocator
        assert_eq!(alloc_a.allocate(), Some(0));
        assert_eq!(alloc_b.allocate(), Some(3));
        assert_eq!(alloc_a.get_num_free_blocks(), 0);
        assert_eq!(alloc_b.get_num_free_blocks(), 0);
    }

    #[test]
    fn exhaust_via_single_loop_then_bulk_free_then_bulk_allocate_verify_set() {
        // Arrange — 7 blocks
        let total = 7;
        let mut alloc = BlockAllocator::new(16, total);

        // Act — exhaust via loop
        let mut blocks = Vec::with_capacity(total);
        while let Some(b) = alloc.allocate() {
            blocks.push(b);
        }
        assert_eq!(blocks.len(), total);
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Bulk free
        for b in &blocks {
            alloc.free(*b);
        }
        assert_eq!(alloc.get_num_free_blocks(), total);

        // Bulk re-allocate
        let mut reclaimed = Vec::with_capacity(total);
        while let Some(b) = alloc.allocate() {
            reclaimed.push(b);
        }

        // Assert — same set of IDs (order may differ due to FIFO, but set is identical)
        assert_eq!(reclaimed.len(), total);
        let mut sorted = reclaimed;
        sorted.sort();
        assert_eq!(sorted, (0..total).collect::<Vec<_>>());
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_block_id_zero_on_fresh_allocator_increments_count_beyond_total() {
        // Arrange — 2 blocks, never allocate anything
        let mut alloc = BlockAllocator::new(16, 2);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Act — "free" block 0 on a fresh allocator (it is still in the free list)
        alloc.free(0);

        // Assert — count goes above total due to duplicate injection
        assert_eq!(alloc.get_num_free_blocks(), 3);
        // FIFO: [0, 1, 0]
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn allocate_one_free_it_then_allocate_two_verify_fifo_ordering() {
        // Arrange — 3 blocks
        let mut alloc = BlockAllocator::new(16, 3);

        // Allocate block 0, then free it
        let b0 = alloc.allocate().unwrap();
        assert_eq!(b0, 0);
        alloc.free(b0);
        // Free list: [1, 2, 0]

        // Act — allocate 2 blocks
        let first = alloc.allocate();
        let second = alloc.allocate();

        // Assert — untouched blocks 1 and 2 served before freed block 0
        assert_eq!(first, Some(1));
        assert_eq!(second, Some(2));
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // The remaining free block is 0
        assert_eq!(alloc.allocate(), Some(0));
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn free_at_boundary_minus_two_accepted_boundary_rejected_across_multiple_sizes() {
        // Arrange — test boundary behavior for several allocator sizes
        for &total in &[1, 2, 5, 10, 50] {
            let mut alloc = BlockAllocator::new(16, total);
            for _ in 0..total {
                alloc.allocate();
            }
            assert_eq!(alloc.get_num_free_blocks(), 0);

            // Act — free total-2 (valid if total >= 2) and total (always invalid)
            if total >= 2 {
                alloc.free(total - 2);
                assert_eq!(alloc.get_num_free_blocks(), 1, "total={}: valid free failed", total);
            }
            alloc.free(total); // always out of range
            assert_eq!(
                alloc.get_num_free_blocks(),
                if total >= 2 { 1 } else { 0 },
                "total={}: invalid free should not count",
                total
            );
        }
    }

    #[test]
    fn partial_allocation_then_free_last_allocated_preserves_front_of_free_list() {
        // Arrange — 5 blocks, allocate 3 (IDs 0,1,2)
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        // Free list: [3, 4]

        // Act — free the most recently allocated block (ID 2)
        alloc.free(2);
        // Free list: [3, 4, 2]
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Assert — front of free list still has untouched blocks
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(2)); // freed block last
    }

    #[test]
    fn single_block_double_free_then_two_allocates_drain_completely() {
        // Arrange — 1 block
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        alloc.free(0);
        alloc.free(0); // double free -> 2 entries
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Act — allocate twice
        let first = alloc.allocate();
        let second = alloc.allocate();
        let third = alloc.allocate();

        // Assert — both succeed (draining duplicates), third fails
        assert_eq!(first, Some(0));
        assert_eq!(second, Some(0));
        assert_eq!(third, None);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn get_total_blocks_returns_constructor_value_for_various_configs() {
        // Arrange & Assert — spot-check several configurations
        for &(bs, tb) in &[(1, 0), (16, 1), (64, 100), (4096, 1000)] {
            let alloc = BlockAllocator::new(bs, tb);
            assert_eq!(alloc.get_total_blocks(), tb, "total_blocks mismatch for bs={}, tb={}", bs, tb);
            assert_eq!(alloc.get_block_size(), bs, "block_size mismatch for bs={}, tb={}", bs, tb);
            assert_eq!(alloc.get_num_free_blocks(), tb, "initial free mismatch for bs={}, tb={}", bs, tb);
        }
    }

    #[test]
    fn free_block_then_alloc_same_block_verify_exact_id_roundtrip() {
        // Arrange — 8 blocks, allocate all, free block 5
        let mut alloc = BlockAllocator::new(16, 8);
        for _ in 0..8 {
            alloc.allocate();
        }
        alloc.free(5);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Act — re-allocate
        let reclaimed = alloc.allocate();

        // Assert — exact roundtrip: freed 5, got back 5
        assert_eq!(reclaimed, Some(5));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn interleaved_three_allocator_lifecycle_verify_full_independence() {
        // Arrange — three allocators with different sizes
        let mut alloc_a = BlockAllocator::new(16, 2);
        let mut alloc_b = BlockAllocator::new(32, 3);
        let mut alloc_c = BlockAllocator::new(64, 1);

        // Act — A: allocate 1, free 0
        let a0 = alloc_a.allocate().unwrap();
        assert_eq!(a0, 0);
        alloc_a.free(a0);

        // B: exhaust, free middle
        let b0 = alloc_b.allocate().unwrap();
        let b1 = alloc_b.allocate().unwrap();
        let b2 = alloc_b.allocate().unwrap();
        alloc_b.free(b1);

        // C: allocate, free, double-free
        alloc_c.allocate();
        alloc_c.free(0);
        alloc_c.free(0);

        // Assert — each allocator's state is independent
        assert_eq!(alloc_a.get_num_free_blocks(), 2);
        assert_eq!(alloc_a.get_block_size(), 16);

        assert_eq!(alloc_b.get_num_free_blocks(), 1);
        assert_eq!(alloc_b.allocate(), Some(1)); // only b1 was freed
        assert_eq!(alloc_b.get_num_free_blocks(), 0);
        assert_eq!(alloc_b.get_block_size(), 32);

        assert_eq!(alloc_c.get_num_free_blocks(), 2); // double free
        assert_eq!(alloc_c.allocate(), Some(0));
        assert_eq!(alloc_c.allocate(), Some(0));
        assert_eq!(alloc_c.allocate(), None);
        assert_eq!(alloc_c.get_block_size(), 64);
    }

    // --- 13 new tests (target: 159 total) ---

    #[test]
    fn allocate_all_then_free_block_zero_then_block_max_verify_fifo_serves_zero_first() {
        // Arrange — 6 blocks, exhaust all
        let mut alloc = BlockAllocator::new(16, 6);
        for _ in 0..6 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 first, then block 5 (max ID)
        alloc.free(0);
        alloc.free(5);

        // Assert — FIFO: block 0 served first, then block 5
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_block_on_fresh_allocator_with_single_block_exceeds_total() {
        // Arrange — 1 block, never allocate
        let mut alloc = BlockAllocator::new(16, 1);
        assert_eq!(alloc.get_num_free_blocks(), 1);

        // Act — free block ID 1 (out of range: 1 >= total_blocks=1)
        alloc.free(1);

        // Assert — count unchanged, only block 0 is available
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn partial_alloc_free_head_then_realloc_verify_unallocated_served_first() {
        // Arrange — 7 blocks, allocate only the first 3 (IDs 0, 1, 2)
        let mut alloc = BlockAllocator::new(16, 7);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        // Free list: [3, 4, 5, 6]

        // Act — free block 0 (appended to back)
        alloc.free(0);
        // Free list: [3, 4, 5, 6, 0]

        // Assert — untouched blocks 3, 4, 5, 6 served before freed block 0
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.allocate(), Some(6));
        assert_eq!(alloc.allocate(), Some(0)); // freed block last
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn four_block_allocator_free_two_non_adjacent_then_verify_only_freed_available() {
        // Arrange — 4 blocks, exhaust all
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free non-adjacent blocks 0 and 3
        alloc.free(0);
        alloc.free(3);

        // Assert — exactly 2 blocks available, IDs 0 and 3 in FIFO order
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn single_block_quadruple_free_yields_four_consecutive_successful_allocations() {
        // Arrange — 1 block, allocate and then free 4 times
        let mut alloc = BlockAllocator::new(16, 1);
        alloc.allocate(); // ID 0
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 four times
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);

        // Assert — 4 entries in free list, all ID 0
        assert_eq!(alloc.get_num_free_blocks(), 4);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn large_pool_allocate_one_verify_block_size_unaffected() {
        // Arrange — 500 blocks with block_size=2048
        let block_size = 2048;
        let total = 500;
        let mut alloc = BlockAllocator::new(block_size, total);

        // Act — allocate a single block
        let b = alloc.allocate();
        assert_eq!(b, Some(0));

        // Assert — block_size remains unchanged, free count decremented by 1
        assert_eq!(alloc.get_block_size(), block_size);
        assert_eq!(alloc.get_total_blocks(), total);
        assert_eq!(alloc.get_num_free_blocks(), total - 1);
    }

    #[test]
    fn exhaust_then_free_last_then_first_verify_fifo_is_last_then_first() {
        // Arrange — 3 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 3);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 2 (last allocated), then block 0 (first allocated)
        alloc.free(2);
        alloc.free(0);

        // Assert — FIFO serves block 2 first, then block 0
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn five_block_allocator_three_exhaust_recover_cycles_with_scatter_pattern() {
        // Arrange — 5 blocks
        let total = 5;
        let mut alloc = BlockAllocator::new(16, total);

        for cycle in 0..3 {
            // Exhaust
            let mut blocks = Vec::with_capacity(total);
            while let Some(b) = alloc.allocate() {
                blocks.push(b);
            }
            assert_eq!(blocks.len(), total, "cycle {} should exhaust", cycle);

            // Act — free in scattered pattern: last, second, first
            alloc.free(blocks[4]);
            alloc.free(blocks[1]);
            alloc.free(blocks[0]);
            alloc.free(blocks[3]);
            alloc.free(blocks[2]);

            // Assert — free count restored
            assert_eq!(alloc.get_num_free_blocks(), total, "cycle {} should recover", cycle);
        }
    }

    #[test]
    fn free_at_boundary_with_two_different_allocator_sizes_back_to_back() {
        // Arrange — first allocator with 3 blocks, second with 10 blocks
        let mut alloc_small = BlockAllocator::new(16, 3);
        let mut alloc_large = BlockAllocator::new(16, 10);

        // Exhaust both
        for _ in 0..3 {
            alloc_small.allocate();
        }
        for _ in 0..10 {
            alloc_large.allocate();
        }

        // Act — free boundary-1 and boundary for each
        alloc_small.free(2); // valid
        alloc_small.free(3); // invalid
        alloc_large.free(9); // valid
        alloc_large.free(10); // invalid

        // Assert — only valid frees counted for each
        assert_eq!(alloc_small.get_num_free_blocks(), 1);
        assert_eq!(alloc_large.get_num_free_blocks(), 1);
        assert_eq!(alloc_small.allocate(), Some(2));
        assert_eq!(alloc_large.allocate(), Some(9));
    }

    #[test]
    fn partial_alloc_three_of_ten_free_middle_verify_free_count_and_order() {
        // Arrange — 10 blocks, allocate 3 (IDs 0, 1, 2)
        let mut alloc = BlockAllocator::new(16, 10);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.allocate(); // 2
        // Free list: [3, 4, 5, 6, 7, 8, 9]
        assert_eq!(alloc.get_num_free_blocks(), 7);

        // Act — free block 1 (from the allocated set)
        alloc.free(1);
        // Free list: [3, 4, 5, 6, 7, 8, 9, 1]

        // Assert — count is 8, untouched blocks served before freed block
        assert_eq!(alloc.get_num_free_blocks(), 8);
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(5));
        // Skip intermediate checks, jump to the freed block
        alloc.allocate(); // 6
        alloc.allocate(); // 7
        alloc.allocate(); // 8
        alloc.allocate(); // 9
        let last = alloc.allocate();
        assert_eq!(last, Some(1)); // the freed block comes last
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn two_blocks_one_freed_verify_only_that_block_returned() {
        // Arrange — 2 blocks, allocate both
        let mut alloc = BlockAllocator::new(16, 2);
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free only block b1
        alloc.free(b1);

        // Assert — exactly block b1 is available, not b0
        assert_eq!(alloc.get_num_free_blocks(), 1);
        assert_eq!(alloc.allocate(), Some(b1));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Cleanup
        alloc.free(b0);
    }

    #[test]
    fn free_block_id_zero_twice_on_two_block_allocator_produces_two_entries() {
        // Arrange — 2 blocks, exhaust
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 twice (double free)
        alloc.free(0);
        alloc.free(0);

        // Assert — count is 2, both entries are block 0
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn total_blocks_invariant_after_partial_free_and_realloc() {
        // Arrange — 5 blocks
        let total = 5;
        let mut alloc = BlockAllocator::new(16, total);

        // Exhaust
        for _ in 0..total {
            alloc.allocate();
        }
        assert_eq!(alloc.get_total_blocks(), total);

        // Act — free 2 blocks
        alloc.free(1);
        alloc.free(3);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Re-allocate 1 block
        let _ = alloc.allocate();

        // Assert — total_blocks unchanged regardless of free/alloc activity
        assert_eq!(alloc.get_total_blocks(), total);
        assert_eq!(alloc.get_num_free_blocks(), 1);
    }

    // --- 10 new tests (target: 169 total) ---

    #[test]
    fn free_list_order_when_untouched_freed_and_injected_blocks_coexist() {
        // Arrange — 5 blocks, allocate only 2 (IDs 0, 1), leaving [2, 3, 4] untouched
        let mut alloc = BlockAllocator::new(16, 5);
        alloc.allocate(); // 0
        alloc.allocate(); // 1

        // Act — free block 0 (allocated block, appended to back)
        alloc.free(0);
        // Free list: [2, 3, 4, 0]

        // Now inject block 3 via free (it's still in the free list — duplicate)
        alloc.free(3);
        // Free list: [2, 3, 4, 0, 3]

        // Assert — FIFO order: untouched 2, original 3, untouched 4, freed 0, injected 3
        assert_eq!(alloc.get_num_free_blocks(), 5);
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn allocate_from_pool_with_two_held_blocks_free_one_hold_two_more() {
        // Arrange — 4 blocks, allocate 2, free 1, allocate 2 more, verify all 4 held
        let mut alloc = BlockAllocator::new(16, 4);

        // Act — phase 1: allocate first 2 (IDs 0, 1)
        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        assert_eq!(b0, 0);
        assert_eq!(b1, 1);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Phase 2: free b0, free list becomes [2, 3, 0]
        alloc.free(b0);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Phase 3: allocate 3 blocks — gets 2, 3, 0
        let c0 = alloc.allocate().unwrap();
        let c1 = alloc.allocate().unwrap();
        let c2 = alloc.allocate().unwrap();
        assert_eq!(c0, 2);
        assert_eq!(c1, 3);
        assert_eq!(c2, 0);
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        // Assert — held IDs are b1=1, c0=2, c1=3, c2=0: the complete set {0,1,2,3}
        let mut held = vec![b1, c0, c1, c2];
        held.sort();
        assert_eq!(held, vec![0, 1, 2, 3]);
    }

    #[test]
    fn free_count_exceeds_total_blocks_after_triple_inject_of_single_id() {
        // Arrange — 2 blocks, exhaust both
        let mut alloc = BlockAllocator::new(16, 2);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 three times (triple inject)
        alloc.free(0);
        alloc.free(0);
        alloc.free(0);

        // Assert — free_count = 3, which exceeds total_blocks = 2
        assert_eq!(alloc.get_num_free_blocks(), 3);
        assert!(alloc.get_num_free_blocks() > alloc.get_total_blocks());

        // All 3 allocations succeed (draining duplicates), then None
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn partial_alloc_free_head_free_tail_verify_fifo_serves_head_first() {
        // Arrange — 6 blocks, allocate all
        let mut alloc = BlockAllocator::new(16, 6);
        for _ in 0..6 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free block 0 (head), then block 5 (tail)
        alloc.free(0);
        alloc.free(5);

        // Assert — FIFO: block 0 served first (freed earlier), then block 5
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn allocate_from_zero_block_allocator_repeatedly_always_none() {
        // Arrange — zero-block allocator
        let mut alloc = BlockAllocator::new(16, 0);

        // Act & Assert — every allocation attempt returns None, count stays 0
        for i in 0..5 {
            assert_eq!(alloc.allocate(), None, "attempt {} should be None", i);
            assert_eq!(alloc.get_num_free_blocks(), 0, "attempt {} free count should be 0", i);
            assert_eq!(alloc.get_total_blocks(), 0);
        }
    }

    #[test]
    fn free_after_partial_alloc_then_alloc_crosses_from_untouched_into_freed() {
        // Arrange — 4 blocks, allocate 2 (IDs 0, 1), free block 0
        // Free list: [2, 3, 0]
        let mut alloc = BlockAllocator::new(16, 4);
        alloc.allocate(); // 0
        alloc.allocate(); // 1
        alloc.free(0);
        assert_eq!(alloc.get_num_free_blocks(), 3);

        // Act — allocate all 3 remaining blocks, crossing from untouched to freed
        let a = alloc.allocate();
        let b = alloc.allocate();
        let c = alloc.allocate();

        // Assert — untouched 2, 3 first, then freed 0
        assert_eq!(a, Some(2));
        assert_eq!(b, Some(3));
        assert_eq!(c, Some(0));
        assert_eq!(alloc.get_num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn double_free_of_middle_block_then_sequential_allocate_drains_both_entries() {
        // Arrange — 4 blocks, exhaust all
        let mut alloc = BlockAllocator::new(16, 4);
        for _ in 0..4 {
            alloc.allocate();
        }

        // Act — double-free block 2 (middle)
        alloc.free(2);
        alloc.free(2);
        assert_eq!(alloc.get_num_free_blocks(), 2);

        // Assert — both entries are block 2, allocated sequentially
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), None);
        assert_eq!(alloc.get_num_free_blocks(), 0);
    }

    #[test]
    fn free_block_id_zero_on_single_block_allocator_boundary_check() {
        // Arrange — 1 block, total_blocks=1 so ID 0 is the only valid ID
        let mut alloc = BlockAllocator::new(16, 1);

        // Act — free block 0 on a fresh allocator (it's in range, so accepted)
        alloc.free(0);
        // Free list now: [0, 0] — original 0 + injected 0

        // Assert — count is 2 (original + injected duplicate)
        assert_eq!(alloc.get_num_free_blocks(), 2);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), None);
    }

    #[test]
    fn two_allocators_interleaved_operations_verify_no_state_leak() {
        // Arrange — two allocators with different capacities
        let mut alloc_a = BlockAllocator::new(16, 2);
        let mut alloc_b = BlockAllocator::new(32, 4);

        // Exhaust both allocators first so free lists are empty
        let a0 = alloc_a.allocate().unwrap(); // 0
        let a1 = alloc_a.allocate().unwrap(); // 1

        let b0 = alloc_b.allocate().unwrap(); // 0
        let b1 = alloc_b.allocate().unwrap(); // 1
        let b2 = alloc_b.allocate().unwrap(); // 2
        let b3 = alloc_b.allocate().unwrap(); // 3

        assert_eq!(alloc_a.get_num_free_blocks(), 0);
        assert_eq!(alloc_b.get_num_free_blocks(), 0);

        // Act — A: free a0, B: free b1 — interleaved free operations
        alloc_a.free(a0);
        alloc_b.free(b1);

        // A: re-allocate — should get a0 back (only freed block in A)
        let a2 = alloc_a.allocate().unwrap();
        assert_eq!(a2, 0, "A should reuse its own freed block a0, not leak state from B");

        // B: re-allocate — should get b1 back (only freed block in B)
        let b4 = alloc_b.allocate().unwrap();
        assert_eq!(b4, 1, "B should reuse its own freed block b1, not leak state from A");

        // Both exhausted again
        assert_eq!(alloc_a.get_num_free_blocks(), 0);
        assert_eq!(alloc_b.get_num_free_blocks(), 0);

        // Metadata unchanged — no cross-allocator state corruption
        assert_eq!(alloc_a.get_block_size(), 16);
        assert_eq!(alloc_b.get_block_size(), 32);
        assert_eq!(alloc_a.get_total_blocks(), 2);
        assert_eq!(alloc_b.get_total_blocks(), 4);
    }

    #[test]
    fn free_all_even_indices_then_all_odd_indices_verify_concatenated_fifo_order() {
        // Arrange — 6 blocks, exhaust all
        let mut alloc = BlockAllocator::new(16, 6);
        for _ in 0..6 {
            alloc.allocate();
        }
        assert_eq!(alloc.get_num_free_blocks(), 0);

        // Act — free all even indices first (0, 2, 4), then all odd (1, 3, 5)
        alloc.free(0);
        alloc.free(2);
        alloc.free(4);
        alloc.free(1);
        alloc.free(3);
        alloc.free(5);

        // Assert — FIFO order is the exact concatenation: evens then odds
        assert_eq!(alloc.get_num_free_blocks(), 6);
        assert_eq!(alloc.allocate(), Some(0));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(4));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(5));
        assert_eq!(alloc.allocate(), None);
    }
}
