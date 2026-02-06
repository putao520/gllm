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
}
