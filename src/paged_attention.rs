use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::collections::HashMap;

/// Tokens per physical block.
pub const BLOCK_SIZE: usize = 16;

/// Physical KV block that stores a fixed number of tokens.
pub struct KVBlock<B: Backend> {
    /// Keys: [num_heads, BLOCK_SIZE, head_dim]
    pub keys: Tensor<B, 3>,
    /// Values: [num_heads, BLOCK_SIZE, head_dim]
    pub values: Tensor<B, 3>,
    /// Number of tokens filled in this block.
    pub num_tokens: usize,
}

impl<B: Backend> KVBlock<B> {
    pub fn new(num_heads: usize, head_dim: usize, device: &B::Device) -> Self {
        Self {
            keys: Tensor::zeros([num_heads, BLOCK_SIZE, head_dim], device),
            values: Tensor::zeros([num_heads, BLOCK_SIZE, head_dim], device),
            num_tokens: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens >= BLOCK_SIZE
    }

    pub fn remaining_capacity(&self) -> usize {
        BLOCK_SIZE - self.num_tokens
    }
}

/// Manages allocation and reuse of physical blocks.
pub struct BlockManager<B: Backend> {
    /// Free block indices.
    free_blocks: Vec<usize>,
    /// All blocks; free_blocks tracks availability.
    blocks: Vec<Option<KVBlock<B>>>,
    /// Configuration.
    num_heads: usize,
    head_dim: usize,
    device: B::Device,
}

impl<B: Backend> BlockManager<B> {
    pub fn new(
        max_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        device: B::Device,
    ) -> Self {
        let mut blocks = Vec::with_capacity(max_blocks);
        let mut free_blocks = Vec::with_capacity(max_blocks);

        for i in 0..max_blocks {
            blocks.push(Some(KVBlock::new(num_heads, head_dim, &device)));
            free_blocks.push(i);
        }

        Self {
            free_blocks,
            blocks,
            num_heads,
            head_dim,
            device,
        }
    }

    /// Allocate a new block, returning its index.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop()
    }

    /// Release a block back to the free list.
    pub fn free(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            if let Some(block) = &mut self.blocks[block_id] {
                block.num_tokens = 0;
            }
            self.free_blocks.push(block_id);
        }
    }

    pub fn get(&self, block_id: usize) -> Option<&KVBlock<B>> {
        self.blocks.get(block_id)?.as_ref()
    }

    pub fn get_mut(&mut self, block_id: usize) -> Option<&mut KVBlock<B>> {
        self.blocks.get_mut(block_id)?.as_mut()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Page table mapping logical blocks to physical blocks.
pub struct PageTable {
    /// block_table[logical_block_idx] = physical_block_id
    block_table: Vec<usize>,
    /// Current sequence length.
    seq_len: usize,
}

impl PageTable {
    pub fn new() -> Self {
        Self {
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    /// Get physical block id and offset for a token index.
    pub fn get_physical_location(&self, token_idx: usize) -> Option<(usize, usize)> {
        if token_idx >= self.seq_len {
            return None;
        }
        let logical_block = token_idx / BLOCK_SIZE;
        let offset = token_idx % BLOCK_SIZE;
        self.block_table
            .get(logical_block)
            .map(|&block_id| (block_id, offset))
    }

    pub fn add_block(&mut self, physical_block_id: usize) {
        self.block_table.push(physical_block_id);
    }

    pub fn extend_seq_len(&mut self, new_tokens: usize) {
        self.seq_len += new_tokens;
    }

    pub fn physical_blocks(&self) -> &[usize] {
        &self.block_table
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn needs_new_block(&self) -> bool {
        let current_capacity = self.block_table.len() * BLOCK_SIZE;
        self.seq_len >= current_capacity
    }
}

/// Paged KV cache with dynamic block allocation.
pub struct PagedKVCache<B: Backend> {
    /// Block manager.
    block_manager: BlockManager<B>,
    /// Page tables per layer and sequence.
    page_tables: Vec<HashMap<usize, PageTable>>,
    /// Number of layers.
    num_layers: usize,
    /// Next available sequence id.
    next_seq_id: usize,
}

impl<B: Backend> PagedKVCache<B> {
    pub fn new(
        max_blocks: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let block_manager = BlockManager::new(max_blocks, num_heads, head_dim, device.clone());
        let page_tables = (0..num_layers).map(|_| HashMap::new()).collect();

        Self {
            block_manager,
            page_tables,
            num_layers,
            next_seq_id: 0,
        }
    }

    /// Allocate a new sequence and return its id.
    pub fn allocate_sequence(&mut self) -> usize {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        for layer in 0..self.num_layers {
            self.page_tables[layer].insert(seq_id, PageTable::new());
        }

        seq_id
    }

    /// Append KV tensors for a given layer and sequence.
    pub fn append(
        &mut self,
        layer: usize,
        seq_id: usize,
        keys: Tensor<B, 3>,   // [num_heads, new_len, head_dim]
        values: Tensor<B, 3>, // [num_heads, new_len, head_dim]
    ) -> Result<(), &'static str> {
        let (block_manager, page_tables) = (&mut self.block_manager, &mut self.page_tables);
        let layer_tables = page_tables.get_mut(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get_mut(&seq_id).ok_or("unknown sequence")?;

        let [num_heads, new_len, head_dim] = keys.dims();
        let values_dims = values.dims();
        if values_dims != [num_heads, new_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        if num_heads != block_manager.num_heads() || head_dim != block_manager.head_dim() {
            return Err("keys/values head dimensions mismatch");
        }
        if new_len == 0 {
            return Ok(());
        }

        let mut offset = 0usize;
        while offset < new_len {
            let needs_block = match page_table.physical_blocks().last() {
                Some(&block_id) => {
                    let block = block_manager.get(block_id).ok_or("block not found")?;
                    block.is_full()
                }
                None => true,
            };

            if needs_block {
                let block_id = block_manager
                    .allocate()
                    .ok_or("no free blocks available")?;
                page_table.add_block(block_id);
            }

            let block_id = *page_table
                .physical_blocks()
                .last()
                .ok_or("no block allocated")?;
            let block = block_manager.get_mut(block_id).ok_or("block not found")?;
            let write_len = (new_len - offset).min(block.remaining_capacity());
            if write_len == 0 {
                return Err("block has no remaining capacity");
            }

            let keys_slice = keys.clone().slice([
                0..num_heads,
                offset..(offset + write_len),
                0..head_dim,
            ]);
            let values_slice = values.clone().slice([
                0..num_heads,
                offset..(offset + write_len),
                0..head_dim,
            ]);

            let block_offset = block.num_tokens;
            block.keys = block.keys.clone().slice_assign(
                [
                    0..num_heads,
                    block_offset..(block_offset + write_len),
                    0..head_dim,
                ],
                keys_slice,
            );
            block.values = block.values.clone().slice_assign(
                [
                    0..num_heads,
                    block_offset..(block_offset + write_len),
                    0..head_dim,
                ],
                values_slice,
            );
            block.num_tokens += write_len;
            page_table.extend_seq_len(write_len);

            offset += write_len;
        }

        Ok(())
    }

    /// Gather all cached KV tensors for a layer/sequence into contiguous tensors.
    pub fn get_kv(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        let seq_len = page_table.seq_len();
        if seq_len == 0 {
            return Err("sequence is empty");
        }

        let mut remaining = seq_len;
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for &block_id in page_table.physical_blocks() {
            if remaining == 0 {
                break;
            }
            let block = self
                .block_manager
                .get(block_id)
                .ok_or("block not found")?;
            let [num_heads, _block_size, head_dim] = block.keys.dims();
            let take = block.num_tokens.min(remaining);
            if take == 0 {
                continue;
            }

            keys.push(
                block
                    .keys
                    .clone()
                    .slice([0..num_heads, 0..take, 0..head_dim]),
            );
            values.push(
                block
                    .values
                    .clone()
                    .slice([0..num_heads, 0..take, 0..head_dim]),
            );
            remaining -= take;
        }

        if keys.is_empty() || remaining != 0 {
            return Err("incomplete kv data");
        }

        Ok((Tensor::cat(keys, 1), Tensor::cat(values, 1)))
    }

    pub fn seq_len(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        Ok(page_table.seq_len())
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_manager.num_free_blocks()
    }

    /// Release all blocks associated with a sequence id.
    pub fn free_sequence(&mut self, seq_id: usize) -> Result<(), &'static str> {
        let mut found = false;
        for layer_tables in &mut self.page_tables {
            if let Some(table) = layer_tables.remove(&seq_id) {
                for &block_id in table.physical_blocks() {
                    self.block_manager.free(block_id);
                }
                found = true;
            }
        }

        if found {
            Ok(())
        } else {
            Err("unknown sequence")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn test_block_manager_allocate_free() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut manager = BlockManager::new(2, 2, 4, device);

        let first = manager.allocate().expect("first allocate");
        assert_eq!(manager.num_free_blocks(), 1);
        let second = manager.allocate().expect("second allocate");
        assert_eq!(manager.num_free_blocks(), 0);
        assert!(manager.allocate().is_none());

        if let Some(block) = manager.get_mut(first) {
            block.num_tokens = BLOCK_SIZE;
        }
        manager.free(first);
        assert_eq!(manager.num_free_blocks(), 1);

        let reused = manager.allocate().expect("reuse allocate");
        let reused_block = manager.get(reused).expect("reused block");
        assert_eq!(reused_block.num_tokens, 0);

        manager.free(second);
        manager.free(reused);
        assert_eq!(manager.num_free_blocks(), 2);
    }

    #[test]
    fn test_page_table_mapping() {
        let mut table = PageTable::new();
        table.add_block(3);
        table.add_block(7);
        table.extend_seq_len(BLOCK_SIZE + 1);

        assert_eq!(table.get_physical_location(0), Some((3, 0)));
        assert_eq!(
            table.get_physical_location(BLOCK_SIZE - 1),
            Some((3, BLOCK_SIZE - 1))
        );
        assert_eq!(
            table.get_physical_location(BLOCK_SIZE),
            Some((7, 0))
        );
        assert_eq!(table.get_physical_location(BLOCK_SIZE + 1), None);
    }

    #[test]
    fn test_paged_kv_cache_basic() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 2, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        cache.append(1, seq_id, keys, values).expect("append");

        assert_eq!(cache.seq_len(1, seq_id).expect("seq len"), BLOCK_SIZE + 3);
        assert_eq!(cache.num_free_blocks(), 2);

        let (k, v) = cache.get_kv(1, seq_id).expect("get kv");
        assert_eq!(k.dims(), [2, BLOCK_SIZE + 3, 4]);
        assert_eq!(v.dims(), [2, BLOCK_SIZE + 3, 4]);

        cache.free_sequence(seq_id).expect("free");
        assert_eq!(cache.num_free_blocks(), 4);
    }
}
