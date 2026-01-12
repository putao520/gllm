use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::{HierarchicalFlashAttention, HierarchicalFlashConfig};
use crate::distributed::{SequenceConfig, SequenceFactory, SequenceKV, SequenceHandle};

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

/// Reference to a KV block without copying data.
/// Used for zero-copy iteration over paged KV cache.
pub struct KVBlockRef<'a, B: Backend> {
    /// Keys slice: [num_heads, valid_tokens, head_dim]
    pub keys: &'a Tensor<B, 3>,
    /// Values slice: [num_heads, valid_tokens, head_dim]
    pub values: &'a Tensor<B, 3>,
    /// Number of valid tokens in this block
    pub num_tokens: usize,
    /// Block index in the sequence (0-based)
    pub block_idx: usize,
}

/// Iterator over KV blocks for a sequence.
/// Enables fused attention without Tensor::cat().
pub struct KVBlockIterator<'a, B: Backend> {
    /// Reference to block manager
    block_manager: &'a BlockManager<B>,
    /// Physical block IDs to iterate
    block_ids: &'a [usize],
    /// Current position in block_ids
    current_idx: usize,
    /// Remaining tokens to yield
    remaining_tokens: usize,
}

impl<'a, B: Backend> Iterator for KVBlockIterator<'a, B> {
    type Item = KVBlockRef<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_tokens == 0 || self.current_idx >= self.block_ids.len() {
            return None;
        }

        let block_id = self.block_ids[self.current_idx];
        let block = self.block_manager.get(block_id)?;

        let valid_tokens = block.num_tokens.min(self.remaining_tokens);
        if valid_tokens == 0 {
            return None;
        }

        let block_ref = KVBlockRef {
            keys: &block.keys,
            values: &block.values,
            num_tokens: valid_tokens,
            block_idx: self.current_idx,
        };

        self.remaining_tokens -= valid_tokens;
        self.current_idx += 1;

        Some(block_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.block_ids.len() - self.current_idx;
        (remaining.min(1), Some(remaining))
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

    /// Iterate over KV blocks for a sequence without concatenation.
    /// This enables O(1) per-block access instead of O(n²) from Tensor::cat().
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `seq_id` - Sequence ID
    ///
    /// # Returns
    /// Iterator yielding KVBlockRef for each block in sequence order.
    pub fn iter_kv_blocks(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<KVBlockIterator<'_, B>, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        let seq_len = page_table.seq_len();

        Ok(KVBlockIterator {
            block_manager: &self.block_manager,
            block_ids: page_table.physical_blocks(),
            current_idx: 0,
            remaining_tokens: seq_len,
        })
    }

    /// Get the number of blocks allocated for a sequence.
    pub fn num_blocks(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        Ok(page_table.physical_blocks().len())
    }

    /// Get block manager configuration.
    pub fn num_heads(&self) -> usize {
        self.block_manager.num_heads()
    }

    pub fn head_dim(&self) -> usize {
        self.block_manager.head_dim()
    }

    pub fn device(&self) -> &B::Device {
        self.block_manager.device()
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

// ============================================================================
// Isolated Paged KV Cache (Thread-Safe via Sequence Isolation)
// ============================================================================

/// Thread-safe paged KV cache using sequence isolation pattern.
/// Each sequence is handled by exactly one thread, eliminating lock contention.
pub struct IsolatedPagedKVCache<B: Backend> {
    /// Sequence factory for creating isolated sequences
    factory: SequenceFactory<B>,
    /// Number of layers
    num_layers: usize,
    /// Hierarchical flash attention configuration for 2M context support
    attention_config: HierarchicalFlashConfig,
}

impl<B: Backend> IsolatedPagedKVCache<B> {
    /// Create a new isolated paged KV cache.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_kv_heads` - Number of KV heads
    /// * `head_dim` - Dimension per head
    /// * `device` - Device for tensor allocation
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: B::Device,
    ) -> Self {
        let seq_config = SequenceConfig {
            shard: Default::default(),
            num_layers,
            num_kv_heads,
            head_dim,
            block_size: BLOCK_SIZE,
        };

        Self {
            factory: SequenceFactory::new(seq_config, device),
            num_layers,
            attention_config: HierarchicalFlashConfig::ultra_long_context(),
        }
    }

    /// Create a new isolated sequence.
    /// Returns a handle (NOT Send/Sync) and the KV storage.
    pub fn create_sequence(&self) -> (SequenceHandle, SequenceKV<B>) {
        self.factory.create()
    }

    /// Get the hierarchical flash attention configuration.
    pub fn attention_config(&self) -> &HierarchicalFlashConfig {
        &self.attention_config
    }

    /// Create a HierarchicalFlashAttention instance.
    pub fn create_attention(&self) -> HierarchicalFlashAttention {
        HierarchicalFlashAttention::new(self.attention_config.clone())
    }

    /// Get the number of sequences created.
    pub fn num_sequences(&self) -> usize {
        self.factory.num_created()
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Wrapper for computing attention with isolated sequences.
/// This is the recommended API for 2M+ context inference.
pub struct IsolatedAttentionContext<'a, B: Backend> {
    /// The sequence KV storage (owned, single-thread access)
    pub kv: &'a mut SequenceKV<B>,
    /// The attention module
    pub attention: HierarchicalFlashAttention,
}

impl<'a, B: Backend> IsolatedAttentionContext<'a, B> {
    /// Create a new isolated attention context.
    pub fn new(kv: &'a mut SequenceKV<B>, config: HierarchicalFlashConfig) -> Self {
        Self {
            kv,
            attention: HierarchicalFlashAttention::new(config),
        }
    }

    /// Compute attention for this sequence.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `query` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `key` - New key tensor [batch, num_heads, seq_len, head_dim]
    /// * `value` - New value tensor [batch, num_heads, seq_len, head_dim]
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Attention output [batch, num_heads, seq_len, head_dim]
    pub fn forward(
        &mut self,
        layer: usize,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        causal: bool,
    ) -> Tensor<B, 4> {
        // Get dimensions
        let [_batch, num_heads, new_seq_len, head_dim] = key.dims();

        // Reshape from [batch=1, num_heads, seq_len, head_dim] to [seq_len, num_heads, head_dim]
        let k_reshaped = key.clone()
            .reshape([num_heads, new_seq_len, head_dim])
            .swap_dims(0, 1);
        let v_reshaped = value.clone()
            .reshape([num_heads, new_seq_len, head_dim])
            .swap_dims(0, 1);
        self.kv.append(layer, k_reshaped, v_reshaped);

        // Get cached KV
        let total_seq_len = self.kv.seq_len();
        let position_offset = total_seq_len.saturating_sub(new_seq_len);

        let (cached_k, cached_v) = self.kv.get_kv(layer).unwrap_or_else(|| {
            let device = self.kv.device().clone();
            (
                Tensor::zeros([0, num_heads, head_dim], &device),
                Tensor::zeros([0, num_heads, head_dim], &device),
            )
        });

        // Reshape cached KV to [batch=1, num_heads, total_seq_len, head_dim]
        let cached_seq_len = cached_k.dims()[0];
        let cached_k = cached_k.swap_dims(0, 1).reshape([1, num_heads, cached_seq_len, head_dim]);
        let cached_v = cached_v.swap_dims(0, 1).reshape([1, num_heads, cached_seq_len, head_dim]);

        // Use hierarchical flash attention
        self.attention.forward(query, cached_k, cached_v, causal, position_offset)
    }

    /// Compute attention using fused iteration over KV blocks.
    /// This is the zero-copy version for maximum efficiency.
    pub fn forward_fused(
        &mut self,
        layer: usize,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        causal: bool,
    ) -> Tensor<B, 4> {
        // Get dimensions
        let [_batch, num_heads, new_seq_len, head_dim] = key.dims();

        // Reshape and append new KV
        let k_reshaped = key.clone()
            .reshape([num_heads, new_seq_len, head_dim])
            .swap_dims(0, 1);
        let v_reshaped = value.clone()
            .reshape([num_heads, new_seq_len, head_dim])
            .swap_dims(0, 1);
        self.kv.append(layer, k_reshaped, v_reshaped);

        // Get total sequence length for position offset
        let total_seq_len = self.kv.seq_len();
        let position_offset = total_seq_len.saturating_sub(new_seq_len);

        // Get KV as iterator (via get_kv for now, fused iteration requires cache redesign)
        let (cached_k, cached_v) = self.kv.get_kv(layer).unwrap_or_else(|| {
            let device = self.kv.device().clone();
            (
                Tensor::zeros([0, num_heads, head_dim], &device),
                Tensor::zeros([0, num_heads, head_dim], &device),
            )
        });

        // Create block iterator from cached KV
        let cached_seq_len = cached_k.dims()[0];
        let block_kv = self.attention.config().block_kv;
        let num_blocks = if cached_seq_len > 0 {
            (cached_seq_len + block_kv - 1) / block_kv
        } else {
            0
        };

        let kv_blocks: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = (0..num_blocks)
            .map(|i| {
                let start = i * block_kv;
                let end = ((i + 1) * block_kv).min(cached_seq_len);
                let k_block = cached_k.clone().slice([start..end, 0..num_heads, 0..head_dim]);
                let v_block = cached_v.clone().slice([start..end, 0..num_heads, 0..head_dim]);
                (k_block, v_block)
            })
            .collect();

        // Use fused iteration
        self.attention.forward_fused_iter(
            query,
            kv_blocks.into_iter(),
            causal,
            position_offset,
            total_seq_len,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn test_block_manager_allocate_free() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut manager = BlockManager::<NdArray<f32>>::new(2, 2, 4, device);

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

    #[test]
    fn test_iter_kv_blocks() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(8, 1, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        // Append enough tokens to span 3 blocks
        let total_tokens = BLOCK_SIZE * 2 + 5;
        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, total_tokens, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, total_tokens, 4], &device);
        cache.append(0, seq_id, keys, values).expect("append");

        // Iterate over blocks
        let iter = cache.iter_kv_blocks(0, seq_id).expect("iter");
        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].num_tokens, BLOCK_SIZE);
        assert_eq!(blocks[0].block_idx, 0);
        assert_eq!(blocks[1].num_tokens, BLOCK_SIZE);
        assert_eq!(blocks[1].block_idx, 1);
        assert_eq!(blocks[2].num_tokens, 5);
        assert_eq!(blocks[2].block_idx, 2);

        // Verify total tokens
        let total: usize = blocks.iter().map(|b| b.num_tokens).sum();
        assert_eq!(total, total_tokens);
    }

    #[test]
    fn test_kv_blocks_to_tensors() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 2, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        cache.append(1, seq_id, keys, values).expect("append");

        // Convert iterator to tensor pairs for attention
        let iter = cache.iter_kv_blocks(1, seq_id).expect("iter");
        let kv_blocks: Vec<_> = iter
            .map(|block| {
                let [num_heads, _, head_dim] = block.keys.dims();
                let k = block.keys.clone().slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                let v = block.values.clone().slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                (k, v)
            })
            .collect();

        assert_eq!(kv_blocks.len(), 2);
    }

    #[test]
    fn test_isolated_paged_kv_cache() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let cache = IsolatedPagedKVCache::<NdArray<f32>>::new(2, 4, 8, device.clone());

        // Create two isolated sequences
        let (handle1, mut kv1) = cache.create_sequence();
        let (handle2, kv2) = cache.create_sequence();

        assert_eq!(handle1.id(), 0);
        assert_eq!(handle2.id(), 1);
        assert_eq!(cache.num_sequences(), 2);

        // Append to kv1
        let k = Tensor::<NdArray<f32>, 3>::zeros([10, 4, 8], &device);
        let v = Tensor::<NdArray<f32>, 3>::zeros([10, 4, 8], &device);
        kv1.append(0, k, v);

        // kv2 should be unaffected (isolation)
        assert_eq!(kv1.seq_len(), 10);
        assert_eq!(kv2.seq_len(), 0);
    }

    #[test]
    fn test_isolated_attention_context() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let cache = IsolatedPagedKVCache::<NdArray<f32>>::new(2, 4, 8, device.clone());

        let (_handle, mut kv) = cache.create_sequence();
        let config = cache.attention_config().clone();

        let mut ctx = IsolatedAttentionContext::new(&mut kv, config);

        // Create input tensors [batch=1, num_heads=4, seq_len=8, head_dim=8]
        let q = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        // Forward pass
        let output = ctx.forward(0, q.clone(), k.clone(), v.clone(), true);
        assert_eq!(output.dims(), [1, 4, 8, 8]);

        // Verify KV was cached
        assert_eq!(ctx.kv.seq_len(), 8);

        // Second forward should use cached KV
        let q2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 4, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let output2 = ctx.forward(0, q2, k2, v2, true);
        assert_eq!(output2.dims(), [1, 4, 4, 8]);
        assert_eq!(ctx.kv.seq_len(), 12); // 8 + 4
    }
}
