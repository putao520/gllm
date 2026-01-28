//! Scheduler implementing ARCH-MEM-TIERING (v2026).
//!
//! Key behaviors:
//! - 3-stage eviction (Zombie -> Idle -> Shared)
//! - Anti-thrashing admission control
//! - Recompute vs Swap cost model
//! - Radix-tree refcounting for shared prefixes

use std::collections::{HashMap, VecDeque};

pub type SequenceId = u64;
pub type BlockId = u64;

/// Scheduler configuration for memory tiering.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Total number of GPU blocks available.
    pub total_gpu_blocks: usize,
    /// Safety margin for anti-thrashing admission control.
    pub safety_margin_steps: usize,
    /// Sequence length (tokens) below which Drop is preferred.
    pub short_sequence_threshold_tokens: usize,
    /// PCIe transfer latency (relative units).
    pub pcie_transfer_latency: f32,
    /// GPU recompute latency (relative units).
    pub gpu_compute_latency: f32,
}

/// Input specification for admitting a sequence.
#[derive(Debug, Clone)]
pub struct SequenceSpec {
    /// Number of GPU blocks required to admit this sequence.
    pub required_blocks: usize,
    /// Predicted resume time used for idle eviction ordering.
    pub predicted_resume_time: u64,
    /// Prefix tokens representing the system prompt.
    pub prefix_tokens: Vec<u32>,
    /// Block size in tokens for mapping prefix tokens to blocks.
    pub block_size_tokens: usize,
    /// Total sequence length in tokens.
    pub sequence_len_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Active,
    Preempted,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictAction {
    /// Release immediately without recompute or swap.
    Release,
    /// Drop from GPU, recompute later if needed.
    Drop,
    /// Swap out to host memory.
    SwapOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStage {
    Zombie,
    Idle,
    Shared,
}

#[derive(Debug, Clone)]
pub struct Eviction {
    pub block_id: BlockId,
    pub sequence_id: SequenceId,
    pub action: EvictAction,
    pub stage: EvictionStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    AdmissionRejected,
    UnknownSequence,
}

#[derive(Debug, Clone)]
pub struct BlockInfo {
    id: BlockId,
    sequence_id: SequenceId,
    is_shared: bool,
    radix_node: Option<NodeId>,
    sequence_len_tokens: usize,
}

#[derive(Debug, Clone)]
struct SequenceInfo {
    id: SequenceId,
    required_blocks: usize,
    predicted_resume_time: u64,
    state: SequenceState,
    block_ids: Vec<BlockId>,
    prefix_tokens: Vec<u32>,
    prefix_nodes: Vec<NodeId>,
    sequence_len_tokens: usize,
    prefix_released: bool,
}

/// Scheduler implementing the ARCH-MEM-TIERING (v2026) policy.
pub struct Scheduler {
    config: SchedulerConfig,
    free_gpu_blocks: usize,
    sequences: HashMap<SequenceId, SequenceInfo>,
    blocks: HashMap<BlockId, BlockInfo>,
    shared_lru: VecDeque<BlockId>,
    radix: RadixTree,
    radix_blocks: HashMap<NodeId, Vec<BlockId>>,
    next_sequence_id: SequenceId,
    next_block_id: BlockId,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let total = config.total_gpu_blocks;
        Self {
            config,
            free_gpu_blocks: total,
            sequences: HashMap::new(),
            blocks: HashMap::new(),
            shared_lru: VecDeque::new(),
            radix: RadixTree::new(),
            radix_blocks: HashMap::new(),
            next_sequence_id: 1,
            next_block_id: 1,
        }
    }

    /// Returns the current number of free GPU blocks.
    pub fn free_gpu_blocks(&self) -> usize {
        self.free_gpu_blocks
    }

    /// Anti-thrashing admission control.
    pub fn can_admit(&self, sequence: &SequenceSpec) -> bool {
        self.free_gpu_blocks
            >= sequence.required_blocks + self.config.safety_margin_steps
    }

    /// Admit a sequence if it passes admission control.
    pub fn admit_sequence(&mut self, sequence: SequenceSpec) -> Result<SequenceId, SchedulerError> {
        if !self.can_admit(&sequence) {
            return Err(SchedulerError::AdmissionRejected);
        }

        let seq_id = self.next_sequence_id;
        self.next_sequence_id = self.next_sequence_id.wrapping_add(1);

        let prefix_nodes = if sequence.prefix_tokens.is_empty() {
            Vec::new()
        } else {
            self.radix.insert(&sequence.prefix_tokens)
        };

        let prefix_block_count = if sequence.block_size_tokens == 0 {
            0
        } else {
            (sequence.prefix_tokens.len() + sequence.block_size_tokens - 1) / sequence.block_size_tokens
        };

        let mut block_ids = Vec::with_capacity(sequence.required_blocks);
        for block_index in 0..sequence.required_blocks {
            let block_id = self.next_block_id;
            self.next_block_id = self.next_block_id.wrapping_add(1);

            let radix_node = if block_index < prefix_block_count && !prefix_nodes.is_empty() {
                let token_index = ((block_index + 1) * sequence.block_size_tokens)
                    .min(sequence.prefix_tokens.len())
                    .saturating_sub(1);
                Some(prefix_nodes[token_index])
            } else {
                None
            };

            let is_shared = radix_node
                .map(|node| self.radix.refcount(node) > 1)
                .unwrap_or(false);

            let info = BlockInfo {
                id: block_id,
                sequence_id: seq_id,
                is_shared,
                radix_node,
                sequence_len_tokens: sequence.sequence_len_tokens,
            };

            if let Some(node) = radix_node {
                self.radix_blocks.entry(node).or_default().push(block_id);
            }

            if is_shared {
                self.shared_lru.push_back(block_id);
            }

            self.blocks.insert(block_id, info);
            block_ids.push(block_id);
        }

        self.free_gpu_blocks = self.free_gpu_blocks.saturating_sub(sequence.required_blocks);

        let info = SequenceInfo {
            id: seq_id,
            required_blocks: sequence.required_blocks,
            predicted_resume_time: sequence.predicted_resume_time,
            state: SequenceState::Active,
            block_ids,
            prefix_tokens: sequence.prefix_tokens,
            prefix_nodes,
            sequence_len_tokens: sequence.sequence_len_tokens,
            prefix_released: false,
        };
        self.sequences.insert(seq_id, info);

        if !self.sequences[&seq_id].prefix_nodes.is_empty() {
            let nodes = self.sequences[&seq_id].prefix_nodes.clone();
            self.refresh_shared_state_for_nodes(&nodes);
        }

        Ok(seq_id)
    }

    /// Mark a sequence as preempted.
    pub fn preempt_sequence(
        &mut self,
        sequence_id: SequenceId,
        predicted_resume_time: u64,
    ) -> Result<(), SchedulerError> {
        let sequence = self
            .sequences
            .get_mut(&sequence_id)
            .ok_or(SchedulerError::UnknownSequence)?;
        sequence.state = SequenceState::Preempted;
        sequence.predicted_resume_time = predicted_resume_time;
        Ok(())
    }

    /// Mark a sequence as active again.
    pub fn resume_sequence(&mut self, sequence_id: SequenceId) -> Result<(), SchedulerError> {
        let sequence = self
            .sequences
            .get_mut(&sequence_id)
            .ok_or(SchedulerError::UnknownSequence)?;
        sequence.state = SequenceState::Active;
        Ok(())
    }

    /// Mark a sequence as finished and release its prefix refcounting.
    pub fn finish_sequence(&mut self, sequence_id: SequenceId) -> Result<(), SchedulerError> {
        let sequence = self
            .sequences
            .get_mut(&sequence_id)
            .ok_or(SchedulerError::UnknownSequence)?;
        sequence.state = SequenceState::Finished;
        if !sequence.prefix_released && !sequence.prefix_tokens.is_empty() {
            let nodes = self.radix.release(&sequence.prefix_tokens);
            sequence.prefix_released = true;
            self.refresh_shared_state_for_nodes(&nodes);
        }
        Ok(())
    }

    /// Touch a block (LRU update for shared blocks).
    pub fn touch_block(&mut self, block_id: BlockId) {
        if let Some(block) = self.blocks.get(&block_id) {
            if block.is_shared {
                self.remove_from_lru(block_id);
                self.shared_lru.push_back(block_id);
            }
        }
    }

    /// Evict blocks until at least `needed_blocks` are freed.
    pub fn evict_blocks(&mut self, needed_blocks: usize) -> Vec<Eviction> {
        let mut remaining = needed_blocks;
        let mut evictions = Vec::new();

        if remaining == 0 {
            return evictions;
        }

        remaining = remaining.saturating_sub(self.evict_zombies(remaining, &mut evictions));
        if remaining == 0 {
            return evictions;
        }

        remaining = remaining.saturating_sub(self.evict_idle(remaining, &mut evictions));
        if remaining == 0 {
            return evictions;
        }

        self.evict_shared(remaining, &mut evictions);
        evictions
    }

    /// Determine eviction strategy for a block (Drop vs SwapOut).
    pub fn decide_eviction_strategy(&self, block: &BlockInfo) -> EvictAction {
        if block.sequence_len_tokens <= self.config.short_sequence_threshold_tokens {
            return EvictAction::Drop;
        }

        if self.config.pcie_transfer_latency > self.config.gpu_compute_latency {
            EvictAction::Drop
        } else {
            EvictAction::SwapOut
        }
    }

    fn evict_zombies(&mut self, needed: usize, evictions: &mut Vec<Eviction>) -> usize {
        if needed == 0 {
            return 0;
        }

        let mut freed = 0;
        let zombie_blocks: Vec<BlockId> = self
            .blocks
            .iter()
            .filter_map(|(&block_id, block)| {
                let sequence = self.sequences.get(&block.sequence_id)?;
                if sequence.state == SequenceState::Finished && !block.is_shared {
                    Some(block_id)
                } else {
                    None
                }
            })
            .collect();

        for block_id in zombie_blocks {
            if freed >= needed {
                break;
            }
            if let Some(block) = self.blocks.get(&block_id) {
                let sequence_id = block.sequence_id;
                self.evict_block(block_id, EvictAction::Release, EvictionStage::Zombie, evictions);
                freed += 1;
                if let Some(sequence) = self.sequences.get(&sequence_id) {
                    if sequence.block_ids.is_empty() {
                        // Optionally clean up finished sequences with no blocks.
                    }
                }
            }
        }

        freed
    }

    fn evict_idle(&mut self, needed: usize, evictions: &mut Vec<Eviction>) -> usize {
        if needed == 0 {
            return 0;
        }

        let mut freed = 0;
        let mut preempted: Vec<SequenceInfo> = self
            .sequences
            .values()
            .filter(|seq| seq.state == SequenceState::Preempted)
            .cloned()
            .collect();

        preempted.sort_by_key(|seq| std::cmp::Reverse(seq.predicted_resume_time));

        for seq in preempted {
            if freed >= needed {
                break;
            }
            for block_id in seq.block_ids {
                if freed >= needed {
                    break;
                }
                let should_evict = self
                    .blocks
                    .get(&block_id)
                    .map(|block| !block.is_shared)
                    .unwrap_or(false);
                if should_evict {
                    let action = {
                        let block = self.blocks.get(&block_id).expect("block exists");
                        self.decide_eviction_strategy(block)
                    };
                    self.evict_block(block_id, action, EvictionStage::Idle, evictions);
                    freed += 1;
                }
            }
        }

        freed
    }

    fn evict_shared(&mut self, needed: usize, evictions: &mut Vec<Eviction>) {
        let mut remaining = needed;
        while remaining > 0 {
            let block_id = match self.shared_lru.pop_front() {
                Some(id) => id,
                None => break,
            };

            let action = {
                let block = match self.blocks.get(&block_id) {
                    Some(block) => block,
                    None => continue,
                };
                self.decide_eviction_strategy(block)
            };

            self.evict_block(block_id, action, EvictionStage::Shared, evictions);
            remaining = remaining.saturating_sub(1);
        }
    }

    fn evict_block(
        &mut self,
        block_id: BlockId,
        action: EvictAction,
        stage: EvictionStage,
        evictions: &mut Vec<Eviction>,
    ) {
        let block = match self.blocks.remove(&block_id) {
            Some(block) => block,
            None => return,
        };

        self.free_gpu_blocks = self.free_gpu_blocks.saturating_add(1);

        if block.is_shared {
            self.remove_from_lru(block_id);
        }

        if let Some(node) = block.radix_node {
            if let Some(list) = self.radix_blocks.get_mut(&node) {
                list.retain(|&id| id != block_id);
                if list.is_empty() {
                    self.radix_blocks.remove(&node);
                }
            }
        }

        if let Some(sequence) = self.sequences.get_mut(&block.sequence_id) {
            sequence.block_ids.retain(|&id| id != block_id);
        }

        evictions.push(Eviction {
            block_id: block.id,
            sequence_id: block.sequence_id,
            action,
            stage,
        });
    }

    fn refresh_shared_state_for_nodes(&mut self, nodes: &[NodeId]) {
        for &node in nodes {
            let shared = self.radix.refcount(node) > 1;
            let block_ids = self
                .radix_blocks
                .get(&node)
                .cloned()
                .unwrap_or_default();
            for block_id in block_ids {
                if let Some(block) = self.blocks.get_mut(&block_id) {
                    if block.is_shared != shared {
                        block.is_shared = shared;
                        if shared {
                            self.shared_lru.push_back(block_id);
                        } else {
                            self.remove_from_lru(block_id);
                        }
                    }
                }
            }
        }
    }

    fn remove_from_lru(&mut self, block_id: BlockId) {
        if let Some(pos) = self.shared_lru.iter().position(|&id| id == block_id) {
            self.shared_lru.remove(pos);
        }
    }
}

type NodeId = usize;

#[derive(Debug, Clone)]
struct RadixNode {
    children: HashMap<u32, NodeId>,
    refcount: usize,
}

#[derive(Debug, Clone)]
struct RadixTree {
    nodes: Vec<RadixNode>,
}

impl RadixTree {
    fn new() -> Self {
        Self {
            nodes: vec![RadixNode {
                children: HashMap::new(),
                refcount: 0,
            }],
        }
    }

    fn insert(&mut self, tokens: &[u32]) -> Vec<NodeId> {
        let mut current = 0;
        let mut path = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let next = if let Some(&child) = self.nodes[current].children.get(&token) {
                child
            } else {
                let child = self.nodes.len();
                self.nodes.push(RadixNode {
                    children: HashMap::new(),
                    refcount: 0,
                });
                self.nodes[current].children.insert(token, child);
                child
            };
            self.nodes[next].refcount += 1;
            path.push(next);
            current = next;
        }
        path
    }

    fn release(&mut self, tokens: &[u32]) -> Vec<NodeId> {
        let mut current = 0;
        let mut path = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let next = match self.nodes[current].children.get(&token).copied() {
                Some(node) => node,
                None => break,
            };
            if self.nodes[next].refcount > 0 {
                self.nodes[next].refcount -= 1;
            }
            path.push(next);
            current = next;
        }
        path
    }

    fn refcount(&self, node: NodeId) -> usize {
        self.nodes.get(node).map(|n| n.refcount).unwrap_or(0)
    }
}
