//! KV cache tracking for executor.

use gllm_kernels::backend_trait::KvCacheHandle;
use gllm_kernels::kernel_types::KvCacheConfig;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheSlot {
    Front,
    Back,
}

impl KvCacheSlot {
    pub fn flip(self) -> Self {
        match self {
            KvCacheSlot::Front => KvCacheSlot::Back,
            KvCacheSlot::Back => KvCacheSlot::Front,
        }
    }
}

#[derive(Debug, Error)]
pub enum KvCacheError {
    #[error("kv cache exhausted: requested {requested}, available {available}")]
    Exhausted { requested: usize, available: usize },
}

pub type KvCacheResult<T> = std::result::Result<T, KvCacheError>;

#[derive(Debug, Clone)]
pub struct KvCacheState {
    handle: KvCacheHandle,
    config: KvCacheConfig,
    used: usize,
}

impl KvCacheState {
    pub fn new(handle: KvCacheHandle, config: KvCacheConfig) -> Self {
        Self {
            handle,
            config,
            used: 0,
        }
    }

    pub fn handle(&self) -> KvCacheHandle {
        self.handle
    }

    pub fn handle_mut(&mut self) -> &mut KvCacheHandle {
        &mut self.handle
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.config.max_seq_len.saturating_sub(self.used)
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn advance(&mut self, tokens: usize) -> KvCacheResult<()> {
        let remaining = self.remaining();
        if tokens > remaining {
            return Err(KvCacheError::Exhausted {
                requested: tokens,
                available: remaining,
            });
        }
        self.used = self.used.saturating_add(tokens);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KvCacheDoubleBuffer {
    front: KvCacheState,
    back: KvCacheState,
}

impl KvCacheDoubleBuffer {
    pub fn new(front: KvCacheState, back: KvCacheState) -> Self {
        Self { front, back }
    }

    pub fn front(&self) -> &KvCacheState {
        &self.front
    }

    pub fn back(&self) -> &KvCacheState {
        &self.back
    }

    pub fn front_mut(&mut self) -> &mut KvCacheState {
        &mut self.front
    }

    pub fn back_mut(&mut self) -> &mut KvCacheState {
        &mut self.back
    }

    pub fn slot(&self, slot: KvCacheSlot) -> &KvCacheState {
        match slot {
            KvCacheSlot::Front => &self.front,
            KvCacheSlot::Back => &self.back,
        }
    }

    pub fn slot_mut(&mut self, slot: KvCacheSlot) -> &mut KvCacheState {
        match slot {
            KvCacheSlot::Front => &mut self.front,
            KvCacheSlot::Back => &mut self.back,
        }
    }

    pub fn reset_all(&mut self) {
        self.front.reset();
        self.back.reset();
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}
