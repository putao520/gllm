//! §13.2 Softmax Centroid-Guided Prefetch
//!
//! Extracts attention probability centroids (argmax token positions) from
//! softmax epilogue telemetry and issues asynchronous memory prefetch hints
//! for KV cache blocks in layer N+1.

use std::collections::VecDeque;

/// Prefetch request for a KV cache block.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    pub layer: usize,
    pub token_idx: usize,
}

/// Prefetch queue for KV cache blocks.
pub struct PrefetchQueue {
    queue: VecDeque<PrefetchRequest>,
    block_size: usize,
}

impl PrefetchQueue {
    pub fn new(block_size: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(32),
            block_size,
        }
    }

    /// Enqueue a prefetch request for layer N+1 based on centroid token index.
    pub fn enqueue(&mut self, layer: usize, centroid_token_idx: usize) {
        self.queue.push_back(PrefetchRequest {
            layer: layer + 1,
            token_idx: centroid_token_idx,
        });
    }

    /// Issue prefetch hints for all queued requests.
    #[cfg(target_arch = "x86_64")]
    pub fn issue_prefetch(&mut self, kv_cache_ptr: *const u8, kv_stride: usize) {
        while let Some(req) = self.queue.pop_front() {
            let offset = req.layer * kv_stride + req.token_idx * self.block_size;
            let ptr = unsafe { kv_cache_ptr.add(offset) };

            unsafe {
                // prefetcht1: prefetch to L2 cache
                std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T1}>(ptr as *const i8);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn issue_prefetch(&mut self, kv_cache_ptr: *const u8, kv_stride: usize) {
        while let Some(req) = self.queue.pop_front() {
            let offset = req.layer * kv_stride + req.token_idx * self.block_size;
            let ptr = unsafe { kv_cache_ptr.add(offset) };

            unsafe {
                // ARM PRFM instruction (prefetch to L2)
                std::arch::asm!(
                    "prfm pldl2keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, preserves_flags)
                );
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn issue_prefetch(&mut self, _kv_cache_ptr: *const u8, _kv_stride: usize) {
        // No-op on unsupported architectures
        self.queue.clear();
    }
}
