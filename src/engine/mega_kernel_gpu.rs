//! BatchContext GPU 扩展区内存布局契约 (SPEC 32 §6.2).
//!
//! 本模块定义 BatchContext 扩展区的内存布局契约:
//! - `DualBatchMeta`: Ping/Pong 双缓冲元数据 (§2.3)，嵌入扩展区
//! - EXT_* 偏移常量: 与 GPU PTX codegen 共享的编译时 ABI 契约
//! - `BATCH_CTX_EXTENSION_SIZE`: 扩展区总大小
//!
//! 生产调用方: `batch_context::BatchContext::with_v2_extension()` +
//! `set_ext_*()` 系列 (line 307-367)。
//!
//! GPU 端 PTX/HIP codegen (cluster.sync, mbarrier, cp.async.bulk 等) 及
//! ring buffer / SM 分区派生由 gllm-kernels JIT 管线实现:
//! - `gllm-kernels::compiler::codegen::vm::mega_kernel_emit::select_mk_variant()`
//! - `gllm-kernels::compiler::codegen::vm::mega_kernel_emit::OutputRingBuffer`
//! - `gllm-kernels::compiler::codegen::vm::isa_profile::Platform` 的 SM 派生

// ═══════════════════════════════════════════════════════════
// §2.3 DualBatchMeta — Ping/Pong 双缓冲元数据
// ═══════════════════════════════════════════════════════════

/// Ping/Pong 双缓冲元数据 (SPEC 32 §2.3).
///
/// 位于 BatchContext 扩展区 (§6.2)，不侵入现有 SPEC 20 字段。
/// GPU 端每个 decode step 结束后 swap(ping, pong) 并递增 step_epoch。
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DualBatchMeta {
    /// Ping: 当前 decode batch 的 seq_meta 起始偏移 (相对 seq_meta_base, 单位 SEQ_META_STRIDE)
    pub ping_seq_offset: u32,
    /// Ping: 当前 decode batch 的活跃序列数
    pub ping_seq_count: u32,
    /// Pong: compact+refill 后新 batch 的 seq_meta 起始偏移
    pub pong_seq_offset: u32,
    /// Pong: 新 batch 的活跃序列数
    pub pong_seq_count: u32,
    /// Barrier epoch: 每步递增。SM90+ cluster.sync 隐式同步;
    /// SM80 grid_sync 需要 epoch 防止跨步同步;
    /// SM70-/CPU ring barrier 用作 arrival count
    pub step_epoch: u32,
    /// SM70-/CPU ring barrier: 每步到达计数器
    pub epoch_arrival_count: u32,
}

impl DualBatchMeta {
    pub const SIZE: usize = 24;

    pub fn new(max_batch_size: u32) -> Self {
        Self {
            ping_seq_offset: 0,
            ping_seq_count: 0,
            pong_seq_offset: max_batch_size,
            pong_seq_count: 0,
            step_epoch: 0,
            epoch_arrival_count: 0,
        }
    }

    /// Swap ping ↔ pong (decode step 结束后调用).
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.ping_seq_offset, &mut self.pong_seq_offset);
        std::mem::swap(&mut self.ping_seq_count, &mut self.pong_seq_count);
        self.step_epoch = self.step_epoch.wrapping_add(1);
        self.epoch_arrival_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════
// §6.2 BatchContext 扩展区偏移常量
// ═══════════════════════════════════════════════════════════

/// BatchContext 扩展区位于 seq_meta 数组之后.
/// 实际偏移 = BATCH_CTX_HEADER_SIZE (96) + max_batch_size × SEQ_META_STRIDE (64).
pub const EXT_REQUEST_QUEUE_PTR: usize = 0;
pub const EXT_OUTPUT_RING_PTR: usize = 8;
pub const EXT_KV_FREE_BITMAP_PTR: usize = 16;
pub const EXT_KV_POOL_TOTAL_PAGES: usize = 24;
pub const EXT_MAX_BATCH_SIZE: usize = 28;
pub const EXT_DUAL_BATCH_META: usize = 32;
pub const EXT_AUTOTUNE_ACTUAL_BATCH: usize = 56;
pub const EXT_POOL_CLUSTER_DSMEM_PTR: usize = 60;
pub const EXT_PENDING_FREE_LIST_PTR: usize = 68;
pub const EXT_PENDING_FREE_COUNT_PTR: usize = 76;
pub const EXT_OUTPUT_PER_CTA_DOORBELL_PTR: usize = 80;
pub const EXT_OUTPUT_EPOCH_FLAG_PTR: usize = 88;
pub const EXT_RESERVED: usize = 92;
// ── REQ-KV-EXT-001: V2 扩展字段 (14 total) ──
/// KvPageHeader stride in bytes (64 for V2, was 56 for V1).
pub const EXT_KV_PAGE_HEADER_STRIDE: usize = 96;
/// Base pointer for ext_id indexed KV extension slots (V2 ext_id → ext slot).
pub const EXT_KV_EXT_ID_BASE_PTR: usize = 104;

/// 扩展区总大小 (REQ-KV-EXT-001: 14 fields, aligned to 128 bytes).
///
/// LEGAL-FIXED: This is a GPU kernel ABI layout constant — the extension area
/// is a fixed-size memory region whose layout (14 offset constants above) is
/// baked into PTX codegen. All EXT_* offsets are compile-time constants that
/// the GPU kernel reads via fixed-address loads/stores. Changing this at
/// runtime would require recompiling the PTX kernel with different offsets,
/// which is the JIT pipeline's job — but the extension layout itself is a
/// stable ABI contract, not a hardware-tunable parameter.
///
/// Current layout: 14 fields occupy 112 bytes (last field at offset 104 + 8B),
/// padded to 128 bytes (16-byte alignment, 16 bytes reserved for future fields).
/// If new extension fields are added that exceed 128 bytes, this constant AND
/// all GPU codegen that references EXT_* offsets must be updated together.
pub const BATCH_CTX_EXTENSION_SIZE: usize = 128;

// ═══════════════════════════════════════════════════════════
// 单元测试 — 仅覆盖生产路径使用的内存布局契约
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── DualBatchMeta ──

    #[test]
    fn dual_batch_meta_new() {
        let meta = DualBatchMeta::new(256);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 256);
        assert_eq!(meta.step_epoch, 0);
    }

    #[test]
    fn dual_batch_meta_swap() {
        let mut meta = DualBatchMeta::new(128);
        meta.ping_seq_offset = 0;
        meta.ping_seq_count = 10;
        meta.pong_seq_offset = 128;
        meta.pong_seq_count = 8;
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 128);
        assert_eq!(meta.ping_seq_count, 8);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 10);
        assert_eq!(meta.step_epoch, 1);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_size() {
        assert_eq!(std::mem::size_of::<DualBatchMeta>(), DualBatchMeta::SIZE);
    }

    #[test]
    fn dual_batch_meta_default_trait() {
        let meta = DualBatchMeta::default();
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
        assert_eq!(meta.step_epoch, 0);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_clone_copy_traits() {
        let meta = DualBatchMeta::new(64);
        let copy = meta;
        assert_eq!(copy.ping_seq_offset, 0);
        assert_eq!(copy.pong_seq_offset, 64);
        let cloned = meta.clone();
        assert_eq!(cloned.ping_seq_offset, 0);
        assert_eq!(cloned.pong_seq_offset, 64);
    }

    #[test]
    fn dual_batch_meta_swap_epoch_wrapping() {
        let mut meta = DualBatchMeta::new(16);
        meta.step_epoch = u32::MAX;
        meta.swap();
        assert_eq!(meta.step_epoch, 0); // wrapping_add wraps to 0
    }

    #[test]
    fn dual_batch_meta_swap_resets_arrival_count() {
        let mut meta = DualBatchMeta::new(16);
        meta.epoch_arrival_count = 42;
        meta.swap();
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_double_swap_identity() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_offset = 10;
        meta.ping_seq_count = 5;
        meta.pong_seq_offset = 74;
        meta.pong_seq_count = 3;
        let original_ping = meta.ping_seq_offset;
        let original_pong = meta.pong_seq_offset;
        meta.swap();
        meta.swap();
        assert_eq!(meta.ping_seq_offset, original_ping);
        assert_eq!(meta.pong_seq_offset, original_pong);
        assert_eq!(meta.step_epoch, 2);
    }

    #[test]
    fn dual_batch_meta_new_zero_batch_size() {
        let meta = DualBatchMeta::new(0);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_new_max_u32_batch_size() {
        let meta = DualBatchMeta::new(u32::MAX);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, u32::MAX);
    }

    #[test]
    fn dual_batch_meta_debug_trait_output() {
        let meta = DualBatchMeta::new(64);
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("ping_seq_offset: 0"));
        assert!(debug_str.contains("pong_seq_offset: 64"));
        assert!(debug_str.contains("step_epoch: 0"));
    }

    #[test]
    fn dual_batch_meta_swap_multiple_epoch_accumulation() {
        let mut meta = DualBatchMeta::new(32);
        assert_eq!(meta.step_epoch, 0);
        for expected_epoch in 1..=5 {
            meta.swap();
            assert_eq!(meta.step_epoch, expected_epoch);
        }
    }

    #[test]
    fn dual_batch_meta_repr_c_alignment() {
        assert_eq!(std::mem::size_of::<DualBatchMeta>(), 24);
        assert_eq!(std::mem::align_of::<DualBatchMeta>(), 4);
    }

    #[test]
    fn dual_batch_meta_swap_preserves_sum_of_counts() {
        let mut meta = DualBatchMeta::new(128);
        meta.ping_seq_count = 37;
        meta.pong_seq_count = 19;
        let total = meta.ping_seq_count + meta.pong_seq_count;
        meta.swap();
        assert_eq!(meta.ping_seq_count + meta.pong_seq_count, total);
    }

    #[test]
    fn dual_batch_meta_swap_preserves_sum_of_offsets() {
        let mut meta = DualBatchMeta::new(256);
        meta.ping_seq_offset = 10;
        meta.pong_seq_offset = 300;
        let sum = meta.ping_seq_offset + meta.pong_seq_offset;
        meta.swap();
        assert_eq!(meta.ping_seq_offset + meta.pong_seq_offset, sum);
    }

    #[test]
    fn dual_batch_meta_many_swaps_epoch_monotonic() {
        let mut meta = DualBatchMeta::new(32);
        let mut prev_epoch = meta.step_epoch;
        for _ in 0..100 {
            meta.swap();
            assert!(meta.step_epoch > prev_epoch || meta.step_epoch == 0);
            prev_epoch = meta.step_epoch;
        }
    }

    #[test]
    fn dual_batch_meta_manual_field_set_and_swap() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_count = 32;
        meta.pong_seq_count = 28;
        meta.epoch_arrival_count = 7;
        meta.swap();
        assert_eq!(meta.ping_seq_count, 28);
        assert_eq!(meta.pong_seq_count, 32);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_swap_with_zero_counts() {
        let mut meta = DualBatchMeta::new(32);
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 32);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_copy_independence_after_swap() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_count = 10;
        let copy = meta;
        meta.swap();
        assert_eq!(copy.ping_seq_offset, 0);
        assert_eq!(copy.ping_seq_count, 10);
        assert_eq!(copy.pong_seq_offset, 64);
    }

    #[test]
    fn dual_batch_meta_new_pong_equals_batch_size() {
        for &batch_size in &[1, 16, 64, 256, 1024, 4096] {
            let meta = DualBatchMeta::new(batch_size);
            assert_eq!(meta.pong_seq_offset, batch_size);
            assert_eq!(meta.ping_seq_offset, 0);
        }
    }

    #[test]
    fn dual_batch_meta_new_non_power_of_two() {
        let meta = DualBatchMeta::new(7);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 7);
        let meta2 = DualBatchMeta::new(99);
        assert_eq!(meta2.pong_seq_offset, 99);
    }

    // ── Extension layout constants ──

    #[test]
    fn batch_ctx_extension_size_alignment() {
        assert_eq!(BATCH_CTX_EXTENSION_SIZE % 128, 0);
        assert!(BATCH_CTX_EXTENSION_SIZE >= 128);
    }

    #[test]
    fn extension_offsets_are_increasing() {
        let offsets: [(usize, &str); 14] = [
            (EXT_REQUEST_QUEUE_PTR, "REQUEST_QUEUE_PTR"),
            (EXT_OUTPUT_RING_PTR, "OUTPUT_RING_PTR"),
            (EXT_KV_FREE_BITMAP_PTR, "KV_FREE_BITMAP_PTR"),
            (EXT_KV_POOL_TOTAL_PAGES, "KV_POOL_TOTAL_PAGES"),
            (EXT_MAX_BATCH_SIZE, "MAX_BATCH_SIZE"),
            (EXT_DUAL_BATCH_META, "DUAL_BATCH_META"),
            (EXT_AUTOTUNE_ACTUAL_BATCH, "AUTOTUNE_ACTUAL_BATCH"),
            (EXT_POOL_CLUSTER_DSMEM_PTR, "POOL_CLUSTER_DSMEM_PTR"),
            (EXT_PENDING_FREE_LIST_PTR, "PENDING_FREE_LIST_PTR"),
            (EXT_PENDING_FREE_COUNT_PTR, "PENDING_FREE_COUNT_PTR"),
            (EXT_OUTPUT_PER_CTA_DOORBELL_PTR, "OUTPUT_PER_CTA_DOORBELL_PTR"),
            (EXT_OUTPUT_EPOCH_FLAG_PTR, "OUTPUT_EPOCH_FLAG_PTR"),
            (EXT_KV_PAGE_HEADER_STRIDE, "KV_PAGE_HEADER_STRIDE"),
            (EXT_KV_EXT_ID_BASE_PTR, "KV_EXT_ID_BASE_PTR"),
        ];
        for i in 1..offsets.len() {
            assert!(
                offsets[i].0 > offsets[i - 1].0,
                "{} ({}) should be greater than {} ({})",
                offsets[i].1,
                offsets[i].0,
                offsets[i - 1].1,
                offsets[i - 1].0,
            );
        }
    }

    #[test]
    fn extension_offsets_within_extension_size() {
        assert!(EXT_REQUEST_QUEUE_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_OUTPUT_RING_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_KV_FREE_BITMAP_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_DUAL_BATCH_META < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_RESERVED < BATCH_CTX_EXTENSION_SIZE);
    }

    #[test]
    fn extension_dual_batch_meta_fits_in_extension() {
        assert!(
            EXT_DUAL_BATCH_META + DualBatchMeta::SIZE <= BATCH_CTX_EXTENSION_SIZE,
            "DualBatchMeta (24B at offset {}) must fit within extension ({}B)",
            EXT_DUAL_BATCH_META,
            BATCH_CTX_EXTENSION_SIZE
        );
    }
}
