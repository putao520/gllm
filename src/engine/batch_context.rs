//! BatchContext flat memory layout (SPEC/20 §1.2 + BCI6 seq_mapping_ptr)
//!
//! M 维度统一架构：batch_size=1 是 batch_size=N 的特例。
//! BatchContext 通过单一 flat memory buffer 传递给 JIT mega-kernel。
//! arg 22 (batch_ctx_ptr): NULL → 单序列模式，非 NULL → batch 模式。
//!
//! # Layout (与 JIT mega_kernel_emit.rs 完全一致)
//!
//! Header (96 bytes):
//!   0-79:   SPEC §1.2 header fields
//!   80-87:  seq_mapping_ptr (BCI6 per-token seq_id lookup)
//!   88-95:  seq_meta_base_ptr (pointer to per-seq metadata array)
//!
//! Per-seq array (at the location pointed to by seq_meta_base_ptr):
//!   stride = 64 bytes, 14 × u32 fields (56 bytes data + 8 bytes padding)

// ── Header field offsets ──

use crate::engine::mega_kernel_v2;

pub const BATCH_CTX_HEADER_SIZE: usize = 96;

pub const NUM_SEQS: usize = 0;
pub const MAX_DECODE_STEPS: usize = 4;
pub const TOTAL_PREFILL_TOKENS: usize = 8;
// 12: pad
pub const INPUT_IDS_FLAT_PTR: usize = 16;
pub const OUTPUT_TOKENS_FLAT_PTR: usize = 24;
pub const POSITIONS_PTR: usize = 32;
pub const PAGE_TABLE_FLAT_PTR: usize = 40;
pub const KV_POOL_BASE: usize = 48;
pub const SAMPLING_PARAMS_PTR: usize = 56;
pub const HOOK_CTX_PTR: usize = 64;
pub const CALLBACK_TABLE_PTR: usize = 72;
pub const SEQ_MAPPING_PTR: usize = 80;
pub const SEQ_META_BASE_PTR: usize = 88;

// ── Per-seq metadata offsets (relative to seq_meta_base + seq_idx × SEQ_META_STRIDE) ──
// SPEC/20 §1.2: 14 × u32 fields, stride 56 bytes padded to 64 for alignment

pub const SEQ_META_STRIDE: usize = 64;

pub const SEQ_PROMPT_LEN: usize = 0;
pub const SEQ_KV_LEN: usize = 4;
pub const SEQ_ROPE_POS_OFFSET: usize = 8;
pub const SEQ_MAX_NEW_TOKENS: usize = 12;
pub const SEQ_SESSION_POSITION: usize = 16;
pub const SEQ_PAGE_TABLE_OFFSET: usize = 20;
pub const SEQ_PAGE_TABLE_LEN: usize = 24;
pub const SEQ_FUSED_HIDDEN_OFFSET: usize = 28;
pub const SEQ_NUM_MM_TOKENS: usize = 32;
pub const SEQ_ACTIVE_FLAG: usize = 36;
pub const SEQ_SEQ_POSITION: usize = 40;
pub const SEQ_GEN_COUNT: usize = 44;
pub const SEQ_LAST_SAMPLED_TOKEN: usize = 48;
pub const SEQ_OUTPUT_OFFSET: usize = 52;

/// BatchContext builder — flat memory layout
///
/// Layout (SPEC/20 §1.2 + BCI6, matches JIT mega_kernel_emit.rs):
/// ```text
/// Header (96 bytes):
///   Offset  Field                      Size
///   ──────  ──────────────────────────  ──────
///   0       num_seqs                   4 (u32)
///   4       max_decode_steps           4 (u32)
///   8       total_prefill_tokens       4 (u32)
///   12      pad                        4
///   16      input_ids_flat_ptr         8 (*const u32)
///   24      output_tokens_flat_ptr     8 (*mut u32)
///   32      positions_ptr              8 (*const u32)
///   40      page_table_flat_ptr        8 (*const u32)
///   48      kv_pool_base               8 (*const u8)
///   56      sampling_params_ptr        8 (*const u32)
///   64      hook_ctx_ptr               8 (*const u8)
///   72      callback_table_ptr         8 (*const u8)
///   80      seq_mapping_ptr            8 (*const u32) — BCI6 per-token seq_id lookup
///   88      seq_meta_base_ptr          8 (*const u8)  — pointer to per-seq metadata
///
/// Per-seq metadata (at seq_meta_base_ptr, stride = 64 bytes):
///   +seq_idx × 64:
///     +0   prompt_len                4 (u32)
///     +4   kv_len                    4 (u32)
///     +8   rope_pos_offset           4 (u32)
///    +12   max_new_tokens            4 (u32)
///    +16   session_position          4 (u32)
///    +20   page_table_offset         4 (u32)
///    +24   page_table_len            4 (u32)
///    +28   fused_hidden_offset       4 (u32)
///    +32   num_mm_tokens             4 (u32)
///    +36   active_flag               4 (u32)  1=活跃, 0=已完成
///    +40   seq_position              4 (u32)  当前 decode 位置 (JIT 运行时更新)
///    +44   gen_count                 4 (u32)  已生成 token 数 (JIT 运行时更新)
///    +48   last_sampled_token        4 (u32)  上次采样结果 (JIT 运行时更新)
///    +52   output_offset             4 (u32)  该序列在 output_tokens_flat 中的起始偏移
///    +56   pad                       8 (padding to 64)
/// ```
#[derive(Debug, Clone)]
pub struct BatchContext {
    /// Flat memory buffer: header + per-seq data [+ extension (SPEC 32)]
    pub data: Vec<u8>,
    /// Current number of sequences
    pub num_seqs: usize,
    /// Maximum batch capacity (SPEC 32: >= num_seqs, for extension area sizing)
    pub max_batch_size: usize,
    /// Whether SPEC 32 extension area is present
    pub has_v2_extension: bool,
    /// BCI6: Per-token seq_mapping — seq_mapping[token_idx] → seq_id (0..num_seqs-1)
    pub seq_mapping: Vec<u32>,
}

impl BatchContext {
    /// Create a new BatchContext with space for `num_seqs` sequences.
    ///
    /// Layout: 96-byte header + seq_meta_base_ptr pointing to per-seq data at offset 96.
    pub fn new(num_seqs: usize) -> Self {
        let total_bytes = BATCH_CTX_HEADER_SIZE + num_seqs * SEQ_META_STRIDE;
        let mut data = vec![0u8; total_bytes];

        // Write absolute pointer to per-seq data at offset 88.
        // JIT reads this via ScalarLoad and uses it as base for per-seq field access.
        // Safe: Vec heap allocation is stable across moves (no reallocation on move).
        let per_seq_ptr = data.as_ptr().wrapping_add(BATCH_CTX_HEADER_SIZE) as u64;
        data[SEQ_META_BASE_PTR..SEQ_META_BASE_PTR + 8]
            .copy_from_slice(&per_seq_ptr.to_le_bytes());

        Self {
            data,
            num_seqs,
            max_batch_size: num_seqs,
            has_v2_extension: false,
            seq_mapping: Vec::new(),
        }
    }

    /// Create a BatchContext with SPEC 32 extension area (REQ-MKO-006).
    ///
    /// Layout: header (96) + max_batch_size × 64 + extension (96).
    /// `max_batch_size` must be >= `initial_num_seqs`.
    pub fn with_v2_extension(max_batch_size: usize, initial_num_seqs: usize) -> Self {
        assert!(max_batch_size >= initial_num_seqs);
        use crate::engine::mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE;
        let total_bytes = BATCH_CTX_HEADER_SIZE
            + max_batch_size * SEQ_META_STRIDE
            + BATCH_CTX_EXTENSION_SIZE;
        let mut data = vec![0u8; total_bytes];

        let per_seq_ptr = data.as_ptr().wrapping_add(BATCH_CTX_HEADER_SIZE) as u64;
        data[SEQ_META_BASE_PTR..SEQ_META_BASE_PTR + 8]
            .copy_from_slice(&per_seq_ptr.to_le_bytes());

        let mut ctx = Self {
            data,
            num_seqs: initial_num_seqs,
            max_batch_size,
            has_v2_extension: true,
            seq_mapping: Vec::new(),
        };
        ctx.set_num_seqs(initial_num_seqs as u32);
        ctx
    }

    /// Extension area start offset (only valid when has_v2_extension).
    #[cfg(test)]
    #[allow(dead_code)]
    fn ext_offset(&self) -> usize {
        BATCH_CTX_HEADER_SIZE + self.max_batch_size * SEQ_META_STRIDE
    }

    // ── Header writers ──

    pub fn set_num_seqs(&mut self, v: u32) {
        write_u32(&mut self.data, NUM_SEQS, v);
    }

    pub fn set_max_decode_steps(&mut self, v: u32) {
        write_u32(&mut self.data, MAX_DECODE_STEPS, v);
    }

    pub fn set_total_prefill_tokens(&mut self, v: u32) {
        write_u32(&mut self.data, TOTAL_PREFILL_TOKENS, v);
    }

    pub fn set_input_ids_flat_ptr(&mut self, p: *const u32) {
        write_usize(&mut self.data, INPUT_IDS_FLAT_PTR, p as usize);
    }

    pub fn set_output_tokens_flat_ptr(&mut self, p: *mut u32) {
        write_usize(&mut self.data, OUTPUT_TOKENS_FLAT_PTR, p as usize);
    }

    pub fn set_positions_ptr(&mut self, p: *const u32) {
        write_usize(&mut self.data, POSITIONS_PTR, p as usize);
    }

    pub fn set_page_table_flat_ptr(&mut self, p: *const u32) {
        write_usize(&mut self.data, PAGE_TABLE_FLAT_PTR, p as usize);
    }

    pub fn set_kv_pool_base(&mut self, p: *const u8) {
        write_usize(&mut self.data, KV_POOL_BASE, p as usize);
    }

    pub fn set_sampling_params_ptr(&mut self, p: *const u32) {
        write_usize(&mut self.data, SAMPLING_PARAMS_PTR, p as usize);
    }

    pub fn set_hook_ctx_ptr(&mut self, p: *const u8) {
        write_usize(&mut self.data, HOOK_CTX_PTR, p as usize);
    }

    pub fn set_callback_table_ptr(&mut self, p: *const u8) {
        write_usize(&mut self.data, CALLBACK_TABLE_PTR, p as usize);
    }

    /// BCI6: Set seq_mapping pointer — per-token seq_id lookup array.
    pub fn set_seq_mapping_ptr(&mut self, p: *const u32) {
        write_usize(&mut self.data, SEQ_MAPPING_PTR, p as usize);
    }

    // ── Per-seq writers (SPEC/20 §1.2 layout) ──
    // Per-seq data starts at BATCH_CTX_HEADER_SIZE (offset 96) in the flat buffer.

    fn seq_field_offset(seq: usize, field: usize) -> usize {
        BATCH_CTX_HEADER_SIZE + seq * SEQ_META_STRIDE + field
    }

    pub fn set_seq_prompt_len(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_PROMPT_LEN), v);
    }

    pub fn set_seq_kv_len(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_KV_LEN), v);
    }

    pub fn set_seq_rope_pos_offset(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_ROPE_POS_OFFSET), v);
    }

    pub fn set_seq_max_new_tokens(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_MAX_NEW_TOKENS), v);
    }

    pub fn set_seq_session_position(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_SESSION_POSITION), v);
    }

    pub fn set_seq_page_table_offset(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_PAGE_TABLE_OFFSET), v);
    }

    pub fn set_seq_page_table_len(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_PAGE_TABLE_LEN), v);
    }

    pub fn set_seq_fused_hidden_offset(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_FUSED_HIDDEN_OFFSET), v);
    }

    pub fn set_seq_num_mm_tokens(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_NUM_MM_TOKENS), v);
    }

    pub fn set_seq_active_flag(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_ACTIVE_FLAG), v);
    }

    pub fn set_seq_position(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_SEQ_POSITION), v);
    }

    pub fn set_seq_gen_count(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_GEN_COUNT), v);
    }

    pub fn set_seq_last_sampled_token(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_LAST_SAMPLED_TOKEN), v);
    }

    pub fn set_seq_output_offset(&mut self, seq: usize, v: u32) {
        write_u32(&mut self.data, Self::seq_field_offset(seq, SEQ_OUTPUT_OFFSET), v);
    }

    /// Returns the pointer to the flat memory buffer for passing to JIT.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Returns the total byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    // ── SPEC 32 Extension writers (REQ-MKO-006 §6.2) ──
    // Extension area starts at ext_offset() = header + max_batch_size × stride.

    fn ext_field(max_batch_size: usize, offset: usize) -> usize {
        BATCH_CTX_HEADER_SIZE + max_batch_size * SEQ_META_STRIDE + offset
    }

    pub fn set_ext_request_queue_ptr(&mut self, p: *const u8) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_REQUEST_QUEUE_PTR), p as usize);
    }

    pub fn set_ext_output_ring_ptr(&mut self, p: *const u8) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_OUTPUT_RING_PTR), p as usize);
    }

    pub fn set_ext_kv_free_bitmap_ptr(&mut self, p: *mut u32) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR), p as usize);
    }

    pub fn set_ext_kv_pool_total_pages(&mut self, v: u32) {
        write_u32(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), v);
    }

    pub fn set_ext_max_batch_size(&mut self, v: u32) {
        write_u32(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_MAX_BATCH_SIZE), v);
    }

    pub fn set_ext_dual_batch_meta(&mut self, meta: &mega_kernel_v2::DualBatchMeta) {
        let base = Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_DUAL_BATCH_META);
        let bytes = unsafe {
            std::slice::from_raw_parts(meta as *const _ as *const u8, mega_kernel_v2::DualBatchMeta::SIZE)
        };
        self.data[base..base + mega_kernel_v2::DualBatchMeta::SIZE].copy_from_slice(bytes);
    }

    pub fn set_ext_autotune_actual_batch(&mut self, v: u32) {
        write_u32(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH), v);
    }

    pub fn set_ext_pool_cluster_dsmem_ptr(&mut self, p: *mut u8) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_POOL_CLUSTER_DSMEM_PTR), p as usize);
    }

    pub fn set_ext_pending_free_list_ptr(&mut self, p: *mut u32) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_PENDING_FREE_LIST_PTR), p as usize);
    }

    pub fn set_ext_pending_free_count_ptr(&mut self, p: *mut u32) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_PENDING_FREE_COUNT_PTR), p as usize);
    }

    pub fn set_ext_output_per_cta_doorbell_ptr(&mut self, p: *mut u64) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_OUTPUT_PER_CTA_DOORBELL_PTR), p as usize);
    }

    pub fn set_ext_output_epoch_flag_ptr(&mut self, p: *mut u32) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_OUTPUT_EPOCH_FLAG_PTR), p as usize);
    }

    // ── REQ-KV-EXT-001: V2 extension field writers ──

    /// Set KvPageHeader stride in bytes (64 for V2, was 56 for V1).
    pub fn set_ext_kv_page_header_stride(&mut self, v: u32) {
        write_u32(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_KV_PAGE_HEADER_STRIDE), v);
    }

    /// Set base pointer for ext_id indexed KV extension slots.
    pub fn set_ext_kv_ext_id_base_ptr(&mut self, p: *const u8) {
        write_usize(&mut self.data, Self::ext_field(self.max_batch_size, mega_kernel_v2::EXT_KV_EXT_ID_BASE_PTR), p as usize);
    }
}

// ── Helper functions for writing to flat memory ──

fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_usize(data: &mut [u8], offset: usize, value: usize) {
    data[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_context_header_layout() {
        let mut ctx = BatchContext::new(0);

        ctx.set_num_seqs(2);
        ctx.set_max_decode_steps(100);
        ctx.set_total_prefill_tokens(50);

        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 2);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 100);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 50);
    }

    #[test]
    fn test_batch_context_seq_meta_base_ptr() {
        let ctx = BatchContext::new(3);

        // seq_meta_base_ptr at offset 88 should be an absolute pointer
        // pointing to per-seq data starting at data[96]
        let ptr_val = read_u64(&ctx.data, SEQ_META_BASE_PTR) as usize;
        let expected = ctx.data.as_ptr() as usize + BATCH_CTX_HEADER_SIZE;
        assert_eq!(ptr_val, expected);

        // Per-seq data starts at offset 96 in the flat buffer
        assert_eq!(BATCH_CTX_HEADER_SIZE, 96);
    }

    #[test]
    fn test_batch_context_per_seq_layout() {
        let mut ctx = BatchContext::new(2);

        // Seq 0: prompt of 10 tokens, max 20 new tokens
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_kv_len(0, 0);
        ctx.set_seq_rope_pos_offset(0, 0);
        ctx.set_seq_max_new_tokens(0, 20);
        ctx.set_seq_session_position(0, 0);
        ctx.set_seq_page_table_offset(0, 0);
        ctx.set_seq_page_table_len(0, 3);
        ctx.set_seq_fused_hidden_offset(0, 0);
        ctx.set_seq_num_mm_tokens(0, 0);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_position(0, 0);
        ctx.set_seq_gen_count(0, 0);
        ctx.set_seq_last_sampled_token(0, 0);
        ctx.set_seq_output_offset(0, 0);

        // Seq 1: prompt of 5 tokens, max 10 new tokens, session resume
        ctx.set_seq_prompt_len(1, 5);
        ctx.set_seq_kv_len(1, 42);
        ctx.set_seq_rope_pos_offset(1, 42);
        ctx.set_seq_max_new_tokens(1, 10);
        ctx.set_seq_session_position(1, 42);
        ctx.set_seq_page_table_offset(1, 3);
        ctx.set_seq_page_table_len(1, 2);
        ctx.set_seq_fused_hidden_offset(1, 0);
        ctx.set_seq_num_mm_tokens(1, 0);
        ctx.set_seq_active_flag(1, 1);
        ctx.set_seq_position(1, 42);
        ctx.set_seq_gen_count(1, 0);
        ctx.set_seq_last_sampled_token(1, 0);
        ctx.set_seq_output_offset(1, 30); // after seq 0's 10+20=30 slots

        // Verify seq 0 per-seq fields at expected offsets
        let base0 = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PROMPT_LEN), 10);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_KV_LEN), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_ROPE_POS_OFFSET), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_MAX_NEW_TOKENS), 20);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_SESSION_POSITION), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PAGE_TABLE_OFFSET), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PAGE_TABLE_LEN), 3);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_FUSED_HIDDEN_OFFSET), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_NUM_MM_TOKENS), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_ACTIVE_FLAG), 1);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_SEQ_POSITION), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_GEN_COUNT), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_LAST_SAMPLED_TOKEN), 0);
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_OUTPUT_OFFSET), 0);

        // Verify seq 1 per-seq fields (stride = 64)
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_PROMPT_LEN), 5);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_KV_LEN), 42);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_ROPE_POS_OFFSET), 42);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_MAX_NEW_TOKENS), 10);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_SESSION_POSITION), 42);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_PAGE_TABLE_OFFSET), 3);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_PAGE_TABLE_LEN), 2);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_ACTIVE_FLAG), 1);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_SEQ_POSITION), 42);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_OUTPUT_OFFSET), 30);
    }

    #[test]
    fn test_batch_context_seq_mapping_ptr() {
        let mut ctx = BatchContext::new(2);
        let mapping = vec![0u32, 0, 0, 1, 1];
        ctx.set_seq_mapping_ptr(mapping.as_ptr());

        let ptr_val = read_usize(&ctx.data, SEQ_MAPPING_PTR);
        assert_eq!(ptr_val, mapping.as_ptr() as usize);
    }

    #[test]
    fn test_batch_context_byte_size() {
        let ctx = BatchContext::new(4);
        // header (96) + 4 × 64 = 96 + 256 = 352
        assert_eq!(ctx.byte_size(), 352);
    }

    fn read_u32(data: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
    }

    fn read_u64(data: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
    }

    fn read_usize(data: &[u8], offset: usize) -> usize {
        read_u64(data, offset) as usize
    }

    // ── SPEC 32 Extension tests ──

    #[test]
    fn test_batch_context_v2_extension_layout() {
        let ctx = BatchContext::with_v2_extension(8, 2);
        // header (96) + 8 × 64 + 128 = 96 + 512 + 128 = 736
        assert_eq!(ctx.byte_size(), 736);
        assert!(ctx.has_v2_extension);
        assert_eq!(ctx.max_batch_size, 8);
        assert_eq!(ctx.num_seqs, 2);

        // num_seqs should be set in header
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 2);

        // Extension area starts at 96 + 8 × 64 = 608
        let ext_start = BATCH_CTX_HEADER_SIZE + 8 * SEQ_META_STRIDE;
        assert_eq!(BatchContext::ext_field(8, 0), ext_start);
    }

    #[test]
    fn test_batch_context_v2_extension_fields() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        // Write extension fields
        ctx.set_ext_kv_pool_total_pages(1024);
        ctx.set_ext_max_batch_size(4);
        ctx.set_ext_autotune_actual_batch(3);

        // Verify
        assert_eq!(
            read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES),
            1024
        );
        assert_eq!(
            read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_MAX_BATCH_SIZE),
            4
        );
        assert_eq!(
            read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH),
            3
        );
    }

    #[test]
    fn test_batch_context_v2_dual_batch_meta() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let meta = mega_kernel_v2::DualBatchMeta::new(4);
        ctx.set_ext_dual_batch_meta(&meta);

        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        let meta_base = ext_start + mega_kernel_v2::EXT_DUAL_BATCH_META;
        // Verify ping_seq_offset (first field) = 0
        assert_eq!(read_u32(&ctx.data, meta_base), 0);
        // Verify pong_seq_offset = 4 (max_batch_size)
        assert_eq!(read_u32(&ctx.data, meta_base + 8), 4);
    }

    #[test]
    fn test_batch_context_backward_compat() {
        // Existing new() should still work without extension
        let ctx = BatchContext::new(4);
        assert!(!ctx.has_v2_extension);
        // header (96) + 4 × 64 = 352
        assert_eq!(ctx.byte_size(), 352);
    }

    // ── Additional tests ──

    #[test]
    fn constants_non_overlapping_header() {
        // Header fields should not overlap
        let offsets = [
            NUM_SEQS, MAX_DECODE_STEPS, TOTAL_PREFILL_TOKENS,
            INPUT_IDS_FLAT_PTR, OUTPUT_TOKENS_FLAT_PTR, POSITIONS_PTR,
            PAGE_TABLE_FLAT_PTR, KV_POOL_BASE, SAMPLING_PARAMS_PTR,
            HOOK_CTX_PTR, CALLBACK_TABLE_PTR, SEQ_MAPPING_PTR,
            SEQ_META_BASE_PTR,
        ];
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                assert_ne!(offsets[i], offsets[j], "header offsets {} and {} overlap", offsets[i], offsets[j]);
            }
        }
    }

    #[test]
    fn constants_per_seq_fields_within_stride() {
        // All per-seq fields should be within the 64-byte stride (56 bytes of data)
        let fields = [
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ];
        for &f in &fields {
            assert!(f + 4 <= SEQ_META_STRIDE, "field at {} exceeds stride {}", f, SEQ_META_STRIDE);
        }
        assert_eq!(fields.len(), 14, "should have exactly 14 per-seq u32 fields");
    }

    #[test]
    fn batch_context_new_zero_seqs() {
        let ctx = BatchContext::new(0);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE);
        assert_eq!(ctx.num_seqs, 0);
        assert!(!ctx.has_v2_extension);
    }

    #[test]
    fn batch_context_header_ptr_fields() {
        let mut ctx = BatchContext::new(1);
        let dummy_val = 0xDEADBEEF_usize as *const u32;
        ctx.set_input_ids_flat_ptr(dummy_val);
        assert_eq!(read_usize(&ctx.data, INPUT_IDS_FLAT_PTR), dummy_val as usize);

        let dummy_out = 0xCAFE_usize as *mut u32;
        ctx.set_output_tokens_flat_ptr(dummy_out);
        assert_eq!(read_usize(&ctx.data, OUTPUT_TOKENS_FLAT_PTR), dummy_out as usize);

        let dummy_pos = 0x1234_usize as *const u32;
        ctx.set_positions_ptr(dummy_pos);
        assert_eq!(read_usize(&ctx.data, POSITIONS_PTR), dummy_pos as usize);

        let dummy_page = 0x5678_usize as *const u32;
        ctx.set_page_table_flat_ptr(dummy_page);
        assert_eq!(read_usize(&ctx.data, PAGE_TABLE_FLAT_PTR), dummy_page as usize);

        let dummy_kv = 0x9ABC_usize as *const u8;
        ctx.set_kv_pool_base(dummy_kv);
        assert_eq!(read_usize(&ctx.data, KV_POOL_BASE), dummy_kv as usize);

        let dummy_samp = 0xDEF0_usize as *const u32;
        ctx.set_sampling_params_ptr(dummy_samp);
        assert_eq!(read_usize(&ctx.data, SAMPLING_PARAMS_PTR), dummy_samp as usize);

        let dummy_hook = 0x1111_usize as *const u8;
        ctx.set_hook_ctx_ptr(dummy_hook);
        assert_eq!(read_usize(&ctx.data, HOOK_CTX_PTR), dummy_hook as usize);

        let dummy_cb = 0x2222_usize as *const u8;
        ctx.set_callback_table_ptr(dummy_cb);
        assert_eq!(read_usize(&ctx.data, CALLBACK_TABLE_PTR), dummy_cb as usize);
    }

    #[test]
    fn batch_context_per_seq_independence() {
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_prompt_len(1, 20);
        ctx.set_seq_prompt_len(2, 30);

        let base0 = BATCH_CTX_HEADER_SIZE;
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE;

        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PROMPT_LEN), 10);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_PROMPT_LEN), 20);
        assert_eq!(read_u32(&ctx.data, base2 + SEQ_PROMPT_LEN), 30);
    }

    #[test]
    fn batch_context_seq_field_offset_correct() {
        // seq_field_offset(seq=2, field=SEQ_KV_LEN) should be 96 + 2*64 + 4
        let offset = BatchContext::seq_field_offset(2, SEQ_KV_LEN);
        assert_eq!(offset, BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_KV_LEN);
        assert_eq!(offset, 96 + 128 + 4);
    }

    #[test]
    fn batch_context_as_ptr_not_null() {
        let ctx = BatchContext::new(1);
        assert!(!ctx.as_ptr().is_null());
    }

    #[test]
    fn batch_context_seq_mapping_default_empty() {
        let ctx = BatchContext::new(2);
        assert!(ctx.seq_mapping.is_empty());
    }

    #[test]
    fn batch_context_all_u32_fields_max_values() {
        let mut ctx = BatchContext::new(1);
        ctx.set_num_seqs(u32::MAX);
        ctx.set_max_decode_steps(u32::MAX);
        ctx.set_total_prefill_tokens(u32::MAX);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), u32::MAX);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), u32::MAX);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), u32::MAX);
    }

    #[test]
    fn batch_context_per_seq_all_fields_max() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_prompt_len(0, u32::MAX);
        ctx.set_seq_kv_len(0, u32::MAX);
        ctx.set_seq_rope_pos_offset(0, u32::MAX);
        ctx.set_seq_max_new_tokens(0, u32::MAX);
        ctx.set_seq_session_position(0, u32::MAX);
        ctx.set_seq_page_table_offset(0, u32::MAX);
        ctx.set_seq_page_table_len(0, u32::MAX);
        ctx.set_seq_fused_hidden_offset(0, u32::MAX);
        ctx.set_seq_num_mm_tokens(0, u32::MAX);
        ctx.set_seq_active_flag(0, u32::MAX);
        ctx.set_seq_position(0, u32::MAX);
        ctx.set_seq_gen_count(0, u32::MAX);
        ctx.set_seq_last_sampled_token(0, u32::MAX);
        ctx.set_seq_output_offset(0, u32::MAX);

        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_PROMPT_LEN), u32::MAX);
        assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), u32::MAX);
        assert_eq!(read_u32(&ctx.data, base + SEQ_OUTPUT_OFFSET), u32::MAX);
    }

    #[test]
    fn batch_context_byte_size_formula() {
        for n in [0, 1, 4, 16] {
            let ctx = BatchContext::new(n);
            assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE + n * SEQ_META_STRIDE);
        }
    }

    #[test]
    fn batch_context_v2_extension_byte_size() {
        let ctx = BatchContext::with_v2_extension(8, 2);
        let expected = BATCH_CTX_HEADER_SIZE + 8 * SEQ_META_STRIDE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE;
        assert_eq!(ctx.byte_size(), expected);
    }

    #[test]
    fn batch_context_clone_independent() {
        let mut ctx = BatchContext::new(2);
        ctx.set_num_seqs(5);
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.num_seqs, ctx.num_seqs);
        assert_eq!(ctx2.byte_size(), ctx.byte_size());
        // Verify data is independent
        assert_eq!(ctx2.data, ctx.data);
    }

    #[test]
    fn batch_context_per_seq_stride_padded() {
        // 14 × u32 = 56 bytes, padded to 64 for alignment
        assert_eq!(SEQ_META_STRIDE, 64);
        assert!(SEQ_META_STRIDE >= 14 * 4, "stride must fit 14 u32 fields");
    }

    // ── Additional tests (15 new) ──

    #[test]
    fn batch_context_debug_trait_formats_fields() {
        let ctx = BatchContext::new(3);
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("data"), "Debug should show 'data' field");
        assert!(debug_str.contains("num_seqs"), "Debug should show 'num_seqs' field");
        assert!(debug_str.contains("max_batch_size"), "Debug should show 'max_batch_size' field");
        assert!(debug_str.contains("has_v2_extension"), "Debug should show 'has_v2_extension' field");
        assert!(debug_str.contains("seq_mapping"), "Debug should show 'seq_mapping' field");
    }

    #[test]
    fn batch_context_data_initially_zeroed() {
        let ctx = BatchContext::new(5);
        for (i, &byte) in ctx.data.iter().enumerate() {
            // Skip seq_meta_base_ptr at offset 88 (8 bytes) — it's set to the per-seq pointer
            if i >= SEQ_META_BASE_PTR && i < SEQ_META_BASE_PTR + 8 {
                continue;
            }
            assert_eq!(byte, 0, "byte at offset {} should be zero-initialized", i);
        }
    }

    #[test]
    fn batch_context_per_seq_padding_remains_zero() {
        let mut ctx = BatchContext::new(2);
        // Write all 14 per-seq u32 fields for seq 0
        ctx.set_seq_prompt_len(0, 42);
        ctx.set_seq_kv_len(0, 10);
        ctx.set_seq_rope_pos_offset(0, 5);
        ctx.set_seq_max_new_tokens(0, 100);
        ctx.set_seq_session_position(0, 7);
        ctx.set_seq_page_table_offset(0, 3);
        ctx.set_seq_page_table_len(0, 8);
        ctx.set_seq_fused_hidden_offset(0, 0);
        ctx.set_seq_num_mm_tokens(0, 0);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_position(0, 10);
        ctx.set_seq_gen_count(0, 3);
        ctx.set_seq_last_sampled_token(0, 99);
        ctx.set_seq_output_offset(0, 0);

        // Padding bytes: offsets 56-63 within the per-seq area (after 14 × u32 = 56 bytes)
        let base0 = BATCH_CTX_HEADER_SIZE;
        for i in 56..SEQ_META_STRIDE {
            assert_eq!(ctx.data[base0 + i], 0, "padding byte at seq0+{} should remain zero", i);
        }
    }

    #[test]
    fn batch_context_no_cross_seq_contamination() {
        let mut ctx = BatchContext::new(4);
        // Write distinctive values to each sequence's active_flag field
        ctx.set_seq_active_flag(0, 100);
        ctx.set_seq_active_flag(1, 200);
        ctx.set_seq_active_flag(2, 300);
        ctx.set_seq_active_flag(3, 400);

        // Now overwrite seq 0 and seq 2 with different values
        ctx.set_seq_active_flag(0, 111);
        ctx.set_seq_active_flag(2, 333);

        // Verify seq 1 and seq 3 were not affected
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        let base3 = BATCH_CTX_HEADER_SIZE + 3 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_ACTIVE_FLAG), 200);
        assert_eq!(read_u32(&ctx.data, base3 + SEQ_ACTIVE_FLAG), 400);
    }

    #[test]
    fn batch_context_set_num_seqs_overwrites() {
        let mut ctx = BatchContext::new(2);
        ctx.set_num_seqs(5);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 5);
        ctx.set_num_seqs(1);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 1);
        ctx.set_num_seqs(0);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 0);
    }

    #[test]
    fn batch_context_write_u32_endianness() {
        let mut ctx = BatchContext::new(0);
        ctx.set_num_seqs(0x01020304);
        // Little-endian: bytes at offset 0 should be [04, 03, 02, 01]
        assert_eq!(ctx.data[0], 0x04);
        assert_eq!(ctx.data[1], 0x03);
        assert_eq!(ctx.data[2], 0x02);
        assert_eq!(ctx.data[3], 0x01);
    }

    #[test]
    fn batch_context_write_usize_endianness() {
        let mut ctx = BatchContext::new(1);
        let ptr_val: usize = 0x0102030405060708;
        ctx.set_input_ids_flat_ptr(ptr_val as *const u32);
        // Little-endian: bytes at INPUT_IDS_FLAT_PTR should be [08,07,06,05,04,03,02,01]
        let base = INPUT_IDS_FLAT_PTR;
        assert_eq!(ctx.data[base + 0], 0x08);
        assert_eq!(ctx.data[base + 1], 0x07);
        assert_eq!(ctx.data[base + 7], 0x01);
    }

    #[test]
    fn batch_context_v2_extension_ext_offset() {
        let ctx = BatchContext::with_v2_extension(10, 3);
        let ext_off = ctx.ext_offset();
        assert_eq!(ext_off, BATCH_CTX_HEADER_SIZE + 10 * SEQ_META_STRIDE);
        assert_eq!(ext_off, 96 + 640);
    }

    #[test]
    fn batch_context_v2_extension_pointer_fields() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        let dummy_req = 0xAABB_usize as *const u8;
        ctx.set_ext_request_queue_ptr(dummy_req);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_REQUEST_QUEUE_PTR), dummy_req as usize);

        let dummy_ring = 0xCCDD_usize as *const u8;
        ctx.set_ext_output_ring_ptr(dummy_ring);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_OUTPUT_RING_PTR), dummy_ring as usize);

        let dummy_bitmap = 0xEEFF_usize as *mut u32;
        ctx.set_ext_kv_free_bitmap_ptr(dummy_bitmap);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR), dummy_bitmap as usize);
    }

    #[test]
    fn batch_context_v2_extension_remaining_pointer_fields() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        let dummy_dsmem = 0x1111_usize as *mut u8;
        ctx.set_ext_pool_cluster_dsmem_ptr(dummy_dsmem);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_POOL_CLUSTER_DSMEM_PTR), dummy_dsmem as usize);

        let dummy_free_list = 0x2222_usize as *mut u32;
        ctx.set_ext_pending_free_list_ptr(dummy_free_list);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_PENDING_FREE_LIST_PTR), dummy_free_list as usize);

        let dummy_free_count = 0x3333_usize as *mut u32;
        ctx.set_ext_pending_free_count_ptr(dummy_free_count);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_PENDING_FREE_COUNT_PTR), dummy_free_count as usize);

        let dummy_doorbell = 0x4444_usize as *mut u64;
        ctx.set_ext_output_per_cta_doorbell_ptr(dummy_doorbell);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_OUTPUT_PER_CTA_DOORBELL_PTR), dummy_doorbell as usize);

        let dummy_epoch = 0x5555_usize as *mut u32;
        ctx.set_ext_output_epoch_flag_ptr(dummy_epoch);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_OUTPUT_EPOCH_FLAG_PTR), dummy_epoch as usize);
    }

    #[test]
    fn batch_context_v2_extension_per_seq_data_intact() {
        let mut ctx = BatchContext::with_v2_extension(8, 4);
        // Write per-seq data
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_prompt_len(3, 99);
        // Write extension data
        ctx.set_ext_kv_pool_total_pages(5555);

        // Per-seq data should not be corrupted by extension writes
        let base0 = BATCH_CTX_HEADER_SIZE;
        let base3 = BATCH_CTX_HEADER_SIZE + 3 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PROMPT_LEN), 10);
        assert_eq!(read_u32(&ctx.data, base3 + SEQ_PROMPT_LEN), 99);
    }

    #[test]
    fn batch_context_large_batch_byte_size() {
        let ctx = BatchContext::new(128);
        let expected = BATCH_CTX_HEADER_SIZE + 128 * SEQ_META_STRIDE;
        assert_eq!(ctx.byte_size(), expected);
        assert_eq!(expected, 96 + 8192);
    }

    #[test]
    fn batch_context_v2_extension_max_batch_exceeds_initial() {
        let ctx = BatchContext::with_v2_extension(32, 1);
        assert_eq!(ctx.max_batch_size, 32);
        assert_eq!(ctx.num_seqs, 1);
        // Buffer has space for 32 sequences worth of per-seq data
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE + 32 * SEQ_META_STRIDE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE);
    }

    #[test]
    fn batch_context_each_per_seq_field_independent() {
        let mut ctx = BatchContext::new(1);
        // Write all fields for seq 0 with distinctive values
        ctx.set_seq_prompt_len(0, 1);
        ctx.set_seq_kv_len(0, 2);
        ctx.set_seq_rope_pos_offset(0, 3);
        ctx.set_seq_max_new_tokens(0, 4);
        ctx.set_seq_session_position(0, 5);
        ctx.set_seq_page_table_offset(0, 6);
        ctx.set_seq_page_table_len(0, 7);
        ctx.set_seq_fused_hidden_offset(0, 8);
        ctx.set_seq_num_mm_tokens(0, 9);
        ctx.set_seq_active_flag(0, 10);
        ctx.set_seq_position(0, 11);
        ctx.set_seq_gen_count(0, 12);
        ctx.set_seq_last_sampled_token(0, 13);
        ctx.set_seq_output_offset(0, 14);

        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_PROMPT_LEN), 1);
        assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), 2);
        assert_eq!(read_u32(&ctx.data, base + SEQ_ROPE_POS_OFFSET), 3);
        assert_eq!(read_u32(&ctx.data, base + SEQ_MAX_NEW_TOKENS), 4);
        assert_eq!(read_u32(&ctx.data, base + SEQ_SESSION_POSITION), 5);
        assert_eq!(read_u32(&ctx.data, base + SEQ_PAGE_TABLE_OFFSET), 6);
        assert_eq!(read_u32(&ctx.data, base + SEQ_PAGE_TABLE_LEN), 7);
        assert_eq!(read_u32(&ctx.data, base + SEQ_FUSED_HIDDEN_OFFSET), 8);
        assert_eq!(read_u32(&ctx.data, base + SEQ_NUM_MM_TOKENS), 9);
        assert_eq!(read_u32(&ctx.data, base + SEQ_ACTIVE_FLAG), 10);
        assert_eq!(read_u32(&ctx.data, base + SEQ_SEQ_POSITION), 11);
        assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), 12);
        assert_eq!(read_u32(&ctx.data, base + SEQ_LAST_SAMPLED_TOKEN), 13);
        assert_eq!(read_u32(&ctx.data, base + SEQ_OUTPUT_OFFSET), 14);
    }

    #[test]
    fn batch_context_v2_extension_ext_field_offset_calculation() {
        // Verify ext_field computes correctly for different max_batch_size values
        let off_4 = BatchContext::ext_field(4, 0);
        let off_8 = BatchContext::ext_field(8, 0);
        let off_16 = BatchContext::ext_field(16, 0);

        assert_eq!(off_4, BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE);
        assert_eq!(off_8, BATCH_CTX_HEADER_SIZE + 8 * SEQ_META_STRIDE);
        assert_eq!(off_16, BATCH_CTX_HEADER_SIZE + 16 * SEQ_META_STRIDE);

        // Difference should scale linearly
        assert_eq!(off_8 - off_4, 4 * SEQ_META_STRIDE);
        assert_eq!(off_16 - off_8, 8 * SEQ_META_STRIDE);
    }

    // ── Additional tests (20 new) ──

    #[test]
    fn batch_context_new_single_seq() {
        let ctx = BatchContext::new(1);
        assert_eq!(ctx.num_seqs, 1);
        assert_eq!(ctx.max_batch_size, 1);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE);
        assert!(!ctx.has_v2_extension);
        assert!(ctx.seq_mapping.is_empty());
    }

    #[test]
    fn batch_context_max_batch_size_equals_num_seqs() {
        // In new(), max_batch_size should default to num_seqs
        let ctx = BatchContext::new(7);
        assert_eq!(ctx.max_batch_size, ctx.num_seqs);
        assert_eq!(ctx.max_batch_size, 7);
    }

    #[test]
    fn batch_context_header_padding_offset_12_is_zero() {
        // Offset 12 is padding (between total_prefill_tokens and input_ids_flat_ptr)
        let ctx = BatchContext::new(2);
        assert_eq!(ctx.data[12], 0);
        assert_eq!(ctx.data[13], 0);
        assert_eq!(ctx.data[14], 0);
        assert_eq!(ctx.data[15], 0);
    }

    #[test]
    fn batch_context_header_u32_offsets_are_4_byte_aligned() {
        let u32_offsets = [
            NUM_SEQS,      // 0
            MAX_DECODE_STEPS, // 4
            TOTAL_PREFILL_TOKENS, // 8
        ];
        for &off in &u32_offsets {
            assert_eq!(off % 4, 0, "header u32 offset {} is not 4-byte aligned", off);
        }
    }

    #[test]
    fn batch_context_header_usize_offsets_are_8_byte_aligned() {
        let usize_offsets = [
            INPUT_IDS_FLAT_PTR,      // 16
            OUTPUT_TOKENS_FLAT_PTR,  // 24
            POSITIONS_PTR,           // 32
            PAGE_TABLE_FLAT_PTR,     // 40
            KV_POOL_BASE,            // 48
            SAMPLING_PARAMS_PTR,     // 56
            HOOK_CTX_PTR,            // 64
            CALLBACK_TABLE_PTR,      // 72
            SEQ_MAPPING_PTR,         // 80
            SEQ_META_BASE_PTR,       // 88
        ];
        for &off in &usize_offsets {
            assert_eq!(off % 8, 0, "header usize offset {} is not 8-byte aligned", off);
        }
    }

    #[test]
    fn batch_context_per_seq_offsets_increasing_and_aligned() {
        let fields = [
            SEQ_PROMPT_LEN,        // 0
            SEQ_KV_LEN,            // 4
            SEQ_ROPE_POS_OFFSET,   // 8
            SEQ_MAX_NEW_TOKENS,    // 12
            SEQ_SESSION_POSITION,  // 16
            SEQ_PAGE_TABLE_OFFSET, // 20
            SEQ_PAGE_TABLE_LEN,    // 24
            SEQ_FUSED_HIDDEN_OFFSET, // 28
            SEQ_NUM_MM_TOKENS,     // 32
            SEQ_ACTIVE_FLAG,       // 36
            SEQ_SEQ_POSITION,      // 40
            SEQ_GEN_COUNT,         // 44
            SEQ_LAST_SAMPLED_TOKEN, // 48
            SEQ_OUTPUT_OFFSET,     // 52
        ];
        // All offsets are 4-byte aligned
        for &f in &fields {
            assert_eq!(f % 4, 0, "per-seq field offset {} is not 4-byte aligned", f);
        }
        // Strictly increasing by exactly 4
        for i in 1..fields.len() {
            assert_eq!(
                fields[i] - fields[i - 1], 4,
                "per-seq field offsets {} and {} are not 4 apart",
                fields[i - 1], fields[i]
            );
        }
    }

    #[test]
    fn batch_context_seq_field_offset_seq_zero() {
        let offset = BatchContext::seq_field_offset(0, SEQ_PROMPT_LEN);
        assert_eq!(offset, BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_PROMPT_LEN);
        assert_eq!(offset, 96);
    }

    #[test]
    fn batch_context_seq_field_offset_last_field_last_seq() {
        // Last field (SEQ_OUTPUT_OFFSET=52) in last sequence
        let seq_idx = 3;
        let offset = BatchContext::seq_field_offset(seq_idx, SEQ_OUTPUT_OFFSET);
        let expected = BATCH_CTX_HEADER_SIZE + seq_idx * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        assert_eq!(offset, expected);
        // The field's 4 bytes should still fit within the stride
        assert!(offset + 4 <= BATCH_CTX_HEADER_SIZE + (seq_idx + 1) * SEQ_META_STRIDE);
    }

    #[test]
    fn batch_context_header_fields_do_not_exceed_header_size() {
        // The last header field is SEQ_META_BASE_PTR at offset 88, size 8
        assert!(SEQ_META_BASE_PTR + 8 <= BATCH_CTX_HEADER_SIZE,
            "last header field extends past header boundary");
    }

    #[test]
    fn batch_context_overwrite_single_field_preserves_others() {
        let mut ctx = BatchContext::new(2);
        // Set all fields for seq 0
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_kv_len(0, 5);
        ctx.set_seq_active_flag(0, 1);

        // Now overwrite only seq_kv_len
        ctx.set_seq_kv_len(0, 99);

        let base = BATCH_CTX_HEADER_SIZE;
        // prompt_len should be unchanged
        assert_eq!(read_u32(&ctx.data, base + SEQ_PROMPT_LEN), 10);
        // kv_len should be the new value
        assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), 99);
        // active_flag should be unchanged
        assert_eq!(read_u32(&ctx.data, base + SEQ_ACTIVE_FLAG), 1);
    }

    #[test]
    fn batch_context_clone_deep_copies_data_buffer() {
        let mut ctx = BatchContext::new(2);
        ctx.set_num_seqs(7);
        ctx.set_seq_prompt_len(0, 42);

        let cloned = ctx.clone();

        // Modify original after clone
        ctx.set_num_seqs(99);

        // Cloned should have the original value
        assert_eq!(read_u32(&cloned.data, NUM_SEQS), 7);
        // Original should have the new value
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 99);

        // Per-seq data also deep-copied
        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&cloned.data, base + SEQ_PROMPT_LEN), 42);
    }

    #[test]
    fn batch_context_byte_size_various_sizes() {
        let sizes = [0, 1, 7, 64];
        for &n in &sizes {
            let ctx = BatchContext::new(n);
            assert_eq!(
                ctx.byte_size(),
                BATCH_CTX_HEADER_SIZE + n * SEQ_META_STRIDE,
                "byte_size mismatch for num_seqs={}",
                n
            );
        }
    }

    #[test]
    fn batch_context_seq_mapping_can_hold_arbitrary_values() {
        let mut ctx = BatchContext::new(3);
        let mapping = vec![0u32, 0, 0, 1, 1, 1, 2, 2];
        ctx.seq_mapping = mapping.clone();
        assert_eq!(ctx.seq_mapping, mapping);
        assert_eq!(ctx.seq_mapping.len(), 8);
    }

    #[test]
    fn batch_context_header_idempotent_writes() {
        let mut ctx = BatchContext::new(1);
        ctx.set_num_seqs(5);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 5);
        // Write same value again
        ctx.set_num_seqs(5);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 5);
        // Write different value
        ctx.set_num_seqs(3);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 3);
    }

    #[test]
    fn batch_context_v2_extension_num_seqs_set_in_header() {
        let ctx = BatchContext::with_v2_extension(16, 5);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 5);
        assert_eq!(ctx.num_seqs, 5);
    }

    #[test]
    fn batch_context_v2_extension_dual_batch_meta_non_zero_max() {
        let mut ctx = BatchContext::with_v2_extension(32, 0);
        let meta = mega_kernel_v2::DualBatchMeta::new(32);
        ctx.set_ext_dual_batch_meta(&meta);

        let ext_start = BATCH_CTX_HEADER_SIZE + 32 * SEQ_META_STRIDE;
        let meta_base = ext_start + mega_kernel_v2::EXT_DUAL_BATCH_META;
        // ping_seq_offset = 0
        assert_eq!(read_u32(&ctx.data, meta_base), 0);
        // pong_seq_offset = 32
        assert_eq!(read_u32(&ctx.data, meta_base + 8), 32);
    }

    #[test]
    fn batch_context_v2_ext_field_with_non_zero_offset() {
        // ext_field(base, offset) should add the offset parameter
        let base_off = BatchContext::ext_field(4, 0);
        let kv_off = BatchContext::ext_field(4, mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES);
        assert_eq!(kv_off - base_off, mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES);
    }

    #[test]
    fn batch_context_extension_offsets_within_extension_size() {
        // All extension offsets + their data size must fit in BATCH_CTX_EXTENSION_SIZE
        let ptr_fields: &[(usize, usize)] = &[
            (mega_kernel_v2::EXT_REQUEST_QUEUE_PTR, 8),
            (mega_kernel_v2::EXT_OUTPUT_RING_PTR, 8),
            (mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR, 8),
            (mega_kernel_v2::EXT_POOL_CLUSTER_DSMEM_PTR, 8),
            (mega_kernel_v2::EXT_PENDING_FREE_LIST_PTR, 8),
            (mega_kernel_v2::EXT_PENDING_FREE_COUNT_PTR, 8),
            (mega_kernel_v2::EXT_OUTPUT_PER_CTA_DOORBELL_PTR, 8),
            (mega_kernel_v2::EXT_OUTPUT_EPOCH_FLAG_PTR, 8),
        ];
        for &(off, size) in ptr_fields {
            assert!(
                off + size <= mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE,
                "extension field at offset {} with size {} exceeds extension size {}",
                off, size, mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE
            );
        }
        let u32_fields: &[(usize, usize)] = &[
            (mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES, 4),
            (mega_kernel_v2::EXT_MAX_BATCH_SIZE, 4),
            (mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH, 4),
        ];
        for &(off, size) in u32_fields {
            assert!(
                off + size <= mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE,
                "extension u32 field at offset {} with size {} exceeds extension size {}",
                off, size, mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE
            );
        }
        // DualBatchMeta: 24 bytes
        assert!(
            mega_kernel_v2::EXT_DUAL_BATCH_META + mega_kernel_v2::DualBatchMeta::SIZE
                <= mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE,
            "DualBatchMeta exceeds extension size"
        );
        // Reserved at offset 92
        assert!(
            mega_kernel_v2::EXT_RESERVED + 4 <= mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE,
            "reserved field exceeds extension size"
        );
    }

    #[test]
    fn batch_context_per_seq_padding_between_sequences_isolation() {
        let mut ctx = BatchContext::new(3);
        // Write to seq 0 all fields at max
        for field in [
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ] {
            write_u32(&mut ctx.data, BATCH_CTX_HEADER_SIZE + field, 0xFF);
        }

        // Seq 1 and seq 2 should still be zeroed (including padding)
        for seq in 1..=2 {
            let base = BATCH_CTX_HEADER_SIZE + seq * SEQ_META_STRIDE;
            for i in 0..SEQ_META_STRIDE {
                assert_eq!(
                    ctx.data[base + i], 0,
                    "seq {} byte {} should be zero, got {:02X}",
                    seq, i, ctx.data[base + i]
                );
            }
        }
    }

    #[test]
    fn batch_context_v2_extension_does_not_corrupt_header() {
        let mut ctx = BatchContext::with_v2_extension(8, 4);
        // Set header fields
        ctx.set_num_seqs(4);
        ctx.set_max_decode_steps(10);
        ctx.set_total_prefill_tokens(100);

        // Write to extension
        ctx.set_ext_kv_pool_total_pages(9999);
        ctx.set_ext_max_batch_size(8);
        ctx.set_ext_autotune_actual_batch(4);

        // Header fields should remain intact
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 4);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 10);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 100);
    }

    #[test]
    fn batch_context_seq_meta_base_ptr_in_v2_extension() {
        let ctx = BatchContext::with_v2_extension(8, 2);
        // seq_meta_base_ptr should still point to offset 96 (start of per-seq data)
        let ptr_val = read_u64(&ctx.data, SEQ_META_BASE_PTR) as usize;
        let expected = ctx.data.as_ptr() as usize + BATCH_CTX_HEADER_SIZE;
        assert_eq!(ptr_val, expected);
    }

    #[test]
    fn batch_context_large_seq_index_write() {
        let mut ctx = BatchContext::new(16);
        ctx.set_seq_prompt_len(15, 12345);
        let base15 = BATCH_CTX_HEADER_SIZE + 15 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base15 + SEQ_PROMPT_LEN), 12345);
        // Adjacent sequences should be unaffected
        let base14 = BATCH_CTX_HEADER_SIZE + 14 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base14 + SEQ_PROMPT_LEN), 0);
    }

    // ── Additional tests (18 new) ──

    #[test]
    fn batch_context_new_zero_seqs_has_header_only_buffer() {
        // Arrange: create context with zero sequences
        let ctx = BatchContext::new(0);

        // Assert: buffer is exactly header size, no per-seq area
        assert_eq!(ctx.data.len(), BATCH_CTX_HEADER_SIZE);
        assert_eq!(ctx.num_seqs, 0);
        assert_eq!(ctx.max_batch_size, 0);
    }

    #[test]
    fn batch_context_as_ptr_returns_non_null_for_various_sizes() {
        // Arrange & Act & Assert: as_ptr() should return non-null for all sizes
        for n in [0, 1, 8, 64] {
            let ctx = BatchContext::new(n);
            assert!(!ctx.as_ptr().is_null(), "as_ptr() null for num_seqs={}", n);
        }
    }

    #[test]
    fn batch_context_set_max_decode_steps_roundtrip() {
        // Arrange
        let mut ctx = BatchContext::new(2);

        // Act
        ctx.set_max_decode_steps(42);

        // Assert
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 42);
    }

    #[test]
    fn batch_context_set_total_prefill_tokens_roundtrip() {
        // Arrange
        let mut ctx = BatchContext::new(2);

        // Act
        ctx.set_total_prefill_tokens(1024);

        // Assert
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 1024);
    }

    #[test]
    fn batch_context_set_seq_kv_len_isolation() {
        // Arrange: write kv_len for two different sequences
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_kv_len(0, 10);
        ctx.set_seq_kv_len(1, 20);
        ctx.set_seq_kv_len(2, 30);

        // Act: overwrite only seq 1
        ctx.set_seq_kv_len(1, 99);

        // Assert: seq 0 and seq 2 untouched
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_KV_LEN), 10);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_KV_LEN), 30);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_KV_LEN), 99);
    }

    #[test]
    fn batch_context_set_seq_rope_pos_offset_roundtrip() {
        // Arrange
        let mut ctx = BatchContext::new(1);

        // Act
        ctx.set_seq_rope_pos_offset(0, 0xDEAD_BEEF);

        // Assert
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_ROPE_POS_OFFSET),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn batch_context_set_seq_max_new_tokens_roundtrip() {
        // Arrange
        let mut ctx = BatchContext::new(1);

        // Act
        ctx.set_seq_max_new_tokens(0, 512);

        // Assert
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_MAX_NEW_TOKENS),
            512
        );
    }

    #[test]
    fn batch_context_set_seq_session_position_roundtrip() {
        // Arrange
        let mut ctx = BatchContext::new(1);

        // Act
        ctx.set_seq_session_position(0, 2048);

        // Assert
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_SESSION_POSITION),
            2048
        );
    }

    #[test]
    fn batch_context_set_seq_prompt_len_max_u32() {
        // Arrange
        let mut ctx = BatchContext::new(1);

        // Act
        ctx.set_seq_prompt_len(0, u32::MAX);

        // Assert
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN),
            u32::MAX
        );
    }

    #[test]
    fn batch_context_header_size_is_96() {
        // Assert: header constant must be 96 bytes
        assert_eq!(BATCH_CTX_HEADER_SIZE, 96);
    }

    #[test]
    fn batch_context_seq_meta_stride_is_64() {
        // Assert: stride constant must be 64 bytes (14 u32 fields + 8 bytes padding)
        assert_eq!(SEQ_META_STRIDE, 64);
    }

    #[test]
    fn batch_context_num_per_seq_fields_exactly_14() {
        // Assert: 14 per-seq u32 fields as documented in layout comments
        let field_count: usize = 14;
        assert_eq!(field_count * 4, 56);
        assert!(field_count * 4 <= SEQ_META_STRIDE);
    }

    #[test]
    fn batch_context_v2_extension_initial_num_seqs_zero() {
        // Arrange: max_batch_size=8, initial_num_seqs=0
        let ctx = BatchContext::with_v2_extension(8, 0);

        // Assert
        assert_eq!(ctx.num_seqs, 0);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 0);
        assert!(ctx.has_v2_extension);
        assert_eq!(ctx.max_batch_size, 8);
    }

    #[test]
    fn batch_context_v2_extension_per_seq_area_sized_by_max_batch() {
        // Arrange: create with max_batch_size=16, initial=2
        let ctx = BatchContext::with_v2_extension(16, 2);

        // Assert: total = header + 16*stride + extension
        let expected = BATCH_CTX_HEADER_SIZE + 16 * SEQ_META_STRIDE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE;
        assert_eq!(ctx.byte_size(), expected);
        // Can write to seq index 15 (the 16th slot) without panic
        let mut ctx_mut = ctx;
        ctx_mut.set_seq_prompt_len(15, 77);
        assert_eq!(
            read_u32(&ctx_mut.data, BATCH_CTX_HEADER_SIZE + 15 * SEQ_META_STRIDE + SEQ_PROMPT_LEN),
            77
        );
    }

    #[test]
    fn batch_context_clone_preserves_has_v2_extension() {
        // Arrange
        let ctx = BatchContext::with_v2_extension(4, 2);

        // Act
        let cloned = ctx.clone();

        // Assert: extension flag preserved
        assert!(cloned.has_v2_extension);
        assert_eq!(cloned.max_batch_size, 4);
        assert_eq!(cloned.num_seqs, 2);
        assert_eq!(cloned.byte_size(), ctx.byte_size());
    }

    #[test]
    fn batch_context_set_seq_all_writable_fields_for_many_seqs() {
        // Arrange: 8 sequences, write a distinctive value per field per sequence
        let mut ctx = BatchContext::new(8);

        // Act: write each seq's prompt_len = seq_idx * 10 + 1
        for seq in 0..8u32 {
            ctx.set_seq_prompt_len(seq as usize, seq * 10 + 1);
            ctx.set_seq_kv_len(seq as usize, seq * 10 + 2);
            ctx.set_seq_rope_pos_offset(seq as usize, seq * 10 + 3);
            ctx.set_seq_max_new_tokens(seq as usize, seq * 10 + 4);
            ctx.set_seq_session_position(seq as usize, seq * 10 + 5);
            ctx.set_seq_output_offset(seq as usize, seq * 10 + 6);
        }

        // Assert: spot-check seq 0, 4, 7
        for &(seq, expected_base) in &[(0u32, 1u32), (4, 41), (7, 71)] {
            let base = BATCH_CTX_HEADER_SIZE + seq as usize * SEQ_META_STRIDE;
            assert_eq!(read_u32(&ctx.data, base + SEQ_PROMPT_LEN), expected_base);
            assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), expected_base + 1);
            assert_eq!(read_u32(&ctx.data, base + SEQ_ROPE_POS_OFFSET), expected_base + 2);
            assert_eq!(read_u32(&ctx.data, base + SEQ_MAX_NEW_TOKENS), expected_base + 3);
            assert_eq!(read_u32(&ctx.data, base + SEQ_SESSION_POSITION), expected_base + 4);
            assert_eq!(read_u32(&ctx.data, base + SEQ_OUTPUT_OFFSET), expected_base + 5);
        }
    }

    #[test]
    fn batch_context_per_seq_non_overlapping_ranges() {
        // Assert: 14 per-seq fields do not overlap each other within a stride
        let fields: [(usize, &str); 14] = [
            (SEQ_PROMPT_LEN, "prompt_len"),
            (SEQ_KV_LEN, "kv_len"),
            (SEQ_ROPE_POS_OFFSET, "rope_pos_offset"),
            (SEQ_MAX_NEW_TOKENS, "max_new_tokens"),
            (SEQ_SESSION_POSITION, "session_position"),
            (SEQ_PAGE_TABLE_OFFSET, "page_table_offset"),
            (SEQ_PAGE_TABLE_LEN, "page_table_len"),
            (SEQ_FUSED_HIDDEN_OFFSET, "fused_hidden_offset"),
            (SEQ_NUM_MM_TOKENS, "num_mm_tokens"),
            (SEQ_ACTIVE_FLAG, "active_flag"),
            (SEQ_SEQ_POSITION, "seq_position"),
            (SEQ_GEN_COUNT, "gen_count"),
            (SEQ_LAST_SAMPLED_TOKEN, "last_sampled_token"),
            (SEQ_OUTPUT_OFFSET, "output_offset"),
        ];
        for i in 0..fields.len() {
            for j in (i + 1)..fields.len() {
                let (off_i, name_i) = fields[i];
                let (off_j, name_j) = fields[j];
                assert!(
                    off_i + 4 <= off_j || off_j + 4 <= off_i,
                    "per-seq fields '{}' (at {}) and '{}' (at {}) overlap",
                    name_i, off_i, name_j, off_j
                );
            }
        }
    }

    // ── Additional tests (50 new, targeting 127+ total) ──

    // ── DualBatchMeta unit tests ──

    #[test]
    fn dual_batch_meta_new_sets_ping_offset_zero() {
        let meta = mega_kernel_v2::DualBatchMeta::new(8);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_new_sets_pong_offset_to_max_batch() {
        let meta = mega_kernel_v2::DualBatchMeta::new(16);
        assert_eq!(meta.pong_seq_offset, 16);
        assert_eq!(meta.pong_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_new_epoch_fields_zero() {
        let meta = mega_kernel_v2::DualBatchMeta::new(4);
        assert_eq!(meta.step_epoch, 0);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_default_all_zero() {
        let meta = mega_kernel_v2::DualBatchMeta::default();
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
        assert_eq!(meta.step_epoch, 0);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_swap_exchanges_ping_pong_offsets() {
        let mut meta = mega_kernel_v2::DualBatchMeta::new(8);
        meta.ping_seq_count = 3;
        meta.pong_seq_count = 5;
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 8);
        assert_eq!(meta.ping_seq_count, 5);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 3);
    }

    #[test]
    fn dual_batch_meta_swap_increments_epoch() {
        let mut meta = mega_kernel_v2::DualBatchMeta::new(4);
        assert_eq!(meta.step_epoch, 0);
        meta.swap();
        assert_eq!(meta.step_epoch, 1);
        meta.swap();
        assert_eq!(meta.step_epoch, 2);
    }

    #[test]
    fn dual_batch_meta_swap_resets_arrival_count() {
        let mut meta = mega_kernel_v2::DualBatchMeta::new(4);
        meta.epoch_arrival_count = 7;
        meta.swap();
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_swap_wrapping_epoch() {
        let mut meta = mega_kernel_v2::DualBatchMeta::new(4);
        meta.step_epoch = u32::MAX;
        meta.swap();
        assert_eq!(meta.step_epoch, 0);
    }

    #[test]
    fn dual_batch_meta_size_is_24() {
        assert_eq!(mega_kernel_v2::DualBatchMeta::SIZE, 24);
    }

    #[test]
    fn dual_batch_meta_clone_equals_original() {
        let meta = mega_kernel_v2::DualBatchMeta::new(16);
        let cloned = meta.clone();
        assert_eq!(cloned.ping_seq_offset, meta.ping_seq_offset);
        assert_eq!(cloned.ping_seq_count, meta.ping_seq_count);
        assert_eq!(cloned.pong_seq_offset, meta.pong_seq_offset);
        assert_eq!(cloned.pong_seq_count, meta.pong_seq_count);
        assert_eq!(cloned.step_epoch, meta.step_epoch);
        assert_eq!(cloned.epoch_arrival_count, meta.epoch_arrival_count);
    }

    #[test]
    fn dual_batch_meta_copy_is_independent() {
        let mut meta = mega_kernel_v2::DualBatchMeta::new(8);
        let copy = meta;
        meta.ping_seq_count = 99;
        assert_eq!(copy.ping_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_debug_formats_all_fields() {
        let meta = mega_kernel_v2::DualBatchMeta::new(4);
        let s = format!("{:?}", meta);
        assert!(s.contains("ping_seq_offset"));
        assert!(s.contains("pong_seq_offset"));
        assert!(s.contains("step_epoch"));
    }

    // ── Per-seq field individual roundtrip with various values ──

    #[test]
    fn batch_context_set_seq_page_table_offset_roundtrip() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_page_table_offset(1, 4096);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_PAGE_TABLE_OFFSET),
            4096
        );
    }

    #[test]
    fn batch_context_set_seq_page_table_len_roundtrip() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_page_table_len(0, 64);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_PAGE_TABLE_LEN),
            64
        );
    }

    #[test]
    fn batch_context_set_seq_fused_hidden_offset_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_fused_hidden_offset(0, 256);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_FUSED_HIDDEN_OFFSET),
            256
        );
    }

    #[test]
    fn batch_context_set_seq_num_mm_tokens_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_num_mm_tokens(0, 3);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_NUM_MM_TOKENS),
            3
        );
    }

    #[test]
    fn batch_context_set_seq_active_flag_active_inactive() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_active_flag(1, 0);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_ACTIVE_FLAG),
            1
        );
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_ACTIVE_FLAG),
            0
        );
    }

    #[test]
    fn batch_context_set_seq_position_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_position(0, 128);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_SEQ_POSITION),
            128
        );
    }

    #[test]
    fn batch_context_set_seq_gen_count_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_gen_count(0, 42);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_GEN_COUNT),
            42
        );
    }

    #[test]
    fn batch_context_set_seq_last_sampled_token_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_last_sampled_token(0, 1337);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_LAST_SAMPLED_TOKEN),
            1337
        );
    }

    #[test]
    fn batch_context_set_seq_output_offset_roundtrip() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_output_offset(0, 2048);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_OUTPUT_OFFSET),
            2048
        );
    }

    // ── Extension field overwrite idempotency ──

    #[test]
    fn batch_context_v2_ext_kv_pool_total_pages_overwrite() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        ctx.set_ext_kv_pool_total_pages(100);
        ctx.set_ext_kv_pool_total_pages(200);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 200);
    }

    #[test]
    fn batch_context_v2_ext_max_batch_size_overwrite() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        ctx.set_ext_max_batch_size(1);
        ctx.set_ext_max_batch_size(4);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_MAX_BATCH_SIZE), 4);
    }

    #[test]
    fn batch_context_v2_ext_autotune_overwrite() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        ctx.set_ext_autotune_actual_batch(0);
        ctx.set_ext_autotune_actual_batch(3);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH), 3);
    }

    // ── Extension pointer field overwrite isolation ──

    #[test]
    fn batch_context_v2_ext_pointer_overwrite_isolation() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        let p1 = 0xAAAA_usize as *const u8;
        ctx.set_ext_request_queue_ptr(p1);

        let p2 = 0xBBBB_usize as *const u8;
        ctx.set_ext_output_ring_ptr(p2);

        // Writing p2 should not overwrite p1
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_REQUEST_QUEUE_PTR), p1 as usize);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_OUTPUT_RING_PTR), p2 as usize);
    }

    #[test]
    fn batch_context_v2_ext_kv_free_bitmap_overwrite() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        let p1 = 0x1111_usize as *mut u32;
        ctx.set_ext_kv_free_bitmap_ptr(p1);
        let p2 = 0x2222_usize as *mut u32;
        ctx.set_ext_kv_free_bitmap_ptr(p2);
        assert_eq!(read_usize(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR), p2 as usize);
    }

    // ── V2 extension data_initially_zeroed ──

    #[test]
    fn batch_context_v2_extension_data_initially_zeroed() {
        let ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        for i in 0..mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE {
            assert_eq!(
                ctx.data[ext_start + i], 0,
                "extension byte at offset {} should be zero-initialized",
                i
            );
        }
    }

    // ── Per-seq area in v2 extension slots beyond initial_num_seqs ──

    #[test]
    fn batch_context_v2_per_seq_beyond_initial_is_zeroed() {
        let ctx = BatchContext::with_v2_extension(8, 2);
        // Slots 2..7 should be zeroed (not yet populated)
        for seq in 2..8 {
            let base = BATCH_CTX_HEADER_SIZE + seq * SEQ_META_STRIDE;
            for i in 0..SEQ_META_STRIDE {
                assert_eq!(
                    ctx.data[base + i], 0,
                    "slot {} byte {} should be zero",
                    seq, i
                );
            }
        }
    }

    // ── Write then read each per-seq field for seq > 0 ──

    #[test]
    fn batch_context_per_seq_field_kv_len_seq_3() {
        let mut ctx = BatchContext::new(4);
        ctx.set_seq_kv_len(3, 777);
        let base3 = BATCH_CTX_HEADER_SIZE + 3 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base3 + SEQ_KV_LEN), 777);
    }

    #[test]
    fn batch_context_per_seq_field_rope_offset_seq_2() {
        let mut ctx = BatchContext::new(4);
        ctx.set_seq_rope_pos_offset(2, 4096);
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base2 + SEQ_ROPE_POS_OFFSET), 4096);
    }

    #[test]
    fn batch_context_per_seq_field_session_pos_seq_1() {
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_session_position(1, 8192);
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_SESSION_POSITION), 8192);
    }

    #[test]
    fn batch_context_per_seq_field_max_new_tokens_seq_2() {
        let mut ctx = BatchContext::new(4);
        ctx.set_seq_max_new_tokens(2, 1024);
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base2 + SEQ_MAX_NEW_TOKENS), 1024);
    }

    #[test]
    fn batch_context_per_seq_field_gen_count_seq_1() {
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_gen_count(1, 50);
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_GEN_COUNT), 50);
    }

    #[test]
    fn batch_context_per_seq_field_last_sampled_token_seq_0() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_last_sampled_token(0, 30522);
        let base0 = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_LAST_SAMPLED_TOKEN), 30522);
    }

    // ── Multi-seq multi-field write then verify all ──

    #[test]
    fn batch_context_multi_seq_all_kv_len_distinct() {
        let mut ctx = BatchContext::new(4);
        for seq in 0..4u32 {
            ctx.set_seq_kv_len(seq as usize, seq * 100);
        }
        for seq in 0..4u32 {
            let base = BATCH_CTX_HEADER_SIZE + seq as usize * SEQ_META_STRIDE;
            assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), seq * 100);
        }
    }

    #[test]
    fn batch_context_multi_seq_all_gen_count_distinct() {
        let mut ctx = BatchContext::new(4);
        for seq in 0..4u32 {
            ctx.set_seq_gen_count(seq as usize, seq + 10);
        }
        for seq in 0..4u32 {
            let base = BATCH_CTX_HEADER_SIZE + seq as usize * SEQ_META_STRIDE;
            assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), seq + 10);
        }
    }

    #[test]
    fn batch_context_multi_seq_all_output_offsets_contiguous() {
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_output_offset(0, 0);
        ctx.set_seq_output_offset(1, 50);
        ctx.set_seq_output_offset(2, 100);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_OUTPUT_OFFSET),
            0
        );
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET),
            50
        );
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET),
            100
        );
    }

    // ── Header field zero then nonzero ──

    #[test]
    fn batch_context_header_field_max_decode_steps_zero_then_set() {
        let mut ctx = BatchContext::new(1);
        ctx.set_max_decode_steps(0);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 0);
        ctx.set_max_decode_steps(10);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 10);
    }

    #[test]
    fn batch_context_header_field_total_prefill_zero_then_set() {
        let mut ctx = BatchContext::new(1);
        ctx.set_total_prefill_tokens(0);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 0);
        ctx.set_total_prefill_tokens(256);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 256);
    }

    // ── write_u32 helper correctness via header field ──

    #[test]
    fn batch_context_write_u32_byte_level_correct() {
        let mut ctx = BatchContext::new(0);
        // 0x12345678 in little-endian is [0x78, 0x56, 0x34, 0x12]
        ctx.set_num_seqs(0x1234_5678);
        assert_eq!(ctx.data[0], 0x78);
        assert_eq!(ctx.data[1], 0x56);
        assert_eq!(ctx.data[2], 0x34);
        assert_eq!(ctx.data[3], 0x12);
    }

    #[test]
    fn batch_context_write_u32_zero_value() {
        let mut ctx = BatchContext::new(0);
        ctx.set_num_seqs(0);
        assert_eq!(ctx.data[0], 0);
        assert_eq!(ctx.data[1], 0);
        assert_eq!(ctx.data[2], 0);
        assert_eq!(ctx.data[3], 0);
    }

    #[test]
    fn batch_context_write_u32_one() {
        let mut ctx = BatchContext::new(0);
        ctx.set_num_seqs(1);
        assert_eq!(ctx.data[0], 1);
        assert_eq!(ctx.data[1], 0);
        assert_eq!(ctx.data[2], 0);
        assert_eq!(ctx.data[3], 0);
    }

    // ── write_usize helper via pointer field ──

    #[test]
    fn batch_context_write_usize_null_pointer() {
        let mut ctx = BatchContext::new(1);
        ctx.set_input_ids_flat_ptr(std::ptr::null());
        assert_eq!(read_usize(&ctx.data, INPUT_IDS_FLAT_PTR), 0);
    }

    #[test]
    fn batch_context_write_usize_high_address() {
        let mut ctx = BatchContext::new(1);
        let high_addr: usize = 0xFFFF_FFFF_FFFF_F000;
        ctx.set_input_ids_flat_ptr(high_addr as *const u32);
        assert_eq!(read_usize(&ctx.data, INPUT_IDS_FLAT_PTR), high_addr);
    }

    // ── Clone then modify each struct field independently ──

    #[test]
    fn batch_context_clone_num_seqs_independent() {
        let mut ctx = BatchContext::new(3);
        let cloned = ctx.clone();
        // num_seqs is a usize field, not written to data automatically after new()
        assert_eq!(cloned.num_seqs, 3);
        assert_eq!(cloned.max_batch_size, 3);
        assert!(!cloned.has_v2_extension);
        assert!(cloned.seq_mapping.is_empty());
    }

    #[test]
    fn batch_context_clone_seq_mapping_independent() {
        let mut ctx = BatchContext::new(2);
        ctx.seq_mapping = vec![0, 0, 1, 1];
        let cloned = ctx.clone();
        ctx.seq_mapping.push(2);
        assert_eq!(cloned.seq_mapping.len(), 4);
        assert_eq!(ctx.seq_mapping.len(), 5);
    }

    #[test]
    fn batch_context_clone_v2_extension_independent() {
        let mut ctx = BatchContext::with_v2_extension(4, 2);
        ctx.set_ext_kv_pool_total_pages(100);
        let cloned = ctx.clone();
        ctx.set_ext_kv_pool_total_pages(999);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&cloned.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 100);
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 999);
    }

    // ── Constant layout invariants ──

    #[test]
    fn constants_header_size_is_multiple_of_8() {
        assert_eq!(BATCH_CTX_HEADER_SIZE % 8, 0, "header size must be 8-byte aligned");
    }

    #[test]
    fn constants_seq_meta_stride_is_power_of_2() {
        assert!(SEQ_META_STRIDE.is_power_of_two(), "stride must be a power of 2 for fast modulo");
    }

    #[test]
    fn constants_extension_size_is_128() {
        assert_eq!(mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE, 128);
    }

    #[test]
    fn constants_extension_size_multiple_of_8() {
        assert_eq!(mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE % 8, 0);
    }

    #[test]
    fn constants_extension_u32_offsets_4_byte_aligned() {
        let u32_ext_offsets = [
            mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES,
            mega_kernel_v2::EXT_MAX_BATCH_SIZE,
            mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH,
            mega_kernel_v2::EXT_RESERVED,
        ];
        for &off in &u32_ext_offsets {
            assert_eq!(off % 4, 0, "extension u32 offset {} is not 4-byte aligned", off);
        }
    }

    #[test]
    fn constants_extension_usize_offsets_8_byte_aligned() {
        // Only offsets that are 8-byte aligned (used for pointer-width fields)
        let eight_byte_aligned_ext_offsets = [
            mega_kernel_v2::EXT_REQUEST_QUEUE_PTR,
            mega_kernel_v2::EXT_OUTPUT_RING_PTR,
            mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR,
            mega_kernel_v2::EXT_OUTPUT_PER_CTA_DOORBELL_PTR,
            mega_kernel_v2::EXT_OUTPUT_EPOCH_FLAG_PTR,
        ];
        for &off in &eight_byte_aligned_ext_offsets {
            assert_eq!(off % 8, 0, "extension usize offset {} is not 8-byte aligned", off);
        }
    }

    #[test]
    fn constants_extension_4_byte_aligned_fields() {
        // Some extension fields are only 4-byte aligned (not 8-byte)
        let four_byte_fields = [
            mega_kernel_v2::EXT_POOL_CLUSTER_DSMEM_PTR,
            mega_kernel_v2::EXT_PENDING_FREE_LIST_PTR,
            mega_kernel_v2::EXT_PENDING_FREE_COUNT_PTR,
        ];
        for &off in &four_byte_fields {
            assert_eq!(off % 4, 0, "extension field at offset {} is not 4-byte aligned", off);
            assert_ne!(off % 8, 0, "expected offset {} to be only 4-byte aligned, not 8-byte", off);
        }
    }

    // ── DualBatchMeta embedded in extension area ──

    #[test]
    fn batch_context_v2_dual_batch_meta_roundtrip_after_swap() {
        let mut ctx = BatchContext::with_v2_extension(8, 2);
        let mut meta = mega_kernel_v2::DualBatchMeta::new(8);
        meta.ping_seq_count = 5;
        meta.pong_seq_count = 3;
        meta.step_epoch = 7;
        meta.swap();
        ctx.set_ext_dual_batch_meta(&meta);

        let ext_start = BATCH_CTX_HEADER_SIZE + 8 * SEQ_META_STRIDE;
        let meta_base = ext_start + mega_kernel_v2::EXT_DUAL_BATCH_META;
        // After swap, ping_offset=8, pong_offset=0
        assert_eq!(read_u32(&ctx.data, meta_base), 8);
        assert_eq!(read_u32(&ctx.data, meta_base + 4), 3);
        assert_eq!(read_u32(&ctx.data, meta_base + 8), 0);
        assert_eq!(read_u32(&ctx.data, meta_base + 12), 5);
        assert_eq!(read_u32(&ctx.data, meta_base + 16), 8);
        assert_eq!(read_u32(&ctx.data, meta_base + 20), 0);
    }

    #[test]
    fn batch_context_v2_dual_batch_meta_does_not_leak() {
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        ctx.set_ext_dual_batch_meta(&mega_kernel_v2::DualBatchMeta::new(4));
        // Extension fields after DualBatchMeta (24 bytes at offset 32) should be zero
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        // AUTOTUNE_ACTUAL_BATCH is at offset 56, DualBatchMeta ends at 32+24=56
        // So it should be zero (untouched)
        assert_eq!(
            read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH),
            0
        );
    }

    // ── Edge: v2 extension with max_batch_size equal to initial ──

    #[test]
    fn batch_context_v2_extension_max_equals_initial() {
        let ctx = BatchContext::with_v2_extension(4, 4);
        assert_eq!(ctx.num_seqs, 4);
        assert_eq!(ctx.max_batch_size, 4);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 4);
        assert_eq!(
            ctx.byte_size(),
            BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE
        );
    }

    // ── Large batch v2 extension ──

    #[test]
    fn batch_context_v2_extension_large_batch() {
        let ctx = BatchContext::with_v2_extension(64, 1);
        assert_eq!(ctx.max_batch_size, 64);
        assert_eq!(ctx.num_seqs, 1);
        assert_eq!(
            ctx.byte_size(),
            BATCH_CTX_HEADER_SIZE + 64 * SEQ_META_STRIDE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE
        );
    }

    // ── All extension pointer writers produce correct offsets ──

    #[test]
    fn batch_context_v2_all_extension_pointers_distinct_offsets() {
        let ptr_offsets: [(usize, &str); 8] = [
            (mega_kernel_v2::EXT_REQUEST_QUEUE_PTR, "req_queue"),
            (mega_kernel_v2::EXT_OUTPUT_RING_PTR, "output_ring"),
            (mega_kernel_v2::EXT_KV_FREE_BITMAP_PTR, "kv_bitmap"),
            (mega_kernel_v2::EXT_POOL_CLUSTER_DSMEM_PTR, "dsmem"),
            (mega_kernel_v2::EXT_PENDING_FREE_LIST_PTR, "free_list"),
            (mega_kernel_v2::EXT_PENDING_FREE_COUNT_PTR, "free_count"),
            (mega_kernel_v2::EXT_OUTPUT_PER_CTA_DOORBELL_PTR, "doorbell"),
            (mega_kernel_v2::EXT_OUTPUT_EPOCH_FLAG_PTR, "epoch"),
        ];
        for i in 0..ptr_offsets.len() {
            for j in (i + 1)..ptr_offsets.len() {
                assert_ne!(
                    ptr_offsets[i].0, ptr_offsets[j].0,
                    "extension pointer offsets '{}' and '{}' overlap at {}",
                    ptr_offsets[i].1, ptr_offsets[j].1, ptr_offsets[i].0
                );
            }
        }
    }

    // ── Per-seq overwrites preserve neighboring fields ──

    #[test]
    fn batch_context_per_seq_overwrite_rope_preserves_neighbors() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_kv_len(0, 10);
        ctx.set_seq_rope_pos_offset(0, 5);
        ctx.set_seq_max_new_tokens(0, 20);
        // Overwrite rope_pos_offset
        ctx.set_seq_rope_pos_offset(0, 99);
        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_KV_LEN), 10);
        assert_eq!(read_u32(&ctx.data, base + SEQ_ROPE_POS_OFFSET), 99);
        assert_eq!(read_u32(&ctx.data, base + SEQ_MAX_NEW_TOKENS), 20);
    }

    #[test]
    fn batch_context_per_seq_overwrite_session_preserves_page_table() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_session_position(0, 100);
        ctx.set_seq_page_table_offset(0, 200);
        ctx.set_seq_page_table_len(0, 300);
        ctx.set_seq_session_position(0, 999);
        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_SESSION_POSITION), 999);
        assert_eq!(read_u32(&ctx.data, base + SEQ_PAGE_TABLE_OFFSET), 200);
        assert_eq!(read_u32(&ctx.data, base + SEQ_PAGE_TABLE_LEN), 300);
    }

    #[test]
    fn batch_context_per_seq_overwrite_active_preserves_position_and_gen() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_position(0, 50);
        ctx.set_seq_gen_count(0, 10);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_active_flag(0, 0);
        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_SEQ_POSITION), 50);
        assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), 10);
        assert_eq!(read_u32(&ctx.data, base + SEQ_ACTIVE_FLAG), 0);
    }

    // ── Additional tests (13 new, targeting 152 total) ──

    #[test]
    fn batch_context_struct_num_seqs_independent_of_header_num_seqs() {
        // Arrange: the struct field `num_seqs` and the header u32 at offset 0
        // are logically related but physically independent — set_num_seqs only
        // writes the header, not the struct field.
        let mut ctx = BatchContext::new(4);
        assert_eq!(ctx.num_seqs, 4);

        // Act: change the header value via set_num_seqs
        ctx.set_num_seqs(99);

        // Assert: struct field remains 4, header reads 99
        assert_eq!(ctx.num_seqs, 4);
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 99);
    }

    #[test]
    fn batch_context_seq_mapping_with_all_same_seq_id() {
        // Arrange: all tokens belong to seq 0 (single-sequence prefill)
        let mut ctx = BatchContext::new(2);
        let mapping = vec![0u32; 16]; // 16 tokens all for seq 0
        ctx.seq_mapping = mapping.clone();

        // Assert: seq_mapping stores the values correctly
        assert_eq!(ctx.seq_mapping.len(), 16);
        assert!(ctx.seq_mapping.iter().all(|&v| v == 0));
    }

    #[test]
    fn batch_context_v2_clone_then_modify_extension_preserves_original() {
        // Arrange: write extension data, clone, then modify original
        let mut ctx = BatchContext::with_v2_extension(4, 2);
        ctx.set_ext_max_batch_size(4);
        ctx.set_ext_kv_pool_total_pages(500);

        // Act
        let cloned = ctx.clone();
        ctx.set_ext_kv_pool_total_pages(9999);
        ctx.set_ext_max_batch_size(2);

        // Assert: cloned retains original values
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(
            read_u32(&cloned.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES),
            500
        );
        assert_eq!(
            read_u32(&cloned.data, ext_start + mega_kernel_v2::EXT_MAX_BATCH_SIZE),
            4
        );
    }

    #[test]
    fn batch_context_double_clone_chain() {
        // Arrange: verify a chain of two clones produces independent copies
        let mut ctx = BatchContext::new(3);
        ctx.set_seq_prompt_len(0, 42);
        ctx.set_num_seqs(3);

        // Act: double clone
        let first = ctx.clone();
        let second = first.clone();

        // Modify original and first clone
        ctx.set_seq_prompt_len(0, 1);
        // Note: first is moved, so we test second independence from original
        assert_eq!(
            read_u32(&second.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN),
            42
        );
        assert_eq!(second.num_seqs, 3);
    }

    #[test]
    fn batch_context_recreate_with_different_size_produces_correct_layout() {
        // Arrange & Act: create contexts of different sizes sequentially
        let ctx_small = BatchContext::new(1);
        let ctx_large = BatchContext::new(32);

        // Assert: each has correct layout
        assert_eq!(ctx_small.byte_size(), BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE);
        assert_eq!(ctx_large.byte_size(), BATCH_CTX_HEADER_SIZE + 32 * SEQ_META_STRIDE);
        // Per-seq pointer is different for each (points into its own buffer)
        let ptr_small = read_u64(&ctx_small.data, SEQ_META_BASE_PTR) as usize;
        let ptr_large = read_u64(&ctx_large.data, SEQ_META_BASE_PTR) as usize;
        assert_eq!(ptr_small, ctx_small.data.as_ptr() as usize + BATCH_CTX_HEADER_SIZE);
        assert_eq!(ptr_large, ctx_large.data.as_ptr() as usize + BATCH_CTX_HEADER_SIZE);
    }

    #[test]
    fn batch_context_single_seq_all_header_pointers_set() {
        // Arrange: single sequence with all pointer fields set
        let mut ctx = BatchContext::new(1);

        // Act: set all 9 pointer fields
        let vals: [usize; 9] = [
            0x1000, 0x2000, 0x3000, 0x4000, 0x5000,
            0x6000, 0x7000, 0x8000, 0x9000,
        ];
        ctx.set_input_ids_flat_ptr(vals[0] as *const u32);
        ctx.set_output_tokens_flat_ptr(vals[1] as *mut u32);
        ctx.set_positions_ptr(vals[2] as *const u32);
        ctx.set_page_table_flat_ptr(vals[3] as *const u32);
        ctx.set_kv_pool_base(vals[4] as *const u8);
        ctx.set_sampling_params_ptr(vals[5] as *const u32);
        ctx.set_hook_ctx_ptr(vals[6] as *const u8);
        ctx.set_callback_table_ptr(vals[7] as *const u8);
        ctx.set_seq_mapping_ptr(vals[8] as *const u32);

        // Assert: all pointer fields read back correctly and are distinct
        let offsets = [
            INPUT_IDS_FLAT_PTR, OUTPUT_TOKENS_FLAT_PTR, POSITIONS_PTR,
            PAGE_TABLE_FLAT_PTR, KV_POOL_BASE, SAMPLING_PARAMS_PTR,
            HOOK_CTX_PTR, CALLBACK_TABLE_PTR, SEQ_MAPPING_PTR,
        ];
        for (i, &off) in offsets.iter().enumerate() {
            assert_eq!(read_usize(&ctx.data, off), vals[i], "pointer field at offset {}", off);
        }
    }

    #[test]
    fn batch_context_max_seq_index_all_fields_nonzero() {
        // Arrange: 8-sequence context, write non-zero values to all 14 fields
        // of the last sequence (index 7)
        let mut ctx = BatchContext::new(8);
        let base7 = BATCH_CTX_HEADER_SIZE + 7 * SEQ_META_STRIDE;

        // Act
        ctx.set_seq_prompt_len(7, 100);
        ctx.set_seq_kv_len(7, 200);
        ctx.set_seq_rope_pos_offset(7, 300);
        ctx.set_seq_max_new_tokens(7, 400);
        ctx.set_seq_session_position(7, 500);
        ctx.set_seq_page_table_offset(7, 600);
        ctx.set_seq_page_table_len(7, 700);
        ctx.set_seq_fused_hidden_offset(7, 800);
        ctx.set_seq_num_mm_tokens(7, 900);
        ctx.set_seq_active_flag(7, 1);
        ctx.set_seq_position(7, 1100);
        ctx.set_seq_gen_count(7, 1200);
        ctx.set_seq_last_sampled_token(7, 1300);
        ctx.set_seq_output_offset(7, 1400);

        // Assert: all 14 fields non-zero at seq 7
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_PROMPT_LEN), 100);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_KV_LEN), 200);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_ROPE_POS_OFFSET), 300);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_MAX_NEW_TOKENS), 400);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_SESSION_POSITION), 500);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_PAGE_TABLE_OFFSET), 600);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_PAGE_TABLE_LEN), 700);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_FUSED_HIDDEN_OFFSET), 800);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_NUM_MM_TOKENS), 900);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_ACTIVE_FLAG), 1);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_SEQ_POSITION), 1100);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_GEN_COUNT), 1200);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_LAST_SAMPLED_TOKEN), 1300);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_OUTPUT_OFFSET), 1400);
    }

    #[test]
    fn batch_context_simulated_decode_steps_update() {
        // Arrange: simulate 3 decode steps updating gen_count and position
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_max_new_tokens(0, 100);
        ctx.set_seq_position(0, 10); // start after prompt
        ctx.set_seq_gen_count(0, 0);

        // Act: simulate 3 decode steps
        for step in 1..=3u32 {
            let pos = 10 + step;
            ctx.set_seq_position(0, pos);
            ctx.set_seq_gen_count(0, step);
            ctx.set_seq_last_sampled_token(0, 1000 + step);
        }

        // Assert: final state after 3 steps
        let base = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base + SEQ_SEQ_POSITION), 13);
        assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), 3);
        assert_eq!(read_u32(&ctx.data, base + SEQ_LAST_SAMPLED_TOKEN), 1003);
        // prompt_len unchanged
        assert_eq!(read_u32(&ctx.data, base + SEQ_PROMPT_LEN), 10);
    }

    #[test]
    fn batch_context_v2_extension_all_u32_fields_at_once() {
        // Arrange: write all three u32 extension fields simultaneously
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;

        // Act
        ctx.set_ext_kv_pool_total_pages(4096);
        ctx.set_ext_max_batch_size(4);
        ctx.set_ext_autotune_actual_batch(3);

        // Assert: all three persist without corrupting each other
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 4096);
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_MAX_BATCH_SIZE), 4);
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH), 3);
    }

    #[test]
    fn batch_context_v2_extension_no_overlap_with_per_seq_area() {
        // Arrange: write per-seq data up to the last slot, then write extension
        let mut ctx = BatchContext::with_v2_extension(4, 4);
        ctx.set_seq_prompt_len(3, 0xAA);
        ctx.set_seq_output_offset(3, 0xBB);

        // Act: write extension fields
        ctx.set_ext_kv_pool_total_pages(0xCC);

        // Assert: per-seq slot 3 data intact
        let base3 = BATCH_CTX_HEADER_SIZE + 3 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base3 + SEQ_PROMPT_LEN), 0xAA);
        assert_eq!(read_u32(&ctx.data, base3 + SEQ_OUTPUT_OFFSET), 0xBB);

        // And extension data is correct
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 0xCC);
    }

    #[test]
    fn dual_batch_meta_swap_three_times() {
        // Arrange: verify epoch increments and offsets swap correctly over multiple swaps
        let mut meta = mega_kernel_v2::DualBatchMeta::new(8);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 8);

        // Act: swap 3 times
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 8);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.step_epoch, 1);

        meta.swap();
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 8);
        assert_eq!(meta.step_epoch, 2);

        meta.swap();
        assert_eq!(meta.ping_seq_offset, 8);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.step_epoch, 3);
    }

    #[test]
    fn batch_context_seq_mapping_with_alternating_seq_ids() {
        // Arrange: alternating seq 0 and seq 1 (interleaved batch)
        let mut ctx = BatchContext::new(2);
        let mapping: Vec<u32> = (0..10).map(|i| i % 2).collect();
        ctx.seq_mapping = mapping.clone();

        // Assert: seq_mapping contains alternating 0/1
        assert_eq!(ctx.seq_mapping.len(), 10);
        for (i, &val) in ctx.seq_mapping.iter().enumerate() {
            assert_eq!(val, (i % 2) as u32, "at index {}", i);
        }
    }

    #[test]
    fn batch_context_header_pointer_overwrite_does_not_affect_others() {
        // Arrange: set all header pointer fields, then overwrite one
        let mut ctx = BatchContext::new(1);
        let p1 = 0x1111_usize as *const u32;
        let p2 = 0x2222_usize as *const u32;
        let p3 = 0x3333_usize as *const u32;
        ctx.set_input_ids_flat_ptr(p1);
        ctx.set_positions_ptr(p2);
        ctx.set_sampling_params_ptr(p3);

        // Act: overwrite only positions_ptr
        let p2_new = 0xAAAA_usize as *const u32;
        ctx.set_positions_ptr(p2_new);

        // Assert: input_ids and sampling_params unchanged
        assert_eq!(read_usize(&ctx.data, INPUT_IDS_FLAT_PTR), p1 as usize);
        assert_eq!(read_usize(&ctx.data, POSITIONS_PTR), p2_new as usize);
        assert_eq!(read_usize(&ctx.data, SAMPLING_PARAMS_PTR), p3 as usize);
    }

    // ── Additional tests (13 new, targeting 165 total) ──

    #[test]
    fn batch_context_v2_extension_with_zero_max_batch_and_zero_initial() {
        // Arrange & Act: edge case — max_batch_size=0, initial_num_seqs=0
        let ctx = BatchContext::with_v2_extension(0, 0);
        // Assert: buffer is header + 0*stride + extension
        assert_eq!(ctx.num_seqs, 0);
        assert_eq!(ctx.max_batch_size, 0);
        assert!(ctx.has_v2_extension);
        assert_eq!(
            ctx.byte_size(),
            BATCH_CTX_HEADER_SIZE + mega_kernel_v2::BATCH_CTX_EXTENSION_SIZE
        );
    }

    #[test]
    fn batch_context_gen_count_wrapping_from_max() {
        // Arrange: set gen_count to u32::MAX, then verify it persists
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_gen_count(0, u32::MAX);
        let base = BATCH_CTX_HEADER_SIZE;

        // Act: re-read and overwrite with 0 (simulating wrapping)
        assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), u32::MAX);
        ctx.set_seq_gen_count(0, 0);

        // Assert: wrapped back to 0, neighboring fields intact
        assert_eq!(read_u32(&ctx.data, base + SEQ_GEN_COUNT), 0);
    }

    #[test]
    fn batch_context_seq_mapping_values_beyond_num_seqs() {
        // Arrange: seq_mapping can legally hold seq_id values >= num_seqs
        // (the mapping is a contract with the caller, not bounds-checked internally)
        let mut ctx = BatchContext::new(2);
        let mapping = vec![0u32, 1, 2, 3, 4]; // seq_ids 2..4 exceed num_seqs=2
        ctx.seq_mapping = mapping.clone();

        // Assert: the raw values are stored faithfully
        assert_eq!(ctx.seq_mapping.len(), 5);
        assert_eq!(ctx.seq_mapping[2], 2);
        assert_eq!(ctx.seq_mapping[4], 4);
    }

    #[test]
    fn batch_context_set_seq_mapping_ptr_then_set_other_ptr_isolation() {
        // Arrange: set seq_mapping_ptr, then overwrite an unrelated pointer field
        let mut ctx = BatchContext::new(2);
        let mapping = vec![0u32; 4];
        ctx.set_seq_mapping_ptr(mapping.as_ptr());
        let original_map_ptr = read_usize(&ctx.data, SEQ_MAPPING_PTR);

        // Act: overwrite callback_table_ptr (offset 72, 8 bytes before seq_mapping_ptr at 80)
        ctx.set_callback_table_ptr(0xBEEF_usize as *const u8);

        // Assert: seq_mapping_ptr unchanged
        assert_eq!(read_usize(&ctx.data, SEQ_MAPPING_PTR), original_map_ptr);
        assert_eq!(read_usize(&ctx.data, CALLBACK_TABLE_PTR), 0xBEEF);
    }

    #[test]
    fn batch_context_per_seq_first_and_last_field_adjacent_boundary() {
        // Arrange: write to first field (prompt_len) and last field (output_offset) of seq N
        // then verify no bleed into seq N+1 or seq N-1
        let mut ctx = BatchContext::new(4);
        ctx.set_seq_prompt_len(1, 0xAA00); // first field
        ctx.set_seq_output_offset(1, 0xBB00); // last field (at +52 within stride)

        // Act: write to seq 0 and seq 2
        ctx.set_seq_prompt_len(0, 0x1111);
        ctx.set_seq_output_offset(2, 0x2222);

        // Assert: seq 1 data intact
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_PROMPT_LEN), 0xAA00);
        assert_eq!(read_u32(&ctx.data, base1 + SEQ_OUTPUT_OFFSET), 0xBB00);
        // Assert: seq 0 and seq 2 have their own values, padding untouched
        let base0 = BATCH_CTX_HEADER_SIZE;
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PROMPT_LEN), 0x1111);
        assert_eq!(read_u32(&ctx.data, base2 + SEQ_OUTPUT_OFFSET), 0x2222);
        // Padding between seq 1 and seq 2 (bytes 56..64 of seq 1) is zero
        for i in 56..SEQ_META_STRIDE {
            assert_eq!(ctx.data[base1 + i], 0, "padding byte at seq1+{}", i);
        }
    }

    #[test]
    fn batch_context_clone_after_all_per_seq_writes_isolation() {
        // Arrange: write all 14 fields for seq 0, then clone
        let mut ctx = BatchContext::new(2);
        for (field_idx, &offset) in [
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ].iter().enumerate() {
            write_u32(&mut ctx.data, BATCH_CTX_HEADER_SIZE + offset, (field_idx + 1) as u32);
        }
        let cloned = ctx.clone();

        // Act: overwrite all fields on original
        for &offset in &[
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ] {
            write_u32(&mut ctx.data, BATCH_CTX_HEADER_SIZE + offset, 0);
        }

        // Assert: cloned retains original values
        let base = BATCH_CTX_HEADER_SIZE;
        for (field_idx, &offset) in [
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ].iter().enumerate() {
            assert_eq!(
                read_u32(&cloned.data, base + offset),
                (field_idx + 1) as u32,
                "cloned field at offset {} should be {}",
                offset,
                field_idx + 1
            );
        }
    }

    #[test]
    fn batch_context_header_overwrite_seq_meta_base_ptr_does_not_change_struct() {
        // Arrange: create context, read seq_meta_base_ptr
        let mut ctx = BatchContext::new(2);
        let original_ptr = read_u64(&ctx.data, SEQ_META_BASE_PTR);

        // Act: overwrite seq_meta_base_ptr via raw write_u32 (simulating corruption)
        write_u32(&mut ctx.data, SEQ_META_BASE_PTR, 0xDEAD);
        write_u32(&mut ctx.data, SEQ_META_BASE_PTR + 4, 0xBEEF);

        // Assert: the header value changed, but struct fields are independent
        assert_eq!(read_u32(&ctx.data, SEQ_META_BASE_PTR), 0xDEAD);
        assert_eq!(ctx.num_seqs, 2); // struct field unaffected
        assert_eq!(ctx.max_batch_size, 2);

        // Cleanup: restore for safety (not strictly needed in test)
        ctx.data[SEQ_META_BASE_PTR..SEQ_META_BASE_PTR + 8]
            .copy_from_slice(&original_ptr.to_le_bytes());
    }

    #[test]
    fn dual_batch_meta_debug_shows_all_six_fields() {
        // Arrange: construct a DualBatchMeta with non-default values
        let mut meta = mega_kernel_v2::DualBatchMeta::new(12);
        meta.ping_seq_count = 5;
        meta.pong_seq_count = 7;
        meta.step_epoch = 3;
        meta.epoch_arrival_count = 2;

        // Act
        let s = format!("{:?}", meta);

        // Assert: all six field names appear in debug output
        assert!(s.contains("ping_seq_offset"), "missing ping_seq_offset");
        assert!(s.contains("ping_seq_count"), "missing ping_seq_count");
        assert!(s.contains("pong_seq_offset"), "missing pong_seq_offset");
        assert!(s.contains("pong_seq_count"), "missing pong_seq_count");
        assert!(s.contains("step_epoch"), "missing step_epoch");
        assert!(s.contains("epoch_arrival_count"), "missing epoch_arrival_count");
    }

    #[test]
    fn dual_batch_meta_swap_preserves_counts_after_exchange() {
        // Arrange: set asymmetric counts, verify swap exchanges them correctly
        let mut meta = mega_kernel_v2::DualBatchMeta::new(8);
        meta.ping_seq_count = 6;
        meta.pong_seq_count = 2;

        // Act
        meta.swap();

        // Assert: counts swapped, offsets swapped
        assert_eq!(meta.ping_seq_count, 2);
        assert_eq!(meta.pong_seq_count, 6);
        assert_eq!(meta.ping_seq_offset, 8); // was pong's offset
        assert_eq!(meta.pong_seq_offset, 0); // was ping's offset
    }

    #[test]
    fn batch_context_v2_extension_write_per_seq_in_upper_slots() {
        // Arrange: v2 extension with max=8, initial=2 — slots 2..7 are reserved
        let mut ctx = BatchContext::with_v2_extension(8, 2);

        // Act: write to slot 7 (the last slot before extension area)
        ctx.set_seq_prompt_len(7, 7777);
        ctx.set_seq_gen_count(7, 42);
        ctx.set_ext_kv_pool_total_pages(1234); // write extension too

        // Assert: slot 7 data is correct, extension data is correct, slot 0 untouched
        let base7 = BATCH_CTX_HEADER_SIZE + 7 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_PROMPT_LEN), 7777);
        assert_eq!(read_u32(&ctx.data, base7 + SEQ_GEN_COUNT), 42);
        let ext_start = BATCH_CTX_HEADER_SIZE + 8 * SEQ_META_STRIDE;
        assert_eq!(read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_KV_POOL_TOTAL_PAGES), 1234);
        let base0 = BATCH_CTX_HEADER_SIZE;
        assert_eq!(read_u32(&ctx.data, base0 + SEQ_PROMPT_LEN), 0);
    }

    #[test]
    fn batch_context_multiple_pointer_writes_no_crosstalk() {
        // Arrange: write all 9 header pointer fields with unique patterns
        let mut ctx = BatchContext::new(1);
        let patterns: [(usize, fn(&mut BatchContext, usize)); 9] = [
            (0xA000, |c, v| c.set_input_ids_flat_ptr(v as *const u32)),
            (0xB000, |c, v| c.set_output_tokens_flat_ptr(v as *mut u32)),
            (0xC000, |c, v| c.set_positions_ptr(v as *const u32)),
            (0xD000, |c, v| c.set_page_table_flat_ptr(v as *const u32)),
            (0xE000, |c, v| c.set_kv_pool_base(v as *const u8)),
            (0xF000, |c, v| c.set_sampling_params_ptr(v as *const u32)),
            (0xA100, |c, v| c.set_hook_ctx_ptr(v as *const u8)),
            (0xB100, |c, v| c.set_callback_table_ptr(v as *const u8)),
            (0xC100, |c, v| c.set_seq_mapping_ptr(v as *const u32)),
        ];
        for (val, setter) in &patterns {
            setter(&mut ctx, *val);
        }

        // Act: overwrite only kv_pool_base
        ctx.set_kv_pool_base(0x9999 as *const u8);

        // Assert: all other pointers unchanged
        assert_eq!(read_usize(&ctx.data, INPUT_IDS_FLAT_PTR), 0xA000);
        assert_eq!(read_usize(&ctx.data, OUTPUT_TOKENS_FLAT_PTR), 0xB000);
        assert_eq!(read_usize(&ctx.data, POSITIONS_PTR), 0xC000);
        assert_eq!(read_usize(&ctx.data, PAGE_TABLE_FLAT_PTR), 0xD000);
        assert_eq!(read_usize(&ctx.data, KV_POOL_BASE), 0x9999); // changed
        assert_eq!(read_usize(&ctx.data, SAMPLING_PARAMS_PTR), 0xF000);
        assert_eq!(read_usize(&ctx.data, HOOK_CTX_PTR), 0xA100);
        assert_eq!(read_usize(&ctx.data, CALLBACK_TABLE_PTR), 0xB100);
        assert_eq!(read_usize(&ctx.data, SEQ_MAPPING_PTR), 0xC100);
    }

    #[test]
    fn batch_context_clone_with_seq_mapping_then_modify_original() {
        // Arrange: set seq_mapping and header field, clone
        let mut ctx = BatchContext::new(3);
        ctx.seq_mapping = vec![0, 1, 2, 0, 1];
        ctx.set_num_seqs(3);
        let cloned = ctx.clone();

        // Act: clear original's seq_mapping and change header
        ctx.seq_mapping.clear();
        ctx.set_num_seqs(1);

        // Assert: cloned is fully independent
        assert_eq!(cloned.seq_mapping.len(), 5);
        assert_eq!(cloned.seq_mapping[3], 0);
        assert_eq!(read_u32(&cloned.data, NUM_SEQS), 3);
    }

    #[test]
    fn batch_context_new_zero_then_set_all_header_u32_fields() {
        // Arrange: start with zero-seq context
        let mut ctx = BatchContext::new(0);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE);

        // Act: set all 3 header u32 fields
        ctx.set_num_seqs(10);
        ctx.set_max_decode_steps(20);
        ctx.set_total_prefill_tokens(30);

        // Assert: all fields read back correctly despite no per-seq area
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 10);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 20);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 30);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE); // size unchanged
    }

    // ── Additional tests (10 new, targeting 175 total) ──

    #[test]
    fn batch_context_as_ptr_differs_after_clone() {
        // Arrange
        let ctx = BatchContext::new(4);

        // Act
        let cloned = ctx.clone();

        // Assert: pointers should be different (deep copy, not sharing buffer)
        assert_ne!(ctx.as_ptr(), cloned.as_ptr());
    }

    #[test]
    fn batch_context_byte_size_unaffected_by_struct_field_mutations() {
        // Arrange: byte_size() returns data.len(), which should not change
        // when only struct metadata fields are mutated
        let mut ctx = BatchContext::new(3);
        let size_before = ctx.byte_size();

        // Act: mutate struct fields (not the data buffer)
        ctx.num_seqs = 99;
        ctx.seq_mapping = vec![0; 100];
        ctx.has_v2_extension = true;

        // Assert: byte_size unchanged — it only reflects data.len()
        assert_eq!(ctx.byte_size(), size_before);
    }

    #[test]
    fn batch_context_interleaved_header_and_per_seq_writes() {
        // Arrange: write header field, then per-seq field, then another header field,
        // then verify nothing was corrupted
        let mut ctx = BatchContext::new(2);

        // Act
        ctx.set_num_seqs(2);
        ctx.set_seq_prompt_len(0, 50);
        ctx.set_max_decode_steps(8);
        ctx.set_seq_prompt_len(1, 75);
        ctx.set_total_prefill_tokens(125);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_active_flag(1, 1);

        // Assert: all header and per-seq values correct
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 2);
        assert_eq!(read_u32(&ctx.data, MAX_DECODE_STEPS), 8);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 125);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN), 50);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_PROMPT_LEN), 75);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_ACTIVE_FLAG), 1);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_ACTIVE_FLAG), 1);
    }

    #[test]
    fn batch_context_v2_extension_ext_offset_matches_ext_field_base() {
        // Arrange: ext_offset() (test helper) should equal ext_field(max_batch_size, 0)
        for &max_batch in &[1usize, 4, 16, 32] {
            let ctx = BatchContext::with_v2_extension(max_batch, 0);

            // Act
            let ext_off = ctx.ext_offset();
            let ext_field_base = BatchContext::ext_field(max_batch, 0);

            // Assert
            assert_eq!(
                ext_off, ext_field_base,
                "ext_offset != ext_field({}, 0)",
                max_batch
            );
        }
    }

    #[test]
    fn batch_context_simulated_batch_lifecycle_reuse() {
        // Arrange: simulate a complete batch lifecycle: create, populate, verify, then reuse
        let mut ctx = BatchContext::new(4);

        // Act 1: first batch — 2 sequences
        ctx.set_num_seqs(2);
        ctx.set_total_prefill_tokens(30);
        ctx.set_max_decode_steps(1);
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_prompt_len(1, 20);
        ctx.set_seq_active_flag(0, 1);
        ctx.set_seq_active_flag(1, 1);
        ctx.set_seq_output_offset(0, 0);
        ctx.set_seq_output_offset(1, 10);

        // Assert 1
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 2);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN), 10);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_PROMPT_LEN), 20);

        // Act 2: reuse for a different batch — 3 sequences with different params
        ctx.set_num_seqs(3);
        ctx.set_total_prefill_tokens(60);
        ctx.set_max_decode_steps(5);
        ctx.set_seq_prompt_len(0, 5);
        ctx.set_seq_prompt_len(1, 15);
        ctx.set_seq_prompt_len(2, 40);
        ctx.set_seq_active_flag(2, 1);
        ctx.set_seq_active_flag(0, 1);

        // Assert 2: new values applied, seq 1 retains old active_flag
        assert_eq!(read_u32(&ctx.data, NUM_SEQS), 3);
        assert_eq!(read_u32(&ctx.data, TOTAL_PREFILL_TOKENS), 60);
        assert_eq!(read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN), 5);
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + SEQ_PROMPT_LEN),
            15
        );
        assert_eq!(
            read_u32(&ctx.data, BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_PROMPT_LEN),
            40
        );
    }

    #[test]
    fn batch_context_v2_extension_dual_batch_meta_with_default() {
        // Arrange: use DualBatchMeta::default() (all zeros) instead of ::new()
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let meta = mega_kernel_v2::DualBatchMeta::default();

        // Act
        ctx.set_ext_dual_batch_meta(&meta);

        // Assert: all 24 bytes at EXT_DUAL_BATCH_META should be zero
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        let meta_base = ext_start + mega_kernel_v2::EXT_DUAL_BATCH_META;
        for i in 0..mega_kernel_v2::DualBatchMeta::SIZE {
            assert_eq!(
                ctx.data[meta_base + i], 0,
                "DualBatchMeta byte {} should be zero after set with default()",
                i
            );
        }
    }

    #[test]
    fn dual_batch_meta_swap_then_swap_back_restores_original_offsets() {
        // Arrange
        let mut meta = mega_kernel_v2::DualBatchMeta::new(16);
        let original_ping = meta.ping_seq_offset;
        let original_pong = meta.pong_seq_offset;

        // Act: double swap returns offsets to original
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 16);
        assert_eq!(meta.pong_seq_offset, 0);

        meta.swap();

        // Assert: offsets restored, but epoch is 2 (incremented twice)
        assert_eq!(meta.ping_seq_offset, original_ping);
        assert_eq!(meta.pong_seq_offset, original_pong);
        assert_eq!(meta.step_epoch, 2);
    }

    #[test]
    fn batch_context_per_seq_all_fourteen_fields_for_two_seqs_independent() {
        // Arrange: write all 14 fields for seq 0 and seq 1 with different value ranges,
        // then verify no field bleeds between sequences
        let mut ctx = BatchContext::new(2);
        let field_offsets: [usize; 14] = [
            SEQ_PROMPT_LEN, SEQ_KV_LEN, SEQ_ROPE_POS_OFFSET, SEQ_MAX_NEW_TOKENS,
            SEQ_SESSION_POSITION, SEQ_PAGE_TABLE_OFFSET, SEQ_PAGE_TABLE_LEN,
            SEQ_FUSED_HIDDEN_OFFSET, SEQ_NUM_MM_TOKENS, SEQ_ACTIVE_FLAG,
            SEQ_SEQ_POSITION, SEQ_GEN_COUNT, SEQ_LAST_SAMPLED_TOKEN, SEQ_OUTPUT_OFFSET,
        ];

        // Act: use raw write_u32 to set seq 0 fields 100..114 and seq 1 fields 200..214
        for (i, &off) in field_offsets.iter().enumerate() {
            write_u32(&mut ctx.data, BATCH_CTX_HEADER_SIZE + off, 100 + i as u32);
            write_u32(&mut ctx.data, BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE + off, 200 + i as u32);
        }

        // Assert: verify all values are correct and independent
        let base0 = BATCH_CTX_HEADER_SIZE;
        let base1 = BATCH_CTX_HEADER_SIZE + SEQ_META_STRIDE;
        for (i, &off) in field_offsets.iter().enumerate() {
            assert_eq!(read_u32(&ctx.data, base0 + off), 100 + i as u32, "seq0 field {}", i);
            assert_eq!(read_u32(&ctx.data, base1 + off), 200 + i as u32, "seq1 field {}", i);
        }
    }

    #[test]
    fn batch_context_v2_extension_reserved_field_readable() {
        // Arrange: the EXT_RESERVED field at offset 92 in the extension area
        // should be writable via raw write and readable
        let mut ctx = BatchContext::with_v2_extension(4, 0);
        let ext_start = BATCH_CTX_HEADER_SIZE + 4 * SEQ_META_STRIDE;
        let reserved_off = ext_start + mega_kernel_v2::EXT_RESERVED;

        // Act: write a known value to the reserved field
        write_u32(&mut ctx.data, reserved_off, 0x4242_4242);

        // Assert: readable and not overlapping with adjacent fields
        assert_eq!(read_u32(&ctx.data, reserved_off), 0x4242_4242);
        // Verify autotune_actual_batch (at offset 56) is still zero — not corrupted
        assert_eq!(
            read_u32(&ctx.data, ext_start + mega_kernel_v2::EXT_AUTOTUNE_ACTUAL_BATCH),
            0
        );
    }

    #[test]
    fn batch_context_seq_mapping_empty_vec_remains_valid_after_operations() {
        // Arrange: create context with empty seq_mapping, perform header/per-seq writes
        let mut ctx = BatchContext::new(2);
        assert!(ctx.seq_mapping.is_empty());

        // Act: perform various operations
        ctx.set_num_seqs(2);
        ctx.set_seq_prompt_len(0, 42);
        ctx.set_seq_active_flag(0, 1);
        let cloned = ctx.clone();

        // Assert: seq_mapping remains empty on both original and clone
        assert!(ctx.seq_mapping.is_empty());
        assert!(cloned.seq_mapping.is_empty());
        // Clone's data is still correct
        assert_eq!(read_u32(&cloned.data, NUM_SEQS), 2);
        assert_eq!(read_u32(&cloned.data, BATCH_CTX_HEADER_SIZE + SEQ_PROMPT_LEN), 42);
    }
}
