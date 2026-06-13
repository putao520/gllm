// Mega-Kernel 执行器 (SPEC §9.1)
//
// ARCH-RUST-IS-CODEGEN 铁律: 推理时 Rust 只做一次 CALL。
// 整个模型（embedding → N 层 → output ops）编译为单一 JIT 机器码，
// 推理时通过 MegaKernelFn 单次 CALL 完成。
// SPEC/39: 所有模型形态（decoder/encoder/embedding/rerank/classify）统一走
// mega-kernel 路径，图拓扑决定编译产物内容，无第二条编译路径。
//
// 无 fallback。编译失败 = 致命错误。


// ============================================================================
// KernelContext ABI (R1: single-pointer fn(ctx: *const u8) -> u32)
// ============================================================================

/// Telemetry bitmask for mega-kernel diagnostic probes.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TelemetryFlagsBitmask(pub u32);

/// Configuration for weight page JIT injection (SPEC/21 §8).
///
/// Controls whether the mega-kernel JIT compiler injects page fault detection
/// and prefetch trigger instructions at weight access points in the generated
/// machine code. When enabled, the JIT emits a page-table lookup before each
/// per-layer weight access and invokes the weight page fault callback on miss.
///
/// REQ-WP-008: Mega-Kernel 权重页 JIT 注入开关。
#[derive(Debug, Clone)]
pub struct WeightPageJitConfig {
    /// Enable weight page fault detection injection in JIT code.
    pub enabled: bool,
    /// Number of page table entries (must match WeightPageTable capacity).
    pub num_pages: usize,
    /// Page size in bytes (must match GlobalMemoryManager page size).
    pub page_size_bytes: usize,
    /// Prefetch distance: how many pages ahead to prefetch (0 = no prefetch).
    pub prefetch_distance: usize,
}

impl Default for WeightPageJitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_pages: 1024,
            page_size_bytes: 64 * 1024 * 1024, // 64 MiB default
            prefetch_distance: 0,
        }
    }
}

/// Configuration for KV page decompress JIT injection (SPEC/22 §7, REQ-COMP11).
///
/// Controls whether the mega-kernel JIT compiler injects decompress instructions
/// at KV page access points. When enabled, the JIT emits a codec check before
/// each KV page read and invokes the corresponding decompress callback
/// (Lz4/BitPackRle/NvcompAns) on the callback table.
///
/// REQ-COMP11: Mega-Kernel JIT 解压注入开关。
#[derive(Debug, Clone)]
pub struct KvPageDecompressConfig {
    /// Enable KV page decompress injection in JIT code.
    pub enabled: bool,
    /// Number of KV pages in the page table (must match page_table length).
    pub num_pages: usize,
    /// Uncompressed KV page size in bytes.
    pub page_size_bytes: usize,
}

impl Default for KvPageDecompressConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_pages: 1024,
            page_size_bytes: 64 * 1024, // 64 KiB default
        }
    }
}

/// Flat parameter block passed via single-pointer ABI to the mega-kernel.
///
/// Layout (0x90 = 144 bytes, all fields naturally aligned):
///
/// ```text
/// offset  size  field
/// 0x00    8     weight_blob_ptr
/// 0x08    8     kv_cache_ptr
/// 0x10    8     output_buffer_ptr
/// 0x18    8     hook_ctx_ptr
/// 0x20    8     seq_len_ptr
/// 0x28    8     rope_freqs_ptr
/// 0x30    8     kv_page_table_ptr
/// 0x38    8     batch_meta_ptr
/// 0x40    4     kv_page_size
/// 0x44    4     kv_num_layers
/// 0x48    4     kv_num_heads
/// 0x4C    4     kv_head_dim
/// 0x50    8     telemetry_ptr
/// 0x58    8     telemetry_flags (u32 + u32 padding)
/// 0x60    8     business_config_ptr
/// 0x68    8     weight_offsets_ptr
/// 0x70    8     weight_offsets_len
/// 0x78    8     callback_table_ptr
/// 0x80    8     scratch_buffer_ptr
/// 0x88    8     batch_ctx_ptr
/// 0x90    8     weight_page_table_ptr
/// 0x98    8     weight_page_fault_cb_ptr
/// 0xA0    4     weight_page_inject_flags
/// 0xA4    4     _pad1
/// 0xA8    8     kv_page_header_ptr      (REQ-COMP11)
/// 0xB0    4     decompress_inject_flags (REQ-COMP11)
/// 0xB4    4     _pad2                   (REQ-COMP11)
/// ```
#[repr(C)]
#[derive(Debug)]
pub struct KernelContext {
    pub weight_blob_ptr: *const u8,
    pub kv_cache_ptr: *mut u8,
    pub output_buffer_ptr: *mut u8,
    pub hook_ctx_ptr: *mut u8,
    pub seq_len_ptr: *const usize,
    pub rope_freqs_ptr: *const f32,
    pub kv_page_table_ptr: *const u32,
    pub batch_meta_ptr: *const u8,
    pub kv_page_size: u32,
    pub kv_num_layers: u32,
    pub kv_num_heads: u32,
    pub kv_head_dim: u32,
    pub telemetry_ptr: *mut u8,
    pub telemetry_flags: u32,
    _pad0: u32,
    pub business_config_ptr: *const u8,
    pub weight_offsets_ptr: *const usize,
    pub weight_offsets_len: usize,
    pub callback_table_ptr: *const u64,
    pub scratch_buffer_ptr: *mut u8,
    pub batch_ctx_ptr: *const u8,
    /// Weight page table for page fault detection (REQ-WP-008).
    /// Points to a flat array of page state entries (one per page).
    pub weight_page_table_ptr: *const u8,
    /// Callback table for weight page fault handling (REQ-WP-008).
    /// Slot 0: weight_page_fault callback (fn(page_id: u32, layer_idx: u32) -> u32).
    pub weight_page_fault_cb_ptr: *const u64,
    /// Weight page inject flags (REQ-WP-008):
    /// bit 0 = enabled, bits 1-31 = reserved.
    pub weight_page_inject_flags: u32,
    _pad1: u32,
    /// REQ-COMP11: Pointer to array of KvPageHeader for KV page decompress injection.
    /// The JIT reads `page.codec` and `page.compressed_size` from this array
    /// before each KV page access to decide whether to invoke a decompress callback.
    /// NULL = no decompress injection (all pages assumed uncompressed).
    pub kv_page_header_ptr: *const u8,
    /// REQ-COMP11: Decompress injection flags:
    /// bit 0 = enabled, bits 1-31 = reserved.
    pub decompress_inject_flags: u32,
    _pad2: u32,
}

impl KernelContext {
    /// Create a zero-initialized context.
    pub fn zeroed() -> Self {
        Self {
            weight_blob_ptr: std::ptr::null(),
            kv_cache_ptr: std::ptr::null_mut(),
            output_buffer_ptr: std::ptr::null_mut(),
            hook_ctx_ptr: std::ptr::null_mut(),
            seq_len_ptr: std::ptr::null(),
            rope_freqs_ptr: std::ptr::null(),
            kv_page_table_ptr: std::ptr::null(),
            batch_meta_ptr: std::ptr::null(),
            kv_page_size: 0,
            kv_num_layers: 0,
            kv_num_heads: 0,
            kv_head_dim: 0,
            telemetry_ptr: std::ptr::null_mut(),
            telemetry_flags: 0,
            _pad0: 0,
            business_config_ptr: std::ptr::null(),
            weight_offsets_ptr: std::ptr::null(),
            weight_offsets_len: 0,
            callback_table_ptr: std::ptr::null(),
            scratch_buffer_ptr: std::ptr::null_mut(),
            batch_ctx_ptr: std::ptr::null(),
            weight_page_table_ptr: std::ptr::null(),
            weight_page_fault_cb_ptr: std::ptr::null(),
            weight_page_inject_flags: 0,
            _pad1: 0,
            kv_page_header_ptr: std::ptr::null(),
            decompress_inject_flags: 0,
            _pad2: 0,
        }
    }

    /// Unified builder for CPU and GPU mega-kernel paths (R2).
    ///
    /// Both CPU and GPU paths call this to construct a fully-populated
    /// KernelContext. The resulting struct is passed via single-pointer ABI
    /// to the JIT-compiled mega-kernel entry point.
    ///
    /// `seq_len` is heap-allocated internally to produce a stable pointer
    /// (`seq_len_ptr`) that remains valid for the duration of the call.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        weight_blob_ptr: *const u8,
        kv_cache_ptr: *mut u8,
        output_buffer_ptr: *mut u8,
        hook_ctx_ptr: *mut u8,
        seq_len: usize,
        rope_freqs_ptr: *const f32,
        kv_page_table_ptr: *const u32,
        batch_meta_ptr: *const u8,
        kv_page_size: u32,
        kv_num_layers: u32,
        kv_num_heads: u32,
        kv_head_dim: u32,
        telemetry_ptr: *mut u8,
        telemetry_flags: u32,
        business_config_ptr: *const u8,
        weight_offsets_ptr: *const usize,
        weight_offsets_len: usize,
        callback_table_ptr: *const u64,
        scratch_buffer_ptr: *mut u8,
        batch_ctx_ptr: *const u8,
        weight_page_table_ptr: *const u8,
        weight_page_fault_cb_ptr: *const u64,
        weight_page_inject_flags: u32,
        kv_page_header_ptr: *const u8,
        decompress_inject_flags: u32,
    ) -> (Self, Box<usize>) {
        let seq_len_boxed = Box::new(seq_len);
        let ctx = Self {
            weight_blob_ptr,
            kv_cache_ptr,
            output_buffer_ptr,
            hook_ctx_ptr,
            seq_len_ptr: &*seq_len_boxed,
            rope_freqs_ptr,
            kv_page_table_ptr,
            batch_meta_ptr,
            kv_page_size,
            kv_num_layers,
            kv_num_heads,
            kv_head_dim,
            telemetry_ptr,
            telemetry_flags,
            _pad0: 0,
            business_config_ptr,
            weight_offsets_ptr,
            weight_offsets_len,
            callback_table_ptr,
            scratch_buffer_ptr,
            batch_ctx_ptr,
            weight_page_table_ptr,
            weight_page_fault_cb_ptr,
            weight_page_inject_flags,
            _pad1: 0,
            kv_page_header_ptr,
            decompress_inject_flags,
            _pad2: 0,
        };
        (ctx, seq_len_boxed)
    }
}

// Safety: KernelContext contains raw pointers but is Send+Sync because the
// mega-kernel only reads from it during execution (single-threaded access).
unsafe impl Send for KernelContext {}
unsafe impl Sync for KernelContext {}

/// Mega-kernel entry point signature: single pointer ABI (R1).
///
/// All parameters are passed through a KernelContext flat struct.
/// The callee reads fields at fixed offsets; Rust fills the struct before CALL.
pub type MegaKernelFn = unsafe extern "C" fn(ctx: *const u8) -> u32;

// ============================================================================
// MegaKernelExecutor
// ============================================================================

/// Mega-Kernel 编译错误
#[derive(Debug, thiserror::Error)]
pub enum MegaKernelError {
    #[error("compilation failed: {0}")]
    Compilation(String),
    #[error("execution failed: {0}")]
    Execution(String),
}

/// True mega-kernel 编译产物。
///
/// 持有完整的 mega-kernel 机器码（embedding → layer loop → logits-producer → sampling → generate loop）
/// + 全模型权重布局 + 缓冲布局。推理时通过单次 CALL 执行。
struct MegaKernelCompiled {
    /// (canonical_name, byte_offset) 对 — 用于诊断查询
    named_offsets: Vec<(String, usize)>,
    /// 运行时缓冲布局（activation ping/pong, logits, sampling workspace）
    buffer_layout: gllm_kernels::compiler::BufferLayout,
    /// Logits 区域在 scratchpad 中的偏移（alloc + RoPE cache 之后）
    logits_scratch_offset: usize,
    /// 预打包的连续权重 blob
    weight_blob: Vec<u8>,
    /// mmap'd 完整 mega-kernel 机器码（generate loop + embedded forward code，单一连续函数）
    exec_code: gllm_kernels::compiler::CompiledLayer,
    /// Legacy ABI function pointer (23-param) emitted by JIT.
    /// R1: KernelContext → unpack to legacy ABI → CALL.
    entry_fn: gllm_kernels::compiler::MegaKernelFn,
    /// RoPE cos/sin 表需求（caller 必须在每次调用前填充 scratchpad）
    rope_cache: Option<gllm_kernels::compiler::codegen::RopeCacheRequirement>,
    /// scratchpad 固定部分大小（intermediate tensors + RoPE cache），不含运行时 logits
    scratchpad_base_bytes: usize,
    /// GPU mega-kernel PTX/HIP 代码（可选，仅 GPU 路径使用）
    gpu_code: Option<Vec<u8>>,
    /// vocab_size — logits 每行元素数
    vocab_size: usize,
    /// hidden_dim — SG scratchpad 需要
    hidden: usize,
    /// 每元素字节数 (compute dtype)
    elem_bytes: usize,
    /// Layer 6: JIT source map — VmInstr → 机器码偏移 → Op 标签映射。
    /// 仅当 debug_jit=true 时生成，供 DAP 调试器使用。
    source_map: Option<gllm_kernels::compiler::codegen::vm::debug_map::JitSourceMap>,
    /// KV cache geometry — used to allocate KV cache buffer at runtime.
    num_kv_heads: usize,
    head_dim: usize,
    /// Compile-time max_seq_len used for KV cache layer stride computation.
    max_seq_len: usize,
    /// MTP depth (0 = MTP disabled, >0 = number of candidate tokens per decode step).
    mtp_depth: usize,
}

impl MegaKernelCompiled {
    /// 计算运行时 scratchpad 大小：固定部分 + logits(max_total 行) + sampling + MTP + SG
    fn runtime_scratchpad_bytes(&self, max_total: usize) -> usize {
        let vocab_bytes = self.vocab_size * self.elem_bytes;
        let logits_bytes = max_total * vocab_bytes;
        let sampling_bytes = vocab_bytes * 4;
        // MTP logits: depth additional vocab-sized rows for MTP candidate generation.
        let mtp_logits_bytes = self.mtp_depth * vocab_bytes;
        let mtp_sampling_bytes = self.mtp_depth * vocab_bytes * 4;
        let sg_end = if self.buffer_layout.sg_data_bytes > 0 {
            let sg_start = (self.logits_scratch_offset + logits_bytes + sampling_bytes
                + mtp_logits_bytes + mtp_sampling_bytes + 63) & !63;
            sg_start + self.hidden * self.elem_bytes * 2
        } else {
            0
        };

        // Mega-kernel scratchpad sizing:
        // - scratchpad_base_bytes (= logits_scratch_offset) covers all intermediate
        //   tensors from the VAM alloc, including activation ping/pong sentinel slots.
        // - logits_bytes uses max_total (actual tokens), NOT max_seq_len — the
        //   mega-kernel never needs logits for the full context window at once.
        // - buffer_layout.total_scratchpad_bytes is NOT used as a lower bound because
        //   it computes logits as max_seq_len * vocab * elem_bytes, which for large
        //   context models (Gemma 4 E2B: 131072 * 262144 * 4 = 128 GB) is grossly
        //   oversized. The alloc-based offsets already account for activation buffers.
        (self.scratchpad_base_bytes + logits_bytes
            + mtp_logits_bytes + mtp_sampling_bytes)
            .max(sg_end)
            .max(64)
    }

    /// Bytes per row of K or V data (num_kv_heads * head_dim * elem_bytes).
    fn kv_row_stride(&self) -> usize {
        self.num_kv_heads * self.head_dim * self.elem_bytes
    }

    /// Bytes per layer (K + V): 2 * max_seq_len * kv_row_stride.
    fn kv_layer_stride(&self) -> usize {
        2 * self.max_seq_len * self.kv_row_stride()
    }

    /// Total KV cache buffer size: num_layers * kv_layer_stride.
    fn kv_cache_bytes(&self, num_layers: usize) -> usize {
        num_layers * self.kv_layer_stride()
    }
}
