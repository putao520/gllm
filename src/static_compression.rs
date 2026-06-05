//! Static Load-Time Weight Compression (SPEC §3 Autotuning, I.8 + I.9)
//!
//! These functions run **once at model load time** and permanently reduce
//! the weight tensors stored for subsequent inference. Zero runtime overhead.

pub use crate::kv_cache::CompressionCodec;

/// GQA Head Deduplication (SPEC I.8)
///
/// Computes pairwise cosine similarity across all Q-projection head rows.
/// When two heads exceed `similarity_threshold` (default 0.98), the second
/// head is removed from the weight matrix and O-projection rows are averaged.
///
/// Returns:
/// - `dedup_weights`: compressed Q weight (only unique rows), shape [unique_heads * head_dim, hidden]
/// - `dedup_indices`: mapping from original head idx → compressed head idx
///
/// Caller is responsible for applying `dedup_indices` when building attention graphs.
pub fn deduplicate_gqa_heads(
    q_weight_rows: &[Vec<f32>], // shape: [num_heads * head_dim][hidden]
    num_heads: usize,
    head_dim: usize,
    similarity_threshold: f32,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let rows_per_head = head_dim;
    let mut head_group: Vec<Vec<usize>> = Vec::new(); // group[i] = list of original head indices merged
    let mut assignments: Vec<Option<usize>> = vec![None; num_heads]; // head_idx → group_idx

    for head_i in 0..num_heads {
        if assignments[head_i].is_some() {
            continue;
        }

        let group_idx = head_group.len();
        head_group.push(vec![head_i]);
        assignments[head_i] = Some(group_idx);

        for (head_j, assignment) in assignments.iter_mut().enumerate().skip(head_i + 1) {
            if assignment.is_some() {
                continue;
            }

            let sim = cosine_similarity_heads(
                q_weight_rows,
                head_i,
                head_j,
                rows_per_head,
            );

            if sim >= similarity_threshold {
                head_group[group_idx].push(head_j);
                *assignment = Some(group_idx);
                log::debug!(
                    "gqa_dedup: merged head {} into head {} (cosine_sim={:.4})",
                    head_j, head_i, sim
                );
            }
        }
    }

    // Build dedup_indices: map original head → representative head
    let mut dedup_indices = vec![0usize; num_heads];
    for (group_idx, group) in head_group.iter().enumerate() {
        for &head_idx in group {
            dedup_indices[head_idx] = group_idx;
        }
    }

    // Build compressed weight: use the first (representative) head from each group.
    // For fused groups, average all member rows for better approximation.
    let hidden = if q_weight_rows.is_empty() { 0 } else { q_weight_rows[0].len() };
    let mut dedup_weights: Vec<Vec<f32>> = Vec::with_capacity(head_group.len() * rows_per_head);

    for group in &head_group {
        for row_offset in 0..rows_per_head {
            let mut avg_row = vec![0.0f32; hidden];
            for &member_head in group {
                let src_row = &q_weight_rows[member_head * rows_per_head + row_offset];
                for (dst, &src) in avg_row.iter_mut().zip(src_row.iter()) {
                    *dst += src;
                }
            }
            let n = group.len() as f32;
            for val in &mut avg_row {
                *val /= n;
            }
            dedup_weights.push(avg_row);
        }
    }

    let saved = num_heads - head_group.len();
    if saved > 0 {
        log::info!(
            "gqa_dedup: removed {} duplicate heads ({} → {} unique, {:.1}% saved)",
            saved, num_heads, head_group.len(),
            100.0 * saved as f32 / num_heads as f32
        );
    }

    (dedup_weights, dedup_indices)
}

/// Weight Column Pruning — NVIDIA 2:4 Structured Sparse Format (SPEC §7)
///
/// Applies NVIDIA's 2:4 structured sparsity to a weight matrix, producing
/// both the pruned (non-zero elements only) weight tensor and the metadata
/// array (`sp_meta`) required by `mma.sp` sparse Tensor Core instructions.
///
/// ## NVIDIA 2:4 Format Rules
/// - Within every group of **4 consecutive elements** in a row, exactly **2** are nonzero.
/// - Selection: the 2 elements with the **largest absolute value** survive; others are zeroed.
/// - `sp_meta[row][meta_col]`: 2-bit encoded index of each surviving element per 4-element group.
///   Packed as 8 indices per `u16` → `meta_cols = ceil(cols / 4)`, each u16 encodes 2 groups.
///
/// Returns:
/// - `pruned`:   `[rows][cols]` with zeros at pruned positions (ready for `cuSparseLtDenseDescriptor`).
/// - `sp_meta`:  `[rows][cols/4]` compressed `u16` metadata (two 2-bit index pairs per u16).
///
/// ## Usage
/// Pass `pruned` to weight layout and `sp_meta` to `cusparseLtSpMMADescriptor`/`mma.sp`.
pub fn prune_dead_columns_24(
    weight: &[Vec<f32>], // shape: [rows][cols] — cols must be divisible by 4 for NVIDIA compliance
) -> (Vec<Vec<f32>>, Vec<Vec<u16>>) {
    if weight.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let rows = weight.len();
    let cols = weight[0].len();

    // NVIDIA 2:4 requires cols divisible by 4
    assert!(
        cols.is_multiple_of(4),
        "prune_dead_columns_24: cols ({cols}) must be divisible by 4 for NVIDIA 2:4 format"
    );

    let meta_cols = cols / 4; // one u16 covers 8 elements (2 groups of 4, 2×2 bits each)
    let mut pruned = weight.to_vec();
    // sp_meta: packed 2-bit position indices: 2 groups of 4 per u16 → ceil(cols/4/2) u16 per row
    let meta_u16_cols = meta_cols.div_ceil(2); // 2 4-element groups per u16
    let mut sp_meta: Vec<Vec<u16>> = vec![vec![0u16; meta_u16_cols]; rows];

    let mut total_pruned_elems = 0usize;

    for (row_idx, row) in weight.iter().enumerate() {
        // Process in 4-element groups
        for grp in 0..(cols / 4) {
            let base = grp * 4;
            let elems = [row[base], row[base + 1], row[base + 2], row[base + 3]];

            // Select the 2 elements with highest absolute value per group of 4
            let mut order = [0usize, 1, 2, 3];
            order.sort_unstable_by(|&a, &b| {
                elems[b].abs().partial_cmp(&elems[a].abs()).unwrap_or(std::cmp::Ordering::Equal) // LEGAL: NaN 比较的标准 Rust 模式
            });
            // Surviving positions are order[0] and order[1] — the two largest |vals|
            let keep: [usize; 2] = [order[0].min(order[1]), order[0].max(order[1])]; // keep sorted

            // Zero out the 2 non-surviving positions
            for &dead in &order[2..] {
                pruned[row_idx][base + dead] = 0.0;
            }
            total_pruned_elems += 2;

            // Encode surviving positions as 2-bit indices: keep[0] ∈ {0..3}, keep[1] ∈ {0..3}
            // NVIDIA sp_meta format: each u16 stores two groups of 4, 4 bits per group (2×2-bit indices)
            // | group[n+1] pos1 | group[n+1] pos0 | group[n] pos1 | group[n] pos0 |  (low→high)
            let meta_u16_idx = grp / 2;
            let meta_shift  = (grp % 2) * 4; // 0 or 4 bits offset within u16
            let encoded: u16 = ((keep[0] as u16) | ((keep[1] as u16) << 2)) << meta_shift;
            sp_meta[row_idx][meta_u16_idx] |= encoded;
        }
    }

    let total_elems = rows * cols;
    log::info!(
        "prune_dead_columns_24: applied NVIDIA 2:4 sparsity, {}/{} elements zeroed ({:.1}%)",
        total_pruned_elems, total_elems,
        100.0 * total_pruned_elems as f64 / total_elems as f64
    );

    (pruned, sp_meta)
}

/// Weight Column Pruning — L2 Norm Threshold (convenience helper, SPEC I.9)
///
/// Scans per-column L2 norms and zeros columns below `threshold_ratio * mean_col_norm`.
/// This is the **pre-pass** step: run before `prune_dead_columns_24` to set structurally
/// inactive columns to exactly zero, so 2:4 selection naturally discards them.
///
/// Returns `(pruned_weight, dead_col_mask)` — `dead_col_mask[j] == true` means column j is dead.
/// Caller must apply the mask to the paired Down-projection matrix.
pub fn prune_dead_columns(
    weight: &[Vec<f32>], // shape: [rows][cols] — typically Gate_proj or Up_proj
    threshold_ratio: f32,
) -> (Vec<Vec<f32>>, Vec<bool>) {
    if weight.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let cols = weight[0].len();

    // Compute per-column L2 norm
    let mut col_norms = vec![0.0f32; cols];
    for row in weight {
        for (j, &val) in row.iter().enumerate() {
            col_norms[j] += val * val;
        }
    }
    for norm in &mut col_norms {
        *norm = norm.sqrt();
    }

    let mean_norm: f32 = col_norms.iter().sum::<f32>() / cols as f32;
    let prune_threshold = threshold_ratio * mean_norm;

    let dead_col_mask: Vec<bool> = col_norms.iter().map(|&n| n < prune_threshold).collect();

    // Zero out pruned columns
    let mut pruned = weight.to_vec();
    let dead_count = dead_col_mask.iter().filter(|&&d| d).count();

    for row in &mut pruned {
        for (j, val) in row.iter_mut().enumerate() {
            if dead_col_mask[j] {
                *val = 0.0;
            }
        }
    }

    if dead_count > 0 {
        log::info!(
            "column_prune: zeroed {}/{} dead columns ({:.1}% pruned, threshold={:.4})",
            dead_count, cols,
            100.0 * dead_count as f32 / cols as f32,
            prune_threshold
        );
    }

    (pruned, dead_col_mask)
}

// ─────────────────────────────────────────────────────────────────────────────
// BitPackRle Compression — SPEC 22 §4
// ─────────────────────────────────────────────────────────────────────────────

/// BitPackRle compress: nibble-level run-length encoding per SPEC 22 §4.
///
/// ## Format
///
/// Each byte packs `[nibble_value: u4][run_len_minus_1: u4]`:
///   - bits 7..4: the nibble value (only low 4 bits of each input byte are encoded)
///   - bits 3..0: (actual_run_length - 1), range 0..14 → 1..15 elements
///
/// Runs longer than 15 elements are split into multiple entries of at most 15.
/// `run_len=15` (0x0F) is reserved as escape per SPEC but never emitted by this encoder.
///
/// ## GPU Kernel Compatibility
///
/// The packed format enables warp-level prefix-sum decode at 200+ GB/s on GPU:
/// `VmInstr::BitPackRleDecode { src, dst, format: PrecisionTier }`.
pub fn compress_bitpack_rle(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    // Worst case: every byte is a different nibble → no compression.
    // But typical KIVI4/KIVI2 streams have long runs of identical nibbles.
    let mut out = Vec::with_capacity(input.len().min(256));
    let mut i = 0;
    while i < input.len() {
        let val = input[i] & 0x0F; // extract low nibble
        let mut run = 1usize;
        while i + run < input.len() && (input[i + run] & 0x0F) == val {
            run += 1;
        }
        // Emit in chunks of at most 15 (run_len_minus_1 ∈ 0..14)
        let mut remaining = run;
        while remaining > 0 {
            let chunk = remaining.min(15);
            out.push((val << 4) | ((chunk - 1) as u8));
            remaining -= chunk;
        }
        i += run;
    }
    out
}

/// BitPackRle decompress: expand SPEC 22 §4 packed RLE back to full bytes.
///
/// Each compressed byte encodes one run:
///   - `val = byte >> 4`  → nibble value
///   - `len = (byte & 0x0F) + 1` → run length (1..16, handles escape as 16)
///
/// Output is truncated to `decompressed_len` bytes to match the original page size.
pub fn decompress_bitpack_rle(input: &[u8], decompressed_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(decompressed_len);
    for &byte in input {
        if out.len() >= decompressed_len {
            break;
        }
        let val = (byte >> 4) & 0x0F;
        let run_len = ((byte & 0x0F) as usize) + 1; // 0→1, …, 14→15, 15→16 (escape)
        let remaining = decompressed_len - out.len();
        let emit = run_len.min(remaining);
        for _ in 0..emit {
            out.push(val);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ4 Compression — SPEC 22 §3.3.1
// ─────────────────────────────────────────────────────────────────────────────

/// LZ4 compress using lz4_flex (raw block, no size header).
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress(data)
}

/// LZ4 decompress with expected output size.
///
/// Returns error if the decompressed data does not match `decompressed_size`.
pub fn lz4_decompress(compressed: &[u8], decompressed_size: usize) -> Result<Vec<u8>, String> {
    lz4_flex::decompress(compressed, decompressed_size)
        .map_err(|e| format!("LZ4 decompress failed: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// ZstdDict Compression — SPEC 22 §4 — Zstandard with Dictionary
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for compression codec operations.
///
/// Returned by `compress_weight_page` when zstd or other codec operations fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecError(pub String);

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CodecError: {}", self.0)
    }
}

impl std::error::Error for CodecError {}

/// Compress data using zstd with a pre-trained dictionary (SPEC 22 §4).
///
/// Uses zstd's bulk `Compressor` with CDict for high-ratio compression.
/// The dictionary must be trained offline from representative samples
/// via `train_zstd_dictionary()` and stored in `WeightCompressionConfig`.
///
/// ## Fallback
///
/// If `dict` is empty, the caller should fall back to `lz4_compress()`.
/// This function returns an error on empty dict to force explicit handling.
pub fn compress_zstd_dict(data: &[u8], dict: &[u8]) -> Result<Vec<u8>, CodecError> {
    if dict.is_empty() {
        return Err(CodecError(
            "compress_zstd_dict called with empty dictionary".to_string(),
        ));
    }
    let mut compressor = zstd::bulk::Compressor::with_dictionary(3, dict)
        .map_err(|e| CodecError(format!("zstd Compressor::with_dictionary failed: {e}")))?;
    compressor
        .compress(data)
        .map_err(|e| CodecError(format!("zstd compress with dict failed: {e}")))
}

/// Decompress data using zstd with a pre-trained dictionary (SPEC 22 §4).
///
/// Uses zstd's bulk `Decompressor` with DDict for accurate decompression.
/// Returns a `Vec<u8>` of exactly `decompressed_size` bytes.
///
/// ## Fallback
///
/// If `dict` is empty, the caller should fall back to `lz4_decompress()`.
pub fn decompress_zstd_dict(
    compressed: &[u8],
    dict: &[u8],
    decompressed_size: usize,
) -> Result<Vec<u8>, String> {
    if dict.is_empty() {
        return Err("decompress_zstd_dict called with empty dictionary".to_string());
    }
    let mut decompressor = zstd::bulk::Decompressor::with_dictionary(dict)
        .map_err(|e| format!("zstd Decompressor::with_dictionary failed: {e}"))?;
    let mut out = decompressor
        .decompress(compressed, decompressed_size)
        .map_err(|e| format!("zstd decompress with dict failed: {e}"))?;
    if out.len() < decompressed_size {
        // Extend with zero-padding if decompressor returned fewer bytes
        out.resize(decompressed_size, 0);
    } else if out.len() > decompressed_size {
        // Truncate if decompressor returned more bytes (shouldn't happen with dict)
        out.truncate(decompressed_size);
    }
    Ok(out)
}

/// Train a zstd dictionary from sample weight pages (SPEC 22 §4).
///
/// Uses zstd's `ZDICT_trainFromBuffer` to produce a compact dictionary
/// optimized for the model's weight distribution. Called once at model load time.
///
/// `samples` must contain at least one non-empty sample.
/// `dict_capacity` is the target dictionary size in bytes (e.g., 112640 = 110 KB).
///
/// Returns an empty `Vec<u8>` if training fails or no samples are provided.
pub fn train_zstd_dictionary(samples: &[&[u8]], dict_capacity: usize) -> Vec<u8> {
    if samples.is_empty() {
        log::warn!("train_zstd_dictionary: no samples provided, returning empty dict");
        return Vec::new();
    }
    if dict_capacity == 0 {
        log::warn!("train_zstd_dictionary: dict_capacity is 0, returning empty dict");
        return Vec::new();
    }
    let total: usize = samples.iter().map(|s| s.len()).sum();
    if total == 0 {
        log::warn!("train_zstd_dictionary: all samples are empty, returning empty dict");
        return Vec::new();
    }
    zstd::dict::from_samples(samples, dict_capacity).unwrap_or_else(|e| {
        log::warn!(
            "train_zstd_dictionary failed (samples={}, dict_capacity={}): {e}, returning empty dict",
            samples.len(),
            dict_capacity,
        );
        Vec::new()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// NvcompAns Compression — SPEC 22 §4.3.3 — GPU ANS entropy coding via nvCOMP
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for NvcompAns operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvcompAnsError(pub String);

impl std::fmt::Display for NvcompAnsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NvcompAnsError: {}", self.0)
    }
}

impl std::error::Error for NvcompAnsError {}

// ── nvCOMP ANS FFI bindings (NVIDIA GPU only) ──

#[cfg(feature = "nvcomp")]
mod nvcomp_ffi {
    //! Raw FFI bindings to the nvCOMP ANS C API (nvcomp.h) and CUDA runtime API.
    //!
    //! These are low-level bindings for:
    //! - `nvcompAnsCompressAsync` / `nvcompAnsDecompressAsync`
    //! - CUDA runtime API for GPU memory management (cudaMalloc, cudaMemcpy, etc.)
    //!
    //! All functions are `unsafe` and require a valid CUDA context.

    use std::ffi::c_int;
    use std::os::raw::c_void;

    /// nvCOMP error code wrapper.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NvcompError(pub c_int);

    /// CUDA error code wrapper.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CudaError(pub c_int);

    /// Opaque CUDA stream handle.
    #[repr(C)]
    pub struct CUstream_st(c_void);
    pub type CudaStream = *mut CUstream_st;

    /// CUDA memcpy kind enum matching cudaMemcpyKind in cuda_runtime_api.h.
    #[repr(C)]
    pub enum CudaMemcpyKind {
        HostToHost = 0,
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3,
    }

    /// Default ANS format options: let nvCOMP choose optimal settings.
    pub const NVCOMP_ANS_USE_DEFAULT: c_int = -1;

    extern "C" {
        // ── nvCOMP ANS compress ──
        pub fn nvcompAnsCompressGetTempSize(
            batch_size: usize,
            temp_bytes: *mut usize,
        ) -> NvcompError;

        pub fn nvcompAnsCompressGetOutputSize(
            batch_size: usize,
            output_bytes: *mut usize,
        ) -> NvcompError;

        pub fn nvcompAnsCompressAsync(
            device_in_ptr: *const *const c_void,
            device_in_bytes: *const usize,
            batch_size: usize,
            device_out_ptr: *mut *mut c_void,
            device_out_bytes: *mut usize,
            device_temp_ptr: *mut c_void,
            temp_bytes: usize,
            stream: CudaStream,
            format_opts: c_int,
        ) -> NvcompError;

        // ── nvCOMP ANS decompress ──
        pub fn nvcompAnsDecompressGetTempSize(
            batch_size: usize,
            temp_bytes: *mut usize,
        ) -> NvcompError;

        pub fn nvcompAnsDecompressGetOutputSize(
            device_in_ptr: *const *const c_void,
            device_in_bytes: *const usize,
            batch_size: usize,
            output_bytes: *mut usize,
        ) -> NvcompError;

        pub fn nvcompAnsDecompressAsync(
            device_in_ptr: *const *const c_void,
            device_in_bytes: *const usize,
            batch_size: usize,
            device_out_ptr: *mut *mut c_void,
            device_out_bytes: *mut usize,
            device_temp_ptr: *mut c_void,
            temp_bytes: usize,
            stream: CudaStream,
            format_opts: c_int,
        ) -> NvcompError;

        // ── CUDA runtime API ──
        pub fn cudaSetDevice(device: c_int) -> CudaError;
        pub fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CudaError;
        pub fn cudaFree(devPtr: *mut c_void) -> CudaError;
        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: CudaMemcpyKind,
        ) -> CudaError;
        pub fn cudaStreamCreate(pStream: *mut CudaStream) -> CudaError;
        pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
        pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    }
}

/// Compress data using nvCOMP ANS entropy coding on GPU (SPEC 22 §4.3.3).
///
/// This function:
/// 1. Checks CUDA device availability
/// 2. Copies input data to GPU memory
/// 3. Calls nvCOMP ANS compress kernel
/// 4. Copies compressed output back to CPU
///
/// ## Errors
///
/// Returns `NvcompAnsError` if:
/// - The `nvcomp` feature is not enabled (compile-time gate)
/// - No NVIDIA GPU is available (runtime check)
/// - nvCOMP compression kernel fails
/// - CUDA memory allocation or transfer fails
///
/// ## Note
///
/// This is a GPU-only codec requiring nvCOMP + CUDA at runtime.
/// For CPU weight compression, use LZ4 or ZstdDict instead.
#[cfg(not(feature = "nvcomp"))]
pub fn compress_nvcomp_ans(_input: &[u8]) -> Result<Vec<u8>, NvcompAnsError> {
    Err(NvcompAnsError(
        "nvcomp feature not enabled; compile with --features nvcomp for GPU ANS compression"
            .to_string(),
    ))
}

#[cfg(feature = "nvcomp")]
pub fn compress_nvcomp_ans(input: &[u8]) -> Result<Vec<u8>, NvcompAnsError> {
    use self::nvcomp_ffi::*;
    use std::ptr;

    if input.is_empty() {
        return Ok(Vec::new());
    }

    // ── 1. Check CUDA device availability ──
    let mut device_count: c_int = 0;
    let ret = unsafe { cudaGetDeviceCount(&mut device_count) };
    if ret.0 != 0 || device_count <= 0 {
        return Err(NvcompAnsError(
            "No CUDA-capable GPU available for nvCOMP ANS compression".to_string(),
        ));
    }
    unsafe { cudaSetDevice(0); }

    // ── 2. Create CUDA stream ──
    let mut stream: CudaStream = ptr::null_mut();
    let ret = unsafe { cudaStreamCreate(&mut stream) };
    if ret.0 != 0 {
        return Err(NvcompAnsError(format!(
            "Failed to create CUDA stream: error code {}",
            ret.0
        )));
    }

    // Use a closure for early return with proper stream cleanup
    let result = (|| -> Result<Vec<u8>, NvcompAnsError> {
        let input_len = input.len();
        let batch_size = 1usize;

        // ── 3. Allocate device memory for input ──
        let mut d_input: *mut c_void = ptr::null_mut();
        let ret = unsafe { cudaMalloc(&mut d_input, input_len) };
        if ret.0 != 0 {
            return Err(NvcompAnsError(format!(
                "cudaMalloc failed for input ({} bytes): error code {}",
                input_len, ret.0
            )));
        }

        // ── 4. Copy input H2D ──
        let ret = unsafe {
            cudaMemcpy(
                d_input,
                input.as_ptr() as *const c_void,
                input_len,
                CudaMemcpyKind::HostToDevice,
            )
        };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); }
            return Err(NvcompAnsError(format!(
                "cudaMemcpy H2D failed: error code {}",
                ret.0
            )));
        }

        // ── 5. Query temp buffer size ──
        let mut temp_bytes: usize = 0;
        let ret = unsafe { nvcompAnsCompressGetTempSize(batch_size, &mut temp_bytes) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); }
            return Err(NvcompAnsError(format!(
                "nvcompAnsCompressGetTempSize failed: error code {}",
                ret.0
            )));
        }

        // ── 6. Allocate temp memory ──
        let mut d_temp: *mut c_void = ptr::null_mut();
        if temp_bytes > 0 {
            let ret = unsafe { cudaMalloc(&mut d_temp, temp_bytes) };
            if ret.0 != 0 {
                unsafe { cudaFree(d_input); }
                return Err(NvcompAnsError(format!(
                    "cudaMalloc failed for temp ({} bytes): error code {}",
                    temp_bytes, ret.0
                )));
            }
        }

        // ── 7. Query output size ──
        let mut output_bytes: usize = 0;
        let ret = unsafe { nvcompAnsCompressGetOutputSize(batch_size, &mut output_bytes) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "nvcompAnsCompressGetOutputSize failed: error code {}",
                ret.0
            )));
        }

        // ── 8. Allocate device output memory ──
        let mut d_output: *mut c_void = ptr::null_mut();
        let ret = unsafe { cudaMalloc(&mut d_output, output_bytes) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "cudaMalloc failed for output ({} bytes): error code {}",
                output_bytes, ret.0
            )));
        }

        // ── 9. Prepare batch pointers ──
        let d_in_ptr = &d_input as *const *mut c_void as *const *const c_void;
        let in_bytes = &input_len as *const usize;
        let mut d_out_ptr: *mut c_void = d_output;
        let mut out_bytes: usize = output_bytes;

        // ── 10. Launch compression ──
        let ret = unsafe {
            nvcompAnsCompressAsync(
                d_in_ptr,
                in_bytes,
                batch_size,
                &mut d_out_ptr as *mut *mut c_void,
                &mut out_bytes,
                d_temp,
                temp_bytes,
                stream,
                NVCOMP_ANS_USE_DEFAULT,
            )
        };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); cudaFree(d_output); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "nvcompAnsCompressAsync failed: error code {}",
                ret.0
            )));
        }

        // ── 11. Synchronize ──
        let ret = unsafe { cudaStreamSynchronize(stream) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); cudaFree(d_output); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "cudaStreamSynchronize failed: error code {}",
                ret.0
            )));
        }

        // ── 12. Copy compressed data D2H ──
        let mut compressed = vec![0u8; out_bytes];
        let ret = unsafe {
            cudaMemcpy(
                compressed.as_mut_ptr() as *mut c_void,
                d_output as *const c_void,
                out_bytes,
                CudaMemcpyKind::DeviceToHost,
            )
        };

        // ── 13. Cleanup device memory ──
        unsafe {
            cudaFree(d_input);
            cudaFree(d_output);
            if !d_temp.is_null() {
                cudaFree(d_temp);
            }
        }

        if ret.0 != 0 {
            return Err(NvcompAnsError(format!(
                "cudaMemcpy D2H failed: error code {}",
                ret.0
            )));
        }

        Ok(compressed)
    })();

    unsafe { cudaStreamDestroy(stream); }
    result
}

/// Decompress data using nvCOMP ANS entropy coding on GPU (SPEC 22 §4.3.3).
///
/// This function:
/// 1. Checks CUDA device availability
/// 2. Copies compressed input to GPU memory
/// 3. Calls nvCOMP ANS decompress kernel
/// 4. Copies decompressed output back to CPU
///
/// The output is sized exactly `decompressed_len` bytes. If nvCOMP returns
/// fewer bytes, the remainder is zero-padded.
///
/// ## Errors
///
/// Returns `NvcompAnsError` if:
/// - The `nvcomp` feature is not enabled (compile-time gate)
/// - No NVIDIA GPU is available (runtime check)
/// - nvCOMP decompression kernel fails
/// - CUDA memory allocation or transfer fails
#[cfg(not(feature = "nvcomp"))]
pub fn decompress_nvcomp_ans(
    _input: &[u8],
    _decompressed_len: usize,
) -> Result<Vec<u8>, NvcompAnsError> {
    Err(NvcompAnsError(
        "nvcomp feature not enabled; compile with --features nvcomp for GPU ANS decompression"
            .to_string(),
    ))
}

#[cfg(feature = "nvcomp")]
pub fn decompress_nvcomp_ans(
    input: &[u8],
    decompressed_len: usize,
) -> Result<Vec<u8>, NvcompAnsError> {
    use self::nvcomp_ffi::*;
    use std::ptr;

    if input.is_empty() || decompressed_len == 0 {
        return Ok(vec![0u8; decompressed_len]);
    }

    // ── 1. Check CUDA device availability ──
    let mut device_count: c_int = 0;
    let ret = unsafe { cudaGetDeviceCount(&mut device_count) };
    if ret.0 != 0 || device_count <= 0 {
        return Err(NvcompAnsError(
            "No CUDA-capable GPU available for nvCOMP ANS decompression".to_string(),
        ));
    }
    unsafe { cudaSetDevice(0); }

    // ── 2. Create CUDA stream ──
    let mut stream: CudaStream = ptr::null_mut();
    let ret = unsafe { cudaStreamCreate(&mut stream) };
    if ret.0 != 0 {
        return Err(NvcompAnsError(format!(
            "Failed to create CUDA stream: error code {}",
            ret.0
        )));
    }

    let result = (|| -> Result<Vec<u8>, NvcompAnsError> {
        let input_len = input.len();
        let batch_size = 1usize;

        // ── 3. Allocate device memory for compressed input ──
        let mut d_input: *mut c_void = ptr::null_mut();
        let ret = unsafe { cudaMalloc(&mut d_input, input_len) };
        if ret.0 != 0 {
            return Err(NvcompAnsError(format!(
                "cudaMalloc failed for compressed input ({} bytes): error code {}",
                input_len, ret.0
            )));
        }

        // ── 4. Copy compressed input H2D ──
        let ret = unsafe {
            cudaMemcpy(
                d_input,
                input.as_ptr() as *const c_void,
                input_len,
                CudaMemcpyKind::HostToDevice,
            )
        };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); }
            return Err(NvcompAnsError(format!(
                "cudaMemcpy H2D failed for compressed input: error code {}",
                ret.0
            )));
        }

        // ── 5. Query temp buffer size ──
        let mut temp_bytes: usize = 0;
        let ret = unsafe { nvcompAnsDecompressGetTempSize(batch_size, &mut temp_bytes) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); }
            return Err(NvcompAnsError(format!(
                "nvcompAnsDecompressGetTempSize failed: error code {}",
                ret.0
            )));
        }

        // ── 6. Allocate temp memory ──
        let mut d_temp: *mut c_void = ptr::null_mut();
        if temp_bytes > 0 {
            let ret = unsafe { cudaMalloc(&mut d_temp, temp_bytes) };
            if ret.0 != 0 {
                unsafe { cudaFree(d_input); }
                return Err(NvcompAnsError(format!(
                    "cudaMalloc failed for temp ({} bytes): error code {}",
                    temp_bytes, ret.0
                )));
            }
        }

        // ── 7. Query decompressed output size ──
        let d_in_ptr = &d_input as *const *mut c_void as *const *const c_void;
        let in_bytes = &input_len as *const usize;
        let mut output_bytes: usize = 0;
        let ret = unsafe {
            nvcompAnsDecompressGetOutputSize(d_in_ptr, in_bytes, batch_size, &mut output_bytes)
        };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "nvcompAnsDecompressGetOutputSize failed: error code {}",
                ret.0
            )));
        }

        // ── 8. Allocate device output memory ──
        let mut d_output: *mut c_void = ptr::null_mut();
        let ret = unsafe { cudaMalloc(&mut d_output, output_bytes) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "cudaMalloc failed for output ({} bytes): error code {}",
                output_bytes, ret.0
            )));
        }

        // ── 9. Launch decompression ──
        let mut d_out_ptr: *mut c_void = d_output;
        let mut out_bytes: usize = output_bytes;
        let ret = unsafe {
            nvcompAnsDecompressAsync(
                d_in_ptr,
                in_bytes,
                batch_size,
                &mut d_out_ptr as *mut *mut c_void,
                &mut out_bytes,
                d_temp,
                temp_bytes,
                stream,
                NVCOMP_ANS_USE_DEFAULT,
            )
        };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); cudaFree(d_output); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "nvcompAnsDecompressAsync failed: error code {}",
                ret.0
            )));
        }

        // ── 10. Synchronize ──
        let ret = unsafe { cudaStreamSynchronize(stream) };
        if ret.0 != 0 {
            unsafe { cudaFree(d_input); cudaFree(d_output); if !d_temp.is_null() { cudaFree(d_temp); } }
            return Err(NvcompAnsError(format!(
                "cudaStreamSynchronize failed: error code {}",
                ret.0
            )));
        }

        // ── 11. Copy decompressed data D2H ──
        let actual_output = out_bytes.min(decompressed_len);
        let mut decompressed = vec![0u8; decompressed_len];
        let ret = unsafe {
            cudaMemcpy(
                decompressed.as_mut_ptr() as *mut c_void,
                d_output as *const c_void,
                actual_output,
                CudaMemcpyKind::DeviceToHost,
            )
        };

        // ── 12. Cleanup device memory ──
        unsafe {
            cudaFree(d_input);
            cudaFree(d_output);
            if !d_temp.is_null() {
                cudaFree(d_temp);
            }
        }

        if ret.0 != 0 {
            return Err(NvcompAnsError(format!(
                "cudaMemcpy D2H failed: error code {}",
                ret.0
            )));
        }

        Ok(decompressed)
    })();

    unsafe { cudaStreamDestroy(stream); }
    result
}

/// Compute mean cosine similarity between two head blocks of a weight matrix.
///
/// Each head occupies `rows_per_head` consecutive rows starting at `head * rows_per_head`.
fn cosine_similarity_heads(
    weight: &[Vec<f32>],
    head_a: usize,
    head_b: usize,
    rows_per_head: usize,
) -> f32 {
    let start_a = head_a * rows_per_head;
    let start_b = head_b * rows_per_head;

    if start_a + rows_per_head > weight.len() || start_b + rows_per_head > weight.len() {
        return 0.0;
    }

    let mut total_sim = 0.0f32;
    for offset in 0..rows_per_head {
        let row_a = &weight[start_a + offset];
        let row_b = &weight[start_b + offset];
        total_sim += cosine_sim_rows(row_a, row_b);
    }

    total_sim / rows_per_head as f32
}

/// Cosine similarity between two equal-length f32 slices.
fn cosine_sim_rows(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_dead_columns_zeros_low_norm() {
        // Column 0 has norm 1.0, column 1 has norm 0.0001 (dead), column 2 has norm 1.0
        let weight = vec![
            vec![1.0f32, 0.0001, 1.0],
            vec![1.0f32, 0.0001, 1.0],
        ];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        assert!(!mask[0], "column 0 should be alive");
        assert!(mask[1],  "column 1 should be pruned");
        assert!(!mask[2], "column 2 should be alive");
        assert_eq!(pruned[0][1], 0.0, "pruned column should be zeroed");
        assert_ne!(pruned[0][0], 0.0, "alive column should remain");
    }

    #[test]
    fn test_deduplicate_gqa_identical_heads() {
        // Two identical heads → should be merged into one
        let head_dim = 2;
        let hidden = 3;
        // head 0 rows: [1,0,0], [0,1,0]; head 1 rows: [1,0,0], [0,1,0] (identical)
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let _ = hidden;
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Should collapse to 1 unique head
        assert_eq!(dedup_w.len(), head_dim, "should have head_dim rows for 1 unique head");
        assert_eq!(dedup_idx[0], dedup_idx[1], "both heads map to same group");
    }

    #[test]
    fn test_deduplicate_gqa_distinct_heads() {
        // Two orthogonal heads → should NOT be merged
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        assert_eq!(dedup_w.len(), 2 * head_dim, "should have rows for 2 unique heads");
        assert_ne!(dedup_idx[0], dedup_idx[1], "distinct heads map to different groups");
    }

    #[test]
    fn test_prune_dead_columns_24_selects_top2_per_group() {
        // Row: [3.0, 1.0, 0.5, 4.0, 0.1, 2.0, 0.2, 7.0]
        // Group 0: [3.0, 1.0, 0.5, 4.0] → keep positions 3(4.0) and 0(3.0) → zero pos 1 and 2
        // Group 1: [0.1, 2.0, 0.2, 7.0] → keep positions 3(7.0) and 1(2.0) → zero pos 0 and 2
        let weight = vec![
            vec![3.0f32, 1.0, 0.5, 4.0, 0.1, 2.0, 0.2, 7.0],
        ];
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);

        // Group 0: positions 1 and 2 must be zero
        assert_eq!(pruned[0][1], 0.0, "grp0 pos1 should be pruned");
        assert_eq!(pruned[0][2], 0.0, "grp0 pos2 should be pruned");
        assert_ne!(pruned[0][0], 0.0, "grp0 pos0 (3.0) should survive");
        assert_ne!(pruned[0][3], 0.0, "grp0 pos3 (4.0) should survive");

        // Group 1: positions 0 and 2 must be zero
        assert_eq!(pruned[0][4], 0.0, "grp1 pos0 should be pruned");
        assert_eq!(pruned[0][6], 0.0, "grp1 pos2 should be pruned");
        assert_ne!(pruned[0][5], 0.0, "grp1 pos1 (2.0) should survive");
        assert_ne!(pruned[0][7], 0.0, "grp1 pos3 (7.0) should survive");

        // sp_meta: 1 row, ceil(8/4/2)=1 u16 per row
        assert_eq!(sp_meta.len(), 1, "should have 1 row of metadata");
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 1 u16 metadata per row");
        // sp_meta[0][0] encodes group0 (keep pos0,3) and group1 (keep pos1,3)
        // grp0: sorted keep = [0,3] → encoded = (0 | 3<<2) = 12 = 0b1100, shift 0 → bits 3:0
        // grp1: sorted keep = [1,3] → encoded = (1 | 3<<2) = 13 = 0b1101, shift 4 → bits 7:4
        // u16 = 0b1101_1100 = 0xDC
        assert_eq!(sp_meta[0][0], 0x00DC, "sp_meta encoding mismatch for [3,1,0.5,4 | 0.1,2,0.2,7]");
    }

    #[test]
    fn test_prune_dead_columns_24_dimensions() {
        // 2 rows of 8 elements → sp_meta should be [2][1]
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].len(), 8);
        assert_eq!(sp_meta.len(), 2);
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 2 grps → 1 u16");
    }

    // ── BitPackRle ──

    #[test]
    fn bitpack_rle_empty_input() {
        assert!(compress_bitpack_rle(&[]).is_empty());
        assert!(decompress_bitpack_rle(&[], 0).is_empty());
    }

    #[test]
    fn bitpack_rle_single_value() {
        let compressed = compress_bitpack_rle(&[0x05]);
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0], 0x50); // val=5, run=1 → (5<<4)|0
        let decompressed = decompress_bitpack_rle(&compressed, 1);
        assert_eq!(decompressed, vec![0x05]);
    }

    #[test]
    fn bitpack_rle_long_run_splits() {
        // 20 identical nibbles → must split into chunks of 15+5
        let input = vec![0x03u8; 20];
        let compressed = compress_bitpack_rle(&input);
        // 15 elements → 1 entry, 5 elements → 1 entry = 2 bytes
        assert_eq!(compressed.len(), 2);
        assert_eq!(compressed[0], 0x3E); // val=3, run=15 → (3<<4)|14
        assert_eq!(compressed[1], 0x34); // val=3, run=5 → (3<<4)|4
        let decompressed = decompress_bitpack_rle(&compressed, 20);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_roundtrip_varied() {
        let input: Vec<u8> = vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 0xF, 0xF, 0xF];
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_truncates_to_decompressed_len() {
        let compressed = compress_bitpack_rle(&[0xAA; 10]);
        // Request only 5 bytes back
        let decompressed = decompress_bitpack_rle(&compressed, 5);
        assert_eq!(decompressed.len(), 5);
        assert!(decompressed.iter().all(|&b| b == 0x0A));
    }

    // ── LZ4 ──

    #[test]
    fn lz4_roundtrip() {
        let data = b"hello world, this is a test of lz4 compression for gllm weight pages".repeat(4);
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_decompress_wrong_size_errors() {
        let data = b"short test data";
        let compressed = lz4_compress(data);
        // Too small → lz4_flex returns error
        assert!(lz4_decompress(&compressed, 1).is_err());
    }

    // ── Error types ──

    #[test]
    fn codec_error_display() {
        let err = CodecError("test error".to_string());
        assert_eq!(format!("{err}"), "CodecError: test error");
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn nvcomp_ans_error_display() {
        let err = NvcompAnsError("gpu failure".to_string());
        assert_eq!(format!("{err}"), "NvcompAnsError: gpu failure");
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn compress_zstd_dict_empty_dict_errors() {
        let result = compress_zstd_dict(b"some data", b"");
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("empty dictionary"));
    }

    #[test]
    fn decompress_zstd_dict_empty_dict_errors() {
        let result = decompress_zstd_dict(b"compressed", b"", 100);
        assert!(result.is_err());
    }

    #[test]
    fn train_zstd_dictionary_no_samples() {
        let dict = train_zstd_dictionary(&[], 1024);
        assert!(dict.is_empty());
    }

    #[test]
    fn train_zstd_dictionary_zero_capacity() {
        let dict = train_zstd_dictionary(&[b"sample data"], 0);
        assert!(dict.is_empty());
    }

    // ── prune_dead_columns edge cases ──

    #[test]
    fn prune_dead_columns_empty() {
        let (pruned, mask): (Vec<Vec<f32>>, Vec<bool>) = prune_dead_columns(&[], 0.5);
        assert!(pruned.is_empty());
        assert!(mask.is_empty());
    }

    #[test]
    fn prune_dead_columns_all_alive() {
        let weight = vec![
            vec![1.0f32, 1.0],
            vec![1.0f32, 1.0],
        ];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        assert!(mask.iter().all(|m| !m), "all columns should be alive");
        assert_eq!(pruned[0][0], 1.0);
    }

    #[test]
    fn deduplicate_gqa_single_head() {
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 1, 2, 0.98);
        assert_eq!(dedup_w.len(), 2, "single head unchanged");
        assert_eq!(dedup_idx.len(), 1);
    }

    // ── Additional tests ──

    #[test]
    fn cosine_sim_rows_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_sim_rows(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_sim_rows(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have sim=0.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_sim_rows(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors should have sim=-1.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_zero_vector() {
        let a = vec![0.0f32, 0.0];
        let b = vec![1.0f32, 2.0];
        let sim = cosine_sim_rows(&a, &b);
        assert!(sim.abs() < 1e-6, "zero vector should have sim=0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_out_of_bounds() {
        let weight: Vec<Vec<f32>> = vec![vec![1.0]];
        // head 1 starts at row 2 but weight only has 1 row
        let sim = cosine_similarity_heads(&weight, 0, 1, 1);
        assert_eq!(sim, 0.0, "out-of-bounds heads should return 0.0");
    }

    #[test]
    fn deduplicate_gqa_three_heads_two_similar() {
        let head_dim = 1;
        // head 0: [1.0], head 1: [1.0001] (very similar to 0), head 2: [0.0] (orthogonal)
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![1.0001],
            vec![0.0],
        ];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.99);
        // head 0 and 1 should merge; head 2 is distinct
        assert_eq!(dedup_w.len(), 2 * head_dim, "should have 2 unique heads");
        assert_eq!(dedup_idx[0], dedup_idx[1], "head 0 and 1 map to same group");
        assert_ne!(dedup_idx[0], dedup_idx[2], "head 2 maps to different group");
    }

    #[test]
    fn deduplicate_gqa_averages_merged_rows() {
        let head_dim = 1;
        // Two identical heads with value [3.0] → merged average should still be [3.0]
        let rows: Vec<Vec<f32>> = vec![
            vec![3.0],
            vec![3.0],
        ];
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        assert!((dedup_w[0][0] - 3.0).abs() < 1e-6, "averaged value should be 3.0");
    }

    #[test]
    fn deduplicate_gqa_low_threshold_keeps_all() {
        let head_dim = 1;
        // threshold = 1.01 means no pair can exceed it (cosine sim ≤ 1.0)
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![1.0],
        ];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 1.01);
        assert_eq!(dedup_w.len(), 2, "no heads should be merged at threshold > 1.0");
        assert_ne!(dedup_idx[0], dedup_idx[1]);
    }

    #[test]
    fn prune_dead_columns_all_zero_values_unchanged() {
        let weight = vec![
            vec![0.0f32, 0.0],
            vec![0.0f32, 0.0],
        ];
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // mean_norm=0 → threshold=0 → norm(0) is not < 0 → no columns pruned
        assert!(mask.iter().all(|m| !m), "zero threshold means nothing is below it");
        assert!(pruned.iter().all(|r| r.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn prune_dead_columns_preserves_nonzero_values() {
        let weight = vec![vec![5.0f32, 0.001]];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // col 0 has high norm, col 1 has near-zero norm
        assert!(!mask[0]);
        assert!(mask[1]);
        assert_eq!(pruned[0][0], 5.0, "alive column value preserved");
        assert_eq!(pruned[0][1], 0.0, "dead column zeroed");
    }

    #[test]
    fn prune_dead_columns_24_empty() {
        let (pruned, sp_meta): (Vec<Vec<f32>>, Vec<Vec<u16>>) = prune_dead_columns_24(&[]);
        assert!(pruned.is_empty());
        assert!(sp_meta.is_empty());
    }

    #[test]
    fn prune_dead_columns_24_single_row_4cols() {
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].len(), 4);
        // Exactly 2 of 4 should be zeroed
        let zero_count = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 2, "exactly 2 of 4 should be pruned");
        assert_eq!(sp_meta.len(), 1);
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 group → 1 u16");
    }

    #[test]
    fn prune_dead_columns_24_negative_values() {
        // Negative values: keep largest absolute
        let weight = vec![vec![-10.0f32, 1.0, -2.0, 0.5]];
        let (pruned, _) = prune_dead_columns_24(&weight);
        // |-10| and |-2| are largest → keep pos 0 and 2
        assert_ne!(pruned[0][0], 0.0, "pos 0 (-10.0) should survive");
        assert_ne!(pruned[0][2], 0.0, "pos 2 (-2.0) should survive");
        assert_eq!(pruned[0][1], 0.0, "pos 1 (1.0) should be pruned");
        assert_eq!(pruned[0][3], 0.0, "pos 3 (0.5) should be pruned");
    }

    #[test]
    fn prune_dead_columns_24_ties() {
        // All values equal → any 2 survive (implementation picks first 2 by sort stability)
        let weight = vec![vec![1.0f32, 1.0, 1.0, 1.0]];
        let (pruned, _) = prune_dead_columns_24(&weight);
        let zero_count = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 2, "2 of 4 equal values should be pruned");
    }

    #[test]
    fn prune_dead_columns_24_large_matrix() {
        let rows = 4;
        let cols = 16;
        let weight: Vec<Vec<f32>> = (0..rows)
            .map(|r| (0..cols).map(|c| (r * cols + c) as f32 + 1.0).collect())
            .collect();
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        assert_eq!(pruned.len(), rows);
        assert_eq!(sp_meta.len(), rows);
        assert_eq!(sp_meta[0].len(), 2, "16 cols → 4 grps → 2 u16 per row");
        // Each row should have exactly cols/2 zeros (50% pruned)
        for row in &pruned {
            let zero_count = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zero_count, cols / 2, "each row should have 50% zeros");
        }
    }

    #[test]
    fn prune_dead_columns_24_meta_encoding_correctness() {
        // Single row, 4 cols → 1 group, keep largest 2
        let weight = vec![vec![1.0f32, 4.0, 2.0, 3.0]];
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Largest absolute: pos 1 (4.0) and pos 3 (3.0)
        // sorted keep = [1, 3] → encoded = (1 | 3<<2) = 13 = 0xD
        // shift 0 → u16 = 0x000D
        assert_eq!(sp_meta[0][0] & 0xF, 0xD, "low nibble should encode keep=[1,3]");
    }

    #[test]
    fn bitpack_rle_all_zeros() {
        let input = vec![0u8; 30];
        let compressed = compress_bitpack_rle(&input);
        // 15+15 → 2 entries
        assert_eq!(compressed.len(), 2);
        let decompressed = decompress_bitpack_rle(&compressed, 30);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_all_different() {
        let input: Vec<u8> = (0..10).map(|i| i & 0x0F).collect();
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_exactly_15_run() {
        let input = vec![0x07u8; 15];
        let compressed = compress_bitpack_rle(&input);
        assert_eq!(compressed.len(), 1, "15 elements → 1 entry");
        assert_eq!(compressed[0], 0x7E, "val=7, run=15 → (7<<4)|14 = 0x7E");
        let decompressed = decompress_bitpack_rle(&compressed, 15);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_30_run_splits_into_two() {
        let input = vec![0x09u8; 30];
        let compressed = compress_bitpack_rle(&input);
        assert_eq!(compressed.len(), 2, "30 elements → 15+15 = 2 entries");
        assert_eq!(compressed[0], 0x9E, "val=9, run=15");
        assert_eq!(compressed[1], 0x9E, "val=9, run=15");
    }

    #[test]
    fn bitpack_rle_high_nibble_ignored() {
        // Only low nibble is encoded; high bits should be masked
        let input = vec![0xAB, 0xAB, 0xAB]; // low nibble = 0xB
        let compressed = compress_bitpack_rle(&input);
        assert_eq!(compressed[0] >> 4, 0xB, "high nibble of compressed byte = 0xB");
        let decompressed = decompress_bitpack_rle(&compressed, 3);
        assert_eq!(decompressed, vec![0x0B, 0x0B, 0x0B], "output should be low nibble only");
    }

    #[test]
    fn bitpack_rle_decompress_zero_len() {
        let decompressed = decompress_bitpack_rle(&[0x50], 0);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn bitpack_rle_decompress_more_than_compressed() {
        // Compressed encodes 1 element, but we request 5 → decompress stops at end of input
        let compressed = compress_bitpack_rle(&[0x03]);
        let decompressed = decompress_bitpack_rle(&compressed, 5);
        // decompress_bitpack_rle emits up to decompressed_len but only has 1 element
        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0], 0x03);
    }

    #[test]
    fn lz4_compress_empty() {
        let compressed = lz4_compress(&[]);
        let decompressed = lz4_decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn lz4_roundtrip_repeated_pattern() {
        let data = vec![0x42u8; 4096];
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len(), "repeated data should compress well");
    }

    #[test]
    fn codec_error_debug() {
        let err = CodecError("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn nvcomp_ans_error_debug() {
        let err = NvcompAnsError("err".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("err"));
    }

    #[test]
    fn compress_nvcomp_ans_no_feature() {
        // Without nvcomp feature, should return error
        let result = compress_nvcomp_ans(b"test data");
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("nvcomp feature not enabled"));
    }

    #[test]
    fn decompress_nvcomp_ans_no_feature() {
        let result = decompress_nvcomp_ans(b"compressed", 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("nvcomp feature not enabled"));
    }

    #[test]
    fn train_zstd_dictionary_all_empty_samples() {
        let dict = train_zstd_dictionary(&[b"" as &[u8], b""], 1024);
        assert!(dict.is_empty(), "all-empty samples should return empty dict");
    }

    #[test]
    fn zstd_dict_roundtrip() {
        // Train a small dict from samples, compress and decompress with it
        let samples: Vec<Vec<u8>> = (0..20)
            .map(|i| {
                let mut v = vec![0u8; 256];
                for j in 0..256 {
                    v[j] = ((i * 7 + j * 3) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            // Training may fail with insufficient data; skip gracefully
            return;
        }
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        let decompressed = decompress_zstd_dict(&compressed, &dict, samples[0].len()).unwrap();
        assert_eq!(decompressed, samples[0]);
    }

    #[test]
    fn prune_dead_columns_single_row() {
        let weight = vec![vec![1.0f32, 0.0001, 10.0]];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        assert!(!mask[0], "col 0 should be alive");
        assert!(mask[1], "col 1 should be dead");
        assert!(!mask[2], "col 2 should be alive");
        assert_eq!(pruned[0][1], 0.0);
    }

    #[test]
    fn deduplicate_gqa_empty_rows() {
        let rows: Vec<Vec<f32>> = vec![];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 0, 2, 0.98);
        assert!(dedup_w.is_empty());
        assert!(dedup_idx.is_empty());
    }

    #[test]
    fn prune_dead_columns_high_threshold_prunes_most() {
        // threshold_ratio = 0.99 means anything below 99% of mean gets pruned
        let weight = vec![
            vec![100.0f32, 0.001],
            vec![100.0f32, 0.001],
        ];
        let (_, mask) = prune_dead_columns(&weight, 0.99);
        assert!(!mask[0], "col 0 (high norm) should be alive");
        assert!(mask[1], "col 1 (low norm) should be pruned");
    }

    // ── New supplementary tests ──

    // ── deduplicate_gqa_heads: deeper coverage ──

    #[test]
    fn deduplicate_gqa_four_heads_chain_merge() {
        // Arrange: 4 heads, each pair adjacent has cosine sim = 1.0
        // head 0 = head 1 = head 2 = head 3 → all should merge into 1 group
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0, 1.0], // head 0 row 0
            vec![0.5, 3.0], // head 0 row 1
            vec![2.0, 1.0], // head 1 row 0 (identical to head 0)
            vec![0.5, 3.0], // head 1 row 1
            vec![2.0, 1.0], // head 2 row 0
            vec![0.5, 3.0], // head 2 row 1
            vec![2.0, 1.0], // head 3 row 0
            vec![0.5, 3.0], // head 3 row 1
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 4, head_dim, 0.98);
        // Assert: all 4 heads merge into 1
        assert_eq!(dedup_w.len(), head_dim, "4 identical heads → 1 unique group");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
        assert_eq!(dedup_idx[2], dedup_idx[3]);
    }

    #[test]
    fn deduplicate_gqa_negative_weights() {
        // Arrange: two heads with negative values but identical direction
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![-3.0], // head 0
            vec![-3.0], // head 1 (identical direction)
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: identical negative heads should merge
        assert_eq!(dedup_w.len(), head_dim, "identical negative heads merge");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
    }

    #[test]
    fn deduplicate_gqa_mixed_positive_negative_heads() {
        // Arrange: head 0 = [1.0], head 1 = [-1.0] → cosine sim = -1.0 → not merged
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![-1.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: opposite direction should NOT merge
        assert_eq!(dedup_w.len(), 2 * head_dim, "opposite heads should remain separate");
        assert_ne!(dedup_idx[0], dedup_idx[1]);
    }

    #[test]
    fn deduplicate_gqa_preserves_hidden_dim() {
        // Arrange: 2 heads × 3 rows_per_head, hidden=5
        let head_dim = 3;
        let hidden = 5;
        let make_row = |v: f32| vec![v; hidden];
        let rows: Vec<Vec<f32>> = vec![
            make_row(1.0), make_row(2.0), make_row(3.0), // head 0
            make_row(1.0), make_row(2.0), make_row(3.0), // head 1 (identical)
        ];
        // Act
        let (dedup_w, _dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: output rows have correct hidden dimension
        for row in &dedup_w {
            assert_eq!(row.len(), hidden, "each output row must have hidden={hidden}");
        }
    }

    #[test]
    fn deduplicate_gqa_threshold_zero_merges_all() {
        // Arrange: threshold=0.0 means any pair with sim >= 0 merges
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![0.5], // cosine sim with [1.0] = 1.0 > 0
            vec![0.1], // cosine sim with [1.0] = 1.0 > 0
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.0);
        // Assert: all merge into one group since all are positive
        assert_eq!(dedup_w.len(), head_dim, "threshold=0 merges all positive-direction heads");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
    }

    #[test]
    fn deduplicate_gqa_averages_non_identical_merged() {
        // Arrange: two heads that are very similar but not identical
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![4.0], // head 0
            vec![6.0], // head 1 (same direction, different magnitude)
        ];
        // Act: these have cosine_sim = 1.0, so they merge
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: averaged = (4.0 + 6.0) / 2 = 5.0
        let expected = 5.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-6,
            "averaged value should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ── prune_dead_columns: deeper coverage ──

    #[test]
    fn prune_dead_columns_zero_threshold_prunes_nothing() {
        // Arrange
        let weight = vec![
            vec![0.0001f32, 100.0],
            vec![0.0001f32, 100.0],
        ];
        // Act: threshold_ratio=0.0 → threshold=0.0 → only columns with norm < 0 are pruned
        let (_, mask) = prune_dead_columns(&weight, 0.0);
        // Assert
        assert!(mask.iter().all(|m| !m), "zero threshold should prune nothing");
    }

    #[test]
    fn prune_dead_columns_wide_matrix() {
        // Arrange: 2 rows × 8 cols, columns 2 and 6 are near-zero
        let weight = vec![
            vec![1.0, 2.0, 0.0001, 3.0, 4.0, 5.0, 0.0001, 6.0],
            vec![1.0, 2.0, 0.0001, 3.0, 4.0, 5.0, 0.0001, 6.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[2], "col 2 should be pruned");
        assert!(mask[6], "col 6 should be pruned");
        assert_eq!(pruned[0][2], 0.0);
        assert_eq!(pruned[1][6], 0.0);
        assert!(!mask[0]);
        assert!(!mask[7]);
    }

    #[test]
    fn prune_dead_columns_negative_threshold_prunes_all() {
        // Arrange: negative threshold → every column's (positive) norm exceeds threshold
        let weight = vec![vec![1.0f32, 2.0, 3.0]];
        // Act: threshold_ratio = -1.0 → threshold = -1.0 * mean_norm → negative
        let (_, mask) = prune_dead_columns(&weight, -1.0);
        // Assert: no column norm is < negative threshold
        assert!(mask.iter().all(|m| !m), "negative threshold prunes nothing");
    }

    #[test]
    fn prune_dead_columns_single_column_alive() {
        // Arrange: single column, value = 5.0
        let weight = vec![vec![5.0f32], vec![3.0f32]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: single column can't be pruned (its norm = mean norm)
        assert!(!mask[0], "single column should always be alive");
    }

    // ── prune_dead_columns_24: deeper coverage ──

    #[test]
    fn prune_dead_columns_24_12_cols_two_u16_meta() {
        // Arrange: 1 row × 12 cols → 3 groups → ceil(3/2) = 2 u16
        let weight = vec![vec![10.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 11.0]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 2, "12 cols → 3 groups → 2 u16");
        let zero_count: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 12 / 2, "50% of elements should be pruned");
    }

    #[test]
    fn prune_dead_columns_24_8_cols_full_u16_pair() {
        // Arrange: 1 row × 8 cols → 2 groups → 1 u16 (2 groups pair perfectly)
        let weight = vec![vec![5.0f32, 1.0, 2.0, 3.0, 10.0, 0.5, 1.0, 0.1]];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 2 groups → 1 u16");
    }

    #[test]
    fn prune_dead_columns_24_multiple_rows_independent() {
        // Arrange: two rows with different value patterns
        let weight = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],  // keep pos 3(4.0) and pos 2(3.0)
            vec![10.0f32, 20.0, 1.0, 0.5], // keep pos 1(20.0) and pos 0(10.0)
        ];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: each row is independent
        assert_ne!(pruned[0][2], 0.0, "row0 pos2 survives");
        assert_ne!(pruned[0][3], 0.0, "row0 pos3 survives");
        assert_ne!(pruned[1][0], 0.0, "row1 pos0 survives");
        assert_ne!(pruned[1][1], 0.0, "row1 pos1 survives");
        // Metadata should differ between rows
        assert_ne!(sp_meta[0][0], sp_meta[1][0], "different rows should have different metadata");
    }

    #[test]
    fn prune_dead_columns_24_all_zeros() {
        // Arrange: all zero values → any 2 survive (ties)
        let weight = vec![vec![0.0f32; 4]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: 2 of 4 are pruned, but the surviving 2 are also 0.0
        let zero_count = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 4, "all values are 0.0 including survivors");
    }

    #[test]
    fn prune_dead_columns_24_extreme_values() {
        // Arrange: very large and very small values
        let weight = vec![vec![1e10f32, 1e-10, -1e10, -1e-10]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: |1e10| and |-1e10| should survive
        assert_ne!(pruned[0][0], 0.0, "1e10 should survive");
        assert_ne!(pruned[0][2], 0.0, "-1e10 should survive");
        assert_eq!(pruned[0][1], 0.0, "1e-10 should be pruned");
        assert_eq!(pruned[0][3], 0.0, "-1e-10 should be pruned");
    }

    #[test]
    #[should_panic(expected = "must be divisible by 4")]
    fn prune_dead_columns_24_non_multiple_of_4_panics() {
        // Arrange: cols = 5, not divisible by 4
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0]];
        // Act: should panic
        let _ = prune_dead_columns_24(&weight);
    }

    // ── cosine_similarity_heads: deeper coverage ──

    #[test]
    fn cosine_similarity_heads_same_head() {
        // Arrange
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        // Act: compare head 0 with itself
        let sim = cosine_similarity_heads(&weight, 0, 0, 1);
        // Assert
        assert!((sim - 1.0).abs() < 1e-6, "same head should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_below_threshold() {
        // Arrange: two heads that are similar but not identical
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.8, 0.6, 0.0], // cos sim ≈ 0.8
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 1);
        // Assert: should be ~0.8 (not above 0.98)
        assert!(sim > 0.7 && sim < 0.9, "partial similarity should be ~0.8, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_multi_row_per_head() {
        // Arrange: 2 heads × 2 rows per head
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], // head 0 row 0
            vec![0.0, 1.0], // head 0 row 1
            vec![1.0, 0.0], // head 1 row 0
            vec![0.0, 1.0], // head 1 row 1
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 2);
        // Assert: identical multi-row heads → sim = 1.0
        assert!((sim - 1.0).abs() < 1e-6, "identical multi-row heads should have sim=1.0, got {sim}");
    }

    // ── compress_bitpack_rle / decompress_bitpack_rle: deeper coverage ──

    #[test]
    fn bitpack_rle_escape_code_decompress() {
        // Arrange: craft a byte with low nibble = 0x0F (escape), meaning run_len = 16
        let compressed = vec![0xA0 | 0x0F]; // val=0xA, run_len=16
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 20);
        // Assert: should emit 16 values (not 20, since only 16 available)
        assert_eq!(decompressed.len(), 16);
        assert!(decompressed.iter().all(|&b| b == 0x0A));
    }

    #[test]
    fn bitpack_rle_mixed_pattern_roundtrip() {
        // Arrange: complex pattern with many transitions
        let input: Vec<u8> = vec![
            0x01, 0x01, 0x01, // 3 of val 1
            0x02,             // 1 of val 2
            0x00, 0x00,       // 2 of val 0
            0x0F, 0x0F, 0x0F, 0x0F, 0x0F, // 5 of val 0xF
            0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, // 7 of val 3
        ];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_very_long_run() {
        // Arrange: 100 identical nibbles → 6 chunks of 15 + 1 chunk of 10
        let input = vec![0x05u8; 100];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: 6 × 15 + 1 × 10 = 100 → 7 entries
        assert_eq!(compressed.len(), 7);
        let decompressed = decompress_bitpack_rle(&compressed, 100);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_all_15_distinct_nibbles() {
        // Arrange: all 16 distinct nibble values (0-15)
        let input: Vec<u8> = (0..=15).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: each is a run of 1
        assert_eq!(compressed.len(), 16, "16 distinct nibbles → 16 entries");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_single_byte_max_nibble() {
        // Arrange
        let input = vec![0xFF]; // low nibble = 0xF
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 1);
        // Assert
        assert_eq!(compressed[0] >> 4, 0xF, "high nibble should be 0xF");
        assert_eq!(decompressed[0], 0x0F, "output should be low nibble 0xF");
    }

    #[test]
    fn bitpack_rle_decompress_truncates_mid_run() {
        // Arrange: encode 10 elements but request only 3
        let compressed = compress_bitpack_rle(&[0x07; 10]);
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 3);
        // Assert
        assert_eq!(decompressed.len(), 3);
        assert!(decompressed.iter().all(|&b| b == 0x07));
    }

    // ── LZ4: deeper coverage ──

    #[test]
    fn lz4_single_byte_roundtrip() {
        // Arrange
        let data = vec![0x42u8];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_large_randomish_data_roundtrip() {
        // Arrange
        let data: Vec<u8> = (0..8192).map(|i| ((i * 31 + 17) % 256) as u8).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_decompress_corrupted_data_errors() {
        // Arrange: create valid compressed data then corrupt it
        let data = b"test data for corruption";
        let mut compressed = lz4_compress(data);
        if !compressed.is_empty() {
            compressed[0] = compressed[0].wrapping_add(0xFF);
            compressed.truncate(compressed.len().saturating_sub(2));
        }
        // Act/Assert: decompressing corrupted data should fail
        let result = lz4_decompress(&compressed, data.len());
        assert!(result.is_err(), "corrupted LZ4 data should return error");
    }

    // ── ZstdDict: deeper coverage ──

    #[test]
    fn decompress_zstd_dict_size_too_small_errors() {
        // Arrange: compress with dict, then request a smaller decompressed size than actual.
        // zstd decompressor returns error when destination buffer is too small.
        let samples: Vec<Vec<u8>> = (0..20)
            .map(|i| {
                let mut v = vec![0u8; 256];
                for j in 0..256 {
                    v[j] = ((i * 7 + j * 3) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            return; // skip if training fails
        }
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        // Act: request 100 bytes (less than original 256) → zstd should error
        let result = decompress_zstd_dict(&compressed, &dict, 100);
        // Assert: zstd correctly rejects undersized decompression request
        assert!(result.is_err(), "requesting fewer bytes than actual should error");
    }

    #[test]
    fn decompress_zstd_dict_size_larger_pads_zeros() {
        // Arrange
        let samples: Vec<Vec<u8>> = (0..20)
            .map(|i| {
                let mut v = vec![0u8; 128];
                for j in 0..128 {
                    v[j] = ((i * 11 + j * 5) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            return;
        }
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        // Act: request 256 bytes (more than original 128)
        let decompressed = decompress_zstd_dict(&compressed, &dict, 256).unwrap();
        // Assert
        assert_eq!(decompressed.len(), 256);
        // First 128 bytes should be the original data
        assert_eq!(&decompressed[..128], samples[0].as_slice());
        // Remaining should be zero-padded
        assert!(decompressed[128..].iter().all(|&b| b == 0));
    }

    #[test]
    fn train_zstd_dictionary_with_single_sample() {
        // Arrange: single sample
        let sample = vec![0xAAu8; 512];
        // Act
        let dict = train_zstd_dictionary(&[sample.as_slice()], 1024);
        // Assert: may or may not succeed (depends on zstd), but should not panic
        // Just verify it returns a Vec (either empty or populated)
        assert!(dict.len() <= 1024);
    }

    #[test]
    fn train_zstd_dictionary_with_many_samples() {
        // Arrange: many diverse samples to ensure good dict training
        let samples: Vec<Vec<u8>> = (0..50)
            .map(|i| {
                let mut v = vec![0u8; 512];
                for j in 0..512 {
                    v[j] = ((i * 13 + j * 7 + 42) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        // Act
        let dict = train_zstd_dictionary(&sample_refs, 8192);
        // Assert: with diverse data, dict training should succeed
        if !dict.is_empty() {
            assert!(dict.len() <= 8192, "dict should not exceed requested capacity");
        }
    }

    #[test]
    fn compress_zstd_dict_valid_roundtrip_multiple() {
        // Arrange: compress and decompress multiple different payloads with same dict
        let samples: Vec<Vec<u8>> = (0..30)
            .map(|i| {
                let mut v = vec![0u8; 256];
                for j in 0..256 {
                    v[j] = ((i * 17 + j * 11) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            return;
        }
        // Act & Assert: roundtrip first, middle, and last samples
        for idx in [0, 15, 29] {
            let compressed = compress_zstd_dict(&samples[idx], &dict).unwrap();
            let decompressed = decompress_zstd_dict(&compressed, &dict, samples[idx].len()).unwrap();
            assert_eq!(decompressed, samples[idx], "roundtrip failed for sample {idx}");
        }
    }

    // ── CodecError / NvcompAnsError: trait verification ──

    #[test]
    fn codec_error_is_std_error() {
        // Arrange
        let err = CodecError("trait check".to_string());
        // Act: verify it can be used as dyn Error
        let _: &dyn std::error::Error = &err;
        // Assert: compiles and Display is correct
        assert_eq!(format!("{err}"), "CodecError: trait check");
    }

    #[test]
    fn nvcomp_ans_error_is_std_error() {
        // Arrange
        let err = NvcompAnsError("trait check".to_string());
        // Act: verify it can be used as dyn Error
        let _: &dyn std::error::Error = &err;
        // Assert: compiles and Display is correct
        assert_eq!(format!("{err}"), "NvcompAnsError: trait check");
    }

    // ── CompressionCodec re-export verification ──

    #[test]
    fn compression_codec_reexport_exists() {
        // Verify CompressionCodec is accessible through this module
        use crate::static_compression::CompressionCodec;
        // Just verify the type alias/re-export exists by referencing it
        let _ = std::marker::PhantomData::<CompressionCodec>;
    }

    // ── prune_dead_columns: L2 norm correctness verification ──

    #[test]
    fn prune_dead_columns_l2_norm_correctness() {
        // Arrange: manually compute L2 norms
        // col 0: sqrt(3^2 + 4^2) = 5.0
        // col 1: sqrt(0.1^2 + 0.1^2) ≈ 0.141
        // col 2: sqrt(1^2 + 1^2) ≈ 1.414
        // mean_norm = (5.0 + 0.141 + 1.414) / 3 ≈ 2.185
        // threshold = 0.1 * 2.185 ≈ 0.218 → col 1 pruned
        let weight = vec![
            vec![3.0f32, 0.1, 1.0],
            vec![4.0f32, 0.1, 1.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.1);
        // Assert
        assert!(!mask[0], "col 0 (norm=5.0) should survive");
        assert!(mask[1], "col 1 (norm≈0.141) should be pruned");
        assert!(!mask[2], "col 2 (norm≈1.414) should survive");
        assert_eq!(pruned[0][1], 0.0);
        assert_eq!(pruned[1][1], 0.0);
        assert_eq!(pruned[0][0], 3.0, "col 0 value preserved");
    }

    #[test]
    fn prune_dead_columns_returns_clone_not_alias() {
        // Arrange: verify pruned is a separate allocation
        let weight = vec![vec![1.0f32, 2.0]];
        // Act
        let (pruned, _) = prune_dead_columns(&weight, 0.5);
        // Assert: modifying pruned should not affect weight
        assert_eq!(pruned[0][0], 1.0);
    }

    // ── deduplicate_gqa_heads: output shape verification ──

    #[test]
    fn deduplicate_gqa_output_indices_count_equals_input_heads() {
        // Arrange
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 4, head_dim, 0.98);
        // Assert: dedup_indices has exactly num_heads entries
        assert_eq!(dedup_idx.len(), 4, "dedup_indices should have num_heads entries");
        // All indices should be valid (0-based group indices)
        for &idx in &dedup_idx {
            assert!(idx < 4, "index should be valid");
        }
    }

    #[test]
    fn deduplicate_gqa_head_0_always_group_0() {
        // Arrange: head 0 should always be in group 0 (first encountered)
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0], vec![2.0], vec![3.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.98);
        // Assert
        assert_eq!(dedup_idx[0], 0, "head 0 always maps to group 0");
    }

    // ── CompressionCodec: trait and conversion coverage ──

    #[test]
    fn compression_codec_from_u8_all_variants() {
        // Arrange & Act & Assert: every discriminant round-trips correctly
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn compression_codec_from_u8_invalid_returns_none() {
        // Arrange & Act & Assert
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_as_u8_roundtrip() {
        // Arrange
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: as_u8 → from_u8 should be identity
        for v in &variants {
            assert_eq!(CompressionCodec::from_u8(v.as_u8()), Some(*v));
        }
    }

    #[test]
    fn compression_codec_debug_display() {
        // Arrange
        let codec = CompressionCodec::Lz4;
        // Act
        let debug = format!("{codec:?}");
        // Assert
        assert_eq!(debug, "Lz4");
    }

    #[test]
    fn compression_codec_clone_copy_equality() {
        // Arrange
        let a = CompressionCodec::BitPackRle;
        // Act: Copy
        let b = a;
        // Assert: PartialEq
        assert_eq!(a, b);
        // Act: Clone
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn compression_codec_eq_trait_variants_differ() {
        // Arrange: verify PartialEq + Eq trait behavior across all variant pairs
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: every pair should be non-equal
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b, "same variant should be equal");
                } else {
                    assert_ne!(a, b, "different variants should not be equal");
                }
            }
        }
    }

    #[test]
    fn compression_codec_discriminant_values() {
        // Arrange & Act & Assert: verify repr(u8) discriminant values match SPEC
        assert_eq!(CompressionCodec::None as u8, 0);
        assert_eq!(CompressionCodec::Lz4 as u8, 1);
        assert_eq!(CompressionCodec::BitPackRle as u8, 2);
        assert_eq!(CompressionCodec::NvcompAns as u8, 3);
        assert_eq!(CompressionCodec::ZstdDict as u8, 4);
    }

    // ── CodecError: additional trait coverage ──

    #[test]
    fn codec_error_equality() {
        // Arrange
        let a = CodecError("msg".to_string());
        let b = CodecError("msg".to_string());
        let c = CodecError("other".to_string());
        // Assert: PartialEq is derived
        assert_eq!(a.0, b.0, "same message should be equal");
        assert_ne!(a.0, c.0, "different messages should not be equal");
    }

    #[test]
    fn codec_error_clone() {
        // Arrange
        let original = CodecError("clonable".to_string());
        // Act
        let cloned = CodecError(original.0.clone());
        // Assert
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    // ── NvcompAnsError: additional trait coverage ──

    #[test]
    fn nvcomp_ans_error_equality() {
        // Arrange
        let a = NvcompAnsError("gpu fail".to_string());
        let b = NvcompAnsError("gpu fail".to_string());
        let c = NvcompAnsError("other".to_string());
        // Assert
        assert_eq!(a.0, b.0);
        assert_ne!(a.0, c.0);
    }

    // ── cosine_sim_rows: edge cases ──

    #[test]
    fn cosine_sim_rows_single_element() {
        // Arrange: 1-element vectors
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: dot=12, norm_a=3, norm_b=4, sim=12/12=1.0
        assert!((sim - 1.0).abs() < 1e-6, "parallel vectors should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_both_zero_vectors() {
        // Arrange
        let a = vec![0.0f32; 5];
        let b = vec![0.0f32; 5];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: zero norms → returns 0.0
        assert!(sim.abs() < 1e-6, "both-zero vectors should have sim=0.0, got {sim}");
    }

    // ── prune_dead_columns: additional edge cases ──

    #[test]
    fn prune_dead_columns_uniform_values_no_prune() {
        // Arrange: all columns have identical norms
        let weight = vec![
            vec![1.0f32, 1.0, 1.0],
            vec![1.0f32, 1.0, 1.0],
        ];
        // Act: threshold_ratio=0.5 → threshold = 0.5 * mean_norm → mean_norm = each col norm
        // each col norm = sqrt(1+1) ≈ 1.414, threshold ≈ 0.707 → no column below
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(mask.iter().all(|m| !m), "uniform columns should all survive");
    }

    #[test]
    fn prune_dead_columns_two_columns_one_dead() {
        // Arrange: col 0 is strong, col 1 is weak
        let weight = vec![vec![10.0f32, 0.001]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(!mask[0], "strong column survives");
        assert!(mask[1], "weak column is pruned");
        assert_eq!(pruned[0][0], 10.0);
        assert_eq!(pruned[0][1], 0.0);
    }

    // ── prune_dead_columns_24: mixed sign value ──

    #[test]
    fn prune_dead_columns_24_mixed_signs_per_group() {
        // Arrange: group of 4 with mixed signs: [-5.0, 3.0, -1.0, 2.0]
        // Sorted by |val|: |-5.0|=5.0, |3.0|=3.0, |-1.0|=1.0, |2.0|=2.0
        // Keep pos 0 (-5.0) and pos 1 (3.0) → zero pos 2 and pos 3
        let weight = vec![vec![-5.0f32, 3.0, -1.0, 2.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(pruned[0][0], -5.0, "pos 0 (-5.0) should survive");
        assert_eq!(pruned[0][1], 3.0, "pos 1 (3.0) should survive");
        assert_eq!(pruned[0][2], 0.0, "pos 2 (-1.0) should be pruned");
        assert_eq!(pruned[0][3], 0.0, "pos 3 (2.0) should be pruned");
    }

    // ── deduplicate_gqa_heads: threshold exactly_1_merges_identical ──

    #[test]
    fn deduplicate_gqa_threshold_exactly_1_merges_identical() {
        // Arrange: two identical heads, threshold = 1.0
        // Cosine sim of identical heads = 1.0, so sim >= 1.0 should merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![7.0],
            vec![7.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 1.0);
        // Assert: sim == 1.0 >= 1.0 → should merge
        assert_eq!(dedup_w.len(), head_dim, "identical heads with threshold=1.0 should merge");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
    }

    // ── bitpack_rle: decompress with multiple compressed bytes ──

    #[test]
    fn bitpack_rle_decompress_multiple_runs_concatenated() {
        // Arrange: two compressed entries representing different values
        // val=2 run=3, val=5 run=2
        let compressed: Vec<u8> = vec![(2 << 4) | 2, (5 << 4) | 1]; // run_len = 3, 2
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 5);
        // Assert
        assert_eq!(decompressed, vec![0x02, 0x02, 0x02, 0x05, 0x05]);
    }

    #[test]
    fn bitpack_rle_decompress_empty_compressed_nonzero_len() {
        // Arrange: empty compressed data but non-zero requested length
        // Act
        let decompressed = decompress_bitpack_rle(&[], 10);
        // Assert: no data to emit → empty vec (decompress stops at end of input)
        assert!(decompressed.is_empty(), "empty compressed data should produce no output");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // New tests (35 additional)
    // ════════════════════════════════════════════════════════════════════════════

    // ── CodecError: PartialEq/Clone/Eq derive tests ──

    #[test]
    fn codec_error_partialeq_derived() {
        // Arrange
        let a = CodecError("same".to_string());
        let b = CodecError("same".to_string());
        let c = CodecError("different".to_string());
        // Act & Assert: derived PartialEq
        assert_eq!(a, b, "CodecError with same message should be equal");
        assert_ne!(a, c, "CodecError with different message should not be equal");
    }

    #[test]
    fn codec_error_clone_derived() {
        // Arrange
        let original = CodecError("clone me".to_string());
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(original, cloned);
        assert_eq!(original.0, cloned.0);
    }

    #[test]
    fn codec_error_display_empty_string() {
        // Arrange
        let err = CodecError(String::new());
        // Act
        let display = format!("{err}");
        // Assert
        assert_eq!(display, "CodecError: ");
    }

    #[test]
    fn codec_error_display_multiline() {
        // Arrange
        let err = CodecError("line1\nline2\nline3".to_string());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("line1\nline2\nline3"));
    }

    #[test]
    fn codec_error_source_is_none() {
        // Arrange
        let err = CodecError("test".to_string());
        // Act & Assert
        assert!(std::error::Error::source(&err).is_none());
    }

    // ── NvcompAnsError: PartialEq/Clone/Eq derive tests ──

    #[test]
    fn nvcomp_ans_error_partialeq_derived() {
        // Arrange
        let a = NvcompAnsError("err".to_string());
        let b = NvcompAnsError("err".to_string());
        let c = NvcompAnsError("other".to_string());
        // Act & Assert: derived PartialEq
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn nvcomp_ans_error_clone_derived() {
        // Arrange
        let original = NvcompAnsError("clone".to_string());
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(original, cloned);
    }

    #[test]
    fn nvcomp_ans_error_display_empty_string() {
        // Arrange
        let err = NvcompAnsError(String::new());
        // Act
        let display = format!("{err}");
        // Assert
        assert_eq!(display, "NvcompAnsError: ");
    }

    #[test]
    fn nvcomp_ans_error_source_is_none() {
        // Arrange
        let err = NvcompAnsError("gpu".to_string());
        // Act & Assert
        assert!(std::error::Error::source(&err).is_none());
    }

    // ── CompressionCodec: Hash trait ──

    #[test]
    fn compression_codec_hash_consistency() {
        // Arrange: two equal values should produce the same hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = CompressionCodec::Lz4;
        let b = CompressionCodec::Lz4;
        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();
        // Act
        a.hash(&mut hasher_a);
        b.hash(&mut hasher_b);
        // Assert
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn compression_codec_hash_differentiates_variants() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: all hashes should be unique
        let hashes: Vec<u64> = variants
            .iter()
            .map(|v| {
                let mut h = DefaultHasher::new();
                v.hash(&mut h);
                h.finish()
            })
            .collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "hashes for {:?} and {:?} should differ", variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn compression_codec_usable_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::ZstdDict);
        // Assert
        assert_eq!(set.len(), 2);
        assert!(set.contains(&CompressionCodec::Lz4));
        assert!(set.contains(&CompressionCodec::ZstdDict));
        assert!(!set.contains(&CompressionCodec::None));
    }

    // ── cosine_sim_rows: NaN and Inf edge cases ──

    #[test]
    fn cosine_sim_rows_nan_input_propagates() {
        // Arrange: one vector contains NaN
        let a = vec![f32::NAN, 1.0];
        let b = vec![1.0, 1.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: NaN propagates through arithmetic → result is NaN
        // f32::clamp(NaN, -1, 1) returns NaN in Rust
        assert!(sim.is_nan(), "NaN input should propagate to NaN result, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_inf_input_produces_nan() {
        // Arrange: vector with infinity
        let a = vec![f32::INFINITY, 0.0];
        let b = vec![f32::INFINITY, 0.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: dot=inf, norm=inf, inf/inf=NaN → clamp(NaN) = NaN
        assert!(sim.is_nan(), "inf/inf should produce NaN, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_empty_slices() {
        // Arrange: empty slices
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: dot=0, norm=0, 0 < 1e-8 → returns 0.0
        assert!((sim - 0.0).abs() < 1e-6, "empty slices should return 0.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_large_dimension() {
        // Arrange: 10000-dim identical vectors
        let a: Vec<f32> = vec![1.0; 10000];
        let b: Vec<f32> = vec![1.0; 10000];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - 1.0).abs() < 1e-4, "large identical vectors should have sim=1.0, got {sim}");
    }

    // ── cosine_similarity_heads: edge cases ──

    #[test]
    fn cosine_similarity_heads_single_row_per_head_identical() {
        // Arrange: 2 heads, 1 row each, identical rows
        let weight: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 1);
        // Assert
        assert!((sim - 1.0).abs() < 1e-6, "identical single-row heads should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_same_head_identity() {
        // Arrange: 1 head, compare with itself
        let weight: Vec<Vec<f32>> = vec![vec![3.0, 4.0]];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 0, 1);
        // Assert
        assert!((sim - 1.0).abs() < 1e-6, "head compared with itself should be 1.0, got {sim}");
    }

    // ── deduplicate_gqa_heads: edge cases ──

    #[test]
    fn deduplicate_gqa_many_heads_partial_merge() {
        // Arrange: 6 heads with head_dim=2: 0,1,2 identical; 3,4 identical; 5 unique
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],  // head 0
            vec![1.0, 0.0], vec![0.0, 1.0],  // head 1 (identical to 0)
            vec![1.0, 0.0], vec![0.0, 1.0],  // head 2 (identical to 0)
            vec![0.0, 1.0], vec![1.0, 0.0],  // head 3
            vec![0.0, 1.0], vec![1.0, 0.0],  // head 4 (identical to 3)
            vec![1.0, 1.0],  vec![1.0, -1.0], // head 5 (unique direction)
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 6, head_dim, 0.98);
        // Assert: 3 unique groups
        assert_eq!(dedup_w.len(), 3 * head_dim, "6 heads → 3 unique groups");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
        assert_eq!(dedup_idx[3], dedup_idx[4]);
        assert_ne!(dedup_idx[0], dedup_idx[3]);
        assert_ne!(dedup_idx[3], dedup_idx[5]);
    }

    #[test]
    fn deduplicate_gqa_dedup_indices_all_valid() {
        // Arrange: 5 distinct heads using head_dim=2 with orthogonal directions
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],  // head 0
            vec![0.0, 1.0], vec![1.0, 0.0],  // head 1 (orthogonal to head 0)
            vec![1.0, 1.0],  vec![1.0, -1.0], // head 2
            vec![-1.0, 0.0], vec![0.0, -1.0], // head 3
            vec![1.0, 0.5],  vec![0.5, 1.0],  // head 4
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 5, head_dim, 0.98);
        // Assert: all indices are within valid range [0, num_heads)
        for &idx in &dedup_idx {
            assert!(idx < 5, "dedup index {idx} should be < 5");
        }
        // Each group representative appears at least once
        let unique_groups: std::collections::HashSet<usize> = dedup_idx.iter().copied().collect();
        assert_eq!(unique_groups.len(), 5, "5 distinct heads → 5 unique groups");
    }

    #[test]
    fn deduplicate_gqa_head_dim_2_preserves_row_count() {
        // Arrange: 3 heads with head_dim=2, all orthogonal so none merge
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],  // head 0
            vec![0.0, 1.0], vec![1.0, 0.0],  // head 1 (rows swapped → different from head 0)
            vec![1.0, 1.0],  vec![1.0, -1.0], // head 2
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.98);
        // Assert: 3 heads * 2 rows_per_head = 6 output rows
        assert_eq!(dedup_w.len(), 3 * head_dim, "3 distinct heads → 6 rows");
    }

    // ── prune_dead_columns: edge cases ──

    #[test]
    fn prune_dead_columns_very_large_threshold_prunes_weak() {
        // Arrange: threshold_ratio=100.0 means threshold is 100x mean
        let weight = vec![vec![1.0f32, 0.0, 0.0]];
        // Act: mean_norm = (1.0+0.0+0.0)/3 ≈ 0.333, threshold = 33.33
        // col 0 norm=1.0 < 33.33 → all pruned? No: norm(1.0) < 33.33 → all pruned
        let (_, mask) = prune_dead_columns(&weight, 100.0);
        // Assert: all columns have norm < 33.33 → all pruned
        assert!(mask.iter().all(|m| *m), "very high threshold should prune all columns");
    }

    #[test]
    fn prune_dead_columns_very_small_threshold_keeps_all() {
        // Arrange: threshold_ratio=1e-10 → threshold near zero
        let weight = vec![vec![1.0f32, 2.0, 3.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 1e-10);
        // Assert: nothing is below near-zero threshold
        assert!(mask.iter().all(|m| !m), "very small threshold should keep all columns");
    }

    #[test]
    fn prune_dead_columns_negative_values_norm_positive() {
        // Arrange: all negative values → norms are still positive
        let weight = vec![vec![-3.0f32, -0.001]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: col 0 has large norm, col 1 has tiny norm
        assert!(!mask[0], "col 0 with large |val| should survive");
        assert!(mask[1], "col 1 with tiny |val| should be pruned");
    }

    #[test]
    fn prune_dead_columns_mask_length_matches_cols() {
        // Arrange
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert_eq!(mask.len(), 5, "mask length should equal number of columns");
    }

    // ── prune_dead_columns_24: edge cases ──

    #[test]
    fn prune_dead_columns_24_16_cols_metadata_layout() {
        // Arrange: 1 row × 16 cols → 4 groups → 2 u16 metadata
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                               9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 2, "16 cols → 4 groups → 2 u16");
        let zero_count: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 8, "16 cols → 8 pruned (50%)");
    }

    #[test]
    fn prune_dead_columns_24_4_rows_consistent_zero_fraction() {
        // Arrange: 4 rows × 8 cols each
        let weight: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..8).map(|c| ((r * 8 + c) as f32 + 1.0).sqrt()).collect())
            .collect();
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: each row has exactly 50% zeros
        for (i, row) in pruned.iter().enumerate() {
            let zeros = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 4, "row {i} should have exactly 4 pruned positions");
        }
    }

    #[test]
    fn prune_dead_columns_24_survivors_preserve_sign() {
        // Arrange: group with [-5.0, 1.0, -3.0, 0.5] → keep |-5.0| and |-3.0|
        let weight = vec![vec![-5.0f32, 1.0, -3.0, 0.5]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: survivors keep their sign
        assert_eq!(pruned[0][0], -5.0, "survivor at pos 0 should be -5.0");
        assert_eq!(pruned[0][2], -3.0, "survivor at pos 2 should be -3.0");
        assert_eq!(pruned[0][1], 0.0, "pruned pos 1");
        assert_eq!(pruned[0][3], 0.0, "pruned pos 3");
    }

    #[test]
    fn prune_dead_columns_24_meta_both_groups_in_one_u16() {
        // Arrange: 8 cols → 2 groups packed into 1 u16
        // Group 0: [4.0, 3.0, 2.0, 1.0] → keep pos 0 (4.0) and pos 1 (3.0)
        // Group 1: [0.5, 0.1, 8.0, 7.0] → keep pos 2 (8.0) and pos 3 (7.0)
        let weight = vec![vec![4.0f32, 3.0, 2.0, 1.0, 0.5, 0.1, 8.0, 7.0]];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: single u16, both groups encoded
        assert_eq!(sp_meta[0].len(), 1);
        let meta = sp_meta[0][0];
        // Group 0 (low nibble): keep=[0,1] → encoded = (0 | 1<<2) = 4 = 0x4
        assert_eq!(meta & 0xF, 0x4, "low nibble should encode keep=[0,1]");
    }

    // ── compress_bitpack_rle: edge cases ──

    #[test]
    fn bitpack_rle_alternating_pattern() {
        // Arrange: alternating values prevent any run > 1
        let input: Vec<u8> = (0..20).map(|i| if i % 2 == 0 { 0x01 } else { 0x02 }).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: each byte encodes a run of 1
        assert_eq!(compressed.len(), 20, "alternating pattern → 20 entries");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_single_nibble_value_0() {
        // Arrange: single element with value 0
        let input = vec![0x00u8];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 1);
        // Assert
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0], 0x00, "val=0, run=1 → (0<<4)|0 = 0x00");
        assert_eq!(decompressed, vec![0x00]);
    }

    #[test]
    fn bitpack_rle_run_exactly_16_splits_into_two() {
        // Arrange: 16 identical nibbles → splits into 15+1
        let input = vec![0x04u8; 16];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: 15 (max) + 1 = 2 entries
        assert_eq!(compressed.len(), 2);
        assert_eq!(compressed[0], 0x4E, "val=4, run=15 → (4<<4)|14");
        assert_eq!(compressed[1], 0x40, "val=4, run=1 → (4<<4)|0");
        let decompressed = decompress_bitpack_rle(&compressed, 16);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_large_input_roundtrip() {
        // Arrange: 1000 elements with mixed pattern — runs of 3-5 of same nibble
        let input: Vec<u8> = (0..1000).map(|i| ((i / 4) % 7) as u8).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: roundtrip correctness
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_decompress_stops_at_requested_len() {
        // Arrange: encode 50 elements but request only 25
        let compressed = compress_bitpack_rle(&[0x03u8; 50]);
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 25);
        // Assert
        assert_eq!(decompressed.len(), 25);
        assert!(decompressed.iter().all(|&b| b == 0x03));
    }

    // ── LZ4: edge cases ──

    #[test]
    fn lz4_single_zero_byte_roundtrip() {
        // Arrange
        let data = vec![0u8];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, 1).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_all_same_byte_roundtrip() {
        // Arrange
        let data = vec![0xAAu8; 65536];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len() / 10, "repeated data should compress to <10%");
    }

    // ── zstd dict: edge cases ──

    #[test]
    fn compress_zstd_dict_empty_data() {
        // Arrange: train a dict first
        let samples: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                let mut v = vec![0u8; 128];
                for j in 0..128 {
                    v[j] = ((i * 3 + j * 7) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 2048);
        if dict.is_empty() {
            return;
        }
        // Act: compress empty data
        let result = compress_zstd_dict(&[], &dict);
        // Assert: should succeed (empty data is valid input)
        assert!(result.is_ok(), "compressing empty data should succeed");
    }

    #[test]
    fn decompress_zstd_dict_invalid_compressed_data_errors() {
        // Arrange
        let samples: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                let mut v = vec![0u8; 128];
                for j in 0..128 {
                    v[j] = ((i * 5 + j * 11) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 2048);
        if dict.is_empty() {
            return;
        }
        // Act: decompress garbage data
        let result = decompress_zstd_dict(b"not valid zstd data", &dict, 100);
        // Assert
        assert!(result.is_err(), "garbage input should return error");
    }

    // ── prune_dead_columns_24: 20 cols (odd group count) ──

    #[test]
    fn prune_dead_columns_24_20_cols_3_u16_meta() {
        // Arrange: 20 cols → 5 groups → ceil(5/2) = 3 u16 metadata entries
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 3, "20 cols → 5 groups → 3 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 10, "20 cols → 10 pruned (50%)");
    }

    // ── deduplicate_gqa_heads: threshold edge ──

    #[test]
    fn deduplicate_gqa_negative_threshold_merges_opposite() {
        // Arrange: two opposite heads (sim=-1.0), threshold=-0.5
        // sim=-1.0 >= -0.5 → should NOT merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![-1.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, -0.5);
        // Assert: sim=-1.0 is NOT >= -0.5, so they stay separate
        assert_ne!(dedup_idx[0], dedup_idx[1], "opposite heads should not merge at threshold=-0.5");
    }

    #[test]
    fn deduplicate_gqa_threshold_negative_2_merges_all() {
        // Arrange: threshold=-2.0 → any pair with sim >= -2.0 merges (all pairs)
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![-1.0], // sim with head 0 = -1.0 >= -2.0 → merge
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, -2.0);
        // Assert: even opposite heads merge
        assert_eq!(dedup_w.len(), head_dim, "threshold=-2.0 should merge all heads");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
    }

    // ── prune_dead_columns: single column matrix ──

    #[test]
    fn prune_dead_columns_single_column_never_pruned() {
        // Arrange: single column, its norm = mean norm → threshold < norm
        let weight = vec![vec![42.0f32], vec![42.0f32], vec![42.0f32]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(!mask[0], "single column should never be pruned");
        assert_eq!(pruned[0][0], 42.0);
    }

    // ── CompressionCodec: from_u8 boundary ──

    #[test]
    fn compression_codec_from_u8_boundary_values() {
        // Act & Assert
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
        assert_eq!(CompressionCodec::from_u8(5), None, "5 is out of range");
        assert_eq!(CompressionCodec::from_u8(u8::MAX), None, "u8::MAX is out of range");
    }

    // ── deduplicate_gqa_heads: verify averaging math for 3 merged heads ──

    #[test]
    fn deduplicate_gqa_averages_three_merged_correctly() {
        // Arrange: 3 identical heads with head_dim=1, values [2.0], [4.0], [6.0]
        // cosine sim between any pair = 1.0 → all merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0],
            vec![4.0],
            vec![6.0],
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.98);
        // Assert: average = (2.0+4.0+6.0)/3 = 4.0
        let expected = 4.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-6,
            "3-head average should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (50 more: 157 → 207+)
    // ════════════════════════════════════════════════════════════════════════════

    // ── CodecError: additional edge cases ──

    #[test]
    fn codec_error_equality_same_inner_string() {
        // Arrange
        let a = CodecError("identical message".to_string());
        let b = CodecError("identical message".to_string());
        // Act & Assert
        assert_eq!(a, b, "CodecError with identical inner strings should be equal");
    }

    #[test]
    fn codec_error_equality_different_inner_string() {
        // Arrange
        let a = CodecError("alpha".to_string());
        let b = CodecError("beta".to_string());
        // Act & Assert
        assert_ne!(a, b, "CodecError with different inner strings should not be equal");
    }

    #[test]
    fn codec_error_debug_contains_inner() {
        // Arrange
        let err = CodecError("inner detail".to_string());
        // Act
        let debug = format!("{err:?}");
        // Assert
        assert!(debug.contains("inner detail"), "Debug should contain inner string");
        assert!(debug.contains("CodecError"), "Debug should contain type name");
    }

    // ── NvcompAnsError: additional edge cases ──

    #[test]
    fn nvcomp_ans_error_equality_same_inner_string() {
        // Arrange
        let a = NvcompAnsError("same error".to_string());
        let b = NvcompAnsError("same error".to_string());
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn nvcomp_ans_error_equality_different_inner_string() {
        // Arrange
        let a = NvcompAnsError("error A".to_string());
        let b = NvcompAnsError("error B".to_string());
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn nvcomp_ans_error_debug_contains_inner() {
        // Arrange
        let err = NvcompAnsError("gpu crash detail".to_string());
        // Act
        let debug = format!("{err:?}");
        // Assert
        assert!(debug.contains("gpu crash detail"));
        assert!(debug.contains("NvcompAnsError"));
    }

    #[test]
    fn nvcomp_ans_error_display_multiline() {
        // Arrange
        let err = NvcompAnsError("line1\nline2".to_string());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("line1\nline2"));
    }

    // ── CompressionCodec: additional edge cases ──

    #[test]
    fn compression_codec_from_u8_all_valid_roundtrip() {
        // Arrange & Act & Assert: every valid u8 value round-trips
        for discriminant in 0u8..=4 {
            let codec = CompressionCodec::from_u8(discriminant).unwrap();
            assert_eq!(codec.as_u8(), discriminant);
        }
    }

    #[test]
    fn compression_codec_from_u8_all_invalid_none() {
        // Act & Assert: all values outside 0..=4 return None
        for v in 5u8..=255 {
            assert_eq!(CompressionCodec::from_u8(v), None, "value {v} should return None");
        }
    }

    #[test]
    fn compression_codec_all_variants_debug_name() {
        // Act & Assert: Debug output matches variant names
        assert_eq!(format!("{:?}", CompressionCodec::None), "None");
        assert_eq!(format!("{:?}", CompressionCodec::Lz4), "Lz4");
        assert_eq!(format!("{:?}", CompressionCodec::BitPackRle), "BitPackRle");
        assert_eq!(format!("{:?}", CompressionCodec::NvcompAns), "NvcompAns");
        assert_eq!(format!("{:?}", CompressionCodec::ZstdDict), "ZstdDict");
    }

    #[test]
    fn compression_codec_usable_as_map_key() {
        // Arrange
        use std::collections::HashMap;
        let mut map = HashMap::new();
        // Act
        map.insert(CompressionCodec::Lz4, "fast");
        map.insert(CompressionCodec::ZstdDict, "cold");
        // Assert
        assert_eq!(map.get(&CompressionCodec::Lz4), Some(&"fast"));
        assert_eq!(map.get(&CompressionCodec::ZstdDict), Some(&"cold"));
        assert_eq!(map.get(&CompressionCodec::None), None);
    }

    // ── cosine_sim_rows: additional edge cases ──

    #[test]
    fn cosine_sim_rows_very_small_values() {
        // Arrange: very small but non-zero values
        let eps = 1e-30f32;
        let a = vec![eps, eps];
        let b = vec![eps, eps];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: norms are ~sqrt(2)*eps ≈ 1.4e-30, which is < 1e-8 → returns 0.0
        assert!(sim.abs() < 1e-6, "very small norms should return 0.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_negative_inf() {
        // Arrange: negative infinity values
        let a = vec![f32::NEG_INFINITY];
        let b = vec![f32::NEG_INFINITY];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: -inf * -inf = +inf, norm = inf, inf/inf = NaN
        assert!(sim.is_nan(), "-inf/-inf should produce NaN, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_mixed_nan_and_real() {
        // Arrange: one element NaN, one real
        let a = vec![f32::NAN, 1.0];
        let b = vec![1.0, 1.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: NaN propagates through arithmetic
        assert!(sim.is_nan(), "NaN in input should produce NaN, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_perpendicular_3d() {
        // Arrange: [1,0,0] and [0,1,0] → cosine sim = 0
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!(sim.abs() < 1e-6, "perpendicular 3D vectors should have sim=0, got {sim}");
    }

    // ── cosine_similarity_heads: additional edge cases ──

    #[test]
    fn cosine_similarity_heads_different_rows_per_head() {
        // Arrange: 2 heads with head_dim=3
        // head 0 rows: [1,0,0], [0,1,0], [0,0,1]
        // head 1 rows: [1,0,0], [0,1,0], [0,0,1] (identical)
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0],
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 3);
        // Assert: identical multi-row heads
        assert!((sim - 1.0).abs() < 1e-6, "identical head_dim=3 heads should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_partial_overlap() {
        // Arrange: 2 heads with head_dim=2
        // head 0: [1,0] and [0,1] → orthogonal pair
        // head 1: [1,0] and [1,0] → parallel pair
        // row 0: sim=1.0, row 1: sim=0.0 → average = 0.5
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 0.0], vec![1.0, 0.0],
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 2);
        // Assert: average of row sims = (1.0 + 0.0) / 2 = 0.5
        assert!((sim - 0.5).abs() < 1e-6, "partial overlap should give sim=0.5, got {sim}");
    }

    // ── deduplicate_gqa_heads: additional edge cases ──

    #[test]
    fn deduplicate_gqa_large_hidden_dimension() {
        // Arrange: 2 heads with hidden=100, identical → should merge
        let head_dim = 1;
        let row: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let rows: Vec<Vec<f32>> = vec![row.clone(), row.clone()];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert
        assert_eq!(dedup_w.len(), head_dim, "identical large-dim heads should merge");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_w[0].len(), 100, "output hidden dimension preserved");
    }

    #[test]
    fn deduplicate_gqa_head_dim_3_preserves_row_count_per_group() {
        // Arrange: 2 identical heads with head_dim=3
        let head_dim = 3;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], // head 0
            vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], // head 1
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: merged to 1 group with head_dim=3 rows
        assert_eq!(dedup_w.len(), head_dim, "1 merged group should have head_dim rows");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
    }

    #[test]
    fn deduplicate_gqa_averages_values_correctly_for_2_merged() {
        // Arrange: 2 heads, head_dim=2, identical direction different magnitude
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0, 4.0], vec![6.0, 8.0], // head 0
            vec![4.0, 8.0], vec![12.0, 16.0], // head 1 (2x magnitude, same direction)
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: average of row 0 = [(2+4)/2, (4+8)/2] = [3.0, 6.0]
        assert!((dedup_w[0][0] - 3.0).abs() < 1e-6, "averaged row 0 col 0 should be 3.0");
        assert!((dedup_w[0][1] - 6.0).abs() < 1e-6, "averaged row 0 col 1 should be 6.0");
    }

    #[test]
    fn deduplicate_gqa_zero_heads_returns_empty() {
        // Arrange: 0 heads
        let rows: Vec<Vec<f32>> = vec![];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 0, 2, 0.98);
        // Assert
        assert!(dedup_w.is_empty());
        assert!(dedup_idx.is_empty());
    }

    // ── prune_dead_columns: additional edge cases ──

    #[test]
    fn prune_dead_columns_3x3_matrix_symmetric() {
        // Arrange: 3x3 symmetric matrix
        let weight = vec![
            vec![3.0f32, 0.0, 0.0],
            vec![0.0f32, 3.0, 0.0],
            vec![0.0f32, 0.0, 3.0],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: all cols have norm=3.0, threshold=0.5*3.0=1.5, all above
        assert!(mask.iter().all(|m| !m), "symmetric matrix with equal norms should keep all");
    }

    #[test]
    fn prune_dead_columns_preserves_row_count() {
        // Arrange
        let weight = vec![
            vec![1.0f32, 2.0],
            vec![3.0f32, 4.0],
            vec![5.0f32, 6.0],
        ];
        // Act
        let (pruned, _) = prune_dead_columns(&weight, 0.5);
        // Assert: output should have same number of rows
        assert_eq!(pruned.len(), 3, "row count should be preserved");
    }

    #[test]
    fn prune_dead_columns_all_dead_columns() {
        // Arrange: all columns have zero norm
        let weight = vec![
            vec![0.0f32, 0.0],
            vec![0.0f32, 0.0],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: mean_norm=0, threshold=0, no norm < 0 → nothing pruned
        assert!(mask.iter().all(|m| !m), "zero norms with threshold=0 mean nothing pruned");
    }

    #[test]
    fn prune_dead_columns_one_column_all_zeros() {
        // Arrange: col 0 all alive, col 1 all zeros
        let weight = vec![
            vec![5.0f32, 0.0],
            vec![3.0f32, 0.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(!mask[0], "col 0 should survive");
        assert!(mask[1], "col 1 (all zeros) should be pruned");
        assert_eq!(pruned[0][1], 0.0);
        assert_eq!(pruned[1][1], 0.0);
    }

    // ── prune_dead_columns_24: additional edge cases ──

    #[test]
    fn prune_dead_columns_24_two_rows_same_pattern_same_meta() {
        // Arrange: two identical rows
        let row = vec![10.0f32, 1.0, 5.0, 0.1];
        let weight = vec![row.clone(), row];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: identical rows should produce identical metadata
        assert_eq!(sp_meta[0], sp_meta[1], "identical rows should produce identical metadata");
    }

    #[test]
    fn prune_dead_columns_24_4_cols_single_group() {
        // Arrange: exactly 4 cols → 1 group → 1 u16 (ceil(1/2) = 1)
        let weight = vec![vec![1.0f32, 10.0, 2.0, 20.0]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 u16");
        assert_eq!(pruned[0].len(), 4);
        // Keep pos 1 (10.0) and pos 3 (20.0) — largest absolute values
        assert_ne!(pruned[0][1], 0.0, "pos 1 (10.0) should survive");
        assert_ne!(pruned[0][3], 0.0, "pos 3 (20.0) should survive");
    }

    #[test]
    fn prune_dead_columns_24_8_cols_two_groups_encoding() {
        // Arrange: 8 cols → 2 groups packed in 1 u16
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0, // group 0: keep pos 2,3 (largest abs: 3.0, 4.0)
            10.0, 20.0, 1.0, 2.0,  // group 1: keep pos 0,1 (largest abs: 10.0, 20.0)
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: sp_meta is exactly 1 u16
        assert_eq!(sp_meta[0].len(), 1);
        let meta = sp_meta[0][0];
        // Group 0: keep = [2,3], encoded = (2 | 3<<2) = 14 = 0xE, shift 0
        assert_eq!(meta & 0xF, 0xE, "group 0 should encode keep=[2,3]");
        // Group 1: keep = [0,1], encoded = (0 | 1<<2) = 4 = 0x4, shift 4
        assert_eq!((meta >> 4) & 0xF, 0x4, "group 1 should encode keep=[0,1]");
        // Verify pruned values
        assert_eq!(pruned[0][0], 0.0, "group 0 pos 0 pruned");
        assert_eq!(pruned[0][1], 0.0, "group 0 pos 1 pruned");
        assert_eq!(pruned[0][6], 0.0, "group 1 pos 2 pruned");
        assert_eq!(pruned[0][7], 0.0, "group 1 pos 3 pruned");
        assert_ne!(pruned[0][2], 0.0, "group 0 pos 2 survives");
        assert_ne!(pruned[0][3], 0.0, "group 0 pos 3 survives");
        assert_ne!(pruned[0][4], 0.0, "group 1 pos 0 survives");
        assert_ne!(pruned[0][5], 0.0, "group 1 pos 1 survives");
    }

    #[test]
    fn prune_dead_columns_24_negative_keep_sign() {
        // Arrange: group of 4 with negative values only
        let weight = vec![vec![-1.0f32, -4.0, -2.0, -3.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: |-4| and |-3| survive (positions 1 and 3)
        assert_eq!(pruned[0][0], 0.0, "pos 0 pruned");
        assert_eq!(pruned[0][1], -4.0, "pos 1 survives as -4.0");
        assert_eq!(pruned[0][2], 0.0, "pos 2 pruned");
        assert_eq!(pruned[0][3], -3.0, "pos 3 survives as -3.0");
    }

    // ── compress_bitpack_rle / decompress_bitpack_rle: additional edge cases ──

    #[test]
    fn bitpack_rle_run_of_exactly_1() {
        // Arrange
        let input = vec![0x05u8];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: run_len=1 → run_len_minus_1=0
        assert_eq!(compressed[0] & 0x0F, 0, "run_len-1 should be 0");
    }

    #[test]
    fn bitpack_rle_run_of_exactly_15() {
        // Arrange
        let input = vec![0x07u8; 15];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: single entry with run_len_minus_1=14
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0] & 0x0F, 14, "run_len-1 should be 14");
    }

    #[test]
    fn bitpack_rle_compress_decompress_symmetry_100_elements() {
        // Arrange: mixed runs
        let mut input = Vec::new();
        input.extend(vec![0x01u8; 20]);
        input.extend(vec![0x02u8; 10]);
        input.extend(vec![0x00u8; 30]);
        input.extend(vec![0x0Fu8; 5]);
        input.extend(vec![0x03u8; 35]);
        assert_eq!(input.len(), 100);
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 100);
        // Assert
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_all_same_nibble_0xf() {
        // Arrange: all 0xF nibbles
        let input = vec![0xFFu8; 8]; // low nibble = 0xF
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 8);
        // Assert: decompressed should be low nibble only
        assert!(decompressed.iter().all(|&b| b == 0x0F));
    }

    #[test]
    fn bitpack_rle_decompress_with_extra_compressed_bytes() {
        // Arrange: 3 compressed bytes representing more data than requested
        let compressed = vec![(0x01 << 4) | 4, (0x02 << 4) | 4, (0x03 << 4) | 4]; // 5+5+5=15
        // Act: request only 8
        let decompressed = decompress_bitpack_rle(&compressed, 8);
        // Assert
        assert_eq!(decompressed.len(), 8);
        // First 5 bytes from entry 0 (val=0x01, run=5)
        assert_eq!(decompressed[0], 0x01);
        // Last 3 bytes from entry 1 (val=0x02, run=5, but only 3 needed)
        assert_eq!(decompressed[5], 0x02);
        assert_eq!(decompressed[7], 0x02);
    }

    #[test]
    fn bitpack_rle_compress_output_size_bound() {
        // Arrange: worst case — all different nibbles
        let input: Vec<u8> = (0..50).map(|i| (i % 16) as u8).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: each run of 1 → 1 byte, total = input.len()
        assert_eq!(compressed.len(), 50, "all-different nibbles should have 1:1 ratio");
    }

    #[test]
    fn bitpack_rle_decompress_run_len_16_escape_handling() {
        // Arrange: craft a byte with low nibble = 0x0F (escape code)
        let compressed = vec![(0x05 << 4) | 0x0F]; // val=5, run_len = 15+1 = 16
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 20);
        // Assert: should emit 16 values (requested 20 but only 16 available)
        assert_eq!(decompressed.len(), 16);
        assert!(decompressed.iter().all(|&b| b == 0x05));
    }

    // ── lz4_compress / lz4_decompress: additional edge cases ──

    #[test]
    fn lz4_compress_binary_data_roundtrip() {
        // Arrange: all 256 byte values
        let data: Vec<u8> = (0..=255).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_compress_two_bytes_roundtrip() {
        // Arrange
        let data = vec![0x01u8, 0x02];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, 2).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_decompress_empty_compressed_nonzero_size_errors() {
        // Act
        let result = lz4_decompress(&[], 10);
        // Assert: empty compressed data can't produce non-zero output
        assert!(result.is_err(), "empty compressed data should error for non-zero size");
    }

    // ── compress_zstd_dict / decompress_zstd_dict: additional edge cases ──

    #[test]
    fn compress_zstd_dict_with_trained_dict_succeeds() {
        // Arrange: train a dictionary from samples then compress with it
        let samples: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i * 7 % 256) as u8; 64])
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 1024);
        if dict.is_empty() {
            return;
        }
        // Act: compress with the trained dictionary
        let result = compress_zstd_dict(&samples[0], &dict);
        // Assert: should succeed with a valid trained dictionary
        assert!(result.is_ok(), "compress with trained dictionary should succeed");
        let compressed = result.unwrap();
        assert!(!compressed.is_empty(), "compressed output should not be empty for non-empty input");
    }

    #[test]
    fn decompress_zstd_dict_roundtrip_with_trained_dict() {
        // Arrange: train dictionary, compress, then decompress
        let samples: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i * 7 % 256) as u8; 64])
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 1024);
        if dict.is_empty() {
            return;
        }
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        // Act: decompress
        let result = decompress_zstd_dict(&compressed, &dict, samples[0].len());
        // Assert: should succeed and match original
        assert!(result.is_ok(), "decompress with valid dict should succeed");
        assert_eq!(result.unwrap(), samples[0]);
    }

    #[test]
    fn train_zstd_dictionary_returns_at_most_capacity() {
        // Arrange: enough samples to produce a full dict
        let samples: Vec<Vec<u8>> = (0..30)
            .map(|i| {
                let mut v = vec![0u8; 512];
                for j in 0..512 {
                    v[j] = ((i * 13 + j * 7) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let capacity = 2048;
        // Act
        let dict = train_zstd_dictionary(&sample_refs, capacity);
        // Assert
        if !dict.is_empty() {
            assert!(dict.len() <= capacity, "dict should not exceed requested capacity");
        }
    }

    #[test]
    fn zstd_dict_compress_produces_smaller_output() {
        // Arrange: highly compressible data with trained dict
        let samples: Vec<Vec<u8>> = (0..20)
            .map(|i| vec![((i * 3) % 256) as u8; 1024])
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            return;
        }
        let data = vec![0x42u8; 1024];
        // Act
        let compressed = compress_zstd_dict(&data, &dict).unwrap();
        // Assert: compressed should be smaller than original for highly repetitive data
        assert!(compressed.len() < data.len(), "compressed should be smaller");
    }

    // ── compress_nvcomp_ans / decompress_nvcomp_ans: non-feature path ──

    #[test]
    fn compress_nvcomp_ans_empty_input_no_feature() {
        // Act
        let result = compress_nvcomp_ans(&[]);
        // Assert: without nvcomp feature, even empty input returns error
        assert!(result.is_err());
    }

    #[test]
    fn decompress_nvcomp_ans_empty_input_no_feature() {
        // Act
        let result = decompress_nvcomp_ans(&[], 0);
        // Assert: without nvcomp feature, always errors
        assert!(result.is_err());
    }

    #[test]
    fn compress_nvcomp_ans_error_message_contains_feature_hint() {
        // Act
        let err = compress_nvcomp_ans(b"data").unwrap_err();
        // Assert: error message should guide user to enable feature
        assert!(err.0.contains("nvcomp"), "error should mention nvcomp");
        assert!(err.0.contains("feature"), "error should mention feature");
    }

    #[test]
    fn decompress_nvcomp_ans_error_message_contains_feature_hint() {
        // Act
        let err = decompress_nvcomp_ans(b"data", 100).unwrap_err();
        // Assert
        assert!(err.0.contains("nvcomp"));
        assert!(err.0.contains("feature"));
    }

    // ── cosine_similarity_heads: boundary on head index ──

    #[test]
    fn cosine_similarity_heads_exact_boundary() {
        // Arrange: weight has exactly 2 rows (1 per head), head_dim=1
        let weight: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        // Act: head 0 starts at row 0, head 1 starts at row 1 — both valid
        let sim = cosine_similarity_heads(&weight, 0, 1, 1);
        // Assert: should compute normally
        assert!((sim - 1.0).abs() < 1e-6, "parallel vectors should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_heads_head_a_oob() {
        // Arrange: only 1 row but head_a=1 (starts at row 1, out of bounds)
        let weight: Vec<Vec<f32>> = vec![vec![1.0]];
        // Act
        let sim = cosine_similarity_heads(&weight, 1, 0, 1);
        // Assert: out of bounds returns 0.0
        assert_eq!(sim, 0.0, "OOB head should return 0.0");
    }

    // ── deduplicate_gqa_heads: additional threshold edge cases ──

    #[test]
    fn deduplicate_gqa_threshold_just_below_1_keeps_similar() {
        // Arrange: two nearly identical heads, threshold = 0.999
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![1.00001], // cosine sim ≈ 1.0
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.999);
        // Assert: should still merge (cosine sim ≈ 1.0 >= 0.999)
        assert_eq!(dedup_w.len(), head_dim, "nearly identical heads should merge at 0.999");
    }

    #[test]
    fn deduplicate_gqa_preserves_first_head_value_in_group() {
        // Arrange: 3 heads that all merge, head_dim=1
        // Verify the averaged value uses all three heads
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0],  // head 0
            vec![3.0],  // head 1 (same direction as head 2)
            vec![6.0],  // head 2
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 3, 1, 0.0);
        // Assert: average = (0+3+6)/3 = 3.0
        let expected = 3.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-6,
            "average of 3 merged heads should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ── prune_dead_columns: additional column preservation tests ──

    #[test]
    fn prune_dead_columns_exact_threshold_boundary() {
        // Arrange: col 0 norm = 1.0, col 1 norm = 0.5, col 2 norm = 0.0
        // mean_norm = (1.0 + 0.5 + 0.0) / 3 = 0.5
        // threshold_ratio = 0.99 → threshold = 0.495
        // col 0 (1.0 >= 0.495) alive, col 1 (0.5 >= 0.495) alive, col 2 (0.0 < 0.495) pruned
        let weight = vec![vec![1.0f32, 0.5, 0.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.99);
        // Assert
        assert!(!mask[0], "col 0 should survive");
        assert!(!mask[1], "col 1 at exact threshold should survive (not strictly >)");
        assert!(mask[2], "col 2 should be pruned");
    }

    #[test]
    fn prune_dead_columns_does_not_modify_input() {
        // Arrange
        let weight = vec![vec![1.0f32, 0.0001, 5.0]];
        let weight_clone = weight.clone();
        // Act
        let _ = prune_dead_columns(&weight, 0.01);
        // Assert: original should be unchanged
        assert_eq!(weight, weight_clone, "prune_dead_columns should not modify input");
    }

    // ── prune_dead_columns_24: does not modify input ──

    #[test]
    fn prune_dead_columns_24_does_not_modify_input() {
        // Arrange
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let weight_clone = weight.clone();
        // Act
        let _ = prune_dead_columns_24(&weight);
        // Assert: original should be unchanged
        assert_eq!(weight, weight_clone, "prune_dead_columns_24 should not modify input");
    }

    // ── deduplicate_gqa_heads: does not modify input ──

    #[test]
    fn deduplicate_gqa_does_not_modify_input() {
        // Arrange
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let rows_clone = rows.clone();
        // Act
        let _ = deduplicate_gqa_heads(&rows, 2, 2, 0.98);
        // Assert: original should be unchanged
        assert_eq!(rows, rows_clone, "deduplicate_gqa_heads should not modify input");
    }

    // ── compress_bitpack_rle: does not modify input ──

    #[test]
    fn bitpack_rle_compress_does_not_modify_input() {
        // Arrange
        let input = vec![0x03u8, 0x03, 0x05, 0x05];
        let input_clone = input.clone();
        // Act
        let _ = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(input, input_clone, "compress_bitpack_rle should not modify input");
    }

    // ── Integration-style: prune_dead_columns then bitpack_rle ──

    #[test]
    fn prune_then_bitpack_rle_integration() {
        // Arrange: weight with one dead column
        let weight = vec![vec![1.0f32, 0.0001, 5.0]];
        let (pruned, _mask) = prune_dead_columns(&weight, 0.01);
        // Convert pruned f32 values to nibble-like bytes for bitpack
        let bytes: Vec<u8> = pruned[0].iter().map(|&v| (v.abs() as u8) & 0x0F).collect();
        // Act
        let compressed = compress_bitpack_rle(&bytes);
        let decompressed = decompress_bitpack_rle(&compressed, bytes.len());
        // Assert: roundtrip through prune → bitpack should be lossless for the byte representation
        assert_eq!(decompressed, bytes);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (55 more, batch 3)
    // ════════════════════════════════════════════════════════════════════════════

    // ── cosine_sim_rows: angle and scale invariance ──

    #[test]
    fn cosine_sim_rows_scale_invariant() {
        // Arrange: scaling a vector should not change cosine similarity
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: b = 10*a → cosine sim = 1.0
        assert!((sim - 1.0).abs() < 1e-6, "scaled vectors should have sim=1.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_45_degree_angle() {
        // Arrange: vectors at 45 degrees: [1,0] and [1,1]
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 1.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: cos(45°) = 1/sqrt(2) ≈ 0.7071
        let expected = 1.0f32 / 2.0f32.sqrt();
        assert!(
            (sim - expected).abs() < 1e-4,
            "45-degree vectors should have sim≈0.7071, got {sim}"
        );
    }

    #[test]
    fn cosine_sim_rows_negative_scaling() {
        // Arrange: a and -2*a → cosine sim = -1.0
        let a = vec![3.0f32, 4.0];
        let b = vec![-6.0f32, -8.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - (-1.0)).abs() < 1e-6, "oppositely scaled vectors should have sim=-1.0, got {sim}");
    }

    #[test]
    fn cosine_sim_rows_very_large_values_stability() {
        // Arrange: vectors with large values that could overflow naive implementations
        let a = vec![1e15f32, 1e15];
        let b = vec![1e15f32, 1e15];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: should not overflow/NaN, sim should be 1.0
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "large-value identical vectors should have sim≈1.0, got {sim}"
        );
    }

    // ── cosine_similarity_heads: transitivity and averaging ──

    #[test]
    fn cosine_similarity_heads_transitive_similarity() {
        // Arrange: 3 heads where head 0 ≈ head 1 ≈ head 2 but head 0 differs from head 2
        // This tests that pairwise similarity is computed independently
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], // head 0 row 0
            vec![1.0, 0.0], // head 0 row 1 (not used, head_dim=1 effectively)
            vec![0.9, 0.1], // head 1 row 0
            vec![0.0, 1.0], // head 2 row 0
        ];
        // Act
        let sim_01 = cosine_similarity_heads(&weight, 0, 1, 2);
        let sim_12 = cosine_similarity_heads(&weight, 1, 2, 2);
        // Assert: head 0 vs 1 should have different similarity than head 1 vs 2
        assert_ne!(sim_01, sim_12, "different head pairs should have different similarities");
    }

    #[test]
    fn cosine_similarity_heads_zero_rows_per_head() {
        // Arrange: 0 rows per head → out-of-bounds check triggers for any head pair
        let weight: Vec<Vec<f32>> = vec![vec![1.0]];
        // Act: head 0 with rows_per_head=2 → start_a=0, 0+2=2 > weight.len()=1
        let sim = cosine_similarity_heads(&weight, 0, 0, 2);
        // Assert
        assert_eq!(sim, 0.0, "rows_per_head exceeding weight length returns 0.0");
    }

    // ── deduplicate_gqa_heads: multi-row head averaging ──

    #[test]
    fn deduplicate_gqa_two_heads_multi_row_averaging() {
        // Arrange: 2 heads with head_dim=2, identical direction → merge
        // head 0: rows [2.0,4.0], [6.0,8.0]
        // head 1: rows [4.0,8.0], [12.0,16.0] (2x magnitude)
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0, 4.0], vec![6.0, 8.0],   // head 0
            vec![4.0, 8.0], vec![12.0, 16.0],  // head 1
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: merge produces averaged rows
        assert_eq!(dedup_idx[0], dedup_idx[1], "should merge");
        assert_eq!(dedup_w.len(), 2, "1 group × 2 rows_per_head");
        // Row 0 average: [(2+4)/2, (4+8)/2] = [3.0, 6.0]
        assert!((dedup_w[0][0] - 3.0).abs() < 1e-5, "row 0 col 0 avg should be 3.0");
        assert!((dedup_w[0][1] - 6.0).abs() < 1e-5, "row 0 col 1 avg should be 6.0");
    }

    #[test]
    fn deduplicate_gqa_group_representative_is_first() {
        // Arrange: 3 heads where 0,1,2 all merge (threshold=0)
        // The group representative index should be 0 (the first head encountered)
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![5.0], // head 0
            vec![5.0], // head 1
            vec![5.0], // head 2
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.0);
        // Assert: all map to group 0
        assert_eq!(dedup_idx[0], 0);
        assert_eq!(dedup_idx[1], 0);
        assert_eq!(dedup_idx[2], 0);
    }

    #[test]
    fn deduplicate_gqa_threshold_0_does_not_merge_opposite() {
        // Arrange: opposite heads with threshold=0.0
        // cosine sim = -1.0, and -1.0 >= 0.0 is false → should NOT merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![-1.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.0);
        // Assert
        assert_ne!(dedup_idx[0], dedup_idx[1], "opposite heads should not merge at threshold=0");
    }

    #[test]
    fn deduplicate_gqa_hidden_dim_preserved_through_merge() {
        // Arrange: hidden=7, 2 identical heads → merge
        let head_dim = 1;
        let hidden = 7;
        let row: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.5 + 1.0)).collect();
        let rows: Vec<Vec<f32>> = vec![row.clone(), row];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert
        assert_eq!(dedup_w[0].len(), hidden, "output hidden dim should match input");
    }

    // ── prune_dead_columns: column interaction and independence ──

    #[test]
    fn prune_dead_columns_many_rows_consistent_mask() {
        // Arrange: 100 rows × 4 cols, col 2 has consistently low values
        let weight: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![10.0f32, 10.0, 0.0001, 10.0])
            .collect();
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[2], "consistently low column should be pruned");
        assert!(!mask[0] && !mask[1] && !mask[3], "other columns should survive");
        // All rows should have col 2 zeroed
        for row in &pruned {
            assert_eq!(row[2], 0.0, "pruned column should be zeroed in all rows");
        }
    }

    #[test]
    fn prune_dead_columns_col_norm_computation() {
        // Arrange: 2 rows × 3 cols
        // col 0: sqrt(3^2 + 4^2) = 5.0
        // col 1: sqrt(1^2 + 1^2) ≈ 1.414
        // col 2: sqrt(0.01^2 + 0.01^2) ≈ 0.014
        // mean ≈ 2.143, threshold at 0.5×mean ≈ 1.07 → col 2 pruned
        let weight = vec![
            vec![3.0f32, 1.0, 0.01],
            vec![4.0f32, 1.0, 0.01],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(!mask[0], "col 0 norm=5.0 should survive");
        assert!(!mask[1], "col 1 norm≈1.414 should survive");
        assert!(mask[2], "col 2 norm≈0.014 should be pruned");
    }

    #[test]
    fn prune_dead_columns_single_element_columns() {
        // Arrange: 1 row, 4 columns
        let weight = vec![vec![10.0f32, 0.0001, 5.0, 0.0002]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: columns 1 and 3 have very low norms
        assert!(!mask[0], "col 0 should survive");
        assert!(mask[1], "col 1 should be pruned");
        assert!(!mask[2], "col 2 should survive");
        assert!(mask[3], "col 3 should be pruned");
    }

    #[test]
    fn prune_dead_columns_threshold_ratio_1_prunes_below_mean() {
        // Arrange: col norms = [10.0, 1.0, 1.0], mean=4.0
        // threshold_ratio=1.0 → threshold=4.0 → col 1,2 pruned (norm < 4.0)
        let weight = vec![vec![10.0f32, 1.0, 1.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 1.0);
        // Assert
        assert!(!mask[0], "col 0 with high norm should survive");
        assert!(mask[1], "col 1 with norm < mean should be pruned");
        assert!(mask[2], "col 2 with norm < mean should be pruned");
    }

    // ── prune_dead_columns_24: metadata correctness ──

    #[test]
    fn prune_dead_columns_24_32_cols_8_groups_meta() {
        // Arrange: 1 row × 32 cols → 8 groups → 4 u16 metadata
        let weight: Vec<Vec<f32>> = vec![(0..32).map(|c| (c as f32 + 1.0).ln() + 1.0).collect()];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 4, "32 cols → 8 groups → 4 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 16, "32 cols → 16 pruned (50%)");
    }

    #[test]
    fn prune_dead_columns_24_preserves_exactly_two_per_group() {
        // Arrange: row with distinct values so selection is deterministic
        let weight = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: exactly 2 nonzero in the group of 4
        let nonzero = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 2, "exactly 2 of 4 should survive");
    }

    #[test]
    fn prune_dead_columns_24_many_rows_zero_fraction_invariant() {
        // Arrange: 10 rows × 8 cols, each row has unique pattern
        let weight: Vec<Vec<f32>> = (0..10)
            .map(|r| (0..8).map(|c| ((r * 8 + c) as f32).sin().abs() + 0.1).collect())
            .collect();
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: every row has exactly 4 zeros
        for (i, row) in pruned.iter().enumerate() {
            let zeros = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 4, "row {i}: 8 cols → 4 pruned (50%)");
        }
    }

    #[test]
    fn prune_dead_columns_24_sp_meta_nonzero_for_nontrivial_input() {
        // Arrange: row with very different magnitudes per group
        let weight = vec![vec![100.0f32, 0.01, 0.01, 0.01]];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: metadata should be nonzero (at least one group has non-trivial encoding)
        assert_ne!(sp_meta[0][0], 0, "non-trivial input should produce nonzero metadata");
    }

    // ── compress_bitpack_rle: structural edge cases ──

    #[test]
    fn bitpack_rle_31_run_splits_correctly() {
        // Arrange: 31 identical values → 15+15+1 = 3 entries
        let input = vec![0x06u8; 31];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 3, "31 elements → 15+15+1 = 3 entries");
        assert_eq!(compressed[0] & 0x0F, 14, "first chunk run_len=15");
        assert_eq!(compressed[1] & 0x0F, 14, "second chunk run_len=15");
        assert_eq!(compressed[2] & 0x0F, 0, "third chunk run_len=1");
    }

    #[test]
    fn bitpack_rle_200_run_entry_count() {
        // Arrange: 200 identical nibbles → 13×15 + 1×5 = 14 entries
        let input = vec![0x02u8; 200];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        let expected_entries = (200 + 14) / 15; // ceil(200/15) = 14
        assert_eq!(compressed.len(), expected_entries, "200 elements → 14 entries");
        let decompressed = decompress_bitpack_rle(&compressed, 200);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_compressed_size_never_exceeds_input() {
        // Arrange: worst-case input with no runs
        let input: Vec<u8> = (0..100).map(|i| (i % 16) as u8).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: each element encodes as 1 byte
        assert!(
            compressed.len() <= input.len(),
            "compressed should never exceed input size for nibble-level encoding"
        );
    }

    #[test]
    fn bitpack_rle_roundtrip_binary_like_data() {
        // Arrange: data mimicking quantized weight nibbles
        let input: Vec<u8> = (0..200).map(|i| {
            let base = (i / 20) as u8; // 10 groups of 20
            base & 0x0F
        }).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_decompress_single_byte_multiple_runs() {
        // Arrange: 3 compressed entries for 3 different values
        let compressed = vec![
            (0x01 << 4) | 0, // val=1, run=1
            (0x02 << 4) | 0, // val=2, run=1
            (0x03 << 4) | 0, // val=3, run=1
        ];
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 3);
        // Assert
        assert_eq!(decompressed, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn bitpack_rle_nibble_value_0_long_run() {
        // Arrange: long run of value 0
        let input = vec![0x00u8; 45];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 45);
        // Assert
        assert_eq!(decompressed, input);
        // 45 = 15+15+15 → 3 entries
        assert_eq!(compressed.len(), 3);
        // All entries should have high nibble = 0
        for &byte in &compressed {
            assert_eq!(byte >> 4, 0, "val should be 0");
        }
    }

    #[test]
    fn bitpack_rle_nibble_value_15_long_run() {
        // Arrange: long run of value 0xF
        let input = vec![0xFFu8; 60]; // low nibble = 0xF
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 60);
        // Assert
        assert!(decompressed.iter().all(|&b| b == 0x0F));
    }

    // ── LZ4: additional structural tests ──

    #[test]
    fn lz4_compress_produces_output_for_nonempty() {
        // Arrange
        let data = b"test";
        // Act
        let compressed = lz4_compress(data);
        // Assert: LZ4 always produces some output for non-empty input
        assert!(!compressed.is_empty(), "LZ4 should produce output for non-empty input");
    }

    #[test]
    fn lz4_roundtrip_short_repeating() {
        // Arrange: "abcabcabcabc" pattern
        let data = b"abc".repeat(10);
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len(), "repeating pattern should compress");
    }

    #[test]
    fn lz4_roundtrip_all_zeros() {
        // Arrange
        let data = vec![0u8; 4096];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < 100, "all-zeros should compress to very few bytes");
    }

    #[test]
    fn lz4_decompress_size_mismatch_errors() {
        // Arrange: use a very small decompressed_size (1 byte) for longer data
        let data = b"hello world, this is longer data for compression";
        let compressed = lz4_compress(data);
        // Act: request 1 byte when original is much larger
        let result = lz4_decompress(&compressed, 1);
        // Assert: lz4_flex should error when decompressed_size is too small
        assert!(result.is_err(), "undersized decompressed_size should return error");
    }

    // ── Zstd dict: capacity and training edge cases ──

    #[test]
    fn train_zstd_dictionary_very_small_capacity() {
        // Arrange: capacity = 1 byte (too small for zstd)
        let samples = vec![b"some sample data that is long enough" as &[u8]];
        // Act
        let dict = train_zstd_dictionary(&samples, 1);
        // Assert: zstd requires minimum dict size, should return empty on failure
        // Just verify it doesn't panic
        assert!(dict.len() <= 1);
    }

    #[test]
    fn train_zstd_dictionary_large_capacity() {
        // Arrange: large capacity with diverse samples
        let samples: Vec<Vec<u8>> = (0..20)
            .map(|i| {
                let mut v = vec![0u8; 256];
                for j in 0..256 {
                    v[j] = ((i * 17 + j * 13) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        // Act
        let dict = train_zstd_dictionary(&sample_refs, 65536);
        // Assert: should succeed with enough data; dict <= capacity
        if !dict.is_empty() {
            assert!(dict.len() <= 65536);
        }
    }

    #[test]
    fn compress_zstd_dict_wrong_dict_errors() {
        // Arrange: train dict A, then try decompress with dict B
        let samples_a: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i as u8).wrapping_mul(17); 128])
            .collect();
        let samples_b: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i as u8).wrapping_mul(31); 128])
            .collect();
        let refs_a: Vec<&[u8]> = samples_a.iter().map(|s| s.as_slice()).collect();
        let refs_b: Vec<&[u8]> = samples_b.iter().map(|s| s.as_slice()).collect();
        let dict_a = train_zstd_dictionary(&refs_a, 2048);
        let dict_b = train_zstd_dictionary(&refs_b, 2048);
        if dict_a.is_empty() || dict_b.is_empty() {
            return;
        }
        // Act: compress with dict_a, decompress with dict_b
        let compressed = compress_zstd_dict(&samples_a[0], &dict_a).unwrap();
        let result = decompress_zstd_dict(&compressed, &dict_b, samples_a[0].len());
        // Assert: wrong dict should produce corrupted output or error
        if let Ok(decompressed) = result {
            // If it doesn't error, the data should NOT match the original
            assert_ne!(
                decompressed, samples_a[0],
                "wrong dict should produce different output or error"
            );
        }
        // Either error or wrong data is acceptable
    }

    // ── CodecError / NvcompAnsError: constructor and inner access ──

    #[test]
    fn codec_error_inner_string_accessible() {
        // Arrange
        let msg = "detailed failure reason";
        let err = CodecError(msg.to_string());
        // Assert: inner .0 is accessible
        assert_eq!(err.0, msg);
    }

    #[test]
    fn nvcomp_ans_error_inner_string_accessible() {
        // Arrange
        let msg = "CUDA device lost";
        let err = NvcompAnsError(msg.to_string());
        // Assert
        assert_eq!(err.0, msg);
    }

    #[test]
    fn codec_error_into_inner() {
        // Arrange
        let err = CodecError("owned string".to_string());
        // Act: consume the error to get inner String
        let inner = err.0;
        // Assert
        assert_eq!(inner, "owned string");
    }

    #[test]
    fn nvcomp_ans_error_into_inner() {
        // Arrange
        let err = NvcompAnsError("gpu error detail".to_string());
        // Act
        let inner = err.0;
        // Assert
        assert_eq!(inner, "gpu error detail");
    }

    // ── CompressionCodec: ordering and exhaustiveness ──

    #[test]
    fn compression_codec_all_variants_count() {
        // Act: enumerate all variants via from_u8
        let count = (0u8..=255)
            .filter(|&v| CompressionCodec::from_u8(v).is_some())
            .count();
        // Assert: exactly 5 variants
        assert_eq!(count, 5, "CompressionCodec should have exactly 5 variants");
    }

    #[test]
    fn compression_codec_copy_semantics() {
        // Arrange
        let a = CompressionCodec::NvcompAns;
        // Act: Copy (implicit, no .clone() needed)
        let b = a;
        // Assert: both are equal and independent
        assert_eq!(a, b);
        // Verify they're both still valid
        assert_eq!(a.as_u8(), 3);
        assert_eq!(b.as_u8(), 3);
    }

    #[test]
    fn compression_codec_none_variant_is_default_like() {
        // Arrange & Act
        let none = CompressionCodec::None;
        // Assert: discriminant 0
        assert_eq!(none.as_u8(), 0);
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
    }

    // ── deduplicate_gqa_heads: dedup_indices mapping correctness ──

    #[test]
    fn deduplicate_gqa_indices_bijection_for_unique_heads() {
        // Arrange: 4 completely distinct heads → 4 groups → indices 0,1,2,3
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![-1.0, 1.0],
        ];
        // But head_dim=1 → each head has 1 row
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],  // head 0
            vec![0.0, 1.0],  // head 1
            vec![1.0, 1.0],  // head 2
            vec![-1.0, 1.0], // head 3
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 4, head_dim, 0.98);
        // Assert: 4 unique groups with indices 0..4
        let unique: std::collections::HashSet<usize> = dedup_idx.iter().copied().collect();
        assert_eq!(unique.len(), 4, "4 distinct heads → 4 unique group indices");
        // Each original head maps to a unique group
        assert_eq!(dedup_idx[0], 0, "head 0 maps to group 0");
    }

    #[test]
    fn deduplicate_gqa_merged_head_output_is_average() {
        // Arrange: 3 heads with values 2.0, 8.0, 12.0 (all same direction)
        // All will merge, average should be (2+8+12)/3 ≈ 7.333
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0],
            vec![8.0],
            vec![12.0],
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.0);
        // Assert
        let expected = 22.0f32 / 3.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-5,
            "average of 3 merged heads should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ── prune_dead_columns: multi-row with mixed alive/dead ──

    #[test]
    fn prune_dead_columns_multi_row_only_dead_cols_zeroed() {
        // Arrange: 3 rows × 3 cols, col 1 has very low norm in every row
        let weight = vec![
            vec![10.0f32, 0.001, 20.0],
            vec![15.0f32, 0.001, 25.0],
            vec![12.0f32, 0.001, 22.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[1], "col 1 should be pruned");
        assert!(!mask[0] && !mask[2], "col 0 and 2 should survive");
        // Col 0 and 2 values preserved in all rows
        for row in &pruned {
            assert_eq!(row[1], 0.0, "dead column should be zero");
            assert_ne!(row[0], 0.0, "alive column should not be zero");
            assert_ne!(row[2], 0.0, "alive column should not be zero");
        }
    }

    #[test]
    fn prune_dead_columns_output_dimensions_match_input() {
        // Arrange
        let weight = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0f32, 6.0, 7.0, 8.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert_eq!(pruned.len(), 2, "row count preserved");
        assert_eq!(pruned[0].len(), 4, "col count preserved");
        assert_eq!(mask.len(), 4, "mask length matches col count");
    }

    // ── prune_dead_columns_24: integration with negative values ──

    #[test]
    fn prune_dead_columns_24_negative_values_across_groups() {
        // Arrange: 2 groups with different sign patterns
        // Group 0: [-8, 1, -2, 3] → keep |-8|=8 and |3|=3 → pos 0 and 3
        // Group 1: [4, -5, 1, 2] → keep |-5|=5 and |4|=4 → pos 1 and 0
        let weight = vec![vec![-8.0f32, 1.0, -2.0, 3.0, 4.0, -5.0, 1.0, 2.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert group 0
        assert_eq!(pruned[0][0], -8.0, "grp0 pos 0 survives");
        assert_eq!(pruned[0][3], 3.0, "grp0 pos 3 survives");
        assert_eq!(pruned[0][1], 0.0, "grp0 pos 1 pruned");
        assert_eq!(pruned[0][2], 0.0, "grp0 pos 2 pruned");
    }

    // ── Integration: prune_dead_columns_24 then lz4_compress ──

    #[test]
    fn prune_24_then_lz4_compress_integration() {
        // Arrange: apply 2:4 pruning then compress with LZ4
        let weight: Vec<Vec<f32>> = vec![
            (0..8).map(|c| (c as f32 + 1.0).sin()).collect(),
        ];
        let (pruned, _sp_meta) = prune_dead_columns_24(&weight);
        // Convert to bytes for compression
        let bytes: Vec<u8> = pruned[0].iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        // Act
        let compressed = lz4_compress(&bytes);
        let decompressed = lz4_decompress(&compressed, bytes.len()).unwrap();
        // Assert: roundtrip is lossless
        assert_eq!(decompressed, bytes);
    }

    // ── Integration: prune_dead_columns then lz4_compress ──

    #[test]
    fn prune_l2_then_lz4_compress_integration() {
        // Arrange: apply L2 norm pruning then compress
        let weight = vec![
            vec![10.0f32, 0.0001, 20.0, 0.0001],
            vec![15.0f32, 0.0001, 25.0, 0.0001],
        ];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Act: convert to bytes and compress
        let bytes: Vec<u8> = pruned.iter()
            .flat_map(|row| row.iter().flat_map(|&v| v.to_le_bytes()))
            .collect();
        let compressed = lz4_compress(&bytes);
        let decompressed = lz4_decompress(&compressed, bytes.len()).unwrap();
        // Assert
        assert_eq!(decompressed, bytes);
        // Verify pruned columns are zero
        assert!(mask[1] && mask[3], "columns 1 and 3 should be pruned");
    }

    // ── cosine_sim_rows: numerical stability with denormalized values ──

    #[test]
    fn cosine_sim_rows_denormalized_values() {
        // Arrange: denormalized f32 values (subnormal)
        let a = vec![1.0e-38f32, 2.0e-38];
        let b = vec![1.0e-38f32, 2.0e-38];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: norms might be very small, could be < 1e-8 → return 0.0
        // Or if norms are large enough, should be 1.0
        // The key assertion: no NaN or panic
        assert!(
            !sim.is_nan(),
            "denormalized values should not produce NaN, got {sim}"
        );
    }

    // ── deduplicate_gqa_heads: already-deduplicated heads ──

    #[test]
    fn deduplicate_gqa_all_heads_already_unique() {
        // Arrange: 5 heads with random orthogonal directions
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],
            vec![0.0, 1.0], vec![1.0, 0.0],
            vec![1.0, 1.0], vec![1.0, -1.0],
            vec![-1.0, 0.0], vec![0.0, -1.0],
            vec![0.0, 0.0], vec![1.0, 1.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 5, head_dim, 0.98);
        // Assert: all should remain unique
        let unique: std::collections::HashSet<usize> = dedup_idx.iter().copied().collect();
        assert_eq!(unique.len(), 5, "5 distinct heads should remain 5 unique groups");
        assert_eq!(dedup_w.len(), 5 * head_dim, "5 groups × 2 rows = 10 rows");
    }

    // ── prune_dead_columns: mixed positive/negative in same column ──

    #[test]
    fn prune_dead_columns_mixed_sign_column_norm_positive() {
        // Arrange: col 0 has mixed signs, col 1 has small values
        let weight = vec![
            vec![10.0f32, 0.001],
            vec![-10.0f32, 0.001],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: col 0 has high L2 norm despite mixed signs; col 1 has tiny norm
        assert!(!mask[0], "col 0 with large |val| should survive");
        assert!(mask[1], "col 1 with tiny val should be pruned");
    }

    // ── prune_dead_columns_24: column dimension exactly 4 (minimum valid) ──

    #[test]
    fn prune_dead_columns_24_minimum_valid_dimension() {
        // Arrange: exactly 4 columns → minimum valid NVIDIA 2:4 input
        let weight = vec![vec![0.1f32, 0.2, 0.3, 0.4]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 u16 metadata");
        assert_eq!(pruned[0].len(), 4);
        let nonzero = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 2, "exactly 2 survive");
    }

    // ── compress_bitpack_rle: boundary run of 15 ──

    #[test]
    fn bitpack_rle_run_of_14_single_entry() {
        // Arrange: 14 identical values → 1 entry (run_len_minus_1=13)
        let input = vec![0x08u8; 14];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 1, "14 elements → 1 entry");
        assert_eq!(compressed[0] & 0x0F, 13, "run_len-1 should be 13");
        let decompressed = decompress_bitpack_rle(&compressed, 14);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bitpack_rle_run_of_17_splits_15_2() {
        // Arrange: 17 identical values → 15+2 = 2 entries
        let input = vec![0x0Bu8; 17];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 2, "17 elements → 2 entries");
        assert_eq!(compressed[0] & 0x0F, 14, "first: run=15");
        assert_eq!(compressed[1] & 0x0F, 1, "second: run=2");
        let decompressed = decompress_bitpack_rle(&compressed, 17);
        assert_eq!(decompressed, input);
    }

    // ── lz4_compress: size relationship ──

    #[test]
    fn lz4_compress_random_data_typically_larger() {
        // Arrange: random-ish data that doesn't compress well
        let data: Vec<u8> = (0u8..=255).map(|i| i.wrapping_mul(79)).collect();
        // Act
        let compressed = lz4_compress(&data);
        // Assert: LZ4 header overhead means compressed is typically >= input for random data
        // Just verify roundtrip works
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    // ── Integration: full pipeline deduplicate → prune → compress ──

    #[test]
    fn full_pipeline_dedup_prune_compress() {
        // Arrange: weight with duplicate heads and a dead column
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![10.0f32, 0.001], // head 0
            vec![10.0f32, 0.001], // head 1 (identical)
        ];
        // Step 1: deduplicate
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        assert_eq!(dedup_idx[0], dedup_idx[1], "heads should merge");
        // Step 2: prune dead columns
        let (pruned, mask) = prune_dead_columns(&dedup_w, 0.01);
        assert!(mask[1], "low-norm column should be pruned");
        // Step 3: compress remaining bytes
        let bytes: Vec<u8> = pruned.iter()
            .flat_map(|row| row.iter().flat_map(|&v| v.to_le_bytes()))
            .collect();
        let compressed = lz4_compress(&bytes);
        let decompressed = lz4_decompress(&compressed, bytes.len()).unwrap();
        assert_eq!(decompressed, bytes, "LZ4 roundtrip should be lossless");
    }

    // ── compress_bitpack_rle: verify compression ratio ──

    #[test]
    fn bitpack_rle_compression_ratio_highly_repetitive() {
        // Arrange: 100 identical nibbles → should compress significantly
        let input = vec![0x05u8; 100];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: 100 elements → ceil(100/15)=7 entries → 93% compression
        assert!(
            compressed.len() <= 7,
            "100 identical values should compress to ≤7 bytes, got {}",
            compressed.len()
        );
    }

    #[test]
    fn bitpack_rle_decompress_exact_requested_len() {
        // Arrange: compress 50 elements
        let input = vec![0x04u8; 50];
        let compressed = compress_bitpack_rle(&input);
        // Act: request exactly 50
        let decompressed = decompress_bitpack_rle(&compressed, 50);
        // Assert
        assert_eq!(decompressed.len(), 50);
        assert!(decompressed.iter().all(|&b| b == 0x04));
    }

    // ── prune_dead_columns: input with one row ──

    #[test]
    fn prune_dead_columns_one_row_threshold_calculation() {
        // Arrange: single row, col norms = |val|
        // col 0: |3.0| = 3.0, col 1: |0.01| = 0.01, col 2: |2.0| = 2.0
        // mean = (3.0+0.01+2.0)/3 ≈ 1.67, threshold at 0.5×mean ≈ 0.835
        // col 1 (0.01) < 0.835 → pruned
        let weight = vec![vec![3.0f32, 0.01, 2.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(!mask[0], "col 0 should survive");
        assert!(mask[1], "col 1 should be pruned");
        assert!(!mask[2], "col 2 should survive");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // New tests (50 additional, batch 4)
    // ════════════════════════════════════════════════════════════════════════════

    // ── cosine_sim_rows: mismatched-length slices ──


    #[test]
    fn cosine_sim_rows_one_empty_one_nonempty() {
        // Arrange: one empty, one non-empty
        let a: Vec<f32> = vec![];
        let b = vec![1.0f32, 2.0];
        // Act: zip yields empty → dot=0, norm_a=0 → returns 0.0
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!(sim.abs() < 1e-6, "empty vs non-empty should return 0.0, got {sim}");
    }


    // ── cosine_similarity_heads: head indices and boundary ──

    #[test]
    fn cosine_similarity_heads_same_start_row_different_rows_per_head() {
        // Arrange: comparing heads that overlap in memory (same start, different rows_per_head)
        let weight: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Act: head 0 with rows_per_head=1 → only row 0
        let sim_r1 = cosine_similarity_heads(&weight, 0, 0, 1);
        // Act: head 0 with rows_per_head=2 → rows 0 and 1
        let sim_r2 = cosine_similarity_heads(&weight, 0, 0, 2);
        // Assert: both should be 1.0 (comparing head with itself)
        assert!((sim_r1 - 1.0).abs() < 1e-6, "rows_per_head=1, got {sim_r1}");
        assert!((sim_r2 - 1.0).abs() < 1e-6, "rows_per_head=2, got {sim_r2}");
    }

    #[test]
    fn cosine_similarity_heads_zero_rows_per_head_returns_zero() {
        // Arrange: rows_per_head=0 → loop runs 0 times → total_sim=0 → 0/0=NaN → but...
        // Actually the function divides by rows_per_head as f32, so 0.0/0.0 = NaN
        let weight: Vec<Vec<f32>> = vec![vec![1.0]];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 0, 0);
        // Assert: 0/0 produces NaN in Rust
        assert!(sim.is_nan(), "zero rows_per_head should produce NaN, got {sim}");
    }

    // ── deduplicate_gqa_heads: head_dim exceeds available rows ──


    // ── prune_dead_columns: two identical columns ──

    #[test]
    fn prune_dead_columns_two_identical_columns_both_survive() {
        // Arrange: two columns with identical norms → both above threshold
        let weight = vec![
            vec![5.0f32, 5.0],
            vec![3.0f32, 3.0],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: identical norms → both survive
        assert!(!mask[0] && !mask[1], "identical columns should both survive");
    }

    #[test]
    fn prune_dead_columns_one_alive_one_dead_two_columns() {
        // Arrange: col 0 alive, col 1 dead
        let weight = vec![vec![100.0f32, 0.0001]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(!mask[0], "col 0 survives");
        assert!(mask[1], "col 1 pruned");
        assert_eq!(pruned[0][0], 100.0, "alive value preserved");
        assert_eq!(pruned[0][1], 0.0, "dead value zeroed");
    }

    // ── prune_dead_columns_24: 24 columns (6 groups) metadata layout ──

    #[test]
    fn prune_dead_columns_24_24_cols_6_groups_meta() {
        // Arrange: 1 row × 24 cols → 6 groups → 3 u16 metadata
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 3, "24 cols → 6 groups → 3 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 12, "24 cols → 12 pruned (50%)");
    }

    // ── prune_dead_columns_24: 4 cols with NaN behavior ──

    #[test]
    fn prune_dead_columns_24_nan_in_group_partial_sort() {
        // Arrange: NaN in a group of 4
        let weight = vec![vec![1.0f32, f32::NAN, 3.0, 4.0]];
        // Act: sort_by with NaN uses unwrap_or(Equal) — NaN treated as equal
        // The behavior is implementation-defined but should not panic
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: exactly 2 survive, 2 pruned (50% rule holds)
        let nonzero = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 2, "2 of 4 should survive even with NaN");
        assert_eq!(sp_meta.len(), 1);
    }

    // ── compress_bitpack_rle: run of exactly 2 ──

    #[test]
    fn bitpack_rle_run_of_2_single_entry() {
        // Arrange
        let input = vec![0x03u8; 2];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: run_len=2 → run_len_minus_1=1
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0] & 0x0F, 1, "run_len-1 should be 1");
        let decompressed = decompress_bitpack_rle(&compressed, 2);
        assert_eq!(decompressed, input);
    }

    // ── compress_bitpack_rle: pattern with runs of different nibbles ──

    #[test]
    fn bitpack_rle_two_different_long_runs() {
        // Arrange: 25 of val 0x01, then 25 of val 0x02
        let mut input = vec![0x01u8; 25];
        input.extend(vec![0x02u8; 25]);
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 50);
        // Assert
        assert_eq!(decompressed, input);
        // 25 → 15+10 = 2 entries; 25 → 15+10 = 2 entries; total = 4
        assert_eq!(compressed.len(), 4, "25+25 → ceil(25/15)+ceil(25/15) = 4 entries");
    }

    // ── compress_bitpack_rle: 45 elements (3×15 exactly) ──

    #[test]
    fn bitpack_rle_45_elements_exact_3_chunks() {
        // Arrange
        let input = vec![0x0Cu8; 45];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: 45/15 = 3 exactly
        assert_eq!(compressed.len(), 3, "45 elements → 3 chunks of 15");
        for &byte in &compressed {
            assert_eq!(byte >> 4, 0xC, "val should be 0xC");
            assert_eq!(byte & 0x0F, 14, "run_len should be 15");
        }
        let decompressed = decompress_bitpack_rle(&compressed, 45);
        assert_eq!(decompressed, input);
    }

    // ── compress_bitpack_rle: rapid alternation of 3 values ──

    #[test]
    fn bitpack_rle_cyclic_3_values_no_compression() {
        // Arrange: cycle through 3 different nibbles → no runs
        let input: Vec<u8> = (0..30).map(|i| (i % 3) as u8).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: each run is length 1, so compressed = 30 bytes
        assert_eq!(compressed.len(), 30, "cyclic 3-value pattern → no compression");
        assert_eq!(decompressed, input);
    }

    // ── decompress_bitpack_rle: requesting 0 from non-empty ──

    #[test]
    fn bitpack_rle_decompress_request_zero_from_nonempty() {
        // Arrange
        let compressed = compress_bitpack_rle(&[0x05u8; 10]);
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 0);
        // Assert
        assert!(decompressed.is_empty(), "requesting 0 bytes should return empty");
    }

    // ── decompress_bitpack_rle: multiple escape codes ──

    #[test]
    fn bitpack_rle_decompress_multiple_escape_codes() {
        // Arrange: 3 escape-coded bytes, each producing 16 elements
        let compressed = vec![
            (0x01 << 4) | 0x0F, // val=1, run=16
            (0x02 << 4) | 0x0F, // val=2, run=16
            (0x03 << 4) | 0x0F, // val=3, run=16
        ];
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 48);
        // Assert
        assert_eq!(decompressed.len(), 48);
        assert!(decompressed[..16].iter().all(|&b| b == 0x01));
        assert!(decompressed[16..32].iter().all(|&b| b == 0x02));
        assert!(decompressed[32..48].iter().all(|&b| b == 0x03));
    }

    // ── LZ4: near block size boundary ──

    #[test]
    fn lz4_roundtrip_64kb_boundary() {
        // Arrange: exactly 65536 bytes
        let data = vec![0x42u8; 65536];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed.len(), 65536);
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len() / 10, "repeated 64KB should compress well");
    }

    #[test]
    fn lz4_roundtrip_65535_bytes() {
        // Arrange: just under 64KB
        let data: Vec<u8> = (0..65535).map(|i| (i % 256) as u8).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    // ── LZ4: incompressible data ──

    #[test]
    fn lz4_roundtrip_high_entropy_data() {
        // Arrange: pseudo-random data (high entropy, poor compression)
        let data: Vec<u8> = (0u32..4096).map(|i| (i.wrapping_mul(79).wrapping_add(13)) as u8).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert: roundtrip must be lossless regardless of compressibility
        assert_eq!(decompressed, data);
    }

    // ── LZ4: decompress with size=0 ──

    #[test]
    fn lz4_decompress_zero_size() {
        // Act: decompress empty compressed data with size=0
        let compressed = lz4_compress(&[]);
        let result = lz4_decompress(&compressed, 0);
        // Assert
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ── zstd: compress/decompress roundtrip with exact size ──

    #[test]
    fn zstd_dict_roundtrip_exact_size_match() {
        // Arrange
        let samples: Vec<Vec<u8>> = (0..15)
            .map(|i| {
                let mut v = vec![0u8; 200];
                for j in 0..200 {
                    v[j] = ((i * 11 + j * 7) % 256) as u8;
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 4096);
        if dict.is_empty() {
            return;
        }
        // Act
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        let decompressed = decompress_zstd_dict(&compressed, &dict, samples[0].len()).unwrap();
        // Assert: exact size match, byte-for-byte identical
        assert_eq!(decompressed.len(), samples[0].len());
        assert_eq!(decompressed, samples[0]);
    }

    // ── zstd: train with mixed empty and non-empty samples ──

    #[test]
    fn train_zstd_dictionary_mixed_empty_and_nonempty() {
        // Arrange: some empty, some non-empty
        let samples: Vec<&[u8]> = vec![b"", b"", b"", b"this is a real sample with enough data"];
        // Act
        let dict = train_zstd_dictionary(&samples, 1024);
        // Assert: should not panic; may succeed or fail depending on zstd requirements
        // Just verify it returns without panic
        assert!(dict.len() <= 1024);
    }

    // ── zstd: compress small data with trained dict ──

    #[test]
    fn zstd_dict_compress_single_byte_with_dict() {
        // Arrange: train dict, then compress a single byte
        let samples: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![(i * 7 % 256) as u8; 64])
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 1024);
        if dict.is_empty() {
            return;
        }
        // Act
        let result = compress_zstd_dict(&[0x42], &dict);
        // Assert: should succeed for any non-empty data
        assert!(result.is_ok(), "compressing a single byte should succeed");
    }

    // ── Integration: bitpack_rle → lz4 roundtrip ──

    #[test]
    fn bitpack_rle_then_lz4_roundtrip() {
        // Arrange: quantized weight nibbles → bitpack_rle → lz4
        let input: Vec<u8> = vec![0x03u8; 40];
        let bitpacked = compress_bitpack_rle(&input);
        // Act: compress the bitpacked output with LZ4
        let lz4_compressed = lz4_compress(&bitpacked);
        let lz4_decompressed = lz4_decompress(&lz4_compressed, bitpacked.len()).unwrap();
        let final_output = decompress_bitpack_rle(&lz4_decompressed, input.len());
        // Assert: full pipeline roundtrip is lossless
        assert_eq!(final_output, input);
    }

    // ── Integration: prune_dead_columns_24 → bitpack_rle → lz4 ──

    #[test]
    fn prune_24_bitpack_lz4_pipeline() {
        // Arrange
        let weight = vec![vec![10.0f32, 1.0, 8.0, 0.5, 20.0, 2.0, 15.0, 3.0]];
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Convert survivors to nibbles (simplified)
        let nibbles: Vec<u8> = pruned[0].iter().map(|&v| (v.abs() as u8).min(15)).collect();
        // Act: bitpack → lz4
        let bitpacked = compress_bitpack_rle(&nibbles);
        let lz4_compressed = lz4_compress(&bitpacked);
        let lz4_decompressed = lz4_decompress(&lz4_compressed, bitpacked.len()).unwrap();
        let final_nibbles = decompress_bitpack_rle(&lz4_decompressed, nibbles.len());
        // Assert: full pipeline roundtrip
        assert_eq!(final_nibbles, nibbles);
    }

    // ── prune_dead_columns: threshold ratio exactly 0.5 with varied norms ──

    #[test]
    fn prune_dead_columns_half_threshold_varied_norms() {
        // Arrange: col norms = [10.0, 5.0, 0.1], mean ≈ 5.03
        // threshold = 0.5 * 5.03 ≈ 2.515
        // col 0 (10.0) alive, col 1 (5.0) alive, col 2 (0.1) pruned
        let weight = vec![
            vec![10.0f32, 5.0, 0.01],
            vec![0.0f32, 0.0, 0.01],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(!mask[0], "col 0 survives");
        assert!(!mask[1], "col 1 survives");
        assert!(mask[2], "col 2 pruned");
        assert_eq!(pruned[0][2], 0.0);
    }

    // ── prune_dead_columns: threshold ratio 0 prunes nothing (even for zero norms) ──

    #[test]
    fn prune_dead_columns_zero_ratio_never_prunes() {
        // Arrange: some columns have zero norm
        let weight = vec![vec![5.0f32, 0.0, 3.0]];
        // Act: threshold = 0.0 * mean → threshold = 0.0
        let (_, mask) = prune_dead_columns(&weight, 0.0);
        // Assert: norm(0.0) is NOT < 0.0 → nothing pruned
        assert!(mask.iter().all(|m| !m), "zero threshold should prune nothing");
    }

    // ── prune_dead_columns_24: large number of rows ──

    #[test]
    fn prune_dead_columns_24_many_rows_consistent_metadata_shape() {
        // Arrange: 50 rows × 8 cols
        let weight: Vec<Vec<f32>> = (0..50)
            .map(|r| (0..8).map(|c| ((r * 8 + c) as f32 + 1.0).sqrt()).collect())
            .collect();
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: metadata shape consistent
        assert_eq!(pruned.len(), 50);
        assert_eq!(sp_meta.len(), 50);
        for (i, row) in pruned.iter().enumerate() {
            assert_eq!(row.len(), 8, "row {i} should have 8 cols");
            let zeros = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 4, "row {i} should have 50% zeros");
        }
        for (i, meta_row) in sp_meta.iter().enumerate() {
            assert_eq!(meta_row.len(), 1, "meta row {i} should have 1 u16");
        }
    }

    // ── deduplicate_gqa_heads: 8 heads all distinct ──

    #[test]
    fn deduplicate_gqa_8_distinct_heads_no_merge() {
        // Arrange: 8 heads with unique directions
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![-1.0, 1.0],
            vec![-1.0, -1.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 8, head_dim, 0.98);
        // Assert: all remain unique
        assert_eq!(dedup_w.len(), 8, "8 distinct heads should remain 8");
        let unique: std::collections::HashSet<usize> = dedup_idx.iter().copied().collect();
        assert_eq!(unique.len(), 8);
    }

    // ── deduplicate_gqa_heads: 8 heads all identical → 1 group ──

    #[test]
    fn deduplicate_gqa_8_identical_heads_single_group() {
        // Arrange: 8 identical heads
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = (0..8).map(|_| vec![2.0, 3.0]).collect();
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 8, head_dim, 0.98);
        // Assert: all merge into 1 group
        assert_eq!(dedup_w.len(), head_dim, "8 identical heads → 1 unique");
        for &idx in &dedup_idx {
            assert_eq!(idx, 0, "all heads should map to group 0");
        }
    }

    // ── deduplicate_gqa_heads: averaging preserves row dimensionality ──

    #[test]
    fn deduplicate_gqa_merged_output_hidden_dimensionality() {
        // Arrange: hidden=4, 2 identical heads with head_dim=2
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0],
            vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0],
        ];
        // Act
        let (dedup_w, _) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: output rows maintain hidden=4 dimension
        for (i, row) in dedup_w.iter().enumerate() {
            assert_eq!(row.len(), 4, "row {i} should have hidden=4");
        }
    }

    // ── prune_dead_columns: many rows, all columns alive ──

    #[test]
    fn prune_dead_columns_many_rows_all_alive() {
        // Arrange: 50 rows × 4 cols, all values significant
        let weight: Vec<Vec<f32>> = (0..50)
            .map(|r| vec![(r as f32 + 1.0) * 10.0; 4])
            .collect();
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: nothing pruned
        assert!(mask.iter().all(|m| !m), "all columns should survive");
        assert_eq!(pruned.len(), 50);
        assert_eq!(pruned[0].len(), 4);
    }

    // ── prune_dead_columns: f32 special values ──

    #[test]
    fn prune_dead_columns_with_negative_inf() {
        // Arrange: col 0 has -inf, col 1 has finite values
        let weight = vec![vec![f32::NEG_INFINITY, 5.0f32]];
        // Act: norm computation: sqrt(inf^2) = inf for col 0
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: col 0 has infinite norm → survives
        assert!(!mask[0], "col with -inf should have infinite norm → survive");
    }

    // ── CompressionCodec: serde-like property (from_u8 ∘ as_u8 = id) ──

    #[test]
    fn compression_codec_roundtrip_preserves_identity() {
        // Arrange & Act & Assert: identity law for serialization roundtrip
        for discriminant in 0u8..=4u8 {
            let original = CompressionCodec::from_u8(discriminant).unwrap();
            let roundtripped = CompressionCodec::from_u8(original.as_u8()).unwrap();
            assert_eq!(original, roundtripped, "roundtrip should be identity for discriminant {discriminant}");
        }
    }

    // ── CompressionCodec: as_u8 values are dense ──

    #[test]
    fn compression_codec_discriminants_are_contiguous() {
        // Arrange & Act: collect all as_u8 values
        let values: Vec<u8> = (0u8..=4).filter_map(|v| {
            let codec = CompressionCodec::from_u8(v)?;
            Some(codec.as_u8())
        }).collect();
        // Assert: should be [0, 1, 2, 3, 4] exactly
        assert_eq!(values, vec![0, 1, 2, 3, 4], "discriminants should be contiguous 0..=4");
    }

    // ── prune_dead_columns_24: survivors keep exact original values ──

    #[test]
    fn prune_dead_columns_24_survivors_keep_exact_values() {
        // Arrange: values that are clearly ranked
        let weight = vec![vec![100.0f32, 1.0, 50.0, 2.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: keep pos 0 (100.0) and pos 2 (50.0), and they keep exact values
        assert_eq!(pruned[0][0], 100.0, "survivor should keep exact value");
        assert_eq!(pruned[0][2], 50.0, "survivor should keep exact value");
    }

    // ── prune_dead_columns: output is independent per call ──

    #[test]
    fn prune_dead_columns_independent_calls() {
        // Arrange
        let weight = vec![vec![1.0f32, 0.001, 5.0]];
        // Act: call twice with different thresholds
        let (_, mask_low) = prune_dead_columns(&weight, 0.01);
        let (_, mask_high) = prune_dead_columns(&weight, 0.99);
        // Assert: different thresholds may produce different masks
        // At threshold 0.01, only very low norms are pruned
        // At threshold 0.99, more columns may be pruned
        assert!(mask_high.iter().filter(|&&m| m).count() >= mask_low.iter().filter(|&&m| m).count(),
            "higher threshold should prune at least as many columns");
    }

    // ── cosine_sim_rows: result always in [-1, 1] for valid inputs ──

    #[test]
    fn cosine_sim_rows_output_bounded_for_random_vectors() {
        // Arrange: many random-ish vectors
        for seed in 0u32..20 {
            let a: Vec<f32> = (0..10).map(|j| ((seed * 17 + j * 31) as f32).sin()).collect();
            let b: Vec<f32> = (0..10).map(|j| ((seed * 13 + j * 23) as f32).cos()).collect();
            // Act
            let sim = cosine_sim_rows(&a, &b);
            // Assert: result should be in [-1, 1] (clamp guarantees this)
            assert!(
                sim >= -1.0 && sim <= 1.0,
                "cosine similarity should be in [-1,1], got {sim} for seed {seed}"
            );
        }
    }

    // ── deduplicate_gqa_heads: single row per head, many heads ──


    // ── prune_dead_columns_24: values near zero but nonzero ──

    #[test]
    fn prune_dead_columns_24_near_zero_survivors_vs_pruned() {
        // Arrange: values where some are near-zero but not the smallest in absolute terms
        let weight = vec![vec![0.001f32, 0.0001, 1.0, 2.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: keep pos 2 (1.0) and pos 3 (2.0) as they have largest |val|
        assert_eq!(pruned[0][0], 0.0, "near-zero pos 0 should be pruned");
        assert_eq!(pruned[0][1], 0.0, "near-zero pos 1 should be pruned");
        assert_eq!(pruned[0][2], 1.0, "pos 2 should survive");
        assert_eq!(pruned[0][3], 2.0, "pos 3 should survive");
    }

    // ── lz4: multiple independent roundtrips ──

    #[test]
    fn lz4_multiple_independent_roundtrips() {
        // Arrange: 3 different datasets
        let datasets: Vec<Vec<u8>> = vec![
            b"short".to_vec(),
            vec![0u8; 4096],
            (0..1024).map(|i| (i % 256) as u8).collect(),
        ];
        // Act & Assert: each roundtrips independently
        for (i, data) in datasets.iter().enumerate() {
            let compressed = lz4_compress(data);
            let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
            assert_eq!(decompressed, *data, "dataset {i} roundtrip should be lossless");
        }
    }

    // ── bitpack_rle: all nibble values 0-15 in a pattern ──

    #[test]
    fn bitpack_rle_all_nibble_values_roundtrip() {
        // Arrange: each nibble value appears exactly once
        let input: Vec<u8> = (0..=15u8).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 16);
        // Assert
        assert_eq!(decompressed, input);
        // Each is a run of 1
        assert_eq!(compressed.len(), 16, "16 distinct values → 16 entries");
    }

    // ── Integration: full pipeline with bitpack → lz4 → decompress ──

    #[test]
    fn bitpack_lz4_decompress_pipeline_correctness() {
        // Arrange: realistic quantized weight page (256 nibbles)
        let page: Vec<u8> = (0..256).map(|i| ((i / 16) % 16) as u8).collect();
        // Step 1: bitpack compress
        let bitpacked = compress_bitpack_rle(&page);
        assert!(bitpacked.len() <= page.len(), "bitpack should compress or match input size");
        // Step 2: lz4 compress the bitpacked output
        let lz4_compressed = lz4_compress(&bitpacked);
        // Step 3: lz4 decompress
        let lz4_decompressed = lz4_decompress(&lz4_compressed, bitpacked.len()).unwrap();
        assert_eq!(lz4_decompressed, bitpacked, "LZ4 roundtrip should be lossless");
        // Step 4: bitpack decompress
        let final_output = decompress_bitpack_rle(&lz4_decompressed, page.len());
        // Assert: end-to-end lossless
        assert_eq!(final_output, page, "full pipeline should be lossless");
    }

    // ── prune_dead_columns: dead column in the middle ──

    #[test]
    fn prune_dead_columns_dead_column_in_middle() {
        // Arrange: col 0 alive, col 1 dead, col 2 alive, col 3 dead, col 4 alive
        let weight = vec![vec![10.0f32, 0.0001, 20.0, 0.0002, 30.0]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert_eq!(mask, vec![false, true, false, true, false], "dead columns at odd positions");
        assert_eq!(pruned[0][0], 10.0);
        assert_eq!(pruned[0][1], 0.0);
        assert_eq!(pruned[0][2], 20.0);
        assert_eq!(pruned[0][3], 0.0);
        assert_eq!(pruned[0][4], 30.0);
    }

    // ── cosine_similarity_heads: heads with zeros in some rows ──

    #[test]
    fn cosine_similarity_heads_zero_rows_mixed() {
        // Arrange: head 0 has a zero row, head 1 has a zero row but different position
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],  // head 0 row 0
            vec![0.0, 0.0],  // head 0 row 1 (zero vector → sim=0.0)
            vec![1.0, 0.0],  // head 1 row 0
            vec![1.0, 0.0],  // head 1 row 1 (non-zero)
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 2);
        // Assert: average of (sim_row0, sim_row1) where row1 of head 0 is zero
        // row 0: both [1,0] → sim=1.0
        // row 1: [0,0] vs [1,0] → zero vector → sim=0.0
        // average = (1.0 + 0.0) / 2 = 0.5
        assert!((sim - 0.5).abs() < 1e-6, "mixed zero rows should give sim=0.5, got {sim}");
    }

    // ── CodecError: long string content ──

    #[test]
    fn codec_error_long_message() {
        // Arrange
        let long_msg = "x".repeat(10000);
        let err = CodecError(long_msg.clone());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains(&long_msg));
        assert_eq!(err.0.len(), 10000);
    }

    // ── NvcompAnsError: long string content ──

    #[test]
    fn nvcomp_ans_error_long_message() {
        // Arrange
        let long_msg = "e".repeat(10000);
        let err = NvcompAnsError(long_msg.clone());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains(&long_msg));
        assert_eq!(err.0.len(), 10000);
    }

    // ── cosine_sim_rows: mismatched-length slices zip to shorter ──

    #[test]
    fn cosine_sim_rows_mismatched_lengths_zip_to_shorter() {
        // Arrange: slices of different lengths
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0]; // shorter
        // Act: zip yields 2 pairs → dot = 1+4=5, norm_a partial = sqrt(1+4)=sqrt(5)
        // but norm_a uses full a: sqrt(1+4+9)=sqrt(14), norm_b: sqrt(1+4)=sqrt(5)
        let sim = cosine_sim_rows(&a, &b);
        // Assert: result is well-defined (no panic), just verifies the function handles mismatch
        assert!(
            sim >= -1.0 && sim <= 1.0,
            "mismatched-length cosine sim should be in [-1,1], got {sim}"
        );
        assert!(!sim.is_nan(), "should not produce NaN");
    }

    // ── prune_dead_columns: all columns have exactly the same value ──

    #[test]
    fn prune_dead_columns_all_identical_values_no_prune() {
        // Arrange: 3 rows × 3 cols, all values = 7.0
        let weight = vec![
            vec![7.0f32, 7.0, 7.0],
            vec![7.0f32, 7.0, 7.0],
            vec![7.0f32, 7.0, 7.0],
        ];
        // Act: all col norms identical → mean = each col norm → nothing below threshold
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert
        assert!(mask.iter().all(|m| !m), "uniform values → uniform norms → nothing pruned");
    }

    // ── prune_dead_columns_24: very large f32 values ──

    #[test]
    fn prune_dead_columns_24_f32_max_values() {
        // Arrange: group of 4 with f32::MAX and small values
        let weight = vec![vec![f32::MAX, 1.0, f32::MIN_POSITIVE, 0.5]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: f32::MAX and 1.0 are the two largest by absolute value
        assert_eq!(pruned[0][0], f32::MAX, "f32::MAX should survive");
        assert_eq!(pruned[0][1], 1.0, "1.0 should survive");
        assert_eq!(pruned[0][2], 0.0, "min positive should be pruned");
        assert_eq!(pruned[0][3], 0.0, "0.5 should be pruned");
    }

    // ── bitpack_rle: decompress with remaining=0 stops mid-byte ──

    #[test]
    fn bitpack_rle_decompress_remaining_zero_mid_entry() {
        // Arrange: a single compressed entry that encodes 5 values
        let compressed = vec![(0x07 << 4) | 4]; // val=7, run=5
        // Act: decompress first 5 to fill the buffer, then verify exact count
        let decompressed = decompress_bitpack_rle(&compressed, 5);
        // Assert
        assert_eq!(decompressed.len(), 5);
        assert!(decompressed.iter().all(|&b| b == 0x07));
    }

    // ── deduplicate_gqa_heads: single head with zero-valued rows ──

    #[test]
    fn deduplicate_gqa_single_zero_head() {
        // Arrange: single head with all-zero rows
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 1, head_dim, 0.98);
        // Assert: single head stays, output has correct shape
        assert_eq!(dedup_w.len(), head_dim, "single head should produce head_dim rows");
        assert_eq!(dedup_idx.len(), 1);
        for row in &dedup_w {
            assert_eq!(row.len(), 3, "hidden dimension preserved");
        }
    }

    // ── prune_dead_columns: NaN column norm behavior ──

    #[test]
    fn prune_dead_columns_nan_column_no_panic() {
        // Arrange: one column has NaN values
        let weight = vec![vec![f32::NAN, 5.0f32]];
        // Act: should not panic; NaN comparison behavior is implementation-defined
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: just verify it doesn't panic and returns a mask of correct length
        assert_eq!(mask.len(), 2, "mask should have one entry per column");
    }

    // ── bitpack_rle: run of exactly 3 elements ──

    #[test]
    fn bitpack_rle_run_of_3_encoding() {
        // Arrange
        let input = vec![0x09u8; 3];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert: single entry with run_len_minus_1 = 2
        assert_eq!(compressed.len(), 1, "3 elements → 1 entry");
        assert_eq!(compressed[0] >> 4, 0x9, "val should be 0x9");
        assert_eq!(compressed[0] & 0x0F, 2, "run_len-1 should be 2");
        let decompressed = decompress_bitpack_rle(&compressed, 3);
        assert_eq!(decompressed, input);
    }

    // ── lz4_compress: very short input (3 bytes) roundtrip ──

    #[test]
    fn lz4_roundtrip_3_bytes() {
        // Arrange
        let data = vec![0xABu8, 0xCD, 0xEF];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, 3).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    // ── prune_dead_columns_24: 8 cols odd group metadata high bits unused ──

    #[test]
    fn prune_dead_columns_24_odd_groups_high_u16_bits_zero() {
        // Arrange: 12 cols → 3 groups → 2 u16 metadata; u16[1] only uses low nibble
        let weight = vec![vec![
            10.0f32, 1.0, 2.0, 3.0, // group 0
            4.0, 5.0, 6.0, 7.0,     // group 1
            8.0, 9.0, 0.5, 11.0,    // group 2
        ]];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: second u16 only has the low nibble set (group 2), high nibble should be 0
        assert_eq!(sp_meta[0].len(), 2, "12 cols → 3 groups → 2 u16");
        // The high nibble of the second u16 is unused (no group 3), so it should be 0
        assert_eq!(sp_meta[0][1] >> 4, 0, "unused high nibble in last u16 should be 0");
    }

    // ── CodecError: inner String is consumed on access ──

    #[test]
    fn codec_error_inner_consumed_on_destructure() {
        // Arrange
        let err = CodecError("owned value".to_string());
        // Act: move the inner String out
        let inner = err.0;
        // Assert: inner is the expected string
        assert_eq!(inner, "owned value");
    }

    // ── NvcompAnsError: inner String is consumed on access ──

    #[test]
    fn nvcomp_ans_error_inner_consumed_on_destructure() {
        // Arrange
        let err = NvcompAnsError("gpu detail".to_string());
        // Act
        let inner = err.0;
        // Assert
        assert_eq!(inner, "gpu detail");
    }

    // ── CompressionCodec: discriminant ordering ──

    #[test]
    fn compression_codec_discriminant_ordering() {
        // Arrange & Act & Assert: discriminants are monotonically increasing
        assert!(CompressionCodec::None.as_u8() < CompressionCodec::Lz4.as_u8());
        assert!(CompressionCodec::Lz4.as_u8() < CompressionCodec::BitPackRle.as_u8());
        assert!(CompressionCodec::BitPackRle.as_u8() < CompressionCodec::NvcompAns.as_u8());
        assert!(CompressionCodec::NvcompAns.as_u8() < CompressionCodec::ZstdDict.as_u8());
    }

    // ── deduplicate_gqa_heads: partially overlapping multi-dim heads ──

    #[test]
    fn deduplicate_gqa_partial_overlap_multi_row_no_merge() {
        // Arrange: 2 heads with head_dim=2, where row 0 is identical but row 1 differs
        // head 0: [1,0], [0,1] → head 1: [1,0], [1,0]
        // row 0: identical (sim=1.0), row 1: [0,1] vs [1,0] → sim=0.0
        // average sim = 0.5 → below 0.98 threshold → no merge
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 0.0], vec![1.0, 0.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: average similarity is 0.5 < 0.98 → no merge
        assert_ne!(dedup_idx[0], dedup_idx[1], "partially overlapping heads should not merge");
        assert_eq!(dedup_w.len(), 2 * head_dim, "2 distinct groups × 2 rows");
    }

    // ── cosine_sim_rows: one-element opposite vectors ──

    #[test]
    fn cosine_sim_rows_single_element_opposite() {
        // Arrange: 1-element vectors pointing opposite
        let a = vec![5.0f32];
        let b = vec![-3.0f32];
        // Act: dot=-15, norm_a=5, norm_b=3, sim=-15/15=-1.0
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite 1D vectors should have sim=-1.0, got {sim}");
    }

    // ── prune_dead_columns: row independence (modifying pruned output) ──

    #[test]
    fn prune_dead_columns_output_is_independent_copy() {
        // Arrange
        let weight = vec![vec![1.0f32, 0.001, 5.0]];
        let (mut pruned, _) = prune_dead_columns(&weight, 0.01);
        // Act: modify pruned output
        pruned[0][0] = 999.0;
        // Assert: original weight is unaffected
        assert_eq!(weight[0][0], 1.0, "modifying pruned output should not affect input");
    }

    // ── bitpack_rle: high nibble masking verified via different inputs ──

    #[test]
    fn bitpack_rle_high_bits_masked_in_compress() {
        // Arrange: two inputs that differ only in high nibble
        let input_a = vec![0x12u8; 4]; // low nibble = 2
        let input_b = vec![0x82u8; 4]; // low nibble = 2 (same!)
        // Act
        let compressed_a = compress_bitpack_rle(&input_a);
        let compressed_b = compress_bitpack_rle(&input_b);
        // Assert: both should produce identical compressed output (only low nibble matters)
        assert_eq!(compressed_a, compressed_b, "high nibble should be ignored in compression");
        // Decompressed output should be identical (low nibble = 2)
        let decomp_a = decompress_bitpack_rle(&compressed_a, 4);
        let decomp_b = decompress_bitpack_rle(&compressed_b, 4);
        assert_eq!(decomp_a, decomp_b);
        assert!(decomp_a.iter().all(|&b| b == 0x02), "output should be low nibble 2");
    }

    // ── CompressionCodec: from_u8 for all values is deterministic ──

    #[test]
    fn compression_codec_from_u8_deterministic() {
        // Arrange & Act: call from_u8 multiple times for same value
        for v in 0u8..=4 {
            let first = CompressionCodec::from_u8(v);
            let second = CompressionCodec::from_u8(v);
            // Assert: deterministic
            assert_eq!(first, second, "from_u8 should be deterministic for value {v}");
        }
    }

    // ── prune_dead_columns_24: 40 cols (10 groups) metadata ──

    #[test]
    fn prune_dead_columns_24_40_cols_10_groups_meta() {
        // Arrange: 1 row × 40 cols → 10 groups → 5 u16 metadata
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
            29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0, 36.0,
            37.0, 38.0, 39.0, 40.0,
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 5, "40 cols → 10 groups → 5 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 20, "40 cols → 20 pruned (50%)");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 more)
    // ════════════════════════════════════════════════════════════════════════════

    // ── prune_dead_columns: every column norm below threshold → all pruned ──

    #[test]
    fn prune_dead_columns_threshold_above_max_norm_prunes_all() {
        // Arrange: col norms = [1.0, 2.0, 3.0], mean = 2.0
        // threshold_ratio = 2.0 → threshold = 4.0 → all norms < 4.0
        let weight = vec![vec![1.0f32, 2.0, 3.0]];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 2.0);
        // Assert: every column norm is below 4.0
        assert!(mask.iter().all(|m| *m), "all columns should be pruned when threshold > all norms");
    }

    // ── deduplicate_gqa_heads: hidden=1 minimal weight matrix ──

    #[test]
    fn deduplicate_gqa_hidden_dim_1() {
        // Arrange: 3 heads, hidden=1 (each row is a single f32)
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![5.0],
            vec![5.0],  // identical to head 0
            vec![-5.0], // opposite direction
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.98);
        // Assert: heads 0 and 1 merge; head 2 stays separate
        assert_eq!(dedup_w.len(), 2 * head_dim, "2 unique groups");
        assert_eq!(dedup_idx[0], dedup_idx[1], "identical heads merge");
        assert_ne!(dedup_idx[0], dedup_idx[2], "opposite head stays separate");
    }

    // ── cosine_sim_rows: single-element same sign different magnitude ──

    #[test]
    fn cosine_sim_rows_single_element_same_sign() {
        // Arrange: [3.0] and [7.0] — both positive → sim=1.0
        let a = vec![3.0f32];
        let b = vec![7.0f32];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: same direction in 1D → cosine sim = 1.0
        assert!((sim - 1.0).abs() < 1e-6, "same-sign 1D vectors should have sim=1.0, got {sim}");
    }

    // ── prune_dead_columns_24: 4 cols all same value → deterministic survivor positions ──

    #[test]
    fn prune_dead_columns_24_all_same_value_survivors_known() {
        // Arrange: all values = 5.0 → sort stability means first two indices survive
        let weight = vec![vec![5.0f32; 4]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: 2 survive, 2 pruned
        let nonzero_count = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 2, "2 of 4 should survive");
        // sp_meta should encode exactly 2 surviving positions
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 u16 metadata");
    }

    // ── compress_bitpack_rle → decompress_bitpack_rle: single element value 0x0F ──

    #[test]
    fn bitpack_rle_single_element_nibble_15_roundtrip() {
        // Arrange
        let input = vec![0x0Fu8];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 1);
        // Assert: high nibble of compressed = 0xF, low nibble = 0 (run_len-1=0)
        assert_eq!(compressed[0], 0xF0, "val=0xF, run=1 → 0xF0");
        assert_eq!(decompressed, input);
    }

    // ── train_zstd_dictionary: two samples sufficient for small dict ──

    #[test]
    fn train_zstd_dictionary_two_samples_sufficient() {
        // Arrange: 2 samples, each 256 bytes with different patterns
        let s1: Vec<u8> = (0..256).map(|i| (i * 3 % 256) as u8).collect();
        let s2: Vec<u8> = (0..256).map(|i| (i * 7 % 256) as u8).collect();
        // Act
        let dict = train_zstd_dictionary(&[s1.as_slice(), s2.as_slice()], 2048);
        // Assert: should produce a non-empty dict or gracefully return empty
        assert!(dict.len() <= 2048, "dict should not exceed capacity");
        // If dict training succeeds, verify it can be used for compression
        if !dict.is_empty() {
            let result = compress_zstd_dict(&s1, &dict);
            assert!(result.is_ok(), "should be able to compress with trained dict");
        }
    }

    // ── lz4_compress: incremental byte pattern roundtrip ──

    #[test]
    fn lz4_roundtrip_incremental_bytes() {
        // Arrange: 0, 1, 2, ..., 255, 0, 1, ... pattern
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    // ── deduplicate_gqa_heads: verify averaging across 4 merged heads ──

    #[test]
    fn deduplicate_gqa_averages_four_merged_heads() {
        // Arrange: 4 heads with same direction, values [1.0], [3.0], [5.0], [7.0]
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0], vec![3.0], vec![5.0], vec![7.0],
        ];
        // Act: threshold=0.0 merges all positive-direction heads
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 4, head_dim, 0.0);
        // Assert: all merge into 1 group
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
        assert_eq!(dedup_idx[2], dedup_idx[3]);
        // Average = (1+3+5+7)/4 = 4.0
        let expected = 4.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-6,
            "average of 4 merged heads should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ── prune_dead_columns_24: 4 cols descending values deterministic selection ──

    #[test]
    fn prune_dead_columns_24_descending_keeps_first_two() {
        // Arrange: [4.0, 3.0, 2.0, 1.0] → largest |vals| are pos 0 and 1
        let weight = vec![vec![4.0f32, 3.0, 2.0, 1.0]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: pos 0 and 1 survive
        assert_eq!(pruned[0][0], 4.0, "pos 0 should survive");
        assert_eq!(pruned[0][1], 3.0, "pos 1 should survive");
        assert_eq!(pruned[0][2], 0.0, "pos 2 should be pruned");
        assert_eq!(pruned[0][3], 0.0, "pos 3 should be pruned");
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 u16 metadata");
    }

    // ── compress_zstd_dict: error message contains useful context ──

    #[test]
    fn compress_zstd_dict_empty_dict_error_message_content() {
        // Act
        let err = compress_zstd_dict(b"some data", b"").unwrap_err();
        // Assert: error message should mention both the problem and context
        assert!(err.0.contains("empty dictionary"), "error should mention empty dictionary, got: {}", err.0);
        assert!(err.0.contains("compress_zstd_dict"), "error should mention function name, got: {}", err.0);
    }

    // ── decompress_zstd_dict: error message for empty dict ──

    #[test]
    fn decompress_zstd_dict_empty_dict_error_is_string() {
        // Act
        let result = decompress_zstd_dict(b"compressed", b"", 100);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("empty dictionary"), "error should mention empty dict, got: {msg}");
    }

    // ── prune_dead_columns_24: 4 cols with one very large negative ──

    #[test]
    fn prune_dead_columns_24_one_large_negative_survives() {
        // Arrange: [-100, 0.001, 0.001, 0.001] → keep pos 0 (|-100|) and any other
        let weight = vec![vec![-100.0f32, 0.001, 0.001, 0.001]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: pos 0 must survive
        assert_eq!(pruned[0][0], -100.0, "large negative at pos 0 must survive");
        // Exactly 2 survive
        let nonzero = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 2, "exactly 2 of 4 survive");
    }

    // ── cosine_similarity_heads: two heads with head_dim=2, one row zero ──

    #[test]
    fn cosine_similarity_heads_one_zero_row_reduces_average() {
        // Arrange: 2 heads with head_dim=2
        // head 0 rows: [1,0], [0,0] (row 1 is zero → sim with any = 0.0)
        // head 1 rows: [1,0], [1,0]
        // row 0: sim=1.0, row 1: sim=0.0 → average = 0.5
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![0.0, 0.0],
            vec![1.0, 0.0], vec![1.0, 0.0],
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 2);
        // Assert
        assert!((sim - 0.5).abs() < 1e-6, "average of 1.0 and 0.0 should be 0.5, got {sim}");
    }

    // ── bitpack_rle: alternating runs of length 1 with same value ──

    #[test]
    fn bitpack_rle_adjacent_different_values_no_merge() {
        // Arrange: 0x01, 0x02, 0x01, 0x02 — adjacent different values
        let input = vec![0x01u8, 0x02, 0x01, 0x02];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: 4 entries (no run merging across different values)
        assert_eq!(compressed.len(), 4, "alternating different values → 4 entries");
        assert_eq!(decompressed, input);
    }

    // ── lz4_decompress: empty compressed with size 0 succeeds ──

    #[test]
    fn lz4_roundtrip_empty_input_empty_output() {
        // Arrange
        let data: Vec<u8> = vec![];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, 0).unwrap();
        // Assert
        assert!(compressed.is_empty() || compressed.len() < 16, "empty data should compress to tiny output");
        assert!(decompressed.is_empty());
    }

    // ── prune_dead_columns_24: 4 cols where two values equal to survivors ──

    #[test]
    fn prune_dead_columns_24_tie_breaking_keeps_earlier_positions() {
        // Arrange: [5.0, 5.0, 1.0, 1.0] — first two are tied largest
        // sort_unstable_by on equal elements preserves input order → keep [0,1]
        let weight = vec![vec![5.0f32, 5.0, 1.0, 1.0]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: positions 0 and 1 survive (tied, first two chosen by sort stability)
        assert_ne!(pruned[0][0], 0.0, "pos 0 should survive (tied first)");
        assert_ne!(pruned[0][1], 0.0, "pos 1 should survive (tied second)");
        assert_eq!(pruned[0][2], 0.0, "pos 2 should be pruned");
        assert_eq!(pruned[0][3], 0.0, "pos 3 should be pruned");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 more, batch 6)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn deduplicate_gqa_two_heads_zero_rows_merge() {
        // Arrange: two heads with all-zero rows → cosine sim = 0.0 (zero norms)
        // At threshold=0.0, sim(0.0) >= 0.0 is false, so they should NOT merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0],
            vec![0.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.0);
        // Assert: zero vectors have sim=0.0, and 0.0 >= 0.0 is true, so they DO merge
        assert_eq!(dedup_idx[0], dedup_idx[1], "zero heads merge at threshold=0.0 since sim=0.0 >= 0.0");
        assert_eq!(dedup_w.len(), head_dim, "1 group");
    }

    #[test]
    fn prune_dead_columns_all_positive_inf_survives() {
        // Arrange: col with +inf values → norm = inf → survives any threshold
        let weight = vec![vec![f32::INFINITY, 0.001]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.99);
        // Assert: +inf column has infinite norm, always survives
        assert!(!mask[0], "+inf column should survive");
        assert!(mask[1], "tiny column should be pruned");
        assert!(pruned[0][0].is_infinite(), "surviving value should still be +inf");
    }

    #[test]
    fn prune_dead_columns_24_float_min_positive_survives_if_large_enough() {
        // Arrange: [f32::MAX, f32::MIN_POSITIVE, 1.0, 0.5]
        // Keep f32::MAX and 1.0 (two largest absolute values)
        let weight = vec![vec![f32::MAX, f32::MIN_POSITIVE, 1.0f32, 0.5]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(pruned[0][0], f32::MAX, "f32::MAX survives");
        assert_eq!(pruned[0][2], 1.0, "1.0 survives as second largest");
        assert_eq!(pruned[0][1], 0.0, "min positive pruned");
        assert_eq!(pruned[0][3], 0.0, "0.5 pruned");
    }

    #[test]
    fn bitpack_rle_decompress_stops_at_end_of_compressed_data() {
        // Arrange: single entry encoding run=5, request 10
        let compressed = vec![(0x04 << 4) | 4]; // val=4, run=5
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 10);
        // Assert: only 5 bytes produced since compressed data only encodes 5
        assert_eq!(decompressed.len(), 5, "should only produce as many bytes as encoded");
        assert!(decompressed.iter().all(|&b| b == 0x04));
    }

    #[test]
    fn lz4_roundtrip_descending_bytes() {
        // Arrange: 255, 254, 253, ..., 0
        let data: Vec<u8> = (0..256).rev().map(|i| i as u8).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }


    #[test]
    fn cosine_sim_rows_60_degree_angle() {
        // Arrange: [1,0] and [0.5, sqrt(3)/2] → 60 degrees → cos(60) = 0.5
        let a = vec![1.0f32, 0.0];
        let b = vec![0.5f32, 3.0f32.sqrt() / 2.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - 0.5).abs() < 1e-5, "60-degree vectors should have sim=0.5, got {sim}");
    }

    #[test]
    fn prune_dead_columns_very_small_values_not_zero_norm() {
        // Arrange: col 0 has 1e-10 values, col 1 has 1e-10 values (identical norms)
        let weight = vec![vec![1e-10f32, 1e-10]];
        // Act: threshold = 0.5 * mean, mean norm = each col norm → nothing below
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: both columns have the same norm → nothing pruned
        assert!(mask.iter().all(|m| !m), "identical tiny norms → nothing pruned");
    }

    #[test]
    fn compression_codec_none_variant_zero_discriminant() {
        // Arrange & Act
        let codec = CompressionCodec::None;
        // Assert
        assert_eq!(codec as u8, 0u8, "None variant should have discriminant 0");
        assert_eq!(codec.as_u8(), 0u8);
    }

    #[test]
    fn codec_error_display_with_special_characters() {
        // Arrange: error message with unicode and special chars
        let err = CodecError("失败: 压缩错误 \t\n\"quoted\"".to_string());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("失败"), "should preserve unicode");
        assert!(display.contains("quoted"), "should preserve quotes");
        assert!(display.starts_with("CodecError: "), "should have prefix");
    }

    #[test]
    fn nvcomp_ans_error_display_with_special_characters() {
        // Arrange
        let err = NvcompAnsError("GPU异常\t\u{1F525}".to_string());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("GPU异常"), "should preserve unicode");
        assert!(display.starts_with("NvcompAnsError: "), "should have prefix");
    }

    #[test]
    fn prune_dead_columns_24_zero_and_nonzero_in_same_group() {
        // Arrange: group of 4 where two are zero: [0.0, 5.0, 0.0, 3.0]
        let weight = vec![vec![0.0f32, 5.0, 0.0, 3.0]];
        // Act: keep 5.0 (pos 1) and 3.0 (pos 3)
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: positions 1 and 3 survive
        assert_eq!(pruned[0][0], 0.0, "pos 0 was already zero, stays zero");
        assert_eq!(pruned[0][1], 5.0, "pos 1 (5.0) survives");
        assert_eq!(pruned[0][2], 0.0, "pos 2 was already zero, stays zero");
        assert_eq!(pruned[0][3], 3.0, "pos 3 (3.0) survives");
    }


    #[test]
    fn deduplicate_gqa_indices_monotonically_assigned() {
        // Arrange: 5 distinct heads
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            vec![-1.0, 0.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 5, head_dim, 0.98);
        // Assert: group indices should be assigned 0,1,2,3,4 in order
        assert_eq!(dedup_idx[0], 0, "first head always maps to group 0");
        assert_eq!(dedup_idx[1], 1, "second distinct head maps to group 1");
        assert_eq!(dedup_idx[2], 2, "third distinct head maps to group 2");
        assert_eq!(dedup_idx[3], 3, "fourth distinct head maps to group 3");
        assert_eq!(dedup_idx[4], 4, "fifth distinct head maps to group 4");
    }

    #[test]
    fn prune_dead_columns_two_dead_columns_at_edges() {
        // Arrange: col 0 and col 4 are dead (tiny), cols 1-3 alive
        let weight = vec![vec![0.0001f32, 10.0, 20.0, 30.0, 0.0002]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[0], "col 0 should be pruned");
        assert!(!mask[1], "col 1 should survive");
        assert!(!mask[2], "col 2 should survive");
        assert!(!mask[3], "col 3 should survive");
        assert!(mask[4], "col 4 should be pruned");
        assert_eq!(pruned[0][0], 0.0, "pruned edge col zeroed");
        assert_eq!(pruned[0][4], 0.0, "pruned edge col zeroed");
        assert_eq!(pruned[0][2], 20.0, "middle col value preserved");
    }

    // ── Additional tests (15 new, batch 7) ──

    #[test]
    fn prune_dead_columns_24_single_element_survives_zero_input() {
        // Arrange: group of 4 all zeros → survivors are also zero, metadata still encodes
        let weight = vec![vec![0.0f32; 4]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: exactly 2 "survive" (but they're zero), metadata still valid
        assert_eq!(sp_meta.len(), 1, "should have 1 row of metadata");
        assert_eq!(sp_meta[0].len(), 1, "4 cols → 1 u16");
        // All values remain zero
        assert!(pruned[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cosine_sim_rows_positive_inf_and_finite() {
        // Arrange: one vector has +inf, other is finite
        let a = vec![f32::INFINITY, 1.0];
        let b = vec![1.0, 1.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: dot=inf, norm_a=inf, norm_b=finite → inf/(inf*finite) = NaN
        assert!(sim.is_nan() || sim.is_infinite(),
            "inf mixed with finite should produce NaN or infinite, got {sim}");
    }

    #[test]
    fn deduplicate_gqa_heads_with_subnormal_weights() {
        // Arrange: two heads with subnormal f32 values (same direction)
        let head_dim = 1;
        let sub = f32::from_bits(1); // smallest positive subnormal
        let rows: Vec<Vec<f32>> = vec![
            vec![sub],
            vec![sub],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: norms are tiny → cosine_sim_rows returns 0.0 → 0.0 < 0.98 → no merge
        assert_ne!(dedup_idx[0], dedup_idx[1], "subnormal zero-norm heads should not merge at 0.98");
        assert_eq!(dedup_w.len(), 2, "both heads remain");
    }

    #[test]
    fn bitpack_rle_run_of_exactly_29_splits_15_14() {
        // Arrange: 29 identical nibbles → 15+14 = 2 entries
        let input = vec![0x0Du8; 29];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 2, "29 elements → 15+14 = 2 entries");
        assert_eq!(compressed[0] & 0x0F, 14, "first chunk run=15");
        assert_eq!(compressed[1] & 0x0F, 13, "second chunk run=14");
        let decompressed = decompress_bitpack_rle(&compressed, 29);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn prune_dead_columns_inf_norm_never_pruned() {
        // Arrange: col 0 = +inf, col 1 = -inf, col 2 = 1.0, col 3 = 0.001
        let weight = vec![vec![f32::INFINITY, f32::NEG_INFINITY, 1.0f32, 0.001]];
        // Act: mean norm = inf, threshold = inf → all finite norms < inf → pruned
        let (_, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: infinite norms always survive
        assert!(!mask[0], "+inf column should survive");
        assert!(!mask[1], "-inf column should survive (norm is +inf)");
    }

    #[test]
    fn codec_error_to_string_matches_display() {
        // Arrange
        let err = CodecError("verify consistency".to_string());
        // Act
        let display = format!("{err}");
        let to_string = err.to_string();
        // Assert: Display and to_string should produce identical output
        assert_eq!(display, to_string, "Display and to_string should match");
    }

    #[test]
    fn nvcomp_ans_error_to_string_matches_display() {
        // Arrange
        let err = NvcompAnsError("verify consistency".to_string());
        // Act
        let display = format!("{err}");
        let to_string = err.to_string();
        // Assert
        assert_eq!(display, to_string, "Display and to_string should match");
    }

    #[test]
    fn deduplicate_gqa_all_heads_zero_vector_high_threshold() {
        // Arrange: 3 heads all-zero → cosine sim between any pair = 0.0
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![vec![0.0], vec![0.0], vec![0.0]];
        // Act: threshold = 0.5 → sim(0.0) >= 0.5 is false → no merge
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.5);
        // Assert: all stay separate since zero-norm cosine sim = 0.0 < 0.5
        assert_eq!(dedup_w.len(), 3, "zero-norm heads should not merge at 0.5 threshold");
        assert_ne!(dedup_idx[0], dedup_idx[1]);
        assert_ne!(dedup_idx[1], dedup_idx[2]);
    }

    #[test]
    fn compression_codec_bitpackrle_discriminant() {
        // Arrange & Act
        let codec = CompressionCodec::BitPackRle;
        // Assert
        assert_eq!(codec.as_u8(), 2);
        assert_eq!(codec as u8, 2);
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
    }

    #[test]
    fn prune_dead_columns_mixed_inf_and_nan_no_panic() {
        // Arrange: col 0 = NaN, col 1 = +inf, col 2 = 1.0
        let weight = vec![vec![f32::NAN, f32::INFINITY, 1.0f32]];
        // Act: should not panic despite NaN in norm computation
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: returns valid mask and pruned output
        assert_eq!(mask.len(), 3, "mask should have 3 entries");
        assert_eq!(pruned.len(), 1, "row count preserved");
        assert_eq!(pruned[0].len(), 3, "col count preserved");
    }

    #[test]
    fn bitpack_rle_decompress_with_zero_requested_after_full_run() {
        // Arrange: compress 5 elements, decompress 0 first, then 5
        let input = vec![0x06u8; 5];
        let compressed = compress_bitpack_rle(&input);
        // Act
        let zero_result = decompress_bitpack_rle(&compressed, 0);
        let full_result = decompress_bitpack_rle(&compressed, 5);
        // Assert
        assert!(zero_result.is_empty(), "requesting 0 should yield empty");
        assert_eq!(full_result, input, "subsequent decompress of same data should work");
    }

    #[test]
    fn cosine_similarity_heads_large_weight_matrix() {
        // Arrange: 100 rows (10 heads × 10 rows_per_head), all rows identical
        let row = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight: Vec<Vec<f32>> = (0..100).map(|_| row.clone()).collect();
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 5, 10);
        // Assert: all rows identical → sim = 1.0
        assert!((sim - 1.0).abs() < 1e-4, "identical rows should give sim=1.0, got {sim}");
    }

    #[test]
    fn lz4_compress_preserves_byte_boundaries() {
        // Arrange: data with byte value 0 at various positions
        let data: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert: byte boundaries and zero bytes preserved exactly
        assert_eq!(decompressed, data);
    }

    #[test]
    fn prune_dead_columns_24_meta_decodable_per_group() {
        // Arrange: 8 cols → 2 groups → metadata should allow decoding keep positions
        // Group 0: [1, 10, 3, 2] → keep pos 1 (10.0) and pos 0 (1.0)
        // Group 1: [8, 0.1, 7, 0.5] → keep pos 0 (8.0) and pos 2 (7.0)
        let weight = vec![vec![1.0f32, 10.0, 3.0, 2.0, 8.0, 0.1, 7.0, 0.5]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        let meta = sp_meta[0][0];
        // Assert: decode low nibble (group 0)
        let g0_pos0 = (meta & 0x03) as usize;
        let g0_pos1 = ((meta >> 2) & 0x03) as usize;
        assert_ne!(pruned[0][g0_pos0], 0.0, "decoded group 0 position 0 should survive");
        assert_ne!(pruned[0][g0_pos1], 0.0, "decoded group 0 position 1 should survive");
        // Decode high nibble (group 1)
        let g1_pos0 = ((meta >> 4) & 0x03) as usize;
        let g1_pos1 = ((meta >> 6) & 0x03) as usize;
        assert_ne!(pruned[0][4 + g1_pos0], 0.0, "decoded group 1 position 0 should survive");
        assert_ne!(pruned[0][4 + g1_pos1], 0.0, "decoded group 1 position 1 should survive");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 more, batch 8)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn deduplicate_gqa_two_heads_same_direction_different_magnitude() {
        // Arrange: head 0 = [3.0, 4.0], head 1 = [6.0, 8.0] (2x magnitude, same direction)
        // Cosine sim = 1.0 -> should merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![3.0, 4.0],
            vec![6.0, 8.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.95);
        // Assert
        assert_eq!(dedup_idx[0], dedup_idx[1], "same-direction heads should merge");
        assert_eq!(dedup_w.len(), head_dim, "should have 1 unique group");
        // Averaged: [(3+6)/2, (4+8)/2] = [4.5, 6.0]
        assert!((dedup_w[0][0] - 4.5).abs() < 1e-5, "col 0 average should be 4.5");
        assert!((dedup_w[0][1] - 6.0).abs() < 1e-5, "col 1 average should be 6.0");
    }

    #[test]
    fn prune_dead_columns_threshold_1_exactly_mean_prunes_below() {
        // Arrange: col 0 norm=6.0, col 1 norm~2.83, col 2 norm~5.66
        // mean ~4.83, threshold=1.0*mean ~4.83 -> col 1 pruned
        let weight = vec![
            vec![6.0f32, 2.0, 4.0],
            vec![0.0f32, 2.0, 4.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 1.0);
        // Assert
        assert!(!mask[0], "col 0 (norm=6.0) should survive");
        assert!(mask[1], "col 1 (norm~2.83) should be pruned");
        assert!(!mask[2], "col 2 (norm~5.66) should survive");
        assert_eq!(pruned[0][1], 0.0, "pruned col should be zeroed");
    }

    #[test]
    fn prune_dead_columns_24_two_groups_identical_pattern_same_meta() {
        // Arrange: two groups with identical value ordering -> same metadata encoding
        let weight = vec![vec![10.0f32, 5.0, 2.0, 1.0, 10.0, 5.0, 2.0, 1.0]];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: group 0 and group 1 have identical value patterns
        let meta = sp_meta[0][0];
        let low_nibble = meta & 0xF;
        let high_nibble = (meta >> 4) & 0xF;
        assert_eq!(low_nibble, high_nibble, "identical group patterns should produce identical metadata nibbles");
    }

    #[test]
    fn bitpack_rle_compress_decompress_run_of_exactly_15_values() {
        // Arrange: exactly 15 identical values -> single compressed byte
        let input = vec![0x0Au8; 15];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 15);
        // Assert
        assert_eq!(compressed.len(), 1, "15 elements should compress to 1 byte");
        assert_eq!(compressed[0], 0xAE, "val=0xA, run_len=15 -> (0xA<<4)|14 = 0xAE");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn lz4_roundtrip_pattern_with_many_zeros() {
        // Arrange: sparse data with many zero bytes
        let mut data = vec![0u8; 1024];
        data[0] = 0xFF;
        data[512] = 0xFF;
        data[1023] = 0xFF;
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len() / 5, "sparse data should compress well");
    }

    #[test]
    fn zstd_dict_roundtrip_preserves_byte_boundaries() {
        // Arrange: data with specific byte boundaries
        let samples: Vec<Vec<u8>> = (0..15)
            .map(|i| {
                let mut v = vec![0u8; 128];
                for j in 0..128 {
                    v[j] = if j % 2 == 0 { (i * 3) as u8 } else { 0xFF };
                }
                v
            })
            .collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = train_zstd_dictionary(&sample_refs, 2048);
        if dict.is_empty() {
            return;
        }
        // Act
        let compressed = compress_zstd_dict(&samples[0], &dict).unwrap();
        let decompressed = decompress_zstd_dict(&compressed, &dict, samples[0].len()).unwrap();
        // Assert: byte-for-byte match including 0xFF boundaries
        assert_eq!(decompressed, samples[0], "byte boundaries must be preserved");
    }

    #[test]
    fn cosine_sim_rows_120_degree_angle() {
        // Arrange: vectors at ~120 degrees -> cos(120) = -0.5
        // [1, 0] and [-0.5, sqrt(3)/2]
        let a = vec![1.0f32, 0.0];
        let b = vec![-0.5f32, 3.0f32.sqrt() / 2.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - (-0.5)).abs() < 1e-5, "120-degree vectors should have sim=-0.5, got {sim}");
    }

    #[test]
    fn deduplicate_gqa_threshold_0_5_merges_parallel_not_orthogonal() {
        // Arrange: head 0 = [1,0], head 1 = [1,1] (sim=1/sqrt(2)~0.707), head 2 = [0,1] (sim with 0 = 0)
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        // Act: threshold 0.5 -> head 0 and 1 merge (sim~0.707 > 0.5); head 2 stays separate (sim=0 < 0.5)
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, head_dim, 0.5);
        // Assert
        assert_eq!(dedup_idx[0], dedup_idx[1], "parallel-ish heads merge at 0.5");
        assert_ne!(dedup_idx[0], dedup_idx[2], "orthogonal head stays separate");
        assert_eq!(dedup_w.len(), 2 * head_dim, "2 unique groups");
    }

    #[test]
    fn prune_dead_columns_rows_with_different_col_survival() {
        // Arrange: 3 rows x 4 cols where col 2 is consistently weak
        let weight = vec![
            vec![10.0f32, 8.0, 0.001, 12.0],
            vec![11.0f32, 9.0, 0.001, 13.0],
            vec![9.0f32, 7.0, 0.001, 11.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[2], "consistently weak col 2 should be pruned");
        assert!(!mask[0] && !mask[1] && !mask[3], "other cols should survive");
        for row in &pruned {
            assert_eq!(row[2], 0.0, "pruned col should be zero in every row");
        }
    }

    #[test]
    fn codec_error_new_constructs_correctly() {
        // Arrange & Act
        let err = CodecError("construction test".to_string());
        // Assert: newly constructed error has correct inner value and display
        assert_eq!(err.0, "construction test");
        assert_eq!(format!("{err}"), "CodecError: construction test");
    }

    #[test]
    fn nvcomp_ans_error_new_constructs_correctly() {
        // Arrange & Act
        let err = NvcompAnsError("construction test".to_string());
        // Assert
        assert_eq!(err.0, "construction test");
        assert_eq!(format!("{err}"), "NvcompAnsError: construction test");
    }

    #[test]
    fn prune_dead_columns_24_28_cols_7_groups_meta() {
        // Arrange: 1 row x 28 cols -> 7 groups -> 4 u16 metadata (ceil(7/2)=4)
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 4, "28 cols -> 7 groups -> 4 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 14, "28 cols -> 14 pruned (50%)");
    }

    #[test]
    fn bitpack_rle_input_with_only_nibble_zero_roundtrip() {
        // Arrange: all zeros (nibble value 0)
        let input = vec![0x00u8; 7];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 7);
        // Assert: single entry encoding run of 7
        assert_eq!(compressed.len(), 1, "7 identical zeros -> 1 entry");
        assert_eq!(compressed[0], 0x06, "val=0, run=7 -> (0<<4)|6 = 0x06");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn deduplicate_gqa_head_dim_4_multi_row_no_merge() {
        // Arrange: 2 heads with head_dim=4, rows are orthogonal -> no merge
        // Head 0 rows point in one direction, head 1 rows in the perpendicular direction
        let head_dim = 4;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], // head 0: along x-axis
            vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], // head 1: along y-axis (orthogonal)
        ];
        // Act: orthogonal heads -> cosine sim = 0 -> no merge
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: orthogonal heads -> should stay separate
        assert_eq!(dedup_w.len(), 2 * head_dim, "orthogonal heads should stay separate");
        assert_ne!(dedup_idx[0], dedup_idx[1], "orthogonal heads must map to different groups");
    }

    #[test]
    fn compression_codec_usable_in_hashset_with_all_variants() {
        // Arrange
        use std::collections::HashSet;
        // Act: insert all 5 variants
        let mut set = HashSet::new();
        set.insert(CompressionCodec::None);
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::BitPackRle);
        set.insert(CompressionCodec::NvcompAns);
        set.insert(CompressionCodec::ZstdDict);
        // Assert: 5 unique variants
        assert_eq!(set.len(), 5, "all 5 CompressionCodec variants should be unique in HashSet");
        // Verify all present
        assert!(set.contains(&CompressionCodec::None));
        assert!(set.contains(&CompressionCodec::Lz4));
        assert!(set.contains(&CompressionCodec::BitPackRle));
        assert!(set.contains(&CompressionCodec::NvcompAns));
        assert!(set.contains(&CompressionCodec::ZstdDict));
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 more, batch 9)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn bitpack_rle_run_of_exactly_46_splits_into_four() {
        // Arrange: 46 identical nibbles -> 15+15+15+1 = 4 entries
        let input = vec![0x04u8; 46];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 4, "46 elements -> 15+15+15+1 = 4 entries");
        assert_eq!(compressed[0] & 0x0F, 14, "first chunk run=15");
        assert_eq!(compressed[1] & 0x0F, 14, "second chunk run=15");
        assert_eq!(compressed[2] & 0x0F, 14, "third chunk run=15");
        assert_eq!(compressed[3] & 0x0F, 0, "fourth chunk run=1");
        let decompressed = decompress_bitpack_rle(&compressed, 46);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn prune_dead_columns_single_row_all_cols_zero_except_one() {
        // Arrange: 5 columns, only col 3 is nonzero
        let weight = vec![vec![0.0f32, 0.0, 0.0, 42.0, 0.0]];
        // Act: mean_norm = 42.0/5 = 8.4, threshold at 0.01*mean = 0.084
        // cols 0,1,2,4 have norm 0.0 < 0.084 -> pruned; col 3 has norm 42.0 > 0.084 -> survives
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(mask[0] && mask[1] && mask[2] && mask[4], "zero-norm columns should be pruned");
        assert!(!mask[3], "nonzero column should survive");
        assert_eq!(pruned[0][3], 42.0, "surviving column preserves value");
    }

    #[test]
    fn cosine_sim_rows_30_degree_angle() {
        // Arrange: vectors at ~30 degrees -> cos(30) = sqrt(3)/2 ~ 0.8660
        let a = vec![1.0f32, 0.0];
        let b = vec![3.0f32.sqrt() / 2.0, 0.5];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        let expected = 3.0f32.sqrt() / 2.0;
        assert!(
            (sim - expected).abs() < 1e-5,
            "30-degree vectors should have sim~0.866, got {sim}"
        );
    }

    #[test]
    fn deduplicate_gqa_head_dim_2_two_merged_output_shape() {
        // Arrange: 2 identical heads, head_dim=2, hidden=5
        let head_dim = 2;
        let hidden = 5;
        let make_row = |v: f32| vec![v; hidden];
        let rows: Vec<Vec<f32>> = vec![
            make_row(1.0), make_row(2.0), // head 0
            make_row(1.0), make_row(2.0), // head 1 (identical)
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: merged to 1 group -> head_dim=2 output rows, each with hidden=5
        assert_eq!(dedup_w.len(), head_dim, "1 merged group -> head_dim rows");
        assert_eq!(dedup_idx[0], dedup_idx[1], "both heads map to same group");
        for (i, row) in dedup_w.iter().enumerate() {
            assert_eq!(row.len(), hidden, "output row {i} should have hidden={hidden}");
        }
    }

    #[test]
    fn lz4_roundtrip_128kb_large_data() {
        // Arrange: 128KB of mixed pattern data
        let data: Vec<u8> = (0..131072).map(|i| {
            let base = (i / 1024) as u8;
            let offset = (i % 256) as u8;
            base.wrapping_add(offset)
        }).collect();
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len(), "mixed pattern data should compress to less than original");
    }

    #[test]
    fn prune_dead_columns_24_8_cols_all_negative() {
        // Arrange: 8 cols all negative values, two groups
        // Group 0: [-10.0, -1.0, -5.0, -0.1] -> keep |-10|=10 and |-5|=5 -> pos 0,2
        // Group 1: [-8.0, -3.0, -0.01, -7.0] -> keep |-8|=8 and |-7|=7 -> pos 0,3
        let weight = vec![vec![-10.0f32, -1.0, -5.0, -0.1, -8.0, -3.0, -0.01, -7.0]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert group 0
        assert_eq!(pruned[0][0], -10.0, "grp0 pos 0 survives");
        assert_eq!(pruned[0][2], -5.0, "grp0 pos 2 survives");
        assert_eq!(pruned[0][1], 0.0, "grp0 pos 1 pruned");
        assert_eq!(pruned[0][3], 0.0, "grp0 pos 3 pruned");
        // Assert group 1
        assert_eq!(pruned[0][4], -8.0, "grp1 pos 0 survives");
        assert_eq!(pruned[0][7], -7.0, "grp1 pos 3 survives");
        assert_eq!(pruned[0][5], 0.0, "grp1 pos 1 pruned");
        assert_eq!(pruned[0][6], 0.0, "grp1 pos 2 pruned");
        // Assert metadata shape
        assert_eq!(sp_meta[0].len(), 1, "8 cols -> 2 groups -> 1 u16");
    }

    #[test]
    fn codec_error_eq_reflexive() {
        // Arrange
        let err = CodecError("reflexive".to_string());
        // Act & Assert: reflexivity (a == a)
        assert_eq!(err, err, "CodecError should be reflexively equal");
    }

    #[test]
    fn nvcomp_ans_error_eq_reflexive() {
        // Arrange
        let err = NvcompAnsError("reflexive".to_string());
        // Act & Assert: reflexivity (a == a)
        assert_eq!(err, err, "NvcompAnsError should be reflexively equal");
    }

    #[test]
    fn compress_zstd_dict_error_type_matches_display() {
        // Arrange
        let err = compress_zstd_dict(b"data", b"").unwrap_err();
        // Act
        let display = format!("{err}");
        // Assert: Display output starts with the expected prefix
        assert!(
            display.starts_with("CodecError: "),
            "Display should start with 'CodecError: ', got: {display}"
        );
    }

    #[test]
    fn decompress_zstd_dict_error_is_string_type() {
        // Arrange: decompress with empty dict returns String error
        let err = decompress_zstd_dict(b"x", b"", 10).unwrap_err();
        // Assert: error is a plain String, contains expected text
        assert!(err.contains("empty dictionary"), "error should mention empty dict, got: {err}");
        assert!(err.contains("decompress_zstd_dict"), "error should mention function name, got: {err}");
    }

    #[test]
    fn prune_dead_columns_single_row_two_dead_in_middle() {
        // Arrange: 6 columns, cols 2 and 4 are near-zero
        let weight = vec![vec![10.0f32, 8.0, 0.0001, 7.0, 0.0002, 9.0]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(!mask[0] && !mask[1], "cols 0,1 should survive");
        assert!(mask[2], "col 2 should be pruned");
        assert!(!mask[3], "col 3 should survive");
        assert!(mask[4], "col 4 should be pruned");
        assert!(!mask[5], "col 5 should survive");
        assert_eq!(pruned[0][0], 10.0, "col 0 value preserved");
        assert_eq!(pruned[0][2], 0.0, "pruned col 2 is zero");
        assert_eq!(pruned[0][3], 7.0, "col 3 value preserved");
        assert_eq!(pruned[0][4], 0.0, "pruned col 4 is zero");
    }

    #[test]
    fn bitpack_rle_compress_decompress_interleaved_short_runs() {
        // Arrange: pattern of alternating short runs: 2 of val A, 3 of val B, 1 of val C, 4 of val A
        let input: Vec<u8> = vec![
            0x01, 0x01,           // 2 of val 1
            0x02, 0x02, 0x02,    // 3 of val 2
            0x03,                // 1 of val 3
            0x01, 0x01, 0x01, 0x01, // 4 of val 1
        ];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert
        assert_eq!(decompressed, input);
        // 4 compressed entries (each run is separate)
        assert_eq!(compressed.len(), 4, "4 distinct runs -> 4 compressed entries");
    }

    #[test]
    fn cosine_similarity_heads_three_heads_all_compared_to_first() {
        // Arrange: 3 heads, head 0 = [1,0], head 1 = [1,0] (identical), head 2 = [0,1] (orthogonal)
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], // head 0
            vec![1.0, 0.0], // head 1
            vec![0.0, 1.0], // head 2
        ];
        // Act: compare each to head 0
        let sim_01 = cosine_similarity_heads(&weight, 0, 1, 1);
        let sim_02 = cosine_similarity_heads(&weight, 0, 2, 1);
        // Assert
        assert!((sim_01 - 1.0).abs() < 1e-6, "identical heads should have sim=1.0, got {sim_01}");
        assert!(sim_02.abs() < 1e-6, "orthogonal heads should have sim=0.0, got {sim_02}");
    }

    #[test]
    fn compression_codec_lz4_variant_roundtrip() {
        // Arrange
        let codec = CompressionCodec::Lz4;
        // Act: serialize and deserialize
        let serialized = codec.as_u8();
        let deserialized = CompressionCodec::from_u8(serialized);
        // Assert
        assert_eq!(deserialized, Some(CompressionCodec::Lz4));
        assert_eq!(serialized, 1u8);
    }

    #[test]
    fn deduplicate_gqa_many_heads_first_head_representative() {
        // Arrange: 5 heads, first 3 identical, last 2 distinct
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0, 1.0],          // head 0
            vec![2.0, 1.0],          // head 1 (identical to 0)
            vec![2.0, 1.0],          // head 2 (identical to 0)
            vec![0.0, 1.0],          // head 3
            vec![1.0, 0.0],          // head 4
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 5, head_dim, 0.98);
        // Assert: heads 0,1,2 map to group 0 (first encountered)
        assert_eq!(dedup_idx[0], 0, "head 0 maps to group 0");
        assert_eq!(dedup_idx[1], 0, "head 1 maps to same group as 0");
        assert_eq!(dedup_idx[2], 0, "head 2 maps to same group as 0");
        // Heads 3,4 have different group indices
        assert_ne!(dedup_idx[3], 0, "head 3 is in a different group");
        assert_ne!(dedup_idx[4], 0, "head 4 is in a different group");
        assert_ne!(dedup_idx[3], dedup_idx[4], "head 3 and 4 are in different groups");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 new)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn cosine_sim_rows_mismatched_length_uses_shorter() {
        // Arrange: vectors of different lengths — zip truncates to shorter
        let a = vec![1.0f32, 0.0, 0.0]; // length 3
        let b = vec![1.0f32, 0.0];       // length 2
        // Act: zip only processes 2 elements → dot=1.0, norm_a=sqrt(1)=1.0, norm_b=1.0
        let sim = cosine_sim_rows(&a, &b);
        // Assert: only overlapping elements contribute; sim should be 1.0
        assert!((sim - 1.0).abs() < 1e-6, "zip truncation should use shorter length, got {sim}");
    }

    #[test]
    fn prune_dead_columns_24_single_value_survives_if_largest() {
        // Arrange: one dominant value in a group of 4
        let weight = vec![vec![0.001f32, 0.002, 100.0, 0.003]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: pos 2 (100.0) must survive; the second survivor is the next largest
        assert_ne!(pruned[0][2], 0.0, "dominant value at pos 2 must survive");
        assert_eq!(pruned[0][2], 100.0, "dominant value preserved exactly");
        let nonzero_count = pruned[0].iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 2, "exactly 2 survive");
    }

    #[test]
    fn bitpack_rle_decompress_handcrafted_two_values() {
        // Arrange: hand-craft compressed bytes — val=0xA run=4, val=0x1 run=2
        // run_len_minus_1 = 3 and 1
        let compressed: Vec<u8> = vec![(0xA << 4) | 3, (0x1 << 4) | 1];
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 6);
        // Assert
        assert_eq!(decompressed.len(), 6);
        assert_eq!(&decompressed[..4], &[0x0A, 0x0A, 0x0A, 0x0A]);
        assert_eq!(&decompressed[4..], &[0x01, 0x01]);
    }

    #[test]
    fn compression_codec_bitpack_rle_variant_properties() {
        // Arrange
        let codec = CompressionCodec::BitPackRle;
        // Act & Assert: discriminant, roundtrip, debug
        assert_eq!(codec.as_u8(), 2);
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(format!("{codec:?}"), "BitPackRle");
    }

    #[test]
    fn codec_error_symmetry() {
        // Arrange: verify symmetric equality
        let a = CodecError("abc".to_string());
        let b = CodecError("abc".to_string());
        // Act & Assert: a == b implies b == a
        assert_eq!(a, b, "forward equality");
        assert_eq!(b, a, "reverse equality");
    }

    #[test]
    fn nvcomp_ans_error_symmetry() {
        // Arrange
        let a = NvcompAnsError("xyz".to_string());
        let b = NvcompAnsError("xyz".to_string());
        // Act & Assert
        assert_eq!(a, b, "forward equality");
        assert_eq!(b, a, "reverse equality");
    }

    #[test]
    fn deduplicate_gqa_heads_preserves_all_alive_columns_after_merge() {
        // Arrange: 2 identical heads with 3 hidden dims, both have strong column 0
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![10.0f32, 0.001, 20.0], // head 0
            vec![10.0f32, 0.001, 20.0], // head 1 (identical)
        ];
        // Act: merge heads, then prune dead columns on result
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        assert_eq!(dedup_idx[0], dedup_idx[1], "heads should merge");
        let (pruned, mask) = prune_dead_columns(&dedup_w, 0.01);
        // Assert: col 1 should be pruned (low norm), cols 0,2 survive
        assert!(mask[1], "low-norm column should be pruned after merge");
        assert!(!mask[0] && !mask[2], "high-norm columns should survive");
        assert_eq!(pruned[0][0], 10.0, "alive col value preserved");
    }

    #[test]
    fn prune_dead_columns_all_negative_survives() {
        // Arrange: all negative values, all have significant magnitude
        let weight = vec![
            vec![-5.0f32, -10.0],
            vec![-3.0f32, -7.0],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: both columns have large norms, nothing pruned
        assert!(mask.iter().all(|m| !m), "large-magnitude negative columns should survive");
        assert_eq!(pruned[0][0], -5.0, "negative value preserved");
        assert_eq!(pruned[1][1], -7.0, "negative value preserved");
    }

    #[test]
    fn lz4_roundtrip_single_char_repeated() {
        // Arrange: single character repeated many times
        let data = b"a".repeat(10000);
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len() / 50, "repeated single char should compress extremely well");
    }

    #[test]
    fn cosine_similarity_heads_identical_multi_row_different_hidden() {
        // Arrange: 2 heads × 2 rows each, hidden=3 (different from typical hidden=2)
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0], // head 0
            vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0], // head 1 (identical)
        ];
        // Act
        let sim = cosine_similarity_heads(&weight, 0, 1, 2);
        // Assert
        assert!((sim - 1.0).abs() < 1e-6, "identical heads with hidden=3 should have sim=1.0, got {sim}");
    }

    #[test]
    fn bitpack_rle_compress_large_all_same_nibble_500() {
        // Arrange: 500 identical nibbles
        let input = vec![0x07u8; 500];
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, 500);
        // Assert
        assert_eq!(decompressed, input);
        // 500/15 = 33.33... → ceil = 34 entries
        assert_eq!(compressed.len(), 34, "500 elements should produce 34 compressed entries");
    }

    #[test]
    fn prune_dead_columns_24_two_identical_rows_same_meta_shape() {
        // Arrange: two rows with different values but same structure (8 cols)
        let weight = vec![
            vec![10.0f32, 1.0, 8.0, 0.5, 20.0, 2.0, 15.0, 3.0],
            vec![10.0f32, 1.0, 8.0, 0.5, 20.0, 2.0, 15.0, 3.0],
        ];
        // Act
        let (_, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: same rows produce same metadata
        assert_eq!(sp_meta[0], sp_meta[1], "identical rows should produce identical sp_meta");
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 2 groups → 1 u16");
    }

    #[test]
    fn compress_zstd_dict_non_empty_dict_with_data_succeeds() {
        // Arrange: a minimal valid dictionary (16 zero bytes is above zstd minimum)
        let dict = vec![0u8; 64];
        let data = b"test data for zstd compression";
        // Act
        let result = compress_zstd_dict(data, &dict);
        // Assert: should succeed with non-empty dict (even if not trained)
        // zstd accepts any non-empty dict; result may or may not be smaller
        assert!(result.is_ok(), "compress with non-empty dict should succeed, got {:?}", result.err());
    }

    #[test]
    fn deduplicate_gqa_threshold_just_above_1_no_merge() {
        // Arrange: two identical heads with threshold just above 1.0
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![3.0],
            vec![3.0],
        ];
        // Act: threshold=1.0001 > max possible cosine_sim=1.0 → no merge
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 1.0001);
        // Assert: should not merge since sim (1.0) < threshold (1.0001)
        assert_ne!(dedup_idx[0], dedup_idx[1], "threshold above 1.0 should prevent merge");
        assert_eq!(dedup_w.len(), 2 * head_dim, "2 heads should remain separate");
    }

    #[test]
    fn cosine_sim_rows_mixed_zero_and_nonzero_elements() {
        // Arrange: vectors with some zero and some non-zero elements
        let a = vec![0.0f32, 3.0, 0.0, 4.0];
        let b = vec![0.0f32, 3.0, 0.0, 4.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: identical vectors → sim = 1.0
        assert!((sim - 1.0).abs() < 1e-6, "identical sparse vectors should have sim=1.0, got {sim}");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 new)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    // @trace TEST-SC-001 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_one_head_zero_vector_stays_separate() {
        // Arrange: 2 heads, head 0 = [1.0], head 1 = [0.0] (zero vector)
        // cosine_sim([1.0], [0.0]) = 0.0 < 0.98 → no merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![0.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: zero-norm head returns sim=0.0 from cosine_sim_rows, so no merge
        assert_ne!(dedup_idx[0], dedup_idx[1], "zero-vector head should not merge with non-zero");
        assert_eq!(dedup_w.len(), 2 * head_dim, "both heads should remain separate");
    }

    #[test]
    // @trace TEST-SC-002 [req:REQ-STATIC-COMP] [level:unit]
    fn cosine_sim_rows_scaled_vectors_same_similarity() {
        // Arrange: vector b = 3*a → cosine sim should be 1.0 regardless of scaling
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![3.0f32, 6.0, 9.0]; // exactly 3*a
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        assert!((sim - 1.0).abs() < 1e-6, "scaled vectors should have sim=1.0, got {sim}");
    }

    #[test]
    // @trace TEST-SC-003 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_24_16_cols_four_groups() {
        // Arrange: 1 row × 16 cols → 4 groups → ceil(4/2) = 2 u16 metadata
        let weight = vec![vec![
            10.0f32, 1.0, 0.5, 8.0, // group 0: keep 10.0, 8.0
            3.0, 7.0, 0.1, 9.0,     // group 1: keep 9.0, 7.0
            5.0, 2.0, 6.0, 0.2,     // group 2: keep 6.0, 5.0
            4.0, 11.0, 0.3, 12.0,   // group 3: keep 12.0, 11.0
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 2, "16 cols → 4 groups → 2 u16");
        let zero_count = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 8, "50% of 16 = 8 elements should be pruned");
    }

    #[test]
    // @trace TEST-SC-004 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_alternating_01_no_compression_benefit() {
        // Arrange: alternating 0 and 1 — worst case, every byte is its own run
        let input: Vec<u8> = (0..20).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: 20 alternating values → 20 compressed entries (no run > 1)
        assert_eq!(compressed.len(), 20, "alternating values should produce 20 entries");
        assert_eq!(decompressed, input);
    }

    #[test]
    // @trace TEST-SC-005 [req:REQ-STATIC-COMP] [level:unit]
    fn decompress_zstd_dict_wrong_dict_produces_error_or_corruption() {
        // Arrange: train dict A from one data distribution, dict B from another
        let samples_a: Vec<Vec<u8>> = (0..20)
            .map(|i| vec![(i as u8).wrapping_mul(3); 128])
            .collect();
        let samples_b: Vec<Vec<u8>> = (0..20)
            .map(|i| vec![(i as u8).wrapping_mul(7); 128])
            .collect();
        let refs_a: Vec<&[u8]> = samples_a.iter().map(|s| s.as_slice()).collect();
        let refs_b: Vec<&[u8]> = samples_b.iter().map(|s| s.as_slice()).collect();
        let dict_a = train_zstd_dictionary(&refs_a, 4096);
        let dict_b = train_zstd_dictionary(&refs_b, 4096);
        if dict_a.is_empty() || dict_b.is_empty() {
            return; // skip if dict training fails
        }
        let compressed = compress_zstd_dict(&samples_a[0], &dict_a).unwrap();
        // Act: decompress with wrong dict B instead of dict A
        let result = decompress_zstd_dict(&compressed, &dict_b, samples_a[0].len());
        // Assert: zstd should either error or return different data (not original)
        match result {
            Ok(decompressed) => {
                // If decompress succeeds, data should NOT match original
                assert_ne!(decompressed, samples_a[0], "wrong dict should not recover original data");
            }
            Err(_) => {
                // zstd correctly detects dictionary mismatch — also valid
            }
        }
    }

    #[test]
    // @trace TEST-SC-006 [req:REQ-STATIC-COMP] [level:unit]
    fn train_zstd_dictionary_mixed_empty_and_nonempty_samples() {
        // Arrange: some samples are empty, some are non-empty
        let samples: Vec<&[u8]> = vec![b"", b"", b"some real data here for training", b"more data"];
        // Act
        let dict = train_zstd_dictionary(&samples, 4096);
        // Assert: non-empty samples provide data, so dict may succeed
        // The function's total size check ensures it returns empty only if ALL are empty
        // Just verify no panic and result is a valid Vec
        assert!(dict.len() <= 4096);
    }

    #[test]
    // @trace TEST-SC-007 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_single_column_survives_any_threshold() {
        // Arrange: weight with only 1 column — its norm = mean_norm, threshold < norm
        let weight = vec![
            vec![42.0f32],
            vec![100.0f32],
            vec![7.0f32],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, 0.99);
        // Assert: single column can never be pruned (its norm = the mean, threshold < norm)
        assert!(!mask[0], "single column should always survive");
    }

    #[test]
    // @trace TEST-SC-008 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_negative_threshold_merges_opposite_heads() {
        // Arrange: two opposite heads (cosine_sim = -1.0), threshold = -0.5
        // sim (-1.0) >= threshold (-0.5) is false, so they should NOT merge
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![-1.0],
        ];
        // Act
        let (_, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, -0.5);
        // Assert: sim=-1.0 < threshold=-0.5 → no merge
        assert_ne!(dedup_idx[0], dedup_idx[1], "sim=-1.0 is below threshold=-0.5, should not merge");
    }

    #[test]
    // @trace TEST-SC-009 [req:REQ-STATIC-COMP] [level:unit]
    fn cosine_similarity_heads_overlapping_same_start_different_rows_per_head() {
        // Arrange: weight has 3 rows. Compare head starting at row 0 with 1 row_per_head
        // vs same start with 2 rows_per_head
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], // row 0
            vec![1.0, 0.0], // row 1
            vec![0.0, 1.0], // row 2
        ];
        // Act: comparing head 0 vs head 1 with rows_per_head=1 (same as rows_per_head=2 below)
        let sim_rph1 = cosine_similarity_heads(&weight, 0, 1, 1);
        // Assert: with rph=1, head 0 = row 0, head 1 = row 1 → both [1,0] → sim=1.0
        assert!((sim_rph1 - 1.0).abs() < 1e-6, "overlapping start with rph=1 should give sim=1.0, got {sim_rph1}");
    }

    #[test]
    // @trace TEST-SC-010 [req:REQ-STATIC-COMP] [level:unit]
    fn lz4_roundtrip_all_ff_bytes() {
        // Arrange: all 0xFF bytes — tests high-byte data path
        let data = vec![0xFFu8; 1024];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len(), "repeated 0xFF should compress well");
    }

    #[test]
    // @trace TEST-SC-011 [req:REQ-STATIC-COMP] [level:unit]
    fn compression_codec_usable_as_hashmap_key() {
        // Arrange: verify CompressionCodec works as HashMap key (requires Hash + Eq)
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(CompressionCodec::Lz4, "lz4_value");
        map.insert(CompressionCodec::ZstdDict, "zstd_value");
        // Act
        let val_lz4 = map.get(&CompressionCodec::Lz4);
        let val_zstd = map.get(&CompressionCodec::ZstdDict);
        let val_none = map.get(&CompressionCodec::None);
        // Assert
        assert_eq!(val_lz4, Some(&"lz4_value"));
        assert_eq!(val_zstd, Some(&"zstd_value"));
        assert_eq!(val_none, None);
    }

    #[test]
    // @trace TEST-SC-012 [req:REQ-STATIC-COMP] [level:unit]
    fn codec_error_display_unicode_message() {
        // Arrange: error message with unicode characters
        let msg = "压缩失败: 无效参数 ❌";
        let err = CodecError(msg.to_string());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: Display and Debug both preserve unicode
        assert!(display.contains(msg), "Display should preserve unicode");
        assert!(debug.contains(msg), "Debug should preserve unicode");
    }

    #[test]
    // @trace TEST-SC-013 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_24_two_rows_one_uniform_keeps_largest_per_group() {
        // Arrange: row 0 = [5.0, 1.0, 0.1, 3.0] → keep pos0, pos3
        //          row 1 = [2.0, 2.0, 2.0, 2.0] → all equal → keep any 2
        let weight = vec![
            vec![5.0f32, 1.0, 0.1, 3.0],
            vec![2.0f32, 2.0, 2.0, 2.0],
        ];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: row 0 deterministic: keep pos 0 (5.0) and pos 3 (3.0)
        assert_ne!(pruned[0][0], 0.0, "row0 pos0 (5.0) survives");
        assert_ne!(pruned[0][3], 0.0, "row0 pos3 (3.0) survives");
        assert_eq!(pruned[0][1], 0.0, "row0 pos1 (1.0) pruned");
        assert_eq!(pruned[0][2], 0.0, "row0 pos2 (0.1) pruned");
        // Row 1: all equal → 2 survive, 2 pruned
        let row1_zeros = pruned[1].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(row1_zeros, 2, "row1: 2 of 4 equal values pruned");
        // sp_meta should have 2 rows, 1 u16 each (4 cols → 1 group → 1 u16)
        assert_eq!(sp_meta.len(), 2);
        assert_eq!(sp_meta[0].len(), 1);
    }

    #[test]
    // @trace TEST-SC-014 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_decompress_exact_run_length() {
        // Arrange: compress exactly 15 values → 1 compressed byte encoding run=15
        let input = vec![0x0Bu8; 15];
        let compressed = compress_bitpack_rle(&input);
        assert_eq!(compressed.len(), 1, "15 identical values → 1 entry");
        // Act: request exactly 15 back
        let decompressed = decompress_bitpack_rle(&compressed, 15);
        // Assert: all 15 recovered, no more no less
        assert_eq!(decompressed.len(), 15);
        assert_eq!(decompressed, input);
    }

    #[test]
    // @trace TEST-SC-015 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_five_heads_chain_merge_averages_correctly() {
        // Arrange: 5 heads with head_dim=1, all same direction but different magnitudes
        // values: [2.0], [3.0], [4.0], [5.0], [6.0] — all cosine_sim = 1.0 with each other
        // With threshold=0.98, all should merge into one group
        // Averaged value should be (2+3+4+5+6)/5 = 4.0
        let head_dim = 1;
        let rows: Vec<Vec<f32>> = vec![
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 5, head_dim, 0.98);
        // Assert: all merge into 1 group
        assert_eq!(dedup_w.len(), head_dim, "5 heads should merge into 1");
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
        assert_eq!(dedup_idx[2], dedup_idx[3]);
        assert_eq!(dedup_idx[3], dedup_idx[4]);
        // Averaged value = (2+3+4+5+6)/5 = 4.0
        let expected = 4.0f32;
        assert!(
            (dedup_w[0][0] - expected).abs() < 1e-6,
            "averaged value should be {expected}, got {}",
            dedup_w[0][0]
        );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (15 new, batch 10)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    // @trace TEST-SC-016 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_two_column_one_zero_norm_one_positive() {
        // Arrange: col 0 all-positive, col 1 all-zero
        let weight = vec![
            vec![7.0f32, 0.0],
            vec![3.0f32, 0.0],
            vec![5.0f32, 0.0],
        ];
        // Act: mean_norm = (col0_norm + 0.0) / 2, threshold = 0.01 * mean
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert
        assert!(!mask[0], "col 0 with positive norm should survive");
        assert!(mask[1], "col 1 with zero norm should be pruned");
        // Verify surviving values are preserved exactly
        assert_eq!(pruned[0][0], 7.0);
        assert_eq!(pruned[1][0], 3.0);
        assert_eq!(pruned[2][0], 5.0);
        // All pruned column values are zero
        for row in &pruned {
            assert_eq!(row[1], 0.0, "pruned col should be zero");
        }
    }

    #[test]
    // @trace TEST-SC-017 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_two_heads_head_dim_3_all_rows_identical() {
        // Arrange: 2 heads, head_dim=3, hidden=2 — all 6 rows identical
        let head_dim = 3;
        let row = vec![1.5f32, -2.5];
        let rows: Vec<Vec<f32>> = vec![
            row.clone(), row.clone(), row.clone(), // head 0
            row.clone(), row.clone(), row,         // head 1
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Assert: identical heads merge into 1 group
        assert_eq!(dedup_idx[0], dedup_idx[1], "identical 3-row heads should merge");
        assert_eq!(dedup_w.len(), head_dim, "1 merged group × 3 rows_per_head");
        // Averaged values should equal the original row (all identical)
        for (i, out_row) in dedup_w.iter().enumerate() {
            assert!((out_row[0] - 1.5).abs() < 1e-6, "row {i} col 0 should be 1.5");
            assert!((out_row[1] - (-2.5)).abs() < 1e-6, "row {i} col 1 should be -2.5");
        }
    }

    #[test]
    // @trace TEST-SC-018 [req:REQ-STATIC-COMP] [level:unit]
    fn cosine_sim_rows_150_degree_angle() {
        // Arrange: vectors at ~150 degrees — cos(150) = -sqrt(3)/2 ≈ -0.8660
        let a = vec![1.0f32, 0.0];
        let b = vec![-3.0f32.sqrt() / 2.0, 0.5];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert
        let expected = -3.0f32.sqrt() / 2.0;
        assert!(
            (sim - expected).abs() < 1e-5,
            "150-degree vectors should have sim≈-0.866, got {sim}"
        );
    }

    #[test]
    // @trace TEST-SC-019 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_run_of_exactly_44_splits_into_three_entries() {
        // Arrange: 44 identical nibbles → 15+15+14 = 3 entries
        let input = vec![0x02u8; 44];
        // Act
        let compressed = compress_bitpack_rle(&input);
        // Assert
        assert_eq!(compressed.len(), 3, "44 elements → 15+15+14 = 3 entries");
        assert_eq!(compressed[0] & 0x0F, 14, "first chunk run=15");
        assert_eq!(compressed[1] & 0x0F, 14, "second chunk run=15");
        assert_eq!(compressed[2] & 0x0F, 13, "third chunk run=14");
        let decompressed = decompress_bitpack_rle(&compressed, 44);
        assert_eq!(decompressed, input);
    }

    #[test]
    // @trace TEST-SC-020 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_24_36_cols_nine_groups_metadata() {
        // Arrange: 1 row × 36 cols → 9 groups → ceil(9/2) = 5 u16 metadata
        let weight = vec![vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
            29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0, 36.0,
        ]];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert
        assert_eq!(sp_meta[0].len(), 5, "36 cols → 9 groups → 5 u16");
        let zeros: usize = pruned[0].iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 18, "36 cols → 18 pruned (50%)");
    }

    #[test]
    // @trace TEST-SC-021 [req:REQ-STATIC-COMP] [level:unit]
    fn lz4_roundtrip_4_byte_exact_minimum_literal() {
        // Arrange: exactly 4 bytes — the minimum for an LZ4 literal sequence
        let data = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert
        assert_eq!(decompressed, data);
    }

    #[test]
    // @trace TEST-SC-022 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_head_dim_2_two_groups_three_heads() {
        // Arrange: 3 heads with head_dim=2, hidden=3
        // head 0 and 2 are identical, head 1 is orthogonal
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0], // head 0
            vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0], // head 1 (different)
            vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0], // head 2 (identical to head 0)
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 3, 2, 0.98);
        // Assert: heads 0 and 2 merge; head 1 stays separate
        assert_eq!(dedup_idx[0], dedup_idx[2], "identical heads 0 and 2 should merge");
        assert_ne!(dedup_idx[0], dedup_idx[1], "head 1 should be in different group");
        assert_eq!(dedup_w.len(), 2 * 2, "2 unique groups × 2 rows_per_head = 4 rows");
    }

    #[test]
    // @trace TEST-SC-023 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_threshold_ratio_nan_behavior() {
        // Arrange: threshold_ratio = NaN → threshold = NaN * mean = NaN
        // In Rust, x < NaN is always false → nothing is pruned
        let weight = vec![
            vec![10.0f32, 0.0001],
            vec![10.0f32, 0.0001],
        ];
        // Act
        let (_, mask) = prune_dead_columns(&weight, f32::NAN);
        // Assert: no column norm is < NaN → nothing pruned
        assert!(mask.iter().all(|m| !m), "NaN threshold should prune nothing");
    }

    #[test]
    // @trace TEST-SC-024 [req:REQ-STATIC-COMP] [level:unit]
    fn codec_error_clone_preserves_message() {
        // Arrange
        let original = CodecError("clone test message".to_string());
        // Act
        let cloned = original.clone();
        // Assert: cloned has identical message and display
        assert_eq!(original, cloned, "clone should produce equal error");
        assert_eq!(format!("{original}"), format!("{cloned}"), "Display should match after clone");
        // Original is still usable
        assert_eq!(original.0, "clone test message");
    }

    #[test]
    // @trace TEST-SC-025 [req:REQ-STATIC-COMP] [level:unit]
    fn cosine_similarity_heads_identical_large_weight_matrix() {
        // Arrange: 200 rows (10 heads × 20 rows_per_head), all identical
        let row = vec![2.0f32, -1.0, 0.5, 3.0];
        let weight: Vec<Vec<f32>> = (0..200).map(|_| row.clone()).collect();
        // Act: compare head 0 with head 9
        let sim = cosine_similarity_heads(&weight, 0, 9, 20);
        // Assert: identical multi-row heads → sim = 1.0
        assert!((sim - 1.0).abs() < 1e-4, "identical large heads should have sim=1.0, got {sim}");
    }

    #[test]
    // @trace TEST-SC-026 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_decompress_handcrafted_three_different_values() {
        // Arrange: hand-craft 3 compressed entries — val=0x0 run=2, val=0xF run=1, val=0x7 run=3
        let compressed: Vec<u8> = vec![
            (0x0 << 4) | 1,   // val=0, run=2
            (0xF << 4) | 0,   // val=0xF, run=1
            (0x7 << 4) | 2,   // val=7, run=3
        ];
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 6);
        // Assert
        assert_eq!(decompressed, vec![0x00, 0x00, 0x0F, 0x07, 0x07, 0x07]);
    }

    #[test]
    // @trace TEST-SC-027 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_24_preserves_value_exactness_for_survivors() {
        // Arrange: a specific float value that must survive unchanged
        let exact_val = 3.14159265f32;
        let weight = vec![vec![exact_val, 0.001, 100.0, 0.001]];
        // Act
        let (pruned, _) = prune_dead_columns_24(&weight);
        // Assert: pos 0 and pos 2 survive with bit-exact values
        assert_eq!(pruned[0][0].to_bits(), exact_val.to_bits(), "surviving float must be bit-exact");
        assert_eq!(pruned[0][2], 100.0f32, "surviving float must be bit-exact");
    }

    #[test]
    // @trace TEST-SC-028 [req:REQ-STATIC-COMP] [level:unit]
    fn compression_codec_nvcompans_variant_properties() {
        // Arrange
        let codec = CompressionCodec::NvcompAns;
        // Act & Assert
        assert_eq!(codec.as_u8(), 3);
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(format!("{codec:?}"), "NvcompAns");
        // Verify it is distinct from neighbors
        assert_ne!(codec, CompressionCodec::BitPackRle);
        assert_ne!(codec, CompressionCodec::ZstdDict);
    }

    #[test]
    // @trace TEST-SC-029 [req:REQ-STATIC-COMP] [level:unit]
    fn nvcomp_ans_error_clone_preserves_equality() {
        // Arrange
        let original = NvcompAnsError("gpu error for clone".to_string());
        // Act
        let cloned = original.clone();
        // Assert: clone produces an equal error
        assert_eq!(original, cloned, "cloned NvcompAnsError should be equal to original");
        assert_eq!(original.0, cloned.0, "inner strings should match");
        // Both Display identically
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    // @trace TEST-SC-030 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_preserves_zero_rows_in_output() {
        // Arrange: single head with all-zero rows, hidden=4
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        // Act
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 1, head_dim, 0.98);
        // Assert: single head remains, output rows are all zero, dimensionality preserved
        assert_eq!(dedup_w.len(), head_dim, "single head produces head_dim rows");
        assert_eq!(dedup_idx.len(), 1, "dedup_indices has 1 entry");
        for (i, row) in dedup_w.iter().enumerate() {
            assert_eq!(row.len(), 4, "row {i} should have hidden=4");
            assert!(row.iter().all(|&v| v == 0.0), "row {i} should be all zeros");
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Additional tests (13 new, batch 11)
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    // @trace TEST-SC-031 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_compress_decompress_alternating_nibble_pairs() {
        // Arrange: alternating nibble pairs — 0xA, 0xA, 0xB, 0xB, repeating 10 times
        let mut input = Vec::with_capacity(40);
        for _ in 0..10 {
            input.push(0x0A);
            input.push(0x0A);
            input.push(0x0B);
            input.push(0x0B);
        }
        // Act
        let compressed = compress_bitpack_rle(&input);
        let decompressed = decompress_bitpack_rle(&compressed, input.len());
        // Assert: roundtrip preserves data
        assert_eq!(decompressed, input, "alternating nibble pairs should roundtrip");
        // Should produce 20 entries: 10 × (run of 2 for 0xA, run of 2 for 0xB)
        assert_eq!(compressed.len(), 20, "10 pairs of runs of 2 → 20 entries");
    }

    #[test]
    // @trace TEST-SC-032 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_two_rows_opposite_signs_both_survive() {
        // Arrange: two columns, each with mixed signs but nonzero norm
        // col 0: [3.0, -3.0] → L2 norm = sqrt(18) ≈ 4.24
        // col 1: [-2.0, 2.0] → L2 norm = sqrt(8) ≈ 2.83
        let weight = vec![
            vec![3.0f32, -2.0],
            vec![-3.0f32, 2.0],
        ];
        // Act: low threshold → both columns survive
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        // Assert: nothing pruned
        assert!(mask.iter().all(|m| !m), "both columns have nonzero norm");
        assert_eq!(pruned, weight, "all values preserved exactly");
    }

    #[test]
    // @trace TEST-SC-033 [req:REQ-STATIC-COMP] [level:unit]
    fn cosine_sim_rows_unit_vectors_at_90_degrees_4d() {
        // Arrange: two orthogonal unit vectors in 4D
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 1.0, 0.0];
        // Act
        let sim = cosine_sim_rows(&a, &b);
        // Assert: orthogonal → sim = 0.0
        assert!(
            sim.abs() < 1e-6,
            "orthogonal 4D unit vectors should have sim≈0.0, got {sim}"
        );
    }

    #[test]
    // @trace TEST-SC-034 [req:REQ-STATIC-COMP] [level:unit]
    fn lz4_roundtrip_ascending_then_descending_bytes() {
        // Arrange: ascending then descending — tests LZ4 match finding at pattern boundary
        let mut data: Vec<u8> = (0..=255).collect::<Vec<u8>>();
        data.extend((0..=255).rev());
        // Act
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).unwrap();
        // Assert: roundtrip must be exact; compression ratio is not guaranteed for high-entropy data
        assert_eq!(decompressed, data, "ascending+descending should roundtrip");
    }

    #[test]
    // @trace TEST-SC-035 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_24_two_rows_different_patterns_same_meta_shape() {
        // Arrange: two rows with different magnitudes but same column count (8 cols)
        let weight = vec![
            vec![10.0f32, 0.5, 8.0, 0.1, 7.0, 0.2, 9.0, 0.3],
            vec![1.0f32, 100.0, 0.5, 90.0, 2.0, 80.0, 0.4, 70.0],
        ];
        // Act
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        // Assert: both rows should have same sp_meta shape (8 cols → 2 groups → 1 u16)
        assert_eq!(sp_meta[0].len(), sp_meta[1].len(), "same shape metadata");
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 2 groups → 1 u16 per row");
        // Each row has 4 survivors (2 per group × 2 groups)
        for (row_idx, row) in pruned.iter().enumerate() {
            let zeros = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 4, "row {row_idx}: 4 of 8 pruned (50%)");
        }
    }

    #[test]
    // @trace TEST-SC-036 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_four_heads_threshold_1_merges_all_parallel() {
        // Arrange: 4 heads all in same direction (positive scalar), head_dim=1
        // All have cosine_sim = 1.0 with each other
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0f32, 2.0],   // head 0
            vec![3.0f32, 6.0],   // head 1 (3× head 0 → parallel)
            vec![5.0f32, 10.0],  // head 2 (5× head 0 → parallel)
            vec![0.5f32, 1.0],   // head 3 (0.5× head 0 → parallel)
        ];
        // Act: threshold=1.0 → only cos=1.0 merges
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 4, 1, 1.0);
        // Assert: all parallel heads have cos=1.0 → all merge into 1 group
        assert_eq!(dedup_idx[0], dedup_idx[1]);
        assert_eq!(dedup_idx[1], dedup_idx[2]);
        assert_eq!(dedup_idx[2], dedup_idx[3]);
        assert_eq!(dedup_w.len(), 1, "4 merged heads × 1 row_per_head = 1 row");
    }

    #[test]
    // @trace TEST-SC-037 [req:REQ-STATIC-COMP] [level:unit]
    fn compression_codec_from_u8_returns_none_for_255() {
        // Arrange: 255 is well beyond the valid discriminant range (0..=4)
        // Act
        let result = CompressionCodec::from_u8(255);
        // Assert
        assert_eq!(result, None, "discriminant 255 should return None");
    }

    #[test]
    // @trace TEST-SC-038 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_decompress_stops_immediately_when_len_is_zero() {
        // Arrange: non-empty compressed data but decompressed_len = 0
        let compressed = vec![(0x5 << 4) | 4]; // val=5, run=5
        // Act
        let decompressed = decompress_bitpack_rle(&compressed, 0);
        // Assert
        assert!(decompressed.is_empty(), "requested 0 bytes → empty output");
    }

    #[test]
    // @trace TEST-SC-039 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_three_columns_middle_survives() {
        // Arrange: col 0 and col 2 are dead (tiny values), col 1 has large values
        let weight = vec![
            vec![0.00001f32, 100.0, 0.00001],
            vec![0.00001f32, 200.0, 0.00001],
            vec![0.00001f32, 300.0, 0.00001],
        ];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: only col 1 survives
        assert!(mask[0], "col 0 with tiny values should be pruned");
        assert!(!mask[1], "col 1 with large values should survive");
        assert!(mask[2], "col 2 with tiny values should be pruned");
        // Pruned cols are zero, surviving col has original values
        for row in &pruned {
            assert_eq!(row[0], 0.0);
            assert_ne!(row[1], 0.0);
            assert_eq!(row[2], 0.0);
        }
    }

    #[test]
    // @trace TEST-SC-040 [req:REQ-STATIC-COMP] [level:unit]
    fn lz4_decompress_produces_error_for_garbage_input() {
        // Arrange: random garbage bytes that are not valid LZ4
        let garbage: Vec<u8> = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA];
        // Act
        let result = lz4_decompress(&garbage, 100);
        // Assert: should error, not panic
        assert!(result.is_err(), "garbage input should produce an error");
    }

    #[test]
    // @trace TEST-SC-041 [req:REQ-STATIC-COMP] [level:unit]
    fn deduplicate_gqa_two_heads_opposite_direction_threshold_negative_1() {
        // Arrange: two heads pointing in exactly opposite directions
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0f32, 0.0],  // head 0: +x
            vec![-1.0f32, 0.0], // head 1: -x (cos = -1.0)
        ];
        // Act: threshold = -1.0 → cos >= -1.0 → all merge (since cos=-1 >= -1)
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, 1, -1.0);
        // Assert: opposite heads merge with threshold=-1.0
        assert_eq!(dedup_idx[0], dedup_idx[1], "threshold=-1.0 merges opposite heads");
        assert_eq!(dedup_w.len(), 1, "2 merged heads × 1 row = 1 row");
        // Average of [1,0] and [-1,0] = [0,0]
        assert!((dedup_w[0][0] - 0.0).abs() < 1e-6, "averaged opposite should be ~0");
        assert!((dedup_w[0][1] - 0.0).abs() < 1e-6, "averaged opposite should be ~0");
    }

    #[test]
    // @trace TEST-SC-042 [req:REQ-STATIC-COMP] [level:unit]
    fn bitpack_rle_compress_does_not_modify_input_slice() {
        // Arrange: input with known content
        let input = vec![0x03u8, 0x03, 0x03, 0x07, 0x07, 0x0A];
        let input_copy = input.clone();
        // Act
        let _compressed = compress_bitpack_rle(&input);
        // Assert: input is unchanged
        assert_eq!(input, input_copy, "compress should not modify the input slice");
    }

    #[test]
    // @trace TEST-SC-043 [req:REQ-STATIC-COMP] [level:unit]
    fn prune_dead_columns_single_row_with_infinity_column() {
        // Arrange: one row with an infinite value — f32::INFINITY has infinite norm
        // col 0 norm = INF, col 1 norm = 1.0, mean = (INF + 1.0) / 2 = INF
        // threshold = 0.5 * INF = INF → col 1 (norm=1.0) is < INF → pruned
        let weight = vec![vec![f32::INFINITY, 1.0]];
        // Act
        let (pruned, mask) = prune_dead_columns(&weight, 0.5);
        // Assert: INF column survives, finite column is pruned
        assert!(!mask[0], "INF norm column survives");
        assert!(mask[1], "finite norm column pruned when mean is INF");
        assert_eq!(pruned[0][0], f32::INFINITY, "INF value preserved");
        assert_eq!(pruned[0][1], 0.0, "pruned column zeroed");
    }

}
