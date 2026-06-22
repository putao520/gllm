#[derive(Debug, Clone)]
pub struct TensorSlice<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Self {
        Self { dtype, shape, data }
    }
}

// upload_native_tensor removed — superseded by convert_tensor_to_f32 (pure conversion)
// + tier-aware upload in process_single_tensor.

/// Convert raw tensor bytes to f32, applying P4/P5 heuristics and layout normalization.
/// Returns the modified meta, the f32 data, and optional sparsity metadata.
fn convert_tensor_to_f32(
    meta: &TensorMeta,
    data: &[u8],
    _format: WeightFormat,
    explicit_transpose_hint: Option<bool>,
) -> Result<(TensorMeta, Vec<f32>, Option<Vec<Vec<u16>>>)> {
    // ARCH-LOADER-PARALLEL-CONVERT: Rayon-parallel dtype→f32 conversion.
    // For large models (e.g. Gemma 4 E2B 9.6 GB BF16), a single-threaded
    // `chunks_exact().map().collect()` takes 60-120s on 4.8B elements. The
    // parallel path pre-allocates the output Vec and uses `par_chunks_mut`
    // so each worker writes into its own disjoint slice — no synchronisation,
    // ~5-10s on a 20-core machine.
    let mut converted_f32: Vec<f32> = match meta.dtype {
        Dtype::F32 => parallel_bytes_to_f32_lossless(data)?,
        Dtype::F16 => parallel_half_to_f32::<half::f16>(data)?,
        Dtype::BF16 => parallel_half_to_f32::<half::bf16>(data)?,
        Dtype::F64 => parallel_f64_to_f32(data)?,
        _ => {
            return Err(LoaderError::Backend(format!(
                "cannot convert {:?} to f32 for heuristics",
                meta.dtype
            )));
        }
    };

    let mut cloned_meta = meta.clone();

    apply_ffn_sparsity_heuristic(&cloned_meta, &mut converted_f32);
    let sp_meta_opt = compress_24_sparsity_heuristic(&mut cloned_meta, &mut converted_f32);
    deduplicate_q_heads_heuristic(&cloned_meta, &mut converted_f32);

    // ARCH-ONNX-MATMUL-TRANSPOSE: ONNX MatMul weights are stored as [K, N]
    // (input_dim, output_dim), while gllm's GemmBias transB=true expects [N, K]
    // (output_dim, input_dim) — the same layout as HF SafeTensors.
    // When explicit_transpose_hint is Some(false), the weight comes from an
    // ONNX MatMul node and needs physical transpose to [N, K] canonical layout.
    // SafeTensors/Gemm-transB=1 weights are already [N, K] → no transpose needed.
    if explicit_transpose_hint == Some(false) {
        normalize_linear_weight_layout(&mut cloned_meta, &mut converted_f32);
    }

    Ok((cloned_meta, converted_f32, sp_meta_opt))
}

/// HF SafeTensors/PyTorch 的 Linear 权重 layout 归一化。
///
/// 问题: HF `nn.Linear.weight` 的内存布局是 `[out_features, in_features]` row-major,
/// 前向为 `y = x @ W.T`。但 gllm-kernels JIT GEMM 的 B 输入约定是 `[K, N]` row-major
/// (ONNX MatMul 语义, `y = x @ B`)。直接用 HF 布局会得到错误结果 (方阵下 shape 一致
/// 但数值错误), 非方阵时 N ≠ K 还会越界 SIGSEGV。
///
/// 根治: 加载边界统一把 HF `[out, in]` 物理转置成 canonical `[in, out]` 布局并更新
/// meta.shape。内部 op 只处理 canonical layout (ARCH-WEIGHT-CANONICAL-LAYOUT)。
///
/// 只对真正的 Linear 权重生效 — 排除 embedding / LayerNorm / bias / 非 2D tensor。
fn normalize_linear_weight_layout(meta: &mut TensorMeta, data: &mut Vec<f32>) {
    if !is_linear_weight(&meta.name, &meta.shape) {
        return;
    }
    let rows = meta.shape[0]; // out_features (HF)
    let cols = meta.shape[1]; // in_features (HF)
    if rows * cols != data.len() {
        log::warn!(
            "normalize_linear_weight_layout: '{}' shape {:?} does not match data len {}, skip",
            meta.name, meta.shape, data.len()
        );
        return;
    }
    // Row-major [rows, cols] → [cols, rows] via cache-blocked transpose.
    let mut out = vec![0.0f32; data.len()];
    cache_blocked_transpose_f32(data, &mut out, rows, cols);
    *data = out;
    meta.shape = vec![cols, rows]; // canonical [in, out] = [K, N]
}

/// Byte-level layout normalization for non-F32 float tensors (BF16/F16).
/// Same logic as `normalize_linear_weight_layout` but operates on raw bytes.
#[allow(dead_code)]
fn normalize_linear_weight_layout_bytes(meta: &mut TensorMeta, data: &mut Vec<u8>, elem_size: usize) {
    if !is_linear_weight(&meta.name, &meta.shape) {
        return;
    }
    let rows = meta.shape[0]; // out_features (HF)
    let cols = meta.shape[1]; // in_features (HF)
    let total_elems = data.len() / elem_size;
    if rows * cols != total_elems {
        log::warn!(
            "normalize_linear_weight_layout_bytes: '{}' shape {:?} does not match data len {} (elem_size={}), skip",
            meta.name, meta.shape, data.len(), elem_size
        );
        return;
    }
    let mut out = vec![0u8; data.len()];
    cache_blocked_transpose_bytes(data, &mut out, rows, cols, elem_size);
    *data = out;
    meta.shape = vec![cols, rows];
}

/// Cache-blocked (tiled) f32 transpose.
///
/// `src` is `[rows, cols]` row-major; `dst` is written as `[cols, rows]`
/// row-major (i.e. `dst[c * rows + r] = src[r * cols + c]`).
///
/// Naive transpose writes with a stride of `rows * 4` bytes → every store
/// misses L1 on typical weight shapes (e.g. 1536 × 12288 → 6144-byte stride).
/// Observed throughput is ~50-200 MB/s.
///
/// A tile-based transpose keeps `TILE × TILE` f32s in L1 (16 KB for 64×64)
/// so reads and writes within a tile both hit L1. We additionally use Rayon
/// to parallelise over the outer row-tile dimension so 20 cores can co-operate
/// on independent chunks of `dst`.
///
/// Both axes are processed in `TILE`-sized blocks; the tail rows/cols inside
/// the last tile are handled by `.min(rows)` / `.min(cols)` inside the inner
/// loops. Produces bit-identical output to a naive transpose for all finite
/// f32 values.
#[allow(dead_code)]
fn cache_blocked_transpose_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    const TILE: usize = 64;
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    // SAFETY: each parallel worker writes to a disjoint set of output rows.
    // We parallelise over `j_tile` (the column dimension of `src`, i.e. the
    // ROW dimension of `dst`). Thread `j_tile` only writes to
    // `dst[jj * rows .. (jj+1) * rows]` for `jj` in `[j_tile*TILE,
    // (j_tile+1)*TILE)`, which is a strictly disjoint row range across
    // threads. We cannot take `&mut [f32]` slices of arbitrary row ranges
    // across the parallel iterator (borrow checker can't prove disjointness
    // through a raw `for_each`), so we pass the base pointer as a `usize`
    // address (which is `Send + Sync`) and re-cast inside each closure.
    let dst_addr = dst.as_mut_ptr() as usize;

    let num_j_tiles = cols.div_ceil(TILE);
    (0..num_j_tiles).into_par_iter().for_each(|j_tile| {
        let j_start = j_tile * TILE;
        let j_end = (j_start + TILE).min(cols);
        let dst_base = dst_addr as *mut f32;
        for i_start in (0..rows).step_by(TILE) {
            let i_end = (i_start + TILE).min(rows);
            for ii in i_start..i_end {
                let src_row = ii * cols;
                for jj in j_start..j_end {
                    // dst[jj * rows + ii] = src[ii * cols + jj]
                    unsafe {
                        let v = *src.get_unchecked(src_row + jj);
                        *dst_base.add(jj * rows + ii) = v;
                    }
                }
            }
        }
    });
}

/// Cache-blocked byte-level transpose for arbitrary element sizes (BF16=2, F16=2, etc.).
/// Generalization of `cache_blocked_transpose_f32` where each element is `elem_size` bytes.
#[allow(dead_code)]
fn cache_blocked_transpose_bytes(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, elem_size: usize) {
    const TILE: usize = 64;
    let row_stride = cols * elem_size;
    let col_stride = rows * elem_size;
    let total = rows * cols * elem_size;
    debug_assert_eq!(src.len(), total);
    debug_assert_eq!(dst.len(), total);

    let dst_addr = dst.as_mut_ptr() as usize;
    let src_addr = src.as_ptr() as usize;

    let num_j_tiles = cols.div_ceil(TILE);
    (0..num_j_tiles).into_par_iter().for_each(|j_tile| {
        let j_start = j_tile * TILE;
        let j_end = (j_start + TILE).min(cols);
        let dst_base = dst_addr as *mut u8;
        let src_base = src_addr as *const u8;
        for i_start in (0..rows).step_by(TILE) {
            let i_end = (i_start + TILE).min(rows);
            for ii in i_start..i_end {
                let src_row_off = ii * row_stride;
                for jj in j_start..j_end {
                    let dst_off = jj * col_stride + ii * elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_base.add(src_row_off + jj * elem_size),
                            dst_base.add(dst_off),
                            elem_size,
                        );
                    }
                }
            }
        }
    });
}

/// Parallel byte-exact `&[u8]` → `Vec<f32>` (for native `F32`) using
/// `par_chunks_exact` so unaligned loads happen across all cores.
///
/// Returns `Err` if `data.len()` is not a multiple of `size_of::<f32>()`.
fn parallel_bytes_to_f32_lossless(data: &[u8]) -> Result<Vec<f32>> {
    let src_size = std::mem::size_of::<f32>();
    if !data.len().is_multiple_of(src_size) {
        return Err(LoaderError::Backend(format!(
            "F32 tensor data length {} is not a multiple of {}",
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    out.par_chunks_mut(1024)
        .zip(data.par_chunks_exact(src_size * 1024).with_min_len(1))
        .for_each(|(out_chunk, in_bytes)| {
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                // SAFETY: sub.len() == src_size, read_unaligned is always valid.
                out_chunk[i] = unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f32) };
            }
        });
    // Handle the tail chunks (where par_chunks_exact leaves a remainder on the
    // input side and par_chunks_mut may expose a smaller last chunk on the
    // output side). `par_chunks_exact` emits exact-sized chunks only — if the
    // total element count is not a multiple of 1024 we still need to cover the
    // final partial chunk manually.
    let completed = (n / 1024) * 1024;
    if completed < n {
        let tail_bytes = &data[completed * src_size..];
        for (i, sub) in tail_bytes.chunks_exact(src_size).enumerate() {
            out[completed + i] =
                unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f32) };
        }
    }
    Ok(out)
}

/// Parallel `&[u8]` → `Vec<f32>` for any 16-bit half-precision type
/// (`half::f16` or `half::bf16`).
pub fn parallel_half_to_f32<H>(data: &[u8]) -> Result<Vec<f32>>
where
    H: Copy + Send + Sync + 'static,
    H: HalfToF32,
{
    let src_size = std::mem::size_of::<H>();
    if !data.len().is_multiple_of(src_size) {
        return Err(LoaderError::Backend(format!(
            "{} tensor data length {} is not a multiple of {}",
            std::any::type_name::<H>(),
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let byte_start = chunk_idx * CHUNK * src_size;
            let byte_end = byte_start + out_chunk.len() * src_size;
            let in_bytes = &data[byte_start..byte_end];
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                // SAFETY: sub.len() == src_size = size_of::<H>().
                let v: H = unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const H) };
                out_chunk[i] = v.to_f32_fast();
            }
        });
    Ok(out)
}

/// Parallel `&[u8]` → `Vec<f32>` for `F64` (narrowing cast).
fn parallel_f64_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    let src_size = std::mem::size_of::<f64>();
    if !data.len().is_multiple_of(src_size) {
        return Err(LoaderError::Backend(format!(
            "F64 tensor data length {} is not a multiple of {}",
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let byte_start = chunk_idx * CHUNK * src_size;
            let byte_end = byte_start + out_chunk.len() * src_size;
            let in_bytes = &data[byte_start..byte_end];
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                let v: f64 =
                    unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f64) };
                out_chunk[i] = v as f32;
            }
        });
    Ok(out)
}

/// Internal trait bridging `half::f16` / `half::bf16` to their `to_f32`
/// implementation inside a generic context.
trait HalfToF32 {
    fn to_f32_fast(self) -> f32;
}

impl HalfToF32 for half::f16 {
    #[inline(always)]
    fn to_f32_fast(self) -> f32 { self.to_f32() }
}

impl HalfToF32 for half::bf16 {
    #[inline(always)]
    fn to_f32_fast(self) -> f32 { self.to_f32() }
}

/// Heuristic: 判断一个 2D tensor 是否是 Linear 权重 (需要 canonical layout 归一化)。
///
/// 两种命名模式均识别:
/// - HF 原始名: `xxx.weight` (以 `.weight` 结尾)
/// - gllm 规范名: `LN.xxx` 或 `classifier.xxx` 等 (不含 `.weight` 后缀)
///
/// 排除 embedding / norm / bias / 非 2D tensor。
fn is_linear_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    // Bias tensors are 1D — already excluded by shape.len() check above,
    // but also skip any name ending in ".bias" as a double-safety guard.
    if name.ends_with(".bias") {
        return false;
    }
    // Embedding weight 是 [vocab, hidden], 用于 Gather 不是 MatMul, 不能转置。
    let excluded_substrings = [
        "embeddings.word_embeddings",
        "embeddings.position_embeddings",
        "embeddings.token_type_embeddings",
        "wte.",                 // GPT-style word/token embedding
        "wpe.",                 // GPT-style position embedding
        "embed_tokens",         // Llama/Qwen/Mistral token embedding
        "token_embd",           // GGUF token embedding
        "LayerNorm",            // LayerNorm.weight (1D 已在 shape 检查排除, 双保险)
        "layer_norm",
        "RMSNorm",
        "rms_norm",
        ".norm.",
        "embed",                // canonical: "embed", "position_embed", "token_type_embed"
        "position_embed",
        "token_type_embed",
    ];
    for ex in &excluded_substrings {
        if name.contains(ex) {
            return false;
        }
    }
    // HF original name: must end with ".weight"
    // gllm canonical name: no ".weight" suffix but is a Linear weight
    //   (e.g. "L0.q_proj", "L0.up_proj", "classifier.dense", "classifier")
    if name.ends_with(".weight") {
        return true;
    }
    // Canonical names: LN.xxx pattern or classifier head
    if name.starts_with('L') && name.contains('.') {
        return true;
    }
    if name.starts_with("classifier") && !name.ends_with(".bias") {
        return true;
    }
    false
}

/// Applies Tier II structural sparsity heuristic on FFN matrices.
/// Identifies and outright zeroes columns (or rows) in `gate_proj` and `up_proj` whose 
/// L2-norm falls below `0.01 * mean_L2`. This structural nullification guarantees 
/// `gate_out` falls to 0.0 and skips dependent computations within `MaskedGemm`.
fn apply_ffn_sparsity_heuristic(meta: &TensorMeta, data: &mut [f32]) {
    if !meta.name.contains("mlp.gate_proj") && !meta.name.contains("mlp.up_proj") {
        return;
    }

    if meta.shape.len() != 2 {
        return;
    }

    let rows = meta.shape[0]; 
    let cols = meta.shape[1];

    let mut l2_norms = Vec::with_capacity(rows);
    let mut sum_l2 = 0.0;

    for r in 0..rows {
        let mut norm_sq = 0.0f32;
        let start = r * cols;
        for c in 0..cols {
            let val = data[start + c];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        l2_norms.push(norm);
        sum_l2 += norm;
    }

    let mean_l2 = sum_l2 / (rows as f32);
    let threshold = 0.01 * mean_l2;

    let mut pruned = 0;
    for (r, norm) in l2_norms.iter().enumerate() {
        if *norm < threshold {
            let start = r * cols;
            for c in 0..cols {
                data[start + c] = 0.0;
            }
            pruned += 1;
        }
    }

    if pruned > 0 {
        log::info!("🧠 Structural Sparsity: Nullified {}/{} rows in {}.", pruned, rows, meta.name);
    }
}

/// Applies NVIDIA 2:4 Structural Sparsity pattern on FFN matrices.
/// Enforces the 2:4 sparsity pattern structurally directly inside the tensor buffer 
/// at model load time to avoid any CPU overhead during the inference hot loop.
/// Shrinks the tensor dimension by 50% and returns the generated sp_meta for Phase D (Sparse MMA).
fn compress_24_sparsity_heuristic(meta: &mut TensorMeta, data: &mut Vec<f32>) -> Option<Vec<Vec<u16>>> {
    // 2:4 structural sparsity compression is ONLY valid for GPU Sparse MMA
    // (NVIDIA Ampere+). On CPU-only JIT builds, the dense GEMM expects full-
    // dimension weights. Compressing here causes the JIT GEMM to read past
    // the buffer boundary → SIGSEGV.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (meta, data);
        None
    }

    #[cfg(feature = "cuda")]
    {
    if !meta.name.contains("mlp.gate_proj") && !meta.name.contains("mlp.up_proj") && !meta.name.contains("experts") {
        return None;
    }
    if meta.shape.len() != 2 {
        return None;
    }
    let rows = meta.shape[0];
    let cols = meta.shape[1];
    if cols % 4 != 0 {
        return None;
    }

    // Convert flat data to Vec<Vec<f32>>
    let rows_data: Vec<Vec<f32>> = data.chunks(cols).map(|c| c.to_vec()).collect();
    let (pruned_rows, sp_meta) = crate::static_compression::prune_dead_columns_24(&rows_data);

    // 物理显存压实 (Physical Memory Shrink):
    // 虽然底层出于接口兼容返回了原尺寸的零填充张量，但在 Loader 我们强制将其抛弃。
    // 我们仅根据生成的 sp_meta 重建紧凑的 50% 内存块。
    let mut compressed_data = Vec::with_capacity(rows * (cols / 2));
    
    for (r_idx, row) in pruned_rows.iter().enumerate() {
        let meta_row = &sp_meta[r_idx];
        for grp in 0..(cols / 4) {
            let base = grp * 4;
            // Decode the 2-bit indices from sp_meta
            let meta_u16_idx = grp / 2;
            let meta_shift = (grp % 2) * 4;
            let encoded = (meta_row[meta_u16_idx] >> meta_shift) & 0x0F;
            
            let keep0 = (encoded & 0x03) as usize;
            let keep1 = ((encoded >> 2) & 0x03) as usize;
            
            compressed_data.push(row[base + keep0]);
            compressed_data.push(row[base + keep1]);
        }
    }

    // UPDATE the tensor shape to reflect the 50% compression!
    // Since columns were compressed by 50%
    meta.shape[1] = cols / 2;
    *data = compressed_data;

    Some(sp_meta)
    } // #[cfg(feature = "jit-cuda")]
}

/// Applies Tier II graph compression for Q-heads.
/// Evaluates cosine similarity between attention heads in `q_proj`. 
/// If `sim > 0.98`, the duplicate head is zeroed out to save VRAM and memory bandwidth,
/// and metadata is generated (conceptually) to scale the runtime accumulator.
fn deduplicate_q_heads_heuristic(meta: &TensorMeta, data: &mut [f32]) {
    if !meta.name.contains("q_proj") && !meta.name.contains("query") {
        return;
    }

    if meta.shape.len() != 2 {
        return;
    }

    let rows = meta.shape[0]; 
    let cols = meta.shape[1];

    // Infer head_dim conservatively (usually 128 or 64). 
    // If cols is not divisible by 128, try 64, else abort heuristic.
    let head_dim = if cols.is_multiple_of(128) { 128 } else if cols.is_multiple_of(64) { 64 } else { return; };
    let num_heads = cols / head_dim;

    if num_heads <= 1 {
        return;
    }

    // data layout: [rows, num_heads * head_dim]
    // A head is a set of columns. 
    // Let's compute the L2 norm for each head.
    let mut head_norms = vec![0.0f32; num_heads];
    for (h, norm_out) in head_norms.iter_mut().enumerate() {
        let mut sq_norm = 0.0f32;
        let start_col = h * head_dim;
        for r in 0..rows {
            let row_offset = r * cols;
            for d in 0..head_dim {
                let val = data[row_offset + start_col + d];
                sq_norm += val * val;
            }
        }
        *norm_out = sq_norm.sqrt();
    }

    let mut merged = 0;
    let mut active = vec![true; num_heads];

    for i in 0..num_heads {
        if !active[i] || head_norms[i] < 1e-6 { continue; }
        
        for j in (i + 1)..num_heads {
            if !active[j] || head_norms[j] < 1e-6 { continue; }

            // Compute dot product between head i and head j
            let mut dot = 0.0f32;
            let start_col_i = i * head_dim;
            let start_col_j = j * head_dim;

            for r in 0..rows {
                let row_offset = r * cols;
                for d in 0..head_dim {
                    let vi = data[row_offset + start_col_i + d];
                    let vj = data[row_offset + start_col_j + d];
                    dot += vi * vj;
                }
            }

            let sim = dot / (head_norms[i] * head_norms[j]);
            if sim > 0.98 {
                // Head j is extremely similar to Head i.
                // Zero out Head j to save memory bandwidth during loading to SRAM.
                for r in 0..rows {
                    let row_offset = r * cols;
                    for d in 0..head_dim {
                        data[row_offset + start_col_j + d] = 0.0;
                    }
                }
                active[j] = false;
                merged += 1;
            }
        }
    }

    if merged > 0 {
        log::info!("🧠 GQA Head Deduplication: Merged {}/{} Q-heads in {}.", merged, num_heads, meta.name);
    }
}

/// 量化配置的伴生张量信息
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompanionConfig {
    /// 量化 scales 张量名称
    pub scales: String,
    /// 量化 zeros 张量名称（可选，某些量化方案不需要）
    pub zeros: Option<String>,
}

/// 量化元数据
///
/// REQ-ARCH-Ω1: 量化配置必须包含完整信息，包括符号位和伴生张量
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// 量化分组大小（某些文档中称为 group_size，这里统一使用 block_size）
    pub block_size: usize,
    /// 量化位宽
    pub bits: u8,
    /// 是否使用激活值降序排列
    #[serde(default)]
    pub desc_act: bool,
    /// 是否使用对称量化
    #[serde(default)]
    pub is_sym: bool,
    /// 是否为有符号量化
    #[serde(default)]
    pub signed: bool,
    /// 伴生张量配置（scales/zeros）
    #[serde(default)]
    pub companions: Option<CompanionConfig>,
}

impl QuantizationMetadata {
    pub fn from_metadata(
        metadata: &HashMap<String, String>,
    ) -> Result<Option<HashMap<String, Self>>> {
        if let Some(json) = metadata.get("gllm.quantization") {
            let map: HashMap<String, Self> = serde_json::from_str(json)?;
            Ok(Some(map))
        } else {
            Ok(None)
        }
    }
}

// --- Legacy Types for Compatibility ---

/// Thinking head tensor names (for models like Qwen3 with thinking capability)
#[derive(Debug, Clone, Default)]
pub struct ThinkingHead {
    pub tensors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct WeightsHandle<B: Backend<E>, E: Element = f32> {
    tensors: HashMap<String, B::Tensor>,
    shapes: HashMap<String, Vec<usize>>,
    pub meta: HashMap<String, TensorMeta>,
    pub thinking_head: Option<ThinkingHead>,
    quantized: HashMap<String, QuantizedTensor>,
    raw_floats: HashMap<String, RawFloatTensor>,
    pub sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
    placements: HashMap<String, crate::compat::backend_trait::WeightPlacement>,
    /// COMP12: Compressed weight pages for cold-storage tiers (SPEC 22 §6).
    /// Populated during `upload_provider` when `WeightCompressionConfig` is set.
    pub compressed_weights: HashMap<String, weight_compress::CompressedWeightPage>,
}

impl<B: Backend<E>, E: Element> WeightsHandle<B, E> {
    pub fn new(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized: HashMap::new(),
            raw_floats: HashMap::new(),
            sparse_24_meta: HashMap::new(),
            placements: HashMap::new(),
            compressed_weights: HashMap::new(),
        }
    }

    pub fn new_with_quantized_and_sparse(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
        quantized: HashMap<String, QuantizedTensor>,
        sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized,
            raw_floats: HashMap::new(),
            sparse_24_meta,
            placements: HashMap::new(),
            compressed_weights: HashMap::new(),
        }
    }

    pub fn new_with_placements(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
        quantized: HashMap<String, QuantizedTensor>,
        raw_floats: HashMap<String, RawFloatTensor>,
        sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
        placements: HashMap<String, crate::compat::backend_trait::WeightPlacement>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized,
            raw_floats,
            sparse_24_meta,
            placements,
            compressed_weights: HashMap::new(),
        }
    }

    /// COMP12: Constructor that includes compressed weights.
    pub fn new_with_compressed(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
        quantized: HashMap<String, QuantizedTensor>,
        raw_floats: HashMap<String, RawFloatTensor>,
        sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
        placements: HashMap<String, crate::compat::backend_trait::WeightPlacement>,
        compressed_weights: HashMap<String, weight_compress::CompressedWeightPage>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized,
            raw_floats,
            sparse_24_meta,
            placements,
            compressed_weights,
        }
    }

    pub fn quantized_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.quantized.get(name)
    }

    pub fn raw_float_tensor(&self, name: &str) -> Option<&RawFloatTensor> {
        self.raw_floats.get(name)
    }

    pub fn raw_floats(&self) -> &HashMap<String, RawFloatTensor> {
        &self.raw_floats
    }

    pub fn is_quantized(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.tensors.get(name)
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }

    /// Return an iterator over all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// All external tensor names (F32 + quantized + BF16/F16 raw_floats).
    pub fn all_tensor_names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.extend(self.quantized.keys());
        names.extend(self.raw_floats.keys());
        names
    }

    /// Build a TensorNameMap without model_kind context.
    ///
    /// **Deprecated**: Use [`name_map_with_kind`] instead to ensure
    /// Reranker models correctly resolve classifier vs lm_head.
    /// BCE-20260619-001: model_kind=None may produce wrong canonical names for
    /// Reranker models.
    #[deprecated(note = "use name_map_with_kind instead")]
    pub fn name_map(&self) -> name_map::TensorNameMap {
        self.name_map_with_kind(None)
    }

    /// Build a TensorNameMap with model_kind context.
    ///
    /// When `model_kind` is `Some(ModelKind::Reranker)`, remaps `lm_head` → `classifier`
    /// because GGUF reranker models rename `score.weight` to `output.weight`.
    pub fn name_map_with_kind(&self, model_kind: Option<crate::manifest::ModelKind>) -> name_map::TensorNameMap {
        let all: Vec<String> = self.all_tensor_names().into_iter().cloned().collect();
        name_map::TensorNameMap::build_from_names(&all, model_kind)
    }

    /// Query the data placement of a tensor (DeviceLocal or HostLocal).
    pub fn placement_of(&self, name: &str) -> Option<crate::compat::backend_trait::WeightPlacement> {
        self.placements.get(name).copied()
    }

    /// ARCH-WEIGHT-BLOB-REMAPPING: 精确释放已编译到 weight_blob 中的权重张量。
    ///
    /// 只释放 `safe_to_release` 集合中的权重（这些权重的数据已完整拷贝到
    /// CompiledNode.weight_blob，运行时不再从 WeightsHandle.tensors 读取）。
    /// 其他权重（attention 节点的 q/k/v/o_proj，因 needs_runtime_weight_pack=true
    /// 而未预打包）必须保留。
    pub fn release_compiled_weights(&mut self, safe_to_release: &std::collections::HashSet<String>) {
        let before = self.tensors.len();
        self.tensors.retain(|name, _| {
            if safe_to_release.contains(name) {
                return false;
            }
            true
        });
        let after = self.tensors.len();
        if before > after {
            log::info!(
                "WeightsHandle::release_compiled_weights: released {}/{} tensors",
                before - after, before
            );
        }
    }

    /// TP 权重分片：按 ParallelConfig 对已加载权重执行 Tensor Parallelism 分片 (REQ-DIST-004)。
    ///
    /// 在 `init_distributed()` 后调用。对每个权重名称：
    /// 1. `infer_shard_strategy(name)` 推断分片策略 (ColumnParallel / RowParallel / None)
    /// 2. 策略为 None → 跳过（embedding/RMSNorm/RoPE 等不分片，所有 rank 持有完整副本）
    /// 3. 策略非 None → `shard_weight(data, rows, cols, config, strategy)` 执行分片
    ///
    /// 分片对象：
    /// - F32 张量 (`self.tensors` 中的 backend tensor): 下载到 host → shard → 重新上传
    /// - BF16/F16 张量 (`self.raw_floats` 中的 RawFloatTensor): 转为 f32 → shard → 转回原 dtype
    /// - 量化张量 (`self.quantized`): 量化权重按量化块边界分片（暂不支持，返回 Err）
    ///
    /// 当 `config.tp_size <= 1` 时立即返回 Ok(())（单机无需分片）。
    ///
    /// # Errors
    /// - 维度不能被 tp_size 整除时返回 Err
    /// - 量化张量分片暂不支持时返回 Err
    // @trace REQ-DIST-004 [entity:ENT-DIST-TP-SHARD] [dataflow:DF-DIST-002]
    #[cfg(feature = "nccl")]
    pub fn shard_for_tp(
        &mut self,
        config: &crate::engine::distributed_config::ParallelConfig,
    ) -> Result<(), String> {
        if config.tp_size <= 1 {
            return Ok(());
        }

        use crate::loader::weight_shard::{infer_shard_strategy, shard_weight, ShardStrategy};

        let mut sharded_count = 0usize;
        let mut replicated_count = 0usize;

        // ── Shard F32 raw_floats (BF16/F16) ──
        // BF16/F16 weights are in self.raw_floats as RawFloatTensor (Vec<u8>).
        // Convert to f32, shard, convert back.
        let raw_names: Vec<String> = self.raw_floats.keys().cloned().collect();
        for name in &raw_names {
            let strategy = infer_shard_strategy(name);
            match strategy {
                None => {
                    replicated_count += 1;
                    continue;
                }
                Some(s) => {
                    let raw = self.raw_floats.get_mut(name).ok_or_else(|| {
                        format!("shard_for_tp: raw_float '{}' not found", name)
                    })?;
                    let shape = raw.shape.clone();
                    if shape.len() != 2 {
                        return Err(format!(
                            "shard_for_tp: raw_float '{}' has non-2D shape {:?}",
                            name, shape
                        ));
                    }
                    let rows = shape[0];
                    let cols = shape[1];
                    let elem_bytes = raw.dtype.size();

                    // Convert raw bytes → f32
                    let mut f32_data = match raw.dtype {
                        Dtype::BF16 => crate::loader::parallel_half_to_f32::<half::bf16>(&raw.data)
                            .map_err(|e| format!("shard_for_tp: BF16→F32 conversion failed for '{}': {}", name, e))?,
                        Dtype::F16 => crate::loader::parallel_half_to_f32::<half::f16>(&raw.data)
                            .map_err(|e| format!("shard_for_tp: F16→F32 conversion failed for '{}': {}", name, e))?,
                        other => {
                            return Err(format!(
                                "shard_for_tp: unsupported raw_float dtype {:?} for '{}'",
                                other, name
                            ));
                        }
                    };

                    // Execute shard
                    shard_weight(&mut f32_data, rows, cols, config, s)?;

                    // Compute new shape
                    let new_shape = match s {
                        ShardStrategy::ColumnParallel => vec![rows, cols / config.tp_size as usize],
                        ShardStrategy::RowParallel => vec![rows / config.tp_size as usize, cols],
                    };

                    // Convert f32 → original dtype bytes
                    let new_bytes = match raw.dtype {
                        Dtype::BF16 => {
                            f32_data.iter()
                                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                                .collect()
                        }
                        Dtype::F16 => {
                            f32_data.iter()
                                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                                .collect()
                        }
                        _ => unreachable!(),
                    };

                    // Update shape in both raw_floats and shapes/meta HashMaps
                    raw.data = new_bytes;
                    raw.shape = new_shape.clone();
                    self.shapes.insert(name.clone(), new_shape.clone());
                    if let Some(m) = self.meta.get_mut(name) {
                        m.shape = new_shape;
                    }

                    sharded_count += 1;
                }
            }
        }

        // ── Shard quantized weights ──
        // Quantized weights require block-boundary-aware sharding.
        // For now, report an error if any quantized weight needs sharding.
        for name in self.quantized.keys() {
            let strategy = infer_shard_strategy(name);
            if strategy.is_some() {
                return Err(format!(
                    "shard_for_tp: quantized weight '{}' requires sharding but block-boundary-aware sharding is not yet supported",
                    name
                ));
            }
        }

        log::info!(
            "shard_for_tp: rank={}, tp_size={}, sharded={}, replicated={}, quantized_skipped={}",
            config.rank, config.tp_size, sharded_count, replicated_count,
            self.quantized.len(),
        );

        Ok(())
    }
}

/// Backward-compatible type alias for f32 weights.
pub type WeightsHandleF32<B> = WeightsHandle<B, f32>;

/// 实现 gllm_kernels::TensorLookup trait
impl<B: Backend<E>, E: Element> crate::compat::backend_trait::TensorLookup<E, B>
    for WeightsHandle<B, E>
{
    fn get_tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.tensor(name)
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        WeightsHandle::tensor_shape(self, name)
    }

    fn get_quantized(&self, name: &str) -> Option<&crate::loader::QuantizedTensor> {
        self.quantized.get(name)
    }

    fn available_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.extend(self.quantized.keys().cloned());
        names.extend(self.raw_floats.keys().cloned());
        names.sort();
        names
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParallelPolicy {
    pub enabled: bool,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone)]
pub struct UploadedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    // backend-specific handle
}

