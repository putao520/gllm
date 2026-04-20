//! MXFP4 pair detection for SafeTensors loaders.
//!
//! OpenAI gpt-oss-20b stores MoE expert weights as **two physical safetensors
//! tensors per logical mxfp4 weight**:
//!
//! - `<prefix>_blocks` — packed e2m1 nibbles (uint8). Two e2m1 floats per byte
//!   (low nibble = even index, high nibble = odd index).
//! - `<prefix>_scales` — one e8m0 byte per block (uint8, biased exponent).
//! - `<prefix>_bias`   — BF16 bias (native, **not** part of the mxfp4 pair).
//!
//! This is different from the GGUF MXFP4 layout (which interleaves
//! `[scale_byte | qs[block_size/2]]` per block in a single buffer). gllm's
//! existing cpu_backend Mxfp4 dequantize path is built around the GGUF layout,
//! so the SafeTensors loader **repacks** the pair into the GGUF-style
//! interleaved buffer at `load_tensor_data()` time, hiding the split layout
//! from the rest of the inference pipeline.
//!
//! # Detection contract
//!
//! A pair is recognized iff:
//! - There exist two tensor names `<prefix>_blocks` and `<prefix>_scales`
//!   sharing the exact same `<prefix>`.
//! - Both tensors are stored as `Dtype::U8`.
//! - `len(blocks_bytes) == len(scales_bytes) * (block_size / 2)` for the
//!   inferred `block_size` (default 32, OCP standard for mxfp4).
//!
//! # Sibling bias
//!
//! gpt-oss MoE expert tensors also carry a `<prefix>_bias` BF16 tensor (e.g.
//! `gate_up_proj_bias`). The bias is **not** part of the mxfp4 pair — it is a
//! plain native float tensor that the loader handles through the regular
//! upload path. We record `bias_name` only as a convenience for downstream
//! consumers (graph templates referencing the bias).

use std::collections::HashMap;

use safetensors::Dtype;

/// OCP-standard mxfp4 block size (elements per block).
///
/// OpenAI gpt-oss-20b uses block_size = 32, which gives 16 bytes of packed
/// nibbles + 1 e8m0 scale byte per block. The constant matches
/// [`gllm_kernels::quant::QuantType::Mxfp4`] default in the rest of the
/// codebase. Detection logic validates the actual byte ratio against this
/// value; mismatched files are rejected.
pub const DEFAULT_MXFP4_BLOCK_SIZE: usize = 32;

/// Suffix marking the packed-nibble half of an mxfp4 pair.
pub const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";

/// Suffix marking the e8m0 scales half of an mxfp4 pair.
pub const MXFP4_SCALES_SUFFIX: &str = "_scales";

/// Suffix marking the BF16 bias sibling (not part of the mxfp4 pair).
pub const MXFP4_BIAS_SUFFIX: &str = "_bias";

/// One logical mxfp4 tensor reconstructed from a safetensors `_blocks` /
/// `_scales` pair.
///
/// `blocks_name` is treated as the canonical (logical) name of the mxfp4
/// tensor — graph templates that reference the packed-expert weight via this
/// name will see an mxfp4-quantized tensor, while the `scales_name` tensor is
/// hidden from the regular `iter_tensors()` enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp4Pair {
    /// Tensor name carrying the packed e2m1 nibbles (canonical logical name).
    pub blocks_name: String,
    /// Tensor name carrying the per-block e8m0 scale bytes (sidecar).
    pub scales_name: String,
    /// Block size (elements per block) — OCP standard is 32.
    pub block_size: usize,
    /// Number of mxfp4 blocks (= length of the scales tensor in bytes).
    pub num_blocks: usize,
    /// Original packed-blocks tensor shape (kept verbatim so downstream layout
    /// — e.g. `[num_experts, ...]` for per-expert dispatch — is preserved).
    pub blocks_shape: Vec<usize>,
    /// Optional BF16 bias sibling (`<prefix>_bias`) — not part of the pair,
    /// recorded for diagnostic / template-binding convenience only.
    pub bias_name: Option<String>,
}

/// Map from `blocks_name` (canonical logical name of the mxfp4 tensor) to its
/// pairing metadata.
pub type Mxfp4PairMap = HashMap<String, Mxfp4Pair>;

/// Set of `_scales` tensor names that have been claimed as sidecars and must
/// be hidden from the regular tensor enumeration / upload path.
pub type Mxfp4ScalesSidecarSet = std::collections::HashSet<String>;

/// Minimal tensor descriptor used by the pairing scanner.
#[derive(Debug, Clone)]
pub struct CandidateTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

/// Result of scanning a tensor list for mxfp4 pairs.
#[derive(Debug, Default)]
pub struct Mxfp4PairScan {
    /// `blocks_name` → pairing metadata.
    pub pairs: Mxfp4PairMap,
    /// `blocks_name` → `scales_name` (the spec calls for this exact map).
    pub blocks_to_scales: HashMap<String, String>,
    /// `_scales` tensor names that must not appear in regular enumeration.
    pub sidecars: Mxfp4ScalesSidecarSet,
}

/// Detect mxfp4 `_blocks` / `_scales` pairs in the given tensor list.
///
/// Detection rules (all must hold for a pair to be recognized):
/// 1. Both tensors are `Dtype::U8`.
/// 2. The names share the exact same prefix `<prefix>`, ending in
///    `_blocks` and `_scales` respectively.
/// 3. `blocks.byte_len == scales.byte_len * (block_size / 2)`, with
///    `block_size = DEFAULT_MXFP4_BLOCK_SIZE` (32).
///
/// A `_bias` sibling sharing the same prefix is recorded but **not** treated
/// as part of the mxfp4 pair (it is a plain BF16 native tensor).
///
/// Lone `_blocks` or lone `_scales` tensors (without a matching counterpart
/// satisfying the contract) are left untouched — downstream sees them as
/// regular U8 tensors. This is conservative: a malformed checkpoint surfaces
/// as a binding/inference error rather than silently re-interpreting bytes.
pub fn scan_mxfp4_pairs<I>(tensors: I) -> Mxfp4PairScan
where
    I: IntoIterator<Item = CandidateTensor>,
{
    // Index by name for O(1) lookup; also keep insertion order for deterministic
    // results (HashMap iteration is unordered, but pair detection only needs
    // hash lookups so order doesn't affect correctness — sorting is the
    // caller's responsibility for any user-visible enumeration).
    let mut by_name: HashMap<String, CandidateTensor> = HashMap::new();
    for t in tensors {
        by_name.insert(t.name.clone(), t);
    }

    let mut scan = Mxfp4PairScan::default();
    let block_size = DEFAULT_MXFP4_BLOCK_SIZE;
    let bytes_per_block = block_size / 2;

    for (name, blocks_t) in &by_name {
        let Some(prefix) = name.strip_suffix(MXFP4_BLOCKS_SUFFIX) else {
            continue;
        };
        if blocks_t.dtype != Dtype::U8 {
            continue;
        }

        let scales_name = format!("{prefix}{MXFP4_SCALES_SUFFIX}");
        let Some(scales_t) = by_name.get(&scales_name) else {
            continue;
        };
        if scales_t.dtype != Dtype::U8 {
            continue;
        }

        let num_blocks = scales_t.byte_len;
        let expected_block_bytes = num_blocks * bytes_per_block;
        if blocks_t.byte_len != expected_block_bytes {
            // Either a non-default block_size or a malformed checkpoint. We
            // refuse to silently guess block_size — log a warning (best-effort)
            // and skip this pair so the loader does not produce wrong bytes.
            log::warn!(
                "mxfp4_pairing: rejecting pair {{blocks: '{name}' ({} bytes), \
                 scales: '{scales_name}' ({} bytes)}}: expected blocks_bytes \
                 == scales_bytes * (block_size/2) = {} for block_size={block_size}, \
                 actual = {}",
                blocks_t.byte_len,
                scales_t.byte_len,
                expected_block_bytes,
                blocks_t.byte_len,
            );
            continue;
        }

        let bias_name = {
            let candidate = format!("{prefix}{MXFP4_BIAS_SUFFIX}");
            if by_name.contains_key(&candidate) {
                Some(candidate)
            } else {
                None
            }
        };

        let pair = Mxfp4Pair {
            blocks_name: name.clone(),
            scales_name: scales_name.clone(),
            block_size,
            num_blocks,
            blocks_shape: blocks_t.shape.clone(),
            bias_name,
        };
        scan.pairs.insert(name.clone(), pair);
        scan.blocks_to_scales.insert(name.clone(), scales_name.clone());
        scan.sidecars.insert(scales_name);
    }

    scan
}

/// Repack a SafeTensors `_blocks` + `_scales` pair into the GGUF-style
/// interleaved layout `[scale_byte | qs[block_size/2]]` per block.
///
/// The cpu_backend `Mxfp4` dequantize path (and the rest of the inference
/// pipeline) is built around this single-buffer layout, so the loader emits
/// it as the canonical bytes for the logical mxfp4 tensor. Two consequences:
///
/// 1. Output length is `num_blocks * (1 + block_size / 2)` bytes
///    (= 17 bytes/block for `block_size = 32`).
/// 2. The `_scales` tensor is consumed by this repack and hidden from the
///    regular tensor enumeration to avoid a stale duplicate upload.
///
/// # Errors
/// Returns `Err` with a descriptive message if the input lengths are
/// inconsistent with the pair metadata. The check is total — there is no
/// silent truncation or padding.
pub fn repack_to_gguf_layout(
    pair: &Mxfp4Pair,
    blocks_bytes: &[u8],
    scales_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let bytes_per_block = pair.block_size / 2;
    if scales_bytes.len() != pair.num_blocks {
        return Err(format!(
            "mxfp4 repack: scales length mismatch for '{}': expected {} bytes \
             (one per block), got {}",
            pair.blocks_name,
            pair.num_blocks,
            scales_bytes.len(),
        ));
    }
    if blocks_bytes.len() != pair.num_blocks * bytes_per_block {
        return Err(format!(
            "mxfp4 repack: blocks length mismatch for '{}': expected {} bytes \
             ({} blocks × {} bytes/block), got {}",
            pair.blocks_name,
            pair.num_blocks * bytes_per_block,
            pair.num_blocks,
            bytes_per_block,
            blocks_bytes.len(),
        ));
    }

    let block_bytes = 1 + bytes_per_block;
    let mut out = Vec::with_capacity(pair.num_blocks * block_bytes);
    for blk in 0..pair.num_blocks {
        out.push(scales_bytes[blk]);
        let qs_start = blk * bytes_per_block;
        out.extend_from_slice(&blocks_bytes[qs_start..qs_start + bytes_per_block]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, dtype: Dtype, shape: Vec<usize>, byte_len: usize) -> CandidateTensor {
        CandidateTensor { name: name.to_string(), dtype, shape, byte_len }
    }

    #[test]
    fn detects_well_formed_pair_with_bias() {
        // num_experts=2, intermediate*2=4, hidden=64 → num_blocks = 2*4*64/32 = 16
        // blocks_bytes = 16 * 16 = 256 ; scales_bytes = 16
        let tensors = vec![
            cand(
                "model.layers.0.mlp.experts.gate_up_proj_blocks",
                Dtype::U8,
                vec![2, 4, 32], // arbitrary shape preserving total = 256
                256,
            ),
            cand(
                "model.layers.0.mlp.experts.gate_up_proj_scales",
                Dtype::U8,
                vec![2, 4, 2],
                16,
            ),
            cand(
                "model.layers.0.mlp.experts.gate_up_proj_bias",
                Dtype::BF16,
                vec![2, 4],
                16, // 8 bf16 elements * 2 bytes
            ),
        ];

        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1, "expected one mxfp4 pair");
        let pair = scan
            .pairs
            .get("model.layers.0.mlp.experts.gate_up_proj_blocks")
            .expect("pair indexed by _blocks name");
        assert_eq!(pair.scales_name, "model.layers.0.mlp.experts.gate_up_proj_scales");
        assert_eq!(pair.bias_name.as_deref(), Some("model.layers.0.mlp.experts.gate_up_proj_bias"));
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.num_blocks, 16);
        assert_eq!(pair.blocks_shape, vec![2, 4, 32]);

        assert!(scan
            .sidecars
            .contains("model.layers.0.mlp.experts.gate_up_proj_scales"));
        assert!(!scan
            .sidecars
            .contains("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        // Bias must NOT be hidden — it's a regular BF16 tensor.
        assert!(!scan
            .sidecars
            .contains("model.layers.0.mlp.experts.gate_up_proj_bias"));

        assert_eq!(
            scan.blocks_to_scales
                .get("model.layers.0.mlp.experts.gate_up_proj_blocks")
                .map(String::as_str),
            Some("model.layers.0.mlp.experts.gate_up_proj_scales"),
        );
    }

    #[test]
    fn ignores_unpaired_blocks_tensor() {
        // _blocks present but no matching _scales → do not pair.
        let tensors = vec![cand("foo.gate_up_proj_blocks", Dtype::U8, vec![16], 16)];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn ignores_unpaired_scales_tensor() {
        let tensors = vec![cand("foo.gate_up_proj_scales", Dtype::U8, vec![1], 1)];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn rejects_pair_with_inconsistent_byte_ratio() {
        // blocks_bytes != scales_bytes * 16 → reject (do not silently mis-pair).
        let tensors = vec![
            cand("a.gate_up_proj_blocks", Dtype::U8, vec![100], 100),
            cand("a.gate_up_proj_scales", Dtype::U8, vec![10], 10),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn rejects_pair_when_dtypes_are_not_u8() {
        let tensors = vec![
            cand("a.gate_up_proj_blocks", Dtype::BF16, vec![16], 32),
            cand("a.gate_up_proj_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn detects_multiple_pairs_independently() {
        // Two layers, each with its own mxfp4 pair.
        let tensors = vec![
            cand("L0.gate_up_proj_blocks", Dtype::U8, vec![16], 16),
            cand("L0.gate_up_proj_scales", Dtype::U8, vec![1], 1),
            cand("L0.down_proj_blocks", Dtype::U8, vec![32], 32),
            cand("L0.down_proj_scales", Dtype::U8, vec![2], 2),
            cand("L1.gate_up_proj_blocks", Dtype::U8, vec![48], 48),
            cand("L1.gate_up_proj_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 3);
        assert_eq!(scan.sidecars.len(), 3);
        for blocks in [
            "L0.gate_up_proj_blocks",
            "L0.down_proj_blocks",
            "L1.gate_up_proj_blocks",
        ] {
            assert!(scan.pairs.contains_key(blocks));
            assert!(!scan.sidecars.contains(blocks));
        }
        for scales in [
            "L0.gate_up_proj_scales",
            "L0.down_proj_scales",
            "L1.gate_up_proj_scales",
        ] {
            assert!(scan.sidecars.contains(scales));
        }
    }

    #[test]
    fn repack_produces_gguf_interleaved_layout() {
        // 2 blocks of 4 elements each (block_size = 4 → 2 bytes packed nibbles).
        // Use block_size = 4 by manually constructing a Pair (bypassing scan).
        let pair = Mxfp4Pair {
            blocks_name: "x_blocks".to_string(),
            scales_name: "x_scales".to_string(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![2, 2],
            bias_name: None,
        };
        let blocks = vec![0xAB, 0xCD, 0xEF, 0x12]; // 2 blocks × 2 bytes
        let scales = vec![0x7F, 0x80]; // 2 scale bytes
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).expect("repack ok");
        // Expected interleaved layout: [scale|qs|qs | scale|qs|qs]
        assert_eq!(out, vec![0x7F, 0xAB, 0xCD, 0x80, 0xEF, 0x12]);
    }

    #[test]
    fn repack_rejects_inconsistent_lengths() {
        let pair = Mxfp4Pair {
            blocks_name: "x_blocks".to_string(),
            scales_name: "x_scales".to_string(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![4, 16],
            bias_name: None,
        };
        // blocks_bytes too short
        let err = repack_to_gguf_layout(&pair, &[0u8; 10], &[0u8; 4]).unwrap_err();
        assert!(err.contains("blocks length mismatch"), "err: {err}");
        // scales_bytes wrong
        let err = repack_to_gguf_layout(&pair, &[0u8; 64], &[0u8; 5]).unwrap_err();
        assert!(err.contains("scales length mismatch"), "err: {err}");
    }
}
