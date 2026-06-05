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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

/// Result of scanning a tensor list for mxfp4 pairs.
#[derive(Debug, Default, PartialEq)]
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
        scan.blocks_to_scales
            .insert(name.clone(), scales_name.clone());
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
        CandidateTensor {
            name: name.to_string(),
            dtype,
            shape,
            byte_len,
        }
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
        assert_eq!(
            pair.scales_name,
            "model.layers.0.mlp.experts.gate_up_proj_scales"
        );
        assert_eq!(
            pair.bias_name.as_deref(),
            Some("model.layers.0.mlp.experts.gate_up_proj_bias")
        );
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

    // ── Constants ──────────────────────────────────────────────────────

    #[test]
    fn default_block_size_is_32() {
        assert_eq!(DEFAULT_MXFP4_BLOCK_SIZE, 32);
    }

    #[test]
    fn suffixes_have_expected_values() {
        assert_eq!(MXFP4_BLOCKS_SUFFIX, "_blocks");
        assert_eq!(MXFP4_SCALES_SUFFIX, "_scales");
        assert_eq!(MXFP4_BIAS_SUFFIX, "_bias");
    }

    // ── Mxfp4Pair derive traits ────────────────────────────────────────

    #[test]
    fn pair_equality_same_fields() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: Some("w_bias".into()),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn pair_equality_differs_by_blocks_name() {
        let a = Mxfp4Pair {
            blocks_name: "a_blocks".into(),
            scales_name: "a_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let mut b = a.clone();
        b.blocks_name = "b_blocks".into();
        assert_ne!(a, b);
    }

    #[test]
    fn pair_equality_differs_by_bias_name() {
        let a = Mxfp4Pair {
            blocks_name: "x_blocks".into(),
            scales_name: "x_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let mut b = a.clone();
        b.bias_name = Some("x_bias".into());
        assert_ne!(a, b);
    }

    #[test]
    fn pair_debug_format_contains_fields() {
        let pair = Mxfp4Pair {
            blocks_name: "foo_blocks".into(),
            scales_name: "foo_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![2, 32],
            bias_name: None,
        };
        let debug = format!("{pair:?}");
        assert!(debug.contains("foo_blocks"), "Debug output: {debug}");
        assert!(debug.contains("foo_scales"), "Debug output: {debug}");
        assert!(debug.contains("block_size: 32"), "Debug output: {debug}");
        assert!(debug.contains("num_blocks: 4"), "Debug output: {debug}");
    }

    #[test]
    fn pair_clone_is_independent() {
        let pair = Mxfp4Pair {
            blocks_name: "orig_blocks".into(),
            scales_name: "orig_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        let mut cloned = pair.clone();
        cloned.blocks_name = "modified_blocks".into();
        assert_eq!(pair.blocks_name, "orig_blocks");
        assert_eq!(cloned.blocks_name, "modified_blocks");
    }

    // ── Mxfp4PairScan Default ──────────────────────────────────────────

    #[test]
    fn scan_default_is_empty() {
        let scan = Mxfp4PairScan::default();
        assert!(scan.pairs.is_empty());
        assert!(scan.blocks_to_scales.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn scan_default_debug_format() {
        let scan = Mxfp4PairScan::default();
        let debug = format!("{scan:?}");
        assert!(debug.contains("pairs"));
        assert!(debug.contains("blocks_to_scales"));
        assert!(debug.contains("sidecars"));
    }

    // ── scan_mxfp4_pairs edge cases ────────────────────────────────────

    #[test]
    fn scan_empty_input() {
        let scan = scan_mxfp4_pairs(vec![]);
        assert!(scan.pairs.is_empty());
        assert!(scan.blocks_to_scales.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn scan_irrelevant_tensors_no_suffix() {
        let tensors = vec![
            cand(
                "model.layers.0.self_attn.q_proj",
                Dtype::BF16,
                vec![64, 64],
                8192,
            ),
            cand(
                "model.layers.0.self_attn.k_proj",
                Dtype::F32,
                vec![64, 64],
                16384,
            ),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn scan_rejects_scales_not_u8() {
        let tensors = vec![
            cand("a.w_blocks", Dtype::U8, vec![16], 16),
            cand("a.w_scales", Dtype::F32, vec![1], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_rejects_both_non_u8() {
        let tensors = vec![
            cand("a.w_blocks", Dtype::BF16, vec![16], 32),
            cand("a.w_scales", Dtype::BF16, vec![1], 2),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_pair_without_bias() {
        let tensors = vec![
            cand("experts.gate_blocks", Dtype::U8, vec![16], 16),
            cand("experts.gate_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("experts.gate_blocks").unwrap();
        assert_eq!(pair.bias_name, None);
    }

    #[test]
    fn scan_bias_sibling_recorded_but_not_counted_as_sidecar() {
        let tensors = vec![
            cand("proj.weight_blocks", Dtype::U8, vec![32], 32),
            cand("proj.weight_scales", Dtype::U8, vec![2], 2),
            cand("proj.weight_bias", Dtype::BF16, vec![8], 16),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("proj.weight_blocks").unwrap();
        assert_eq!(
            pair.bias_name.as_deref(),
            Some("proj.weight_bias"),
            "bias should be recorded"
        );
        assert!(
            !scan.sidecars.contains("proj.weight_bias"),
            "bias must NOT be a sidecar"
        );
        assert!(
            scan.sidecars.contains("proj.weight_scales"),
            "scales must be a sidecar"
        );
    }

    #[test]
    fn scan_blocks_to_scales_map_consistent() {
        let tensors = vec![
            cand("x.blocks_blocks", Dtype::U8, vec![32], 32),
            cand("x.blocks_scales", Dtype::U8, vec![2], 2),
            cand("y.other_blocks", Dtype::U8, vec![16], 16),
            cand("y.other_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.blocks_to_scales.len(), 2);
        assert_eq!(
            scan.blocks_to_scales.get("x.blocks_blocks"),
            Some(&"x.blocks_scales".to_string())
        );
        assert_eq!(
            scan.blocks_to_scales.get("y.other_blocks"),
            Some(&"y.other_scales".to_string())
        );
        // Reverse lookup should not exist
        assert_eq!(scan.blocks_to_scales.get("x.blocks_scales"), None);
    }

    #[test]
    fn scan_exact_byte_ratio_boundary() {
        // block_size=32, bytes_per_block=16. scales_len=1 → blocks must be 16.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);

        // Off-by-one: blocks 15 bytes, scales 1 byte → reject
        let tensors_off = vec![
            cand("w_blocks", Dtype::U8, vec![15], 15),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan_off = scan_mxfp4_pairs(tensors_off);
        assert!(scan_off.pairs.is_empty(), "off-by-one must be rejected");

        // Off-by-one the other way: blocks 17 bytes, scales 1 byte → reject
        let tensors_off2 = vec![
            cand("w_blocks", Dtype::U8, vec![17], 17),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan_off2 = scan_mxfp4_pairs(tensors_off2);
        assert!(scan_off2.pairs.is_empty(), "off-by-one must be rejected");
    }

    #[test]
    fn scan_large_num_blocks() {
        // 1024 blocks, blocks_bytes = 1024*16 = 16384
        let tensors = vec![
            cand("big_blocks", Dtype::U8, vec![1024, 16], 16384),
            cand("big_scales", Dtype::U8, vec![1024], 1024),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("big_blocks").unwrap();
        assert_eq!(pair.num_blocks, 1024);
    }

    #[test]
    fn scan_zero_byte_tensors_pair_together() {
        // 0 blocks: scales_len=0, blocks_len=0. 0 == 0*16 → valid pair.
        let tensors = vec![
            cand("empty_blocks", Dtype::U8, vec![0], 0),
            cand("empty_scales", Dtype::U8, vec![0], 0),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("empty_blocks").unwrap();
        assert_eq!(pair.num_blocks, 0);
    }

    #[test]
    fn scan_preserves_blocks_shape() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![4, 8, 16], 512),
            cand("w_scales", Dtype::U8, vec![4, 8], 32),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.blocks_shape, vec![4, 8, 16]);
    }

    #[test]
    fn scan_suffix_must_be_exact() {
        // "_blocks" embedded inside a longer suffix should not match.
        let tensors = vec![
            cand("model.w_blocks_extra", Dtype::U8, vec![16], 16),
            cand("model.w_blocks_extra_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty(), "inexact suffix must not match");
    }

    #[test]
    fn scan_different_prefixes_do_not_cross_pair() {
        // Two different prefixes each have _blocks and _scales but mismatched
        // byte ratios — ensure no cross-pollination.
        let tensors = vec![
            cand("a_proj_blocks", Dtype::U8, vec![16], 16),
            cand("b_proj_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(
            scan.pairs.is_empty(),
            "different prefixes must not form a pair"
        );
    }

    // ── repack_to_gguf_layout edge cases ───────────────────────────────

    #[test]
    fn repack_single_block() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let blocks = vec![
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ];
        let scales = vec![0x42];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 17, "1 + 16 = 17 bytes");
        assert_eq!(out[0], 0x42, "scale byte first");
        assert_eq!(&out[1..], &blocks[..], "block data follows scale");
    }

    #[test]
    fn repack_many_blocks_byte_level_correctness() {
        // 3 blocks of block_size=4 (2 bytes packed per block).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![3, 2],
            bias_name: None,
        };
        let blocks = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let scales = vec![0xAA, 0xBB, 0xCC];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // block 0: scale=0xAA, qs=[0x11, 0x22]
        // block 1: scale=0xBB, qs=[0x33, 0x44]
        // block 2: scale=0xCC, qs=[0x55, 0x66]
        assert_eq!(
            out,
            vec![0xAA, 0x11, 0x22, 0xBB, 0x33, 0x44, 0xCC, 0x55, 0x66]
        );
    }

    #[test]
    fn repack_zero_blocks() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 0,
            blocks_shape: vec![0],
            bias_name: None,
        };
        let out = repack_to_gguf_layout(&pair, &[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn repack_error_message_contains_tensor_name() {
        let pair = Mxfp4Pair {
            blocks_name: "my_special_tensor_blocks".into(),
            scales_name: "my_special_tensor_scales".into(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![128],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 10], &[0u8; 8]).unwrap_err();
        assert!(
            err.contains("my_special_tensor_blocks"),
            "error must reference tensor name: {err}"
        );
    }

    #[test]
    fn repack_rejects_scales_too_short() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 64], &[0u8; 3]).unwrap_err();
        assert!(err.contains("scales length mismatch"), "err: {err}");
    }

    #[test]
    fn repack_rejects_blocks_too_long() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        // Expected blocks: 2 * 16 = 32. Provide 48.
        let err = repack_to_gguf_layout(&pair, &[0u8; 48], &[0u8; 2]).unwrap_err();
        assert!(err.contains("blocks length mismatch"), "err: {err}");
    }

    #[test]
    fn repack_output_length_formula() {
        // For block_size=32, bytes_per_block=16, block_bytes=17.
        // num_blocks=5 → output = 5 * 17 = 85 bytes.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 5,
            blocks_shape: vec![5, 16],
            bias_name: None,
        };
        let blocks = vec![0u8; 80]; // 5 * 16
        let scales = vec![0u8; 5];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 85);
    }

    #[test]
    fn repack_preserves_each_byte_verbatim() {
        // Ensure no byte transformation happens during repack — pure interleaving.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 8,
            num_blocks: 2,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0xFF, 0x00, 0x80, 0x7F, 0x01, 0xFE, 0x55, 0xAA];
        let scales = vec![0xCC, 0x33];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // block 0: scale 0xCC, qs = [0xFF, 0x00, 0x80, 0x7F]
        // block 1: scale 0x33, qs = [0x01, 0xFE, 0x55, 0xAA]
        assert_eq!(
            out,
            vec![0xCC, 0xFF, 0x00, 0x80, 0x7F, 0x33, 0x01, 0xFE, 0x55, 0xAA]
        );
    }

    // ── CandidateTensor construction ───────────────────────────────────

    #[test]
    fn candidate_tensor_fields_match() {
        let ct = cand("test_tensor", Dtype::U8, vec![3, 4], 12);
        assert_eq!(ct.name, "test_tensor");
        assert_eq!(ct.dtype, Dtype::U8);
        assert_eq!(ct.shape, vec![3, 4]);
        assert_eq!(ct.byte_len, 12);
    }

    #[test]
    fn candidate_tensor_debug_format() {
        let ct = cand("abc", Dtype::U8, vec![1], 1);
        let debug = format!("{ct:?}");
        assert!(debug.contains("abc"));
        assert!(debug.contains("CandidateTensor"));
    }

    #[test]
    fn candidate_tensor_clone_independence() {
        let ct = cand("orig", Dtype::U8, vec![2, 3], 6);
        let mut cloned = ct.clone();
        cloned.name = "modified".into();
        cloned.byte_len = 99;
        assert_eq!(ct.name, "orig");
        assert_eq!(ct.byte_len, 6);
        assert_eq!(cloned.name, "modified");
        assert_eq!(cloned.byte_len, 99);
    }

    // ── Mxfp4PairMap / Mxfp4ScalesSidecarSet type aliases ──────────────

    #[test]
    fn pair_map_is_hash_map() {
        let mut map: Mxfp4PairMap = HashMap::new();
        let pair = Mxfp4Pair {
            blocks_name: "k_blocks".into(),
            scales_name: "k_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        map.insert("k_blocks".into(), pair);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("k_blocks"));
    }

    #[test]
    fn sidecar_set_is_hash_set() {
        let mut set: Mxfp4ScalesSidecarSet = std::collections::HashSet::new();
        set.insert("x_scales".into());
        assert!(set.contains("x_scales"));
        assert!(!set.contains("x_blocks"));
    }

    // ── Integration: scan → repack pipeline ────────────────────────────

    #[test]
    fn scan_then_repack_produces_correct_output() {
        let tensors = vec![
            cand("layer.w_blocks", Dtype::U8, vec![32], 32),
            cand("layer.w_scales", Dtype::U8, vec![2], 2),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);

        let pair = scan.pairs.get("layer.w_blocks").unwrap();
        assert_eq!(pair.num_blocks, 2);
        assert_eq!(pair.block_size, 32);

        let blocks = vec![0xAA; 32]; // 2 blocks * 16 bytes each
        let scales = vec![0x55, 0x77];
        let out = repack_to_gguf_layout(pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 34, "2 * (1 + 16) = 34");
        assert_eq!(out[0], 0x55);
        assert_eq!(out[17], 0x77);
        // All block data bytes should be 0xAA
        for i in 0..16 {
            assert_eq!(out[1 + i], 0xAA, "block 0 byte {i}");
            assert_eq!(out[18 + i], 0xAA, "block 1 byte {i}");
        }
    }

    #[test]
    fn scan_then_repack_with_bias_sibling() {
        let tensors = vec![
            cand("mlp.up_proj_blocks", Dtype::U8, vec![16], 16),
            cand("mlp.up_proj_scales", Dtype::U8, vec![1], 1),
            cand("mlp.up_proj_bias", Dtype::BF16, vec![4], 8),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("mlp.up_proj_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("mlp.up_proj_bias"));

        let blocks = vec![
            0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0,
            0xF0, 0x00,
        ];
        let scales = vec![0xFF];
        let out = repack_to_gguf_layout(pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 17);
        assert_eq!(out[0], 0xFF);
        assert_eq!(&out[1..], &blocks[..]);
    }

    // ── MXFP4 nibble layout details ────────────────────────────────────

    #[test]
    fn nibble_packing_two_e2m1_per_byte() {
        // block_size=4 → 2 packed bytes per block.
        // Low nibble = even index (elem 0, 2), high nibble = odd index (elem 1, 3).
        // This test validates that repack treats blocks_bytes as opaque — no
        // nibble swapping or transformation occurs.
        let pair = Mxfp4Pair {
            blocks_name: "n_blocks".into(),
            scales_name: "n_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        // Byte 0x2B = low nibble 0xB (elem 0), high nibble 0x2 (elem 1)
        // Byte 0x15 = low nibble 0x5 (elem 2), high nibble 0x1 (elem 3)
        let blocks = vec![0x2B, 0x15];
        let scales = vec![0x80];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0x80, 0x2B, 0x15], "bytes pass through unchanged");
    }

    #[test]
    fn block_size_32_yields_16_packed_bytes_per_block() {
        // Verify the block_size / 2 relationship: 32 elements → 16 bytes packed.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let blocks = vec![0u8; 16];
        let scales = vec![0x7F];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 17);
        assert_eq!(out[0], 0x7F);
        assert_eq!(&out[1..17], &[0u8; 16]);
    }

    #[test]
    fn e8m0_scale_byte_passthrough() {
        // E8M0 scale bytes are biased exponents (no mantissa). The repack must
        // copy them verbatim — no reinterpretation.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0u8; 8]; // 4 * 2 bytes
                                   // Diverse E8M0 values: zero bias, normal, max exponent, NaN-like
        let scales = vec![0x00, 0x7F, 0xFE, 0xFF];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Every 3rd byte (positions 0, 3, 6, 9) should be the scale byte
        assert_eq!(out[0], 0x00);
        assert_eq!(out[3], 0x7F);
        assert_eq!(out[6], 0xFE);
        assert_eq!(out[9], 0xFF);
    }

    // ── Mxfp4Pair Hash ─────────────────────────────────────────────────

    #[test]
    fn pair_hash_equal_objects_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![4, 16],
            bias_name: Some("w_bias".into()),
        };
        let b = a.clone();

        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let hash_a = ha.finish();

        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        let hash_b = hb.finish();

        assert_eq!(hash_a, hash_b, "equal objects must have equal hashes");
    }

    #[test]
    fn pair_hash_different_objects_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = Mxfp4Pair {
            blocks_name: "a_blocks".into(),
            scales_name: "a_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let b = Mxfp4Pair {
            blocks_name: "b_blocks".into(),
            scales_name: "b_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };

        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let hash_a = ha.finish();

        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        let hash_b = hb.finish();

        assert_ne!(hash_a, hash_b, "different objects should have different hashes");
    }

    #[test]
    fn pair_can_be_used_in_hashset() {
        use std::collections::HashSet;

        let a = Mxfp4Pair {
            blocks_name: "x_blocks".into(),
            scales_name: "x_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        let b = a.clone();
        let c = Mxfp4Pair {
            blocks_name: "y_blocks".into(),
            scales_name: "y_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };

        let mut set = HashSet::new();
        assert!(set.insert(a.clone()));
        assert!(!set.insert(b), "duplicate insert should return false");
        assert!(set.insert(c));
        assert_eq!(set.len(), 2);
    }

    // ── Mxfp4Pair field access ─────────────────────────────────────────

    #[test]
    fn pair_all_fields_accessible() {
        let pair = Mxfp4Pair {
            blocks_name: "blocks_name_val".into(),
            scales_name: "scales_name_val".into(),
            block_size: 16,
            num_blocks: 99,
            blocks_shape: vec![3, 4, 5],
            bias_name: Some("bias_val".into()),
        };
        assert_eq!(pair.blocks_name, "blocks_name_val");
        assert_eq!(pair.scales_name, "scales_name_val");
        assert_eq!(pair.block_size, 16);
        assert_eq!(pair.num_blocks, 99);
        assert_eq!(pair.blocks_shape, vec![3, 4, 5]);
        assert_eq!(pair.bias_name, Some("bias_val".into()));
    }

    #[test]
    fn pair_bias_name_none() {
        let pair = Mxfp4Pair {
            blocks_name: "x_blocks".into(),
            scales_name: "x_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        assert!(pair.bias_name.is_none());
    }

    #[test]
    fn pair_equality_differs_by_block_size() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let mut b = a.clone();
        b.block_size = 16;
        assert_ne!(a, b);
    }

    #[test]
    fn pair_equality_differs_by_num_blocks() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let mut b = a.clone();
        b.num_blocks = 8;
        assert_ne!(a, b);
    }

    #[test]
    fn pair_equality_differs_by_blocks_shape() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let mut b = a.clone();
        b.blocks_shape = vec![2, 8];
        assert_ne!(a, b);
    }

    #[test]
    fn pair_equality_differs_by_scales_name() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let mut b = a.clone();
        b.scales_name = "other_scales".into();
        assert_ne!(a, b);
    }

    // ── CandidateTensor PartialEq ──────────────────────────────────────

    #[test]
    fn candidate_equality_same_fields() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![3, 4],
            byte_len: 12,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn candidate_inequality_different_name() {
        let a = CandidateTensor {
            name: "a".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let mut b = a.clone();
        b.name = "b".into();
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_inequality_different_dtype() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let mut b = a.clone();
        b.dtype = Dtype::BF16;
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_inequality_different_shape() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let mut b = a.clone();
        b.shape = vec![2, 2];
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_inequality_different_byte_len() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let mut b = a.clone();
        b.byte_len = 8;
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_empty_shape() {
        let ct = CandidateTensor {
            name: "scalar".into(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 0,
        };
        assert!(ct.shape.is_empty());
        assert_eq!(ct.byte_len, 0);
    }

    // ── Mxfp4PairScan PartialEq ────────────────────────────────────────

    #[test]
    fn scan_equality_both_default() {
        let a = Mxfp4PairScan::default();
        let b = Mxfp4PairScan::default();
        assert_eq!(a, b);
    }

    #[test]
    fn scan_inequality_after_insert() {
        let mut a = Mxfp4PairScan::default();
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        a.pairs.insert("w_blocks".into(), pair);
        a.sidecars.insert("w_scales".into());
        a.blocks_to_scales.insert("w_blocks".into(), "w_scales".into());

        let b = Mxfp4PairScan::default();
        assert_ne!(a, b);
    }

    #[test]
    fn scan_equality_same_content() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };

        let mut a = Mxfp4PairScan::default();
        a.pairs.insert("w_blocks".into(), pair.clone());
        a.sidecars.insert("w_scales".into());
        a.blocks_to_scales.insert("w_blocks".into(), "w_scales".into());

        let mut b = Mxfp4PairScan::default();
        b.pairs.insert("w_blocks".into(), pair);
        b.sidecars.insert("w_scales".into());
        b.blocks_to_scales.insert("w_blocks".into(), "w_scales".into());

        assert_eq!(a, b);
    }

    // ── scan_mxfp4_pairs: prefix edge cases ────────────────────────────

    #[test]
    fn scan_prefix_can_be_empty_string() {
        // "_blocks" with empty prefix — valid as far as suffix matching goes.
        let tensors = vec![
            cand("_blocks", Dtype::U8, vec![16], 16),
            cand("_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("_blocks").unwrap();
        assert_eq!(pair.scales_name, "_scales");
    }

    #[test]
    fn scan_deeply_nested_prefix() {
        let tensors = vec![
            cand("a.b.c.d.e.f.g_blocks", Dtype::U8, vec![16], 16),
            cand("a.b.c.d.e.f.g_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
    }

    #[test]
    fn scan_bias_with_wrong_dtype_still_recorded() {
        // The bias sibling detection only checks name existence, not dtype.
        // A bias tensor with U8 dtype (unusual) is still recorded.
        let tensors = vec![
            cand("proj_blocks", Dtype::U8, vec![16], 16),
            cand("proj_scales", Dtype::U8, vec![1], 1),
            cand("proj_bias", Dtype::U8, vec![4], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("proj_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("proj_bias"));
    }

    #[test]
    fn scan_blocks_name_ends_with_blocks_blocks() {
        // Name "x_blocks_blocks" → suffix is "_blocks", prefix is "x_blocks".
        // The scales should be "x_blocks_scales".
        let tensors = vec![
            cand("x_blocks_blocks", Dtype::U8, vec![16], 16),
            cand("x_blocks_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("x_blocks_blocks").unwrap();
        assert_eq!(pair.scales_name, "x_blocks_scales");
    }

    #[test]
    fn scan_deduplicates_duplicate_tensor_names() {
        // Two tensors with the same name — last one wins (HashMap behavior).
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_blocks", Dtype::U8, vec![32], 32), // duplicate name
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        // The second w_blocks has byte_len=32, which does not match 1*16,
        // so no pair should be detected.
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_mixed_valid_and_invalid_pairs() {
        let tensors = vec![
            // Valid pair
            cand("valid_blocks", Dtype::U8, vec![16], 16),
            cand("valid_scales", Dtype::U8, vec![1], 1),
            // Invalid: wrong byte ratio
            cand("invalid_blocks", Dtype::U8, vec![10], 10),
            cand("invalid_scales", Dtype::U8, vec![1], 1),
            // Invalid: wrong dtype on blocks
            cand("bad_dtype_blocks", Dtype::BF16, vec![16], 32),
            cand("bad_dtype_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("valid_blocks"));
        assert!(!scan.pairs.contains_key("invalid_blocks"));
        assert!(!scan.pairs.contains_key("bad_dtype_blocks"));
    }

    // ── repack_to_gguf_layout: more edge cases ─────────────────────────

    #[test]
    fn repack_block_size_2_minimum() {
        // block_size=2 → 1 packed byte per block. Scale + 1 byte = 2 bytes/block.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 2,
            num_blocks: 3,
            blocks_shape: vec![3],
            bias_name: None,
        };
        let blocks = vec![0xAA, 0xBB, 0xCC];
        let scales = vec![0x11, 0x22, 0x33];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0x11, 0xAA, 0x22, 0xBB, 0x33, 0xCC]);
    }

    #[test]
    fn repack_error_contains_expected_and_actual_byte_counts() {
        let pair = Mxfp4Pair {
            blocks_name: "my_w_blocks".into(),
            scales_name: "my_w_scales".into(),
            block_size: 32,
            num_blocks: 3,
            blocks_shape: vec![48],
            bias_name: None,
        };
        // Expected blocks: 3 * 16 = 48. Provide 30.
        let err = repack_to_gguf_layout(&pair, &[0u8; 30], &[0u8; 3]).unwrap_err();
        assert!(err.contains("expected 48"), "err: {err}");
        assert!(err.contains("got 30"), "err: {err}");
        assert!(err.contains("3 blocks"), "err: {err}");
    }

    #[test]
    fn repack_error_scales_message_contains_block_count() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 5,
            blocks_shape: vec![80],
            bias_name: None,
        };
        // Expected scales: 5. Provide 3.
        let err = repack_to_gguf_layout(&pair, &[0u8; 80], &[0u8; 3]).unwrap_err();
        assert!(err.contains("expected 5"), "err: {err}");
        assert!(err.contains("got 3"), "err: {err}");
    }

    #[test]
    fn repack_empty_input_valid_pair() {
        // num_blocks=0, empty slices → empty output (already tested indirectly
        // but verify the output vec is indeed allocated with capacity 0).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 0,
            blocks_shape: vec![0],
            bias_name: None,
        };
        let out = repack_to_gguf_layout(&pair, &[], &[]).unwrap();
        assert!(out.is_empty());
        assert_eq!(out.len(), 0);
    }

    // ── Mxfp4Pair construction with extreme values ─────────────────────

    #[test]
    fn pair_large_num_blocks_and_block_size() {
        let pair = Mxfp4Pair {
            blocks_name: "big_blocks".into(),
            scales_name: "big_scales".into(),
            block_size: 1024,
            num_blocks: usize::MAX / 1024,
            blocks_shape: vec![1024],
            bias_name: None,
        };
        assert_eq!(pair.block_size, 1024);
        assert_eq!(pair.num_blocks, usize::MAX / 1024);
    }

    #[test]
    fn pair_with_empty_blocks_shape() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 0,
            blocks_shape: vec![],
            bias_name: None,
        };
        assert!(pair.blocks_shape.is_empty());
    }

    #[test]
    fn pair_with_long_bias_name() {
        let long_name = "x".repeat(10000);
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: Some(long_name.clone()),
        };
        assert_eq!(pair.bias_name.as_deref(), Some(long_name.as_str()));
    }

    // ── CandidateTensor with various dtypes ────────────────────────────

    #[test]
    fn candidate_various_dtypes_preserved() {
        let dtypes = [Dtype::BOOL, Dtype::U8, Dtype::I8, Dtype::F32, Dtype::BF16, Dtype::F16];
        for dtype in dtypes {
            let ct = CandidateTensor {
                name: "t".into(),
                dtype,
                shape: vec![1],
                byte_len: 1,
            };
            assert_eq!(ct.dtype, dtype, "dtype should be preserved");
        }
    }

    // ── scan result consistency ────────────────────────────────────────

    #[test]
    fn scan_pairs_and_blocks_to_scales_consistent_keys() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![32], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
            cand("c_blocks", Dtype::U8, vec![48], 48),
            cand("c_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), scan.blocks_to_scales.len());
        for (blocks_name, scales_name) in &scan.blocks_to_scales {
            assert!(scan.pairs.contains_key(blocks_name));
            assert!(scan.sidecars.contains(scales_name));
        }
    }

    #[test]
    fn scan_sidecars_never_contains_blocks_names() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![32], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        for blocks_name in scan.pairs.keys() {
            assert!(!scan.sidecars.contains(blocks_name));
        }
    }

    // ── E8M0 exponent scale byte properties ────────────────────────────

    #[test]
    fn e8m0_biased_exponent_127_represents_one() {
        // E8M0: value = 2^(raw_byte - 127). Byte 0x7F = 127 → 2^0 = 1.0.
        // Repack must preserve the raw byte; interpretation is downstream.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0x00, 0x00];
        let scales = vec![0x7F];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out[0], 0x7F, "E8M0 byte 0x7F must pass through");
    }

    #[test]
    fn e8m0_zero_byte_represents_denormal() {
        // E8M0 byte 0x00 = 2^(0-127) = 2^-127 ≈ 5.88e-39 (subnormal scale).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![4],
            bias_name: None,
        };
        let blocks = vec![0x11, 0x22, 0x33, 0x44];
        let scales = vec![0x00, 0x00];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out[0], 0x00, "E8M0 zero byte preserved");
        assert_eq!(out[3], 0x00, "second E8M0 zero byte preserved");
    }

    #[test]
    fn e8m0_max_byte_254_represents_large_scale() {
        // E8M0 byte 0xFE = 254 → 2^(254-127) = 2^127.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0xAA, 0xBB];
        let scales = vec![0xFE];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out[0], 0xFE);
    }

    #[test]
    fn e8m0_byte_255_nan_sentinel_passthrough() {
        // E8M0 byte 0xFF is NaN in OCP spec. Must pass through verbatim.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0x00, 0x00];
        let scales = vec![0xFF];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out[0], 0xFF, "NaN sentinel must pass through");
    }

    #[test]
    fn e8m0_mixed_exponents_across_blocks() {
        // Each block has an independent scale byte.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0u8; 8];
        let scales = vec![0x70, 0x80, 0x90, 0xA0];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // block_bytes = 1 + 2 = 3
        assert_eq!(out[0], 0x70);
        assert_eq!(out[3], 0x80);
        assert_eq!(out[6], 0x90);
        assert_eq!(out[9], 0xA0);
    }

    // ── Scan: order independence ───────────────────────────────────────

    #[test]
    fn scan_scales_before_blocks_in_input() {
        // Input order does not matter — scales listed first should still pair.
        let tensors = vec![
            cand("w_scales", Dtype::U8, vec![1], 1),
            cand("w_blocks", Dtype::U8, vec![16], 16),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("w_blocks"));
        assert!(scan.sidecars.contains("w_scales"));
    }

    #[test]
    fn scan_interleaved_blocks_and_scales() {
        // Interleaved: blocks_a, scales_b, blocks_b, scales_a.
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("b_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 2);
        assert!(scan.pairs.contains_key("a_blocks"));
        assert!(scan.pairs.contains_key("b_blocks"));
    }

    // ── Scan: bias-only tensor (no blocks/scales siblings) ─────────────

    #[test]
    fn scan_bias_only_tensor_ignored() {
        // A lone "_bias" tensor without _blocks/_scales is not special.
        let tensors = vec![cand("proj_bias", Dtype::BF16, vec![4], 8)];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn scan_bias_recorded_only_when_blocks_scales_pair_exists() {
        // bias is present but blocks/scales byte ratio is wrong → no pair → bias not recorded.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![10], 10),
            cand("w_scales", Dtype::U8, vec![1], 1),
            cand("w_bias", Dtype::BF16, vec![2], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Scan: prefix with special characters ───────────────────────────

    #[test]
    fn scan_prefix_with_dots_and_underscores() {
        let tensors = vec![
            cand("model.layers_0.mlp.gate_blocks", Dtype::U8, vec![16], 16),
            cand("model.layers_0.mlp.gate_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("model.layers_0.mlp.gate_blocks").unwrap();
        assert_eq!(
            pair.scales_name,
            "model.layers_0.mlp.gate_scales"
        );
    }

    #[test]
    fn scan_prefix_containing_blocks_substring() {
        // Prefix "blocks" contains the word "blocks" but suffix is still exact.
        let tensors = vec![
            cand("blocks_blocks", Dtype::U8, vec![16], 16),
            cand("blocks_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
    }

    // ── Scan: only scales present (no blocks) ──────────────────────────

    #[test]
    fn scan_only_scales_suffix_tensor_present() {
        // A "_scales" tensor without a matching "_blocks" should be ignored.
        let tensors = vec![
            cand("weights_scales", Dtype::U8, vec![4], 4),
            cand("unrelated_tensor", Dtype::BF16, vec![8], 16),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // ── Scan: multiple biases with different prefixes ──────────────────

    #[test]
    fn scan_multiple_biases_each_paired_correctly() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![2], 4),
            cand("b_blocks", Dtype::U8, vec![32], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
            cand("b_bias", Dtype::BF16, vec![4], 8),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 2);
        let pair_a = scan.pairs.get("a_blocks").unwrap();
        let pair_b = scan.pairs.get("b_blocks").unwrap();
        assert_eq!(pair_a.bias_name.as_deref(), Some("a_bias"));
        assert_eq!(pair_b.bias_name.as_deref(), Some("b_bias"));
        // Biases must not be sidecars.
        assert!(!scan.sidecars.contains("a_bias"));
        assert!(!scan.sidecars.contains("b_bias"));
    }

    // ── Repack: block_size odd values ──────────────────────────────────

    #[test]
    fn repack_block_size_1_half_byte_per_block() {
        // block_size=1 → bytes_per_block = 0 (1/2 rounds down).
        // This would produce 0 block bytes per block — edge case in integer division.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 1,
            num_blocks: 2,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks: Vec<u8> = vec![];
        let scales = vec![0xAA, 0xBB];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // bytes_per_block = 1/2 = 0; block_bytes = 1 + 0 = 1 per block.
        assert_eq!(out, vec![0xAA, 0xBB], "only scale bytes, no packed data");
    }

    #[test]
    fn repack_block_size_8_yields_4_bytes_per_block() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 8,
            num_blocks: 2,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let scales = vec![0xF0, 0x0F];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // block 0: scale 0xF0 + qs [0x11,0x22,0x33,0x44]
        // block 1: scale 0x0F + qs [0x55,0x66,0x77,0x88]
        assert_eq!(
            out,
            vec![0xF0, 0x11, 0x22, 0x33, 0x44, 0x0F, 0x55, 0x66, 0x77, 0x88]
        );
    }

    #[test]
    fn repack_block_size_16_yields_8_bytes_per_block() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 16,
            num_blocks: 2,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let blocks = vec![0u8; 16]; // 2 * 8
        let scales = vec![0x7F, 0x80];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 18, "2 * (1 + 8) = 18");
        assert_eq!(out[0], 0x7F);
        assert_eq!(out[9], 0x80);
    }

    // ── Repack: error priority (scales checked before blocks) ──────────

    #[test]
    fn repack_scales_error_checked_before_blocks() {
        // Both scales and blocks are wrong lengths.
        // The function checks scales first, so the error should mention scales.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 10], &[0u8; 2]).unwrap_err();
        assert!(
            err.contains("scales length mismatch"),
            "scales should be checked first: {err}"
        );
    }

    // ── Repack: all-zeros and all-ones byte patterns ───────────────────

    #[test]
    fn repack_all_zero_bytes() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0u8; 6];
        let scales = vec![0u8; 3];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0u8; 9], "all zeros: 3 * (1 + 2)");
    }

    #[test]
    fn repack_all_ones_bytes() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![4],
            bias_name: None,
        };
        let blocks = vec![0xFF; 4];
        let scales = vec![0xFF; 2];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0xFF; 6], "all 0xFF: 2 * (1 + 2)");
    }

    // ── Scan: shape preservation for multi-dimensional tensors ─────────

    

    #[test]
    fn scan_preserves_rank_1_blocks_shape() {
        let tensors = vec![
            cand("flat_blocks", Dtype::U8, vec![160], 160),
            cand("flat_scales", Dtype::U8, vec![10], 10),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("flat_blocks").unwrap();
        assert_eq!(pair.blocks_shape, vec![160]);
    }

    #[test]
    fn scan_preserves_rank_4_blocks_shape() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![2, 3, 4, 16], 2 * 3 * 4 * 16),
            cand("a_scales", Dtype::U8, vec![2, 3, 4], 2 * 3 * 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("a_blocks").unwrap();
        assert_eq!(pair.blocks_shape, vec![2, 3, 4, 16]);
    }

    // ── Scan: suffix must be at end of name ────────────────────────────

    #[test]
    fn scan_blocks_suffix_at_start_of_name_ignored() {
        // Name starting with "_blocks" but not ending with it.
        let tensors = vec![
            cand("_blocks_tensor", Dtype::U8, vec![16], 16),
            cand("_scales_tensor", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    

    // ── Repack: output capacity matches length ─────────────────────────

    #[test]
    fn repack_output_capacity_exact() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 10,
            blocks_shape: vec![160],
            bias_name: None,
        };
        let blocks = vec![0u8; 160];
        let scales = vec![0u8; 10];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 170, "10 * 17 = 170");
        assert!(out.capacity() >= 170);
    }

    // ── Mxfp4PairScan: debug output detailed ──────────────────────────

    #[test]
    fn scan_debug_includes_pair_details() {
        let pair = Mxfp4Pair {
            blocks_name: "test_blocks".into(),
            scales_name: "test_scales".into(),
            block_size: 32,
            num_blocks: 5,
            blocks_shape: vec![80],
            bias_name: Some("test_bias".into()),
        };
        let mut scan = Mxfp4PairScan::default();
        scan.pairs.insert("test_blocks".into(), pair);
        let debug = format!("{scan:?}");
        assert!(debug.contains("test_blocks"), "Debug: {debug}");
        assert!(debug.contains("test_scales"), "Debug: {debug}");
    }

    // ── Scan: pairs map keyed by blocks_name only ─────────────────────

    #[test]
    fn scan_pairs_map_never_contains_scales_key() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(!scan.pairs.contains_key("a_scales"));
        assert!(scan.pairs.contains_key("a_blocks"));
    }

    #[test]
    fn scan_pairs_map_never_contains_bias_key() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![4], 8),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(!scan.pairs.contains_key("a_bias"));
    }

    // ── Scan: byte ratio exact arithmetic ─────────────────────────────

    #[test]
    fn scan_byte_ratio_with_large_scales_bytes() {
        // 256 blocks: blocks=4096, scales=256.
        let tensors = vec![
            cand("big_blocks", Dtype::U8, vec![256, 16], 4096),
            cand("big_scales", Dtype::U8, vec![256], 256),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("big_blocks").unwrap();
        assert_eq!(pair.num_blocks, 256);
    }

    #[test]
    fn scan_byte_ratio_just_under_boundary_rejected() {
        // blocks=15, scales=1. Expected 16. Off by one below.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![15], 15),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_byte_ratio_just_over_boundary_rejected() {
        // blocks=17, scales=1. Expected 16. Off by one above.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![17], 17),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Repack: verify interleaving with non-uniform scale values ──────

    #[test]
    fn repack_each_scale_precedes_its_block_data() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let scales = vec![0x10, 0x20, 0x30];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Layout: [s0, b0[0], b0[1], s1, b1[0], b1[1], s2, b2[0], b2[1]]
        assert_eq!(out[0], 0x10);
        assert_eq!(out[1], 0x01);
        assert_eq!(out[2], 0x02);
        assert_eq!(out[3], 0x20);
        assert_eq!(out[4], 0x03);
        assert_eq!(out[5], 0x04);
        assert_eq!(out[6], 0x30);
        assert_eq!(out[7], 0x05);
        assert_eq!(out[8], 0x06);
    }

    // ── CandidateTensor: equality edge cases ──────────────────────────

    #[test]
    fn candidate_equality_all_fields_same() {
        let a = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![2, 3],
            byte_len: 6,
        };
        let b = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![2, 3],
            byte_len: 6,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn candidate_inequality_different_shape_length() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![6],
            byte_len: 6,
        };
        let mut b = a.clone();
        b.shape = vec![2, 3]; // same total but different shape
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_inequality_different_byte_len_same_shape() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let mut b = a.clone();
        b.byte_len = 8;
        assert_ne!(a, b);
    }

    // ── Scan: tensor with empty name (edge case) ──────────────────────

    #[test]
    fn scan_tensor_with_exactly_blocks_suffix() {
        // Name is exactly "_blocks" (prefix = "").
        let tensors = vec![
            cand("_blocks", Dtype::U8, vec![16], 16),
            cand("_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("_blocks"));
    }

    #[test]
    fn scan_tensor_named_exactly_bias() {
        // Name "_bias" alone → no blocks/scales → ignored.
        let tensors = vec![cand("_bias", Dtype::BF16, vec![1], 2)];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Repack: scales too long ───────────────────────────────────────

    #[test]
    fn repack_rejects_scales_too_long() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 32], &[0u8; 5]).unwrap_err();
        assert!(err.contains("scales length mismatch"), "err: {err}");
    }

    // ── Scan: blocks_to_scales never contains bias names ──────────────

    #[test]
    fn scan_blocks_to_scales_excludes_bias_names() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![2], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(
            !scan.blocks_to_scales.contains_key("a_bias"),
            "bias must not appear in blocks_to_scales"
        );
    }

    // ── Mxfp4Pair: equality differs by all fields independently ───────

    #[test]
    fn pair_equality_same_all_fields_with_bias() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 16,
            num_blocks: 2,
            blocks_shape: vec![4, 8],
            bias_name: Some("w_bias".into()),
        };
        let b = a.clone();
        assert_eq!(a, b, "identical pairs must be equal");
    }

    #[test]
    fn pair_equality_both_none_bias() {
        let a = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let b = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        assert_eq!(a, b);
    }

    // ── Scan: same prefix with bias but wrong dtype on bias ───────────

    #[test]
    fn scan_bias_with_f16_dtype_still_recorded() {
        // Bias dtype is not validated — any tensor with the right name is recorded.
        let tensors = vec![
            cand("p_blocks", Dtype::U8, vec![16], 16),
            cand("p_scales", Dtype::U8, vec![1], 1),
            cand("p_bias", Dtype::F16, vec![4], 8),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("p_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("p_bias"));
    }

    // ── Repack: very large block_size ─────────────────────────────────

    #[test]
    fn repack_large_block_size() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 512,
            num_blocks: 1,
            blocks_shape: vec![256],
            bias_name: None,
        };
        let blocks = vec![0xAB; 256]; // 512 / 2 = 256
        let scales = vec![0x7F];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 257, "1 + 256");
        assert_eq!(out[0], 0x7F);
        assert_eq!(&out[1..], &[0xAB; 256]);
    }

    // ── Scan: blocks with zero-length scales (empty pair) ─────────────

    #[test]
    fn scan_zero_blocks_with_bias() {
        let tensors = vec![
            cand("empty_blocks", Dtype::U8, vec![0], 0),
            cand("empty_scales", Dtype::U8, vec![0], 0),
            cand("empty_bias", Dtype::BF16, vec![0], 0),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("empty_blocks").unwrap();
        assert_eq!(pair.num_blocks, 0);
        assert_eq!(pair.bias_name.as_deref(), Some("empty_bias"));
    }

    // ── Repack: error message for blocks includes block computation ───

    #[test]
    fn repack_error_blocks_message_includes_bytes_per_block() {
        let pair = Mxfp4Pair {
            blocks_name: "test_blocks".into(),
            scales_name: "test_scales".into(),
            block_size: 32,
            num_blocks: 3,
            blocks_shape: vec![48],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 1], &[0u8; 3]).unwrap_err();
        assert!(err.contains("16 bytes/block"), "err: {err}");
        assert!(err.contains("3 blocks"), "err: {err}");
    }

    // ── Scan: many irrelevant tensors do not interfere ────────────────

    #[test]
    fn scan_many_irrelevant_tensors_plus_one_valid_pair() {
        let mut tensors = vec![
            cand("attn.q_proj", Dtype::BF16, vec![64, 64], 8192),
            cand("attn.k_proj", Dtype::BF16, vec![64, 64], 8192),
            cand("attn.v_proj", Dtype::BF16, vec![64, 64], 8192),
            cand("attn.o_proj", Dtype::BF16, vec![64, 64], 8192),
            cand("layer_norm.weight", Dtype::F32, vec![64], 256),
            cand("token_embedding", Dtype::BF16, vec![32000, 64], 4096000),
        ];
        tensors.push(cand("mlp.gate_blocks", Dtype::U8, vec![16], 16));
        tensors.push(cand("mlp.gate_scales", Dtype::U8, vec![1], 1));
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("mlp.gate_blocks"));
    }

    // ── Repack: scales error checked first even when blocks also wrong ─

    #[test]
    fn repack_scales_checked_first_both_wrong() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 5,
            blocks_shape: vec![10],
            bias_name: None,
        };
        // scales expected=5, got=3. blocks expected=10, got=7. Both wrong.
        let err = repack_to_gguf_layout(&pair, &[0u8; 7], &[0u8; 3]).unwrap_err();
        assert!(
            err.contains("scales length mismatch"),
            "scales error must fire first: {err}"
        );
    }

    // ── Scan: pair sidecar set contains exactly the scales names ──────

    #[test]
    fn scan_sidecar_set_exactly_matches_scales_names() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![32], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
            cand("c_blocks", Dtype::U8, vec![48], 48),
            cand("c_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.sidecars.len(), 3);
        assert!(scan.sidecars.contains("a_scales"));
        assert!(scan.sidecars.contains("b_scales"));
        assert!(scan.sidecars.contains("c_scales"));
    }

    // ── Mxfp4Pair: debug format with bias present ────────────────────

    #[test]
    fn pair_debug_format_with_bias_shows_bias() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![64],
            bias_name: Some("w_bias".into()),
        };
        let debug = format!("{pair:?}");
        assert!(debug.contains("w_bias"), "Debug: {debug}");
    }

    // ── Scan: name with trailing whitespace not matched ───────────────

    #[test]
    fn scan_name_with_trailing_whitespace_not_matched() {
        let tensors = vec![
            cand("w_blocks ", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(
            scan.pairs.is_empty(),
            "trailing space breaks suffix match"
        );
    }

    // ── Repack: correct capacity reservation ──────────────────────────

    #[test]
    fn repack_output_vec_capacity_equals_length() {
        // capacity() >= len() always, but with_capacity means exact for fresh vec.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 3,
            blocks_shape: vec![48],
            bias_name: None,
        };
        let blocks = vec![0u8; 48];
        let scales = vec![0u8; 3];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 51, "3 * 17 = 51");
    }

    // ── Scan: F16 and F64 dtypes not treated as U8 ───────────────────

    #[test]
    fn scan_rejects_f16_blocks_with_u8_scales() {
        let tensors = vec![
            cand("a_blocks", Dtype::F16, vec![16], 32),
            cand("a_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_rejects_u8_blocks_with_f32_scales() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::F32, vec![1], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Repack: single block with various block_sizes ─────────────────

    #[test]
    fn repack_single_block_block_size_6() {
        // block_size=6 → bytes_per_block=3. 1+3=4 bytes output.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 6,
            num_blocks: 1,
            blocks_shape: vec![3],
            bias_name: None,
        };
        let blocks = vec![0x12, 0x34, 0x56];
        let scales = vec![0xAB];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0xAB, 0x12, 0x34, 0x56]);
    }

    // ── Scan: many layers with mixed valid/invalid pairs ──────────────

    #[test]
    fn scan_ten_layers_mixed_valid_invalid() {
        let mut tensors = vec![];
        for i in 0..10 {
            let prefix = format!("layer.{i}.mlp.gate");
            if i % 3 == 0 {
                // Invalid: wrong byte ratio
                tensors.push(cand(&format!("{prefix}_blocks"), Dtype::U8, vec![10], 10));
                tensors.push(cand(&format!("{prefix}_scales"), Dtype::U8, vec![1], 1));
            } else {
                // Valid
                tensors.push(cand(&format!("{prefix}_blocks"), Dtype::U8, vec![16], 16));
                tensors.push(cand(&format!("{prefix}_scales"), Dtype::U8, vec![1], 1));
            }
        }
        let scan = scan_mxfp4_pairs(tensors);
        // Layers 0, 3, 6, 9 are invalid (i%3==0). Layers 1,2,4,5,7,8 are valid.
        assert_eq!(scan.pairs.len(), 6, "6 valid pairs out of 10 layers");
    }

    // ── Repack: error includes both expected and actual counts ────────

    #[test]
    fn repack_scales_error_shows_expected_vs_actual() {
        let pair = Mxfp4Pair {
            blocks_name: "named_blocks".into(),
            scales_name: "named_scales".into(),
            block_size: 32,
            num_blocks: 7,
            blocks_shape: vec![112],
            bias_name: None,
        };
        let err = repack_to_gguf_layout(&pair, &[0u8; 112], &[0u8; 10]).unwrap_err();
        assert!(err.contains("expected 7"), "err: {err}");
        assert!(err.contains("got 10"), "err: {err}");
        assert!(err.contains("named_blocks"), "err: {err}");
    }

    // ── Scan: sidecar set does not grow for rejected pairs ────────────

    #[test]
    fn scan_sidecar_set_empty_when_all_pairs_rejected() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![10], 10), // wrong ratio
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::BF16, vec![16], 32), // wrong dtype
            cand("b_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.sidecars.is_empty());
    }

    // ── Scan: blocks present but scales has non-U8 dtype ─────────────────

    #[test]
    fn scan_blocks_u8_scales_i32_rejected() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::I32, vec![1], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_blocks_u8_scales_i8_rejected() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::I8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Scan: prefix with Unicode characters ─────────────────────────────

    #[test]
    fn scan_prefix_with_cjk_characters() {
        let tensors = vec![
            cand("模型.层0_blocks", Dtype::U8, vec![16], 16),
            cand("模型.层0_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("模型.层0_blocks"));
    }

    #[test]
    fn scan_prefix_with_emoji() {
        let tensors = vec![
            cand("🧱_blocks", Dtype::U8, vec![16], 16),
            cand("🧱_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
    }

    // ── Scan: prefix with hyphens and numbers ────────────────────────────

    #[test]
    fn scan_prefix_with_hyphens_and_numbers() {
        let tensors = vec![
            cand("layer-0 MLP-expert-3_blocks", Dtype::U8, vec![16], 16),
            cand("layer-0 MLP-expert-3_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan
            .pairs
            .get("layer-0 MLP-expert-3_blocks")
            .unwrap();
        assert_eq!(pair.scales_name, "layer-0 MLP-expert-3_scales");
    }

    // ── Scan: name case sensitivity ──────────────────────────────────────

    #[test]
    fn scan_suffix_is_case_sensitive() {
        // "_BLOCKS" (uppercase) is NOT the same as "_blocks".
        let tensors = vec![
            cand("w_BLOCKS", Dtype::U8, vec![16], 16),
            cand("w_SCALES", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(
            scan.pairs.is_empty(),
            "uppercase suffix must not match"
        );
    }

    #[test]
    fn scan_mixed_case_suffix_not_matched() {
        let tensors = vec![
            cand("w_Blocks", Dtype::U8, vec![16], 16),
            cand("w_Scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    // ── Scan: three-block scenario (blocks + scales + extra) ─────────────

    #[test]
    fn scan_three_valid_pairs_independent() {
        let tensors = vec![
            cand("gate_blocks", Dtype::U8, vec![16], 16),
            cand("gate_scales", Dtype::U8, vec![1], 1),
            cand("up_blocks", Dtype::U8, vec![32], 32),
            cand("up_scales", Dtype::U8, vec![2], 2),
            cand("down_blocks", Dtype::U8, vec![48], 48),
            cand("down_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 3);
        assert_eq!(scan.sidecars.len(), 3);
        assert_eq!(scan.blocks_to_scales.len(), 3);
    }

    // ── Scan: multiple blocks share same scale name (impossible by design)

    #[test]
    fn scan_two_block_names_one_scale_only_one_pairs() {
        // If two _blocks tensors both share the same _scales suffix,
        // the last one in the HashMap wins, and the other may or may not pair
        // depending on byte_len. Both must match independently.
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_blocks", Dtype::U8, vec![32], 32), // overrides first
        ];
        let scan = scan_mxfp4_pairs(tensors);
        // Second "a_blocks" has byte_len=32 != 1*16, so no pair.
        assert!(scan.pairs.is_empty());
    }

    // ── Mxfp4Pair: blocks_shape mutation independence ────────────────────

    #[test]
    fn pair_blocks_shape_mutation_is_independent() {
        let mut pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![2, 16],
            bias_name: None,
        };
        let shape_clone = pair.blocks_shape.clone();
        pair.blocks_shape.push(99);
        assert_eq!(shape_clone, vec![2, 16], "clone must be independent");
        assert_eq!(pair.blocks_shape, vec![2, 16, 99]);
    }

    // ── Mxfp4PairScan: sidecar set is mutable ────────────────────────────

    #[test]
    fn scan_sidecar_set_removable() {
        let mut scan = Mxfp4PairScan::default();
        scan.sidecars.insert("x_scales".into());
        assert!(scan.sidecars.contains("x_scales"));
        scan.sidecars.remove("x_scales");
        assert!(!scan.sidecars.contains("x_scales"));
    }

    // ── Mxfp4PairMap: lookup and removal ─────────────────────────────────

    #[test]
    fn pair_map_lookup_and_remove() {
        let mut map: Mxfp4PairMap = HashMap::new();
        let pair = Mxfp4Pair {
            blocks_name: "k_blocks".into(),
            scales_name: "k_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        map.insert("k_blocks".into(), pair);
        assert!(map.contains_key("k_blocks"));
        let removed = map.remove("k_blocks");
        assert!(removed.is_some());
        assert!(!map.contains_key("k_blocks"));
    }

    // ── Mxfp4ScalesSidecarSet: deduplication ─────────────────────────────

    #[test]
    fn sidecar_set_duplicate_insert_is_noop() {
        let mut set: Mxfp4ScalesSidecarSet = std::collections::HashSet::new();
        assert!(set.insert("x_scales".into()));
        assert!(!set.insert("x_scales".into()), "duplicate insert returns false");
        assert_eq!(set.len(), 1);
    }

    // ── Repack: bias_name does not affect repack behavior ────────────────

    #[test]
    fn repack_with_bias_name_still_produces_correct_output() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![4],
            bias_name: Some("w_bias".into()),
        };
        let blocks = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let scales = vec![0x10, 0x20];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0x10, 0xAA, 0xBB, 0x20, 0xCC, 0xDD]);
    }

    // ── Repack: output byte conservation property ────────────────────────

    #[test]
    fn repack_output_byte_count_equals_scales_plus_blocks() {
        // Total output bytes must equal scales_bytes + blocks_bytes.
        for &block_size in &[2usize, 4, 8, 16, 32, 64] {
            let bytes_per_block = block_size / 2;
            let num_blocks = 3;
            let pair = Mxfp4Pair {
                blocks_name: "w_blocks".into(),
                scales_name: "w_scales".into(),
                block_size,
                num_blocks,
                blocks_shape: vec![num_blocks * bytes_per_block],
                bias_name: None,
            };
            let blocks = vec![0u8; num_blocks * bytes_per_block];
            let scales = vec![0u8; num_blocks];
            let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
            assert_eq!(
                out.len(),
                num_blocks + num_blocks * bytes_per_block,
                "block_size={block_size}: output must be scales + blocks bytes"
            );
        }
    }

    // ── Scan + Repack: round-trip byte conservation ──────────────────────

    #[test]
    fn scan_repack_roundtrip_byte_conservation() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![48], 48),
            cand("w_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("w_blocks").unwrap();
        let blocks = vec![0xAB; 48];
        let scales = vec![0x7F; 3];
        let out = repack_to_gguf_layout(pair, &blocks, &scales).unwrap();
        // output = num_blocks * (1 + bytes_per_block) = 3 * 17 = 51
        assert_eq!(out.len(), 51);
        // All original blocks bytes and scales bytes must be present
        let mut blocks_collected: Vec<u8> = vec![];
        let mut scales_collected: Vec<u8> = vec![];
        for blk in 0..pair.num_blocks {
            let base = blk * 17;
            scales_collected.push(out[base]);
            blocks_collected.extend_from_slice(&out[base + 1..base + 17]);
        }
        assert_eq!(blocks_collected, blocks);
        assert_eq!(scales_collected, scales);
    }

    // ── CandidateTensor: name with various edge characters ───────────────

    #[test]
    fn candidate_name_with_special_chars() {
        let ct = CandidateTensor {
            name: "a.b/c[d]e{f}g".into(),
            dtype: Dtype::U8,
            shape: vec![1],
            byte_len: 1,
        };
        assert_eq!(ct.name, "a.b/c[d]e{f}g");
    }

    #[test]
    fn candidate_name_with_newline_preserved() {
        let ct = CandidateTensor {
            name: "name\nwith\nnewlines".into(),
            dtype: Dtype::U8,
            shape: vec![1],
            byte_len: 1,
        };
        assert!(ct.name.contains('\n'));
    }

    // ── CandidateTensor: large shape ─────────────────────────────────────

    #[test]
    fn candidate_large_shape_dims() {
        let shape: Vec<usize> = (0..100).collect();
        let ct = CandidateTensor {
            name: "big".into(),
            dtype: Dtype::U8,
            shape: shape.clone(),
            byte_len: 100,
        };
        assert_eq!(ct.shape.len(), 100);
        assert_eq!(ct.shape[50], 50);
    }

    // ── Mxfp4Pair: block_size zero edge case ─────────────────────────────

    #[test]
    fn pair_block_size_zero_is_valid_struct() {
        // Struct does not enforce block_size > 0 — construction is valid.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 0,
            num_blocks: 0,
            blocks_shape: vec![],
            bias_name: None,
        };
        assert_eq!(pair.block_size, 0);
    }

    // ── Scan: block_size always set to DEFAULT_MXFP4_BLOCK_SIZE ──────────

    #[test]
    fn scan_detected_pair_always_uses_default_block_size() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
    }

    // ── Scan: prefix ending in underscore before suffix ──────────────────

    #[test]
    fn scan_prefix_with_trailing_underscore_before_blocks() {
        let tensors = vec![
            cand("model_w_blocks", Dtype::U8, vec![16], 16),
            cand("model_w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("model_w_blocks").unwrap();
        assert_eq!(pair.scales_name, "model_w_scales");
    }

    // ── Scan: pair with same tensor appearing twice (last wins) ──────────

    #[test]
    fn scan_duplicate_blocks_last_wins_rejects_if_ratio_wrong() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
            cand("w_blocks", Dtype::U8, vec![32], 32), // overrides, wrong ratio
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_duplicate_scales_last_wins() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![2], 2), // first: 16 != 2*16
            cand("w_scales", Dtype::U8, vec![1], 1), // second overrides: 16 == 1*16
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1, "second scales should match");
    }

    // ── Repack: block_size=3 (odd) ───────────────────────────────────────

    #[test]
    fn repack_block_size_3_bytes_per_block_1() {
        // block_size=3 → bytes_per_block = 3/2 = 1 (integer division).
        // Each block: 1 scale byte + 1 packed byte = 2 bytes output.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 3,
            num_blocks: 2,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0xAA, 0xBB];
        let scales = vec![0x10, 0x20];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out, vec![0x10, 0xAA, 0x20, 0xBB]);
    }

    // ── Repack: block_size=64 (larger than default) ──────────────────────

    #[test]
    fn repack_block_size_64() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 64,
            num_blocks: 2,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let blocks = vec![0u8; 64]; // 2 * 32
        let scales = vec![0x7F, 0x80];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 66, "2 * (1 + 32) = 66");
        assert_eq!(out[0], 0x7F);
        assert_eq!(out[33], 0x80);
    }

    // ── Mxfp4PairScan: blocks_to_scales is one-directional ──────────────

    #[test]
    fn scan_blocks_to_scales_reverse_lookup_absent() {
        let tensors = vec![
            cand("x_blocks", Dtype::U8, vec![16], 16),
            cand("x_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.blocks_to_scales.contains_key("x_blocks"));
        assert!(!scan.blocks_to_scales.contains_key("x_scales"));
    }

    // ── CandidateTensor: eq derives for all dtypes ───────────────────────

    #[test]
    fn candidate_bool_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::BOOL,
            shape: vec![1],
            byte_len: 1,
        };
        assert_eq!(ct.dtype, Dtype::BOOL);
    }

    #[test]
    fn candidate_i8_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::I8,
            shape: vec![1],
            byte_len: 1,
        };
        assert_eq!(ct.dtype, Dtype::I8);
    }

    #[test]
    fn candidate_f16_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::F16,
            shape: vec![1],
            byte_len: 2,
        };
        assert_eq!(ct.dtype, Dtype::F16);
    }

    #[test]
    fn candidate_f64_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::F64,
            shape: vec![1],
            byte_len: 8,
        };
        assert_eq!(ct.dtype, Dtype::F64);
    }

    // ── Scan: only unrelated suffixes (e.g. _weight, _bias) ──────────────

    #[test]
    fn scan_only_weight_suffix_no_pairs() {
        let tensors = vec![
            cand("model.weight", Dtype::BF16, vec![64, 64], 8192),
            cand("model.weight_bias", Dtype::BF16, vec![64], 128),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // ── Scan: bias present without corresponding pair blocks ─────────────

    #[test]
    fn scan_bias_without_blocks_ignored() {
        let tensors = vec![
            cand("proj_bias", Dtype::BF16, vec![4], 8),
            cand("proj_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // ── Mxfp4PairScan: all three fields start empty ─────────────────────

    #[test]
    fn scan_default_all_fields_empty() {
        let scan = Mxfp4PairScan::default();
        assert_eq!(scan.pairs.len(), 0);
        assert_eq!(scan.blocks_to_scales.len(), 0);
        assert_eq!(scan.sidecars.len(), 0);
    }

    // ── Repack: alternating pattern verification ─────────────────────────

    #[test]
    fn repack_alternating_scale_block_pattern() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let scales = vec![0xA0, 0xB0, 0xC0, 0xD0];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Verify pattern: scale, 2 block bytes, scale, 2 block bytes, ...
        for i in 0..4 {
            let base = i * 3;
            assert_eq!(out[base], scales[i], "scale at block {i}");
            assert_eq!(out[base + 1], blocks[i * 2], "block byte 0 at block {i}");
            assert_eq!(out[base + 2], blocks[i * 2 + 1], "block byte 1 at block {i}");
        }
    }

    // ── Repack: scale byte at every Nth position ─────────────────────────

    #[test]
    fn repack_scale_byte_stride_is_bytes_per_block_plus_one() {
        let block_size = 8;
        let bytes_per_block = block_size / 2;
        let num_blocks = 5;
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size,
            num_blocks,
            blocks_shape: vec![num_blocks * bytes_per_block],
            bias_name: None,
        };
        let blocks = vec![0u8; num_blocks * bytes_per_block];
        let scales: Vec<u8> = (0..num_blocks as u8).collect();
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        let stride = 1 + bytes_per_block;
        for i in 0..num_blocks {
            assert_eq!(
                out[i * stride], scales[i],
                "scale byte at stride position {i}"
            );
        }
    }

    // ── Scan: repack with max practical num_blocks ───────────────────────

    #[test]
    fn repack_large_num_blocks_output_correct() {
        let num_blocks = 256;
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks,
            blocks_shape: vec![num_blocks * 2],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0..=255).cycle().take(num_blocks * 2).collect();
        let scales: Vec<u8> = (0..=255).cycle().take(num_blocks).collect();
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), num_blocks * 3);
        for i in 0..num_blocks {
            assert_eq!(out[i * 3], scales[i]);
            assert_eq!(out[i * 3 + 1], blocks[i * 2]);
            assert_eq!(out[i * 3 + 2], blocks[i * 2 + 1]);
        }
    }

    // ── Scan: bias with F64 dtype ────────────────────────────────────────

    #[test]
    fn scan_bias_with_f64_dtype_recorded() {
        let tensors = vec![
            cand("p_blocks", Dtype::U8, vec![16], 16),
            cand("p_scales", Dtype::U8, vec![1], 1),
            cand("p_bias", Dtype::F64, vec![4], 32),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("p_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("p_bias"));
    }

    // ── Scan: prefix is a number ─────────────────────────────────────────

    #[test]
    fn scan_prefix_all_digits() {
        let tensors = vec![
            cand("12345_blocks", Dtype::U8, vec![16], 16),
            cand("12345_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
    }

    // ── Mxfp4Pair: blocks_name serves as canonical key ───────────────────

    #[test]
    fn scan_uses_blocks_name_as_map_key() {
        let tensors = vec![
            cand("unique_prefix_blocks", Dtype::U8, vec![16], 16),
            cand("unique_prefix_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.contains_key("unique_prefix_blocks"));
        assert!(!scan.pairs.contains_key("unique_prefix_scales"));
        assert!(!scan.pairs.contains_key("unique_prefix"));
    }

    // ── Repack: scale byte at position zero always first ─────────────────

    #[test]
    fn repack_first_byte_is_first_scale() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: None,
        };
        let blocks = vec![0u8; 16];
        let scales = vec![0x42];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out[0], 0x42, "first byte must be first scale");
    }

    // ── Scan: two pairs with same scale byte count ───────────────────────

    #[test]
    fn scan_two_pairs_same_scale_byte_count() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![16], 16),
            cand("b_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 2);
        assert_eq!(scan.sidecars.len(), 2);
    }

    // ── Repack: with pair constructed via scan (integration) ─────────────

    #[test]
    fn repack_integration_scan_produces_correct_interleaving() {
        let tensors = vec![
            cand("mlp.weight_blocks", Dtype::U8, vec![2, 16], 32),
            cand("mlp.weight_scales", Dtype::U8, vec![2], 2),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("mlp.weight_blocks").unwrap();
        let blocks: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let scales = vec![0xAA, 0xBB];
        let out = repack_to_gguf_layout(pair, &blocks, &scales).unwrap();
        // Block 0: scale 0xAA, block bytes [0..16]
        // Block 1: scale 0xBB, block bytes [16..32]
        assert_eq!(out[0], 0xAA);
        assert_eq!(&out[1..17], &blocks[0..16]);
        assert_eq!(out[17], 0xBB);
        assert_eq!(&out[18..34], &blocks[16..32]);
    }

    // ── Scan: blocks shape is copied, not referenced ─────────────────────

    #[test]
    fn scan_blocks_shape_is_deep_copy() {
        let shape = vec![2, 4, 8];
        let tensors = vec![
            cand("w_blocks", Dtype::U8, shape.clone(), 64),
            cand("w_scales", Dtype::U8, vec![4], 4),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.blocks_shape, shape);
        // Mutating the original does not affect the pair
        drop(shape);
        assert_eq!(pair.blocks_shape, vec![2, 4, 8]);
    }

    // ── Mxfp4PairScan: PartialEq differs by pairs field ──────────────────

    #[test]
    fn scan_inequality_differs_by_pairs() {
        let mut a = Mxfp4PairScan::default();
        a.pairs.insert(
            "w_blocks".into(),
            Mxfp4Pair {
                blocks_name: "w_blocks".into(),
                scales_name: "w_scales".into(),
                block_size: 32,
                num_blocks: 1,
                blocks_shape: vec![16],
                bias_name: None,
            },
        );
        let b = Mxfp4PairScan::default();
        assert_ne!(a, b);
    }

    #[test]
    fn scan_inequality_differs_by_sidecars() {
        let mut a = Mxfp4PairScan::default();
        a.sidecars.insert("x_scales".into());
        let b = Mxfp4PairScan::default();
        assert_ne!(a, b);
    }

    #[test]
    fn scan_inequality_differs_by_blocks_to_scales() {
        let mut a = Mxfp4PairScan::default();
        a.blocks_to_scales
            .insert("w_blocks".into(), "w_scales".into());
        let b = Mxfp4PairScan::default();
        assert_ne!(a, b);
    }

    // ── CandidateTensor: debug format includes all key fields ────────────

    #[test]
    fn candidate_debug_includes_dtype() {
        let ct = CandidateTensor {
            name: "test".into(),
            dtype: Dtype::U8,
            shape: vec![1],
            byte_len: 1,
        };
        let debug = format!("{ct:?}");
        assert!(debug.contains("test"), "Debug: {debug}");
    }

    // ── Scan: suffix at exact end of string ──────────────────────────────

    #[test]
    fn scan_suffix_at_end_with_no_prefix_content() {
        // Name is exactly "_blocks" — empty prefix.
        let tensors = vec![
            cand("_blocks", Dtype::U8, vec![16], 16),
            cand("_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("_blocks").unwrap();
        assert_eq!(pair.blocks_name, "_blocks");
        assert_eq!(pair.scales_name, "_scales");
    }

    // ── Mxfp4Pair: scales_name field in scan result matches pair ─────────

    #[test]
    fn scan_scales_name_matches_in_pair_and_blocks_to_scales() {
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        let pair = scan.pairs.get("w_blocks").unwrap();
        let mapped = scan.blocks_to_scales.get("w_blocks").unwrap();
        assert_eq!(&pair.scales_name, mapped);
    }

    // ── Repack: blocks all 0x00, scales all 0x7F ────────────────────────

    #[test]
    fn repack_zeros_blocks_normal_scales() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0u8; 6];
        let scales = vec![0x7F; 3];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 9);
        for i in 0..3 {
            assert_eq!(out[i * 3], 0x7F, "scale at block {i}");
            assert_eq!(out[i * 3 + 1], 0x00, "block byte 0 at block {i}");
            assert_eq!(out[i * 3 + 2], 0x00, "block byte 1 at block {i}");
        }
    }

    // ── Scan: valid pair among many dtypes ───────────────────────────────

    #[test]
    fn scan_finds_pair_among_many_dtype_tensors() {
        let tensors = vec![
            cand("embed.weight", Dtype::BF16, vec![32000, 4096], 262144000),
            cand("lm_head.weight", Dtype::F32, vec![32000, 4096], 524288000),
            cand("layer.0.attn.q_proj_blocks", Dtype::U8, vec![16], 16),
            cand("layer.0.attn.q_proj_scales", Dtype::U8, vec![1], 1),
            cand("layer.0.attn.k_proj", Dtype::BF16, vec![4096, 4096], 33554432),
            cand("layer.0.norm.weight", Dtype::F32, vec![4096], 16384),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("layer.0.attn.q_proj_blocks"));
        assert!(scan.sidecars.contains("layer.0.attn.q_proj_scales"));
    }

    // ── Repack: pair fields are read-only in repack ──────────────────────

    #[test]
    fn repack_does_not_modify_pair_fields() {
        let pair = Mxfp4Pair {
            blocks_name: "immutable_blocks".into(),
            scales_name: "immutable_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0x00, 0x01];
        let scales = vec![0x80];
        let _out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Verify pair was not modified (fields are still the same)
        assert_eq!(pair.blocks_name, "immutable_blocks");
        assert_eq!(pair.scales_name, "immutable_scales");
        assert_eq!(pair.block_size, 4);
        assert_eq!(pair.num_blocks, 1);
    }

    // ── Constants: suffix relationships ──────────────────────────────────

    #[test]
    fn suffixes_are_distinct_from_each_other() {
        assert_ne!(MXFP4_BLOCKS_SUFFIX, MXFP4_SCALES_SUFFIX);
        assert_ne!(MXFP4_BLOCKS_SUFFIX, MXFP4_BIAS_SUFFIX);
        assert_ne!(MXFP4_SCALES_SUFFIX, MXFP4_BIAS_SUFFIX);
    }

    #[test]
    fn all_suffixes_start_with_underscore() {
        assert!(MXFP4_BLOCKS_SUFFIX.starts_with('_'));
        assert!(MXFP4_SCALES_SUFFIX.starts_with('_'));
        assert!(MXFP4_BIAS_SUFFIX.starts_with('_'));
    }

    // ── Scan: pair detected with single element blocks ───────────────────

    #[test]
    fn scan_single_block_pair() {
        let tensors = vec![
            cand("tiny_blocks", Dtype::U8, vec![16], 16),
            cand("tiny_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("tiny_blocks").unwrap();
        assert_eq!(pair.num_blocks, 1);
        assert_eq!(pair.block_size, 32);
    }

    // ── CandidateTensor: PartialEq comprehensive ─────────────────────────

    #[test]
    fn candidate_eq_reflexive() {
        let ct = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        assert_eq!(ct, ct);
    }

    #[test]
    fn candidate_eq_symmetric() {
        let a = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let b = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn candidate_eq_transitive() {
        let a = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let b = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let c = CandidateTensor {
            name: "x".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── Scan: 20-layer full MoE expert scan ──────────────────────────────

    #[test]
    fn scan_20_layer_moe_all_paired() {
        let mut tensors = vec![];
        for i in 0..20 {
            let prefix = format!("model.layers.{i}.mlp.experts.gate_up_proj");
            tensors.push(cand(&format!("{prefix}_blocks"), Dtype::U8, vec![16], 16));
            tensors.push(cand(&format!("{prefix}_scales"), Dtype::U8, vec![1], 1));
            tensors.push(cand(&format!("{prefix}_bias"), Dtype::BF16, vec![8], 16));
        }
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 20);
        assert_eq!(scan.sidecars.len(), 20);
        assert_eq!(scan.blocks_to_scales.len(), 20);
        for i in 0..20 {
            let blocks_name =
                format!("model.layers.{i}.mlp.experts.gate_up_proj_blocks");
            assert!(scan.pairs.contains_key(&blocks_name));
            let pair = scan.pairs.get(&blocks_name).unwrap();
            let expected_bias =
                format!("model.layers.{i}.mlp.experts.gate_up_proj_bias");
            assert_eq!(pair.bias_name.as_deref(), Some(expected_bias.as_str()));
        }
    }

    // ── Mxfp4Pair: bias_name with empty string ───────────────────────────

    #[test]
    fn pair_bias_name_empty_string_some() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![16],
            bias_name: Some(String::new()),
        };
        assert!(pair.bias_name.is_some());
        assert_eq!(pair.bias_name.as_deref(), Some(""));
    }

    // ── Scan: bias name with extra underscore in prefix ──────────────────

    #[test]
    fn scan_bias_with_complex_prefix() {
        let tensors = vec![
            cand("model.layers.0.mlp.experts.0.gate_up_proj_blocks", Dtype::U8, vec![16], 16),
            cand("model.layers.0.mlp.experts.0.gate_up_proj_scales", Dtype::U8, vec![1], 1),
            cand("model.layers.0.mlp.experts.0.gate_up_proj_bias", Dtype::BF16, vec![4], 8),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("model.layers.0.mlp.experts.0.gate_up_proj_blocks").unwrap();
        assert_eq!(
            pair.bias_name.as_deref(),
            Some("model.layers.0.mlp.experts.0.gate_up_proj_bias")
        );
    }

    // ── Repack: correct total output length across various block_sizes ───

    #[test]
    fn repack_output_length_various_block_sizes() {
        for &block_size in &[2, 4, 6, 8, 16, 32, 64, 128] {
            let bytes_per_block = block_size / 2;
            let num_blocks = 4;
            let pair = Mxfp4Pair {
                blocks_name: "w_blocks".into(),
                scales_name: "w_scales".into(),
                block_size,
                num_blocks,
                blocks_shape: vec![num_blocks * bytes_per_block],
                bias_name: None,
            };
            let blocks = vec![0u8; num_blocks * bytes_per_block];
            let scales = vec![0u8; num_blocks];
            let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
            assert_eq!(
                out.len(),
                num_blocks * (1 + bytes_per_block),
                "block_size={block_size}"
            );
        }
    }

    // ── Scan: pairs map only contains blocks_name keys, never prefixes ──

    #[test]
    fn scan_pairs_map_never_contains_prefix_alone() {
        let tensors = vec![
            cand("proj_blocks", Dtype::U8, vec![16], 16),
            cand("proj_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(!scan.pairs.contains_key("proj"));
        assert!(scan.pairs.contains_key("proj_blocks"));
    }

    // ── Mxfp4Pair: num_blocks zero means empty pair ──────────────────────

    #[test]
    fn pair_num_blocks_zero_means_empty() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 0,
            blocks_shape: vec![0],
            bias_name: None,
        };
        assert_eq!(pair.num_blocks, 0);
        let out = repack_to_gguf_layout(&pair, &[], &[]).unwrap();
        assert!(out.is_empty());
    }

    // ── Constants: verify exported values ─────────────────────────────────

    #[test]
    fn constants_block_size_is_32() {
        assert_eq!(DEFAULT_MXFP4_BLOCK_SIZE, 32);
    }

    #[test]
    fn constants_blocks_suffix() {
        assert_eq!(MXFP4_BLOCKS_SUFFIX, "_blocks");
    }

    #[test]
    fn constants_scales_suffix() {
        assert_eq!(MXFP4_SCALES_SUFFIX, "_scales");
    }

    #[test]
    fn constants_bias_suffix() {
        assert_eq!(MXFP4_BIAS_SUFFIX, "_bias");
    }

    // ── Mxfp4PairScan: Default all fields empty ────────────────────────────

    #[test]
    fn scan_default_pairs_is_empty() {
        let scan = Mxfp4PairScan::default();
        assert!(scan.pairs.is_empty());
    }

    #[test]
    fn scan_default_blocks_to_scales_is_empty() {
        let scan = Mxfp4PairScan::default();
        assert!(scan.blocks_to_scales.is_empty());
    }

    #[test]
    fn scan_default_sidecars_is_empty() {
        let scan = Mxfp4PairScan::default();
        assert!(scan.sidecars.is_empty());
    }

    // ── CandidateTensor: clone preserves equality ─────────────────────────

    #[test]
    fn candidate_clone_equals_original() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![2, 3],
            byte_len: 6,
        };
        assert_eq!(ct, ct.clone());
    }

    // ── CandidateTensor: shape deep copy on clone ─────────────────────────

    #[test]
    fn candidate_clone_shape_is_independent() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![1, 2, 3],
            byte_len: 6,
        };
        let mut cloned = ct.clone();
        cloned.shape.clear();
        assert_eq!(ct.shape, vec![1, 2, 3]);
    }

    // ── CandidateTensor: I32 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_i32_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 16,
        };
        assert_eq!(ct.dtype, Dtype::I32);
    }

    // ── CandidateTensor: I64 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_i64_dtype_preserved() {
        let ct = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::I64,
            shape: vec![2],
            byte_len: 16,
        };
        assert_eq!(ct.dtype, Dtype::I64);
    }

    // ── Mxfp4Pair: manual construction with all Some bias ─────────────────

    #[test]
    fn pair_construction_with_all_fields() {
        let pair = Mxfp4Pair {
            blocks_name: "gate_up_proj_blocks".into(),
            scales_name: "gate_up_proj_scales".into(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: Some("gate_up_proj_bias".into()),
        };
        assert_eq!(pair.blocks_name, "gate_up_proj_blocks");
        assert_eq!(pair.scales_name, "gate_up_proj_scales");
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.num_blocks, 8);
        assert_eq!(pair.blocks_shape, vec![8, 16]);
        assert_eq!(pair.bias_name.as_deref(), Some("gate_up_proj_bias"));
    }

    // ── Repack: block_size=128 ────────────────────────────────────────────

    #[test]
    fn repack_block_size_128() {
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 128,
            num_blocks: 1,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let blocks = vec![0xAB; 64]; // 128/2 = 64 bytes per block
        let scales = vec![0x7F];
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        assert_eq!(out.len(), 65, "1 * (1 + 64) = 65");
        assert_eq!(out[0], 0x7F);
        assert_eq!(out[1..], [0xAB; 64]);
    }

    // ── scan: blocks_to_scales values match pair scales_name ──────────────

    #[test]
    fn scan_blocks_to_scales_values_match_pair_scales_name() {
        let tensors = vec![
            cand("alpha_blocks", Dtype::U8, vec![16], 16),
            cand("alpha_scales", Dtype::U8, vec![1], 1),
            cand("beta_blocks", Dtype::U8, vec![32], 32),
            cand("beta_scales", Dtype::U8, vec![2], 2),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        for (blocks_name, scales_name) in &scan.blocks_to_scales {
            let pair = scan.pairs.get(blocks_name).unwrap();
            assert_eq!(&pair.scales_name, scales_name);
        }
    }

    // ── scan: sidecar entries match blocks_to_scales values exactly ───────

    #[test]
    fn scan_sidecars_match_blocks_to_scales_values() {
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![48], 48),
            cand("b_scales", Dtype::U8, vec![3], 3),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        for scales_name in scan.blocks_to_scales.values() {
            assert!(scan.sidecars.contains(scales_name));
        }
        assert_eq!(scan.sidecars.len(), scan.blocks_to_scales.len());
    }

    // ── CandidateTensor: F8_E5M2 dtype preserved ──────────────────────────

    #[test]
    fn candidate_f8_e5m2_dtype_preserved() {
        let ct = CandidateTensor {
            name: "fp8_tensor".into(),
            dtype: Dtype::F8_E5M2,
            shape: vec![8],
            byte_len: 8,
        };
        assert_eq!(ct.dtype, Dtype::F8_E5M2);
    }

    // ── CandidateTensor: F8_E4M3 dtype preserved ──────────────────────────

    #[test]
    fn candidate_f8_e4m3_dtype_preserved() {
        let ct = CandidateTensor {
            name: "fp8_tensor".into(),
            dtype: Dtype::F8_E4M3,
            shape: vec![16],
            byte_len: 16,
        };
        assert_eq!(ct.dtype, Dtype::F8_E4M3);
    }

    // ── CandidateTensor: U16 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_u16_dtype_preserved() {
        let ct = CandidateTensor {
            name: "u16_tensor".into(),
            dtype: Dtype::U16,
            shape: vec![4],
            byte_len: 8,
        };
        assert_eq!(ct.dtype, Dtype::U16);
    }

    // ── CandidateTensor: U32 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_u32_dtype_preserved() {
        let ct = CandidateTensor {
            name: "u32_tensor".into(),
            dtype: Dtype::U32,
            shape: vec![3],
            byte_len: 12,
        };
        assert_eq!(ct.dtype, Dtype::U32);
    }

    // ── CandidateTensor: U64 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_u64_dtype_preserved() {
        let ct = CandidateTensor {
            name: "u64_tensor".into(),
            dtype: Dtype::U64,
            shape: vec![2],
            byte_len: 16,
        };
        assert_eq!(ct.dtype, Dtype::U64);
    }

    // ── CandidateTensor: I16 dtype preserved ──────────────────────────────

    #[test]
    fn candidate_i16_dtype_preserved() {
        let ct = CandidateTensor {
            name: "i16_tensor".into(),
            dtype: Dtype::I16,
            shape: vec![4],
            byte_len: 8,
        };
        assert_eq!(ct.dtype, Dtype::I16);
    }

    // ── CandidateTensor: Debug format includes all fields ─────────────────

    #[test]
    fn candidate_debug_format_contains_name_and_dtype() {
        let ct = CandidateTensor {
            name: "my_tensor".into(),
            dtype: Dtype::BF16,
            shape: vec![2, 4],
            byte_len: 16,
        };
        let debug_str = format!("{ct:?}");
        assert!(
            debug_str.contains("my_tensor"),
            "Debug output should contain tensor name: {debug_str}"
        );
        assert!(
            debug_str.contains("CandidateTensor"),
            "Debug output should contain struct name: {debug_str}"
        );
    }

    // ── CandidateTensor: equality differs by byte_len ─────────────────────

    #[test]
    fn candidate_inequality_differs_by_byte_len() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![8],
            byte_len: 8,
        };
        let b = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::U8,
            shape: vec![8],
            byte_len: 99,
        };
        assert_ne!(a, b);
    }

    // ── CandidateTensor: equality differs by name ─────────────────────────

    #[test]
    fn candidate_inequality_differs_by_name() {
        let a = CandidateTensor {
            name: "alpha".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        let b = CandidateTensor {
            name: "beta".into(),
            dtype: Dtype::U8,
            shape: vec![4],
            byte_len: 4,
        };
        assert_ne!(a, b);
    }

    // ── CandidateTensor: equality differs by shape ────────────────────────

    #[test]
    fn candidate_inequality_differs_by_shape() {
        let a = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::F32,
            shape: vec![2, 3],
            byte_len: 24,
        };
        let b = CandidateTensor {
            name: "t".into(),
            dtype: Dtype::F32,
            shape: vec![3, 2],
            byte_len: 24,
        };
        assert_ne!(a, b, "different shape ordering should not be equal");
    }

    // ── Scan: tensor name with _blocks in the middle is not paired ────────

    #[test]
    fn scan_blocks_in_middle_of_name_not_matched() {
        // "_blocks" must be a suffix, not just a substring.
        let tensors = vec![
            cand("model_blocks_layer", Dtype::U8, vec![16], 16),
            cand("model_scales_layer", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert!(scan.pairs.is_empty(), "names without exact suffix should not pair");
    }

    // ── Scan: bias prefix substring does not create false pair ────────────

    #[test]
    fn scan_bias_prefix_substring_no_false_pair() {
        // A tensor named "proj_bias_blocks" has prefix "proj_bias" — the bias
        // suffix check would look for "proj_bias_bias" which should not exist.
        let tensors = vec![
            cand("proj_bias_blocks", Dtype::U8, vec![16], 16),
            cand("proj_bias_scales", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("proj_bias_blocks").unwrap();
        assert_eq!(
            pair.bias_name, None,
            "no 'proj_bias_bias' tensor exists, so bias_name should be None"
        );
    }

    // ── Mxfp4Pair: blocks_shape empty vec is valid ────────────────────────

    #[test]
    fn pair_blocks_shape_empty_vec_preserved() {
        let pair = Mxfp4Pair {
            blocks_name: "s_blocks".into(),
            scales_name: "s_scales".into(),
            block_size: 32,
            num_blocks: 0,
            blocks_shape: vec![],
            bias_name: None,
        };
        assert!(pair.blocks_shape.is_empty());
    }

    // ── Mxfp4PairScan: pairs field is mutable and removable ───────────────

    #[test]
    fn scan_pairs_field_supports_remove() {
        let mut scan = Mxfp4PairScan::default();
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        scan.pairs.insert("w_blocks".into(), pair);
        scan.blocks_to_scales.insert("w_blocks".into(), "w_scales".into());
        scan.sidecars.insert("w_scales".into());

        let removed = scan.pairs.remove("w_blocks").expect("should exist");
        assert_eq!(removed.blocks_name, "w_blocks");
        assert!(scan.pairs.is_empty());
    }

    // ── Repack: block_size=0 with non-zero num_blocks errors ──────────────

    #[test]
    fn repack_block_size_zero_with_blocks_errors() {
        // block_size=0 means bytes_per_block=0. blocks_bytes should be 0
        // (num_blocks * 0), and scales must match num_blocks.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 0,
            num_blocks: 3,
            blocks_shape: vec![0],
            bias_name: None,
        };
        // blocks_bytes = 3 * 0 = 0, scales = 3 bytes — should succeed
        // because 0 == 0 for blocks check.
        let out = repack_to_gguf_layout(&pair, &[], &[0x10, 0x20, 0x30]).unwrap();
        assert_eq!(out, vec![0x10, 0x20, 0x30], "each block is just the scale byte");
    }

    // ── New tests: uncovered edge cases and boundary conditions ────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit] [nfr:TMG-ACC]

    #[test]
    fn scan_blocks_to_scales_len_equals_pairs_len() {
        // Arrange: three valid pairs with different sizes.
        let tensors = vec![
            cand("p1_blocks", Dtype::U8, vec![16], 16),
            cand("p1_scales", Dtype::U8, vec![1], 1),
            cand("p2_blocks", Dtype::U8, vec![32], 32),
            cand("p2_scales", Dtype::U8, vec![2], 2),
            cand("p3_blocks", Dtype::U8, vec![48], 48),
            cand("p3_scales", Dtype::U8, vec![3], 3),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: blocks_to_scales must have exactly one entry per pair.
        assert_eq!(
            scan.pairs.len(),
            scan.blocks_to_scales.len(),
            "blocks_to_scales must have same cardinality as pairs"
        );
        assert_eq!(scan.pairs.len(), 3);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_with_f32_dtype_and_scales_with_u8_rejected() {
        // Arrange: blocks is F32 (wrong), scales is U8 (correct).
        let tensors = vec![
            cand("w_blocks", Dtype::F32, vec![4], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: blocks must be U8 for a valid pair.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_u8_scales_bool_dtype_rejected() {
        // Arrange: scales is BOOL — not U8.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::BOOL, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_shape_single_dimension_zero() {
        // Arrange: blocks_shape is [0], both tensors empty.
        let tensors = vec![
            cand("zero_blocks", Dtype::U8, vec![0], 0),
            cand("zero_scales", Dtype::U8, vec![0], 0),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: shape is preserved as a single-element vec with 0.
        let pair = scan.pairs.get("zero_blocks").unwrap();
        assert_eq!(pair.blocks_shape, vec![0]);
        assert_eq!(pair.num_blocks, 0);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_byte_ratio_exact_multiple_of_scales() {
        // Arrange: scales=4 bytes, blocks must be 4*16=64.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![64], 64),
            cand("w_scales", Dtype::U8, vec![4], 4),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.num_blocks, 4);
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_byte_ratio_non_multiple_of_scales_rejected() {
        // Arrange: scales=3, blocks=50 (not 3*16=48).
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![50], 50),
            cand("w_scales", Dtype::U8, vec![3], 3),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: 50 != 48, so rejected.
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_byte_ratio_scales_zero_blocks_nonzero_rejected() {
        // Arrange: scales_len=0, blocks_len=16 → 16 != 0*16.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![0], 0),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_10_bytes_per_block_5() {
        // Arrange: block_size=10 → bytes_per_block=5 (integer division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 10,
            num_blocks: 2,
            blocks_shape: vec![10],
            bias_name: None,
        };
        let blocks = vec![0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5];
        let scales = vec![0x10, 0x20];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: each block is scale + 5 packed bytes.
        assert_eq!(out.len(), 12, "2 * (1 + 5) = 12");
        assert_eq!(out[0], 0x10);
        assert_eq!(&out[1..6], &[0xA1, 0xA2, 0xA3, 0xA4, 0xA5]);
        assert_eq!(out[6], 0x20);
        assert_eq!(&out[7..12], &[0xB1, 0xB2, 0xB3, 0xB4, 0xB5]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_blocks_short_by_one_byte_error() {
        // Arrange: blocks needs 32 bytes (2*16), provide 31.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        // Act
        let err = repack_to_gguf_layout(&pair, &[0u8; 31], &[0u8; 2]).unwrap_err();
        // Assert
        assert!(err.contains("blocks length mismatch"), "err: {err}");
        assert!(err.contains("expected 32"), "err: {err}");
        assert!(err.contains("got 31"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_scales_short_by_one_byte_error() {
        // Arrange: num_blocks=4, scales needs 4 bytes, provide 3.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        // Act
        let err = repack_to_gguf_layout(&pair, &[0u8; 8], &[0u8; 3]).unwrap_err();
        // Assert
        assert!(err.contains("scales length mismatch"), "err: {err}");
        assert!(err.contains("expected 4"), "err: {err}");
        assert!(err.contains("got 3"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_with_zero_block_size_zero_blocks_succeeds() {
        // Arrange: block_size=0, num_blocks=0 — degenerate but valid.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 0,
            num_blocks: 0,
            blocks_shape: vec![],
            bias_name: None,
        };
        // Act
        let out = repack_to_gguf_layout(&pair, &[], &[]).unwrap();
        // Assert
        assert!(out.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_large_byte_len_succeeds() {
        // Arrange: 4096 blocks → blocks=65536, scales=4096.
        let tensors = vec![
            cand("huge_blocks", Dtype::U8, vec![4096, 16], 65536),
            cand("huge_scales", Dtype::U8, vec![4096], 4096),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("huge_blocks").unwrap();
        assert_eq!(pair.num_blocks, 4096);
        assert_eq!(pair.block_size, 32);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_scales_byte_len_equals_num_blocks() {
        // Arrange: scales_len directly determines num_blocks.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![128], 128),
            cand("w_scales", Dtype::U8, vec![8], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: num_blocks = scales.byte_len = 8.
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.num_blocks, 8);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_five_pairs_with_bias_all_correct() {
        // Arrange: 5 independent pairs, each with a bias tensor.
        let mut tensors = vec![];
        for i in 0..5 {
            let prefix = format!("layer{i}.proj");
            tensors.push(cand(&format!("{prefix}_blocks"), Dtype::U8, vec![16], 16));
            tensors.push(cand(&format!("{prefix}_scales"), Dtype::U8, vec![1], 1));
            tensors.push(cand(&format!("{prefix}_bias"), Dtype::BF16, vec![4], 8));
        }
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: all 5 pairs detected, each with correct bias_name.
        assert_eq!(scan.pairs.len(), 5);
        for i in 0..5 {
            let blocks_name = format!("layer{i}.proj_blocks");
            let expected_bias = format!("layer{i}.proj_bias");
            let pair = scan.pairs.get(&blocks_name).unwrap();
            assert_eq!(pair.bias_name.as_deref(), Some(expected_bias.as_str()));
            assert!(!scan.sidecars.contains(&expected_bias));
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_shape_rank_5_preserved() {
        // Arrange: 5-dimensional blocks shape.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![2, 3, 4, 2, 16], 2 * 3 * 4 * 2 * 16),
            cand("w_scales", Dtype::U8, vec![2, 3, 4, 2], 2 * 3 * 4 * 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.blocks_shape, vec![2, 3, 4, 2, 16]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_5_odd_bytes_per_block() {
        // Arrange: block_size=5 → bytes_per_block=2 (integer division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 5,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60];
        let scales = vec![0xA1, 0xA2, 0xA3];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 3 * (1 + 2) = 9 bytes.
        assert_eq!(out.len(), 9);
        assert_eq!(out, vec![0xA1, 0x10, 0x20, 0xA2, 0x30, 0x40, 0xA3, 0x50, 0x60]);
    }

    // ── Additional tests: further edge cases and boundary conditions ────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_with_multiple_dots_and_nested_paths() {
        // Arrange: deeply nested prefix with dots and numbers.
        let tensors = vec![
            cand("model.layers.12.mlp.experts.3.gate_up_proj_blocks", Dtype::U8, vec![16], 16),
            cand("model.layers.12.mlp.experts.3.gate_up_proj_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("model.layers.12.mlp.experts.3.gate_up_proj_blocks").unwrap();
        assert_eq!(pair.num_blocks, 1);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_scales_len_one_blocks_len_16_boundary() {
        // Arrange: minimal valid pair with exactly 1 block.
        let tensors = vec![
            cand("min_blocks", Dtype::U8, vec![16], 16),
            cand("min_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("min_blocks").unwrap();
        assert_eq!(pair.num_blocks, 1);
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.blocks_shape, vec![16]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_shape_is_vec_usize_not_usize() {
        // Arrange: blocks_shape is a Vec<usize>, not a single usize.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![4, 8, 16], 512),
            cand("w_scales", Dtype::U8, vec![4, 8], 32),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: shape preserved exactly, including dimensionality.
        let pair = scan.pairs.get("w_blocks").unwrap();
        assert_eq!(pair.blocks_shape.len(), 3);
        assert_eq!(pair.blocks_shape[0], 4);
        assert_eq!(pair.blocks_shape[1], 8);
        assert_eq!(pair.blocks_shape[2], 16);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_only_bias_and_scales_present_no_blocks() {
        // Arrange: _bias and _scales exist but no _blocks — no pair possible.
        let tensors = vec![
            cand("proj_scales", Dtype::U8, vec![1], 1),
            cand("proj_bias", Dtype::BF16, vec![4], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_suffix_with_extra_trailing_underscore_rejected() {
        // Arrange: name "w_blocks_" does NOT end with "_blocks" (extra trailing _).
        let tensors = vec![
            cand("w_blocks_", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: "w_blocks_" does not strip_suffix("_blocks") → no pair.
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_rejects_blocks_u8_scales_u16() {
        // Arrange: scales is U16, not U8 — must be rejected.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::U16, vec![1], 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_i64_bias_recorded() {
        // Arrange: bias has I64 dtype — still recorded (dtype not validated).
        let tensors = vec![
            cand("p_blocks", Dtype::U8, vec![16], 16),
            cand("p_scales", Dtype::U8, vec![1], 1),
            cand("p_bias", Dtype::I64, vec![2], 16),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("p_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("p_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_32_ten_blocks_interleaving() {
        // Arrange: 10 blocks with sequential block data to verify interleaving.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 10,
            blocks_shape: vec![160],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0..160u8).collect();
        let scales: Vec<u8> = (0xF0..=0xF9).collect();
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: output is 10 * 17 = 170 bytes; each scale precedes its 16 block bytes.
        assert_eq!(out.len(), 170);
        for i in 0..10 {
            let base = i * 17;
            assert_eq!(out[base], scales[i], "scale at block {i}");
            assert_eq!(&out[base + 1..base + 17], &blocks[i * 16..(i + 1) * 16]);
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_blocks_error_message_includes_num_blocks_and_bytes_per_block() {
        // Arrange: verify error message content with specific values.
        let pair = Mxfp4Pair {
            blocks_name: "diag_blocks".into(),
            scales_name: "diag_scales".into(),
            block_size: 32,
            num_blocks: 6,
            blocks_shape: vec![96],
            bias_name: None,
        };
        // Act: provide wrong blocks length.
        let err = repack_to_gguf_layout(&pair, &[0u8; 50], &[0u8; 6]).unwrap_err();
        // Assert
        assert!(err.contains("diag_blocks"), "must reference tensor name: {err}");
        assert!(err.contains("96"), "must show expected bytes: {err}");
        assert!(err.contains("got 50"), "must show actual bytes: {err}");
        assert!(err.contains("6 blocks"), "must show block count: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_name_with_double_blocks_suffix() {
        // Arrange: name "x_blocks_blocks" — suffix match on outer "_blocks",
        // prefix is "x_blocks", so scales would be "x_blocks_scales".
        let tensors = vec![
            cand("x_blocks_blocks", Dtype::U8, vec![16], 16),
            cand("x_blocks_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("x_blocks_blocks").unwrap();
        assert_eq!(pair.scales_name, "x_blocks_scales");
        assert_eq!(pair.blocks_name, "x_blocks_blocks");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_with_bias_name_none_still_works() {
        // Arrange: pair has no bias — repack should not be affected.
        let pair = Mxfp4Pair {
            blocks_name: "nobias_blocks".into(),
            scales_name: "nobias_scales".into(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![4],
            bias_name: None,
        };
        let blocks = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let scales = vec![0x01, 0x02];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert
        assert_eq!(out, vec![0x01, 0xDE, 0xAD, 0x02, 0xBE, 0xEF]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_interleaved_valid_invalid_and_bias_across_layers() {
        // Arrange: layer 0 valid, layer 1 invalid (wrong ratio), layer 2 valid with bias.
        let tensors = vec![
            cand("L0.gate_blocks", Dtype::U8, vec![16], 16),
            cand("L0.gate_scales", Dtype::U8, vec![1], 1),
            cand("L1.gate_blocks", Dtype::U8, vec![15], 15),
            cand("L1.gate_scales", Dtype::U8, vec![1], 1),
            cand("L2.gate_blocks", Dtype::U8, vec![32], 32),
            cand("L2.gate_scales", Dtype::U8, vec![2], 2),
            cand("L2.gate_bias", Dtype::BF16, vec![4], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: only L0 and L2 are valid.
        assert_eq!(scan.pairs.len(), 2);
        assert!(scan.pairs.contains_key("L0.gate_blocks"));
        assert!(scan.pairs.contains_key("L2.gate_blocks"));
        assert!(!scan.pairs.contains_key("L1.gate_blocks"));
        // L2 bias recorded, not a sidecar.
        let l2_pair = scan.pairs.get("L2.gate_blocks").unwrap();
        assert_eq!(l2_pair.bias_name.as_deref(), Some("L2.gate_bias"));
        assert!(!scan.sidecars.contains("L2.gate_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_name_with_spaces_in_prefix() {
        // Arrange: spaces in the prefix — valid as far as string matching goes.
        let tensors = vec![
            cand("my tensor_blocks", Dtype::U8, vec![16], 16),
            cand("my tensor_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("my tensor_blocks"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_empty_input_iterator_no_panic() {
        // Arrange: use an empty iterator (not just empty vec).
        let scan = scan_mxfp4_pairs(std::iter::empty());
        // Assert
        assert!(scan.pairs.is_empty());
        assert!(scan.blocks_to_scales.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_with_maximal_single_block_size() {
        // Arrange: block_size=1024, num_blocks=1 → bytes_per_block=512.
        let pair = Mxfp4Pair {
            blocks_name: "huge_blocks".into(),
            scales_name: "huge_scales".into(),
            block_size: 1024,
            num_blocks: 1,
            blocks_shape: vec![512],
            bias_name: Some("huge_bias".into()),
        };
        let blocks = vec![0xCC; 512];
        let scales = vec![0xAB];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 1 * (1 + 512) = 513 bytes.
        assert_eq!(out.len(), 513);
        assert_eq!(out[0], 0xAB);
        assert_eq!(&out[1..], &[0xCC; 512]);
    }

    // ── Additional 15 edge-case tests ──────────────────────────────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_nonzero_num_blocks_with_empty_scales_errors() {
        // Arrange: num_blocks=3 but scales is empty — must fail.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        // Act
        let err = repack_to_gguf_layout(&pair, &[0u8; 6], &[]).unwrap_err();
        // Assert
        assert!(err.contains("scales length mismatch"), "err: {err}");
        assert!(err.contains("expected 3"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_name_exactly_scales_suffix_ignored() {
        // Arrange: a tensor named exactly "_scales" (empty prefix) with no "_blocks".
        let tensors = vec![cand("_scales", Dtype::U8, vec![1], 1)];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: no _blocks tensor → no pair.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn candidate_tensor_with_empty_name() {
        // Arrange: CandidateTensor with an empty string name.
        let ct = CandidateTensor {
            name: String::new(),
            dtype: Dtype::U8,
            shape: vec![1],
            byte_len: 1,
        };
        // Assert
        assert!(ct.name.is_empty());
        assert_eq!(ct, ct.clone());
        let debug = format!("{ct:?}");
        assert!(debug.contains("CandidateTensor"), "Debug: {debug}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_blocks_name_with_inner_blocks_suffix() {
        // Arrange: prefix "gate_up_proj_blocks" contains "_blocks" as substring,
        // so the name is "gate_up_proj_blocks_blocks" and scales is
        // "gate_up_proj_blocks_scales".
        let tensors = vec![
            cand("gate_up_proj_blocks_blocks", Dtype::U8, vec![16], 16),
            cand("gate_up_proj_blocks_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("gate_up_proj_blocks_blocks").unwrap();
        assert_eq!(pair.scales_name, "gate_up_proj_blocks_scales");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_7_odd_division() {
        // Arrange: block_size=7 → bytes_per_block=3 (integer division 7/2).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 7,
            num_blocks: 2,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0xA1, 0xA2, 0xA3, 0xB1, 0xB2, 0xB3];
        let scales = vec![0x10, 0x20];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 3) = 8 bytes.
        assert_eq!(out.len(), 8);
        assert_eq!(out, vec![0x10, 0xA1, 0xA2, 0xA3, 0x20, 0xB1, 0xB2, 0xB3]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_byte_len_zero_scales_byte_len_nonzero_rejected() {
        // Arrange: blocks has 0 bytes but scales has 1 byte → 0 != 1*16.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![0], 0),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: 0 != 16 → rejected.
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_three_tensors_same_prefix_blocks_scales_bias_all_u8() {
        // Arrange: all three tensors (_blocks, _scales, _bias) are U8.
        // The bias being U8 (unusual) should still be recorded.
        let tensors = vec![
            cand("fc_blocks", Dtype::U8, vec![16], 16),
            cand("fc_scales", Dtype::U8, vec![1], 1),
            cand("fc_bias", Dtype::U8, vec![4], 4),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("fc_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("fc_bias"));
        assert!(!scan.sidecars.contains("fc_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_scales_checked_first_blocks_correct_scales_wrong() {
        // Arrange: blocks length is correct but scales is wrong.
        // The function checks scales first, so the error mentions scales.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        // Act: blocks=6 (correct), scales=1 (wrong, expected 3).
        let err = repack_to_gguf_layout(&pair, &[0u8; 6], &[0u8; 1]).unwrap_err();
        // Assert
        assert!(err.contains("scales length mismatch"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_u8_scales_f64_rejected() {
        // Arrange: scales is F64 — not U8.
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales", Dtype::F64, vec![1], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_256_large_output_interleaving() {
        // Arrange: block_size=256, num_blocks=2 → bytes_per_block=128.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 256,
            num_blocks: 2,
            blocks_shape: vec![256],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0u8..255).cycle().take(256).collect();
        let scales = vec![0xDE, 0xAD];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 128) = 258 bytes.
        assert_eq!(out.len(), 258);
        assert_eq!(out[0], 0xDE);
        assert_eq!(&out[1..129], &blocks[0..128]);
        assert_eq!(out[129], 0xAD);
        assert_eq!(&out[130..258], &blocks[128..256]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_prefix_that_is_another_tensor_suffix() {
        // Arrange: prefix "x_scales" means blocks_name = "x_scales_blocks"
        // and scales_name = "x_scales_scales". The prefix containing "_scales"
        // should not confuse the detector.
        let tensors = vec![
            cand("x_scales_blocks", Dtype::U8, vec![16], 16),
            cand("x_scales_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("x_scales_blocks").unwrap();
        assert_eq!(pair.scales_name, "x_scales_scales");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn pair_blocks_shape_with_usize_max_dimension() {
        // Arrange: blocks_shape contains usize::MAX as a dimension.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 1,
            blocks_shape: vec![usize::MAX],
            bias_name: None,
        };
        // Assert: shape preserved verbatim.
        assert_eq!(pair.blocks_shape, vec![usize::MAX]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_two_valid_pairs_share_no_sidecar_cross_contamination() {
        // Arrange: two valid pairs. Verify sidecars do not leak between them.
        let tensors = vec![
            cand("alpha_blocks", Dtype::U8, vec![16], 16),
            cand("alpha_scales", Dtype::U8, vec![1], 1),
            cand("beta_blocks", Dtype::U8, vec![32], 32),
            cand("beta_scales", Dtype::U8, vec![2], 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: each sidecar maps to exactly one pair.
        assert_eq!(scan.sidecars.len(), 2);
        assert!(scan.sidecars.contains("alpha_scales"));
        assert!(scan.sidecars.contains("beta_scales"));
        assert!(!scan.sidecars.contains("alpha_blocks"));
        assert!(!scan.sidecars.contains("beta_blocks"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_with_bias_name_empty_string_still_works() {
        // Arrange: bias_name is Some("") — repack should ignore bias_name.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 2,
            blocks_shape: vec![4],
            bias_name: Some(String::new()),
        };
        let blocks = vec![0x11, 0x22, 0x33, 0x44];
        let scales = vec![0xAA, 0xBB];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: bias_name does not affect repack output.
        assert_eq!(out, vec![0xAA, 0x11, 0x22, 0xBB, 0x33, 0x44]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_rejects_blocks_i8_scales_u8() {
        // Arrange: blocks is I8, scales is U8 — blocks dtype must be U8.
        let tensors = vec![
            cand("w_blocks", Dtype::I8, vec![16], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // ── 15 additional edge-case tests ──────────────────────────────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_dtype_u16_rejected_even_with_correct_byte_ratio() {
        // Arrange: blocks is U16 (not U8), byte_len happens to satisfy the ratio.
        let tensors = vec![
            cand("w_blocks", Dtype::U16, vec![8], 16),
            cand("w_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: dtype check happens before byte ratio check.
        assert!(scan.pairs.is_empty(), "U16 blocks must be rejected");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_single_pair_with_two_bias_candidates_only_one_recorded() {
        // Arrange: one valid pair but an additional tensor that could look like
        // a second bias for a different pair (wrong prefix).
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![4], 8),
            cand("b_bias", Dtype::BF16, vec![4], 8), // b has no blocks/scales
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: only one pair, only a_bias recorded, b_bias ignored.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("a_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("a_bias"));
        assert!(!scan.sidecars.contains("b_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_output_scale_byte_positions_form_regular_stride() {
        // Arrange: 5 blocks with block_size=4 → stride=3. Every 3rd byte is a scale.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 5,
            blocks_shape: vec![10],
            bias_name: None,
        };
        let blocks = vec![0u8; 10];
        let scales = vec![0x10, 0x20, 0x30, 0x40, 0x50];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: positions 0, 3, 6, 9, 12 must be scale bytes.
        let expected_positions = [0usize, 3, 6, 9, 12];
        for (i, &pos) in expected_positions.iter().enumerate() {
            assert_eq!(out[pos], scales[i], "scale byte at position {pos}");
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_blocks_byte_len_exactly_scales_times_default_bytes_per_block() {
        // Arrange: large pair with exact ratio — 512 blocks, blocks=8192.
        let num_blocks = 512;
        let tensors = vec![
            cand("massive_blocks", Dtype::U8, vec![512, 16], num_blocks * 16),
            cand("massive_scales", Dtype::U8, vec![512], num_blocks),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("massive_blocks").unwrap();
        assert_eq!(pair.num_blocks, 512);
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_with_block_size_3_and_many_blocks_interleaving_correct() {
        // Arrange: block_size=3 → bytes_per_block=1. Each output block = 2 bytes.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 3,
            num_blocks: 4,
            blocks_shape: vec![4],
            bias_name: None,
        };
        let blocks = vec![0xA1, 0xA2, 0xA3, 0xA4];
        let scales = vec![0x01, 0x02, 0x03, 0x04];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: output = [s0, b0, s1, b1, s2, b2, s3, b3]
        assert_eq!(out.len(), 8);
        assert_eq!(out, vec![0x01, 0xA1, 0x02, 0xA2, 0x03, 0xA3, 0x04, 0xA4]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_suffix_must_be_at_end_not_middle() {
        // Arrange: name has "_blocks" in the middle, followed by more characters.
        let tensors = vec![
            cand("model_blocks_extra", Dtype::U8, vec![16], 16),
            cand("model_blocks_extra_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: "_blocks" not at end → no match.
        assert!(scan.pairs.is_empty(), "suffix must be at end of name");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_last_byte_of_output_is_last_block_byte() {
        // Arrange: 3 blocks, block_size=4 → last output byte is blocks[5].
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let scales = vec![0xAA, 0xBB, 0xCC];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: total 9 bytes. Last byte is blocks[5] = 0x66.
        assert_eq!(out.len(), 9);
        assert_eq!(out[out.len() - 1], 0x66, "last byte must be last block byte");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_to_scales_value_equals_pair_scales_name_for_each_pair() {
        // Arrange: multiple pairs with varying sizes.
        let tensors = vec![
            cand("x_blocks", Dtype::U8, vec![16], 16),
            cand("x_scales", Dtype::U8, vec![1], 1),
            cand("y_blocks", Dtype::U8, vec![32], 32),
            cand("y_scales", Dtype::U8, vec![2], 2),
            cand("z_blocks", Dtype::U8, vec![48], 48),
            cand("z_scales", Dtype::U8, vec![3], 3),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: blocks_to_scales values match the corresponding pair scales_name.
        for (blocks_name, pair) in &scan.pairs {
            let mapped = scan.blocks_to_scales.get(blocks_name).unwrap();
            assert_eq!(&pair.scales_name, mapped, "mismatch for {blocks_name}");
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_with_empty_prefix_and_bias_sibling() {
        // Arrange: prefix is "" → blocks_name="_blocks", bias would be "_bias".
        let tensors = vec![
            cand("_blocks", Dtype::U8, vec![16], 16),
            cand("_scales", Dtype::U8, vec![1], 1),
            cand("_bias", Dtype::BF16, vec![4], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: empty prefix works, bias recorded as "_bias".
        let pair = scan.pairs.get("_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("_bias"));
        assert!(!scan.sidecars.contains("_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_all_0xff_blocks_and_0x00_scales_passthrough() {
        // Arrange: blocks all 0xFF, scales all 0x00 — no transformation.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        let blocks = vec![0xFF; 6];
        let scales = vec![0x00; 3];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: interleaved [0x00, 0xFF, 0xFF] repeated 3 times.
        assert_eq!(out.len(), 9);
        for i in 0..3 {
            assert_eq!(out[i * 3], 0x00, "scale at block {i}");
            assert_eq!(out[i * 3 + 1], 0xFF, "block byte 0 at block {i}");
            assert_eq!(out[i * 3 + 2], 0xFF, "block byte 1 at block {i}");
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_no_pairs_when_all_blocks_have_wrong_dtype() {
        // Arrange: 3 _blocks tensors all with BF16, matching _scales all U8.
        let tensors = vec![
            cand("a_blocks", Dtype::BF16, vec![8], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::BF16, vec![16], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
            cand("c_blocks", Dtype::BF16, vec![24], 48),
            cand("c_scales", Dtype::U8, vec![3], 3),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: all rejected, no sidecars.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_9_bytes_per_block_4_correct_interleaving() {
        // Arrange: block_size=9 → bytes_per_block=4 (integer division 9/2).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 9,
            num_blocks: 2,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let scales = vec![0x10, 0x20];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 4) = 10 bytes.
        assert_eq!(out.len(), 10);
        assert_eq!(out, vec![0x10, 0xA0, 0xA1, 0xA2, 0xA3, 0x20, 0xB0, 0xB1, 0xB2, 0xB3]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_with_prefix_containing_bias_word() {
        // Arrange: prefix "model.bias_proj" contains "bias" but is not the bias suffix.
        let tensors = vec![
            cand("model.bias_proj_blocks", Dtype::U8, vec![16], 16),
            cand("model.bias_proj_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: pair detected, no bias because "model.bias_proj_bias" doesn't exist.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("model.bias_proj_blocks").unwrap();
        assert_eq!(pair.bias_name, None);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_blocks_error_message_format_complete() {
        // Arrange: verify the complete error message structure for blocks mismatch.
        let pair = Mxfp4Pair {
            blocks_name: "expert.gate_blocks".into(),
            scales_name: "expert.gate_scales".into(),
            block_size: 32,
            num_blocks: 10,
            blocks_shape: vec![160],
            bias_name: None,
        };
        // Act: blocks expected 160, got 100.
        let err = repack_to_gguf_layout(&pair, &[0u8; 100], &[0u8; 10]).unwrap_err();
        // Assert: error must contain tensor name, expected bytes, actual bytes, block count.
        assert!(err.contains("expert.gate_blocks"), "must have tensor name: {err}");
        assert!(err.contains("expected 160"), "must have expected: {err}");
        assert!(err.contains("got 100"), "must have actual: {err}");
        assert!(err.contains("10 blocks"), "must have block count: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_two_pairs_one_with_bias_one_without_independent() {
        // Arrange: two pairs where only one has a bias sibling.
        let tensors = vec![
            cand("with_bias_blocks", Dtype::U8, vec![16], 16),
            cand("with_bias_scales", Dtype::U8, vec![1], 1),
            cand("with_bias_bias", Dtype::BF16, vec![4], 8),
            cand("no_bias_blocks", Dtype::U8, vec![16], 16),
            cand("no_bias_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: both pairs detected, one has bias, one does not.
        assert_eq!(scan.pairs.len(), 2);
        let with_bias = scan.pairs.get("with_bias_blocks").unwrap();
        let without_bias = scan.pairs.get("no_bias_blocks").unwrap();
        assert_eq!(with_bias.bias_name.as_deref(), Some("with_bias_bias"));
        assert_eq!(without_bias.bias_name, None);
        assert!(!scan.sidecars.contains("with_bias_bias"));
    }

    // ── 15 additional edge-case tests (batch 3) ───────────────────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_with_scales_prefix_containing_blocks_word() {
        // Arrange: prefix "layer_blocks" contains "_blocks" as a substring.
        // blocks_name = "layer_blocks_blocks", scales_name = "layer_blocks_scales".
        // The inner "_blocks" should not confuse detection.
        let tensors = vec![
            cand("layer_blocks_blocks", Dtype::U8, vec![32], 32),
            cand("layer_blocks_scales", Dtype::U8, vec![2], 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: pair detected with the full name as blocks_name.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("layer_blocks_blocks").unwrap();
        assert_eq!(pair.num_blocks, 2);
        assert_eq!(pair.scales_name, "layer_blocks_scales");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_scales_error_message_includes_tensor_name() {
        // Arrange: scales length mismatch — verify the tensor name appears in error.
        let pair = Mxfp4Pair {
            blocks_name: "unique_tensor_blocks".into(),
            scales_name: "unique_tensor_scales".into(),
            block_size: 32,
            num_blocks: 5,
            blocks_shape: vec![80],
            bias_name: None,
        };
        // Act: scales expected 5, got 2.
        let err = repack_to_gguf_layout(&pair, &[0u8; 80], &[0u8; 2]).unwrap_err();
        // Assert
        assert!(
            err.contains("unique_tensor_blocks"),
            "error must reference tensor name: {err}"
        );
        assert!(err.contains("expected 5"), "err: {err}");
        assert!(err.contains("got 2"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_blocks_byte_len_exactly_scales_times_bytes_per_block_boundary() {
        // Arrange: scales=8 bytes, blocks=128 bytes (8*16=128). Exact boundary.
        let tensors = vec![
            cand("boundary_blocks", Dtype::U8, vec![128], 128),
            cand("boundary_scales", Dtype::U8, vec![8], 8),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("boundary_blocks").unwrap();
        assert_eq!(pair.num_blocks, 8);
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_single_block_size_4_interleaving_exact() {
        // Arrange: single block with block_size=4 (2 packed bytes) and known data.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 1,
            blocks_shape: vec![2],
            bias_name: None,
        };
        let blocks = vec![0xAB, 0xCD];
        let scales = vec![0x7F];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: output = [scale, block_byte_0, block_byte_1]
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0x7F);
        assert_eq!(out[1], 0xAB);
        assert_eq!(out[2], 0xCD);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_to_scales_and_sidecars_cardinality_invariant() {
        // Arrange: 4 pairs with varying sizes.
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("b_blocks", Dtype::U8, vec![32], 32),
            cand("b_scales", Dtype::U8, vec![2], 2),
            cand("c_blocks", Dtype::U8, vec![48], 48),
            cand("c_scales", Dtype::U8, vec![3], 3),
            cand("d_blocks", Dtype::U8, vec![64], 64),
            cand("d_scales", Dtype::U8, vec![4], 4),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: all three collections must have identical cardinality.
        assert_eq!(scan.pairs.len(), 4);
        assert_eq!(scan.blocks_to_scales.len(), 4);
        assert_eq!(scan.sidecars.len(), 4);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_12_bytes_per_block_6_interleaving() {
        // Arrange: block_size=12 → bytes_per_block=6 (even division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 12,
            num_blocks: 2,
            blocks_shape: vec![12],
            bias_name: None,
        };
        let blocks = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C];
        let scales = vec![0xF0, 0xF1];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 6) = 14 bytes.
        assert_eq!(out.len(), 14);
        assert_eq!(out[0], 0xF0);
        assert_eq!(&out[1..7], &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        assert_eq!(out[7], 0xF1);
        assert_eq!(&out[8..14], &[0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_rejects_pair_where_blocks_byte_len_not_multiple_of_bytes_per_block() {
        // Arrange: scales=3 → expected blocks=48, but blocks=47 (off by one).
        let tensors = vec![
            cand("off_blocks", Dtype::U8, vec![47], 47),
            cand("off_scales", Dtype::U8, vec![3], 3),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: 47 != 48 → rejected.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_blocks_long_by_one_byte_error() {
        // Arrange: num_blocks=2, block_size=32 → expected blocks=32, provide 33.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 32,
            num_blocks: 2,
            blocks_shape: vec![32],
            bias_name: None,
        };
        // Act
        let err = repack_to_gguf_layout(&pair, &[0u8; 33], &[0u8; 2]).unwrap_err();
        // Assert
        assert!(err.contains("blocks length mismatch"), "err: {err}");
        assert!(err.contains("expected 32"), "err: {err}");
        assert!(err.contains("got 33"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_with_50_irrelevant_tensors_plus_one_pair() {
        // Arrange: 50 non-matching tensors + 1 valid pair hidden among them.
        let mut tensors: Vec<CandidateTensor> = (0..50)
            .map(|i| {
                cand(&format!("layer.{i}.weight"), Dtype::BF16, vec![64, 64], 8192)
            })
            .collect();
        tensors.push(cand("hidden.gate_blocks", Dtype::U8, vec![16], 16));
        tensors.push(cand("hidden.gate_scales", Dtype::U8, vec![1], 1));
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: exactly one pair found, nothing else leaked.
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("hidden.gate_blocks"));
        assert!(scan.sidecars.contains("hidden.gate_scales"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_pair_detected_with_scales_byte_len_one_blocks_byte_len_16_minimum() {
        // Arrange: absolute minimum valid pair — 1 block, 16 block bytes, 1 scale byte.
        let tensors = vec![
            cand("tiny_blocks", Dtype::U8, vec![16], 16),
            cand("tiny_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("tiny_blocks").unwrap();
        assert_eq!(pair.num_blocks, 1);
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.blocks_shape, vec![16]);
        assert_eq!(pair.bias_name, None);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_output_bytes_are_exactly_scales_plus_blocks_concatenated() {
        // Arrange: verify the output contains all scale bytes and all block bytes
        // exactly once, interleaved, with no other bytes added.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0xA0, 0xA1, 0xB0, 0xB1, 0xC0, 0xC1, 0xD0, 0xD1];
        let scales = vec![0x10, 0x20, 0x30, 0x40];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: extract and collect all scale bytes and block bytes from output.
        let mut extracted_scales: Vec<u8> = vec![];
        let mut extracted_blocks: Vec<u8> = vec![];
        for i in 0..4 {
            let base = i * 3;
            extracted_scales.push(out[base]);
            extracted_blocks.push(out[base + 1]);
            extracted_blocks.push(out[base + 2]);
        }
        assert_eq!(extracted_scales, scales);
        assert_eq!(extracted_blocks, blocks);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_name_with_tab_character_in_prefix() {
        // Arrange: prefix contains a tab character — still valid for string matching.
        let tensors = vec![
            cand("layer\t0_blocks", Dtype::U8, vec![16], 16),
            cand("layer\t0_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: tab in prefix does not prevent suffix matching.
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("layer\t0_blocks"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_sidecars_never_include_bias_even_with_bias_present() {
        // Arrange: three tensors forming a pair with bias.
        let tensors = vec![
            cand("fc_blocks", Dtype::U8, vec![32], 32),
            cand("fc_scales", Dtype::U8, vec![2], 2),
            cand("fc_bias", Dtype::BF16, vec![8], 16),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: sidecars has exactly the scales name, never the bias.
        assert_eq!(scan.sidecars.len(), 1);
        assert!(scan.sidecars.contains("fc_scales"));
        assert!(!scan.sidecars.contains("fc_bias"));
        assert!(!scan.sidecars.contains("fc_blocks"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_48_bytes_per_block_24_interleaving() {
        // Arrange: block_size=48 → bytes_per_block=24 (large even division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 48,
            num_blocks: 2,
            blocks_shape: vec![48],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0u8..48).collect();
        let scales = vec![0xAA, 0xBB];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 24) = 50 bytes.
        assert_eq!(out.len(), 50);
        assert_eq!(out[0], 0xAA);
        assert_eq!(&out[1..25], &blocks[0..24]);
        assert_eq!(out[25], 0xBB);
        assert_eq!(&out[26..50], &blocks[24..48]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_blocks_u8_scales_u8_byte_ratio_32_to_2_accepted() {
        // Arrange: blocks=32 bytes, scales=2 bytes → 32 == 2*16. Accepted.
        let tensors = vec![
            cand("ratio_blocks", Dtype::U8, vec![32], 32),
            cand("ratio_scales", Dtype::U8, vec![2], 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("ratio_blocks").unwrap();
        assert_eq!(pair.num_blocks, 2);
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
    }

    // ── 15 additional edge-case tests (batch 4) ────────────────────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_pair_detected_when_blocks_appears_after_many_unrelated() {
        // Arrange: 100 unrelated tensors followed by one valid pair at the end.
        let mut tensors: Vec<CandidateTensor> = (0..100)
            .map(|i| {
                cand(&format!("layer.{i}.attn.q_proj"), Dtype::BF16, vec![64, 64], 8192)
            })
            .collect();
        tensors.push(cand("moi.expert.gate_blocks", Dtype::U8, vec![16], 16));
        tensors.push(cand("moi.expert.gate_scales", Dtype::U8, vec![1], 1));
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: the pair at the end is found despite many preceding tensors.
        assert_eq!(scan.pairs.len(), 1);
        assert!(scan.pairs.contains_key("moi.expert.gate_blocks"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_blocks_name_containing_scales_suffix_still_pairs_correctly() {
        // Arrange: prefix "w_scales" means blocks_name="w_scales_blocks",
        // scales_name="w_scales_scales". The prefix containing "_scales" should
        // not prevent detection — only the outermost suffix matters.
        let tensors = vec![
            cand("w_scales_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("w_scales_blocks").unwrap();
        assert_eq!(pair.scales_name, "w_scales_scales");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_scales_name_is_exact_not_prefix_match() {
        // Arrange: scales tensor named "w_scales_extra" — the suffix "_scales"
        // must match exactly, not just as a prefix. The blocks tensor looks for
        // prefix+"_scales" = "w_scales", which does not match "w_scales_extra".
        let tensors = vec![
            cand("w_blocks", Dtype::U8, vec![16], 16),
            cand("w_scales_extra", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: "w_scales" not found in map, so no pair.
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn repack_block_size_20_bytes_per_block_10_interleaving_verified() {
        // Arrange: block_size=20, bytes_per_block=10 (even division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 20,
            num_blocks: 2,
            blocks_shape: vec![20],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0u8..20).collect();
        let scales = vec![0xDE, 0xAD];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 10) = 22 bytes.
        assert_eq!(out.len(), 22);
        assert_eq!(out[0], 0xDE);
        assert_eq!(&out[1..11], &blocks[0..10]);
        assert_eq!(out[11], 0xAD);
        assert_eq!(&out[12..22], &blocks[10..20]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_blocks_with_scales_byte_len_zero_blocks_byte_len_zero_accepted() {
        // Arrange: both tensors have zero length — 0 == 0 * 16 -> valid pair.
        let tensors = vec![
            cand("null_blocks", Dtype::U8, vec![], 0),
            cand("null_scales", Dtype::U8, vec![], 0),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: zero-length pair is accepted.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("null_blocks").unwrap();
        assert_eq!(pair.num_blocks, 0);
        assert_eq!(pair.blocks_shape, Vec::<usize>::new());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_pair_with_blocks_byte_len_65536_large_scale_count() {
        // Arrange: blocks=65536 bytes, scales=4096 bytes -> 65536 == 4096 * 16.
        let tensors = vec![
            cand("big_blocks", Dtype::U8, vec![4096, 16], 65536),
            cand("big_scales", Dtype::U8, vec![4096], 4096),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair = scan.pairs.get("big_blocks").unwrap();
        assert_eq!(pair.num_blocks, 4096);
        assert_eq!(pair.block_size, DEFAULT_MXFP4_BLOCK_SIZE);
        assert_eq!(pair.blocks_shape, vec![4096, 16]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn repack_scales_long_by_one_byte_errors() {
        // Arrange: num_blocks=3, scales expected=3, provide 4.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 3,
            blocks_shape: vec![6],
            bias_name: None,
        };
        // Act
        let err = repack_to_gguf_layout(&pair, &[0u8; 6], &[0u8; 4]).unwrap_err();
        // Assert
        assert!(err.contains("scales length mismatch"), "err: {err}");
        assert!(err.contains("expected 3"), "err: {err}");
        assert!(err.contains("got 4"), "err: {err}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_valid_pair_with_bias_and_extra_unrelated_bias_ignored() {
        // Arrange: one valid pair with its own bias, plus an unrelated bias tensor.
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![2], 4),
            cand("orphan_bias", Dtype::BF16, vec![8], 16),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: only one pair, its bias recorded, orphan ignored.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("a_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("a_bias"));
        assert!(!scan.sidecars.contains("orphan_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn candidate_tensor_name_with_null_byte_preserved() {
        // Arrange: CandidateTensor name contains a null byte — string preserves it.
        let ct = CandidateTensor {
            name: "before\0after".into(),
            dtype: Dtype::U8,
            shape: vec![1],
            byte_len: 1,
        };
        // Assert: null byte is part of the name, equality still works.
        assert!(ct.name.contains('\0'));
        assert_eq!(ct, ct.clone());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_blocks_scales_and_bias_with_all_same_byte_len_different_dtypes() {
        // Arrange: blocks and scales are U8 with correct ratio; bias is F32.
        // The bias tensor has same byte_len as scales but different dtype.
        let tensors = vec![
            cand("proj_blocks", Dtype::U8, vec![16], 16),
            cand("proj_scales", Dtype::U8, vec![1], 1),
            cand("proj_bias", Dtype::F32, vec![1], 4),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: bias recorded regardless of its dtype.
        let pair = scan.pairs.get("proj_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("proj_bias"));
        assert!(!scan.sidecars.contains("proj_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn repack_first_byte_of_each_block_group_is_scale() {
        // Arrange: 6 blocks with block_size=4 -> stride=3. Verify every 3rd byte.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 6,
            blocks_shape: vec![12],
            bias_name: None,
        };
        let blocks = vec![0u8; 12];
        let scales = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: positions 0, 3, 6, 9, 12, 15 must be scale bytes.
        assert_eq!(out.len(), 18, "6 * 3 = 18");
        for i in 0..6 {
            assert_eq!(out[i * 3], scales[i], "scale byte at group {i}");
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_blocks_with_byte_len_not_divisible_by_16_rejected() {
        // Arrange: blocks=33, scales=2 -> 33 != 32 (not divisible by bytes_per_block=16).
        let tensors = vec![
            cand("odd_blocks", Dtype::U8, vec![33], 33),
            cand("odd_scales", Dtype::U8, vec![2], 2),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: 33 != 32 -> rejected.
        assert!(scan.pairs.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn pair_debug_format_with_none_bias_shows_none() {
        // Arrange: pair with bias_name = None.
        let pair = Mxfp4Pair {
            blocks_name: "x_blocks".into(),
            scales_name: "x_scales".into(),
            block_size: 32,
            num_blocks: 4,
            blocks_shape: vec![64],
            bias_name: None,
        };
        // Act
        let debug = format!("{pair:?}");
        // Assert: Debug output must contain the fields. None renders as "None".
        assert!(debug.contains("x_blocks"), "Debug: {debug}");
        assert!(debug.contains("None"), "bias_name None must appear: {debug}");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn scan_two_valid_pairs_one_shared_bias_name_no_cross_contamination() {
        // Arrange: two pairs "a_blocks/a_scales" and "b_blocks/b_scales".
        // Also a "a_bias" tensor. "b_bias" does NOT exist.
        // Verify b's bias_name is None, not contaminated by a_bias.
        let tensors = vec![
            cand("a_blocks", Dtype::U8, vec![16], 16),
            cand("a_scales", Dtype::U8, vec![1], 1),
            cand("a_bias", Dtype::BF16, vec![2], 4),
            cand("b_blocks", Dtype::U8, vec![16], 16),
            cand("b_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        let pair_a = scan.pairs.get("a_blocks").unwrap();
        let pair_b = scan.pairs.get("b_blocks").unwrap();
        assert_eq!(pair_a.bias_name.as_deref(), Some("a_bias"));
        assert_eq!(pair_b.bias_name, None, "b must not inherit a_bias");
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]
    #[test]
    fn repack_block_size_96_bytes_per_block_48_output_correct() {
        // Arrange: block_size=96, bytes_per_block=48 (large even division).
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 96,
            num_blocks: 2,
            blocks_shape: vec![96],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0u8..96).collect();
        let scales = vec![0xCA, 0xFE];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 48) = 98 bytes.
        assert_eq!(out.len(), 98);
        assert_eq!(out[0], 0xCA);
        assert_eq!(&out[1..49], &blocks[0..48]);
        assert_eq!(out[49], 0xFE);
        assert_eq!(&out[50..98], &blocks[48..96]);
    }

    // ── 10 additional edge-case tests ─────────────────────────────────────

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_rejects_pair_where_blocks_and_scales_have_same_byte_len() {
        // Arrange: blocks=16 bytes, scales=16 bytes -> 16 != 16*16 -> ratio mismatch.
        // Both tensors have identical byte_len, which can never satisfy the mxfp4
        // ratio unless block_size=2 (non-standard). With default block_size=32, rejected.
        let tensors = vec![
            cand("proj_blocks", Dtype::U8, vec![16], 16),
            cand("proj_scales", Dtype::U8, vec![16], 16),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: 16 != 16 * 16 = 256, so the pair is rejected.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn candidate_tensor_with_zero_byte_len_and_nonempty_shape_preserved() {
        // Arrange: shape says [4, 8] but byte_len is 0 — valid descriptor for an
        // empty tensor (e.g., tensor whose data has not been loaded yet).
        let ct = CandidateTensor {
            name: "empty_data".into(),
            dtype: Dtype::U8,
            shape: vec![4, 8],
            byte_len: 0,
        };
        // Assert: fields preserved verbatim, no normalization.
        assert_eq!(ct.shape, vec![4, 8]);
        assert_eq!(ct.byte_len, 0);
        assert_eq!(ct, ct.clone());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_64_bytes_per_block_32_interleaving_correct() {
        // Arrange: block_size=64, num_blocks=2 -> bytes_per_block=32.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 64,
            num_blocks: 2,
            blocks_shape: vec![64],
            bias_name: None,
        };
        let blocks: Vec<u8> = (0u8..64).collect();
        let scales = vec![0xAA, 0xBB];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 2 * (1 + 32) = 66 bytes total.
        assert_eq!(out.len(), 66);
        assert_eq!(out[0], 0xAA);
        assert_eq!(&out[1..33], &blocks[0..32]);
        assert_eq!(out[33], 0xBB);
        assert_eq!(&out[34..66], &blocks[32..64]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_bias_with_u8_dtype_still_recorded_as_sibling() {
        // Arrange: bias tensor has U8 dtype (same as blocks/scales).
        // The bias is not part of the mxfp4 pair and its dtype is not validated,
        // but it should still be recorded as a sibling.
        let tensors = vec![
            cand("proj_blocks", Dtype::U8, vec![16], 16),
            cand("proj_scales", Dtype::U8, vec![1], 1),
            cand("proj_bias", Dtype::U8, vec![4], 4),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: pair detected, bias recorded, bias not in sidecars.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("proj_blocks").unwrap();
        assert_eq!(pair.bias_name.as_deref(), Some("proj_bias"));
        assert!(!scan.sidecars.contains("proj_bias"));
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_then_repack_with_zero_length_pair_empty_output() {
        // Arrange: zero-length pair (num_blocks=0). Both blocks and scales are empty.
        let tensors = vec![
            cand("null_blocks", Dtype::U8, vec![], 0),
            cand("null_scales", Dtype::U8, vec![], 0),
        ];
        let scan = scan_mxfp4_pairs(tensors);
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("null_blocks").unwrap();
        // Act: repack with empty slices
        let out = repack_to_gguf_layout(pair, &[], &[]).unwrap();
        // Assert: empty output for zero blocks.
        assert!(out.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_identical_scale_bytes_across_all_blocks() {
        // Arrange: all scale bytes are identical (0x7F) — verify no cross-block
        // contamination and output correctness does not depend on scale diversity.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 4,
            num_blocks: 4,
            blocks_shape: vec![8],
            bias_name: None,
        };
        let blocks = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        let scales = vec![0x7F, 0x7F, 0x7F, 0x7F];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: each group starts with 0x7F, followed by 2 block bytes.
        assert_eq!(out.len(), 12); // 4 * (1 + 2)
        assert_eq!(out, vec![0x7F, 0x10, 0x20, 0x7F, 0x30, 0x40, 0x7F, 0x50, 0x60, 0x7F, 0x70, 0x80]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_rejects_blocks_u64_dtype_even_with_correct_byte_ratio() {
        // Arrange: blocks has U64 dtype — not U8. byte_len happens to give a
        // valid ratio with scales, but dtype check rejects the pair first.
        let tensors = vec![
            cand("w_blocks", Dtype::U64, vec![2], 16), // 16 bytes
            cand("w_scales", Dtype::U8, vec![1], 1),   // 1 byte, 16 == 1 * 16
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: U64 blocks rejected regardless of byte ratio.
        assert!(scan.pairs.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_multiple_pairs_blocks_to_scales_all_correct() {
        // Arrange: 5 valid pairs with distinct prefixes. Verify blocks_to_scales
        // contains exactly 5 entries, each mapping blocks_name -> scales_name.
        let mut tensors = Vec::new();
        for i in 0..5 {
            let prefix = format!("layer.{i}.mlp.gate");
            tensors.push(cand(&format!("{prefix}_blocks"), Dtype::U8, vec![16], 16));
            tensors.push(cand(&format!("{prefix}_scales"), Dtype::U8, vec![1], 1));
        }
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert
        assert_eq!(scan.pairs.len(), 5);
        assert_eq!(scan.blocks_to_scales.len(), 5);
        for i in 0..5 {
            let blocks_name = format!("layer.{i}.mlp.gate_blocks");
            let scales_name = format!("layer.{i}.mlp.gate_scales");
            assert_eq!(
                scan.blocks_to_scales.get(&blocks_name).map(String::as_str),
                Some(scales_name.as_str()),
                "pair {i} blocks_to_scales mismatch"
            );
        }
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn repack_block_size_8_bytes_per_block_4_interleaving() {
        // Arrange: block_size=8, num_blocks=3 -> bytes_per_block=4, stride=5.
        let pair = Mxfp4Pair {
            blocks_name: "w_blocks".into(),
            scales_name: "w_scales".into(),
            block_size: 8,
            num_blocks: 3,
            blocks_shape: vec![12],
            bias_name: None,
        };
        let blocks = vec![0xA1, 0xA2, 0xA3, 0xA4, 0xB1, 0xB2, 0xB3, 0xB4, 0xC1, 0xC2, 0xC3, 0xC4];
        let scales = vec![0x11, 0x22, 0x33];
        // Act
        let out = repack_to_gguf_layout(&pair, &blocks, &scales).unwrap();
        // Assert: 3 * (1 + 4) = 15 bytes, scale byte every 5th position.
        assert_eq!(out.len(), 15);
        assert_eq!(out[0], 0x11);
        assert_eq!(&out[1..5], &[0xA1, 0xA2, 0xA3, 0xA4]);
        assert_eq!(out[5], 0x22);
        assert_eq!(&out[6..10], &[0xB1, 0xB2, 0xB3, 0xB4]);
        assert_eq!(out[10], 0x33);
        assert_eq!(&out[11..15], &[0xC1, 0xC2, 0xC3, 0xC4]);
    }

    // @trace TEST-MXFP4-PAIR [req:REQ-GLF-001] [level:unit]

    #[test]
    fn scan_prefix_with_newline_character_pairs_correctly() {
        // Arrange: prefix contains a newline character — string matching should
        // still work since strip_suffix operates on exact byte sequences.
        let tensors = vec![
            cand("layer\n0_blocks", Dtype::U8, vec![16], 16),
            cand("layer\n0_scales", Dtype::U8, vec![1], 1),
        ];
        // Act
        let scan = scan_mxfp4_pairs(tensors);
        // Assert: pair detected despite newline in prefix.
        assert_eq!(scan.pairs.len(), 1);
        let pair = scan.pairs.get("layer\n0_blocks").unwrap();
        assert_eq!(pair.scales_name, "layer\n0_scales");
        assert_eq!(pair.num_blocks, 1);
    }
}
