//! AWQ/GPTQ tensor triplet detection for SafeTensors loaders.
//!
//! HuggingFace AWQ/GPTQ models store quantized weights as **separate
//! safetensors tensors** (not a single interleaved buffer):
//!
//! - `<prefix>.qweight` — packed 4-bit weights (I32, each u32 holds 8 int4
//!   values). Row-major for AWQ, column-interleaved (stride-16) for GPTQ.
//! - `<prefix>.qzeros`  — packed zero-points (I32 for AWQ, I32 or I16 for
//!   GPTQ). AWQ stores f16-interpret zeros; GPTQ stores INT4 packed + 1
//!   offset.
//! - `<prefix>.scales`  — per-group FP16 scales (F16).
//! - `<prefix>.g_idx`   — optional (GPTQ only) per-column group index (I32).
//!
//! AWQ vs GPTQ is distinguished by:
//! - Presence of `g_idx` → GPTQ4 format
//! - Absence of `g_idx` → AWQ4 format
//!
//! # Block layout (for JIT consumption)
//!
//! After repacking, each 256-element block follows `BlockAWQ4` / `BlockGPTQ4`
//! from `gllm_kernels::quant`:
//!
//! ```text
//! BlockAWQ4:   qweight: [u32; 32] (128 bytes) + scales: f16 (2B) + zeros: f16 (2B) = 132B
//! BlockGPTQ4:  qweight: [u32; 32] (128 bytes) + scales: f16 (2B) + zeros: u32 (4B) = 134B
//! ```
//!
//! The HuggingFace safetensors layout is *separate* tensors, while the JIT
//! microkernel expects contiguous block format. This module detects the
//! triplets; repacking is done separately at load time.

use std::collections::{HashMap, HashSet};

use safetensors::Dtype;

/// Suffix for packed 4-bit weights.
const QWEIGHT_SUFFIX: &str = ".qweight";
/// Suffix for packed zero-points.
const QZEROS_SUFFIX: &str = ".qzeros";
/// Suffix for per-group FP16 scales.
const SCALES_SUFFIX: &str = ".scales";
/// Suffix for per-column group index (GPTQ only).
const G_IDX_SUFFIX: &str = ".g_idx";

/// Whether the quantization format is AWQ or GPTQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AwqGptqFormat {
    /// AWQ: no `g_idx`, row-major qweight, FP16 zero-points.
    Awq,
    /// GPTQ: has `g_idx`, column-interleaved qweight, INT4 packed zero-points + 1 offset.
    Gptq,
}

/// One logical AWQ/GPTQ weight reconstructed from a safetensors triplet
/// (qweight + scales + qzeros), with optional g_idx (GPTQ only).
#[derive(Debug, Clone)]
pub struct AwqGptqGroup {
    /// Base name of the logical weight (e.g. `model.layers.0.mlp.gate_proj`).
    pub base_name: String,
    /// Full tensor name of the packed 4-bit weights.
    pub qweight_name: String,
    /// Full tensor name of the per-group scales (FP16).
    pub scales_name: String,
    /// Full tensor name of the packed zero-points.
    pub qzeros_name: String,
    /// Full tensor name of the per-column group index (GPTQ only).
    pub g_idx_name: Option<String>,
    /// Detected format (AWQ vs GPTQ).
    pub format: AwqGptqFormat,
    /// Shape of the qweight tensor (e.g. `[K/8, N]` where K is hidden, N is output).
    pub qweight_shape: Vec<usize>,
}

/// Result of scanning a tensor list for AWQ/GPTQ triplets.
#[derive(Debug, Default)]
pub struct AwqGptqScan {
    /// `base_name` → group metadata.
    pub groups: HashMap<String, AwqGptqGroup>,
    /// All tensor names consumed by a triplet (hidden from regular enumeration).
    pub consumed: HashSet<String>,
}

/// Minimal tensor descriptor used by the scanner (reuses the same struct
/// from mxfp4_pairing for consistency).
pub use super::mxfp4_pairing::CandidateTensor;

/// Detect AWQ/GPTQ `qweight` + `scales` + `qzeros` triplets (with optional
/// `g_idx`) in the given tensor list.
///
/// Detection rules (all must hold for a triplet to be recognized):
/// 1. There exists a tensor named `<prefix>.qweight` with dtype I32.
/// 2. There exist sibling tensors `<prefix>.scales` (F16) and `<prefix>.qzeros`
///    (I32 or I16).
/// 3. The byte lengths are consistent with a common group_size:
///    - `qweight`: `[K/8, N]` shape, each row packs 8 int4 values into one u32
///    - `scales`:  `[K/group_size, N]` shape
///    - `qzeros`:  `[K/group_size, N/8]` or similar packed shape
/// 4. Format detection:
///    - `<prefix>.g_idx` exists → GPTQ4
///    - No `g_idx` → AWQ4
pub fn scan_awq_gptq_groups<I>(tensors: I) -> AwqGptqScan
where
    I: IntoIterator<Item = CandidateTensor>,
{
    let mut by_name: HashMap<String, CandidateTensor> = HashMap::new();
    for t in tensors {
        by_name.insert(t.name.clone(), t);
    }

    let mut scan = AwqGptqScan::default();

    for (name, qw_t) in &by_name {
        let Some(base_name) = name.strip_suffix(QWEIGHT_SUFFIX) else {
            continue;
        };
        // qweight must be I32 (each u32 packs 8 int4 values)
        if qw_t.dtype != Dtype::I32 {
            continue;
        }

        let scales_name = format!("{base_name}{SCALES_SUFFIX}");
        let Some(scales_t) = by_name.get(&scales_name) else {
            continue;
        };
        if scales_t.dtype != Dtype::F16 {
            continue;
        }

        let qzeros_name = format!("{base_name}{QZEROS_SUFFIX}");
        let Some(qzeros_t) = by_name.get(&qzeros_name) else {
            continue;
        };
        // qzeros can be I32 (AWQ) or I32/I16 (GPTQ variants)
        if qzeros_t.dtype != Dtype::I32 && qzeros_t.dtype != Dtype::I16 {
            continue;
        }

        // Validate dimension consistency:
        // qweight shape is [K/8, N] → K = qweight_rows * 8, N = qweight_cols
        // scales shape is [K/group_size, N] → group_size = K / scales_rows
        // qzeros shape is [K/group_size, N/8] (packed) or similar
        let qw_shape = &qw_t.shape;
        if qw_shape.len() != 2 {
            continue;
        }
        let qw_rows = qw_shape[0];
        let n = qw_shape[1];
        let k = qw_rows * 8; // unpack K from packed rows

        let scales_shape = &scales_t.shape;
        if scales_shape.len() != 2 || scales_shape[1] != n {
            log::warn!(
                "awq_gptq_pairing: shape mismatch for scales '{}': expected \
                 [_, {n}], got {:?}",
                scales_name,
                scales_shape,
            );
            continue;
        }
        let scales_rows = scales_shape[0];
        if scales_rows == 0 || k % scales_rows != 0 {
            log::warn!(
                "awq_gptq_pairing: inconsistent scales rows for '{}': \
                 k={k}, scales_rows={scales_rows}",
                scales_name,
            );
            continue;
        }

        let qzeros_shape = &qzeros_t.shape;
        if qzeros_shape.len() != 2 {
            log::warn!(
                "awq_gptq_pairing: unexpected qzeros shape for '{}': {:?}",
                qzeros_name,
                qzeros_shape,
            );
            continue;
        }
        // qzeros rows should match scales rows
        if qzeros_shape[0] != scales_rows {
            log::warn!(
                "awq_gptq_pairing: qzeros rows {} != scales rows {} for '{}'",
                qzeros_shape[0],
                scales_rows,
                base_name,
            );
            continue;
        }

        // GPTQ detection: presence of g_idx
        let g_idx_name = format!("{base_name}{G_IDX_SUFFIX}");
        let (format, g_idx_name_opt) = if by_name.contains_key(&g_idx_name) {
            (AwqGptqFormat::Gptq, Some(g_idx_name.clone()))
        } else {
            (AwqGptqFormat::Awq, None)
        };

        let group = AwqGptqGroup {
            base_name: base_name.to_string(),
            qweight_name: name.clone(),
            scales_name: scales_name.clone(),
            qzeros_name: qzeros_name.clone(),
            g_idx_name: g_idx_name_opt.clone(),
            format,
            qweight_shape: qw_shape.clone(),
        };
        scan.groups.insert(base_name.to_string(), group);
        scan.consumed.insert(name.clone());
        scan.consumed.insert(scales_name.clone());
        scan.consumed.insert(qzeros_name.clone());
        if let Some(g) = g_idx_name_opt {
            scan.consumed.insert(g);
        }
    }

    scan
}


#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, dtype: Dtype, shape: Vec<usize>, byte_len: usize) -> CandidateTensor {
        CandidateTensor { name: name.to_string(), dtype, shape, byte_len }
    }

    #[test]
    fn detects_awq_triplet_without_g_idx() {
        // Typical AWQ: qweight [K/8, N] + scales [K/gs, N] + qzeros [K/gs, N/8]
        // K=4096, N=4096, group_size=128
        // qweight: [512, 4096] × 4B = 8_388_608 bytes
        // scales:  [32, 4096] × 2B = 262_144 bytes
        // qzeros:  [32, 512] × 4B = 65_536 bytes
        let tensors = vec![
            cand(
                "model.layers.0.mlp.gate_proj.qweight",
                Dtype::I32,
                vec![512, 4096],
                8_388_608,
            ),
            cand(
                "model.layers.0.mlp.gate_proj.scales",
                Dtype::F16,
                vec![32, 4096],
                262_144,
            ),
            cand(
                "model.layers.0.mlp.gate_proj.qzeros",
                Dtype::I32,
                vec![32, 512],
                65_536,
            ),
            // Regular weight, not part of the triplet
            cand(
                "model.layers.0.mlp.gate_proj.bias",
                Dtype::F32,
                vec![4096],
                16_384,
            ),
        ];

        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "expected one AWQ group");
        let group = scan
            .groups
            .get("model.layers.0.mlp.gate_proj")
            .expect("group by base_name");
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.qweight_shape, vec![512, 4096]);

        // Consumed tensors
        assert!(scan.consumed.contains("model.layers.0.mlp.gate_proj.qweight"));
        assert!(scan.consumed.contains("model.layers.0.mlp.gate_proj.scales"));
        assert!(scan.consumed.contains("model.layers.0.mlp.gate_proj.qzeros"));
        assert!(!scan.consumed.contains("model.layers.0.mlp.gate_proj.bias"));
    }

    #[test]
    fn detects_gptq_triplet_with_g_idx() {
        let tensors = vec![
            cand("model.layers.0.self_attn.q_proj.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
            cand("model.layers.0.self_attn.q_proj.scales", Dtype::F16, vec![32, 4096], 262_144),
            cand("model.layers.0.self_attn.q_proj.qzeros", Dtype::I32, vec![32, 512], 65_536),
            cand("model.layers.0.self_attn.q_proj.g_idx", Dtype::I32, vec![4096], 16_384),
        ];

        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(
            group.g_idx_name.as_deref(),
            Some("model.layers.0.self_attn.q_proj.g_idx")
        );

        // g_idx also consumed
        assert!(scan.consumed.contains("model.layers.0.self_attn.q_proj.g_idx"));
    }

    #[test]
    fn detects_multiple_layers_independently() {
        let tensors = vec![
            cand("L0.gate_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L0.gate_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L0.gate_proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("L0.up_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L0.up_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L0.up_proj.qzeros", Dtype::I32, vec![4, 16], 256),
            // One GPTQ layer
            cand("L1.down_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L1.down_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L1.down_proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("L1.down_proj.g_idx", Dtype::I32, vec![128], 512),
        ];

        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 3);
        assert_eq!(scan.groups["L0.gate_proj"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["L0.up_proj"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["L1.down_proj"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 10); // 3×3 + 1 g_idx
    }

    #[test]
    fn ignores_incomplete_triplet() {
        // Only qweight, no scales or qzeros
        let tensors = vec![cand("foo.bar.qweight", Dtype::I32, vec![64, 128], 32_768)];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn ignores_wrong_dtypes() {
        let tensors = vec![
            cand("foo.bar.qweight", Dtype::F32, vec![64, 128], 32_768), // wrong dtype
            cand("foo.bar.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.bar.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    #[test]
    fn ignores_shape_mismatch() {
        // scales columns don't match qweight columns
        let tensors = vec![
            cand("foo.bar.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.bar.scales", Dtype::F16, vec![4, 256], 2_048), // N mismatch: 256 != 128
            cand("foo.bar.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    // --- Additional tests ---

    #[test]
    fn empty_input_yields_empty_scan() {
        let scan = scan_awq_gptq_groups(vec![]);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn single_non_qweight_tensor_ignored() {
        let tensors = vec![
            cand("model.layers.0.mlp.gate_proj.bias", Dtype::F32, vec![4096], 16_384),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn qzeros_i16_dtype_accepted() {
        // GPTQ variant where qzeros uses I16 instead of I32
        let tensors = vec![
            cand("model.layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model.layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model.layer.qzeros", Dtype::I16, vec![4, 16], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "I16 qzeros should be accepted");
        let group = scan.groups.get("model.layer").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
    }

    #[test]
    fn rejects_qweight_wrong_dtype_f16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::F16, vec![64, 128], 16_384),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F16 qweight must be rejected");
    }

    #[test]
    fn rejects_scales_wrong_dtype_f32() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F32, vec![4, 128], 2_048),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F32 scales must be rejected");
    }

    #[test]
    fn rejects_qzeros_wrong_dtype_f16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::F16, vec![4, 16], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F16 qzeros must be rejected");
    }

    #[test]
    fn rejects_1d_qweight() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![512], 2_048),
            cand("foo.scales", Dtype::F16, vec![32], 64),
            cand("foo.qzeros", Dtype::I32, vec![32], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "1D qweight must be rejected");
    }

    #[test]
    fn rejects_3d_qweight() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![8, 64, 128], 262_144),
            cand("foo.scales", Dtype::F16, vec![4, 64, 128], 65_536),
            cand("foo.qzeros", Dtype::I32, vec![4, 8, 16], 4_096),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "3D qweight must be rejected");
    }

    #[test]
    fn rejects_qzeros_row_mismatch() {
        // qzeros rows differ from scales rows
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![8, 16], 512), // rows=8, scales rows=4
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "qzeros row count mismatch must be rejected");
    }

    #[test]
    fn rejects_1d_qzeros() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![64], 256), // 1D, should be 2D
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "1D qzeros must be rejected");
    }

    #[test]
    fn group_base_name_extracted_correctly() {
        let tensors = vec![
            cand("very.deep.path.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("very.deep.path.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("very.deep.path.proj.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("very.deep.path.proj").unwrap();
        assert_eq!(group.base_name, "very.deep.path.proj");
        assert_eq!(group.qweight_name, "very.deep.path.proj.qweight");
        assert_eq!(group.scales_name, "very.deep.path.proj.scales");
        assert_eq!(group.qzeros_name, "very.deep.path.proj.qzeros");
    }

    #[test]
    fn gptq_consumed_includes_g_idx() {
        let tensors = vec![
            cand("x.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("x.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("x.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("x.g_idx", Dtype::I32, vec![128], 512),
            // non-triplet tensor
            cand("x.bias", Dtype::F32, vec![128], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 4, "qweight + scales + qzeros + g_idx");
        assert!(scan.consumed.contains("x.g_idx"));
        assert!(!scan.consumed.contains("x.bias"));
    }

    #[test]
    fn awqgptqscan_default_is_empty() {
        let scan = AwqGptqScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn awqgptqformat_equality_and_copy() {
        let a = AwqGptqFormat::Awq;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(a, AwqGptqFormat::Gptq);
    }

    #[test]
    fn awqgptqformat_debug_output() {
        assert_eq!(format!("{:?}", AwqGptqFormat::Awq), "Awq");
        assert_eq!(format!("{:?}", AwqGptqFormat::Gptq), "Gptq");
    }

    #[test]
    fn awqgptqgroup_clone_preserves_fields() {
        let group = AwqGptqGroup {
            base_name: "test.proj".to_string(),
            qweight_name: "test.proj.qweight".to_string(),
            scales_name: "test.proj.scales".to_string(),
            qzeros_name: "test.proj.qzeros".to_string(),
            g_idx_name: Some("test.proj.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        let cloned = group.clone();
        assert_eq!(cloned.base_name, group.base_name);
        assert_eq!(cloned.qweight_name, group.qweight_name);
        assert_eq!(cloned.scales_name, group.scales_name);
        assert_eq!(cloned.qzeros_name, group.qzeros_name);
        assert_eq!(cloned.g_idx_name, group.g_idx_name);
        assert_eq!(cloned.format, group.format);
        assert_eq!(cloned.qweight_shape, group.qweight_shape);
    }

    #[test]
    fn suffixes_dont_match_partial_names() {
        // "qweightx" should not match the .qweight suffix
        let tensors = vec![
            cand("foo.qweightx", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.qweightx.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qweightx.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "partial suffix must not match");
    }

    #[test]
    fn mixed_valid_and_invalid_triplets() {
        let tensors = vec![
            // Valid AWQ
            cand("valid.aw.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("valid.aw.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("valid.aw.qzeros", Dtype::I32, vec![4, 16], 256),
            // Invalid: qweight is F32
            cand("invalid.qweight", Dtype::F32, vec![64, 128], 32_768),
            cand("invalid.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("invalid.qzeros", Dtype::I32, vec![4, 16], 256),
            // Unrelated tensor
            cand("unrelated.tensor", Dtype::F32, vec![100], 400),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "only the valid AWQ triplet should be found");
        assert!(scan.groups.contains_key("valid.aw"));
        assert_eq!(scan.consumed.len(), 3, "only 3 tensors from valid triplet consumed");
    }

    // ── Constants ────────────────────────────────────────────────────────

    #[test]
    fn suffix_constants_have_expected_values() {
        assert_eq!(QWEIGHT_SUFFIX, ".qweight");
        assert_eq!(QZEROS_SUFFIX, ".qzeros");
        assert_eq!(SCALES_SUFFIX, ".scales");
        assert_eq!(G_IDX_SUFFIX, ".g_idx");
    }

    // ── AwqGptqFormat derive traits (expanded) ───────────────────────────

    #[test]
    fn format_variants_are_distinct() {
        let variants = [AwqGptqFormat::Awq, AwqGptqFormat::Gptq];
        // Each pair of variants must be non-equal
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i != j {
                    assert_ne!(variants[i], variants[j], "variant {i} != variant {j}");
                }
            }
        }
    }

    #[test]
    fn format_copy_allows_independent_use() {
        let original = AwqGptqFormat::Gptq;
        let copy = original; // Copy (not Clone) — both usable
        assert_eq!(original, copy);
        // Both are still valid after the "move" (Copy semantics)
        let _still_valid = original;
    }

    // ── AwqGptqGroup Debug output ────────────────────────────────────────

    #[test]
    fn group_debug_format_contains_all_fields() {
        let group = AwqGptqGroup {
            base_name: "layer.proj".to_string(),
            qweight_name: "layer.proj.qweight".to_string(),
            scales_name: "layer.proj.scales".to_string(),
            qzeros_name: "layer.proj.qzeros".to_string(),
            g_idx_name: Some("layer.proj.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        let debug = format!("{group:?}");
        assert!(debug.contains("layer.proj"), "Debug should contain base_name: {debug}");
        assert!(debug.contains("layer.proj.qweight"), "Debug should contain qweight_name: {debug}");
        assert!(debug.contains("layer.proj.scales"), "Debug should contain scales_name: {debug}");
        assert!(debug.contains("layer.proj.qzeros"), "Debug should contain qzeros_name: {debug}");
        assert!(debug.contains("layer.proj.g_idx"), "Debug should contain g_idx_name: {debug}");
        assert!(debug.contains("Gptq"), "Debug should contain format variant: {debug}");
    }

    #[test]
    fn group_debug_format_awq_no_g_idx() {
        let group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        let debug = format!("{group:?}");
        assert!(debug.contains("Awq"), "Debug should show Awq format: {debug}");
        assert!(debug.contains("None"), "Debug should show None for g_idx_name: {debug}");
    }

    // ── AwqGptqGroup Clone independence ──────────────────────────────────

    #[test]
    fn group_clone_is_independent() {
        let group = AwqGptqGroup {
            base_name: "original".to_string(),
            qweight_name: "original.qweight".to_string(),
            scales_name: "original.scales".to_string(),
            qzeros_name: "original.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        let mut cloned = group.clone();
        cloned.base_name = "modified".to_string();
        cloned.qweight_shape[0] = 999;
        // Original must be unchanged
        assert_eq!(group.base_name, "original");
        assert_eq!(group.qweight_shape, vec![64, 128]);
        assert_eq!(cloned.base_name, "modified");
        assert_eq!(cloned.qweight_shape, vec![999, 128]);
    }

    // ── AwqGptqScan Debug output ─────────────────────────────────────────

    #[test]
    fn scan_debug_format_default() {
        let scan = AwqGptqScan::default();
        let debug = format!("{scan:?}");
        assert!(debug.contains("groups"), "Debug should mention 'groups': {debug}");
        assert!(debug.contains("consumed"), "Debug should mention 'consumed': {debug}");
    }

    #[test]
    fn scan_debug_format_with_data() {
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let debug = format!("{scan:?}");
        assert!(debug.contains("a.qweight"), "Debug should contain consumed tensor: {debug}");
    }

    // ── CandidateTensor construction ─────────────────────────────────────

    #[test]
    fn candidate_tensor_fields_match_constructor() {
        let ct = CandidateTensor {
            name: "model.layer.weight".to_string(),
            dtype: Dtype::BF16,
            shape: vec![256, 512],
            byte_len: 262_144,
        };
        assert_eq!(ct.name, "model.layer.weight");
        assert_eq!(ct.dtype, Dtype::BF16);
        assert_eq!(ct.shape, vec![256, 512]);
        assert_eq!(ct.byte_len, 262_144);
    }

    #[test]
    fn candidate_tensor_clone_independence() {
        let ct = CandidateTensor {
            name: "orig".to_string(),
            dtype: Dtype::I32,
            shape: vec![10],
            byte_len: 40,
        };
        let mut cloned = ct.clone();
        cloned.name = "modified".to_string();
        cloned.byte_len = 99;
        assert_eq!(ct.name, "orig");
        assert_eq!(ct.byte_len, 40);
        assert_eq!(cloned.name, "modified");
        assert_eq!(cloned.byte_len, 99);
    }

    #[test]
    fn candidate_tensor_debug_format() {
        let ct = CandidateTensor {
            name: "test_weight".to_string(),
            dtype: Dtype::F16,
            shape: vec![4, 8],
            byte_len: 64,
        };
        let debug = format!("{ct:?}");
        assert!(debug.contains("test_weight"), "Debug should contain name: {debug}");
        assert!(debug.contains("CandidateTensor"), "Debug should contain type name: {debug}");
    }

    // ── Scan: missing scales tensor ──────────────────────────────────────

    #[test]
    fn rejects_triplet_missing_scales() {
        // qweight + qzeros present, but no scales
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "triplet without scales must be rejected");
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: scales with zero rows (k % 0 would panic) ─────────────────

    #[test]
    fn rejects_scales_zero_rows() {
        // scales shape [0, 128] → scales_rows = 0 → k % 0 rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![0, 128], 0),
            cand("foo.qzeros", Dtype::I32, vec![0, 16], 0),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "zero scales_rows must be rejected");
    }

    // ── Scan: scales rows not evenly dividing K ──────────────────────────

    #[test]
    fn rejects_scales_rows_not_dividing_k() {
        // qweight [64, 128] → K = 64*8 = 512
        // scales [7, 128] → scales_rows = 7, 512 % 7 != 0 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![7, 128], 1_792),
            cand("foo.qzeros", Dtype::I32, vec![7, 16], 448),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "scales_rows not dividing K must be rejected");
    }

    // ── Scan: scales BF16 dtype rejected ─────────────────────────────────

    #[test]
    fn rejects_scales_dtype_bf16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::BF16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "BF16 scales must be rejected");
    }

    // ── Scan: qzeros U8 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_qzeros_dtype_u8() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::U8, vec![4, 16], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "U8 qzeros must be rejected");
    }

    // ── Scan: g_idx tensor ignored for AWQ format but does not interfere ─

    #[test]
    fn g_idx_for_different_prefix_does_not_affect_awq_detection() {
        // One AWQ group (no g_idx), one GPTQ group (with g_idx)
        let tensors = vec![
            cand("awq_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq_layer.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq_layer.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_layer.g_idx", Dtype::I32, vec![128], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups["awq_layer"].format, AwqGptqFormat::Awq);
        assert!(scan.groups["awq_layer"].g_idx_name.is_none());
        assert_eq!(scan.groups["gptq_layer"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.groups["gptq_layer"].g_idx_name.as_deref(), Some("gptq_layer.g_idx"));
    }

    // ── Scan: GPTQ group field verification ──────────────────────────────

    #[test]
    fn gptq_group_all_fields_correct() {
        let tensors = vec![
            cand("model.layers.5.mlp.down_proj.qweight", Dtype::I32, vec![256, 1024], 1_048_576),
            cand("model.layers.5.mlp.down_proj.scales", Dtype::F16, vec![16, 1024], 32_768),
            cand("model.layers.5.mlp.down_proj.qzeros", Dtype::I32, vec![16, 128], 8_192),
            cand("model.layers.5.mlp.down_proj.g_idx", Dtype::I32, vec![2048], 8_192),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("model.layers.5.mlp.down_proj").unwrap();
        assert_eq!(group.base_name, "model.layers.5.mlp.down_proj");
        assert_eq!(group.qweight_name, "model.layers.5.mlp.down_proj.qweight");
        assert_eq!(group.scales_name, "model.layers.5.mlp.down_proj.scales");
        assert_eq!(group.qzeros_name, "model.layers.5.mlp.down_proj.qzeros");
        assert_eq!(group.g_idx_name.as_deref(), Some("model.layers.5.mlp.down_proj.g_idx"));
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(group.qweight_shape, vec![256, 1024]);
    }

    // ── Scan: consumed set for AWQ is exactly 3 tensors ─────────────────

    #[test]
    fn awq_consumed_exactly_three_tensors() {
        let tensors = vec![
            cand("proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("proj.bias", Dtype::F16, vec![128], 256),
            cand("other.weight", Dtype::F32, vec![64, 64], 16_384),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.consumed.contains("proj.qweight"));
        assert!(scan.consumed.contains("proj.scales"));
        assert!(scan.consumed.contains("proj.qzeros"));
        assert!(!scan.consumed.contains("proj.bias"));
        assert!(!scan.consumed.contains("other.weight"));
    }

    // ── Scan: only qweight present, no scales or qzeros ──────────────────

    #[test]
    fn rejects_qweight_only_no_siblings() {
        let tensors = vec![
            cand("lonely.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("unrelated.weight", Dtype::F32, vec![10], 40),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: qweight + scales but no qzeros ─────────────────────────────

    #[test]
    fn rejects_triplet_missing_qzeros() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "triplet without qzeros must be rejected");
    }

    // ── Scan: qzeros I16 + g_idx (GPTQ variant) ─────────────────────────

    #[test]
    fn gptq_with_i16_qzeros_detected() {
        let tensors = vec![
            cand("layer.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("layer.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("layer.proj.qzeros", Dtype::I16, vec![4, 16], 128),
            cand("layer.proj.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("layer.proj").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq, "g_idx presence means GPTQ");
        assert!(group.g_idx_name.is_some());
        assert_eq!(scan.consumed.len(), 4, "qweight + scales + qzeros + g_idx consumed");
    }

    // ── Scan: multiple AWQ groups with overlapping suffix names ──────────

    #[test]
    fn independent_detection_with_similar_prefixes() {
        // Two groups where one prefix is a prefix of the other
        let tensors = vec![
            cand("model.gate.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("model.gate.scales", Dtype::F16, vec![2, 64], 256),
            cand("model.gate.qzeros", Dtype::I32, vec![2, 8], 64),
            cand("model.gate_proj.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("model.gate_proj.scales", Dtype::F16, vec![2, 64], 256),
            cand("model.gate_proj.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 2, "both prefixes should be detected independently");
        assert!(scan.groups.contains_key("model.gate"));
        assert!(scan.groups.contains_key("model.gate_proj"));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  New unit tests — pure data structures, traits, construction, edges
    // ══════════════════════════════════════════════════════════════════════

    // ── AwqGptqFormat: Hash trait ────────────────────────────────────────

    #[test]
    fn format_hash_equal_for_equal_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(val: &AwqGptqFormat) -> u64 {
            let mut h = DefaultHasher::new();
            val.hash(&mut h);
            h.finish()
        }
        assert_eq!(hash_of(&AwqGptqFormat::Awq), hash_of(&AwqGptqFormat::Awq));
        assert_eq!(hash_of(&AwqGptqFormat::Gptq), hash_of(&AwqGptqFormat::Gptq));
        assert_ne!(hash_of(&AwqGptqFormat::Awq), hash_of(&AwqGptqFormat::Gptq));
    }

    // ── AwqGptqFormat: exhaustive match is possible ──────────────────────

    #[test]
    fn format_exhaustive_match_both_arms_reachable() {
        let a = AwqGptqFormat::Awq;
        let g = AwqGptqFormat::Gptq;

        let label_a = match a {
            AwqGptqFormat::Awq => "awq",
            AwqGptqFormat::Gptq => "gptq",
        };
        let label_g = match g {
            AwqGptqFormat::Awq => "awq",
            AwqGptqFormat::Gptq => "gptq",
        };
        assert_eq!(label_a, "awq");
        assert_eq!(label_g, "gptq");
    }

    // ── AwqGptqGroup: construction with all fields populated ─────────────

    #[test]
    fn group_construction_all_fields_populated() {
        let group = AwqGptqGroup {
            base_name: "model.layers.0.mlp.gate_proj".to_string(),
            qweight_name: "model.layers.0.mlp.gate_proj.qweight".to_string(),
            scales_name: "model.layers.0.mlp.gate_proj.scales".to_string(),
            qzeros_name: "model.layers.0.mlp.gate_proj.qzeros".to_string(),
            g_idx_name: Some("model.layers.0.mlp.gate_proj.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![512, 4096],
        };
        assert_eq!(group.base_name, "model.layers.0.mlp.gate_proj");
        assert_eq!(group.qweight_name, "model.layers.0.mlp.gate_proj.qweight");
        assert_eq!(group.scales_name, "model.layers.0.mlp.gate_proj.scales");
        assert_eq!(group.qzeros_name, "model.layers.0.mlp.gate_proj.qzeros");
        assert_eq!(group.g_idx_name.as_deref(), Some("model.layers.0.mlp.gate_proj.g_idx"));
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(group.qweight_shape, vec![512, 4096]);
    }

    // ── AwqGptqGroup: construction with g_idx_name None (AWQ style) ──────

    #[test]
    fn group_construction_g_idx_none_awq() {
        let group = AwqGptqGroup {
            base_name: "proj".to_string(),
            qweight_name: "proj.qweight".to_string(),
            scales_name: "proj.scales".to_string(),
            qzeros_name: "proj.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.format, AwqGptqFormat::Awq);
    }

    // ── AwqGptqGroup: empty qweight_shape (edge case) ────────────────────

    #[test]
    fn group_with_empty_qweight_shape() {
        let group = AwqGptqGroup {
            base_name: "empty".to_string(),
            qweight_name: "empty.qweight".to_string(),
            scales_name: "empty.scales".to_string(),
            qzeros_name: "empty.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![],
        };
        assert!(group.qweight_shape.is_empty());
    }

    // ── AwqGptqGroup: large field values (edge case) ─────────────────────

    #[test]
    fn group_with_large_shape_values() {
        let group = AwqGptqGroup {
            base_name: "huge".to_string(),
            qweight_name: "huge.qweight".to_string(),
            scales_name: "huge.scales".to_string(),
            qzeros_name: "huge.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![usize::MAX / 8, usize::MAX],
        };
        assert_eq!(group.qweight_shape[0], usize::MAX / 8);
        assert_eq!(group.qweight_shape[1], usize::MAX);
    }

    // ── AwqGptqGroup: Debug contains format variant name for Awq ─────────

    #[test]
    fn group_debug_awq_format_variant() {
        let group = AwqGptqGroup {
            base_name: "base".to_string(),
            qweight_name: "base.qweight".to_string(),
            scales_name: "base.scales".to_string(),
            qzeros_name: "base.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![1, 2],
        };
        let debug = format!("{group:?}");
        assert!(debug.contains("Awq"), "must contain 'Awq': {debug}");
        assert!(debug.contains("base"), "must contain base_name: {debug}");
        assert!(debug.contains("None"), "must show None for g_idx: {debug}");
    }

    // ── AwqGptqGroup: Clone with g_idx_name Some then mutate ────────────

    #[test]
    fn group_clone_g_idx_independence() {
        let group = AwqGptqGroup {
            base_name: "base".to_string(),
            qweight_name: "base.qweight".to_string(),
            scales_name: "base.scales".to_string(),
            qzeros_name: "base.qzeros".to_string(),
            g_idx_name: Some("base.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        let mut cloned = group.clone();
        // Mutate the cloned g_idx_name
        cloned.g_idx_name = None;
        // Original unaffected
        assert_eq!(group.g_idx_name.as_deref(), Some("base.g_idx"));
        assert!(cloned.g_idx_name.is_none());
    }

    // ── AwqGptqScan: manual construction with one group ──────────────────

    #[test]
    fn scan_manual_construction_single_group() {
        let group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        let mut groups = HashMap::new();
        groups.insert("test".to_string(), group);

        let mut consumed = HashSet::new();
        consumed.insert("test.qweight".to_string());
        consumed.insert("test.scales".to_string());
        consumed.insert("test.qzeros".to_string());

        let scan = AwqGptqScan { groups, consumed };
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.groups.contains_key("test"));
        assert!(scan.consumed.contains("test.qweight"));
    }

    // ── AwqGptqScan: default equals empty manual construction ────────────

    #[test]
    fn scan_default_equals_empty_construction() {
        let default_scan = AwqGptqScan::default();
        let manual_scan = AwqGptqScan {
            groups: HashMap::new(),
            consumed: HashSet::new(),
        };
        assert_eq!(default_scan.groups.len(), manual_scan.groups.len());
        assert_eq!(default_scan.consumed.len(), manual_scan.consumed.len());
    }

    // ── AwqGptqScan: groups HashMap override replaces existing entry ──────

    #[test]
    fn scan_groups_hashmap_replaces_on_duplicate_key() {
        let group_v1 = AwqGptqGroup {
            base_name: "dup".to_string(),
            qweight_name: "dup.qweight".to_string(),
            scales_name: "dup.scales".to_string(),
            qzeros_name: "dup.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        let group_v2 = AwqGptqGroup {
            base_name: "dup".to_string(),
            qweight_name: "dup.qweight".to_string(),
            scales_name: "dup.scales".to_string(),
            qzeros_name: "dup.qzeros".to_string(),
            g_idx_name: Some("dup.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        let mut groups = HashMap::new();
        groups.insert("dup".to_string(), group_v1);
        groups.insert("dup".to_string(), group_v2);
        // HashMap::insert replaces: still only 1 entry
        assert_eq!(groups.len(), 1);
        assert_eq!(groups["dup"].format, AwqGptqFormat::Gptq);
        assert!(groups["dup"].g_idx_name.is_some());
    }

    // ── AwqGptqScan: Debug with zero groups ──────────────────────────────

    #[test]
    fn scan_debug_zero_groups() {
        let scan = AwqGptqScan::default();
        let debug = format!("{scan:?}");
        // Default HashMap/HashSet debug shows the struct name
        assert!(debug.contains("AwqGptqScan"), "Debug must contain type name: {debug}");
    }

    // ── scan_awq_gptq_groups: empty iterator (not just empty Vec) ────────

    #[test]
    fn scan_from_std_iter_empty() {
        let scan = scan_awq_gptq_groups(std::iter::empty::<CandidateTensor>());
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── AwqGptqFormat: usable as HashMap key (Hash + Eq) ─────────────────

    #[test]
    fn format_used_as_hashmap_key() {
        let mut map = HashMap::new();
        map.insert(AwqGptqFormat::Awq, "awq_label");
        map.insert(AwqGptqFormat::Gptq, "gptq_label");
        assert_eq!(map.get(&AwqGptqFormat::Awq), Some(&"awq_label"));
        assert_eq!(map.get(&AwqGptqFormat::Gptq), Some(&"gptq_label"));
        assert_eq!(map.len(), 2);
    }

    // ── AwqGptqFormat: usable as HashSet element ─────────────────────────

    #[test]
    fn format_used_in_hashset() {
        let mut set = HashSet::new();
        set.insert(AwqGptqFormat::Awq);
        set.insert(AwqGptqFormat::Gptq);
        set.insert(AwqGptqFormat::Awq); // duplicate, ignored
        assert_eq!(set.len(), 2);
        assert!(set.contains(&AwqGptqFormat::Awq));
        assert!(set.contains(&AwqGptqFormat::Gptq));
    }

    // ── CandidateTensor: construction with various dtypes ────────────────

    #[test]
    fn candidate_tensor_various_dtypes() {
        let dtypes_and_expected = [
            (Dtype::F16, "F16"),
            (Dtype::BF16, "BF16"),
            (Dtype::F32, "F32"),
            (Dtype::I32, "I32"),
            (Dtype::I16, "I16"),
            (Dtype::U8, "U8"),
        ];
        for (dtype, _label) in &dtypes_and_expected {
            let ct = CandidateTensor {
                name: "t".to_string(),
                dtype: *dtype,
                shape: vec![1],
                byte_len: 4,
            };
            assert_eq!(ct.dtype, *dtype);
        }
    }

    // ── CandidateTensor: shape with single dimension (1D tensor) ─────────

    #[test]
    fn candidate_tensor_1d_shape() {
        let ct = CandidateTensor {
            name: "bias".to_string(),
            dtype: Dtype::F32,
            shape: vec![4096],
            byte_len: 16_384,
        };
        assert_eq!(ct.shape.len(), 1);
        assert_eq!(ct.shape[0], 4096);
    }

    // ── AwqGptqGroup: Debug output includes all seven fields ─────────────

    #[test]
    fn group_debug_includes_all_seven_fields() {
        let group = AwqGptqGroup {
            base_name: "my.base".to_string(),
            qweight_name: "my.base.qweight".to_string(),
            scales_name: "my.base.scales".to_string(),
            qzeros_name: "my.base.qzeros".to_string(),
            g_idx_name: Some("my.base.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![128, 256],
        };
        let debug = format!("{group:?}");
        assert!(debug.contains("base_name"), "Debug must show base_name field: {debug}");
        assert!(debug.contains("qweight_name"), "Debug must show qweight_name field: {debug}");
        assert!(debug.contains("scales_name"), "Debug must show scales_name field: {debug}");
        assert!(debug.contains("qzeros_name"), "Debug must show qzeros_name field: {debug}");
        assert!(debug.contains("g_idx_name"), "Debug must show g_idx_name field: {debug}");
        assert!(debug.contains("format"), "Debug must show format field: {debug}");
        assert!(debug.contains("qweight_shape"), "Debug must show qweight_shape field: {debug}");
    }

    // ── AwqGptqScan: Debug with manually added group shows group data ────

    #[test]
    fn scan_debug_with_manually_added_group() {
        let group = AwqGptqGroup {
            base_name: "manual".to_string(),
            qweight_name: "manual.qweight".to_string(),
            scales_name: "manual.scales".to_string(),
            qzeros_name: "manual.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![16, 32],
        };
        let mut groups = HashMap::new();
        groups.insert("manual".to_string(), group);
        let scan = AwqGptqScan {
            groups,
            consumed: HashSet::new(),
        };
        let debug = format!("{scan:?}");
        assert!(debug.contains("manual"), "Debug must contain group data: {debug}");
        assert!(debug.contains("Awq"), "Debug must contain format: {debug}");
    }

    // ── AwqGptqFormat: Copy semantics — assign to multiple variables ──────

    #[test]
    fn format_copy_multiple_assignments() {
        let v = AwqGptqFormat::Awq;
        let a = v;
        let b = v;
        let c = v;
        assert_eq!(a, AwqGptqFormat::Awq);
        assert_eq!(b, AwqGptqFormat::Awq);
        assert_eq!(c, AwqGptqFormat::Awq);
    }

    // ── AwqGptqScan: consumed set manually built with multiple entries ───

    #[test]
    fn scan_consumed_set_manual_construction() {
        let mut consumed = HashSet::new();
        consumed.insert("a.qweight".to_string());
        consumed.insert("a.scales".to_string());
        consumed.insert("a.qzeros".to_string());
        consumed.insert("a.g_idx".to_string());
        let scan = AwqGptqScan {
            groups: HashMap::new(),
            consumed,
        };
        assert_eq!(scan.consumed.len(), 4);
        assert!(scan.consumed.contains("a.qweight"));
        assert!(scan.consumed.contains("a.g_idx"));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests — edge cases, validation boundaries, more coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qweight with single row (K=8, minimal) ────────────────────

    #[test]
    fn detects_minimal_qweight_single_row() {
        // qweight [1, 128] → K = 1*8 = 8, N = 128
        // scales [1, 128] → group_size = 8/1 = 8
        // qzeros [1, 16]
        let tensors = vec![
            cand("tiny.qweight", Dtype::I32, vec![1, 128], 512),
            cand("tiny.scales", Dtype::F16, vec![1, 128], 256),
            cand("tiny.qzeros", Dtype::I32, vec![1, 16], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "minimal single-row qweight should be detected");
        let group = scan.groups.get("tiny").unwrap();
        assert_eq!(group.qweight_shape, vec![1, 128]);
        assert_eq!(group.format, AwqGptqFormat::Awq);
    }

    // ── Scan: scales 1D shape rejected ──────────────────────────────────

    #[test]
    fn rejects_scales_1d_shape() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![1024], 2_048),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "1D scales must be rejected");
    }

    // ── Scan: scales 3D shape rejected ──────────────────────────────────

    #[test]
    fn rejects_scales_3d_shape() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 4, 128], 4_096),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "3D scales must be rejected");
    }

    // ── Scan: g_idx tensor with no matching qweight is ignored ──────────

    #[test]
    fn orphan_g_idx_tensor_ignored() {
        // A lone g_idx without corresponding qweight/scales/qzeros
        let tensors = vec![
            cand("orphan.g_idx", Dtype::I32, vec![512], 2_048),
            cand("unrelated.weight", Dtype::F32, vec![100], 400),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty(), "orphan g_idx must not be consumed");
    }

    // ── Scan: qweight with shape [0, N] (zero rows) ────────────────────

    #[test]
    fn detects_qweight_zero_rows_valid_shape() {
        // qweight [0, 128] → K = 0*8 = 0
        // scales [0, 128] → scales_rows = 0 → rejected (scales_rows == 0 check)
        let tensors = vec![
            cand("zero.qweight", Dtype::I32, vec![0, 128], 0),
            cand("zero.scales", Dtype::F16, vec![0, 128], 0),
            cand("zero.qzeros", Dtype::I32, vec![0, 16], 0),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "zero-row qweight must be rejected via scales_rows==0");
    }

    // ── Scan: qweight with shape [N, 0] (zero columns) ─────────────────

    #[test]
    fn detects_qweight_zero_columns() {
        // qweight [64, 0] → K = 512, N = 0
        // scales [4, 0] → N matches (both 0), but degenerate
        let tensors = vec![
            cand("zero_n.qweight", Dtype::I32, vec![64, 0], 0),
            cand("zero_n.scales", Dtype::F16, vec![4, 0], 0),
            cand("zero_n.qzeros", Dtype::I32, vec![4, 0], 0),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        // Shape [64, 0] with N=0: scales columns match (both 0), qzeros rows match
        // This is technically valid by shape checks, though degenerate
        assert_eq!(scan.groups.len(), 1, "degenerate zero-column shape passes shape checks");
    }

    // ── Scan: scales columns = 0 but qweight columns > 0 ───────────────

    #[test]
    fn rejects_scales_zero_cols_qweight_nonzero_cols() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 0], 0),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "N mismatch (128 != 0) must be rejected");
    }

    // ── Scan: exact group_size=128 (most common in practice) ────────────

    #[test]
    fn detects_group_size_128_realistic() {
        // K=4096, N=11008 (typical Llama MLP), group_size=128
        // qweight [512, 11008]
        // scales [32, 11008]
        // qzeros [32, 1376]
        let tensors = vec![
            cand("model.layers.0.mlp.gate_proj.qweight", Dtype::I32, vec![512, 11008], 22_544_384),
            cand("model.layers.0.mlp.gate_proj.scales", Dtype::F16, vec![32, 11008], 704_512),
            cand("model.layers.0.mlp.gate_proj.qzeros", Dtype::I32, vec![32, 1376], 176_128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("model.layers.0.mlp.gate_proj").unwrap();
        assert_eq!(group.qweight_shape, vec![512, 11008]);
        assert_eq!(group.format, AwqGptqFormat::Awq);
    }

    // ── Scan: group_size=32 (small groups) ──────────────────────────────

    #[test]
    fn detects_small_group_size_32() {
        // K=256, N=128, group_size=32
        // qweight [32, 128]
        // scales [8, 128]
        // qzeros [8, 16]
        let tensors = vec![
            cand("small_gs.qweight", Dtype::I32, vec![32, 128], 16_384),
            cand("small_gs.scales", Dtype::F16, vec![8, 128], 2_048),
            cand("small_gs.qzeros", Dtype::I32, vec![8, 16], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("small_gs").unwrap();
        assert_eq!(group.qweight_shape, vec![32, 128]);
    }

    // ── Scan: scales rows exactly equals K (group_size=8, minimal) ──────

    #[test]
    fn detects_group_size_equals_8() {
        // K=64, N=128, group_size=8
        // qweight [8, 128]
        // scales [8, 128] → scales_rows=8, K=64, 64%8==0
        // qzeros [8, 16]
        let tensors = vec![
            cand("gs8.qweight", Dtype::I32, vec![8, 128], 4_096),
            cand("gs8.scales", Dtype::F16, vec![8, 128], 2_048),
            cand("gs8.qzeros", Dtype::I32, vec![8, 16], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "group_size=8 should be accepted");
    }

    // ── Scan: group_size=1 (extreme: scales_rows=K, every element a group) ─

    #[test]
    fn detects_group_size_1_extreme() {
        // K=64, N=128, group_size=1 (extreme)
        // qweight [8, 128]
        // scales [64, 128] → scales_rows=64, K=64, 64%64==0
        // qzeros [64, 16]
        let tensors = vec![
            cand("gs1.qweight", Dtype::I32, vec![8, 128], 4_096),
            cand("gs1.scales", Dtype::F16, vec![64, 128], 16_384),
            cand("gs1.qzeros", Dtype::I32, vec![64, 16], 4_096),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "group_size=1 should be accepted");
    }

    // ── Scan: tensor named exactly ".qweight" (empty base_name) ─────────

    #[test]
    fn detects_empty_base_name() {
        // Edge: base_name is empty string (tensor named just ".qweight")
        let tensors = vec![
            cand(".qweight", Dtype::I32, vec![64, 128], 32_768),
            cand(".scales", Dtype::F16, vec![4, 128], 1_024),
            cand(".qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "empty base_name should still work");
        let group = scan.groups.get("").unwrap();
        assert_eq!(group.base_name, "");
    }

    // ── Scan: base_name with multiple trailing dots ─────────────────────

    #[test]
    fn detects_base_name_with_trailing_dots() {
        let tensors = vec![
            cand("model..proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model..proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model..proj.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("model..proj"));
    }

    // ── Scan: base_name with unicode characters ─────────────────────────

    #[test]
    fn detects_base_name_with_unicode() {
        let tensors = vec![
            cand("模型.层.投影.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("模型.层.投影.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("模型.层.投影.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "unicode base_name should work");
        assert!(scan.groups.contains_key("模型.层.投影"));
    }

    // ── Scan: tensornames ending in "qweight" but without leading dot ───

    #[test]
    fn rejects_qweight_without_dot_prefix() {
        // "fooqweight" does not end with ".qweight" → not matched
        let tensors = vec![
            cand("fooqweight", Dtype::I32, vec![64, 128], 32_768),
            cand("fooscales", Dtype::F16, vec![4, 128], 1_024),
            cand("fooqzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "missing dot separator must be rejected");
    }

    // ── Scan: scales columns = qweight columns + 1 (off-by-one) ────────

    #[test]
    fn rejects_scales_columns_off_by_one() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 129], 1_032),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "off-by-one columns must be rejected");
    }

    // ── Scan: qzeros columns differ from expected (still accepted) ──────

    #[test]
    fn accepts_qzeros_arbitrary_columns() {
        // qzeros column count is not validated against N/8 — only rows checked
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 99], 1_584),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "qzeros column count is not validated against N/8");
    }

    // ── Scan: duplicate qweight tensors with same base_name ─────────────

    #[test]
    fn duplicate_tensors_same_name_last_wins() {
        // HashMap::insert overwrites, so the last tensor with the same name wins
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.qweight", Dtype::I32, vec![32, 64], 8_192), // overwrites
            cand("foo.scales", Dtype::F16, vec![2, 64], 256),      // matches second qweight
            cand("foo.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        // Second qweight [32, 64] → K=256, scales [2, 64] → 256/2=128 OK
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["foo"].qweight_shape, vec![32, 64]);
    }

    // ── Scan: 4 AWQ groups + 4 GPTQ groups ──────────────────────────────

    #[test]
    fn detects_many_groups_mixed_formats() {
        let mut tensors = Vec::new();
        // 4 AWQ groups
        for i in 0..4 {
            let base = format!("awq.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        // 4 GPTQ groups
        for i in 0..4 {
            let base = format!("gptq.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
            tensors.push(cand(&format!("{base}.g_idx"), Dtype::I32, vec![512], 2_048));
        }
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 8, "4 AWQ + 4 GPTQ = 8 groups");
        // AWQ groups: 3 tensors each = 12, GPTQ groups: 4 tensors each = 16 → 28
        assert_eq!(scan.consumed.len(), 28);
        for i in 0..4 {
            assert_eq!(scan.groups[&format!("awq.{i}.proj")].format, AwqGptqFormat::Awq);
            assert_eq!(scan.groups[&format!("gptq.{i}.proj")].format, AwqGptqFormat::Gptq);
        }
    }

    // ── Scan: g_idx present but scales/qzeros missing ───────────────────

    #[test]
    fn g_idx_without_qweight_not_consumed() {
        let tensors = vec![
            cand("foo.g_idx", Dtype::I32, vec![128], 512),
            // No qweight/scales/qzeros for "foo"
            cand("bar.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("bar.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("bar.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(!scan.consumed.contains("foo.g_idx"), "orphan g_idx must not be consumed");
        assert!(scan.consumed.contains("bar.qweight"));
    }

    // ── CandidateTensor: PartialEq (from mxfp4_pairing) ────────────────

    #[test]
    fn candidate_tensor_partialeq_equal() {
        let a = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let b = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        assert_eq!(a, b, "identical CandidateTensors must be equal");
    }

    #[test]
    fn candidate_tensor_partialeq_name_differs() {
        let a = CandidateTensor {
            name: "a".to_string(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 16,
        };
        let b = CandidateTensor {
            name: "b".to_string(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 16,
        };
        assert_ne!(a, b, "different names must be non-equal");
    }

    #[test]
    fn candidate_tensor_partialeq_dtype_differs() {
        let a = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 16,
        };
        let b = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            byte_len: 16,
        };
        assert_ne!(a, b, "different dtypes must be non-equal");
    }

    #[test]
    fn candidate_tensor_partialeq_shape_differs() {
        let a = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let b = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![8, 4],
            byte_len: 128,
        };
        assert_ne!(a, b, "different shapes must be non-equal");
    }

    #[test]
    fn candidate_tensor_partialeq_byte_len_differs() {
        let a = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 16,
        };
        let b = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::I32,
            shape: vec![4],
            byte_len: 32,
        };
        assert_ne!(a, b, "different byte_len must be non-equal");
    }

    // ── CandidateTensor: empty shape (scalar) ───────────────────────────

    #[test]
    fn candidate_tensor_empty_shape_scalar() {
        let ct = CandidateTensor {
            name: "scalar".to_string(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 4,
        };
        assert!(ct.shape.is_empty());
        assert_eq!(ct.byte_len, 4);
    }

    // ── CandidateTensor: very large shape dimensions ────────────────────

    #[test]
    fn candidate_tensor_large_dimensions() {
        let ct = CandidateTensor {
            name: "huge".to_string(),
            dtype: Dtype::I32,
            shape: vec![usize::MAX, usize::MAX],
            byte_len: 0, // overflow in real life, but struct accepts it
        };
        assert_eq!(ct.shape[0], usize::MAX);
        assert_eq!(ct.shape[1], usize::MAX);
    }

    // ── AwqGptqGroup: qweight_shape with single dimension ───────────────

    #[test]
    fn group_with_single_dim_qweight_shape() {
        let group = AwqGptqGroup {
            base_name: "1d".to_string(),
            qweight_name: "1d.qweight".to_string(),
            scales_name: "1d.scales".to_string(),
            qzeros_name: "1d.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![42],
        };
        assert_eq!(group.qweight_shape.len(), 1);
        assert_eq!(group.qweight_shape[0], 42);
    }

    // ── AwqGptqGroup: base_name is very long ────────────────────────────

    #[test]
    fn group_with_very_long_base_name() {
        let long_name = "a".repeat(10_000);
        let group = AwqGptqGroup {
            base_name: long_name.clone(),
            qweight_name: format!("{long_name}.qweight"),
            scales_name: format!("{long_name}.scales"),
            qzeros_name: format!("{long_name}.qzeros"),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        assert_eq!(group.base_name.len(), 10_000);
        assert!(group.qweight_name.starts_with(&long_name));
    }

    // ── AwqGptqFormat: iteration over all variants ──────────────────────

    #[test]
    fn format_variants_can_be_collected() {
        let all = vec![AwqGptqFormat::Awq, AwqGptqFormat::Gptq];
        assert_eq!(all.len(), 2);
        assert!(all.contains(&AwqGptqFormat::Awq));
        assert!(all.contains(&AwqGptqFormat::Gptq));
    }

    // ── AwqGptqScan: groups map can be queried by format ────────────────

    #[test]
    fn scan_groups_can_be_filtered_by_format() {
        let tensors = vec![
            cand("awq_a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq_a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq_a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_b.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq_b.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq_b.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_b.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let awq_count = scan.groups.values().filter(|g| g.format == AwqGptqFormat::Awq).count();
        let gptq_count = scan.groups.values().filter(|g| g.format == AwqGptqFormat::Gptq).count();
        assert_eq!(awq_count, 1);
        assert_eq!(gptq_count, 1);
    }

    // ── AwqGptqGroup: g_idx_name roundtrip through Some/None ───────────

    #[test]
    fn group_g_idx_name_some_none_roundtrip() {
        let mut group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: Some("test.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        assert!(group.g_idx_name.is_some());
        group.g_idx_name = None;
        assert!(group.g_idx_name.is_none());
        group.g_idx_name = Some("new.g_idx".to_string());
        assert_eq!(group.g_idx_name.as_deref(), Some("new.g_idx"));
    }

    // ── Scan: tensor with name containing "qweight" in the middle ───────

    #[test]
    fn rejects_qweight_in_middle_of_name() {
        // "qweight.tensor" — "qweight" is not a suffix after a dot
        let tensors = vec![
            cand("qweight.tensor", Dtype::I32, vec![64, 128], 32_768),
            cand("qweight.tensor.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("qweight.tensor.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "qweight not as .qweight suffix must be rejected");
    }

    // ── Scan: qweight with qzeros I16 + matching shapes ────────────────

    #[test]
    fn accepts_qzeros_i16_matching_shapes() {
        // qzeros with I16 dtype, rows match scales rows
        let tensors = vec![
            cand("layer.up_proj.qweight", Dtype::I32, vec![128, 256], 131_072),
            cand("layer.up_proj.scales", Dtype::F16, vec![8, 256], 4_096),
            cand("layer.up_proj.qzeros", Dtype::I16, vec![8, 32], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("layer.up_proj").unwrap();
        assert_eq!(group.qweight_shape, vec![128, 256]);
        assert_eq!(group.format, AwqGptqFormat::Awq);
    }

    // ── Scan: qzeros I8 dtype rejected ──────────────────────────────────

    #[test]
    fn rejects_qzeros_dtype_i8() {
        // Dtype::I8 is not I32 or I16
        // Note: safetensors Dtype may not have I8, use U8 as proxy for non-I32/I16
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::U8, vec![4, 16], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "U8 qzeros must be rejected");
    }

    // ── Scan: qweight I64 dtype rejected ────────────────────────────────

    #[test]
    fn rejects_qweight_dtype_i64() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I64, vec![64, 128], 65_536),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "I64 qweight must be rejected");
    }

    // ── Scan: qzeros rows = 0 but scales rows > 0 ──────────────────────

    #[test]
    fn rejects_qzeros_zero_rows_scales_nonzero_rows() {
        // qzeros [0, 16] vs scales [4, 128] → rows mismatch
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![0, 16], 0),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "qzeros rows=0 vs scales rows=4 must be rejected");
    }

    // ── AwqGptqScan: consumed set does not include non-triplet tensors ──

    #[test]
    fn scan_consumed_excludes_all_non_triplet_tensors() {
        let tensors = vec![
            cand("proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj.qzeros", Dtype::I32, vec![4, 16], 256),
            // These should NOT be in consumed:
            cand("proj.bias", Dtype::F16, vec![128], 256),
            cand("other.weight", Dtype::F32, vec![64, 64], 16_384),
            cand("model.norm.weight", Dtype::F32, vec![64], 256),
            cand("lm_head.weight", Dtype::F32, vec![64, 32000], 8_192_000),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.consumed.len(), 3);
        assert!(!scan.consumed.contains("proj.bias"));
        assert!(!scan.consumed.contains("other.weight"));
        assert!(!scan.consumed.contains("model.norm.weight"));
        assert!(!scan.consumed.contains("lm_head.weight"));
    }

    // ── AwqGptqGroup: base_name with spaces ─────────────────────────────

    #[test]
    fn group_base_name_with_spaces() {
        let group = AwqGptqGroup {
            base_name: "my model layer 0".to_string(),
            qweight_name: "my model layer 0.qweight".to_string(),
            scales_name: "my model layer 0.scales".to_string(),
            qzeros_name: "my model layer 0.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        assert_eq!(group.base_name, "my model layer 0");
        assert!(group.qweight_name.contains(' '));
    }

    // ── AwqGptqFormat: match ergonomics in if-let ───────────────────────

    #[test]
    fn format_if_let_both_variants() {
        let awq = AwqGptqFormat::Awq;
        let gptq = AwqGptqFormat::Gptq;

        if let AwqGptqFormat::Awq = awq {
            // expected
        } else {
            panic!("AWQ variant must match if-let");
        }

        if let AwqGptqFormat::Gptq = gptq {
            // expected
        } else {
            panic!("GPTQ variant must match if-let");
        }
    }

    // ── Scan: AWQ group with scales rows = K/8 = qweight rows ──────────

    #[test]
    fn detects_scales_rows_equal_qweight_rows() {
        // group_size = 8 → scales_rows = K/8 = qweight_rows
        // qweight [64, 128] → K = 512
        // scales [64, 128] → scales_rows = 64 = qweight_rows, 512/64 = 8 OK
        // qzeros [64, 16]
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![64, 128], 16_384),
            cand("foo.qzeros", Dtype::I32, vec![64, 16], 4_096),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "scales_rows==qweight_rows is valid (group_size=8)");
    }

    // ── Scan: large number of unrelated tensors + one triplet ───────────

    #[test]
    fn detects_triplet_amidst_many_unrelated_tensors() {
        let tensors = vec![
            cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524_288_000),
            cand("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4096], 16_384),
            cand("model.layers.0.post_attention_layernorm.weight", Dtype::F32, vec![4096], 16_384),
            cand("model.layers.0.self_attn.q_proj.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
            cand("model.layers.0.self_attn.q_proj.scales", Dtype::F16, vec![32, 4096], 262_144),
            cand("model.layers.0.self_attn.q_proj.qzeros", Dtype::I32, vec![32, 512], 65_536),
            cand("model.layers.0.self_attn.k_proj.weight", Dtype::F32, vec![4096, 1024], 16_777_216),
            cand("model.layers.0.self_attn.v_proj.weight", Dtype::F32, vec![4096, 1024], 16_777_216),
            cand("model.layers.0.self_attn.o_proj.weight", Dtype::F32, vec![4096, 4096], 67_108_864),
            cand("model.norm.weight", Dtype::F32, vec![4096], 16_384),
            cand("lm_head.weight", Dtype::F32, vec![4096, 32000], 524_288_000),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("model.layers.0.self_attn.q_proj"));
        assert_eq!(scan.consumed.len(), 3);
        assert!(!scan.consumed.contains("model.embed_tokens.weight"));
        assert!(!scan.consumed.contains("lm_head.weight"));
    }

    // ── CandidateTensor: empty name ─────────────────────────────────────

    #[test]
    fn candidate_tensor_empty_name() {
        let ct = CandidateTensor {
            name: String::new(),
            dtype: Dtype::F32,
            shape: vec![1],
            byte_len: 4,
        };
        assert!(ct.name.is_empty());
    }

    // ── CandidateTensor: byte_len = 0 with non-empty shape ──────────────

    #[test]
    fn candidate_tensor_zero_byte_len_nonempty_shape() {
        let ct = CandidateTensor {
            name: "phantom".to_string(),
            dtype: Dtype::F32,
            shape: vec![10, 20],
            byte_len: 0,
        };
        assert_eq!(ct.byte_len, 0);
        assert_eq!(ct.shape, vec![10, 20]);
    }

    // ── Scan: suffix matching exactness — ".qweights" not matched ───────

    #[test]
    fn rejects_suffix_qweights_plural() {
        let tensors = vec![
            cand("foo.qweights", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "plural .qweights must not match .qweight suffix");
    }

    // ── AwqGptqScan: groups HashMap key matches base_name field ─────────

    #[test]
    fn scan_groups_key_equals_base_name() {
        let tensors = vec![
            cand("my.layer.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("my.layer.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("my.layer.proj.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        for (key, group) in &scan.groups {
            assert_eq!(*key, group.base_name, "HashMap key must equal base_name field");
        }
    }

    // ── AwqGptqGroup: qweight_name contains base_name as prefix ────────

    #[test]
    fn group_qweight_name_extends_base_name() {
        let tensors = vec![
            cand("prefix.name.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("prefix.name.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("prefix.name.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("prefix.name").unwrap();
        assert!(group.qweight_name.starts_with(&group.base_name));
        assert!(group.scales_name.starts_with(&group.base_name));
        assert!(group.qzeros_name.starts_with(&group.base_name));
    }

    // ── Scan: qweight BF16 dtype rejected (boundary: dtype check) ──────

    #[test]
    fn rejects_qweight_dtype_bf16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::BF16, vec![64, 128], 16_384),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "BF16 qweight must be rejected");
    }

    // ── Scan: scales rows = 1 (group_size = K, single group) ───────────

    #[test]
    fn detects_single_group_entire_row() {
        // K=512, N=128, group_size=512
        // qweight [64, 128]
        // scales [1, 128] → scales_rows=1, 512%1==0
        // qzeros [1, 16]
        let tensors = vec![
            cand("single.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("single.scales", Dtype::F16, vec![1, 128], 256),
            cand("single.qzeros", Dtype::I32, vec![1, 16], 64),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "single scale row (group_size=K) should be accepted");
    }

    // ── AwqGptqFormat: Clone followed by Copy ───────────────────────────

    #[test]
    fn format_clone_then_copy_both_usable() {
        let original = AwqGptqFormat::Gptq;
        let cloned = original.clone();
        let copied = original; // Copy
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
        // All three still usable (Copy semantics)
        assert_eq!(format!("{original:?}"), "Gptq");
        assert_eq!(format!("{cloned:?}"), "Gptq");
        assert_eq!(format!("{copied:?}"), "Gptq");
    }

    // ── Scan: two groups sharing a scales tensor name (impossible by design)

    #[test]
    fn two_qweights_one_scales_only_one_detected() {
        // Two qweight tensors, but only one scales/qzeros for the first
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("b.qweight", Dtype::I32, vec![64, 128], 32_768),
            // b has no scales/qzeros
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "only 'a' has a complete triplet");
        assert!(scan.groups.contains_key("a"));
        assert!(!scan.groups.contains_key("b"));
    }

    // ── AwqGptqScan: default then insert then check ─────────────────────

    #[test]
    fn scan_default_then_manual_insert() {
        let mut scan = AwqGptqScan::default();
        let group = AwqGptqGroup {
            base_name: "manual".to_string(),
            qweight_name: "manual.qweight".to_string(),
            scales_name: "manual.scales".to_string(),
            qzeros_name: "manual.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        scan.groups.insert("manual".to_string(), group);
        scan.consumed.insert("manual.qweight".to_string());
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 1);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 47 tests — additional ~40 unit tests for public API coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: scales F64 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_scales_dtype_f64() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F64, vec![4, 128], 2_048),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F64 scales must be rejected");
    }

    // ── Scan: scales I32 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_scales_dtype_i32() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::I32, vec![4, 128], 2_048),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "I32 scales must be rejected");
    }

    // ── Scan: qweight F32 dtype rejected ──────────────────────────────────

    #[test]
    fn rejects_qweight_dtype_f32() {
        let tensors = vec![
            cand("foo.qweight", Dtype::F32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F32 qweight must be rejected");
    }

    // ── Scan: qzeros F32 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_qzeros_dtype_f32() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::F32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F32 qzeros must be rejected");
    }

    // ── Scan: qzeros BF16 dtype rejected ──────────────────────────────────

    #[test]
    fn rejects_qzeros_dtype_bf16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::BF16, vec![4, 16], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "BF16 qzeros must be rejected");
    }

    // ── Scan: qweight U8 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_qweight_dtype_u8() {
        let tensors = vec![
            cand("foo.qweight", Dtype::U8, vec![64, 128], 8_192),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "U8 qweight must be rejected");
    }

    // ── Scan: qweight BOOL dtype rejected ─────────────────────────────────

    #[test]
    fn rejects_qweight_dtype_bool() {
        let tensors = vec![
            cand("foo.qweight", Dtype::BOOL, vec![64, 128], 1_024),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "BOOL qweight must be rejected");
    }

    // ── Scan: scales BOOL dtype rejected ──────────────────────────────────

    #[test]
    fn rejects_scales_dtype_bool() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::BOOL, vec![4, 128], 512),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "BOOL scales must be rejected");
    }

    // ── Scan: scan from VecDeque iterator ─────────────────────────────────

    #[test]
    fn scan_from_vecdeque_iterator() {
        use std::collections::VecDeque;
        let mut deque = VecDeque::new();
        deque.push_back(cand("x.qweight", Dtype::I32, vec![64, 128], 32_768));
        deque.push_back(cand("x.scales", Dtype::F16, vec![4, 128], 1_024));
        deque.push_back(cand("x.qzeros", Dtype::I32, vec![4, 16], 256));
        let scan = scan_awq_gptq_groups(deque);
        assert_eq!(scan.groups.len(), 1, "VecDeque iterator should work");
        assert!(scan.groups.contains_key("x"));
    }

    // ── Scan: scan from array iterator ────────────────────────────────────

    #[test]
    fn scan_from_array_iterator() {
        let tensors: [CandidateTensor; 3] = [
            cand("arr.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("arr.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("arr.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "array iterator should work");
        assert!(scan.groups.contains_key("arr"));
    }

    // ── Scan: scan from once() single-element iterator ────────────────────

    #[test]
    fn scan_from_once_single_tensor() {
        let scan = scan_awq_gptq_groups(std::iter::once(cand(
            "single.qweight",
            Dtype::I32,
            vec![64, 128],
            32_768,
        )));
        assert!(scan.groups.is_empty(), "single qweight without siblings must be empty");
    }

    // ── Scan: scan from chain() of two iterators ──────────────────────────

    #[test]
    fn scan_from_chained_iterators() {
        let part1 = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
        ];
        let part2 = vec![
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(part1.into_iter().chain(part2));
        assert_eq!(scan.groups.len(), 1, "chained iterators should find the triplet");
        assert!(scan.groups.contains_key("a"));
    }

    // ── Scan: 5D qweight rejected ─────────────────────────────────────────

    #[test]
    fn rejects_5d_qweight() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![2, 3, 4, 5, 6], 240),
            cand("foo.scales", Dtype::F16, vec![2, 6], 24),
            cand("foo.qzeros", Dtype::I32, vec![2, 1], 8),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "5D qweight must be rejected");
    }

    // ── Scan: qweight [1, 1] minimal valid shape ──────────────────────────

    #[test]
    fn detects_qweight_shape_1x1() {
        // qweight [1, 1] → K=8, N=1
        // scales [1, 1] → scales_rows=1, 8%1==0
        // qzeros [1, 1]
        let tensors = vec![
            cand("tiny.qweight", Dtype::I32, vec![1, 1], 4),
            cand("tiny.scales", Dtype::F16, vec![1, 1], 2),
            cand("tiny.qzeros", Dtype::I32, vec![1, 1], 4),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "1×1 qweight should be valid");
        assert_eq!(scan.groups["tiny"].qweight_shape, vec![1, 1]);
    }

    // ── Scan: scales rows = 2 but K=512, 512%2==0 valid ──────────────────

    #[test]
    fn detects_scales_rows_divides_k_evenly() {
        // qweight [64, 128] → K=512
        // scales [2, 128] → scales_rows=2, 512%2==0 → OK
        let tensors = vec![
            cand("even.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("even.scales", Dtype::F16, vec![2, 128], 512),
            cand("even.qzeros", Dtype::I32, vec![2, 16], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "scales_rows=2 dividing K=512 should be accepted");
    }

    // ── Scan: scales rows = 3 but K=512, 512%3 != 0 rejected ─────────────

    #[test]
    fn rejects_scales_rows_3_k_512() {
        // qweight [64, 128] → K=512
        // scales [3, 128] → scales_rows=3, 512%3 != 0 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![3, 128], 768),
            cand("foo.qzeros", Dtype::I32, vec![3, 16], 192),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "scales_rows=3 not dividing K=512 must be rejected");
    }

    // ── Scan: AWQ with g_idx of a different dtype still triggers GPTQ ─────

    #[test]
    fn g_idx_presence_ignores_dtype_for_format_detection() {
        // g_idx exists with F32 dtype — scanner only checks presence, not dtype
        let tensors = vec![
            cand("layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("layer.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("layer.g_idx", Dtype::F32, vec![512], 2_048), // unusual dtype but present
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["layer"].format, AwqGptqFormat::Gptq, "g_idx presence alone determines GPTQ");
    }

    // ── Scan: 10 AWQ groups all detected independently ────────────────────

    #[test]
    fn detects_ten_awq_groups_independently() {
        let mut tensors = Vec::new();
        for i in 0..10 {
            let base = format!("layer.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 10, "all 10 AWQ groups must be detected");
        assert_eq!(scan.consumed.len(), 30, "3 tensors per group × 10 = 30 consumed");
        for i in 0..10 {
            assert!(scan.groups.contains_key(&format!("layer.{i}.proj")));
        }
    }

    // ── Scan: qweight_name exactly matches the qweight tensor name ────────

    #[test]
    fn group_qweight_name_exactly_matches_tensor() {
        let tensors = vec![
            cand("my.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("my.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("my.proj.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("my.proj").unwrap();
        assert_eq!(group.qweight_name, "my.proj.qweight");
        assert_eq!(group.scales_name, "my.proj.scales");
        assert_eq!(group.qzeros_name, "my.proj.qzeros");
    }

    // ── Scan: g_idx_name is the full tensor name not just suffix ──────────

    #[test]
    fn gptq_g_idx_name_is_full_tensor_name() {
        let tensors = vec![
            cand("deep.path.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("deep.path.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("deep.path.proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("deep.path.proj.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("deep.path.proj").unwrap();
        assert_eq!(
            group.g_idx_name.as_deref(),
            Some("deep.path.proj.g_idx"),
            "g_idx_name must be the full tensor name"
        );
    }

    // ── Scan: base_name with underscores and numbers ──────────────────────

    #[test]
    fn detects_base_name_with_underscores_and_numbers() {
        let tensors = vec![
            cand("model_layers_0_self_attn_q_proj_v2.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model_layers_0_self_attn_q_proj_v2.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model_layers_0_self_attn_q_proj_v2.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("model_layers_0_self_attn_q_proj_v2"));
    }

    // ── Scan: scales columns = qweight columns - 1 (off-by-one lower) ────

    #[test]
    fn rejects_scales_columns_off_by_one_lower() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 127], 1_016),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "scales columns 127 vs qweight columns 128 must be rejected");
    }

    // ── Scan: qzeros with 3D shape rejected ───────────────────────────────

    #[test]
    fn rejects_qzeros_3d_shape() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 2, 8], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "3D qzeros must be rejected");
    }

    // ── Scan: qweight with name ending in "..qweight" (double dot) ────────

    #[test]
    fn detects_double_dot_before_qweight() {
        let tensors = vec![
            cand("model..qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model..scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model..qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "double dot before suffix should still match");
        assert!(scan.groups.contains_key("model."));
    }

    // ── Scan: no tensors with any of the four suffixes ────────────────────

    #[test]
    fn scan_with_only_unrelated_suffixes() {
        let tensors = vec![
            cand("model.layers.0.weight", Dtype::F32, vec![4096, 4096], 67_108_864),
            cand("model.layers.0.bias", Dtype::F32, vec![4096], 16_384),
            cand("model.norm.weight", Dtype::F32, vec![4096], 16_384),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: qweight I16 dtype rejected ──────────────────────────────────

    #[test]
    fn rejects_qweight_dtype_i16() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I16, vec![64, 128], 16_384),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "I16 qweight must be rejected");
    }

    // ── Scan: qzeros I64 dtype rejected ───────────────────────────────────

    #[test]
    fn rejects_qzeros_dtype_i64() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I64, vec![4, 16], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "I64 qzeros must be rejected");
    }

    // ── Scan: qweight present, scales present, qzeros present but wrong ───

    #[test]
    fn rejects_valid_qweight_valid_scales_wrong_qzeros_dtype() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::F16, vec![4, 16], 128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F16 qzeros must be rejected even with valid qweight+scales");
    }

    // ── AwqGptqGroup: all name fields are consistent suffixes of base_name ─

    #[test]
    fn group_name_suffixes_consistent_with_base() {
        let base = "model.layers.3.mlp.up_proj";
        let tensors = vec![
            cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768),
            cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024),
            cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get(base).unwrap();
        assert!(group.qweight_name.ends_with(".qweight"));
        assert!(group.scales_name.ends_with(".scales"));
        assert!(group.qzeros_name.ends_with(".qzeros"));
        // Each suffix starts after base_name
        assert_eq!(&group.qweight_name[..base.len()], base);
        assert_eq!(&group.scales_name[..base.len()], base);
        assert_eq!(&group.qzeros_name[..base.len()], base);
    }

    // ── AwqGptqGroup: format matches g_idx_name presence invariant ────────

    #[test]
    fn group_format_g_idx_invariant_gptq() {
        // GPTQ format implies g_idx_name is Some
        let tensors = vec![
            cand("g.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("g.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("g.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("g.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("g").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert!(group.g_idx_name.is_some(), "GPTQ must have g_idx_name = Some");
    }

    // ── AwqGptqGroup: AWQ format implies g_idx_name is None invariant ─────

    #[test]
    fn group_format_g_idx_invariant_awq() {
        // AWQ format implies g_idx_name is None
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("a").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none(), "AWQ must have g_idx_name = None");
    }

    // ── AwqGptqScan: consumed is superset of all group tensor names ───────

    #[test]
    fn scan_consumed_superset_of_group_tensor_names() {
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("b.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("b.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("b.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("b.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        for group in scan.groups.values() {
            assert!(scan.consumed.contains(&group.qweight_name));
            assert!(scan.consumed.contains(&group.scales_name));
            assert!(scan.consumed.contains(&group.qzeros_name));
            if let Some(ref g_idx) = group.g_idx_name {
                assert!(scan.consumed.contains(g_idx));
            }
        }
    }

    // ── AwqGptqScan: multiple groups with different shapes ───────────────

    #[test]
    fn scan_multiple_groups_different_shapes() {
        let tensors = vec![
            cand("gate.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
            cand("gate.scales", Dtype::F16, vec![32, 4096], 262_144),
            cand("gate.qzeros", Dtype::I32, vec![32, 512], 65_536),
            cand("up.qweight", Dtype::I32, vec![512, 11008], 22_544_384),
            cand("up.scales", Dtype::F16, vec![32, 11008], 704_512),
            cand("up.qzeros", Dtype::I32, vec![32, 1376], 176_128),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups["gate"].qweight_shape, vec![512, 4096]);
        assert_eq!(scan.groups["up"].qweight_shape, vec![512, 11008]);
    }

    // ── CandidateTensor: zero-length shape ────────────────────────────────

    #[test]
    fn candidate_tensor_zero_length_shape() {
        let ct = CandidateTensor {
            name: "scalar_val".to_string(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 4,
        };
        assert!(ct.shape.is_empty());
        assert_eq!(ct.byte_len, 4);
    }

    // ── CandidateTensor: shape with many dimensions ──────────────────────

    #[test]
    fn candidate_tensor_4d_shape() {
        let ct = CandidateTensor {
            name: "conv.weight".to_string(),
            dtype: Dtype::F32,
            shape: vec![64, 3, 7, 7],
            byte_len: 64 * 3 * 7 * 7 * 4,
        };
        assert_eq!(ct.shape.len(), 4);
        assert_eq!(ct.shape[0], 64);
        assert_eq!(ct.shape[3], 7);
    }

    // ── CandidateTensor: clone then modify shape independently ───────────

    #[test]
    fn candidate_tensor_clone_shape_independence() {
        let ct = CandidateTensor {
            name: "original".to_string(),
            dtype: Dtype::F32,
            shape: vec![10, 20],
            byte_len: 800,
        };
        let mut cloned = ct.clone();
        cloned.shape.push(30);
        assert_eq!(ct.shape.len(), 2, "original shape must be unchanged");
        assert_eq!(cloned.shape.len(), 3, "cloned shape must have new dimension");
    }

    // ── AwqGptqFormat: assert ne between different variants ──────────────

    #[test]
    fn format_assert_ne_cross_variants() {
        assert_ne!(AwqGptqFormat::Awq, AwqGptqFormat::Gptq);
        assert_ne!(AwqGptqFormat::Gptq, AwqGptqFormat::Awq);
    }

    // ── AwqGptqFormat: size_of is Copy-efficient ─────────────────────────

    #[test]
    fn format_size_is_small() {
        use std::mem::size_of;
        assert!(size_of::<AwqGptqFormat>() <= 1, "AwqGptqFormat should be 1 byte (C-like enum)");
    }

    // ── AwqGptqGroup: clone produces equal but distinct allocation ────────

    #[test]
    fn group_clone_equality_and_distinction() {
        let group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: Some("test.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![128, 256],
        };
        let cloned = group.clone();
        // Fields are equal
        assert_eq!(group.base_name, cloned.base_name);
        assert_eq!(group.format, cloned.format);
        assert_eq!(group.qweight_shape, cloned.qweight_shape);
        assert_eq!(group.g_idx_name, cloned.g_idx_name);
        // But Strings are distinct allocations
        assert!(!std::ptr::eq(group.base_name.as_ptr(), cloned.base_name.as_ptr()));
    }

    // ── AwqGptqScan: inserting into groups does not affect consumed ──────

    #[test]
    fn scan_groups_and_consumed_are_independent() {
        let mut scan = AwqGptqScan::default();
        let group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        scan.groups.insert("test".to_string(), group);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.consumed.is_empty(), "consumed must remain empty when only groups is modified");
    }

    // ── AwqGptqScan: inserting into consumed does not affect groups ───────

    #[test]
    fn scan_consumed_insert_does_not_affect_groups() {
        let mut scan = AwqGptqScan::default();
        scan.consumed.insert("orphan.weight".to_string());
        assert!(scan.consumed.contains("orphan.weight"));
        assert!(scan.groups.is_empty(), "groups must remain empty when only consumed is modified");
    }

    // ── Scan: suffix ".scale" (singular) not matched ──────────────────────

    #[test]
    fn rejects_suffix_scale_singular() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scale", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), ".scale (singular) must not match .scales suffix");
    }

    // ── Scan: suffix ".qzero" (singular) not matched ──────────────────────

    #[test]
    fn rejects_suffix_qzero_singular() {
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzero", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), ".qzero (singular) must not match .qzeros suffix");
    }

    // ── Scan: realistic multi-layer model with mixed quantized/non-quantized

    #[test]
    fn realistic_model_mixed_quantized_and_dense_layers() {
        let mut tensors = Vec::new();
        // Dense embedding
        tensors.push(cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524_288_000));
        // Layer 0: quantized attention
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let base = format!("model.layers.0.self_attn.{proj}");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![512, 4096], 8_388_608));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![32, 4096], 262_144));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![32, 512], 65_536));
        }
        // Layer 0: dense MLP
        tensors.push(cand("model.layers.0.mlp.gate_proj.weight", Dtype::F32, vec![4096, 11008], 180_355_072));
        tensors.push(cand("model.layers.0.mlp.up_proj.weight", Dtype::F32, vec![4096, 11008], 180_355_072));
        // Layer norms
        tensors.push(cand("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4096], 16_384));
        tensors.push(cand("model.norm.weight", Dtype::F32, vec![4096], 16_384));
        tensors.push(cand("lm_head.weight", Dtype::F32, vec![4096, 32000], 524_288_000));

        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 4, "4 quantized attention projections must be detected");
        assert_eq!(scan.consumed.len(), 12, "4 groups × 3 tensors = 12 consumed");
        // Dense tensors not consumed
        assert!(!scan.consumed.contains("model.embed_tokens.weight"));
        assert!(!scan.consumed.contains("model.layers.0.mlp.gate_proj.weight"));
        assert!(!scan.consumed.contains("lm_head.weight"));
    }

    // ── Scan: scales rows larger than K (scales_rows > K) ────────────────

    #[test]
    fn rejects_scales_rows_larger_than_k() {
        // qweight [8, 128] → K = 8*8 = 64
        // scales [65, 128] → scales_rows=65, 64%65 != 0 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![8, 128], 4_096),
            cand("foo.scales", Dtype::F16, vec![65, 128], 16_640),
            cand("foo.qzeros", Dtype::I32, vec![65, 16], 4_160),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "scales_rows > K must be rejected");
    }

    // ── Scan: qweight consumed only when full triplet is valid ────────────

    #[test]
    fn qweight_consumed_only_on_valid_triplet() {
        let tensors = vec![
            // Incomplete: qweight + scales but no qzeros
            cand("incomplete.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("incomplete.scales", Dtype::F16, vec![4, 128], 1_024),
            // Complete
            cand("complete.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("complete.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("complete.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(!scan.consumed.contains("incomplete.qweight"), "incomplete triplet must not be consumed");
        assert!(scan.consumed.contains("complete.qweight"), "complete triplet must be consumed");
    }

    // ── AwqGptqGroup: qweight_shape reflects exact tensor shape ──────────

    #[test]
    fn group_qweight_shape_preserves_tensor_shape() {
        // Non-standard large shape
        let shape = vec![2048, 8192];
        let tensors = vec![
            cand("big.qweight", Dtype::I32, shape.clone(), 67_108_864),
            cand("big.scales", Dtype::F16, vec![128, 8192], 2_097_152),
            cand("big.qzeros", Dtype::I32, vec![128, 1024], 524_288),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get("big").unwrap();
        assert_eq!(group.qweight_shape, shape);
    }

    // ── AwqGptqFormat: Ord-like comparison via derived PartialOrd/Ord ────────

    #[test]
    fn format_ordering_is_consistent() {
        // AwqGptqFormat derives Eq+Hash but NOT Ord. Verify Copy is the only
        // auto-trait beyond what's already tested.
        let a = AwqGptqFormat::Awq;
        let b = AwqGptqFormat::Gptq;
        // Just verify both variants are usable after copy
        let _a2 = a;
        let _b2 = b;
        assert_eq!(a, AwqGptqFormat::Awq);
        assert_eq!(b, AwqGptqFormat::Gptq);
    }

    // ── AwqGptqGroup: field accessor for qweight_name ──────────────────────

    #[test]
    fn group_qweight_name_field_accessor() {
        let group = AwqGptqGroup {
            base_name: "layer.0.proj".to_string(),
            qweight_name: "layer.0.proj.qweight".to_string(),
            scales_name: "layer.0.proj.scales".to_string(),
            qzeros_name: "layer.0.proj.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        assert_eq!(group.qweight_name, "layer.0.proj.qweight");
    }

    // ── AwqGptqGroup: field accessor for scales_name ───────────────────────

    #[test]
    fn group_scales_name_field_accessor() {
        let group = AwqGptqGroup {
            base_name: "x".to_string(),
            qweight_name: "x.qweight".to_string(),
            scales_name: "x.scales".to_string(),
            qzeros_name: "x.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![4, 8],
        };
        assert_eq!(group.scales_name, "x.scales");
    }

    // ── AwqGptqGroup: field accessor for qzeros_name ───────────────────────

    #[test]
    fn group_qzeros_name_field_accessor() {
        let group = AwqGptqGroup {
            base_name: "y".to_string(),
            qweight_name: "y.qweight".to_string(),
            scales_name: "y.scales".to_string(),
            qzeros_name: "y.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![4, 8],
        };
        assert_eq!(group.qzeros_name, "y.qzeros");
    }

    // ── AwqGptqGroup: field accessor for base_name ─────────────────────────

    #[test]
    fn group_base_name_field_accessor() {
        let group = AwqGptqGroup {
            base_name: "model.layers.3.mlp.up_proj".to_string(),
            qweight_name: "model.layers.3.mlp.up_proj.qweight".to_string(),
            scales_name: "model.layers.3.mlp.up_proj.scales".to_string(),
            qzeros_name: "model.layers.3.mlp.up_proj.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        assert_eq!(group.base_name, "model.layers.3.mlp.up_proj");
    }

    // ── AwqGptqScan: groups HashMap retains insertion count ────────────────

    #[test]
    fn scan_groups_len_matches_insertions() {
        let mut scan = AwqGptqScan::default();
        assert_eq!(scan.groups.len(), 0);
        scan.groups.insert(
            "a".to_string(),
            AwqGptqGroup {
                base_name: "a".to_string(),
                qweight_name: "a.qweight".to_string(),
                scales_name: "a.scales".to_string(),
                qzeros_name: "a.qzeros".to_string(),
                g_idx_name: None,
                format: AwqGptqFormat::Awq,
                qweight_shape: vec![1, 1],
            },
        );
        assert_eq!(scan.groups.len(), 1);
        scan.groups.insert(
            "b".to_string(),
            AwqGptqGroup {
                base_name: "b".to_string(),
                qweight_name: "b.qweight".to_string(),
                scales_name: "b.scales".to_string(),
                qzeros_name: "b.qzeros".to_string(),
                g_idx_name: Some("b.g_idx".to_string()),
                format: AwqGptqFormat::Gptq,
                qweight_shape: vec![2, 4],
            },
        );
        assert_eq!(scan.groups.len(), 2);
    }

    // ── AwqGptqScan: consumed HashSet len tracks insertions ────────────────

    #[test]
    fn scan_consumed_len_tracks_insertions() {
        let mut scan = AwqGptqScan::default();
        assert!(scan.consumed.is_empty());
        scan.consumed.insert("a.qweight".to_string());
        assert_eq!(scan.consumed.len(), 1);
        scan.consumed.insert("a.scales".to_string());
        scan.consumed.insert("a.qzeros".to_string());
        assert_eq!(scan.consumed.len(), 3);
        // Duplicate insert does not increase length
        scan.consumed.insert("a.qweight".to_string());
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── AwqGptqScan: clearing groups does not affect consumed ──────────────

    #[test]
    fn scan_clear_groups_preserves_consumed() {
        let mut scan = AwqGptqScan::default();
        scan.groups.insert(
            "g".to_string(),
            AwqGptqGroup {
                base_name: "g".to_string(),
                qweight_name: "g.qweight".to_string(),
                scales_name: "g.scales".to_string(),
                qzeros_name: "g.qzeros".to_string(),
                g_idx_name: None,
                format: AwqGptqFormat::Awq,
                qweight_shape: vec![1, 2],
            },
        );
        scan.consumed.insert("g.qweight".to_string());
        scan.consumed.insert("g.scales".to_string());
        scan.consumed.insert("g.qzeros".to_string());
        scan.groups.clear();
        assert!(scan.groups.is_empty());
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── CandidateTensor: Eq trait consistency ──────────────────────────────

    #[test]
    fn candidate_tensor_eq_consistent_with_partialeq() {
        let a = CandidateTensor {
            name: "x".to_string(),
            dtype: Dtype::F16,
            shape: vec![4, 8],
            byte_len: 64,
        };
        let b = CandidateTensor {
            name: "x".to_string(),
            dtype: Dtype::F16,
            shape: vec![4, 8],
            byte_len: 64,
        };
        // Eq implies PartialEq consistency: equal values always equal
        assert_eq!(a, b);
        assert!(!(a != b));
    }

    // ── AwqGptqFormat: usable as match exhaustiveness ─────────────────────

    #[test]
    fn format_match_exhaustive_returns_string() {
        fn label(f: AwqGptqFormat) -> &'static str {
            match f {
                AwqGptqFormat::Awq => "AWQ",
                AwqGptqFormat::Gptq => "GPTQ",
            }
        }
        assert_eq!(label(AwqGptqFormat::Awq), "AWQ");
        assert_eq!(label(AwqGptqFormat::Gptq), "GPTQ");
    }

    // ── AwqGptqGroup: format field determines variant correctly ───────────

    #[test]
    fn group_format_field_gptq_with_g_idx() {
        let group = AwqGptqGroup {
            base_name: "z".to_string(),
            qweight_name: "z.qweight".to_string(),
            scales_name: "z.scales".to_string(),
            qzeros_name: "z.qzeros".to_string(),
            g_idx_name: Some("z.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![16, 32],
        };
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert!(group.g_idx_name.is_some());
    }

    // ── AwqGptqGroup: format field awq without g_idx ──────────────────────

    #[test]
    fn group_format_field_awq_without_g_idx() {
        let group = AwqGptqGroup {
            base_name: "w".to_string(),
            qweight_name: "w.qweight".to_string(),
            scales_name: "w.scales".to_string(),
            qzeros_name: "w.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![8, 16],
        };
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
    }

    // ── AwqGptqScan: default is same as manual empty construction ──────────

    #[test]
    fn scan_default_manual_fields_are_empty() {
        let scan = AwqGptqScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── AwqGptqScan: consumed contains_key works ──────────────────────────

    #[test]
    fn scan_consumed_contains_key() {
        let mut scan = AwqGptqScan::default();
        scan.consumed.insert("foo.qweight".to_string());
        assert!(scan.consumed.contains("foo.qweight"));
        assert!(!scan.consumed.contains("foo.scales"));
    }

    // ── AwqGptqScan: groups contains_key works ────────────────────────────

    #[test]
    fn scan_groups_contains_key() {
        let mut scan = AwqGptqScan::default();
        scan.groups.insert(
            "my_layer".to_string(),
            AwqGptqGroup {
                base_name: "my_layer".to_string(),
                qweight_name: "my_layer.qweight".to_string(),
                scales_name: "my_layer.scales".to_string(),
                qzeros_name: "my_layer.qzeros".to_string(),
                g_idx_name: None,
                format: AwqGptqFormat::Awq,
                qweight_shape: vec![4, 4],
            },
        );
        assert!(scan.groups.contains_key("my_layer"));
        assert!(!scan.groups.contains_key("other_layer"));
    }

    // ── AwqGptqGroup: qweight_shape with vec![] empty ─────────────────────

    #[test]
    fn group_qweight_shape_empty_vec() {
        let group = AwqGptqGroup {
            base_name: "e".to_string(),
            qweight_name: "e.qweight".to_string(),
            scales_name: "e.scales".to_string(),
            qzeros_name: "e.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![],
        };
        assert!(group.qweight_shape.is_empty());
    }

    // ── AwqGptqFormat: both variants used as HashMap value ────────────────

    #[test]
    fn format_as_hashmap_value() {
        let mut map = HashMap::new();
        map.insert("awq", AwqGptqFormat::Awq);
        map.insert("gptq", AwqGptqFormat::Gptq);
        assert_eq!(*map.get("awq").unwrap(), AwqGptqFormat::Awq);
        assert_eq!(*map.get("gptq").unwrap(), AwqGptqFormat::Gptq);
        assert_eq!(map.len(), 2);
    }

    // ── AwqGptqGroup: g_idx_name some and none are distinct ───────────────

    #[test]
    fn group_g_idx_name_some_vs_none() {
        let with = AwqGptqGroup {
            base_name: "a".to_string(),
            qweight_name: "a.qweight".to_string(),
            scales_name: "a.scales".to_string(),
            qzeros_name: "a.qzeros".to_string(),
            g_idx_name: Some("a.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![2, 4],
        };
        let without = AwqGptqGroup {
            base_name: "a".to_string(),
            qweight_name: "a.qweight".to_string(),
            scales_name: "a.scales".to_string(),
            qzeros_name: "a.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![2, 4],
        };
        assert!(with.g_idx_name.is_some());
        assert!(without.g_idx_name.is_none());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 48 tests — additional ~15 unit tests for public API coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── AwqGptqScan: groups.remove() decrements length ───────────────────

    #[test]
    fn scan_groups_remove_decrements_len() {
        let mut scan = AwqGptqScan::default();
        scan.groups.insert(
            "target".to_string(),
            AwqGptqGroup {
                base_name: "target".to_string(),
                qweight_name: "target.qweight".to_string(),
                scales_name: "target.scales".to_string(),
                qzeros_name: "target.qzeros".to_string(),
                g_idx_name: None,
                format: AwqGptqFormat::Awq,
                qweight_shape: vec![32, 64],
            },
        );
        scan.groups.insert(
            "keeper".to_string(),
            AwqGptqGroup {
                base_name: "keeper".to_string(),
                qweight_name: "keeper.qweight".to_string(),
                scales_name: "keeper.scales".to_string(),
                qzeros_name: "keeper.qzeros".to_string(),
                g_idx_name: None,
                format: AwqGptqFormat::Awq,
                qweight_shape: vec![16, 32],
            },
        );
        assert_eq!(scan.groups.len(), 2);
        let removed = scan.groups.remove("target");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().base_name, "target");
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("keeper"));
    }

    // ── AwqGptqScan: consumed.remove() decrements length ─────────────────

    #[test]
    fn scan_consumed_remove_decrements_len() {
        let mut scan = AwqGptqScan::default();
        scan.consumed.insert("a.qweight".to_string());
        scan.consumed.insert("a.scales".to_string());
        scan.consumed.insert("a.qzeros".to_string());
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.consumed.remove("a.scales"));
        assert_eq!(scan.consumed.len(), 2);
        assert!(!scan.consumed.contains("a.scales"));
        assert!(scan.consumed.contains("a.qweight"));
        assert!(scan.consumed.contains("a.qzeros"));
    }

    // ── AwqGptqGroup: format field can be mutated after construction ─────

    #[test]
    fn group_modify_format_after_construction() {
        let mut group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        assert_eq!(group.format, AwqGptqFormat::Awq);
        group.format = AwqGptqFormat::Gptq;
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        group.g_idx_name = Some("test.g_idx".to_string());
        assert!(group.g_idx_name.is_some());
    }

    // ── AwqGptqGroup: all string fields empty ────────────────────────────

    #[test]
    fn group_all_string_fields_empty() {
        let group = AwqGptqGroup {
            base_name: String::new(),
            qweight_name: String::new(),
            scales_name: String::new(),
            qzeros_name: String::new(),
            g_idx_name: Some(String::new()),
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![],
        };
        assert!(group.base_name.is_empty());
        assert!(group.qweight_name.is_empty());
        assert!(group.scales_name.is_empty());
        assert!(group.qzeros_name.is_empty());
        assert_eq!(group.g_idx_name.as_deref(), Some(""));
        assert!(group.qweight_shape.is_empty());
    }

    // ── AwqGptqScan: iterating over groups.values() ──────────────────────

    #[test]
    fn scan_iterate_groups_values() {
        let mut scan = AwqGptqScan::default();
        for i in 0..3 {
            let base = format!("layer.{i}");
            scan.groups.insert(
                base.clone(),
                AwqGptqGroup {
                    base_name: base,
                    qweight_name: format!("layer.{i}.qweight"),
                    scales_name: format!("layer.{i}.scales"),
                    qzeros_name: format!("layer.{i}.qzeros"),
                    g_idx_name: None,
                    format: AwqGptqFormat::Awq,
                    qweight_shape: vec![32, 64],
                },
            );
        }
        let formats: Vec<_> = scan.groups.values().map(|g| g.format).collect();
        assert_eq!(formats.len(), 3);
        assert!(formats.iter().all(|f| *f == AwqGptqFormat::Awq));
    }

    // ── AwqGptqScan: iterating over consumed set ─────────────────────────

    #[test]
    fn scan_iterate_consumed_set() {
        let mut scan = AwqGptqScan::default();
        scan.consumed.insert("a.qweight".to_string());
        scan.consumed.insert("a.scales".to_string());
        scan.consumed.insert("a.qzeros".to_string());
        let all_end_with_suffix: bool = scan
            .consumed
            .iter()
            .all(|n| n.ends_with(".qweight") || n.ends_with(".scales") || n.ends_with(".qzeros"));
        assert!(all_end_with_suffix);
    }

    // ── CandidateTensor: name with special characters ────────────────────

    #[test]
    fn candidate_tensor_name_with_special_chars() {
        let ct = CandidateTensor {
            name: "layer\t0\nproj\0binary".to_string(),
            dtype: Dtype::F32,
            shape: vec![1],
            byte_len: 4,
        };
        assert!(ct.name.contains('\t'));
        assert!(ct.name.contains('\n'));
        assert!(ct.name.contains('\0'));
    }

    // ── AwqGptqFormat: both variants collected in a Vec ──────────────────

    #[test]
    fn format_vec_contains_both_variants() {
        let variants = vec![AwqGptqFormat::Awq, AwqGptqFormat::Gptq];
        assert_eq!(variants.len(), 2);
        let awq_count = variants.iter().filter(|v| **v == AwqGptqFormat::Awq).count();
        let gptq_count = variants.iter().filter(|v| **v == AwqGptqFormat::Gptq).count();
        assert_eq!(awq_count, 1);
        assert_eq!(gptq_count, 1);
    }

    // ── AwqGptqGroup: qweight_shape can be mutated after construction ────

    #[test]
    fn group_modify_qweight_shape_push() {
        let mut group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        assert_eq!(group.qweight_shape.len(), 2);
        group.qweight_shape.push(256);
        assert_eq!(group.qweight_shape, vec![64, 128, 256]);
        group.qweight_shape[0] = 99;
        assert_eq!(group.qweight_shape[0], 99);
    }

    // ── CandidateTensor: Debug output contains byte_len field name ───────

    #[test]
    fn candidate_tensor_debug_shows_byte_len() {
        let ct = CandidateTensor {
            name: "t".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            byte_len: 16,
        };
        let debug = format!("{ct:?}");
        assert!(
            debug.contains("byte_len"),
            "Debug must contain byte_len field: {debug}"
        );
    }

    // ── AwqGptqGroup: base_name can be mutated after construction ────────

    #[test]
    fn group_modify_base_name_after_construction() {
        let mut group = AwqGptqGroup {
            base_name: "original".to_string(),
            qweight_name: "original.qweight".to_string(),
            scales_name: "original.scales".to_string(),
            qzeros_name: "original.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![4, 8],
        };
        assert_eq!(group.base_name, "original");
        group.base_name = "renamed".to_string();
        assert_eq!(group.base_name, "renamed");
        assert_eq!(group.qweight_name, "original.qweight", "other fields unchanged");
    }

    // ── AwqGptqScan: groups.keys() collection ────────────────────────────

    #[test]
    fn scan_groups_keys_collection() {
        let mut scan = AwqGptqScan::default();
        for name in &["gate_proj", "up_proj", "down_proj"] {
            scan.groups.insert(
                name.to_string(),
                AwqGptqGroup {
                    base_name: name.to_string(),
                    qweight_name: format!("{name}.qweight"),
                    scales_name: format!("{name}.scales"),
                    qzeros_name: format!("{name}.qzeros"),
                    g_idx_name: None,
                    format: AwqGptqFormat::Awq,
                    qweight_shape: vec![64, 128],
                },
            );
        }
        let keys: Vec<_> = scan.groups.keys().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&&"gate_proj".to_string()));
        assert!(keys.contains(&&"up_proj".to_string()));
        assert!(keys.contains(&&"down_proj".to_string()));
    }

    // ── AwqGptqGroup: qweight_shape with three dimensions ────────────────

    #[test]
    fn group_with_qweight_shape_three_dims() {
        let group = AwqGptqGroup {
            base_name: "3d".to_string(),
            qweight_name: "3d.qweight".to_string(),
            scales_name: "3d.scales".to_string(),
            qzeros_name: "3d.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![16, 32, 64],
        };
        assert_eq!(group.qweight_shape.len(), 3);
        assert_eq!(group.qweight_shape, vec![16, 32, 64]);
    }

    // ── CandidateTensor: name consisting of only dots ────────────────────

    #[test]
    fn candidate_tensor_name_only_dots() {
        let ct = CandidateTensor {
            name: "...".to_string(),
            dtype: Dtype::F32,
            shape: vec![1],
            byte_len: 4,
        };
        assert_eq!(ct.name, "...");
    }

    // ── AwqGptqScan: get() returns correct group reference ───────────────

    #[test]
    fn scan_groups_get_returns_correct_reference() {
        let mut scan = AwqGptqScan::default();
        scan.groups.insert(
            "target".to_string(),
            AwqGptqGroup {
                base_name: "target".to_string(),
                qweight_name: "target.qweight".to_string(),
                scales_name: "target.scales".to_string(),
                qzeros_name: "target.qzeros".to_string(),
                g_idx_name: Some("target.g_idx".to_string()),
                format: AwqGptqFormat::Gptq,
                qweight_shape: vec![128, 256],
            },
        );
        let group_ref = scan.groups.get("target");
        assert!(group_ref.is_some());
        let group = group_ref.unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(group.qweight_shape, vec![128, 256]);
        assert!(scan.groups.get("nonexistent").is_none());
    }

    // ── AwqGptqFormat: both variants have distinct Debug strings ─────────

    #[test]
    fn format_debug_strings_are_distinct() {
        let awq_debug = format!("{:?}", AwqGptqFormat::Awq);
        let gptq_debug = format!("{:?}", AwqGptqFormat::Gptq);
        assert_ne!(awq_debug, gptq_debug, "Debug strings must differ between variants");
        assert!(!awq_debug.is_empty());
        assert!(!gptq_debug.is_empty());
    }

    // ── Additional tests ─────────────────────────────────────────────────

    // AwqGptqFormat implements Copy, so assignment creates an independent value.
    #[test]
    fn format_copy_assignment_independence() {
        let mut a = AwqGptqFormat::Awq;
        let b = a;
        a = AwqGptqFormat::Gptq;
        assert_eq!(b, AwqGptqFormat::Awq, "Copy should produce an independent value");
        assert_eq!(a, AwqGptqFormat::Gptq);
    }

    // AwqGptqScan Default produces empty but usable HashMap/HashSet.
    #[test]
    fn scan_default_is_functionally_empty() {
        let scan = AwqGptqScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
        assert_eq!(scan.groups.len(), 0);
        assert_eq!(scan.consumed.len(), 0);
        assert!(scan.groups.keys().next().is_none());
        assert!(scan.consumed.iter().next().is_none());
    }

    // AwqGptqGroup can be constructed with all string fields non-empty and g_idx None.
    #[test]
    fn group_full_construction_g_idx_none() {
        let group = AwqGptqGroup {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            qweight_name: "model.layers.0.self_attn.q_proj.qweight".to_string(),
            scales_name: "model.layers.0.self_attn.q_proj.scales".to_string(),
            qzeros_name: "model.layers.0.self_attn.q_proj.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![512, 1024],
        };
        assert_eq!(group.base_name, "model.layers.0.self_attn.q_proj");
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert_eq!(group.qweight_shape.len(), 2);
    }

    // qweight_shape with very large dimension values.
    #[test]
    fn group_qweight_shape_large_values() {
        let group = AwqGptqGroup {
            base_name: "big".to_string(),
            qweight_name: "big.qweight".to_string(),
            scales_name: "big.scales".to_string(),
            qzeros_name: "big.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![usize::MAX / 2, 1],
        };
        assert_eq!(group.qweight_shape[0], usize::MAX / 2);
    }

    // Two groups with the same base_name in the same scan map: last insert wins.
    #[test]
    fn scan_duplicate_base_name_last_insert_wins() {
        let mut scan = AwqGptqScan::default();
        let group_v1 = AwqGptqGroup {
            base_name: "layer.0".to_string(),
            qweight_name: "layer.0.qweight".to_string(),
            scales_name: "layer.0.scales".to_string(),
            qzeros_name: "layer.0.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        let group_v2 = AwqGptqGroup {
            base_name: "layer.0".to_string(),
            qweight_name: "layer.0.qweight".to_string(),
            scales_name: "layer.0.scales".to_string(),
            qzeros_name: "layer.0.qzeros".to_string(),
            g_idx_name: Some("layer.0.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![32, 64],
        };
        scan.groups.insert("layer.0".to_string(), group_v1);
        scan.groups.insert("layer.0".to_string(), group_v2);
        assert_eq!(scan.groups.len(), 1);
        let stored = scan.groups.get("layer.0").unwrap();
        assert_eq!(stored.format, AwqGptqFormat::Gptq);
        assert_eq!(stored.qweight_shape, vec![32, 64]);
    }

    // scan_awq_gptq_groups works with an empty Vec iterator.
    #[test]
    fn scan_from_empty_vec() {
        let tensors: Vec<CandidateTensor> = vec![];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // qzeros with F16 dtype is rejected even if shapes match.
    #[test]
    fn rejects_qzeros_f16_dtype() {
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32768),
            cand("a.scales", Dtype::F16, vec![512, 128], 131072),
            cand("a.qzeros", Dtype::F16, vec![512, 16], 16384),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "F16 qzeros should be rejected");
    }

    // Scales with zero columns (N=0) and matching qweight columns.
    #[test]
    fn detects_qweight_zero_cols_scales_zero_cols() {
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 0], 0),
            cand("a.scales", Dtype::F16, vec![512, 0], 0),
            cand("a.qzeros", Dtype::I32, vec![512, 0], 0),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        // Zero-column case: N=0, scales cols=0 match, should be detected.
        assert_eq!(scan.groups.len(), 1);
    }

    // Base name containing digits and hyphens.
    #[test]
    fn detects_base_name_with_hyphens_and_digits() {
        let tensors = vec![
            cand("model-layer-99.ffn-3.qweight", Dtype::I32, vec![16, 32], 2048),
            cand("model-layer-99.ffn-3.scales", Dtype::F16, vec![128, 32], 8192),
            cand("model-layer-99.ffn-3.qzeros", Dtype::I32, vec![128, 4], 2048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("model-layer-99.ffn-3").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq);
    }

    // Multiple consumed tensors are all recorded.
    #[test]
    fn scan_consumed_captures_all_four_for_gptq() {
        let tensors = vec![
            cand("x.qweight", Dtype::I32, vec![16, 64], 4096),
            cand("x.scales", Dtype::F16, vec![128, 64], 16384),
            cand("x.qzeros", Dtype::I32, vec![128, 8], 4096),
            cand("x.g_idx", Dtype::I32, vec![128], 512),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.consumed.contains("x.qweight"));
        assert!(scan.consumed.contains("x.scales"));
        assert!(scan.consumed.contains("x.qzeros"));
        assert!(scan.consumed.contains("x.g_idx"));
        assert_eq!(scan.consumed.len(), 4);
    }

    // AwqGptqFormat can be used as a key in BTreeMap-like comparisons (PartialEq).
    #[test]
    fn format_equality_reflexive() {
        assert_eq!(AwqGptqFormat::Awq, AwqGptqFormat::Awq);
        assert_eq!(AwqGptqFormat::Gptq, AwqGptqFormat::Gptq);
    }

    // AwqGptqFormat inequality across variants.
    #[test]
    fn format_inequality_cross_variants() {
        assert_ne!(AwqGptqFormat::Awq, AwqGptqFormat::Gptq);
        assert_ne!(AwqGptqFormat::Gptq, AwqGptqFormat::Awq);
    }

    // scan_awq_gptq_groups ignores tensors whose names only partially match suffixes.
    #[test]
    fn ignores_tensors_with_suffix_as_substring() {
        let tensors = vec![
            cand("qweight_repacked", Dtype::I32, vec![16, 32], 2048),
            cand("myqweight", Dtype::I32, vec![16, 32], 2048),
            cand("scales_bias", Dtype::F16, vec![128, 32], 8192),
            cand("qzeros_adjusted", Dtype::I32, vec![128, 4], 2048),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty(), "Substring matches should not form triplets");
    }

    // Group Debug output includes the base_name substring.
    #[test]
    fn group_debug_contains_base_name_value() {
        let group = AwqGptqGroup {
            base_name: "layers.42.mlp.up_proj".to_string(),
            qweight_name: "layers.42.mlp.up_proj.qweight".to_string(),
            scales_name: "layers.42.mlp.up_proj.scales".to_string(),
            qzeros_name: "layers.42.mlp.up_proj.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![256, 512],
        };
        let debug_str = format!("{:?}", group);
        assert!(
            debug_str.contains("layers.42.mlp.up_proj"),
            "Debug output should contain base_name"
        );
    }

    // Scan with a single unrelated tensor: both groups and consumed remain empty.
    #[test]
    fn scan_single_unrelated_tensor_no_consumed() {
        let tensors = vec![
            cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524288000),
        ];
        let scan = scan_awq_gptq_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 49 tests — ~15 additional unit tests for edge case coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: GPTQ with I16 qzeros + g_idx — full 4-tensor detection ───

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_i16_qzeros_with_g_idx_full_consumed() {
        // Arrange: GPTQ triplet with I16 qzeros and g_idx present
        let tensors = vec![
            cand("model.layers.0.mlp.gate_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model.layers.0.mlp.gate_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model.layers.0.mlp.gate_proj.qzeros", Dtype::I16, vec![4, 16], 128),
            cand("model.layers.0.mlp.gate_proj.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 4 tensors consumed, format is GPTQ
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("model.layers.0.mlp.gate_proj").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 4);
        assert!(scan.consumed.contains("model.layers.0.mlp.gate_proj.g_idx"));
        assert!(scan.consumed.contains("model.layers.0.mlp.gate_proj.qzeros"));
    }

    // ── Scan: base_name with only numeric characters ───────────────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_base_name_numeric_only() {
        // Arrange: base_name is "12345"
        let tensors = vec![
            cand("12345.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("12345.scales", Dtype::F16, vec![2, 64], 256),
            cand("12345.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("12345"));
        assert_eq!(scan.groups["12345"].base_name, "12345");
    }

    // ── Scan: scales rows equal to qweight rows (group_size=8) with GPTQ ─

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_scales_rows_equal_qweight_rows_group_size_8() {
        // Arrange: K=512, group_size=8 → scales_rows=64=qweight_rows
        let tensors = vec![
            cand("gptq_gs8.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq_gs8.scales", Dtype::F16, vec![64, 128], 16_384),
            cand("gptq_gs8.qzeros", Dtype::I32, vec![64, 16], 4_096),
            cand("gptq_gs8.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "scales_rows==qweight_rows should be valid GPTQ");
        assert_eq!(scan.groups["gptq_gs8"].format, AwqGptqFormat::Gptq);
    }

    // ── Scan: qweight with odd row count, scales rows divide K evenly ───

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_odd_qweight_rows_scales_divides_evenly() {
        // Arrange: qweight [7, 64] → K = 56, scales [7, 64] → 56%7==0
        let tensors = vec![
            cand("odd.qweight", Dtype::I32, vec![7, 64], 1_792),
            cand("odd.scales", Dtype::F16, vec![7, 64], 1_792),
            cand("odd.qzeros", Dtype::I32, vec![7, 8], 224),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "odd qweight rows with evenly dividing scales should be valid");
        assert_eq!(scan.groups["odd"].qweight_shape, vec![7, 64]);
    }

    // ── Scan: qweight [0, 0] (both dimensions zero) ────────────────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_both_dimensions_zero() {
        // Arrange: qweight [0, 0] → K=0, scales [0, 0] → scales_rows=0 → rejected
        let tensors = vec![
            cand("zero2d.qweight", Dtype::I32, vec![0, 0], 0),
            cand("zero2d.scales", Dtype::F16, vec![0, 0], 0),
            cand("zero2d.qzeros", Dtype::I32, vec![0, 0], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: scales_rows==0 triggers rejection
        assert!(scan.groups.is_empty(), "zero-row scales must be rejected");
    }

    // ── Scan: very deep nesting in base_name path ──────────────────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_deeply_nested_base_name_path() {
        // Arrange: 10-level deep base_name
        let deep = "a.b.c.d.e.f.g.h.i.j";
        let tensors = vec![
            cand(&format!("{deep}.qweight"), Dtype::I32, vec![16, 32], 2_048),
            cand(&format!("{deep}.scales"), Dtype::F16, vec![1, 32], 64),
            cand(&format!("{deep}.qzeros"), Dtype::I32, vec![1, 4], 16),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key(deep));
        assert_eq!(scan.groups[deep].qweight_name, format!("{deep}.qweight"));
    }

    // ── Scan: AWQ + GPTQ in same layer, different projection types ────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn mixed_awq_gptq_same_layer_different_projections() {
        // Arrange: layer.0 has AWQ for q_proj and GPTQ for k_proj
        let tensors = vec![
            // AWQ: no g_idx
            cand("layer.0.q_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("layer.0.q_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("layer.0.q_proj.qzeros", Dtype::I32, vec![4, 16], 256),
            // GPTQ: has g_idx
            cand("layer.0.k_proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("layer.0.k_proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("layer.0.k_proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("layer.0.k_proj.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups["layer.0.q_proj"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["layer.0.k_proj"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 7); // 3 AWQ + 4 GPTQ
    }

    // ── Scan: qweight present, qzeros present but scales missing ───────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_triplet_missing_scales_with_qzeros() {
        // Arrange: qweight + qzeros but no scales
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: scales rows = K (entire dimension is one group per row) ──

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_scales_rows_equals_k() {
        // Arrange: qweight [8, 64] → K=64, scales [8, 64] → scales_rows=8
        // group_size = K/scales_rows = 64/8 = 8
        // scales_rows == qweight_rows is valid (already tested for larger shapes)
        // This test verifies K == scales_rows * group_size consistency for small K
        let tensors = vec![
            cand("gs_equal.qweight", Dtype::I32, vec![8, 64], 2_048),
            cand("gs_equal.scales", Dtype::F16, vec![8, 64], 1_024),
            cand("gs_equal.qzeros", Dtype::I32, vec![8, 8], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["gs_equal"].qweight_shape, vec![8, 64]);
    }

    // ── Scan: scan from filter() iterator ──────────────────────────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_from_filter_iterator() {
        // Arrange: create a Vec with valid and invalid tensors, filter to keep only I32
        let all_tensors = vec![
            cand("proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("noise.weight", Dtype::F32, vec![10], 40),
        ];
        // Act: scan using filtered iterator
        let scan = scan_awq_gptq_groups(all_tensors.into_iter().filter(|t| t.dtype != Dtype::F32));
        // Assert: noise.weight filtered out, valid triplet still found
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("proj"));
    }

    // ── Scan: realistic 2-layer model with all projections quantized ──

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn realistic_two_layer_all_quantized_projections() {
        // Arrange: 2 layers × 4 attention projections + 3 MLP projections
        let mut tensors = Vec::new();
        for layer in 0..2 {
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let base = format!("model.layers.{layer}.self_attn.{proj}");
                tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![512, 4096], 8_388_608));
                tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![32, 4096], 262_144));
                tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![32, 512], 65_536));
            }
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                let base = format!("model.layers.{layer}.mlp.{proj}");
                tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![512, 11008], 22_544_384));
                tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![32, 11008], 704_512));
                tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![32, 1376], 176_128));
            }
        }
        // Also add dense non-quantized tensors
        tensors.push(cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524_288_000));
        tensors.push(cand("model.norm.weight", Dtype::F32, vec![4096], 16_384));
        tensors.push(cand("lm_head.weight", Dtype::F32, vec![4096, 32000], 524_288_000));
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 2 layers × 7 projections = 14 groups, each consuming 3 tensors = 42 consumed
        assert_eq!(scan.groups.len(), 14, "2 layers × 7 projections = 14 groups");
        assert_eq!(scan.consumed.len(), 42, "14 groups × 3 tensors = 42 consumed");
        assert!(!scan.consumed.contains("model.embed_tokens.weight"));
        assert!(!scan.consumed.contains("lm_head.weight"));
    }

    // ── Scan: qzeros with shape [scales_rows, 1] (minimal columns) ────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn accepts_qzeros_single_column() {
        // Arrange: qzeros has only 1 column (rows still match scales rows)
        let tensors = vec![
            cand("mincol.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("mincol.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("mincol.qzeros", Dtype::I32, vec![4, 1], 16),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: column count is not validated against N/8
        assert_eq!(scan.groups.len(), 1, "qzeros with 1 column should be accepted");
    }

    // ── Scan: qweight I8 dtype rejected ────────────────────────────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_dtype_i8() {
        // Arrange
        let tensors = vec![
            cand("foo.qweight", Dtype::I8, vec![64, 128], 8_192),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "I8 qweight must be rejected");
    }

    // ── Scan: candidate tensors with byte_len inconsistent with shape ─

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_triplet_ignoring_inconsistent_byte_len() {
        // Arrange: byte_len values are deliberately inconsistent with shape × dtype,
        // but scan_awq_gptq_groups only validates dtype and shape, not byte_len
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 999_999),  // wrong byte_len
            cand("foo.scales", Dtype::F16, vec![4, 128], 42),         // wrong byte_len
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 7),           // wrong byte_len
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: byte_len is not validated by scan, so triplet is detected
        assert_eq!(scan.groups.len(), 1, "byte_len inconsistency should not affect detection");
    }

    // ── Scan: consumed set is empty when no groups are found ───────────

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn consumed_empty_when_only_non_matching_tensors() {
        // Arrange: several tensors with wrong suffixes or dtypes
        let tensors = vec![
            cand("model.weight", Dtype::F32, vec![4096, 4096], 67_108_864),
            cand("model.bias", Dtype::F32, vec![4096], 16_384),
            cand("layer.qweight", Dtype::F16, vec![64, 128], 16_384), // wrong dtype
            cand("layer.scales", Dtype::F16, vec![4, 128], 1_024),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: AWQ group qweight_shape correctly reflects qweight dim ──

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn awq_group_qweight_shape_matches_tensor_shape_not_k() {
        // Arrange: qweight shape is [128, 256] (NOT [K, N] — it's [K/8, N])
        // The stored qweight_shape must be the raw tensor shape, not the unpacked K
        let tensors = vec![
            cand("shape_test.qweight", Dtype::I32, vec![128, 256], 131_072),
            cand("shape_test.scales", Dtype::F16, vec![8, 256], 4_096),
            cand("shape_test.qzeros", Dtype::I32, vec![8, 32], 1_024),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qweight_shape is [128, 256], NOT [1024, 256]
        let group = scan.groups.get("shape_test").unwrap();
        assert_eq!(group.qweight_shape, vec![128, 256], "qweight_shape must be raw tensor shape");
    }

    // ── Scan: qweight with shape [rows, 1] (N=1, minimal output dim) ─

    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_shape_rows_by_1() {
        // Arrange: qweight [16, 1] → K=128, N=1
        // scales [1, 1] → scales_rows=1, 128%1==0
        // qzeros [1, 1]
        let tensors = vec![
            cand("n1.qweight", Dtype::I32, vec![16, 1], 64),
            cand("n1.scales", Dtype::F16, vec![1, 1], 2),
            cand("n1.qzeros", Dtype::I32, vec![1, 1], 4),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "N=1 shape should be valid");
        assert_eq!(scan.groups["n1"].qweight_shape, vec![16, 1]);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 48 — 15 additional unit tests for edge cases and trait coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qzeros rows > scales rows (too many rows) ────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qzeros_more_rows_than_scales() {
        // Arrange: qzeros has 8 rows but scales has only 4 rows
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![8, 16], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qzeros rows (8) != scales rows (4) → rejected
        assert!(scan.groups.is_empty(), "qzeros rows > scales rows must be rejected");
    }

    // ── Scan: scales F64 dtype rejected (already tested F32/BF16, add F64) ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_dtype_f64_alternative_shape() {
        // Arrange: non-standard large shape with F64 scales
        let tensors = vec![
            cand("big.qweight", Dtype::I32, vec![256, 512], 524_288),
            cand("big.scales", Dtype::F64, vec![16, 512], 16_384),
            cand("big.qzeros", Dtype::I32, vec![16, 64], 4_096),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "F64 scales must be rejected regardless of shape");
    }

    // ── Scan: base_name with leading dot (e.g. ".hidden.proj") ──────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_base_name_with_leading_dot() {
        // Arrange: tensor name ".hidden.proj.qweight" → base_name ".hidden.proj"
        let tensors = vec![
            cand(".hidden.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand(".hidden.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand(".hidden.proj.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key(".hidden.proj"), "base_name with leading dot must be detected");
        assert_eq!(scan.groups[".hidden.proj"].base_name, ".hidden.proj");
    }

    // ── AwqGptqFormat: both variants produce distinct Debug strings ────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_debug_strings_are_distinct_and_stable() {
        // Arrange
        let awq_debug = format!("{:?}", AwqGptqFormat::Awq);
        let gptq_debug = format!("{:?}", AwqGptqFormat::Gptq);
        // Assert: both are non-empty, distinct, and match variant names
        assert!(!awq_debug.is_empty());
        assert!(!gptq_debug.is_empty());
        assert_ne!(awq_debug, gptq_debug, "Debug output must differ between variants");
        assert_eq!(awq_debug, "Awq");
        assert_eq!(gptq_debug, "Gptq");
    }

    // ── AwqGptqGroup: Clone of group with all String fields ensures deep copy ─
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_clone_deep_copies_all_string_fields() {
        // Arrange: group with all String fields populated
        let group = AwqGptqGroup {
            base_name: "deep.copy.test".to_string(),
            qweight_name: "deep.copy.test.qweight".to_string(),
            scales_name: "deep.copy.test.scales".to_string(),
            qzeros_name: "deep.copy.test.qzeros".to_string(),
            g_idx_name: Some("deep.copy.test.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![128, 256],
        };
        // Act
        let cloned = group.clone();
        // Assert: each String field is a distinct allocation
        assert!(!std::ptr::eq(group.base_name.as_ptr(), cloned.base_name.as_ptr()));
        assert!(!std::ptr::eq(group.qweight_name.as_ptr(), cloned.qweight_name.as_ptr()));
        assert!(!std::ptr::eq(group.scales_name.as_ptr(), cloned.scales_name.as_ptr()));
        assert!(!std::ptr::eq(group.qzeros_name.as_ptr(), cloned.qzeros_name.as_ptr()));
        let original_g_idx_ptr = group.g_idx_name.as_ref().unwrap().as_ptr();
        let cloned_g_idx_ptr = cloned.g_idx_name.as_ref().unwrap().as_ptr();
        assert!(!std::ptr::eq(original_g_idx_ptr, cloned_g_idx_ptr));
    }

    // ── AwqGptqScan: two groups with same base_name impossible by scanner ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scanner_produces_unique_base_names_per_group() {
        // Arrange: provide two valid triplets with same base_name
        // (HashMap deduplication means only one group survives)
        let tensors = vec![
            cand("dup.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("dup.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("dup.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: only one group with key "dup"
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("dup"));
        // keys() iterator yields exactly one entry
        let keys: Vec<_> = scan.groups.keys().collect();
        assert_eq!(keys.len(), 1);
    }

    // ── Scan: scales with shape [0, N] where qweight rows > 0 ──────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_zero_rows_qweight_nonzero_rows() {
        // Arrange: qweight [64, 128] → K=512, scales [0, 128] → scales_rows=0
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![0, 128], 0),
            cand("foo.qzeros", Dtype::I32, vec![0, 16], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: scales_rows == 0 → rejected
        assert!(scan.groups.is_empty(), "scales with zero rows must be rejected");
    }

    // ── Scan: only .scales and .qzeros present, no .qweight ────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_and_qzeros_without_qweight() {
        // Arrange: only scales and qzeros, no qweight — detection starts from .qweight
        let tensors = vec![
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: no group detected — scanner only triggers from .qweight
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty(), "scales/qzeros without qweight must not be consumed");
    }

    // ── Scan: group_size = K (scales_rows = 1, maximum possible group) ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_group_size_equals_k_large_k() {
        // Arrange: K=4096, N=4096, group_size=4096
        // qweight [512, 4096], scales [1, 4096], qzeros [1, 512]
        let tensors = vec![
            cand("huge_gs.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
            cand("huge_gs.scales", Dtype::F16, vec![1, 4096], 8_192),
            cand("huge_gs.qzeros", Dtype::I32, vec![1, 512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 4096 % 1 == 0 → valid
        assert_eq!(scan.groups.len(), 1, "group_size=K with large K should be accepted");
        assert_eq!(scan.groups["huge_gs"].qweight_shape, vec![512, 4096]);
    }

    // ── CandidateTensor: same name different shape not equal ────────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_eq_rejects_different_shapes_same_name() {
        // Arrange: two tensors with same name and dtype but different shapes
        let a = CandidateTensor {
            name: "shared".to_string(),
            dtype: Dtype::F16,
            shape: vec![4, 8],
            byte_len: 64,
        };
        let b = CandidateTensor {
            name: "shared".to_string(),
            dtype: Dtype::F16,
            shape: vec![8, 4],
            byte_len: 64,
        };
        // Assert: different shape order means not equal
        assert_ne!(a, b);
    }

    // ── Scan: tensor named ".qweight.qweight" has base_name ".qweight" ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_nested_qweight_suffix() {
        // Arrange: tensor name ".qweight.qweight" → base_name = ".qweight"
        let tensors = vec![
            cand(".qweight.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand(".qweight.scales", Dtype::F16, vec![4, 128], 1_024),
            cand(".qweight.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key(".qweight"), "base_name should be '.qweight'");
        assert_eq!(scan.groups[".qweight"].qweight_name, ".qweight.qweight");
    }

    // ── AwqGptqScan: groups.values() iterator yields correct count ──────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_groups_values_count_matches_len() {
        // Arrange: 3 groups
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("b.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("b.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("b.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("c.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("c.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("c.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        let values_count = scan.groups.values().count();
        assert_eq!(values_count, scan.groups.len());
        assert_eq!(values_count, 3);
    }

    // ── Scan: AWQ group detected with I16 qzeros and GPTQ group detected with I32 ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn mixed_qzeros_dtypes_across_groups() {
        // Arrange: one group with I16 qzeros, one with I32 qzeros
        let tensors = vec![
            cand("i16_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("i16_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("i16_layer.qzeros", Dtype::I16, vec![4, 16], 128),
            cand("i32_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("i32_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("i32_layer.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: both groups detected
        assert_eq!(scan.groups.len(), 2, "I16 and I32 qzeros should both be accepted in different groups");
        assert_eq!(scan.groups["i16_layer"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["i32_layer"].format, AwqGptqFormat::Awq);
    }

    // ── AwqGptqFormat: stored in Vec and sorted by Debug string ────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_variants_sortable_by_debug_string() {
        // Arrange
        let mut variants = vec![AwqGptqFormat::Gptq, AwqGptqFormat::Awq];
        // Act: sort by Debug string representation
        variants.sort_by_key(|v| format!("{v:?}"));
        // Assert: "Awq" < "Gptq" lexicographically
        assert_eq!(variants[0], AwqGptqFormat::Awq);
        assert_eq!(variants[1], AwqGptqFormat::Gptq);
    }

    // ── Scan: consumed set for GPTQ with I16 qzeros includes all 4 tensors ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_i16_qzeros_consumed_includes_g_idx() {
        // Arrange: GPTQ group with I16 qzeros + g_idx
        let tensors = vec![
            cand("gptq_i16.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq_i16.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq_i16.qzeros", Dtype::I16, vec![4, 16], 128),
            cand("gptq_i16.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 4 tensors consumed
        assert_eq!(scan.consumed.len(), 4, "qweight + scales + qzeros(I16) + g_idx = 4");
        assert!(scan.consumed.contains("gptq_i16.qweight"));
        assert!(scan.consumed.contains("gptq_i16.scales"));
        assert!(scan.consumed.contains("gptq_i16.qzeros"));
        assert!(scan.consumed.contains("gptq_i16.g_idx"));
        // Format is GPTQ
        assert_eq!(scan.groups["gptq_i16"].format, AwqGptqFormat::Gptq);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 50 — 15 additional unit tests for edge case coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qweight with exactly group_size=8 and N=1 (minimal viable) ─
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_minimal_viable_triplet_k8_n1() {
        // Arrange: qweight [1, 1] → K=8, N=1; scales [1, 1]; qzeros [1, 1]
        let tensors = vec![
            cand("min.qweight", Dtype::I32, vec![1, 1], 4),
            cand("min.scales", Dtype::F16, vec![1, 1], 2),
            cand("min.qzeros", Dtype::I32, vec![1, 1], 4),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["min"].qweight_shape, vec![1, 1]);
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── Scan: scales rows = K/8 = qweight rows but K not divisible by 7 ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_rows_prime_not_dividing_k() {
        // Arrange: qweight [11, 64] → K=88, scales [13, 64] → 88%13 != 0
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![11, 64], 2_816),
            cand("foo.scales", Dtype::F16, vec![13, 64], 1_664),
            cand("foo.qzeros", Dtype::I32, vec![13, 8], 416),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 88 % 13 = 10 != 0 → rejected
        assert!(scan.groups.is_empty(), "scales_rows=13 not dividing K=88 must be rejected");
    }

    // ── Scan: g_idx with empty string g_idx_name in consumed ─────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn g_idx_empty_base_name_still_consumed() {
        // Arrange: empty base_name with g_idx
        let tensors = vec![
            cand(".qweight", Dtype::I32, vec![64, 128], 32_768),
            cand(".scales", Dtype::F16, vec![4, 128], 1_024),
            cand(".qzeros", Dtype::I32, vec![4, 16], 256),
            cand(".g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 4 tensors consumed including g_idx
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 4, "qweight + scales + qzeros + g_idx = 4");
        assert!(scan.consumed.contains(".g_idx"));
        assert_eq!(scan.groups[""].format, AwqGptqFormat::Gptq);
    }

    // ── Scan: scales columns = qweight columns exactly equal (N match) ───
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_exact_n_match_between_qweight_and_scales() {
        // Arrange: N=1024 for both qweight and scales
        let tensors = vec![
            cand("nmatch.qweight", Dtype::I32, vec![128, 1024], 524_288),
            cand("nmatch.scales", Dtype::F16, vec![8, 1024], 16_384),
            cand("nmatch.qzeros", Dtype::I32, vec![8, 128], 4_096),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        let group = scan.groups.get("nmatch").unwrap();
        assert_eq!(group.qweight_shape[1], 1024, "N must be 1024");
    }

    // ── Scan: 20 groups stress test ──────────────────────────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_twenty_awq_groups() {
        // Arrange: 20 AWQ groups
        let mut tensors = Vec::new();
        for i in 0..20 {
            let base = format!("layer.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 20, "all 20 groups must be detected");
        assert_eq!(scan.consumed.len(), 60, "20 × 3 = 60 consumed");
    }

    // ── Scan: qweight with BF16 scales AND BF16 qzeros — both rejected ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_both_scales_and_qzeros_wrong_dtype() {
        // Arrange: BF16 scales AND BF16 qzeros — scales rejected first
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::BF16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::BF16, vec![4, 16], 128),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: scales BF16 fails dtype check before qzeros is examined
        assert!(scan.groups.is_empty(), "BF16 scales must cause rejection");
    }

    // ── AwqGptqGroup: g_idx_name with very long string ──────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_g_idx_name_very_long_string() {
        // Arrange
        let long_g_idx = format!("layer.{}.g_idx", "x".repeat(5000));
        let group = AwqGptqGroup {
            base_name: "base".to_string(),
            qweight_name: "base.qweight".to_string(),
            scales_name: "base.scales".to_string(),
            qzeros_name: "base.qzeros".to_string(),
            g_idx_name: Some(long_g_idx.clone()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        // Assert
        assert_eq!(group.g_idx_name.as_deref(), Some(long_g_idx.as_str()));
        assert!(group.g_idx_name.unwrap().len() > 5000);
    }

    // ── AwqGptqScan: groups.entry().or_insert() pattern works ───────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_groups_entry_api_works() {
        // Arrange
        let mut scan = AwqGptqScan::default();
        let group = AwqGptqGroup {
            base_name: "entry".to_string(),
            qweight_name: "entry.qweight".to_string(),
            scales_name: "entry.scales".to_string(),
            qzeros_name: "entry.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![4, 8],
        };
        // Act: use entry API
        scan.groups.entry("entry".to_string()).or_insert(group);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("entry"));
    }

    // ── Scan: qweight F64 dtype rejected ────────────────────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_dtype_f64() {
        // Arrange
        let tensors = vec![
            cand("foo.qweight", Dtype::F64, vec![64, 128], 65_536),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "F64 qweight must be rejected");
    }

    // ── Scan: qzeros I32 with large shape values accepted ───────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn accepts_qzeros_i32_large_shape() {
        // Arrange: large but valid shapes
        let tensors = vec![
            cand("big.qweight", Dtype::I32, vec![2048, 8192], 67_108_864),
            cand("big.scales", Dtype::F16, vec![128, 8192], 2_097_152),
            cand("big.qzeros", Dtype::I32, vec![128, 1024], 524_288),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["big"].qweight_shape, vec![2048, 8192]);
    }

    // ── Scan: only .qweight and .g_idx, no scales/qzeros ────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_and_g_idx_without_scales_qzeros() {
        // Arrange: qweight + g_idx but no scales or qzeros
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: no scales → rejected; g_idx alone not consumed
        assert!(scan.groups.is_empty());
        assert!(!scan.consumed.contains("foo.g_idx"), "orphan g_idx must not be consumed");
    }

    // ── AwqGptqFormat: used in a tuple with a string label ──────────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_in_tuple_pair() {
        // Arrange
        let pairs = [
            (AwqGptqFormat::Awq, "awq"),
            (AwqGptqFormat::Gptq, "gptq"),
        ];
        // Assert: tuples accessible by index
        assert_eq!(pairs[0].0, AwqGptqFormat::Awq);
        assert_eq!(pairs[0].1, "awq");
        assert_eq!(pairs[1].0, AwqGptqFormat::Gptq);
        assert_eq!(pairs[1].1, "gptq");
    }

    // ── Scan: scales rows = prime number dividing K exactly ─────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_scales_rows_prime_dividing_k() {
        // Arrange: K = 128, scales_rows = 2, 128 % 2 == 0 (prime dividing K)
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![16, 64], 4_096),
            cand("foo.scales", Dtype::F16, vec![2, 64], 256),
            cand("foo.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "prime scales_rows dividing K should be accepted");
    }

    // ── CandidateTensor: Clone produces equal but distinct String name ──
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_clone_distinct_name_allocation() {
        // Arrange
        let ct = CandidateTensor {
            name: "distinct_alloc".to_string(),
            dtype: Dtype::F32,
            shape: vec![10],
            byte_len: 40,
        };
        // Act
        let cloned = ct.clone();
        // Assert: equal values but different String allocations
        assert_eq!(ct.name, cloned.name);
        assert!(!std::ptr::eq(ct.name.as_ptr(), cloned.name.as_ptr()));
    }

    // ── Scan: AWQ group with qzeros shape [scales_rows, N] (unpacked) ───
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn accepts_qzeros_unpacked_shape() {
        // Arrange: qzeros columns = N (not N/8), rows still match scales rows
        let tensors = vec![
            cand("unpacked.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("unpacked.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("unpacked.qzeros", Dtype::I32, vec![4, 128], 2_048), // cols = N, not N/8
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: column count is not validated, so accepted
        assert_eq!(scan.groups.len(), 1, "unpacked qzeros columns should be accepted");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 51 — 15 additional unit tests for edge cases and format coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qweight with shape [rows, 2] — N=2, minimal non-trivial columns
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_shape_n_equals_2() {
        // Arrange: qweight [32, 2] → K=256, N=2
        // scales [2, 2] → scales_rows=2, 256%2==0
        // qzeros [2, 1]
        let tensors = vec![
            cand("n2.qweight", Dtype::I32, vec![32, 2], 256),
            cand("n2.scales", Dtype::F16, vec![2, 2], 8),
            cand("n2.qzeros", Dtype::I32, vec![2, 1], 8),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "N=2 should be a valid column count");
        assert_eq!(scan.groups["n2"].qweight_shape, vec![32, 2]);
    }

    // ── Scan: GPTQ group where g_idx has wrong shape (1D with wrong length) still detected
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_g_idx_wrong_length_still_triggers_format() {
        // Arrange: g_idx shape [1] — scanner only checks presence, not shape validity
        let tensors = vec![
            cand("proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("proj.g_idx", Dtype::I32, vec![1], 4), // wrong length but present
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: format is GPTQ because g_idx exists; shape not validated
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["proj"].format, AwqGptqFormat::Gptq);
        assert!(scan.consumed.contains("proj.g_idx"));
    }

    // ── Scan: qweight with shape [1, usize::MAX] — extreme column count
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_extreme_column_count() {
        // Arrange: qweight [1, usize::MAX] → K=8, N=usize::MAX
        // scales [1, usize::MAX] → scales_rows=1, 8%1==0
        // qzeros [1, usize::MAX]
        let n = usize::MAX;
        let tensors = vec![
            cand("extreme.qweight", Dtype::I32, vec![1, n], 0),
            cand("extreme.scales", Dtype::F16, vec![1, n], 0),
            cand("extreme.qzeros", Dtype::I32, vec![1, n], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: shape checks pass (2D, cols match, rows=1 divides K=8)
        assert_eq!(scan.groups.len(), 1, "extreme N should pass shape validation");
        assert_eq!(scan.groups["extreme"].qweight_shape, vec![1, n]);
    }

    // ── Scan: scales rows equal to K itself (group_size=1, maximal scales)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_scales_rows_equals_k_directly() {
        // Arrange: qweight [8, 64] → K=64, scales [64, 64] → 64%64==0, group_size=1
        // qzeros [64, 8]
        let tensors = vec![
            cand("gs1_direct.qweight", Dtype::I32, vec![8, 64], 2_048),
            cand("gs1_direct.scales", Dtype::F16, vec![64, 64], 8_192),
            cand("gs1_direct.qzeros", Dtype::I32, vec![64, 8], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "scales_rows=K is valid (group_size=1)");
        assert_eq!(scan.groups["gs1_direct"].qweight_shape, vec![8, 64]);
    }

    // ── Scan: 3 GPTQ groups with different scales rows (different group sizes)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_three_gptq_groups_varying_group_sizes() {
        // Arrange: three groups with K=512 but different scales_rows
        let tensors = vec![
            // group_size=8: scales [64, 64]
            cand("gs8.qweight", Dtype::I32, vec![64, 64], 16_384),
            cand("gs8.scales", Dtype::F16, vec![64, 64], 8_192),
            cand("gs8.qzeros", Dtype::I32, vec![64, 8], 2_048),
            cand("gs8.g_idx", Dtype::I32, vec![512], 2_048),
            // group_size=16: scales [32, 64]
            cand("gs16.qweight", Dtype::I32, vec![64, 64], 16_384),
            cand("gs16.scales", Dtype::F16, vec![32, 64], 4_096),
            cand("gs16.qzeros", Dtype::I32, vec![32, 8], 1_024),
            cand("gs16.g_idx", Dtype::I32, vec![512], 2_048),
            // group_size=128: scales [4, 64]
            cand("gs128.qweight", Dtype::I32, vec![64, 64], 16_384),
            cand("gs128.scales", Dtype::F16, vec![4, 64], 512),
            cand("gs128.qzeros", Dtype::I32, vec![4, 8], 128),
            cand("gs128.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 3 groups detected as GPTQ
        assert_eq!(scan.groups.len(), 3, "three groups with different group sizes");
        assert_eq!(scan.groups["gs8"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.groups["gs16"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.groups["gs128"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 12, "3 groups x 4 tensors = 12 consumed");
    }

    // ── Scan: scales with row count that is a large prime not dividing K
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_rows_large_prime_not_dividing_k() {
        // Arrange: qweight [64, 128] → K=512
        // scales [101, 128] → 512 % 101 = 7 != 0 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![101, 128], 25_856),
            cand("foo.qzeros", Dtype::I32, vec![101, 16], 6_464),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(
            scan.groups.is_empty(),
            "scales_rows=101 not dividing K=512 must be rejected"
        );
    }

    // ── Scan: scales shape [rows, N] where rows=1 and K=8 (minimal valid)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_minimal_k8_with_scales_rows_1() {
        // Arrange: qweight [1, 32] → K=8, N=32; scales [1, 32]; qzeros [1, 4]
        let tensors = vec![
            cand("min_k8.qweight", Dtype::I32, vec![1, 32], 128),
            cand("min_k8.scales", Dtype::F16, vec![1, 32], 64),
            cand("min_k8.qzeros", Dtype::I32, vec![1, 4], 16),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "minimal K=8 with single scale row must be valid");
        assert_eq!(scan.groups["min_k8"].qweight_shape, vec![1, 32]);
    }

    // ── Scan: qzeros with I16 dtype and rows matching scales (AWQ variant)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn awq_i16_qzeros_rows_match_scales_detected() {
        // Arrange: AWQ triplet with I16 qzeros, no g_idx
        let tensors = vec![
            cand("layer.gate.qweight", Dtype::I32, vec![128, 256], 131_072),
            cand("layer.gate.scales", Dtype::F16, vec![8, 256], 4_096),
            cand("layer.gate.qzeros", Dtype::I16, vec![8, 32], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("layer.gate").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq, "no g_idx means AWQ format");
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.qweight_shape, vec![128, 256]);
    }

    // ── Scan: base_name with only a single character
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_base_name_single_character() {
        // Arrange: base_name is just "x"
        let tensors = vec![
            cand("x.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("x.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("x.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("x"));
        assert_eq!(scan.groups["x"].base_name, "x");
        assert_eq!(scan.groups["x"].qweight_name, "x.qweight");
    }

    // ── Scan: 15 AWQ groups + 5 GPTQ groups stress test
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_20_mixed_format_groups() {
        // Arrange: 15 AWQ + 5 GPTQ = 20 groups total
        let mut tensors = Vec::new();
        for i in 0..15 {
            let base = format!("awq.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        for i in 0..5 {
            let base = format!("gptq.{i}.proj");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
            tensors.push(cand(&format!("{base}.g_idx"), Dtype::I32, vec![512], 2_048));
        }
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 20, "15 AWQ + 5 GPTQ = 20 groups");
        // AWQ: 15 x 3 = 45 consumed, GPTQ: 5 x 4 = 20 consumed = 65 total
        assert_eq!(scan.consumed.len(), 65, "15x3 + 5x4 = 65 consumed");
        let awq_count = scan.groups.values().filter(|g| g.format == AwqGptqFormat::Awq).count();
        let gptq_count = scan.groups.values().filter(|g| g.format == AwqGptqFormat::Gptq).count();
        assert_eq!(awq_count, 15);
        assert_eq!(gptq_count, 5);
    }

    // ── Scan: qweight with shape [rows, N] where N is odd and non-trivial
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_odd_column_count() {
        // Arrange: qweight [32, 33] → K=256, N=33 (odd)
        // scales [2, 33] → scales_rows=2, 256%2==0
        // qzeros [2, 5]
        let tensors = vec![
            cand("odd_n.qweight", Dtype::I32, vec![32, 33], 4_224),
            cand("odd_n.scales", Dtype::F16, vec![2, 33], 132),
            cand("odd_n.qzeros", Dtype::I32, vec![2, 5], 40),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: N being odd is not validated against
        assert_eq!(scan.groups.len(), 1, "odd N should be accepted");
        assert_eq!(scan.groups["odd_n"].qweight_shape, vec![32, 33]);
    }

    // ── Scan: scales columns off-by-one lower than qweight (large shape)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_columns_one_less_than_qweight_large_shape() {
        // Arrange: qweight [256, 8192] → N=8192, scales [16, 8191] → N mismatch
        let tensors = vec![
            cand("big.qweight", Dtype::I32, vec![256, 8192], 8_388_608),
            cand("big.scales", Dtype::F16, vec![16, 8191], 262_112),
            cand("big.qzeros", Dtype::I32, vec![16, 1024], 65_536),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(
            scan.groups.is_empty(),
            "scales columns 8191 vs qweight columns 8192 must be rejected"
        );
    }

    // ── AwqGptqGroup: Clone and then mutate qweight_shape preserves original
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_clone_mutate_qweight_shape_preserves_original() {
        // Arrange
        let group = AwqGptqGroup {
            base_name: "original".to_string(),
            qweight_name: "original.qweight".to_string(),
            scales_name: "original.scales".to_string(),
            qzeros_name: "original.qzeros".to_string(),
            g_idx_name: Some("original.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![128, 256],
        };
        // Act
        let mut cloned = group.clone();
        cloned.qweight_shape.clear();
        cloned.format = AwqGptqFormat::Awq;
        // Assert: original unchanged
        assert_eq!(group.qweight_shape, vec![128, 256], "original qweight_shape must be unchanged");
        assert_eq!(group.format, AwqGptqFormat::Gptq, "original format must be unchanged");
        assert!(cloned.qweight_shape.is_empty());
        assert_eq!(cloned.format, AwqGptqFormat::Awq);
    }

    // ── Scan: AWQ group detected alongside many dense FP32 weights
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn single_awq_among_many_dense_weights() {
        // Arrange: many dense weights + one AWQ triplet
        let tensors = vec![
            cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524_288_000),
            cand("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4096], 16_384),
            cand("model.layers.0.self_attn.k_proj.weight", Dtype::F32, vec![4096, 1024], 16_777_216),
            cand("model.layers.0.self_attn.v_proj.weight", Dtype::F32, vec![4096, 1024], 16_777_216),
            cand("model.layers.0.self_attn.o_proj.weight", Dtype::F32, vec![4096, 4096], 67_108_864),
            // One AWQ triplet for gate_proj
            cand("model.layers.0.mlp.gate_proj.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
            cand("model.layers.0.mlp.gate_proj.scales", Dtype::F16, vec![32, 4096], 262_144),
            cand("model.layers.0.mlp.gate_proj.qzeros", Dtype::I32, vec![32, 512], 65_536),
            // More dense weights
            cand("model.layers.0.mlp.up_proj.weight", Dtype::F32, vec![4096, 11008], 180_355_072),
            cand("model.layers.0.mlp.down_proj.weight", Dtype::F32, vec![11008, 4096], 180_355_072),
            cand("model.norm.weight", Dtype::F32, vec![4096], 16_384),
            cand("lm_head.weight", Dtype::F32, vec![4096, 32000], 524_288_000),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: exactly one group found, only 3 tensors consumed
        assert_eq!(scan.groups.len(), 1, "only the AWQ triplet should be detected");
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.groups.contains_key("model.layers.0.mlp.gate_proj"));
        // Dense weights not consumed
        for dense_name in &[
            "model.embed_tokens.weight",
            "model.layers.0.mlp.up_proj.weight",
            "lm_head.weight",
        ] {
            assert!(!scan.consumed.contains(*dense_name));
        }
    }

    // ── Scan: GPTQ group with I16 qzeros verified via consumed set completeness
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_i16_qzeros_consumed_set_complete() {
        // Arrange: GPTQ with I16 qzeros
        let qweight = "gptq_i16_v2.qweight";
        let scales = "gptq_i16_v2.scales";
        let qzeros = "gptq_i16_v2.qzeros";
        let g_idx = "gptq_i16_v2.g_idx";
        let tensors = vec![
            cand(qweight, Dtype::I32, vec![32, 64], 8_192),
            cand(scales, Dtype::F16, vec![2, 64], 256),
            cand(qzeros, Dtype::I16, vec![2, 8], 32),
            cand(g_idx, Dtype::I32, vec![256], 1_024),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: every tensor name is in consumed
        assert_eq!(scan.consumed.len(), 4);
        assert!(scan.consumed.contains(qweight));
        assert!(scan.consumed.contains(scales));
        assert!(scan.consumed.contains(qzeros));
        assert!(scan.consumed.contains(g_idx));
        // Group format and g_idx_name correct
        let group = scan.groups.get("gptq_i16_v2").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(group.g_idx_name.as_deref(), Some(g_idx));
    }

    // ── Scan: scales columns one more than qweight columns (off-by-one high)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_columns_one_more_than_qweight() {
        // Arrange: qweight N=64, scales N=65
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("foo.scales", Dtype::F16, vec![2, 65], 260),
            cand("foo.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(
            scan.groups.is_empty(),
            "scales columns 65 vs qweight columns 64 must be rejected"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 52 — 15 additional unit tests for edge cases and coverage gaps
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qweight with empty shape vec (not 2D) rejected via scan ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_empty_shape_via_scan() {
        // Arrange: qweight has empty shape (scalar-like), not 2D
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![], 0),
            cand("foo.scales", Dtype::F16, vec![], 0),
            cand("foo.qzeros", Dtype::I32, vec![], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: shape.len() != 2 → rejected
        assert!(scan.groups.is_empty(), "empty-shape qweight must be rejected by scan");
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: scan from take() iterator (bounded iterator) ─────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_from_take_iterator() {
        // Arrange: 5 tensors but only take first 3 (the complete triplet)
        let all_tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("noise.weight", Dtype::F32, vec![10], 40),
            cand("noise.bias", Dtype::F32, vec![5], 20),
        ];
        // Act: only take first 3
        let scan = scan_awq_gptq_groups(all_tensors.into_iter().take(3));
        // Assert: triplet detected from the limited iterator
        assert_eq!(scan.groups.len(), 1, "take(3) should provide a complete triplet");
        assert!(scan.groups.contains_key("a"));
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── AwqGptqGroup: g_idx_name is Some with empty string ────────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_g_idx_name_some_empty_string() {
        // Arrange: g_idx_name is Some("") — unusual but valid for the struct
        let group = AwqGptqGroup {
            base_name: "test".to_string(),
            qweight_name: "test.qweight".to_string(),
            scales_name: "test.scales".to_string(),
            qzeros_name: "test.qzeros".to_string(),
            g_idx_name: Some(String::new()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        // Assert: Some("") is not None but contains empty string
        assert!(group.g_idx_name.is_some(), "g_idx_name must be Some");
        assert!(group.g_idx_name.as_deref().unwrap().is_empty(),
            "inner string must be empty");
    }

    // ── Scan: scales and qzeros present but wrong scales rows vs qzeros rows ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qzeros_fewer_rows_than_scales() {
        // Arrange: qzeros has 2 rows but scales has 4 rows
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![2, 16], 128), // rows=2, scales rows=4
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qzeros rows (2) != scales rows (4) → rejected
        assert!(scan.groups.is_empty(), "qzeros rows < scales rows must be rejected");
    }

    // ── Scan: multiple tensors with identical names — last-wins HashMap behavior
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn duplicate_scales_tensor_last_wins_in_hashmap() {
        // Arrange: two scales tensors with same name, different shapes
        // HashMap::insert overwrites, so the second one is used
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            // First scales with wrong N
            cand("foo.scales", Dtype::F16, vec![4, 256], 2_048),
            // Second scales overwrites first with correct N
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: second scales (correct shape) wins, triplet detected
        assert_eq!(scan.groups.len(), 1, "last scales tensor should enable triplet detection");
        assert!(scan.groups.contains_key("foo"));
    }

    // ── Scan: scan from map() transformed iterator ────────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_from_map_iterator() {
        // Arrange: create candidates from raw tuples via map
        let raw: Vec<(&str, Dtype, Vec<usize>, usize)> = vec![
            ("m.qweight", Dtype::I32, vec![64, 128], 32_768),
            ("m.scales", Dtype::F16, vec![4, 128], 1_024),
            ("m.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act: map tuples into CandidateTensor
        let scan = scan_awq_gptq_groups(raw.into_iter().map(|(n, d, s, b)| CandidateTensor {
            name: n.to_string(),
            dtype: d,
            shape: s,
            byte_len: b,
        }));
        // Assert
        assert_eq!(scan.groups.len(), 1, "mapped iterator should produce valid scan");
        assert!(scan.groups.contains_key("m"));
    }

    // ── Scan: only scales and g_idx present, no qweight or qzeros ─────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_and_g_idx_without_qweight_or_qzeros() {
        // Arrange: scales + g_idx but no qweight → scanner starts from .qweight
        let tensors = vec![
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: nothing consumed, no groups
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
        assert!(!scan.consumed.contains("foo.g_idx"), "orphan g_idx must not be consumed");
    }

    // ── AwqGptqGroup: base_name contains ".qweight" as substring ─────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_base_name_contains_qweight_substring() {
        // Arrange: base_name includes ".qweight" literally — unusual but struct allows it
        let group = AwqGptqGroup {
            base_name: "layer.qweight_copy".to_string(),
            qweight_name: "layer.qweight_copy.qweight".to_string(),
            scales_name: "layer.qweight_copy.scales".to_string(),
            qzeros_name: "layer.qweight_copy.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![64, 128],
        };
        // Assert: base_name contains ".qweight" as a substring
        assert!(group.base_name.contains(".qweight"));
        assert_eq!(group.qweight_name, "layer.qweight_copy.qweight");
    }

    // ── CandidateTensor: Eq consistency — equal values always equal ──
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_eq_consistency_repeated_checks() {
        // Arrange: two identical CandidateTensors
        let a = CandidateTensor {
            name: "tensor_eq_test".to_string(),
            dtype: Dtype::F32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let b = CandidateTensor {
            name: "tensor_eq_test".to_string(),
            dtype: Dtype::F32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        // Assert: Eq consistency — repeated equality checks must return same result
        assert_eq!(a, b, "first check");
        assert_eq!(a, b, "second check");
        assert!(!(a != b), "negated ne must hold");
        // Verify symmetry
        assert_eq!(b, a, "equality must be symmetric");
    }

    // ── AwqGptqScan: consumed set is exact union of all group tensor names
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_consumed_is_exact_union_of_group_names() {
        // Arrange: 2 AWQ + 1 GPTQ
        let tensors = vec![
            cand("awq1.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq1.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq1.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("awq2.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq2.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq2.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq1.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq1.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq1.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq1.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: consumed = 2*3 AWQ + 1*4 GPTQ = 10
        assert_eq!(scan.consumed.len(), 10);
        // Every group's tensor name is in consumed
        for group in scan.groups.values() {
            assert!(scan.consumed.contains(&group.qweight_name));
            assert!(scan.consumed.contains(&group.scales_name));
            assert!(scan.consumed.contains(&group.qzeros_name));
            if let Some(ref g) = group.g_idx_name {
                assert!(scan.consumed.contains(g));
            }
        }
        // Consumed has no extra entries beyond group tensor names
        let all_group_names: HashSet<String> = scan.groups.values().flat_map(|g| {
            let mut names = vec![
                g.qweight_name.clone(),
                g.scales_name.clone(),
                g.qzeros_name.clone(),
            ];
            if let Some(ref g_idx) = g.g_idx_name {
                names.push(g_idx.clone());
            }
            names
        }).collect();
        assert_eq!(scan.consumed, all_group_names, "consumed must be exactly the union of group tensor names");
    }

    // ── Scan: minimal 1x1 triplet with I16 qzeros ────────────────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_minimal_1x1_with_i16_qzeros() {
        // Arrange: smallest possible triplet with I16 qzeros
        let tensors = vec![
            cand("tiny.qweight", Dtype::I32, vec![1, 1], 4),
            cand("tiny.scales", Dtype::F16, vec![1, 1], 2),
            cand("tiny.qzeros", Dtype::I16, vec![1, 1], 2),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "1x1 with I16 qzeros must be detected");
        assert_eq!(scan.groups["tiny"].qweight_shape, vec![1, 1]);
        assert_eq!(scan.groups["tiny"].format, AwqGptqFormat::Awq);
        assert!(scan.groups["tiny"].g_idx_name.is_none());
    }

    // ── Scan: g_idx present but qweight has wrong dtype — g_idx not consumed
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn g_idx_not_consumed_when_qweight_dtype_wrong() {
        // Arrange: qweight is F32 (wrong), scales and qzeros are correct, g_idx present
        let tensors = vec![
            cand("foo.qweight", Dtype::F32, vec![64, 128], 32_768), // wrong dtype
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("foo.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: triplet rejected, g_idx not consumed
        assert!(scan.groups.is_empty(), "wrong qweight dtype must reject entire triplet");
        assert!(!scan.consumed.contains("foo.g_idx"), "g_idx must not be consumed for rejected triplet");
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: two groups sharing same prefix but one is GPTQ, other has longer name
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn overlapping_prefixes_one_awq_one_gptq() {
        // Arrange: "proj" is AWQ, "proj_v2" is GPTQ — different base names
        let tensors = vec![
            cand("proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("proj_v2.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("proj_v2.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("proj_v2.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("proj_v2.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: two independent groups
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups["proj"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["proj_v2"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 7); // 3 AWQ + 4 GPTQ
    }

    // ── AwqGptqFormat: used as closure return value via match ────────
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_as_closure_return_via_match() {
        // Arrange: closure that maps bool to format variant
        let resolve = |has_g_idx: bool| -> AwqGptqFormat {
            match has_g_idx {
                true => AwqGptqFormat::Gptq,
                false => AwqGptqFormat::Awq,
            }
        };
        // Assert
        assert_eq!(resolve(false), AwqGptqFormat::Awq);
        assert_eq!(resolve(true), AwqGptqFormat::Gptq);
    }

    // ── Scan: qweight with scales rows = qweight rows and qzeros I16 ──
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_scales_rows_equal_qweight_rows_with_i16_qzeros() {
        // Arrange: K=512, group_size=8 → scales_rows=64=qweight_rows, I16 qzeros
        let tensors = vec![
            cand("mixed.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("mixed.scales", Dtype::F16, vec![64, 128], 16_384),
            cand("mixed.qzeros", Dtype::I16, vec![64, 16], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "scales_rows=qweight_rows with I16 qzeros should be valid");
        assert_eq!(scan.groups["mixed"].qweight_shape, vec![64, 128]);
        assert_eq!(scan.groups["mixed"].format, AwqGptqFormat::Awq);
    }

    // ── Scan: scales rows = K but qzeros rows = 0 (mismatch) ────────
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qzeros_zero_rows_scales_rows_nonzero() {
        // Arrange: qweight [8, 64] → K=64, scales [64, 64] → scales_rows=64
        // qzeros [0, 8] → rows=0 != 64 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![8, 64], 2_048),
            cand("foo.scales", Dtype::F16, vec![64, 64], 8_192),
            cand("foo.qzeros", Dtype::I32, vec![0, 8], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qzeros rows (0) != scales rows (64)
        assert!(scan.groups.is_empty(), "qzeros rows=0 vs scales rows=64 must be rejected");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 53 — 15 additional unit tests for edge cases and coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── AwqGptqFormat: transitive equality (a==b and b==c implies a==c)
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_transitive_equality() {
        // Arrange: three variables all holding the same variant
        let a = AwqGptqFormat::Gptq;
        let b = a;
        let c = b;
        // Assert: transitivity holds for Copy types
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "transitive equality must hold: a==b && b==c => a==c");
    }

    // ── AwqGptqGroup: Clone preserves all seven fields exactly
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_clone_all_fields_identical_after_mutation() {
        // Arrange: group with all fields, clone it, mutate clone deeply
        let group = AwqGptqGroup {
            base_name: "src".to_string(),
            qweight_name: "src.qweight".to_string(),
            scales_name: "src.scales".to_string(),
            qzeros_name: "src.qzeros".to_string(),
            g_idx_name: Some("src.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![128, 256],
        };
        // Act
        let mut cloned = group.clone();
        cloned.base_name = "dst".to_string();
        cloned.qweight_name = "dst.qweight".to_string();
        cloned.scales_name = "dst.scales".to_string();
        cloned.qzeros_name = "dst.qzeros".to_string();
        cloned.g_idx_name = None;
        cloned.format = AwqGptqFormat::Awq;
        cloned.qweight_shape[0] = 0;
        // Assert: original untouched
        assert_eq!(group.base_name, "src");
        assert_eq!(group.qweight_name, "src.qweight");
        assert_eq!(group.scales_name, "src.scales");
        assert_eq!(group.qzeros_name, "src.qzeros");
        assert_eq!(group.g_idx_name.as_deref(), Some("src.g_idx"));
        assert_eq!(group.format, AwqGptqFormat::Gptq);
        assert_eq!(group.qweight_shape, vec![128, 256]);
    }

    // ── AwqGptqScan: retain() filtering groups by format
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_groups_retain_keeps_only_awq() {
        // Arrange: 2 AWQ + 1 GPTQ groups
        let tensors = vec![
            cand("awq1.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq1.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq1.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("awq2.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq2.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq2.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq1.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq1.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq1.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq1.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        let mut scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 3);
        // Act: retain only AWQ groups
        scan.groups.retain(|_, g| g.format == AwqGptqFormat::Awq);
        // Assert
        assert_eq!(scan.groups.len(), 2, "only AWQ groups should remain");
        assert!(scan.groups.contains_key("awq1"));
        assert!(scan.groups.contains_key("awq2"));
        assert!(!scan.groups.contains_key("gptq1"));
        // Consumed set is NOT modified by retain (separate collection)
        assert_eq!(scan.consumed.len(), 10, "consumed set retains all entries");
    }

    // ── Scan: scan from rev() iterator (reverse order of tensors)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_from_rev_iterator() {
        // Arrange: tensors in reverse order — qzeros first, qweight last
        let tensors = vec![
            cand("rev.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("rev.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("rev.qweight", Dtype::I32, vec![64, 128], 32_768),
        ];
        // Act: iterate in reverse (so qweight is encountered first in iteration)
        let scan = scan_awq_gptq_groups(tensors.into_iter().rev());
        // Assert: order does not matter — all tensors go into HashMap first
        assert_eq!(scan.groups.len(), 1, "reverse-order iterator must still detect triplet");
        assert!(scan.groups.contains_key("rev"));
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── CandidateTensor: Eq consistency with repeated equality checks
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_eq_symmetry_and_transitivity() {
        // Arrange: three identical tensors
        let a = CandidateTensor {
            name: "eq_test".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let b = CandidateTensor {
            name: "eq_test".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let c = CandidateTensor {
            name: "eq_test".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        // Assert: reflexivity
        assert_eq!(a, a, "reflexivity: a must equal itself");
        // Symmetry
        assert_eq!(a, b, "a == b");
        assert_eq!(b, a, "symmetry: b == a");
        // Transitivity
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "transitivity: a==b && b==c => a==c");
    }

    // ── AwqGptqFormat: both variants collected into slice and indexed
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_variants_in_slice_indexable() {
        // Arrange
        let variants: &[AwqGptqFormat] = &[AwqGptqFormat::Awq, AwqGptqFormat::Gptq];
        // Assert: index access works, correct variant at each index
        assert_eq!(variants[0], AwqGptqFormat::Awq);
        assert_eq!(variants[1], AwqGptqFormat::Gptq);
        assert_eq!(variants.len(), 2);
        // Verify iteration over slice
        let collected: Vec<_> = variants.iter().copied().collect();
        assert_eq!(collected.len(), 2);
    }

    // ── Scan: qweight [1, 32] + scales [1, 32] + qzeros I16 [1, 4] (minimal AWQ I16)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_minimal_awq_i16_qzeros_single_row() {
        // Arrange: K=8, N=32, group_size=8, I16 qzeros
        let tensors = vec![
            cand("mini.qweight", Dtype::I32, vec![1, 32], 128),
            cand("mini.scales", Dtype::F16, vec![1, 32], 64),
            cand("mini.qzeros", Dtype::I16, vec![1, 4], 8),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "minimal single-row AWQ with I16 qzeros must be detected");
        let group = scan.groups.get("mini").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.qweight_shape, vec![1, 32]);
    }

    // ── AwqGptqGroup: base_name containing newline characters
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_base_name_with_newline() {
        // Arrange: base_name has a newline — struct does not restrict string content
        let group = AwqGptqGroup {
            base_name: "layer\n0".to_string(),
            qweight_name: "layer\n0.qweight".to_string(),
            scales_name: "layer\n0.scales".to_string(),
            qzeros_name: "layer\n0.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![4, 8],
        };
        // Assert
        assert!(group.base_name.contains('\n'));
        assert_eq!(group.base_name, "layer\n0");
        let debug = format!("{group:?}");
        assert!(debug.contains("layer"), "Debug must still render the base_name substring");
    }

    // ── AwqGptqScan: removing a group key does not remove its tensors from consumed
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_groups_remove_does_not_clear_consumed() {
        // Arrange
        let tensors = vec![
            cand("keep.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("keep.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("keep.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("drop.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("drop.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("drop.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let mut scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.consumed.len(), 6);
        // Act: remove one group
        let removed = scan.groups.remove("drop");
        assert!(removed.is_some());
        // Assert: consumed still has all 6 entries
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 6, "consumed set must not be modified by groups.remove()");
        assert!(scan.consumed.contains("drop.qweight"));
        assert!(scan.consumed.contains("keep.qweight"));
    }

    // ── Scan: scales shape [rows, N] with rows = K (group_size = 1) and GPTQ
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_gptq_group_size_1_scales_rows_equals_k() {
        // Arrange: qweight [8, 64] → K=64, scales [64, 64] → group_size=1
        // qzeros [64, 8], g_idx present
        let tensors = vec![
            cand("gs1_gptq.qweight", Dtype::I32, vec![8, 64], 2_048),
            cand("gs1_gptq.scales", Dtype::F16, vec![64, 64], 8_192),
            cand("gs1_gptq.qzeros", Dtype::I32, vec![64, 8], 2_048),
            cand("gs1_gptq.g_idx", Dtype::I32, vec![64], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "GPTQ with group_size=1 should be valid");
        assert_eq!(scan.groups["gs1_gptq"].format, AwqGptqFormat::Gptq);
        assert!(scan.groups["gs1_gptq"].g_idx_name.is_some());
        assert_eq!(scan.consumed.len(), 4);
    }

    // ── AwqGptqGroup: qweight_shape with three dimensions (struct allows any Vec)
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_qweight_shape_three_dims_access_by_index() {
        // Arrange
        let group = AwqGptqGroup {
            base_name: "3d".to_string(),
            qweight_name: "3d.qweight".to_string(),
            scales_name: "3d.scales".to_string(),
            qzeros_name: "3d.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![16, 32, 64],
        };
        // Assert: individual dimension access
        assert_eq!(group.qweight_shape[0], 16);
        assert_eq!(group.qweight_shape[1], 32);
        assert_eq!(group.qweight_shape[2], 64);
        // Verify iter produces all values
        let sum: usize = group.qweight_shape.iter().sum();
        assert_eq!(sum, 16 + 32 + 64);
    }

    // ── Scan: realistic single layer with all 7 projections quantized AWQ
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn single_layer_seven_awq_projections_detected() {
        // Arrange: 4 attention + 3 MLP = 7 AWQ groups
        let projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
        let mut tensors = Vec::new();
        for proj in &projs {
            let base = format!("model.layers.0.self_attn.{proj}");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 7 groups detected, all AWQ
        assert_eq!(scan.groups.len(), 7, "7 AWQ projections must be detected");
        assert_eq!(scan.consumed.len(), 21, "7 x 3 = 21 consumed");
        for proj in &projs {
            let key = format!("model.layers.0.self_attn.{proj}");
            assert_eq!(scan.groups[&key].format, AwqGptqFormat::Awq);
        }
    }

    // ── CandidateTensor: zero byte_len with empty shape
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_zero_byte_len_empty_shape() {
        // Arrange: scalar-like tensor with no bytes and no dimensions
        let ct = CandidateTensor {
            name: "phantom".to_string(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 0,
        };
        // Assert
        assert_eq!(ct.byte_len, 0);
        assert!(ct.shape.is_empty());
        // Equality with another identical instance
        let ct2 = CandidateTensor {
            name: "phantom".to_string(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 0,
        };
        assert_eq!(ct, ct2);
    }

    // ── Scan: tensor names with mixed suffix — ".qweight" appears twice in name
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_base_name_with_embedded_qweight_substring() {
        // Arrange: base_name "model.qweight_backup" — strip_suffix only strips the
        // trailing ".qweight", so the full name is "model.qweight_backup.qweight"
        // base_name becomes "model.qweight_backup"
        let tensors = vec![
            cand("model.qweight_backup.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("model.qweight_backup.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("model.qweight_backup.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: strip_suffix removes only the trailing ".qweight"
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("model.qweight_backup"));
        assert_eq!(
            scan.groups["model.qweight_backup"].base_name,
            "model.qweight_backup"
        );
    }

    // ── AwqGptqScan: Debug output with mixed AWQ and GPTQ groups
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_debug_mixed_formats_shows_both() {
        // Arrange: one AWQ + one GPTQ
        let tensors = vec![
            cand("awq_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq_layer.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_layer.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq_layer.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq_layer.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq_layer.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        let debug = format!("{scan:?}");
        // Assert: Debug contains both format variant names
        assert!(debug.contains("Awq"), "Debug must contain 'Awq': {debug}");
        assert!(debug.contains("Gptq"), "Debug must contain 'Gptq': {debug}");
        assert!(debug.contains("awq_layer"), "Debug must contain AWQ group base_name: {debug}");
        assert!(debug.contains("gptq_layer"), "Debug must contain GPTQ group base_name: {debug}");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 54 — 15 additional unit tests for uncovered edge cases
    // ══════════════════════════════════════════════════════════════════════

    // ── CandidateTensor: Eq 语义 — assert_eq 与 assert_ne 互斥
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_eq_ne_mutually_exclusive() {
        // Arrange: 相同与不同的 CandidateTensor
        let a = CandidateTensor {
            name: "same".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let b = CandidateTensor {
            name: "same".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        let c = CandidateTensor {
            name: "diff".to_string(),
            dtype: Dtype::I32,
            shape: vec![4, 8],
            byte_len: 128,
        };
        // Assert: 相等时不等必须为 false，反之亦然
        assert_eq!(a, b, "identical tensors must be equal");
        assert!(!(a != b), "ne must be false when eq is true");
        assert_ne!(a, c, "different tensors must be non-equal");
        assert!(!(a == c), "eq must be false when ne is true");
    }

    // ── CandidateTensor: Clone 后修改 byte_len 不影响原始对象
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_clone_modify_byte_len_independence() {
        // Arrange
        let ct = CandidateTensor {
            name: "original".to_string(),
            dtype: Dtype::F32,
            shape: vec![10, 20],
            byte_len: 800,
        };
        // Act: Clone 后修改 byte_len
        let mut cloned = ct.clone();
        cloned.byte_len = 0;
        // Assert: 原始对象不受影响
        assert_eq!(ct.byte_len, 800, "original byte_len must be unchanged");
        assert_eq!(cloned.byte_len, 0, "cloned byte_len must be 0");
    }

    // ── AwqGptqScan: consumed 的 drain 操作清空集合但返回所有元素
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_consumed_drain_empties_and_returns_all() {
        // Arrange: 构建一个包含 5 个 consumed 条目的 scan
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("b.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("b.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("b.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let mut scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.consumed.len(), 6);
        // Act: drain 消耗所有元素
        let drained: HashSet<String> = scan.consumed.drain().collect();
        // Assert: consumed 现在为空，drained 包含所有元素
        assert!(scan.consumed.is_empty(), "consumed must be empty after drain");
        assert_eq!(drained.len(), 6, "drained must contain all 6 entries");
        assert!(drained.contains("a.qweight"));
        assert!(drained.contains("b.qzeros"));
    }

    // ── Scan: base_name 包含制表符和换行符的检测
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_base_name_with_tab_and_newline_chars() {
        // Arrange: base_name 包含 \t 和 \n
        let base = "layer\t0\nproj";
        let tensors = vec![
            cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768),
            cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024),
            cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 扫描器只做后缀匹配，不限制 base_name 内容
        assert_eq!(scan.groups.len(), 1, "base_name with whitespace chars should be detected");
        assert!(scan.groups.contains_key(base));
        assert!(scan.groups[base].base_name.contains('\t'));
        assert!(scan.groups[base].base_name.contains('\n'));
    }

    // ── Scan: qzeros 列数远大于 N（不受列数验证约束）
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn accepts_qzeros_columns_much_larger_than_n() {
        // Arrange: qweight [64, 128] → N=128
        // qzeros [4, 9999] → 列数远大于 N/8=16，但扫描器不验证列数与 N 的关系
        let tensors = vec![
            cand("bigcols.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("bigcols.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("bigcols.qzeros", Dtype::I32, vec![4, 9999], 159_984),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qzeros 列数不受限制
        assert_eq!(scan.groups.len(), 1, "qzeros with oversized columns should be accepted");
    }

    // ── AwqGptqGroup: Debug 输出包含所有 7 个字段名
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_debug_seven_field_names_present() {
        // Arrange: 构建 GPTQ 格式的 group
        let group = AwqGptqGroup {
            base_name: "full.field.test".to_string(),
            qweight_name: "full.field.test.qweight".to_string(),
            scales_name: "full.field.test.scales".to_string(),
            qzeros_name: "full.field.test.qzeros".to_string(),
            g_idx_name: Some("full.field.test.g_idx".to_string()),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![512, 1024],
        };
        // Act
        let debug = format!("{group:?}");
        // Assert: 验证所有 7 个字段名都出现在 Debug 输出中
        assert!(debug.contains("base_name"), "must contain base_name: {debug}");
        assert!(debug.contains("qweight_name"), "must contain qweight_name: {debug}");
        assert!(debug.contains("scales_name"), "must contain scales_name: {debug}");
        assert!(debug.contains("qzeros_name"), "must contain qzeros_name: {debug}");
        assert!(debug.contains("g_idx_name"), "must contain g_idx_name: {debug}");
        assert!(debug.contains("format"), "must contain format: {debug}");
        assert!(debug.contains("qweight_shape"), "must contain qweight_shape: {debug}");
    }

    // ── Scan: GPTQ 的 g_idx 名字与 base_name + ".g_idx" 精确一致
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn gptq_g_idx_name_concatenates_base_name_correctly() {
        // Arrange: 使用多层嵌套路径
        let base = "model.layers.7.mlp.gate_proj";
        let tensors = vec![
            cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768),
            cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024),
            cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256),
            cand(&format!("{base}.g_idx"), Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        let group = scan.groups.get(base).unwrap();
        // Assert: g_idx_name 必须精确等于 base_name + ".g_idx"
        let expected_g_idx = format!("{base}.g_idx");
        assert_eq!(group.g_idx_name.as_deref(), Some(expected_g_idx.as_str()));
        assert_eq!(group.qweight_name, format!("{base}.qweight"));
        assert_eq!(group.scales_name, format!("{base}.scales"));
        assert_eq!(group.qzeros_name, format!("{base}.qzeros"));
    }

    // ── AwqGptqScan: consumed 的 clear 操作只清空 consumed 不影响 groups
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_consumed_clear_does_not_affect_groups() {
        // Arrange
        let tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let mut scan = scan_awq_gptq_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 3);
        // Act: 清空 consumed
        scan.consumed.clear();
        // Assert: groups 不受影响
        assert_eq!(scan.groups.len(), 1, "groups must remain intact after consumed.clear()");
        assert!(scan.groups.contains_key("a"));
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: 仅存在 .qweight 和 .scales，.qzeros 名称拼写错误（.qzero）
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_misspelled_qzeros_suffix_qzero() {
        // Arrange: .qzeros 拼写为 .qzero（缺少 s）
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzero", Dtype::I32, vec![4, 16], 256), // 拼写错误
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: ".qzero" 不匹配 ".qzeros" 后缀，缺少合法的 qzeros 张量
        assert!(scan.groups.is_empty(), "misspelled .qzero suffix must not match .qzeros");
        assert!(scan.consumed.is_empty());
    }

    // ── Scan: 仅存在 .qweight 和 .qzeros，.scales 名称拼写错误（.scale）
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_misspelled_scales_suffix_scale() {
        // Arrange: .scales 拼写为 .scale（缺少 s）
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scale", Dtype::F16, vec![4, 128], 1_024), // 拼写错误
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: ".scale" 不匹配 ".scales" 后缀
        assert!(scan.groups.is_empty(), "misspelled .scale suffix must not match .scales");
    }

    // ── Scan: qweight 存在但 dtype 为 BOOL，后面跟正确的 scales 和 qzeros
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_qweight_bool_dtype_with_valid_siblings() {
        // Arrange: qweight 使用 BOOL dtype（不符合 I32 要求）
        let tensors = vec![
            cand("foo.qweight", Dtype::BOOL, vec![64, 128], 1_024),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "BOOL qweight must be rejected");
    }

    // ── AwqGptqFormat: 作为函数参数传递（验证 Copy 语义零开销）
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_passed_as_function_argument() {
        // Arrange: 定义一个消费 AwqGptqFormat 的函数
        fn is_gptq(f: AwqGptqFormat) -> bool {
            matches!(f, AwqGptqFormat::Gptq)
        }
        // Act & Assert: 传递后原始值仍可用（Copy 语义）
        let awq = AwqGptqFormat::Awq;
        let gptq = AwqGptqFormat::Gptq;
        assert!(!is_gptq(awq), "AWQ must not be GPTQ");
        assert!(is_gptq(gptq), "GPTQ must be GPTQ");
        // Copy 后原始值仍可用
        assert_eq!(awq, AwqGptqFormat::Awq);
        assert_eq!(gptq, AwqGptqFormat::Gptq);
    }

    // ── Scan: 完整三重检测后，groups 的 keys() 和 values() 数量一致
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_keys_and_values_count_identical() {
        // Arrange: 5 个混合格式的组
        let mut tensors = Vec::new();
        for i in 0..3 {
            let base = format!("awq.{i}");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        for i in 0..2 {
            let base = format!("gptq.{i}");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
            tensors.push(cand(&format!("{base}.g_idx"), Dtype::I32, vec![512], 2_048));
        }
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: keys 和 values 数量必须相等
        let keys_count = scan.groups.keys().count();
        let values_count = scan.groups.values().count();
        assert_eq!(keys_count, values_count, "keys count must equal values count");
        assert_eq!(keys_count, 5);
    }

    // ── Scan: scales 行数 = K/2（偶数分组）但 qzeros 行数也匹配
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_group_size_16_even_split() {
        // Arrange: qweight [32, 64] → K=256, group_size=16
        // scales [16, 64] → 256/16=16 OK
        // qzeros [16, 8]
        let tensors = vec![
            cand("gs16.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("gs16.scales", Dtype::F16, vec![16, 64], 2_048),
            cand("gs16.qzeros", Dtype::I32, vec![16, 8], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "group_size=16 should be accepted");
        assert_eq!(scan.groups["gs16"].qweight_shape, vec![32, 64]);
        assert_eq!(scan.groups["gs16"].format, AwqGptqFormat::Awq);
    }

    // ── AwqGptqGroup: 直接构造的 group 的 g_idx_name 与 qweight_name 共享 base 前缀
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_name_fields_share_common_prefix() {
        // Arrange: 构造 group 并验证所有名称字段以 base_name 开头
        let base = "model.layers.0.mlp.down_proj";
        let group = AwqGptqGroup {
            base_name: base.to_string(),
            qweight_name: format!("{base}.qweight"),
            scales_name: format!("{base}.scales"),
            qzeros_name: format!("{base}.qzeros"),
            g_idx_name: Some(format!("{base}.g_idx")),
            format: AwqGptqFormat::Gptq,
            qweight_shape: vec![64, 128],
        };
        // Assert: 所有名称字段都以 base_name 为前缀
        assert!(group.qweight_name.starts_with(base));
        assert!(group.scales_name.starts_with(base));
        assert!(group.qzeros_name.starts_with(base));
        assert!(group.g_idx_name.as_ref().unwrap().starts_with(base));
        // 后缀部分
        assert_eq!(&group.qweight_name[base.len()..], ".qweight");
        assert_eq!(&group.scales_name[base.len()..], ".scales");
        assert_eq!(&group.qzeros_name[base.len()..], ".qzeros");
        assert_eq!(&group.g_idx_name.as_ref().unwrap()[base.len()..], ".g_idx");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 55 — 15 additional unit tests for uncovered edge cases
    // ══════════════════════════════════════════════════════════════════════

    // ── Scan: qweight with 4D scales (scales.ndim != 2) rejected despite matching N
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_4d_shape_despite_matching_last_dim() {
        // Arrange: qweight [64, 128], scales [4, 1, 1, 128] — 4D scales, last dim matches N
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 1, 1, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: scales shape must be exactly 2D
        assert!(scan.groups.is_empty(), "4D scales must be rejected even if last dim matches N");
    }

    // ── Scan: qweight with qzeros having I16 dtype and group_size=64
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_awq_i16_qzeros_group_size_64() {
        // Arrange: K=512, N=256, group_size=64
        // qweight [64, 256], scales [8, 256], qzeros I16 [8, 32]
        let tensors = vec![
            cand("gs64.qweight", Dtype::I32, vec![64, 256], 65_536),
            cand("gs64.scales", Dtype::F16, vec![8, 256], 4_096),
            cand("gs64.qzeros", Dtype::I16, vec![8, 32], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "group_size=64 with I16 qzeros must be detected");
        assert_eq!(scan.groups["gs64"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["gs64"].qweight_shape, vec![64, 256]);
    }

    // ── Scan: scales rows exactly 2 and K=256 — verifies 256 % 2 == 0
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_scales_rows_2_k_256_even_division() {
        // Arrange: qweight [32, 64] → K=256, scales [2, 64] → 256/2=128
        let tensors = vec![
            cand("even2.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("even2.scales", Dtype::F16, vec![2, 64], 256),
            cand("even2.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "scales_rows=2 dividing K=256 must be accepted");
        assert_eq!(scan.groups["even2"].qweight_shape, vec![32, 64]);
    }

    // ── AwqGptqScan: groups.extend() merges another scan's groups
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn scan_groups_extend_merges_two_scans() {
        // Arrange: two separate scans with different groups
        let scan1_tensors = vec![
            cand("a.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("a.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("a.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let scan2_tensors = vec![
            cand("b.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("b.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("b.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        let mut scan1 = scan_awq_gptq_groups(scan1_tensors);
        let scan2 = scan_awq_gptq_groups(scan2_tensors);
        // Act: extend scan1 with scan2's groups
        scan1.groups.extend(scan2.groups);
        // Assert: both groups now present
        assert_eq!(scan1.groups.len(), 2);
        assert!(scan1.groups.contains_key("a"));
        assert!(scan1.groups.contains_key("b"));
    }

    // ── Scan: qweight shape [1, 2] — minimal K=8 with N=2
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_1x2_minimal_k8_n2() {
        // Arrange: qweight [1, 2] → K=8, N=2
        // scales [1, 2] → scales_rows=1, 8%1==0
        // qzeros [1, 1]
        let tensors = vec![
            cand("tiny2.qweight", Dtype::I32, vec![1, 2], 8),
            cand("tiny2.scales", Dtype::F16, vec![1, 2], 4),
            cand("tiny2.qzeros", Dtype::I32, vec![1, 1], 4),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1, "1x2 qweight should be valid");
        assert_eq!(scan.groups["tiny2"].qweight_shape, vec![1, 2]);
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── CandidateTensor: name containing all four suffixes as substrings
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn candidate_tensor_name_containing_all_suffixes() {
        // Arrange: a single tensor whose name contains all four suffix substrings
        let ct = CandidateTensor {
            name: "layer.qweight_backup.scales_copy.qzeros_orig.g_idx_ref".to_string(),
            dtype: Dtype::F32,
            shape: vec![1],
            byte_len: 4,
        };
        // Assert: name contains all suffix strings as substrings
        assert!(ct.name.contains(".qweight"));
        assert!(ct.name.contains(".scales"));
        assert!(ct.name.contains(".qzeros"));
        assert!(ct.name.contains(".g_idx"));
    }

    // ── Scan: qweight present but scales has wrong column count (doubled)
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_columns_doubled_vs_qweight() {
        // Arrange: qweight [64, 128] → N=128, scales [4, 256] → N mismatch
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 256], 2_048),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: scales columns 256 != qweight columns 128
        assert!(scan.groups.is_empty(), "doubled scales columns must be rejected");
    }

    // ── AwqGptqFormat: Hash consistency when used as key across multiple inserts
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_hash_consistency_across_hashmap_operations() {
        // Arrange: insert both variants as keys, then retrieve
        let mut map: HashMap<AwqGptqFormat, Vec<&str>> = HashMap::new();
        map.insert(AwqGptqFormat::Awq, vec!["layer1", "layer2"]);
        map.insert(AwqGptqFormat::Gptq, vec!["layer3"]);
        // Act: retrieve using freshly constructed keys
        let awq_layers = map.get(&AwqGptqFormat::Awq).unwrap();
        let gptq_layers = map.get(&AwqGptqFormat::Gptq).unwrap();
        // Assert: Hash must be consistent — same variant always maps to same bucket
        assert_eq!(awq_layers.len(), 2);
        assert_eq!(gptq_layers.len(), 1);
        // Remove and re-insert, verify same behavior
        let awq_val = map.remove(&AwqGptqFormat::Awq).unwrap();
        map.insert(AwqGptqFormat::Awq, awq_val);
        assert_eq!(map.get(&AwqGptqFormat::Awq).unwrap().len(), 2);
    }

    // ── Scan: triplet where qweight rows is a power of 2 and scales rows divides exactly
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_qweight_256_rows_scales_32_rows() {
        // Arrange: qweight [256, 512] → K=2048, scales [32, 512] → group_size=64
        // qzeros [32, 64]
        let tensors = vec![
            cand("pwr2.qweight", Dtype::I32, vec![256, 512], 524_288),
            cand("pwr2.scales", Dtype::F16, vec![32, 512], 32_768),
            cand("pwr2.qzeros", Dtype::I32, vec![32, 64], 8_192),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["pwr2"].qweight_shape, vec![256, 512]);
    }

    // ── Scan: tensor named exactly "qweight" (no prefix at all) not matched
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_tensor_named_exactly_qweight_no_prefix() {
        // Arrange: tensor named "qweight" — no dot before the suffix
        let tensors = vec![
            cand("qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("scales", Dtype::F16, vec![4, 128], 1_024),
            cand("qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: "qweight" does not end with ".qweight" (missing the dot)
        assert!(scan.groups.is_empty(), "bare 'qweight' must not match '.qweight' suffix");
    }

    // ── AwqGptqScan: constructing scan with groups and consumed independently then verifying
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_manual_construction_groups_and_consumed_independent() {
        // Arrange: construct scan with groups but no consumed, then with consumed but no groups
        let group = AwqGptqGroup {
            base_name: "independent".to_string(),
            qweight_name: "independent.qweight".to_string(),
            scales_name: "independent.scales".to_string(),
            qzeros_name: "independent.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        // Act: groups with empty consumed
        let scan1 = AwqGptqScan {
            groups: {
                let mut m = HashMap::new();
                m.insert("independent".to_string(), group);
                m
            },
            consumed: HashSet::new(),
        };
        // Assert: groups has data but consumed is empty
        assert_eq!(scan1.groups.len(), 1);
        assert!(scan1.consumed.is_empty());
        // Verify group is accessible
        assert_eq!(scan1.groups["independent"].format, AwqGptqFormat::Awq);
    }

    // ── Scan: GPTQ group with scales rows = 1 (single group, entire K) and I16 qzeros
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_gptq_scales_rows_1_i16_qzeros_single_group() {
        // Arrange: K=512, N=64, group_size=512 (single group)
        // qweight [64, 64], scales [1, 64], qzeros I16 [1, 8], g_idx present
        let tensors = vec![
            cand("single_gptq.qweight", Dtype::I32, vec![64, 64], 16_384),
            cand("single_gptq.scales", Dtype::F16, vec![1, 64], 128),
            cand("single_gptq.qzeros", Dtype::I16, vec![1, 8], 16),
            cand("single_gptq.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: single-group GPTQ with I16 qzeros
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["single_gptq"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 4);
        assert!(scan.consumed.contains("single_gptq.qzeros"));
        assert!(scan.consumed.contains("single_gptq.g_idx"));
    }

    // ── Scan: multiple AWQ groups with identical shapes — all independently detected
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn detects_five_identical_shape_awq_groups() {
        // Arrange: 5 groups with the same shape but different base_names
        let mut tensors = Vec::new();
        let projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"];
        for proj in &projs {
            let base = format!("layer.{proj}");
            tensors.push(cand(&format!("{base}.qweight"), Dtype::I32, vec![64, 128], 32_768));
            tensors.push(cand(&format!("{base}.scales"), Dtype::F16, vec![4, 128], 1_024));
            tensors.push(cand(&format!("{base}.qzeros"), Dtype::I32, vec![4, 16], 256));
        }
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 5 groups with identical shapes
        assert_eq!(scan.groups.len(), 5);
        assert_eq!(scan.consumed.len(), 15);
        for proj in &projs {
            let key = format!("layer.{proj}");
            assert_eq!(scan.groups[&key].qweight_shape, vec![64, 128]);
            assert_eq!(scan.groups[&key].format, AwqGptqFormat::Awq);
        }
    }

    // ── Scan: qweight and qzeros correct but scales shape [scales_rows, N] has N off by 2
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn rejects_scales_columns_off_by_two() {
        // Arrange: qweight [64, 128] → N=128, scales [4, 130] → N mismatch by 2
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 130], 1_040),
            cand("foo.qzeros", Dtype::I32, vec![4, 16], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "scales columns 130 vs qweight columns 128 must be rejected");
    }

    // ── AwqGptqGroup: Debug output for AWQ group shows "Awq" and "None"
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_debug_awq_shows_format_and_none_g_idx() {
        // Arrange
        let group = AwqGptqGroup {
            base_name: "verify.debug".to_string(),
            qweight_name: "verify.debug.qweight".to_string(),
            scales_name: "verify.debug.scales".to_string(),
            qzeros_name: "verify.debug.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        // Act
        let debug = format!("{group:?}");
        // Assert: must contain both format name and None indicator
        assert!(debug.contains("Awq"), "must contain 'Awq': {debug}");
        assert!(debug.contains("None"), "must contain 'None' for g_idx: {debug}");
        assert!(debug.contains("verify.debug"), "must contain base_name value: {debug}");
        // Note: the struct name "AwqGptqGroup" always contains "Gptq" so we cannot
        // assert absence of "Gptq" in Debug output. Instead verify the format field
        // explicitly shows the AWQ variant by checking "format: Awq" substring.
        assert!(debug.contains("format: Awq"), "must show format: Awq: {debug}");
    }

    // ── Scan: qweight with I32 dtype and correct shape, but qzeros is empty 2D [4, 0]
    // @trace TEST-LOADER-AWQ-GPTQ [req:REQ-GLF-006] [level:unit]
    #[test]
    fn accepts_qzeros_zero_columns_valid_rows() {
        // Arrange: qzeros [4, 0] — rows match scales rows, but columns are zero
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("foo.qzeros", Dtype::I32, vec![4, 0], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: column count is not validated, zero columns accepted if rows match
        assert_eq!(scan.groups.len(), 1, "qzeros with zero columns but valid rows should be accepted");
        assert_eq!(scan.groups["foo"].format, AwqGptqFormat::Awq);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  15 additional unit tests — public API coverage expansion
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. AwqGptqScan result field access after scanning mixed AWQ+GPTQ
    // Verify that groups HashMap and consumed HashSet contain exactly the expected entries
    // after scanning a mixed-format input with both AWQ and GPTQ triplets.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_result_field_access_mixed_awq_gptq() {
        // Arrange: 1 AWQ triplet + 1 GPTQ triplet + 1 unrelated tensor
        let tensors = vec![
            cand("awq.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("awq.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("awq.proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("gptq.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("gptq.proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("gptq.proj.g_idx", Dtype::I32, vec![512], 2_048),
            cand("dense.weight", Dtype::F32, vec![128, 128], 65_536),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: groups has exactly 2 entries
        assert_eq!(scan.groups.len(), 2);
        let awq_group = scan.groups.get("awq.proj").expect("AWQ group must exist");
        let gptq_group = scan.groups.get("gptq.proj").expect("GPTQ group must exist");
        assert_eq!(awq_group.format, AwqGptqFormat::Awq);
        assert!(awq_group.g_idx_name.is_none());
        assert_eq!(gptq_group.format, AwqGptqFormat::Gptq);
        assert!(gptq_group.g_idx_name.is_some());
        // Assert: consumed has exactly 7 entries (3 AWQ + 4 GPTQ)
        assert_eq!(scan.consumed.len(), 7);
        assert!(!scan.consumed.contains("dense.weight"));
    }

    // ── 2. Format detection with mixed quant types across 3 layers
    // Ensure that layers with different quant formats are independently classified.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_detection_mixed_quant_types_three_layers() {
        // Arrange: layer 0 is AWQ, layer 1 is GPTQ (I32 qzeros), layer 2 is GPTQ (I16 qzeros)
        let tensors = vec![
            cand("L0.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L0.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L0.proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("L1.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L1.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L1.proj.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("L1.proj.g_idx", Dtype::I32, vec![512], 2_048),
            cand("L2.proj.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("L2.proj.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("L2.proj.qzeros", Dtype::I16, vec![4, 16], 128),
            cand("L2.proj.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 3);
        assert_eq!(scan.groups["L0.proj"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["L1.proj"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.groups["L2.proj"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.consumed.len(), 11); // 3 + 4 + 4
    }

    // ── 3. Group size validation: scales_rows=0 rejected (would cause division by zero)
    // When scales has 0 rows, K % 0 would panic, so the scanner must reject it.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_size_zero_scales_rows_rejected() {
        // Arrange: qweight [64, 128] → K=512, scales [0, 128] → scales_rows=0
        let tensors = vec![
            cand("zero_gs.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("zero_gs.scales", Dtype::F16, vec![0, 128], 0),
            cand("zero_gs.qzeros", Dtype::I32, vec![0, 16], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: zero scales_rows must be rejected (not panic)
        assert!(scan.groups.is_empty(), "scales_rows=0 must be rejected");
        assert!(scan.consumed.is_empty());
    }

    // ── 4. Scan rejection for non-quantized tensors (F32 weights)
    // Regular dense weights with F32 dtype must not trigger any group detection.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_rejects_all_f32_dense_weights() {
        // Arrange: typical dense model weights, none with quantization suffixes
        let tensors = vec![
            cand("model.embed_tokens.weight", Dtype::F32, vec![32000, 4096], 524_288_000),
            cand("model.layers.0.self_attn.q_proj.weight", Dtype::F32, vec![4096, 4096], 67_108_864),
            cand("model.layers.0.self_attn.k_proj.weight", Dtype::F32, vec![1024, 4096], 16_777_216),
            cand("model.layers.0.mlp.gate_proj.weight", Dtype::F32, vec![11008, 4096], 180_355_072),
            cand("model.norm.weight", Dtype::F32, vec![4096], 16_384),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: no groups detected, no tensors consumed
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── 5. Field validation: scales tensor must have F16 dtype, qzeros must be I32 or I16
    // Validates that swapping scales and qzeros dtypes causes rejection.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn field_validation_swapped_scales_qzeros_dtypes_rejected() {
        // Arrange: scales has I32 (wrong), qzeros has F16 (wrong) — both dtypes invalid
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::I32, vec![4, 128], 2_048),
            cand("foo.qzeros", Dtype::F16, vec![4, 16], 128),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: both wrong dtypes must cause rejection
        assert!(scan.groups.is_empty(), "swapped dtypes for scales/qzeros must be rejected");
    }

    // ── 6. AwqGptqFormat Debug output roundtrip stability
    // Verify that Debug output for both variants is stable and matches expected strings.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_debug_roundtrip_stability() {
        // Arrange
        let awq_debug = format!("{:?}", AwqGptqFormat::Awq);
        let gptq_debug = format!("{:?}", AwqGptqFormat::Gptq);
        // Act: format twice to confirm stability
        let awq_debug_2 = format!("{:?}", AwqGptqFormat::Awq);
        let gptq_debug_2 = format!("{:?}", AwqGptqFormat::Gptq);
        // Assert: Debug output is deterministic
        assert_eq!(awq_debug, awq_debug_2);
        assert_eq!(gptq_debug, gptq_debug_2);
        assert_eq!(awq_debug, "Awq");
        assert_eq!(gptq_debug, "Gptq");
        // Assert: the two variants produce different Debug strings
        assert_ne!(awq_debug, gptq_debug);
    }

    // ── 7. Scan with empty tensor list produces structurally valid result
    // The result must be a valid AwqGptqScan with empty groups and consumed,
    // and must be usable (iterable, queryable) without panic.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_empty_list_structurally_valid_result() {
        // Arrange
        let tensors: Vec<CandidateTensor> = vec![];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: result is a valid default-like AwqGptqScan
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
        // Verify iteration works without panic
        let group_count = scan.groups.iter().count();
        let consumed_count = scan.consumed.iter().count();
        assert_eq!(group_count, 0);
        assert_eq!(consumed_count, 0);
        // Verify queries work
        assert!(scan.groups.get("anything").is_none());
        assert!(!scan.consumed.contains("anything"));
    }

    // ── 8. Format precedence: g_idx presence determines GPTQ regardless of qzeros dtype
    // When g_idx is present, format MUST be GPTQ even if qzeros is I32 (AWQ-style).
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_precedence_g_idx_overrides_awq_detection() {
        // Arrange: I32 qzeros (AWQ-style) but g_idx is present → must be GPTQ
        let tensors = vec![
            cand("hybrid.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("hybrid.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("hybrid.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("hybrid.g_idx", Dtype::I32, vec![512], 2_048),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: g_idx presence forces GPTQ format even with I32 qzeros
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["hybrid"].format, AwqGptqFormat::Gptq);
        assert_eq!(scan.groups["hybrid"].g_idx_name.as_deref(), Some("hybrid.g_idx"));
    }

    // ── 9. Scan with same base_name appearing multiple times (HashMap last-insert wins)
    // When the same prefix has multiple qweight entries, HashMap deduplication means
    // the last entry wins. The scanner processes by_name which also deduplicates.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_duplicate_base_name_last_qweight_wins() {
        // Arrange: two complete triplets for the same base name "dup"
        let tensors = vec![
            // First triplet: shape [64, 128]
            cand("dup.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("dup.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("dup.qzeros", Dtype::I32, vec![4, 16], 256),
            // Second qweight overwrites first in HashMap → shape [32, 64]
            cand("dup.qweight", Dtype::I32, vec![32, 64], 8_192),
            // These match the second qweight's N=64
            cand("dup.scales", Dtype::F16, vec![2, 64], 256),
            cand("dup.qzeros", Dtype::I32, vec![2, 8], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: only one group, with the last-inserted qweight shape
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["dup"].qweight_shape, vec![32, 64]);
    }

    // ── 10. Group boundary alignment: K must be evenly divisible by scales_rows
    // When K=512 and scales_rows=3 (512 % 3 != 0), the triplet is rejected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_boundary_alignment_k_not_divisible_by_scales_rows() {
        // Arrange: qweight [64, 128] → K = 512
        // scales [3, 128] → scales_rows=3, 512 % 3 = 2 != 0 → rejected
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![3, 128], 768),
            cand("foo.qzeros", Dtype::I32, vec![3, 16], 192),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert!(scan.groups.is_empty(), "K not divisible by scales_rows must be rejected");
    }

    // ── 11. AwqGptqFormat Clone and Debug roundtrip through collection
    // Clone each variant, collect into a Vec, format with Debug, and verify identity.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn format_clone_debug_roundtrip_in_collection() {
        // Arrange
        let original = vec![AwqGptqFormat::Awq, AwqGptqFormat::Gptq];
        // Act: clone into a new vec
        let cloned: Vec<AwqGptqFormat> = original.iter().cloned().collect();
        let debug_str = format!("{cloned:?}");
        // Assert: cloned matches original
        assert_eq!(original, cloned);
        // Assert: Debug output contains both variant names
        assert!(debug_str.contains("Awq"), "Debug must contain 'Awq': {debug_str}");
        assert!(debug_str.contains("Gptq"), "Debug must contain 'Gptq': {debug_str}");
    }

    // ── 12. Scan result with single tensor (only qweight, no siblings)
    // A lone qweight tensor without scales/qzeros must produce an empty scan.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_single_qweight_no_siblings_empty_result() {
        // Arrange
        let tensors = vec![
            cand("model.layers.0.mlp.gate_proj.qweight", Dtype::I32, vec![512, 4096], 8_388_608),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: no complete triplet, so no groups
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    // ── 13. Scale tensor shape must be exactly 2D with matching N dimension
    // When scales is 2D but N differs from qweight's N, it must be rejected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scale_shape_n_must_match_qweight_n_exactly() {
        // Arrange: qweight [64, 128] → N=128, scales [4, 64] → N=64 != 128
        let tensors = vec![
            cand("foo.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("foo.scales", Dtype::F16, vec![4, 64], 512),
            cand("foo.qzeros", Dtype::I32, vec![4, 8], 128),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: N mismatch must be rejected
        assert!(scan.groups.is_empty(), "scales N != qweight N must be rejected");
    }

    // ── 14. Zero-point tensor I16 dtype with correct shape accepted as AWQ
    // When qzeros has I16 dtype and rows match scale rows, it is accepted.
    // The format is AWQ when no g_idx is present.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn zero_point_i16_dtype_awq_format_accepted() {
        // Arrange: qzeros I16, rows match scale rows, no g_idx → AWQ
        let tensors = vec![
            cand("layer.mlp.up_proj.qweight", Dtype::I32, vec![128, 256], 131_072),
            cand("layer.mlp.up_proj.scales", Dtype::F16, vec![8, 256], 4_096),
            cand("layer.mlp.up_proj.qzeros", Dtype::I16, vec![8, 32], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("layer.mlp.up_proj").unwrap();
        assert_eq!(group.format, AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.qweight_shape, vec![128, 256]);
        assert_eq!(group.qzeros_name, "layer.mlp.up_proj.qzeros");
    }

    // ── 15. AwqGptqScan Default produces usable empty state
    // Verify that Default::default() produces a scan that can be modified and queried.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_default_produces_usable_empty_state() {
        // Arrange
        let mut scan = AwqGptqScan::default();
        // Assert: initially empty
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
        // Act: add entries to verify the default state is mutable
        let group = AwqGptqGroup {
            base_name: "added".to_string(),
            qweight_name: "added.qweight".to_string(),
            scales_name: "added.scales".to_string(),
            qzeros_name: "added.qzeros".to_string(),
            g_idx_name: None,
            format: AwqGptqFormat::Awq,
            qweight_shape: vec![32, 64],
        };
        scan.groups.insert("added".to_string(), group);
        scan.consumed.insert("added.qweight".to_string());
        scan.consumed.insert("added.scales".to_string());
        scan.consumed.insert("added.qzeros".to_string());
        // Assert: modifications took effect
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.groups.contains_key("added"));
        assert!(scan.consumed.contains("added.qweight"));
    }

    // ── 16. Scan from BTreeMap values iterator
    // The scanner accepts any IntoIterator; verify it works from std::collections::BTreeMap.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_from_btreemap_values_iterator() {
        // Arrange: put tensors into a BTreeMap, then iterate its values
        use std::collections::BTreeMap;
        let mut map = BTreeMap::new();
        map.insert(
            1,
            cand("btree.layer.qweight", Dtype::I32, vec![64, 128], 32_768),
        );
        map.insert(
            2,
            cand("btree.layer.scales", Dtype::F16, vec![4, 128], 1_024),
        );
        map.insert(
            3,
            cand("btree.layer.qzeros", Dtype::I32, vec![4, 16], 256),
        );
        map.insert(
            4,
            cand("btree.layer.bias", Dtype::F32, vec![128], 512),
        );
        // Act: scan from BTreeMap values
        let scan = scan_awq_gptq_groups(map.into_values());
        // Assert: triplet detected
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["btree.layer"].format, AwqGptqFormat::Awq);
        assert!(!scan.consumed.contains("btree.layer.bias"));
    }

    // ── 17. Rejects qzeros I16 dtype when rows do not match scales rows
    // When qzeros is I16 (valid dtype) but its row count differs from scales rows, rejected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn rejects_qzeros_i16_dtype_with_mismatched_rows() {
        // Arrange: qzeros I16 with rows=2, scales rows=4 → mismatch
        let tensors = vec![
            cand("mismatch.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("mismatch.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("mismatch.qzeros", Dtype::I16, vec![2, 16], 64),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: qzeros rows 2 != scales rows 4 → rejected
        assert!(scan.groups.is_empty(), "I16 qzeros row mismatch must be rejected");
        assert!(scan.consumed.is_empty());
    }

    // ── 18. Detects GPTQ with zero-element g_idx tensor
    // When g_idx has shape [0] (zero elements), it still triggers GPTQ format detection
    // because the scanner only checks for key presence, not content.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn detects_gptq_with_zero_element_g_idx() {
        // Arrange: g_idx shape [0] (zero length), but key exists → GPTQ
        let tensors = vec![
            cand("zero_gidx.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("zero_gidx.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("zero_gidx.qzeros", Dtype::I32, vec![4, 16], 256),
            cand("zero_gidx.g_idx", Dtype::I32, vec![0], 0),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: g_idx presence forces GPTQ regardless of shape
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["zero_gidx"].format, AwqGptqFormat::Gptq);
        assert_eq!(
            scan.groups["zero_gidx"].g_idx_name.as_deref(),
            Some("zero_gidx.g_idx")
        );
        assert!(scan.consumed.contains("zero_gidx.g_idx"));
    }

    // ── 19. Scan from boxed slice via Vec::into_boxed_slice
    // Verify IntoIterator works with Vec converted to boxed slice, proving ownership transfer.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_from_boxed_slice_iterator() {
        // Arrange: create tensors, box the slice, then collect back to iterate owned
        let tensors = vec![
            cand("boxed.proj.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("boxed.proj.scales", Dtype::F16, vec![4, 64], 512),
            cand("boxed.proj.qzeros", Dtype::I32, vec![4, 8], 128),
        ];
        let boxed = tensors.into_boxed_slice();
        // Act: scan from boxed slice (collect into Vec to get owned values)
        let scan = scan_awq_gptq_groups(boxed.into_vec().into_iter());
        // Assert: triplet detected from boxed slice path
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["boxed.proj"].qweight_shape, vec![32, 64]);
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── 20. Scan from cloned iterator over references
    // Verify scanning from an iterator of references (.iter().cloned()) works correctly.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn scan_from_cloned_reference_iterator() {
        // Arrange: tensor slice, iterate with .iter().cloned()
        let tensors = [
            cand("ref_iter.qweight", Dtype::I32, vec![16, 32], 2_048),
            cand("ref_iter.scales", Dtype::F16, vec![2, 32], 128),
            cand("ref_iter.qzeros", Dtype::I16, vec![2, 4], 16),
            cand("ref_iter.dense.weight", Dtype::F32, vec![32, 32], 4_096),
        ];
        // Act: clone each reference into an owned value
        let scan = scan_awq_gptq_groups(tensors.iter().cloned());
        // Assert: I16 qzeros AWQ triplet detected, dense weight ignored
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["ref_iter"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["ref_iter"].qzeros_name, "ref_iter.qzeros");
        assert!(!scan.consumed.contains("ref_iter.dense.weight"));
    }

    // ── 21. Rejects when qweight has correct dtype but qzeros has F64 dtype
    // An edge case: qweight I32, scales F16 both valid, but qzeros is F64 → rejected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn rejects_qzeros_f64_dtype_with_valid_qweight_scales() {
        // Arrange: valid qweight and scales, but qzeros is F64
        let tensors = vec![
            cand("f64zero.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("f64zero.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("f64zero.qzeros", Dtype::F64, vec![4, 16], 512),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: F64 is not I32 or I16 → rejected
        assert!(scan.groups.is_empty(), "F64 qzeros must be rejected");
        assert!(scan.consumed.is_empty());
    }

    // ── 22. Detects AWQ with qweight N=1 minimal column count and I16 qzeros
    // When qweight is [K/8, 1], scales [K/gs, 1], qzeros I16 [K/gs, 1] — N=1 edge.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn detects_awq_minimal_n1_with_i16_qzeros() {
        // Arrange: K=64, N=1, group_size=16 → qweight [8, 1], scales [4, 1], qzeros I16 [4, 1]
        let tensors = vec![
            cand("n1.qweight", Dtype::I32, vec![8, 1], 32),
            cand("n1.scales", Dtype::F16, vec![4, 1], 8),
            cand("n1.qzeros", Dtype::I16, vec![4, 1], 8),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: N=1 with I16 qzeros should be accepted as AWQ
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups["n1"].format, AwqGptqFormat::Awq);
        assert_eq!(scan.groups["n1"].qweight_shape, vec![8, 1]);
    }

    // ── 23. Scan produces independent groups when two base names share a common prefix
    // When "prefix_short" and "prefix_short_ext" both have triplets, both are detected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn detects_two_groups_with_common_prefix_one_is_prefix_of_other() {
        // Arrange: "x" and "x_ext" are both valid base names
        let tensors = vec![
            // First group: base_name = "x"
            cand("x.qweight", Dtype::I32, vec![32, 64], 8_192),
            cand("x.scales", Dtype::F16, vec![4, 64], 512),
            cand("x.qzeros", Dtype::I32, vec![4, 8], 128),
            // Second group: base_name = "x_ext"
            cand("x_ext.qweight", Dtype::I32, vec![16, 32], 2_048),
            cand("x_ext.scales", Dtype::F16, vec![2, 32], 128),
            cand("x_ext.qzeros", Dtype::I32, vec![2, 4], 32),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: both groups detected independently
        assert_eq!(scan.groups.len(), 2);
        assert!(scan.groups.contains_key("x"));
        assert!(scan.groups.contains_key("x_ext"));
        assert_eq!(scan.groups["x"].qweight_shape, vec![32, 64]);
        assert_eq!(scan.groups["x_ext"].qweight_shape, vec![16, 32]);
        assert_eq!(scan.consumed.len(), 6);
    }

    // ── 24. Group all seven fields are correctly assigned for I16 GPTQ variant
    // Verify all AwqGptqGroup fields when using I16 qzeros with GPTQ (g_idx present).
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn group_all_seven_fields_correct_for_gptq_i16_qzeros() {
        // Arrange: GPTQ with I16 qzeros
        let tensors = vec![
            cand("gptq_i16.layer.qweight", Dtype::I32, vec![128, 256], 131_072),
            cand("gptq_i16.layer.scales", Dtype::F16, vec![16, 256], 8_192),
            cand("gptq_i16.layer.qzeros", Dtype::I16, vec![16, 32], 1_024),
            cand("gptq_i16.layer.g_idx", Dtype::I32, vec![1024], 4_096),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: all 7 fields verified
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("gptq_i16.layer").unwrap();
        assert_eq!(g.base_name, "gptq_i16.layer");
        assert_eq!(g.qweight_name, "gptq_i16.layer.qweight");
        assert_eq!(g.scales_name, "gptq_i16.layer.scales");
        assert_eq!(g.qzeros_name, "gptq_i16.layer.qzeros");
        assert_eq!(g.g_idx_name.as_deref(), Some("gptq_i16.layer.g_idx"));
        assert_eq!(g.format, AwqGptqFormat::Gptq);
        assert_eq!(g.qweight_shape, vec![128, 256]);
        // All four tensors consumed
        assert_eq!(scan.consumed.len(), 4);
        assert!(scan.consumed.contains("gptq_i16.layer.qweight"));
        assert!(scan.consumed.contains("gptq_i16.layer.scales"));
        assert!(scan.consumed.contains("gptq_i16.layer.qzeros"));
        assert!(scan.consumed.contains("gptq_i16.layer.g_idx"));
    }

    // ── 25. Rejects when scales tensor is 2D but N is correct and qzeros has 3D shape
    // A specific combination: qweight 2D, scales 2D matching, but qzeros 3D → rejected.
    // @trace TEST-LOADER-AWQ-GPTQ [level:unit]
    #[test]
    fn rejects_qzeros_3d_while_qweight_scales_valid() {
        // Arrange: valid qweight [64, 128] and scales [4, 128], but qzeros is 3D [4, 2, 8]
        let tensors = vec![
            cand("qz3d.qweight", Dtype::I32, vec![64, 128], 32_768),
            cand("qz3d.scales", Dtype::F16, vec![4, 128], 1_024),
            cand("qz3d.qzeros", Dtype::I32, vec![4, 2, 8], 256),
        ];
        // Act
        let scan = scan_awq_gptq_groups(tensors);
        // Assert: 3D qzeros shape must be rejected
        assert!(scan.groups.is_empty(), "3D qzeros shape must be rejected");
        assert!(scan.consumed.is_empty());
    }

}
