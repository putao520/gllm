//! NVFP4 tensor pairing detection for SafeTensors loaders.
//!
//! HuggingFace NVFP4 models (e.g. NVIDIA TensorRT-Model-Optimizer outputs) store
//! quantized weights as **two safetensors tensors**:
//!
//! - `<prefix>.weight`        — packed E2M1 4-bit values (U8, each byte holds 2 nibbles).
//! - `<prefix>.weight_scale`  — UE4M3 sub-block scales (U8, one byte per 16-element sub-block).
//!
//! An optional global FP32 scale tensor (`<prefix>.weight_scale_2`) may be present;
//! gllm folds it into the block-level descriptor by multiplying into the UE4M3 scales
//! at load time (NVFP4 SPEC §2.2 uses two-level scaling: global × sub-block × E2M1).
//!
//! # Detection rules
//!
//! 1. Match `*.weight` tensors that are U8 dtype.
//! 2. Look up `<base>.weight_scale` — must also be U8 (UE4M3 stored as raw byte).
//! 3. If found → register NvfpGroup with `base = <prefix>`.
//!
//! # Block layout (for JIT consumption)
//!
//! After repacking, each 64-element NVFP4 block matches `BlockNvfp4`:
//!
//! ```text
//! d[4]: 4 bytes UE4M3 sub-block scales (one per 16-element sub-block)
//! qs[32]: 32 bytes packed E2M1 nibbles (2 per byte)
//! → 36 bytes per block
//! ```

use std::collections::{HashMap, HashSet};

use safetensors::Dtype;

const WEIGHT_SUFFIX: &str = ".weight";
const SCALE_SUFFIX: &str = ".weight_scale";
const GLOBAL_SCALE_SUFFIX: &str = ".weight_scale_2";

/// Lightweight description of a candidate tensor used during scan.
#[derive(Debug, Clone)]
pub struct NvfpCandidate {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

/// One logical NVFP4 weight reconstructed from a SafeTensors pair (+ optional global scale).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NvfpGroup {
    pub base_name: String,
    pub weight_name: String,
    pub scale_name: String,
    /// Optional FP32 global scale tensor (multiplied into UE4M3 sub-block scales at repack time).
    pub global_scale_name: Option<String>,
    /// `[N, K/2]` shape of the packed weight tensor.
    pub weight_shape: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct NvfpScan {
    pub groups: HashMap<String, NvfpGroup>,
    /// Names of tensors consumed by detected groups (hidden from regular enumeration).
    pub consumed: HashSet<String>,
}

/// Scan a tensor candidate set for NVFP4 weight+scale pairs.
pub fn scan_nvfp4_groups<I>(tensors: I) -> NvfpScan
where
    I: IntoIterator<Item = NvfpCandidate>,
{
    let by_name: HashMap<String, NvfpCandidate> = tensors
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();

    let mut scan = NvfpScan::default();

    for (name, w_t) in &by_name {
        let Some(base_name) = name.strip_suffix(WEIGHT_SUFFIX) else {
            continue;
        };
        if w_t.dtype != Dtype::U8 {
            continue;
        }

        let scale_name = format!("{base_name}{SCALE_SUFFIX}");
        let Some(scale_t) = by_name.get(&scale_name) else {
            continue;
        };
        if scale_t.dtype != Dtype::U8 {
            continue;
        }

        // weight shape [N, K/2] (each byte = 2 nibbles).
        if w_t.shape.len() != 2 {
            continue;
        }
        let n = w_t.shape[0];
        let k_half = w_t.shape[1];
        let k = k_half * 2;

        // scale shape must be [N, K/16] (one UE4M3 byte per 16-element sub-block).
        let sub_blocks = k / 16;
        if scale_t.shape.len() != 2
            || scale_t.shape[0] != n
            || scale_t.shape[1] != sub_blocks
        {
            continue;
        }

        let global_scale_name = format!("{base_name}{GLOBAL_SCALE_SUFFIX}");
        let global_scale_present = by_name
            .get(&global_scale_name)
            .map(|g| g.dtype == Dtype::F32)
            .unwrap_or(false);

        let group = NvfpGroup {
            base_name: base_name.to_string(),
            weight_name: name.clone(),
            scale_name: scale_name.clone(),
            global_scale_name: if global_scale_present {
                Some(global_scale_name.clone())
            } else {
                None
            },
            weight_shape: w_t.shape.clone(),
        };

        scan.consumed.insert(name.clone());
        scan.consumed.insert(scale_name);
        if global_scale_present {
            scan.consumed.insert(global_scale_name);
        }
        scan.groups.insert(base_name.to_string(), group);
    }

    scan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, dtype: Dtype, shape: Vec<usize>, byte_len: usize) -> NvfpCandidate {
        NvfpCandidate { name: name.to_string(), dtype, shape, byte_len }
    }

    #[test]
    fn detects_nvfp4_pair_without_global_scale() {
        // Typical NVFP4: weight [N=64, K/2=32], scale [N=64, K/16=4]
        // K=64, N=64, sub_block=16
        let tensors = vec![
            cand("layer.0.mlp.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("layer.0.mlp.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];

        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("layer.0.mlp").expect("group by base_name");
        assert_eq!(group.weight_name, "layer.0.mlp.weight");
        assert_eq!(group.scale_name, "layer.0.mlp.weight_scale");
        assert!(group.global_scale_name.is_none());
        assert!(scan.consumed.contains("layer.0.mlp.weight"));
        assert!(scan.consumed.contains("layer.0.mlp.weight_scale"));
    }

    #[test]
    fn detects_nvfp4_pair_with_global_scale() {
        let tensors = vec![
            cand("attn.qkv.weight", Dtype::U8, vec![128, 64], 128 * 64),
            cand("attn.qkv.weight_scale", Dtype::U8, vec![128, 8], 128 * 8),
            cand("attn.qkv.weight_scale_2", Dtype::F32, vec![1], 4),
        ];

        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let group = scan.groups.get("attn.qkv").expect("group");
        assert_eq!(group.global_scale_name.as_deref(), Some("attn.qkv.weight_scale_2"));
        assert!(scan.consumed.contains("attn.qkv.weight_scale_2"));
    }

    #[test]
    fn rejects_mismatched_scale_shape() {
        // scale shape [N, K/8] is wrong (should be K/16 for NVFP4)
        let tensors = vec![
            cand("bad.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("bad.weight_scale", Dtype::U8, vec![64, 8], 64 * 8),
        ];

        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "mismatched scale shape must not detect");
    }

    #[test]
    fn ignores_non_u8_weight() {
        // F32 weight is not NVFP4 — skip
        let tensors = vec![
            cand("f32.weight", Dtype::F32, vec![64, 64], 64 * 64 * 4),
            cand("f32.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];

        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    #[test]
    fn rejects_non_u8_scale() {
        let tensors = vec![
            cand("x.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("x.weight_scale", Dtype::F32, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "non-U8 scale must not detect");
    }

    #[test]
    fn rejects_1d_weight() {
        let tensors = vec![
            cand("flat.weight", Dtype::U8, vec![64], 64),
            cand("flat.weight_scale", Dtype::U8, vec![4], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "1D weight must not detect");
    }

    #[test]
    fn ignores_weight_without_scale_suffix() {
        let tensors = vec![
            cand("layer.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("layer.bias", Dtype::U8, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    #[test]
    fn detects_multiple_groups() {
        let tensors = vec![
            cand("a.weight", Dtype::U8, vec![32, 16], 32 * 16),
            cand("a.weight_scale", Dtype::U8, vec![32, 2], 32 * 2),
            cand("b.weight", Dtype::U8, vec![16, 8], 16 * 8),
            cand("b.weight_scale", Dtype::U8, vec![16, 1], 16),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 2);
        assert!(scan.groups.contains_key("a"));
        assert!(scan.groups.contains_key("b"));
        assert_eq!(scan.consumed.len(), 4);
    }

    #[test]
    fn nvfp_group_equality() {
        let a = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn nvfp_scan_default_is_empty() {
        let scan = NvfpScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn global_scale_wrong_dtype_ignored() {
        let tensors = vec![
            cand("m.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("m.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            cand("m.weight_scale_2", Dtype::U8, vec![1], 1),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let group = scan.groups.get("m").unwrap();
        assert!(group.global_scale_name.is_none(), "non-F32 global scale must be ignored");
    }

    // ── New tests ──────────────────────────────────────────────────────────

    #[test]
    fn nvfp_candidate_clone_preserves_fields() {
        let original = cand("test.tensor", Dtype::BF16, vec![3, 4], 24);
        let cloned = original.clone();
        assert_eq!(cloned.name, "test.tensor");
        assert_eq!(cloned.dtype, Dtype::BF16);
        assert_eq!(cloned.shape, vec![3, 4]);
        assert_eq!(cloned.byte_len, 24);
    }

    #[test]
    fn nvfp_candidate_debug_format_includes_fields() {
        let c = cand("my.weight", Dtype::U8, vec![2, 3], 6);
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("my.weight"), "Debug must include name");
        assert!(debug_str.contains("U8"), "Debug must include dtype variant");
    }

    #[test]
    fn nvfp_group_inequality_different_base_name() {
        let a = NvfpGroup {
            base_name: "layer.a".into(),
            weight_name: "layer.a.weight".into(),
            scale_name: "layer.a.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "layer.b".into(),
            weight_name: "layer.b.weight".into(),
            scale_name: "layer.b.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn nvfp_group_inequality_different_weight_shape() {
        let a = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![128, 64],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn nvfp_group_inequality_global_scale_presence() {
        let without = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let with = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: Some("x.weight_scale_2".into()),
            weight_shape: vec![64, 32],
        };
        assert_ne!(without, with);
    }

    #[test]
    fn nvfp_group_clone_is_equal() {
        let original = NvfpGroup {
            base_name: "blk.0".into(),
            weight_name: "blk.0.weight".into(),
            scale_name: "blk.0.weight_scale".into(),
            global_scale_name: Some("blk.0.weight_scale_2".into()),
            weight_shape: vec![32, 16],
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn nvfp_group_debug_format_includes_key_fields() {
        let group = NvfpGroup {
            base_name: "model.layer".into(),
            weight_name: "model.layer.weight".into(),
            scale_name: "model.layer.weight_scale".into(),
            global_scale_name: Some("model.layer.weight_scale_2".into()),
            weight_shape: vec![64, 32],
        };
        let debug = format!("{:?}", group);
        assert!(debug.contains("model.layer"), "Debug must contain base_name");
        assert!(debug.contains("weight_scale_2"), "Debug must show global_scale_name");
    }

    #[test]
    fn nvfp_scan_clone_preserves_contents() {
        let mut original = NvfpScan::default();
        original.groups.insert(
            "x".into(),
            NvfpGroup {
                base_name: "x".into(),
                weight_name: "x.weight".into(),
                scale_name: "x.weight_scale".into(),
                global_scale_name: None,
                weight_shape: vec![64, 32],
            },
        );
        original.consumed.insert("x.weight".into());
        original.consumed.insert("x.weight_scale".into());

        let cloned = original.clone();
        assert_eq!(cloned.groups.len(), 1);
        assert!(cloned.groups.contains_key("x"));
        assert_eq!(cloned.consumed.len(), 2);
        assert!(cloned.consumed.contains("x.weight"));
    }

    #[test]
    fn nvfp_scan_debug_format() {
        let scan = NvfpScan::default();
        let debug = format!("{:?}", scan);
        assert!(debug.contains("groups") || debug.contains("NvfpScan"), "Debug must include struct identity");
    }

    #[test]
    fn rejects_3d_weight_tensor() {
        let tensors = vec![
            cand("cube.weight", Dtype::U8, vec![8, 64, 32], 8 * 64 * 32),
            cand("cube.weight_scale", Dtype::U8, vec![8, 64, 4], 8 * 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "3D weight must not form a group");
    }

    #[test]
    fn rejects_scale_with_wrong_n_dimension() {
        // weight [N=64, K/2=32] expects scale [64, K/16=4], but scale has N=32
        let tensors = vec![
            cand("mismatch.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("mismatch.weight_scale", Dtype::U8, vec![32, 4], 32 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "scale N mismatch must be rejected");
    }

    #[test]
    fn rejects_scale_with_wrong_k_dimension() {
        // weight [N=64, K/2=32] → K=64, K/16=4, but scale has dim1=2
        let tensors = vec![
            cand("kbad.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("kbad.weight_scale", Dtype::U8, vec![64, 2], 64 * 2),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "scale K dimension mismatch must be rejected");
    }

    #[test]
    fn empty_input_yields_empty_scan() {
        let scan = scan_nvfp4_groups(vec![]);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn rejects_bf16_weight_dtype() {
        let tensors = vec![
            cand("bf16.weight", Dtype::BF16, vec![64, 32], 64 * 32 * 2),
            cand("bf16.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "BF16 weight must not be detected as NVFP4");
    }

    #[test]
    fn global_scale_f16_dtype_ignored() {
        let tensors = vec![
            cand("f16gs.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("f16gs.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            cand("f16gs.weight_scale_2", Dtype::F16, vec![1], 2),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let group = scan.groups.get("f16gs").unwrap();
        assert!(
            group.global_scale_name.is_none(),
            "F16 global scale must be ignored — only F32 is valid"
        );
        // F16 global scale must not be in consumed set
        assert!(
            !scan.consumed.contains("f16gs.weight_scale_2"),
            "non-F32 global scale must not be consumed"
        );
    }

    #[test]
    fn weight_shape_reflects_packed_layout() {
        // K=128 original features → packed K/2=64 columns in weight
        let tensors = vec![
            cand("shaped.weight", Dtype::U8, vec![16, 64], 16 * 64),
            cand("shaped.weight_scale", Dtype::U8, vec![16, 8], 16 * 8),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let group = scan.groups.get("shaped").unwrap();
        assert_eq!(group.weight_shape, vec![16, 64], "weight_shape must be [N, K/2]");
    }

    #[test]
    fn consumed_set_excludes_non_f32_global_scale() {
        let tensors = vec![
            cand("gs.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("gs.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            cand("gs.weight_scale_2", Dtype::BF16, vec![1], 2),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(!scan.consumed.contains("gs.weight_scale_2"), "BF16 global scale not consumed");
        assert!(scan.consumed.contains("gs.weight"), "weight consumed");
        assert!(scan.consumed.contains("gs.weight_scale"), "scale consumed");
    }

    // ── Struct construction & field access ──────────────────────────────────

    #[test]
    fn nvfp_candidate_field_access() {
        let c = NvfpCandidate {
            name: "layer.0.weight".into(),
            dtype: Dtype::U8,
            shape: vec![128, 64],
            byte_len: 8192,
        };
        assert_eq!(c.name, "layer.0.weight");
        assert_eq!(c.dtype, Dtype::U8);
        assert_eq!(c.shape, vec![128, 64]);
        assert_eq!(c.byte_len, 8192);
    }

    #[test]
    fn nvfp_candidate_with_empty_shape() {
        let c = NvfpCandidate {
            name: "scalar".into(),
            dtype: Dtype::F32,
            shape: vec![],
            byte_len: 0,
        };
        assert!(c.shape.is_empty());
        assert_eq!(c.byte_len, 0);
    }

    #[test]
    fn nvfp_candidate_with_zero_byte_len() {
        let c = NvfpCandidate {
            name: "empty".into(),
            dtype: Dtype::U8,
            shape: vec![0, 0],
            byte_len: 0,
        };
        assert_eq!(c.byte_len, 0);
    }

    #[test]
    fn nvfp_group_with_global_scale_field_access() {
        let g = NvfpGroup {
            base_name: "blk.3.mlp".into(),
            weight_name: "blk.3.mlp.weight".into(),
            scale_name: "blk.3.mlp.weight_scale".into(),
            global_scale_name: Some("blk.3.mlp.weight_scale_2".into()),
            weight_shape: vec![256, 128],
        };
        assert_eq!(g.base_name, "blk.3.mlp");
        assert!(g.global_scale_name.is_some());
        assert_eq!(g.global_scale_name.as_deref(), Some("blk.3.mlp.weight_scale_2"));
        assert_eq!(g.weight_shape, vec![256, 128]);
    }

    #[test]
    fn nvfp_group_without_global_scale_field_access() {
        let g = NvfpGroup {
            base_name: "simple".into(),
            weight_name: "simple.weight".into(),
            scale_name: "simple.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![32, 16],
        };
        assert!(g.global_scale_name.is_none());
    }

    // ── Hash trait ──────────────────────────────────────────────────────────

    #[test]
    fn nvfp_group_hash_equal_for_equal_values() {
        use std::collections::HashSet;
        let a = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let mut set = HashSet::new();
        assert!(set.insert(a.clone()));
        assert!(!set.insert(b), "equal group must not be inserted twice into HashSet");
    }

    #[test]
    fn nvfp_group_hash_different_for_different_values() {
        use std::collections::HashSet;
        let a = NvfpGroup {
            base_name: "alpha".into(),
            weight_name: "alpha.weight".into(),
            scale_name: "alpha.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "beta".into(),
            weight_name: "beta.weight".into(),
            scale_name: "beta.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2, "different groups must both exist in HashSet");
    }

    // ── scan_nvfp4_groups boundary cases ────────────────────────────────────

    #[test]
    fn scan_with_zero_rows() {
        let tensors = vec![
            cand("zero_n.weight", Dtype::U8, vec![0, 32], 0),
            cand("zero_n.weight_scale", Dtype::U8, vec![0, 4], 0),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "N=0 should still form a valid group");
        let g = scan.groups.get("zero_n").unwrap();
        assert_eq!(g.weight_shape, vec![0, 32]);
    }

    #[test]
    fn scan_with_zero_columns() {
        // K/2=0 => K=0, sub_blocks=0, scale shape must be [N, 0]
        let tensors = vec![
            cand("zero_k.weight", Dtype::U8, vec![64, 0], 0),
            cand("zero_k.weight_scale", Dtype::U8, vec![64, 0], 0),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "K=0 should form a valid group");
        let g = scan.groups.get("zero_k").unwrap();
        assert_eq!(g.weight_shape, vec![64, 0]);
    }

    #[test]
    fn scan_rejects_scale_with_1d_shape() {
        let tensors = vec![
            cand("s1d.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("s1d.weight_scale", Dtype::U8, vec![64], 64),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "1D scale shape must not form a group");
    }

    #[test]
    fn scan_rejects_scale_with_3d_shape() {
        let tensors = vec![
            cand("s3d.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("s3d.weight_scale", Dtype::U8, vec![1, 64, 4], 1 * 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "3D scale shape must not form a group");
    }

    #[test]
    fn scan_ignores_tensor_without_weight_suffix() {
        let tensors = vec![
            cand("model.norm", Dtype::U8, vec![64, 32], 64 * 32),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    #[test]
    fn scan_ignores_scale_without_matching_weight() {
        let tensors = vec![
            cand("orphan.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
    }

    #[test]
    fn scan_with_deeply_nested_prefix() {
        let tensors = vec![
            cand("model.layers.12.self_attn.q_proj.weight", Dtype::U8, vec![128, 64], 128 * 64),
            cand("model.layers.12.self_attn.q_proj.weight_scale", Dtype::U8, vec![128, 8], 128 * 8),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("model.layers.12.self_attn.q_proj").unwrap();
        assert_eq!(g.base_name, "model.layers.12.self_attn.q_proj");
    }

    #[test]
    fn scan_last_tensor_wins_when_duplicate_names() {
        // IntoIterator with duplicate keys: HashMap::from_iter keeps the last one
        let tensors = vec![
            cand("dup.weight", Dtype::U8, vec![32, 16], 32 * 16),
            cand("dup.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("dup.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let g = scan.groups.get("dup").unwrap();
        assert_eq!(g.weight_shape, vec![64, 32]);
    }

    #[test]
    fn scan_consumed_includes_all_three_tensors_with_global_scale() {
        let tensors = vec![
            cand("full.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("full.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            cand("full.weight_scale_2", Dtype::F32, vec![1], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.consumed.len(), 3);
        assert!(scan.consumed.contains("full.weight"));
        assert!(scan.consumed.contains("full.weight_scale"));
        assert!(scan.consumed.contains("full.weight_scale_2"));
    }

    #[test]
    fn scan_consumed_exactly_two_without_global_scale() {
        let tensors = vec![
            cand("pair.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("pair.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.consumed.len(), 2);
    }

    #[test]
    fn scan_non_consumed_tensors_remain_outside() {
        let tensors = vec![
            cand("good.weight", Dtype::U8, vec![64, 32], 64 * 32),
            cand("good.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            cand("unrelated.bias", Dtype::F32, vec![64], 256),
            cand("other.weight", Dtype::F32, vec![64, 64], 64 * 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 2);
        assert!(!scan.consumed.contains("unrelated.bias"));
        assert!(!scan.consumed.contains("other.weight"));
    }

    #[test]
    fn scan_with_various_non_u8_weight_dtypes() {
        for dtype in [Dtype::I8, Dtype::I16, Dtype::I32, Dtype::I64, Dtype::F64, Dtype::BOOL] {
            let tensors = vec![
                cand("dt.weight", dtype, vec![64, 32], 64 * 32),
                cand("dt.weight_scale", Dtype::U8, vec![64, 4], 64 * 4),
            ];
            let scan = scan_nvfp4_groups(tensors);
            assert!(scan.groups.is_empty(), "weight dtype {:?} must not form NVFP4 group", dtype);
        }
    }

    #[test]
    fn nvfp_scan_default_and_manual_construction() {
        let scan = NvfpScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());

        let mut manual = NvfpScan {
            groups: HashMap::new(),
            consumed: HashSet::new(),
        };
        manual.consumed.insert("test".into());
        assert_eq!(manual.consumed.len(), 1);
    }

    // ── Additional edge-case & coverage tests ────────────────────────────────

    #[test]
    fn scan_k_half_not_multiple_of_8_still_passes_shape_check() {
        // K/2=3 => K=6, sub_blocks = 6/16 = 0 (integer division truncates to 0).
        // Scale must be [N, 0] to match. This tests the integer division behavior.
        let tensors = vec![
            cand("odd.weight", Dtype::U8, vec![4, 3], 12),
            cand("odd.weight_scale", Dtype::U8, vec![4, 0], 0),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "K=6, sub_blocks=0, scale [4,0] should match");
    }

    #[test]
    fn scan_k_half_not_multiple_of_8_rejects_mismatched_scale() {
        // K/2=3 => K=6, sub_blocks=0, but scale has dim1=1 instead of 0
        let tensors = vec![
            cand("oddbad.weight", Dtype::U8, vec![4, 3], 12),
            cand("oddbad.weight_scale", Dtype::U8, vec![4, 1], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "scale shape mismatch with non-aligned K must reject");
    }

    #[test]
    fn scan_exact_minimum_valid_dimensions() {
        // Smallest valid NVFP4: N=1, K/2=8 => K=16, sub_blocks=1
        let tensors = vec![
            cand("tiny.weight", Dtype::U8, vec![1, 8], 8),
            cand("tiny.weight_scale", Dtype::U8, vec![1, 1], 1),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("tiny").unwrap();
        assert_eq!(g.weight_shape, vec![1, 8]);
        assert_eq!(g.base_name, "tiny");
    }

    #[test]
    fn nvfp_candidate_with_single_element_shape() {
        let c = NvfpCandidate {
            name: "bias".into(),
            dtype: Dtype::F32,
            shape: vec![1],
            byte_len: 4,
        };
        assert_eq!(c.shape, vec![1]);
        // A 1D tensor won't be matched by the scanner (needs 2D weight)
    }

    #[test]
    fn nvfp_group_equality_requires_all_fields_identical() {
        let base = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        // Different scale_name
        let diff_scale = NvfpGroup {
            scale_name: "x.other_scale".into(),
            ..base.clone()
        };
        assert_ne!(base, diff_scale);

        // Different weight_name
        let diff_weight = NvfpGroup {
            weight_name: "x.other_weight".into(),
            ..base.clone()
        };
        assert_ne!(base, diff_weight);
    }

    #[test]
    fn nvfp_group_with_empty_weight_shape() {
        let g = NvfpGroup {
            base_name: "empty".into(),
            weight_name: "empty.weight".into(),
            scale_name: "empty.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![],
        };
        assert!(g.weight_shape.is_empty());
    }

    #[test]
    fn scan_ignores_tensors_named_weight_scale_without_weight() {
        // A tensor named "layer.weight_scale" without a corresponding "layer.weight" U8 tensor
        let tensors = vec![
            cand("layer.weight_scale", Dtype::U8, vec![64, 4], 256),
            cand("layer.weight", Dtype::F32, vec![64, 64], 64 * 64 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "F32 weight must not trigger NVFP4 detection");
    }

    #[test]
    fn scan_group_base_name_excludes_weight_suffix() {
        let tensors = vec![
            cand("transformer.h.3.attn.weight", Dtype::U8, vec![64, 32], 2048),
            cand("transformer.h.3.attn.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let g = scan.groups.get("transformer.h.3.attn").unwrap();
        assert_eq!(g.base_name, "transformer.h.3.attn");
        assert!(!g.base_name.ends_with(".weight"));
    }

    #[test]
    fn scan_with_global_scale_f32_accepts() {
        // Verify that a valid F32 global scale is accepted
        let tensors = vec![
            cand("gs_ok.weight", Dtype::U8, vec![32, 16], 512),
            cand("gs_ok.weight_scale", Dtype::U8, vec![32, 2], 64),
            cand("gs_ok.weight_scale_2", Dtype::F32, vec![4], 16),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let g = scan.groups.get("gs_ok").unwrap();
        assert_eq!(g.global_scale_name.as_deref(), Some("gs_ok.weight_scale_2"));
    }

    #[test]
    fn scan_large_dimensions() {
        // Large model dimensions: N=4096, K/2=2048 => K=4096, sub_blocks=256
        let tensors = vec![
            cand("big.weight", Dtype::U8, vec![4096, 2048], 4096 * 2048),
            cand("big.weight_scale", Dtype::U8, vec![4096, 256], 4096 * 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("big").unwrap();
        assert_eq!(g.weight_shape, vec![4096, 2048]);
    }

    #[test]
    fn scan_mixed_valid_and_invalid_groups() {
        let tensors = vec![
            // Valid group A
            cand("a.weight", Dtype::U8, vec![64, 32], 2048),
            cand("a.weight_scale", Dtype::U8, vec![64, 4], 256),
            // Invalid: F32 weight (no pair)
            cand("b.weight", Dtype::F32, vec![64, 64], 16384),
            cand("b.weight_scale", Dtype::U8, vec![64, 4], 256),
            // Valid group C
            cand("c.weight", Dtype::U8, vec![32, 16], 512),
            cand("c.weight_scale", Dtype::U8, vec![32, 2], 64),
            // Orphan scale (no matching weight)
            cand("d.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 2);
        assert!(scan.groups.contains_key("a"));
        assert!(scan.groups.contains_key("c"));
        assert!(!scan.groups.contains_key("b"));
        assert!(!scan.groups.contains_key("d"));
    }

    #[test]
    fn scan_candidate_with_byte_len_zero_and_valid_shape() {
        // byte_len is not used in detection logic, so even 0 byte_len should form a group
        let tensors = vec![
            cand("zero_byte.weight", Dtype::U8, vec![64, 32], 0),
            cand("zero_byte.weight_scale", Dtype::U8, vec![64, 4], 0),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
    }

    #[test]
    fn nvfp_candidate_name_with_dots_and_underscores() {
        let c = NvfpCandidate {
            name: "model.transformer.layer_0.attn.q_proj.weight".into(),
            dtype: Dtype::U8,
            shape: vec![64, 32],
            byte_len: 2048,
        };
        assert!(c.name.contains('.') && c.name.contains('_'));
    }

    #[test]
    fn scan_weight_suffix_at_end_only() {
        // "x.weight_extra" should NOT match ".weight" suffix
        let tensors = vec![
            cand("x.weight_extra", Dtype::U8, vec![64, 32], 2048),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "suffix must be exactly '.weight'");
    }

    #[test]
    fn scan_global_scale_consumed_only_when_f32() {
        // Two groups: one with valid F32 global scale, one without
        let tensors = vec![
            cand("has_gs.weight", Dtype::U8, vec![64, 32], 2048),
            cand("has_gs.weight_scale", Dtype::U8, vec![64, 4], 256),
            cand("has_gs.weight_scale_2", Dtype::F32, vec![1], 4),
            cand("no_gs.weight", Dtype::U8, vec![64, 32], 2048),
            cand("no_gs.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.consumed.len(), 5); // 2+1 for has_gs + 2 for no_gs
        assert!(scan.consumed.contains("has_gs.weight_scale_2"));
    }

    #[test]
    fn nvfp_scan_debug_includes_groups_and_consumed() {
        let mut scan = NvfpScan::default();
        scan.groups.insert(
            "test".into(),
            NvfpGroup {
                base_name: "test".into(),
                weight_name: "test.weight".into(),
                scale_name: "test.weight_scale".into(),
                global_scale_name: None,
                weight_shape: vec![64, 32],
            },
        );
        scan.consumed.insert("test.weight".into());
        let debug = format!("{:?}", scan);
        assert!(debug.contains("groups"), "Debug output must include 'groups' field");
        assert!(debug.contains("consumed"), "Debug output must include 'consumed' field");
    }

    #[test]
    fn nvfp_group_base_name_matches_hashmap_key() {
        let tensors = vec![
            cand("layer.0.mlp.weight", Dtype::U8, vec![64, 32], 2048),
            cand("layer.0.mlp.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        for (key, group) in &scan.groups {
            assert_eq!(*key, group.base_name, "HashMap key must equal group.base_name");
        }
    }

    // ── Additional coverage tests (wave-15) ──────────────────────────────────

    #[test]
    fn scan_weight_named_exactly_dot_weight_yields_empty_base() {
        // Edge case: tensor named ".weight" → base_name = "" (empty string)
        let tensors = vec![
            cand(".weight", Dtype::U8, vec![64, 32], 2048),
            cand(".weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "empty base_name is still a valid group key");
        let g = scan.groups.get("").expect("group keyed by empty string");
        assert_eq!(g.base_name, "");
        assert_eq!(g.weight_name, ".weight");
        assert_eq!(g.scale_name, ".weight_scale");
    }

    #[test]
    fn scan_single_row_with_standard_k() {
        // N=1, K/2=32 => K=64, sub_blocks=4. Tests minimal row count with full feature dim.
        let tensors = vec![
            cand("row1.weight", Dtype::U8, vec![1, 32], 32),
            cand("row1.weight_scale", Dtype::U8, vec![1, 4], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("row1").unwrap();
        assert_eq!(g.weight_shape, vec![1, 32]);
    }

    #[test]
    fn scan_rejects_scale_with_bf16_dtype() {
        let tensors = vec![
            cand("bf16scale.weight", Dtype::U8, vec![64, 32], 2048),
            cand("bf16scale.weight_scale", Dtype::BF16, vec![64, 4], 512),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "BF16 scale must not form NVFP4 group");
    }

    #[test]
    fn scan_rejects_global_scale_with_various_wrong_dtypes() {
        for (label, dtype) in [
            ("I32", Dtype::I32),
            ("BOOL", Dtype::BOOL),
            ("F64", Dtype::F64),
        ] {
            let tensors = vec![
                cand("gdt.weight", Dtype::U8, vec![64, 32], 2048),
                cand("gdt.weight_scale", Dtype::U8, vec![64, 4], 256),
                cand("gdt.weight_scale_2", dtype, vec![1], 4),
            ];
            let scan = scan_nvfp4_groups(tensors);
            let g = scan.groups.get("gdt").expect("group must exist regardless of global scale dtype");
            assert!(
                g.global_scale_name.is_none(),
                "global scale dtype {} ({:?}) must be ignored",
                label,
                dtype
            );
            assert!(
                !scan.consumed.contains("gdt.weight_scale_2"),
                "non-F32 global scale ({}) must not be consumed",
                label
            );
        }
    }

    #[test]
    fn nvfp_group_with_one_element_weight_shape() {
        let g = NvfpGroup {
            base_name: "tiny".into(),
            weight_name: "tiny.weight".into(),
            scale_name: "tiny.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![1],
        };
        assert_eq!(g.weight_shape.len(), 1);
        assert_eq!(g.weight_shape[0], 1);
    }

    #[test]
    fn scan_two_groups_one_with_global_scale_one_without() {
        let tensors = vec![
            cand("with_gs.weight", Dtype::U8, vec![32, 16], 512),
            cand("with_gs.weight_scale", Dtype::U8, vec![32, 2], 64),
            cand("with_gs.weight_scale_2", Dtype::F32, vec![1], 4),
            cand("no_gs.weight", Dtype::U8, vec![32, 16], 512),
            cand("no_gs.weight_scale", Dtype::U8, vec![32, 2], 64),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 2);

        let with_gs = scan.groups.get("with_gs").expect("with_gs group");
        assert_eq!(
            with_gs.global_scale_name.as_deref(),
            Some("with_gs.weight_scale_2")
        );

        let no_gs = scan.groups.get("no_gs").expect("no_gs group");
        assert!(no_gs.global_scale_name.is_none());

        // Consumed: 3 for with_gs (weight + scale + global_scale) + 2 for no_gs = 5
        assert_eq!(scan.consumed.len(), 5);
        assert!(scan.consumed.contains("with_gs.weight_scale_2"));
    }

    #[test]
    fn nvfp_candidate_with_large_byte_len() {
        // Simulate a large tensor: 4096 x 4096 = 16M elements
        let c = NvfpCandidate {
            name: "huge.weight".into(),
            dtype: Dtype::U8,
            shape: vec![4096, 4096],
            byte_len: 4096 * 4096,
        };
        assert_eq!(c.byte_len, 16_777_216);
        assert_eq!(c.shape, vec![4096, 4096]);
    }

    #[test]
    fn scan_scale_partial_name_match_does_not_trigger() {
        // "proj.weight_scale" should not match "proj_gated.weight" because
        // the scanner looks up "<base>.weight_scale" where base is stripped from ".weight".
        // "proj_gated.weight" → base = "proj_gated" → scale_name = "proj_gated.weight_scale"
        // The existing "proj.weight_scale" is for a different base "proj" which has no weight tensor.
        let tensors = vec![
            cand("proj_gated.weight", Dtype::U8, vec![64, 32], 2048),
            cand("proj.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(
            scan.groups.is_empty(),
            "scale with different base prefix must not match"
        );
    }

    #[test]
    fn scan_only_weight_tensor_without_any_scale() {
        // A lone U8 weight tensor with no scale tensor at all
        let tensors = vec![
            cand("lonely.weight", Dtype::U8, vec![64, 32], 2048),
            cand("lonely.bias", Dtype::F32, vec![64], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "weight without scale must not form group");
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn scan_k_just_large_enough_for_one_sub_block() {
        // K/2=8 => K=16, sub_blocks=16/16=1. Minimum K that produces exactly 1 sub-block.
        let tensors = vec![
            cand("k16.weight", Dtype::U8, vec![4, 8], 32),
            cand("k16.weight_scale", Dtype::U8, vec![4, 1], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("k16").unwrap();
        assert_eq!(g.weight_shape, vec![4, 8]);
    }

    #[test]
    fn scan_k_just_below_one_sub_block_rejects_nonzero_scale() {
        // K/2=7 => K=14, sub_blocks=14/16=0. Scale must be [N, 0].
        // Supplying scale [N, 1] must reject.
        let tensors = vec![
            cand("k14.weight", Dtype::U8, vec![4, 7], 28),
            cand("k14.weight_scale", Dtype::U8, vec![4, 1], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "K=14 sub_blocks=0, scale dim1=1 must reject");
    }

    #[test]
    fn scan_rejects_f16_weight_dtype() {
        let tensors = vec![
            cand("f16w.weight", Dtype::F16, vec![64, 32], 4096),
            cand("f16w.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "F16 weight must not be detected as NVFP4");
    }

    #[test]
    fn scan_consumed_set_tracks_all_grouped_tensors_across_multiple_groups() {
        // g3 needs K >= 16 for valid sub_blocks; use [8, 8] so K=16, sub_blocks=1
        let tensors = vec![
            cand("g1.weight", Dtype::U8, vec![32, 16], 512),
            cand("g1.weight_scale", Dtype::U8, vec![32, 2], 64),
            cand("g2.weight", Dtype::U8, vec![16, 16], 256),
            cand("g2.weight_scale", Dtype::U8, vec![16, 2], 32),
            cand("g3.weight", Dtype::U8, vec![8, 8], 64),
            cand("g3.weight_scale", Dtype::U8, vec![8, 1], 8),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 3);
        assert_eq!(scan.consumed.len(), 6, "3 groups x 2 tensors each = 6 consumed");
        for name in [
            "g1.weight", "g1.weight_scale",
            "g2.weight", "g2.weight_scale",
            "g3.weight", "g3.weight_scale",
        ] {
            assert!(scan.consumed.contains(name), "{} must be in consumed set", name);
        }
    }

    #[test]
    fn nvfp_group_equality_different_global_scale_values() {
        let a = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: Some("x.weight_scale_2".into()),
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "x".into(),
            weight_name: "x.weight".into(),
            scale_name: "x.weight_scale".into(),
            global_scale_name: Some("x.other_global".into()),
            weight_shape: vec![64, 32],
        };
        assert_ne!(a, b, "different global_scale_name values must not be equal");
    }

    #[test]
    fn scan_with_non_u8_various_scale_dtypes_rejection() {
        // Verify that several non-U8 scale dtypes are all rejected
        for (label, dtype) in [
            ("F16", Dtype::F16),
            ("BF16", Dtype::BF16),
            ("I64", Dtype::I64),
            ("I32", Dtype::I32),
            ("F64", Dtype::F64),
        ] {
            let tensors = vec![
                cand("sdtype.weight", Dtype::U8, vec![64, 32], 2048),
                cand("sdtype.weight_scale", dtype, vec![64, 4], 256),
            ];
            let scan = scan_nvfp4_groups(tensors);
            assert!(
                scan.groups.is_empty(),
                "scale dtype {} ({:?}) must be rejected",
                label,
                dtype
            );
        }
    }

    // ── Wave-16 edge-case & boundary tests ──────────────────────────────────────

    #[test]
    fn scan_k_half_equals_1_sub_blocks_zero_requires_empty_scale_dim() {
        // K/2=1 => K=2, sub_blocks=2/16=0. Scale must be [N, 0].
        let tensors = vec![
            cand("k2.weight", Dtype::U8, vec![8, 1], 8),
            cand("k2.weight_scale", Dtype::U8, vec![8, 0], 0),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "K=2, sub_blocks=0, scale [8,0] should form group");
    }

    #[test]
    fn scan_k_half_equals_1_sub_blocks_zero_rejects_nonzero_scale() {
        // K/2=1 => K=2, sub_blocks=0, but scale has dim1=1 → mismatch
        let tensors = vec![
            cand("k2bad.weight", Dtype::U8, vec![8, 1], 8),
            cand("k2bad.weight_scale", Dtype::U8, vec![8, 1], 8),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "K=2, sub_blocks=0, scale [8,1] must reject");
    }

    #[test]
    fn scan_k_32_produces_two_sub_blocks() {
        // K/2=16 => K=32, sub_blocks=32/16=2. Tests power-of-2 K with 2 sub-blocks.
        let tensors = vec![
            cand("k32.weight", Dtype::U8, vec![4, 16], 64),
            cand("k32.weight_scale", Dtype::U8, vec![4, 2], 8),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("k32").unwrap();
        assert_eq!(g.weight_shape, vec![4, 16]);
    }

    #[test]
    fn scan_k_between_sub_block_boundaries_truncates_sub_blocks() {
        // K/2=12 => K=24, sub_blocks=24/16=1 (integer division). Scale must be [N, 1].
        let tensors = vec![
            cand("k24.weight", Dtype::U8, vec![2, 12], 24),
            cand("k24.weight_scale", Dtype::U8, vec![2, 1], 2),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "K=24, sub_blocks=1 via truncation should form group");
    }

    #[test]
    fn scan_scale_dim1_off_by_one_rejected() {
        // Correct scale dim1 for weight [64, 32] (K=64, sub_blocks=4) is 4.
        // Supply scale dim1=3 → must reject.
        let tensors = vec![
            cand("off1.weight", Dtype::U8, vec![64, 32], 2048),
            cand("off1.weight_scale", Dtype::U8, vec![64, 3], 192),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "scale dim1 off by 1 must be rejected");
    }

    #[test]
    fn scan_scale_dim0_correct_dim1_off_by_one_rejected() {
        // Correct scale for weight [64, 32] (K=64) is [64, 4]. Supply [64, 5] → reject.
        let tensors = vec![
            cand("off1hi.weight", Dtype::U8, vec![64, 32], 2048),
            cand("off1hi.weight_scale", Dtype::U8, vec![64, 5], 320),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty(), "scale dim1=5 (expected 4) must be rejected");
    }

    #[test]
    fn scan_global_scale_with_multi_element_shape_accepted() {
        // Global scale tensor shape is not validated — any F32 shape is accepted.
        let tensors = vec![
            cand("multigs.weight", Dtype::U8, vec![32, 16], 512),
            cand("multigs.weight_scale", Dtype::U8, vec![32, 2], 64),
            cand("multigs.weight_scale_2", Dtype::F32, vec![2, 3], 24),
        ];
        let scan = scan_nvfp4_groups(tensors);
        let g = scan.groups.get("multigs").expect("group must exist");
        assert_eq!(
            g.global_scale_name.as_deref(),
            Some("multigs.weight_scale_2"),
            "multi-element F32 global scale must be accepted"
        );
        assert!(scan.consumed.contains("multigs.weight_scale_2"));
    }

    #[test]
    fn scan_bias_tensor_not_consumed_alongside_valid_group() {
        let tensors = vec![
            cand("blk.weight", Dtype::U8, vec![64, 32], 2048),
            cand("blk.weight_scale", Dtype::U8, vec![64, 4], 256),
            cand("blk.bias", Dtype::F32, vec![64], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.consumed.contains("blk.weight"));
        assert!(scan.consumed.contains("blk.weight_scale"));
        assert!(!scan.consumed.contains("blk.bias"), "bias must not be in consumed set");
    }

    #[test]
    fn scan_three_groups_all_with_global_scale() {
        let tensors = vec![
            cand("a.weight", Dtype::U8, vec![16, 8], 128),
            cand("a.weight_scale", Dtype::U8, vec![16, 1], 16),
            cand("a.weight_scale_2", Dtype::F32, vec![1], 4),
            cand("b.weight", Dtype::U8, vec![8, 8], 64),
            cand("b.weight_scale", Dtype::U8, vec![8, 1], 8),
            cand("b.weight_scale_2", Dtype::F32, vec![1], 4),
            cand("c.weight", Dtype::U8, vec![4, 8], 32),
            cand("c.weight_scale", Dtype::U8, vec![4, 1], 4),
            cand("c.weight_scale_2", Dtype::F32, vec![1], 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 3);
        // 3 groups × 3 tensors each = 9 consumed
        assert_eq!(scan.consumed.len(), 9);
        for prefix in ["a", "b", "c"] {
            let g = scan.groups.get(prefix).expect("group must exist");
            assert!(g.global_scale_name.is_some(), "{} must have global scale", prefix);
        }
    }

    #[test]
    fn nvfp_group_equality_ignores_nothing_all_fields_matter() {
        // Verify that PartialEq checks weight_name, not just base_name
        let a = NvfpGroup {
            base_name: "same".into(),
            weight_name: "same.weight".into(),
            scale_name: "same.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = NvfpGroup {
            base_name: "same".into(),
            weight_name: "same.weight_alt".into(),
            scale_name: "same.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        assert_ne!(a, b, "different weight_name must produce inequality");
    }

    #[test]
    fn scan_k_near_sub_block_boundary_15_elements() {
        // K/2=8..15 range: K/2=15 => K=30, sub_blocks=30/16=1. Scale must be [N, 1].
        let tensors = vec![
            cand("k30.weight", Dtype::U8, vec![2, 15], 30),
            cand("k30.weight_scale", Dtype::U8, vec![2, 1], 2),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1, "K=30, sub_blocks=1 should form group");
    }

    #[test]
    fn scan_k_near_sub_block_boundary_31_elements() {
        // K/2=16 => K=32, sub_blocks=2. K/2=15 => K=30, sub_blocks=1.
        // At the boundary: K=31 is impossible since K/2 must be integer (weight shape is [N, K/2]).
        // So test K=48: K/2=24, sub_blocks=48/16=3.
        let tensors = vec![
            cand("k48.weight", Dtype::U8, vec![2, 24], 48),
            cand("k48.weight_scale", Dtype::U8, vec![2, 3], 6),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("k48").unwrap();
        assert_eq!(g.weight_shape, vec![2, 24]);
    }

    #[test]
    fn nvfp_candidate_independent_mutation() {
        // Cloning a candidate and mutating the clone must not affect the original.
        let original = cand("orig.weight", Dtype::U8, vec![64, 32], 2048);
        let mut cloned = original.clone();
        cloned.name = "mutated.weight".to_string();
        cloned.byte_len = 0;
        assert_eq!(original.name, "orig.weight", "original name must be unchanged");
        assert_eq!(original.byte_len, 2048, "original byte_len must be unchanged");
        assert_eq!(cloned.name, "mutated.weight");
        assert_eq!(cloned.byte_len, 0);
    }

    // ── Wave-17 additional coverage tests ──────────────────────────────────────

    #[test]
    fn suffix_constants_are_stable() {
        // Verify the constant values used for detection logic
        assert_eq!(WEIGHT_SUFFIX, ".weight");
        assert_eq!(SCALE_SUFFIX, ".weight_scale");
        assert_eq!(GLOBAL_SCALE_SUFFIX, ".weight_scale_2");
    }

    #[test]
    fn nvfp_group_partialeq_symmetry() {
        // If a == b then b == a (symmetry property of PartialEq)
        let a = NvfpGroup {
            base_name: "sym".into(),
            weight_name: "sym.weight".into(),
            scale_name: "sym.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![64, 32],
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, a, "PartialEq must be symmetric");
    }

    #[test]
    fn nvfp_group_partialeq_transitivity() {
        // If a == b and b == c then a == c
        let mk = || NvfpGroup {
            base_name: "trans".into(),
            weight_name: "trans.weight".into(),
            scale_name: "trans.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![32, 16],
        };
        let a = mk();
        let b = mk();
        let c = mk();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "PartialEq must be transitive");
    }

    #[test]
    fn nvfp_scan_clone_is_independent() {
        // Mutating a cloned NvfpScan must not affect the original
        let mut original = NvfpScan::default();
        original.groups.insert(
            "orig".into(),
            NvfpGroup {
                base_name: "orig".into(),
                weight_name: "orig.weight".into(),
                scale_name: "orig.weight_scale".into(),
                global_scale_name: None,
                weight_shape: vec![64, 32],
            },
        );
        original.consumed.insert("orig.weight".into());

        let mut cloned = original.clone();
        cloned.groups.remove("orig");
        cloned.consumed.insert("extra".into());

        assert_eq!(original.groups.len(), 1, "original groups must be unchanged");
        assert_eq!(original.consumed.len(), 1, "original consumed must be unchanged");
        assert!(cloned.groups.is_empty());
        assert_eq!(cloned.consumed.len(), 2);
    }

    #[test]
    fn nvfp_candidate_debug_includes_all_fields() {
        let c = NvfpCandidate {
            name: "test.field".into(),
            dtype: Dtype::BF16,
            shape: vec![2, 3],
            byte_len: 12,
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("test.field"), "Debug must include name");
        assert!(debug.contains("BF16"), "Debug must include dtype");
        assert!(debug.contains("shape"), "Debug must include shape field name");
        assert!(debug.contains("byte_len"), "Debug must include byte_len field name");
    }

    #[test]
    fn nvfp_group_debug_includes_all_fields() {
        let g = NvfpGroup {
            base_name: "dbg".into(),
            weight_name: "dbg.weight".into(),
            scale_name: "dbg.weight_scale".into(),
            global_scale_name: Some("dbg.weight_scale_2".into()),
            weight_shape: vec![32, 16],
        };
        let debug = format!("{:?}", g);
        assert!(debug.contains("base_name"), "Debug must show base_name field");
        assert!(debug.contains("weight_name"), "Debug must show weight_name field");
        assert!(debug.contains("scale_name"), "Debug must show scale_name field");
        assert!(debug.contains("global_scale_name"), "Debug must show global_scale_name field");
        assert!(debug.contains("weight_shape"), "Debug must show weight_shape field");
    }

    #[test]
    fn scan_consumed_excludes_tensors_from_rejected_groups() {
        // A group with wrong scale shape should NOT add any tensors to consumed
        let tensors = vec![
            cand("bad.weight", Dtype::U8, vec![64, 32], 2048),
            cand("bad.weight_scale", Dtype::U8, vec![32, 4], 128), // wrong N
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(
            scan.consumed.is_empty(),
            "rejected group must not add any tensor to consumed"
        );
    }

    #[test]
    fn scan_with_only_scale_tensor_no_weight() {
        // An orphan scale tensor (U8, correct 2D shape) but no matching weight
        let tensors = vec![
            cand("orphan.weight_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn scan_valid_pair_plus_unrelated_f32_weight() {
        // A valid NVFP4 pair alongside an unrelated F32 weight tensor
        let tensors = vec![
            cand("nvfp.weight", Dtype::U8, vec![64, 32], 2048),
            cand("nvfp.weight_scale", Dtype::U8, vec![64, 4], 256),
            cand("dense.weight", Dtype::F32, vec![128, 256], 128 * 256 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.groups.contains_key("nvfp"));
        assert!(!scan.consumed.contains("dense.weight"));
    }

    #[test]
    fn nvfp_group_with_three_element_weight_shape() {
        // weight_shape can legally hold any vec (struct doesn't enforce 2D)
        let g = NvfpGroup {
            base_name: "conv".into(),
            weight_name: "conv.weight".into(),
            scale_name: "conv.weight_scale".into(),
            global_scale_name: None,
            weight_shape: vec![3, 64, 32],
        };
        assert_eq!(g.weight_shape.len(), 3);
        assert_eq!(g.weight_shape[0], 3);
    }

    #[test]
    fn scan_k64_standard_sub_block_count() {
        // K=64 is the canonical NVFP4 block size: sub_blocks = 64/16 = 4
        let tensors = vec![
            cand("stdk.weight", Dtype::U8, vec![128, 32], 128 * 32),
            cand("stdk.weight_scale", Dtype::U8, vec![128, 4], 128 * 4),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert_eq!(scan.groups.len(), 1);
        let g = scan.groups.get("stdk").unwrap();
        assert_eq!(g.weight_shape, vec![128, 32]);
        assert_eq!(g.base_name, "stdk");
    }

    #[test]
    fn scan_weight_suffix_embedded_in_name_not_matched() {
        // A tensor named "my.weight.extra" does NOT end with ".weight"
        let tensors = vec![
            cand("my.weight.extra", Dtype::U8, vec![64, 32], 2048),
            cand("my.weight.extra_scale", Dtype::U8, vec![64, 4], 256),
        ];
        let scan = scan_nvfp4_groups(tensors);
        assert!(
            scan.groups.is_empty(),
            "tensor name with '.weight' not at the end must not match"
        );
    }

    #[test]
    fn nvfp_group_with_global_scale_some_display() {
        // Verify that the Some variant is properly constructed and accessible
        let g = NvfpGroup {
            base_name: "gs_test".into(),
            weight_name: "gs_test.weight".into(),
            scale_name: "gs_test.weight_scale".into(),
            global_scale_name: Some("gs_test.weight_scale_2".into()),
            weight_shape: vec![64, 32],
        };
        let inner = g.global_scale_name.as_deref().expect("should be Some");
        assert_eq!(inner, "gs_test.weight_scale_2");
        assert!(inner.ends_with("_scale_2"));
    }
}
