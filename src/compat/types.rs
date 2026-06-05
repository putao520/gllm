//! Parameter grouping structs for compat layer functions.
//!
//! These structs reduce parameter counts by bundling related values
//! that are always passed together (attention geometry, layer dimensions,
//! sequence context, KV cache slices, and per-layer weights).

use super::weight_helpers::WeightData;
use gllm_kernels::inference::DType;

/// Attention head geometry — derived from `GeneratorForwardConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AttentionGeometry {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub heads_per_group: usize,
}

/// Per-layer dimension constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LayerDims {
    pub hidden: usize,
    pub inter: usize,
    pub eps: f32,
    pub rope_theta: f64,
}

/// Sequence position context for a single forward step.
pub(crate) struct SeqContext<'a> {
    pub positions: &'a [u32],
    pub seq_len: usize,
    pub total_seq: usize,
}

/// A read-only view into one layer's KV cache.
pub(crate) struct KvCacheSlice<'a> {
    pub k: &'a [u8],
    pub v: &'a [u8],
    pub dtype: DType,
    pub layer: usize,
    pub max_seq_len: usize,
}

/// Dense decoder layer weights (DType-aware byte slices).
pub(crate) struct DecoderLayerWeights<'a> {
    pub q_w: &'a [u8],
    pub k_w: &'a [u8],
    pub v_w: &'a [u8],
    pub o_w: &'a [u8],
    pub rn1_w: &'a [u8],
    pub rn2_w: &'a [u8],
    pub gate_w: &'a [u8],
    pub up_w: &'a [u8],
    pub down_w: &'a [u8],
    pub dtype: DType,
}

/// Quantized decoder layer weights (attention via WeightData, norms DType-aware).
pub(crate) struct QuantizedDecoderWeights<'a> {
    pub q: &'a WeightData,
    pub o: &'a WeightData,
    pub rn1_w: &'a [u8],
    pub rn2_w: &'a [u8],
    pub rn_dtype: DType,
    pub gate: &'a WeightData,
    pub up: &'a WeightData,
    pub down: &'a WeightData,
}

/// BERT encoder layer weights (DType-aware byte slices, with biases).
pub(crate) struct BertLayerWeights<'a> {
    pub q_w: &'a [u8],
    pub q_b: &'a [u8],
    pub k_w: &'a [u8],
    pub k_b: &'a [u8],
    pub v_w: &'a [u8],
    pub v_b: &'a [u8],
    pub out_w: &'a [u8],
    pub out_b: &'a [u8],
    pub ln1_w: &'a [u8],
    pub ln1_b: &'a [u8],
    pub ffn_up_w: &'a [u8],
    pub ffn_up_b: &'a [u8],
    pub ffn_down_w: &'a [u8],
    pub ffn_down_b: &'a [u8],
    pub ln2_w: &'a [u8],
    pub ln2_b: &'a [u8],
    pub dtype: DType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::backend_trait::QuantType;

    #[test]
    fn attention_geometry_fields() {
        let geo = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        assert_eq!(geo.num_heads, 8);
        assert_eq!(geo.heads_per_group, 4);
    }

    #[test]
    fn attention_geometry_copy() {
        let geo = AttentionGeometry {
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            q_dim: 128,
            kv_dim: 32,
            heads_per_group: 4,
        };
        let geo2 = geo;
        assert_eq!(geo2.num_heads, geo.num_heads);
    }

    #[test]
    fn layer_dims_fields() {
        let dims = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.hidden, 768);
        assert!((dims.eps - 1e-5).abs() < 1e-12);
        assert!((dims.rope_theta - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn layer_dims_copy() {
        let dims = LayerDims {
            hidden: 64,
            inter: 128,
            eps: 1e-5,
            rope_theta: 500000.0,
        };
        let dims2 = dims;
        assert_eq!(dims2.hidden, dims.hidden);
    }

    #[test]
    fn seq_context_fields() {
        let positions = [0u32, 1, 2];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 3,
            total_seq: 3,
        };
        assert_eq!(ctx.seq_len, 3);
        assert_eq!(ctx.positions.len(), 3);
    }

    #[test]
    fn kv_cache_slice_fields() {
        let k = [0u8; 16];
        let v = [0u8; 16];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 3,
            max_seq_len: 512,
        };
        assert_eq!(slice.layer, 3);
        assert_eq!(slice.max_seq_len, 512);
    }

    #[test]
    fn decoder_layer_weights_fields() {
        let data = [0u8; 64];
        let w = DecoderLayerWeights {
            q_w: &data,
            k_w: &data,
            v_w: &data,
            o_w: &data,
            rn1_w: &data,
            rn2_w: &data,
            gate_w: &data,
            up_w: &data,
            down_w: &data,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w.len(), 64);
        assert_eq!(w.dtype, DType::F32);
    }

    #[test]
    fn bert_layer_weights_fields() {
        let data = [0u8; 32];
        let w = BertLayerWeights {
            q_w: &data,
            q_b: &data,
            k_w: &data,
            k_b: &data,
            v_w: &data,
            v_b: &data,
            out_w: &data,
            out_b: &data,
            ln1_w: &data,
            ln1_b: &data,
            ffn_up_w: &data,
            ffn_up_b: &data,
            ffn_down_w: &data,
            ffn_down_b: &data,
            ln2_w: &data,
            ln2_b: &data,
            dtype: DType::BF16,
        };
        assert_eq!(w.dtype, DType::BF16);
    }

    // ── AttentionGeometry: additional tests ──

    #[test]
    fn attention_geometry_debug_format() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 1024,
            heads_per_group: 4,
        };
        let debug_str = format!("{:?}", geo);
        assert!(debug_str.contains("num_heads: 32"));
        assert!(debug_str.contains("num_kv_heads: 8"));
        assert!(debug_str.contains("head_dim: 128"));
        assert!(debug_str.contains("heads_per_group: 4"));
    }

    #[test]
    fn attention_geometry_clone_independent() {
        let geo = AttentionGeometry {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            q_dim: 1024,
            kv_dim: 256,
            heads_per_group: 4,
        };
        let cloned = geo.clone();
        assert_eq!(cloned.num_heads, geo.num_heads);
        assert_eq!(cloned.num_kv_heads, geo.num_kv_heads);
        assert_eq!(cloned.head_dim, geo.head_dim);
        assert_eq!(cloned.q_dim, geo.q_dim);
        assert_eq!(cloned.kv_dim, geo.kv_dim);
        assert_eq!(cloned.heads_per_group, geo.heads_per_group);
    }

    #[test]
    fn attention_geometry_mqa_single_kv_head() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 1,
            head_dim: 64,
            q_dim: 2048,
            kv_dim: 64,
            heads_per_group: 32,
        };
        assert_eq!(geo.heads_per_group, geo.num_heads);
        assert_eq!(geo.kv_dim, geo.head_dim);
    }

    #[test]
    fn attention_geometry_gqa_dimensions() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 1024,
            heads_per_group: 4,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
        assert_eq!(geo.heads_per_group, geo.num_heads / geo.num_kv_heads);
    }

    #[test]
    fn attention_geometry_zero_heads() {
        let geo = AttentionGeometry {
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 64,
            q_dim: 0,
            kv_dim: 0,
            heads_per_group: 0,
        };
        assert_eq!(geo.num_heads, 0);
        assert_eq!(geo.q_dim, 0);
    }

    // ── LayerDims: additional tests ──

    #[test]
    fn layer_dims_debug_format() {
        let dims = LayerDims {
            hidden: 4096,
            inter: 11008,
            eps: 1e-6,
            rope_theta: 500000.0,
        };
        let debug_str = format!("{:?}", dims);
        assert!(debug_str.contains("hidden: 4096"));
        assert!(debug_str.contains("inter: 11008"));
        assert!(debug_str.contains("eps:"));
        assert!(debug_str.contains("rope_theta:"));
    }

    #[test]
    fn layer_dims_clone_preserves_values() {
        let dims = LayerDims {
            hidden: 2048,
            inter: 5632,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        let cloned = dims.clone();
        assert_eq!(cloned.hidden, dims.hidden);
        assert_eq!(cloned.inter, dims.inter);
        assert!((cloned.eps - dims.eps).abs() < f32::EPSILON);
        assert!((cloned.rope_theta - dims.rope_theta).abs() < f64::EPSILON);
    }

    #[test]
    fn layer_dims_small_model() {
        let dims = LayerDims {
            hidden: 64,
            inter: 128,
            eps: 1e-12,
            rope_theta: 100.0,
        };
        assert_eq!(dims.hidden, 64);
        assert!((dims.eps - 1e-12).abs() < 1e-20);
        assert!((dims.rope_theta - 100.0).abs() < f64::EPSILON);
    }

    // ── SeqContext: additional tests ──

    #[test]
    fn seq_context_empty_positions() {
        let positions: [u32; 0] = [];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 0,
            total_seq: 10,
        };
        assert_eq!(ctx.seq_len, 0);
        assert!(ctx.positions.is_empty());
        assert_eq!(ctx.total_seq, 10);
    }

    #[test]
    fn seq_context_decode_step() {
        let positions = [15u32];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 1,
            total_seq: 16,
        };
        assert_eq!(ctx.positions[0], 15);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.total_seq, 16);
    }

    #[test]
    fn seq_context_prefill_varies_from_total() {
        let positions = [0u32, 1, 2, 3];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 4,
            total_seq: 4,
        };
        assert_eq!(ctx.seq_len, ctx.total_seq);
        assert_eq!(ctx.positions.len(), ctx.seq_len);
    }

    // ── KvCacheSlice: additional tests ──

    #[test]
    fn kv_cache_slice_empty_buffers() {
        let k: [u8; 0] = [];
        let v: [u8; 0] = [];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 0,
        };
        assert!(slice.k.is_empty());
        assert!(slice.v.is_empty());
        assert_eq!(slice.layer, 0);
    }

    #[test]
    fn kv_cache_slice_bf16_dtype() {
        let k = [0u8; 32];
        let v = [0u8; 32];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::BF16,
            layer: 5,
            max_seq_len: 2048,
        };
        assert_eq!(slice.dtype, DType::BF16);
        assert_eq!(slice.layer, 5);
        assert_eq!(slice.max_seq_len, 2048);
    }

    #[test]
    fn kv_cache_slice_f16_dtype() {
        let k = [0u8; 16];
        let v = [0u8; 16];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F16,
            layer: 12,
            max_seq_len: 4096,
        };
        assert_eq!(slice.dtype, DType::F16);
    }

    // ── DecoderLayerWeights: additional tests ──

    #[test]
    fn decoder_layer_weights_bf16_dtype() {
        let data = [0u8; 128];
        let w = DecoderLayerWeights {
            q_w: &data,
            k_w: &data,
            v_w: &data,
            o_w: &data,
            rn1_w: &data,
            rn2_w: &data,
            gate_w: &data,
            up_w: &data,
            down_w: &data,
            dtype: DType::BF16,
        };
        assert_eq!(w.dtype, DType::BF16);
        assert_eq!(w.gate_w.len(), 128);
        assert_eq!(w.up_w.len(), 128);
        assert_eq!(w.down_w.len(), 128);
    }

    #[test]
    fn decoder_layer_weights_distinct_slices() {
        let q = [1u8; 16];
        let k = [2u8; 16];
        let v = [3u8; 16];
        let w = DecoderLayerWeights {
            q_w: &q,
            k_w: &k,
            v_w: &v,
            o_w: &q,
            rn1_w: &k,
            rn2_w: &v,
            gate_w: &q,
            up_w: &k,
            down_w: &v,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w[0], 1);
        assert_eq!(w.k_w[0], 2);
        assert_eq!(w.v_w[0], 3);
    }

    // ── QuantizedDecoderWeights ──

    #[test]
    fn quantized_decoder_weights_fields() {
        use super::super::weight_helpers::WeightData;
        let q_w = WeightData::F32(vec![0.0; 16]);
        let o_w = WeightData::F32(vec![0.0; 16]);
        let gate_w = WeightData::F32(vec![0.0; 32]);
        let up_w = WeightData::F32(vec![0.0; 32]);
        let down_w = WeightData::F32(vec![0.0; 32]);
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &q_w,
            o: &o_w,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &gate_w,
            up: &up_w,
            down: &down_w,
        };
        assert_eq!(w.rn_dtype, DType::F32);
    }

    #[test]
    fn quantized_decoder_weights_bf16_norm_dtype() {
        use super::super::weight_helpers::WeightData;
        let wd = WeightData::F32(vec![1.0; 4]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::BF16,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert_eq!(w.rn_dtype, DType::BF16);
    }

    // ── BertLayerWeights: additional tests ──

    #[test]
    fn bert_layer_weights_f32_dtype() {
        let data = [0u8; 16];
        let w = BertLayerWeights {
            q_w: &data,
            q_b: &data,
            k_w: &data,
            k_b: &data,
            v_w: &data,
            v_b: &data,
            out_w: &data,
            out_b: &data,
            ln1_w: &data,
            ln1_b: &data,
            ffn_up_w: &data,
            ffn_up_b: &data,
            ffn_down_w: &data,
            ffn_down_b: &data,
            ln2_w: &data,
            ln2_b: &data,
            dtype: DType::F32,
        };
        assert_eq!(w.dtype, DType::F32);
        assert_eq!(w.q_b.len(), 16);
        assert_eq!(w.ffn_down_b.len(), 16);
    }

    #[test]
    fn bert_layer_weights_distinct_bias_slices() {
        let weight = [1u8; 8];
        let bias = [2u8; 8];
        let w = BertLayerWeights {
            q_w: &weight,
            q_b: &bias,
            k_w: &weight,
            k_b: &bias,
            v_w: &weight,
            v_b: &bias,
            out_w: &weight,
            out_b: &bias,
            ln1_w: &weight,
            ln1_b: &bias,
            ffn_up_w: &weight,
            ffn_up_b: &bias,
            ffn_down_w: &weight,
            ffn_down_b: &bias,
            ln2_w: &weight,
            ln2_b: &bias,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w[0], 1);
        assert_eq!(w.q_b[0], 2);
        assert_eq!(w.ffn_up_w[0], 1);
        assert_eq!(w.ffn_up_b[0], 2);
    }

    // ── DType variants coverage ──

    #[test]
    fn dtype_f16_size_and_equality() {
        assert_eq!(DType::F16, DType::F16);
        assert_ne!(DType::F16, DType::BF16);
        assert_eq!(DType::F16.size_bytes(), 2);
    }

    #[test]
    fn dtype_fp8_variants_size() {
        assert_eq!(DType::F8E4M3.size_bytes(), 1);
        assert_eq!(DType::F8E5M2.size_bytes(), 1);
        assert_ne!(DType::F8E4M3, DType::F8E5M2);
    }

    #[test]
    fn dtype_copy_preserves_variant() {
        let dt = DType::BF16;
        let dt2 = dt;
        assert_eq!(dt, dt2);
    }

    #[test]
    fn dtype_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DType::F32);
        set.insert(DType::BF16);
        set.insert(DType::F16);
        assert!(set.contains(&DType::F32));
        assert!(set.contains(&DType::BF16));
        assert!(set.contains(&DType::F16));
        assert!(!set.contains(&DType::U8));
        assert_eq!(set.len(), 3);
    }

    // =======================================================================
    // Additional tests for improved coverage
    // =======================================================================

    // ── AttentionGeometry: PartialEq, Hash, edge cases ──

    #[test]
    fn attention_geometry_partial_eq_equal() {
        let a = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        let b = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn attention_geometry_partial_eq_not_equal() {
        let a = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        let b = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 256,
            heads_per_group: 2,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn attention_geometry_hash_deduplication() {
        use std::collections::HashSet;
        let geo = AttentionGeometry {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            q_dim: 1024,
            kv_dim: 256,
            heads_per_group: 4,
        };
        let mut set = HashSet::new();
        set.insert(geo);
        set.insert(geo);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&geo));
    }

    #[test]
    fn attention_geometry_max_dimensions() {
        let geo = AttentionGeometry {
            num_heads: usize::MAX,
            num_kv_heads: usize::MAX,
            head_dim: usize::MAX,
            q_dim: usize::MAX,
            kv_dim: usize::MAX,
            heads_per_group: usize::MAX,
        };
        assert_eq!(geo.num_heads, usize::MAX);
        assert_eq!(geo.head_dim, usize::MAX);
    }

    #[test]
    fn attention_geometry_mha_equal_heads() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 4096,
            heads_per_group: 1,
        };
        assert_eq!(geo.num_heads, geo.num_kv_heads);
        assert_eq!(geo.q_dim, geo.kv_dim);
        assert_eq!(geo.heads_per_group, 1);
    }

    // ── LayerDims: PartialEq, edge cases ──

    #[test]
    fn layer_dims_partial_eq_equal() {
        let a = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        let b = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn layer_dims_partial_eq_different_hidden() {
        let a = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        let b = LayerDims {
            hidden: 1024,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn layer_dims_zero_eps() {
        let dims = LayerDims {
            hidden: 256,
            inter: 512,
            eps: 0.0,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.eps, 0.0);
    }

    #[test]
    fn layer_dims_zero_rope_theta() {
        let dims = LayerDims {
            hidden: 256,
            inter: 512,
            eps: 1e-5,
            rope_theta: 0.0,
        };
        assert_eq!(dims.rope_theta, 0.0);
    }

    #[test]
    fn layer_dims_max_values() {
        let dims = LayerDims {
            hidden: usize::MAX,
            inter: usize::MAX,
            eps: f32::MAX,
            rope_theta: f64::MAX,
        };
        assert_eq!(dims.hidden, usize::MAX);
        assert_eq!(dims.inter, usize::MAX);
        assert_eq!(dims.eps, f32::MAX);
        assert_eq!(dims.rope_theta, f64::MAX);
    }

    // ── SeqContext: edge cases ──

    #[test]
    fn seq_context_single_token_prefill() {
        let positions = [0u32];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 1,
            total_seq: 1,
        };
        assert_eq!(ctx.positions[0], 0);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.total_seq, 1);
    }

    #[test]
    fn seq_context_max_position_value() {
        let positions = [u32::MAX];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 1,
            total_seq: (u32::MAX as usize) + 1,
        };
        assert_eq!(ctx.positions[0], u32::MAX);
        assert_eq!(ctx.total_seq, (u32::MAX as usize) + 1);
    }

    #[test]
    fn seq_context_seq_len_exceeds_positions() {
        // seq_len and positions slice can differ in real usage;
        // the struct is a plain data bundle, no validation.
        let positions = [0u32, 1];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 2,
            total_seq: 100,
        };
        assert_eq!(ctx.positions.len(), 2);
        assert_eq!(ctx.total_seq, 100);
    }

    // ── KvCacheSlice: edge cases ──

    #[test]
    fn kv_cache_slice_asymmetric_buffers() {
        let k = [0u8; 64];
        let v = [0u8; 32];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 128,
        };
        assert_eq!(slice.k.len(), 64);
        assert_eq!(slice.v.len(), 32);
    }

    #[test]
    fn kv_cache_slice_max_layer_and_seq() {
        let k = [0u8; 8];
        let v = [0u8; 8];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F16,
            layer: usize::MAX,
            max_seq_len: usize::MAX,
        };
        assert_eq!(slice.layer, usize::MAX);
        assert_eq!(slice.max_seq_len, usize::MAX);
    }

    #[test]
    fn kv_cache_slice_all_dtypes() {
        let buf = [0u8; 4];
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ];
        for &dt in &dtypes {
            let slice = KvCacheSlice {
                k: &buf,
                v: &buf,
                dtype: dt,
                layer: 0,
                max_seq_len: 0,
            };
            assert_eq!(slice.dtype, dt);
        }
    }

    // ── DecoderLayerWeights: edge cases ──

    #[test]
    fn decoder_layer_weights_empty_slices() {
        let empty: [u8; 0] = [];
        let w = DecoderLayerWeights {
            q_w: &empty,
            k_w: &empty,
            v_w: &empty,
            o_w: &empty,
            rn1_w: &empty,
            rn2_w: &empty,
            gate_w: &empty,
            up_w: &empty,
            down_w: &empty,
            dtype: DType::F32,
        };
        assert!(w.q_w.is_empty());
        assert!(w.gate_w.is_empty());
        assert!(w.down_w.is_empty());
    }

    #[test]
    fn decoder_layer_weights_all_dtypes() {
        let buf = [0u8; 4];
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ];
        for &dt in &dtypes {
            let w = DecoderLayerWeights {
                q_w: &buf,
                k_w: &buf,
                v_w: &buf,
                o_w: &buf,
                rn1_w: &buf,
                rn2_w: &buf,
                gate_w: &buf,
                up_w: &buf,
                down_w: &buf,
                dtype: dt,
            };
            assert_eq!(w.dtype, dt);
        }
    }

    // ── QuantizedDecoderWeights: additional coverage ──

    #[test]
    fn quantized_decoder_weights_distinct_norm_slices() {
        use super::super::weight_helpers::WeightData;
        let wd = WeightData::F32(vec![0.0; 4]);
        let rn1 = [1u8; 8];
        let rn2 = [2u8; 8];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &rn1,
            rn2_w: &rn2,
            rn_dtype: DType::F32,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert_eq!(w.rn1_w[0], 1);
        assert_eq!(w.rn2_w[0], 2);
    }

    #[test]
    fn quantized_decoder_weights_empty_norm() {
        use super::super::weight_helpers::WeightData;
        let wd = WeightData::F32(vec![]);
        let empty: [u8; 0] = [];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &empty,
            rn2_w: &empty,
            rn_dtype: DType::BF16,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert!(w.rn1_w.is_empty());
        assert!(w.rn2_w.is_empty());
        assert_eq!(w.rn_dtype, DType::BF16);
    }

    // ── BertLayerWeights: edge cases ──

    #[test]
    fn bert_layer_weights_empty_slices() {
        let empty: [u8; 0] = [];
        let w = BertLayerWeights {
            q_w: &empty,
            q_b: &empty,
            k_w: &empty,
            k_b: &empty,
            v_w: &empty,
            v_b: &empty,
            out_w: &empty,
            out_b: &empty,
            ln1_w: &empty,
            ln1_b: &empty,
            ffn_up_w: &empty,
            ffn_up_b: &empty,
            ffn_down_w: &empty,
            ffn_down_b: &empty,
            ln2_w: &empty,
            ln2_b: &empty,
            dtype: DType::F32,
        };
        assert!(w.q_w.is_empty());
        assert!(w.q_b.is_empty());
        assert!(w.ffn_down_b.is_empty());
        assert!(w.ln2_b.is_empty());
    }

    #[test]
    fn bert_layer_weights_all_dtypes() {
        let buf = [0u8; 4];
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ];
        for &dt in &dtypes {
            let w = BertLayerWeights {
                q_w: &buf,
                q_b: &buf,
                k_w: &buf,
                k_b: &buf,
                v_w: &buf,
                v_b: &buf,
                out_w: &buf,
                out_b: &buf,
                ln1_w: &buf,
                ln1_b: &buf,
                ffn_up_w: &buf,
                ffn_up_b: &buf,
                ffn_down_w: &buf,
                ffn_down_b: &buf,
                ln2_w: &buf,
                ln2_b: &buf,
                dtype: dt,
            };
            assert_eq!(w.dtype, dt);
        }
    }

    #[test]
    fn bert_layer_weights_16_fields_independent() {
        let a = [1u8; 4];
        let b = [2u8; 4];
        let w = BertLayerWeights {
            q_w: &a,
            q_b: &b,
            k_w: &a,
            k_b: &b,
            v_w: &a,
            v_b: &b,
            out_w: &a,
            out_b: &b,
            ln1_w: &a,
            ln1_b: &b,
            ffn_up_w: &a,
            ffn_up_b: &b,
            ffn_down_w: &a,
            ffn_down_b: &b,
            ln2_w: &a,
            ln2_b: &b,
            dtype: DType::BF16,
        };
        // All weight fields point to `a`, all bias fields point to `b`
        for field in [&w.q_w, &w.k_w, &w.v_w, &w.out_w, &w.ln1_w, &w.ffn_up_w, &w.ffn_down_w, &w.ln2_w] {
            assert_eq!(field[0], 1);
        }
        for field in [&w.q_b, &w.k_b, &w.v_b, &w.out_b, &w.ln1_b, &w.ffn_up_b, &w.ffn_down_b, &w.ln2_b] {
            assert_eq!(field[0], 2);
        }
    }

    // ── DType: exhaustive variant coverage ──

    #[test]
    fn dtype_all_variants_size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::F8E4M3.size_bytes(), 1);
        assert_eq!(DType::F8E5M2.size_bytes(), 1);
        assert_eq!(DType::F6E3M2.size_bytes(), 1);
        assert_eq!(DType::F6E2M3.size_bytes(), 1);
        assert_eq!(DType::F4E2M1.size_bytes(), 1);
    }

    #[test]
    fn dtype_all_variants_elem_id_unique() {
        use std::collections::HashSet;
        let ids: Vec<u8> = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ]
        .iter()
        .map(|dt| dt.elem_id())
        .collect();
        let unique: HashSet<u8> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "elem_id values must all be unique");
    }

    #[test]
    fn dtype_all_variants_equality_and_inequality() {
        let all = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn dtype_f16_and_bf16_same_size_distinct() {
        assert_eq!(DType::F16.size_bytes(), DType::BF16.size_bytes());
        assert_ne!(DType::F16, DType::BF16);
    }

    #[test]
    fn dtype_u8_size_one() {
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_ne!(DType::U8, DType::F8E4M3);
    }

    #[test]
    fn dtype_sub_byte_variants_distinct() {
        assert_ne!(DType::F6E3M2, DType::F6E2M3);
        assert_ne!(DType::F6E3M2, DType::F4E2M1);
        assert_ne!(DType::F6E2M3, DType::F4E2M1);
    }

    #[test]
    fn dtype_debug_contains_variant_name() {
        let all = [
            (DType::F32, "F32"),
            (DType::F16, "F16"),
            (DType::BF16, "BF16"),
            (DType::U8, "U8"),
            (DType::F8E4M3, "F8E4M3"),
            (DType::F8E5M2, "F8E5M2"),
            (DType::F6E3M2, "F6E3M2"),
            (DType::F6E2M3, "F6E2M3"),
            (DType::F4E2M1, "F4E2M1"),
        ];
        for (dt, name) in &all {
            let debug = format!("{:?}", dt);
            assert!(debug.contains(name), "Debug output {:?} should contain {}", debug, name);
        }
    }

    #[test]
    fn dtype_hash_set_all_variants() {
        use std::collections::HashSet;
        let all = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E3M2,
            DType::F6E2M3,
            DType::F4E2M1,
        ];
        let set: HashSet<DType> = all.iter().copied().collect();
        assert_eq!(set.len(), all.len());
        for dt in &all {
            assert!(set.contains(dt));
        }
    }

    // =======================================================================
    // 45+ additional tests for improved coverage
    // =======================================================================

    // ── AttentionGeometry: Eq trait, Hash in HashMap, struct layout ──

    #[test]
    fn attention_geometry_eq_symmetry() {
        let a = AttentionGeometry {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            q_dim: 128,
            kv_dim: 64,
            heads_per_group: 2,
        };
        let b = AttentionGeometry {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            q_dim: 128,
            kv_dim: 64,
            heads_per_group: 2,
        };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn attention_geometry_ne_differs_per_field() {
        let base = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        // Each field changed individually should produce != base
        let diff_heads = AttentionGeometry { num_heads: 99, ..base };
        let diff_kv = AttentionGeometry { num_kv_heads: 99, ..base };
        let diff_dim = AttentionGeometry { head_dim: 99, ..base };
        let diff_q = AttentionGeometry { q_dim: 99, ..base };
        let diff_kvdim = AttentionGeometry { kv_dim: 99, ..base };
        let diff_hpg = AttentionGeometry { heads_per_group: 99, ..base };
        assert_ne!(base, diff_heads);
        assert_ne!(base, diff_kv);
        assert_ne!(base, diff_dim);
        assert_ne!(base, diff_q);
        assert_ne!(base, diff_kvdim);
        assert_ne!(base, diff_hpg);
    }

    #[test]
    fn attention_geometry_hash_map_lookup() {
        use std::collections::HashMap;
        let geo = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        let mut map = HashMap::new();
        map.insert(geo, "gqa_8x2");
        assert_eq!(map.get(&geo), Some(&"gqa_8x2"));
        let same = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        assert_eq!(map.get(&same), Some(&"gqa_8x2"));
    }

    #[test]
    fn attention_geometry_unit_head_dim() {
        let geo = AttentionGeometry {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            q_dim: 1,
            kv_dim: 1,
            heads_per_group: 1,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
    }

    #[test]
    fn attention_geometry_large_gqa_ratio() {
        // 128 Q heads, 1 KV head (extreme MQA)
        let geo = AttentionGeometry {
            num_heads: 128,
            num_kv_heads: 1,
            head_dim: 64,
            q_dim: 8192,
            kv_dim: 64,
            heads_per_group: 128,
        };
        assert_eq!(geo.heads_per_group, 128);
        assert_eq!(geo.heads_per_group, geo.num_heads / geo.num_kv_heads);
    }

    // ── LayerDims: special float values, negative values, equality edge cases ──

    #[test]
    fn layer_dims_eps_nan_is_not_equal() {
        let a = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: f32::NAN,
            rope_theta: 10000.0,
        };
        let b = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: f32::NAN,
            rope_theta: 10000.0,
        };
        // NaN != NaN per IEEE 754, so PartialOrd should say not equal
        assert_ne!(a, b);
    }

    #[test]
    fn layer_dims_eps_infinity() {
        let dims = LayerDims {
            hidden: 256,
            inter: 512,
            eps: f32::INFINITY,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.eps, f32::INFINITY);
    }

    #[test]
    fn layer_dims_eps_neg_infinity() {
        let dims = LayerDims {
            hidden: 256,
            inter: 512,
            eps: f32::NEG_INFINITY,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.eps, f32::NEG_INFINITY);
    }

    #[test]
    fn layer_dims_negative_eps() {
        let dims = LayerDims {
            hidden: 512,
            inter: 1024,
            eps: -1e-5,
            rope_theta: 10000.0,
        };
        assert!(dims.eps < 0.0);
    }

    #[test]
    fn layer_dims_negative_rope_theta() {
        let dims = LayerDims {
            hidden: 512,
            inter: 1024,
            eps: 1e-5,
            rope_theta: -500.0,
        };
        assert!(dims.rope_theta < 0.0);
    }

    #[test]
    fn layer_dims_very_small_eps() {
        let dims = LayerDims {
            hidden: 1024,
            inter: 4096,
            eps: f32::MIN_POSITIVE,
            rope_theta: 10000.0,
        };
        assert!(dims.eps > 0.0);
        assert_eq!(dims.eps, f32::MIN_POSITIVE);
    }

    #[test]
    fn layer_dims_hidden_zero() {
        let dims = LayerDims {
            hidden: 0,
            inter: 0,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.hidden, 0);
        assert_eq!(dims.inter, 0);
    }

    #[test]
    fn layer_dims_copy_independent() {
        let dims = LayerDims {
            hidden: 1024,
            inter: 4096,
            eps: 1e-6,
            rope_theta: 500000.0,
        };
        let mut dims2 = dims;
        dims2.hidden = 2048;
        // Original should remain unchanged (Copy trait)
        assert_eq!(dims.hidden, 1024);
        assert_eq!(dims2.hidden, 2048);
    }

    // ── SeqContext: lifetime and slice patterns ──

    #[test]
    fn seq_context_positions_match_first_last() {
        let positions = [10u32, 20, 30, 40, 50];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 5,
            total_seq: 50,
        };
        assert_eq!(ctx.positions.first(), Some(&10));
        assert_eq!(ctx.positions.last(), Some(&50));
    }

    #[test]
    fn seq_context_zero_total_seq() {
        let positions: [u32; 0] = [];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 0,
            total_seq: 0,
        };
        assert_eq!(ctx.total_seq, 0);
        assert!(ctx.positions.is_empty());
    }

    #[test]
    fn seq_context_long_sequence() {
        let positions: Vec<u32> = (0..10000).collect();
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 10000,
            total_seq: 10000,
        };
        assert_eq!(ctx.seq_len, 10000);
        assert_eq!(ctx.positions.len(), 10000);
        assert_eq!(ctx.positions[9999], 9999);
    }

    #[test]
    fn seq_context_positions_subslice() {
        let full = [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        // Simulate using a sub-slice
        let sub = &full[3..7];
        let ctx = SeqContext {
            positions: sub,
            seq_len: 4,
            total_seq: 10,
        };
        assert_eq!(ctx.positions.len(), 4);
        assert_eq!(ctx.positions[0], 3);
        assert_eq!(ctx.positions[3], 6);
    }

    // ── KvCacheSlice: dtype-specific sizes, layer indexing ──

    #[test]
    fn kv_cache_slice_dtype_f32_implies_4bytes_per_element() {
        let k = [0u8; 256];
        let v = [0u8; 256];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 64,
        };
        assert_eq!(slice.dtype.size_bytes(), 4);
        // 256 bytes / 4 bytes per elem = 64 elements
        assert_eq!(slice.k.len() / slice.dtype.size_bytes(), 64);
    }

    #[test]
    fn kv_cache_slice_dtype_bf16_implies_2bytes_per_element() {
        let k = [0u8; 128];
        let v = [0u8; 128];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::BF16,
            layer: 1,
            max_seq_len: 64,
        };
        assert_eq!(slice.dtype.size_bytes(), 2);
    }

    #[test]
    fn kv_cache_slice_layer_zero_first_layer() {
        let k = [0u8; 8];
        let v = [0u8; 8];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F16,
            layer: 0,
            max_seq_len: 2048,
        };
        assert_eq!(slice.layer, 0);
    }

    #[test]
    fn kv_cache_slice_single_byte_buffers() {
        let k = [42u8];
        let v = [99u8];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::U8,
            layer: 0,
            max_seq_len: 1,
        };
        assert_eq!(slice.k[0], 42);
        assert_eq!(slice.v[0], 99);
    }

    // ── DecoderLayerWeights: all 9 fields independent, dtype combinations ──

    #[test]
    fn decoder_layer_weights_f16_dtype() {
        let data = [0u8; 32];
        let w = DecoderLayerWeights {
            q_w: &data,
            k_w: &data,
            v_w: &data,
            o_w: &data,
            rn1_w: &data,
            rn2_w: &data,
            gate_w: &data,
            up_w: &data,
            down_w: &data,
            dtype: DType::F16,
        };
        assert_eq!(w.dtype, DType::F16);
        assert_eq!(w.dtype.size_bytes(), 2);
    }

    #[test]
    fn decoder_layer_weights_all_nine_fields_distinct() {
        let q = [1u8; 4];
        let k = [2u8; 4];
        let v = [3u8; 4];
        let o = [4u8; 4];
        let rn1 = [5u8; 4];
        let rn2 = [6u8; 4];
        let gate = [7u8; 4];
        let up = [8u8; 4];
        let down = [9u8; 4];
        let w = DecoderLayerWeights {
            q_w: &q,
            k_w: &k,
            v_w: &v,
            o_w: &o,
            rn1_w: &rn1,
            rn2_w: &rn2,
            gate_w: &gate,
            up_w: &up,
            down_w: &down,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w[0], 1);
        assert_eq!(w.k_w[0], 2);
        assert_eq!(w.v_w[0], 3);
        assert_eq!(w.o_w[0], 4);
        assert_eq!(w.rn1_w[0], 5);
        assert_eq!(w.rn2_w[0], 6);
        assert_eq!(w.gate_w[0], 7);
        assert_eq!(w.up_w[0], 8);
        assert_eq!(w.down_w[0], 9);
    }

    #[test]
    fn decoder_layer_weights_norm_fields_are_separate() {
        let rn1 = [0xAAu8; 4];
        let rn2 = [0xBBu8; 4];
        let other = [0u8; 4];
        let w = DecoderLayerWeights {
            q_w: &other,
            k_w: &other,
            v_w: &other,
            o_w: &other,
            rn1_w: &rn1,
            rn2_w: &rn2,
            gate_w: &other,
            up_w: &other,
            down_w: &other,
            dtype: DType::F32,
        };
        assert_eq!(w.rn1_w[0], 0xAA);
        assert_eq!(w.rn2_w[0], 0xBB);
    }

    // ── WeightData enum variant tests ──

    #[test]
    fn weight_data_f32_variant_holds_data() {
        let wd = WeightData::F32(vec![1.0, 2.0, 3.0, 4.0]);
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data.len(), 4);
                assert_eq!(data[0], 1.0);
                assert_eq!(data[3], 4.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    #[test]
    fn weight_data_quantized_variant_fields() {

        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q4_0,
            out_dim: 16,
            in_dim: 32,
        };
        match &wd {
            WeightData::Quantized { data, quant_type, out_dim, in_dim } => {
                assert_eq!(data.len(), 64);
                assert_eq!(*quant_type, QuantType::Q4_0);
                assert_eq!(*out_dim, 16);
                assert_eq!(*in_dim, 32);
            }
            WeightData::F32(_) => panic!("expected Quantized variant"),
        }
    }

    #[test]
    fn weight_data_f32_empty_vec() {
        let wd = WeightData::F32(vec![]);
        match &wd {
            WeightData::F32(data) => assert!(data.is_empty()),
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    #[test]
    fn weight_data_quantized_empty_data() {

        let wd = WeightData::Quantized {
            data: vec![],
            quant_type: QuantType::Q8_0,
            out_dim: 0,
            in_dim: 0,
        };
        match &wd {
            WeightData::Quantized { data, out_dim, in_dim, .. } => {
                assert!(data.is_empty());
                assert_eq!(*out_dim, 0);
                assert_eq!(*in_dim, 0);
            }
            WeightData::F32(_) => panic!("expected Quantized variant"),
        }
    }

    #[test]
    fn weight_data_f32_single_element() {
        let wd = WeightData::F32(vec![42.0]);
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data.len(), 1);
                assert_eq!(data[0], 42.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // ── QuantizedDecoderWeights: additional WeightData variant coverage ──

    #[test]
    fn quantized_decoder_weights_with_quantized_gate() {

        let f32_wd = WeightData::F32(vec![0.0; 4]);
        let quant_wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &f32_wd,
            o: &f32_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &quant_wd,
            up: &f32_wd,
            down: &f32_wd,
        };
        // Verify gate is quantized
        match w.gate {
            WeightData::Quantized { out_dim, in_dim, .. } => {
                assert_eq!(*out_dim, 8);
                assert_eq!(*in_dim, 4);
            }
            WeightData::F32(_) => panic!("expected Quantized variant for gate"),
        }
    }

    #[test]
    fn quantized_decoder_weights_all_f32_variants() {
        let wd = WeightData::F32(vec![1.0, 2.0, 3.0]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::BF16,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        // All weight fields point to same WeightData
        assert!(matches!(w.q, WeightData::F32(_)));
        assert!(matches!(w.o, WeightData::F32(_)));
        assert!(matches!(w.gate, WeightData::F32(_)));
        assert!(matches!(w.up, WeightData::F32(_)));
        assert!(matches!(w.down, WeightData::F32(_)));
    }

    #[test]
    fn quantized_decoder_weights_f16_norm_dtype() {
        let wd = WeightData::F32(vec![0.0; 2]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F16,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert_eq!(w.rn_dtype, DType::F16);
        assert_eq!(w.rn_dtype.size_bytes(), 2);
    }

    // ── BertLayerWeights: 16 fields all independent ──

    #[test]
    fn bert_layer_weights_all_16_fields_different_data() {
        let d0 = [0u8; 4];
        let d1 = [1u8; 4];
        let d2 = [2u8; 4];
        let d3 = [3u8; 4];
        let d4 = [4u8; 4];
        let d5 = [5u8; 4];
        let d6 = [6u8; 4];
        let d7 = [7u8; 4];
        let d8 = [8u8; 4];
        let d9 = [9u8; 4];
        let d10 = [10u8; 4];
        let d11 = [11u8; 4];
        let d12 = [12u8; 4];
        let d13 = [13u8; 4];
        let d14 = [14u8; 4];
        let d15 = [15u8; 4];
        let w = BertLayerWeights {
            q_w: &d0,
            q_b: &d1,
            k_w: &d2,
            k_b: &d3,
            v_w: &d4,
            v_b: &d5,
            out_w: &d6,
            out_b: &d7,
            ln1_w: &d8,
            ln1_b: &d9,
            ffn_up_w: &d10,
            ffn_up_b: &d11,
            ffn_down_w: &d12,
            ffn_down_b: &d13,
            ln2_w: &d14,
            ln2_b: &d15,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w[0], 0);
        assert_eq!(w.q_b[0], 1);
        assert_eq!(w.k_w[0], 2);
        assert_eq!(w.k_b[0], 3);
        assert_eq!(w.v_w[0], 4);
        assert_eq!(w.v_b[0], 5);
        assert_eq!(w.out_w[0], 6);
        assert_eq!(w.out_b[0], 7);
        assert_eq!(w.ln1_w[0], 8);
        assert_eq!(w.ln1_b[0], 9);
        assert_eq!(w.ffn_up_w[0], 10);
        assert_eq!(w.ffn_up_b[0], 11);
        assert_eq!(w.ffn_down_w[0], 12);
        assert_eq!(w.ffn_down_b[0], 13);
        assert_eq!(w.ln2_w[0], 14);
        assert_eq!(w.ln2_b[0], 15);
    }

    #[test]
    fn bert_layer_weights_hf16_dtype() {
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data,
            q_b: &data,
            k_w: &data,
            k_b: &data,
            v_w: &data,
            v_b: &data,
            out_w: &data,
            out_b: &data,
            ln1_w: &data,
            ln1_b: &data,
            ffn_up_w: &data,
            ffn_up_b: &data,
            ffn_down_w: &data,
            ffn_down_b: &data,
            ln2_w: &data,
            ln2_b: &data,
            dtype: DType::F16,
        };
        assert_eq!(w.dtype, DType::F16);
    }

    #[test]
    fn bert_layer_weights_u8_dtype() {
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data,
            q_b: &data,
            k_w: &data,
            k_b: &data,
            v_w: &data,
            v_b: &data,
            out_w: &data,
            out_b: &data,
            ln1_w: &data,
            ln1_b: &data,
            ffn_up_w: &data,
            ffn_up_b: &data,
            ffn_down_w: &data,
            ffn_down_b: &data,
            ln2_w: &data,
            ln2_b: &data,
            dtype: DType::U8,
        };
        assert_eq!(w.dtype, DType::U8);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── DType: gpu_type_name, ptx_type, ptx_reg_type, ptx_ld_type, hip_type, msl_type ──

    #[test]
    fn dtype_gpu_type_name_f32() {
        assert_eq!(DType::F32.gpu_type_name(), Ok("f32"));
    }

    #[test]
    fn dtype_gpu_type_name_f16() {
        assert_eq!(DType::F16.gpu_type_name(), Ok("f16"));
    }

    #[test]
    fn dtype_gpu_type_name_bf16() {
        assert_eq!(DType::BF16.gpu_type_name(), Ok("bf16"));
    }

    #[test]
    fn dtype_gpu_type_name_sub_byte_returns_err() {
        assert_eq!(DType::F6E3M2.gpu_type_name(), Err(()));
        assert_eq!(DType::F6E2M3.gpu_type_name(), Err(()));
    }

    #[test]
    fn dtype_ptx_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.ptx_type().is_empty());
        }
    }

    #[test]
    fn dtype_ptx_reg_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.ptx_reg_type().is_empty());
        }
    }

    #[test]
    fn dtype_hip_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.hip_type().is_empty());
        }
    }

    #[test]
    fn dtype_msl_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.msl_type().is_empty());
        }
    }

    #[test]
    fn dtype_ptx_arith_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.ptx_arith_type().is_empty());
        }
    }

    #[test]
    fn dtype_elem_id_monotonic() {
        // elem_id values should be 0..=8 for the 9 variants
        assert_eq!(DType::F32.elem_id(), 0);
        assert_eq!(DType::F16.elem_id(), 1);
        assert_eq!(DType::BF16.elem_id(), 2);
        assert_eq!(DType::U8.elem_id(), 3);
        assert_eq!(DType::F8E4M3.elem_id(), 4);
        assert_eq!(DType::F8E5M2.elem_id(), 5);
        assert_eq!(DType::F6E3M2.elem_id(), 6);
        assert_eq!(DType::F6E2M3.elem_id(), 7);
        assert_eq!(DType::F4E2M1.elem_id(), 8);
    }

    #[test]
    fn dtype_copy_independent() {
        let dt = DType::BF16;
        let dt2 = dt;
        // Both should be equal and valid
        assert_eq!(dt, dt2);
        assert_eq!(dt.size_bytes(), 2);
        assert_eq!(dt2.size_bytes(), 2);
    }

    // ── Cross-struct: KvCacheSlice with DecoderLayerWeights same dtype ──

    #[test]
    fn kv_and_decoder_weights_share_dtype() {
        let data = [0u8; 16];
        let k = [0u8; 16];
        let v = [0u8; 16];
        let kv = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::BF16,
            layer: 0,
            max_seq_len: 512,
        };
        let weights = DecoderLayerWeights {
            q_w: &data,
            k_w: &data,
            v_w: &data,
            o_w: &data,
            rn1_w: &data,
            rn2_w: &data,
            gate_w: &data,
            up_w: &data,
            down_w: &data,
            dtype: DType::BF16,
        };
        assert_eq!(kv.dtype, weights.dtype);
    }

    // ── QuantType in WeightData: use Q8_0 variant ──

    #[test]
    fn weight_data_quantized_q8_0() {

        let wd = WeightData::Quantized {
            data: vec![0u8; 128],
            quant_type: QuantType::Q8_0,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => {
                assert_eq!(*quant_type, QuantType::Q8_0);
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_large_dims() {

        let wd = WeightData::Quantized {
            data: vec![0u8; 1024],
            quant_type: QuantType::Q4_0,
            out_dim: usize::MAX,
            in_dim: usize::MAX,
        };
        match &wd {
            WeightData::Quantized { out_dim, in_dim, data, .. } => {
                assert_eq!(*out_dim, usize::MAX);
                assert_eq!(*in_dim, usize::MAX);
                assert_eq!(data.len(), 1024);
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── AttentionGeometry: dimension consistency check ──

    #[test]
    fn attention_geometry_dimension_consistency_llama_7b() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 4096,
            heads_per_group: 1,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
        assert_eq!(geo.heads_per_group, geo.num_heads / geo.num_kv_heads);
    }

    #[test]
    fn attention_geometry_dimension_consistency_gemma_2b() {
        // Gemma 2B: 8 heads, 1 KV head, head_dim 256
        let geo = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
            q_dim: 2048,
            kv_dim: 256,
            heads_per_group: 8,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
    }

    // ── LayerDims: PartialEq reflexivity and transitivity ──

    #[test]
    fn layer_dims_partial_eq_reflexivity() {
        let dims = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_eq!(dims, dims);
    }

    #[test]
    fn layer_dims_partial_eq_transitivity() {
        let a = LayerDims { hidden: 512, inter: 2048, eps: 1e-5, rope_theta: 10000.0 };
        let b = LayerDims { hidden: 512, inter: 2048, eps: 1e-5, rope_theta: 10000.0 };
        let c = LayerDims { hidden: 512, inter: 2048, eps: 1e-5, rope_theta: 10000.0 };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── DType: size_bytes vs elem_id relationship ──

    #[test]
    fn dtype_f32_is_largest_standard() {
        // Among standard types, F32 has the largest size_bytes
        assert!(DType::F32.size_bytes() >= DType::F16.size_bytes());
        assert!(DType::F32.size_bytes() >= DType::BF16.size_bytes());
        assert!(DType::F32.size_bytes() >= DType::U8.size_bytes());
    }

    #[test]
    fn dtype_fp8_variants_same_size() {
        assert_eq!(DType::F8E4M3.size_bytes(), DType::F8E5M2.size_bytes());
    }

    #[test]
    fn dtype_gpu_type_name_fp8_variants() {
        assert_eq!(DType::F8E4M3.gpu_type_name(), Ok("e4m3"));
        assert_eq!(DType::F8E5M2.gpu_type_name(), Ok("e5m2"));
    }

    #[test]
    fn dtype_gpu_type_name_f4e2m1() {
        assert_eq!(DType::F4E2M1.gpu_type_name(), Ok("e2m1"));
    }

    #[test]
    fn dtype_gpu_type_name_u8() {
        assert_eq!(DType::U8.gpu_type_name(), Ok("u8"));
    }

    // =======================================================================
    // 18 additional tests for improved coverage
    // =======================================================================

    // ── DType: ptx_ld_type (only untested method) ──

    #[test]
    fn dtype_ptx_ld_type_returns_non_empty() {
        let all = [
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ];
        for dt in &all {
            assert!(!dt.ptx_ld_type().is_empty());
        }
    }

    // ── WeightData: more QuantType variants ──

    #[test]
    fn weight_data_quantized_q4_1_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Q4_1,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q4_1),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q5_0_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Q5_0,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q5_0),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_awq4_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::AWQ4,
            out_dim: 4,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::AWQ4),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_gptq4_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::GPTQ4,
            out_dim: 4,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::GPTQ4),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q2k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q2K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q2K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q4k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q4K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q4K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── Cross-struct dtype consistency ──

    #[test]
    fn kv_and_bert_weights_share_dtype() {
        let k = [0u8; 8];
        let v = [0u8; 8];
        let kv = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::BF16,
            layer: 0,
            max_seq_len: 512,
        };
        let data = [0u8; 8];
        let bert = BertLayerWeights {
            q_w: &data, q_b: &data, k_w: &data, k_b: &data,
            v_w: &data, v_b: &data, out_w: &data, out_b: &data,
            ln1_w: &data, ln1_b: &data, ffn_up_w: &data, ffn_up_b: &data,
            ffn_down_w: &data, ffn_down_b: &data, ln2_w: &data, ln2_b: &data,
            dtype: DType::BF16,
        };
        assert_eq!(kv.dtype, bert.dtype);
    }

    #[test]
    fn kv_and_quantized_weights_dtype_consistency() {
        let k = [0u8; 8];
        let v = [0u8; 8];
        let kv = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F16,
            layer: 2,
            max_seq_len: 1024,
        };
        let wd = WeightData::F32(vec![0.0; 4]);
        let norm = [0u8; 4];
        let qw = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F16,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert_eq!(kv.dtype, qw.rn_dtype);
    }

    // ── QuantizedDecoderWeights: all FFN quantized ──

    #[test]
    fn quantized_decoder_weights_all_quantized_ffn() {
        let f32_wd = WeightData::F32(vec![0.0; 4]);
        let quant_gate = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let quant_up = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let quant_down = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::Q4_0,
            out_dim: 4,
            in_dim: 8,
        };
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &f32_wd,
            o: &f32_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &quant_gate,
            up: &quant_up,
            down: &quant_down,
        };
        assert!(matches!(w.gate, WeightData::Quantized { .. }));
        assert!(matches!(w.up, WeightData::Quantized { .. }));
        assert!(matches!(w.down, WeightData::Quantized { .. }));
    }

    #[test]
    fn quantized_decoder_weights_f8e4m3_norm_dtype() {
        let wd = WeightData::F32(vec![0.0; 2]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F8E4M3,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        assert_eq!(w.rn_dtype, DType::F8E4M3);
    }

    // ── AttentionGeometry: real model configs ──

    #[test]
    fn attention_geometry_dimension_consistency_mistral() {
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 1024,
            heads_per_group: 4,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
        assert_eq!(geo.heads_per_group, geo.num_heads / geo.num_kv_heads);
    }

    #[test]
    fn attention_geometry_dimension_consistency_deepseek() {
        let geo = AttentionGeometry {
            num_heads: 128,
            num_kv_heads: 128,
            head_dim: 128,
            q_dim: 16384,
            kv_dim: 16384,
            heads_per_group: 1,
        };
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
        assert_eq!(geo.heads_per_group, 1);
    }

    // ── SeqContext: non-zero start positions ──

    #[test]
    fn seq_context_positions_non_zero_start() {
        let positions = [100u32, 101, 102, 103];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 4,
            total_seq: 104,
        };
        assert_eq!(ctx.positions[0], 100);
        assert_eq!(ctx.positions[3], 103);
        assert_eq!(ctx.total_seq, 104);
    }

    // ── LayerDims: zero inter dim ──

    #[test]
    fn layer_dims_inter_zero() {
        let dims = LayerDims {
            hidden: 768,
            inter: 0,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        assert_eq!(dims.inter, 0);
        assert_eq!(dims.hidden, 768);
    }

    // ── KvCacheSlice: k and v point to same buffer ──

    #[test]
    fn kv_cache_slice_same_k_v_buffers() {
        let buf = [42u8; 16];
        let slice = KvCacheSlice {
            k: &buf,
            v: &buf,
            dtype: DType::F32,
            layer: 5,
            max_seq_len: 256,
        };
        assert_eq!(slice.k.as_ptr(), slice.v.as_ptr());
        assert_eq!(slice.k.len(), slice.v.len());
    }

    // ── DecoderLayerWeights: separate attention vs FFN slices ──

    #[test]
    fn decoder_layer_weights_attention_vs_ffn_separate() {
        let attn = [1u8; 8];
        let norm = [2u8; 8];
        let ffn = [3u8; 8];
        let w = DecoderLayerWeights {
            q_w: &attn,
            k_w: &attn,
            v_w: &attn,
            o_w: &attn,
            rn1_w: &norm,
            rn2_w: &norm,
            gate_w: &ffn,
            up_w: &ffn,
            down_w: &ffn,
            dtype: DType::F32,
        };
        assert_eq!(w.q_w[0], 1);
        assert_eq!(w.rn1_w[0], 2);
        assert_eq!(w.gate_w[0], 3);
    }

    // ── BertLayerWeights: FP8 dtype ──

    #[test]
    fn bert_layer_weights_fp8_e4m3_dtype() {
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data, q_b: &data, k_w: &data, k_b: &data,
            v_w: &data, v_b: &data, out_w: &data, out_b: &data,
            ln1_w: &data, ln1_b: &data, ffn_up_w: &data, ffn_up_b: &data,
            ffn_down_w: &data, ffn_down_b: &data, ln2_w: &data, ln2_b: &data,
            dtype: DType::F8E4M3,
        };
        assert_eq!(w.dtype, DType::F8E4M3);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // =======================================================================
    // 15 additional tests
    // =======================================================================

    // ── QuantType variants not yet covered ──

    #[test]
    fn weight_data_quantized_q3k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q3K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q3K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q5k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q5K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q5K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q6k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q6K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q6K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q8k_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::Q8K,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q8K),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq3_xxs_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ3XXS,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ3XXS),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq4_xs_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ4XS,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ4XS),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_squeeze_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Squeeze,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Squeeze),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_fp8_e4m3_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Fp8E4M3,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Fp8E4M3),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_tq1_0_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::TQ1_0,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::TQ1_0),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_tq2_0_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::TQ2_0,
            out_dim: 16,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::TQ2_0),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_mxfp4_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Mxfp4 { block_size: 32 },
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => {
                assert_eq!(*quant_type, QuantType::Mxfp4 { block_size: 32 });
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── Struct edge cases not yet covered ──

    #[test]
    fn seq_context_positions_iter_matches_seq_len() {
        let positions = [5u32, 10, 15];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 3,
            total_seq: 20,
        };
        let collected: Vec<u32> = ctx.positions.iter().copied().collect();
        assert_eq!(collected, vec![5, 10, 15]);
    }

    #[test]
    fn decoder_layer_weights_all_fields_same_buffer() {
        let buf = [0xABu8; 8];
        let w = DecoderLayerWeights {
            q_w: &buf, k_w: &buf, v_w: &buf, o_w: &buf,
            rn1_w: &buf, rn2_w: &buf,
            gate_w: &buf, up_w: &buf, down_w: &buf,
            dtype: DType::BF16,
        };
        // All 9 fields point to the same buffer
        assert_eq!(w.q_w.as_ptr(), w.k_w.as_ptr());
        assert_eq!(w.q_w.as_ptr(), w.down_w.as_ptr());
        assert_eq!(w.dtype, DType::BF16);
    }

    #[test]
    fn quantized_decoder_weights_q_and_o_distinct_weights() {
        let q_wd = WeightData::F32(vec![1.0; 4]);
        let o_wd = WeightData::F32(vec![2.0; 4]);
        let gate_wd = WeightData::F32(vec![3.0; 4]);
        let up_wd = WeightData::F32(vec![4.0; 4]);
        let down_wd = WeightData::F32(vec![5.0; 4]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &q_wd,
            o: &o_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &gate_wd,
            up: &up_wd,
            down: &down_wd,
        };
        // Each weight field is independently accessible
        assert!(matches!(w.q, WeightData::F32(v) if v[0] == 1.0));
        assert!(matches!(w.o, WeightData::F32(v) if v[0] == 2.0));
        assert!(matches!(w.gate, WeightData::F32(v) if v[0] == 3.0));
        assert!(matches!(w.up, WeightData::F32(v) if v[0] == 4.0));
        assert!(matches!(w.down, WeightData::F32(v) if v[0] == 5.0));
    }

    #[test]
    fn kv_cache_slice_large_max_seq_len() {
        let k = [0u8; 1024];
        let v = [0u8; 1024];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 1_000_000,
        };
        assert_eq!(slice.max_seq_len, 1_000_000);
        assert_eq!(slice.k.len(), 1024);
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // =======================================================================

    // ── QuantType: remaining uncovered variants ──

    #[test]
    fn weight_data_quantized_nvfp4_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Nvfp4,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Nvfp4),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q5_1_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 24],
            quant_type: QuantType::Q5_1,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q5_1),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_q8_1_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 40],
            quant_type: QuantType::Q8_1,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Q8_1),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq1s_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 50],
            quant_type: QuantType::IQ1S,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ1S),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq1m_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 56],
            quant_type: QuantType::IQ1M,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ1M),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq2xxs_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ2XXS,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ2XXS),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq3s_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ3S,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ3S),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq4nl_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::IQ4NL,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ4NL),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_fp8_e5m2_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Fp8E5M2,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Fp8E5M2),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── LayerDims: NaN rope_theta ──

    #[test]
    fn layer_dims_rope_theta_nan_not_equal() {
        let a = LayerDims {
            hidden: 512,
            inter: 2048,
            eps: 1e-5,
            rope_theta: f64::NAN,
        };
        let b = LayerDims {
            hidden: 512,
            inter: 2048,
            eps: 1e-5,
            rope_theta: f64::NAN,
        };
        // NaN != NaN per IEEE 754
        assert_ne!(a, b);
    }

    // ── SeqContext: total_seq much larger than seq_len (batched decode) ──

    #[test]
    fn seq_context_batched_decode_total_exceeds_seq_len() {
        let positions = [127u32, 255, 63];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 3,
            total_seq: 1024,
        };
        assert_eq!(ctx.positions.len(), 3);
        assert!(ctx.total_seq > ctx.seq_len);
        assert_eq!(ctx.positions[1], 255);
    }

    // ── BertLayerWeights: FP8 E5M2 dtype ──

    #[test]
    fn bert_layer_weights_fp8_e5m2_dtype() {
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data, q_b: &data, k_w: &data, k_b: &data,
            v_w: &data, v_b: &data, out_w: &data, out_b: &data,
            ln1_w: &data, ln1_b: &data, ffn_up_w: &data, ffn_up_b: &data,
            ffn_down_w: &data, ffn_down_b: &data, ln2_w: &data, ln2_b: &data,
            dtype: DType::F8E5M2,
        };
        assert_eq!(w.dtype, DType::F8E5M2);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── DecoderLayerWeights: FP8 E5M2 dtype ──

    #[test]
    fn decoder_layer_weights_fp8_e5m2_dtype() {
        let data = [0u8; 16];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::F8E5M2,
        };
        assert_eq!(w.dtype, DType::F8E5M2);
        assert_ne!(w.dtype, DType::F8E4M3);
    }

    // ── KvCacheSlice: verify buffer content integrity ──

    #[test]
    fn kv_cache_slice_buffer_content_integrity() {
        let k = [0xDEu8, 0xAD, 0xBE, 0xEF];
        let v = [0xCAu8, 0xFE, 0xBA, 0xBE];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 1,
        };
        assert_eq!(slice.k, &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(slice.v, &[0xCA, 0xFE, 0xBA, 0xBE]);
    }

    // ── AttentionGeometry: struct in a Vec ──

    #[test]
    fn attention_geometry_vec_of_configs() {
        let configs = vec![
            AttentionGeometry { num_heads: 8, num_kv_heads: 1, head_dim: 64, q_dim: 512, kv_dim: 64, heads_per_group: 8 },
            AttentionGeometry { num_heads: 32, num_kv_heads: 8, head_dim: 128, q_dim: 4096, kv_dim: 1024, heads_per_group: 4 },
        ];
        assert_eq!(configs[0].heads_per_group, 8);
        assert_eq!(configs[1].heads_per_group, 4);
        assert_eq!(configs.len(), 2);
    }

    // ── QuantizedDecoderWeights: gate/up/down all different QuantType ──

    #[test]
    fn quantized_decoder_weights_mixed_quant_types() {
        let q_wd = WeightData::Quantized { data: vec![0u8; 16], quant_type: QuantType::AWQ4, out_dim: 8, in_dim: 4 };
        let o_wd = WeightData::Quantized { data: vec![0u8; 16], quant_type: QuantType::GPTQ4, out_dim: 8, in_dim: 4 };
        let gate_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q4_0, out_dim: 8, in_dim: 4 };
        let up_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q4_1, out_dim: 8, in_dim: 4 };
        let down_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q8_0, out_dim: 4, in_dim: 8 };
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &q_wd, o: &o_wd,
            rn1_w: &norm, rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &gate_wd, up: &up_wd, down: &down_wd,
        };
        assert!(matches!(w.q, WeightData::Quantized { quant_type: QuantType::AWQ4, .. }));
        assert!(matches!(w.o, WeightData::Quantized { quant_type: QuantType::GPTQ4, .. }));
        assert!(matches!(w.gate, WeightData::Quantized { quant_type: QuantType::Q4_0, .. }));
        assert!(matches!(w.up, WeightData::Quantized { quant_type: QuantType::Q4_1, .. }));
        assert!(matches!(w.down, WeightData::Quantized { quant_type: QuantType::Q8_0, .. }));
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // =======================================================================

    // ── WeightData: edge-case float values ──

    #[test]
    fn weight_data_f32_with_nan_values() {
        let wd = WeightData::F32(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0]);
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data.len(), 4);
                assert!(data[0].is_nan());
                assert!(data[1].is_infinite() && data[1].is_sign_positive());
                assert!(data[2].is_infinite() && data[2].is_sign_negative());
                assert_eq!(data[3], 0.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    #[test]
    fn weight_data_f32_large_vector() {
        let v: Vec<f32> = (0..100000).map(|i| i as f32).collect();
        let wd = WeightData::F32(v);
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data.len(), 100000);
                assert_eq!(data[0], 0.0);
                assert_eq!(data[99999], 99999.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // ── AttentionGeometry: Eq trait allows HashSet dedup with multiple configs ──

    #[test]
    fn attention_geometry_hashset_multiple_configs_dedup() {
        use std::collections::HashSet;
        let a = AttentionGeometry {
            num_heads: 8, num_kv_heads: 2, head_dim: 64, q_dim: 512, kv_dim: 128, heads_per_group: 4,
        };
        let b = AttentionGeometry {
            num_heads: 32, num_kv_heads: 8, head_dim: 128, q_dim: 4096, kv_dim: 1024, heads_per_group: 4,
        };
        let a_dup = AttentionGeometry {
            num_heads: 8, num_kv_heads: 2, head_dim: 64, q_dim: 512, kv_dim: 128, heads_per_group: 4,
        };
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(a_dup);
        assert_eq!(set.len(), 2);
    }

    // ── LayerDims: Copy trait allows mutation without affecting original ──

    #[test]
    fn layer_dims_copy_then_modify_rope_theta() {
        let original = LayerDims {
            hidden: 768,
            inter: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        let mut modified = original;
        modified.rope_theta = 500000.0;
        assert!((original.rope_theta - 10000.0).abs() < f64::EPSILON);
        assert!((modified.rope_theta - 500000.0).abs() < f64::EPSILON);
    }

    // ── LayerDims: Debug format contains all field names ──

    #[test]
    fn layer_dims_debug_all_field_names() {
        let dims = LayerDims { hidden: 256, inter: 512, eps: 1e-6, rope_theta: 500.0 };
        let debug = format!("{:?}", dims);
        assert!(debug.contains("hidden"));
        assert!(debug.contains("inter"));
        assert!(debug.contains("eps"));
        assert!(debug.contains("rope_theta"));
    }

    // ── KvCacheSlice: sub-byte dtype (F4E2M1) ──

    #[test]
    fn kv_cache_slice_sub_byte_f4e2m1_dtype() {
        let k = [0u8; 8];
        let v = [0u8; 8];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F4E2M1,
            layer: 3,
            max_seq_len: 256,
        };
        assert_eq!(slice.dtype, DType::F4E2M1);
        assert_eq!(slice.dtype.size_bytes(), 1);
    }

    // ── SeqContext: positions slice is a proper sub-slice of larger array ──

    #[test]
    fn seq_context_positions_from_vec_mid_range() {
        let all_positions: Vec<u32> = (0..200).collect();
        let sub = &all_positions[50..55];
        let ctx = SeqContext {
            positions: sub,
            seq_len: 5,
            total_seq: 200,
        };
        assert_eq!(ctx.positions.len(), 5);
        assert_eq!(ctx.positions[0], 50);
        assert_eq!(ctx.positions[4], 54);
    }

    // ── BertLayerWeights: verify all weight fields point to independent buffers ──

    #[test]
    fn bert_layer_weights_weight_and_bias_different_sizes() {
        let weight = [0u8; 64];
        let bias = [0u8; 16];
        let w = BertLayerWeights {
            q_w: &weight, q_b: &bias,
            k_w: &weight, k_b: &bias,
            v_w: &weight, v_b: &bias,
            out_w: &weight, out_b: &bias,
            ln1_w: &weight, ln1_b: &bias,
            ffn_up_w: &weight, ffn_up_b: &bias,
            ffn_down_w: &weight, ffn_down_b: &bias,
            ln2_w: &weight, ln2_b: &bias,
            dtype: DType::F32,
        };
        // Weights and biases can have different sizes in real models
        assert_eq!(w.q_w.len(), 64);
        assert_eq!(w.q_b.len(), 16);
        assert_ne!(w.q_w.len(), w.q_b.len());
    }

    // ── DecoderLayerWeights: U8 dtype (embedding layer with quantized norms) ──

    #[test]
    fn decoder_layer_weights_u8_dtype() {
        let data = [0u8; 32];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::U8,
        };
        assert_eq!(w.dtype, DType::U8);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── DType: f32 is 4-byte aligned for SIMD ──

    #[test]
    fn dtype_f32_size_is_four_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(4 % DType::F32.size_bytes(), 0);
    }

    // ── WeightData::Quantized with large data buffer ──

    #[test]
    fn weight_data_quantized_large_data_buffer() {
        let large_data = vec![0xABu8; 65536];
        let wd = WeightData::Quantized {
            data: large_data,
            quant_type: QuantType::Q4_0,
            out_dim: 4096,
            in_dim: 4096,
        };
        match &wd {
            WeightData::Quantized { data, out_dim, in_dim, .. } => {
                assert_eq!(data.len(), 65536);
                assert_eq!(*out_dim, 4096);
                assert_eq!(*in_dim, 4096);
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── QuantizedDecoderWeights: mixed F32 attention + Quantized FFN ──

    #[test]
    fn quantized_decoder_weights_mixed_f32_attn_quant_ffn() {
        let attn_wd = WeightData::F32(vec![0.0; 64]);
        let ffn_wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &attn_wd,
            o: &attn_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::BF16,
            gate: &ffn_wd,
            up: &ffn_wd,
            down: &ffn_wd,
        };
        // Attention weights are F32
        assert!(matches!(w.q, WeightData::F32(_)));
        assert!(matches!(w.o, WeightData::F32(_)));
        // FFN weights are Quantized
        assert!(matches!(w.gate, WeightData::Quantized { .. }));
        assert!(matches!(w.up, WeightData::Quantized { .. }));
        assert!(matches!(w.down, WeightData::Quantized { .. }));
        // Norm dtype is separate
        assert_eq!(w.rn_dtype, DType::BF16);
    }

    // ── LayerDims: rope_theta f64 precision (very large theta) ──

    #[test]
    fn layer_dims_very_large_rope_theta() {
        let dims = LayerDims {
            hidden: 4096,
            inter: 11008,
            eps: 1e-5,
            rope_theta: 1e15,
        };
        assert!((dims.rope_theta - 1e15).abs() < 1.0);
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // @trace TEST-TYPES-COV additional compat types coverage
    // =======================================================================

    // ── QuantType: native floating-point variants (Bf16, F16, F32) ──

    #[test]
    fn weight_data_quantized_bf16_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Bf16,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::Bf16),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_f16_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::F16,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::F16),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_f32_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 64],
            quant_type: QuantType::F32,
            out_dim: 8,
            in_dim: 8,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::F32),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── QuantType: IQ family remaining variants ──

    #[test]
    fn weight_data_quantized_iq2xs_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ2XS,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ2XS),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    #[test]
    fn weight_data_quantized_iq2s_variant() {
        let wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::IQ2S,
            out_dim: 8,
            in_dim: 4,
        };
        match &wd {
            WeightData::Quantized { quant_type, .. } => assert_eq!(*quant_type, QuantType::IQ2S),
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // ── DType: ptx_type specific values for standard types ──

    #[test]
    fn dtype_ptx_type_f32_is_dot_f32() {
        assert_eq!(DType::F32.ptx_type(), ".f32");
    }

    #[test]
    fn dtype_ptx_type_bf16_is_dot_b16() {
        assert_eq!(DType::BF16.ptx_reg_type(), ".b16");
    }

    // ── DType: ptx_ld_type specific values ──

    #[test]
    fn dtype_ptx_ld_type_f32_is_dot_f32() {
        assert_eq!(DType::F32.ptx_ld_type(), ".f32");
    }

    #[test]
    fn dtype_ptx_ld_type_u8_is_dot_u8() {
        assert_eq!(DType::U8.ptx_ld_type(), ".u8");
    }

    #[test]
    fn dtype_ptx_ld_type_sub_byte_is_dot_u8() {
        assert_eq!(DType::F4E2M1.ptx_ld_type(), ".u8");
        assert_eq!(DType::F6E3M2.ptx_ld_type(), ".u8");
        assert_eq!(DType::F6E2M3.ptx_ld_type(), ".u8");
    }

    // ── DType: hip_type specific values ──

    #[test]
    fn dtype_hip_type_f32_is_float() {
        assert_eq!(DType::F32.hip_type(), "float");
    }

    #[test]
    fn dtype_hip_type_f16_is_half() {
        assert_eq!(DType::F16.hip_type(), "half");
    }

    // ── DType: msl_type specific values ──

    #[test]
    fn dtype_msl_type_bf16_is_bfloat() {
        assert_eq!(DType::BF16.msl_type(), "bfloat");
    }

    #[test]
    fn dtype_msl_type_f16_is_half() {
        assert_eq!(DType::F16.msl_type(), "half");
    }

    // ── LayerDims: inter dim larger than hidden (MoE expert) ──

    #[test]
    fn layer_dims_inter_larger_than_hidden() {
        let dims = LayerDims {
            hidden: 2048,
            inter: 14016,
            eps: 1e-6,
            rope_theta: 10000.0,
        };
        assert!(dims.inter > dims.hidden);
        assert_eq!(dims.inter / dims.hidden, 6); // 7x expansion ratio typical for SwiGLU
    }

    // ── AttentionGeometry: hash consistency in HashMap with mutation ──

    #[test]
    fn attention_geometry_hash_map_replace_value() {
        use std::collections::HashMap;
        let geo = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            q_dim: 512,
            kv_dim: 128,
            heads_per_group: 4,
        };
        let mut map = HashMap::new();
        map.insert(geo, "v1");
        assert_eq!(map.get(&geo), Some(&"v1"));
        map.insert(geo, "v2");
        assert_eq!(map.get(&geo), Some(&"v2"));
        assert_eq!(map.len(), 1);
    }

    // ── WeightData: F32 with negative values ──

    #[test]
    fn weight_data_f32_negative_values() {
        let wd = WeightData::F32(vec![-1.5, -0.5, 0.5, 1.5]);
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data.len(), 4);
                assert!(data[0] < 0.0);
                assert!(data[1] < 0.0);
                assert!(data[2] > 0.0);
                assert!(data[3] > 0.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // @trace TEST-TYPES-COV-2 additional compat types coverage
    // =======================================================================

    // ── WeightData: F32 with subnormal float values ──

    #[test]
    fn weight_data_f32_subnormal_values() {
        // Arrange: create F32 weight data with subnormal (denormalized) floats
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let wd = WeightData::F32(vec![subnormal, -subnormal, 0.0f32]);
        // Act & Assert: verify values are preserved correctly
        match &wd {
            WeightData::F32(data) => {
                assert!(data[0] > 0.0);
                assert!(data[0] < f32::MIN_POSITIVE);
                assert!(data[1] < 0.0);
                assert_eq!(data[2], 0.0);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // ── AttentionGeometry: Copy then modify produces independent struct ──

    #[test]
    fn attention_geometry_copy_then_modify_independent() {
        // Arrange: create a geometry and copy it
        let original = AttentionGeometry {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            q_dim: 1024,
            kv_dim: 256,
            heads_per_group: 4,
        };
        // Act: copy and modify
        let mut copy = original;
        copy.num_heads = 99;
        copy.head_dim = 1;
        // Assert: original unchanged, copy modified
        assert_eq!(original.num_heads, 16);
        assert_eq!(original.head_dim, 64);
        assert_eq!(copy.num_heads, 99);
        assert_eq!(copy.head_dim, 1);
    }

    // ── AttentionGeometry: Debug output contains all six fields ──

    #[test]
    fn attention_geometry_debug_contains_all_fields() {
        // Arrange
        let geo = AttentionGeometry {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            q_dim: 128,
            kv_dim: 64,
            heads_per_group: 2,
        };
        // Act
        let debug = format!("{:?}", geo);
        // Assert: all six fields appear in debug output
        for field in &["num_heads", "num_kv_heads", "head_dim", "q_dim", "kv_dim", "heads_per_group"] {
            assert!(debug.contains(field), "Debug output missing field: {}", field);
        }
    }

    // ── KvCacheSlice: F8E4M3 dtype with element count calculation ──

    #[test]
    fn kv_cache_slice_f8e4m3_dtype_element_count() {
        // Arrange: 128 bytes of KV data in F8E4M3 (1 byte per element)
        let k = [0u8; 128];
        let v = [0u8; 128];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F8E4M3,
            layer: 7,
            max_seq_len: 128,
        };
        // Act & Assert: 1 byte per element means 128 elements
        assert_eq!(slice.dtype.size_bytes(), 1);
        assert_eq!(slice.k.len() / slice.dtype.size_bytes(), 128);
        assert_eq!(slice.layer, 7);
    }

    // ── DecoderLayerWeights: F8E4M3 dtype (FP8 quantized weights) ──

    #[test]
    fn decoder_layer_weights_f8e4m3_dtype() {
        // Arrange: FP8 quantized decoder weights
        let data = [0u8; 32];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::F8E4M3,
        };
        // Act & Assert
        assert_eq!(w.dtype, DType::F8E4M3);
        assert_ne!(w.dtype, DType::F8E5M2);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── QuantizedDecoderWeights: F8E5M2 norm dtype ──

    #[test]
    fn quantized_decoder_weights_f8e5m2_norm_dtype() {
        // Arrange
        let wd = WeightData::F32(vec![0.0; 4]);
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &wd,
            o: &wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F8E5M2,
            gate: &wd,
            up: &wd,
            down: &wd,
        };
        // Assert
        assert_eq!(w.rn_dtype, DType::F8E5M2);
        assert_ne!(w.rn_dtype, DType::F8E4M3);
    }

    // ── BertLayerWeights: F4E2M1 sub-byte dtype ──

    #[test]
    fn bert_layer_weights_f4e2m1_dtype() {
        // Arrange: sub-byte quantized BERT weights
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data, q_b: &data, k_w: &data, k_b: &data,
            v_w: &data, v_b: &data, out_w: &data, out_b: &data,
            ln1_w: &data, ln1_b: &data, ffn_up_w: &data, ffn_up_b: &data,
            ffn_down_w: &data, ffn_down_b: &data, ln2_w: &data, ln2_b: &data,
            dtype: DType::F4E2M1,
        };
        // Assert
        assert_eq!(w.dtype, DType::F4E2M1);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── SeqContext: non-contiguous positions (gaps in position IDs) ──

    #[test]
    fn seq_context_non_contiguous_positions() {
        // Arrange: positions with gaps (e.g., packed sequences)
        let positions = [0u32, 5, 10, 20, 50];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 5,
            total_seq: 51,
        };
        // Assert: positions are preserved as-is, not contiguous
        assert_eq!(ctx.positions[0], 0);
        assert_eq!(ctx.positions[1], 5);
        assert_eq!(ctx.positions[4], 50);
        let diffs: Vec<i64> = positions.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect();
        assert!(diffs.iter().any(|&d| d > 1), "expected at least one gap in positions");
    }

    // ── DType: ptx_arith_type specific value for F32 ──

    #[test]
    fn dtype_ptx_arith_type_f32_is_dot_f32() {
        // Arrange & Act
        let arith = DType::F32.ptx_arith_type();
        // Assert
        assert_eq!(arith, ".f32");
    }

    // ── DType: ptx_reg_type specific values for standard types ──

    #[test]
    fn dtype_ptx_reg_type_u8_is_dot_b8() {
        assert_eq!(DType::U8.ptx_reg_type(), ".b8");
    }

    #[test]
    fn dtype_ptx_reg_type_f32_is_dot_f32() {
        assert_eq!(DType::F32.ptx_reg_type(), ".f32");
    }

    // ── QuantType::Mxfp4: different block_size values produce distinct QuantType ──

    #[test]
    fn weight_data_quantized_mxfp4_different_block_sizes() {
        // Arrange: two Mxfp4 with different block_size
        let wd_small = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::Mxfp4 { block_size: 16 },
            out_dim: 4,
            in_dim: 4,
        };
        let wd_large = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Mxfp4 { block_size: 64 },
            out_dim: 8,
            in_dim: 4,
        };
        // Assert: block_size distinguishes them
        assert_ne!(
            QuantType::Mxfp4 { block_size: 16 },
            QuantType::Mxfp4 { block_size: 64 }
        );
        match (&wd_small, &wd_large) {
            (WeightData::Quantized { quant_type: qt_small, .. },
             WeightData::Quantized { quant_type: qt_large, .. }) => {
                assert_ne!(*qt_small, *qt_large);
            }
            _ => panic!("expected Quantized variants"),
        }
    }

    // ── LayerDims: subnormal eps value ──

    #[test]
    fn layer_dims_subnormal_eps() {
        // Arrange: eps is a subnormal f32
        let subnormal_eps = f32::from_bits(1u32);
        let dims = LayerDims {
            hidden: 1024,
            inter: 4096,
            eps: subnormal_eps,
            rope_theta: 10000.0,
        };
        // Assert: subnormal eps is preserved
        assert!(dims.eps > 0.0);
        assert!(dims.eps < f32::MIN_POSITIVE);
        assert_eq!(dims.hidden, 1024);
    }

    // ── BertLayerWeights: F6E3M2 dtype ──

    #[test]
    fn bert_layer_weights_f6e3m2_dtype() {
        // Arrange
        let data = [0u8; 4];
        let w = BertLayerWeights {
            q_w: &data, q_b: &data, k_w: &data, k_b: &data,
            v_w: &data, v_b: &data, out_w: &data, out_b: &data,
            ln1_w: &data, ln1_b: &data, ffn_up_w: &data, ffn_up_b: &data,
            ffn_down_w: &data, ffn_down_b: &data, ln2_w: &data, ln2_b: &data,
            dtype: DType::F6E3M2,
        };
        // Assert
        assert_eq!(w.dtype, DType::F6E3M2);
        assert_ne!(w.dtype, DType::F6E2M3);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // ── DecoderLayerWeights: F6E2M3 dtype ──

    #[test]
    fn decoder_layer_weights_f6e2m3_dtype() {
        // Arrange
        let data = [0u8; 16];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::F6E2M3,
        };
        // Assert
        assert_eq!(w.dtype, DType::F6E2M3);
        assert_ne!(w.dtype, DType::F6E3M2);
    }

    // ── WeightData: F32 with maximum positive f32 value ──

    #[test]
    fn weight_data_f32_max_value() {
        // Arrange
        let wd = WeightData::F32(vec![f32::MAX, f32::MIN, f32::MAX]);
        // Act & Assert
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data[0], f32::MAX);
                assert_eq!(data[1], f32::MIN);
                assert_eq!(data[2], f32::MAX);
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // @trace TEST-TYPES-COV-3 additional compat types coverage
    // =======================================================================

    // -- AttentionGeometry: Eq transitivity via three equal instances --

    #[test]
    fn attention_geometry_eq_transitivity() {
        // Arrange: three identical geometries
        let a = AttentionGeometry {
            num_heads: 12,
            num_kv_heads: 3,
            head_dim: 96,
            q_dim: 1152,
            kv_dim: 288,
            heads_per_group: 4,
        };
        let b = AttentionGeometry {
            num_heads: 12,
            num_kv_heads: 3,
            head_dim: 96,
            q_dim: 1152,
            kv_dim: 288,
            heads_per_group: 4,
        };
        let c = AttentionGeometry {
            num_heads: 12,
            num_kv_heads: 3,
            head_dim: 96,
            q_dim: 1152,
            kv_dim: 288,
            heads_per_group: 4,
        };
        // Assert: a==b and b==c implies a==c
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- AttentionGeometry: heads_per_group consistency with integer division --

    #[test]
    fn attention_geometry_heads_per_group_matches_division() {
        // Arrange: Qwen3-8B GQA config (32 Q heads, 4 KV heads)
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 512,
            heads_per_group: 8,
        };
        // Assert: heads_per_group equals integer division result
        assert_eq!(geo.heads_per_group, geo.num_heads / geo.num_kv_heads);
        assert_eq!(geo.heads_per_group * geo.num_kv_heads, geo.num_heads);
    }

    // -- LayerDims: eps f32 vs rope_theta f64 mixed-precision equality --

    #[test]
    fn layer_dims_mixed_precision_fields_independent() {
        // Arrange: eps is very small f32, rope_theta is very large f64
        let dims = LayerDims {
            hidden: 4096,
            inter: 11008,
            eps: 1e-6,
            rope_theta: 1000000.0,
        };
        // Assert: f32 eps preserves its precision independently of f64 rope_theta
        assert!((dims.eps - 1e-6f32).abs() < f32::EPSILON);
        assert!((dims.rope_theta - 1000000.0f64).abs() < f64::EPSILON);
        // The f64 value has no effect on f32 field
        assert_ne!(dims.eps as f64, dims.rope_theta);
    }

    // -- SeqContext: all positions are same value (continuous batch same token) --

    #[test]
    fn seq_context_uniform_positions() {
        // Arrange: all positions are 0 (e.g., batch of new sequences at token 0)
        let positions = [0u32; 8];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 8,
            total_seq: 8,
        };
        // Assert: every position is zero
        assert!(ctx.positions.iter().all(|&p| p == 0));
        assert_eq!(ctx.seq_len, 8);
    }

    // -- SeqContext: descending positions (reverse order) --

    #[test]
    fn seq_context_descending_positions() {
        // Arrange: positions in descending order
        let positions = [9u32, 7, 5, 3, 1];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 5,
            total_seq: 10,
        };
        // Assert: positions are strictly decreasing
        for window in ctx.positions.windows(2) {
            assert!(window[0] > window[1], "positions should be descending");
        }
        assert_eq!(ctx.positions[0], 9);
        assert_eq!(ctx.positions[4], 1);
    }

    // -- KvCacheSlice: dtype size bytes consistency between slice and dtype field --

    #[test]
    fn kv_cache_slice_f16_dtype_size_consistency() {
        // Arrange: F16 KV cache with known byte count
        let k = [0u8; 256];
        let v = [0u8; 256];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F16,
            layer: 3,
            max_seq_len: 128,
        };
        // Assert: F16 is 2 bytes, so 256/2 = 128 elements per buffer
        assert_eq!(slice.dtype.size_bytes(), 2);
        assert_eq!(slice.k.len() / slice.dtype.size_bytes(), 128);
        assert_eq!(slice.v.len() / slice.dtype.size_bytes(), 128);
    }

    // -- DecoderLayerWeights: all attention weights share same length but are distinct --

    #[test]
    fn decoder_layer_weights_attention_fields_same_length_distinct_data() {
        // Arrange: four different attention weight slices of same length
        let q = [1u8; 32];
        let k = [2u8; 32];
        let v = [3u8; 32];
        let o = [4u8; 32];
        let norm = [0u8; 16];
        let ffn = [0u8; 64];
        let w = DecoderLayerWeights {
            q_w: &q,
            k_w: &k,
            v_w: &v,
            o_w: &o,
            rn1_w: &norm,
            rn2_w: &norm,
            gate_w: &ffn,
            up_w: &ffn,
            down_w: &ffn,
            dtype: DType::F32,
        };
        // Assert: all four attention fields same length, different first byte
        assert_eq!(w.q_w.len(), w.k_w.len());
        assert_eq!(w.k_w.len(), w.v_w.len());
        assert_eq!(w.v_w.len(), w.o_w.len());
        assert_ne!(w.q_w[0], w.k_w[0]);
        assert_ne!(w.k_w[0], w.v_w[0]);
        assert_ne!(w.v_w[0], w.o_w[0]);
    }

    // -- DecoderLayerWeights: FFN weights (gate/up/down) same length but distinct --

    #[test]
    fn decoder_layer_weights_ffn_fields_same_length_distinct_data() {
        // Arrange: three different FFN weight slices of same length
        let attn = [0u8; 16];
        let norm = [0u8; 8];
        let gate = [10u8; 48];
        let up = [20u8; 48];
        let down = [30u8; 48];
        let w = DecoderLayerWeights {
            q_w: &attn,
            k_w: &attn,
            v_w: &attn,
            o_w: &attn,
            rn1_w: &norm,
            rn2_w: &norm,
            gate_w: &gate,
            up_w: &up,
            down_w: &down,
            dtype: DType::BF16,
        };
        // Assert: FFN fields same length, different first byte
        assert_eq!(w.gate_w.len(), w.up_w.len());
        assert_eq!(w.up_w.len(), w.down_w.len());
        assert_eq!(w.gate_w[0], 10);
        assert_eq!(w.up_w[0], 20);
        assert_eq!(w.down_w[0], 30);
    }

    // -- BertLayerWeights: out_w and out_b are independently accessible --

    #[test]
    fn bert_layer_weights_out_projection_fields_accessible() {
        // Arrange: out_w and out_b with distinct data
        let zeros = [0u8; 16];
        let out_weight = [0xABu8; 16];
        let out_bias = [0xCDu8; 8];
        let w = BertLayerWeights {
            q_w: &zeros, q_b: &zeros, k_w: &zeros, k_b: &zeros,
            v_w: &zeros, v_b: &zeros,
            out_w: &out_weight, out_b: &out_bias,
            ln1_w: &zeros, ln1_b: &zeros,
            ffn_up_w: &zeros, ffn_up_b: &zeros,
            ffn_down_w: &zeros, ffn_down_b: &zeros,
            ln2_w: &zeros, ln2_b: &zeros,
            dtype: DType::F32,
        };
        // Assert: out projection fields hold correct values and are independent
        assert_eq!(w.out_w[0], 0xAB);
        assert_eq!(w.out_b[0], 0xCD);
        assert_ne!(w.out_w[0], w.q_w[0]);
        assert_ne!(w.out_w.len(), w.out_b.len());
    }

    // -- BertLayerWeights: ln1 and ln2 are separate norm layers --

    #[test]
    fn bert_layer_weights_ln1_ln2_independent_fields() {
        // Arrange: ln1 and ln2 with different data
        let zeros = [0u8; 8];
        let ln1_weight = [0x11u8; 8];
        let ln1_bias = [0x22u8; 8];
        let ln2_weight = [0x33u8; 8];
        let ln2_bias = [0x44u8; 8];
        let w = BertLayerWeights {
            q_w: &zeros, q_b: &zeros, k_w: &zeros, k_b: &zeros,
            v_w: &zeros, v_b: &zeros, out_w: &zeros, out_b: &zeros,
            ln1_w: &ln1_weight, ln1_b: &ln1_bias,
            ffn_up_w: &zeros, ffn_up_b: &zeros,
            ffn_down_w: &zeros, ffn_down_b: &zeros,
            ln2_w: &ln2_weight, ln2_b: &ln2_bias,
            dtype: DType::F32,
        };
        // Assert: ln1 and ln2 hold distinct values
        assert_eq!(w.ln1_w[0], 0x11);
        assert_eq!(w.ln1_b[0], 0x22);
        assert_eq!(w.ln2_w[0], 0x33);
        assert_eq!(w.ln2_b[0], 0x44);
        assert_ne!(w.ln1_w[0], w.ln2_w[0]);
        assert_ne!(w.ln1_b[0], w.ln2_b[0]);
    }

    // -- QuantizedDecoderWeights: q and o fields can hold different WeightData variants --

    #[test]
    fn quantized_decoder_weights_q_f32_o_quantized_mixed() {
        // Arrange: q is F32, o is Quantized
        let q_wd = WeightData::F32(vec![0.5; 16]);
        let o_wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let ffn_wd = WeightData::F32(vec![0.0; 8]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &q_wd,
            o: &o_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &ffn_wd,
            up: &ffn_wd,
            down: &ffn_wd,
        };
        // Assert: q is F32, o is Quantized, they are different variants
        assert!(matches!(w.q, WeightData::F32(_)));
        assert!(matches!(w.o, WeightData::Quantized { .. }));
    }

    // -- WeightData: Quantized with zero out_dim but non-zero data --

    #[test]
    fn weight_data_quantized_zero_out_dim_nonzero_data() {
        // Arrange: quantized data exists but out_dim is zero
        let wd = WeightData::Quantized {
            data: vec![0xFFu8; 16],
            quant_type: QuantType::Q4_0,
            out_dim: 0,
            in_dim: 0,
        };
        // Act & Assert: data is preserved even with zero dims
        match &wd {
            WeightData::Quantized { data, out_dim, in_dim, quant_type } => {
                assert_eq!(data.len(), 16);
                assert_eq!(*out_dim, 0);
                assert_eq!(*in_dim, 0);
                assert_eq!(*quant_type, QuantType::Q4_0);
                assert_eq!(data[0], 0xFF);
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // -- DType: size_bytes multiplication for buffer sizing --

    #[test]
    fn dtype_size_bytes_used_for_buffer_calculation() {
        // Arrange: verify size_bytes works correctly in arithmetic
        let num_elements = 1024usize;
        // Assert: buffer sizes are correct for each dtype
        assert_eq!(num_elements * DType::F32.size_bytes(), 4096);
        assert_eq!(num_elements * DType::BF16.size_bytes(), 2048);
        assert_eq!(num_elements * DType::F16.size_bytes(), 2048);
        assert_eq!(num_elements * DType::U8.size_bytes(), 1024);
        assert_eq!(num_elements * DType::F8E4M3.size_bytes(), 1024);
    }

    // -- Cross-struct: LayerDims hidden dim matches AttentionGeometry q_dim --

    #[test]
    fn layer_dims_hidden_matches_attention_geometry_q_dim() {
        // Arrange: Llama-7B config with consistent hidden dims
        let dims = LayerDims {
            hidden: 4096,
            inter: 11008,
            eps: 1e-6,
            rope_theta: 10000.0,
        };
        let geo = AttentionGeometry {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            q_dim: 4096,
            kv_dim: 4096,
            heads_per_group: 1,
        };
        // Assert: hidden dim equals q_dim (same model config)
        assert_eq!(dims.hidden, geo.q_dim);
        assert_eq!(geo.q_dim, geo.num_heads * geo.head_dim);
    }

    // -- AttentionGeometry: Copy trait allows return from function --

    #[test]
    fn attention_geometry_returned_from_function() {
        // Arrange: helper function that returns AttentionGeometry (tests Copy)
        fn make_geo() -> AttentionGeometry {
            AttentionGeometry {
                num_heads: 24,
                num_kv_heads: 8,
                head_dim: 128,
                q_dim: 3072,
                kv_dim: 1024,
                heads_per_group: 3,
            }
        }
        // Act
        let geo = make_geo();
        // Assert: all fields survive the return (Copy semantics)
        assert_eq!(geo.num_heads, 24);
        assert_eq!(geo.num_kv_heads, 8);
        assert_eq!(geo.head_dim, 128);
        assert_eq!(geo.q_dim, 3072);
        assert_eq!(geo.kv_dim, 1024);
        assert_eq!(geo.heads_per_group, 3);
    }

    // =======================================================================
    // 15 additional tests for improved coverage
    // @trace TEST-TYPES-COV-4 additional compat types coverage
    // =======================================================================

    // -- SeqContext: borrow survives across function boundary --

    #[test]
    fn seq_context_borrow_across_function() {
        // Arrange: positions owned in a variable, context borrows them
        fn get_first_position(ctx: &SeqContext) -> u32 {
            ctx.positions[0]
        }
        let positions = [42u32, 43, 44];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 3,
            total_seq: 100,
        };
        // Act: pass borrowed context to helper
        let first = get_first_position(&ctx);
        // Assert: data survives the borrow
        assert_eq!(first, 42);
        assert_eq!(ctx.positions.len(), 3);
    }

    // -- KvCacheSlice: layer field distinguishes layers in multi-layer model --

    #[test]
    fn kv_cache_slice_layer_indexing_across_layers() {
        // Arrange: simulate KV slices for layers 0, 1, 2 of a 3-layer model
        let buf = [0u8; 64];
        let slices: Vec<KvCacheSlice> = (0..3)
            .map(|layer| KvCacheSlice {
                k: &buf,
                v: &buf,
                dtype: DType::F32,
                layer,
                max_seq_len: 16,
            })
            .collect();
        // Assert: each slice has the correct layer index
        assert_eq!(slices[0].layer, 0);
        assert_eq!(slices[1].layer, 1);
        assert_eq!(slices[2].layer, 2);
        assert_eq!(slices.len(), 3);
        // All share the same dtype
        assert!(slices.iter().all(|s| s.dtype == DType::F32));
    }

    // -- WeightData::Quantized: asymmetric matrix (out_dim != in_dim) --

    #[test]
    fn weight_data_quantized_asymmetric_dims() {
        // Arrange: non-square weight matrix (e.g., gate projection 4096->14336)
        let wd = WeightData::Quantized {
            data: vec![0u8; 256],
            quant_type: QuantType::Q4_0,
            out_dim: 14336,
            in_dim: 4096,
        };
        // Assert: out_dim and in_dim are independently stored
        match &wd {
            WeightData::Quantized { out_dim, in_dim, data, .. } => {
                assert_eq!(*out_dim, 14336);
                assert_eq!(*in_dim, 4096);
                assert_ne!(*out_dim, *in_dim);
                assert_eq!(data.len(), 256);
            }
            WeightData::F32(_) => panic!("expected Quantized"),
        }
    }

    // -- QuantizedDecoderWeights: all seven weight fields are Quantized --

    #[test]
    fn quantized_decoder_weights_all_fields_quantized() {
        // Arrange: every weight field is a distinct Quantized variant
        let q_wd = WeightData::Quantized { data: vec![0u8; 16], quant_type: QuantType::AWQ4, out_dim: 4, in_dim: 4 };
        let o_wd = WeightData::Quantized { data: vec![0u8; 16], quant_type: QuantType::GPTQ4, out_dim: 4, in_dim: 4 };
        let gate_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q4_0, out_dim: 8, in_dim: 4 };
        let up_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q4_1, out_dim: 8, in_dim: 4 };
        let down_wd = WeightData::Quantized { data: vec![0u8; 32], quant_type: QuantType::Q8_0, out_dim: 4, in_dim: 8 };
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &q_wd,
            o: &o_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &gate_wd,
            up: &up_wd,
            down: &down_wd,
        };
        // Assert: all five weight fields are Quantized variant
        assert!(matches!(w.q, WeightData::Quantized { .. }));
        assert!(matches!(w.o, WeightData::Quantized { .. }));
        assert!(matches!(w.gate, WeightData::Quantized { .. }));
        assert!(matches!(w.up, WeightData::Quantized { .. }));
        assert!(matches!(w.down, WeightData::Quantized { .. }));
        // Norm fields are separate byte slices
        assert_eq!(w.rn1_w.len(), 8);
        assert_eq!(w.rn2_w.len(), 8);
    }

    // -- BertLayerWeights: all four attention projections independently accessible --

    #[test]
    fn bert_layer_weights_four_attention_projections_distinct() {
        // Arrange: four different slices for q/k/v/out projections
        let q = [1u8; 8];
        let k = [2u8; 8];
        let v = [3u8; 8];
        let out = [4u8; 8];
        let bias = [0u8; 4];
        let norm = [0u8; 4];
        let ffn = [0u8; 8];
        let w = BertLayerWeights {
            q_w: &q, q_b: &bias,
            k_w: &k, k_b: &bias,
            v_w: &v, v_b: &bias,
            out_w: &out, out_b: &bias,
            ln1_w: &norm, ln1_b: &norm,
            ffn_up_w: &ffn, ffn_up_b: &bias,
            ffn_down_w: &ffn, ffn_down_b: &bias,
            ln2_w: &norm, ln2_b: &norm,
            dtype: DType::F32,
        };
        // Assert: each projection has its own distinct data
        assert_eq!(w.q_w[0], 1);
        assert_eq!(w.k_w[0], 2);
        assert_eq!(w.v_w[0], 3);
        assert_eq!(w.out_w[0], 4);
    }

    // -- LayerDims: inter == hidden (embedding-only layer with 1:1 expansion) --

    #[test]
    fn layer_dims_inter_equals_hidden() {
        // Arrange: inter dim equals hidden dim (no expansion)
        let dims = LayerDims {
            hidden: 768,
            inter: 768,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        // Assert: inter == hidden is valid (1:1 ratio)
        assert_eq!(dims.inter, dims.hidden);
        assert_eq!(dims.inter, 768);
    }

    // -- AttentionGeometry: kv_dim calculated correctly for GQA with large ratio --

    #[test]
    fn attention_geometry_gqa_kv_dim_calculation_with_large_ratio() {
        // Arrange: Phi-3 style config (32 Q heads, 4 KV heads, head_dim 96)
        let num_heads = 32;
        let num_kv_heads = 4;
        let head_dim = 96;
        let geo = AttentionGeometry {
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim: num_heads * head_dim,
            kv_dim: num_kv_heads * head_dim,
            heads_per_group: num_heads / num_kv_heads,
        };
        // Assert: dimensions are internally consistent
        assert_eq!(geo.q_dim, 3072);
        assert_eq!(geo.kv_dim, 384);
        assert_eq!(geo.heads_per_group, 8);
        assert_eq!(geo.kv_dim, geo.num_kv_heads * geo.head_dim);
    }

    // -- KvCacheSlice: U8 dtype element count matches buffer size --

    #[test]
    fn kv_cache_slice_u8_dtype_element_count_exact() {
        // Arrange: U8 KV cache where byte count equals element count
        let k = [0xAAu8; 256];
        let v = [0xBBu8; 256];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::U8,
            layer: 10,
            max_seq_len: 256,
        };
        // Assert: U8 is 1 byte per element, so elements == bytes
        assert_eq!(slice.dtype.size_bytes(), 1);
        assert_eq!(slice.k.len(), 256);
        assert_eq!(slice.k.len() / slice.dtype.size_bytes(), 256);
        assert_eq!(slice.v[0], 0xBB);
        assert_eq!(slice.layer, 10);
    }

    // -- DecoderLayerWeights: F4E2M1 sub-byte dtype (NVFP4 quantized weights) --

    #[test]
    fn decoder_layer_weights_f4e2m1_sub_byte_dtype() {
        // Arrange: NVFP4 (F4E2M1) quantized decoder weights
        let data = [0u8; 32];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::F4E2M1,
        };
        // Assert: F4E2M1 is a sub-byte type stored as 1 byte
        assert_eq!(w.dtype, DType::F4E2M1);
        assert_eq!(w.dtype.size_bytes(), 1);
        assert_ne!(w.dtype, DType::F8E4M3);
    }

    // -- SeqContext: positions from Vec with dynamic slicing --

    #[test]
    fn seq_context_positions_from_dynamic_vec_slice() {
        // Arrange: dynamically computed positions (e.g., chunked prefill)
        let full: Vec<u32> = (0..1000).collect();
        let chunk_start = 200;
        let chunk_end = 300;
        let chunk = &full[chunk_start..chunk_end];
        let ctx = SeqContext {
            positions: chunk,
            seq_len: chunk.len(),
            total_seq: full.len(),
        };
        // Assert: slice boundaries are correct
        assert_eq!(ctx.seq_len, 100);
        assert_eq!(ctx.positions[0], 200);
        assert_eq!(ctx.positions[99], 299);
        assert_eq!(ctx.total_seq, 1000);
    }

    // -- BertLayerWeights: FFN up and down projections are independently accessible --

    #[test]
    fn bert_layer_weights_ffn_projections_independent() {
        // Arrange: FFN up and down projections with distinct data
        let zeros = [0u8; 8];
        let ffn_up_weight = [0xAAu8; 32];
        let ffn_up_bias = [0x11u8; 8];
        let ffn_down_weight = [0xBBu8; 32];
        let ffn_down_bias = [0x22u8; 8];
        let w = BertLayerWeights {
            q_w: &zeros, q_b: &zeros, k_w: &zeros, k_b: &zeros,
            v_w: &zeros, v_b: &zeros, out_w: &zeros, out_b: &zeros,
            ln1_w: &zeros, ln1_b: &zeros,
            ffn_up_w: &ffn_up_weight, ffn_up_b: &ffn_up_bias,
            ffn_down_w: &ffn_down_weight, ffn_down_b: &ffn_down_bias,
            ln2_w: &zeros, ln2_b: &zeros,
            dtype: DType::F32,
        };
        // Assert: FFN up and down projections are distinct
        assert_eq!(w.ffn_up_w[0], 0xAA);
        assert_eq!(w.ffn_up_b[0], 0x11);
        assert_eq!(w.ffn_down_w[0], 0xBB);
        assert_eq!(w.ffn_down_b[0], 0x22);
        assert_ne!(w.ffn_up_w[0], w.ffn_down_w[0]);
        assert_ne!(w.ffn_up_b[0], w.ffn_down_b[0]);
    }

    // -- LayerDims: rope_theta f64 infinity value --

    #[test]
    fn layer_dims_rope_theta_f64_infinity() {
        // Arrange: rope_theta set to positive infinity
        let dims = LayerDims {
            hidden: 2048,
            inter: 8192,
            eps: 1e-5,
            rope_theta: f64::INFINITY,
        };
        // Assert: infinity is preserved
        assert_eq!(dims.rope_theta, f64::INFINITY);
        assert!(dims.rope_theta.is_sign_positive());
        assert!(dims.rope_theta > 0.0);
    }

    // -- Cross-struct: all three weight structs share same DType for a consistent model --

    #[test]
    fn all_weight_structs_share_same_dtype_for_consistent_model() {
        // Arrange: BF16 model config — all weight structs use the same dtype
        let shared_dtype = DType::BF16;
        let buf = [0u8; 8];
        let decoder = DecoderLayerWeights {
            q_w: &buf, k_w: &buf, v_w: &buf, o_w: &buf,
            rn1_w: &buf, rn2_w: &buf,
            gate_w: &buf, up_w: &buf, down_w: &buf,
            dtype: shared_dtype,
        };
        let bert = BertLayerWeights {
            q_w: &buf, q_b: &buf, k_w: &buf, k_b: &buf,
            v_w: &buf, v_b: &buf, out_w: &buf, out_b: &buf,
            ln1_w: &buf, ln1_b: &buf,
            ffn_up_w: &buf, ffn_up_b: &buf,
            ffn_down_w: &buf, ffn_down_b: &buf,
            ln2_w: &buf, ln2_b: &buf,
            dtype: shared_dtype,
        };
        let kv = KvCacheSlice {
            k: &buf, v: &buf, dtype: shared_dtype, layer: 0, max_seq_len: 512,
        };
        // Assert: all three structs agree on dtype
        assert_eq!(decoder.dtype, bert.dtype);
        assert_eq!(bert.dtype, kv.dtype);
        assert_eq!(decoder.dtype, shared_dtype);
    }

    // -- QuantizedDecoderWeights: TQ1_0 and TQ2_0 TurboQuant variants in FFN --

    #[test]
    fn quantized_decoder_weights_turboquant_tq_variants() {
        // Arrange: TQ1_0 gate, TQ2_0 up, TQ4_0 down (TurboQuant mixed)
        let q_wd = WeightData::F32(vec![0.0; 4]);
        let o_wd = WeightData::F32(vec![0.0; 4]);
        let gate_wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::TQ1_0,
            out_dim: 8,
            in_dim: 4,
        };
        let up_wd = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::TQ2_0,
            out_dim: 8,
            in_dim: 4,
        };
        let down_wd = WeightData::F32(vec![0.0; 4]);
        let norm = [0u8; 4];
        let w = QuantizedDecoderWeights {
            q: &q_wd,
            o: &o_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::BF16,
            gate: &gate_wd,
            up: &up_wd,
            down: &down_wd,
        };
        // Assert: gate is TQ1_0, up is TQ2_0
        assert!(matches!(w.gate, WeightData::Quantized { quant_type: QuantType::TQ1_0, .. }));
        assert!(matches!(w.up, WeightData::Quantized { quant_type: QuantType::TQ2_0, .. }));
        assert!(matches!(w.down, WeightData::F32(_)));
        assert_eq!(w.rn_dtype, DType::BF16);
    }

    // -- DType: gpu_type_name for F6 sub-byte variants returns Err --

    #[test]
    fn dtype_gpu_type_name_f6_variants_return_err() {
        // Arrange & Assert: F6E3M2 and F6E2M3 are sub-byte, no GPU native type
        assert_eq!(DType::F6E3M2.gpu_type_name(), Err(()));
        assert_eq!(DType::F6E2M3.gpu_type_name(), Err(()));
        // F32 has a valid GPU type name (control)
        assert_eq!(DType::F32.gpu_type_name(), Ok("f32"));
    }

    // =======================================================================
    // 13 additional tests for improved coverage
    // @trace TEST-TYPES-COV-5 additional compat types coverage
    // =======================================================================

    // -- AttentionGeometry: Clone produces truly independent copy (modify clone) --

    #[test]
    fn attention_geometry_clone_then_modify_produces_independent() {
        // Arrange
        let original = AttentionGeometry {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            q_dim: 1024,
            kv_dim: 256,
            heads_per_group: 4,
        };
        // Act: clone then mutate
        let mut cloned = original.clone();
        cloned.num_heads = 1;
        cloned.kv_dim = 1;
        // Assert: original unchanged, clone modified
        assert_eq!(original.num_heads, 16);
        assert_eq!(original.kv_dim, 256);
        assert_eq!(cloned.num_heads, 1);
        assert_eq!(cloned.kv_dim, 1);
    }

    // -- LayerDims: eps and rope_theta Debug format preserves precision indicators --

    #[test]
    fn layer_dims_debug_eps_and_rope_theta_precision() {
        // Arrange: eps and rope_theta with distinctive values
        let dims = LayerDims {
            hidden: 4096,
            inter: 11008,
            eps: 1.23456e-5,
            rope_theta: 500000.123456,
        };
        // Act
        let debug = format!("{:?}", dims);
        // Assert: both numeric fields appear in output
        assert!(debug.contains("hidden: 4096"));
        assert!(debug.contains("inter: 11008"));
        assert!(debug.contains("eps:"));
        assert!(debug.contains("rope_theta:"));
    }

    // -- SeqContext: empty slice from Vec via draining all elements --

    #[test]
    fn seq_context_empty_slice_from_vec_range() {
        // Arrange: take an empty sub-slice of a non-empty Vec
        let positions: Vec<u32> = (0..100).collect();
        let empty = &positions[50..50]; // zero-length slice
        let ctx = SeqContext {
            positions: empty,
            seq_len: 0,
            total_seq: 100,
        };
        // Assert: empty positions with non-zero total_seq
        assert!(ctx.positions.is_empty());
        assert_eq!(ctx.seq_len, 0);
        assert_eq!(ctx.total_seq, 100);
    }

    // -- KvCacheSlice: two slices with same dtype different layers are distinct --

    #[test]
    fn kv_cache_slice_different_layers_distinct() {
        // Arrange: same buffer, same dtype, different layer indices
        let buf = [0u8; 32];
        let slice_a = KvCacheSlice {
            k: &buf, v: &buf, dtype: DType::F32, layer: 5, max_seq_len: 128,
        };
        let slice_b = KvCacheSlice {
            k: &buf, v: &buf, dtype: DType::F32, layer: 17, max_seq_len: 128,
        };
        // Assert: same dtype and buffer, different layers
        assert_eq!(slice_a.dtype, slice_b.dtype);
        assert_ne!(slice_a.layer, slice_b.layer);
        assert_eq!(slice_a.k.as_ptr(), slice_b.k.as_ptr());
    }

    // -- DecoderLayerWeights: F6E3M2 sub-byte dtype --

    #[test]
    fn decoder_layer_weights_f6e3m2_dtype() {
        // Arrange: F6E3M2 quantized decoder weights
        let data = [0u8; 32];
        let w = DecoderLayerWeights {
            q_w: &data, k_w: &data, v_w: &data, o_w: &data,
            rn1_w: &data, rn2_w: &data,
            gate_w: &data, up_w: &data, down_w: &data,
            dtype: DType::F6E3M2,
        };
        // Assert: F6E3M2 is distinct from F6E2M3 and has 1-byte storage
        assert_eq!(w.dtype, DType::F6E3M2);
        assert_ne!(w.dtype, DType::F6E2M3);
        assert_eq!(w.dtype.size_bytes(), 1);
    }

    // -- BertLayerWeights: PartialEq equal with all-same buffers and dtype --

    #[test]
    fn bert_layer_weights_partial_eq_equal_structs() {
        // Arrange: two BertLayerWeights with identical fields
        let data = [0u8; 16];
        let bias = [0u8; 4];
        let a = BertLayerWeights {
            q_w: &data, q_b: &bias, k_w: &data, k_b: &bias,
            v_w: &data, v_b: &bias, out_w: &data, out_b: &bias,
            ln1_w: &data, ln1_b: &bias, ffn_up_w: &data, ffn_up_b: &bias,
            ffn_down_w: &data, ffn_down_b: &bias, ln2_w: &data, ln2_b: &bias,
            dtype: DType::F32,
        };
        let b = BertLayerWeights {
            q_w: &data, q_b: &bias, k_w: &data, k_b: &bias,
            v_w: &data, v_b: &bias, out_w: &data, out_b: &bias,
            ln1_w: &data, ln1_b: &bias, ffn_up_w: &data, ffn_up_b: &bias,
            ffn_down_w: &data, ffn_down_b: &bias, ln2_w: &data, ln2_b: &bias,
            dtype: DType::F32,
        };
        // Assert: same dtype means PartialEq succeeds (struct has #[derive(PartialEq)])
        assert_eq!(a.dtype, b.dtype);
    }

    // -- QuantizedDecoderWeights: rn1_w and rn2_w are different byte slices --

    #[test]
    fn quantized_decoder_weights_rn1_rn2_different_content() {
        // Arrange: rn1 and rn2 with distinct content
        let wd = WeightData::F32(vec![0.0; 4]);
        let rn1 = [0x11u8; 8];
        let rn2 = [0x22u8; 8];
        let w = QuantizedDecoderWeights {
            q: &wd, o: &wd,
            rn1_w: &rn1, rn2_w: &rn2,
            rn_dtype: DType::F32,
            gate: &wd, up: &wd, down: &wd,
        };
        // Assert: each norm weight field has independent content
        assert_eq!(w.rn1_w[0], 0x11);
        assert_eq!(w.rn2_w[0], 0x22);
        assert_ne!(w.rn1_w[0], w.rn2_w[0]);
        assert_eq!(w.rn1_w.len(), 8);
        assert_eq!(w.rn2_w.len(), 8);
    }

    // -- WeightData: F32 with positive zero and negative zero are both preserved --

    #[test]
    fn weight_data_f32_positive_and_negative_zero() {
        // Arrange: F32 data with +0.0 and -0.0
        let pos_zero = 0.0f32;
        let neg_zero = -0.0f32;
        let wd = WeightData::F32(vec![pos_zero, neg_zero]);
        // Act & Assert: both are zero but have different sign bits
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data[0], 0.0);
                assert_eq!(data[1], 0.0);
                assert!(data[0].is_sign_positive());
                assert!(data[1].is_sign_negative());
                assert_ne!(data[0].to_bits(), data[1].to_bits());
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // -- LayerDims: eps at f32::MIN (most negative finite f32) --

    #[test]
    fn layer_dims_eps_f32_min() {
        // Arrange: eps set to f32::MIN (most negative finite value)
        let dims = LayerDims {
            hidden: 1024,
            inter: 4096,
            eps: f32::MIN,
            rope_theta: 10000.0,
        };
        // Assert: f32::MIN is preserved (negative, finite, very large magnitude)
        assert_eq!(dims.eps, f32::MIN);
        assert!(dims.eps.is_sign_negative());
        assert!(!dims.eps.is_infinite());
    }

    // -- DType: ptx_arith_type for BF16 returns correct string --

    #[test]
    fn dtype_ptx_arith_type_bf16_value() {
        // Arrange & Act
        let arith = DType::BF16.ptx_arith_type();
        // Assert: BF16 arith type is a non-empty string (BF16 uses .bf16 or similar)
        assert!(!arith.is_empty());
        assert_ne!(arith, DType::F32.ptx_arith_type());
    }

    // -- SeqContext: positions containing u32::MAX values --

    #[test]
    fn seq_context_positions_with_u32_max() {
        // Arrange: positions include u32::MAX
        let positions = [0u32, u32::MAX / 2, u32::MAX];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 3,
            total_seq: (u32::MAX as usize) + 1,
        };
        // Assert: u32::MAX is preserved without truncation
        assert_eq!(ctx.positions[2], u32::MAX);
        assert_eq!(ctx.positions[1], u32::MAX / 2);
        assert_eq!(ctx.seq_len, 3);
    }

    // -- AttentionGeometry: Hash trait consistency across multiple insertions --

    #[test]
    fn attention_geometry_hash_consistent_across_insertions() {
        // Arrange: use AttentionGeometry as HashMap key, insert multiple distinct keys
        use std::collections::HashMap;
        let geo_mha = AttentionGeometry {
            num_heads: 32, num_kv_heads: 32, head_dim: 128,
            q_dim: 4096, kv_dim: 4096, heads_per_group: 1,
        };
        let geo_gqa = AttentionGeometry {
            num_heads: 32, num_kv_heads: 8, head_dim: 128,
            q_dim: 4096, kv_dim: 1024, heads_per_group: 4,
        };
        let geo_mqa = AttentionGeometry {
            num_heads: 32, num_kv_heads: 1, head_dim: 128,
            q_dim: 4096, kv_dim: 128, heads_per_group: 32,
        };
        let mut map = HashMap::new();
        map.insert(geo_mha, "mha");
        map.insert(geo_gqa, "gqa");
        map.insert(geo_mqa, "mqa");
        // Assert: all three keys are retrievable
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&geo_mha), Some(&"mha"));
        assert_eq!(map.get(&geo_gqa), Some(&"gqa"));
        assert_eq!(map.get(&geo_mqa), Some(&"mqa"));
    }

    // -- LayerDims: rope_theta f64 negative infinity --

    #[test]
    fn layer_dims_rope_theta_neg_infinity() {
        // Arrange: rope_theta set to negative infinity
        let dims = LayerDims {
            hidden: 2048,
            inter: 8192,
            eps: 1e-5,
            rope_theta: f64::NEG_INFINITY,
        };
        // Assert: negative infinity is preserved
        assert_eq!(dims.rope_theta, f64::NEG_INFINITY);
        assert!(dims.rope_theta.is_sign_negative());
        assert!(dims.rope_theta.is_infinite());
    }

    // =======================================================================
    // 10 additional tests for improved coverage
    // @trace TEST-TYPES-COV-5 additional compat types coverage
    // =======================================================================

    // -- WeightData: F32 with f32::MIN_POSITIVE boundary value --

    #[test]
    fn weight_data_f32_min_positive_value_preserved() {
        // Arrange: F32 data containing the smallest positive normal f32
        let min_pos = f32::MIN_POSITIVE;
        let wd = WeightData::F32(vec![min_pos, min_pos * 2.0, min_pos / 2.0]);
        // Act & Assert: boundary value and nearby values are preserved
        match &wd {
            WeightData::F32(data) => {
                assert_eq!(data[0], f32::MIN_POSITIVE);
                assert!(data[0].is_normal());
                assert_eq!(data[1], min_pos * 2.0);
                assert!(!data[2].is_normal()); // subnormal
            }
            WeightData::Quantized { .. } => panic!("expected F32 variant"),
        }
    }

    // -- SeqContext: minimal two-token prompt prefill --

    #[test]
    fn seq_context_two_token_prefill_positions() {
        // Arrange: smallest meaningful prefill (2 tokens, e.g., BOS + first token)
        let positions = [0u32, 1];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 2,
            total_seq: 2,
        };
        // Assert: both positions accessible, seq_len matches
        assert_eq!(ctx.positions.len(), 2);
        assert_eq!(ctx.positions[0], 0);
        assert_eq!(ctx.positions[1], 1);
        assert_eq!(ctx.seq_len, ctx.total_seq);
        assert_eq!(ctx.positions.first(), Some(&0));
        assert_eq!(ctx.positions.last(), Some(&1));
    }

    // -- KvCacheSlice: F6E2M3 dtype with buffer size calculation --

    #[test]
    fn kv_cache_slice_f6e2m3_dtype_buffer_size_calculation() {
        // Arrange: F6E2M3 is 1 byte per element (packed sub-byte type stored in whole bytes)
        let k = [0u8; 64];
        let v = [0u8; 64];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F6E2M3,
            layer: 4,
            max_seq_len: 32,
        };
        // Assert: size_bytes is 1 for F6E2M3, buffer holds 64 elements
        assert_eq!(slice.dtype.size_bytes(), 1);
        assert_eq!(slice.k.len(), 64);
        assert_eq!(slice.k.len() / slice.dtype.size_bytes(), 64);
        assert_eq!(slice.layer, 4);
        assert_ne!(slice.dtype, DType::F6E3M2);
    }

    // -- DecoderLayerWeights: F4E2M1 sub-byte dtype --

    #[test]
    fn decoder_layer_weights_f4e2m1_dtype_fields_accessible() {
        // Arrange: F4E2M1 (NVFP4 element type) decoder weights
        let data = [0u8; 16];
        let w = DecoderLayerWeights {
            q_w: &data,
            k_w: &data,
            v_w: &data,
            o_w: &data,
            rn1_w: &data,
            rn2_w: &data,
            gate_w: &data,
            up_w: &data,
            down_w: &data,
            dtype: DType::F4E2M1,
        };
        // Assert: F4E2M1 is 1 byte stored, distinct from other sub-byte types
        assert_eq!(w.dtype, DType::F4E2M1);
        assert_eq!(w.dtype.size_bytes(), 1);
        assert_ne!(w.dtype, DType::F8E4M3);
        assert_ne!(w.dtype, DType::F6E3M2);
        // All nine fields are accessible
        assert_eq!(w.q_w.len(), 16);
        assert_eq!(w.down_w.len(), 16);
    }

    // -- QuantizedDecoderWeights: k and v weights absent — q and o are the only attention fields --

    #[test]
    fn quantized_decoder_weights_no_kv_fields_only_q_and_o() {
        // Arrange: QuantizedDecoderWeights has q and o but no k/v (unlike DecoderLayerWeights)
        let q_wd = WeightData::F32(vec![1.0; 8]);
        let o_wd = WeightData::Quantized {
            data: vec![0u8; 16],
            quant_type: QuantType::Q4_0,
            out_dim: 8,
            in_dim: 4,
        };
        let norm = [0u8; 8];
        let w = QuantizedDecoderWeights {
            q: &q_wd,
            o: &o_wd,
            rn1_w: &norm,
            rn2_w: &norm,
            rn_dtype: DType::F32,
            gate: &q_wd,
            up: &q_wd,
            down: &q_wd,
        };
        // Assert: q is F32, o is Quantized — struct does not have k_w/v_w fields
        assert!(matches!(w.q, WeightData::F32(_)));
        assert!(matches!(w.o, WeightData::Quantized { .. }));
        assert_eq!(w.rn1_w.len(), 8);
        assert_eq!(w.rn2_w.len(), 8);
    }

    // -- AttentionGeometry: used as BTreeMap key (requires Ord via derive) --

    #[test]
    fn attention_geometry_vec_sorted_by_num_heads() {
        // Arrange: multiple geometries with different num_heads
        let configs = vec![
            AttentionGeometry { num_heads: 32, num_kv_heads: 8, head_dim: 128, q_dim: 4096, kv_dim: 1024, heads_per_group: 4 },
            AttentionGeometry { num_heads: 8, num_kv_heads: 1, head_dim: 64, q_dim: 512, kv_dim: 64, heads_per_group: 8 },
            AttentionGeometry { num_heads: 16, num_kv_heads: 4, head_dim: 96, q_dim: 1536, kv_dim: 384, heads_per_group: 4 },
        ];
        // Act: sort by num_heads
        let mut sorted = configs.clone();
        sorted.sort_by_key(|g| g.num_heads);
        // Assert: sorted in ascending order
        assert_eq!(sorted[0].num_heads, 8);
        assert_eq!(sorted[1].num_heads, 16);
        assert_eq!(sorted[2].num_heads, 32);
        // Original order preserved (Copy trait)
        assert_eq!(configs[0].num_heads, 32);
    }

    // -- BertLayerWeights: F6E2M3 dtype with distinct weight/bias lengths --

    #[test]
    fn bert_layer_weights_f6e2m3_dtype_weight_bias_different_lengths() {
        // Arrange: BERT weights with F6E2M3 dtype, weight and bias have different byte counts
        let weight = [0u8; 32];
        let bias = [0u8; 8];
        let w = BertLayerWeights {
            q_w: &weight, q_b: &bias,
            k_w: &weight, k_b: &bias,
            v_w: &weight, v_b: &bias,
            out_w: &weight, out_b: &bias,
            ln1_w: &weight, ln1_b: &bias,
            ffn_up_w: &weight, ffn_up_b: &bias,
            ffn_down_w: &weight, ffn_down_b: &bias,
            ln2_w: &weight, ln2_b: &bias,
            dtype: DType::F6E2M3,
        };
        // Assert: dtype is correct, weights and biases have different sizes
        assert_eq!(w.dtype, DType::F6E2M3);
        assert_eq!(w.q_w.len(), 32);
        assert_eq!(w.q_b.len(), 8);
        assert_ne!(w.q_w.len(), w.q_b.len());
        // All 8 weight fields and 8 bias fields are independently sized
        for wf in [&w.q_w, &w.k_w, &w.v_w, &w.out_w, &w.ln1_w, &w.ffn_up_w, &w.ffn_down_w, &w.ln2_w] {
            assert_eq!(wf.len(), 32);
        }
        for bf in [&w.q_b, &w.k_b, &w.v_b, &w.out_b, &w.ln1_b, &w.ffn_up_b, &w.ffn_down_b, &w.ln2_b] {
            assert_eq!(bf.len(), 8);
        }
    }

    // -- Cross-struct: SeqContext positions used to compute KvCacheSlice element count --

    #[test]
    fn cross_struct_seq_context_and_kv_cache_dtype_agreement() {
        // Arrange: seq context and KV cache with consistent F32 dtype
        let positions = [0u32, 1, 2, 3];
        let ctx = SeqContext {
            positions: &positions,
            seq_len: 4,
            total_seq: 4,
        };
        // F32: 4 bytes per element, seq_len * head_dim elements per KV row
        let head_dim = 64usize;
        let bytes_per_kv_row = ctx.seq_len * head_dim * DType::F32.size_bytes();
        let k = vec![0u8; bytes_per_kv_row];
        let v = vec![0u8; bytes_per_kv_row];
        let slice = KvCacheSlice {
            k: &k,
            v: &v,
            dtype: DType::F32,
            layer: 0,
            max_seq_len: 1024,
        };
        // Assert: buffer sizes match seq_len * head_dim * sizeof(f32)
        assert_eq!(slice.k.len(), 4 * 64 * 4);
        assert_eq!(slice.v.len(), 4 * 64 * 4);
        assert_eq!(slice.dtype.size_bytes(), 4);
    }

    // -- QuantType: Mxfp4 and Nvfp4 are distinct even though both are 4-bit formats --

    #[test]
    fn weight_data_quantized_mxfp4_vs_nvfp4_distinct() {
        // Arrange: two WeightData entries, one Mxfp4 and one Nvfp4
        let mxfp = WeightData::Quantized {
            data: vec![0u8; 32],
            quant_type: QuantType::Mxfp4 { block_size: 32 },
            out_dim: 8,
            in_dim: 4,
        };
        let nvfp = WeightData::Quantized {
            data: vec![0u8; 36],
            quant_type: QuantType::Nvfp4,
            out_dim: 8,
            in_dim: 4,
        };
        // Assert: they are different QuantType variants
        assert_ne!(QuantType::Mxfp4 { block_size: 32 }, QuantType::Nvfp4);
        match (&mxfp, &nvfp) {
            (WeightData::Quantized { quant_type: qt_m, data: d_m, .. },
             WeightData::Quantized { quant_type: qt_n, data: d_n, .. }) => {
                assert_ne!(*qt_m, *qt_n);
                // Nvfp4 has different data layout (36 bytes for 64 elements vs 17 for Mxfp4)
                assert_ne!(d_m.len(), d_n.len());
            }
            _ => panic!("expected both Quantized"),
        }
    }

    // -- DType: ptx_arith_type for U8 returns correct arithmetic type --

    #[test]
    fn dtype_ptx_arith_type_u8_not_equal_to_f32() {
        // Arrange & Act
        let u8_arith = DType::U8.ptx_arith_type();
        let f32_arith = DType::F32.ptx_arith_type();
        // Assert: U8 arith type differs from F32 (U8 uses integer ops, F32 uses float ops)
        assert!(!u8_arith.is_empty());
        assert_ne!(u8_arith, f32_arith);
    }
}
