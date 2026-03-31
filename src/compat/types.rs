//! Parameter grouping structs for compat layer functions.
//!
//! These structs reduce parameter counts by bundling related values
//! that are always passed together (attention geometry, layer dimensions,
//! sequence context, KV cache slices, and per-layer weights).

use super::weight_helpers::WeightData;
use gllm_kernels::inference::DType;

/// Attention head geometry — derived from `GeneratorForwardConfig`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionGeometry {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub heads_per_group: usize,
}

/// Per-layer dimension constants.
#[derive(Debug, Clone, Copy)]
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
