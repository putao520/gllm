use std::collections::HashSet;

use super::model::OnnxGraph;
use super::Result;

mod attention;
mod index;
mod rope;
mod swiglu;

#[derive(Debug, Clone)]
pub struct FusedGraph {
    pub ops: Vec<FusedOp>,
}

#[derive(Debug, Clone)]
pub struct FusedOp {
    pub kind: FusedKernel,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum FusedKernel {
    FlashAttention(FlashAttentionSpec),
    SwiGlu(SwiGluSpec),
    Rope(RopeSpec),
    FusedQkvRope(FusedQkvRopeSpec),
    Atomic(AtomicOp),
}

#[derive(Debug, Clone)]
pub struct FlashAttentionSpec {
    pub q: String,
    pub k: String,
    pub v: String,
    pub output: String,
    pub scale: Option<f32>,
    pub causal: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SwiGluSpec {
    pub gate: String,
    pub up: String,
    pub output: String,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub input: String,
    pub output: String,
    pub rotary_dim: Option<i64>,
    pub base: Option<f32>,
    pub scale: Option<f32>,
    pub interleaved: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct FusedQkvRopeSpec {
    pub input: String,
    pub weight: String,
    pub bias: Option<String>,
    pub output: String,
    pub rotary_dim: Option<i64>,
    pub base: Option<f32>,
    pub scale: Option<f32>,
    pub interleaved: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct AtomicOp {
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub(super) fn build_fused_graph(graph: &OnnxGraph) -> Result<FusedGraph> {
    let index = index::GraphIndex::new(graph);
    let mut consumed = HashSet::new();
    let mut ops = Vec::new();

    ops.extend(attention::match_attention(graph, &index, &mut consumed)?);
    ops.extend(rope::match_rope(graph, &index, &mut consumed)?);
    ops.extend(swiglu::match_swiglu(graph, &index, &mut consumed)?);
    ops.extend(index::collect_atomic_ops(graph, &consumed));

    Ok(FusedGraph { ops })
}
