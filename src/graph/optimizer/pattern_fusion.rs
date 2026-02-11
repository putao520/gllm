//! 模式融合 Pass (REQ-OPT-002)
//!
//! 识别并融合常见的算子模式。

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::{
    FlashAttentionConfig, FusedGraph, FusedNode, FusedOp, FusedQkvRopeConfig,
    FusedRMSLinearConfig, SwiGLUConfig,
};
#[cfg(test)]
use crate::graph::types::AtomicOp;

#[derive(Debug)]
pub struct FlashAttentionFusionPass;

impl OptimizationPass for FlashAttentionFusionPass {
    fn name(&self) -> &'static str {
        "FlashAttentionFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        if !ctx.supports_flash_attention() {
            return Ok(graph);
        }

        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 4, |window| {
            let [qk, scale, softmax, av] = window else {
                return None;
            };
            if !is_atomic_op(qk, "MatMul")
                || !is_atomic_op(av, "MatMul")
                || !is_softmax(softmax)
                || !is_scale_like(scale)
            {
                return None;
            }

            let config = FlashAttentionConfig {
                num_heads: ctx.num_heads,
                num_kv_heads: ctx.num_kv_heads,
                head_dim: ctx.head_dim,
                scale: Some(1.0 / (ctx.head_dim as f32).sqrt()),
                causal: true,
            };
            let mut fused = FusedNode::new(
                format!("{}_flash_attn", qk.name),
                FusedOp::FlashAttention(config),
            );
            fused.inputs = qk.inputs.clone();
            fused.outputs = av.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.flash_attention_fusions = fusions;
        Ok(out)
    }

    fn enabled(&self, ctx: &OptimizationContext) -> bool {
        ctx.supports_flash_attention()
    }

    fn priority(&self) -> i32 {
        10
    }
}

#[derive(Debug)]
pub struct SwiGLUFusionPass;

impl OptimizationPass for SwiGLUFusionPass {
    fn name(&self) -> &'static str {
        "SwiGLUFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        if !ctx.supports_swiglu() {
            return Ok(graph);
        }

        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 4, |window| {
            let [gate, up, silu, mul] = window else {
                return None;
            };
            if !is_gate_proj(gate) || !is_up_proj(up) || !is_silu(silu) || !is_mul(mul) {
                return None;
            }

            let mut fused = FusedNode::new(
                format!("{}_swiglu", gate.name),
                FusedOp::SwiGLU(SwiGLUConfig::default()),
            );
            fused.inputs = gate.inputs.clone();
            fused.outputs = mul.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.swiglu_fusions = fusions;
        Ok(out)
    }

    fn enabled(&self, ctx: &OptimizationContext) -> bool {
        ctx.supports_swiglu()
    }

    fn priority(&self) -> i32 {
        20
    }
}

#[derive(Debug)]
pub struct FusedQkvRopeFusionPass;

impl OptimizationPass for FusedQkvRopeFusionPass {
    fn name(&self) -> &'static str {
        "FusedQkvRopeFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 4, |window| {
            let [q, k, v, rope] = window else {
                return None;
            };
            if !is_q_proj(q) || !is_k_proj(k) || !is_v_proj(v) || !is_rope(rope) {
                return None;
            }

            let mut fused = FusedNode::new(
                format!("{}_fused_qkv_rope", q.name),
                FusedOp::FusedQkvRope(FusedQkvRopeConfig::default()),
            );
            fused.inputs = q
                .inputs
                .iter()
                .chain(k.inputs.iter())
                .chain(v.inputs.iter())
                .cloned()
                .collect();
            fused.outputs = rope.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.qkv_rope_fusions = fusions;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        15
    }
}

#[derive(Debug)]
pub struct FusedRMSLinearFusionPass;

impl OptimizationPass for FusedRMSLinearFusionPass {
    fn name(&self) -> &'static str {
        "FusedRMSLinearFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 2, |window| {
            let [rms, linear] = window else {
                return None;
            };
            if !is_rms_norm(rms) || !is_linear(linear) {
                return None;
            }
            let mut fused = FusedNode::new(
                format!("{}_fused_rms_linear", rms.name),
                FusedOp::FusedRMSLinear(FusedRMSLinearConfig::default()),
            );
            fused.inputs = rms.inputs.clone();
            fused.outputs = linear.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.rms_linear_fusions = fusions;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        25
    }
}

fn fuse_window<F>(nodes: Vec<FusedNode>, window: usize, fuse: F) -> (Vec<FusedNode>, usize)
where
    F: Fn(&[&FusedNode]) -> Option<FusedNode>,
{
    if nodes.len() < window || window == 0 {
        return (nodes, 0);
    }

    let mut out = Vec::with_capacity(nodes.len());
    let mut i = 0;
    let mut fusions = 0;

    while i < nodes.len() {
        if i + window <= nodes.len() {
            let slice = nodes[i..i + window].iter().collect::<Vec<_>>();
            if let Some(fused) = fuse(&slice) {
                out.push(fused);
                i += window;
                fusions += 1;
                continue;
            }
        }
        out.push(nodes[i].clone());
        i += 1;
    }

    (out, fusions)
}

fn is_atomic_op(node: &FusedNode, op_type: &str) -> bool {
    matches!(&node.op, FusedOp::Atomic(op) if op.op_type.eq_ignore_ascii_case(op_type))
}

fn is_softmax(node: &FusedNode) -> bool {
    is_atomic_op(node, "Softmax")
}

fn is_scale_like(node: &FusedNode) -> bool {
    is_atomic_op(node, "Scale") || is_atomic_op(node, "Div") || is_atomic_op(node, "Mul")
}

fn is_linear(node: &FusedNode) -> bool {
    is_atomic_op(node, "MatMul") || is_atomic_op(node, "Gemm")
}

fn is_rms_norm(node: &FusedNode) -> bool {
    is_atomic_op(node, "RmsNorm")
        || is_atomic_op(node, "RMSNorm")
        || is_atomic_op(node, "SimplifiedLayerNormalization")
}

fn is_gate_proj(node: &FusedNode) -> bool {
    is_linear(node) && node.name.contains("gate")
}

fn is_up_proj(node: &FusedNode) -> bool {
    is_linear(node) && (node.name.contains("up") || node.name.contains("ffn"))
}

fn is_silu(node: &FusedNode) -> bool {
    is_atomic_op(node, "SiLU") || is_atomic_op(node, "Silu")
}

fn is_mul(node: &FusedNode) -> bool {
    is_atomic_op(node, "Mul")
}

fn is_q_proj(node: &FusedNode) -> bool {
    is_linear(node) && (node.name.contains("q_proj") || node.name.ends_with("_q"))
}

fn is_k_proj(node: &FusedNode) -> bool {
    is_linear(node) && (node.name.contains("k_proj") || node.name.ends_with("_k"))
}

fn is_v_proj(node: &FusedNode) -> bool {
    is_linear(node) && (node.name.contains("v_proj") || node.name.ends_with("_v"))
}

fn is_rope(node: &FusedNode) -> bool {
    is_atomic_op(node, "RotaryEmbedding") || is_atomic_op(node, "Rope")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atomic_node(name: &str, op_type: &str) -> FusedNode {
        FusedNode::new(name, FusedOp::Atomic(AtomicOp::new(op_type)))
            .with_outputs(vec![format!("{}_out", name)])
    }

    #[test]
    fn swiglu_fusion_detected() {
        let pass = SwiGLUFusionPass;
        let ctx = OptimizationContext::default();
        let graph = FusedGraph {
            nodes: vec![
                make_atomic_node("layer_0_gate", "MatMul"),
                make_atomic_node("layer_0_up", "MatMul"),
                make_atomic_node("layer_0_silu", "SiLU"),
                make_atomic_node("layer_0_mul", "Mul"),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.swiglu_fusions, 1);
        assert!(matches!(fused.nodes[0].op, FusedOp::SwiGLU(_)));
    }

    #[test]
    fn fused_qkv_rope_detected() {
        let pass = FusedQkvRopeFusionPass;
        let ctx = OptimizationContext::default();
        let graph = FusedGraph {
            nodes: vec![
                make_atomic_node("layer_0_q_proj", "MatMul"),
                make_atomic_node("layer_0_k_proj", "MatMul"),
                make_atomic_node("layer_0_v_proj", "MatMul"),
                make_atomic_node("layer_0_rope", "RotaryEmbedding"),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.qkv_rope_fusions, 1);
        assert!(matches!(fused.nodes[0].op, FusedOp::FusedQkvRope(_)));
    }
}
