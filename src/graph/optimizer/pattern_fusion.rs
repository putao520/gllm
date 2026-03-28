//! 模式融合 Pass (REQ-OPT-002)
//!
//! 识别并融合常见的算子模式。

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::{
    FlashAttentionConfig, FusedGraph, FusedNode, FusedOp, FusedQkvRopeConfig,
    FusedRMSLinearConfig, GQAConfig, MoERoutingConfig, SwiGLUConfig,
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
                scale: qk
                    .inputs
                    .iter()
                    .find_map(|input| out.quantization_info.get(input).map(|q| q.scale))
                    .or(Some(1.0 / (ctx.head_dim as f32).sqrt())),
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

            let rope_theta = rope.attributes.get("theta").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f as f64),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f64),
                _ => None,
            }).unwrap_or(10000.0);

            let config = FusedQkvRopeConfig {
                num_heads: _ctx.num_heads,
                num_kv_heads: _ctx.num_kv_heads,
                head_dim: _ctx.head_dim,
                rope_theta,
            };

            let mut fused = FusedNode::new(
                format!("{}_fused_qkv_rope", q.name),
                FusedOp::FusedQkvRope(config),
            );
            fused.inputs = vec![
                q.inputs[0].clone(), // activation input
                q.inputs[1].clone(), // w_q
                k.inputs[1].clone(), // w_k
                v.inputs[1].clone(), // w_v
            ];
            // Outputs: Q_rope, K_rope, V_proj
            let mut fused_outputs = rope.outputs.clone();
            // Append V projection output (it's not part of the RoPE node's outputs)
            fused_outputs.extend(v.outputs.iter().cloned());
            fused.outputs = fused_outputs;
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
            let eps = rms.attributes.get("eps").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f32),
                _ => None,
            }).unwrap_or(1e-5);

            let mut fused = FusedNode::new(
                format!("{}_fused_rms_linear", rms.name),
                FusedOp::FusedRMSLinear(FusedRMSLinearConfig {
                    hidden_size: _ctx.hidden_size,
                    eps,
                }),
            );
            fused.inputs = rms
                .inputs
                .iter()
                .chain(linear.inputs.iter().skip(1))
                .cloned()
                .collect();
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

#[derive(Debug)]
pub struct GQAFusionPass;

impl OptimizationPass for GQAFusionPass {
    fn name(&self) -> &'static str {
        "GQAFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        if ctx.num_kv_heads >= ctx.num_heads || ctx.num_kv_heads == 0 {
            return Ok(graph);
        }

        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 4, |window| {
            let [q, k, v, attn] = window else {
                return None;
            };
            if !is_q_proj(q) || !is_k_proj(k) || !is_v_proj(v) || !is_attention(attn) {
                return None;
            }

            let config = GQAConfig {
                num_heads: ctx.num_heads,
                num_kv_heads: ctx.num_kv_heads,
                num_groups: ctx.num_heads / ctx.num_kv_heads,
                head_dim: ctx.head_dim,
            };
            let mut fused = FusedNode::new(format!("{}_gqa", q.name), FusedOp::GQA(config));
            fused.inputs = q
                .inputs
                .iter()
                .chain(k.inputs.iter())
                .chain(v.inputs.iter())
                .cloned()
                .collect();
            fused.outputs = attn.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.gqa_fusions = fusions;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        14
    }
}

#[derive(Debug)]
pub struct CanonicalizeAttentionPass;

impl OptimizationPass for CanonicalizeAttentionPass {
    fn name(&self) -> &'static str {
        "CanonicalizeAttention"
    }

    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        let mut canonicalized = 0;

        for node in &mut out.nodes {
            if is_attention(node) {
                // Determine whether to use FlashAttention or GQA
                if ctx.supports_flash_attention() {
                    let config = FlashAttentionConfig {
                        num_heads: ctx.num_heads,
                        num_kv_heads: ctx.num_kv_heads,
                        head_dim: ctx.head_dim,
                        scale: Some(1.0 / (ctx.head_dim as f32).sqrt()),
                        causal: true,
                    };
                    node.op = FusedOp::FlashAttention(config);
                } else {
                    let config = GQAConfig {
                        num_heads: ctx.num_heads,
                        num_kv_heads: ctx.num_kv_heads,
                        num_groups: ctx.num_heads / ctx.num_kv_heads,
                        head_dim: ctx.head_dim,
                    };
                    node.op = FusedOp::GQA(config);
                }
                canonicalized += 1;
            }
        }

        // We overload gqa_fusions stat to count canonicalized attention ops too
        out.stats.gqa_fusions += canonicalized;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        5 // High priority to run before window-based fusions
    }
}

#[derive(Debug)]
pub struct MoERoutingFusionPass;

impl OptimizationPass for MoERoutingFusionPass {
    fn name(&self) -> &'static str {
        "MoERoutingFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 4, |window| {
            let [router, topk, softmax, dispatch] = window else {
                return None;
            };
            if !is_router(router)
                || !is_atomic_op(topk, "TopK")
                || !is_softmax(softmax)
                || !is_dispatch(dispatch)
            {
                return None;
            }

            let num_experts = find_num_experts(router).unwrap_or_else(|| {
                log::warn!(
                    "MoE pattern fusion: could not determine num_experts from router node '{}', using 0 (runtime-determined)",
                    router.name
                );
                0
            });
            let top_k = find_top_k(topk).unwrap_or(2);
            let capacity_factor = find_capacity_factor(dispatch).unwrap_or(1.0);

            let mut fused = FusedNode::new(
                format!("{}_moe_routing", router.name),
                FusedOp::MoERouting(MoERoutingConfig {
                    num_experts,
                    top_k,
                    capacity_factor,
                }),
            );
            fused.inputs = router.inputs.clone();
            fused.outputs = dispatch.outputs.clone();
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.moe_routing_fusions = fusions;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        22
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

fn is_attention(node: &FusedNode) -> bool {
    is_atomic_op(node, "Attention") || is_atomic_op(node, "GroupedQueryAttention")
}

fn is_router(node: &FusedNode) -> bool {
    is_linear(node) && node.name.to_ascii_lowercase().contains("router")
}

fn is_dispatch(node: &FusedNode) -> bool {
    let name = node.name.to_ascii_lowercase();
    is_atomic_op(node, "Dispatch")
        || is_atomic_op(node, "Scatter")
        || (is_atomic_op(node, "Gather") && name.contains("dispatch"))
}

fn find_num_experts(node: &FusedNode) -> Option<usize> {
    node.attributes
        .get("num_experts")
        .and_then(|value| match value {
            crate::graph::types::AttrValue::Int(v) => usize::try_from(*v).ok(),
            _ => None,
        })
}

fn find_top_k(node: &FusedNode) -> Option<usize> {
    node.attributes.get("k").and_then(|value| match value {
        crate::graph::types::AttrValue::Int(v) => usize::try_from(*v).ok(),
        _ => None,
    })
}

fn find_capacity_factor(node: &FusedNode) -> Option<f32> {
    node.attributes
        .get("capacity_factor")
        .and_then(|value| match value {
            crate::graph::types::AttrValue::Float(v) => Some(*v),
            _ => None,
        })
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

    #[test]
    fn gqa_fusion_detected() {
        let pass = GQAFusionPass;
        let ctx = OptimizationContext {
            num_heads: 32,
            num_kv_heads: 8,
            ..OptimizationContext::default()
        };
        let graph = FusedGraph {
            nodes: vec![
                make_atomic_node("layer_0_q_proj", "MatMul"),
                make_atomic_node("layer_0_k_proj", "MatMul"),
                make_atomic_node("layer_0_v_proj", "MatMul"),
                make_atomic_node("layer_0_attn", "Attention"),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.gqa_fusions, 1);
        assert!(matches!(fused.nodes[0].op, FusedOp::GQA(_)));
    }

    #[test]
    fn moe_routing_fusion_detected() {
        let pass = MoERoutingFusionPass;
        let ctx = OptimizationContext::default();

        let mut topk = make_atomic_node("router_topk", "TopK");
        topk.attributes
            .insert("k".to_string(), crate::graph::types::AttrValue::Int(2));
        let mut dispatch = make_atomic_node("expert_dispatch", "Dispatch");
        dispatch.attributes.insert(
            "capacity_factor".to_string(),
            crate::graph::types::AttrValue::Float(1.25),
        );

        let graph = FusedGraph {
            nodes: vec![
                make_atomic_node("moe_router", "MatMul"),
                topk,
                make_atomic_node("router_softmax", "Softmax"),
                dispatch,
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.moe_routing_fusions, 1);
        assert!(matches!(fused.nodes[0].op, FusedOp::MoERouting(_)));
    }

    #[test]
    fn flash_attention_reads_quantization_scale() {
        let pass = FlashAttentionFusionPass;
        let ctx = OptimizationContext::cuda((8, 0));
        let mut graph = FusedGraph {
            nodes: vec![
                make_atomic_node("qk", "MatMul")
                    .with_inputs(vec!["q_in".to_string(), "k_in".to_string()]),
                make_atomic_node("scale", "Mul"),
                make_atomic_node("softmax", "Softmax"),
                make_atomic_node("av", "MatMul"),
            ],
            ..FusedGraph::new()
        };
        graph.quantization_info.insert(
            "q_in".to_string(),
            crate::graph::types::QuantizationInfo {
                scale: 0.03125,
                zero_point: 0,
                axis: None,
            },
        );

        let fused = pass.run(graph, &ctx).unwrap();
        match &fused.nodes[0].op {
            FusedOp::FlashAttention(config) => {
                let scale = config.scale.expect("scale");
                assert!((scale - 0.03125).abs() < f32::EPSILON);
            }
            other => panic!("unexpected op: {other:?}"),
        }
    }
}
