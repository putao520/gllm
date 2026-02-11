//! 硬件感知融合 Pass (REQ-OPT-003)

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::backend::BackendType;
use crate::graph::types::{AtomicOp, FusedGraph, FusedOp};

#[derive(Debug)]
pub struct HardwareFusionPass;

impl OptimizationPass for HardwareFusionPass {
    fn name(&self) -> &'static str {
        "HardwareFusion"
    }

    fn run(
        &self,
        mut graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        for node in &mut graph.nodes {
            match &node.op {
                FusedOp::FlashAttention(_) if !supports_flash_attention(ctx) => {
                    node.op = FusedOp::Atomic(AtomicOp::new("Attention"));
                }
                FusedOp::FusedQkvRope(_) if !supports_qkv_rope(ctx) => {
                    node.op = FusedOp::Atomic(AtomicOp::new("QkvRope"));
                }
                FusedOp::FusedRMSLinear(_) if !supports_rms_linear(ctx) => {
                    node.op = FusedOp::Atomic(AtomicOp::new("RmsNormLinear"));
                }
                _ => {}
            }
        }
        Ok(graph)
    }

    fn priority(&self) -> i32 {
        40
    }
}

fn supports_flash_attention(ctx: &OptimizationContext) -> bool {
    if ctx.backend_type != BackendType::Cuda {
        return false;
    }
    ctx.cuda_sm_version
        .map(|(major, _)| major >= 8)
        .unwrap_or(false)
}

fn supports_qkv_rope(ctx: &OptimizationContext) -> bool {
    if ctx.backend_type != BackendType::Cuda {
        return true;
    }
    ctx.cuda_sm_version
        .map(|(major, _)| major >= 7)
        .unwrap_or(false)
}

fn supports_rms_linear(ctx: &OptimizationContext) -> bool {
    match ctx.backend_type {
        BackendType::Cuda => ctx
            .cuda_sm_version
            .map(|(major, _)| major >= 8)
            .unwrap_or(false),
        BackendType::Cpu => true,
        BackendType::Rocm | BackendType::Metal => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{FlashAttentionConfig, FusedNode};

    #[test]
    fn downgrade_flash_attention_for_cpu() {
        let pass = HardwareFusionPass;
        let ctx = OptimizationContext::cpu();
        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "attn",
                FusedOp::FlashAttention(FlashAttentionConfig::default()),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert!(matches!(out.nodes[0].op, FusedOp::Atomic(_)));
    }
}
