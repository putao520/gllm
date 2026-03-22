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
    match ctx.backend_type {
        BackendType::Cuda => {
            // FlashAttention requires SM >= 8.0 (Ampere+)
            // F16/BF16: SM >= 8.0 (tensor core HMMA)
            // F32: SM >= 8.0 (TF32 path)
            ctx.cuda_sm_version
                .map(|(major, _)| major >= 8)
                .unwrap_or(false)
        }
        BackendType::Rocm => {
            // ROCm supports FlashAttention on CDNA2+ (gfx90a+)
            // Approximated by SM version mapping: treat as supported
            true
        }
        BackendType::Metal => {
            // Metal supports FlashAttention via MSL threadgroup memory
            true
        }
        BackendType::Cpu => false,
    }
}

fn supports_qkv_rope(ctx: &OptimizationContext) -> bool {
    match ctx.backend_type {
        BackendType::Cuda => ctx.cuda_sm_version
            .map(|(major, _)| major >= 7)
            .unwrap_or(false),
        // All other backends support fused QKV+RoPE
        _ => true,
    }
}

fn supports_rms_linear(ctx: &OptimizationContext) -> bool {
    match ctx.backend_type {
        BackendType::Cuda => ctx
            .cuda_sm_version
            .map(|(major, _)| major >= 7)
            .unwrap_or(false),
        BackendType::Cpu => true,
        BackendType::Rocm => true,
        BackendType::Metal => true,
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
