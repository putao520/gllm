//! 硬件感知融合 Pass (REQ-OPT-003)
//!
//! ⚠️ NO_HW_DEGRADATION 铁律：此 Pass 不再执行降级操作。
//! 硬件差异由 codegen 层处理（x86_64/AArch64/GPU 各自生成最优路径）。
//! 融合算子在所有硬件上保持融合形态，仅 JIT 指令选择不同。

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::FusedGraph;

#[derive(Debug)]
pub struct HardwareFusionPass;

impl OptimizationPass for HardwareFusionPass {
    fn name(&self) -> &'static str {
        "HardwareFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        // NO_HW_DEGRADATION: 不降级融合算子。
        // codegen 层根据 DeviceProfile 为每种硬件生成最优代码。
        Ok(graph)
    }

    fn priority(&self) -> i32 {
        40
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{FlashAttentionConfig, FusedNode, FusedOp};

    #[test]
    fn no_downgrade_flash_attention_for_cpu() {
        // NO_HW_DEGRADATION: FlashAttention 保持融合形态，不降级
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
        // FlashAttention 应保持融合形态
        assert!(matches!(out.nodes[0].op, FusedOp::FlashAttention(_)));
    }
}
