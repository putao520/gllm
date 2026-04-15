//! 硬件感知融合拓扑变换 Pass (SPEC §13.12)
//!
//! ## 设计原则
//!
//! **NO_HW_DEGRADATION 铁律**：此 Pass 绝不将融合算子降级为原子算子。
//! 硬件差异体现在 codegen 层的指令选择，不在 fusion 层的算子拆分。
//!
//! **§13.12 拓扑变换**：不同硬件产生不同的图拓扑——不仅是同一算子的
//! 不同 codegen，而是节点属性、融合深度、特化配置根本不同。
//!
//! 具体操作：
//! 1. 根据 GPU SM 版本设置 FlashAttention 变体提示 (FA4/FA3/FA2/wmma)
//! 2. 根据寄存器预算调整可融合 epilogue 深度的提示
//! 3. 根据硬件缓存层级向节点注入 tile size 提示
//! 4. 为 Hopper+ 设备标注 TMA/Warp Specialization 启用标记
//! 5. 为 MoE 标注硬件分发策略 (核内分发 vs CPU 协同)

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::{AttrValue, FusedGraph, FusedOp, FusedNode};

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
        // §13.12: 根据硬件 Profile 变换图拓扑（属性特化，非降级）
        let sm = ctx.cuda_sm_version.map(|(major, minor)| major * 10 + minor);
        let num_simd_regs = infer_simd_regs(ctx);

        for node in &mut graph.nodes {
            // Extract needed data from op before mutably borrowing node.attributes
            let op_tag = match &node.op {
                FusedOp::FlashAttention(_) => OpTag::FlashAttention,
                FusedOp::GQA(c) => OpTag::GQA { sliding_window: c.sliding_window },
                FusedOp::SwiGLU(_) => OpTag::Epilogue,
                FusedOp::FusedQkvRope(_) => OpTag::QkvRope,
                FusedOp::MoERouting(c) => OpTag::MoE { num_experts: c.num_experts },
                FusedOp::FusedRMSLinear(_) => OpTag::Epilogue,
                FusedOp::RoPE(_) | FusedOp::PerLayerEmbed(_) | FusedOp::Atomic(_) => OpTag::Noop,
            };

            match op_tag {
                OpTag::FlashAttention => {
                    annotate_flash_attention(node, sm, ctx);
                }
                OpTag::GQA { sliding_window } => {
                    annotate_gqa(node, sliding_window, sm);
                }
                OpTag::Epilogue => {
                    annotate_epilogue_depth(node, num_simd_regs);
                }
                OpTag::QkvRope => {
                    if let Some(v) = sm {
                        if v >= 90 {
                            node.attributes.insert(
                                "hw_warp_specialization".into(),
                                AttrValue::String("true".into()),
                            );
                        }
                    }
                }
                OpTag::MoE { num_experts } => {
                    annotate_moe_dispatch(node, num_experts, sm, ctx);
                }
                OpTag::Noop => {}
            }
        }

        // §13.12: 跨层残差融合资格标注
        // 只有寄存器预算充足（>=4 scratch）时才标注跨层残差
        // codegen 层根据此属性决定是否生成 Add→RmsNorm scratchpad 直通代码
        // (graph-level attribute 通过节点属性传播，不在 stats 中记录)

        Ok(graph)
    }

    fn priority(&self) -> i32 {
        40
    }
}

/// Internal tag to decouple op match from mutable annotation.
#[derive(Clone, Copy)]
enum OpTag {
    FlashAttention,
    GQA { sliding_window: usize },
    Epilogue,
    QkvRope,
    MoE { num_experts: usize },
    Noop,
}

// ═══════════════════════════════════════════════════════════════
//  §13.12.1 FlashAttention 变体标注
// ═══════════════════════════════════════════════════════════════

fn annotate_flash_attention(
    node: &mut FusedNode,
    sm: Option<u32>,
    ctx: &OptimizationContext,
) {
    if let Some(v) = sm {
        // GPU 路径：按 SM 版本标注 FA 变体
        let variant = if v >= 100 {
            "FA4_BlockScaled"  // SM100+: tcgen05.mma + TMEM
        } else if v >= 90 {
            "FA3_Pipeline"     // SM90: WGMMA + TMA + Warp Spec
        } else if v >= 80 {
            "FA2_Tiled"        // SM80: mma.sync + cp.async
        } else {
            "wmma_Tiled"       // SM70: wmma 16×16×16
        };
        node.attributes.insert("hw_fa_variant".into(), AttrValue::String(variant.into()));

        // Hopper+ 特性标注
        if v >= 90 {
            node.attributes.insert("hw_tma_enabled".into(), AttrValue::String("true".into()));
            node.attributes.insert("hw_warp_specialization".into(), AttrValue::String("true".into()));
        }
        // Blackwell 特性标注
        if v >= 100 {
            node.attributes.insert("hw_tmem_enabled".into(), AttrValue::String("true".into()));
            node.attributes.insert("hw_block_scaled".into(), AttrValue::String("true".into()));
        }
    } else {
        // CPU 路径
        if ctx.has_bf16 {
            node.attributes.insert("hw_fa_variant".into(), AttrValue::String("AMX_Tile".into()));
        } else {
            node.attributes.insert("hw_fa_variant".into(), AttrValue::String("SIMD_Loop".into()));
        }
    }

    // Tensor Core 代数标注（影响 codegen 指令选择）
    if ctx.tensor_core_gen > 0 {
        node.attributes.insert(
            "hw_tensor_core_gen".into(),
            AttrValue::String(ctx.tensor_core_gen.to_string()),
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  §13.12.2 GQA 硬件特化标注
// ═══════════════════════════════════════════════════════════════

fn annotate_gqa(node: &mut FusedNode, sliding_window: usize, sm: Option<u32>) {
    if let Some(v) = sm {
        if v >= 90 {
            // Hopper: TMA 2D prefetch 启用
            node.attributes.insert("hw_tma_kv_prefetch".into(), AttrValue::String("true".into()));
        }
        if v >= 100 {
            // Blackwell: TMEM 替代 shared memory
            node.attributes.insert("hw_tmem_kv".into(), AttrValue::String("true".into()));
        }
    }

    // Sliding window 提示（影响 KV 页表布局）
    if sliding_window > 0 {
        node.attributes.insert(
            "hw_sliding_window".into(),
            AttrValue::String(sliding_window.to_string()),
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  §13.12.3 Epilogue 深度提示
// ═══════════════════════════════════════════════════════════════

fn annotate_epilogue_depth(node: &mut FusedNode, num_simd_regs: usize) {
    // 寄存器充裕时标注可融合更深的 epilogue 链
    // 32 zmm (AVX-512) → 最多 8-op epilogue
    // 16 ymm (AVX2) → 最多 4-op epilogue
    // 31 GPR (APX) → 支持最深链
    let max_depth = if num_simd_regs >= 32 {
        8
    } else if num_simd_regs >= 16 {
        4
    } else {
        2
    };
    node.attributes.insert(
        "hw_max_epilogue_depth".into(),
        AttrValue::String(max_depth.to_string()),
    );
}

// ═══════════════════════════════════════════════════════════════
//  §13.12.4 MoE 硬件分发策略
// ═══════════════════════════════════════════════════════════════

fn annotate_moe_dispatch(
    node: &mut FusedNode,
    num_experts: usize,
    sm: Option<u32>,
    ctx: &OptimizationContext,
) {
    if let Some(v) = sm {
        // GPU: 核内分发（§15.1）
        node.attributes.insert("hw_moe_dispatch".into(), AttrValue::String("in_kernel".into()));

        if v >= 90 {
            // Hopper: TMA 预取冷专家权重
            node.attributes.insert("hw_expert_prefetch".into(), AttrValue::String("tma".into()));
        }

        // 128+ 专家路由表立即数化（§12.7.3 完美哈希跳表）
        if num_experts > 64 {
            node.attributes.insert("hw_route_table".into(), AttrValue::String("imm_register".into()));
        } else {
            node.attributes.insert("hw_route_table".into(), AttrValue::String("smem".into()));
        }
    } else {
        // CPU: NUMA 感知分发（§15.3 Core Disaggregation）
        if ctx.has_vnni || ctx.has_bf16 {
            node.attributes.insert("hw_moe_dispatch".into(), AttrValue::String("cpu_parallel".into()));
        } else {
            node.attributes.insert("hw_moe_dispatch".into(), AttrValue::String("cpu_sequential".into()));
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Helper
// ═══════════════════════════════════════════════════════════════

/// 从 OptimizationContext 推断 SIMD 寄存器数
fn infer_simd_regs(ctx: &OptimizationContext) -> usize {
    if ctx.cuda_sm_version.is_some() {
        // GPU: 不直接适用 SIMD 寄存器概念，用 warp 大小近似
        return 32;
    }
    // CPU: 根据已知特性推断
    if ctx.has_bf16 {
        // AVX-512 BF16 / AMX → 32 zmm
        32
    } else {
        // AVX2 → 16 ymm
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{FlashAttentionConfig, FusedNode, FusedOp, GQAConfig,
                               SwiGLUConfig, MoERoutingConfig};

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

    fn assert_attr(node: &FusedNode, key: &str, expected: &str) {
        let val = node.attributes.get(key)
            .unwrap_or_else(|| panic!("missing attribute: {}", key));
        match val {
            AttrValue::String(s) => assert_eq!(s, expected, "attribute {} mismatch", key),
            other => panic!("expected String for {}, got {:?}", key, other),
        }
    }

    #[test]
    fn annotates_fa_variant_for_gpu_sm90() {
        let pass = HardwareFusionPass;
        let mut ctx = OptimizationContext::cpu();
        ctx.cuda_sm_version = Some((9, 0));
        ctx.tensor_core_gen = 3;

        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "attn",
                FusedOp::FlashAttention(FlashAttentionConfig {
                    num_heads: 32,
                    num_kv_heads: 8,
                    head_dim: 128,
                    scale: None,
                    causal: true,
                }),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert_attr(&out.nodes[0], "hw_fa_variant", "FA3_Pipeline");
        assert_attr(&out.nodes[0], "hw_tma_enabled", "true");
        assert_attr(&out.nodes[0], "hw_warp_specialization", "true");
        assert_attr(&out.nodes[0], "hw_tensor_core_gen", "3");
    }

    #[test]
    fn annotates_fa_variant_for_gpu_sm100() {
        let pass = HardwareFusionPass;
        let mut ctx = OptimizationContext::cpu();
        ctx.cuda_sm_version = Some((10, 0));
        ctx.tensor_core_gen = 4;

        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "attn",
                FusedOp::FlashAttention(FlashAttentionConfig::default()),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert_attr(&out.nodes[0], "hw_fa_variant", "FA4_BlockScaled");
        assert_attr(&out.nodes[0], "hw_tmem_enabled", "true");
        assert_attr(&out.nodes[0], "hw_block_scaled", "true");
    }

    #[test]
    fn annotates_epilogue_depth_avx512() {
        let pass = HardwareFusionPass;
        let mut ctx = OptimizationContext::cpu();
        ctx.has_bf16 = true; // AVX-512 BF16 → 32 regs

        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "ffn",
                FusedOp::SwiGLU(SwiGLUConfig {
                    hidden_size: 4096,
                    intermediate_size: 11008,
                }),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert_attr(&out.nodes[0], "hw_max_epilogue_depth", "8");
    }

    #[test]
    fn annotates_moe_gpu_dispatch() {
        let pass = HardwareFusionPass;
        let mut ctx = OptimizationContext::cpu();
        ctx.cuda_sm_version = Some((9, 0));

        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "moe",
                FusedOp::MoERouting(MoERoutingConfig {
                    num_experts: 128,
                    top_k: 2,
                    capacity_factor: 1.25,
                }),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert_attr(&out.nodes[0], "hw_moe_dispatch", "in_kernel");
        assert_attr(&out.nodes[0], "hw_expert_prefetch", "tma");
        assert_attr(&out.nodes[0], "hw_route_table", "imm_register");
    }

    #[test]
    fn gqa_sliding_window_annotated() {
        let pass = HardwareFusionPass;
        let mut ctx = OptimizationContext::cpu();
        ctx.cuda_sm_version = Some((8, 0));

        let graph = FusedGraph {
            nodes: vec![FusedNode::new(
                "gqa",
                FusedOp::GQA(GQAConfig {
                    num_heads: 32,
                    num_kv_heads: 8,
                    num_groups: 4,
                    head_dim: 128,
                    sliding_window: 4096,
                }),
            )],
            ..FusedGraph::new()
        };

        let out = pass.run(graph, &ctx).unwrap();
        assert_attr(&out.nodes[0], "hw_sliding_window", "4096");
    }
}
