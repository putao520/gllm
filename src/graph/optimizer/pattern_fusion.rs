//! 模式融合 Pass (REQ-OPT-002)
//!
//! 识别并融合常见的算子模式。

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::{
    FlashAttentionConfig, FusedGraph, FusedNode, FusedOp, FusedQkvNormRopeConfig,
    FusedQkvRopeConfig, FusedRMSLinearConfig, GQAConfig, MoERoutingConfig, SwiGLUConfig,
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
                num_heads: ctx.num_heads(),
                num_kv_heads: ctx.num_kv_heads(),
                head_dim: ctx.head_dim(),
                scale: qk
                    .inputs
                    .iter()
                    .find_map(|input| out.quantization_info.get(input).map(|q| q.scale))
                    .or(Some(1.0 / (ctx.head_dim() as f32).sqrt())),
                causal: !ctx.is_encoder(),
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

            // WII (Weight-in-Inputs) architecture — same pattern as FusedQkvRope:
            // inputs = [activation, gate_proj.weight, up_proj.weight]
            // The build_fused_swiglu_graph will create 2×Gemm + Silu + Mul internally.
            let mut fused = FusedNode::new(
                format!("{}_swiglu", gate.name),
                FusedOp::SwiGLU(SwiGLUConfig {
                    hidden_size: ctx.geometry.hidden_size,
                    intermediate_size: ctx.geometry.intermediate_size,
                }),
            );
            // gate.inputs = [activation, gate_weight]
            // up.inputs   = [activation, up_weight]
            // Merged: [activation, gate_weight, up_weight]
            let mut inputs = Vec::with_capacity(3);
            inputs.push(gate.inputs[0].clone()); // activation (shared)
            if gate.inputs.len() > 1 {
                inputs.push(gate.inputs[1].clone()); // gate_proj.weight
            }
            if up.inputs.len() > 1 {
                inputs.push(up.inputs[1].clone()); // up_proj.weight
            }
            fused.inputs = inputs;
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

/// Gemma 4 pattern: QKV projection + QkNorm(Q,K) + ValueNorm(V) + RoPE(Q) + RoPE(K)
///
/// 识别窗口 (7 个节点, 严格顺序):
/// ```text
/// q_proj : MatMul  (inputs=[act, w_q])  -> q
/// k_proj : MatMul  (inputs=[act, w_k])  -> k
/// v_proj : MatMul  (inputs=[act, w_v])  -> v
/// qk_norm: QkNorm  (inputs=[q, k])      -> q_normed, k_normed
/// v_norm : ValueNorm (inputs=[v])       -> v_normed
/// rope_q : RotaryEmbedding (inputs=[q_normed], ...) -> q_rope
/// rope_k : RotaryEmbedding (inputs=[k_normed], ...) -> k_rope
/// ```
///
/// 对应 gllm-kernels `FusionMode::FusedQkvNormRope`。仅在 QkNorm / ValueNorm
/// 同时出现且 Q/K RoPE 分别消费 Q-normed/K-normed 时才触发融合 (标准 QKV 无
/// norm 的模型走 `FusedQkvRopeFusionPass` 或保持 atomic)。
///
/// 优先级 12: 早于 FusedQkvRope(15), 因为 FusedQkvNormRope 是更严格的超集,
/// 必须先尝试匹配, 否则前三个 MatMul 会被 FusedQkvRope 抢占。
#[derive(Debug)]
pub struct FusedQkvNormRopeFusionPass;

impl OptimizationPass for FusedQkvNormRopeFusionPass {
    fn name(&self) -> &'static str {
        "FusedQkvNormRopeFusion"
    }

    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 7, |window| {
            let [q, k, v, qk_norm, v_norm, rope_q, rope_k] = window else {
                return None;
            };

            // 1. 前三个节点必须是 Q/K/V MatMul, 共享同一个激活输入
            if !is_q_proj(q) || !is_k_proj(k) || !is_v_proj(v) {
                return None;
            }
            if q.inputs.len() < 2 || k.inputs.len() < 2 || v.inputs.len() < 2 {
                return None;
            }
            let activation = &q.inputs[0];
            if k.inputs[0] != *activation || v.inputs[0] != *activation {
                return None;
            }

            // 2. QkNorm: inputs=[q_out, k_out], outputs=[q_normed, k_normed]
            if !is_atomic_op(qk_norm, "QkNorm") {
                return None;
            }
            if qk_norm.inputs.len() != 2 || qk_norm.outputs.len() != 2 {
                return None;
            }
            if qk_norm.inputs[0] != q.outputs[0] || qk_norm.inputs[1] != k.outputs[0] {
                return None;
            }

            // 3. ValueNorm: inputs=[v_out], outputs=[v_normed]
            if !is_atomic_op(v_norm, "ValueNorm") {
                return None;
            }
            if v_norm.inputs.len() != 1 || v_norm.outputs.len() != 1 {
                return None;
            }
            if v_norm.inputs[0] != v.outputs[0] {
                return None;
            }

            // 4. Q-RoPE: inputs=[q_normed], outputs=[q_rope]
            if !is_rope(rope_q) || rope_q.inputs.is_empty() {
                return None;
            }
            if rope_q.inputs[0] != qk_norm.outputs[0] {
                return None;
            }

            // 5. K-RoPE: inputs=[k_normed], outputs=[k_rope]
            if !is_rope(rope_k) || rope_k.inputs.is_empty() {
                return None;
            }
            if rope_k.inputs[0] != qk_norm.outputs[1] {
                return None;
            }

            // ── 配置读取 ──
            // rope_theta / rope_partial 从 rope_q 读取 (Q/K 两个 RoPE 同层共享参数,
            // 取 Q 的即可; template.rs::expand_dual_rope 保证了这一点)。
            let rope_theta = rope_q.attributes.get("theta").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f as f64),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f64),
                _ => None,
            }).unwrap_or(10000.0); // LEGAL: rope_theta=10000.0 是 RoPE 的行业标准默认值

            let rope_partial = rope_q.attributes.get("partial").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f32),
                _ => None,
            }).unwrap_or(1.0); // LEGAL: partial=1.0 是标准 RoPE (全维度旋转)

            // norm_eps 优先从 v_norm 节点 attributes 读 (ValueNorm 可能显式指定 eps),
            // 缺省使用 ModelGeometry.norm_eps (来自模型配置)。
            let norm_eps = v_norm.attributes.get("eps").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f32),
                _ => None,
            }).unwrap_or(ctx.geometry.norm_eps as f32);

            let config = FusedQkvNormRopeConfig {
                num_heads: ctx.num_heads(),
                num_kv_heads: ctx.num_kv_heads(),
                head_dim: ctx.head_dim(),
                rope_theta,
                rope_partial,
                norm_eps,
            };

            // ── 构造融合节点 ──
            // inputs: [activation, w_q, w_k, w_v]
            // outputs: [q_rope, k_rope, v_normed]
            let mut fused = FusedNode::new(
                format!("{}_fused_qkv_norm_rope", q.name),
                FusedOp::FusedQkvNormRope(config),
            );
            fused.inputs = vec![
                activation.clone(),
                q.inputs[1].clone(),
                k.inputs[1].clone(),
                v.inputs[1].clone(),
            ];
            fused.outputs = vec![
                rope_q.outputs[0].clone(),
                rope_k.outputs[0].clone(),
                v_norm.outputs[0].clone(),
            ];
            Some(fused)
        });

        out.nodes = nodes;
        out.stats.qkv_norm_rope_fusions = fusions;
        Ok(out)
    }

    fn priority(&self) -> i32 {
        12
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
            }).unwrap_or(10000.0); // LEGAL: rope_theta=10000.0 是 RoPE 的行业标准默认值

            let config = FusedQkvRopeConfig {
                num_heads: _ctx.num_heads(),
                num_kv_heads: _ctx.num_kv_heads(),
                head_dim: _ctx.head_dim(),
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
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut out = graph;
        // lm_head 是 hidden→vocab 的非方阵线性, 当前 FusedRMSLinearConfig 只有
        // hidden_size 一个维度参数, build_fused_rms_linear_graph 会把 linear_w
        // 构造成 [hidden, hidden] — 对 lm_head 必然形状错配,执行后输出维度退化
        // 为 hidden_size 而非 vocab_size (ARCH-RMSLINEAR-SQUARE-ONLY)。
        //
        // 用图输出集识别"终点线性 (lm_head)": 若 linear.outputs 在 graph.outputs
        // 中, 跳过融合, 让 final_norm 和 lm_head 各自走 Standalone 路径。
        let graph_output_set: std::collections::HashSet<&str> =
            out.outputs.iter().map(|s| s.as_str()).collect();
        let (nodes, fusions) = fuse_window(std::mem::take(&mut out.nodes), 2, |window| {
            let [rms, linear] = window else {
                return None;
            };
            if !is_rms_norm(rms) || !is_linear(linear) {
                return None;
            }
            // 跳过指向 graph.outputs 的 linear (lm_head / classification head)
            if linear.outputs.iter().any(|o| graph_output_set.contains(o.as_str())) {
                return None;
            }
            let eps = rms.attributes.get("eps").and_then(|val| match val {
                crate::graph::types::AttrValue::Float(f) => Some(*f),
                crate::graph::types::AttrValue::Int(i) => Some(*i as f32),
                _ => None,
            }).unwrap_or(1e-5); // LEGAL: eps=1e-5 是 RMSNorm 的行业标准默认值

            let mut fused = FusedNode::new(
                format!("{}_fused_rms_linear", rms.name),
                FusedOp::FusedRMSLinear(FusedRMSLinearConfig {
                    hidden_size: ctx.hidden_size(),
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
        if ctx.num_kv_heads() >= ctx.num_heads() || ctx.num_kv_heads() == 0 {
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
                num_heads: ctx.num_heads(),
                num_kv_heads: ctx.num_kv_heads(),
                num_groups: ctx.num_heads() / ctx.num_kv_heads(),
                head_dim: ctx.head_dim(),
                sliding_window: 0,
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
                // GQA 的 FusedOp 定义隐式假设 causal=true (decoder-only)，对 encoder
                // 模型错误。因此 encoder 无论是否支持硬件 FlashAttention，都统一走
                // FlashAttention(causal=false) 分支，让 JIT 在 lower_mha 里用 causal
                // 参数控制 attention mask (ARCH-ENCODER-ATTENTION-NOCAUSAL)。
                if ctx.supports_flash_attention() || ctx.is_encoder() {
                    let config = FlashAttentionConfig {
                        num_heads: ctx.num_heads(),
                        num_kv_heads: ctx.num_kv_heads(),
                        head_dim: ctx.head_dim(),
                        scale: Some(1.0 / (ctx.head_dim() as f32).sqrt()),
                        causal: !ctx.is_encoder(),
                    };
                    node.op = FusedOp::FlashAttention(config);
                } else {
                    let config = GQAConfig {
                        num_heads: ctx.num_heads(),
                        num_kv_heads: ctx.num_kv_heads(),
                        num_groups: ctx.num_heads() / ctx.num_kv_heads(),
                        head_dim: ctx.head_dim(),
                        sliding_window: 0,
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
                0 // LEGAL: num_experts=0 表示运行时动态确定专家数
            });
            let top_k = find_top_k(topk).unwrap_or(2); // LEGAL: top_k=2 是 MoE 的行业标准默认值
            let capacity_factor = find_capacity_factor(dispatch).unwrap_or(1.0); // LEGAL: capacity_factor=1.0 是 MoE 的行业标准默认值

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
            .with_inputs(vec![format!("{}_in", name)])
            .with_outputs(vec![format!("{}_out", name)])
    }

    fn make_linear_node(name: &str, op_type: &str, activation: &str, weight: &str) -> FusedNode {
        FusedNode::new(name, FusedOp::Atomic(AtomicOp::new(op_type)))
            .with_inputs(vec![activation.to_string(), weight.to_string()])
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
    fn fused_qkv_norm_rope_detected() {
        // Gemma 4 pattern: QKV + QkNorm + ValueNorm + Q-RoPE + K-RoPE 融合识别。
        // 构造节点时所有 input/output 名严格对齐,模拟 template.rs::expand_dual_rope
        // 已把 DualRotaryEmbedding 拆成独立 Q/K RoPE 后的图形态。
        let pass = FusedQkvNormRopeFusionPass;
        let ctx = OptimizationContext::default();

        let mk = |name: &str,
                  op_type: &str,
                  inputs: Vec<&str>,
                  outputs: Vec<&str>,
                  attrs: Vec<(&str, crate::graph::types::AttrValue)>| {
            let mut node = FusedNode::new(name, FusedOp::Atomic(AtomicOp::new(op_type)))
                .with_inputs(inputs.into_iter().map(String::from).collect())
                .with_outputs(outputs.into_iter().map(String::from).collect());
            for (k, v) in attrs {
                node.attributes.insert(k.to_string(), v);
            }
            node
        };

        let graph = FusedGraph {
            nodes: vec![
                mk("layer_0_q_proj", "MatMul", vec!["hidden", "w_q"], vec!["q"], vec![]),
                mk("layer_0_k_proj", "MatMul", vec!["hidden", "w_k"], vec!["k"], vec![]),
                mk("layer_0_v_proj", "MatMul", vec!["hidden", "w_v"], vec!["v"], vec![]),
                mk("layer_0_qk_norm", "QkNorm",
                    vec!["q", "k"], vec!["q_normed", "k_normed"], vec![]),
                mk("layer_0_v_norm", "ValueNorm",
                    vec!["v"], vec!["v_normed"], vec![]),
                mk("layer_0_rope_q", "RotaryEmbedding",
                    vec!["q_normed"], vec!["q_rope"], vec![
                        ("theta", crate::graph::types::AttrValue::Float(1_000_000.0)),
                        ("partial", crate::graph::types::AttrValue::Float(0.25)),
                    ]),
                mk("layer_0_rope_k", "RotaryEmbedding",
                    vec!["k_normed"], vec!["k_rope"], vec![
                        ("theta", crate::graph::types::AttrValue::Float(1_000_000.0)),
                        ("partial", crate::graph::types::AttrValue::Float(0.25)),
                    ]),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.qkv_norm_rope_fusions, 1,
            "应识别出 1 个 FusedQkvNormRope 融合");
        assert_eq!(fused.nodes.len(), 1, "7 个 atomic 节点应被融合为 1 个 FusedQkvNormRope 节点");

        match &fused.nodes[0].op {
            FusedOp::FusedQkvNormRope(cfg) => {
                assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
                assert!((cfg.rope_partial - 0.25).abs() < 1e-6);
            }
            other => panic!("期望 FusedQkvNormRope, 实际 {other:?}"),
        }

        // 融合节点 inputs 应为 [activation, w_q, w_k, w_v]
        assert_eq!(fused.nodes[0].inputs, vec!["hidden", "w_q", "w_k", "w_v"]);
        // 输出为 [q_rope, k_rope, v_normed]
        assert_eq!(fused.nodes[0].outputs, vec!["q_rope", "k_rope", "v_normed"]);
    }

    #[test]
    fn fused_qkv_norm_rope_rejects_when_norm_missing() {
        // 没有 QkNorm/ValueNorm 的标准 QKV+RoPE 序列不应触发 FusedQkvNormRope
        // (交给下游 FusedQkvRopeFusionPass 处理)。
        let pass = FusedQkvNormRopeFusionPass;
        let ctx = OptimizationContext::default();

        let graph = FusedGraph {
            nodes: vec![
                make_linear_node("layer_0_q_proj", "MatMul", "hidden", "w_q"),
                make_linear_node("layer_0_k_proj", "MatMul", "hidden", "w_k"),
                make_linear_node("layer_0_v_proj", "MatMul", "hidden", "w_v"),
                make_atomic_node("layer_0_rope", "RotaryEmbedding"),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.qkv_norm_rope_fusions, 0,
            "不含 QkNorm/ValueNorm 的序列不应触发 FusedQkvNormRope");
        // 节点全部保持 atomic
        assert_eq!(fused.nodes.len(), 4);
        for node in &fused.nodes {
            assert!(matches!(node.op, FusedOp::Atomic(_)),
                "不匹配时应保持 atomic 形态: {:?}", node.op);
        }
    }

    #[test]
    fn fused_qkv_norm_rope_detected_via_graph_optimizer() {
        // 集成测试: 通过 GraphOptimizer 真实调用路径验证 pass 已被 register 并生效
        // (非孤岛模块铁律 NO_ISLAND_MODULE)。
        use crate::graph::optimizer::GraphOptimizer;
        use crate::loader::onnx::{OnnxAttribute, OnnxAttributeValue, OnnxGraph, OnnxNode};

        let mk_node = |name: &str,
                       op_type: &str,
                       inputs: Vec<&str>,
                       outputs: Vec<&str>,
                       attrs: Vec<(&str, OnnxAttributeValue)>| {
            let mut attributes = std::collections::HashMap::new();
            for (k, v) in attrs {
                attributes.insert(
                    k.to_string(),
                    OnnxAttribute {
                        name: k.to_string(),
                        value: v,
                        doc_string: String::new(),
                        ref_attr_name: None,
                        attr_type: None,
                    },
                );
            }
            OnnxNode {
                name: name.to_string(),
                op_type: op_type.to_string(),
                domain: String::new(),
                inputs: inputs.into_iter().map(String::from).collect(),
                outputs: outputs.into_iter().map(String::from).collect(),
                attributes,
            }
        };

        // graph.outputs 必须声明 q_rope/k_rope/v_normed, 否则 DeadCodeElimination
        // pass 会删除融合节点 (其输出不在 graph.outputs 中视为死代码)。
        let mk_value_info = |name: &str| crate::loader::onnx::OnnxValueInfo {
            name: name.to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: std::collections::HashMap::new(),
        };

        let onnx_graph = OnnxGraph {
            name: "gemma4_layer".to_string(),
            doc_string: String::new(),
            nodes: vec![
                mk_node("layer_0_q_proj", "MatMul", vec!["hidden", "w_q"], vec!["q"], vec![]),
                mk_node("layer_0_k_proj", "MatMul", vec!["hidden", "w_k"], vec!["k"], vec![]),
                mk_node("layer_0_v_proj", "MatMul", vec!["hidden", "w_v"], vec!["v"], vec![]),
                mk_node("layer_0_qk_norm", "QkNorm",
                    vec!["q", "k"], vec!["q_normed", "k_normed"], vec![]),
                mk_node("layer_0_v_norm", "ValueNorm",
                    vec!["v"], vec!["v_normed"], vec![]),
                mk_node("layer_0_rope_q", "RotaryEmbedding",
                    vec!["q_normed"], vec!["q_rope"], vec![
                        ("theta", OnnxAttributeValue::Float(10000.0)),
                        ("partial", OnnxAttributeValue::Float(1.0)),
                    ]),
                mk_node("layer_0_rope_k", "RotaryEmbedding",
                    vec!["k_normed"], vec!["k_rope"], vec![
                        ("theta", OnnxAttributeValue::Float(10000.0)),
                        ("partial", OnnxAttributeValue::Float(1.0)),
                    ]),
            ],
            inputs: vec![],
            outputs: vec![
                mk_value_info("q_rope"),
                mk_value_info("k_rope"),
                mk_value_info("v_normed"),
            ],
            value_info: vec![],
            initializers: std::collections::HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: std::collections::HashMap::new(),
        };

        let optimizer = GraphOptimizer::new(OptimizationContext::default());
        let fused = optimizer.optimize(&onnx_graph).unwrap();

        assert_eq!(fused.stats.qkv_norm_rope_fusions, 1,
            "GraphOptimizer 注册路径必须触发 FusedQkvNormRope 融合");
        assert!(
            fused.nodes.iter().any(|n| matches!(n.op, FusedOp::FusedQkvNormRope(_))),
            "融合后应存在 FusedQkvNormRope 节点"
        );
    }

    #[test]
    fn fused_qkv_rope_detected() {
        let pass = FusedQkvRopeFusionPass;
        let ctx = OptimizationContext::default();
        let graph = FusedGraph {
            nodes: vec![
                make_linear_node("layer_0_q_proj", "MatMul", "hidden", "w_q"),
                make_linear_node("layer_0_k_proj", "MatMul", "hidden", "w_k"),
                make_linear_node("layer_0_v_proj", "MatMul", "hidden", "w_v"),
                make_atomic_node("layer_0_rope", "RotaryEmbedding"),
            ],
            ..FusedGraph::new()
        };

        let fused = pass.run(graph, &ctx).unwrap();
        assert_eq!(fused.stats.qkv_rope_fusions, 1);
        assert!(matches!(fused.nodes[0].op, FusedOp::FusedQkvRope(_)));
        // Verify fused inputs: [hidden, w_q, w_k, w_v]
        assert_eq!(fused.nodes[0].inputs, vec!["hidden", "w_q", "w_k", "w_v"]);
    }

    #[test]
    fn gqa_fusion_detected() {
        let pass = GQAFusionPass;
        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            num_heads: 32,
            num_kv_heads: 8,
            ..{
                let d = OptimizationContext::default();
                (*d.geometry).clone()
            }
        });
        let ctx = OptimizationContext {
            geometry,
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
