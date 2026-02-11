//! 图类型定义

use std::collections::HashMap;
use std::collections::HashSet;

use crate::loader::TensorProvider;

/// 融合后的图 - 优化器输出
#[derive(Debug, Clone)]
pub struct FusedGraph {
    /// 融合后的节点列表
    pub nodes: Vec<FusedNode>,
    /// 输入名称列表
    pub inputs: Vec<String>,
    /// 输出名称列表
    pub outputs: Vec<String>,
    /// 零拷贝权重绑定（仅保存名称/元信息，数据由 provider 按需提供）
    pub weight_bindings: HashMap<String, WeightBinding>,
    /// 优化统计信息
    pub stats: OptimizationStats,
}

impl FusedGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weight_bindings: HashMap::new(),
            stats: OptimizationStats::default(),
        }
    }

    /// 节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 融合算子数量
    pub fn fused_op_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| !matches!(n.op, FusedOp::Atomic(_)))
            .count()
    }

    /// REQ-ARCH-003: 零拷贝权重绑定
    ///
    /// 仅记录图输入中“需要作为权重”的张量名与元信息，不复制任何权重字节。
    pub fn bind_weights<P: TensorProvider>(&mut self, provider: &P) -> usize {
        let mut produced = HashSet::new();
        for node in &self.nodes {
            for output in &node.outputs {
                produced.insert(output.as_str());
            }
        }

        let graph_inputs: HashSet<&str> = self.inputs.iter().map(String::as_str).collect();
        let mut bound = 0usize;

        for node in &self.nodes {
            for input in &node.inputs {
                if input.is_empty()
                    || produced.contains(input.as_str())
                    || graph_inputs.contains(input.as_str())
                {
                    continue;
                }
                if self.weight_bindings.contains_key(input) {
                    continue;
                }
                if let Some(meta) = provider.tensor_info(input) {
                    self.weight_bindings.insert(
                        input.clone(),
                        WeightBinding {
                            source_name: meta.name,
                            shape: meta.shape,
                            dtype: meta.dtype,
                        },
                    );
                    bound += 1;
                }
            }
        }

        bound
    }
}

impl Default for FusedGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// 融合后的节点
#[derive(Debug, Clone)]
pub struct FusedNode {
    /// 节点名称
    pub name: String,
    /// 融合算子
    pub op: FusedOp,
    /// 输入名称列表
    pub inputs: Vec<String>,
    /// 输出名称列表
    pub outputs: Vec<String>,
    /// 节点属性
    pub attributes: HashMap<String, AttrValue>,
}

impl FusedNode {
    pub fn new(name: impl Into<String>, op: FusedOp) -> Self {
        Self {
            name: name.into(),
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_inputs(mut self, inputs: Vec<String>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<String>) -> Self {
        self.outputs = outputs;
        self
    }
}

/// 融合算子类型
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOp {
    /// FlashAttention 融合
    FlashAttention(FlashAttentionConfig),
    /// SwiGLU 融合 (gate * silu(up))
    SwiGLU(SwiGLUConfig),
    /// RoPE 融合
    RoPE(RoPEConfig),
    /// QKV + RoPE 融合
    FusedQkvRope(FusedQkvRopeConfig),
    /// RMSNorm + Linear 融合
    FusedRMSLinear(FusedRMSLinearConfig),
    /// 原子操作（未融合）
    Atomic(AtomicOp),
}

impl FusedOp {
    /// 是否是融合算子
    pub fn is_fused(&self) -> bool {
        !matches!(self, FusedOp::Atomic(_))
    }

    /// 获取算子名称
    pub fn name(&self) -> &str {
        match self {
            FusedOp::FlashAttention(_) => "FlashAttention",
            FusedOp::SwiGLU(_) => "SwiGLU",
            FusedOp::RoPE(_) => "RoPE",
            FusedOp::FusedQkvRope(_) => "FusedQkvRope",
            FusedOp::FusedRMSLinear(_) => "FusedRMSLinear",
            FusedOp::Atomic(op) => &op.op_type,
        }
    }
}

/// FlashAttention 配置
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FlashAttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: Option<f32>,
    pub causal: bool,
}

/// SwiGLU 配置
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SwiGLUConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

/// RoPE 配置
#[derive(Debug, Clone, PartialEq)]
pub struct RoPEConfig {
    pub head_dim: usize,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub interleaved: bool,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            interleaved: false,
        }
    }
}

/// QKV + RoPE 融合配置
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FusedQkvRopeConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
}

/// RMSNorm + Linear 融合配置
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FusedRMSLinearConfig {
    pub hidden_size: usize,
    pub eps: f32,
}

/// 原子操作（未融合的 ONNX 算子）
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicOp {
    pub op_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightBinding {
    pub source_name: String,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
}

impl AtomicOp {
    pub fn new(op_type: impl Into<String>) -> Self {
        Self {
            op_type: op_type.into(),
        }
    }
}

/// 属性值
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

/// 优化统计信息
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// 原始节点数
    pub original_nodes: usize,
    /// 优化后节点数
    pub optimized_nodes: usize,
    /// FlashAttention 融合数
    pub flash_attention_fusions: usize,
    /// SwiGLU 融合数
    pub swiglu_fusions: usize,
    /// RoPE 融合数
    pub rope_fusions: usize,
    /// QKV+RoPE 融合数
    pub qkv_rope_fusions: usize,
    /// RMSNorm+Linear 融合数
    pub rms_linear_fusions: usize,
    /// 消除的死代码节点数
    pub dead_code_eliminated: usize,
}

impl OptimizationStats {
    /// 总融合数
    pub fn total_fusions(&self) -> usize {
        self.flash_attention_fusions
            + self.swiglu_fusions
            + self.rope_fusions
            + self.qkv_rope_fusions
            + self.rms_linear_fusions
    }

    /// 节点减少率
    pub fn reduction_ratio(&self) -> f32 {
        if self.original_nodes == 0 {
            0.0
        } else {
            1.0 - (self.optimized_nodes as f32 / self.original_nodes as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_graph_default() {
        let graph = FusedGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.fused_op_count(), 0);
    }

    #[test]
    fn fused_op_names() {
        assert_eq!(
            FusedOp::FlashAttention(Default::default()).name(),
            "FlashAttention"
        );
        assert_eq!(FusedOp::SwiGLU(Default::default()).name(), "SwiGLU");
        assert_eq!(FusedOp::Atomic(AtomicOp::new("MatMul")).name(), "MatMul");
    }

    #[test]
    fn optimization_stats_ratio() {
        let stats = OptimizationStats {
            original_nodes: 100,
            optimized_nodes: 60,
            flash_attention_fusions: 10,
            swiglu_fusions: 5,
            ..Default::default()
        };
        assert_eq!(stats.total_fusions(), 15);
        assert!((stats.reduction_ratio() - 0.4).abs() < 0.01);
    }
}
