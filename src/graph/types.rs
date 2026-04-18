//! 图类型定义

use std::collections::HashMap;
use std::collections::HashSet;

use crate::loader::TensorProvider;

/// 判断一个 tensor 是否作为 MatMul/Gemm 节点的 weight (input index 1) 使用。
///
/// ARCH-WEIGHT-ROLE-BY-NODE: 权重角色 (Linear weight / Embedding table / bias /
/// Norm param) 由节点 op_type 和 input 位置确定, **不按 tensor 名字 grep**。
/// 这避免与具体模型命名约定 (XLM-R 的 `embeddings.word_embeddings`, LLaMA 的
/// `embed_tokens`, GGUF 的 `token_embd` 等) 耦合。
///
/// 对 Linear weight → true (HF SafeTensors 需要 transpose 到 canonical [K, N])。
/// 对 Gather table / bias / norm param → false (不 transpose)。
/// FusedOp (FlashAttention/SwiGLU/RoPE 等) 携带多个权重, 当前保守处理: 若任意
/// 使用点是 MatMul-like fused op 且 input index > 0, 视为 Linear weight。
pub(crate) fn is_linear_matmul_weight(
    input_name: &str,
    role_map: &HashMap<&str, Vec<(&FusedOp, usize)>>,
) -> bool {
    let Some(roles) = role_map.get(input_name) else { return false; };
    for (op, idx) in roles {
        let is_linear_role = match op {
            // Atomic MatMul/Gemm: input[1] 是 Linear weight。
            FusedOp::Atomic(atomic) if *idx > 0 => {
                matches!(atomic.op_type.as_str(), "MatMul" | "Gemm")
            }
            // Fused ops 携带多个 Linear 权重 (qkv_proj / gate_proj / up_proj / down_proj
            // 等)。保守认为 idx > 0 的 input 都是 Linear weight。
            // 注: 这些 fused op 目前也会带 bias 参数, bias 在 shape.len()==2 过滤中
            // 已被排除, 故此处无需进一步细分。
            FusedOp::FlashAttention(_)
            | FusedOp::SwiGLU(_)
            | FusedOp::FusedQkvRope(_)
            | FusedOp::FusedRMSLinear(_)
            | FusedOp::GQA(_)
            | FusedOp::MoERouting(_)
            | FusedOp::PerLayerEmbed(_)
                if *idx > 0 => true,
            _ => false,
        };
        if is_linear_role {
            return true;
        }
    }
    false
}

/// 融合后的图 - 优化器输出
#[derive(Debug, Clone, PartialEq)]
pub struct FusedGraph {
    /// 融合后的节点列表
    pub nodes: Vec<FusedNode>,
    /// 输入名称列表
    pub inputs: Vec<String>,
    /// 输出名称列表
    pub outputs: Vec<String>,
    /// 零拷贝权重绑定（仅保存名称/元信息，数据由 provider 按需提供）
    pub weight_bindings: HashMap<String, WeightBinding>,
    /// 量化标注信息
    pub quantization_info: HashMap<String, QuantizationInfo>,
    /// 稀疏张量绑定信息
    pub sparse_tensors: HashMap<String, SparseTensorBinding>,
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
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
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
    ///
    /// `format_needs_transpose`: 该格式的 Linear weight 是否需要语义转置到 `[K, N]`
    /// (SafeTensors/PyTorch: true, ONNX/GGUF: false)。该 flag 只对在 **MatMul/Gemm
    /// 节点作为 weight 角色** 的 2D 参数生效; Gather 表 / bias / LayerNorm 参数
    /// 不转置。角色识别**按 node.op 的 op_type 和 input 位置** (架构元数据),
    /// **不按 tensor 名字 grep** (避免与具体模型命名约定耦合)。
    pub fn bind_weights<P: TensorProvider>(&mut self, provider: &P, format_needs_transpose: bool) -> usize {
        let mut produced = HashSet::new();
        for node in &self.nodes {
            for output in &node.outputs {
                produced.insert(output.as_str());
            }
        }

        let graph_inputs: HashSet<&str> = self.inputs.iter().map(String::as_str).collect();

        // Pass 1: 建立 input_name → Vec<(node_op_kind, input_index)> 映射,
        // 让 bind 阶段能按语义角色 (Linear weight / embedding table / bias) 决定 transpose。
        let mut role_map: HashMap<&str, Vec<(&FusedOp, usize)>> = HashMap::new();
        for node in &self.nodes {
            for (i, input) in node.inputs.iter().enumerate() {
                role_map.entry(input.as_str()).or_default().push((&node.op, i));
            }
        }

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
                    let needs_transpose = format_needs_transpose
                        && meta.shape.len() == 2
                        && is_linear_matmul_weight(input.as_str(), &role_map);
                    self.weight_bindings.insert(
                        input.clone(),
                        WeightBinding {
                            source_name: meta.name,
                            shape: meta.shape,
                            dtype: meta.dtype,
                            data: None,
                            ptr: None,
                            shape_needs_transpose: needs_transpose,
                        },
                    );
                    bound += 1;
                }
            }
        }

        bound
    }

    /// Bind weight shapes with fuzzy prefix matching.
    ///
    /// For YAML-template graphs, node inputs use canonical names (e.g.
    /// `model.layers.0.self_attn.q_proj.weight`) but the actual tensors in
    /// the provider may use different prefixes (or no prefix at all).
    ///
    /// This method tries exact match first, then strips/adds known architecture
    /// prefixes to find the matching tensor metadata.
    pub fn bind_weight_shapes_fuzzy<P: TensorProvider>(&mut self, provider: &P, format_needs_transpose: bool) -> usize {
        const PREFIXES: &[&str] = &["model.", "roberta.", "bert.", "encoder.", "transformer."];

        let mut produced = HashSet::new();
        for node in &self.nodes {
            for output in &node.outputs {
                produced.insert(output.as_str());
            }
        }
        let graph_inputs: HashSet<&str> = self.inputs.iter().map(String::as_str).collect();
        let mut bound = 0usize;

        // 建立 input_name → Vec<(node_op, input_index)> 映射,
        // bind 阶段按 node 语义角色 (MatMul weight / Gather table / bias / norm)
        // 决定是否 transpose。
        let mut role_map: HashMap<&str, Vec<(&FusedOp, usize)>> = HashMap::new();
        for node in &self.nodes {
            for (i, input) in node.inputs.iter().enumerate() {
                role_map.entry(input.as_str()).or_default().push((&node.op, i));
            }
        }

        for node in &self.nodes {
            for input in &node.inputs {
                if input.is_empty()
                    || produced.contains(input.as_str())
                    || graph_inputs.contains(input.as_str())
                {
                    continue;
                }
                // Skip if already bound with a valid shape
                if let Some(existing) = self.weight_bindings.get(input) {
                    if !existing.shape.is_empty() {
                        continue;
                    }
                }

                // Try exact match
                let meta = provider.tensor_info(input)
                    .or_else(|| {
                        // Try stripping prefixes
                        for prefix in PREFIXES {
                            if let Some(stripped) = input.strip_prefix(prefix) {
                                if let Some(m) = provider.tensor_info(stripped) {
                                    return Some(m);
                                }
                            }
                        }
                        // Try adding prefixes
                        for prefix in PREFIXES {
                            let prefixed = format!("{prefix}{input}");
                            if let Some(m) = provider.tensor_info(&prefixed) {
                                return Some(m);
                            }
                        }
                        // Decoder layer HF → GGUF name translation.
                        // E.g. model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
                        for alias in crate::weight_names::all_decoder_weight_aliases(input) {
                            if let Some(m) = provider.tensor_info(&alias) {
                                return Some(m);
                            }
                        }
                        // Decoder global weight aliases (final norm, lm_head, embedding).
                        // These are not layer-indexed and don't fit the layer alias table.
                        let global_aliases: &[&dyn Fn() -> Vec<String>] = &[
                            &crate::weight_names::decoder_final_norm_aliases,
                            &crate::weight_names::lm_head_aliases,
                            &crate::weight_names::decoder_embed_aliases,
                        ];
                        for alias_fn in global_aliases {
                            let aliases = alias_fn();
                            if aliases.contains(input) {
                                // `input` is one of the canonical names; try all siblings.
                                for alias in &aliases {
                                    if alias != input {
                                        if let Some(m) = provider.tensor_info(alias) {
                                            return Some(m);
                                        }
                                    }
                                }
                            }
                        }
                        None
                    });

                if let Some(meta) = meta {
                    // ARCH-WEIGHT-CANONICAL-LAYOUT: shape_needs_transpose 只对
                    // Linear (2D MatMul/Gemm 权重) 生效。Embedding 表 (Gather weight)
                    // 的 shape [vocab, hidden] 不走 MatMul 语义, 不能 transpose,
                    // 否则 vocab 和 hidden 被互换 → Gather 输出 shape 爆炸/错位。
                    // LayerNorm / bias / 1D 参数已通过 shape.len()==2 过滤。
                    let needs_transpose = format_needs_transpose
                        && meta.shape.len() == 2
                        && is_linear_matmul_weight(input.as_str(), &role_map);
                    self.weight_bindings.insert(
                        input.clone(),
                        WeightBinding {
                            source_name: meta.name,
                            shape: meta.shape,
                            dtype: meta.dtype,
                            data: None,
                            ptr: None,
                            shape_needs_transpose: needs_transpose,
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
#[derive(Debug, Clone, PartialEq)]
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
    /// GQA 融合
    GQA(GQAConfig),
    /// MoE routing 融合
    MoERouting(MoERoutingConfig),
    /// Per-Layer Embedding 融合 (Gemma 4 E2B/E4B)
    PerLayerEmbed(PleConfig),
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
            FusedOp::GQA(_) => "GQA",
            FusedOp::MoERouting(_) => "MoERouting",
            FusedOp::PerLayerEmbed(_) => "PerLayerEmbed",
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
    /// 部分旋转比例 (0.0~1.0)。1.0 = 全维度旋转 (标准 RoPE)。
    /// Gemma 4 global 层使用 0.25 (p-RoPE: 仅旋转前 25% 维度)。
    pub partial_ratio: f32,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            interleaved: false,
            partial_ratio: 1.0,
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

/// GQA 融合配置
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GQAConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    /// Sliding-window 注意力窗口大小 (0 = global/full attention)。
    /// Gemma 4 根据 `attention_pattern[layer_idx]` 选择 0 或 config.sliding_window。
    pub sliding_window: usize,
}

/// MoE 路由融合配置
#[derive(Debug, Clone, PartialEq)]
pub struct MoERoutingConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for MoERoutingConfig {
    fn default() -> Self {
        Self {
            num_experts: 0,
            top_k: 2,
            capacity_factor: 1.0,
        }
    }
}

/// Per-Layer Embedding (PLE) 配置 — Gemma 4 E2B/E4B
///
/// PLE 在每个 transformer 层后注入条件信号:
/// ```text
/// ple_token = per_layer_embed_weight[:, layer_i * dim : (layer_i+1) * dim]
/// ple_ctx   = linear_proj(main_embedding)
/// signal    = (ple_ctx + ple_token × √dim) / √2
/// hidden    = hidden + post_mlp_proj(signal)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PleConfig {
    /// 每层注入的 embedding 维度
    pub dim_per_layer: usize,
    /// 模型总层数 (用于 slice per_layer_embed_weight)
    pub num_layers: usize,
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
    /// Embedded weight bytes (constant weights loaded at graph-build time).
    pub data: Option<Vec<u8>>,
    /// Runtime weight pointer injected by the caller (e.g. from WeightsHandle).
    /// When set, takes priority over `data` during execution.
    /// The caller guarantees the pointed-to memory outlives any execution using
    /// this binding.
    pub ptr: Option<*const f32>,
    /// ARCH-WEIGHT-CANONICAL-LAYOUT: 该权重的 shape 是否仍是 HF `[out, in]` 而
    /// 需要语义转置到 `[K, N]`。由 loader 按 WeightFormat 设置:
    ///   SafeTensors/PyTorch: true  (HF 原生 [out, in])
    ///   ONNX: false                (原生 [K, N])
    ///   GGUF: 走独立量化路径, 通常 false
    /// 此 flag 与实际 data layout 独立 — upload 阶段 `normalize_linear_weight_layout`
    /// 已对 data 做物理转置, 但 meta.shape 未同步更新, 这个 flag 捕获这个
    /// 元数据差异, 由 executor 在 shape 推导时按需 swap。
    pub shape_needs_transpose: bool,
}

// Safety: WeightBinding only stores a raw pointer; it does not own the data.
// The caller is responsible for ensuring the pointer remains valid.
unsafe impl Send for WeightBinding {}
unsafe impl Sync for WeightBinding {}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationInfo {
    pub scale: f32,
    pub zero_point: i64,
    pub axis: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseFormat {
    Coo,
    Csr,
    Csc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensorBinding {
    pub format: SparseFormat,
    pub indices: String,
    pub values: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorAttrValue {
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
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
    Strings(Vec<String>),
    Tensor(TensorAttrValue),
    /// Subgraph (used by If/Loop/Scan control flow operators)
    Graph(Box<FusedGraph>),
    /// Multiple subgraphs
    Graphs(Vec<FusedGraph>),
}

/// 优化统计信息
#[derive(Debug, Clone, Default, PartialEq)]
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
    /// GQA 融合数
    pub gqa_fusions: usize,
    /// MoE routing 融合数
    pub moe_routing_fusions: usize,
    /// PLE 融合数
    pub ple_fusions: usize,
    /// 常量折叠节点数
    pub constant_folded_nodes: usize,
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
            + self.gqa_fusions
            + self.moe_routing_fusions
            + self.ple_fusions
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
        assert_eq!(FusedOp::GQA(Default::default()).name(), "GQA");
        assert_eq!(FusedOp::PerLayerEmbed(PleConfig { dim_per_layer: 128, num_layers: 26 }).name(), "PerLayerEmbed");
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

// ============================================================================
// LayeredRequestControl — 逐层请求控制 (§9.3 残差总线)
// ============================================================================

/// 逐层请求控制信号
///
/// 由 RequestState 转换而来，供 PolymorphicExecutor 在逐层执行时
/// 检查 Early-Exit / 层跳过等控制信号。
pub struct LayeredRequestControl {
    /// 目标层索引（Early-Exit 截断点）
    pub target_layer: u32,
    /// 退出标志（非零 = 请求已终止）
    pub exit_flag: std::sync::atomic::AtomicU32,
}

impl LayeredRequestControl {
    /// 检查是否应在指定层退出
    pub fn should_exit_at(&self, layer_idx: u32) -> bool {
        self.exit_flag.load(std::sync::atomic::Ordering::Relaxed) != 0
            || (self.target_layer > 0 && layer_idx >= self.target_layer)
    }
}
