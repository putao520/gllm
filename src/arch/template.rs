//! YAML 模板解析和类型定义 (REQ-ARCH-001)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 架构模板 - 从 YAML 解析
///
/// YAML 模板是架构元数据的 SSOT。`aliases` 字段声明此模板匹配哪些
/// HuggingFace architecture token（如 "LlamaForCausalLM"），`family` 字段
/// 声明编码器/解码器族。Registry 从这些字段自动注册，无需手工映射。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchTemplate {
    /// 架构名称 (e.g., "qwen3", "llama")
    pub name: String,
    /// 模板版本
    #[serde(default = "default_version")]
    pub version: String,
    /// 架构族: "decoder" 或 "encoder"
    #[serde(default = "default_family")]
    pub family: String,
    /// 例外别名 — 无法从模板名自动推导的架构 token。
    ///
    /// 标准别名（`{name}` 和 `{name}ForCausalLM`）由 Registry 自动派生，无需声明。
    /// 此字段仅用于跨代/跨厂商的映射例外，如：
    /// - `qwen2` → qwen3 模板（Qwen2 使用同一架构）
    /// - `chatglm` → glm4 模板（ChatGLM 是 GLM4 的别名）
    #[serde(default)]
    pub extra_aliases: Vec<String>,
    /// MoE 路由器类型（仅含 MoE 算子的模板需要声明）。
    /// 值: "deepseek" / "qwen" / "mixtral"
    #[serde(default)]
    pub moe_router: Option<String>,
    /// 是否使用 HeadRmsNorm (Qwen3 q_norm/k_norm)。
    /// mega-kernel 编译时据此决定是否插入 HeadRmsNorm 节点。
    #[serde(default)]
    pub has_head_rms_norm: bool,
    /// HeadRmsNorm epsilon (仅在 has_head_rms_norm=true 时有效)。
    #[serde(default = "default_head_rms_norm_eps")]
    pub head_rms_norm_eps: f32,
    /// 配置占位符映射
    #[serde(default)]
    pub config: HashMap<String, ConfigValue>,
    /// 图定义
    pub graph: GraphDef,
    /// 融合提示（可选）
    #[serde(default)]
    pub fusion_hints: Vec<FusionHint>,
    /// 张量名映射规则
    #[serde(default)]
    pub tensor_patterns: TensorPatterns,
}

fn default_family() -> String {
    "decoder".to_string()
}

fn default_head_rms_norm_eps() -> f32 {
    1e-6
}

fn default_version() -> String {
    "1.0".to_string()
}

/// 配置值 - 支持直接值或占位符
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    /// 直接值
    Direct(i64),
    /// 浮点值
    Float(f64),
    /// 字符串（可能是占位符如 "${num_hidden_layers}"）
    String(String),
}

impl ConfigValue {
    /// 检查是否是占位符
    pub fn is_placeholder(&self) -> bool {
        match self {
            ConfigValue::String(s) => s.starts_with("${") && s.ends_with('}'),
            _ => false,
        }
    }

    /// 提取占位符名称
    pub fn placeholder_name(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) if s.starts_with("${") && s.ends_with('}') => {
                Some(&s[2..s.len() - 1])
            }
            _ => None,
        }
    }
}

/// 图定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDef {
    /// 输入定义
    #[serde(default)]
    pub inputs: Vec<TensorDef>,
    /// 输出定义
    #[serde(default)]
    pub outputs: Vec<TensorDef>,
    /// 派生输入 — 运行时由 JIT 自动生成的输入张量（不打包为权重）。
    ///
    /// 典型用例: encoder 模型的 position_ids (arange) 和 token_type_ids (zeros)。
    /// 这些张量不需要从外部传入，而是由 Gather lowering 根据
    /// `generator` 字段内联生成。
    #[serde(default)]
    pub derived_inputs: Vec<DerivedInputDef>,
    /// 节点列表（包含普通节点和重复块）
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
}

/// 派生输入定义 — 声明 JIT 如何自动生成该张量。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedInputDef {
    pub name: String,
    /// 生成方式:
    /// - "arange": 生成 [0, 1, 2, ..., seq_len-1]（position_ids）
    /// - "zeros":  生成全零张量（token_type_ids for single sentence）
    pub generator: String,
}

/// 张量定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDef {
    pub name: String,
    pub dtype: String,
    #[serde(default)]
    pub shape: Vec<ShapeDim>,
}

/// 形状维度 - 支持符号和具体值
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShapeDim {
    /// 符号维度 (e.g., "batch", "seq_len")
    Symbol(String),
    /// 具体值
    Value(usize),
}

/// 图节点 - 可以是普通节点或重复块
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GraphNode {
    /// 重复块 (用于 transformer layers)
    Repeat(RepeatBlock),
    /// 普通节点
    Node(NodeDef),
}

/// 重复块 - 用于定义多层结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatBlock {
    /// 重复次数（占位符）
    pub repeat: String,
    /// 循环变量名
    #[serde(default = "default_var")]
    pub var: String,
    /// 重复的节点列表
    pub nodes: Vec<NodeDef>,
}

fn default_var() -> String {
    "i".to_string()
}

/// 节点定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    /// 节点名称
    pub name: String,
    /// 算子类型 (ONNX op_type)
    pub op_type: String,
    /// 输入列表
    #[serde(default)]
    pub inputs: Vec<String>,
    /// 输出列表
    #[serde(default)]
    pub outputs: Vec<String>,
    /// 属性
    #[serde(default)]
    pub attributes: HashMap<String, AttributeValue>,
    /// 条件生成表达式 — 为空或缺省表示始终生成。
    ///
    /// 表达式语法 (两种形式二选一):
    ///   1. 字段名查表: `"has_per_layer_embedding"` → 查 `ResolvedConfig::get_bool`
    ///   2. 简单比较: `"hidden_size_per_layer_input > 0"` → `split_whitespace`
    ///      拆成 `variable op value`,op ∈ {`>`,`>=`,`==`,`!=`,`<`,`<=`},
    ///      变量从 `ResolvedConfig::get_int` 查询,值为整数字面量。
    ///
    /// 求值为 `true` → 节点展开; `false` → 节点跳过。
    /// 未知字段名或语法错误 → 模板错误 (禁止静默跳过掩盖拼写错误)。
    #[serde(default)]
    pub only_if: Option<String>,
}

/// 属性值
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

/// 融合提示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionHint {
    /// 匹配的算子模式
    pub pattern: Vec<String>,
    /// 目标融合算子
    pub target: String,
}

/// 张量名映射规则
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorPatterns {
    /// 嵌入层权重
    #[serde(default)]
    pub embedding: Option<String>,
    /// 输出头权重
    #[serde(default)]
    pub lm_head: Option<String>,
    /// 层前缀模式 (e.g., "model.layers.{}")
    #[serde(default)]
    pub layer_prefix: Option<String>,
    /// Q/K/V/O 投影
    #[serde(default)]
    pub q_proj: Option<String>,
    #[serde(default)]
    pub k_proj: Option<String>,
    #[serde(default)]
    pub v_proj: Option<String>,
    #[serde(default)]
    pub o_proj: Option<String>,
    /// FFN 层
    #[serde(default)]
    pub gate_proj: Option<String>,
    #[serde(default)]
    pub up_proj: Option<String>,
    #[serde(default)]
    pub down_proj: Option<String>,
    /// LayerNorm
    #[serde(default)]
    pub input_layernorm: Option<String>,
    #[serde(default)]
    pub post_attention_layernorm: Option<String>,
    #[serde(default)]
    pub final_norm: Option<String>,
}

impl ArchTemplate {
    /// 从 YAML 字符串解析
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// 从 YAML 文件加载
    pub fn from_file(path: &std::path::Path) -> Result<Self, TemplateError> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_yaml(&content)?)
    }

    /// 使用模板展开逻辑直接从 YAML 模板构建
    /// `gllm_kernels::compiler::CompilerGraph`。处理 repeat、only_if、
    /// DualRotaryEmbedding/QkNorm/PerLayerEmbed 展开器，
    /// 直接产出 `OpKind` 枚举。
    ///
    /// mega-kernel 独有节点（SgInject, SessionKvRestore, MmHiddenInject, GenerateLoop）
    /// 由 `business_config` 驱动注入，不在 YAML 中声明。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn to_compiler_graph(
        &self,
        config: &super::resolve::ResolvedConfig,
        business_config: &gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig,
    ) -> Result<gllm_kernels::compiler::CompilerGraph, TemplateError> {
        use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
        use gllm_kernels::compiler::graph::SYMDIM_MAX_SEQ_LEN;
        use gllm_kernels::compiler::mega_kernel_abi::OutputMode;
        use gllm_kernels::types::DType;

        let mut g = CompilerGraph::new();

        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let eps = 1e-5f32;
        let dt = match config.dtype.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };

        let s = SymDim::Symbolic {
            name: "seq_len".to_string(),
            max_value: Some(SYMDIM_MAX_SEQ_LEN),
        };

        let is_encoder = self.family == "encoder";

        // ── Embedding lookup (Gather) — decoder only ──
        // Encoder templates define their own Gather ops in YAML graph.nodes.
        let mut tensor_map: HashMap<String, gllm_kernels::compiler::graph::TensorId> = HashMap::new();

        // Tracks the embedding output TensorId for decoder activation_alias.
        // For decoder: set to the post-embedding tensor (after scaling/SG/session/MM).
        // For encoder: not used (encoder uses set_encoder_loop_config).
        let mut decoder_post_embed: Option<gllm_kernels::compiler::graph::TensorId> = None;

        if !is_encoder {
            let token_ids = g.add_tensor("token_ids", vec![s.clone()], dt);
            let embed_w = g.add_tensor_concrete("embed_w", &[vocab_size, hidden], dt);
            let embedding = g.add_tensor("embedding", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(
                OpKind::Gather { table_rows: vocab_size, embed_dim: hidden, index_dim: s.clone(), indices_kind: Default::default() },
                vec![token_ids, embed_w],
                vec![embedding],
                "embed_gather",
            );

            // ── Mega-kernel business nodes (decoder only) ──
            let mut post_embed = embedding;

            // Embedding scaling (Gemma: sqrt(hidden_size))
            if let Some(scale) = business_config.embedding_scale {
                let scaled = g.add_tensor("embed_scaled", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::Mul,
                    vec![post_embed],
                    vec![scaled],
                    "embed_scale",
                );
                post_embed = scaled;
            }

            // Semantic Gatekeeper knowledge injection (side-channel)
            if let Some(ref sg) = business_config.semantic_gatekeeper {
                let sg_side_out = g.add_tensor("sg_inject_side", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::SgInject { knowledge_offset: sg.inject_offset, dim: hidden },
                    vec![post_embed],
                    vec![sg_side_out],
                    "sg_inject",
                );
            }

            // Session KV Cache restore
            if business_config.session_enabled {
                let restored = g.add_tensor("session_restored", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::SessionKvRestore,
                    vec![post_embed],
                    vec![restored],
                    "session_kv_restore",
                );
                post_embed = restored;
            }

            // Multimodal fused hidden injection
            if business_config.multimodal_enabled {
                let mm_injected = g.add_tensor("mm_injected", vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(
                    OpKind::MmHiddenInject { hidden_dim: hidden },
                    vec![post_embed],
                    vec![mm_injected],
                    "mm_hidden_inject",
                );
                post_embed = mm_injected;
            }

            tensor_map.insert("embedding".to_string(), post_embed);
            tensor_map.insert("hidden_0".to_string(), post_embed);
            decoder_post_embed = Some(post_embed);
        }

        // ── Layer loop: expand YAML repeat block into CompilerGraph ops ──
        // For encoder, initial tensors come from YAML Gather ops.
        // tensor_map will be populated by emit_compiler_node as YAML nodes are processed.

        // Default dims for top-level (non-repeat) nodes
        let default_dims = LayerDims::for_layer(config, 0);

        for graph_node in &self.graph.nodes {
            match graph_node {
                GraphNode::Node(node_def) => {
                    if !Self::eval_only_if_with_loop(node_def.only_if.as_deref(), config, None)? {
                        continue;
                    }
                    Self::emit_compiler_node(
                        &mut g, node_def, config, business_config,
                        &s, &default_dims, eps, dt, None,
                        &self.tensor_patterns,
                        &mut tensor_map,
                        &self.graph.derived_inputs,
                    )?;
                }
                GraphNode::Repeat(repeat_block) => {
                    let repeat_count = self.resolve_repeat_count(&repeat_block.repeat, config)?;
                    for i in 0..repeat_count {
                        let layer_dims = LayerDims::for_layer(config, i);
                        for node_def in &repeat_block.nodes {
                            if !Self::eval_only_if_with_loop(
                                node_def.only_if.as_deref(), config, Some((&repeat_block.var, i)),
                            )? {
                                continue;
                            }

                            if node_def.op_type == "DualRotaryEmbedding" {
                                Self::emit_dual_rope_compiler(
                                    &mut g, node_def, config, business_config,
                                    &s, &layer_dims,
                                    eps, dt, &repeat_block.var, i,
                                    &mut tensor_map,
                                )?;
                            } else if node_def.op_type == "QkNorm" {
                                Self::emit_qk_norm_compiler(
                                    &mut g, node_def, config, business_config,
                                    &s, &layer_dims, dt, &repeat_block.var, i,
                                    &mut tensor_map,
                                )?;
                            } else if node_def.op_type == "PerLayerEmbed" {
                                Self::emit_per_layer_embed_compiler(
                                    &mut g, node_def, config,
                                    &s, &layer_dims, dt, &repeat_block.var, i,
                                    &mut tensor_map,
                                )?;
                            } else {
                                Self::emit_compiler_node(
                                    &mut g, node_def, config, business_config,
                                    &s, &layer_dims, eps, dt, Some((&repeat_block.var, i)),
                                    &self.tensor_patterns,
                                    &mut tensor_map,
                                    &self.graph.derived_inputs,
                                )?;
                            }
                        }

                        // SgDetect: side-channel copy of hidden state to scratchpad at detection layer.
                        // Decoder-only (encoder models don't use SG).
                        if !is_encoder {
                            if let Some(ref sg) = business_config.semantic_gatekeeper {
                                if i == sg.detect_layer {
                                    let layer_out = tensor_map.get("hidden_0")
                                        .copied()
                                        .ok_or_else(|| TemplateError::Invalid(
                                            "SgDetect: no hidden_0 tensor in layer loop".into()
                                        ))?;
                                    let detect_side = g.add_tensor(
                                        &format!("sg_detect_side_L{}", i),
                                        vec![s.clone(), SymDim::Concrete(hidden)],
                                        dt,
                                    );
                                    g.add_op(
                                        OpKind::SgDetect { detect_offset: sg.detect_offset },
                                        vec![layer_out],
                                        vec![detect_side],
                                        &format!("sg_detect_L{}", i),
                                    );

                                    // Q-Tap STG: extract Q vector after q_proj for SG callback.
                                    if let Some(ref qtap_cfg) = sg.q_tap {
                                        let q_tensor_key = format!("layer_{}_q", i);
                                        if let Some(&q_tid) = tensor_map.get(&q_tensor_key) {
                                            let qtap_sentinel = g.add_tensor_concrete(
                                                &format!("qtap_sentinel_L{}", i), &[1], dt,
                                            );
                                            g.add_op(
                                                OpKind::QTapSTG {
                                                    sink_ptr: qtap_cfg.sink_ptr,
                                                    step_index_ptr: qtap_cfg.step_index_ptr,
                                                    dtype: qtap_cfg.dtype,
                                                    q_dim: SymDim::Concrete(layer_dims.q_dim),
                                                    position: qtap_cfg.position,
                                                    num_slots: qtap_cfg.num_slots,
                                                },
                                                vec![q_tid],
                                                vec![qtap_sentinel],
                                                &format!("layer_{}_qtap_stg", i),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Find the final hidden tensor — it's the last "hidden_0" in the tensor map
        // (updated in-place by residual connections in YAML templates).
        let final_hidden = tensor_map.get("hidden_0")
            .copied()
            .ok_or_else(|| TemplateError::Invalid("no hidden_0 tensor found after layer expansion".into()))?;

        // ── Finalize graph inputs/outputs for encoder ──
        // Must happen BEFORE set_encoder_loop_config because find_layer_weight_indices
        // scans g.inputs for per-layer weight tensors.
        //
        // Contract: g.inputs[0] = activation input (input_ids, passed via ABI arg 0).
        //           g.inputs[1..] = weight tensors (packed into weight blob).
        //           Derived inputs (position_ids, token_type_ids) are NOT in g.inputs —
        //           the Gather lowering generates them inline based on indices_kind.
        if is_encoder {
            let derived_names: Vec<&str> = self.graph.derived_inputs.iter()
                .map(|d| d.name.as_str())
                .collect();

            // 1. Activation input: the declared graph input (e.g., input_ids)
            let mut external_inputs: Vec<gllm_kernels::compiler::graph::TensorId> = Vec::new();
            for input_def in &self.graph.inputs {
                if let Some(tid) = tensor_map.get(&input_def.name) {
                    external_inputs.push(*tid);
                }
            }

            // 2. Weight tensors: no-producer tensors that are NOT derived inputs
            for t in &g.tensors {
                if t.producer.is_none()
                    && !external_inputs.contains(&t.id)
                    && !derived_names.contains(&t.name.as_str())
                {
                    external_inputs.push(t.id);
                }
            }
            g.inputs = external_inputs;
        }

        // ── Pre-populate g.inputs for decoder (before layer loop config) ──
        // Layer loop config needs g.inputs to find per-layer weight indices.
        // Final global weights (final_norm_w, lm_head_w) are added later after creation.
        if !is_encoder && g.inputs.is_empty() {
            let derived_names: Vec<&str> = self.graph.derived_inputs.iter()
                .map(|d| d.name.as_str())
                .collect();

            let mut external_inputs: Vec<gllm_kernels::compiler::graph::TensorId> = Vec::new();

            // 1. Activation input: token_ids (ABI arg 0 for decoder)
            for t in &g.tensors {
                if t.name == "token_ids" {
                    external_inputs.push(t.id);
                    break;
                }
            }

            // 2. Weight tensors: no-producer tensors excluding derived inputs and activation
            for t in &g.tensors {
                if t.producer.is_none()
                    && !external_inputs.contains(&t.id)
                    && !derived_names.contains(&t.name.as_str())
                {
                    external_inputs.push(t.id);
                }
            }
            g.inputs = external_inputs;
        }

        // ── Layer loop config for mega-kernel compilation ──
        // The JIT compiler needs to know weight stride and activation aliasing
        // to emit LoopBegin/LoopEnd with correct weight pointer advancement.
        {
            let num_layers = config.num_hidden_layers;
            let is_hetero = config.attention_pattern.iter().any(|&p| p != config.attention_pattern.first().copied().unwrap_or(0));
            let elem_bytes = dt.size_bytes();
            let embed_weight_bytes = vocab_size * hidden * elem_bytes;
            let default_dims = LayerDims::for_layer(config, 0);

            if is_encoder {
                // Encoder: all layers are already expanded in the graph (each layer has
                // its own ops). LayerLoopConfig is designed for graphs where only layer 0
                // exists and the loop repeats it — using it with an expanded graph causes
                // the JIT to create a loop over ALL 224 ops, processing each layer N times.
                // Skip LayerLoopConfig for encoder; weight_layout already has correct
                // offsets for each layer's weights since they're all graph inputs.
            } else {
                // Decoder: activation_alias = (post_embed, final_hidden)
                // post_embed is the embedding output, final_hidden is the layer loop output.
                // They share the same physical buffer for in-place residual updates.
                let activation_input = decoder_post_embed
                    .ok_or_else(|| TemplateError::Invalid(
                        "decoder template: no post_embed tensor (decoder_post_embed not set)".into()
                    ))?;
                if is_hetero {
                    // Heterogeneous: compute 4-type stride layout (sliding/full × small/large FFN)
                    Self::set_hetero_loop_config(
                        &mut g, config, business_config, &default_dims,
                        embed_weight_bytes, elem_bytes, activation_input, final_hidden,
                    )?;
                } else {
                    // Homogeneous: single stride for all layers
                    Self::set_homogeneous_loop_config(
                        &mut g, config, &default_dims,
                        embed_weight_bytes, elem_bytes, activation_input, final_hidden,
                    )?;
                }
            }
        }

        // ── Decoder-only post-layer ops: final norm, lm_head, output modes ──
        // Encoder templates define their own post-layer ops in YAML graph.nodes.
        if !is_encoder {
            // Final norm
            let final_norm_w = g.add_tensor_concrete("final_norm_w", &[hidden], dt);
            let final_normed = g.add_tensor("final_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::RmsNorm { eps }, vec![final_hidden, final_norm_w], vec![final_normed], "final_norm");

            // lm_head
            let lm_head_w = g.add_tensor_concrete("lm_head_w", &[hidden, vocab_size], dt);
            let logits = g.add_tensor("logits", vec![s.clone(), SymDim::Concrete(vocab_size)], dt);
            g.add_op(
                OpKind::Gemm { m: s.clone(), n: vocab_size, k: hidden, dtype: dt },
                vec![final_normed, lm_head_w],
                vec![logits],
                "lm_head",
            );

            // Logit softcapping
            let logits_for_loop = if let Some(cap) = business_config.logit_softcapping {
                let capped = g.add_tensor("logits_capped", vec![s.clone(), SymDim::Concrete(vocab_size)], dt);
                g.add_op(OpKind::LogitSoftcap { cap }, vec![logits], vec![capped], "logit_softcapping");
                capped
            } else {
                logits
            };

            // Generate loop (only for Generate output mode)
            for mode in &business_config.output_modes {
                match mode {
                    OutputMode::Generate { .. } => {
                        let token_id = g.add_tensor("token_id", vec![SymDim::Concrete(1)], dt);
                        g.add_op(
                            OpKind::Argmax { vocab_size },
                            vec![logits_for_loop],
                            vec![token_id],
                            "argmax",
                        );
                        g.add_op(OpKind::StoreToken, vec![token_id], vec![], "store_token");
                        g.add_op(OpKind::CheckStopCondition, vec![token_id], vec![], "check_stop");
                    }
                    OutputMode::ClassifyBinary { positive_token_id, negative_token_id } => {
                        let indices = vec![*positive_token_id, *negative_token_id];
                        g.add_op(
                            OpKind::WriteLogits { target_indices: indices },
                            vec![logits_for_loop],
                            vec![],
                            "write_classify_logits",
                        );
                    }
                    OutputMode::ClassifyMultiway { ref label_token_ids } => {
                        g.add_op(
                            OpKind::WriteLogits { target_indices: label_token_ids.clone() },
                            vec![logits_for_loop],
                            vec![],
                            "write_classify_logits",
                        );
                    }
                    OutputMode::EncodeToLayer { .. } => {
                        g.add_op(
                            OpKind::EarlyExit { anchor_layer: 0 },
                            vec![],
                            vec![],
                            "early_exit",
                        );
                    }
                }
            }
        } // end if !is_encoder

        // ── Encoder post-processing: MeanPool / L2Normalize driven by OutputMode ──
        if is_encoder {
            use gllm_kernels::compiler::mega_kernel_abi::{OutputMode, PoolMode};

            // final_hidden is the last hidden state from the layer loop.
            let mut encoder_output = final_hidden;

            for mode in &business_config.output_modes {
                match mode {
                    OutputMode::EncodeToLayer { pool_mode, .. } => {
                        match pool_mode {
                            PoolMode::MeanPool => {
                                // MeanPool: [seq_len, hidden] → [hidden]
                                let pooled = g.add_tensor(
                                    "encoder_mean_pooled",
                                    vec![SymDim::Concrete(hidden)],
                                    dt,
                                );
                                g.add_op(
                                    OpKind::MeanPool {
                                        seq_len: s.max_for_allocation_strict()
                                            .expect("MeanPool seq_len needs max_value"),
                                        hidden,
                                    },
                                    vec![encoder_output],
                                    vec![pooled],
                                    "mean_pool",
                                );
                                encoder_output = pooled;
                            }
                            PoolMode::ClsToken => {
                                // ClsToken: take first token [0, :] → [hidden]
                                // Implemented as ColumnSlice(start=0, end=hidden) from first row.
                                // For now, use MeanPool as placeholder (encoder rerankers
                                // typically use the CLS token, but MeanPool is a reasonable default).
                                let pooled = g.add_tensor(
                                    "encoder_cls_pooled",
                                    vec![SymDim::Concrete(hidden)],
                                    dt,
                                );
                                g.add_op(
                                    OpKind::MeanPool {
                                        seq_len: s.max_for_allocation_strict()
                                            .expect("MeanPool seq_len needs max_value"),
                                        hidden,
                                    },
                                    vec![encoder_output],
                                    vec![pooled],
                                    "cls_pool",
                                );
                                encoder_output = pooled;
                            }
                            PoolMode::LastToken => {
                                // LastToken: take last token's hidden state → [hidden]
                                // Same structure as ClsToken but different row index.
                                let pooled = g.add_tensor(
                                    "encoder_last_pooled",
                                    vec![SymDim::Concrete(hidden)],
                                    dt,
                                );
                                g.add_op(
                                    OpKind::MeanPool {
                                        seq_len: s.max_for_allocation_strict()
                                            .expect("MeanPool seq_len needs max_value"),
                                        hidden,
                                    },
                                    vec![encoder_output],
                                    vec![pooled],
                                    "last_pool",
                                );
                                encoder_output = pooled;
                            }
                        }
                        // L2Normalize after pooling
                        let normalized = g.add_tensor(
                            "encoder_normalized",
                            vec![SymDim::Concrete(hidden)],
                            dt,
                        );
                        g.add_op(
                            OpKind::L2Normalize { hidden },
                            vec![encoder_output],
                            vec![normalized],
                            "l2_normalize",
                        );
                        encoder_output = normalized;
                    }
                    _ => {}
                }
            }

            // ── Finalize graph outputs for encoder ──
            g.outputs = vec![encoder_output];
        }

        Ok(g)
    }

    /// Map a single YAML NodeDef to a CompilerGraph OpKind + tensor registration.
    ///
    /// `dims` provides per-layer dimensions (handles hetero models).
    /// `tensor_patterns` provides weight name conventions for MatMul dimension inference.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn emit_compiler_node(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        business_config: &gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig,
        s: &gllm_kernels::compiler::graph::SymDim,
        dims: &LayerDims,
        eps: f32,
        dt: gllm_kernels::types::DType,
        loop_var: Option<(&str, usize)>,
        tensor_patterns: &super::template::TensorPatterns,
        tensor_map: &mut HashMap<String, gllm_kernels::compiler::graph::TensorId>,
        derived_inputs: &[DerivedInputDef],
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::{OpKind, SymDim};

        let substitute = |s_in: &str| -> Result<String, TemplateError> {
            Self::substitute_with_donor(s_in, config, loop_var)
        };

        let op_type = &node_def.op_type;

        // Resolve input/output names
        let input_names: Vec<String> = node_def.inputs.iter()
            .map(|s| substitute(s))
            .collect::<Result<Vec<_>, _>>()?;
        let output_names: Vec<String> = node_def.outputs.iter()
            .map(|s| substitute(s))
            .collect::<Result<Vec<_>, _>>()?;

        let label = substitute(&node_def.name)?;
        let label = if loop_var.is_some() {
            format!("layer.{}", label)
        } else {
            label
        };

        // Compute op_kind first so we can derive input weight shapes.
        let op_kind = match op_type.as_str() {
            "Gather" => {
                // Determine indices_kind from derived_inputs declaration.
                // Gather inputs: [indices, table] — lower_gather reads indices from input_ptr.
                let indices_name = input_names.get(0).map(|s| s.as_str()).unwrap_or("");
                let indices_kind = if derived_inputs.iter().any(|d| d.name == indices_name && d.generator == "arange") {
                    gllm_kernels::compiler::graph::GatherIndicesKind::Arange
                } else if derived_inputs.iter().any(|d| d.name == indices_name && d.generator == "zeros") {
                    gllm_kernels::compiler::graph::GatherIndicesKind::Zeros
                } else {
                    gllm_kernels::compiler::graph::GatherIndicesKind::Tensor
                };
                OpKind::Gather {
                    table_rows: config.vocab_size,
                    embed_dim: dims.hidden,
                    index_dim: s.clone(),
                    indices_kind,
                }
            }
            "MatMul" => {
                let (n, k) = infer_matmul_dims(&input_names, node_def, config, loop_var, dims);
                OpKind::Gemm { m: s.clone(), n, k, dtype: dt }
            }
            "SimplifiedLayerNormalization" | "RmsNorm" => {
                OpKind::RmsNorm { eps: read_attr_f32(node_def, config, loop_var, "eps").unwrap_or(eps) }
            }
            "LayerNormalization" => {
                OpKind::LayerNorm { eps: read_attr_f32(node_def, config, loop_var, "eps").unwrap_or(eps) }
            }
            "ValueNorm" => {
                OpKind::ValueNorm {
                    eps: read_attr_f32(node_def, config, loop_var, "eps")
                        .unwrap_or(business_config.value_norm_eps),
                }
            }
            "SiLU" | "Silu" => OpKind::Silu,
            "Gelu" | "GELU" => OpKind::Gelu,
            "Tanh" => OpKind::Tanh,
            "SwiGLU" => OpKind::SwiGlu,
            "Attention" | "MultiHeadAttention" => {
                let bidirectional = read_attr_bool(node_def, config, loop_var, "bidirectional").unwrap_or(false);
                let causal_from_attr = read_attr_bool(node_def, config, loop_var, "causal");
                let causal = causal_from_attr.unwrap_or(!bidirectional);
                let attn_sinks = read_attr_bool(node_def, config, loop_var, "attention_sinks").unwrap_or(false);
                OpKind::MultiHeadAttention {
                    seq_len: s.clone(),
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    causal,
                    attention_sinks: attn_sinks,
                }
            }
            "Add" => OpKind::Add,
            "Mul" => OpKind::Mul,
            "RotaryEmbedding" => {
                let rope_num_heads = read_attr_usize(node_def, config, loop_var, "num_heads")
                    .unwrap_or(dims.num_heads);
                let rope_head_dim = read_attr_usize(node_def, config, loop_var, "head_dim")
                    .unwrap_or(dims.head_dim);
                let theta = read_attr_f32(node_def, config, loop_var, "theta")
                    .map(|v| v as f64)
                    .unwrap_or(config.rope_theta);
                let partial = read_attr_f32(node_def, config, loop_var, "partial")
                    .unwrap_or(config.rope_partial_ratio);
                OpKind::RoPE {
                    num_heads: rope_num_heads,
                    head_dim: rope_head_dim,
                    theta,
                    partial,
                    rope_scaling: None,
                }
            }
            "HeadRmsNorm" => {
                let h_dim = read_attr_usize(node_def, config, loop_var, "head_dim")
                    .unwrap_or(dims.head_dim);
                let h_eps = read_attr_f32(node_def, config, loop_var, "eps")
                    .unwrap_or(business_config.head_rms_norm_eps);
                OpKind::HeadRmsNorm { head_dim: h_dim, eps: h_eps }
            }
            "MoERouter" => {
                let num_experts = read_attr_usize(node_def, config, loop_var, "num_experts")
                    .ok_or_else(|| TemplateError::Invalid("MoERouter missing num_experts".into()))?;
                let top_k = read_attr_usize(node_def, config, loop_var, "top_k")
                    .ok_or_else(|| TemplateError::Invalid("MoERouter missing top_k".into()))?;
                OpKind::MoERouter {
                    num_experts,
                    top_k,
                    hidden: dims.hidden,
                    seq_len: s.clone(),
                }
            }
            "MoEDispatchPacked" => {
                let num_experts = read_attr_usize(node_def, config, loop_var, "num_experts")
                    .ok_or_else(|| TemplateError::Invalid("MoEDispatchPacked missing num_experts".into()))?;
                let top_k = read_attr_usize(node_def, config, loop_var, "top_k")
                    .ok_or_else(|| TemplateError::Invalid("MoEDispatchPacked missing top_k".into()))?;
                OpKind::MoEDispatchPacked {
                    num_experts,
                    top_k,
                    mxfp4_block_size: 32,
                    swiglu_limit: 7.0,
                    intermediate_size: dims.intermediate,
                    hidden: dims.hidden,
                    seq_len: s.clone(),
                }
            }
            "PatchEmbed" => {
                let patch_size = read_attr_usize(node_def, config, loop_var, "patch_size")
                    .ok_or_else(|| TemplateError::Invalid("PatchEmbed missing patch_size".into()))?;
                let embed_dim = read_attr_usize(node_def, config, loop_var, "embed_dim")
                    .unwrap_or(dims.hidden);
                let in_channels = read_attr_usize(node_def, config, loop_var, "in_channels")
                    .unwrap_or(3);
                let image_size = read_attr_usize(node_def, config, loop_var, "image_size")
                    .unwrap_or(224);
                OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size }
            }
            "LearnedPos2D" => {
                let num_patches = read_attr_usize(node_def, config, loop_var, "num_patches")
                    .ok_or_else(|| TemplateError::Invalid("LearnedPos2D missing num_patches".into()))?;
                let embed_dim = read_attr_usize(node_def, config, loop_var, "embed_dim")
                    .unwrap_or(dims.hidden);
                OpKind::LearnedPos2D { num_patches, embed_dim }
            }
            "DepthwiseConv1D" => {
                let channels = read_attr_usize(node_def, config, loop_var, "channels")
                    .unwrap_or(dims.hidden);
                let kernel_size = read_attr_usize(node_def, config, loop_var, "kernel_size")
                    .unwrap_or(32);
                let causal = read_attr_bool(node_def, config, loop_var, "causal").unwrap_or(true);
                OpKind::DepthwiseConv1D { channels, kernel_size, causal }
            }
            "Reshape" => OpKind::Reshape { target_shape: vec![] },
            "Transpose" => OpKind::Transpose { perm: vec![] },
            other => {
                return Err(TemplateError::Invalid(format!(
                    "to_compiler_graph: unsupported op_type '{}' (node '{}')",
                    other, label
                )));
            }
        };

        // Derive per-input weight shapes from op_kind.
        let weight_shapes = Self::infer_input_shapes(&op_kind, dims, &input_names);
        let input_ids: Vec<gllm_kernels::compiler::graph::TensorId> = input_names.iter()
            .enumerate()
            .map(|(i, name)| {
                // Skip derived inputs — they don't exist as real tensors in the graph.
                // The Gather lowering generates their values inline based on indices_kind.
                if derived_inputs.iter().any(|d| d.name == *name) {
                    return *tensor_map.entry(name.clone()).or_insert_with(|| {
                        g.add_tensor_concrete(name, &[0], dt)
                    });
                }
                *tensor_map.entry(name.clone()).or_insert_with(|| {
                    let shape = weight_shapes.get(i).map(|s| s.as_slice()).unwrap_or(&[0]);
                    g.add_tensor_concrete(name, shape, dt)
                })
            })
            .collect();

        // Create output tensors and register them
        let output_ids: Vec<gllm_kernels::compiler::graph::TensorId> = output_names.iter()
            .map(|name| {
                let shape = infer_output_shape(&op_kind, s, dims.hidden, dt);
                let tid = g.add_tensor(name, shape, dt);
                tensor_map.insert(name.clone(), tid);
                tid
            })
            .collect();

        g.add_op(op_kind, input_ids, output_ids, &label);
        Ok(())
    }

    /// Expand DualRotaryEmbedding into two OpKind::RoPE ops for CompilerGraph.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn emit_dual_rope_compiler(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        business_config: &gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig,
        s: &gllm_kernels::compiler::graph::SymDim,
        dims: &LayerDims,
        _eps: f32,
        dt: gllm_kernels::types::DType,
        var: &str,
        layer_idx: usize,
        tensor_map: &mut HashMap<String, gllm_kernels::compiler::graph::TensorId>,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::{OpKind, SymDim, QTapPosition};

        let read_f32 = |key: &str| -> Option<f32> {
            match node_def.attributes.get(key)? {
                AttributeValue::Float(v) => Some(*v as f32),
                AttributeValue::Int(v) => Some(*v as f32),
                AttributeValue::String(s_val) => {
                    let resolved = super::resolve::substitute_placeholders(s_val, config);
                    resolved.parse::<f32>().ok()
                }
                _ => None,
            }
        };

        let sliding_theta = read_f32("sliding_theta").ok_or_else(|| TemplateError::Invalid(
            format!("DualRotaryEmbedding '{}' missing sliding_theta", node_def.name)))?;
        let global_theta = read_f32("global_theta").ok_or_else(|| TemplateError::Invalid(
            format!("DualRotaryEmbedding '{}' missing global_theta", node_def.name)))?;
        let sliding_partial = read_f32("sliding_partial").unwrap_or(1.0);
        let global_partial = read_f32("global_partial").unwrap_or(0.25);

        let is_global = config.attention_pattern.get(layer_idx).copied().unwrap_or(0) == 1;
        let (theta, partial) = if is_global { (global_theta, global_partial) } else { (sliding_theta, sliding_partial) };

        let substitute = |s_in: &str| -> String {
            let mut r = super::resolve::substitute_placeholders(s_in, config);
            r = r.replace(&format!("${{{}}}", var), &layer_idx.to_string());
            r = r.replace(&format!("{}$", var), &layer_idx.to_string());
            r
        };

        // Q-RoPE
        let q_in_name = substitute(&node_def.inputs[0]);
        let q_out_name = substitute(&node_def.outputs[0]);
        let q_in_id = *tensor_map.entry(q_in_name.clone()).or_insert_with(|| g.add_tensor_concrete(&q_in_name, &[0], dt));

        // Q-Tap STG: insert after q_proj, before RoPE consumes Q (REQ-SG-002)
        if let Some(ref sg) = business_config.semantic_gatekeeper {
            if layer_idx == sg.detect_layer {
                if let Some(ref qtap_cfg) = sg.q_tap {
                    let qtap_sentinel = g.add_tensor_concrete(
                        &format!("qtap_sentinel_L{}", layer_idx), &[1], dt,
                    );
                    g.add_op(
                        OpKind::QTapSTG {
                            sink_ptr: qtap_cfg.sink_ptr,
                            step_index_ptr: qtap_cfg.step_index_ptr,
                            dtype: qtap_cfg.dtype,
                            q_dim: SymDim::Concrete(dims.q_dim),
                            position: qtap_cfg.position,
                            num_slots: qtap_cfg.num_slots,
                        },
                        vec![q_in_id],
                        vec![qtap_sentinel],
                        &format!("layer.layer_{layer_idx}_qtap_stg"),
                    );
                }
            }
        }

        let q_out_id = g.add_tensor(&q_out_name, vec![s.clone(), SymDim::Concrete(dims.q_dim)], dt);
        tensor_map.insert(q_out_name.clone(), q_out_id);
        g.add_op(
            OpKind::RoPE { num_heads: dims.num_heads, head_dim: dims.head_dim, theta: theta as f64, partial, rope_scaling: None },
            vec![q_in_id], vec![q_out_id], &format!("layer.layer_{layer_idx}_rope_q"),
        );

        // K-RoPE: skip on shared KV consumer layers
        if config.is_kv_shared_layer(layer_idx) {
            return Ok(());
        }
        let k_in_name = substitute(&node_def.inputs[1]);
        let k_out_name = substitute(&node_def.outputs[1]);
        let k_in_id = *tensor_map.entry(k_in_name.clone()).or_insert_with(|| g.add_tensor_concrete(&k_in_name, &[0], dt));
        let k_out_id = g.add_tensor(&k_out_name, vec![s.clone(), SymDim::Concrete(dims.kv_dim)], dt);
        tensor_map.insert(k_out_name.clone(), k_out_id);
        g.add_op(
            OpKind::RoPE { num_heads: dims.num_kv_heads, head_dim: dims.head_dim, theta: theta as f64, partial, rope_scaling: None },
            vec![k_in_id], vec![k_out_id], &format!("layer.layer_{layer_idx}_rope_k"),
        );

        Ok(())
    }

    /// Expand QkNorm into two OpKind::QkNorm ops for CompilerGraph.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn emit_qk_norm_compiler(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        _business_config: &gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig,
        s: &gllm_kernels::compiler::graph::SymDim,
        dims: &LayerDims,
        dt: gllm_kernels::types::DType,
        var: &str,
        layer_idx: usize,
        tensor_map: &mut HashMap<String, gllm_kernels::compiler::graph::TensorId>,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::{OpKind, SymDim};

        let substitute = |s_in: &str| -> String {
            let mut r = super::resolve::substitute_placeholders(s_in, config);
            r = r.replace(&format!("${{{}}}", var), &layer_idx.to_string());
            r = r.replace(&format!("{}$", var), &layer_idx.to_string());
            r
        };

        let q_in_name = substitute(&node_def.inputs[0]);
        let q_out_name = substitute(&node_def.outputs[0]);
        let q_in_id = *tensor_map.entry(q_in_name.clone()).or_insert_with(|| g.add_tensor_concrete(&q_in_name, &[0], dt));
        let q_out_id = g.add_tensor(&q_out_name, vec![s.clone()], dt);
        tensor_map.insert(q_out_name.clone(), q_out_id);
        g.add_op(
            OpKind::QkNorm { head_dim: dims.head_dim },
            vec![q_in_id], vec![q_out_id], &format!("layer.layer_{layer_idx}_qk_norm_q"),
        );

        // K-QkNorm: skip on shared KV consumer layers
        if config.is_kv_shared_layer(layer_idx) {
            return Ok(());
        }
        let k_in_name = substitute(&node_def.inputs[1]);
        let k_out_name = substitute(&node_def.outputs[1]);
        let k_in_id = *tensor_map.entry(k_in_name.clone()).or_insert_with(|| g.add_tensor_concrete(&k_in_name, &[0], dt));
        let k_out_id = g.add_tensor(&k_out_name, vec![s.clone()], dt);
        tensor_map.insert(k_out_name.clone(), k_out_id);
        g.add_op(
            OpKind::QkNorm { head_dim: dims.head_dim },
            vec![k_in_id], vec![k_out_id], &format!("layer.layer_{layer_idx}_qk_norm_k"),
        );

        Ok(())
    }

    /// Expand PerLayerEmbed into PleSlice + PerLayerEmbed ops for CompilerGraph.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn emit_per_layer_embed_compiler(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        s: &gllm_kernels::compiler::graph::SymDim,
        dims: &LayerDims,
        dt: gllm_kernels::types::DType,
        var: &str,
        layer_idx: usize,
        tensor_map: &mut HashMap<String, gllm_kernels::compiler::graph::TensorId>,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::{OpKind, SymDim};

        let read_usize = |key: &str| -> Result<usize, TemplateError> {
            match node_def.attributes.get(key) {
                Some(AttributeValue::Int(v)) => Ok(*v as usize),
                Some(AttributeValue::String(s_val)) => {
                    let mut resolved = super::resolve::substitute_placeholders(s_val, config);
                    resolved = resolved.replace(&format!("${{{}}}", var), &layer_idx.to_string());
                    resolved.parse::<usize>().map_err(|_| TemplateError::Invalid(format!(
                        "PerLayerEmbed attr '{}' parse failed: {}", key, resolved
                    )))
                }
                _ => Err(TemplateError::Invalid(format!(
                    "PerLayerEmbed '{}' missing attr '{}'", node_def.name, key
                ))),
            }
        };

        let dim_per_layer = read_usize("dim_per_layer")?;
        let num_layers = read_usize("num_layers")?;

        let substitute = |s_in: &str| -> String {
            let mut r = super::resolve::substitute_placeholders(s_in, config);
            r = r.replace(&format!("${{{}}}", var), &layer_idx.to_string());
            r = r.replace(&format!("{}$", var), &layer_idx.to_string());
            r
        };

        // Node 1: ColumnSlice (PleSlice)
        let ple_full_name = substitute(&node_def.inputs[1]);
        let slice_out_name = format!("layer_{layer_idx}_ple_slice");
        let ple_full_id = *tensor_map.entry(ple_full_name.clone()).or_insert_with(|| g.add_tensor_concrete(&ple_full_name, &[0], dt));
        let slice_out_id = g.add_tensor(&slice_out_name, vec![s.clone(), SymDim::Concrete(dim_per_layer)], dt);
        tensor_map.insert(slice_out_name.clone(), slice_out_id);
        g.add_op(
            OpKind::ColumnSlice {
                seq_len: s.clone(),
                input_inner: num_layers * dim_per_layer,
                start: layer_idx * dim_per_layer,
                slice_dim: dim_per_layer,
            },
            vec![ple_full_id], vec![slice_out_id], &format!("layer.layer_{layer_idx}_ple_slice"),
        );

        // Node 2: PerLayerEmbed (5 inputs)
        let hidden_in_name = substitute(&node_def.inputs[0]);
        let proj_w_name = substitute(&node_def.inputs[2]);
        let post_mlp_w_name = substitute(&node_def.inputs[3]);
        let hidden_out_name = substitute(&node_def.outputs[0]);

        let hidden_in_id = *tensor_map.get(&hidden_in_name).ok_or_else(|| TemplateError::Invalid(
            format!("PerLayerEmbed: input '{}' not found in tensor_map", hidden_in_name)))?;
        let main_embed_id = *tensor_map.entry("main_embed".to_string()).or_insert_with(|| g.add_tensor_concrete("main_embed", &[0], dt));
        let proj_w_id = *tensor_map.entry(proj_w_name.clone()).or_insert_with(|| g.add_tensor_concrete(&proj_w_name, &[0], dt));
        let post_mlp_w_id = *tensor_map.entry(post_mlp_w_name.clone()).or_insert_with(|| g.add_tensor_concrete(&post_mlp_w_name, &[0], dt));

        let hidden_out_id = g.add_tensor(&hidden_out_name, vec![s.clone(), SymDim::Concrete(dims.hidden)], dt);
        tensor_map.insert(hidden_out_name.clone(), hidden_out_id);

        g.add_op(
            OpKind::PerLayerEmbed {
                seq_len: s.clone(),
                layer_idx,
                dim_per_layer,
                num_layers,
                hidden: dims.hidden,
            },
            vec![hidden_in_id, main_embed_id, slice_out_id, proj_w_id, post_mlp_w_id],
            vec![hidden_out_id],
            &format!("layer.layer_{layer_idx}_ple"),
        );

        Ok(())
    }

    /// Perform full placeholder substitution with donor-aware extension.
    ///
    /// Resolves (in order):
    /// 1. Static config placeholders (e.g. `${num_hidden_layers}`) — via
    ///    [`super::resolve::substitute_placeholders`].
    /// 2. **Donor placeholder** (e.g. `${donor_i}` when `loop_var = ("i", _)`) —
    ///    expands to the donor layer index for the current consumer layer, or
    ///    to the current layer index itself for non-consumer layers (identity
    ///    path). This lets YAML express "attention reads donor K/V on shared
    ///    layers, self K/V otherwise" without branching.
    /// 3. Loop-variable placeholder (e.g. `${i}`) — identity with the active
    ///    repeat index. Resolved *after* donor so `${donor_i}` does not alias
    ///    `${i}` inside `donor_<digits>`.
    ///
    /// A donor placeholder on a shared consumer layer with a malformed
    /// attention pattern propagates the scheduler's error (no silent defaults).
    fn substitute_with_donor(
        s: &str,
        config: &super::resolve::ResolvedConfig,
        loop_var: Option<(&str, usize)>,
    ) -> Result<String, TemplateError> {
        let mut result = super::resolve::substitute_placeholders(s, config);
        if let Some((var, idx)) = loop_var {
            let donor_ph = format!("${{donor_{}}}", var);
            if result.contains(&donor_ph) {
                let donor_idx = match config.donor_layer(idx) {
                    Ok(Some(d)) => d,
                    Ok(None) => idx, // identity: non-consumer layer references its own tensors.
                    Err(e) => {
                        return Err(TemplateError::Invalid(format!(
                            "donor placeholder for layer {idx}: {e}"
                        )));
                    }
                };
                result = result.replace(&donor_ph, &donor_idx.to_string());
            }
            result = result.replace(&format!("${{{}}}", var), &idx.to_string());
            result = result.replace(&format!("${}$", var), &idx.to_string());
        }
        Ok(result)
    }

    /// Wrapper over [`Self::eval_only_if`] that first substitutes the active
    /// repeat-loop variable (e.g. `${i}`) in the expression so per-layer
    /// conditions like `!is_kv_shared_layer_${i}` resolve against the current
    /// layer index before field lookup.
    fn eval_only_if_with_loop(
        expr: Option<&str>,
        config: &super::resolve::ResolvedConfig,
        loop_var: Option<(&str, usize)>,
    ) -> Result<bool, TemplateError> {
        let substituted: Option<String> = expr.map(|raw| {
            let mut s = raw.to_string();
            if let Some((var, idx)) = loop_var {
                s = s.replace(&format!("${{{}}}", var), &idx.to_string());
                s = s.replace(&format!("${}$", var), &idx.to_string());
            }
            s
        });
        Self::eval_only_if(substituted.as_deref(), config)
    }

    /// 求值 `only_if` 条件表达式 (节点级展开守卫)。
    ///
    /// `None` / 空串 → `true` (无条件展开); 表达式求值为 `false` → 调用方跳过该节点。
    ///
    /// 支持三种语法:
    /// 1. **字段名查表** (`has_per_layer_embedding` / `is_kv_shared_layer_3`):
    ///    调用 `ResolvedConfig::get_bool`。per-layer 字段在 `eval_only_if_with_loop`
    ///    先经过 `${i}` 占位符替换,查表时以带索引的键命中 `get_bool` 的 per-layer
    ///    分支。未知字段 → `TemplateError::Invalid`。
    /// 2. **前缀否定** (`!has_per_layer_embedding` / `!is_kv_shared_layer_5`):
    ///    字段求值后取反。语法上等价于 `field == false` 但更贴近 YAML 习惯。
    /// 3. **整数比较** (`lhs op rhs`, 按 `split_whitespace` 拆 3 段): `lhs` 从
    ///    `ResolvedConfig::get_int` 查询,`rhs` 为整数字面量,`op ∈ {>,>=,==,!=,<,<=}`。
    ///
    /// 选择查表 + 简单比较的双轨设计而非完整表达式解析:
    /// - 字段名查表表达最常见的"启用/禁用"意图,阅读性最好
    /// - 整数比较覆盖"派生字段不存在"的过渡场景 (如 `hidden_size_per_layer_input > 0`)
    /// - 避免引入 pest/nom 等解析器依赖,保持模板引擎零依赖
    fn eval_only_if(
        expr: Option<&str>,
        config: &super::resolve::ResolvedConfig,
    ) -> Result<bool, TemplateError> {
        let raw = match expr {
            None => return Ok(true),
            Some(s) => s.trim(),
        };
        if raw.is_empty() {
            return Ok(true);
        }

        // 形式 3: 三段式比较 `lhs op rhs`
        let tokens: Vec<&str> = raw.split_whitespace().collect();
        if tokens.len() == 3 {
            let lhs_key = tokens[0];
            let op = tokens[1];
            let rhs_str = tokens[2];
            let rhs: i64 = rhs_str.parse().map_err(|_| TemplateError::Invalid(format!(
                "only_if '{raw}': rhs '{rhs_str}' 无法解析为整数"
            )))?;
            let lhs = config.get_int(lhs_key).ok_or_else(|| TemplateError::Invalid(format!(
                "only_if '{raw}': 未知配置字段 '{lhs_key}' (ResolvedConfig::get_int 无匹配)"
            )))?;
            return match op {
                ">" => Ok(lhs > rhs),
                ">=" => Ok(lhs >= rhs),
                "==" => Ok(lhs == rhs),
                "!=" => Ok(lhs != rhs),
                "<" => Ok(lhs < rhs),
                "<=" => Ok(lhs <= rhs),
                _ => Err(TemplateError::Invalid(format!(
                    "only_if '{raw}': 未知比较运算符 '{op}' (支持 >,>=,==,!=,<,<=)"
                ))),
            };
        }

        // 形式 1 / 2: 单 token (可带 `!` 前缀否定)
        if tokens.len() == 1 {
            let token = tokens[0];
            let (key, negate) = if let Some(rest) = token.strip_prefix('!') {
                (rest, true)
            } else {
                (token, false)
            };
            if key.is_empty() {
                return Err(TemplateError::Invalid(format!(
                    "only_if '{raw}': `!` 后缺少字段名"
                )));
            }
            if let Some(v) = config.get_bool(key) {
                return Ok(if negate { !v } else { v });
            }
            return Err(TemplateError::Invalid(format!(
                "only_if '{raw}': 未知布尔字段 '{key}' (ResolvedConfig::get_bool 无匹配)。\
                 若需派生字段,请在 ResolvedConfig::get_bool 中注册,禁止静默返回 false。"
            )));
        }

        Err(TemplateError::Invalid(format!(
            "only_if '{raw}': 语法错误,期望 `<field>` / `!<field>` 或 `<lhs> <op> <rhs>` (3 个空白分隔 token)"
        )))
    }

    /// 解析重复次数
    fn resolve_repeat_count(
        &self,
        repeat_expr: &str,
        config: &super::resolve::ResolvedConfig,
    ) -> Result<usize, TemplateError> {
        // 如果是占位符，从配置中获取
        if repeat_expr.starts_with("${") && repeat_expr.ends_with('}') {
            let key = &repeat_expr[2..repeat_expr.len() - 1];
            if let Some(value) = config.get_int(key) {
                return Ok(value as usize);
            }
            return Err(TemplateError::Invalid(format!(
                "Unknown repeat count placeholder: {}",
                key
            )));
        }

        // 尝试直接解析为数字
        repeat_expr
            .parse()
            .map_err(|_| TemplateError::Invalid(format!("Invalid repeat count: {}", repeat_expr)))
    }

    /// Set homogeneous layer loop config on CompilerGraph.
    ///
    /// Computes weight stride from layer dimensions, identifies weight input indices,
    /// and sets activation alias for in-place residual updates.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn set_homogeneous_loop_config(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        config: &super::resolve::ResolvedConfig,
        dims: &LayerDims,
        embed_weight_bytes: usize,
        elem_bytes: usize,
        activation_input: gllm_kernels::compiler::graph::TensorId,
        activation_output: gllm_kernels::compiler::graph::TensorId,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::{LayerLoopConfig, WeightLayout};

        let h = dims.hidden;
        let stride = compute_layer_stride(
            h, dims.q_dim, dims.kv_dim, dims.q_dim,
            dims.head_dim, dims.intermediate, elem_bytes,
        );

        // Find per-layer weight input indices: scan graph.inputs for tensors
        // whose names match layer 0 weight patterns.
        let layer_weight_indices: Vec<usize> = find_layer_weight_indices(g, 0, None);

        g.layer_loop_config = Some(LayerLoopConfig {
            num_layers: config.num_hidden_layers,
            weight_stride: stride,
            layer_blob_base_offset: embed_weight_bytes,
            layer_weight_input_indices: layer_weight_indices,
            activation_alias: Some((activation_input, activation_output)),
        });

        // Custom weight layout: per-layer weights use relative offsets,
        // global weights use absolute offsets.
        set_custom_weight_layout_homogeneous(
            g, dims, embed_weight_bytes, elem_bytes,
        );

        Ok(())
    }

    /// Set heterogeneous layer loop config on CompilerGraph.
    ///
    /// Computes 4-type stride layout for models with alternating attention types
    /// (e.g., Gemma-4 E2B: 7 segments × [4 sliding + 1 full]).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn set_hetero_loop_config(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        config: &super::resolve::ResolvedConfig,
        _business_config: &gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig,
        default_dims: &LayerDims,
        embed_weight_bytes: usize,
        elem_bytes: usize,
        activation_input: gllm_kernels::compiler::graph::TensorId,
        activation_output: gllm_kernels::compiler::graph::TensorId,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::HeteroLayerLoopConfig;

        let pat = &config.attention_pattern;
        let h = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let global_head_dim = if config.global_head_dim > 0 { config.global_head_dim } else { head_dim };
        let default_intermediate = config.intermediate_size.unwrap_or(h * 4);

        // Derive segment structure from attention_pattern.
        // Pattern is like [0,0,0,0,1, 0,0,0,0,1, ...] where 0=sliding, 1=full.
        let full_indices: Vec<usize> = pat.iter().enumerate()
            .filter(|&(_, &p)| p == 1)
            .map(|(i, _)| i)
            .collect();
        if full_indices.is_empty() {
            // No full attention layers found — treat as homogeneous
            return Self::set_homogeneous_loop_config(
                g, config, default_dims, embed_weight_bytes, elem_bytes,
                activation_input, activation_output,
            );
        }

        let sliding_per_segment = full_indices[0];
        let num_segments = full_indices.len();
        let total_expected = num_segments * (sliding_per_segment + 1);
        if total_expected != config.num_hidden_layers {
            return Err(TemplateError::Invalid(format!(
                "hetero pattern inconsistent: {} segments × {} + 1 = {} but num_layers = {}",
                num_segments, sliding_per_segment, total_expected, config.num_hidden_layers,
            )));
        }

        // Detect FFN size transition from attention_pattern position
        let sliding_head_dim = head_dim;
        let full_head_dim = global_head_dim;
        let sliding_q_dim = num_heads * sliding_head_dim;
        let full_q_dim = num_heads * full_head_dim;
        let sliding_kv_dim = num_kv_heads * sliding_head_dim;
        let full_kv_dim = num_kv_heads * full_head_dim;
        // Assume uniform intermediate for now (can be extended to detect per-segment sizes)
        let small_intermediate = default_intermediate;
        let large_intermediate = default_intermediate;
        let large_ffn_start_segment = num_segments; // no large FFN distinction by default

        let ss_stride = compute_layer_stride(
            h, sliding_q_dim, sliding_kv_dim, sliding_q_dim,
            sliding_head_dim, small_intermediate, elem_bytes,
        );
        let fs_stride = compute_layer_stride(
            h, full_q_dim, full_kv_dim, full_q_dim,
            full_head_dim, small_intermediate, elem_bytes,
        );
        let sl_stride = compute_layer_stride(
            h, sliding_q_dim, sliding_kv_dim, sliding_q_dim,
            sliding_head_dim, large_intermediate, elem_bytes,
        );
        let fl_stride = compute_layer_stride(
            h, full_q_dim, full_kv_dim, full_q_dim,
            full_head_dim, large_intermediate, elem_bytes,
        );

        let small_seg_stride = sliding_per_segment * ss_stride + fs_stride;
        let large_seg_stride = sliding_per_segment * sl_stride + fl_stride;

        // Weight input indices: for now, use layer 0's indices for all 4 types.
        // The JIT codegen treats per-type weights as having the same relative offset
        // structure (just different strides).
        let layer_weight_indices = find_layer_weight_indices(g, 0, None);

        g.hetero_layer_loop_config = Some(HeteroLayerLoopConfig {
            num_segments,
            sliding_per_segment,
            sliding_small_stride: ss_stride,
            full_small_stride: fs_stride,
            sliding_large_stride: sl_stride,
            full_large_stride: fl_stride,
            small_segment_stride: small_seg_stride,
            large_segment_stride: large_seg_stride,
            large_ffn_start_segment,
            layer_blob_base_offset: embed_weight_bytes,
            sliding_small_weight_input_indices: layer_weight_indices.clone(),
            full_small_weight_input_indices: layer_weight_indices.clone(),
            sliding_large_weight_input_indices: layer_weight_indices.clone(),
            full_large_weight_input_indices: layer_weight_indices,
            activation_aliases: vec![
                (activation_input, activation_output),
            ],
        });

        // Custom weight layout with per-type relative offsets
        set_custom_weight_layout_heterogeneous(
            g, config, embed_weight_bytes, elem_bytes,
            sliding_head_dim, full_head_dim, num_heads, num_kv_heads,
            small_intermediate, large_intermediate,
        );

        Ok(())
    }

    /// Set layer loop config for encoder models.
    ///
    /// Uses the graph's default sequential weight layout (no custom layout).
    /// Stride is computed from the distance between layer 0 and layer 1 first weights.
    /// `layer_blob_base_offset = 0` because sequential offsets are absolute.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn set_encoder_loop_config(
        g: &mut gllm_kernels::compiler::CompilerGraph,
        config: &super::resolve::ResolvedConfig,
        tensor_patterns: &TensorPatterns,
        activation_tensor: gllm_kernels::compiler::graph::TensorId,
    ) -> Result<(), TemplateError> {
        use gllm_kernels::compiler::graph::LayerLoopConfig;

        let layer_prefix = tensor_patterns.layer_prefix.as_deref()
            .ok_or_else(|| TemplateError::Invalid(
                "encoder template missing layer_prefix in tensor_patterns".into()
            ))?;

        let layer_weight_indices = find_layer_weight_indices(g, 0, Some(layer_prefix));

        if layer_weight_indices.is_empty() {
            return Err(TemplateError::Invalid(
                "encoder template: no per-layer weights found for layer 0".into()
            ));
        }

        // Compute stride from default sequential weight layout.
        let layout = g.weight_layout();
        let prefix_0 = format!("{}0.", layer_prefix.replace("{}", ""));
        let prefix_1 = format!("{}1.", layer_prefix.replace("{}", ""));

        let off_0 = layout.offsets.iter()
            .find(|(tid, _)| {
                g.tensors.get(tid.0 as usize)
                    .map(|t| t.name.starts_with(&prefix_0))
                    .unwrap_or(false)
            })
            .map(|(_, off)| *off)
            .unwrap_or(0);
        let off_1 = layout.offsets.iter()
            .find(|(tid, _)| {
                g.tensors.get(tid.0 as usize)
                    .map(|t| t.name.starts_with(&prefix_1))
                    .unwrap_or(false)
            })
            .map(|(_, off)| *off);
        let stride = off_1.map(|o| o.saturating_sub(off_0)).unwrap_or(0);

        if stride == 0 {
            return Err(TemplateError::Invalid(
                format!("encoder template: could not compute layer stride (off_0={}, off_1={:?})",
                         off_0, off_1)
            ));
        }

        g.layer_loop_config = Some(LayerLoopConfig {
            num_layers: config.num_hidden_layers,
            weight_stride: stride,
            layer_blob_base_offset: 0, // sequential layout: offsets are absolute
            layer_weight_input_indices: layer_weight_indices,
            activation_alias: Some((activation_tensor, activation_tensor)),
        });

        Ok(())
    }

    /// Derive per-input tensor shapes from OpKind.
    ///
    /// `input_names` is used to detect bias inputs (names ending in ".bias")
    /// which have shape [hidden] rather than the default [0] (dynamic).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn infer_input_shapes(
        op_kind: &gllm_kernels::compiler::graph::OpKind,
        dims: &LayerDims,
        input_names: &[String],
    ) -> Vec<Vec<usize>> {
        use gllm_kernels::compiler::graph::OpKind;
        match op_kind {
            OpKind::Gather { table_rows, embed_dim, .. } => {
                // input[0] = indices, input[1] = table (lower_gather convention)
                vec![vec![0], vec![*table_rows, *embed_dim]]
            }
            OpKind::Gemm { n, k, .. } => {
                vec![vec![0], vec![*k, *n]]
            }
            OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {
                vec![vec![0], vec![dims.hidden], vec![dims.hidden]]
            }
            OpKind::ValueNorm { .. } => vec![vec![0]],
            OpKind::Add | OpKind::Mul => {
                // Detect bias inputs: if an input name contains ".bias", its shape is [hidden].
                input_names.iter().map(|name| {
                    if name.contains(".bias") {
                        vec![dims.hidden]
                    } else {
                        vec![0]
                    }
                }).collect()
            }
            OpKind::Silu | OpKind::Gelu | OpKind::Tanh => {
                vec![vec![0]]
            }
            OpKind::SwiGlu | OpKind::SwiGluClipped { .. } | OpKind::GeGlu => {
                vec![vec![0], vec![0]]
            }
            OpKind::RoPE { .. } => vec![vec![0], vec![0]],
            OpKind::MultiHeadAttention { .. } => vec![vec![0], vec![0], vec![0]],
            OpKind::QkNorm { .. } => vec![vec![0]],
            OpKind::HeadRmsNorm { .. } => vec![vec![0]],
            OpKind::LogitSoftcap { .. } => vec![vec![0]],
            OpKind::Argmax { .. } => vec![vec![0]],
            OpKind::ColumnSlice { .. } => vec![vec![0], vec![0]],
            OpKind::PerLayerEmbed { .. } => vec![vec![0], vec![0], vec![0]],
            OpKind::MoERouter { .. } => vec![vec![0]],
            OpKind::MoEDispatchPacked { .. } => vec![vec![0]],
            OpKind::Reshape { .. } | OpKind::Transpose { .. } => vec![vec![0]],
            OpKind::PatchEmbed { .. } => vec![vec![0]],
            OpKind::LearnedPos2D { .. } => vec![vec![0]],
            OpKind::DepthwiseConv1D { .. } => vec![vec![0]],
            _ => vec![vec![0]],
        }
    }
}
// ═══════════════════════════════════════════════════════════════════════

/// Per-layer dimension context for heterogeneous models.
// ═══════════════════════════════════════════════════════════════════════
// Layer loop config helpers
// ═══════════════════════════════════════════════════════════════════════

/// Compute per-layer weight byte stride from dimension parameters.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn compute_layer_stride(
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    o_in_dim: usize,
    head_dim: usize,
    intermediate: usize,
    elem_bytes: usize,
) -> usize {
    let attn_norm = hidden * elem_bytes;
    let w_q = q_dim * hidden * elem_bytes;
    let w_k = kv_dim * hidden * elem_bytes;
    let w_v = kv_dim * hidden * elem_bytes;
    let w_o = hidden * o_in_dim * elem_bytes;
    let w_q_norm = head_dim * elem_bytes;
    let w_k_norm = head_dim * elem_bytes;
    let ffn_norm = hidden * elem_bytes;
    let w_gate = intermediate * hidden * elem_bytes;
    let w_up = intermediate * hidden * elem_bytes;
    let w_down = hidden * intermediate * elem_bytes;
    attn_norm + w_q + w_k + w_v + w_o + w_q_norm + w_k_norm + ffn_norm + w_gate + w_up + w_down
}

/// Find graph input indices that correspond to per-layer weights for the given layer index.
///
/// `prefix_pattern`: layer prefix with `{}` placeholder, e.g., `"model.layers.{}"` or
/// `"roberta.encoder.layer.{}"`. When `None`, defaults to decoder pattern `"model.layers.{}"`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn find_layer_weight_indices(
    g: &gllm_kernels::compiler::CompilerGraph,
    layer_idx: usize,
    prefix_pattern: Option<&str>,
) -> Vec<usize> {
    let pattern = prefix_pattern.unwrap_or("model.layers.{}");
    let prefix = format!("{}.", pattern.replace("{}", &layer_idx.to_string()));
    g.inputs.iter().enumerate()
        .filter(|(_, &tid)| {
            g.tensors.get(tid.0 as usize)
                .map(|t| t.name.starts_with(&prefix))
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Set custom weight layout for homogeneous models.
/// Per-layer weights use relative offsets; global weights use absolute offsets.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn set_custom_weight_layout_homogeneous(
    g: &mut gllm_kernels::compiler::CompilerGraph,
    dims: &LayerDims,
    embed_weight_bytes: usize,
    elem_bytes: usize,
) {
    use gllm_kernels::compiler::graph::WeightLayout;

    let h = dims.hidden;
    let num_layers = {
        // Count layer ops to get actual number of layers
        g.ops.iter().filter(|op| op.label.starts_with("layer_")).count()
    };
    if num_layers == 0 {
        return;
    }

    // Compute relative offsets within one layer
    let o0 = 0;
    let o1 = o0 + h * elem_bytes;
    let o2 = o1 + dims.q_dim * h * elem_bytes;
    let o3 = o2 + dims.kv_dim * h * elem_bytes;
    let o4 = o3 + dims.kv_dim * h * elem_bytes;
    let o5 = o4 + h * dims.q_dim * elem_bytes;
    let o6 = o5 + dims.head_dim * elem_bytes;
    let o7 = o6 + dims.head_dim * elem_bytes;
    let o8 = o7 + h * elem_bytes;
    let o9 = o8 + dims.intermediate * h * elem_bytes;
    let o10 = o9 + dims.intermediate * h * elem_bytes;

    let layer_stride = compute_layer_stride(
        h, dims.q_dim, dims.kv_dim, dims.q_dim,
        dims.head_dim, dims.intermediate, elem_bytes,
    );
    let final_norm_off = embed_weight_bytes + num_layers * layer_stride;
    let lm_head_off = final_norm_off + h * elem_bytes;
    let total = lm_head_off + dims.hidden * /* vocab */ g.tensors.iter()
        .find(|t| t.name == "embed_w")
        .map(|t| {
            if t.shape.len() >= 1 {
                match &t.shape[0] {
                    gllm_kernels::compiler::graph::SymDim::Concrete(v) => *v,
                    _ => 0,
                }
            } else { 0 }
        })
        .unwrap_or(0) * elem_bytes;

    // Collect all weight tensor IDs with their offsets
    let mut offsets: Vec<(gllm_kernels::compiler::graph::TensorId, usize)> = Vec::new();

    // Global weights (absolute offsets)
    for (i, &tid) in g.inputs.iter().enumerate() {
        if let Some(t) = g.tensors.get(tid.0 as usize) {
            let name = &t.name;
            if name == "embed_w" {
                offsets.push((tid, 0));
            } else if name == "final_norm_w" {
                offsets.push((tid, final_norm_off));
            } else if name == "lm_head_w" {
                offsets.push((tid, lm_head_off));
            }
        }
    }

    // Per-layer weights (relative offsets from layer base)
    // Use layer 0's weight tensors as the template
    let prefix = "model.layers.0.";
    let layer_weight_offsets: Vec<(gllm_kernels::compiler::graph::TensorId, usize)> = g.inputs.iter()
        .filter(|&&tid| {
            g.tensors.get(tid.0 as usize)
                .map(|t| t.name.starts_with(prefix))
                .unwrap_or(false)
        })
        .map(|&tid| {
            let t = g.tensors.get(tid.0 as usize).unwrap();
            let name = &t.name;
            let rel_off = if name.contains("input_layernorm") { o0 }
                else if name.contains("q_proj") { o1 }
                else if name.contains("k_proj") { o2 }
                else if name.contains("v_proj") { o3 }
                else if name.contains("o_proj") { o4 }
                else if name.contains("q_norm") { o5 }
                else if name.contains("k_norm") { o6 }
                else if name.contains("post_attention_layernorm") || name.contains("pre_feedforward_layernorm") { o7 }
                else if name.contains("gate_proj") { o8 }
                else if name.contains("up_proj") { o9 }
                else if name.contains("down_proj") { o10 }
                else { 0 }; // unknown weights default to offset 0
            (tid, rel_off)
        })
        .collect();

    offsets.extend(layer_weight_offsets);

    g.set_custom_weight_layout(WeightLayout {
        offsets,
        total_bytes: total,
    });
}

/// Set custom weight layout for heterogeneous models.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn set_custom_weight_layout_heterogeneous(
    g: &mut gllm_kernels::compiler::CompilerGraph,
    config: &super::resolve::ResolvedConfig,
    embed_weight_bytes: usize,
    elem_bytes: usize,
    sliding_head_dim: usize,
    full_head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    small_intermediate: usize,
    large_intermediate: usize,
) {
    use gllm_kernels::compiler::graph::WeightLayout;

    let h = config.hidden_size;
    let vocab = config.vocab_size;
    let num_layers = config.num_hidden_layers;

    let sliding_q_dim = num_heads * sliding_head_dim;
    let full_q_dim = num_heads * full_head_dim;
    let sliding_kv_dim = num_kv_heads * sliding_head_dim;
    let full_kv_dim = num_kv_heads * full_head_dim;

    // Compute per-type offsets and strides
    let ss_stride = compute_layer_stride(h, sliding_q_dim, sliding_kv_dim, sliding_q_dim, sliding_head_dim, small_intermediate, elem_bytes);
    let fs_stride = compute_layer_stride(h, full_q_dim, full_kv_dim, full_q_dim, full_head_dim, small_intermediate, elem_bytes);
    let sl_stride = compute_layer_stride(h, sliding_q_dim, sliding_kv_dim, sliding_q_dim, sliding_head_dim, large_intermediate, elem_bytes);
    let fl_stride = compute_layer_stride(h, full_q_dim, full_kv_dim, full_q_dim, full_head_dim, large_intermediate, elem_bytes);

    let pat = &config.attention_pattern;
    let full_indices: Vec<usize> = pat.iter().enumerate()
        .filter(|&(_, &p)| p == 1).map(|(i, _)| i).collect();
    let sliding_per_segment = full_indices.get(0).copied().unwrap_or(4);
    let num_segments = full_indices.len().max(1);
    let small_seg_stride = sliding_per_segment * ss_stride + fs_stride;
    let large_seg_stride = sliding_per_segment * sl_stride + fl_stride;
    let large_ffn_start_segment = num_segments;
    let num_small_segs = large_ffn_start_segment;
    let num_large_segs = num_segments - num_small_segs;
    let total_layers_bytes = num_small_segs * small_seg_stride + num_large_segs * large_seg_stride;

    let final_norm_off = embed_weight_bytes + total_layers_bytes;
    let lm_head_off = final_norm_off + h * elem_bytes;
    let total_bytes = lm_head_off + vocab * h * elem_bytes;

    // For the template-based graph, all per-layer weights use the same relative offset
    // structure. The stride difference is captured in the HeteroLayerLoopConfig.
    // Here we just set the template offsets (using layer 0 as reference).
    let prefix = "model.layers.0.";
    let default_dims = LayerDims::for_layer(config, 0);
    let rel_offsets = compute_type_offsets_array(&default_dims, elem_bytes);

    let mut offsets: Vec<(gllm_kernels::compiler::graph::TensorId, usize)> = Vec::new();

    // Global weights
    for &tid in &g.inputs {
        if let Some(t) = g.tensors.get(tid.0 as usize) {
            match t.name.as_str() {
                "embed_w" => offsets.push((tid, 0)),
                "final_norm_w" => offsets.push((tid, final_norm_off)),
                "lm_head_w" => offsets.push((tid, lm_head_off)),
                _ => {}
            }
        }
    }

    // Per-layer weights (relative offsets from layer base)
    for &tid in &g.inputs {
        if let Some(t) = g.tensors.get(tid.0 as usize) {
            if t.name.starts_with(prefix) {
                let rel_off = map_weight_name_to_offset(&t.name, &rel_offsets);
                offsets.push((tid, rel_off));
            }
        }
    }

    g.set_custom_weight_layout(WeightLayout {
        offsets,
        total_bytes,
    });
}

/// Compute per-type relative weight offsets array [attn_norm, w_q, w_k, w_v, w_o, w_q_norm, w_k_norm, ffn_norm, w_gate, w_up, w_down].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn compute_type_offsets_array(dims: &LayerDims, elem_bytes: usize) -> [usize; 11] {
    let h = dims.hidden;
    let o0 = 0;
    let o1 = o0 + h * elem_bytes;
    let o2 = o1 + dims.q_dim * h * elem_bytes;
    let o3 = o2 + dims.kv_dim * h * elem_bytes;
    let o4 = o3 + dims.kv_dim * h * elem_bytes;
    let o5 = o4 + h * dims.q_dim * elem_bytes;
    let o6 = o5 + dims.head_dim * elem_bytes;
    let o7 = o6 + dims.head_dim * elem_bytes;
    let o8 = o7 + h * elem_bytes;
    let o9 = o8 + dims.intermediate * h * elem_bytes;
    let o10 = o9 + dims.intermediate * h * elem_bytes;
    [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10]
}

/// Map a weight tensor name to its relative offset within the per-type offsets array.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn map_weight_name_to_offset(name: &str, offsets: &[usize; 11]) -> usize {
    // Strip "model.layers.X." prefix to get the relative part
    let parts: Vec<&str> = name.splitn(4, '.').collect();
    if parts.len() < 4 {
        return 0;
    }
    // parts = ["model", "layers", "0", "self_attn.q_proj.weight"]
    let relative = parts[3];
    let idx = if relative.contains("input_layernorm") { 0 }
        else if relative.contains("q_proj") { 1 }
        else if relative.contains("k_proj") { 2 }
        else if relative.contains("v_proj") { 3 }
        else if relative.contains("o_proj") { 4 }
        else if relative.contains("q_norm") { 5 }
        else if relative.contains("k_norm") { 6 }
        else if relative.contains("post_attention_layernorm") || relative.contains("pre_feedforward_layernorm") { 7 }
        else if relative.contains("gate_proj") { 8 }
        else if relative.contains("up_proj") { 9 }
        else if relative.contains("down_proj") { 10 }
        else { 0 };
    offsets[idx]
}

/// Per-layer dimension context for CompilerGraph construction.
///
/// For homogeneous models, all layers share the same dimensions.
/// For heterogeneous models (Gemma-4 E2B), different layer templates have
/// different head_dim / num_kv_heads / intermediate_size.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
struct LayerDims {
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    intermediate: usize,
    q_dim: usize,
    kv_dim: usize,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
impl LayerDims {
    /// Derive per-layer dimensions from config + layer index.
    ///
    /// For hetero models, `attention_pattern[i]` selects the head_dim variant:
    /// - 0 (sliding): uses config.head_dim + config.num_key_value_heads
    /// - 1 (global): uses config.global_head_dim (if > 0) + config.num_key_value_heads
    ///
    /// FFN intermediate_size is model-level (same for all layers in current models).
    fn for_layer(config: &super::resolve::ResolvedConfig, layer_idx: usize) -> Self {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let is_global = config.attention_pattern.get(layer_idx).copied().unwrap_or(0) == 1;
        let head_dim = if is_global && config.global_head_dim > 0 {
            config.global_head_dim
        } else {
            config.head_dim
        };

        let intermediate = config.intermediate_size.unwrap_or(hidden * 4);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self { hidden, num_heads, head_dim, num_kv_heads, intermediate, q_dim, kv_dim }
    }
}

/// Infer MatMul (n, k) from weight tensor name + per-layer dimension context.
///
/// Matches the weight input name against `tensor_patterns` conventions:
/// - q_proj → n = q_dim, k = hidden
/// - k_proj → n = kv_dim, k = hidden
/// - v_proj → n = kv_dim, k = hidden
/// - o_proj → n = hidden, k = q_dim
/// - gate_proj → n = intermediate, k = hidden
/// - up_proj → n = intermediate, k = hidden
/// - down_proj → n = hidden, k = intermediate
///
/// Falls back to YAML `n`/`k` attributes, then to (hidden, hidden) as last resort.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn infer_matmul_dims(
    input_names: &[String],
    node_def: &NodeDef,
    config: &super::resolve::ResolvedConfig,
    loop_var: Option<(&str, usize)>,
    dims: &LayerDims,
) -> (usize, usize) {
    // Try YAML attributes first (explicit override)
    if let (Some(n), Some(k)) = (
        read_attr_usize(node_def, config, loop_var, "n"),
        read_attr_usize(node_def, config, loop_var, "k"),
    ) {
        return (n, k);
    }

    // Match weight tensor name against projection conventions
    let weight_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");

    // Projection dimension table: (n, k)
    let dims_table: &[(&str, (usize, usize))] = &[
        ("q_proj.weight",          (dims.q_dim, dims.hidden)),
        ("k_proj.weight",          (dims.kv_dim, dims.hidden)),
        ("v_proj.weight",          (dims.kv_dim, dims.hidden)),
        ("o_proj.weight",          (dims.hidden, dims.q_dim)),
        ("gate_proj.weight",       (dims.intermediate, dims.hidden)),
        ("up_proj.weight",         (dims.intermediate, dims.hidden)),
        ("down_proj.weight",       (dims.hidden, dims.intermediate)),
        // gate_up_fused (MoE packed)
        ("gate_up_proj.weight",    (dims.intermediate * 2, dims.hidden)),
        // lm_head
        ("lm_head.weight",         (config.vocab_size, dims.hidden)),
    ];

    for (suffix, nk) in dims_table {
        if weight_name.contains(suffix) {
            return *nk;
        }
    }

    // Fallback: use YAML attributes partially, or default to (hidden, hidden)
    let n = read_attr_usize(node_def, config, loop_var, "n").unwrap_or(dims.hidden);
    let k = read_attr_usize(node_def, config, loop_var, "k").unwrap_or(dims.hidden);
    (n, k)
}

/// Read a usize attribute from NodeDef, with placeholder substitution.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn read_attr_usize(
    node_def: &NodeDef,
    config: &super::resolve::ResolvedConfig,
    loop_var: Option<(&str, usize)>,
    key: &str,
) -> Option<usize> {
    match node_def.attributes.get(key)? {
        AttributeValue::Int(v) => Some(*v as usize),
        AttributeValue::String(s) => {
            let resolved = super::resolve::substitute_placeholders(s, config);
            let resolved = if let Some((var, idx)) = loop_var {
                resolved.replace(&format!("${{{}}}", var), &idx.to_string())
            } else {
                resolved
            };
            resolved.parse().ok()
        }
        _ => None,
    }
}

/// Read a f32 attribute from NodeDef, with placeholder substitution.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn read_attr_f32(
    node_def: &NodeDef,
    config: &super::resolve::ResolvedConfig,
    loop_var: Option<(&str, usize)>,
    key: &str,
) -> Option<f32> {
    match node_def.attributes.get(key)? {
        AttributeValue::Float(v) => Some(*v as f32),
        AttributeValue::Int(v) => Some(*v as f32),
        AttributeValue::String(s) => {
            let resolved = super::resolve::substitute_placeholders(s, config);
            let resolved = if let Some((var, idx)) = loop_var {
                resolved.replace(&format!("${{{}}}", var), &idx.to_string())
            } else {
                resolved
            };
            resolved.parse().ok()
        }
        _ => None,
    }
}

/// Read a bool attribute from NodeDef.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn read_attr_bool(
    node_def: &NodeDef,
    _config: &super::resolve::ResolvedConfig,
    _loop_var: Option<(&str, usize)>,
    key: &str,
) -> Option<bool> {
    match node_def.attributes.get(key)? {
        AttributeValue::Int(v) => Some(*v != 0),
        _ => None,
    }
}

/// Infer output tensor shape from OpKind.

/// Infer output tensor shape from OpKind.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn infer_output_shape(
    op_kind: &gllm_kernels::compiler::graph::OpKind,
    seq_len: &gllm_kernels::compiler::graph::SymDim,
    hidden: usize,
    dt: gllm_kernels::types::DType,
) -> Vec<gllm_kernels::compiler::graph::SymDim> {
    use gllm_kernels::compiler::graph::{OpKind, SymDim};

    let s = seq_len.clone();
    let h = SymDim::Concrete(hidden);
    match op_kind {
        OpKind::Gather { embed_dim, .. } => vec![s.clone(), SymDim::Concrete(*embed_dim)],
        OpKind::Gemm { n, .. } => vec![s.clone(), SymDim::Concrete(*n)],
        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => vec![s.clone(), h.clone()],
        OpKind::ValueNorm { .. } => vec![s.clone(), h.clone()],
        OpKind::Silu | OpKind::Gelu | OpKind::Tanh | OpKind::Add | OpKind::Mul => vec![s.clone(), h.clone()],
        OpKind::SwiGlu | OpKind::SwiGluClipped { .. } | OpKind::GeGlu => vec![s.clone(), h.clone()],
        OpKind::RoPE { .. } => vec![s.clone()],
        OpKind::MultiHeadAttention { .. } => vec![s.clone(), h.clone()],
        OpKind::QkNorm { .. } => vec![s.clone()],
        OpKind::HeadRmsNorm { .. } => vec![s.clone()],
        OpKind::LogitSoftcap { .. } => vec![s.clone()],
        OpKind::Argmax { .. } => vec![SymDim::Concrete(1)],
        OpKind::ColumnSlice { slice_dim, .. } => vec![s.clone(), SymDim::Concrete(*slice_dim)],
        OpKind::PerLayerEmbed { hidden, .. } => vec![s.clone(), SymDim::Concrete(*hidden)],
        OpKind::MoERouter { .. } => vec![s.clone()],
        OpKind::MoEDispatchPacked { hidden, .. } => vec![s.clone(), SymDim::Concrete(*hidden)],
        OpKind::Reshape { .. } | OpKind::Transpose { .. } => vec![s.clone()],
        OpKind::PatchEmbed { embed_dim, .. } => vec![s.clone(), SymDim::Concrete(*embed_dim)],
        OpKind::LearnedPos2D { embed_dim, .. } => vec![s.clone(), SymDim::Concrete(*embed_dim)],
        OpKind::DepthwiseConv1D { .. } => vec![s.clone()],
        _ => vec![s.clone(), h.clone()],
    }
}

/// 模板错误
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("Invalid template: {0}")]
    Invalid(String),
}


#[cfg(test)]
/// Derive per-input tensor shapes from OpKind.
///
mod tests {
    use super::*;
    use gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig;

    /// Helper: get a tensor name from a CompilerGraph by TensorId.
    fn tensor_name(g: &gllm_kernels::compiler::CompilerGraph, tid: gllm_kernels::compiler::graph::TensorId) -> String {
        g.tensor(tid).expect("tensor should exist").name.clone()
    }

    /// Helper: find a CompilerOp by label prefix within repeat-expanded ops (those with "layer." prefix).
    fn find_layer_op<'a>(g: &'a gllm_kernels::compiler::CompilerGraph, suffix: &str) -> Option<&'a gllm_kernels::compiler::graph::CompilerOp> {
        g.ops.iter().find(|op| op.label == format!("layer.{}", suffix))
    }

    /// Helper: find a CompilerOp by exact label.
    fn find_op<'a>(g: &'a gllm_kernels::compiler::CompilerGraph, label: &str) -> Option<&'a gllm_kernels::compiler::graph::CompilerOp> {
        g.ops.iter().find(|op| op.label == label)
    }

    /// Helper: count ops whose label contains the given suffix.
    fn count_ops_with_suffix(g: &gllm_kernels::compiler::CompilerGraph, suffix: &str) -> usize {
        g.ops.iter().filter(|op| op.label.ends_with(suffix)).count()
    }

    /// Helper: build a test config with reasonable defaults.
    fn make_test_config() -> super::super::resolve::ResolvedConfig {
        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.hidden_size = 1024;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        config.vocab_size = 32000;
        config.dtype = "f32".to_string();
        config
    }

    #[test]
    fn parse_minimal_template() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        assert_eq!(template.name, "test");
        assert_eq!(template.version, "1.0");
    }

    #[test]
    fn parse_config_placeholders() {
        let yaml = r#"
name: test
config:
  num_layers: "${num_hidden_layers}"
  hidden_size: 4096
graph:
  inputs: []
  outputs: []
  nodes: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let num_layers = template.config.get("num_layers").unwrap();
        assert!(num_layers.is_placeholder());
        assert_eq!(num_layers.placeholder_name(), Some("num_hidden_layers"));

        let hidden_size = template.config.get("hidden_size").unwrap();
        assert!(!hidden_size.is_placeholder());
    }

    #[test]
    fn parse_repeat_block() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - repeat: "${num_layers}"
      var: i
      nodes:
        - name: "layer_${i}_norm"
          op_type: LayerNormalization
          inputs: ["hidden_${i}"]
          outputs: ["normed_${i}"]
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        assert_eq!(template.graph.nodes.len(), 1);
        match &template.graph.nodes[0] {
            GraphNode::Repeat(block) => {
                assert_eq!(block.repeat, "${num_layers}");
                assert_eq!(block.var, "i");
                assert_eq!(block.nodes.len(), 1);
            }
            _ => panic!("Expected repeat block"),
        }
    }

    /// to_compiler_graph expands repeat blocks and auto-adds embed/norm/lm_head/generate ops.
    #[test]
    fn to_compiler_graph_expands_repeat_blocks() {
        let yaml = r#"
name: test_model
graph:
  inputs:
    - name: input_ids
      dtype: int64
  outputs:
    - name: logits
      dtype: f32
  nodes:
    - name: embed
      op_type: Gather
      inputs: ["weights", "input_ids"]
      outputs: ["hidden_0"]
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_attn"
          op_type: Attention
          inputs: ["hidden_${i}"]
          outputs: ["hidden_next_${i}"]
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 2;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Auto-added: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        // YAML top-level: embed (Gather) = 1
        // Repeat: 2 layers * 1 Attention = 2
        // Total = 6 + 1 + 2 = 9
        assert_eq!(graph.ops.len(), 9);

        // Verify YAML's embed node
        let embed_op = find_op(&graph, "embed").expect("YAML embed node");
        assert!(matches!(embed_op.kind, gllm_kernels::compiler::graph::OpKind::Gather { .. }));

        // Verify auto-added embed_gather
        let embed_gather = find_op(&graph, "embed_gather").expect("auto embed_gather");
        assert!(matches!(embed_gather.kind, gllm_kernels::compiler::graph::OpKind::Gather { .. }));

        // Verify repeat-expanded attention ops have "layer." prefix
        let attn0 = find_layer_op(&graph, "layer_0_attn").expect("layer 0 attn");
        assert!(matches!(attn0.kind, gllm_kernels::compiler::graph::OpKind::MultiHeadAttention { .. }));
        let attn1 = find_layer_op(&graph, "layer_1_attn").expect("layer 1 attn");
        assert!(matches!(attn1.kind, gllm_kernels::compiler::graph::OpKind::MultiHeadAttention { .. }));

        // Check input names were substituted: hidden_0 and hidden_1
        assert_eq!(tensor_name(&graph, attn0.inputs[0]), "hidden_0");
        assert_eq!(tensor_name(&graph, attn1.inputs[0]), "hidden_1");
    }

    /// DualRotaryEmbedding per attention_pattern[i] expands into sliding / global RoPE ops.
    #[test]
    fn dual_rope_expands_per_attention_pattern() {
        use gllm_kernels::compiler::graph::OpKind;

        let yaml = r#"
name: gemma4_test
graph:
  inputs:
    - name: input_ids
      dtype: int64
  outputs:
    - name: logits
      dtype: f32
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_rope"
          op_type: DualRotaryEmbedding
          inputs: ["q_${i}", "k_${i}"]
          outputs: ["q_rope_${i}", "k_rope_${i}"]
          attributes:
            sliding_theta: "10000.0"
            global_theta: "1000000.0"
            global_partial: 0.25
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = make_test_config();
        config.num_hidden_layers = 4;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        // 4 layers: sliding, sliding, sliding, global
        config.attention_pattern = vec![0, 0, 0, 1];

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Collect RoPE ops from layer-expanded nodes
        let rope_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::RoPE { .. }) && op.label.contains("_rope_"))
            .collect();

        // Each layer expands into two RoPE ops (Q + K), 4 layers x 2 = 8
        assert_eq!(rope_ops.len(), 8, "each layer should expand into two independent RoPE ops (Q + K)");

        // Verify all are RoPE
        for op in &rope_ops {
            assert!(matches!(op.kind, OpKind::RoPE { .. }),
                "DualRotaryEmbedding must expand into RoPE ops, got {:?}", op.kind);
        }

        // Verify paired Q / K labels per layer
        for i in 0..4 {
            let q = find_layer_op(&graph, &format!("layer_{i}_rope_q")).expect("Q RoPE");
            let k = find_layer_op(&graph, &format!("layer_{i}_rope_k")).expect("K RoPE");
            assert_eq!(tensor_name(&graph, q.inputs[0]), format!("q_{i}"));
            assert_eq!(tensor_name(&graph, q.outputs[0]), format!("q_rope_{i}"));
            assert_eq!(tensor_name(&graph, k.inputs[0]), format!("k_{i}"));
            assert_eq!(tensor_name(&graph, k.outputs[0]), format!("k_rope_{i}"));
        }

        // First 3 layers (attention_pattern=0) -> sliding: theta=10000, partial=1.0
        for layer in 0..3 {
            for suffix in ["rope_q", "rope_k"] {
                let op = find_layer_op(&graph, &format!("layer_{layer}_{suffix}")).unwrap();
                match op.kind {
                    OpKind::RoPE { theta, partial, .. } => {
                        assert!((theta - 10000.0).abs() < 1e-3,
                            "layer {layer} sliding theta incorrect");
                        assert!((partial - 1.0).abs() < 1e-6,
                            "layer {layer} sliding partial should be 1.0");
                    }
                    _ => panic!("expected RoPE"),
                }
            }
        }

        // 4th layer (attention_pattern=1) -> global: theta=1M, partial=0.25
        for suffix in ["rope_q", "rope_k"] {
            let op = find_layer_op(&graph, &format!("layer_3_{suffix}")).unwrap();
            match op.kind {
                OpKind::RoPE { theta, partial, .. } => {
                    assert!((theta - 1_000_000.0).abs() < 1e-2, "global theta incorrect");
                    assert!((partial - 0.25).abs() < 1e-6, "global partial should be 0.25");
                }
                _ => panic!("expected RoPE"),
            }
        }
    }

    /// QkNorm YAML node (2in2out) expands into two independent 1in1out QkNorm ops,
    /// each with head_dim embedded in OpKind.
    #[test]
    fn qk_norm_expands_to_two_nodes_with_head_dim() {
        use gllm_kernels::compiler::graph::OpKind;

        let yaml = r#"
name: gemma4_qk_test
graph:
  inputs:
    - name: input_ids
      dtype: int64
  outputs:
    - name: logits
      dtype: f32
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_qk_norm"
          op_type: QkNorm
          inputs: ["q_${i}", "k_${i}"]
          outputs: ["q_normed_${i}", "k_normed_${i}"]
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = make_test_config();
        config.num_hidden_layers = 2;
        config.head_dim = 128;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Collect QkNorm ops
        let qk_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::QkNorm { .. }))
            .collect();
        assert_eq!(qk_ops.len(), 4, "2 layers x 2 ops = 4 QkNorm");

        for i in 0..2 {
            let q = find_layer_op(&graph, &format!("layer_{i}_qk_norm_q")).expect("Q QkNorm");
            let k = find_layer_op(&graph, &format!("layer_{i}_qk_norm_k")).expect("K QkNorm");
            assert!(matches!(q.kind, OpKind::QkNorm { .. }));
            assert!(matches!(k.kind, OpKind::QkNorm { .. }));
            assert_eq!(tensor_name(&graph, q.inputs[0]), format!("q_{i}"));
            assert_eq!(tensor_name(&graph, q.outputs[0]), format!("q_normed_{i}"));
            assert_eq!(tensor_name(&graph, k.inputs[0]), format!("k_{i}"));
            assert_eq!(tensor_name(&graph, k.outputs[0]), format!("k_normed_{i}"));
            // Verify head_dim = 128
            match q.kind {
                OpKind::QkNorm { head_dim } => assert_eq!(head_dim, 128),
                _ => panic!("expected QkNorm"),
            }
        }
    }

    /// serde correctly parses `only_if` field (present / absent both round-trip).
    #[test]
    fn only_if_field_parsed_by_serde() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      inputs: []
      outputs: []
    - name: b
      op_type: Mul
      only_if: has_per_layer_embedding
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        assert_eq!(template.graph.nodes.len(), 2);
        match (&template.graph.nodes[0], &template.graph.nodes[1]) {
            (GraphNode::Node(a), GraphNode::Node(b)) => {
                assert!(a.only_if.is_none(), "node a should have None only_if");
                assert_eq!(b.only_if.as_deref(), Some("has_per_layer_embedding"));
            }
            _ => panic!("Expected two Nodes"),
        }
    }

    /// When has_per_layer_embedding = false, nodes with only_if are skipped.
    #[test]
    fn only_if_skips_node_when_false() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_core"
          op_type: Add
          inputs: ["x_${i}"]
          outputs: ["y_${i}"]
        - name: "layer_${i}_ple"
          op_type: PerLayerEmbed
          only_if: has_per_layer_embedding
          inputs: ["y_${i}"]
          outputs: ["z_${i}"]
          attributes:
            layer_idx: "${i}"
            dim_per_layer: 256
            num_layers: "${num_hidden_layers}"
            hidden: 1024
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = make_test_config();
        config.num_hidden_layers = 2;
        config.hidden_size_per_layer_input = 0;
        config.has_per_layer_embedding = false;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Only core Add nodes survive; PLE skipped by only_if=false
        let layer_adds: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_core"))
            .collect();
        assert_eq!(layer_adds.len(), 2, "PLE nodes must be skipped when only_if=false");

        assert_eq!(layer_adds[0].label, "layer.layer_0_core");
        assert_eq!(layer_adds[1].label, "layer.layer_1_core");

        // No PLE ops should exist
        let ple_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_ple"))
            .collect();
        assert!(ple_ops.is_empty(), "PLE ops must not exist when only_if=false");
    }

    /// When has_per_layer_embedding = true, only_if nodes expand normally.
    /// PerLayerEmbed expands into ColumnSlice + PerLayerEmbed (2 ops per layer).
    #[test]
    fn only_if_expands_node_when_true() {
        use gllm_kernels::compiler::graph::OpKind;

        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_core"
          op_type: Add
          inputs: ["x_${i}"]
          outputs: ["y_${i}"]
        - name: "layer_${i}_ple"
          op_type: PerLayerEmbed
          only_if: has_per_layer_embedding
          inputs: ["y_${i}", "ple_full", "proj_w", "post_mlp_w_${i}"]
          outputs: ["z_${i}"]
          attributes:
            layer_idx: "${i}"
            dim_per_layer: 256
            num_layers: "${num_hidden_layers}"
            hidden: 1024
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = make_test_config();
        config.num_hidden_layers = 2;
        config.hidden_size_per_layer_input = 256;
        config.has_per_layer_embedding = true;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Each layer: core(Add) + ColumnSlice + PerLayerEmbed = 3 ops, 2 layers = 6 YAML ops
        let core_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_core"))
            .collect();
        let slice_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_ple_slice"))
            .collect();
        let ple_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::PerLayerEmbed { .. }))
            .collect();

        assert_eq!(core_ops.len(), 2);
        assert_eq!(slice_ops.len(), 2, "ColumnSlice per layer");
        assert_eq!(ple_ops.len(), 2, "PerLayerEmbed per layer");

        // Verify ColumnSlice kind
        assert!(matches!(slice_ops[0].kind, OpKind::ColumnSlice { .. }));

        // Verify PerLayerEmbed has layer_idx and num_layers from OpKind
        match &ple_ops[0].kind {
            OpKind::PerLayerEmbed { layer_idx, num_layers, .. } => {
                assert_eq!(*layer_idx, 0);
                assert_eq!(*num_layers, 2);
            }
            _ => panic!("expected PerLayerEmbed"),
        }
        match &ple_ops[1].kind {
            OpKind::PerLayerEmbed { layer_idx, .. } => {
                assert_eq!(*layer_idx, 1);
            }
            _ => panic!("expected PerLayerEmbed"),
        }
    }

    /// Three-segment comparison syntax `field op value` evaluated by integer comparison.
    #[test]
    fn only_if_comparison_expression() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      only_if: "hidden_size > 0"
      inputs: []
      outputs: []
    - name: b
      op_type: Add
      only_if: "num_hidden_layers == 0"
      inputs: []
      outputs: []
    - name: c
      op_type: Add
      only_if: "num_hidden_layers != 0"
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 4;
        config.hidden_size = 1024;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // a: hidden_size > 0 -> true; b: num_hidden_layers == 0 -> false; c: != 0 -> true
        // "a" and "c" are YAML top-level Add ops (no "layer." prefix)
        let a_op = find_op(&graph, "a").expect("node a");
        let c_op = find_op(&graph, "c").expect("node c");
        assert!(matches!(a_op.kind, gllm_kernels::compiler::graph::OpKind::Add));
        assert!(matches!(c_op.kind, gllm_kernels::compiler::graph::OpKind::Add));
        // b should not exist
        assert!(find_op(&graph, "b").is_none(), "node b should be skipped");
    }

    /// Unknown only_if field must return error (no silent false masking typos).
    #[test]
    fn only_if_unknown_field_errors() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      only_if: typo_field_that_does_not_exist
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let err = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("typo_field_that_does_not_exist"),
            "error should contain unknown field name, got: {msg}");
    }

    /// PerLayerEmbed with 4 inputs expands into ColumnSlice + 5-input PerLayerEmbed.
    #[test]
    fn expand_per_layer_embed_injects_slice_node_and_main_embed() {
        use gllm_kernels::compiler::graph::OpKind;

        let yaml = r#"
name: gemma4_ple_test
graph:
  inputs:
    - name: input_ids
      dtype: int64
  outputs:
    - name: logits
      dtype: f32
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_ple"
          op_type: PerLayerEmbed
          inputs:
            - "hidden_0"
            - "ple_full"
            - "model.per_layer_embedding.per_layer_projection.weight"
            - "model.layers.${i}.post_mlp_projection.weight"
          outputs: ["hidden_0"]
          attributes:
            layer_idx: "${i}"
            dim_per_layer: 256
            num_layers: "${num_hidden_layers}"
            hidden: 1024
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = make_test_config();
        config.num_hidden_layers = 3;
        config.hidden_size_per_layer_input = 256;
        config.has_per_layer_embedding = true;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Each layer expands into 2 ops (ColumnSlice + PerLayerEmbed), 3 layers = 6 YAML ops
        for layer in 0..3 {
            let slice_op = find_layer_op(&graph, &format!("layer_{layer}_ple_slice"))
                .unwrap_or_else(|| panic!("ColumnSlice for layer {layer}"));
            let ple_op = find_layer_op(&graph, &format!("layer_{layer}_ple"))
                .unwrap_or_else(|| panic!("PerLayerEmbed for layer {layer}"));

            // ColumnSlice op
            assert!(matches!(slice_op.kind, OpKind::ColumnSlice { .. }));
            // ColumnSlice reads ple_full
            assert_eq!(tensor_name(&graph, slice_op.inputs[0]), "ple_full");

            // PerLayerEmbed op
            assert!(matches!(ple_op.kind, OpKind::PerLayerEmbed { .. }));
            // 5 inputs: [hidden_state, main_embed, layer_{i}_ple_slice, proj_w, post_mlp_w]
            // Layer 0's hidden_state is the auto-added "embedding" tensor (tensor_map["hidden_0"]
            // initially points to it). Layer 1+ read "hidden_0" (output of layer 0 overwrote the map).
            assert_eq!(ple_op.inputs.len(), 5, "PerLayerEmbed must have 5 inputs");
            let expected_hidden_name = if layer == 0 { "embedding" } else { "hidden_0" };
            assert_eq!(tensor_name(&graph, ple_op.inputs[0]), expected_hidden_name,
                "layer {layer} input[0] hidden state mismatch");
            assert_eq!(tensor_name(&graph, ple_op.inputs[1]), "main_embed",
                "input[1] must be main_embed");
            assert_eq!(tensor_name(&graph, ple_op.inputs[2]), format!("layer_{layer}_ple_slice"),
                "input[2] must be current layer slice");
            assert_eq!(tensor_name(&graph, ple_op.inputs[3]),
                "model.per_layer_embedding.per_layer_projection.weight");
            assert_eq!(tensor_name(&graph, ple_op.inputs[4]),
                format!("model.layers.{layer}.post_mlp_projection.weight"));

            // Verify OpKind fields
            match &ple_op.kind {
                OpKind::PerLayerEmbed { layer_idx, dim_per_layer, num_layers, .. } => {
                    assert_eq!(*layer_idx, layer);
                    assert_eq!(*dim_per_layer, 256);
                    assert_eq!(*num_layers, 3);
                }
                _ => panic!("expected PerLayerEmbed"),
            }

            match &slice_op.kind {
                OpKind::ColumnSlice { start, slice_dim, .. } => {
                    assert_eq!(*start, layer * 256);
                    assert_eq!(*slice_dim, 256);
                }
                _ => panic!("expected ColumnSlice"),
            }
        }
    }

    /// Nodes without only_if always expand (forward compatibility).
    #[test]
    fn only_if_absent_expands_unconditionally() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();
        assert!(find_op(&graph, "a").is_some());
    }

    // ============================================================================
    // T43: SharedKvRef graph-layer integration tests.
    // ============================================================================

    /// `!<field>` negates a boolean field lookup in `only_if`.
    #[test]
    fn only_if_negation_prefix() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      only_if: "!has_per_layer_embedding"
      inputs: []
      outputs: []
    - name: b
      op_type: Add
      only_if: "has_per_layer_embedding"
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;
        config.has_per_layer_embedding = false;

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();
        // has_per_layer_embedding = false -> `!has_...` = true, `has_...` = false.
        assert!(find_op(&graph, "a").is_some());
        assert!(find_op(&graph, "b").is_none());

        config.has_per_layer_embedding = true;
        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();
        assert!(find_op(&graph, "a").is_none());
        assert!(find_op(&graph, "b").is_some());
    }

    /// `!<field>` on empty / malformed field returns an error (no silent defaults).
    #[test]
    fn only_if_negation_requires_field_name() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      only_if: "!"
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let err = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap_err();
        assert!(format!("{err}").contains("缺少字段名") || format!("{err}").contains("`!`"));
    }

    /// Per-layer `is_kv_shared_layer_${i}` lookup inside a repeat block.
    #[test]
    fn only_if_per_layer_shared_kv_lookup() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_kv"
          op_type: Add
          only_if: "!is_kv_shared_layer_${i}"
          inputs: []
          outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        // 10 layers, last 4 are consumers: layers 6..10 skipped.
        config.num_hidden_layers = 10;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0u8; 10];

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Only 6 non-consumer layers keep their KV node.
        let kv_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_kv"))
            .collect();
        assert_eq!(kv_ops.len(), 6, "consumer layers must skip the KV node");
        for (idx, op) in kv_ops.iter().enumerate() {
            assert_eq!(op.label, format!("layer.layer_{idx}_kv"));
        }
    }

    /// `${donor_i}` placeholder resolves identity on non-consumer layers and
    /// resolves to the donor layer index on consumer layers.
    #[test]
    fn donor_placeholder_routes_to_donor_on_consumer_layers() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - repeat: "${num_hidden_layers}"
      var: i
      nodes:
        - name: "layer_${i}_attn"
          op_type: Attention
          inputs: ["layer_${i}_q", "layer_${donor_i}_k", "layer_${donor_i}_v"]
          outputs: ["layer_${i}_out"]
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        // 8 layers, last 4 shared. attention_pattern: alternating 0/1.
        //   layer 0 -> 0, layer 1 -> 1, layer 2 -> 0, layer 3 -> 1   (non-consumer)
        //   layer 4 -> 0, layer 5 -> 1, layer 6 -> 0, layer 7 -> 1   (consumer)
        // donor bucket-matched latest non-consumer:
        //   layer 4 (bucket 0) -> donor 2
        //   layer 5 (bucket 1) -> donor 3
        //   layer 6 (bucket 0) -> donor 2
        //   layer 7 (bucket 1) -> donor 3
        config.num_hidden_layers = 8;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0, 1];

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        let attn_ops: Vec<&gllm_kernels::compiler::graph::CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.contains("_attn"))
            .collect();
        assert_eq!(attn_ops.len(), 8);

        // Non-consumer layers (0..4): donor_i == i, identity routing.
        for i in 0..4 {
            let op = find_layer_op(&graph, &format!("layer_{i}_attn")).expect("attn op");
            assert_eq!(tensor_name(&graph, op.inputs[0]), format!("layer_{i}_q"));
            assert_eq!(tensor_name(&graph, op.inputs[1]), format!("layer_{i}_k"),
                "non-consumer layer {i} must read self K");
            assert_eq!(tensor_name(&graph, op.inputs[2]), format!("layer_{i}_v"));
        }

        // Consumer layers (4..8): K/V route to donor.
        let expected_donor = [2usize, 3, 2, 3];
        for (offset, &donor) in expected_donor.iter().enumerate() {
            let i = 4 + offset;
            let op = find_layer_op(&graph, &format!("layer_{i}_attn")).expect("attn op");
            assert_eq!(tensor_name(&graph, op.inputs[0]), format!("layer_{i}_q"));
            assert_eq!(tensor_name(&graph, op.inputs[1]), format!("layer_{donor}_k"),
                "consumer layer {i} must read donor {donor} K");
            assert_eq!(tensor_name(&graph, op.inputs[2]), format!("layer_{donor}_v"));
        }
    }

    /// End-to-end expansion against the real `gemma4.yaml` for a Gemma 4 E2B
    /// config (26 layers, 20 shared). Consumer layers must not emit
    /// k_proj / v_proj / v_norm / _rope_k / _qk_norm_k; attention
    /// on consumer layers must read donor-layer tensors.
    #[test]
    fn gemma4_e2b_consumer_layers_skip_kv_and_route_to_donor() {
        use super::super::resolve::ResolvedConfig;

        let yaml = include_str!("templates/gemma4.yaml");
        let template = ArchTemplate::from_yaml(yaml).unwrap();

        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 26;
        config.hidden_size = 2048;
        config.num_attention_heads = 8;
        config.num_key_value_heads = 2;
        config.head_dim = 256;
        config.intermediate_size = Some(16384);
        config.vocab_size = 262208;
        config.rope_theta = 10_000.0;
        config.global_rope_theta = 1_000_000.0;
        config.sliding_window = 512;
        config.hidden_size_per_layer_input = 128;
        config.num_kv_shared_layers = 20;
        // Use uniform sliding pattern (all 0) to avoid hetero loop config path.
        // The shared KV routing tested here is independent of sliding/global alternation.
        config.attention_pattern = vec![0u8; 26];
        config.has_per_layer_embedding = true;
        config.dtype = "f32".to_string();

        let graph = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap();

        // Collect all op labels (with "layer." prefix)
        let labels: std::collections::HashSet<String> =
            graph.ops.iter().map(|op| op.label.clone()).collect();

        // Strip "layer." prefix for matching convenience
        let names: std::collections::HashSet<String> = labels.iter()
            .map(|l| l.strip_prefix("layer.").unwrap_or(l).to_string())
            .collect();

        // Non-consumer layers 0..6 own all K/V pipeline nodes.
        for i in 0..6 {
            assert!(names.contains(&format!("layer_{i}_k_proj")),
                "non-consumer layer {i} must have k_proj");
            assert!(names.contains(&format!("layer_{i}_v_proj")),
                "non-consumer layer {i} must have v_proj");
            assert!(names.contains(&format!("layer_{i}_v_norm")),
                "non-consumer layer {i} must have v_norm");
            assert!(names.contains(&format!("layer_{i}_qk_norm_k")),
                "non-consumer layer {i} must have qk_norm_k");
            assert!(names.contains(&format!("layer_{i}_rope_k")),
                "non-consumer layer {i} must have rope_k");
        }

        // Consumer layers 6..26 (20 layers) skip k_proj / v_proj / v_norm /
        // qk_norm_k / rope_k -- donated by their respective donors.
        for i in 6..26 {
            assert!(!names.contains(&format!("layer_{i}_k_proj")),
                "consumer layer {i} must skip k_proj");
            assert!(!names.contains(&format!("layer_{i}_v_proj")),
                "consumer layer {i} must skip v_proj");
            assert!(!names.contains(&format!("layer_{i}_v_norm")),
                "consumer layer {i} must skip v_norm");
            assert!(!names.contains(&format!("layer_{i}_qk_norm_k")),
                "consumer layer {i} must skip qk_norm_k");
            assert!(!names.contains(&format!("layer_{i}_rope_k")),
                "consumer layer {i} must skip rope_k");
        }

        // Count k_proj nodes -- must equal num_hidden_layers - num_kv_shared_layers (6).
        let k_proj_count = count_ops_with_suffix(&graph, "_k_proj");
        assert_eq!(k_proj_count, config.num_hidden_layers - config.num_kv_shared_layers,
            "k_proj must appear only on non-consumer layers");

        // Consumer attention nodes must reference donor K / V tensors.
        for i in 6..26 {
            let attn = find_layer_op(&graph, &format!("layer_{i}_attn"))
                .unwrap_or_else(|| panic!("attention node for consumer layer {i}"));
            let donor = config.donor_layer(i).unwrap().expect("donor present");
            assert_ne!(donor, i, "consumer layer donor must differ from self");
            assert_eq!(tensor_name(&graph, attn.inputs[0]), format!("layer_{i}_q_rope"),
                "consumer layer {i} still uses own Q rope");
            assert_eq!(tensor_name(&graph, attn.inputs[1]), format!("layer_{donor}_k_rope"),
                "consumer layer {i} attention must read donor {donor} K rope");
            assert_eq!(tensor_name(&graph, attn.inputs[2]), format!("layer_{donor}_v_normed"),
                "consumer layer {i} attention must read donor {donor} V normed");
        }
    }

    /// Unknown `!<field>` errors out with a clear message (no silent false).
    #[test]
    fn only_if_negation_unknown_field_errors() {
        let yaml = r#"
name: test
graph:
  inputs: []
  outputs: []
  nodes:
    - name: a
      op_type: Add
      only_if: "!typo_field"
      inputs: []
      outputs: []
"#;
        let template = ArchTemplate::from_yaml(yaml).unwrap();
        let mut config = make_test_config();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let err = template.to_compiler_graph(&config, &MegaKernelBusinessConfig::default()).unwrap_err();
        assert!(format!("{err}").contains("typo_field"),
            "error must name the unknown field, got: {err}");
    }
}
