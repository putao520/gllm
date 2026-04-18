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
    /// 节点列表（包含普通节点和重复块）
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
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

    /// 将架构模板转换为 OnnxGraph (REQ-EXEC-001)
    ///
    /// 使用已解析的配置替换占位符，展开重复块，生成可执行的图。
    pub fn to_onnx_graph(
        &self,
        config: &super::resolve::ResolvedConfig,
    ) -> Result<crate::loader::onnx::OnnxGraph, TemplateError> {
        use crate::loader::onnx::{OnnxGraph, OnnxValueInfo};
        use std::collections::HashMap;

        // 1. 构建输入
        let inputs: Vec<OnnxValueInfo> = self
            .graph
            .inputs
            .iter()
            .map(|def| OnnxValueInfo {
                name: super::resolve::substitute_placeholders(&def.name, config),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            })
            .collect();

        // 2. 构建输出
        let outputs: Vec<OnnxValueInfo> = self
            .graph
            .outputs
            .iter()
            .map(|def| OnnxValueInfo {
                name: super::resolve::substitute_placeholders(&def.name, config),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            })
            .collect();

        // 3. 展开节点（处理重复块）
        let mut nodes = Vec::new();
        for graph_node in &self.graph.nodes {
            match graph_node {
                GraphNode::Node(node_def) => {
                    if !Self::eval_only_if(node_def.only_if.as_deref(), config)? {
                        continue;
                    }
                    nodes.push(self.node_def_to_onnx(node_def, config, None)?);
                }
                GraphNode::Repeat(repeat_block) => {
                    let repeat_count = self.resolve_repeat_count(&repeat_block.repeat, config)?;
                    for i in 0..repeat_count {
                        for node_def in &repeat_block.nodes {
                            if !Self::eval_only_if(node_def.only_if.as_deref(), config)? {
                                continue;
                            }
                            // DualRotaryEmbedding 是 per-layer 双轨 RoPE 的逻辑节点,按
                            // `config.attention_pattern[i]` 在模板展开时解析为标准
                            // RotaryEmbedding + 对应 (theta, partial)。
                            //
                            //   attention_pattern[i] == 0 → sliding: theta=sliding_theta, partial=1.0
                            //   attention_pattern[i] == 1 → global : theta=global_theta, partial=global_partial
                            //
                            // 展开为一个 RotaryEmbedding 节点,其 attributes 由 sliding/global
                            // 两组属性中选一组注入。这保持"YAML 为 SSOT + 模板引擎做语义展开"
                            // 原则,不在 Rust 源码里硬编码层映射。
                            if node_def.op_type == "DualRotaryEmbedding" {
                                let expanded = self.expand_dual_rope(node_def, config, &repeat_block.var, i)?;
                                nodes.extend(expanded);
                            } else {
                                nodes.push(self.node_def_to_onnx(
                                    node_def,
                                    config,
                                    Some((&repeat_block.var, i)),
                                )?);
                            }
                        }
                    }
                }
            }
        }

        Ok(OnnxGraph {
            name: self.name.clone(),
            doc_string: format!("Generated from template {} v{}", self.name, self.version),
            nodes,
            inputs,
            outputs,
            value_info: Vec::new(),
            initializers: HashMap::new(),
            sparse_initializers: Vec::new(),
            quantization_annotation: Vec::new(),
            metadata_props: HashMap::new(),
        })
    }

    /// 将 NodeDef 转换为 OnnxNode
    fn node_def_to_onnx(
        &self,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        loop_var: Option<(&str, usize)>,
    ) -> Result<crate::loader::onnx::OnnxNode, TemplateError> {
        use crate::loader::onnx::{OnnxAttribute, OnnxAttributeValue, OnnxNode};
        use std::collections::HashMap;

        let substitute = |s: &str| -> String {
            let mut result = super::resolve::substitute_placeholders(s, config);
            if let Some((var, idx)) = loop_var {
                result = result.replace(&format!("${{{}}}", var), &idx.to_string());
                result = result.replace(&format!("${}$", var), &idx.to_string());
            }
            result
        };

        let inputs: Vec<String> = node_def.inputs.iter().map(|s| substitute(s)).collect();
        let outputs: Vec<String> = node_def.outputs.iter().map(|s| substitute(s)).collect();

        let mut attributes = HashMap::new();
        for (key, value) in &node_def.attributes {
            let attr_value = match value {
                AttributeValue::Int(v) => OnnxAttributeValue::Int(*v),
                AttributeValue::Float(v) => OnnxAttributeValue::Float(*v as f32),
                AttributeValue::String(s) => {
                    // YAML 的 `attr: "${i}"` / `attr: "${num_hidden_layers}"` 经占位符替换
                    // 后得到纯数字字符串(如 "3" / "32"),下游 `atomic_op_to_kind::require_usize`
                    // 要求 `AttrValue::Int`,如果保留为 String 会导致 "缺少必需属性" 错误。
                    //
                    // 规则: 替换后的字符串若能解析为 i64,则以 Int 发射; 否则保留为 String。
                    // 这样既保持 YAML 语法简洁性(不需要额外类型标签),又让属性在下游以
                    // 正确类型传递。对真正需要字符串语义的属性(如 dtype="f32")无影响。
                    let substituted = substitute(s);
                    if let Ok(n) = substituted.parse::<i64>() {
                        OnnxAttributeValue::Int(n)
                    } else {
                        OnnxAttributeValue::String(substituted)
                    }
                }
                AttributeValue::Ints(v) => OnnxAttributeValue::Ints(v.clone()),
                AttributeValue::Floats(v) => {
                    OnnxAttributeValue::Floats(v.iter().map(|f| *f as f32).collect())
                }
            };
            let attr = OnnxAttribute {
                name: key.clone(),
                value: attr_value,
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            };
            attributes.insert(key.clone(), attr);
        }

        Ok(OnnxNode {
            name: substitute(&node_def.name),
            op_type: node_def.op_type.clone(),
            domain: String::new(),
            inputs,
            outputs,
            attributes,
        })
    }

    /// 按 `config.attention_pattern[layer_idx]` 把 DualRotaryEmbedding 展开为
    /// **两个独立的** RotaryEmbedding 节点 (分别作用于 Q / K),并把 sliding/global
    /// 其中一组属性物化到 theta/partial。
    ///
    /// 拆分成两个节点是下游 FusedQkvNormRope pattern_fusion 的要求 —
    /// gllm-kernels 的 `FusionMode::FusedQkvNormRope` 显式要求 `rope_q` 和 `rope_k`
    /// 为两个独立的 OpId,因此图层必须结构化地暴露 Q-RoPE 与 K-RoPE 两个节点。
    ///
    /// 输入 DualRotaryEmbedding 节点形态 (由 gemma4.yaml 提供):
    ///   inputs  = [q_normed, k_normed]
    ///   outputs = [q_rope,   k_rope]
    ///
    /// 展开后 (节点名在原名基础上加 `_q` / `_k` 后缀):
    ///   RotaryEmbedding("{name}_q"): inputs=[q_normed], outputs=[q_rope]
    ///   RotaryEmbedding("{name}_k"): inputs=[k_normed], outputs=[k_rope]
    ///
    /// 输入 YAML attributes 约定:
    /// - `sliding_theta`, `sliding_partial` (缺省 1.0)
    /// - `global_theta`, `global_partial` (缺省 0.25)
    ///
    /// 若 `config.attention_pattern` 为空或 layer_idx 越界,默认为 sliding 语义。
    fn expand_dual_rope(
        &self,
        node_def: &NodeDef,
        config: &super::resolve::ResolvedConfig,
        var: &str,
        layer_idx: usize,
    ) -> Result<Vec<crate::loader::onnx::OnnxNode>, TemplateError> {
        use crate::loader::onnx::{OnnxAttribute, OnnxAttributeValue, OnnxNode};

        let substitute = |s: &str| -> String {
            let mut result = super::resolve::substitute_placeholders(s, config);
            result = result.replace(&format!("${{{}}}", var), &layer_idx.to_string());
            result = result.replace(&format!("${}$", var), &layer_idx.to_string());
            result
        };

        // 读原始 attributes 的两组 (sliding / global) 参数,与 node_def_to_onnx 一致地
        // 做占位符替换。AttributeValue::String 的 sliding_theta=${rope_theta} 会被解析。
        let read_f32 = |key: &str| -> Option<f32> {
            match node_def.attributes.get(key)? {
                AttributeValue::Float(v) => Some(*v as f32),
                AttributeValue::Int(v) => Some(*v as f32),
                AttributeValue::String(s) => {
                    let resolved = super::resolve::substitute_placeholders(s, config);
                    resolved.parse::<f32>().ok()
                }
                _ => None,
            }
        };

        let sliding_theta = read_f32("sliding_theta").ok_or_else(|| TemplateError::Invalid(
            format!("DualRotaryEmbedding '{}' 缺少 sliding_theta 属性", node_def.name)))?;
        let sliding_partial = read_f32("sliding_partial").unwrap_or(1.0);
        let global_theta = read_f32("global_theta").ok_or_else(|| TemplateError::Invalid(
            format!("DualRotaryEmbedding '{}' 缺少 global_theta 属性", node_def.name)))?;
        let global_partial = read_f32("global_partial").unwrap_or(0.25);

        // 选择层变体: 0=sliding, 1=global。attention_pattern 缺失时按 sliding 处理。
        let is_global = config.attention_pattern.get(layer_idx).copied().unwrap_or(0) == 1;
        let (theta, partial) = if is_global {
            (global_theta, global_partial)
        } else {
            (sliding_theta, sliding_partial)
        };

        // num_heads / head_dim: RoPE 标量参数,从模型几何推导
        let num_heads = config.get_int("num_attention_heads").ok_or_else(|| TemplateError::Invalid(
            "DualRotaryEmbedding 展开时配置缺少 num_attention_heads".into()))? as i64;
        let head_dim = config.get_int("head_dim").ok_or_else(|| TemplateError::Invalid(
            "DualRotaryEmbedding 展开时配置缺少 head_dim".into()))? as i64;

        let mk_attrs = || -> HashMap<String, OnnxAttribute> {
            let mut attributes = HashMap::new();
            let mk_attr = |name: &str, value: OnnxAttributeValue| OnnxAttribute {
                name: name.to_string(), value, doc_string: String::new(),
                ref_attr_name: None, attr_type: None,
            };
            attributes.insert("num_heads".into(), mk_attr("num_heads", OnnxAttributeValue::Int(num_heads)));
            attributes.insert("head_dim".into(),  mk_attr("head_dim",  OnnxAttributeValue::Int(head_dim)));
            attributes.insert("theta".into(),     mk_attr("theta",     OnnxAttributeValue::Float(theta)));
            attributes.insert("partial".into(),   mk_attr("partial",   OnnxAttributeValue::Float(partial)));
            attributes
        };

        // DualRotaryEmbedding 必须形态: inputs=[q, k], outputs=[q_rope, k_rope]
        if node_def.inputs.len() != 2 || node_def.outputs.len() != 2 {
            return Err(TemplateError::Invalid(format!(
                "DualRotaryEmbedding '{}' 必须有 2 个输入 (q, k) 和 2 个输出 (q_rope, k_rope), 实际 inputs={} outputs={}",
                node_def.name, node_def.inputs.len(), node_def.outputs.len(),
            )));
        }

        let base_name = substitute(&node_def.name);
        let q_in = substitute(&node_def.inputs[0]);
        let k_in = substitute(&node_def.inputs[1]);
        let q_out = substitute(&node_def.outputs[0]);
        let k_out = substitute(&node_def.outputs[1]);

        let q_node = OnnxNode {
            name: format!("{}_q", base_name),
            op_type: "RotaryEmbedding".into(),
            domain: String::new(),
            inputs: vec![q_in],
            outputs: vec![q_out],
            attributes: mk_attrs(),
        };
        let k_node = OnnxNode {
            name: format!("{}_k", base_name),
            op_type: "RotaryEmbedding".into(),
            domain: String::new(),
            inputs: vec![k_in],
            outputs: vec![k_out],
            attributes: mk_attrs(),
        };

        Ok(vec![q_node, k_node])
    }

    /// 求值 `only_if` 条件表达式 (节点级展开守卫)。
    ///
    /// `None` / 空串 → `true` (无条件展开); 表达式求值为 `false` → 调用方跳过该节点。
    ///
    /// 支持两种语法:
    /// 1. **字段名查表** (`has_per_layer_embedding`): 调用 `ResolvedConfig::get_bool`。
    ///    未知字段 → `TemplateError::Invalid` (不做静默 false 掩盖拼写错误)。
    /// 2. **整数比较** (`lhs op rhs`, 按 `split_whitespace` 拆 3 段): `lhs` 从
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

        // 形式 2: 三段式比较 `lhs op rhs`
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

        // 形式 1: 单字段名查 bool
        if tokens.len() == 1 {
            let key = tokens[0];
            if let Some(v) = config.get_bool(key) {
                return Ok(v);
            }
            return Err(TemplateError::Invalid(format!(
                "only_if '{raw}': 未知布尔字段 (ResolvedConfig::get_bool 无匹配)。\
                 若需派生字段,请在 ResolvedConfig::get_bool 中注册,禁止静默返回 false。"
            )));
        }

        Err(TemplateError::Invalid(format!(
            "only_if '{raw}': 语法错误,期望 `<field>` 或 `<lhs> <op> <rhs>` (3 个空白分隔 token)"
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
mod tests {
    use super::*;

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

    #[test]
    fn to_onnx_graph_expands_repeat_blocks() {
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
          outputs: ["attn_${i}"]
        - name: "layer_${i}_ffn"
          op_type: FFN
          inputs: ["attn_${i}"]
          outputs: ["hidden_${ next }"]
          attributes:
            layer_idx: ${i}
"#;
        // Note: The template uses ${i} which won't parse as attribute, so simplify
        let yaml_simple = r#"
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
        let template = ArchTemplate::from_yaml(yaml_simple).unwrap();

        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.hidden_size = 768;
        config.vocab_size = 50000;

        let graph = template.to_onnx_graph(&config).unwrap();

        // embed + 2 layers * 1 node = 3 nodes
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].name, "embed");
        assert_eq!(graph.nodes[1].name, "layer_0_attn");
        assert_eq!(graph.nodes[2].name, "layer_1_attn");

        // Check inputs were substituted
        assert_eq!(graph.nodes[1].inputs, vec!["hidden_0".to_string()]);
        assert_eq!(graph.nodes[2].inputs, vec!["hidden_1".to_string()]);
    }

    /// DualRotaryEmbedding 按 attention_pattern[i] 展开为 sliding / global
    /// 对应参数的 RotaryEmbedding 节点。
    #[test]
    fn dual_rope_expands_per_attention_pattern() {
        use crate::loader::onnx::OnnxAttributeValue;

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

        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        // 4 层: sliding, sliding, sliding, global
        config.attention_pattern = vec![0, 0, 0, 1];

        let graph = template.to_onnx_graph(&config).unwrap();
        // 每层展开为两个 RoPE 节点 (Q / K), 4 层 × 2 = 8 个节点
        assert_eq!(graph.nodes.len(), 8, "每层应展开为两个独立的 RoPE 节点 (Q + K)");

        // 每个节点 op_type 都应是 RotaryEmbedding (被 expand 改写)
        for n in &graph.nodes {
            assert_eq!(n.op_type, "RotaryEmbedding",
                "DualRotaryEmbedding 必须展开为 RotaryEmbedding,实际 op_type={}", n.op_type);
        }

        // 成对验证每层的 Q / K 节点 (节点名 `layer_{i}_rope_q` / `layer_{i}_rope_k`)
        for i in 0..4 {
            let q = &graph.nodes[i * 2];
            let k = &graph.nodes[i * 2 + 1];
            assert_eq!(q.name, format!("layer_{i}_rope_q"));
            assert_eq!(k.name, format!("layer_{i}_rope_k"));
            assert_eq!(q.inputs, vec![format!("q_{i}")]);
            assert_eq!(q.outputs, vec![format!("q_rope_{i}")]);
            assert_eq!(k.inputs, vec![format!("k_{i}")]);
            assert_eq!(k.outputs, vec![format!("k_rope_{i}")]);
        }

        let read_f32 = |attr_map: &std::collections::HashMap<String, crate::loader::onnx::OnnxAttribute>, key: &str| {
            match attr_map.get(key).map(|a| &a.value) {
                Some(OnnxAttributeValue::Float(v)) => *v,
                other => panic!("属性 {key} 缺失或类型错误: {:?}", other),
            }
        };

        // 前三层 (attention_pattern=0) → sliding; Q / K 节点 attributes 一致
        for layer in 0..3 {
            for node_idx in [layer * 2, layer * 2 + 1] {
                assert!((read_f32(&graph.nodes[node_idx].attributes, "theta") - 10000.0).abs() < 1e-3,
                    "layer {layer} node {node_idx} sliding theta 不正确");
                assert!((read_f32(&graph.nodes[node_idx].attributes, "partial") - 1.0).abs() < 1e-6,
                    "layer {layer} node {node_idx} sliding partial 应为 1.0");
            }
        }
        // 第 4 层 (attention_pattern=1) → global
        for node_idx in [6, 7] {
            assert!((read_f32(&graph.nodes[node_idx].attributes, "theta") - 1_000_000.0).abs() < 1e-2);
            assert!((read_f32(&graph.nodes[node_idx].attributes, "partial") - 0.25).abs() < 1e-6);
        }
    }

    /// serde 正确解析 `only_if` 字段 (存在 / 缺省都能 round-trip)。
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
                assert!(a.only_if.is_none(), "节点 a 缺省 only_if 应为 None");
                assert_eq!(b.only_if.as_deref(), Some("has_per_layer_embedding"));
            }
            _ => panic!("期望两个 Node"),
        }
    }

    /// `has_per_layer_embedding = false` 时带 only_if 的节点被跳过。
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

        // has_per_layer_embedding 必须基于 hidden_size_per_layer_input > 0 推导
        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.hidden_size = 1024;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        config.vocab_size = 32000;
        config.hidden_size_per_layer_input = 0;
        config.has_per_layer_embedding = false;

        let graph = template.to_onnx_graph(&config).unwrap();
        // 只有核心 Add 节点, PLE 被跳过
        assert_eq!(graph.nodes.len(), 2, "PLE 节点必须在 only_if=false 时跳过");
        assert_eq!(graph.nodes[0].name, "layer_0_core");
        assert_eq!(graph.nodes[1].name, "layer_1_core");
    }

    /// `has_per_layer_embedding = true` 时 only_if 节点正常展开。
    #[test]
    fn only_if_expands_node_when_true() {
        use crate::loader::onnx::OnnxAttributeValue;

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

        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.hidden_size = 1024;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        config.vocab_size = 32000;
        config.hidden_size_per_layer_input = 256;
        config.has_per_layer_embedding = true;

        let graph = template.to_onnx_graph(&config).unwrap();
        // 每层 core + ple = 2 节点, 2 层共 4 节点
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.nodes[0].name, "layer_0_core");
        assert_eq!(graph.nodes[1].name, "layer_0_ple");
        assert_eq!(graph.nodes[2].name, "layer_1_core");
        assert_eq!(graph.nodes[3].name, "layer_1_ple");

        // layer_idx / num_layers 占位符替换后必须以 Int 形式发射 (否则下游
        // atomic_op_to_kind::require_usize 会因 AttrValue::String 失败)。
        let read_int = |attrs: &std::collections::HashMap<String, crate::loader::onnx::OnnxAttribute>, key: &str| {
            match attrs.get(key).map(|a| &a.value) {
                Some(OnnxAttributeValue::Int(v)) => *v,
                other => panic!("属性 {key} 期望 Int, 实际: {:?}", other),
            }
        };
        assert_eq!(read_int(&graph.nodes[1].attributes, "layer_idx"), 0);
        assert_eq!(read_int(&graph.nodes[1].attributes, "num_layers"), 2);
        assert_eq!(read_int(&graph.nodes[3].attributes, "layer_idx"), 1);
    }

    /// 三段比较语法 `field op value` 按整数比较求值。
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
        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.hidden_size = 1024;
        config.num_attention_heads = 8;
        config.head_dim = 128;
        config.vocab_size = 32000;

        let graph = template.to_onnx_graph(&config).unwrap();
        // a: hidden_size > 0 → true; b: num_hidden_layers == 0 → false; c: != 0 → true
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].name, "a");
        assert_eq!(graph.nodes[1].name, "c");
    }

    /// 未知 only_if 字段必须返回错误 (禁止静默 false 掩盖拼写错误)。
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
        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let err = template.to_onnx_graph(&config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("typo_field_that_does_not_exist"),
            "错误消息应包含未知字段名, 实际: {msg}");
    }

    /// 现有无 only_if 的节点始终展开 (向前兼容验证)。
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
        let mut config = super::super::resolve::ResolvedConfig::default();
        config.num_hidden_layers = 1;
        config.hidden_size = 1;
        config.num_attention_heads = 1;
        config.vocab_size = 1;

        let graph = template.to_onnx_graph(&config).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].name, "a");
    }
}
