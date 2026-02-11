//! YAML 模板解析和类型定义 (REQ-ARCH-001)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 架构模板 - 从 YAML 解析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchTemplate {
    /// 架构名称 (e.g., "qwen3", "llama")
    pub name: String,
    /// 模板版本
    #[serde(default = "default_version")]
    pub version: String,
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
                    nodes.push(self.node_def_to_onnx(node_def, config, None)?);
                }
                GraphNode::Repeat(repeat_block) => {
                    let repeat_count = self.resolve_repeat_count(&repeat_block.repeat, config)?;
                    for i in 0..repeat_count {
                        for node_def in &repeat_block.nodes {
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
                AttributeValue::String(s) => OnnxAttributeValue::String(substitute(s)),
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
}
