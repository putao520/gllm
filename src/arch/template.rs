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
}
