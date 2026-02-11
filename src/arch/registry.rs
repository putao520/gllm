//! 架构模板注册表 (REQ-ARCH-002)

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::manifest::ModelArchitecture;

use super::template::ArchTemplate;

/// 全局模板注册表
static REGISTRY: OnceLock<ArchRegistry> = OnceLock::new();

/// 架构注册表
#[derive(Debug, Default)]
pub struct ArchRegistry {
    templates: HashMap<String, ArchTemplate>,
    arch_mapping: HashMap<ModelArchitecture, String>,
}

impl ArchRegistry {
    /// 创建空注册表
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册模板
    pub fn register(&mut self, template: ArchTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// 注册架构到模板名的映射
    pub fn map_arch(&mut self, arch: ModelArchitecture, template_name: &str) {
        self.arch_mapping
            .insert(arch, template_name.to_string());
    }

    /// 获取模板（按名称）
    pub fn get(&self, name: &str) -> Option<&ArchTemplate> {
        self.templates.get(name)
    }

    /// 获取模板（按架构）
    pub fn get_by_arch(&self, arch: ModelArchitecture) -> Option<&ArchTemplate> {
        self.arch_mapping
            .get(&arch)
            .and_then(|name| self.templates.get(name))
    }

    /// 列出所有模板名
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

/// 获取全局模板
pub fn get_template(name: &str) -> Option<&'static ArchTemplate> {
    REGISTRY.get().and_then(|r| r.get(name))
}

/// 按架构获取模板
pub fn get_template_by_arch(arch: ModelArchitecture) -> Option<&'static ArchTemplate> {
    REGISTRY.get().and_then(|r| r.get_by_arch(arch))
}

/// 注册内置模板
pub fn register_builtin_templates() {
    let _ = REGISTRY.get_or_init(|| {
        let mut registry = ArchRegistry::new();

        // 注册 Qwen3
        if let Ok(qwen3) = ArchTemplate::from_yaml(QWEN3_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Qwen3, "qwen3");
            registry.register(qwen3);
        }

        // 注册 Llama
        if let Ok(llama) = ArchTemplate::from_yaml(LLAMA_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Llama4, "llama");
            registry.register(llama);
        }

        registry
    });
}

// ============================================================================
// 内置模板定义
// ============================================================================

/// Qwen3 架构模板 (REQ-ARCH-004)
const QWEN3_TEMPLATE: &str = r#"
name: qwen3
version: "1.0"

config:
  num_layers: "${num_hidden_layers}"
  hidden_size: "${hidden_size}"
  num_heads: "${num_attention_heads}"
  num_kv_heads: "${num_key_value_heads}"
  head_dim: "${head_dim}"
  intermediate_size: "${intermediate_size}"
  vocab_size: "${vocab_size}"
  rope_theta: "${rope_theta}"

tensor_patterns:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  layer_prefix: "model.layers.{}"
  q_proj: "self_attn.q_proj.weight"
  k_proj: "self_attn.k_proj.weight"
  v_proj: "self_attn.v_proj.weight"
  o_proj: "self_attn.o_proj.weight"
  gate_proj: "mlp.gate_proj.weight"
  up_proj: "mlp.up_proj.weight"
  down_proj: "mlp.down_proj.weight"
  input_layernorm: "input_layernorm.weight"
  post_attention_layernorm: "post_attention_layernorm.weight"
  final_norm: "model.norm.weight"

graph:
  inputs:
    - name: input_ids
      dtype: int64
      shape:
        - batch
        - seq_len

  outputs:
    - name: logits
      dtype: "${dtype}"
      shape:
        - batch
        - seq_len
        - "${vocab_size}"

  nodes:
    - name: embed
      op_type: Gather
      inputs:
        - "model.embed_tokens.weight"
        - input_ids
      outputs:
        - hidden_states

fusion_hints:
  - pattern:
      - q_proj
      - k_proj
      - v_proj
      - rope
    target: FusedQkvRope
  - pattern:
      - gate
      - up
      - silu
      - mul
    target: SwiGLU
"#;

/// Llama 架构模板
const LLAMA_TEMPLATE: &str = r#"
name: llama
version: "1.0"

config:
  num_layers: "${num_hidden_layers}"
  hidden_size: "${hidden_size}"
  num_heads: "${num_attention_heads}"
  num_kv_heads: "${num_key_value_heads}"
  head_dim: "${head_dim}"
  intermediate_size: "${intermediate_size}"
  vocab_size: "${vocab_size}"
  rope_theta: "${rope_theta}"

tensor_patterns:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  layer_prefix: "model.layers.{}"
  q_proj: "self_attn.q_proj.weight"
  k_proj: "self_attn.k_proj.weight"
  v_proj: "self_attn.v_proj.weight"
  o_proj: "self_attn.o_proj.weight"
  gate_proj: "mlp.gate_proj.weight"
  up_proj: "mlp.up_proj.weight"
  down_proj: "mlp.down_proj.weight"
  input_layernorm: "input_layernorm.weight"
  post_attention_layernorm: "post_attention_layernorm.weight"
  final_norm: "model.norm.weight"

graph:
  inputs:
    - name: input_ids
      dtype: int64
      shape: [batch, seq_len]

  outputs:
    - name: logits
      dtype: "${dtype}"
      shape: [batch, seq_len, "${vocab_size}"]

  nodes: []

fusion_hints:
  - pattern: [q_proj, k_proj, v_proj, rope]
    target: FusedQkvRope
  - pattern: [gate, up, silu, mul]
    target: SwiGLU
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_qwen3_parses() {
        let template = ArchTemplate::from_yaml(QWEN3_TEMPLATE).unwrap();
        assert_eq!(template.name, "qwen3");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_llama_parses() {
        let template = ArchTemplate::from_yaml(LLAMA_TEMPLATE).unwrap();
        assert_eq!(template.name, "llama");
    }

    #[test]
    fn registry_lookup_works() {
        register_builtin_templates();
        assert!(get_template("qwen3").is_some());
        assert!(get_template("llama").is_some());
        assert!(get_template("nonexistent").is_none());
    }
}
