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
        self.arch_mapping.insert(arch, template_name.to_string());
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
            registry.map_arch(ModelArchitecture::Qwen3MoE, "qwen3");
            // Qwen2.5 shares the same GQA+SwiGLU+RoPE architecture as Qwen3
            registry.map_arch(ModelArchitecture::Qwen2_5, "qwen3");
            registry.register(qwen3);
        }

        // 注册 Llama
        if let Ok(llama) = ArchTemplate::from_yaml(LLAMA_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Llama4, "llama");
            registry.register(llama);
        }

        // SmolLM2 and InternLM3 share the Llama template
        registry.map_arch(ModelArchitecture::SmolLM2, "llama");
        registry.map_arch(ModelArchitecture::InternLM3, "llama");

        // 注册 Mistral3 / Ministral
        if let Ok(mistral3) = ArchTemplate::from_yaml(MISTRAL3_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Mistral3, "mistral3");
            registry.map_arch(ModelArchitecture::Ministral, "mistral3");
            registry.register(mistral3);
        }

        // 注册 GLM4 / GLM5
        if let Ok(glm4) = ArchTemplate::from_yaml(GLM4_TEMPLATE) {
            registry.map_arch(ModelArchitecture::GLM4, "glm4");
            registry.map_arch(ModelArchitecture::GLM5, "glm4");
            registry.register(glm4);
        }

        // 注册 Phi4
        if let Ok(phi4) = ArchTemplate::from_yaml(PHI4_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Phi4, "phi4");
            registry.register(phi4);
        }

        // 注册 Gemma2
        if let Ok(gemma2) = ArchTemplate::from_yaml(GEMMA2_TEMPLATE) {
            registry.map_arch(ModelArchitecture::Gemma2, "gemma2");
            registry.register(gemma2);
        }

        // 注册 GPT2Next
        if let Ok(gpt2next) = ArchTemplate::from_yaml(GPT2NEXT_TEMPLATE) {
            registry.map_arch(ModelArchitecture::GPT2Next, "gpt2next");
            registry.register(gpt2next);
        }

        // 注册 XLM-R / XLM-R Next
        if let Ok(xlmr) = ArchTemplate::from_yaml(XLMR_TEMPLATE) {
            registry.map_arch(ModelArchitecture::XlmR, "xlmr");
            registry.map_arch(ModelArchitecture::XlmRNext, "xlmr");
            registry.register(xlmr);
        }

        // 注册 DeepSeek
        if let Ok(deepseek) = ArchTemplate::from_yaml(DEEPSEEK_TEMPLATE) {
            registry.map_arch(ModelArchitecture::DeepSeek, "deepseek");
            registry.register(deepseek);
        }

        registry
    });
}

// ============================================================================
// 内置模板定义
// ============================================================================

/// Qwen3 架构模板 (REQ-ARCH-004)
const QWEN3_TEMPLATE: &str = include_str!("templates/qwen3.yaml");

/// Mistral3 架构模板
const MISTRAL3_TEMPLATE: &str = include_str!("templates/mistral3.yaml");

/// GLM4 架构模板
const GLM4_TEMPLATE: &str = include_str!("templates/glm4.yaml");

/// Phi4 架构模板
const PHI4_TEMPLATE: &str = include_str!("templates/phi4.yaml");

/// Gemma2 架构模板
const GEMMA2_TEMPLATE: &str = include_str!("templates/gemma2.yaml");

/// GPT2Next 架构模板
const GPT2NEXT_TEMPLATE: &str = include_str!("templates/gpt2next.yaml");

/// XLM-R 架构模板
const XLMR_TEMPLATE: &str = include_str!("templates/xlmr.yaml");

/// DeepSeek 架构模板
const DEEPSEEK_TEMPLATE: &str = include_str!("templates/deepseek.yaml");

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
    fn builtin_mistral3_parses() {
        let template = ArchTemplate::from_yaml(MISTRAL3_TEMPLATE).unwrap();
        assert_eq!(template.name, "mistral3");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_glm4_parses() {
        let template = ArchTemplate::from_yaml(GLM4_TEMPLATE).unwrap();
        assert_eq!(template.name, "glm4");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_phi4_parses() {
        let template = ArchTemplate::from_yaml(PHI4_TEMPLATE).unwrap();
        assert_eq!(template.name, "phi4");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_gemma2_parses() {
        let template = ArchTemplate::from_yaml(GEMMA2_TEMPLATE).unwrap();
        assert_eq!(template.name, "gemma2");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_gpt2next_parses() {
        let template = ArchTemplate::from_yaml(GPT2NEXT_TEMPLATE).unwrap();
        assert_eq!(template.name, "gpt2next");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_xlmr_parses() {
        let template = ArchTemplate::from_yaml(XLMR_TEMPLATE).unwrap();
        assert_eq!(template.name, "xlmr");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn builtin_deepseek_parses() {
        let template = ArchTemplate::from_yaml(DEEPSEEK_TEMPLATE).unwrap();
        assert_eq!(template.name, "deepseek");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.embedding.is_some());
    }

    #[test]
    fn registry_lookup_works() {
        register_builtin_templates();
        assert!(get_template("qwen3").is_some());
        assert!(get_template("llama").is_some());
        assert!(get_template("mistral3").is_some());
        assert!(get_template("glm4").is_some());
        assert!(get_template("phi4").is_some());
        assert!(get_template("gemma2").is_some());
        assert!(get_template("gpt2next").is_some());
        assert!(get_template("xlmr").is_some());
        assert!(get_template("deepseek").is_some());
        assert!(get_template("nonexistent").is_none());
    }

    #[test]
    fn registry_arch_mapping_works() {
        register_builtin_templates();
        assert!(get_template_by_arch(ModelArchitecture::Mistral3).is_some());
        assert!(get_template_by_arch(ModelArchitecture::Ministral).is_some());
        assert!(get_template_by_arch(ModelArchitecture::GLM4).is_some());
        assert!(get_template_by_arch(ModelArchitecture::GLM5).is_some());
        assert!(get_template_by_arch(ModelArchitecture::Phi4).is_some());
        assert!(get_template_by_arch(ModelArchitecture::Gemma2).is_some());
        assert!(get_template_by_arch(ModelArchitecture::GPT2Next).is_some());
        assert!(get_template_by_arch(ModelArchitecture::XlmR).is_some());
        assert!(get_template_by_arch(ModelArchitecture::XlmRNext).is_some());
        assert!(get_template_by_arch(ModelArchitecture::DeepSeek).is_some());
    }
}
