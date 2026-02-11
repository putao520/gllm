//! Layer 1: Manifest types (SSOT).

use std::borrow::Cow;
use std::collections::HashMap;

/// Role of a tensor in the model architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    Embedding,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    OutputHead,
    LayerNorm,
    Rope,
}

/// HuggingFace file rename map.
///
/// Each pair is (logical_name, repo_name).
pub type FileMap = &'static [(&'static str, &'static str)];

pub const EMPTY_FILE_MAP: FileMap = &[];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Qwen2_5,
    Qwen3,
    Qwen3MoE,
    Llama4,
    Mistral3,
    Ministral,
    GLM4,
    GLM5,
    GPT2Next,
    Phi4,
    Gemma2,
    XlmR,
    XlmRNext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorNamingRule {
    Qwen3,
    Llama4,
    Mistral3,
    Ministral,
    GLM4,
    GLM5,
    GPT2Next,
    Phi4,
    Gemma2,
    XlmR,
    XlmRNext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Chat,
    Embedding,
    Reranker,
}

impl ModelKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat" | "generation" | "generator" | "text-generation" => Some(Self::Chat),
            "embedding" | "embeddings" | "embed" => Some(Self::Embedding),
            "rerank" | "reranker" | "re-ranker" | "re-rank" => Some(Self::Reranker),
            _ => None,
        }
    }
}

impl std::str::FromStr for ModelKind {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouterType {
    Qwen,
    Mixtral,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoEConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub router_type: RouterType,
}

/// Model manifest (runtime data + optional overrides).
#[derive(Debug, Clone)]
pub struct ModelManifest {
    // Identity (HF Model ID)
    pub model_id: Cow<'static, str>,
    // Optional file rename overrides for non-standard repos.
    pub file_map: FileMap,

    // Architecture
    pub arch: ModelArchitecture,
    pub tensor_rules: TensorNamingRule,
    pub kind: ModelKind,

    // Inference overrides (None means use config.json)
    pub rope_base_override: Option<f32>,
    pub max_context_override: Option<usize>,

    // MoE specific configuration (None means read from config.json)
    pub moe_config: Option<MoEConfig>,

    // Tensor map for dynamic loading
    pub tensor_map: HashMap<TensorRole, String>,
}

impl ModelManifest {
    pub fn is_moe(&self) -> bool {
        self.moe_config.is_some()
            || matches!(
                self.arch,
                ModelArchitecture::Qwen3MoE | ModelArchitecture::Llama4
            )
    }
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            model_id: Cow::Borrowed("default"),
            file_map: EMPTY_FILE_MAP,
            arch: ModelArchitecture::Llama4,
            tensor_rules: TensorNamingRule::Llama4,
            kind: ModelKind::Chat,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        }
    }
}

/// Map architecture token string to ModelArchitecture enum.
///
/// Supports various formats from config.json (architectures field) and GGUF metadata.
pub fn map_architecture_token(token: &str) -> Option<ModelArchitecture> {
    match normalize_architecture_token(token).as_str() {
        "ministral" | "ministralforcausallm" => Some(ModelArchitecture::Ministral),
        "mistral" | "mistralforcausallm" => Some(ModelArchitecture::Mistral3),
        "qwen3_moe" | "qwen3moe" | "qwen3moeforcausallm" => Some(ModelArchitecture::Qwen3MoE),
        "qwen3" | "qwen3forcausallm" => Some(ModelArchitecture::Qwen3),
        "qwen2_5" | "qwen2_5forcausallm" => Some(ModelArchitecture::Qwen2_5),
        "qwen2" | "qwen2forcausallm" => Some(ModelArchitecture::Qwen2_5),
        "llama" | "llamaforcausallm" => Some(ModelArchitecture::Llama4),
        "phi3" | "phi3forcausallm" | "phi4" | "phi4forcausallm" => Some(ModelArchitecture::Phi4),
        "gemma" | "gemmaforcausallm" | "gemma2" | "gemma2forcausallm" => {
            Some(ModelArchitecture::Gemma2)
        }
        "glm5" | "glm5forcausallm" => Some(ModelArchitecture::GLM5),
        "glm4" | "glm4forcausallm" | "chatglm" | "chatglmforcausallm" => {
            Some(ModelArchitecture::GLM4)
        }
        "glm" | "glmforcausallm" => Some(ModelArchitecture::GLM5),
        "gpt2" | "gpt2lmheadmodel" | "gpt_oss" | "gptoss" => Some(ModelArchitecture::GPT2Next),
        "xlm_roberta" | "xlm_roberta_model" | "xlmr" | "roberta" | "bert" => {
            Some(ModelArchitecture::XlmR)
        }
        _ => None,
    }
}

/// Normalize architecture token for matching.
fn normalize_architecture_token(token: &str) -> String {
    token
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .map(|ch| match ch {
            '-' | '.' => '_',
            _ => ch.to_ascii_lowercase(),
        })
        .collect()
}

/// Get tensor naming rules for a given architecture.
pub fn tensor_rules_for_arch(arch: ModelArchitecture) -> TensorNamingRule {
    match arch {
        ModelArchitecture::Qwen2_5 => TensorNamingRule::Llama4,
        ModelArchitecture::Qwen3 => TensorNamingRule::Qwen3,
        ModelArchitecture::Qwen3MoE => TensorNamingRule::Qwen3,
        ModelArchitecture::Llama4 => TensorNamingRule::Llama4,
        ModelArchitecture::Mistral3 => TensorNamingRule::Mistral3,
        ModelArchitecture::Ministral => TensorNamingRule::Ministral,
        ModelArchitecture::GLM4 => TensorNamingRule::GLM4,
        ModelArchitecture::GLM5 => TensorNamingRule::GLM5,
        ModelArchitecture::GPT2Next => TensorNamingRule::GPT2Next,
        ModelArchitecture::Phi4 => TensorNamingRule::Phi4,
        ModelArchitecture::Gemma2 => TensorNamingRule::Gemma2,
        ModelArchitecture::XlmR => TensorNamingRule::XlmR,
        ModelArchitecture::XlmRNext => TensorNamingRule::XlmRNext,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_architecture_token_uses_exact_normalized_matching() {
        assert_eq!(
            map_architecture_token("LlamaForCausalLM"),
            Some(ModelArchitecture::Llama4)
        );
        assert_eq!(
            map_architecture_token("Qwen2ForCausalLM"),
            Some(ModelArchitecture::Qwen2_5)
        );
        assert_eq!(
            map_architecture_token("Qwen2.5ForCausalLM"),
            Some(ModelArchitecture::Qwen2_5)
        );
        assert_eq!(
            map_architecture_token("MistralForCausalLM"),
            Some(ModelArchitecture::Mistral3)
        );
        assert_eq!(
            map_architecture_token("Gemma2ForCausalLM"),
            Some(ModelArchitecture::Gemma2)
        );
        assert_eq!(map_architecture_token("custom-llama-adapter"), None);
    }
}
