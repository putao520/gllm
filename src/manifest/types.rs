//! Layer 1: Manifest types (SSOT).

use std::borrow::Cow;

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
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat" | "generation" | "generator" | "text-generation" => Some(Self::Chat),
            "embedding" | "embeddings" | "embed" => Some(Self::Embedding),
            "rerank" | "reranker" | "re-ranker" | "re-rank" => Some(Self::Reranker),
            _ => None,
        }
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

/// Override-only manifest entries (registry layer).
#[derive(Debug, Clone)]
pub struct ManifestOverride {
    pub model_id: &'static str,
    pub file_map: FileMap,
    pub rope_base_override: Option<f32>,
    pub max_context_override: Option<usize>,
    pub moe_config: Option<MoEConfig>,
    pub kind: Option<ModelKind>,
}
