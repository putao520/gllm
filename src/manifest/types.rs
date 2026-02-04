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

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnownModel {
    Qwen2_5_0_5B,
    Qwen3_7B,
    Qwen3_MoE_A22B,
    Qwen3_Thinking,
    Llama4_8B,
    Llama4_Scout_17B,
    SmolLM2_135M,
    SmolLM3_3B,
    Internlm3_8B,
    Ministral_8B,
    MistralSmall_3,
    GLM4_7_Flash,
    GLM5_9B,
    GptOss_1_5B,
    GptOss_12B,
    GPT2,
    Phi4,
    Phi4_Mini,
    Gemma2_2B_It,
    Gemma2_9B,
    Gemma2_27B,
    Qwen3_Embed,
    Bge_M3,
    Bge_M4,
    E5_Small,
    E5_Base,
    E5_Large,
    M3e_Base,
    JinaEmbeddingsV2_Base,
    JinaEmbeddingsV2_Small,
    JinaEmbeddingsV4,
    Qwen3_Rerank,
    Bge_Rerank_V3,
    Bge_Rerank_V2_M3,
}

impl KnownModel {
    /// 判断是否为生成模型 (Generator)
    pub fn is_generator(&self) -> bool {
        matches!(
            self,
            Self::Qwen2_5_0_5B
                | Self::Qwen3_7B
                | Self::Qwen3_MoE_A22B
                | Self::Qwen3_Thinking
                | Self::Llama4_8B
                | Self::Llama4_Scout_17B
                | Self::SmolLM2_135M
                | Self::SmolLM3_3B
                | Self::Internlm3_8B
                | Self::Ministral_8B
                | Self::MistralSmall_3
                | Self::GLM4_7_Flash
                | Self::GLM5_9B
                | Self::GptOss_1_5B
                | Self::GptOss_12B
                | Self::GPT2
                | Self::Phi4
                | Self::Phi4_Mini
                | Self::Gemma2_2B_It
                | Self::Gemma2_9B
                | Self::Gemma2_27B
        )
    }

    /// 判断是否为嵌入模型 (Embedding)
    pub fn is_embedding(&self) -> bool {
        matches!(
            self,
            Self::Qwen3_Embed
                | Self::Bge_M3
                | Self::Bge_M4
                | Self::E5_Small
                | Self::E5_Base
                | Self::E5_Large
                | Self::M3e_Base
                | Self::JinaEmbeddingsV2_Base
                | Self::JinaEmbeddingsV2_Small
                | Self::JinaEmbeddingsV4
        )
    }

    /// 判断是否为重排序模型 (Reranker)
    pub fn is_reranker(&self) -> bool {
        matches!(
            self,
            Self::Qwen3_Rerank | Self::Bge_Rerank_V3 | Self::Bge_Rerank_V2_M3
        )
    }
}
