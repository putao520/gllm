//! Layer 1: Static model manifests (SSOT).

use std::borrow::Cow;

use super::types::{
    KnownModel, ModelArchitecture, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};

pub const QWEN2_5_0_5B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen2.5-0.5B-Instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen2_5,
    tensor_rules: TensorNamingRule::Llama4, // Qwen2.5 使用类似 Llama 的命名规则
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_7B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-0.6B"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

/// Qwen3-1.7B (实际存在的小型 Qwen3 模型)
pub const QWEN3_1_7B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-1.7B"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_MOE_A22B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-235B-A22B-Instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3MoE,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_THINKING_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-Max-Thinking"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const LLAMA4_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("meta-llama/Llama-4-8B-Instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const LLAMA4_SCOUT_17B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("meta-llama/Llama-4-Scout"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const SMOLLM2_135M_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("HuggingFaceTB/SmolLM2-135M-Instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const SMOLLM3_3B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("HuggingFaceTB/SmolLM3-3B"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const INTERNLM3_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("internlm/internlm3-8b-instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const MINISTRAL_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("mistralai/Ministral-8B-Instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Ministral,
    tensor_rules: TensorNamingRule::Ministral,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const MISTRAL_SMALL_3_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("mistralai/Mistral-Small-3.2"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Mistral3,
    tensor_rules: TensorNamingRule::Mistral3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GLM_4_7_FLASH_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("THUDM/glm-4.7-flash"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GLM4,
    tensor_rules: TensorNamingRule::GLM4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GLM_5_9B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("THUDM/glm-5-9b-chat"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GLM5,
    tensor_rules: TensorNamingRule::GLM5,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

/// GLM-4-9B - smaller GLM-4 model
pub const GLM_4_9B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("zai-org/glm-4-9b"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GLM4,
    tensor_rules: TensorNamingRule::GLM4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GPT_OSS_1_5B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("openai/gpt-oss-1.5b"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GPT2Next,
    tensor_rules: TensorNamingRule::GPT2Next,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GPT_OSS_12B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("openai/gpt-oss-12b"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GPT2Next,
    tensor_rules: TensorNamingRule::GPT2Next,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

/// GPT-2 (124M) - the original GPT-2 small model
pub const GPT2_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("openai-community/gpt2"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GPT2Next,
    tensor_rules: TensorNamingRule::GPT2Next,
    rope_base_override: None,
    max_context_override: Some(1024),
    moe_config: None,
};

pub const PHI4_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("microsoft/Phi-4"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Phi4,
    tensor_rules: TensorNamingRule::Phi4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const PHI4_MINI_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("microsoft/Phi-4-mini-instruct"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Phi4,
    tensor_rules: TensorNamingRule::Phi4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_2B_IT_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("google/gemma-2-2b-it"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_9B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("google/gemma-2-9b-it"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_27B_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("google/gemma-2-27b-it"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_EMBED_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-Embedding"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_M3_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("BAAI/bge-m3"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_M4_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("BAAI/bge-m4"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmRNext,
    tensor_rules: TensorNamingRule::XlmRNext,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_SMALL_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("intfloat/e5-small"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("intfloat/e5-base"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_LARGE_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("intfloat/e5-large"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const M3E_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("moka-ai/m3e-base"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V2_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("jinaai/jina-embeddings-v2-base-en"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V2_SMALL_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("jinaai/jina-embeddings-v2-small-en"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V4_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("jinaai/jina-embeddings-v3"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_RERANK_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("Qwen/Qwen3-Reranker"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_RERANK_V3_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("BAAI/bge-reranker-v3"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmRNext,
    tensor_rules: TensorNamingRule::XlmRNext,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_RERANK_V2_M3_MANIFEST: ModelManifest = ModelManifest {
    model_id: Cow::Borrowed("BAAI/bge-reranker-v2-m3"),
    file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const ALL_MANIFESTS: &[&ModelManifest] = &[
    &QWEN2_5_0_5B_MANIFEST,
    &QWEN3_7B_MANIFEST,
    &QWEN3_1_7B_MANIFEST,
    &QWEN3_MOE_A22B_MANIFEST,
    &QWEN3_THINKING_MANIFEST,
    &LLAMA4_8B_MANIFEST,
    &LLAMA4_SCOUT_17B_MANIFEST,
    &SMOLLM2_135M_MANIFEST,
    &SMOLLM3_3B_MANIFEST,
    &INTERNLM3_8B_MANIFEST,
    &MINISTRAL_8B_MANIFEST,
    &MISTRAL_SMALL_3_MANIFEST,
    &GLM_4_7_FLASH_MANIFEST,
    &GLM_5_9B_MANIFEST,
    &GLM_4_9B_MANIFEST,
    &GPT_OSS_1_5B_MANIFEST,
    &GPT_OSS_12B_MANIFEST,
    &GPT2_MANIFEST,
    &PHI4_MANIFEST,
    &PHI4_MINI_MANIFEST,
    &GEMMA2_2B_IT_MANIFEST,
    &GEMMA2_9B_MANIFEST,
    &GEMMA2_27B_MANIFEST,
    &QWEN3_EMBED_MANIFEST,
    &BGE_M3_MANIFEST,
    &BGE_M4_MANIFEST,
    &E5_SMALL_MANIFEST,
    &E5_BASE_MANIFEST,
    &E5_LARGE_MANIFEST,
    &M3E_BASE_MANIFEST,
    &JINA_EMBEDDINGS_V2_BASE_MANIFEST,
    &JINA_EMBEDDINGS_V2_SMALL_MANIFEST,
    &JINA_EMBEDDINGS_V4_MANIFEST,
    &QWEN3_RERANK_MANIFEST,
    &BGE_RERANK_V3_MANIFEST,
    &BGE_RERANK_V2_M3_MANIFEST,
];

pub fn manifest_by_id(model: KnownModel) -> &'static ModelManifest {
    match model {
        KnownModel::Qwen2_5_0_5B => &QWEN2_5_0_5B_MANIFEST,
        KnownModel::Qwen3_7B => &QWEN3_7B_MANIFEST,
        KnownModel::Qwen3_MoE_A22B => &QWEN3_MOE_A22B_MANIFEST,
        KnownModel::Qwen3_Thinking => &QWEN3_THINKING_MANIFEST,
        KnownModel::Llama4_8B => &LLAMA4_8B_MANIFEST,
        KnownModel::Llama4_Scout_17B => &LLAMA4_SCOUT_17B_MANIFEST,
        KnownModel::SmolLM2_135M => &SMOLLM2_135M_MANIFEST,
        KnownModel::SmolLM3_3B => &SMOLLM3_3B_MANIFEST,
        KnownModel::Internlm3_8B => &INTERNLM3_8B_MANIFEST,
        KnownModel::Ministral_8B => &MINISTRAL_8B_MANIFEST,
        KnownModel::MistralSmall_3 => &MISTRAL_SMALL_3_MANIFEST,
        KnownModel::GLM4_7_Flash => &GLM_4_7_FLASH_MANIFEST,
        KnownModel::GLM5_9B => &GLM_5_9B_MANIFEST,
        KnownModel::GptOss_1_5B => &GPT_OSS_1_5B_MANIFEST,
        KnownModel::GptOss_12B => &GPT_OSS_12B_MANIFEST,
        KnownModel::GPT2 => &GPT2_MANIFEST,
        KnownModel::Phi4 => &PHI4_MANIFEST,
        KnownModel::Phi4_Mini => &PHI4_MINI_MANIFEST,
        KnownModel::Gemma2_2B_It => &GEMMA2_2B_IT_MANIFEST,
        KnownModel::Gemma2_9B => &GEMMA2_9B_MANIFEST,
        KnownModel::Gemma2_27B => &GEMMA2_27B_MANIFEST,
        KnownModel::Qwen3_Embed => &QWEN3_EMBED_MANIFEST,
        KnownModel::Bge_M3 => &BGE_M3_MANIFEST,
        KnownModel::Bge_M4 => &BGE_M4_MANIFEST,
        KnownModel::E5_Small => &E5_SMALL_MANIFEST,
        KnownModel::E5_Base => &E5_BASE_MANIFEST,
        KnownModel::E5_Large => &E5_LARGE_MANIFEST,
        KnownModel::M3e_Base => &M3E_BASE_MANIFEST,
        KnownModel::JinaEmbeddingsV2_Base => &JINA_EMBEDDINGS_V2_BASE_MANIFEST,
        KnownModel::JinaEmbeddingsV2_Small => &JINA_EMBEDDINGS_V2_SMALL_MANIFEST,
        KnownModel::JinaEmbeddingsV4 => &JINA_EMBEDDINGS_V4_MANIFEST,
        KnownModel::Qwen3_Rerank => &QWEN3_RERANK_MANIFEST,
        KnownModel::Bge_Rerank_V3 => &BGE_RERANK_V3_MANIFEST,
        KnownModel::Bge_Rerank_V2_M3 => &BGE_RERANK_V2_M3_MANIFEST,
    }
}
