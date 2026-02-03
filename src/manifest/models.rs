//! Layer 1: Static model manifests (SSOT).

use super::types::{
    KnownModel, ModelArchitecture, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};

pub const QWEN3_7B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_7B,
    aliases: &["qwen3-7b"],
    hf_repo: "Qwen/Qwen3-7B-Instruct",
    model_scope_repo: Some("qwen/Qwen3-7B-Instruct"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

/// Qwen3-1.7B (实际存在的小型 Qwen3 模型)
pub const QWEN3_1_7B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_7B, // 复用相同的架构
    aliases: &["qwen3-1.7b", "qwen3-1.7b-instruct"],
    hf_repo: "Qwen/Qwen3-1.7B",
    model_scope_repo: Some("qwen/Qwen3-1.7B"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_MOE_A22B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_MoE_A22B,
    aliases: &["qwen3-moe"],
    hf_repo: "Qwen/Qwen3-235B-A22B-Instruct",
    model_scope_repo: Some("qwen/Qwen3-235B-A22B-Instruct"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3MoE,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_THINKING_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_Thinking,
    aliases: &["qwen3-thinking"],
    hf_repo: "Qwen/Qwen3-Max-Thinking",
    model_scope_repo: Some("qwen/Qwen3-Max-Thinking"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const LLAMA4_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Llama4_8B,
    aliases: &["llama-4-8b"],
    hf_repo: "meta-llama/Llama-4-8B-Instruct",
    model_scope_repo: Some("LLM-Research/Llama-4-8B-Instruct"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const LLAMA4_SCOUT_17B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Llama4_Scout_17B,
    aliases: &["llama-4-scout"],
    hf_repo: "meta-llama/Llama-4-Scout",
    model_scope_repo: Some("LLM-Research/Llama-4-Scout"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const SMOLLM2_135M_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::SmolLM2_135M,
    aliases: &["smollm2-135m"],
    hf_repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const SMOLLM3_3B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::SmolLM3_3B,
    aliases: &["smollm3-3b"],
    hf_repo: "HuggingFaceTB/SmolLM3-3B",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const INTERNLM3_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Internlm3_8B,
    aliases: &["internlm3-8b"],
    hf_repo: "internlm/internlm3-8b-instruct",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Llama4,
    tensor_rules: TensorNamingRule::Llama4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const MINISTRAL_8B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Ministral_8B,
    aliases: &["ministral-8b"],
    hf_repo: "mistralai/Ministral-8B-Instruct",
    model_scope_repo: Some("AI-ModelScope/Ministral-8B-Instruct"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Ministral,
    tensor_rules: TensorNamingRule::Ministral,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const MISTRAL_SMALL_3_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::MistralSmall_3,
    aliases: &["mistral-small-3"],
    hf_repo: "mistralai/Mistral-Small-3.2",
    model_scope_repo: Some("AI-ModelScope/Mistral-Small-3.2"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Mistral3,
    tensor_rules: TensorNamingRule::Mistral3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GLM_4_7_FLASH_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::GLM4_7_Flash,
    aliases: &["glm-4.7-flash"],
    hf_repo: "THUDM/glm-4.7-flash",
    model_scope_repo: Some("ZhipuAI/glm-4.7-flash"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GLM4,
    tensor_rules: TensorNamingRule::GLM4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GLM_5_9B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::GLM5_9B,
    aliases: &["glm-5-9b"],
    hf_repo: "THUDM/glm-5-9b-chat",
    model_scope_repo: Some("ZhipuAI/glm-5-9b-chat"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GLM5,
    tensor_rules: TensorNamingRule::GLM5,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GPT_OSS_1_5B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::GptOss_1_5B,
    aliases: &["gpt-oss-1.5b"],
    hf_repo: "openai/gpt-oss-1.5b",
    model_scope_repo: Some("openai-mirror/gpt-oss-1.5b"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GPT2Next,
    tensor_rules: TensorNamingRule::GPT2Next,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GPT_OSS_12B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::GptOss_12B,
    aliases: &["gpt-oss-12b"],
    hf_repo: "openai/gpt-oss-12b",
    model_scope_repo: Some("openai-mirror/gpt-oss-12b"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::GPT2Next,
    tensor_rules: TensorNamingRule::GPT2Next,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const PHI4_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Phi4,
    aliases: &["phi-4"],
    hf_repo: "microsoft/Phi-4",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Phi4,
    tensor_rules: TensorNamingRule::Phi4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const PHI4_MINI_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Phi4_Mini,
    aliases: &["phi-4-mini"],
    hf_repo: "microsoft/Phi-4-mini-instruct",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Phi4,
    tensor_rules: TensorNamingRule::Phi4,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_2B_IT_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Gemma2_2B_It,
    aliases: &["gemma-2-2b-it"],
    hf_repo: "google/gemma-2-2b-it",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_9B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Gemma2_9B,
    aliases: &["gemma-2-9b"],
    hf_repo: "google/gemma-2-9b-it",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const GEMMA2_27B_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Gemma2_27B,
    aliases: &["gemma-2-27b"],
    hf_repo: "google/gemma-2-27b-it",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Gemma2,
    tensor_rules: TensorNamingRule::Gemma2,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_EMBED_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_Embed,
    aliases: &["qwen3-embed"],
    hf_repo: "Qwen/Qwen3-Embedding",
    model_scope_repo: Some("qwen/Qwen3-Embedding"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_M3_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Bge_M3,
    aliases: &["bge-m3"],
    hf_repo: "BAAI/bge-m3",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_M4_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Bge_M4,
    aliases: &["bge-m4"],
    hf_repo: "BAAI/bge-m4",
    model_scope_repo: Some("Xorbits/bge-m4"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmRNext,
    tensor_rules: TensorNamingRule::XlmRNext,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_SMALL_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::E5_Small,
    aliases: &["e5-small"],
    hf_repo: "intfloat/e5-small",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::E5_Base,
    aliases: &["e5-base"],
    hf_repo: "intfloat/e5-base",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const E5_LARGE_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::E5_Large,
    aliases: &["e5-large"],
    hf_repo: "intfloat/e5-large",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const M3E_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::M3e_Base,
    aliases: &["m3e-base"],
    hf_repo: "moka-ai/m3e-base",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V2_BASE_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::JinaEmbeddingsV2_Base,
    aliases: &["jina-embeddings-v2-base"],
    hf_repo: "jinaai/jina-embeddings-v2-base-en",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V2_SMALL_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::JinaEmbeddingsV2_Small,
    aliases: &["jina-embeddings-v2-small"],
    hf_repo: "jinaai/jina-embeddings-v2-small-en",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const JINA_EMBEDDINGS_V4_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::JinaEmbeddingsV4,
    aliases: &["jina-embeddings-v4"],
    hf_repo: "jinaai/jina-embeddings-v4",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const QWEN3_RERANK_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Qwen3_Rerank,
    aliases: &["qwen3-rerank"],
    hf_repo: "Qwen/Qwen3-Reranker",
    model_scope_repo: Some("qwen/Qwen3-Reranker"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::Qwen3,
    tensor_rules: TensorNamingRule::Qwen3,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_RERANK_V3_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Bge_Rerank_V3,
    aliases: &["bge-rerank-v3"],
    hf_repo: "BAAI/bge-reranker-v3",
    model_scope_repo: Some("Xorbits/bge-reranker-v3"),
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmRNext,
    tensor_rules: TensorNamingRule::XlmRNext,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const BGE_RERANK_V2_M3_MANIFEST: ModelManifest = ModelManifest {
    model_id: KnownModel::Bge_Rerank_V2_M3,
    aliases: &["bge-reranker-v2-m3"],
    hf_repo: "BAAI/bge-reranker-v2-m3",
    model_scope_repo: None,
    hf_file_map: EMPTY_FILE_MAP,
    arch: ModelArchitecture::XlmR,
    tensor_rules: TensorNamingRule::XlmR,
    rope_base_override: None,
    max_context_override: None,
    moe_config: None,
};

pub const ALL_MANIFESTS: &[&ModelManifest] = &[
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
    &GPT_OSS_1_5B_MANIFEST,
    &GPT_OSS_12B_MANIFEST,
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
