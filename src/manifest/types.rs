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

/// 架构族：决定权重命名约定和推理路径（真正的类型约束，不随配置扩展）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchFamily {
    /// BERT/XLM-R 编码器族：绝对位置编码，双向注意力
    Encoder,
    /// LLaMA/Qwen/GPT 解码器族：RoPE，因果注意力
    Decoder,
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
    DeepSeek,
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

    // Architecture — 模板名字符串，运行时由注册表校验。
    // 值域由 `src/arch/templates/*.yaml` 文件名决定，不硬编码。
    pub arch: String,
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
    /// 获取架构族 (从注册表查询)
    pub fn family(&self) -> ArchFamily {
        crate::arch::resolve_family(&self.arch).unwrap_or(ArchFamily::Decoder)
    }

    pub fn is_moe(&self) -> bool {
        self.moe_config.is_some()
    }
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            model_id: Cow::Borrowed("default"),
            file_map: EMPTY_FILE_MAP,
            arch: "llama".to_string(),
            kind: ModelKind::Chat,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        }
    }
}

/// Map architecture token string to template name.
///
/// 委托给 `arch::registry` 的别名查找，返回模板名字符串。
pub fn map_architecture_token(token: &str) -> Option<String> {
    crate::arch::register_builtin_templates();
    crate::arch::resolve_template_name(token).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_architecture_token_delegates_to_registry() {
        assert_eq!(map_architecture_token("LlamaForCausalLM").as_deref(), Some("llama"));
        assert_eq!(map_architecture_token("Qwen2ForCausalLM").as_deref(), Some("qwen3"));
        assert_eq!(map_architecture_token("MistralForCausalLM").as_deref(), Some("mistral3"));
        assert_eq!(map_architecture_token("Gemma2ForCausalLM").as_deref(), Some("gemma2"));
        assert_eq!(map_architecture_token("DeepseekV3ForCausalLM").as_deref(), Some("deepseek"));
        assert_eq!(map_architecture_token("GPTOSSForCausalLM").as_deref(), Some("gpt2next"));
        assert_eq!(map_architecture_token("custom-llama-adapter"), None);
    }

    #[test]
    fn manifest_family_from_registry() {
        crate::arch::register_builtin_templates();
        let mut m = ModelManifest::default();
        assert_eq!(m.family(), ArchFamily::Decoder);
        m.arch = "xlmr".to_string();
        assert_eq!(m.family(), ArchFamily::Encoder);
    }
}
