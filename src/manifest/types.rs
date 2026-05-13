//! Layer 1: Manifest types (SSOT).

use std::borrow::Cow;
use std::collections::HashMap;

/// Role of a tensor in the model architecture.
///
/// Used by `match_tensor_role()` for 100% precise segment-sequence matching:
/// every variant maps to an exact suffix pattern (no `contains()` heuristics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    // ── Global (layer_idx == None) ──
    Embedding,
    OutputHead,
    FinalNorm,
    ClassifierDense,
    ClassifierOutProj,
    PatchEmbed,
    PositionEmbedding,

    // ── Per-layer attention ──
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionFusedQkv,
    AttentionOutput,
    AttentionQNorm,
    AttentionKNorm,
    AttentionSinks,

    // ── Per-layer norm ──
    InputNorm,
    PostAttnNorm,
    LayerNorm, // legacy — kept for backward compat

    // ── Per-layer FFN ──
    FfnGate,
    FfnUp,
    FfnDown,

    // ── MoE ──
    MoEGate,         // router: mlp.gate / ffn_gate_inp
    MoESharedExpert, // shared_experts.*
    MoEExpert,       // experts.*.*

    // ── Audio/Vision special ──
    DepthwiseConv,

    // ── Misc ──
    Rope,
}

impl TensorRole {
    /// Map this role to its canonical name, optionally prefixed with layer index.
    pub fn to_canonical_name(&self, layer: Option<usize>) -> String {
        let base = match self {
            Self::Embedding => "embed",
            Self::OutputHead => "lm_head",
            Self::FinalNorm => "final_norm",
            Self::ClassifierDense => "classifier.dense",
            Self::ClassifierOutProj => "classifier",
            Self::PatchEmbed => "patch_embed",
            Self::PositionEmbedding => "position_embed",
            Self::AttentionFusedQkv => "qkv_proj",
            Self::AttentionQuery => "q_proj",
            Self::AttentionKey => "k_proj",
            Self::AttentionValue => "v_proj",
            Self::AttentionOutput => "o_proj",
            Self::AttentionQNorm => "q_norm",
            Self::AttentionKNorm => "k_norm",
            Self::AttentionSinks => "attn_sinks",
            Self::InputNorm => "input_norm",
            Self::PostAttnNorm => "post_attn_norm",
            Self::LayerNorm => "input_norm",
            Self::FfnGate => "gate_proj",
            Self::FfnUp => "up_proj",
            Self::FfnDown => "down_proj",
            Self::MoEGate => "moe_gate",
            Self::MoESharedExpert => "shared_expert",
            Self::MoEExpert => "expert",
            Self::DepthwiseConv => "depthwise_conv",
            Self::Rope => "rope",
        };
        match layer {
            Some(l) => format!("L{}.{base}", l),
            None => base.to_string(),
        }
    }
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
    Classifier,
}

impl ModelKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat" | "generation" | "generator" | "text-generation" => Some(Self::Chat),
            "embedding" | "embeddings" | "embed" => Some(Self::Embedding),
            "rerank" | "reranker" | "re-ranker" | "re-rank" => Some(Self::Reranker),
            "classifier" | "classification" | "classify" | "sequence-classification"
            | "text-classification" => Some(Self::Classifier),
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
    GptOss,
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
    /// 获取架构族 (从别名表查询)
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

/// Map architecture token string to canonical name.
pub fn map_architecture_token(token: &str) -> Option<String> {
    crate::arch::resolve_template_name(token).map(|s| s.to_string())
}

/// Map architecture token string to canonical name, respecting model kind.
pub fn map_architecture_token_for_kind(token: &str, _kind: ModelKind) -> Option<String> {
    // auto_graph derives everything from tensor names at runtime.
    // The canonical name is still useful for model identification.
    crate::arch::resolve_template_name(token).map(|s| s.to_string())
}

/// Map template name respecting model kind (no-op for auto_graph).
pub fn map_kind_template(template_name: &str, _kind: ModelKind) -> Option<String> {
    Some(template_name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_architecture_token_delegates_to_registry() {
        assert_eq!(map_architecture_token("qwen3"), Some("qwen3".to_string()));
        assert_eq!(map_architecture_token("LlamaForCausalLM"), Some("llama".to_string()));
        assert_eq!(map_architecture_token("xlmr"), Some("xlmr".to_string()));
        assert_eq!(map_architecture_token("custom-token-that-never-exists"), None);
    }

    #[test]
    fn manifest_family_from_registry() {
        let mut m = ModelManifest::default();
        assert_eq!(m.family(), ArchFamily::Decoder);
        m.arch = "xlmr".to_string();
        assert_eq!(m.family(), ArchFamily::Encoder);
    }
}
