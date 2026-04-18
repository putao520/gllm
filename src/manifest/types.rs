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
        crate::arch::register_builtin_templates();
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

/// Model-kind-specific template override table.
///
/// When a GGUF architecture token resolves to a generative template but the
/// `ModelKind` requires a different specialised template, we reroute to the
/// matching template here. This is necessary because GGUF stores the same
/// `general.architecture` value for generative, embedding, and reranker
/// variants (e.g. all Qwen3 variants report `general.architecture = qwen3`).
///
/// Entry format: (base_template, model_kind, override_template)
static KIND_TEMPLATE_OVERRIDE: &[(&str, ModelKind, &str)] = &[
    ("qwen3", ModelKind::Embedding, "qwen3-embed"),
    ("qwen3", ModelKind::Reranker, "qwen3-reranker"),
    // XLM-R / BERT / RoBERTa 架构的 reranker (如 bge-reranker-v2-m3) 在
    // encoder 之后附加 RobertaClassificationHead (dense + tanh + out_proj)，
    // 使用独立模板 xlmr-reranker 表达完整图，不能用 encoder-only 的 xlmr。
    ("xlmr", ModelKind::Reranker, "xlmr-reranker"),
];

/// Map architecture token string to template name, respecting model kind.
///
/// For specialised `ModelKind` variants (Embedding, Reranker), applies
/// kind-specific template overrides so that (for example) `qwen3` with
/// `ModelKind::Embedding` → `qwen3-embed` instead of the generative `qwen3`
/// template, and `qwen3` with `ModelKind::Reranker` → `qwen3-reranker`.
pub fn map_architecture_token_for_kind(token: &str, kind: ModelKind) -> Option<String> {
    crate::arch::register_builtin_templates();
    let base = crate::arch::resolve_template_name(token)?.to_string();
    if let Some(&(_, _, override_template)) = KIND_TEMPLATE_OVERRIDE
        .iter()
        .find(|&&(t, k, _)| t == base && k == kind)
    {
        // Only override if the specialised template is registered.
        if crate::arch::is_valid_template(override_template) {
            return Some(override_template.to_string());
        }
    }
    Some(base)
}

/// 按 ModelKind 对已解析的 template name 做 override (SafeTensors/ONNX 路径)。
///
/// `map_architecture_token_for_kind` 从 HF token 解析，适用于 GGUF;
/// 本函数从**已确定的模板名**做 kind override，适用于 SafeTensors/ONNX/PyTorch
/// 路径 (loader.detect_architecture 返回 template name 时)。
///
/// 返回 None 表示无需 override，调用方应保持原模板名。
pub fn map_kind_template(template_name: &str, kind: ModelKind) -> Option<String> {
    crate::arch::register_builtin_templates();
    let (_, _, override_template) = KIND_TEMPLATE_OVERRIDE
        .iter()
        .find(|&&(t, k, _)| t == template_name && k == kind)?;
    if crate::arch::is_valid_template(override_template) {
        Some(override_template.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_architecture_token_delegates_to_registry() {
        // SSOT 测试:不硬编码 token 映射。从 YAML 注册表读取每个模板的
        // `name` / extra_aliases,验证 map_architecture_token 能正确反查。
        crate::arch::register_builtin_templates();
        let registry = crate::arch::ARCH_REGISTRY.get().unwrap();
        for (name, template) in &registry.templates {
            assert_eq!(map_architecture_token(name).as_deref(), Some(name.as_str()),
                "template name '{name}' 必须自反查");
            assert_eq!(
                map_architecture_token(&format!("{name}ForCausalLM")).as_deref(),
                Some(name.as_str()),
                "自动派生 token '{name}ForCausalLM' 必须反查到 '{name}'"
            );
            for alias in &template.extra_aliases {
                assert_eq!(map_architecture_token(alias).as_deref(), Some(name.as_str()),
                    "extra_alias '{alias}' (YAML {}) 必须反查到 '{name}'", name);
            }
        }
        assert_eq!(map_architecture_token("custom-token-that-never-exists"), None);
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
