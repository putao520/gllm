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
    TokenTypeEmbedding, // BERT: token_type_embeddings [type_vocab_size, H]
    EmbedNorm,          // BERT: embeddings.LayerNorm (pre-encoder norm)

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

    // ── MLA (Multi-head Latent Attention) ──
    MlaQCompress,    // W_QA — query 低秩压缩 (q_a_proj, DeepSeek V3)
    MlaQExpand,      // W_QB — query 低秩还原 (q_b_proj, DeepSeek V3)
    MlaKvCompress,   // W_DKV — KV 低秩压缩权重
    MlaKeyAbsorb,    // W_UK  — K 吸收权重 (Matrix Absorption)
    MlaValueAbsorb,  // W_UV  — V 还原权重
    MlaRopeKey,      // W_KR  — 解耦 RoPE key 权重

    // ── MTP (Multi-Token Prediction) ──
    MtpProjection,   // mtp_proj / mtp_head / model.mtp.* projection weights

    // ── Sandwich Norms (Gemma 4) ──
    PostAttentionSandwichNorm, // post_attention_norm — norm after attention, before residual
    PostFfwSandwichNorm,       // post_ffw_norm — norm after FFN, before residual
    PostLayerNorm,             // post_norm — norm after AltUp/PLE correction

    // ── PLE / AltUp (Gemma 4 E2B/E4B) ──
    PerLayerEmbedding,         // per_layer_token_embd
    PerLayerGate,              // inp_gate
    PerLayerProj,              // proj
    PerLayerModelProj,         // per_layer_model_proj
    PerLayerProjNorm,          // per_layer_proj_norm
    LayerOutputScale,          // layer_output_scale
    OutputScale,               // output_scale (global)

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
            Self::TokenTypeEmbedding => "token_type_embed",
            Self::EmbedNorm => "embed_norm",
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
            Self::MlaQCompress => "q_a_proj",
            Self::MlaQExpand => "q_b_proj",
            Self::MlaKvCompress => "kv_b_proj",
            Self::MlaKeyAbsorb => "k_b_proj",
            Self::MlaValueAbsorb => "v_b_proj",
            Self::MlaRopeKey => "k_pe_proj",
            Self::MtpProjection => "mtp_proj",
            Self::PostAttentionSandwichNorm => "post_attention_norm",
            Self::PostFfwSandwichNorm => "post_ffw_norm",
            Self::PostLayerNorm => "post_norm",
            Self::PerLayerEmbedding => "per_layer_token_embd",
            Self::PerLayerGate => "inp_gate",
            Self::PerLayerProj => "proj",
            Self::PerLayerModelProj => "per_layer_model_proj",
            Self::PerLayerProjNorm => "per_layer_proj_norm",
            Self::LayerOutputScale => "layer_output_scale",
            Self::OutputScale => "output_scale",
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

    // Architecture — canonical name 字符串，运行时由注册表校验。
    // 值域由 ARCH_TABLE 中的 canonical name 列决定，不硬编码。
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

/// Map canonical name respecting model kind (no-op for auto_graph).
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

    #[test]
    fn model_kind_parse_variants() {
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("Generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("EMBEDDING"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("text-classification"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("unknown"), None);
    }

    #[test]
    fn model_kind_from_str() {
        assert!("chat".parse::<ModelKind>().is_ok());
        assert!("invalid".parse::<ModelKind>().is_err());
    }

    #[test]
    fn tensor_role_canonical_names() {
        assert_eq!(TensorRole::Embedding.to_canonical_name(None), "embed");
        assert_eq!(TensorRole::AttentionQuery.to_canonical_name(Some(3)), "L3.q_proj");
        assert_eq!(TensorRole::FinalNorm.to_canonical_name(None), "final_norm");
        assert_eq!(TensorRole::MlaQCompress.to_canonical_name(Some(0)), "L0.q_a_proj");
        assert_eq!(TensorRole::MtpProjection.to_canonical_name(Some(5)), "L5.mtp_proj");
    }

    #[test]
    fn manifest_default_values() {
        let m = ModelManifest::default();
        assert_eq!(&*m.model_id, "default");
        assert_eq!(m.arch, "llama");
        assert_eq!(m.kind, ModelKind::Chat);
        assert!(m.rope_base_override.is_none());
        assert!(m.max_context_override.is_none());
        assert!(!m.is_moe());
    }

    #[test]
    fn manifest_is_moe() {
        let mut m = ModelManifest::default();
        assert!(!m.is_moe());
        m.moe_config = Some(MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        });
        assert!(m.is_moe());
    }

    // ── Additional ModelKind::parse edge cases ──

    #[test]
    fn model_kind_parse_aliases() {
        assert_eq!(ModelKind::parse("generator"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("text-generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("embed"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("re-ranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-rank"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("classify"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("sequence-classification"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_whitespace() {
        assert_eq!(ModelKind::parse("  chat  "), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse(" embedding "), Some(ModelKind::Embedding));
    }

    #[test]
    fn model_kind_parse_empty() {
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("  "), None);
    }

    // ── RouterType variants ──

    #[test]
    fn router_type_equality() {
        assert_eq!(RouterType::Qwen, RouterType::Qwen);
        assert_eq!(RouterType::Unknown, RouterType::Unknown);
        assert_ne!(RouterType::Qwen, RouterType::Mixtral);
        assert_ne!(RouterType::DeepSeek, RouterType::GptOss);
    }

    #[test]
    fn router_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RouterType::Qwen);
        set.insert(RouterType::Mixtral);
        set.insert(RouterType::Qwen);
        assert_eq!(set.len(), 2);
    }

    // ── MoEConfig ──

    #[test]
    fn moe_config_fields() {
        let cfg = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::Mixtral,
        };
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.router_type, RouterType::Mixtral);
    }

    // ── TensorRole canonical names ──

    #[test]
    fn tensor_role_layer_canonical_names() {
        assert_eq!(TensorRole::AttentionKey.to_canonical_name(Some(0)), "L0.k_proj");
        assert_eq!(TensorRole::AttentionValue.to_canonical_name(Some(2)), "L2.v_proj");
        assert_eq!(TensorRole::AttentionOutput.to_canonical_name(Some(1)), "L1.o_proj");
        assert_eq!(TensorRole::InputNorm.to_canonical_name(Some(0)), "L0.input_norm");
        assert_eq!(TensorRole::PostAttnNorm.to_canonical_name(Some(0)), "L0.post_attn_norm");
        assert_eq!(TensorRole::FfnGate.to_canonical_name(Some(0)), "L0.gate_proj");
        assert_eq!(TensorRole::FfnUp.to_canonical_name(Some(0)), "L0.up_proj");
        assert_eq!(TensorRole::FfnDown.to_canonical_name(Some(0)), "L0.down_proj");
    }

    #[test]
    fn tensor_role_output_head_canonical() {
        assert_eq!(TensorRole::OutputHead.to_canonical_name(None), "lm_head");
    }

    // ── ArchFamily ──

    #[test]
    fn arch_family_variants() {
        assert_eq!(ArchFamily::Encoder, ArchFamily::Encoder);
        assert_eq!(ArchFamily::Decoder, ArchFamily::Decoder);
        assert_ne!(ArchFamily::Encoder, ArchFamily::Decoder);
    }

    // ── FileMap ──

    #[test]
    fn empty_file_map() {
        let fm: FileMap = EMPTY_FILE_MAP;
        assert!(fm.is_empty());
    }

    // ── Additional coverage ──

    #[test]
    fn tensor_role_debug() {
        let debug = format!("{:?}", TensorRole::Embedding);
        assert!(debug.contains("Embedding"));
        let debug = format!("{:?}", TensorRole::MoEGate);
        assert!(debug.contains("MoEGate"));
    }

    #[test]
    fn tensor_role_copy() {
        let a = TensorRole::FfnGate;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_role_mla_canonical_names() {
        assert_eq!(TensorRole::MlaQExpand.to_canonical_name(Some(1)), "L1.q_b_proj");
        assert_eq!(TensorRole::MlaKvCompress.to_canonical_name(Some(2)), "L2.kv_b_proj");
        assert_eq!(TensorRole::MlaKeyAbsorb.to_canonical_name(Some(3)), "L3.k_b_proj");
        assert_eq!(TensorRole::MlaValueAbsorb.to_canonical_name(Some(4)), "L4.v_b_proj");
        assert_eq!(TensorRole::MlaRopeKey.to_canonical_name(Some(5)), "L5.k_pe_proj");
    }

    #[test]
    fn tensor_role_moe_canonical_names() {
        assert_eq!(TensorRole::MoEGate.to_canonical_name(Some(0)), "L0.moe_gate");
        assert_eq!(TensorRole::MoESharedExpert.to_canonical_name(Some(1)), "L1.shared_expert");
        assert_eq!(TensorRole::MoEExpert.to_canonical_name(Some(2)), "L2.expert");
    }

    #[test]
    fn tensor_role_vision_audio_canonical() {
        assert_eq!(TensorRole::PatchEmbed.to_canonical_name(None), "patch_embed");
        assert_eq!(TensorRole::PositionEmbedding.to_canonical_name(None), "position_embed");
        assert_eq!(TensorRole::DepthwiseConv.to_canonical_name(Some(0)), "L0.depthwise_conv");
    }

    #[test]
    fn model_kind_debug() {
        assert!(format!("{:?}", ModelKind::Chat).contains("Chat"));
        assert!(format!("{:?}", ModelKind::Reranker).contains("Reranker"));
    }

    #[test]
    fn model_kind_copy() {
        let a = ModelKind::Embedding;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_debug() {
        assert!(format!("{:?}", ArchFamily::Encoder).contains("Encoder"));
        assert!(format!("{:?}", ArchFamily::Decoder).contains("Decoder"));
    }

    #[test]
    fn manifest_clone() {
        let m = ModelManifest::default();
        let cloned = m.clone();
        assert_eq!(cloned.arch, "llama");
        assert_eq!(cloned.kind, ModelKind::Chat);
    }

    #[test]
    fn manifest_debug() {
        let m = ModelManifest::default();
        let debug = format!("{m:?}");
        assert!(debug.contains("model_id"));
        assert!(debug.contains("arch"));
    }

    #[test]
    fn moe_config_copy() {
        let cfg = MoEConfig {
            num_experts: 8,
            num_experts_per_tok: 2,
            router_type: RouterType::Qwen,
        };
        let copied = cfg;
        assert_eq!(copied.num_experts, 8);
    }

    #[test]
    fn map_architecture_token_for_kind_delegates() {
        assert_eq!(
            map_architecture_token_for_kind("qwen3", ModelKind::Chat),
            Some("qwen3".to_string())
        );
    }

    #[test]
    fn map_kind_template_passthrough() {
        assert_eq!(
            map_kind_template("llama", ModelKind::Chat),
            Some("llama".to_string())
        );
    }

    // ── TensorRole trait coverage ──

    #[test]
    fn tensor_role_clone() {
        let a = TensorRole::AttentionFusedQkv;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_role_hash_set_dedup() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TensorRole::Embedding);
        set.insert(TensorRole::Embedding);
        set.insert(TensorRole::OutputHead);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn tensor_role_all_variants_no_layer() {
        // Every variant should produce a non-empty canonical name without layer
        let roles = [
            TensorRole::Embedding,
            TensorRole::OutputHead,
            TensorRole::FinalNorm,
            TensorRole::ClassifierDense,
            TensorRole::ClassifierOutProj,
            TensorRole::PatchEmbed,
            TensorRole::PositionEmbedding,
            TensorRole::AttentionQuery,
            TensorRole::AttentionKey,
            TensorRole::AttentionValue,
            TensorRole::AttentionFusedQkv,
            TensorRole::AttentionOutput,
            TensorRole::AttentionQNorm,
            TensorRole::AttentionKNorm,
            TensorRole::AttentionSinks,
            TensorRole::InputNorm,
            TensorRole::PostAttnNorm,
            TensorRole::LayerNorm,
            TensorRole::FfnGate,
            TensorRole::FfnUp,
            TensorRole::FfnDown,
            TensorRole::MoEGate,
            TensorRole::MoESharedExpert,
            TensorRole::MoEExpert,
            TensorRole::DepthwiseConv,
            TensorRole::MlaQCompress,
            TensorRole::MlaQExpand,
            TensorRole::MlaKvCompress,
            TensorRole::MlaKeyAbsorb,
            TensorRole::MlaValueAbsorb,
            TensorRole::MlaRopeKey,
            TensorRole::MtpProjection,
            TensorRole::Rope,
        ];
        for role in &roles {
            let name = role.to_canonical_name(None);
            assert!(!name.is_empty(), "TensorRole {:?} produced empty name", role);
        }
    }

    #[test]
    fn tensor_role_layer_zero_vs_none() {
        // Layer 0 should be prefixed; None should not
        let with_layer = TensorRole::AttentionQuery.to_canonical_name(Some(0));
        let no_layer = TensorRole::AttentionQuery.to_canonical_name(None);
        assert_eq!(with_layer, "L0.q_proj");
        assert_eq!(no_layer, "q_proj");
        assert_ne!(with_layer, no_layer);
    }

    #[test]
    fn tensor_role_layer_norm_alias() {
        // LayerNorm maps to same canonical as InputNorm
        assert_eq!(
            TensorRole::LayerNorm.to_canonical_name(Some(0)),
            TensorRole::InputNorm.to_canonical_name(Some(0))
        );
        assert_eq!(
            TensorRole::LayerNorm.to_canonical_name(None),
            TensorRole::InputNorm.to_canonical_name(None)
        );
    }

    #[test]
    fn tensor_role_attention_sinks_canonical() {
        assert_eq!(TensorRole::AttentionSinks.to_canonical_name(Some(0)), "L0.attn_sinks");
        assert_eq!(TensorRole::AttentionSinks.to_canonical_name(None), "attn_sinks");
    }

    #[test]
    fn tensor_role_rope_canonical() {
        assert_eq!(TensorRole::Rope.to_canonical_name(None), "rope");
        assert_eq!(TensorRole::Rope.to_canonical_name(Some(3)), "L3.rope");
    }

    #[test]
    fn tensor_role_classifier_canonical() {
        assert_eq!(TensorRole::ClassifierDense.to_canonical_name(None), "classifier.dense");
        assert_eq!(TensorRole::ClassifierOutProj.to_canonical_name(None), "classifier");
    }

    #[test]
    fn tensor_role_qnorm_knorm_canonical() {
        assert_eq!(TensorRole::AttentionQNorm.to_canonical_name(Some(0)), "L0.q_norm");
        assert_eq!(TensorRole::AttentionKNorm.to_canonical_name(Some(0)), "L0.k_norm");
    }

    // ── ArchFamily trait coverage ──

    #[test]
    fn arch_family_clone() {
        let a = ArchFamily::Encoder;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_copy() {
        let a = ArchFamily::Decoder;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ArchFamily::Encoder);
        set.insert(ArchFamily::Decoder);
        set.insert(ArchFamily::Encoder);
        assert_eq!(set.len(), 2);
    }

    // ── ModelKind trait coverage ──

    #[test]
    fn model_kind_clone() {
        let a = ModelKind::Chat;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn model_kind_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelKind::Chat);
        set.insert(ModelKind::Chat);
        set.insert(ModelKind::Embedding);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn model_kind_from_str_ok() {
        assert_eq!("chat".parse::<ModelKind>(), Ok(ModelKind::Chat));
        assert_eq!("embedding".parse::<ModelKind>(), Ok(ModelKind::Embedding));
        assert_eq!("rerank".parse::<ModelKind>(), Ok(ModelKind::Reranker));
        assert_eq!("classifier".parse::<ModelKind>(), Ok(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_from_str_err() {
        assert_eq!("".parse::<ModelKind>(), Err(()));
        assert_eq!("foo".parse::<ModelKind>(), Err(()));
        assert_eq!("123".parse::<ModelKind>(), Err(()));
    }

    // ── RouterType trait coverage ──

    #[test]
    fn router_type_clone() {
        let a = RouterType::DeepSeek;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn router_type_copy() {
        let a = RouterType::GptOss;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn router_type_debug() {
        assert!(format!("{:?}", RouterType::Qwen).contains("Qwen"));
        assert!(format!("{:?}", RouterType::Mixtral).contains("Mixtral"));
        assert!(format!("{:?}", RouterType::DeepSeek).contains("DeepSeek"));
        assert!(format!("{:?}", RouterType::GptOss).contains("GptOss"));
        assert!(format!("{:?}", RouterType::Unknown).contains("Unknown"));
    }

    #[test]
    fn router_type_all_variants_distinct() {
        let variants = [
            RouterType::Qwen,
            RouterType::Mixtral,
            RouterType::DeepSeek,
            RouterType::GptOss,
            RouterType::Unknown,
        ];
        use std::collections::HashSet;
        let set: HashSet<_> = variants.iter().collect();
        assert_eq!(set.len(), variants.len());
    }

    // ── MoEConfig trait + edge cases ──

    #[test]
    fn moe_config_clone() {
        let cfg = MoEConfig {
            num_experts: 16,
            num_experts_per_tok: 4,
            router_type: RouterType::Qwen,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.num_experts, 16);
        assert_eq!(cloned.num_experts_per_tok, 4);
        assert_eq!(cloned.router_type, RouterType::Qwen);
    }

    #[test]
    fn moe_config_debug() {
        let cfg = MoEConfig {
            num_experts: 8,
            num_experts_per_tok: 2,
            router_type: RouterType::Mixtral,
        };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("num_experts"));
        assert!(debug.contains("router_type"));
    }

    #[test]
    fn moe_config_hash_and_eq() {
        use std::collections::HashSet;
        let a = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        let b = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn moe_config_zero_experts() {
        let cfg = MoEConfig {
            num_experts: 0,
            num_experts_per_tok: 0,
            router_type: RouterType::Unknown,
        };
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.num_experts_per_tok, 0);
    }

    #[test]
    fn moe_config_experts_per_tok_exceeds_total() {
        // No validation constraint in struct; just verify field storage
        let cfg = MoEConfig {
            num_experts: 4,
            num_experts_per_tok: 8,
            router_type: RouterType::Mixtral,
        };
        assert_eq!(cfg.num_experts, 4);
        assert_eq!(cfg.num_experts_per_tok, 8);
    }

    #[test]
    fn moe_config_max_values() {
        let cfg = MoEConfig {
            num_experts: usize::MAX,
            num_experts_per_tok: usize::MAX,
            router_type: RouterType::GptOss,
        };
        assert_eq!(cfg.num_experts, usize::MAX);
        assert_eq!(cfg.num_experts_per_tok, usize::MAX);
    }

    // ── ModelManifest construction & field access ──

    #[test]
    fn manifest_construct_borrowed_cow() {
        let m = ModelManifest {
            model_id: Cow::Borrowed("test-model"),
            file_map: EMPTY_FILE_MAP,
            arch: "qwen3".to_string(),
            kind: ModelKind::Embedding,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        };
        assert_eq!(&*m.model_id, "test-model");
        assert_eq!(m.arch, "qwen3");
        assert_eq!(m.kind, ModelKind::Embedding);
    }

    #[test]
    fn manifest_construct_owned_cow() {
        let dynamic_id = format!("dynamic-{}", 42);
        let m = ModelManifest {
            model_id: Cow::Owned(dynamic_id),
            file_map: EMPTY_FILE_MAP,
            arch: "llama".to_string(),
            kind: ModelKind::Chat,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        };
        assert_eq!(&*m.model_id, "dynamic-42");
    }

    #[test]
    fn manifest_with_overrides() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(500000.0);
        m.max_context_override = Some(32768);
        assert_eq!(m.rope_base_override, Some(500000.0));
        assert_eq!(m.max_context_override, Some(32768));
    }

    #[test]
    fn manifest_tensor_map_operations() {
        let mut m = ModelManifest::default();
        assert!(m.tensor_map.is_empty());
        m.tensor_map.insert(TensorRole::Embedding, "model.embed_tokens".to_string());
        m.tensor_map.insert(TensorRole::OutputHead, "lm_head".to_string());
        assert_eq!(m.tensor_map.len(), 2);
        assert_eq!(
            m.tensor_map.get(&TensorRole::Embedding),
            Some(&"model.embed_tokens".to_string())
        );
        assert!(m.tensor_map.contains_key(&TensorRole::OutputHead));
        assert!(!m.tensor_map.contains_key(&TensorRole::Rope));
    }

    #[test]
    fn manifest_file_map_nonempty() {
        let fm: FileMap = &[("model.safetensors", "model-00001-of-00002.safetensors")];
        let m = ModelManifest {
            model_id: Cow::Borrowed("test"),
            file_map: fm,
            arch: "llama".to_string(),
            kind: ModelKind::Chat,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        };
        assert_eq!(m.file_map.len(), 1);
        assert_eq!(m.file_map[0].0, "model.safetensors");
        assert_eq!(m.file_map[0].1, "model-00001-of-00002.safetensors");
    }

    #[test]
    fn manifest_is_moe_false_when_none() {
        let m = ModelManifest {
            moe_config: None,
            ..ModelManifest::default()
        };
        assert!(!m.is_moe());
    }

    #[test]
    fn manifest_is_moe_true_when_some() {
        let m = ModelManifest {
            moe_config: Some(MoEConfig {
                num_experts: 1,
                num_experts_per_tok: 1,
                router_type: RouterType::Unknown,
            }),
            ..ModelManifest::default()
        };
        assert!(m.is_moe());
    }

    #[test]
    fn manifest_family_unknown_arch_falls_back_to_decoder() {
        let m = ModelManifest {
            arch: "totally-unknown-arch-xyz".to_string(),
            ..ModelManifest::default()
        };
        // Unknown arch resolves to Decoder as fallback
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    #[test]
    fn manifest_clone_independence() {
        let mut original = ModelManifest::default();
        original.tensor_map.insert(TensorRole::Embedding, "embed".to_string());
        let cloned = original.clone();
        original.tensor_map.insert(TensorRole::Rope, "rope".to_string());
        // Clone is independent (HashMap clone is deep)
        assert_eq!(cloned.tensor_map.len(), 1);
        assert_eq!(original.tensor_map.len(), 2);
    }

    // ── map_architecture_token_for_kind edge cases ──

    #[test]
    fn map_architecture_token_for_kind_unknown_returns_none() {
        assert_eq!(
            map_architecture_token_for_kind("nonexistent-arch-xyz", ModelKind::Chat),
            None
        );
    }

    #[test]
    fn map_architecture_token_for_kind_with_all_kinds() {
        // All ModelKind values produce the same result (kind is unused)
        for kind in [ModelKind::Chat, ModelKind::Embedding, ModelKind::Reranker, ModelKind::Classifier] {
            assert_eq!(
                map_architecture_token_for_kind("qwen3", kind),
                Some("qwen3".to_string())
            );
        }
    }

    // ── map_kind_template edge cases ──

    #[test]
    fn map_kind_template_empty_string() {
        assert_eq!(
            map_kind_template("", ModelKind::Chat),
            Some(String::new())
        );
    }

    #[test]
    fn map_kind_template_preserves_input() {
        assert_eq!(
            map_kind_template("deepseek-v3", ModelKind::Embedding),
            Some("deepseek-v3".to_string())
        );
    }

    // ── ModelKind parse: every valid alias ──

    #[test]
    fn model_kind_parse_all_chat_aliases() {
        for alias in &["chat", "generation", "generator", "text-generation"] {
            assert_eq!(
                ModelKind::parse(alias),
                Some(ModelKind::Chat),
                "alias {:?} should parse as Chat",
                alias
            );
        }
    }

    #[test]
    fn model_kind_parse_all_embedding_aliases() {
        for alias in &["embedding", "embeddings", "embed"] {
            assert_eq!(
                ModelKind::parse(alias),
                Some(ModelKind::Embedding),
                "alias {:?} should parse as Embedding",
                alias
            );
        }
    }

    #[test]
    fn model_kind_parse_all_reranker_aliases() {
        for alias in &["rerank", "reranker", "re-ranker", "re-rank"] {
            assert_eq!(
                ModelKind::parse(alias),
                Some(ModelKind::Reranker),
                "alias {:?} should parse as Reranker",
                alias
            );
        }
    }

    #[test]
    fn model_kind_parse_all_classifier_aliases() {
        for alias in &["classifier", "classification", "classify", "sequence-classification", "text-classification"] {
            assert_eq!(
                ModelKind::parse(alias),
                Some(ModelKind::Classifier),
                "alias {:?} should parse as Classifier",
                alias
            );
        }
    }

    // ── TensorRole canonical names with large layer indices ──

    #[test]
    fn tensor_role_canonical_name_large_layer() {
        assert_eq!(TensorRole::FfnGate.to_canonical_name(Some(999)), "L999.gate_proj");
        assert_eq!(TensorRole::Embedding.to_canonical_name(Some(usize::MAX)), format!("L{}.embed", usize::MAX));
    }

    #[test]
    fn tensor_role_canonical_name_all_with_layer_zero() {
        let roles_with_expected = [
            (TensorRole::Embedding, "L0.embed"),
            (TensorRole::OutputHead, "L0.lm_head"),
            (TensorRole::FinalNorm, "L0.final_norm"),
            (TensorRole::ClassifierDense, "L0.classifier.dense"),
            (TensorRole::ClassifierOutProj, "L0.classifier"),
            (TensorRole::Rope, "L0.rope"),
        ];
        for (role, expected) in &roles_with_expected {
            assert_eq!(role.to_canonical_name(Some(0)), *expected);
        }
    }

    // ── TensorRole Hash + equality in HashMap key scenario ──

    #[test]
    fn tensor_role_as_hashmap_key() {
        let mut map = HashMap::new();
        map.insert(TensorRole::AttentionQuery, "q_weight".to_string());
        map.insert(TensorRole::AttentionKey, "k_weight".to_string());
        assert_eq!(map.get(&TensorRole::AttentionQuery), Some(&"q_weight".to_string()));
        assert_eq!(map.get(&TensorRole::AttentionKey), Some(&"k_weight".to_string()));
        assert_eq!(map.get(&TensorRole::AttentionValue), None);
    }

    // ── TensorRole inequality between all major categories ──

    #[test]
    fn tensor_role_distinct_variants_are_not_equal() {
        assert_ne!(TensorRole::Embedding, TensorRole::OutputHead);
        assert_ne!(TensorRole::AttentionQuery, TensorRole::AttentionKey);
        assert_ne!(TensorRole::FfnGate, TensorRole::FfnUp);
        assert_ne!(TensorRole::MoEGate, TensorRole::MoEExpert);
        assert_ne!(TensorRole::MlaQCompress, TensorRole::MlaKvCompress);
        assert_ne!(TensorRole::InputNorm, TensorRole::PostAttnNorm);
    }

    // ── TensorRole: canonical names are unique across all global roles ──

    #[test]
    fn tensor_role_global_canonical_names_are_distinct() {
        let global_roles = [
            TensorRole::Embedding,
            TensorRole::OutputHead,
            TensorRole::FinalNorm,
            TensorRole::ClassifierDense,
            TensorRole::ClassifierOutProj,
            TensorRole::PatchEmbed,
            TensorRole::PositionEmbedding,
            TensorRole::Rope,
        ];
        let names: Vec<String> = global_roles.iter().map(|r| r.to_canonical_name(None)).collect();
        let unique: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
        assert_eq!(unique.len(), global_roles.len(), "Global TensorRole canonical names must be unique");
    }

    // ── ModelKind parse: mixed case and unicode-ish ──

    #[test]
    fn model_kind_parse_mixed_case() {
        assert_eq!(ModelKind::parse("Chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("CHAT"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("Reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("RERANKER"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("Embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("EMBEDDING"), Some(ModelKind::Embedding));
    }

    #[test]
    fn model_kind_parse_non_ascii_returns_none() {
        // Strings that don't match any alias return None
        assert_eq!(ModelKind::parse("chàt"), None);
        assert_eq!(ModelKind::parse("émbédding"), None);
    }

    // ── ModelKind FromStr: type annotation forms ──

    #[test]
    fn model_kind_from_str_generation_alias() {
        let result: Result<ModelKind, ()> = "generation".parse();
        assert_eq!(result, Ok(ModelKind::Chat));
    }

    #[test]
    fn model_kind_from_str_rerank_alias() {
        let result: Result<ModelKind, ()> = "rerank".parse();
        assert_eq!(result, Ok(ModelKind::Reranker));
    }

    // ── ModelKind: test all 4 variants are distinct ──

    #[test]
    fn model_kind_all_variants_distinct() {
        use std::collections::HashSet;
        let variants = [ModelKind::Chat, ModelKind::Embedding, ModelKind::Reranker, ModelKind::Classifier];
        let set: HashSet<_> = variants.iter().collect();
        assert_eq!(set.len(), variants.len());
    }

    // ── RouterType: ensure Unknown is different from all named types ──

    #[test]
    fn router_type_unknown_differs_from_all_named() {
        assert_ne!(RouterType::Unknown, RouterType::Qwen);
        assert_ne!(RouterType::Unknown, RouterType::Mixtral);
        assert_ne!(RouterType::Unknown, RouterType::DeepSeek);
        assert_ne!(RouterType::Unknown, RouterType::GptOss);
    }

    // ── RouterType Hash dedup ──

    #[test]
    fn router_type_hash_set_dedup_full() {
        use std::collections::HashSet;
        let all = [RouterType::Qwen, RouterType::Mixtral, RouterType::DeepSeek, RouterType::GptOss, RouterType::Unknown];
        let mut set = HashSet::new();
        for r in &all {
            set.insert(*r);
            set.insert(*r); // insert twice
        }
        assert_eq!(set.len(), all.len());
    }

    // ── MoEConfig: field equality is structural ──

    #[test]
    fn moe_config_equality_same_fields() {
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        assert_eq!(a, b);
    }

    #[test]
    fn moe_config_inequality_different_num_experts() {
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 16, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        assert_ne!(a, b);
    }

    #[test]
    fn moe_config_inequality_different_per_tok() {
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 4, router_type: RouterType::Qwen };
        assert_ne!(a, b);
    }

    #[test]
    fn moe_config_inequality_different_router() {
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        assert_ne!(a, b);
    }

    // ── MoEConfig: single expert edge case ──

    #[test]
    fn moe_config_single_expert() {
        let cfg = MoEConfig { num_experts: 1, num_experts_per_tok: 1, router_type: RouterType::Unknown };
        assert_eq!(cfg.num_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 1);
    }

    // ── ModelManifest: rope_base_override special float values ──

    #[test]
    fn manifest_rope_override_nan() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(f32::NAN);
        assert!(m.rope_base_override.unwrap().is_nan());
    }

    #[test]
    fn manifest_rope_override_infinity() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(f32::INFINITY);
        assert!(m.rope_base_override.unwrap().is_infinite());
        assert!(m.rope_base_override.unwrap().is_sign_positive());
    }

    #[test]
    fn manifest_rope_override_neg_infinity() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(f32::NEG_INFINITY);
        assert!(m.rope_base_override.unwrap().is_infinite());
        assert!(m.rope_base_override.unwrap().is_sign_negative());
    }

    #[test]
    fn manifest_rope_override_zero() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(0.0);
        assert_eq!(m.rope_base_override, Some(0.0));
        assert!(m.rope_base_override.unwrap().is_sign_positive());
    }

    #[test]
    fn manifest_rope_override_negative_zero() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(-0.0);
        assert_eq!(m.rope_base_override, Some(-0.0));
        assert!(m.rope_base_override.unwrap().is_sign_negative());
    }

    // ── ModelManifest: max_context_override edge cases ──

    #[test]
    fn manifest_max_context_zero() {
        let mut m = ModelManifest::default();
        m.max_context_override = Some(0);
        assert_eq!(m.max_context_override, Some(0));
    }

    #[test]
    fn manifest_max_context_one() {
        let mut m = ModelManifest::default();
        m.max_context_override = Some(1);
        assert_eq!(m.max_context_override, Some(1));
    }

    #[test]
    fn manifest_max_context_max() {
        let mut m = ModelManifest::default();
        m.max_context_override = Some(usize::MAX);
        assert_eq!(m.max_context_override, Some(usize::MAX));
    }

    // ── ModelManifest: arch field with various strings ──

    #[test]
    fn manifest_arch_empty_string() {
        let m = ModelManifest { arch: String::new(), ..ModelManifest::default() };
        assert!(m.arch.is_empty());
        // Empty arch falls back to Decoder
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    #[test]
    fn manifest_arch_with_special_chars() {
        let m = ModelManifest { arch: "model/v2-alpha".to_string(), ..ModelManifest::default() };
        assert_eq!(m.arch, "model/v2-alpha");
    }

    // ── ModelManifest: model_id Cow variants ──

    #[test]
    fn manifest_model_id_empty_borrowed() {
        let m = ModelManifest { model_id: Cow::Borrowed(""), ..ModelManifest::default() };
        assert!(m.model_id.is_empty());
    }

    #[test]
    fn manifest_model_id_empty_owned() {
        let m = ModelManifest { model_id: Cow::Owned(String::new()), ..ModelManifest::default() };
        assert!(m.model_id.is_empty());
    }

    #[test]
    fn manifest_model_id_long_owned() {
        let long_id = "x".repeat(10000);
        let m = ModelManifest { model_id: Cow::Owned(long_id.clone()), ..ModelManifest::default() };
        assert_eq!(m.model_id.len(), 10000);
    }

    // ── ModelManifest: tensor_map edge cases ──

    #[test]
    fn manifest_tensor_map_insert_overwrite() {
        let mut m = ModelManifest::default();
        m.tensor_map.insert(TensorRole::Embedding, "first".to_string());
        m.tensor_map.insert(TensorRole::Embedding, "second".to_string());
        assert_eq!(m.tensor_map.get(&TensorRole::Embedding), Some(&"second".to_string()));
        assert_eq!(m.tensor_map.len(), 1);
    }

    #[test]
    fn manifest_tensor_map_remove() {
        let mut m = ModelManifest::default();
        m.tensor_map.insert(TensorRole::Embedding, "embed".to_string());
        assert!(m.tensor_map.remove(&TensorRole::Embedding).is_some());
        assert!(m.tensor_map.is_empty());
    }

    #[test]
    fn manifest_tensor_map_entry_keyed_by_role_identity() {
        let mut m = ModelManifest::default();
        m.tensor_map.insert(TensorRole::AttentionQuery, "q".to_string());
        assert!(m.tensor_map.contains_key(&TensorRole::AttentionQuery));
        assert!(!m.tensor_map.contains_key(&TensorRole::AttentionKey));
    }

    #[test]
    fn manifest_tensor_map_multiple_roles() {
        let mut m = ModelManifest::default();
        m.tensor_map.insert(TensorRole::Embedding, "e".to_string());
        m.tensor_map.insert(TensorRole::OutputHead, "o".to_string());
        m.tensor_map.insert(TensorRole::FinalNorm, "n".to_string());
        m.tensor_map.insert(TensorRole::FfnGate, "g".to_string());
        m.tensor_map.insert(TensorRole::FfnUp, "u".to_string());
        assert_eq!(m.tensor_map.len(), 5);
    }

    // ── ModelManifest: clone preserves all fields ──

    #[test]
    fn manifest_clone_preserves_overrides() {
        let mut m = ModelManifest::default();
        m.rope_base_override = Some(100000.0);
        m.max_context_override = Some(8192);
        m.moe_config = Some(MoEConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            router_type: RouterType::DeepSeek,
        });
        m.tensor_map.insert(TensorRole::Embedding, "embed".to_string());
        let c = m.clone();
        assert_eq!(c.rope_base_override, Some(100000.0));
        assert_eq!(c.max_context_override, Some(8192));
        assert!(c.is_moe());
        assert_eq!(c.tensor_map.get(&TensorRole::Embedding), Some(&"embed".to_string()));
    }

    // ── ModelManifest: Debug output spot checks ──

    #[test]
    fn manifest_debug_shows_kind() {
        let m = ModelManifest { kind: ModelKind::Reranker, ..ModelManifest::default() };
        let debug = format!("{m:?}");
        assert!(debug.contains("Reranker"));
    }

    #[test]
    fn manifest_debug_shows_arch_string() {
        let m = ModelManifest { arch: "deepseek-v3".to_string(), ..ModelManifest::default() };
        let debug = format!("{m:?}");
        assert!(debug.contains("deepseek-v3"));
    }

    // ── FileMap: non-empty iteration ──

    #[test]
    fn file_map_nonempty_iteration() {
        let fm: FileMap = &[
            ("a.safetensors", "a-001.safetensors"),
            ("b.safetensors", "b-002.safetensors"),
            ("c.safetensors", "c-003.safetensors"),
        ];
        assert_eq!(fm.len(), 3);
        let first = fm.iter().next();
        assert!(first.is_some());
        assert_eq!(first.unwrap().0, "a.safetensors");
    }

    #[test]
    fn file_map_empty_vs_nonempty() {
        let empty: FileMap = EMPTY_FILE_MAP;
        let nonempty: FileMap = &[("a", "b")];
        assert!(empty.is_empty());
        assert!(!nonempty.is_empty());
    }

    // ── map_architecture_token edge cases ──

    #[test]
    fn map_architecture_token_empty_string() {
        assert_eq!(map_architecture_token(""), None);
    }

    // ── map_kind_template with various kinds ──

    #[test]
    fn map_kind_template_all_kinds_passthrough() {
        let template = "my_template";
        for kind in [ModelKind::Chat, ModelKind::Embedding, ModelKind::Reranker, ModelKind::Classifier] {
            assert_eq!(
                map_kind_template(template, kind),
                Some(template.to_string())
            );
        }
    }

    // ── TensorRole: every variant with Some(0) produces a non-empty L0-prefixed name ──

    #[test]
    fn tensor_role_all_variants_with_layer_zero() {
        let roles = [
            TensorRole::Embedding, TensorRole::OutputHead, TensorRole::FinalNorm,
            TensorRole::ClassifierDense, TensorRole::ClassifierOutProj,
            TensorRole::PatchEmbed, TensorRole::PositionEmbedding,
            TensorRole::AttentionQuery, TensorRole::AttentionKey, TensorRole::AttentionValue,
            TensorRole::AttentionFusedQkv, TensorRole::AttentionOutput,
            TensorRole::AttentionQNorm, TensorRole::AttentionKNorm, TensorRole::AttentionSinks,
            TensorRole::InputNorm, TensorRole::PostAttnNorm, TensorRole::LayerNorm,
            TensorRole::FfnGate, TensorRole::FfnUp, TensorRole::FfnDown,
            TensorRole::MoEGate, TensorRole::MoESharedExpert, TensorRole::MoEExpert,
            TensorRole::DepthwiseConv,
            TensorRole::MlaQCompress, TensorRole::MlaQExpand, TensorRole::MlaKvCompress,
            TensorRole::MlaKeyAbsorb, TensorRole::MlaValueAbsorb, TensorRole::MlaRopeKey,
            TensorRole::MtpProjection, TensorRole::Rope,
        ];
        for role in &roles {
            let name = role.to_canonical_name(Some(0));
            assert!(name.starts_with("L0."), "TensorRole {:?} with layer 0 should start with L0., got: {}", role, name);
        }
    }

    // ── TensorRole: canonical name format consistency ──

    #[test]
    fn tensor_role_canonical_name_format_with_layer() {
        // With layer: format is "L{n}.{base}"
        let name = TensorRole::FfnDown.to_canonical_name(Some(42));
        assert!(name.starts_with("L42."));
        assert!(name.ends_with("down_proj"));
    }

    #[test]
    fn tensor_role_canonical_name_format_without_layer() {
        // Without layer: just the base name, no "L" prefix, no dot
        let name = TensorRole::FfnDown.to_canonical_name(None);
        assert!(!name.contains('.'));
        assert_eq!(name, "down_proj");
    }

    // ── ArchFamily: Debug format exact check ──

    #[test]
    fn arch_family_debug_format_exact() {
        assert_eq!(format!("{:?}", ArchFamily::Encoder), "Encoder");
        assert_eq!(format!("{:?}", ArchFamily::Decoder), "Decoder");
    }

    // ── ModelKind: Debug format exact check ──

    #[test]
    fn model_kind_debug_format_exact() {
        assert_eq!(format!("{:?}", ModelKind::Chat), "Chat");
        assert_eq!(format!("{:?}", ModelKind::Embedding), "Embedding");
        assert_eq!(format!("{:?}", ModelKind::Reranker), "Reranker");
        assert_eq!(format!("{:?}", ModelKind::Classifier), "Classifier");
    }

    // ── RouterType: Debug format exact check ──

    #[test]
    fn router_type_debug_format_exact() {
        assert_eq!(format!("{:?}", RouterType::Qwen), "Qwen");
        assert_eq!(format!("{:?}", RouterType::Mixtral), "Mixtral");
        assert_eq!(format!("{:?}", RouterType::DeepSeek), "DeepSeek");
        assert_eq!(format!("{:?}", RouterType::GptOss), "GptOss");
        assert_eq!(format!("{:?}", RouterType::Unknown), "Unknown");
    }

    // ── MoEConfig: Debug contains all field names ──

    #[test]
    fn moe_config_debug_all_fields() {
        let cfg = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::GptOss };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("num_experts"));
        assert!(debug.contains("num_experts_per_tok"));
        assert!(debug.contains("router_type"));
        assert!(debug.contains("GptOss"));
    }

    // ── ModelManifest: family() with encoder arch ──

    #[test]
    fn manifest_family_xlmr_is_encoder() {
        let m = ModelManifest { arch: "xlmr".to_string(), ..ModelManifest::default() };
        assert_eq!(m.family(), ArchFamily::Encoder);
    }

    #[test]
    fn manifest_family_llama_is_decoder() {
        let m = ModelManifest { arch: "llama".to_string(), ..ModelManifest::default() };
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    // ── MoEConfig: hash consistency ──

    #[test]
    fn moe_config_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    // ── ModelManifest: is_moe toggles ──

    #[test]
    fn manifest_is_moe_toggle() {
        let mut m = ModelManifest::default();
        assert!(!m.is_moe());
        m.moe_config = Some(MoEConfig { num_experts: 1, num_experts_per_tok: 1, router_type: RouterType::Unknown });
        assert!(m.is_moe());
        m.moe_config = None;
        assert!(!m.is_moe());
    }

    // ── NEW: 15 additional tests ──

    // 1. TensorRole: all per-layer canonical names are distinct within a single layer
    #[test]
    fn tensor_role_all_per_layer_canonical_names_unique_in_hashset() {
        use std::collections::HashSet;
        let roles = [
            TensorRole::AttentionQuery, TensorRole::AttentionKey, TensorRole::AttentionValue,
            TensorRole::AttentionFusedQkv, TensorRole::AttentionOutput,
            TensorRole::AttentionQNorm, TensorRole::AttentionKNorm, TensorRole::AttentionSinks,
            TensorRole::InputNorm, TensorRole::PostAttnNorm, TensorRole::LayerNorm,
            TensorRole::FfnGate, TensorRole::FfnUp, TensorRole::FfnDown,
            TensorRole::MoEGate, TensorRole::MoESharedExpert, TensorRole::MoEExpert,
            TensorRole::DepthwiseConv,
            TensorRole::MlaQCompress, TensorRole::MlaQExpand, TensorRole::MlaKvCompress,
            TensorRole::MlaKeyAbsorb, TensorRole::MlaValueAbsorb, TensorRole::MlaRopeKey,
            TensorRole::MtpProjection, TensorRole::Rope,
        ];
        let names: HashSet<String> = roles.iter().map(|r| r.to_canonical_name(Some(7))).collect();
        // LayerNorm aliases InputNorm, so expect roles.len() - 1 unique names
        assert_eq!(names.len(), roles.len() - 1, "Per-layer canonical names should be unique (except LayerNorm alias)");
    }

    // 2. TensorRole: canonical name with layer=1 produces correct prefix
    #[test]
    fn tensor_role_canonical_name_layer_one() {
        assert_eq!(TensorRole::FfnUp.to_canonical_name(Some(1)), "L1.up_proj");
        assert_eq!(TensorRole::MoEGate.to_canonical_name(Some(1)), "L1.moe_gate");
        assert_eq!(TensorRole::AttentionFusedQkv.to_canonical_name(Some(1)), "L1.qkv_proj");
    }

    // 3. TensorRole: MtpProjection canonical name without layer
    #[test]
    fn tensor_role_mtp_projection_no_layer() {
        assert_eq!(TensorRole::MtpProjection.to_canonical_name(None), "mtp_proj");
    }

    // 4. TensorRole: canonical names for attention roles are all distinct
    #[test]
    fn tensor_role_attention_roles_distinct_canonical_names() {
        use std::collections::HashSet;
        let attention_roles = [
            TensorRole::AttentionQuery, TensorRole::AttentionKey, TensorRole::AttentionValue,
            TensorRole::AttentionFusedQkv, TensorRole::AttentionOutput,
            TensorRole::AttentionQNorm, TensorRole::AttentionKNorm, TensorRole::AttentionSinks,
        ];
        let names: HashSet<String> = attention_roles.iter().map(|r| r.to_canonical_name(None)).collect();
        assert_eq!(names.len(), attention_roles.len());
    }

    // 5. ModelKind: tab and mixed whitespace in parse
    #[test]
    fn model_kind_parse_tab_and_mixed_whitespace() {
        assert_eq!(ModelKind::parse("\tchat\t"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("  \t  embedding  \t  "), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("\n"), None);
    }

    // 6. ModelKind: parse "embeddings" (plural) alias
    #[test]
    fn model_kind_parse_embeddings_plural() {
        assert_eq!(ModelKind::parse("embeddings"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("Embeddings"), Some(ModelKind::Embedding));
    }

    // 7. RouterType: all named pairwise inequalities
    #[test]
    fn router_type_all_named_pairwise_inequality() {
        let named = [RouterType::Qwen, RouterType::Mixtral, RouterType::DeepSeek, RouterType::GptOss];
        for i in 0..named.len() {
            for j in (i + 1)..named.len() {
                assert_ne!(named[i], named[j], "{:?} should differ from {:?}", named[i], named[j]);
            }
        }
    }

    // 8. MoEConfig: Copy trait works (value semantics)
    #[test]
    fn moe_config_copy_semantics() {
        let original = MoEConfig { num_experts: 32, num_experts_per_tok: 4, router_type: RouterType::DeepSeek };
        let copied = original;
        // Both are independent copies
        assert_eq!(original.num_experts, 32);
        assert_eq!(copied.num_experts, 32);
        assert_eq!(original, copied);
    }

    // 9. MoEConfig: hash mismatch for different configs
    #[test]
    fn moe_config_hash_differs_for_different_router() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        // Hash may collide but extremely unlikely for these two
        assert_ne!(ha.finish(), hb.finish(), "Different RouterType should produce different hashes");
    }

    // 10. ArchFamily: works as HashMap key
    #[test]
    fn arch_family_as_hashmap_key() {
        let mut map = HashMap::new();
        map.insert(ArchFamily::Encoder, "bert-style");
        map.insert(ArchFamily::Decoder, "gpt-style");
        assert_eq!(map.get(&ArchFamily::Encoder), Some(&"bert-style"));
        assert_eq!(map.get(&ArchFamily::Decoder), Some(&"gpt-style"));
        assert_eq!(map.len(), 2);
    }

    // 11. ModelManifest: family() with qwen3 returns Decoder
    #[test]
    fn manifest_family_qwen3_is_decoder() {
        let m = ModelManifest { arch: "qwen3".to_string(), ..ModelManifest::default() };
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    // 12. ModelManifest: file_map preserved through clone
    #[test]
    fn manifest_clone_preserves_file_map() {
        let fm: FileMap = &[("a.bin", "a-001.bin"), ("b.bin", "b-002.bin")];
        let m = ModelManifest {
            file_map: fm,
            ..ModelManifest::default()
        };
        let cloned = m.clone();
        assert_eq!(cloned.file_map.len(), 2);
        assert_eq!(cloned.file_map[0].0, "a.bin");
        assert_eq!(cloned.file_map[1].0, "b.bin");
    }

    // 13. FileMap: index access returns correct pairs
    #[test]
    fn file_map_index_access() {
        let fm: FileMap = &[
            ("model-001.safetensors", "part-001"),
            ("model-002.safetensors", "part-002"),
        ];
        assert_eq!(fm[0].0, "model-001.safetensors");
        assert_eq!(fm[1].1, "part-002");
        assert_eq!(fm.len(), 2);
    }

    // 14. map_architecture_token_for_kind: empty string returns None
    #[test]
    fn map_architecture_token_for_kind_empty_string() {
        assert_eq!(map_architecture_token_for_kind("", ModelKind::Chat), None);
    }

    // 15. ModelManifest: model_id Cow conversion from static str
    #[test]
    fn manifest_model_id_cow_borrowed_from_static() {
        static STATIC_ID: &str = "org/model-v2";
        let m = ModelManifest {
            model_id: Cow::Borrowed(STATIC_ID),
            ..ModelManifest::default()
        };
        assert_eq!(&*m.model_id, "org/model-v2");
    }

    // ── NEW: 10 additional tests (batch 2) ──

    // 16. MoEConfig used as HashMap key — structural equality drives lookup
    #[test]
    fn moe_config_as_hashmap_key() {
        let mut map = HashMap::new();
        let cfg = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        map.insert(cfg, "mixtral-8x2");
        let lookup = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        assert_eq!(map.get(&lookup), Some(&"mixtral-8x2"));
        let wrong = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        assert_eq!(map.get(&wrong), None);
    }

    // 17. MoEConfig: hash differs for different num_experts (same router and per_tok)
    #[test]
    fn moe_config_hash_differs_for_different_num_experts() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let b = MoEConfig { num_experts: 16, num_experts_per_tok: 2, router_type: RouterType::Qwen };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish(), "Different num_experts should produce different hashes");
    }

    // 18. ModelManifest: family() returns Decoder for known decoder architectures
    #[test]
    fn manifest_family_deepseek_v3_is_decoder() {
        let m = ModelManifest { arch: "deepseek-v3".to_string(), ..ModelManifest::default() };
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    // 19. ModelManifest: is_moe() independence through clone
    #[test]
    fn manifest_is_moe_independent_through_clone() {
        let mut original = ModelManifest::default();
        original.moe_config = Some(MoEConfig { num_experts: 4, num_experts_per_tok: 2, router_type: RouterType::DeepSeek });
        let cloned = original.clone();
        assert!(original.is_moe());
        assert!(cloned.is_moe());
        original.moe_config = None;
        assert!(!original.is_moe(), "Clearing original should not affect clone");
        assert!(cloned.is_moe(), "Clone should retain its own moe_config");
    }

    // 20. TensorRole: MLA roles produce pairwise distinct canonical names (no alias collisions)
    #[test]
    fn tensor_role_mla_all_distinct_canonical_names() {
        use std::collections::HashSet;
        let mla_roles = [
            TensorRole::MlaQCompress,
            TensorRole::MlaQExpand,
            TensorRole::MlaKvCompress,
            TensorRole::MlaKeyAbsorb,
            TensorRole::MlaValueAbsorb,
            TensorRole::MlaRopeKey,
        ];
        let names: HashSet<String> = mla_roles.iter().map(|r| r.to_canonical_name(None)).collect();
        assert_eq!(names.len(), mla_roles.len(), "All MLA roles must have distinct canonical names");
    }

    // 21. TensorRole: global roles never start with "L" when layer is None
    #[test]
    fn tensor_role_global_roles_no_L_prefix_without_layer() {
        let global_roles = [
            TensorRole::Embedding, TensorRole::OutputHead, TensorRole::FinalNorm,
            TensorRole::ClassifierDense, TensorRole::ClassifierOutProj,
            TensorRole::PatchEmbed, TensorRole::PositionEmbedding, TensorRole::Rope,
        ];
        for role in &global_roles {
            let name = role.to_canonical_name(None);
            assert!(!name.starts_with('L'), "Global role {:?} should not have L prefix, got: {}", role, name);
        }
    }

    // 22. ModelKind: FromStr error type is unit ()
    #[test]
    fn model_kind_from_str_error_type_is_unit() {
        let result: Result<ModelKind, ()> = "invalid".parse();
        assert_eq!(result, Err(()));
    }

    // 23. FileMap: iter on empty map yields no elements
    #[test]
    fn file_map_iter_empty_yields_none() {
        let fm: FileMap = EMPTY_FILE_MAP;
        assert!(fm.iter().next().is_none());
    }

    // 24. ModelManifest: tensor_map clone independence verified by mutation
    #[test]
    fn manifest_tensor_map_clone_independence() {
        let mut original = ModelManifest::default();
        original.tensor_map.insert(TensorRole::FfnGate, "gate_v1".to_string());
        let cloned = original.clone();
        original.tensor_map.insert(TensorRole::FfnUp, "up_v1".to_string());
        assert_eq!(cloned.tensor_map.len(), 1, "Clone should not see mutations to original");
        assert_eq!(original.tensor_map.len(), 2);
        assert_eq!(cloned.tensor_map.get(&TensorRole::FfnGate), Some(&"gate_v1".to_string()));
        assert!(cloned.tensor_map.get(&TensorRole::FfnUp).is_none());
    }

    // 25. ModelManifest: Default trait correctness — all fields match expected defaults
    #[test]
    fn manifest_default_comprehensive() {
        let m = ModelManifest::default();
        assert_eq!(&*m.model_id, "default");
        assert!(matches!(&m.model_id, std::borrow::Cow::Borrowed(_)), "Default model_id should be Borrowed Cow");
        assert_eq!(m.file_map as *const _, EMPTY_FILE_MAP as *const _, "file_map should be EMPTY_FILE_MAP");
        assert_eq!(m.arch, "llama");
        assert_eq!(m.kind, ModelKind::Chat);
        assert_eq!(m.rope_base_override, None);
        assert_eq!(m.max_context_override, None);
        assert_eq!(m.moe_config, None);
        assert!(m.tensor_map.is_empty());
    }
}
