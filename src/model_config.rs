//! Model config loader (metadata/tensor-driven, no config.json fallback).
#![allow(clippy::manual_checked_ops)]

use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

use crate::loader::{
    gguf::GgufReader as GgufLoader, match_tensor_role, Loader, TensorMeta, TensorProvider,
    WeightFormat,
};
use crate::manifest::{ModelManifest, TensorRole};
use gllm_kernels::types::DType;

/// Model geometric constants — single source of truth.
/// Created once from ModelConfig, immutable, Arc-shared across all subsystems.
/// Eliminates field copying between GeneratorForwardConfig, KvCacheConfig,
/// AttentionTopology, ResolvedConfig, OptimizationContext, etc.
#[derive(Debug, Clone)]
pub struct ModelGeometry {
    // ── Core dimensions ──
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,

    // ── Attention ──
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,

    // ── RoPE ──
    pub rope_theta: f64,
    pub rope_scale: f64,
    pub rope_interleaved: bool,

    // ── Gemma 4: Dual RoPE ──
    /// Global attention 层的 RoPE θ (Gemma 4: 1,000,000)。
    /// 为 0 时表示不使用 dual RoPE (所有层使用 rope_theta)。
    pub global_rope_theta: f64,
    /// Global attention 层的 RoPE partial 旋转比例 (Gemma 4: 0.25 = p-RoPE)。
    /// 1.0 表示全维度旋转 (标准 RoPE)。
    pub rope_partial_ratio: f32,

    // ── Gemma 4: Per-layer attention pattern ──
    /// 每层注意力类型: 0=sliding-window, 1=global。
    /// 空 Vec 表示所有层使用同一类型。
    pub attention_pattern: Vec<u8>,
    /// Sliding-window 注意力的窗口大小 (token 数)。
    pub sliding_window: usize,

    // ── Gemma 4: Shared KV ──
    /// 后 N 层复用前一个非共享层的 KV cache。0 = 不共享。
    pub num_kv_shared_layers: usize,

    // ── Gemma 4: Global attention head_dim ──
    /// Global attention 层的 head_dim (Gemma 4: 512，sliding 用 256)。
    /// 0 表示与 head_dim 相同。
    pub global_head_dim: usize,

    // ── Gemma 4: PLE ──
    /// Per-Layer Embedding 每层注入维度。0 = 不使用 PLE。
    pub hidden_size_per_layer_input: usize,

    // ── Tokenizer/Embedding ──
    /// Position ID 偏移量 (RoBERTa=2, BERT=0, GPT=0)。
    /// 从 tokenizer_config.json 的 pad_token_id + 1 推导，或模型 config 显式指定。
    pub position_offset: Option<usize>,

    // ── Precision ──
    pub dtype: DType,
    pub norm_eps: f32,

    // ── MoE ──
    pub num_experts: usize,
    pub moe_top_k: usize,
    pub expert_intermediate_size: usize,
}

impl ModelGeometry {
    /// Create from ModelConfig + manifest MoE info.
    /// This is the ONLY place model geometry is derived.
    pub fn from_config(config: &ModelConfig, moe_config: Option<crate::manifest::MoEConfig>) -> Self {
        let intermediate_size = config.intermediate_size.unwrap_or(config.hidden_size * 4);
        let num_experts = moe_config.map(|c| c.num_experts).unwrap_or(0);
        let moe_top_k = moe_config.map(|c| c.num_experts_per_tok).unwrap_or(0);
        let expert_intermediate_size = config.expert_intermediate_size.unwrap_or(intermediate_size);

        Self {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            intermediate_size,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_position_embeddings,
            rope_theta: config.rope_theta as f64,
            rope_scale: config.rope_scale as f64,
            rope_interleaved: config.rope_interleaved,
            global_rope_theta: config.global_rope_theta.unwrap_or(0.0) as f64,
            rope_partial_ratio: config.rope_partial_ratio.unwrap_or(1.0),
            attention_pattern: config.attention_pattern.clone().unwrap_or_default(),
            sliding_window: config.sliding_window.unwrap_or(0),
            num_kv_shared_layers: config.num_kv_shared_layers.unwrap_or(0),
            global_head_dim: config.global_head_dim.unwrap_or(0),
            hidden_size_per_layer_input: config.hidden_size_per_layer_input.unwrap_or(0),
            dtype: config.dtype,
            norm_eps: config.layer_norm_epsilon.unwrap_or(1e-12),
            // position_offset = pad_token_id + 1 (RoBERTa: pad=1→offset=2, BERT: pad=0→offset=1)
            // None 表示模型不需要 position offset（GPT-style decoder 从 0 开始）
            position_offset: config.pad_token_id.map(|pad| (pad + 1) as usize),
            num_experts,
            moe_top_k,
            expert_intermediate_size,
        }
    }

    /// Whether this is a MoE model.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// KV cache bytes per token (for memory estimation).
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * self.num_layers * self.dtype.size_bytes()
    }

    /// Expert weight bytes (gate + up + down matrices).
    pub fn expert_weight_bytes(&self) -> usize {
        self.hidden_size * self.expert_intermediate_size * 3 * self.dtype.size_bytes()
    }
}

#[derive(Debug, Error)]
pub enum ModelConfigError {
    #[error("metadata-driven config unavailable")]
    MissingConfig,
    #[error("metadata-driven config unavailable: {0}")]
    MissingConfigAndMetadata(String),
    #[error("invalid or incomplete metadata config: {0}")]
    InvalidConfig(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type ModelConfigResult<T> = std::result::Result<T, ModelConfigError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeScalingType {
    Linear,
    Dynamic,
    Yarn,
    LongRope,
    NtkAware,
    Llama3,
    Unknown(String),
}

impl RopeScalingType {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "linear" => Self::Linear,
            "dynamic" | "dynamic_ntk" | "ntk" => Self::Dynamic,
            "ntk_aware" | "ntk-aware" => Self::NtkAware,
            "yarn" => Self::Yarn,
            "longrope" | "long_rope" => Self::LongRope,
            "llama3" | "llama_3" => Self::Llama3,
            other => Self::Unknown(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeScalingConfig {
    pub scaling_type: Option<RopeScalingType>,
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    pub factors: Option<Vec<f32>>,
    pub base: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub ext_factor: Option<f32>,
    pub attn_factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
}

impl RopeScalingConfig {
    fn has_any_value(&self) -> bool {
        self.scaling_type.is_some()
            || self.rope_type.is_some()
            || self.factor.is_some()
            || self.factors.is_some()
            || self.base.is_some()
            || self.original_max_position_embeddings.is_some()
            || self.ext_factor.is_some()
            || self.attn_factor.is_some()
            || self.beta_fast.is_some()
            || self.beta_slow.is_some()
    }

    fn runtime_factor(&self) -> Option<f32> {
        self.factor.filter(|v| v.is_finite() && *v > 0.0)
    }
}

/// Activation function used in FFN layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HiddenAct {
    Silu,
    Gelu,
    GeluNew,
    Relu,
    Swish,
    QuickGelu,
    Unknown(String),
}

impl HiddenAct {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "silu" | "swiglu" => Self::Silu,
            "gelu" => Self::Gelu,
            "gelu_new" | "gelu_pytorch_tanh" => Self::GeluNew,
            "relu" => Self::Relu,
            "swish" => Self::Swish,
            "quick_gelu" | "gelu_fast" => Self::QuickGelu,
            other => Self::Unknown(other.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub expert_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub rope_interleaved: bool,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub kv_cache_block_size: usize,
    pub head_dim: usize,
    /// Model weight dtype (F32/F16/BF16).
    pub dtype: DType,
    pub use_cache: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub attention_dropout: Option<f32>,
    pub hidden_act: Option<HiddenAct>,
    pub layer_norm_epsilon: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub tensor_map: HashMap<TensorRole, String>,

    // ── Gemma 4 specific ──
    /// Global attention 层的 RoPE θ (Gemma 4: 1e6)
    pub global_rope_theta: Option<f32>,
    /// Global attention 层的 RoPE partial 旋转比例 (Gemma 4: 0.25)
    pub rope_partial_ratio: Option<f32>,
    /// 每层注意力类型: 0=sliding, 1=global
    pub attention_pattern: Option<Vec<u8>>,
    /// Sliding-window 注意力窗口大小
    pub sliding_window: Option<usize>,
    /// 后 N 层共享 KV cache
    pub num_kv_shared_layers: Option<usize>,
    /// Global 层 head_dim (Gemma 4: 512)
    pub global_head_dim: Option<usize>,
    /// PLE 每层注入维度
    pub hidden_size_per_layer_input: Option<usize>,

    /// MTP (Multi-Token Prediction) 预测深度
    /// 从 config.json 的 `num_nextn_predict_layers` 或 `mtp_depth` 解析
    pub mtp_depth: Option<usize>,

    // ── Multimodal: Vision Encoder (SigLIP) ──
    /// Vision encoder configuration parsed from `"vision_config"` sub-object.
    /// Present only for multimodal models (e.g. Gemma 4).
    pub vision_config: Option<crate::compat::vision_forward::VisionConfig>,

    /// Multimodal special token IDs (image / audio / eoi / eoa).
    ///
    /// **Source rule (T58)**: Must come from the model config or tokenizer
    /// `special_tokens_map.json`; not hard-coded in Rust. When this is
    /// `None`, the model is treated as text-only and calls to
    /// `.image()` / `.audio()` on the generation builder fail fast with
    /// `ClientError::InvalidModelType`.
    pub multimodal_token_ids: Option<crate::compat::multimodal::MultimodalTokenIds>,
}

impl ModelConfig {
    pub fn from_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let mut gguf_metadata_error: Option<ModelConfigError> = None;
        let mut safetensors_metadata_error: Option<ModelConfigError> = None;

        // GGUF 格式：优先从张量+元数据推导配置
        if loader.weight_format() == WeightFormat::Gguf {
            match Self::from_gguf_loader(manifest, loader) {
                Ok(config) => return Ok(config),
                Err(err) => gguf_metadata_error = Some(err),
            }
        }

        // SafeTensors / ONNX 格式：统一张量驱动推导 (REQ-LOADER-022, REQ-LOADER-023)
        if matches!(loader.weight_format(), WeightFormat::SafeTensors | WeightFormat::Onnx) {
            match Self::from_tensor_driven(manifest, loader) {
                Ok(config) => return Ok(config),
                Err(e) => {
                    safetensors_metadata_error = Some(e);
                }
            }
        }

        // Fallback: load from config.json file (HuggingFace standard format)
        if let Some(config_path) = loader.config_path() {
            if config_path.exists() {
                let content = std::fs::read_to_string(config_path)?;
                let value: Value = serde_json::from_str(&content)?;
                let weight_dtype = loader
                    .detect_weight_dtype()
                    .ok()
                    .flatten();
                return Self::from_value(manifest, &value, weight_dtype);
            }
        }

        if let Some(err) = gguf_metadata_error {
            return Err(ModelConfigError::MissingConfigAndMetadata(err.to_string()));
        }
        if let Some(err) = safetensors_metadata_error {
            return Err(ModelConfigError::MissingConfigAndMetadata(err.to_string()));
        }

        Err(ModelConfigError::MissingConfig)
    }

    /// REQ-LOADER-022 / REQ-LOADER-023: 统一张量驱动配置推导 (SafeTensors + ONNX)
    ///
    /// 遵循 ARCH-TENSOR-DRIVEN 原则：
    /// - 优先级：张量形状 > 元数据
    /// - 从 embedding 张量推导 vocab_size（取较大维度）
    /// - 从 Q 投影张量推导 head_dim（q_out / num_heads）
    /// - 从权重张量 dtype 推导 dtype_size
    fn from_tensor_driven(
        manifest: &ModelManifest,
        loader: &mut Loader,
    ) -> ModelConfigResult<Self> {
        // 1. Extract hints (SafeTensors: from config.json; ONNX: default)
        let hints = match loader.weight_format() {
            WeightFormat::SafeTensors => Self::extract_head_dim_hint_from_config(loader),
            _ => TensorDeriveHints::default(),
        };

        // 2. Derive config from tensors via the active TensorProvider
        let derived = match loader.weight_format() {
            WeightFormat::SafeTensors => {
                let st = loader.safetensors_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access SafeTensors loader: {err}"
                    ))
                })?;
                derive_config_from_tensors_with_hints(st, hints)?
            }
            WeightFormat::Onnx => {
                let onnx = loader.onnx_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access ONNX loader: {err}"
                    ))
                })?;
                derive_config_from_tensors_with_hints(onnx, hints)?
            }
            _ => unreachable!("from_tensor_driven only called for SafeTensors/ONNX"),
        };

        // 3. Extract metadata value (SafeTensors: gllm.config; ONNX: onnx metadata)
        let base_value = match loader.weight_format() {
            WeightFormat::SafeTensors => loader
                .safetensors_gllm_config()
                .map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to load safetensors gllm.config metadata: {err}"
                    ))
                })?
                .cloned()
                .ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "tensor-driven derivation requires gllm.config metadata for non-tensor fields"
                            .to_string(),
                    )
                })?,
            WeightFormat::Onnx => {
                let onnx = loader.onnx_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access ONNX loader: {err}"
                    ))
                })?;
                onnx_config_from_metadata(onnx)?.ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "tensor-driven ONNX derivation requires metadata attributes for non-tensor fields"
                            .to_string(),
                    )
                })?
            }
            _ => unreachable!(),
        };

        // 4. Build config
        let base = Self::from_value(manifest, &base_value, Some(derived.dtype))?;
        apply_tensor_derived(base, derived)
    }

    /// Extract head_dim hint from config.json's num_attention_heads field.
    /// Returns default hints if config.json is unavailable or missing the field.
    fn extract_head_dim_hint_from_config(loader: &Loader) -> TensorDeriveHints {
        let config_path = match loader.config_path() {
            Some(p) => p,
            None => return TensorDeriveHints::default(),
        };
        let data = match std::fs::read_to_string(config_path) {
            Ok(d) => d,
            Err(e) => {
                log::debug!("config.json read failed, using default tensor hints: {}", e);
                return TensorDeriveHints::default();
            }
        };
        let json: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                log::debug!("config.json parse failed, using default tensor hints: {}", e);
                return TensorDeriveHints::default();
            }
        };
        let num_heads = json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0);
        let hidden_size = json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0);
        match (num_heads, hidden_size) {
            (Some(nh), Some(hs)) if (hs as usize).is_multiple_of(nh as usize) => TensorDeriveHints {
                head_dim: Some(hs as usize / nh as usize),
            },
            _ => TensorDeriveHints::default(),
        }
    }

    fn from_gguf_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let (
            derived,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern_metadata,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
        ) = {
            let reader = loader.gguf_reader().map_err(|err| {
                ModelConfigError::InvalidConfig(format!("failed to load GGUF metadata: {err}"))
            })?;
            let arch = reader.architecture().map_err(|err| {
                ModelConfigError::InvalidConfig(format!(
                    "missing GGUF architecture metadata: {err}"
                ))
            })?;
            let metadata_hidden_size =
                optional_gguf_usize(reader.embedding_length(), "embedding_length")?;
            let metadata_num_attention_heads =
                optional_gguf_usize(reader.head_count(), "attention.head_count")?;
            let metadata_num_kv_heads =
                optional_gguf_usize(reader.head_count_kv(), "attention.head_count_kv")?;
            let metadata_head_dim = gguf_arch_usize(reader, arch, "attention.head_dim")
                .or_else(|| {
                    reader
                        .rope_dimension_count()
                        .and_then(|value| usize::try_from(value).ok())
                })
                .or_else(|| {
                    let hidden = metadata_hidden_size?;
                    let heads = metadata_num_attention_heads?;
                    (heads > 0 && hidden % heads == 0).then_some(hidden / heads)
                });
            let mut derived = derive_config_from_tensors_with_hints(
                reader,
                TensorDeriveHints {
                    head_dim: metadata_head_dim,
                },
            )?;

            // Ω1: Tensors are truth. Capture the physical Q-proj output size derived from tensors.
            // If metadata overrides num_heads, we MUST adjust head_dim to maintain this physical invariant.
            // q_out = n_head * head_dim
            let tensor_q_out = derived.num_attention_heads * derived.head_dim;

            if let Some(value) = metadata_num_attention_heads {
                derived.num_attention_heads = value;
                if derived.num_attention_heads > 0 {
                    derived.head_dim = tensor_q_out / derived.num_attention_heads;
                }
            }
            if let Some(value) = metadata_num_kv_heads {
                derived.num_key_value_heads = value;
            }
            if let Some(value) = metadata_head_dim {
                // If metadata provides head_dim, check if it conflicts with tensor physics
                if value != derived.head_dim {
                    // We trust tensor physics (derived.head_dim calculated from q_out) over metadata here
                    // because CpuBackend validation relies on tensor shapes.
                    // However, if q_out was derived using this hint, they should match unless
                    // num_heads was also changed.
                }
            }

            if derived.num_attention_heads == 0
                || derived.num_key_value_heads == 0
                || derived.head_dim == 0
                || derived.num_attention_heads % derived.num_key_value_heads != 0
            {
                return Err(ModelConfigError::InvalidConfig(
                    "invalid GGUF attention topology after tensor derivation".to_string(),
                ));
            }

            let intermediate_size =
                optional_gguf_usize(reader.feed_forward_length(), "feed_forward_length")?;
            let num_experts = optional_gguf_usize(reader.num_experts(), "num_experts")?;
            if matches!(num_experts, Some(0)) {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: num_experts".to_string(),
                ));
            }
            let num_experts_per_tok = optional_gguf_usize(reader.num_experts_per_tok(), "expert_used_count")?;
            let expert_intermediate_size = optional_gguf_usize(
                reader.expert_intermediate_size(),
                "expert_intermediate_size",
            )?;
            if matches!(expert_intermediate_size, Some(0)) {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: expert_intermediate_size".to_string(),
                ));
            }
            let max_position_embeddings =
                require_gguf_usize(reader.context_length(), "context_length")?;
            let rope_scaling = rope_scaling_from_gguf(reader, arch)?;
            let rope_theta = rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| require_gguf_f32(reader.rope_freq_base(), "rope.freq_base").ok())
                .ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "GGUF metadata missing: rope.freq_base (rope_theta)".to_string(),
                    )
                })?;

            // Ω1: GGUF 模型必须有 attention（已在 line 353 验证 num_attention_heads > 0）
            // 因此 rope_theta 必须 > 0
            if rope_theta == 0.0 {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata invalid: rope.freq_base (rope_theta) must be > 0".to_string(),
                ));
            }

            let rope_scale = gguf_arch_f32(reader, arch, "rope.scale")
                .or_else(|| {
                    rope_scaling
                        .as_ref()
                        .and_then(RopeScalingConfig::runtime_factor)
                })
                // Ω1: rope_scale = 1.0 is the industry standard (no scaling)
                .unwrap_or(1.0); // LEGAL: GGUF 元数据可选字段，缺失时使用行业标准默认值
            if !rope_scale.is_finite() || rope_scale <= 0.0 {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: rope.scale".to_string(),
                ));
            }

            let rope_interleaved =
                gguf_arch_bool(reader, arch, "rope.interleaved").unwrap_or(false); // LEGAL: GGUF 元数据可选字段，缺失时使用行业标准默认值
            let attention_dropout =
                gguf_arch_f32(reader, arch, "attention.dropout").filter(|v| v.is_finite());
            let hidden_act = gguf_arch_str(reader, arch, "feed_forward.activation")
                .or_else(|| gguf_arch_str(reader, arch, "hidden_act"))
                .map(|v| HiddenAct::parse(v));
            let layer_norm_epsilon = gguf_arch_f32(reader, arch, "layer_norm_epsilon")
                .or_else(|| gguf_arch_f32(reader, arch, "layer_norm_rms_epsilon"))
                .or_else(|| gguf_arch_f32(reader, arch, "attention.layer_norm_rms_epsilon"))
                .filter(|v| v.is_finite() && *v > 0.0);

            // ── Gemma 4 / dual-attention family metadata ───────────────────
            // Ω1: load from GGUF when present. Rope/partial/global-head/PLE keys
            // are sensitive (affect numerics) so 0/absent stays as `None` at
            // this layer — `ModelGeometry::from_config` keeps the "0 means not
            // enabled" contract explicit.
            //
            // Community key conventions (as of 2026 Q1, llama.cpp/unsloth):
            //   {arch}.attention.sliding_window
            //   {arch}.attention.num_kv_shared_layers
            //   {arch}.attention.global_head_dim
            //   {arch}.attention.pattern                 (ARRAY[U8])
            //   {arch}.rope.global.freq_base             (Gemma 4 dual-RoPE global θ)
            //   {arch}.rope.partial_ratio                (p-RoPE fraction, 0..=1)
            //   {arch}.embedding.per_layer_input         (PLE injection width)
            let global_rope_theta = gguf_arch_f32(reader, arch, "rope.global.freq_base")
                .or_else(|| gguf_arch_f32(reader, arch, "global_rope_theta"))
                .filter(|v| v.is_finite() && *v > 0.0);
            let rope_partial_ratio = gguf_arch_f32(reader, arch, "rope.partial_ratio")
                .or_else(|| gguf_arch_f32(reader, arch, "rope.global.partial_ratio"))
                .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0);
            let attention_pattern_metadata = gguf_arch_array_u8(reader, arch, "attention.pattern")
                .or_else(|| gguf_arch_array_u8(reader, arch, "attention_pattern"));
            let sliding_window = gguf_arch_usize(reader, arch, "attention.sliding_window")
                .or_else(|| gguf_arch_usize(reader, arch, "sliding_window"));
            let num_kv_shared_layers =
                gguf_arch_usize(reader, arch, "attention.num_kv_shared_layers")
                    .or_else(|| gguf_arch_usize(reader, arch, "num_kv_shared_layers"));
            let global_head_dim = gguf_arch_usize(reader, arch, "attention.global_head_dim")
                .or_else(|| gguf_arch_usize(reader, arch, "global_head_dim"));
            let hidden_size_per_layer_input =
                gguf_arch_usize(reader, arch, "embedding.per_layer_input")
                    .or_else(|| gguf_arch_usize(reader, arch, "hidden_size_per_layer_input"));

            (
                derived,
                intermediate_size,
                num_experts,
                num_experts_per_tok,
                expert_intermediate_size,
                max_position_embeddings,
                rope_theta,
                rope_scale,
                rope_interleaved,
                rope_scaling,
                attention_dropout,
                hidden_act,
                layer_norm_epsilon,
                reader.bos_token_id(),
                reader.eos_token_id(),
                reader
                    .get_metadata_u64("tokenizer.ggml.padding_token_id")
                    .and_then(|v| u32::try_from(v).ok()),
                global_rope_theta,
                rope_partial_ratio,
                attention_pattern_metadata,
                sliding_window,
                num_kv_shared_layers,
                global_head_dim,
                hidden_size_per_layer_input,
            )
        };

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings); // LEGAL: manifest 可选字段，缺失时使用推导值
        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }
        let rope_theta = manifest.rope_base_override.unwrap_or(rope_theta); // LEGAL: manifest 可选字段，缺失时使用推导值

        let base_derived = derived.clone();

        // Gemma 4 attention_pattern fallback: GGUF 未提供时，按 SPEC 默认模式
        // (每 6 层第 6 层 global) 派生。仅当 PLE/global-rope 等信号表明确为
        // Gemma-4 家族时启用；否则保持 None 以免污染其他 arch。
        let enable_gemma4_pattern_default = hidden_size_per_layer_input
            .map(|v| v > 0)
            .unwrap_or(false)
            || global_rope_theta.is_some()
            || sliding_window.map(|v| v > 0).unwrap_or(false);
        let attention_pattern = attention_pattern_metadata.or_else(|| {
            if enable_gemma4_pattern_default && base_derived.num_hidden_layers > 0 {
                Some(derive_default_attention_pattern(
                    base_derived.num_hidden_layers,
                ))
            } else {
                None
            }
        });

        let base = Self {
            hidden_size: base_derived.hidden_size,
            num_attention_heads: base_derived.num_attention_heads,
            num_key_value_heads: base_derived.num_key_value_heads,
            num_hidden_layers: base_derived.num_hidden_layers,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            vocab_size: base_derived.vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            kv_cache_block_size: base_derived.head_dim.max(base_derived.num_key_value_heads),
            head_dim: base_derived.head_dim,
            dtype: base_derived.dtype,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            tensor_map: HashMap::new(),
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
            mtp_depth: None,
            vision_config: None,
            multimodal_token_ids: None,
        };
        apply_tensor_derived(base, derived)
    }

    pub fn from_value(
        manifest: &ModelManifest,
        value: &Value,
        weight_dtype: Option<DType>,
    ) -> ModelConfigResult<Self> {
        let hidden_size = require_usize(value, &["hidden_size", "n_embd", "d_model"])?;
        let num_attention_heads =
            require_usize(value, &["num_attention_heads", "n_head", "num_heads"])?;
        let num_key_value_heads = find_usize(
            value,
            &[
                "num_key_value_heads",
                "num_kv_heads",
                "n_kv_head",
                "attention.num_key_value_heads",
            ],
        )
        .unwrap_or_else(|| {
            log::debug!("num_key_value_heads not found: defaulting to num_attention_heads = {}", num_attention_heads);
            num_attention_heads // LEGAL: num_key_value_heads 默认等于 num_attention_heads（非 GQA 模型）
        });
        let num_hidden_layers =
            require_usize(value, &["num_hidden_layers", "n_layer", "num_layers"])?;
        let intermediate_size = find_usize(
            value,
            &["intermediate_size", "n_inner", "ffn_inter_dim", "d_ff"],
        );
        let num_experts = find_usize(value, &["num_experts", "moe.num_experts", "num_local_experts", "n_routed_experts"]);
        let num_experts_per_tok = find_usize(value, &["num_experts_per_tok", "num_selected_experts", "num_experts_per_token", "moe.num_experts_per_tok"]);
        let expert_intermediate_size = find_usize(
            value,
            &[
                "expert_intermediate_size",
                "moe.expert_intermediate_size",
                "moe_intermediate_size",
            ],
        );
        if matches!(num_experts, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "num_experts must be > 0 when provided".to_string(),
            ));
        }
        if matches!(expert_intermediate_size, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "expert_intermediate_size must be > 0 when provided".to_string(),
            ));
        }
        let vocab_size = require_usize(value, &["vocab_size"])?;
        let rope_scaling = rope_scaling_from_metadata_json(value)?;

        let max_position_embeddings = find_usize(
            value,
            &[
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "rope_scaling.original_max_position_embeddings",
            ]
        )
        .unwrap_or_else(|| {
            log::debug!("max_position_embeddings not found: defaulting to 0 prior to manifest override");
            0 // LEGAL: 默认 0，后续由 manifest.override 或错误检查处理
        });

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings); // LEGAL: manifest 可选字段，缺失时使用推导值

        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }

        // Ω1: rope_theta 从模型配置或 manifest 中读取。
        // Encoder 模型（BERT/XLM-R）使用绝对位置编码，不含 rope_theta → 默认 0.0（表示无 RoPE）。
        // 下游 executor 通过 PositionEncoding::None 正确处理 rope_theta == 0.0 的 encoder 模型。
        let rope_theta = if let Some(override_value) = manifest.rope_base_override {
            override_value
        } else {
            rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| find_f32(value, &["rope_theta", "rope_base", "rope_base_value"]))
                .unwrap_or_else(|| {
                    log::debug!("rope_theta not found: defaulting to 0.0 (model uses absolute position embeddings)");
                    0.0 // LEGAL: encoder 模型（BERT/XLM-R）无 RoPE，0.0 表示不使用旋转位置编码
                })
        };

        // RoPE 缩放系数优先读取完整 rope_scaling 对象；缺失时保持无缩放 (1.0)。
        // Ω1: rope_scale = 1.0 is the industry standard (no rotation scaling)
        let rope_scale = find_f32(value, &["rope_scale", "rope_factor", "rope.scaling.factor"])
            .or_else(|| {
                rope_scaling
                    .as_ref()
                    .and_then(RopeScalingConfig::runtime_factor)
            })
            .unwrap_or_else(|| {
                log::debug!("rope_scale not found: defaulting to 1.0 (no scaling)");
                1.0 // LEGAL: rope_scale=1.0 是 RoPE 的行业标准默认值（无缩放）
            });
        if !rope_scale.is_finite() || rope_scale <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scale must be positive".to_string(),
            ));
        }

        let rope_interleaved = find_bool(
            value,
            &[
                "rope_interleaved",
                "rotary_interleaved",
                "interleaved_rotary",
                "rope_scaling.interleaved",
            ]
        )
        .unwrap_or_else(|| {
            log::debug!("rope_interleaved not found: defaulting to false");
            false // LEGAL: rope_interleaved=false 是 RoPE 的行业标准默认值（非交错）
        });

        // Ω1: head_dim 从元数据读取，或使用标准公式计算
        // 注意：Embedding 模型可能没有 num_attention_heads，此时 head_dim = 0
        let head_dim = if num_attention_heads > 0 {
            find_usize(value, &["attention.head_dim", "head_dim", "kv_channels"])
                .unwrap_or_else(|| {
                    let derived = hidden_size / num_attention_heads;
                    log::debug!("head_dim not found: deriving from hidden_size/num_heads => {}", derived);
                    derived // LEGAL: head_dim 可由 hidden_size/num_heads 推导（标准公式）
                })
        } else {
            // Embedding 模型没有 attention heads
            0
        };

        if num_attention_heads > 0 && head_dim == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid head_dim for non-embedding model".to_string(),
            ));
        }

        // Ω1: rope_theta == 0.0 合法 — encoder 模型（BERT/XLM-R）有 attention 但不使用 RoPE。
        // 下游 executor 会根据 (ModelKind, rope_theta) 选择 PositionEncoding::None 或 Rope。

        let kv_cache_block_size = find_usize(
            value,
            &["kv_cache_block_size", "kv_block_size", "page_size"],
        )
        .unwrap_or_else(|| {
            let derived = head_dim.max(num_key_value_heads);
            log::debug!("kv_cache_block_size not found: deriving from head_dim.max(num_key_value_heads) => {}", derived);
            derived // LEGAL: kv_cache_block_size 可由 head_dim.max(num_kv_heads) 推导（标准公式）
        });
        if kv_cache_block_size == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid kv_cache_block_size".to_string(),
            ));
        }

        // Ω1: dtype 大小必须从实际权重中读取，不从外部配置推断。
        let weight_dtype = weight_dtype.ok_or_else(|| {
            ModelConfigError::InvalidConfig(
                "无法确定模型 dtype：权重中缺少可识别浮点 dtype".to_string(),
            )
        })?;

        // Derive dtype: prefer torch_dtype from config.json, fall back to detected weight dtype.
        let dtype = find_string(value, &["torch_dtype", "dtype"])
            .and_then(|s| match s.as_str() {
                "bfloat16" => Some(DType::BF16),
                "float16" => Some(DType::F16),
                "float32" => Some(DType::F32),
                "float64" => Some(DType::F32), // f64 降级到 f32
                other => {
                    log::warn!("Unknown torch_dtype: {}, using detected weight dtype", other);
                    None
                }
            })
            .unwrap_or(weight_dtype); // LEGAL: 配置缺失时使用从权重检测的 dtype


        let attention_dropout = find_f32(value, &["attention_dropout", "attention.dropout"])
            .filter(|v| v.is_finite() && *v >= 0.0);
        let layer_norm_epsilon = find_f32(
            value,
            &[
                "layer_norm_epsilon",
                "layer_norm_eps",
                "rms_norm_eps",
                "norm_epsilon",
            ],
        )
        .filter(|v| v.is_finite() && *v > 0.0);

        // ── Gemma 4 specific fields ──
        let global_rope_theta = find_f32(value, &["global_rope_theta"]);
        let rope_partial_ratio = find_f32(value, &["rope_partial_ratio", "global_rope_partial"]);
        let attention_pattern = value.get("attention_pattern")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect());
        let sliding_window = find_usize(value, &["sliding_window", "sliding_window_size"]);
        let num_kv_shared_layers = find_usize(value, &["num_kv_shared_layers"]);
        let global_head_dim = find_usize(value, &["global_head_dim"]);
        let hidden_size_per_layer_input = find_usize(value, &["hidden_size_per_layer_input"]);

        // ── MTP (Multi-Token Prediction) ──
        let mtp_depth = find_usize(value, &["num_nextn_predict_layers", "mtp_depth"]);

        // ── Multimodal: Vision Encoder (SigLIP) ──
        let vision_config = value.get("vision_config").and_then(|vc| {
            let image_size = find_usize(vc, &["image_size"])?;
            let patch_size = find_usize(vc, &["patch_size"])?;
            let vis_hidden = find_usize(vc, &["hidden_size"])?;
            let num_layers = find_usize(vc, &["num_hidden_layers", "num_layers"])?;
            let num_heads = find_usize(vc, &["num_attention_heads", "num_heads"])?;
            let vis_intermediate = find_usize(vc, &["intermediate_size"])?;
            Some(crate::compat::vision_forward::VisionConfig {
                image_size,
                patch_size,
                hidden_size: vis_hidden,
                num_layers,
                num_heads,
                intermediate_size: vis_intermediate,
            })
        });

        // ── Multimodal special token IDs (T58) ──
        // 优先从 config 顶层字段读取（兼容 Gemma 4 `boi_token_id` /
        // `image_token_id` 两种命名），否则从 tokenizer special_tokens_map
        // 读取（由 loader 侧合并到 value 时生效）。T58 铁律：禁止在 Rust
        // 源码里硬编码 ID。仅当 `vision_config` 声明存在但顶层 token 字段
        // 缺失时，才回退到 Gemma-4 默认值作为兜底，避免模型声明了多模态
        // 能力却无法路由。
        let image_tok = find_u32(value, &["image_token_id", "boi_token_id"]);
        let audio_tok = find_u32(value, &["audio_token_id", "boa_token_id"]);
        let eoi_tok = find_u32(value, &["eoi_token_id", "image_end_token_id"]);
        let eoa_tok = find_u32(value, &["eoa_token_id", "audio_end_token_id"]);
        let multimodal_token_ids = match (image_tok, audio_tok) {
            (Some(img), Some(aud)) => Some(crate::compat::multimodal::MultimodalTokenIds {
                image_token_id: img,
                audio_token_id: aud,
                eoi_token_id: eoi_tok.unwrap_or(img + 2),
                eoa_token_id: eoa_tok.unwrap_or(aud + 2),
            }),
            _ => {
                if vision_config.is_some() {
                    Some(crate::compat::multimodal::MultimodalTokenIds::gemma4_defaults())
                } else {
                    None
                }
            }
        };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            kv_cache_block_size,
            head_dim,
            dtype,
            use_cache: find_bool(value, &["use_cache"]),
            tie_word_embeddings: find_bool(value, &["tie_word_embeddings"]),
            attention_dropout,
            hidden_act: find_string(value, &["hidden_act", "hidden_activation"]).map(|s| HiddenAct::parse(&s)),
            layer_norm_epsilon,
            bos_token_id: find_u32(value, &["bos_token_id"]),
            eos_token_id: find_u32(value, &["eos_token_id"]),
            pad_token_id: find_u32(value, &["pad_token_id"]),
            tensor_map: manifest.tensor_map.clone(),
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
            mtp_depth,
            vision_config,
            multimodal_token_ids,
        })
    }

    /// Build MoEConfig from extracted metadata, if this is a MoE model.
    pub fn build_moe_config(&self, arch: &str) -> Option<crate::manifest::MoEConfig> {
        let num_experts = self.num_experts?;
        if num_experts <= 1 {
            return None;
        }
        let num_experts_per_tok = self.num_experts_per_tok.unwrap_or(2); // LEGAL: num_experts_per_tok=2 是 MoE 的行业标准默认值
        crate::arch::register_builtin_templates();
        let router_type = crate::arch::resolve_moe_router(arch)
            .unwrap_or(crate::manifest::RouterType::Mixtral);
        Some(crate::manifest::MoEConfig {
            num_experts,
            num_experts_per_tok,
            router_type,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorDerivedConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub tensor_map: HashMap<TensorRole, String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TensorDeriveHints {
    pub head_dim: Option<usize>,
}

const VALID_HEAD_DIMS: [usize; 6] = [32, 64, 80, 96, 128, 256];

pub(crate) fn derive_config_from_tensors_with_hints<P: TensorProvider>(
    provider: &P,
    hints: TensorDeriveHints,
) -> ModelConfigResult<TensorDerivedConfig> {
    let metas: Vec<TensorMeta> = provider.iter_tensors().collect();
    if metas.is_empty() {
        return Err(ModelConfigError::InvalidConfig(
            "tensor provider returned no tensors".to_string(),
        ));
    }

    // 1. Group tensors by role
    let mut role_map: HashMap<(TensorRole, Option<usize>), &TensorMeta> = HashMap::new();
    let mut max_layer_idx = 0;
    let mut has_layers = false;

    for meta in &metas {
        if let Some((role, layer)) = match_tensor_role(&meta.name) {
            role_map.insert((role, layer), meta);
            if let Some(idx) = layer {
                if idx > max_layer_idx {
                    max_layer_idx = idx;
                }
                has_layers = true;
            }
        }
    }

    // 2. Derive Vocab Size & Hidden Size from Embedding
    // Prefer explicitly identified embedding role
    let embedding_meta = role_map
        .get(&(TensorRole::Embedding, None))
        .or_else(|| role_map.get(&(TensorRole::Embedding, Some(0))));

    let (vocab_size, hidden_size) = if let Some(meta) = embedding_meta {
        if meta.shape.len() < 2 {
            return Err(ModelConfigError::InvalidConfig(format!(
                "embedding tensor {} must be at least 2D",
                meta.name
            )));
        }
        // Heuristic: Vocab size is usually larger than hidden size
        // If they are equal, it doesn't matter for validation purposes
        let a = meta.shape[0];
        let b = meta.shape[1];
        (a.max(b), a.min(b))
    } else {
        return Err(ModelConfigError::InvalidConfig(
            "cannot derive config: no embedding tensor found".to_string(),
        ));
    };

    // 3. Derive Layer Count
    let num_hidden_layers = if has_layers { max_layer_idx + 1 } else { 0 };

    // 4. Derive Heads and Head Dim
    // We check Layer 0 for consistency
    let q_meta = role_map.get(&(TensorRole::AttentionQuery, Some(0)));
    let k_meta = role_map.get(&(TensorRole::AttentionKey, Some(0)));

    let (q_out, k_out) = match (q_meta, k_meta) {
        (Some(q), Some(k)) => {
            let q_out = projection_out_dim(q, hidden_size, "Q projection")?;
            let k_out = projection_out_dim(k, hidden_size, "K projection")?;
            (q_out, k_out)
        }
        _ => {
            // Fallback for models that might not have layer 0 or use different structure?
            // For now, strict validation: if we detected layers, we expect Q/K in layer 0.
            if num_hidden_layers > 0 {
                return Err(ModelConfigError::InvalidConfig(
                    "cannot derive attention params: missing Q/K projection in layer 0".to_string(),
                ));
            } else {
                // No layers? (Embedding model?)
                (0, 0)
            }
        }
    };

    // Cross-layer consistency check (Ω1: True Source Principle)
    // All layers must have consistent Q/K projection dimensions
    if num_hidden_layers > 1 && q_out > 0 {
        for layer_idx in 1..num_hidden_layers {
            if let Some(q) = role_map.get(&(TensorRole::AttentionQuery, Some(layer_idx))) {
                let layer_q_out = projection_out_dim(q, hidden_size, "Q projection")?;
                if layer_q_out != q_out {
                    return Err(ModelConfigError::InvalidConfig(format!(
                        "cross-layer mismatch: layer 0 Q projection has dim {q_out}, but layer {layer_idx} has dim {layer_q_out}"
                    )));
                }
            }
            if let Some(k) = role_map.get(&(TensorRole::AttentionKey, Some(layer_idx))) {
                let layer_k_out = projection_out_dim(k, hidden_size, "K projection")?;
                if layer_k_out != k_out {
                    return Err(ModelConfigError::InvalidConfig(format!(
                        "cross-layer mismatch: layer 0 K projection has dim {k_out}, but layer {layer_idx} has dim {layer_k_out}"
                    )));
                }
            }
        }
    }

    let mut head_candidates = Vec::new();
    if q_out > 0 && k_out > 0 {
        for head_dim in VALID_HEAD_DIMS {
            if hints.head_dim.is_some_and(|hint| head_dim != hint) {
                continue;
            }
            if q_out % head_dim != 0 || k_out % head_dim != 0 {
                continue;
            }
            let n_head = q_out / head_dim;
            let n_kv = k_out / head_dim;

            // Basic sanity checks
            if n_head == 0 || n_kv == 0 {
                continue;
            }
            if n_head % n_kv != 0 {
                continue;
            } // Grouped Query Attention constraint

            head_candidates.push((n_head, n_kv, head_dim));
        }
    } else if num_hidden_layers == 0 {
        // Embedding model without attention
        head_candidates.push((0, 0, 0));
    }

    head_candidates.sort_unstable();
    head_candidates.dedup();

    if head_candidates.is_empty() {
        return Err(ModelConfigError::InvalidConfig(format!(
            "cannot derive head_dim from tensors (q_out={q_out}, k_out={k_out})"
        )));
    }
    if head_candidates.len() > 1 {
        // Ω1: Ambiguity must be rejected
        return Err(ModelConfigError::InvalidConfig(format!(
            "ambiguous head_dim candidates: {:?}",
            head_candidates
        )));
    }

    let (num_attention_heads, num_key_value_heads, head_dim) = head_candidates[0];

    // 5. Intermediate Size
    let mut intermediate_size = None;
    if let Some(gate) = role_map.get(&(TensorRole::FfnGate, Some(0))) {
        let out = projection_out_dim(gate, hidden_size, "FFN Gate")?;
        intermediate_size = Some(out);
    } else if let Some(up) = role_map.get(&(TensorRole::FfnUp, Some(0))) {
        let out = projection_out_dim(up, hidden_size, "FFN Up")?;
        intermediate_size = Some(out);
    }

    let dtype = derive_dtype(&metas)?;

    // 6. Build Tensor Map
    // We only need to store the pattern for each role once.
    let mut tensor_map = HashMap::new();
    for ((role, layer), meta) in &role_map {
        if tensor_map.contains_key(role) {
            continue;
        }

        let pattern = if let Some(idx) = layer {
            anonymize_layer_index(&meta.name, *idx)
        } else {
            meta.name.clone()
        };
        tensor_map.insert(*role, pattern);
    }

    Ok(TensorDerivedConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        num_hidden_layers,
        intermediate_size,
        vocab_size,
        head_dim,
        dtype,
        tensor_map,
    })
}

fn anonymize_layer_index(name: &str, layer_idx: usize) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut new_parts = Vec::new();
    let idx_str = layer_idx.to_string();

    // Use the same logic as match_tensor_role to locate the layer index
    let mut replaced = false;
    for (i, part) in parts.iter().enumerate() {
        if !replaced && *part == idx_str {
            // Check context like match_tensor_role
            if i > 0 {
                let prefix = parts[i - 1];
                if matches!(
                    prefix,
                    "layers" | "blk" | "blocks" | "h" | "layer" | "block"
                ) {
                    new_parts.push("{}");
                    replaced = true;
                    continue;
                }
            }
            // Fallback: if we didn't match prefix but exact string matches and we haven't replaced yet,
            // we might consider it. But match_tensor_role is strict about prefix.
            // If match_tensor_role found it, it must have matched the prefix check.
        }
        new_parts.push(part);
    }

    // If exact logic failed (maybe due to case sensitivity in match_tensor_role vs here?),
    // try simpler heuristic if not replaced?
    // match_tensor_role uses to_ascii_lowercase() but we have original name here.
    // Let's assume the original name preserves case but structure is standard.

    if !replaced {
        // Second pass: just find the number segment if it wasn't replaced
        new_parts.clear();
        for part in parts.iter() {
            if !replaced && *part == idx_str {
                new_parts.push("{}");
                replaced = true;
            } else {
                new_parts.push(part);
            }
        }
    }

    new_parts.join(".")
}

fn apply_tensor_derived(
    mut base: ModelConfig,
    derived: TensorDerivedConfig,
) -> ModelConfigResult<ModelConfig> {
    base.hidden_size = derived.hidden_size;
    base.num_attention_heads = derived.num_attention_heads;
    base.num_key_value_heads = derived.num_key_value_heads;
    base.num_hidden_layers = derived.num_hidden_layers;
    if let Some(intermediate_size) = derived.intermediate_size {
        base.intermediate_size = Some(intermediate_size);
    }
    base.vocab_size = derived.vocab_size;
    base.head_dim = derived.head_dim;
    base.dtype = derived.dtype;
    base.kv_cache_block_size = base.head_dim.max(base.num_key_value_heads);
    // Ω1: Update tensor map with derived patterns
    base.tensor_map = derived.tensor_map;

    if base.head_dim == 0 {
        return Err(ModelConfigError::InvalidConfig(
            "invalid tensor-derived head_dim".to_string(),
        ));
    }
    if base.kv_cache_block_size == 0 {
        return Err(ModelConfigError::InvalidConfig(
            "invalid tensor-derived kv_cache_block_size".to_string(),
        ));
    }

    Ok(base)
}

fn onnx_config_from_metadata(
    loader: &crate::loader::OnnxLoader,
) -> ModelConfigResult<Option<Value>> {
    let model_props = &loader.model().metadata.metadata_props;
    let graph_props = &loader.graph().metadata_props;

    for props in [graph_props, model_props] {
        for key in ["gllm.config", "_gllm_config"] {
            if let Some(raw) = props.get(key) {
                let parsed: Value = serde_json::from_str(raw).map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "invalid ONNX metadata json for {key}: {err}"
                    ))
                })?;
                return Ok(Some(parsed));
            }
        }
    }

    let mut root = serde_json::Map::new();
    let mut has_value = false;
    for props in [model_props, graph_props] {
        for (key, raw) in props {
            if key == "gllm.config" || key == "_gllm_config" {
                continue;
            }
            if key.trim().is_empty() {
                continue;
            }
            has_value = true;
            let parsed = parse_metadata_string_value(raw);
            insert_json_path(&mut root, key, parsed);
        }
    }

    if has_value {
        Ok(Some(Value::Object(root)))
    } else {
        Ok(None)
    }
}

fn parse_metadata_string_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
        return parsed;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    if let Ok(v) = trimmed.parse::<u64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = trimmed.parse::<i64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        if let Some(number) = serde_json::Number::from_f64(v) {
            return Value::Number(number);
        }
    }
    Value::String(trimmed.to_string())
}

fn insert_json_path(root: &mut serde_json::Map<String, Value>, path: &str, value: Value) {
    let segments = path
        .split('.')
        .filter(|segment| !segment.trim().is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return;
    }
    insert_json_path_segments(root, &segments, value);
}

fn insert_json_path_segments(
    root: &mut serde_json::Map<String, Value>,
    segments: &[&str],
    value: Value,
) {
    if segments.len() == 1 {
        root.insert(segments[0].to_string(), value);
        return;
    }

    let entry = root
        .entry(segments[0].to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(serde_json::Map::new());
    }
    let child = entry
        .as_object_mut()
        .expect("insert_json_path ensures object");
    insert_json_path_segments(child, &segments[1..], value);
}

#[allow(dead_code)]
fn derive_dtype_size(metas: &[TensorMeta]) -> ModelConfigResult<usize> {
    let mut float_sizes = Vec::new();
    let mut all_sizes = Vec::new();

    for meta in metas {
        let Some(size) = dtype_size_from_dtype(meta.dtype) else {
            continue;
        };
        all_sizes.push(size);
        if is_floating_dtype(meta.dtype) {
            float_sizes.push(size);
        }
    }

    if !float_sizes.is_empty() {
        return unique_mode(&float_sizes, "dtype_size");
    }
    if !all_sizes.is_empty() {
        return unique_mode(&all_sizes, "dtype_size");
    }

    Err(ModelConfigError::InvalidConfig(
        "cannot derive dtype_size from tensor dtypes".to_string(),
    ))
}

/// Derive the dominant floating-point dtype string from tensor metadata.
/// Returns "f32" as default when no floating tensors are found.
fn derive_dtype(metas: &[TensorMeta]) -> ModelConfigResult<DType> {
    let mut bf16_count = 0usize;
    let mut f16_count = 0usize;
    let mut f32_count = 0usize;

    for meta in metas {
        if !is_floating_dtype(meta.dtype) {
            continue;
        }
        match meta.dtype {
            safetensors::Dtype::BF16 => bf16_count += 1,
            safetensors::Dtype::F16 => f16_count += 1,
            safetensors::Dtype::F32 => f32_count += 1,
            safetensors::Dtype::F64 => f32_count += 1, // f64 降级到 f32
            _ => {}
        }
    }

    // Pick the dominant dtype by count
    let max = bf16_count.max(f16_count).max(f32_count);
    if max == 0 {
        return Ok(DType::F32); // 默认 F32
    }
    if bf16_count == max {
        Ok(DType::BF16)
    } else if f16_count == max {
        Ok(DType::F16)
    } else {
        Ok(DType::F32)
    }
}

#[allow(dead_code)]
fn dtype_size_from_dtype(dtype: safetensors::Dtype) -> Option<usize> {
    match dtype {
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => Some(8),
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => Some(4),
        safetensors::Dtype::F16
        | safetensors::Dtype::BF16
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => Some(2),
        safetensors::Dtype::F8_E5M2
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::I8
        | safetensors::Dtype::U8
        | safetensors::Dtype::BOOL => Some(1),
        _ => None,
    }
}

fn is_floating_dtype(dtype: safetensors::Dtype) -> bool {
    matches!(
        dtype,
        safetensors::Dtype::F64
            | safetensors::Dtype::F32
            | safetensors::Dtype::F16
            | safetensors::Dtype::BF16
            | safetensors::Dtype::F8_E5M2
            | safetensors::Dtype::F8_E4M3
    )
}

#[allow(dead_code)]
fn unique_mode(values: &[usize], field: &str) -> ModelConfigResult<usize> {
    if values.is_empty() {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} cannot be derived: no candidates"
        )));
    }

    let mut counts = HashMap::<usize, usize>::new();
    for value in values {
        *counts.entry(*value).or_default() += 1;
    }

    let max_count =
        counts.values().copied().max().ok_or_else(|| {
            ModelConfigError::InvalidConfig(format!("{field} has no valid count"))
        })?;
    let mut winners = counts
        .into_iter()
        .filter_map(|(value, count)| (count == max_count).then_some(value))
        .collect::<Vec<_>>();
    winners.sort_unstable();
    if winners.len() != 1 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} is ambiguous: candidates={winners:?}"
        )));
    }
    let winner = winners[0];
    if winner == 0 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} resolved to invalid zero"
        )));
    }
    Ok(winner)
}

fn projection_out_dim(
    meta: &TensorMeta,
    hidden_size: usize,
    role: &str,
) -> ModelConfigResult<usize> {
    if meta.shape.len() < 2 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} must be at least 2D",
            meta.name
        )));
    }

    let a = meta.shape[0];
    let b = meta.shape[1];
    let out = if a == hidden_size && b != hidden_size {
        b
    } else if b == hidden_size && a != hidden_size {
        a
    } else if a == hidden_size && b == hidden_size {
        hidden_size
    } else {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} shape {:?} does not contain hidden_size {}",
            meta.name, meta.shape, hidden_size
        )));
    };

    if out == 0 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} resolved zero output dimension",
            meta.name
        )));
    }
    Ok(out)
}

fn require_usize(value: &Value, keys: &[&str]) -> ModelConfigResult<usize> {
    find_usize(value, keys).ok_or_else(|| ModelConfigError::InvalidConfig(keys[0].to_string()))
}

fn require_gguf_usize(value: Option<u64>, field: &str) -> ModelConfigResult<usize> {
    let value = value.ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })?;
    usize::try_from(value).map_err(|_| {
        ModelConfigError::InvalidConfig(format!("GGUF metadata field overflow: {field}"))
    })
}

fn optional_gguf_usize(value: Option<u64>, field: &str) -> ModelConfigResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = usize::try_from(value).map_err(|_| {
        ModelConfigError::InvalidConfig(format!("GGUF metadata field overflow: {field}"))
    })?;
    Ok(Some(parsed))
}

fn require_gguf_f32(value: Option<f32>, field: &str) -> ModelConfigResult<f32> {
    value.filter(|v| v.is_finite()).ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })
}

fn find_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value_at_path(value, key))
}

fn value_at_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn find_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn find_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn find_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    find_value(value, keys).and_then(|v| v.as_f64().map(|num| num as f32))
}

fn find_f32_array(value: &Value, keys: &[&str]) -> Option<Vec<f32>> {
    let values = find_value(value, keys)?.as_array()?;
    let mut out = Vec::with_capacity(values.len());
    for item in values {
        let value = item.as_f64()? as f32;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

fn find_string(value: &Value, keys: &[&str]) -> Option<String> {
    find_value(value, keys)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn find_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    find_value(value, keys).and_then(|v| {
        v.as_bool().or_else(|| {
            v.as_u64().and_then(|num| match num {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
        })
    })
}

fn rope_scaling_from_metadata_json(value: &Value) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig::default();

    if let Some(scaling) = value.get("rope_scaling") {
        if let Some(obj) = scaling.as_object() {
            if let Some(raw) = obj
                .get("type")
                .or_else(|| obj.get("scaling_type"))
                .and_then(Value::as_str)
            {
                config.scaling_type = Some(RopeScalingType::parse(raw));
            }
            config.rope_type = obj
                .get("rope_type")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            config.factor = obj.get("factor").and_then(|v| v.as_f64().map(|n| n as f32));
            let parse_array = |key: &str| -> ModelConfigResult<Option<Vec<f32>>> {
                obj.get(key)
                    .and_then(Value::as_array)
                    .map(|values| to_f32_array(values.as_slice()))
                    .transpose()
            };
            config.factors = parse_array("factors")?
                .or_else(|| parse_array("short_factor").ok().flatten())
                .or_else(|| parse_array("long_factor").ok().flatten());
            config.base = obj
                .get("base")
                .and_then(|v| v.as_f64().map(|n| n as f32))
                .filter(|v| v.is_finite() && *v > 0.0);
            config.original_max_position_embeddings = obj
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());
            config.ext_factor = obj
                .get("ext_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.attn_factor = obj
                .get("attn_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_fast = obj
                .get("beta_fast")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_slow = obj
                .get("beta_slow")
                .and_then(|v| v.as_f64().map(|n| n as f32));
        } else if let Some(factor) = scaling.as_f64().map(|n| n as f32) {
            config.factor = Some(factor);
        } else if let Some(raw) = scaling.as_str() {
            config.scaling_type = Some(RopeScalingType::parse(raw));
        }
    }

    if config.scaling_type.is_none() {
        config.scaling_type = find_string(value, &["rope_type", "rope_scaling_type"])
            .map(|v| RopeScalingType::parse(&v));
    }
    if config.factor.is_none() {
        config.factor = find_f32(value, &["rope_scaling.factor"]);
    }
    if config.factors.is_none() {
        config.factors = find_f32_array(value, &["rope_scaling.factors"]);
    }
    if config.base.is_none() {
        config.base = find_f32(value, &["rope_scaling.base", "rope_base", "rope_theta"]);
    }
    if config.original_max_position_embeddings.is_none() {
        config.original_max_position_embeddings =
            find_usize(value, &["rope_scaling.original_max_position_embeddings"]);
    }

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factor must be positive".to_string(),
            ));
        }
    }
    if let Some(factors) = &config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must contain positive finite values".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn to_f32_array(values: &[Value]) -> ModelConfigResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(v) = value.as_f64().map(|n| n as f32) else {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be numeric".to_string(),
            ));
        };
        if !v.is_finite() {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be finite".to_string(),
            ));
        }
        out.push(v);
    }
    Ok(out)
}

fn rope_scaling_from_gguf(
    reader: &GgufLoader,
    arch: &str,
) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig {
        scaling_type: gguf_arch_str(reader, arch, "rope.scaling.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling"))
            .map(RopeScalingType::parse),
        rope_type: gguf_arch_str(reader, arch, "rope.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling.rope_type"))
            .map(|v| v.to_string()),
        factor: gguf_arch_f32(reader, arch, "rope.scaling.factor"),
        factors: gguf_arch_array_f32(reader, arch, "rope.scaling.factors")
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.short_factor"))
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.long_factor")),
        base: gguf_arch_f32(reader, arch, "rope.scaling.base")
            .or_else(|| gguf_arch_f32(reader, arch, "rope.freq_base")),
        original_max_position_embeddings: gguf_arch_usize(
            reader,
            arch,
            "rope.scaling.original_max_position_embeddings",
        )
        .or_else(|| gguf_arch_usize(reader, arch, "rope.scaling.original_context_length")),
        ext_factor: gguf_arch_f32(reader, arch, "rope.ext_factor"),
        attn_factor: gguf_arch_f32(reader, arch, "rope.attn_factor"),
        beta_fast: gguf_arch_f32(reader, arch, "rope.beta_fast"),
        beta_slow: gguf_arch_f32(reader, arch, "rope.beta_slow"),
    };

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factor".to_string(),
            ));
        }
    }

    if let Some(factors) = &mut config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factors".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn gguf_arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

fn gguf_arch_u64(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<u64> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_u64(&key)
}

fn gguf_arch_usize(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<usize> {
    gguf_arch_u64(reader, arch, suffix).and_then(|v| usize::try_from(v).ok())
}

fn gguf_arch_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<f32> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_f32(&key)
}

fn gguf_arch_str<'a>(reader: &'a GgufLoader, arch: &str, suffix: &str) -> Option<&'a str> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_str(&key)
}

fn gguf_arch_bool(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<bool> {
    let key = gguf_arch_key(arch, suffix);
    let value = reader.get(&key)?;
    value.as_bool().or_else(|| value.as_u64().map(|v| v != 0))
}

fn gguf_arch_array_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<Vec<f32>> {
    let key = gguf_arch_key(arch, suffix);
    let array = reader.get_metadata_array(&key)?;
    let mut out = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let value = item.as_f32()?;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

/// Read a `{arch}.{suffix}` ARRAY metadata as `Vec<u8>`.
///
/// Used for Gemma 4 `attention.pattern` (0=sliding, 1=global) and similar
/// per-layer tag arrays. Items are accepted from any integer GGUF type as long
/// as they fit in u8 (0..=255). Any out-of-range or non-integer item causes the
/// whole read to return `None` (Ω1: no silent truncation).
fn gguf_arch_array_u8(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<Vec<u8>> {
    let key = gguf_arch_key(arch, suffix);
    let array = reader.get_metadata_array(&key)?;
    let mut out = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let value = item.as_u64()?;
        let byte = u8::try_from(value).ok()?;
        out.push(byte);
    }
    Some(out)
}

/// Gemma 4 fallback: derive per-layer attention pattern when GGUF metadata is
/// absent. SPEC §Gemma4: every 6th layer (1-indexed) is global, others are
/// sliding-window. I.e. `(i + 1) % 6 == 0 → 1 (global)`, else `0 (sliding)`.
///
/// This is the ONLY place the default pattern is synthesised. Loader paths,
/// executor fixtures and tests must all go through this helper so the fallback
/// stays centralised (per CLAUDE.md: fallback 逻辑集中在一个函数).
pub fn derive_default_attention_pattern(num_layers: usize) -> Vec<u8> {
    (0..num_layers)
        .map(|i| if (i + 1) % 6 == 0 { 1u8 } else { 0u8 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[derive(Debug)]
    struct MockTensorProvider {
        tensors: Vec<TensorMeta>,
    }

    impl TensorProvider for MockTensorProvider {
        fn tensor_info(&self, name: &str) -> Option<TensorMeta> {
            self.tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .cloned()
        }

        fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
            self.tensors.clone().into_iter()
        }

        fn load_tensor_data(&self, name: &str) -> crate::loader::Result<Cow<'_, [u8]>> {
            let meta = self.tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| crate::loader::LoaderError::MissingTensor(name.to_string()))?;
            let element_size = match meta.dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::U8 => 1,
                safetensors::Dtype::I64 => 8,
                safetensors::Dtype::I32 => 4,
                safetensors::Dtype::F64 => 8,
                safetensors::Dtype::BOOL => 1,
                _ => 2,
            };
            let total_elements: usize = meta.shape.iter().product();
            Ok(Cow::Owned(vec![0u8; total_elements * element_size]))
        }
    }

    fn tensor(name: &str, shape: &[usize]) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: safetensors::Dtype::F16,
        }
    }

    #[test]
    fn derive_config_from_tensors_succeeds_with_unique_head_dim() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000, 2816]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.0.mlp.gate_proj.weight", &[11264, 2816]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.1.mlp.gate_proj.weight", &[11264, 2816]),
            ],
        };

        let derived = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default()).expect("tensor-driven derivation");
        assert_eq!(derived.hidden_size, 2816);
        assert_eq!(derived.vocab_size, 50000);
        assert_eq!(derived.head_dim, 32);
        assert_eq!(derived.num_attention_heads, 88);
        assert_eq!(derived.num_key_value_heads, 11);
        assert_eq!(derived.num_hidden_layers, 2);
        assert_eq!(derived.intermediate_size, Some(11264));
        assert_eq!(derived.dtype, DType::F16);
    }

    #[test]
    fn derive_config_from_tensors_rejects_ambiguous_head_dim() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[32000, 4096]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[1024, 4096]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[4096, 4096]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[1024, 4096]),
                tensor("model.layers.0.mlp.gate_proj.weight", &[11008, 4096]),
                tensor("model.layers.1.mlp.gate_proj.weight", &[11008, 4096]),
            ],
        };

        let err = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default()).expect_err("must reject ambiguity");
        assert!(
            err.to_string().contains("ambiguous"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn derive_config_from_tensors_rejects_cross_layer_mismatch() {
        let provider = MockTensorProvider {
            tensors: vec![
                tensor("model.embed_tokens.weight", &[50000, 2816]),
                tensor("model.layers.0.self_attn.q_proj.weight", &[2816, 2816]),
                tensor("model.layers.0.self_attn.k_proj.weight", &[352, 2816]),
                tensor("model.layers.1.self_attn.q_proj.weight", &[3072, 2816]),
                tensor("model.layers.1.self_attn.k_proj.weight", &[352, 2816]),
            ],
        };

        let err = derive_config_from_tensors_with_hints(&provider, TensorDeriveHints::default()).expect_err("must reject mismatch");
        assert!(
            err.to_string().contains("cross-layer") || err.to_string().contains("ambiguous"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn build_moe_config_deepseek() {
        use crate::manifest::RouterType;
        let cfg = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_hidden_layers: 28,
            intermediate_size: Some(10944),
            num_experts: Some(64),
            num_experts_per_tok: Some(6),
            expert_intermediate_size: Some(1408),
            vocab_size: 102400,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 128,
            head_dim: 128,
            dtype: DType::BF16,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            hidden_act: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            sliding_window: None,
            num_kv_shared_layers: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            mtp_depth: None,
            vision_config: None,
            multimodal_token_ids: None,
        };
        let moe = cfg.build_moe_config("deepseek").unwrap();
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.num_experts_per_tok, 6);
        assert_eq!(moe.router_type, RouterType::DeepSeek);
    }

    // ── GGUF Gemma 4 extraction scaffolding ──────────────────────────────
    //
    // Build a minimal GGUF v3 byte stream with the given KV metadata and no
    // tensors, write it to a temp file, and open it via `GgufReader::open`.
    // This exercises the real public loader surface so any future changes to
    // GGUF parsing stay covered by the same path production uses.
    fn make_gguf_with_meta(kvs: &[(&str, GgufMetaValue)]) -> std::path::PathBuf {
        use crate::loader::gguf::GGUF_MAGIC;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            write_kv_str(&mut buf, k);
            v.write(&mut buf);
        }
        let pos = buf.len();
        let aligned = (pos + 31) & !31;
        buf.resize(aligned, 0u8);

        let mut path = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!(
            "gllm_gguf_gemma4_test_{ts}_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &buf).expect("write temp GGUF");
        path
    }

    fn write_kv_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    enum GgufMetaValue {
        Str(&'static str),
        U64(u64),
        F32(f32),
        ArrU8(Vec<u8>),
    }

    impl GgufMetaValue {
        fn write(&self, buf: &mut Vec<u8>) {
            use crate::loader::gguf::GgufValueType;
            match self {
                Self::Str(s) => {
                    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    write_kv_str(buf, s);
                }
                Self::U64(v) => {
                    buf.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Self::F32(v) => {
                    buf.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                Self::ArrU8(bytes) => {
                    buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    buf.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
                    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                    buf.extend_from_slice(bytes);
                }
            }
        }
    }

    /// TEST-GGUF-GEMMA4-ARCH-U8: gguf_arch_array_u8 正确解码 attention.pattern
    #[test]
    fn gguf_arch_array_u8_reads_attention_pattern() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("gemma4")),
            (
                "gemma4.attention.pattern",
                GgufMetaValue::ArrU8(vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
            ),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        let pattern = gguf_arch_array_u8(&reader, "gemma4", "attention.pattern");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            pattern,
            Some(vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
            "u8 array helper must round-trip attention.pattern bytes"
        );
    }

    /// TEST-GGUF-GEMMA4-ARCH-SCALARS: 各类 arch scalar 读取 (u64 + f32)
    #[test]
    fn gguf_arch_scalars_read_gemma4_fields() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[
            ("general.architecture", GgufMetaValue::Str("gemma4")),
            ("gemma4.attention.sliding_window", GgufMetaValue::U64(512)),
            (
                "gemma4.attention.num_kv_shared_layers",
                GgufMetaValue::U64(4),
            ),
            ("gemma4.attention.global_head_dim", GgufMetaValue::U64(512)),
            (
                "gemma4.rope.global.freq_base",
                GgufMetaValue::F32(1_000_000.0),
            ),
            ("gemma4.rope.partial_ratio", GgufMetaValue::F32(0.25)),
            ("gemma4.embedding.per_layer_input", GgufMetaValue::U64(128)),
        ]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.sliding_window"),
            Some(512)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.num_kv_shared_layers"),
            Some(4)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "attention.global_head_dim"),
            Some(512)
        );
        assert_eq!(
            gguf_arch_f32(&reader, "gemma4", "rope.global.freq_base"),
            Some(1_000_000.0)
        );
        assert_eq!(
            gguf_arch_f32(&reader, "gemma4", "rope.partial_ratio"),
            Some(0.25)
        );
        assert_eq!(
            gguf_arch_usize(&reader, "gemma4", "embedding.per_layer_input"),
            Some(128)
        );
        let _ = std::fs::remove_file(&path);
    }

    /// TEST-GGUF-GEMMA4-ARCH-MISSING: 缺 key 时 helper 返回 None，不 panic
    #[test]
    fn gguf_arch_helpers_return_none_when_missing() {
        use crate::loader::gguf::GgufReader as GgufLoader;
        let path = make_gguf_with_meta(&[(
            "general.architecture",
            GgufMetaValue::Str("gemma4"),
        )]);
        let reader = GgufLoader::open(&path).expect("open test GGUF");
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.sliding_window").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.num_kv_shared_layers").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "attention.global_head_dim").is_none());
        assert!(gguf_arch_f32(&reader, "gemma4", "rope.global.freq_base").is_none());
        assert!(gguf_arch_f32(&reader, "gemma4", "rope.partial_ratio").is_none());
        assert!(gguf_arch_usize(&reader, "gemma4", "embedding.per_layer_input").is_none());
        assert!(gguf_arch_array_u8(&reader, "gemma4", "attention.pattern").is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// TEST-GGUF-GEMMA4-001: derive_default_attention_pattern 每 6 层第 6 层 global
    #[test]
    fn derive_default_attention_pattern_matches_gemma4_rule() {
        // 26 层（Gemma 4 E2B）: idx 5, 11, 17, 23 为 global (即第 6/12/18/24 层)
        let pattern = derive_default_attention_pattern(26);
        assert_eq!(pattern.len(), 26);
        for (i, &v) in pattern.iter().enumerate() {
            let expect = if (i + 1) % 6 == 0 { 1u8 } else { 0u8 };
            assert_eq!(v, expect, "layer {i} expected {expect} got {v}");
        }
        // 少于 6 层 → 全部 sliding
        let small = derive_default_attention_pattern(5);
        assert_eq!(small, vec![0, 0, 0, 0, 0]);
        // 边界: 0 层 → 空 Vec
        assert!(derive_default_attention_pattern(0).is_empty());
        // 恰好 6 层 → 最后一层 global
        let six = derive_default_attention_pattern(6);
        assert_eq!(six, vec![0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn build_moe_config_none_for_dense() {
        let cfg = ModelConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_hidden_layers: 28,
            intermediate_size: Some(10944),
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
            vocab_size: 102400,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 128,
            head_dim: 128,
            dtype: DType::BF16,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            hidden_act: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            sliding_window: None,
            num_kv_shared_layers: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            mtp_depth: None,
            vision_config: None,
            multimodal_token_ids: None,
        };
        assert!(cfg.build_moe_config("llama").is_none());
    }
}
