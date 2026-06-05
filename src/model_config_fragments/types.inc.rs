/// MLA (Multi-head Latent Attention) configuration.
/// DeepSeek V3/R1/Kimi-K2 low-rank KV compression + Matrix Absorption.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// KV compression latent dimension (DeepSeek V3 = 512)
    pub d_c: usize,
    /// Decoupled RoPE dimension (DeepSeek V3 = 64)
    pub d_rope: usize,
    /// Un-absorbed prefill threshold (tokens; short prefill ≤ this → KV restore path)
    pub unabsorbed_threshold: usize,
}

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
    /// Model storage dtype (from config.json torch_dtype or detected from weights).
    pub dtype: DType,
    /// Compute dtype for inference. Defaults to `dtype` but can be overridden by user
    /// to enable native mixed precision (e.g., BF16 model → FP8 compute, F32 → BF16).
    /// All weights are dequantized/converted to this dtype before JIT compilation.
    pub compute_dtype: DType,
    pub norm_eps: f32,

    // ── MoE ──
    pub num_experts: usize,
    pub moe_top_k: usize,
    pub expert_intermediate_size: usize,

    // ── RoPE scaling ──
    pub rope_scaling: Option<RopeScalingConfig>,

    // ── Logit softcapping ──
    /// Final logit softcapping value from config.json.
    pub final_logit_softcapping: Option<f32>,

    // ── FFN activation type ──
    /// From config.json hidden_act / hidden_activation.
    pub hidden_act: Option<HiddenAct>,

    // ── MLA (Multi-head Latent Attention) ──
    /// KV compression latent dimension. 0 = not MLA.
    pub mla_d_c: usize,
    /// Decoupled RoPE dimension. 0 = not MLA.
    pub mla_d_rope: usize,
    /// Un-absorbed prefill threshold. 0 = not MLA.
    pub mla_unabsorbed_threshold: usize,
}

impl ModelGeometry {
    /// Create from ModelConfig + manifest MoE info.
    /// This is the ONLY place model geometry is derived.
    pub fn from_config(config: &ModelConfig, moe_config: Option<crate::manifest::MoEConfig>) -> Self {
        let intermediate_size = match config.intermediate_size {
            Some(is) => is,
            None if config.use_double_wide_mlp.unwrap_or(false) => {
                ((config.hidden_size as f64 * 8.0 / 3.0).round() as usize / 256) * 256
            }
            None => config.hidden_size * 4,
        };
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
            compute_dtype: config.compute_dtype.unwrap_or(config.dtype),
            norm_eps: config.layer_norm_epsilon.unwrap_or(1e-12),
            // position_offset = pad_token_id + 1 (RoBERTa: pad=1→offset=2, BERT: pad=0→offset=1)
            // None 表示模型不需要 position offset（GPT-style decoder 从 0 开始）
            position_offset: config.pad_token_id.map(|pad| (pad + 1) as usize),
            num_experts,
            moe_top_k,
            expert_intermediate_size,
            rope_scaling: config.rope_scaling.clone(),
            final_logit_softcapping: config.final_logit_softcapping,
            hidden_act: config.hidden_act.clone(),
            mla_d_c: config.mla_config.as_ref().map(|c| c.d_c).unwrap_or(0),
            mla_d_rope: config.mla_config.as_ref().map(|c| c.d_rope).unwrap_or(0),
            mla_unabsorbed_threshold: config.mla_config.as_ref().map(|c| c.unabsorbed_threshold).unwrap_or(0),
        }
    }

    /// Whether this is a MoE model.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Whether this is an MLA (Multi-head Latent Attention) model.
    pub fn is_mla(&self) -> bool {
        self.mla_d_c > 0
    }

    /// Per-head KV dimension: standard = num_kv_heads * head_dim, MLA = d_c + d_rope.
    /// This is the total dimension stored per token per layer in the KV cache.
    pub fn kv_dim(&self) -> usize {
        if self.is_mla() {
            self.mla_d_c + self.mla_d_rope
        } else {
            self.num_kv_heads * self.head_dim
        }
    }

    /// KV cache bytes per token (for memory estimation).
    /// Standard: 2 (K+V) × num_kv_heads × head_dim × num_layers × elem_bytes.
    /// MLA: (d_c + d_rope) × num_layers × elem_bytes (single compressed vector, no K/V split).
    pub fn kv_bytes_per_token(&self) -> usize {
        if self.is_mla() {
            self.kv_dim() * self.num_layers * self.dtype.size_bytes()
        } else {
            2 * self.kv_dim() * self.num_layers * self.dtype.size_bytes()
        }
    }

    /// Expert weight bytes (gate + up + down matrices).
    pub fn expert_weight_bytes(&self) -> usize {
        self.hidden_size * self.expert_intermediate_size * 3 * self.dtype.size_bytes()
    }

    /// SharedKvRef (Gemma 4): how many physical KV-cache layers we actually
    /// allocate (= `num_layers - num_kv_shared_layers`). All callers that
    /// stride into the KV cache buffer MUST use this — using the raw
    /// `num_layers` would over-allocate, and using the raw per-op layer index
    /// without donor remapping would read past the buffer for any layer in
    /// the shared tail.
    pub fn effective_kv_layers(&self) -> usize {
        self.num_layers.saturating_sub(self.num_kv_shared_layers).max(1)
    }

    /// SharedKvRef: map a raw per-op layer index (0..num_layers) to its
    /// effective KV-cache layer index (0..effective_kv_layers).
    ///
    /// Non-shared layers return their own index. Shared layers (the last
    /// `num_kv_shared_layers` of the model) are mapped to the nearest
    /// preceding non-shared layer of the same attention type
    /// (sliding vs global), mirroring `KvCacheBuffer::build_kv_donor_map`.
    /// If no matching donor exists (degenerate case), we clamp to the last
    /// effective layer rather than returning an out-of-range index — every
    /// call site writes/reads using `effective * num_kv_heads * max_seq *
    /// head_dim * elem_bytes`, so a clamp keeps the offset inside the
    /// allocated buffer.
    pub fn effective_kv_layer(&self, layer: usize) -> usize {
        let shared_start = self.num_layers.saturating_sub(self.num_kv_shared_layers);
        if layer < shared_start {
            return layer.min(self.effective_kv_layers().saturating_sub(1));
        }
        // Shared layer: find the nearest non-shared layer of the same type.
        let this_type = self.attention_pattern.get(layer).copied().unwrap_or(0);
        for j in (0..shared_start).rev() {
            let j_type = self.attention_pattern.get(j).copied().unwrap_or(0);
            if j_type == this_type {
                return j;
            }
        }
        // Fallback: clamp to the last effective layer.
        self.effective_kv_layers().saturating_sub(1)
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

    pub fn as_str(&self) -> &str {
        match self {
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::GeluNew => "gelu_new",
            Self::Relu => "relu",
            Self::Swish => "swish",
            Self::QuickGelu => "quick_gelu",
            Self::Unknown(s) => s.as_str(),
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
    /// User-specified compute dtype. When Some, overrides `dtype` for weight
    /// dequantization target. Enables native mixed precision:
    /// BF16 model → FP8 compute, F32 model → BF16 compute, etc.
    /// None = use `dtype` (model native precision).
    pub compute_dtype: Option<DType>,
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

    /// MLA (Multi-head Latent Attention) 配置
    /// 从 config.json 的 `kv_lora_rank` + `q_lora_rank` 或 GGUF `deepseek_mla.*` 解析
    pub mla_config: Option<MlaConfig>,

    // ── Multimodal: Vision Encoder (SigLIP) ──
    /// Vision encoder configuration parsed from `"vision_config"` sub-object.
    /// Present only for multimodal models (e.g. Gemma 4).
    pub vision_config: Option<crate::compat::vision_forward::VisionConfig>,

    // ── Multimodal: Audio Encoder (USM Conformer) ──
    /// Audio encoder configuration parsed from `"audio_config"` sub-object.
    /// Present only for multimodal models that expose an audio tower
    /// (e.g. Gemma 4 with USM Conformer).
    pub audio_config: Option<crate::compat::audio_forward::AudioConfig>,

    /// Multimodal special token IDs (image / audio / eoi / eoa).
    ///
    /// **Source rule (T58)**: Must come from the model config or tokenizer
    /// `special_tokens_map.json`; not hard-coded in Rust. When this is
    /// `None`, the model is treated as text-only and calls to
    /// `.image()` / `.audio()` on the generation builder fail fast with
    /// `ClientError::InvalidModelType`.
    pub multimodal_token_ids: Option<crate::compat::multimodal::MultimodalTokenIds>,

    /// Final logit softcapping value (Gemma 4: 30.0, GPT-OSS: 24.0).
    /// From config.json `final_logit_softcapping`.
    pub final_logit_softcapping: Option<f32>,

    /// Whether FFN uses double-wide MLP (intermediate = hidden * 8/3 rounded).
    /// From config.json `use_double_wide_mlp`.
    pub use_double_wide_mlp: Option<bool>,

    /// Whether to add special tokens (BOS/EOS) during tokenization.
    /// From config.json `add_bos_token` or `add_special_tokens`.
    /// Defaults to `true` when not specified.
    pub add_special_tokens: Option<bool>,
}
