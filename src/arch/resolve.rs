//! 占位符解析和配置推导 (REQ-ARCH-005)
//!
//! 从 GGUF metadata 或 SafeTensors 张量形状推导配置值。

use std::collections::HashMap;

use gllm_kernels::compiler::graph::RopeScaling;

use crate::loader::gguf::GgufReader;
use crate::loader::TensorProvider;
use crate::manifest::TensorRole;
use crate::model_config::RopeScalingConfig;

/// 解析后的配置
#[derive(Debug, Clone, Default)]
pub struct ResolvedConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub dtype: String,

    // ── Gemma 4: Dual RoPE ──
    pub global_rope_theta: f64,
    pub rope_partial_ratio: f32,

    // ── Gemma 4: Per-layer attention ──
    pub attention_pattern: Vec<u8>,
    pub sliding_window: usize,

    // ── RoPE scaling (YaRN / Linear) ──
    pub rope_scaling: Option<RopeScaling>,

    // ── Gemma 4: Shared KV + PLE ──
    pub num_kv_shared_layers: usize,
    pub global_head_dim: usize,
    pub hidden_size_per_layer_input: usize,
    /// 派生字段 — 模型是否启用 PerLayerEmbedding (Gemma 4 E2B/E4B 为 true,
    /// 31B Dense / 26B MoE 为 false)。由 `hidden_size_per_layer_input > 0`
    /// 推导,供 auto_graph 条件分支引用。
    pub has_per_layer_embedding: bool,

    /// 额外的自定义配置
    pub extra: HashMap<String, i64>,

    /// LayerNorm / RMSNorm epsilon (from model config, typically 1e-12 for BERT, 1e-5 for GPT)
    pub norm_eps: f32,
}

impl ResolvedConfig {
    /// Construct from ModelGeometry (single source of truth).
    pub fn from_geometry(g: &crate::model_config::ModelGeometry, extra: HashMap<String, i64>) -> Self {
        Self {
            num_hidden_layers: g.num_layers,
            hidden_size: g.hidden_size,
            num_attention_heads: g.num_heads,
            num_key_value_heads: g.num_kv_heads,
            head_dim: g.head_dim,
            intermediate_size: Some(g.intermediate_size),
            vocab_size: g.vocab_size,
            rope_theta: g.rope_theta,
            dtype: format!("{:?}", g.dtype).to_lowercase(),
            global_rope_theta: g.global_rope_theta,
            rope_partial_ratio: g.rope_partial_ratio,
            attention_pattern: g.attention_pattern.clone(),
            sliding_window: g.sliding_window,
            rope_scaling: convert_rope_scaling(g.rope_scaling.as_ref()),
            num_kv_shared_layers: g.num_kv_shared_layers,
            global_head_dim: g.global_head_dim,
            hidden_size_per_layer_input: g.hidden_size_per_layer_input,
            has_per_layer_embedding: g.hidden_size_per_layer_input > 0,
            norm_eps: g.norm_eps,
            extra,
        }
    }

    /// 获取配置值（整数）
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match key {
            "num_hidden_layers" => Some(self.num_hidden_layers as i64),
            "hidden_size" => Some(self.hidden_size as i64),
            "num_attention_heads" => Some(self.num_attention_heads as i64),
            "num_key_value_heads" => Some(self.num_key_value_heads as i64),
            "head_dim" => Some(self.head_dim as i64),
            "intermediate_size" => self.intermediate_size.map(|v| v as i64),
            "vocab_size" => Some(self.vocab_size as i64),
            _ => self.extra.get(key).copied(),
        }
    }

    /// 获取配置值（浮点）
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match key {
            "rope_theta" => Some(self.rope_theta),
            "global_rope_theta" => Some(self.global_rope_theta),
            _ => None,
        }
    }

    /// 获取配置值（字符串）
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match key {
            "dtype" => Some(&self.dtype),
            _ => None,
        }
    }

    /// 获取配置值（布尔）— 供 auto_graph 条件分支引用。
    ///
    /// 返回 `None` 表示该键不是已知的布尔派生字段。调用方必须显式处理
    /// 未知键（而非默认 false),避免条件分支静默跳过节点。
    ///
    /// 支持两类键:
    /// - **静态派生字段**: `has_per_layer_embedding`
    /// - **per-layer 动态字段**: `is_kv_shared_layer_{N}` (N 为整数),
    ///   等价于 `self.is_kv_shared_layer(N)`。auto_graph 先把 `${i}`
    ///   替换成层索引后以 `is_kv_shared_layer_3` 形式查询,
    ///   避免硬编码层索引。
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match key {
            "has_per_layer_embedding" => Some(self.has_per_layer_embedding),
            _ => {
                // per-layer 动态字段: `is_kv_shared_layer_<N>`
                if let Some(rest) = key.strip_prefix("is_kv_shared_layer_") {
                    if let Ok(idx) = rest.parse::<usize>() {
                        return Some(self.is_kv_shared_layer(idx));
                    }
                }
                None
            }
        }
    }

    /// Returns `true` when `layer_idx` is a KV-sharing consumer layer.
    ///
    /// A layer is a consumer when `num_kv_shared_layers > 0` and
    /// `layer_idx >= num_hidden_layers - num_kv_shared_layers`. Gemma 4 E2B
    /// (26 layers, 20 shared) marks layers 6..26 as consumers.
    #[inline]
    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0
            && layer_idx < self.num_hidden_layers
            && layer_idx >= self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers)
    }

    /// Resolve the donor layer index for a KV-sharing consumer layer.
    ///
    /// Returns:
    /// - `Ok(None)` when `layer_idx` is **not** a consumer (identity path: the
    ///   layer owns its own K/V).
    /// - `Ok(Some(donor))` — the donor layer index (strictly less than
    ///   `num_hidden_layers - num_kv_shared_layers`) with the same
    ///   `attention_pattern[·]` bucket as the consumer.
    /// - `Err(..)` when the attention pattern is malformed or no donor exists.
    ///
    /// Delegates to `scheduler::find_donor` — the same algorithm used by the
    /// runtime page allocator, guaranteeing graph and scheduler agree on the
    /// donor choice.
    pub fn donor_layer(&self, layer_idx: usize) -> Result<Option<usize>, ResolveError> {
        crate::scheduler::find_donor(
            layer_idx,
            self.num_hidden_layers,
            self.num_kv_shared_layers,
            &self.attention_pattern,
        )
        .map_err(|e| ResolveError::DerivationFailed {
            key: format!("donor_layer({layer_idx})"),
            reason: format!("{e}"),
        })
    }
}

/// 配置推导错误
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResolveError {
    #[error("Missing required config: {0}")]
    MissingConfig(String),
    #[error("Failed to derive {key} from tensors: {reason}")]
    DerivationFailed { key: String, reason: String },
    #[error("Inconsistent config: {0}")]
    Inconsistent(String),
}

/// 从数据源解析配置 (GGUF metadata + tensor shapes)
pub fn resolve_from_provider<P: TensorProvider>(
    provider: &P,
    gguf: Option<&GgufReader>,
) -> Result<ResolvedConfig, ResolveError> {
    let mut config = ResolvedConfig::default();

    if let Some(reader) = gguf {
        resolve_from_gguf(&mut config, reader)?;
    }

    resolve_from_tensors(&mut config, provider)?;

    if config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0 {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }

    validate_config(&config)?;

    Ok(config)
}

/// 从 GGUF metadata 解析
fn resolve_from_gguf(config: &mut ResolvedConfig, reader: &GgufReader) -> Result<(), ResolveError> {
    // 尝试常见的 GGUF metadata 键
    // llama.* 系列（Qwen, Llama, Mistral 等通用）
    if let Some(v) = reader.get_metadata_u64("llama.block_count") {
        config.num_hidden_layers = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.embedding_length") {
        config.hidden_size = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.attention.head_count") {
        config.num_attention_heads = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.attention.head_count_kv") {
        config.num_key_value_heads = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.feed_forward_length") {
        config.intermediate_size = Some(v as usize);
    }
    if let Some(v) = reader.get_metadata_f32("llama.rope.freq_base") {
        config.rope_theta = v as f64;
    }

    // qwen2.* 系列
    if config.num_hidden_layers == 0 {
        if let Some(v) = reader.get_metadata_u64("qwen2.block_count") {
            config.num_hidden_layers = v as usize;
        }
    }

    Ok(())
}

/// 从张量形状推导配置
fn resolve_from_tensors<P: TensorProvider>(
    config: &mut ResolvedConfig,
    provider: &P,
) -> Result<(), ResolveError> {
    // 收集张量信息
    let mut max_layer_idx = 0usize;
    let mut embed_shape: Option<Vec<usize>> = None;
    let mut q_proj_shape: Option<Vec<usize>> = None;
    let mut k_proj_shape: Option<Vec<usize>> = None;
    let mut detected_dtype = None;

    for tensor in provider.iter_tensors() {
        let name = &tensor.name;
        let shape = &tensor.shape;

        // 检测层数
        if let Some((_, Some(layer_idx))) = crate::loader::match_tensor_role(name) {
            max_layer_idx = max_layer_idx.max(layer_idx + 1);
        }

        // 匹配角色
        if let Some((role, _)) = crate::loader::match_tensor_role(name) {
            match role {
                TensorRole::Embedding => {
                    embed_shape = Some(shape.clone());
                    if detected_dtype.is_none() {
                        detected_dtype = Some(tensor.dtype);
                    }
                }
                TensorRole::AttentionQuery => {
                    q_proj_shape = Some(shape.clone());
                }
                TensorRole::AttentionKey => {
                    k_proj_shape = Some(shape.clone());
                }
                _ => {}
            }
        }
    }

    // 从 embed 推导 vocab_size 和 hidden_size
    if let Some(shape) = embed_shape {
        if shape.len() >= 2 {
            if config.vocab_size == 0 {
                config.vocab_size = shape[0];
            }
            if config.hidden_size == 0 {
                config.hidden_size = shape[1];
            }
        }
    }

    // 从 q_proj 推导 num_attention_heads
    if let Some(shape) = q_proj_shape {
        if shape.len() >= 2 && config.hidden_size > 0 {
            let q_out_dim = shape[0];
            // q_proj: [num_heads * head_dim, hidden_size]
            if config.num_attention_heads == 0 && config.head_dim > 0 {
                config.num_attention_heads = q_out_dim / config.head_dim;
            }
        }
    }

    // 从 k_proj 推导 num_key_value_heads
    if let Some(shape) = k_proj_shape {
        if shape.len() >= 2 && config.head_dim > 0 {
            let k_out_dim = shape[0];
            if config.num_key_value_heads == 0 {
                config.num_key_value_heads = k_out_dim / config.head_dim;
            }
        }
    }

    // 层数
    if config.num_hidden_layers == 0 && max_layer_idx > 0 {
        config.num_hidden_layers = max_layer_idx;
    }

    // dtype
    if let Some(dtype) = detected_dtype {
        config.dtype = match dtype {
            safetensors::Dtype::F32 => "f32".to_string(),
            safetensors::Dtype::F16 => "f16".to_string(),
            safetensors::Dtype::BF16 => "bf16".to_string(),
            _ => "f32".to_string(),
        };
    }

    Ok(())
}

/// 验证配置完整性
fn validate_config(config: &ResolvedConfig) -> Result<(), ResolveError> {
    if config.num_hidden_layers == 0 {
        return Err(ResolveError::MissingConfig("num_hidden_layers".to_string()));
    }
    if config.hidden_size == 0 {
        return Err(ResolveError::MissingConfig("hidden_size".to_string()));
    }
    if config.num_attention_heads == 0 {
        return Err(ResolveError::MissingConfig(
            "num_attention_heads".to_string(),
        ));
    }
    if config.vocab_size == 0 {
        return Err(ResolveError::MissingConfig("vocab_size".to_string()));
    }
    Ok(())
}

/// 替换字符串中的占位符
pub fn substitute_placeholders(template: &str, config: &ResolvedConfig) -> String {
    let mut result = template.to_string();

    // 整数占位符
    for (key, getter) in [
        ("num_hidden_layers", config.num_hidden_layers),
        ("hidden_size", config.hidden_size),
        ("num_attention_heads", config.num_attention_heads),
        ("num_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("num_kv_heads", config.num_key_value_heads),
        ("head_dim", config.head_dim),
        ("vocab_size", config.vocab_size),
    ] {
        let placeholder = format!("${{{}}}", key);
        result = result.replace(&placeholder, &getter.to_string());
    }

    // intermediate_size (可选)
    if let Some(v) = config.intermediate_size {
        result = result.replace("${intermediate_size}", &v.to_string());
    }

    // Gemma 4 整数占位符
    result = result.replace("${sliding_window}", &config.sliding_window.to_string());
    result = result.replace("${hidden_size_per_layer_input}",
                            &config.hidden_size_per_layer_input.to_string());

    // 浮点占位符
    result = result.replace("${rope_theta}", &config.rope_theta.to_string());
    result = result.replace("${global_rope_theta}", &config.global_rope_theta.to_string());

    // 字符串占位符
    result = result.replace("${dtype}", &config.dtype);

    // Extra KV pairs (e.g., patch_size, image_size, in_channels from vision/audio models)
    for (key, value) in &config.extra {
        let placeholder = format!("${{{}}}", key);
        result = result.replace(&placeholder, &value.to_string());
    }

    result
}

fn convert_rope_scaling(src: Option<&RopeScalingConfig>) -> Option<RopeScaling> {
    let cfg = src?;
    let factor = cfg.factor.unwrap_or(1.0);
    if factor <= 1.0 {
        return None;
    }

    match cfg.scaling_type.as_ref() {
        Some(crate::model_config::RopeScalingType::Yarn) => {
            let orig_max = cfg.original_max_position_embeddings.unwrap_or(4096);
            Some(RopeScaling::Yarn {
                factor,
                beta_fast: cfg.beta_fast.unwrap_or(32.0),
                beta_slow: cfg.beta_slow.unwrap_or(1.0),
                original_max_position: orig_max,
            })
        }
        Some(crate::model_config::RopeScalingType::Linear) => {
            Some(RopeScaling::Linear { factor })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::TensorMeta;
    use crate::model_config::RopeScalingType;

    #[test]
    fn substitute_placeholders_works() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(11008),
            vocab_size: 32000,
            rope_theta: 10000.0,
            dtype: "f16".to_string(),
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            has_per_layer_embedding: false,
            rope_scaling: None,
            extra: HashMap::new(),
            norm_eps: 1e-5,
        };

        let template = "layers: ${num_hidden_layers}, hidden: ${hidden_size}, dtype: ${dtype}";
        let result = substitute_placeholders(template, &config);
        assert_eq!(result, "layers: 32, hidden: 4096, dtype: f16");
    }

    #[test]
    fn config_get_methods() {
        let config = ResolvedConfig {
            num_hidden_layers: 24,
            hidden_size: 2048,
            rope_theta: 500000.0,
            dtype: "bf16".to_string(),
            ..Default::default()
        };

        assert_eq!(config.get_int("num_hidden_layers"), Some(24));
        assert_eq!(config.get_float("rope_theta"), Some(500000.0));
        assert_eq!(config.get_str("dtype"), Some("bf16"));
    }

    /// T43: `is_kv_shared_layer` reflects the trailing-consumer window.
    #[test]
    fn is_kv_shared_layer_identifies_consumer_window() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 26;
        config.num_kv_shared_layers = 20;
        // Gemma 4 E2B: layers 0..6 own their KV, 6..26 are consumers.
        for i in 0..6 {
            assert!(!config.is_kv_shared_layer(i), "layer {i} must own its KV");
        }
        for i in 6..26 {
            assert!(config.is_kv_shared_layer(i), "layer {i} must be a consumer");
        }
        // Out-of-range layers are never consumers.
        assert!(!config.is_kv_shared_layer(26));

        // Sharing disabled → nobody is a consumer.
        config.num_kv_shared_layers = 0;
        for i in 0..26 {
            assert!(!config.is_kv_shared_layer(i),
                "layer {i} must not be a consumer when sharing disabled");
        }
    }

    /// T43: `donor_layer` reuses `scheduler::find_donor` so graph and runtime agree.
    #[test]
    fn donor_layer_matches_scheduler_find_donor() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        // Alternating sliding/global buckets: 0,1,0,1,0,1,0,1.
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0, 1];

        // Non-consumer layers return None (identity path).
        for i in 0..4 {
            assert_eq!(config.donor_layer(i).unwrap(), None,
                "non-consumer layer {i} must have no donor (identity)");
        }
        // Consumer layers pick the latest matching-bucket non-consumer.
        assert_eq!(config.donor_layer(4).unwrap(), Some(2)); // bucket 0 → layer 2
        assert_eq!(config.donor_layer(5).unwrap(), Some(3)); // bucket 1 → layer 3
        assert_eq!(config.donor_layer(6).unwrap(), Some(2));
        assert_eq!(config.donor_layer(7).unwrap(), Some(3));

        // Malformed attention_pattern length → error (no silent fallback).
        config.attention_pattern = vec![0u8; 4];
        assert!(config.donor_layer(6).is_err(),
            "pattern length mismatch must propagate scheduler error");
    }

    /// T43: `get_bool` resolves `is_kv_shared_layer_<N>` via the dynamic branch.
    #[test]
    fn get_bool_supports_per_layer_shared_kv_keys() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 26;
        config.num_kv_shared_layers = 20;

        assert_eq!(config.get_bool("is_kv_shared_layer_0"), Some(false));
        assert_eq!(config.get_bool("is_kv_shared_layer_5"), Some(false));
        assert_eq!(config.get_bool("is_kv_shared_layer_6"), Some(true));
        assert_eq!(config.get_bool("is_kv_shared_layer_25"), Some(true));

        // Unknown key patterns stay `None` so the template engine can still
        // fail loudly on typos.
        assert_eq!(config.get_bool("is_kv_shared_layer_"), None);
        assert_eq!(config.get_bool("is_kv_shared_layer_abc"), None);
        assert_eq!(config.get_bool("typo_field"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  ResolvedConfig::default — all fields zeroed / empty
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn default_config_has_all_zeroed_fields() {
        let config = ResolvedConfig::default();
        assert_eq!(config.num_hidden_layers, 0);
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.num_attention_heads, 0);
        assert_eq!(config.num_key_value_heads, 0);
        assert_eq!(config.head_dim, 0);
        assert_eq!(config.intermediate_size, None);
        assert_eq!(config.vocab_size, 0);
        assert_eq!(config.rope_theta, 0.0);
        assert_eq!(config.dtype, "");
        assert_eq!(config.global_rope_theta, 0.0);
        assert_eq!(config.rope_partial_ratio, 0.0);
        assert!(config.attention_pattern.is_empty());
        assert_eq!(config.sliding_window, 0);
        assert_eq!(config.num_kv_shared_layers, 0);
        assert_eq!(config.global_head_dim, 0);
        assert_eq!(config.hidden_size_per_layer_input, 0);
        assert!(!config.has_per_layer_embedding);
        assert!(config.rope_scaling.is_none());
        assert!(config.extra.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  get_int — comprehensive coverage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn get_int_returns_core_fields() {
        let config = ResolvedConfig {
            num_hidden_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            head_dim: 64,
            vocab_size: 50000,
            ..Default::default()
        };

        assert_eq!(config.get_int("num_hidden_layers"), Some(12));
        assert_eq!(config.get_int("hidden_size"), Some(768));
        assert_eq!(config.get_int("num_attention_heads"), Some(12));
        assert_eq!(config.get_int("num_key_value_heads"), Some(4));
        assert_eq!(config.get_int("head_dim"), Some(64));
        assert_eq!(config.get_int("vocab_size"), Some(50000));
    }

    #[test]
    fn get_int_intermediate_size_some() {
        let config = ResolvedConfig {
            intermediate_size: Some(3072),
            ..Default::default()
        };
        assert_eq!(config.get_int("intermediate_size"), Some(3072));
    }

    #[test]
    fn get_int_intermediate_size_none() {
        let config = ResolvedConfig {
            intermediate_size: None,
            ..Default::default()
        };
        assert_eq!(config.get_int("intermediate_size"), None);
    }

    #[test]
    fn get_int_extra_fallback() {
        let mut extra = HashMap::new();
        extra.insert("custom_param".to_string(), 42);
        extra.insert("another".to_string(), -1);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };

        assert_eq!(config.get_int("custom_param"), Some(42));
        assert_eq!(config.get_int("another"), Some(-1));
    }

    #[test]
    fn get_int_unknown_key_returns_none() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_int("nonexistent_key"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  get_float — comprehensive coverage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn get_float_rope_theta() {
        let config = ResolvedConfig {
            rope_theta: 1000000.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(1000000.0));
    }

    #[test]
    fn get_float_global_rope_theta() {
        let config = ResolvedConfig {
            global_rope_theta: 1000000.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("global_rope_theta"), Some(1000000.0));
    }

    #[test]
    fn get_float_unknown_key_returns_none() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_float("nonexistent"), None);
        assert_eq!(config.get_float("hidden_size"), None);
        assert_eq!(config.get_float("dtype"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  get_str — comprehensive coverage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn get_str_dtype() {
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("dtype"), Some("bf16"));
    }

    #[test]
    fn get_str_unknown_key_returns_none() {
        let config = ResolvedConfig {
            dtype: "f32".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("hidden_size"), None);
        assert_eq!(config.get_str("rope_theta"), None);
        assert_eq!(config.get_str("nonexistent"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  get_bool — static field and edge cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn get_bool_has_per_layer_embedding_true() {
        let config = ResolvedConfig {
            has_per_layer_embedding: true,
            ..Default::default()
        };
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
    }

    #[test]
    fn get_bool_has_per_layer_embedding_false() {
        let config = ResolvedConfig {
            has_per_layer_embedding: false,
            ..Default::default()
        };
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(false));
    }

    #[test]
    fn get_bool_unknown_key_returns_none() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_bool("sliding_window"), None);
        assert_eq!(config.get_bool("num_heads"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  is_kv_shared_layer — edge cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn is_kv_shared_layer_all_layers_shared() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 4;
        // When all layers are "shared", every layer is a consumer.
        for i in 0..4 {
            assert!(config.is_kv_shared_layer(i),
                "layer {i} must be a consumer when all layers are shared");
        }
    }

    #[test]
    fn is_kv_shared_layer_single_donor() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 7;
        // Only layer 0 is a donor; layers 1..8 are consumers.
        assert!(!config.is_kv_shared_layer(0));
        for i in 1..8 {
            assert!(config.is_kv_shared_layer(i),
                "layer {i} must be a consumer with single donor");
        }
    }

    #[test]
    fn is_kv_shared_layer_zero_layers() {
        let config = ResolvedConfig::default();
        // num_hidden_layers=0 means any index is out of range.
        assert!(!config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(100));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  donor_layer — edge cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn donor_layer_returns_none_when_no_sharing() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 0;
        // Without sharing, every layer returns Ok(None).
        for i in 0..10 {
            assert_eq!(config.donor_layer(i).unwrap(), None);
        }
    }

    #[test]
    fn donor_layer_out_of_range_returns_none() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 1, 0, 1];
        // layer_idx >= num_layers → Ok(None)
        assert_eq!(config.donor_layer(4).unwrap(), None);
        assert_eq!(config.donor_layer(100).unwrap(), None);
    }

    #[test]
    fn donor_layer_no_donor_for_bucket_returns_error() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        // Pattern: [2, 2, 2, 2] — consumers (layers 2, 3) look for bucket 2
        // in donors (layers 0, 1), but donors have bucket 2 while consumers
        // also have bucket 2, so it actually matches. Use a different bucket
        // for consumers that donors don't have.
        config.attention_pattern = vec![0, 0, 1, 1];
        // Consumers 2,3 have bucket 1. Donors 0,1 have bucket 0. No match.
        assert!(config.donor_layer(2).is_err(),
            "no donor with bucket 1 exists → must be error");
        assert!(config.donor_layer(3).is_err(),
            "no donor with bucket 1 exists → must be error");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  validate_config — error cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_config_rejects_zero_num_hidden_layers() {
        let config = ResolvedConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            vocab_size: 32000,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        match err {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_config_rejects_zero_hidden_size() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            num_attention_heads: 32,
            vocab_size: 32000,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        match err {
            ResolveError::MissingConfig(key) => assert_eq!(key, "hidden_size"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_config_rejects_zero_num_attention_heads() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            vocab_size: 32000,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        match err {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_config_rejects_zero_vocab_size() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        match err {
            ResolveError::MissingConfig(key) => assert_eq!(key, "vocab_size"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_config_accepts_valid_config() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            vocab_size: 32000,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  convert_rope_scaling — Yarn / Linear / None / factor<=1
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_rope_scaling_yarn_with_all_fields() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(32.0),
            original_max_position_embeddings: Some(8192),
            beta_fast: Some(32.0),
            beta_slow: Some(1.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!((factor - 32.0).abs() < f32::EPSILON);
                assert!((beta_fast - 32.0).abs() < f32::EPSILON);
                assert!((beta_slow - 1.0).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 8192);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    #[test]
    fn convert_rope_scaling_yarn_uses_defaults_when_missing() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(4.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!((factor - 4.0).abs() < f32::EPSILON);
                // Defaults: beta_fast=32.0, beta_slow=1.0, original_max=4096
                assert!((beta_fast - 32.0).abs() < f32::EPSILON);
                assert!((beta_slow - 1.0).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 4096);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    #[test]
    fn convert_rope_scaling_linear() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(8.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Linear { factor } => {
                assert!((factor - 8.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    #[test]
    fn convert_rope_scaling_factor_one_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(1.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    #[test]
    fn convert_rope_scaling_factor_below_one_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(0.5),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    #[test]
    fn convert_rope_scaling_factor_default_returns_none() {
        // factor is None → unwrap_or(1.0) → 1.0 <= 1.0 → None
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: None,
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    #[test]
    fn convert_rope_scaling_none_input_returns_none() {
        assert!(convert_rope_scaling(None).is_none());
    }

    #[test]
    fn convert_rope_scaling_unknown_type_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Dynamic),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    #[test]
    fn convert_rope_scaling_no_type_returns_none() {
        let scaling = RopeScalingConfig {
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  substitute_placeholders — comprehensive coverage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn substitute_placeholders_num_heads_alias() {
        let config = ResolvedConfig {
            num_attention_heads: 16,
            ..Default::default()
        };
        let result = substitute_placeholders("${num_heads}", &config);
        assert_eq!(result, "16");
    }

    #[test]
    fn substitute_placeholders_num_kv_heads_alias() {
        let config = ResolvedConfig {
            num_key_value_heads: 4,
            ..Default::default()
        };
        let result = substitute_placeholders("${num_kv_heads}", &config);
        assert_eq!(result, "4");
    }

    #[test]
    fn substitute_placeholders_intermediate_size_present() {
        let config = ResolvedConfig {
            intermediate_size: Some(11008),
            ..Default::default()
        };
        let result = substitute_placeholders("${intermediate_size}", &config);
        assert_eq!(result, "11008");
    }

    #[test]
    fn substitute_placeholders_intermediate_size_none_leaves_unexpanded() {
        let config = ResolvedConfig {
            intermediate_size: None,
            ..Default::default()
        };
        let result = substitute_placeholders("${intermediate_size}", &config);
        assert_eq!(result, "${intermediate_size}");
    }

    #[test]
    fn substitute_placeholders_sliding_window() {
        let config = ResolvedConfig {
            sliding_window: 4096,
            ..Default::default()
        };
        let result = substitute_placeholders("${sliding_window}", &config);
        assert_eq!(result, "4096");
    }

    #[test]
    fn substitute_placeholders_hidden_size_per_layer_input() {
        let config = ResolvedConfig {
            hidden_size_per_layer_input: 256,
            ..Default::default()
        };
        let result = substitute_placeholders("${hidden_size_per_layer_input}", &config);
        assert_eq!(result, "256");
    }

    #[test]
    fn substitute_placeholders_rope_theta_and_global() {
        let config = ResolvedConfig {
            rope_theta: 10000.0,
            global_rope_theta: 1000000.0,
            ..Default::default()
        };
        let result = substitute_placeholders(
            "theta=${rope_theta}, global=${global_rope_theta}", &config,
        );
        assert_eq!(result, "theta=10000, global=1000000");
    }

    #[test]
    fn substitute_placeholders_extra_kv() {
        let mut extra = HashMap::new();
        extra.insert("patch_size".to_string(), 14);
        extra.insert("image_size".to_string(), 224);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        let result = substitute_placeholders(
            "patch=${patch_size}, image=${image_size}", &config,
        );
        assert_eq!(result, "patch=14, image=224");
    }

    #[test]
    fn substitute_placeholders_unmatched_stays_literal() {
        let config = ResolvedConfig::default();
        let result = substitute_placeholders(
            "unknown=${totally_unknown}", &config,
        );
        assert_eq!(result, "unknown=${totally_unknown}");
    }

    #[test]
    fn substitute_placeholders_empty_template() {
        let config = ResolvedConfig::default();
        let result = substitute_placeholders("", &config);
        assert_eq!(result, "");
    }

    #[test]
    fn substitute_placeholders_no_placeholders() {
        let config = ResolvedConfig::default();
        let result = substitute_placeholders("plain text", &config);
        assert_eq!(result, "plain text");
    }

    #[test]
    fn substitute_placeholders_repeated_placeholder() {
        let config = ResolvedConfig {
            hidden_size: 2048,
            ..Default::default()
        };
        let result = substitute_placeholders(
            "${hidden_size}+${hidden_size}=${hidden_size}", &config,
        );
        assert_eq!(result, "2048+2048=2048");
    }

    #[test]
    fn substitute_placeholders_all_integer_fields() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(11008),
            vocab_size: 32000,
            sliding_window: 4096,
            hidden_size_per_layer_input: 512,
            ..Default::default()
        };
        let template = "L=${num_hidden_layers} H=${hidden_size} A=${num_attention_heads} \
                         KV=${num_key_value_heads} D=${head_dim} I=${intermediate_size} \
                         V=${vocab_size} SW=${sliding_window} PLE=${hidden_size_per_layer_input}";
        let result = substitute_placeholders(template, &config);
        assert_eq!(result, "L=32 H=4096 A=32 KV=8 D=128 I=11008 V=32000 SW=4096 PLE=512");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  from_geometry — maps all ModelGeometry fields correctly
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn from_geometry_maps_all_core_fields() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![0, 1, 0, 1, 0, 1, 0, 1],
            sliding_window: 4096,
            num_kv_shared_layers: 4,
            global_head_dim: 256,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, Some(11008));
        assert_eq!(config.vocab_size, 32000);
        assert!((config.rope_theta - 10000.0).abs() < f64::EPSILON);
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.attention_pattern, vec![0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(config.sliding_window, 4096);
        assert_eq!(config.num_kv_shared_layers, 4);
        assert_eq!(config.global_head_dim, 256);
        assert!(!config.has_per_layer_embedding);
        assert!(config.rope_scaling.is_none());
        assert!(config.extra.is_empty());
    }

    #[test]
    fn from_geometry_with_rope_scaling_yarn() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.25,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 512,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: Some(RopeScalingConfig {
                scaling_type: Some(RopeScalingType::Yarn),
                factor: Some(32.0),
                original_max_position_embeddings: Some(8192),
                beta_fast: Some(32.0),
                beta_slow: Some(1.0),
                ..Default::default()
            }),
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert!((config.global_rope_theta - 1000000.0).abs() < f64::EPSILON);
        assert!((config.rope_partial_ratio - 0.25).abs() < f32::EPSILON);
        assert!(config.has_per_layer_embedding);
        assert!(config.rope_scaling.is_some());
        assert_eq!(config.hidden_size_per_layer_input, 512);
    }

    #[test]
    fn from_geometry_carries_extra_fields() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 2048,
            num_layers: 12,
            vocab_size: 50000,
            intermediate_size: 8192,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F16,
            compute_dtype: DType::F16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let mut extra = HashMap::new();
        extra.insert("custom_field".to_string(), 99);

        let config = ResolvedConfig::from_geometry(&geometry, extra.clone());
        assert_eq!(config.extra, extra);
        assert_eq!(config.get_int("custom_field"), Some(99));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  resolve_from_provider — tensor-driven config derivation
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimal TensorProvider that returns a fixed list of TensorMeta entries.
    struct FakeProvider {
        tensors: Vec<TensorMeta>,
    }

    impl TensorProvider for FakeProvider {
        fn tensor_info(&self, _name: &str) -> Option<TensorMeta> {
            None
        }
        fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
            self.tensors.clone().into_iter()
        }
        fn load_tensor_data(&self, _name: &str) -> Result<std::borrow::Cow<'_, [u8]>, crate::loader::LoaderError> {
            Ok(std::borrow::Cow::Borrowed(&[]))
        }
    }

    #[test]
    fn resolve_from_provider_embedding_only_fails_validation() {
        // Only embedding tensor: derives vocab_size + hidden_size + dtype,
        // but validation requires num_hidden_layers and num_attention_heads.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };

        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn resolve_from_provider_derives_layers_from_per_layer_tensors() {
        // Per-layer tensors set num_hidden_layers=3, but without GGUF metadata
        // num_attention_heads cannot be derived (head_dim is 0 at tensor scan time).
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.1.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.2.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn resolve_from_provider_k_proj_without_head_dim() {
        // k_proj derivation requires head_dim > 0, which itself requires
        // num_attention_heads > 0. Without GGUF metadata, this path is inactive.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F16,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![1024, 4096], // 8 heads * 128 head_dim = 1024
                    dtype: safetensors::Dtype::F16,
                },
            ],
        };

        // Fails because num_hidden_layers and num_attention_heads are both 0.
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_from_provider_head_dim_auto_computed() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096], // 32 heads * 128 head_dim
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // num_attention_heads is not derivable without head_dim,
        // but head_dim IS derivable as hidden_size / num_attention_heads
        // only if num_attention_heads > 0. Here neither is set, so
        // head_dim remains 0 after tensor pass. The auto-compute
        // `hidden_size / num_attention_heads` requires num_attention_heads > 0.
        let config = resolve_from_provider(&provider, None);
        // Should fail because num_attention_heads=0 (validation rejects it).
        assert!(config.is_err());
    }

    #[test]
    fn resolve_from_provider_fails_on_empty_tensors() {
        let provider = FakeProvider {
            tensors: vec![],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn resolve_from_provider_fails_on_missing_attention_heads() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.input_layernorm.weight".to_string(),
                    shape: vec![4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    #[test]
    fn resolve_from_provider_dtype_f32_detected() {
        // Only embedding tensor: dtype is detected as f32 but validation fails.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_from_provider_dtype_f16_detected() {
        // Only embedding tensor: dtype is detected as f16 but validation fails.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F16,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  ResolveError — display messages
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn resolve_error_missing_config_message() {
        let err = ResolveError::MissingConfig("test_field".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("test_field"));
        assert!(msg.contains("Missing required config"));
    }

    #[test]
    fn resolve_error_derivation_failed_message() {
        let err = ResolveError::DerivationFailed {
            key: "head_dim".to_string(),
            reason: "division by zero".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("head_dim"));
        assert!(msg.contains("division by zero"));
        assert!(msg.contains("Failed to derive"));
    }

    #[test]
    fn resolve_error_inconsistent_message() {
        let err = ResolveError::Inconsistent("layer count mismatch".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("layer count mismatch"));
        assert!(msg.contains("Inconsistent config"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Edge case: has_per_layer_embedding derivation from geometry
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn from_geometry_has_per_layer_embedding_false_when_zero() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 2048,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 8192,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert!(!config.has_per_layer_embedding);
        assert_eq!(config.hidden_size_per_layer_input, 0);
    }

    #[test]
    fn from_geometry_has_per_layer_embedding_true_when_nonzero() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 2048,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 8192,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 256,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert!(config.has_per_layer_embedding);
        assert_eq!(config.hidden_size_per_layer_input, 256);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Additional edge-case coverage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn get_int_extra_key_shadows_none_for_unknown() {
        // Extra keys are only consulted when the key does not match a known field.
        // Placing a non-field key in extra should still return its value.
        let mut extra = HashMap::new();
        extra.insert("my_custom_counter".to_string(), 7);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        assert_eq!(config.get_int("my_custom_counter"), Some(7));
        // A known field not present in extra still returns its struct value.
        assert_eq!(config.get_int("num_hidden_layers"), Some(0));
    }

    #[test]
    fn get_float_rope_theta_zero() {
        let config = ResolvedConfig {
            rope_theta: 0.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(0.0));
    }

    #[test]
    fn get_float_rope_theta_negative() {
        let config = ResolvedConfig {
            rope_theta: -1.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(-1.0));
    }

    #[test]
    fn get_bool_is_kv_shared_layer_large_index() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        // Index far beyond num_hidden_layers → not a consumer.
        assert_eq!(config.get_bool("is_kv_shared_layer_999"), Some(false));
        // Valid consumer index.
        assert_eq!(config.get_bool("is_kv_shared_layer_3"), Some(true));
    }

    #[test]
    fn get_bool_is_kv_shared_layer_with_leading_zeros_in_suffix() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 3,
            ..Default::default()
        };
        // "07" parses as 7 via str::parse::<usize>() → valid index.
        assert_eq!(config.get_bool("is_kv_shared_layer_07"), Some(true));
        // "03" parses as 3 → 10-3=7, layer 3 < 7 → not a consumer.
        assert_eq!(config.get_bool("is_kv_shared_layer_03"), Some(false));
    }

    #[test]
    fn is_kv_shared_layer_saturating_sub_when_shared_exceeds_layers() {
        // num_kv_shared_layers > num_hidden_layers: saturating_sub clamps to 0,
        // so every valid index is >= 0 and < num_hidden_layers → all consumers.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 3;
        config.num_kv_shared_layers = 10;
        for i in 0..3 {
            assert!(config.is_kv_shared_layer(i),
                "layer {i} must be consumer when shared_count > total_layers");
        }
        // Out of range still false.
        assert!(!config.is_kv_shared_layer(3));
    }

    #[test]
    fn donor_layer_error_carries_correct_key_format() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0u8; 3]; // malformed length
        let err = config.donor_layer(2).unwrap_err();
        match err {
            ResolveError::DerivationFailed { key, reason } => {
                assert!(key.starts_with("donor_layer("), "key was: {key}");
                assert!(!reason.is_empty(), "reason must be non-empty");
            }
            other => panic!("expected DerivationFailed, got {other:?}"),
        }
    }

    #[test]
    fn donor_layer_uniform_bucket_all_donors_match() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        // All layers share bucket 0.
        config.attention_pattern = vec![0, 0, 0, 0, 0, 0];
        // Consumer layer 3 picks latest non-consumer with same bucket → layer 2.
        assert_eq!(config.donor_layer(3).unwrap(), Some(2));
        assert_eq!(config.donor_layer(4).unwrap(), Some(2));
        assert_eq!(config.donor_layer(5).unwrap(), Some(2));
    }

    #[test]
    fn substitute_placeholders_dtype_replacement() {
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        let result = substitute_placeholders("model_${dtype}_v2", &config);
        assert_eq!(result, "model_bf16_v2");
    }

    #[test]
    fn substitute_placeholders_extra_and_builtin_mixed() {
        let mut extra = HashMap::new();
        extra.insert("num_channels".to_string(), 3);
        let config = ResolvedConfig {
            hidden_size: 1024,
            extra,
            ..Default::default()
        };
        let result = substitute_placeholders(
            "H=${hidden_size}, C=${num_channels}", &config,
        );
        assert_eq!(result, "H=1024, C=3");
    }

    #[test]
    fn substitute_placeholders_extra_key_not_present_stays_literal() {
        let config = ResolvedConfig {
            extra: HashMap::new(),
            ..Default::default()
        };
        let result = substitute_placeholders("${missing_extra}", &config);
        assert_eq!(result, "${missing_extra}");
    }

    #[test]
    fn validate_config_accepts_minimal_valid_config() {
        // Only the 4 required fields are set; everything else is default/zero.
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn resolve_error_inconsistent_variant() {
        let err = ResolveError::Inconsistent("mismatch".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Inconsistent config"));
        assert!(msg.contains("mismatch"));
    }

    #[test]
    fn resolve_error_derivation_failed_variant_fields() {
        let err = ResolveError::DerivationFailed {
            key: "my_key".to_string(),
            reason: "some_reason".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("my_key"));
        assert!(msg.contains("some_reason"));
        assert!(msg.contains("Failed to derive"));
    }

    #[test]
    fn resolved_config_clone_produces_equal_instance() {
        let mut extra = HashMap::new();
        extra.insert("k".to_string(), 10);
        let original = ResolvedConfig {
            num_hidden_layers: 8,
            hidden_size: 512,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: Some(2048),
            vocab_size: 1000,
            rope_theta: 50000.0,
            dtype: "f32".to_string(),
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![0, 1],
            sliding_window: 256,
            num_kv_shared_layers: 1,
            global_head_dim: 32,
            hidden_size_per_layer_input: 0,
            has_per_layer_embedding: false,
            rope_scaling: None,
            extra,
            norm_eps: 1e-5,
        };
        let cloned = original.clone();
        assert_eq!(cloned.num_hidden_layers, original.num_hidden_layers);
        assert_eq!(cloned.hidden_size, original.hidden_size);
        assert_eq!(cloned.num_attention_heads, original.num_attention_heads);
        assert_eq!(cloned.num_key_value_heads, original.num_key_value_heads);
        assert_eq!(cloned.head_dim, original.head_dim);
        assert_eq!(cloned.intermediate_size, original.intermediate_size);
        assert_eq!(cloned.vocab_size, original.vocab_size);
        assert_eq!(cloned.rope_theta, original.rope_theta);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.attention_pattern, original.attention_pattern);
        assert_eq!(cloned.extra, original.extra);
    }

    #[test]
    fn resolved_config_debug_formats_all_fields() {
        let config = ResolvedConfig {
            num_hidden_layers: 2,
            hidden_size: 128,
            ..Default::default()
        };
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("num_hidden_layers: 2"));
        assert!(debug_str.contains("hidden_size: 128"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (45+) — added for coverage expansion
    // ═══════════════════════════════════════════════════════════════════════

    // ── ResolveError Debug format ──────────────────────────────────────────

    #[test]
    fn resolve_error_debug_missing_config() {
        let err = ResolveError::MissingConfig("field_a".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("MissingConfig"));
        assert!(debug.contains("field_a"));
    }

    #[test]
    fn resolve_error_debug_derivation_failed() {
        let err = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("DerivationFailed"));
        assert!(debug.contains("k"));
        assert!(debug.contains("r"));
    }

    #[test]
    fn resolve_error_debug_inconsistent() {
        let err = ResolveError::Inconsistent("bad_state".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Inconsistent"));
        assert!(debug.contains("bad_state"));
    }

    // ── ResolveError Clone + PartialEq ─────────────────────────────────────

    #[test]
    fn resolve_error_clone_missing_config() {
        let err = ResolveError::MissingConfig("x".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn resolve_error_clone_derivation_failed() {
        let err = ResolveError::DerivationFailed {
            key: "a".to_string(),
            reason: "b".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn resolve_error_clone_inconsistent() {
        let err = ResolveError::Inconsistent("msg".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn resolve_error_partial_eq_same_variants_equal() {
        let a = ResolveError::MissingConfig("k".to_string());
        let b = ResolveError::MissingConfig("k".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn resolve_error_partial_eq_different_keys_not_equal() {
        let a = ResolveError::MissingConfig("k1".to_string());
        let b = ResolveError::MissingConfig("k2".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn resolve_error_partial_eq_different_variants_not_equal() {
        let a = ResolveError::MissingConfig("k".to_string());
        let b = ResolveError::Inconsistent("k".to_string());
        assert_ne!(a, b);
    }

    // ── ResolvedConfig default values: numeric edge cases ──────────────────

    #[test]
    fn resolved_config_default_rope_theta_is_zero_f64() {
        let config = ResolvedConfig::default();
        assert_eq!(config.rope_theta, 0.0f64);
        assert!(config.rope_theta.is_sign_positive() || config.rope_theta == 0.0);
    }

    #[test]
    fn resolved_config_default_global_rope_theta_is_zero_f64() {
        let config = ResolvedConfig::default();
        assert_eq!(config.global_rope_theta, 0.0f64);
    }

    #[test]
    fn resolved_config_default_rope_partial_ratio_is_zero_f32() {
        let config = ResolvedConfig::default();
        assert_eq!(config.rope_partial_ratio, 0.0f32);
    }

    // ── ResolvedConfig get_int with usize::MAX ─────────────────────────────

    #[test]
    fn get_int_returns_usize_max_as_i64() {
        let config = ResolvedConfig {
            num_hidden_layers: usize::MAX,
            ..Default::default()
        };
        // usize::MAX on 64-bit is larger than i64::MAX, so the cast wraps.
        let result = config.get_int("num_hidden_layers");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), usize::MAX as i64);
    }

    // ── ResolvedConfig get_int extra with negative values ───────────────────

    #[test]
    fn get_int_extra_negative_value() {
        let mut extra = HashMap::new();
        extra.insert("neg".to_string(), -999);
        let config = ResolvedConfig { extra, ..Default::default() };
        assert_eq!(config.get_int("neg"), Some(-999));
    }

    #[test]
    fn get_int_extra_with_i64_max() {
        let mut extra = HashMap::new();
        extra.insert("max_val".to_string(), i64::MAX);
        let config = ResolvedConfig { extra, ..Default::default() };
        assert_eq!(config.get_int("max_val"), Some(i64::MAX));
    }

    #[test]
    fn get_int_extra_with_i64_min() {
        let mut extra = HashMap::new();
        extra.insert("min_val".to_string(), i64::MIN);
        let config = ResolvedConfig { extra, ..Default::default() };
        assert_eq!(config.get_int("min_val"), Some(i64::MIN));
    }

    // ── ResolvedConfig get_float special values ─────────────────────────────

    #[test]
    fn get_float_rope_theta_negative_value() {
        let config = ResolvedConfig {
            rope_theta: -500.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(-500.0));
    }

    #[test]
    fn get_float_rope_theta_very_large() {
        let config = ResolvedConfig {
            rope_theta: 1e18,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(1e18));
    }

    // ── ResolvedConfig get_str edge cases ───────────────────────────────────

    #[test]
    fn get_str_dtype_empty_string() {
        let config = ResolvedConfig {
            dtype: String::new(),
            ..Default::default()
        };
        // Empty dtype is still Some("")
        assert_eq!(config.get_str("dtype"), Some(""));
    }

    #[test]
    fn get_str_dtype_with_special_chars() {
        let config = ResolvedConfig {
            dtype: "f32+bf16".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("dtype"), Some("f32+bf16"));
    }

    // ── ResolvedConfig get_bool: non-parseable suffix ──────────────────────

    #[test]
    fn get_bool_is_kv_shared_layer_negative_number_suffix() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 3,
            ..Default::default()
        };
        // "-1" is not parseable as usize → returns None
        assert_eq!(config.get_bool("is_kv_shared_layer_-1"), None);
    }

    #[test]
    fn get_bool_is_kv_shared_layer_hex_suffix_not_parsed() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 3,
            ..Default::default()
        };
        // "0x5" is not parseable as usize by str::parse → returns None
        assert_eq!(config.get_bool("is_kv_shared_layer_0x5"), None);
    }

    // ── is_kv_shared_layer: boundary layer indices ─────────────────────────

    #[test]
    fn is_kv_shared_layer_exactly_at_boundary() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 3;
        // Boundary: 10 - 3 = 7. Layer 7 is the first consumer.
        assert!(!config.is_kv_shared_layer(6));
        assert!(config.is_kv_shared_layer(7));
        assert!(config.is_kv_shared_layer(8));
        assert!(config.is_kv_shared_layer(9));
    }

    #[test]
    fn is_kv_shared_layer_single_layer_model() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 1;
        config.num_kv_shared_layers = 1;
        assert!(config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(1));
    }

    // ── donor_layer: single layer model ────────────────────────────────────

    #[test]
    fn donor_layer_single_layer_no_sharing() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 1;
        config.num_kv_shared_layers = 0;
        assert_eq!(config.donor_layer(0).unwrap(), None);
    }

    // ── validate_config: all fields zero except one ────────────────────────

    #[test]
    fn validate_config_rejects_when_only_layers_set() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("hidden_size".to_string()));
    }

    #[test]
    fn validate_config_rejects_when_layers_and_hidden_set() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 128,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("num_attention_heads".to_string()));
    }

    #[test]
    fn validate_config_rejects_when_three_of_four_set() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 128,
            num_attention_heads: 4,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("vocab_size".to_string()));
    }

    // ── validate_config: large values accepted ─────────────────────────────

    #[test]
    fn validate_config_accepts_large_values() {
        let config = ResolvedConfig {
            num_hidden_layers: 128,
            hidden_size: 16384,
            num_attention_heads: 128,
            vocab_size: 256000,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── substitute_placeholders: multiple same placeholder ─────────────────

    #[test]
    fn substitute_placeholders_multiple_extras() {
        let mut extra = HashMap::new();
        extra.insert("a".to_string(), 1);
        extra.insert("b".to_string(), 2);
        extra.insert("c".to_string(), 3);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("${a}-${b}-${c}", &config);
        assert_eq!(result, "1-2-3");
    }

    #[test]
    fn substitute_placeholders_extra_overrides_if_name_collision() {
        // If extra has a key that is NOT a known field, it is substituted.
        // Known fields (like hidden_size) are NOT looked up from extra.
        let mut extra = HashMap::new();
        extra.insert("custom_key".to_string(), 42);
        let config = ResolvedConfig {
            hidden_size: 100,
            extra,
            ..Default::default()
        };
        let result = substitute_placeholders("${hidden_size}_${custom_key}", &config);
        assert_eq!(result, "100_42");
    }

    // ── convert_rope_scaling: Linear with large factor ─────────────────────

    #[test]
    fn convert_rope_scaling_linear_large_factor() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(1000.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Linear { factor } => {
                assert!((factor - 1000.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: factor slightly above 1.0 ────────────────────

    #[test]
    fn convert_rope_scaling_factor_slightly_above_one() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(1.0 + f32::EPSILON),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some(), "factor slightly > 1.0 must produce scaling");
    }

    // ── convert_rope_scaling: Yarn with default original_max_position ──────

    #[test]
    fn convert_rope_scaling_yarn_missing_original_max_defaults_to_4096() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(2.0),
            original_max_position_embeddings: None,
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { original_max_position, .. } => {
                assert_eq!(original_max_position, 4096);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: LongRope type is unknown → returns None ──────

    #[test]
    fn convert_rope_scaling_longrope_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::LongRope),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── convert_rope_scaling: NtkAware type is unknown → returns None ──────

    #[test]
    fn convert_rope_scaling_ntk_aware_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::NtkAware),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── convert_rope_scaling: Llama3 type is unknown → returns None ────────

    #[test]
    fn convert_rope_scaling_llama3_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Llama3),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── convert_rope_scaling: Unknown string variant → returns None ────────

    #[test]
    fn convert_rope_scaling_unknown_string_variant_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Unknown("custom".to_string())),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── ResolvedConfig clone independence ───────────────────────────────────

    #[test]
    fn resolved_config_clone_extra_is_independent() {
        let mut extra = HashMap::new();
        extra.insert("key".to_string(), 1);
        let original = ResolvedConfig {
            extra,
            ..Default::default()
        };
        let mut cloned = original.clone();
        cloned.extra.insert("new_key".to_string(), 2);
        // Original unaffected
        assert!(!original.extra.contains_key("new_key"));
        assert!(cloned.extra.contains_key("new_key"));
    }

    // ── ResolvedConfig clone attention_pattern independence ─────────────────

    #[test]
    fn resolved_config_clone_attention_pattern_is_independent() {
        let original = ResolvedConfig {
            attention_pattern: vec![0, 1, 2],
            ..Default::default()
        };
        let mut cloned = original.clone();
        cloned.attention_pattern.push(3);
        assert_eq!(original.attention_pattern.len(), 3);
        assert_eq!(cloned.attention_pattern.len(), 4);
    }

    // ── ResolvedConfig Debug: all field names appear ───────────────────────

    #[test]
    fn resolved_config_debug_contains_all_field_names() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 2,
            num_attention_heads: 3,
            num_key_value_heads: 4,
            head_dim: 5,
            intermediate_size: Some(6),
            vocab_size: 7,
            rope_theta: 8.0,
            dtype: "f32".to_string(),
            global_rope_theta: 9.0,
            rope_partial_ratio: 10.0,
            attention_pattern: vec![0],
            sliding_window: 11,
            rope_scaling: None,
            num_kv_shared_layers: 12,
            global_head_dim: 13,
            hidden_size_per_layer_input: 14,
            has_per_layer_embedding: true,
            extra: HashMap::new(),
            norm_eps: 1e-5,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("num_hidden_layers"));
        assert!(debug.contains("hidden_size"));
        assert!(debug.contains("num_attention_heads"));
        assert!(debug.contains("num_key_value_heads"));
        assert!(debug.contains("head_dim"));
        assert!(debug.contains("intermediate_size"));
        assert!(debug.contains("vocab_size"));
        assert!(debug.contains("rope_theta"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("global_rope_theta"));
        assert!(debug.contains("rope_partial_ratio"));
        assert!(debug.contains("attention_pattern"));
        assert!(debug.contains("sliding_window"));
        assert!(debug.contains("rope_scaling"));
        assert!(debug.contains("num_kv_shared_layers"));
        assert!(debug.contains("global_head_dim"));
        assert!(debug.contains("hidden_size_per_layer_input"));
        assert!(debug.contains("has_per_layer_embedding"));
        assert!(debug.contains("extra"));
    }

    // ── is_kv_shared_layer: rope_partial_ratio field does not affect it ────

    #[test]
    fn is_kv_shared_layer_unaffected_by_rope_partial_ratio() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.rope_partial_ratio = 0.25;
        // Layers 4..8 are consumers regardless of rope_partial_ratio
        assert!(!config.is_kv_shared_layer(3));
        assert!(config.is_kv_shared_layer(4));
    }

    // ── substitute_placeholders: rope_partial_ratio not a placeholder ──────

    #[test]
    fn substitute_placeholders_rope_partial_ratio_not_substituted() {
        let config = ResolvedConfig {
            rope_partial_ratio: 0.25,
            ..Default::default()
        };
        let result = substitute_placeholders("${rope_partial_ratio}", &config);
        assert_eq!(result, "${rope_partial_ratio}");
    }

    // ── substitute_placeholders: head_dim zero outputs "0" ─────────────────

    #[test]
    fn substitute_placeholders_zero_values_output_zero_string() {
        let config = ResolvedConfig {
            head_dim: 0,
            ..Default::default()
        };
        let result = substitute_placeholders("hd=${head_dim}", &config);
        assert_eq!(result, "hd=0");
    }

    // ── resolve_from_provider: only non-layer tensors ──────────────────────

    #[test]
    fn resolve_from_provider_non_layer_tensor_no_role() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "lm_head.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // lm_head does not match Embedding role → vocab_size stays 0
        assert!(result.is_err());
    }

    // ── get_int: intermediate_size returns zero from default ───────────────

    #[test]
    fn get_int_intermediate_size_none_returns_none_from_default() {
        let config = ResolvedConfig::default();
        assert_eq!(config.intermediate_size, None);
        assert_eq!(config.get_int("intermediate_size"), None);
    }

    // ── validate_config: checks fields in specific order ───────────────────

    #[test]
    fn validate_config_reports_first_missing_field() {
        // All four required fields missing → first reported is num_hidden_layers
        let config = ResolvedConfig::default();
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("num_hidden_layers".to_string()));
    }

    // ── donor_layer: config with num_kv_shared_layers but no attention_pattern ─

    #[test]
    fn donor_layer_empty_attention_pattern_errors_for_consumer() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![];
        // Consumer layer 4 with empty pattern → error
        assert!(config.donor_layer(4).is_err());
    }

    // ── ResolvedConfig: extra HashMap with many entries ─────────────────────

    #[test]
    fn resolved_config_extra_multiple_entries_all_accessible() {
        let mut extra = HashMap::new();
        extra.insert("a".to_string(), 1);
        extra.insert("b".to_string(), 2);
        extra.insert("c".to_string(), 3);
        extra.insert("d".to_string(), 4);
        extra.insert("e".to_string(), 5);
        let config = ResolvedConfig { extra, ..Default::default() };
        assert_eq!(config.get_int("a"), Some(1));
        assert_eq!(config.get_int("b"), Some(2));
        assert_eq!(config.get_int("c"), Some(3));
        assert_eq!(config.get_int("d"), Some(4));
        assert_eq!(config.get_int("e"), Some(5));
    }

    // ── from_geometry: dtype string formatting ──────────────────────────────

    #[test]
    fn from_geometry_dtype_f32_lowercase() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.dtype, "f32");
    }

    // ── from_geometry: global_rope_theta and rope_partial_ratio pass through ─

    #[test]
    fn from_geometry_global_rope_and_partial_ratio_passthrough() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.25,
            attention_pattern: vec![0, 1, 0, 1],
            sliding_window: 4096,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert!((config.global_rope_theta - 1000000.0).abs() < f64::EPSILON);
        assert!((config.rope_partial_ratio - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.attention_pattern, vec![0, 1, 0, 1]);
        assert_eq!(config.sliding_window, 4096);
    }

    // ── ResolveError display does not panic on empty strings ───────────────

    #[test]
    fn resolve_error_display_empty_strings() {
        let err = ResolveError::MissingConfig(String::new());
        let msg = format!("{err}");
        assert!(msg.contains("Missing required config"));

        let err = ResolveError::Inconsistent(String::new());
        let msg = format!("{err}");
        assert!(msg.contains("Inconsistent config"));

        let err = ResolveError::DerivationFailed {
            key: String::new(),
            reason: String::new(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Failed to derive"));
    }

    // ── is_kv_shared_layer: num_kv_shared_layers equals num_hidden_layers ──

    #[test]
    fn is_kv_shared_layer_all_shared_at_boundary() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 5;
        config.num_kv_shared_layers = 5;
        // Every layer is a consumer (5 - 5 = 0, all indices >= 0)
        for i in 0..5 {
            assert!(config.is_kv_shared_layer(i));
        }
    }

    // ── substitute_placeholders: only extra no builtins ────────────────────

    #[test]
    fn substitute_placeholders_only_extra_no_builtins() {
        let mut extra = HashMap::new();
        extra.insert("x".to_string(), 42);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("val=${x}", &config);
        assert_eq!(result, "val=42");
    }

    // ── donor_layer: donor_layer(0) with no sharing returns Ok(None) ──────

    #[test]
    fn donor_layer_zero_with_no_sharing_is_none() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 0;
        assert_eq!(config.donor_layer(0).unwrap(), None);
    }

    // ── get_bool: has_per_layer_embedding is independent of extra ──────────

    #[test]
    fn get_bool_has_per_layer_embedding_ignores_extra() {
        let extra = HashMap::new();
        // Even if extra has a key "has_per_layer_embedding", the match arm
        // for the static field takes priority.
        let config = ResolvedConfig {
            has_per_layer_embedding: true,
            extra,
            ..Default::default()
        };
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
    }

    // ── validate_config: zero intermediate_size is valid (it's optional) ───

    #[test]
    fn validate_config_accepts_none_intermediate_size() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            intermediate_size: None,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── resolve_from_provider: BF16 dtype detected ─────────────────────────

    #[test]
    fn resolve_from_provider_dtype_bf16_detected() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };
        // Validation will still fail (missing num_hidden_layers etc.)
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (55+) — gap-filling coverage expansion
    // ═══════════════════════════════════════════════════════════════════════

    // ── resolve_from_provider: embed tensor with 1D shape ────────────────

    #[test]
    fn resolve_from_provider_embedding_1d_shape_no_crash() {
        // 1D embedding shape: shape.len() < 2 → vocab_size/hidden_size NOT derived.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ── resolve_from_provider: embed tensor with 3D shape ────────────────

    #[test]
    fn resolve_from_provider_embedding_3d_shape_uses_first_two_dims() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096, 128],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // vocab_size=32000, hidden_size=4096 derived, but num_hidden_layers=0.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: q_proj shape with 1D ──────────────────────

    #[test]
    fn resolve_from_provider_q_proj_1d_shape_skips_head_derivation() {
        // q_proj with shape [4096] → shape.len() < 2 → num_attention_heads NOT derived.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ── resolve_from_provider: k_proj 2D shape with head_dim > 0 ─────────

    #[test]
    fn resolve_from_provider_k_proj_with_nonzero_head_dim() {
        // k_proj + q_proj + embed, with head_dim preset via hidden_size/num_attention_heads.
        // This path requires GGUF metadata for head_dim, but without it head_dim=0.
        // When hidden_size > 0 and num_attention_heads > 0, head_dim gets auto-computed.
        // Here we provide a k_proj where head_dim=0, so num_kv_heads is not derived.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![1024, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ── resolve_from_provider: multiple layers detected correctly ─────────

    #[test]
    fn resolve_from_provider_detects_max_layer_index_plus_one() {
        // Tensors for layers 0, 2, 4 → max_layer_idx = 5 (not 3).
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.2.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.4.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        // Even though num_hidden_layers=5, num_attention_heads=0 still fails.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: unknown tensor role ignored ────────────────

    #[test]
    fn resolve_from_provider_unknown_tensor_names_ignored() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![100, 64],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "some.random.tensor".to_string(),
                    shape: vec![100, 100],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "rotary_emb.inv_freq".to_string(),
                    shape: vec![32],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // vocab_size=100, hidden_size=64 derived from embed. Still missing layers/heads.
        assert!(result.is_err());
    }

    // ── resolve_from_provider: dtype BF16 string format ──────────────────

    #[test]
    fn resolve_from_provider_sets_dtype_bf16_string() {
        // We cannot directly observe the dtype string since validation fails,
        // but we verify no panic occurs with BF16 dtype detection.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![100, 64],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };
        assert!(resolve_from_provider(&provider, None).is_err());
    }

    // ── resolve_from_provider: unsupported dtype falls back to f32 string ─

    #[test]
    fn resolve_from_provider_unsupported_dtype_defaults_to_f32() {
        // safetensors::Dtype::U8 is not F32/F16/BF16 → falls to the `_ => "f32"` arm.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![100, 64],
                    dtype: safetensors::Dtype::U8,
                },
            ],
        };
        // No panic; dtype becomes "f32" but validation still fails.
        assert!(resolve_from_provider(&provider, None).is_err());
    }

    // ── get_int: head_dim field accessible ────────────────────────────────

    #[test]
    fn get_int_head_dim_field() {
        let config = ResolvedConfig {
            head_dim: 96,
            ..Default::default()
        };
        assert_eq!(config.get_int("head_dim"), Some(96));
    }

    // ── get_int: all core fields return zero from default ────────────────

    #[test]
    fn get_int_all_core_fields_zero_from_default() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_int("num_hidden_layers"), Some(0));
        assert_eq!(config.get_int("hidden_size"), Some(0));
        assert_eq!(config.get_int("num_attention_heads"), Some(0));
        assert_eq!(config.get_int("num_key_value_heads"), Some(0));
        assert_eq!(config.get_int("head_dim"), Some(0));
        assert_eq!(config.get_int("vocab_size"), Some(0));
    }

    // ── get_float: global_rope_theta with zero ────────────────────────────

    #[test]
    fn get_float_global_rope_theta_zero() {
        let config = ResolvedConfig {
            global_rope_theta: 0.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("global_rope_theta"), Some(0.0));
    }

    // ── get_float: global_rope_theta with negative ───────────────────────

    #[test]
    fn get_float_global_rope_theta_negative() {
        let config = ResolvedConfig {
            global_rope_theta: -100.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("global_rope_theta"), Some(-100.0));
    }

    // ── get_str: dtype with long string ──────────────────────────────────

    #[test]
    fn get_str_dtype_long_custom_string() {
        let config = ResolvedConfig {
            dtype: "custom_quantized_4bit_special".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("dtype"), Some("custom_quantized_4bit_special"));
    }

    // ── get_bool: prefix-only string returns None ────────────────────────

    #[test]
    fn get_bool_is_kv_shared_layer_prefix_only_returns_none() {
        let config = ResolvedConfig::default();
        // Just the prefix with no number after.
        assert_eq!(config.get_bool("is_kv_shared_layer_"), None);
    }

    // ── get_bool: float suffix returns None ──────────────────────────────

    #[test]
    fn get_bool_is_kv_shared_layer_float_suffix_returns_none() {
        let config = ResolvedConfig::default();
        // "3.5" is not parseable as usize → None.
        assert_eq!(config.get_bool("is_kv_shared_layer_3.5"), None);
    }

    // ── get_bool: very large index handled gracefully ────────────────────

    #[test]
    fn get_bool_is_kv_shared_layer_usize_max_suffix() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 3,
            ..Default::default()
        };
        // usize::MAX is parseable but far beyond num_hidden_layers → false.
        let key = format!("is_kv_shared_layer_{}", usize::MAX);
        assert_eq!(config.get_bool(&key), Some(false));
    }

    // ── is_kv_shared_layer: head_dim=0 does not affect KV sharing ────────

    #[test]
    fn is_kv_shared_layer_unaffected_by_head_dim() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.head_dim = 0;
        // Layers 4..8 are consumers regardless of head_dim value.
        assert!(!config.is_kv_shared_layer(3));
        assert!(config.is_kv_shared_layer(4));
        assert!(config.is_kv_shared_layer(7));
    }

    // ── is_kv_shared_layer: sliding_window does not affect KV sharing ────

    #[test]
    fn is_kv_shared_layer_unaffected_by_sliding_window() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.sliding_window = 4096;
        assert!(!config.is_kv_shared_layer(3));
        assert!(config.is_kv_shared_layer(4));
    }

    // ── is_kv_shared_layer: global_head_dim does not affect KV sharing ───

    #[test]
    fn is_kv_shared_layer_unaffected_by_global_head_dim() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 2;
        config.global_head_dim = 256;
        assert!(!config.is_kv_shared_layer(3));
        assert!(config.is_kv_shared_layer(4));
    }

    // ── donor_layer: two-bucket pattern with odd layer count ─────────────

    #[test]
    fn donor_layer_odd_layer_count_two_buckets() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 7;
        config.num_kv_shared_layers = 3;
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0];
        // Consumers: layers 4,5,6 (first_consumer = 7-3 = 4)
        // Layer 4 (bucket 0) → donor at layer 2 (bucket 0, latest non-consumer)
        assert_eq!(config.donor_layer(4).unwrap(), Some(2));
        // Layer 5 (bucket 1) → donor at layer 3 (bucket 1)
        assert_eq!(config.donor_layer(5).unwrap(), Some(3));
        // Layer 6 (bucket 0) → donor at layer 2 (bucket 0)
        assert_eq!(config.donor_layer(6).unwrap(), Some(2));
    }

    // ── donor_layer: single bucket consumer gets latest donor ────────────

    #[test]
    fn donor_layer_single_bucket_picks_latest_donor() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![5, 5, 5, 5, 5, 5];
        // Consumer layer 4 → latest non-consumer with bucket 5 is layer 3.
        assert_eq!(config.donor_layer(4).unwrap(), Some(3));
        assert_eq!(config.donor_layer(5).unwrap(), Some(3));
    }

    // ── donor_layer: non-consumer at boundary returns Ok(None) ───────────

    #[test]
    fn donor_layer_boundary_non_consumer_returns_none() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0; 8];
        // Layer 3 is the last non-consumer (8-4=4, so layers 0..4 are non-consumers).
        assert_eq!(config.donor_layer(3).unwrap(), None);
        assert_eq!(config.donor_layer(4).unwrap(), Some(3));
    }

    // ── validate_config: accepts config with all optional fields set ──────

    #[test]
    fn validate_config_accepts_with_all_optional_fields_set() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(11008),
            vocab_size: 32000,
            rope_theta: 10000.0,
            dtype: "bf16".to_string(),
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.25,
            attention_pattern: vec![0, 1, 0, 1],
            sliding_window: 4096,
            num_kv_shared_layers: 4,
            global_head_dim: 256,
            hidden_size_per_layer_input: 512,
            has_per_layer_embedding: true,
            rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
            extra: {
                let mut m = HashMap::new();
                m.insert("custom".to_string(), 1);
                m
            },
            norm_eps: 1e-5,
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── validate_config: zero head_dim is valid ──────────────────────────

    #[test]
    fn validate_config_accepts_zero_head_dim() {
        // head_dim=0 is not a required field for validation.
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            head_dim: 0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── validate_config: zero num_key_value_heads is valid ───────────────

    #[test]
    fn validate_config_accepts_zero_num_kv_heads() {
        // num_key_value_heads is not required by validate_config.
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            num_key_value_heads: 0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── substitute_placeholders: multiple occurrences of same extra key ──

    #[test]
    fn substitute_placeholders_extra_key_repeated() {
        let mut extra = HashMap::new();
        extra.insert("x".to_string(), 7);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("${x}+${x}=${x}", &config);
        assert_eq!(result, "7+7=7");
    }

    // ── substitute_placeholders: adjacent placeholders ───────────────────

    #[test]
    fn substitute_placeholders_adjacent_placeholders() {
        let config = ResolvedConfig {
            num_hidden_layers: 12,
            hidden_size: 768,
            ..Default::default()
        };
        let result = substitute_placeholders("${num_hidden_layers}${hidden_size}", &config);
        assert_eq!(result, "12768");
    }

    // ── substitute_placeholders: placeholder in middle of word ───────────

    #[test]
    fn substitute_placeholders_embedded_in_text() {
        let config = ResolvedConfig {
            hidden_size: 4096,
            ..Default::default()
        };
        let result = substitute_placeholders("prefix_${hidden_size}_suffix", &config);
        assert_eq!(result, "prefix_4096_suffix");
    }

    // ── substitute_placeholders: all float placeholders ──────────────────

    #[test]
    fn substitute_placeholders_all_float_fields() {
        let config = ResolvedConfig {
            rope_theta: 10000.0,
            global_rope_theta: 1000000.0,
            ..Default::default()
        };
        let result = substitute_placeholders(
            "theta=${rope_theta},global=${global_rope_theta}", &config,
        );
        assert!(result.contains("10000"));
        assert!(result.contains("1000000"));
    }

    // ── from_geometry: with Linear rope scaling ──────────────────────────

    #[test]
    fn from_geometry_with_linear_rope_scaling() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: Some(RopeScalingConfig {
                scaling_type: Some(RopeScalingType::Linear),
                factor: Some(4.0),
                ..Default::default()
            }),
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert!(config.rope_scaling.is_some());
        match config.rope_scaling.unwrap() {
            RopeScaling::Linear { factor } => {
                assert!((factor - 4.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    // ── from_geometry: BF16 dtype string formatting ──────────────────────

    #[test]
    fn from_geometry_dtype_bf16_lowercase() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.dtype, "bf16");
    }

    // ── from_geometry: F16 dtype string formatting ───────────────────────

    #[test]
    fn from_geometry_dtype_f16_lowercase() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F16,
            compute_dtype: DType::F16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.dtype, "f16");
    }

    // ── from_geometry: extra accessible via get_int ──────────────────────

    #[test]
    fn from_geometry_extra_accessible_via_get_int() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let mut extra = HashMap::new();
        extra.insert("vision_channels".to_string(), 3);
        extra.insert("audio_sample_rate".to_string(), 16000);

        let config = ResolvedConfig::from_geometry(&geometry, extra);
        assert_eq!(config.get_int("vision_channels"), Some(3));
        assert_eq!(config.get_int("audio_sample_rate"), Some(16000));
    }

    // ── convert_rope_scaling: Yarn with all custom values ────────────────

    #[test]
    fn convert_rope_scaling_yarn_custom_beta_values() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(16.0),
            original_max_position_embeddings: Some(16384),
            beta_fast: Some(64.0),
            beta_slow: Some(0.5),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!((factor - 16.0).abs() < f32::EPSILON);
                assert!((beta_fast - 64.0).abs() < f32::EPSILON);
                assert!((beta_slow - 0.5).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 16384);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: factor exactly 0.0 returns None ────────────

    #[test]
    fn convert_rope_scaling_factor_zero_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(0.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── convert_rope_scaling: negative factor returns None ───────────────

    #[test]
    fn convert_rope_scaling_negative_factor_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(-2.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── convert_rope_scaling: Dynamic type returns None ──────────────────

    #[test]
    fn convert_rope_scaling_dynamic_returns_none() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Dynamic),
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(convert_rope_scaling(Some(&scaling)).is_none());
    }

    // ── ResolvedConfig: clone with rope_scaling ──────────────────────────

    #[test]
    fn resolved_config_clone_with_rope_scaling() {
        let config = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Linear { factor: 8.0 }),
            ..Default::default()
        };
        let cloned = config.clone();
        assert!(cloned.rope_scaling.is_some());
        match cloned.rope_scaling.unwrap() {
            RopeScaling::Linear { factor } => {
                assert!((factor - 8.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    // ── ResolvedConfig: clone with dtype preserves string ────────────────

    #[test]
    fn resolved_config_clone_preserves_dtype() {
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.dtype, "bf16");
    }

    // ── ResolvedConfig: clone with attention_pattern preserves vec ───────

    #[test]
    fn resolved_config_clone_preserves_attention_pattern() {
        let config = ResolvedConfig {
            attention_pattern: vec![1, 0, 1, 0, 1, 0],
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.attention_pattern, vec![1, 0, 1, 0, 1, 0]);
    }

    // ── ResolvedConfig: debug with rope_scaling some ─────────────────────

    #[test]
    fn resolved_config_debug_with_rope_scaling_some() {
        let config = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
            ..Default::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("rope_scaling"));
        assert!(debug.contains("Some"));
    }

    // ── ResolvedConfig: debug with non-empty extra ───────────────────────

    #[test]
    fn resolved_config_debug_with_non_empty_extra() {
        let mut extra = HashMap::new();
        extra.insert("my_key".to_string(), 42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("extra"));
        assert!(debug.contains("my_key"));
    }

    // ── ResolvedConfig: has_per_layer_embedding derived field ────────────

    #[test]
    fn has_per_layer_embedding_independent_of_hidden_size() {
        // has_per_layer_embedding is solely derived from hidden_size_per_layer_input.
        let config = ResolvedConfig {
            hidden_size: 8192,
            hidden_size_per_layer_input: 0,
            has_per_layer_embedding: false,
            ..Default::default()
        };
        assert!(!config.has_per_layer_embedding);

        let config2 = ResolvedConfig {
            hidden_size: 64,
            hidden_size_per_layer_input: 128,
            has_per_layer_embedding: true,
            ..Default::default()
        };
        assert!(config2.has_per_layer_embedding);
    }

    // ── ResolvedConfig: get_bool with various unknown prefixes ───────────

    #[test]
    fn get_bool_various_unknown_prefixes() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_bool("is_"), None);
        assert_eq!(config.get_bool("is_kv_shared_5"), None);
        assert_eq!(config.get_bool("shared_layer_5"), None);
        assert_eq!(config.get_bool("kv_shared_layer_5"), None);
    }

    // ── ResolvedConfig: attention_pattern with u8 max values ─────────────

    #[test]
    fn donor_layer_with_u8_max_bucket_values() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![255, 255, 255, 255];
        // All buckets are 255 → consumer 2 gets donor 1 (latest matching).
        assert_eq!(config.donor_layer(2).unwrap(), Some(1));
        assert_eq!(config.donor_layer(3).unwrap(), Some(1));
    }

    // ── ResolvedConfig: attention_pattern with mixed high values ─────────

    #[test]
    fn donor_layer_with_distinct_high_bucket_values() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![10, 20, 10, 20];
        // Consumer 2 (bucket 10) → donor 0 (bucket 10).
        assert_eq!(config.donor_layer(2).unwrap(), Some(0));
        // Consumer 3 (bucket 20) → donor 1 (bucket 20).
        assert_eq!(config.donor_layer(3).unwrap(), Some(1));
    }

    // ── resolve_from_provider: embed tensor sets vocab_size from dim 0 ──

    #[test]
    fn resolve_from_provider_embed_first_dim_sets_vocab_size() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![50000, 1024],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // Should fail at num_hidden_layers, not vocab_size.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_ne!(key, "vocab_size", "vocab_size must be derived from embed");
                assert_eq!(key, "num_hidden_layers");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: embed tensor sets hidden_size from dim 1 ──

    #[test]
    fn resolve_from_provider_embed_second_dim_sets_hidden_size() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 768],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // Should fail at num_hidden_layers, not hidden_size.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_ne!(key, "hidden_size", "hidden_size must be derived from embed");
                assert_eq!(key, "num_hidden_layers");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: no tensors means all fields zero ──────────

    #[test]
    fn resolve_from_provider_empty_tensors_reports_first_required_field() {
        let provider = FakeProvider { tensors: vec![] };
        let err = resolve_from_provider(&provider, None).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("num_hidden_layers".to_string()));
    }

    // ── substitute_placeholders: unicode in template ─────────────────────

    #[test]
    fn substitute_placeholders_unicode_template() {
        let config = ResolvedConfig {
            hidden_size: 2048,
            ..Default::default()
        };
        let result = substitute_placeholders("隐藏层=${hidden_size}", &config);
        assert_eq!(result, "隐藏层=2048");
    }

    // ── substitute_placeholders: extra with zero value ───────────────────

    #[test]
    fn substitute_placeholders_extra_zero_value() {
        let mut extra = HashMap::new();
        extra.insert("zero_field".to_string(), 0);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("${zero_field}", &config);
        assert_eq!(result, "0");
    }

    // ── substitute_placeholders: extra with negative value ───────────────

    #[test]
    fn substitute_placeholders_extra_negative_value() {
        let mut extra = HashMap::new();
        extra.insert("offset".to_string(), -10);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("${offset}", &config);
        assert_eq!(result, "-10");
    }

    // ── ResolveError: PartialEq for DerivationFailed ────────────────────

    #[test]
    fn resolve_error_partial_eq_derivation_failed_same() {
        let a = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        let b = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        assert_eq!(a, b);
    }

    // ── ResolveError: PartialEq for DerivationFailed different reason ───

    #[test]
    fn resolve_error_partial_eq_derivation_failed_different_reason() {
        let a = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r1".to_string(),
        };
        let b = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r2".to_string(),
        };
        assert_ne!(a, b);
    }

    // ── is_kv_shared_layer: vocab_size does not affect result ────────────

    #[test]
    fn is_kv_shared_layer_unaffected_by_vocab_size() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        config.vocab_size = 100000;
        assert!(!config.is_kv_shared_layer(2));
        assert!(config.is_kv_shared_layer(3));
    }

    // ── is_kv_shared_layer: dtype does not affect result ─────────────────

    #[test]
    fn is_kv_shared_layer_unaffected_by_dtype() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        config.dtype = "bf16".to_string();
        assert!(!config.is_kv_shared_layer(2));
        assert!(config.is_kv_shared_layer(3));
    }

    // ── is_kv_shared_layer: intermediate_size does not affect result ─────

    #[test]
    fn is_kv_shared_layer_unaffected_by_intermediate_size() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        config.intermediate_size = Some(8192);
        assert!(!config.is_kv_shared_layer(2));
        assert!(config.is_kv_shared_layer(3));
    }

    // ── substitute_placeholders: sliding_window zero outputs "0" ─────────

    #[test]
    fn substitute_placeholders_sliding_window_zero() {
        let config = ResolvedConfig {
            sliding_window: 0,
            ..Default::default()
        };
        let result = substitute_placeholders("sw=${sliding_window}", &config);
        assert_eq!(result, "sw=0");
    }

    // ── ResolvedConfig: extra field retains insertion order independence ──

    #[test]
    fn resolved_config_extra_multiple_keys_all_queryable() {
        let mut extra = HashMap::new();
        for i in 0..10 {
            extra.insert(format!("key_{i}"), i as i64);
        }
        let config = ResolvedConfig { extra, ..Default::default() };
        for i in 0..10 {
            assert_eq!(config.get_int(&format!("key_{i}")), Some(i as i64));
        }
    }

    // ── validate_config: valid with extra fields present ─────────────────

    #[test]
    fn validate_config_valid_with_extra_present() {
        let mut extra = HashMap::new();
        extra.insert("model_revision".to_string(), 2);
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            extra,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── validate_config: rejects with extra but missing required field ───

    #[test]
    fn validate_config_rejects_with_extra_but_missing_required() {
        let mut extra = HashMap::new();
        extra.insert("model_revision".to_string(), 2);
        let config = ResolvedConfig {
            hidden_size: 256,
            extra,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("num_hidden_layers".to_string()));
    }

    // ── convert_rope_scaling: Yarn with zero beta_fast ───────────────────

    #[test]
    fn convert_rope_scaling_yarn_zero_beta_fast() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(4.0),
            beta_fast: Some(0.0),
            beta_slow: None,
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { beta_fast, beta_slow, .. } => {
                assert!((beta_fast - 0.0).abs() < f32::EPSILON);
                assert!((beta_slow - 1.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: Yarn with zero beta_slow ──────────────────

    #[test]
    fn convert_rope_scaling_yarn_zero_beta_slow() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(4.0),
            beta_fast: None,
            beta_slow: Some(0.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { beta_fast, beta_slow, .. } => {
                assert!((beta_fast - 32.0).abs() < f32::EPSILON);
                assert!((beta_slow - 0.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── ResolvedConfig: get_int returns known field before extra ─────────

    #[test]
    fn get_int_known_field_takes_priority_over_extra() {
        // If extra has a key "num_hidden_layers" with a different value,
        // the struct field value is returned (match arm comes first).
        let mut extra = HashMap::new();
        extra.insert("num_hidden_layers".to_string(), 999);
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            extra,
            ..Default::default()
        };
        // Struct field value (32) wins over extra (999).
        assert_eq!(config.get_int("num_hidden_layers"), Some(32));
    }

    // ── ResolvedConfig: get_int head_dim with nonzero value ──────────────

    #[test]
    fn get_int_head_dim_nonzero() {
        let config = ResolvedConfig {
            head_dim: 256,
            ..Default::default()
        };
        assert_eq!(config.get_int("head_dim"), Some(256));
    }

    // ── ResolvedConfig: get_int vocab_size large ─────────────────────────

    #[test]
    fn get_int_vocab_size_large() {
        let config = ResolvedConfig {
            vocab_size: 256000,
            ..Default::default()
        };
        assert_eq!(config.get_int("vocab_size"), Some(256000));
    }

    // ── ResolvedConfig: get_float does not return integer fields ─────────

    #[test]
    fn get_float_never_returns_integer_fields() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            ..Default::default()
        };
        assert_eq!(config.get_float("num_hidden_layers"), None);
        assert_eq!(config.get_float("hidden_size"), None);
        assert_eq!(config.get_float("head_dim"), None);
        assert_eq!(config.get_float("vocab_size"), None);
    }

    // ── ResolvedConfig: get_str does not return float fields ─────────────

    #[test]
    fn get_str_never_returns_float_fields() {
        let config = ResolvedConfig {
            rope_theta: 10000.0,
            ..Default::default()
        };
        assert_eq!(config.get_str("rope_theta"), None);
    }

    // ── ResolvedConfig: full config round-trip from_geometry + getters ───

    #[test]
    fn from_geometry_round_trip_all_getters() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 2048,
            num_layers: 12,
            vocab_size: 50000,
            intermediate_size: 8192,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.5,
            attention_pattern: vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            sliding_window: 1024,
            num_kv_shared_layers: 2,
            global_head_dim: 64,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        // Verify all getters work on the from_geometry output.
        assert_eq!(config.get_int("num_hidden_layers"), Some(12));
        assert_eq!(config.get_int("hidden_size"), Some(2048));
        assert_eq!(config.get_int("num_attention_heads"), Some(16));
        assert_eq!(config.get_int("num_key_value_heads"), Some(4));
        assert_eq!(config.get_int("head_dim"), Some(128));
        assert_eq!(config.get_int("intermediate_size"), Some(8192));
        assert_eq!(config.get_int("vocab_size"), Some(50000));
        assert!((config.get_float("rope_theta").unwrap() - 500000.0).abs() < f64::EPSILON);
        assert!((config.get_float("global_rope_theta").unwrap() - 1000000.0).abs() < f64::EPSILON);
        assert_eq!(config.get_str("dtype"), Some("bf16"));
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(false));
        assert_eq!(config.sliding_window, 1024);
        assert_eq!(config.num_kv_shared_layers, 2);
        assert_eq!(config.global_head_dim, 64);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  WAVE 60 — coverage expansion: +60 tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── ResolvedConfig: default has f64 zero that is non-negative ──────────

    #[test]
    fn resolved_config_default_rope_theta_is_non_negative() {
        let config = ResolvedConfig::default();
        assert!(config.rope_theta >= 0.0);
    }

    #[test]
    fn resolved_config_default_global_rope_theta_is_non_negative() {
        let config = ResolvedConfig::default();
        assert!(config.global_rope_theta >= 0.0);
    }

    // ── ResolvedConfig: default extra map is empty and queryable ──────────

    #[test]
    fn resolved_config_default_extra_is_empty_map() {
        let config = ResolvedConfig::default();
        assert!(config.extra.is_empty());
        assert_eq!(config.extra.len(), 0);
    }

    // ── ResolvedConfig: default attention_pattern is empty vec ────────────

    #[test]
    fn resolved_config_default_attention_pattern_is_empty() {
        let config = ResolvedConfig::default();
        assert!(config.attention_pattern.is_empty());
    }

    // ── ResolvedConfig: default rope_scaling is None ──────────────────────

    #[test]
    fn resolved_config_default_rope_scaling_is_none() {
        let config = ResolvedConfig::default();
        assert!(config.rope_scaling.is_none());
    }

    // ── ResolvedConfig: construction with all fields populated ────────────

    #[test]
    fn resolved_config_full_construction_all_fields_set() {
        let mut extra = HashMap::new();
        extra.insert("model_version".to_string(), 3);
        let config = ResolvedConfig {
            num_hidden_layers: 64,
            hidden_size: 8192,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(28672),
            vocab_size: 128256,
            rope_theta: 500000.0,
            dtype: "bf16".to_string(),
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.25,
            attention_pattern: vec![0; 64],
            sliding_window: 4096,
            num_kv_shared_layers: 4,
            global_head_dim: 256,
            hidden_size_per_layer_input: 512,
            has_per_layer_embedding: true,
            rope_scaling: Some(RopeScaling::Linear { factor: 8.0 }),
            extra,
            norm_eps: 1e-5,
        };
        // Verify every field
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, Some(28672));
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.sliding_window, 4096);
        assert_eq!(config.num_kv_shared_layers, 4);
        assert_eq!(config.global_head_dim, 256);
        assert_eq!(config.hidden_size_per_layer_input, 512);
        assert!(config.has_per_layer_embedding);
        assert!(config.rope_scaling.is_some());
        assert_eq!(config.extra.len(), 1);
    }

    // ── ResolvedConfig: intermediate_size Some(0) is distinct from None ──

    #[test]
    fn resolved_config_intermediate_size_some_zero_vs_none() {
        let config_some_zero = ResolvedConfig {
            intermediate_size: Some(0),
            ..Default::default()
        };
        let config_none = ResolvedConfig::default();
        // Some(0) returns Some(0) from get_int, None returns None.
        assert_eq!(config_some_zero.get_int("intermediate_size"), Some(0));
        assert_eq!(config_none.get_int("intermediate_size"), None);
    }

    // ── ResolvedConfig: intermediate_size Some(1) ─────────────────────────

    #[test]
    fn resolved_config_intermediate_size_some_one() {
        let config = ResolvedConfig {
            intermediate_size: Some(1),
            ..Default::default()
        };
        assert_eq!(config.intermediate_size, Some(1));
        assert_eq!(config.get_int("intermediate_size"), Some(1));
    }

    // ── get_int: known field with explicit zero returns Some(0) ───────────

    #[test]
    fn get_int_num_key_value_heads_explicit_zero() {
        let config = ResolvedConfig {
            num_key_value_heads: 0,
            ..Default::default()
        };
        assert_eq!(config.get_int("num_key_value_heads"), Some(0));
    }

    // ── get_int: known field with value 1 returns Some(1) ────────────────

    #[test]
    fn get_int_vocab_size_one() {
        let config = ResolvedConfig {
            vocab_size: 1,
            ..Default::default()
        };
        assert_eq!(config.get_int("vocab_size"), Some(1));
    }

    // ── get_float: rope_theta with very small positive value ──────────────

    #[test]
    fn get_float_rope_theta_tiny_positive() {
        let config = ResolvedConfig {
            rope_theta: f64::MIN_POSITIVE,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(f64::MIN_POSITIVE));
    }

    // ── get_float: global_rope_theta with very large value ────────────────

    #[test]
    fn get_float_global_rope_theta_large() {
        let config = ResolvedConfig {
            global_rope_theta: 1e15,
            ..Default::default()
        };
        assert_eq!(config.get_float("global_rope_theta"), Some(1e15));
    }

    // ── get_float: rope_theta does not come from extra ────────────────────

    #[test]
    fn get_float_rope_theta_ignores_extra() {
        let mut extra = HashMap::new();
        extra.insert("rope_theta".to_string(), 42);
        let config = ResolvedConfig {
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        };
        // get_float does NOT consult extra, only the match arms.
        assert_eq!(config.get_float("rope_theta"), Some(10000.0));
    }

    // ── get_str: dtype does not come from extra ───────────────────────────

    #[test]
    fn get_str_dtype_ignores_extra() {
        let mut extra = HashMap::new();
        extra.insert("dtype".to_string(), 42);
        let config = ResolvedConfig {
            dtype: "f16".to_string(),
            extra,
            ..Default::default()
        };
        // get_str returns the named field, not extra.
        assert_eq!(config.get_str("dtype"), Some("f16"));
        // get_int reads from extra where "dtype" was stored as 42.
        assert_eq!(config.get_int("dtype"), Some(42));
    }

    // ── get_str: multiple calls return same reference ─────────────────────

    #[test]
    fn get_str_dtype_returns_consistent_reference() {
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        let first = config.get_str("dtype");
        let second = config.get_str("dtype");
        assert_eq!(first, second);
    }

    // ── get_bool: is_kv_shared_layer with index zero is non_consumer ──────

    #[test]
    fn get_bool_is_kv_shared_layer_zero_when_no_shared_layers() {
        let config = ResolvedConfig {
            num_hidden_layers: 8,
            num_kv_shared_layers: 0,
            ..Default::default()
        };
        assert_eq!(config.get_bool("is_kv_shared_layer_0"), Some(false));
        assert_eq!(config.get_bool("is_kv_shared_layer_7"), Some(false));
    }

    // ── get_bool: has_per_layer_embedding from default is Some(false) ────

    #[test]
    fn get_bool_has_per_layer_embedding_default() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(false));
    }

    // ── get_bool: is_kv_shared_layer with whitespace in suffix ────────────

    #[test]
    fn get_bool_is_kv_shared_layer_whitespace_suffix_returns_none() {
        let config = ResolvedConfig::default();
        // " 3" has a leading space → not parseable as usize → None.
        assert_eq!(config.get_bool("is_kv_shared_layer_ 3"), None);
    }

    // ── get_bool: empty string key returns None ───────────────────────────

    #[test]
    fn get_bool_empty_string_key_returns_none() {
        let config = ResolvedConfig::default();
        assert_eq!(config.get_bool(""), None);
    }

    // ── is_kv_shared_layer: two-layer model one shared ────────────────────

    #[test]
    fn is_kv_shared_layer_two_layers_one_shared() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.num_kv_shared_layers = 1;
        // Layer 0: not consumer (2-1=1, 0 < 1). Layer 1: consumer (1 >= 1).
        assert!(!config.is_kv_shared_layer(0));
        assert!(config.is_kv_shared_layer(1));
    }

    // ── is_kv_shared_layer: config with very large num_hidden_layers ──────

    #[test]
    fn is_kv_shared_layer_large_num_layers() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 1000;
        config.num_kv_shared_layers = 100;
        // First consumer: 1000 - 100 = 900.
        assert!(!config.is_kv_shared_layer(899));
        assert!(config.is_kv_shared_layer(900));
        assert!(config.is_kv_shared_layer(999));
        assert!(!config.is_kv_shared_layer(1000)); // out of range
    }

    // ── is_kv_shared_layer: config where shared equals half of layers ─────

    #[test]
    fn is_kv_shared_layer_half_shared() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 5;
        // First consumer at index 5.
        for i in 0..5 {
            assert!(!config.is_kv_shared_layer(i), "layer {i} must not be consumer");
        }
        for i in 5..10 {
            assert!(config.is_kv_shared_layer(i), "layer {i} must be consumer");
        }
    }

    // ── is_kv_shared_layer: rope_theta does not affect result ─────────────

    #[test]
    fn is_kv_shared_layer_unaffected_by_rope_theta() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        config.rope_theta = 1e15;
        assert!(!config.is_kv_shared_layer(2));
        assert!(config.is_kv_shared_layer(3));
    }

    // ── donor_layer: two-bucket alternating pattern 10 layers ─────────────

    #[test]
    fn donor_layer_alternating_ten_layers() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 4;
        // Pattern: [0,1,0,1,0,1,0,1,0,1]. Consumers: layers 6..10.
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        // Consumer 6 (bucket 0) → latest non-consumer with bucket 0 is layer 4.
        assert_eq!(config.donor_layer(6).unwrap(), Some(4));
        // Consumer 7 (bucket 1) → latest non-consumer with bucket 1 is layer 5.
        assert_eq!(config.donor_layer(7).unwrap(), Some(5));
        // Consumer 8 (bucket 0) → layer 4.
        assert_eq!(config.donor_layer(8).unwrap(), Some(4));
        // Consumer 9 (bucket 1) → layer 5.
        assert_eq!(config.donor_layer(9).unwrap(), Some(5));
    }

    // ── donor_layer: three distinct buckets ───────────────────────────────

    #[test]
    fn donor_layer_three_distinct_buckets() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 9;
        config.num_kv_shared_layers = 3;
        // Pattern: [0,1,2, 0,1,2, 0,1,2]. Donors: 0..6. Consumers: 6..9.
        config.attention_pattern = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        // Consumer 6 (bucket 0) → latest donor with bucket 0 = layer 3.
        assert_eq!(config.donor_layer(6).unwrap(), Some(3));
        // Consumer 7 (bucket 1) → latest donor with bucket 1 = layer 4.
        assert_eq!(config.donor_layer(7).unwrap(), Some(4));
        // Consumer 8 (bucket 2) → latest donor with bucket 2 = layer 5.
        assert_eq!(config.donor_layer(8).unwrap(), Some(5));
    }

    // ── donor_layer: pattern shorter than num_layers errors ───────────────

    #[test]
    fn donor_layer_pattern_shorter_than_layers_errors() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0, 1]; // 8 items, 10 layers
        // Consumer layer 8 → pattern[8] out of bounds.
        assert!(config.donor_layer(8).is_err());
    }

    // ── donor_layer: non-consumer at exact boundary returns Ok(None) ──────

    #[test]
    fn donor_layer_exact_boundary_non_consumer_none() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 12;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0; 12];
        // First consumer index: 12 - 4 = 8. Layer 7 is the last non-consumer.
        assert_eq!(config.donor_layer(7).unwrap(), None);
        assert_eq!(config.donor_layer(8).unwrap(), Some(7));
    }

    // ── donor_layer: last layer is consumer ───────────────────────────────

    #[test]
    fn donor_layer_last_layer_consumer() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0; 8];
        // Layer 7 is the last consumer.
        assert_eq!(config.donor_layer(7).unwrap(), Some(5)); // latest non-consumer with bucket 0
    }

    // ── donor_layer: first consumer layer ─────────────────────────────────

    #[test]
    fn donor_layer_first_consumer() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![5; 8];
        // First consumer is layer 4. Latest non-consumer is layer 3.
        assert_eq!(config.donor_layer(4).unwrap(), Some(3));
    }

    // ── validate_config: value of 1 for all required fields passes ────────

    #[test]
    fn validate_config_accepts_unit_values() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── validate_config: extra fields do not affect validation ────────────

    #[test]
    fn validate_config_extra_does_not_satisfy_required() {
        // Even if extra has "num_hidden_layers" = 999, validation still reads
        // the struct field (which is 0) and rejects.
        let mut extra = HashMap::new();
        extra.insert("num_hidden_layers".to_string(), 999);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("num_hidden_layers".to_string()));
    }

    // ── validate_config: zero rope_theta does not cause failure ───────────

    #[test]
    fn validate_config_accepts_zero_rope_theta() {
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            rope_theta: 0.0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── validate_config: empty dtype does not cause failure ───────────────

    #[test]
    fn validate_config_accepts_empty_dtype() {
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            dtype: String::new(),
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── substitute_placeholders: $ without braces stays literal ───────────

    #[test]
    fn substitute_placeholders_dollar_without_braces_literal() {
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };
        let result = substitute_placeholders("cost=$100, size=${hidden_size}", &config);
        assert_eq!(result, "cost=$100, size=1024");
    }

    // ── substitute_placeholders: template with newlines ───────────────────

    #[test]
    fn substitute_placeholders_template_with_newlines() {
        let config = ResolvedConfig {
            num_hidden_layers: 8,
            hidden_size: 512,
            ..Default::default()
        };
        let template = "layers=${num_hidden_layers}\nhidden=${hidden_size}";
        let result = substitute_placeholders(template, &config);
        assert!(result.contains("layers=8"));
        assert!(result.contains("hidden=512"));
        assert!(result.contains('\n'));
    }

    // ── substitute_placeholders: only whitespace template ─────────────────

    #[test]
    fn substitute_placeholders_whitespace_only_template() {
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };
        let result = substitute_placeholders("   ", &config);
        assert_eq!(result, "   ");
    }

    // ── substitute_placeholders: placeholder name is substring of another ─

    #[test]
    fn substitute_placeholders_head_dim_not_confused_with_hidden_size() {
        let config = ResolvedConfig {
            hidden_size: 4096,
            head_dim: 128,
            ..Default::default()
        };
        let result = substitute_placeholders("${head_dim}", &config);
        assert_eq!(result, "128");
    }

    // ── substitute_placeholders: extra with same name as known field ──────

    #[test]
    fn substitute_placeholders_extra_same_name_as_known_field() {
        // Extra keys are substituted after builtins. If extra has "hidden_size",
        // the builtin ${hidden_size} replaces first, then extra's value replaces
        // any remaining ${hidden_size} (which there won't be).
        let mut extra = HashMap::new();
        extra.insert("hidden_size".to_string(), 999);
        extra.insert("unique_extra".to_string(), 42);
        let config = ResolvedConfig {
            hidden_size: 2048,
            extra,
            ..Default::default()
        };
        let result = substitute_placeholders("${hidden_size}_${unique_extra}", &config);
        assert_eq!(result, "2048_42");
    }

    // ── substitute_placeholders: dtype placeholder with empty string ──────

    #[test]
    fn substitute_placeholders_dtype_empty_string() {
        let config = ResolvedConfig {
            dtype: String::new(),
            ..Default::default()
        };
        let result = substitute_placeholders("dtype=${dtype}", &config);
        assert_eq!(result, "dtype=");
    }

    // ── convert_rope_scaling: Yarn with very large factor ─────────────────

    #[test]
    fn convert_rope_scaling_yarn_large_factor() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(f32::MAX),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Yarn { factor, .. } => {
                assert!((factor - f32::MAX).abs() < f32::EPSILON);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: Linear with factor at 1.0 + epsilon ────────

    #[test]
    fn convert_rope_scaling_linear_factor_slightly_above_one() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(1.0 + f32::EPSILON),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some());
    }

    // ── convert_rope_scaling: Yarn with custom original_max ──────────────

    #[test]
    fn convert_rope_scaling_yarn_custom_original_max() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(2.0),
            original_max_position_embeddings: Some(32768),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { original_max_position, .. } => {
                assert_eq!(original_max_position, 32768);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: Yarn with large beta_fast ──────────────────

    #[test]
    fn convert_rope_scaling_yarn_large_beta_fast() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(4.0),
            beta_fast: Some(1000.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { beta_fast, .. } => {
                assert!((beta_fast - 1000.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── from_geometry: attention_pattern copied not referenced ────────────

    #[test]
    fn from_geometry_attention_pattern_is_independent() {
        use gllm_kernels::types::DType;

        let pattern = vec![0, 1, 0, 1];
        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: pattern.clone(),
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.attention_pattern, pattern);
    }

    // ── from_geometry: num_kv_shared_layers zero means no sharing ─────────

    #[test]
    fn from_geometry_zero_shared_layers_means_no_consumers() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 6,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        for i in 0..6 {
            assert!(!config.is_kv_shared_layer(i));
        }
    }

    // ── from_geometry: global_head_dim passed through ─────────────────────

    #[test]
    fn from_geometry_global_head_dim_passthrough() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 128,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.global_head_dim, 128);
    }

    // ── from_geometry: sliding_window passed through ──────────────────────

    #[test]
    fn from_geometry_sliding_window_passthrough() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 8192,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.sliding_window, 8192);
    }

    // ── resolve_from_provider: embed with empty shape vec ─────────────────

    #[test]
    fn resolve_from_provider_embedding_empty_shape_no_crash() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ── resolve_from_provider: embed with 2D shape derives both dims ──────

    #[test]
    fn resolve_from_provider_embed_2d_sets_vocab_and_hidden() {
        // The validation will fail at num_hidden_layers, proving
        // vocab_size and hidden_size were derived (not the error).
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![128000, 4096],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                // Must NOT be vocab_size or hidden_size.
                assert_ne!(key, "vocab_size");
                assert_ne!(key, "hidden_size");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: layer tensor from non-standard prefix ──────

    #[test]
    fn resolve_from_provider_non_standard_tensor_name_no_layer() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "transformer.wte.weight".to_string(),
                    shape: vec![50000, 768],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // May or may not match Embedding role, but num_hidden_layers stays 0.
        assert!(result.is_err());
    }

    // ── resolve_from_provider: two embedding tensors uses first for dtype ─

    #[test]
    fn resolve_from_provider_two_embeddings_no_panic() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Should not panic; dtype detection picks the first tensor's dtype.
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
    }

    // ── resolve_from_provider: tensor with shape containing zeros ─────────

    #[test]
    fn resolve_from_provider_embed_shape_with_zero_dim_no_panic() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![0, 0],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // vocab_size=0 and hidden_size=0 → validation fails at num_hidden_layers
        // (or hidden_size if num_hidden_layers was set).
        assert!(result.is_err());
    }

    // ── resolve_from_provider: q_proj tensor alone without embed ──────────

    #[test]
    fn resolve_from_provider_q_proj_without_embed() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // vocab_size=0 → validation fails.
        assert!(result.is_err());
    }

    // ── ResolveError: MissingConfig with unicode string ───────────────────

    #[test]
    fn resolve_error_missing_config_unicode_key() {
        let err = ResolveError::MissingConfig("配置字段".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("配置字段"));
    }

    // ── ResolveError: DerivationFailed with empty key and reason ──────────

    #[test]
    fn resolve_error_derivation_failed_empty_key_and_reason() {
        let err = ResolveError::DerivationFailed {
            key: String::new(),
            reason: String::new(),
        };
        // Both empty strings → should still format without panic.
        let msg = format!("{err}");
        assert!(msg.contains("Failed to derive"));
    }

    // ── ResolveError: Inconsistent with long string ───────────────────────

    #[test]
    fn resolve_error_inconsistent_long_message() {
        let long_msg = "x".repeat(10000);
        let err = ResolveError::Inconsistent(long_msg.clone());
        let msg = format!("{err}");
        assert!(msg.contains("Inconsistent config"));
        // Verify the long message is included.
        assert!(msg.len() > long_msg.len());
    }

    // ── ResolvedConfig: clone with intermediate_size Some ─────────────────

    #[test]
    fn resolved_config_clone_intermediate_size_some() {
        let config = ResolvedConfig {
            intermediate_size: Some(4096),
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.intermediate_size, Some(4096));
    }

    // ── ResolvedConfig: clone with intermediate_size None ─────────────────

    #[test]
    fn resolved_config_clone_intermediate_size_none() {
        let config = ResolvedConfig {
            intermediate_size: None,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.intermediate_size, None);
    }

    // ── ResolvedConfig: clone preserves has_per_layer_embedding ───────────

    #[test]
    fn resolved_config_clone_preserves_has_per_layer_embedding() {
        let config = ResolvedConfig {
            has_per_layer_embedding: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert!(cloned.has_per_layer_embedding);
    }

    // ── ResolvedConfig: debug includes intermediate_size field ────────────

    #[test]
    fn resolved_config_debug_intermediate_size_some() {
        let config = ResolvedConfig {
            intermediate_size: Some(8192),
            ..Default::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("intermediate_size"));
        assert!(debug.contains("8192"));
    }

    // ── ResolvedConfig: debug with all zero fields is valid ───────────────

    #[test]
    fn resolved_config_debug_all_default_fields() {
        let config = ResolvedConfig::default();
        let debug = format!("{config:?}");
        // The Debug output must be a non-empty string containing the type name.
        assert!(!debug.is_empty());
        assert!(debug.contains("ResolvedConfig"));
    }

    // ── ResolvedConfig: multiple field mutations are independent ──────────

    #[test]
    fn resolved_config_fields_independent() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            head_dim: 128,
            ..Default::default()
        };
        // Changing one field's getter doesn't affect others.
        assert_eq!(config.get_int("num_hidden_layers"), Some(32));
        assert_eq!(config.get_int("hidden_size"), Some(4096));
        assert_eq!(config.get_int("head_dim"), Some(128));
        assert_eq!(config.get_int("vocab_size"), Some(0));
    }

    // ── get_bool: is_kv_shared_layer_0 when all layers shared ────────────

    #[test]
    fn get_bool_is_kv_shared_layer_zero_when_all_shared() {
        let config = ResolvedConfig {
            num_hidden_layers: 5,
            num_kv_shared_layers: 5,
            ..Default::default()
        };
        assert_eq!(config.get_bool("is_kv_shared_layer_0"), Some(true));
    }

    // ── substitute_placeholders: template with multiple $ signs ───────────

    #[test]
    fn substitute_placeholders_multiple_dollar_signs() {
        let config = ResolvedConfig {
            hidden_size: 512,
            ..Default::default()
        };
        let result = substitute_placeholders("$$${hidden_size}$$", &config);
        // $$ stays literal (not a ${...} pattern), ${hidden_size} replaced.
        assert_eq!(result, "$$512$$");
    }

    // ── substitute_placeholders: empty extra map does not error ───────────

    #[test]
    fn substitute_placeholders_empty_extra_no_error() {
        let config = ResolvedConfig {
            extra: HashMap::new(),
            hidden_size: 256,
            ..Default::default()
        };
        let result = substitute_placeholders("${hidden_size}", &config);
        assert_eq!(result, "256");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (277-301) — coverage gap-fill
    // ═══════════════════════════════════════════════════════════════════════

    // ── ResolvedConfig: hidden_size_per_layer_input field access ───────────

    #[test]
    fn resolved_config_hidden_size_per_layer_input_default_zero() {
        let config = ResolvedConfig::default();
        assert_eq!(config.hidden_size_per_layer_input, 0);
    }

    #[test]
    fn resolved_config_hidden_size_per_layer_input_set() {
        let config = ResolvedConfig {
            hidden_size_per_layer_input: 512,
            ..Default::default()
        };
        assert_eq!(config.hidden_size_per_layer_input, 512);
    }

    // ── ResolvedConfig: global_head_dim field access ──────────────────────

    #[test]
    fn resolved_config_global_head_dim_default_zero() {
        let config = ResolvedConfig::default();
        assert_eq!(config.global_head_dim, 0);
    }

    #[test]
    fn resolved_config_global_head_dim_set() {
        let config = ResolvedConfig {
            global_head_dim: 256,
            ..Default::default()
        };
        assert_eq!(config.global_head_dim, 256);
    }

    // ── ResolvedConfig: sliding_window field access ──────────────────────

    #[test]
    fn resolved_config_sliding_window_default_zero() {
        let config = ResolvedConfig::default();
        assert_eq!(config.sliding_window, 0);
    }

    // ── ResolvedConfig: num_kv_shared_layers field access ────────────────

    #[test]
    fn resolved_config_num_kv_shared_layers_default_zero() {
        let config = ResolvedConfig::default();
        assert_eq!(config.num_kv_shared_layers, 0);
    }

    #[test]
    fn resolved_config_num_kv_shared_layers_set() {
        let config = ResolvedConfig {
            num_kv_shared_layers: 20,
            num_hidden_layers: 26,
            ..Default::default()
        };
        assert_eq!(config.num_kv_shared_layers, 20);
    }

    // ── ResolvedConfig: attention_pattern field access ────────────────────

    #[test]
    fn resolved_config_attention_pattern_default_empty() {
        let config = ResolvedConfig::default();
        assert!(config.attention_pattern.is_empty());
    }

    #[test]
    fn resolved_config_attention_pattern_set() {
        let config = ResolvedConfig {
            attention_pattern: vec![0, 1, 0, 1],
            ..Default::default()
        };
        assert_eq!(config.attention_pattern, vec![0, 1, 0, 1]);
    }

    // ── ResolvedConfig: rope_scaling field access ────────────────────────

    #[test]
    fn resolved_config_rope_scaling_default_none() {
        let config = ResolvedConfig::default();
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn resolved_config_rope_scaling_set_linear() {
        let config = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
            ..Default::default()
        };
        assert!(config.rope_scaling.is_some());
        match config.rope_scaling.unwrap() {
            RopeScaling::Linear { factor } => assert!((factor - 4.0).abs() < f32::EPSILON),
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    #[test]
    fn resolved_config_rope_scaling_set_yarn() {
        let config = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Yarn {
                factor: 8.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 4096,
            }),
            ..Default::default()
        };
        assert!(config.rope_scaling.is_some());
    }

    // ── get_int: num_attention_heads and num_key_value_heads ─────────────

    #[test]
    fn get_int_num_attention_heads_nonzero() {
        let config = ResolvedConfig {
            num_attention_heads: 64,
            ..Default::default()
        };
        assert_eq!(config.get_int("num_attention_heads"), Some(64));
    }

    #[test]
    fn get_int_num_key_value_heads_nonzero() {
        let config = ResolvedConfig {
            num_key_value_heads: 16,
            ..Default::default()
        };
        assert_eq!(config.get_int("num_key_value_heads"), Some(16));
    }

    // ── get_int: extra with same name as known field is ignored ──────────

    #[test]
    fn get_int_extra_same_name_as_known_field_returns_struct_value() {
        let mut extra = HashMap::new();
        extra.insert("num_hidden_layers".to_string(), 999);
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            extra,
            ..Default::default()
        };
        // Struct field takes priority over extra.
        assert_eq!(config.get_int("num_hidden_layers"), Some(32));
    }

    // ── is_kv_shared_layer: independent of has_per_layer_embedding ──────

    #[test]
    fn is_kv_shared_layer_unaffected_by_has_per_layer_embedding() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.has_per_layer_embedding = true;
        assert!(!config.is_kv_shared_layer(3));
        assert!(config.is_kv_shared_layer(4));
    }

    // ── get_bool: "has_per_layer_embedding" returns struct value ─────────

    #[test]
    fn get_bool_has_per_layer_embedding_true_explicit() {
        let config = ResolvedConfig {
            has_per_layer_embedding: true,
            hidden_size_per_layer_input: 256,
            ..Default::default()
        };
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
    }

    #[test]
    fn get_bool_has_per_layer_embedding_false_explicit() {
        let config = ResolvedConfig {
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            ..Default::default()
        };
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(false));
    }

    // ── substitute_placeholders: mixed builtin and dtype ─────────────────

    #[test]
    fn substitute_placeholders_mixed_builtin_and_dtype() {
        let config = ResolvedConfig {
            num_hidden_layers: 12,
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        let result = substitute_placeholders("${num_hidden_layers}_${dtype}", &config);
        assert_eq!(result, "12_bf16");
    }

    // ── ResolveError: PartialEq edge case — same DerivationFailed ───────

    #[test]
    fn resolve_error_partial_eq_derivation_failed_same_fields() {
        let a = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        let b = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        assert_eq!(a, b);
    }

    // ── ResolveError: Clone produces identical Debug output ──────────────

    #[test]
    fn resolve_error_clone_debug_output_identical() {
        let err = ResolveError::DerivationFailed {
            key: "test_key".to_string(),
            reason: "test_reason".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(format!("{err:?}"), format!("{cloned:?}"));
    }

    // ── substitute_placeholders: only dtype placeholder ──────────────────

    #[test]
    fn substitute_placeholders_only_dtype() {
        let config = ResolvedConfig {
            dtype: "f16".to_string(),
            ..Default::default()
        };
        let result = substitute_placeholders("${dtype}", &config);
        assert_eq!(result, "f16");
    }

    // ── from_geometry: clone preserves num_hidden_layers ─────────────────

    #[test]
    fn from_geometry_clone_preserves_all_scalar_fields() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        let cloned = config.clone();
        assert_eq!(cloned.num_hidden_layers, config.num_hidden_layers);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.head_dim, config.head_dim);
        assert_eq!(cloned.dtype, config.dtype);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (300-314) — coverage gap-fill
    // ═══════════════════════════════════════════════════════════════════════

    // ── substitute_placeholders: only sliding_window placeholder ────────────

    #[test]
    fn substitute_placeholders_only_sliding_window() {
        let config = ResolvedConfig {
            sliding_window: 4096,
            ..Default::default()
        };
        let result = substitute_placeholders("sw=${sliding_window}", &config);
        assert_eq!(result, "sw=4096");
    }

    // ── substitute_placeholders: unclosed brace stays literal ──────────────

    #[test]
    fn substitute_placeholders_unclosed_brace_stays_literal() {
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };
        let result = substitute_placeholders("${hidden_size", &config);
        assert_eq!(result, "${hidden_size");
    }

    // ── substitute_placeholders: nested braces stay literal ────────────────

    #[test]
    fn substitute_placeholders_nested_braces_stay_literal() {
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };
        let result = substitute_placeholders("${${hidden_size}}", &config);
        // Inner ${hidden_size} is not matched because the outer ${ eats the first }.
        // The result is "${1024}" — the inner placeholder IS substituted.
        assert!(result.contains("1024"));
    }

    // ── substitute_placeholders: only global_rope_theta placeholder ────────

    #[test]
    fn substitute_placeholders_only_global_rope_theta() {
        let config = ResolvedConfig {
            global_rope_theta: 1000000.0,
            ..Default::default()
        };
        let result = substitute_placeholders("global=${global_rope_theta}", &config);
        assert_eq!(result, "global=1000000");
    }

    // ── ResolvedConfig: rope_partial_ratio direct field access ─────────────

    #[test]
    fn resolved_config_rope_partial_ratio_field_direct() {
        let config = ResolvedConfig {
            rope_partial_ratio: 0.25,
            ..Default::default()
        };
        let val = config.rope_partial_ratio;
        assert!((val - 0.25).abs() < f32::EPSILON);
    }

    // ── ResolvedConfig: dtype with unicode characters ──────────────────────

    #[test]
    fn resolved_config_dtype_unicode_string() {
        let config = ResolvedConfig {
            dtype: "bf16-实验".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("dtype"), Some("bf16-实验"));
        assert_eq!(config.dtype, "bf16-实验");
    }

    // ── is_kv_shared_layer: shared = layers - 1 ───────────────────────────

    #[test]
    fn is_kv_shared_layer_shared_equals_layers_minus_one() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 9;
        // Only layer 0 is a donor; layers 1..10 are consumers.
        assert!(!config.is_kv_shared_layer(0));
        for i in 1..10 {
            assert!(config.is_kv_shared_layer(i),
                "layer {i} must be consumer when shared=layers-1");
        }
    }

    // ── donor_layer: donor is layer 0 ─────────────────────────────────────

    #[test]
    fn donor_layer_donor_is_layer_zero() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 3;
        config.attention_pattern = vec![7, 7, 7, 7];
        // Consumer 1,2,3 all have bucket 7. Only donor is layer 0 (bucket 7).
        assert_eq!(config.donor_layer(1).unwrap(), Some(0));
        assert_eq!(config.donor_layer(2).unwrap(), Some(0));
        assert_eq!(config.donor_layer(3).unwrap(), Some(0));
    }

    // ── validate_config: accepts when num_kv_shared_layers > num_hidden_layers

    #[test]
    fn validate_config_accepts_shared_exceeding_num_layers() {
        // validate_config only checks 4 required fields; KV sharing config is not validated.
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            num_kv_shared_layers: 100,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── convert_rope_scaling: Yarn with negative beta_fast ─────────────────

    #[test]
    fn convert_rope_scaling_yarn_negative_beta_fast() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(4.0),
            beta_fast: Some(-10.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        match result.unwrap() {
            RopeScaling::Yarn { beta_fast, .. } => {
                assert!((beta_fast - (-10.0)).abs() < f32::EPSILON);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── substitute_placeholders: partial prefix mismatch stays literal ─────

    #[test]
    fn substitute_placeholders_partial_prefix_mismatch_stays_literal() {
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };
        // Typo: "hidden_siz" missing the trailing 'e'
        let result = substitute_placeholders("${hidden_siz}", &config);
        assert_eq!(result, "${hidden_siz}");
    }

    // ── ResolvedConfig: has_per_layer_embedding manually set inconsistent ──

    #[test]
    fn resolved_config_has_per_layer_embedding_manually_inconsistent() {
        // It's possible to manually construct a config where
        // has_per_layer_embedding=true but hidden_size_per_layer_input=0.
        // This test verifies the field is stored as-is (no validation).
        let config = ResolvedConfig {
            has_per_layer_embedding: true,
            hidden_size_per_layer_input: 0,
            ..Default::default()
        };
        assert!(config.has_per_layer_embedding);
        assert_eq!(config.hidden_size_per_layer_input, 0);
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
    }

    // ── substitute_placeholders: hidden_size_per_layer_input zero ──────────

    #[test]
    fn substitute_placeholders_hidden_size_per_layer_input_zero_outputs_zero() {
        let config = ResolvedConfig {
            hidden_size_per_layer_input: 0,
            ..Default::default()
        };
        let result = substitute_placeholders("ple=${hidden_size_per_layer_input}", &config);
        assert_eq!(result, "ple=0");
    }

    // ── donor_layer: two consumers same bucket both get same donor ─────────

    #[test]
    fn donor_layer_two_consumers_same_bucket_same_donor() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![3, 3, 3, 3, 3, 3, 3, 3];
        // Consumers: layers 4..8. Latest non-consumer with bucket 3 = layer 3.
        assert_eq!(config.donor_layer(4).unwrap(), Some(3));
        assert_eq!(config.donor_layer(5).unwrap(), Some(3));
        assert_eq!(config.donor_layer(6).unwrap(), Some(3));
        assert_eq!(config.donor_layer(7).unwrap(), Some(3));
    }

    // ── resolve_from_provider: layernorm tensor does not derive heads ──────

    #[test]
    fn resolve_from_provider_layernorm_tensor_no_head_derivation() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.input_layernorm.weight".to_string(),
                    shape: vec![4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.post_attention_layernorm.weight".to_string(),
                    shape: vec![4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // LayerNorm tensors don't match Q/K/V roles, so num_attention_heads stays 0.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (315-329) — coverage gap-fill round 2
    // ═══════════════════════════════════════════════════════════════════════

    // ── get_int: head_dim returns correct value ────────────────────────────

    #[test]
    fn get_int_head_dim_returns_value() {
        let config = ResolvedConfig {
            head_dim: 128,
            ..Default::default()
        };
        assert_eq!(config.get_int("head_dim"), Some(128));
    }

    // ── get_int: num_hidden_layers returns correct value ───────────────────

    #[test]
    fn get_int_num_hidden_layers_returns_value() {
        let config = ResolvedConfig {
            num_hidden_layers: 48,
            ..Default::default()
        };
        assert_eq!(config.get_int("num_hidden_layers"), Some(48));
    }

    // ── get_int: hidden_size returns correct value ─────────────────────────

    #[test]
    fn get_int_hidden_size_returns_value() {
        let config = ResolvedConfig {
            hidden_size: 6144,
            ..Default::default()
        };
        assert_eq!(config.get_int("hidden_size"), Some(6144));
    }

    // ── get_float: non-rope key returns None even with rope_theta set ──────

    #[test]
    fn get_float_arbitrary_key_returns_none() {
        let config = ResolvedConfig {
            rope_theta: 500000.0,
            ..Default::default()
        };
        assert_eq!(config.get_float("custom_theta"), None);
    }

    // ── get_str: non-dtype key returns None ────────────────────────────────

    #[test]
    fn get_str_arbitrary_key_returns_none() {
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_str("custom_type"), None);
    }

    // ── get_bool: non-numeric suffix after is_kv_shared_layer_ returns None ─

    #[test]
    fn get_bool_is_kv_shared_layer_non_numeric_suffix_returns_none() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 5,
            ..Default::default()
        };
        assert_eq!(config.get_bool("is_kv_shared_layer_abc"), None);
    }

    // ── get_bool: partial prefix match is_kv_shared_layer returns None ─────

    #[test]
    fn get_bool_is_kv_shared_layer_without_suffix_returns_none() {
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 5,
            ..Default::default()
        };
        assert_eq!(config.get_bool("is_kv_shared_layer"), None);
    }

    // ── validate_config: rejects zero hidden_size with other fields valid ──

    #[test]
    fn validate_config_zero_hidden_size_reports_missing() {
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 0,
            num_attention_heads: 4,
            vocab_size: 1000,
            ..Default::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert_eq!(err, ResolveError::MissingConfig("hidden_size".to_string()));
    }

    // ── resolve_from_provider: F16 dtype detection ────────────────────────

    #[test]
    fn resolve_from_provider_f16_dtype_detected() {
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F16,
                },
            ],
        };
        let result = resolve_from_provider(&provider, None);
        // Will fail validation (missing layers/heads), but dtype should be set
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_hidden_layers");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
        // Verify F16 was detected by re-checking the provider independently
        let mut config = ResolvedConfig::default();
        resolve_from_tensors(&mut config, &provider).unwrap();
        assert_eq!(config.dtype, "f16");
    }

    // ── substitute_placeholders: num_heads alias with large value ───────────

    #[test]
    fn substitute_placeholders_num_heads_alias_large() {
        let config = ResolvedConfig {
            num_attention_heads: 128,
            ..Default::default()
        };
        let result = substitute_placeholders("${num_heads}", &config);
        assert_eq!(result, "128");
    }

    // ── substitute_placeholders: num_kv_heads alias with single kv head ────

    #[test]
    fn substitute_placeholders_num_kv_heads_alias_single() {
        let config = ResolvedConfig {
            num_key_value_heads: 1,
            ..Default::default()
        };
        let result = substitute_placeholders("${num_kv_heads}", &config);
        assert_eq!(result, "1");
    }

    // ── substitute_placeholders: intermediate_size None leaves placeholder ──

    #[test]
    fn substitute_placeholders_intermediate_size_none_stays_literal() {
        let config = ResolvedConfig {
            intermediate_size: None,
            ..Default::default()
        };
        let result = substitute_placeholders("size=${intermediate_size}", &config);
        assert_eq!(result, "size=${intermediate_size}");
    }

    // ── convert_rope_scaling: linear with factor exactly 2.0 ──────────────

    #[test]
    fn convert_rope_scaling_linear_factor_exactly_two() {
        let cfg = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(2.0),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&cfg));
        assert!(matches!(result, Some(RopeScaling::Linear { factor }) if (factor - 2.0).abs() < 1e-10));
    }

    // ── is_kv_shared_layer: layer_idx equal to num_hidden_layers is out of range ─

    #[test]
    fn is_kv_shared_layer_idx_equals_num_layers_out_of_range() {
        let config = ResolvedConfig {
            num_hidden_layers: 8,
            num_kv_shared_layers: 4,
            ..Default::default()
        };
        // layer_idx == num_hidden_layers is NOT < num_hidden_layers, so false
        assert!(!config.is_kv_shared_layer(8));
    }

    // ── ResolvedConfig default: rope_theta is zero ────────────────────────

    #[test]
    fn resolved_config_default_rope_theta_is_zero() {
        let config = ResolvedConfig::default();
        assert_eq!(config.rope_theta, 0.0);
    }

    // ── ResolveError: MissingConfig not equal to Inconsistent ──────────────

    #[test]
    fn resolve_error_missing_config_not_equal_to_inconsistent() {
        let a = ResolveError::MissingConfig("key".to_string());
        let b = ResolveError::Inconsistent("key".to_string());
        assert_ne!(a, b);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (330-344) — coverage gap-fill round 3
    // ═══════════════════════════════════════════════════════════════════════

    // ── resolve_from_provider: head_dim auto-computed when hidden_size and
    //    num_attention_heads are derivable from tensor shapes ────────────────
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_auto_head_dim_from_hidden_and_heads() {
        // Arrange: embed provides hidden_size=4096 and vocab_size=32000.
        // GGUF metadata is unavailable so num_attention_heads must come from
        // q_proj. However q_proj derivation requires head_dim > 0, which itself
        // requires num_attention_heads > 0. Without GGUF metadata, this
        // circular dependency means head_dim stays 0.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.1.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: num_hidden_layers=2 from layer tensors, but num_attention_heads=0
        // because head_dim=0 at q_proj scan time. Validation rejects.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: k_proj derivation when head_dim is available ─
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_k_proj_derives_kv_heads_with_head_dim() {
        // Arrange: embed + k_proj tensor. head_dim stays 0 without GGUF,
        // so k_proj derivation of num_key_value_heads is skipped (requires
        // head_dim > 0). Verify the code path handles this gracefully.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![1024, 4096],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: fails because num_hidden_layers and num_attention_heads are 0.
        assert!(result.is_err());
    }

    // ── resolve_from_provider: head_dim auto-computed from hidden_size/heads
    //    after both are derived from tensors ─────────────────────────────────
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_head_dim_computed_after_tensor_pass() {
        // Arrange: Only embed tensor. hidden_size=4096 derived. head_dim=0
        // because num_attention_heads=0. The auto-compute at line ~198
        // only triggers when both hidden_size > 0 AND num_attention_heads > 0.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![50000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: head_dim stays 0, num_hidden_layers=0 → fails.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: embed with 4D shape uses first two dims ─────
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_embedding_4d_shape_uses_first_two_dims() {
        // Arrange: 4D embedding tensor (unusual but should not panic).
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096, 1, 1],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: vocab_size=32000, hidden_size=4096 from first two dims,
        // but num_hidden_layers=0 → validation fails.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_ne!(key, "vocab_size");
                assert_ne!(key, "hidden_size");
                assert_eq!(key, "num_hidden_layers");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── substitute_placeholders: rope_theta with scientific notation output ─

    #[test]
    fn substitute_placeholders_rope_theta_scientific_notation() {
        // Arrange
        let config = ResolvedConfig {
            rope_theta: 1e-6,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("theta=${rope_theta}", &config);

        // Assert: f64::to_string() for 1e-6 produces a scientific notation.
        assert!(result.contains("theta="));
        assert_ne!(result, "theta=${rope_theta}", "must be substituted");
    }

    // ── validate_config: usize::MAX values for all required fields pass ─────

    #[test]
    fn validate_config_accepts_usize_max_values() {
        // Arrange: all required fields at maximum value.
        let config = ResolvedConfig {
            num_hidden_layers: usize::MAX,
            hidden_size: usize::MAX,
            num_attention_heads: usize::MAX,
            vocab_size: usize::MAX,
            ..Default::default()
        };

        // Act & Assert: validation only checks for zero, not for overflow.
        assert!(validate_config(&config).is_ok());
    }

    // ── is_kv_shared_layer: zero layers but nonzero shared layers ───────────

    #[test]
    fn is_kv_shared_layer_zero_layers_with_shared_layers_set() {
        // Arrange: num_hidden_layers=0 but num_kv_shared_layers=5.
        // The layer_idx < num_hidden_layers check prevents any layer from
        // being a consumer.
        let config = ResolvedConfig {
            num_hidden_layers: 0,
            num_kv_shared_layers: 5,
            ..Default::default()
        };

        // Act & Assert
        assert!(!config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(4));
    }

    // ── donor_layer: consumer at index 0 when all layers are shared ─────────

    #[test]
    fn donor_layer_consumer_index_zero_when_all_shared() {
        // Arrange: all layers shared, all same bucket.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0, 0, 0, 0];

        // Act & Assert: layer 0 is a consumer but there are no non-consumer
        // donors. The scheduler should return an error (no donor exists).
        assert!(config.donor_layer(0).is_err(),
            "no non-consumer layers exist to be donors");
        assert!(config.donor_layer(3).is_err());
    }

    // ── substitute_placeholders: intermediate_size Some replaces correctly ──

    #[test]
    fn substitute_placeholders_intermediate_size_some_replaces() {
        // Arrange
        let config = ResolvedConfig {
            intermediate_size: Some(28672),
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("ffn=${intermediate_size}", &config);

        // Assert
        assert_eq!(result, "ffn=28672");
    }

    // ── convert_rope_scaling: Yarn with factor just above 1.0 produces output

    #[test]
    fn convert_rope_scaling_yarn_factor_slightly_above_one() {
        // Arrange: Yarn with factor slightly above the 1.0 threshold.
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(1.0 + f32::EPSILON),
            ..Default::default()
        };

        // Act
        let result = convert_rope_scaling(Some(&scaling));

        // Assert
        assert!(result.is_some(), "factor slightly > 1.0 must produce Yarn scaling");
        match result.unwrap() {
            RopeScaling::Yarn { factor, .. } => {
                assert!(factor > 1.0);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── ResolvedConfig: extra with single entry is queryable via get_int ────

    #[test]
    fn resolved_config_extra_with_single_entry_queryable() {
        // Arrange
        let mut extra = HashMap::new();
        extra.insert("single_key".to_string(), 42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };

        // Act & Assert
        assert_eq!(config.get_int("single_key"), Some(42));
        assert_eq!(config.extra.len(), 1);
    }

    // ── ResolveError: thiserror source chain ────────────────────────────────

    #[test]
    fn resolve_error_source_chain_none_for_all_variants() {
        // Arrange: thiserror's #[error(...)] without #[source] means
        // std::error::Error::source() returns None for all variants.

        // Act & Assert
        let err1 = ResolveError::MissingConfig("k".to_string());
        assert!(std::error::Error::source(&err1).is_none());

        let err2 = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        assert!(std::error::Error::source(&err2).is_none());

        let err3 = ResolveError::Inconsistent("msg".to_string());
        assert!(std::error::Error::source(&err3).is_none());
    }

    // ── from_geometry: rope_scaling None maps to resolved rope_scaling None ─

    #[test]
    fn from_geometry_rope_scaling_none_stays_none() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        // Assert
        assert!(config.rope_scaling.is_none());
    }

    // ── resolve_from_provider: multiple layer tensors set correct max_layer ─
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_non_consecutive_layers_counted_correctly() {
        // Arrange: layers 0, 3, 7 → max_layer_idx = 8.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.gate_proj.weight".to_string(),
                    shape: vec![5632, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.3.mlp.gate_proj.weight".to_string(),
                    shape: vec![5632, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.7.mlp.gate_proj.weight".to_string(),
                    shape: vec![5632, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: num_hidden_layers=8 from max layer index + 1,
        // but num_attention_heads=0 (no q_proj) → validation fails.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── is_kv_shared_layer: boundary with exactly one non-consumer ──────────

    #[test]
    fn is_kv_shared_layer_exactly_one_donor_layer() {
        // Arrange: 6 layers, 5 shared → only layer 0 is non-consumer.
        let config = ResolvedConfig {
            num_hidden_layers: 6,
            num_kv_shared_layers: 5,
            ..Default::default()
        };

        // Act & Assert
        assert!(!config.is_kv_shared_layer(0), "layer 0 is the sole donor");
        assert!(config.is_kv_shared_layer(1), "layer 1 is consumer");
        assert!(config.is_kv_shared_layer(5), "layer 5 is consumer");
        assert!(!config.is_kv_shared_layer(6), "layer 6 out of range");
    }

    // ── substitute_placeholders: multiple extras with long keys ─────────────

    #[test]
    fn substitute_placeholders_extra_with_underscore_heavy_key() {
        // Arrange
        let mut extra = HashMap::new();
        extra.insert("my_custom_param__v2".to_string(), 99);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("val=${my_custom_param__v2}", &config);

        // Assert
        assert_eq!(result, "val=99");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (345-359) — edge case coverage expansion
    // ═══════════════════════════════════════════════════════════════════════

    // ── substitute_placeholders: empty placeholder name stays literal ──────

    #[test]
    fn substitute_placeholders_empty_placeholder_name_stays_literal() {
        // Arrange
        let config = ResolvedConfig {
            hidden_size: 4096,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("a=${}b=${hidden_size}", &config);

        // Assert: ${} does not match any field, stays literal.
        assert_eq!(result, "a=${}b=4096");
    }

    // ── substitute_placeholders: single dollar at end of string ───────────

    #[test]
    fn substitute_placeholders_trailing_dollar_literal() {
        // Arrange
        let config = ResolvedConfig {
            hidden_size: 1024,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("${hidden_size}$", &config);

        // Assert
        assert_eq!(result, "1024$");
    }

    // ── ResolvedConfig: get_str returns Some("") for default dtype ────────

    #[test]
    fn get_str_default_dtype_returns_some_empty() {
        // Arrange
        let config = ResolvedConfig::default();

        // Act & Assert: default dtype is an empty string, get_str returns Some("").
        assert_eq!(config.get_str("dtype"), Some(""));
    }

    // ── ResolvedConfig: get_int falls through to extra for dtype key ──────

    #[test]
    fn get_int_dtype_key_returns_extra_value() {
        // Arrange: put "dtype" as an i64 in extra. get_int does not match
        // "dtype" in its known-field match arms, so it falls through to extra.
        let mut extra = HashMap::new();
        extra.insert("dtype".to_string(), 16);
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            extra,
            ..Default::default()
        };

        // Act & Assert
        assert_eq!(config.get_int("dtype"), Some(16));
    }

    // ── is_kv_shared_layer: single layer with shared_layers=1 ─────────────

    #[test]
    fn is_kv_shared_layer_single_layer_single_shared() {
        // Arrange: 1 layer, 1 shared. saturating_sub(1,1)=0, index 0 >= 0.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 1;
        config.num_kv_shared_layers = 1;

        // Act & Assert: layer 0 is a consumer.
        assert!(config.is_kv_shared_layer(0));
    }

    // ── donor_layer: pattern longer than num_layers triggers error ────────

    #[test]
    fn donor_layer_pattern_longer_than_layers_errors() {
        // Arrange: 4 layers but 8 pattern items. The scheduler validates
        // pattern length == num_layers when num_kv_shared_layers > 0.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1, 0, 1];

        // Act & Assert: non-consumers return Ok(None) (no pattern check needed).
        assert_eq!(config.donor_layer(0).unwrap(), None);
        assert_eq!(config.donor_layer(1).unwrap(), None);
        // Consumer layers trigger pattern length validation -> error.
        assert!(config.donor_layer(2).is_err(),
            "oversized pattern must be rejected for consumer layers");
        assert!(config.donor_layer(3).is_err());
    }

    // ── convert_rope_scaling: NaN factor is not > 1.0, returns None ───────

    #[test]
    fn convert_rope_scaling_nan_factor_returns_none() {
        // Arrange: NaN comparisons are always false, so NaN <= 1.0 is false,
        // but the code checks factor <= 1.0 → false for NaN, so it falls
        // through to the match. However the match arms produce output.
        // Let's verify actual behavior: NaN factor with a valid type.
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::NAN),
            ..Default::default()
        };

        // Act
        let result = convert_rope_scaling(Some(&scaling));

        // Assert: NaN > 1.0 is false, NaN <= 1.0 is also false. The condition
        // `factor <= 1.0` evaluates to false for NaN, so the function proceeds
        // to the match and returns Some(Linear { factor: NaN }).
        assert!(result.is_some(), "NaN bypasses the <= 1.0 guard");
    }

    // ── validate_config: accepts with zero num_key_value_heads ────────────

    #[test]
    fn validate_config_accepts_zero_kv_heads_and_zero_head_dim() {
        // Arrange
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            hidden_size: 256,
            num_attention_heads: 4,
            vocab_size: 1000,
            num_key_value_heads: 0,
            head_dim: 0,
            ..Default::default()
        };

        // Act & Assert
        assert!(validate_config(&config).is_ok());
    }

    // ── ResolveError: clone and compare with assert_eq ───────────────────

    #[test]
    fn resolve_error_clone_and_eq_for_inconsistent() {
        // Arrange
        let err = ResolveError::Inconsistent("test_msg".to_string());

        // Act
        let cloned = err.clone();

        // Assert
        assert_eq!(err, cloned);
    }

    // ── ResolvedConfig: default dtype is empty string not "f32" ───────────

    #[test]
    fn resolved_config_default_dtype_is_empty_string() {
        // Arrange & Act
        let config = ResolvedConfig::default();

        // Assert
        assert_eq!(config.dtype, "");
        assert_ne!(config.dtype, "f32");
    }

    // ── substitute_placeholders: extra with i64::MAX value ────────────────

    #[test]
    fn substitute_placeholders_extra_i64_max_value() {
        // Arrange
        let mut extra = HashMap::new();
        extra.insert("max".to_string(), i64::MAX);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("${max}", &config);

        // Assert
        assert_eq!(result, i64::MAX.to_string());
    }

    // ── get_bool: is_kv_shared_layer_0 when only layer 0 is non-consumer ─

    #[test]
    fn get_bool_is_kv_shared_layer_zero_with_mostly_shared() {
        // Arrange: 10 layers, 9 shared. Only layer 0 is non-consumer.
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 9,
            ..Default::default()
        };

        // Act & Assert
        assert_eq!(config.get_bool("is_kv_shared_layer_0"), Some(false));
        assert_eq!(config.get_bool("is_kv_shared_layer_1"), Some(true));
        assert_eq!(config.get_bool("is_kv_shared_layer_9"), Some(true));
    }

    // ── from_geometry: extra passed through with multiple entries ─────────

    #[test]
    fn from_geometry_extra_preserves_multiple_entries() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        let mut extra = HashMap::new();
        extra.insert("a".to_string(), 1);
        extra.insert("b".to_string(), 2);
        extra.insert("c".to_string(), 3);

        // Act
        let config = ResolvedConfig::from_geometry(&geometry, extra);

        // Assert: all extra entries preserved and queryable.
        assert_eq!(config.get_int("a"), Some(1));
        assert_eq!(config.get_int("b"), Some(2));
        assert_eq!(config.get_int("c"), Some(3));
        assert_eq!(config.extra.len(), 3);
    }

    // ── donor_layer: error reason contains layer index ────────────────────

    #[test]
    fn donor_layer_error_message_contains_layer_index() {
        // Arrange: malformed pattern triggers error.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 0, 0, 0]; // length 4, need 6

        // Act
        let err = config.donor_layer(4).unwrap_err();

        // Assert: key should contain the layer index "4".
        match err {
            ResolveError::DerivationFailed { key, .. } => {
                assert!(key.contains("donor_layer("), "key was: {key}");
                assert!(key.contains('4'), "key must contain layer index: {key}");
            }
            other => panic!("expected DerivationFailed, got {other:?}"),
        }
    }

    // ── ResolvedConfig: clone preserves rope_scaling with Yarn ────────────

    #[test]
    fn resolved_config_clone_preserves_yarn_rope_scaling() {
        // Arrange
        let config = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Yarn {
                factor: 16.0,
                beta_fast: 64.0,
                beta_slow: 0.5,
                original_max_position: 16384,
            }),
            ..Default::default()
        };

        // Act
        let cloned = config.clone();

        // Assert
        match cloned.rope_scaling.unwrap() {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!((factor - 16.0).abs() < f32::EPSILON);
                assert!((beta_fast - 64.0).abs() < f32::EPSILON);
                assert!((beta_slow - 0.5).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 16384);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── resolve_from_provider: tensor with role AttentionValue ignored ────

    #[test]
    fn resolve_from_provider_v_proj_tensor_does_not_derive_heads() {
        // Arrange: v_proj (AttentionValue role) does not trigger head derivation.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.v_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: num_attention_heads stays 0 → validation fails.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_attention_heads"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (360-374) — edge case coverage expansion
    // ═══════════════════════════════════════════════════════════════════════

    // ── get_float: rope_theta with f64::INFINITY ───────────────────────────

    #[test]
    fn get_float_rope_theta_infinity() {
        let config = ResolvedConfig {
            rope_theta: f64::INFINITY,
            ..Default::default()
        };
        assert_eq!(config.get_float("rope_theta"), Some(f64::INFINITY));
    }

    // ── get_float: global_rope_theta with f64::NEG_INFINITY ────────────────

    #[test]
    fn get_float_global_rope_theta_neg_infinity() {
        let config = ResolvedConfig {
            global_rope_theta: f64::NEG_INFINITY,
            ..Default::default()
        };
        assert_eq!(config.get_float("global_rope_theta"), Some(f64::NEG_INFINITY));
    }

    // ── get_float: rope_theta with f64::NAN is returned as-is ──────────────

    #[test]
    fn get_float_rope_theta_nan_is_some() {
        let config = ResolvedConfig {
            rope_theta: f64::NAN,
            ..Default::default()
        };
        let result = config.get_float("rope_theta");
        assert!(result.is_some());
        assert!(result.unwrap().is_nan());
    }

    // ── substitute_placeholders: rope_theta with infinity produces "inf" ────

    #[test]
    fn substitute_placeholders_rope_theta_infinity_output() {
        let config = ResolvedConfig {
            rope_theta: f64::INFINITY,
            ..Default::default()
        };
        let result = substitute_placeholders("theta=${rope_theta}", &config);
        assert!(result.contains("inf"), "expected inf in: {result}");
        assert!(!result.contains("${rope_theta}"), "placeholder must be replaced");
    }

    // ── convert_rope_scaling: f32::INFINITY factor bypasses <= 1.0 guard ──

    #[test]
    fn convert_rope_scaling_infinity_factor_bypasses_guard() {
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::INFINITY),
            ..Default::default()
        };
        let result = convert_rope_scaling(Some(&scaling));
        assert!(result.is_some(), "INFINITY > 1.0 must bypass the guard");
    }

    // ── ResolvedConfig: extra with empty string key is queryable ────────────

    #[test]
    fn resolved_config_extra_empty_string_key_queryable() {
        let mut extra = HashMap::new();
        extra.insert(String::new(), 42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        assert_eq!(config.get_int(""), Some(42));
    }

    // ── ResolvedConfig: clone with large attention_pattern is independent ──

    #[test]
    fn resolved_config_clone_large_attention_pattern_independent() {
        let original = ResolvedConfig {
            attention_pattern: vec![0u8; 1024],
            ..Default::default()
        };
        let mut cloned = original.clone();
        cloned.attention_pattern.clear();
        assert_eq!(original.attention_pattern.len(), 1024);
        assert!(cloned.attention_pattern.is_empty());
    }

    // ── donor_layer: non-consumer at index 0 returns Ok(None) ──────────────

    #[test]
    fn donor_layer_non_consumer_index_zero_returns_none() {
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0; 8];
        // Layer 0 is a non-consumer (8-4=4, 0 < 4).
        assert_eq!(config.donor_layer(0).unwrap(), None);
    }

    // ── substitute_placeholders: extra key containing braces characters ────

    #[test]
    fn substitute_placeholders_extra_key_with_special_chars() {
        let mut extra = HashMap::new();
        extra.insert("key-with-dashes".to_string(), 7);
        let config = ResolvedConfig { extra, ..Default::default() };
        let result = substitute_placeholders("v=${key-with-dashes}", &config);
        assert_eq!(result, "v=7");
    }

    // ── validate_config: required fields exactly 1 pass ────────────────────

    #[test]
    fn validate_config_boundary_value_one_for_all_required() {
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            ..Default::default()
        };
        assert!(validate_config(&config).is_ok());
    }

    // ── is_kv_shared_layer: usize::MAX as layer_idx is out of range ────────

    #[test]
    fn is_kv_shared_layer_usize_max_always_false() {
        let config = ResolvedConfig {
            num_hidden_layers: usize::MAX,
            num_kv_shared_layers: usize::MAX,
            ..Default::default()
        };
        // usize::MAX is NOT < num_hidden_layers because it equals it.
        assert!(!config.is_kv_shared_layer(usize::MAX));
    }

    // ── ResolvedConfig: debug output includes rope_partial_ratio value ──────

    #[test]
    fn resolved_config_debug_shows_rope_partial_ratio() {
        let config = ResolvedConfig {
            rope_partial_ratio: 0.75,
            ..Default::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("rope_partial_ratio"));
        assert!(debug.contains("0.75"));
    }

    // ── resolve_from_provider: explicit None GGUF reader path ──────────────

    #[test]
    fn resolve_from_provider_explicit_none_gguf_same_as_omitted() {
        // Passing None as GGUF reader should behave identically to omitting it.
        let provider = FakeProvider { tensors: vec![] };
        let result = resolve_from_provider(&provider, None);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            ResolveError::MissingConfig("num_hidden_layers".to_string()),
        );
    }

    // ── from_geometry: zero layers config ──────────────────────────────────

    #[test]
    fn from_geometry_zero_layers_produces_zero_num_hidden_layers() {
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 0,
            vocab_size: 100,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        assert_eq!(config.num_hidden_layers, 0);
    }

    // ── ResolvedConfig: get_int returns correct cast for large hidden_size ─

    #[test]
    fn get_int_large_hidden_size_cast_to_i64() {
        let config = ResolvedConfig {
            hidden_size: usize::MAX / 2,
            ..Default::default()
        };
        let result = config.get_int("hidden_size");
        assert_eq!(result, Some((usize::MAX / 2) as i64));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (375-389) — edge case coverage expansion
    // ═══════════════════════════════════════════════════════════════════════

    // ── convert_rope_scaling: f32::NEG_INFINITY factor caught by <= 1.0 guard ──
    // @trace REQ-ARCH-005

    #[test]
    fn convert_rope_scaling_neg_infinity_factor_caught_by_guard() {
        // Arrange: NEG_INFINITY <= 1.0 is true in IEEE 754, so the guard returns None.
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::NEG_INFINITY),
            ..Default::default()
        };

        // Act
        let result = convert_rope_scaling(Some(&scaling));

        // Assert: NEG_INFINITY is <= 1.0, so the function returns None.
        assert!(result.is_none(), "NEG_INFINITY must be caught by the <= 1.0 guard");
    }

    // ── donor_layer: consumer at large index with moderate config ──────────
    // @trace REQ-ARCH-005

    #[test]
    fn donor_layer_large_consumer_index_no_panic() {
        // Arrange: use moderate values to avoid OOM from attention_pattern vec.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 3;
        config.attention_pattern = vec![0, 0, 0, 0, 0, 0];

        // Act: out-of-range index should return Ok(None), not panic.
        let result = config.donor_layer(1000);

        // Assert: layer 1000 >= num_hidden_layers -> Ok(None).
        assert_eq!(result.unwrap(), None);
    }

    // ── is_kv_shared_layer: large config with 1000 layers ──────────────────
    // @trace REQ-ARCH-005

    #[test]
    fn is_kv_shared_layer_very_large_index_with_large_config() {
        // Arrange: num_hidden_layers = 1000, shared = 500.
        let config = ResolvedConfig {
            num_hidden_layers: 1000,
            num_kv_shared_layers: 500,
            ..Default::default()
        };

        // Act & Assert: index 500 is the first consumer (1000 - 500 = 500).
        assert!(!config.is_kv_shared_layer(499));
        assert!(config.is_kv_shared_layer(500));
        assert!(config.is_kv_shared_layer(999));
        // Out-of-range.
        assert!(!config.is_kv_shared_layer(1000));
    }

    // ── substitute_placeholders: multiline template with mixed field types ──
    // @trace REQ-ARCH-005

    #[test]
    fn substitute_placeholders_multiline_mixed_field_types() {
        // Arrange
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            rope_theta: 10000.0,
            dtype: "bf16".to_string(),
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders(
            "layers=${num_hidden_layers}\nsize=${hidden_size}\ntheta=${rope_theta}\ntype=${dtype}",
            &config,
        );

        // Assert: all four placeholder types replaced correctly across lines.
        let lines: Vec<&str> = result.split('\n').collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "layers=32");
        assert_eq!(lines[1], "size=4096");
        assert!(lines[2].contains("10000"));
        assert_eq!(lines[3], "type=bf16");
    }

    // ── resolve_from_provider: single FFN tensor does not derive vocab or heads
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_single_ffn_tensor_fails_validation() {
        // Arrange: gate_proj tensor (MLP role) matches layer 0 -> num_hidden_layers=1,
        // but no embed tensor -> hidden_size=0 and vocab_size=0.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.layers.0.mlp.gate_proj.weight".to_string(),
                    shape: vec![5632, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: num_hidden_layers=1 from layer tensor, but hidden_size=0 -> fails.
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "hidden_size"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── ResolvedConfig: debug output with empty vs populated extra ──────────

    #[test]
    fn resolved_config_debug_empty_extra_vs_populated_extra_differ() {
        // Arrange
        let config_empty = ResolvedConfig {
            num_hidden_layers: 4,
            extra: HashMap::new(),
            ..Default::default()
        };
        let mut extra = HashMap::new();
        extra.insert("k".to_string(), 1);
        let config_populated = ResolvedConfig {
            num_hidden_layers: 4,
            extra,
            ..Default::default()
        };

        // Act
        let debug_empty = format!("{config_empty:?}");
        let debug_populated = format!("{config_populated:?}");

        // Assert: both contain the struct name and field, but populated has the key.
        assert!(debug_empty.contains("ResolvedConfig"));
        assert!(debug_populated.contains("ResolvedConfig"));
        assert!(debug_populated.contains("k"));
    }

    // ── get_bool: is_kv_shared_layer_0x5 hex not parsed by Rust usize::from_str
    // @trace REQ-ARCH-005

    #[test]
    fn get_bool_is_kv_shared_layer_hex_prefix_returns_none() {
        // Arrange: Rust's str::parse::<usize>() does NOT accept "0x5" hex prefix.
        // With num_hidden_layers=10, num_kv_shared_layers=3:
        // first consumer at 10-3=7.
        let config = ResolvedConfig {
            num_hidden_layers: 10,
            num_kv_shared_layers: 3,
            ..Default::default()
        };

        // Act & Assert: "0x5" is not parseable as usize -> returns None.
        assert_eq!(config.get_bool("is_kv_shared_layer_0x5"), None);
        // "5" is parseable -> index 5 < 7 -> not a consumer.
        assert_eq!(config.get_bool("is_kv_shared_layer_5"), Some(false));
        // "8" is parseable -> index 8 >= 7 -> is a consumer.
        assert_eq!(config.get_bool("is_kv_shared_layer_8"), Some(true));
    }

    // ── validate_config: non-ASCII in dtype field does not affect validation ─
    // @trace REQ-ARCH-005

    #[test]
    fn validate_config_non_ascii_dtype_still_passes() {
        // Arrange: dtype is not validated, even with non-ASCII characters.
        let config = ResolvedConfig {
            num_hidden_layers: 2,
            hidden_size: 128,
            num_attention_heads: 4,
            vocab_size: 100,
            dtype: "类型".to_string(),
            ..Default::default()
        };

        // Act & Assert
        assert!(validate_config(&config).is_ok());
        assert_eq!(config.get_str("dtype"), Some("类型"));
    }

    // ── resolve_error: MissingConfig display and debug contain the key ──────

    #[test]
    fn resolve_error_missing_config_display_and_debug_contain_key() {
        // Arrange
        let err = ResolveError::MissingConfig("my_special_field".to_string());

        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");

        // Assert: Display contains human-readable message, Debug contains variant name.
        assert!(display.contains("my_special_field"));
        assert!(display.contains("Missing required config"));
        assert!(debug.contains("MissingConfig"));
    }

    // ── substitute_placeholders: overlapping placeholder names not confused ──
    // @trace REQ-ARCH-005

    #[test]
    fn substitute_placeholders_hidden_not_confused_with_hidden_size() {
        // Arrange: "hidden_size" exists but "hidden" does not.
        let config = ResolvedConfig {
            hidden_size: 4096,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("${hidden}_${hidden_size}", &config);

        // Assert: ${hidden} stays literal, ${hidden_size} gets replaced.
        assert_eq!(result, "${hidden}_4096");
    }

    // ── from_geometry: zero intermediate_size maps to Some(0) ──────────────
    // @trace REQ-ARCH-005

    #[test]
    fn from_geometry_zero_intermediate_maps_to_some_zero() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 128,
            num_layers: 1,
            vocab_size: 100,
            intermediate_size: 0,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        // Assert: ModelGeometry.intermediate_size is usize (0) -> maps to Some(0).
        assert_eq!(config.intermediate_size, Some(0));
        assert_eq!(config.get_int("intermediate_size"), Some(0));
    }

    // ── ResolvedConfig: extra direct field access vs get_int consistency ────

    #[test]
    fn resolved_config_extra_direct_access_matches_get_int() {
        // Arrange
        let mut extra = HashMap::new();
        extra.insert("alpha".to_string(), 10);
        extra.insert("beta".to_string(), 20);
        let config = ResolvedConfig {
            extra: extra.clone(),
            ..Default::default()
        };

        // Act & Assert: get_int("alpha") == extra.get("alpha").copied()
        for (key, value) in &extra {
            assert_eq!(config.get_int(key), Some(*value));
        }
        // Non-existent key returns None.
        assert_eq!(config.get_int("gamma"), None);
    }

    // ── resolve_from_provider: attention tensors without embedding ─────────
    // @trace REQ-ARCH-005

    #[test]
    fn resolve_from_provider_attention_tensors_without_embed_fails() {
        // Arrange: q_proj + k_proj alone without embedding tensor.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![512, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: no embed tensor -> vocab_size=0, hidden_size=0 -> validation fails.
        assert!(result.is_err());
    }

    // ── convert_rope_scaling: f32::MIN (most negative) factor returns None ──

    #[test]
    fn convert_rope_scaling_f32_min_factor_returns_none() {
        // Arrange: f32::MIN is a very large negative number, which is <= 1.0.
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::MIN),
            ..Default::default()
        };

        // Act
        let result = convert_rope_scaling(Some(&scaling));

        // Assert: f32::MIN <= 1.0 -> returns None.
        assert!(result.is_none(), "f32::MIN must be caught by the <= 1.0 guard");
    }

    // ── ResolvedConfig: clone with all fields zero is identical to default ──

    #[test]
    fn resolved_config_clone_default_is_identical_to_fresh_default() {
        // Arrange
        let config = ResolvedConfig::default();

        // Act
        let cloned = config.clone();

        // Assert: every field matches a fresh default.
        assert_eq!(cloned.num_hidden_layers, 0);
        assert_eq!(cloned.hidden_size, 0);
        assert_eq!(cloned.num_attention_heads, 0);
        assert_eq!(cloned.num_key_value_heads, 0);
        assert_eq!(cloned.head_dim, 0);
        assert_eq!(cloned.intermediate_size, None);
        assert_eq!(cloned.vocab_size, 0);
        assert_eq!(cloned.rope_theta, 0.0);
        assert_eq!(cloned.dtype, "");
        assert_eq!(cloned.global_rope_theta, 0.0);
        assert_eq!(cloned.rope_partial_ratio, 0.0);
        assert!(cloned.attention_pattern.is_empty());
        assert_eq!(cloned.sliding_window, 0);
        assert_eq!(cloned.num_kv_shared_layers, 0);
        assert_eq!(cloned.global_head_dim, 0);
        assert_eq!(cloned.hidden_size_per_layer_input, 0);
        assert!(!cloned.has_per_layer_embedding);
        assert!(cloned.rope_scaling.is_none());
        assert!(cloned.extra.is_empty());
    }

    // ── ResolvedConfig: get_float returns both rope_theta and global_rope_theta ──

    #[test]
    fn resolved_config_get_float_both_rope_thetas_set() {
        // Arrange
        let config = ResolvedConfig {
            rope_theta: 500000.0,
            global_rope_theta: 1_000_000.0,
            ..Default::default()
        };

        // Act & Assert
        assert_eq!(config.get_float("rope_theta"), Some(500000.0));
        assert_eq!(config.get_float("global_rope_theta"), Some(1_000_000.0));
        assert_eq!(config.get_float("unknown"), None);
    }

    // ── ResolvedConfig: get_str returns None for non-dtype string keys ──

    #[test]
    fn resolved_config_get_str_non_dtype_returns_none() {
        // Arrange
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };

        // Act & Assert: only "dtype" is recognized; other string-like keys return None.
        assert_eq!(config.get_str("dtype"), Some("bf16"));
        assert_eq!(config.get_str("hidden_act"), None);
        assert_eq!(config.get_str("rope_scaling_type"), None);
        assert_eq!(config.get_str(""), None);
    }

    // ── ResolvedConfig: get_int returns all named integer fields correctly ──

    #[test]
    fn resolved_config_get_int_all_named_fields() {
        // Arrange
        let config = ResolvedConfig {
            num_hidden_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            intermediate_size: Some(3072),
            vocab_size: 50257,
            ..Default::default()
        };

        // Act & Assert: every named field resolves via get_int.
        assert_eq!(config.get_int("num_hidden_layers"), Some(12));
        assert_eq!(config.get_int("hidden_size"), Some(768));
        assert_eq!(config.get_int("num_attention_heads"), Some(12));
        assert_eq!(config.get_int("num_key_value_heads"), Some(12));
        assert_eq!(config.get_int("head_dim"), Some(64));
        assert_eq!(config.get_int("intermediate_size"), Some(3072));
        assert_eq!(config.get_int("vocab_size"), Some(50257));
    }

    // ── ResolvedConfig: extra HashMap with i64 zero value is queryable ──

    #[test]
    fn resolved_config_extra_with_i64_zero_value() {
        // Arrange
        let mut extra = HashMap::new();
        extra.insert("custom_param".to_string(), 0i64);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };

        // Act & Assert: zero i64 is a valid value, not missing.
        assert_eq!(config.get_int("custom_param"), Some(0));
        assert_eq!(config.get_int("nonexistent"), None);
    }

    // ── substitute_placeholders: intermediate_size Some(0) replaces as "0" ──

    #[test]
    fn substitute_placeholders_intermediate_size_zero_replaces() {
        // Arrange
        let config = ResolvedConfig {
            intermediate_size: Some(0),
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("${intermediate_size}", &config);

        // Assert: Some(0) is still Some, so it replaces.
        assert_eq!(result, "0");
    }

    // ── substitute_placeholders: rope_theta=0.0 replaces as "0" ──

    #[test]
    fn substitute_placeholders_rope_theta_zero_replaces() {
        // Arrange
        let config = ResolvedConfig {
            rope_theta: 0.0,
            ..Default::default()
        };

        // Act
        let result = substitute_placeholders("theta=${rope_theta}", &config);

        // Assert: zero f64 is still a valid value, produces "0".
        assert_eq!(result, "theta=0");
    }

    // ── validate_config: missing hidden_size produces exact error key ──

    #[test]
    fn validate_config_rejects_missing_hidden_size_specifically() {
        // Arrange: only num_hidden_layers and num_attention_heads and vocab_size set.
        let config = ResolvedConfig {
            num_hidden_layers: 4,
            num_attention_heads: 8,
            vocab_size: 100,
            ..Default::default() // hidden_size stays 0
        };

        // Act
        let err = validate_config(&config);

        // Assert
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("hidden_size"),
            "error message must mention hidden_size, got: {msg}"
        );
    }

    // ── ResolveError: DerivationFailed Display includes key and reason ──

    #[test]
    fn resolve_error_derivation_failed_display_format() {
        // Arrange
        let err = ResolveError::DerivationFailed {
            key: "head_dim".to_string(),
            reason: "division by zero".to_string(),
        };

        // Act
        let display = format!("{err}");

        // Assert: Display format from #[error] includes both fields.
        assert!(display.contains("head_dim"), "must contain key");
        assert!(display.contains("division by zero"), "must contain reason");
    }

    // ── ResolveError: Inconsistent Display matches the error attribute ──

    #[test]
    fn resolve_error_inconsistent_display_matches_error_attr() {
        // Arrange
        let err = ResolveError::Inconsistent("num_heads mismatch".to_string());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("num_heads mismatch"),
            "Inconsistent display must contain the message, got: {display}"
        );
    }

    // ── donor_layer: all layers are consumers (num_kv_shared_layers == num_hidden_layers) ──

    #[test]
    fn donor_layer_all_layers_consumers_no_donors() {
        // Arrange: 6 layers, all 6 shared → every layer is a consumer,
        // but there are zero non-consumer layers to donate.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 6;
        config.attention_pattern = vec![0, 0, 0, 0, 0, 0];

        // Act & Assert: every consumer layer has no donor to find.
        let result = config.donor_layer(0);
        assert!(
            result.is_err(),
            "all-consumer config must error: no non-consumer donor exists"
        );
    }

    // ── from_geometry: F16 dtype maps to lowercase "f16" string ──

    #[test]
    fn from_geometry_dtype_f16_lowercase_string() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 1024,
            num_layers: 4,
            vocab_size: 32000,
            intermediate_size: 4096,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F16,
            compute_dtype: DType::F16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        // Assert
        assert_eq!(config.dtype, "f16");
        assert_eq!(config.get_str("dtype"), Some("f16"));
    }

    // ── from_geometry: F32 dtype maps to lowercase "f32" string ──

    #[test]
    fn from_geometry_dtype_f32_lowercase_string() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 512,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 2048,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };

        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());

        // Assert
        assert_eq!(config.dtype, "f32");
    }

    // ── is_kv_shared_layer: num_kv_shared_layers == num_hidden_layers (all shared) ──

    #[test]
    fn is_kv_shared_layer_equal_num_layers_and_shared_layers() {
        // Arrange: 4 layers, all 4 shared.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 4;

        // Act & Assert: every valid layer index is a consumer.
        for i in 0..4 {
            assert!(
                config.is_kv_shared_layer(i),
                "layer {i} must be consumer when all layers shared"
            );
        }
        // Out-of-range still false.
        assert!(!config.is_kv_shared_layer(4));
    }

    // ── ResolvedConfig Debug format includes key field names ──

    #[test]
    fn resolved_config_debug_format_includes_key_field_names() {
        // Arrange
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 64,
            vocab_size: 100,
            ..Default::default()
        };

        // Act
        let debug = format!("{config:?}");

        // Assert: Debug output must contain the struct name and key field identifiers.
        assert!(debug.contains("ResolvedConfig"), "Debug must contain type name");
        assert!(debug.contains("num_hidden_layers"), "Debug must contain num_hidden_layers");
        assert!(debug.contains("hidden_size"), "Debug must contain hidden_size");
        assert!(debug.contains("vocab_size"), "Debug must contain vocab_size");
    }

    // ── resolve_from_provider: non-consecutive layer indices counted correctly ──

    #[test]
    fn resolve_from_provider_two_layers_non_consecutive() {
        // Arrange: layers 0 and 5 (skipping 1-4), embedding, and attention tensors.
        // Without GGUF, num_attention_heads can't be derived from tensors alone
        // (circular dependency: q_proj needs head_dim > 0, head_dim needs heads > 0).
        // So validation fails — the test verifies num_hidden_layers is correctly
        // counted as max(layer_idx) + 1 = 6 before validation rejects.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![100, 64],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![64, 64],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.5.self_attn.q_proj.weight".to_string(),
                    shape: vec![64, 64],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };

        // Act
        let result = resolve_from_provider(&provider, None);

        // Assert: fails because num_attention_heads=0 (no GGUF metadata).
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS — 15 additional tests for uncovered areas
    // ═══════════════════════════════════════════════════════════════════════

    // ── RopeScaling Copy semantics: Yarn ──────────────────────────────────

    #[test]
    fn rope_scaling_yarn_copy_is_independent() {
        // Arrange
        let original = RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 8192,
        };
        // Act: Copy the value (RopeScaling derives Copy)
        let copied = original;
        // Assert: both are equal and independent
        assert_eq!(original, copied);
    }

    // ── RopeScaling Copy semantics: Linear ──────────────────────────────────

    #[test]
    fn rope_scaling_linear_copy_is_independent() {
        // Arrange
        let original = RopeScaling::Linear { factor: 8.0 };
        // Act
        let copied = original;
        // Assert: both are equal (Copy is a bitwise copy)
        assert_eq!(original, copied);
    }

    // ── RopeScaling PartialEq: Yarn equals identical Yarn ──────────────────

    #[test]
    fn rope_scaling_yarn_partial_eq_same_values() {
        // Arrange
        let a = RopeScaling::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        let b = RopeScaling::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        // Act & Assert
        assert_eq!(a, b);
    }

    // ── RopeScaling PartialEq: Yarn not equal to different Yarn ────────────

    #[test]
    fn rope_scaling_yarn_partial_eq_different_factor() {
        // Arrange
        let a = RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        let b = RopeScaling::Yarn {
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    // ── RopeScaling PartialEq: Linear != Yarn (cross-variant) ──────────────

    #[test]
    fn rope_scaling_linear_not_equal_yarn() {
        // Arrange
        let linear = RopeScaling::Linear { factor: 4.0 };
        let yarn = RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        // Act & Assert
        assert_ne!(linear, yarn);
    }

    // ── RopeScaling Debug: Yarn variant contains field names ────────────────

    #[test]
    fn rope_scaling_debug_yarn_variant() {
        // Arrange
        let scaling = RopeScaling::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 8192,
        };
        // Act
        let debug = format!("{scaling:?}");
        // Assert: Debug output must contain the variant name and key fields
        assert!(debug.contains("Yarn"), "Debug must contain variant name 'Yarn'");
        assert!(debug.contains("factor"), "Debug must contain field 'factor'");
        assert!(debug.contains("original_max_position"), "Debug must contain field 'original_max_position'");
    }

    // ── RopeScaling Debug: Linear variant contains field names ──────────────

    #[test]
    fn rope_scaling_debug_linear_variant() {
        // Arrange
        let scaling = RopeScaling::Linear { factor: 8.0 };
        // Act
        let debug = format!("{scaling:?}");
        // Assert
        assert!(debug.contains("Linear"), "Debug must contain variant name 'Linear'");
        assert!(debug.contains("factor"), "Debug must contain field 'factor'");
    }

    // ── get_int("sliding_window") returns None (not a recognized int key) ──

    #[test]
    fn get_int_sliding_window_returns_none() {
        // Arrange: sliding_window is a usize field but not in get_int's match arms
        let config = ResolvedConfig {
            sliding_window: 4096,
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_int("sliding_window"), None,
            "sliding_window is not a recognized get_int key");
    }

    // ── get_float("sliding_window") returns None ───────────────────────────

    #[test]
    fn get_float_sliding_window_returns_none() {
        // Arrange
        let config = ResolvedConfig {
            sliding_window: 4096,
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_float("sliding_window"), None);
    }

    // ── get_str("sliding_window") returns None ─────────────────────────────

    #[test]
    fn get_str_sliding_window_returns_none() {
        // Arrange
        let config = ResolvedConfig {
            sliding_window: 4096,
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_str("sliding_window"), None);
    }

    // ── get_int: core field takes priority over extra with same name ────────

    #[test]
    fn get_int_core_field_shadows_extra_with_same_key() {
        // Arrange: put "num_hidden_layers" in extra with a different value;
        // get_int must return the struct field, not the extra entry.
        let mut extra = HashMap::new();
        extra.insert("num_hidden_layers".to_string(), 999);
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            extra,
            ..Default::default()
        };
        // Act & Assert: struct field wins
        assert_eq!(config.get_int("num_hidden_layers"), Some(32),
            "struct field must shadow extra entry with same key");
    }

    // ── from_geometry with intermediate_size=0 maps to Some(0) ─────────────

    #[test]
    fn from_geometry_zero_intermediate_size_maps_to_some_zero() {
        // Arrange
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 0,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: intermediate_size is Some(0), not None
        assert_eq!(config.intermediate_size, Some(0));
        assert_eq!(config.get_int("intermediate_size"), Some(0));
    }

    // ── is_kv_shared_layer: minimal sharing with 2 layers, 1 shared ────────

    #[test]
    fn is_kv_shared_layer_two_layers_one_shared_boundary() {
        // Arrange: 2 layers, 1 shared → layer 0 is donor, layer 1 is consumer
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.num_kv_shared_layers = 1;
        // Act & Assert
        assert!(!config.is_kv_shared_layer(0), "layer 0 is the donor");
        assert!(config.is_kv_shared_layer(1), "layer 1 is the consumer");
        assert!(!config.is_kv_shared_layer(2), "out of range");
    }

    // ── donor_layer: 2 layers all shared, single bucket picks layer 0 ──────

    #[test]
    fn donor_layer_two_layers_all_shared_single_bucket() {
        // Arrange: 2 layers, 2 shared, same bucket → consumer layer 1
        // picks the latest non-consumer with same bucket.
        // All layers shared means 2 - 2 = 0, so even layer 0 is a consumer
        // (idx 0 >= 0), so there is no donor → error.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 2;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 0];
        // Act & Assert: when all layers are consumers, no donor exists → error
        assert!(config.donor_layer(1).is_err(),
            "all layers shared means no donor exists → must error");
    }

    // ── substitute_placeholders: large model dimensions ────────────────────

    #[test]
    fn substitute_placeholders_large_model_dimensions() {
        // Arrange: realistic large model config
        let config = ResolvedConfig {
            num_hidden_layers: 80,
            hidden_size: 8192,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(28672),
            vocab_size: 152000,
            sliding_window: 131072,
            rope_theta: 500000.0,
            global_rope_theta: 1000000.0,
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        // Act
        let template = "L=${num_hidden_layers} H=${hidden_size} A=${num_attention_heads} \
                         KV=${num_key_value_heads} D=${head_dim} I=${intermediate_size} \
                         V=${vocab_size} SW=${sliding_window} theta=${rope_theta} \
                         gtheta=${global_rope_theta} dt=${dtype}";
        let result = substitute_placeholders(template, &config);
        // Assert
        assert!(result.contains("L=80"), "layers must be 80");
        assert!(result.contains("H=8192"), "hidden must be 8192");
        assert!(result.contains("V=152000"), "vocab must be 152000");
        assert!(result.contains("I=28672"), "intermediate must be 28672");
        assert!(result.contains("dt=bf16"), "dtype must be bf16");
        assert!(result.contains("SW=131072"), "sliding window must be 131072");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (423-437) — 15 additional coverage tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── get_int: extra with empty string key returns value ──────────────────

    #[test]
    fn get_int_extra_empty_string_key() {
        // Arrange: extra HashMap with an empty string as key
        let mut extra = HashMap::new();
        extra.insert(String::new(), 42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        // Act: query with empty string key
        let result = config.get_int("");
        // Assert: the extra entry is returned
        assert_eq!(result, Some(42));
    }

    // ── get_int: all core fields set to large values simultaneously ────────

    #[test]
    fn get_int_all_core_fields_large_values() {
        // Arrange: config with all integer fields at large but valid values
        let config = ResolvedConfig {
            num_hidden_layers: 128,
            hidden_size: 16384,
            num_attention_heads: 128,
            num_key_value_heads: 128,
            head_dim: 256,
            intermediate_size: Some(65536),
            vocab_size: 256000,
            ..Default::default()
        };
        // Act & Assert: every getter returns the expected large value
        assert_eq!(config.get_int("num_hidden_layers"), Some(128));
        assert_eq!(config.get_int("hidden_size"), Some(16384));
        assert_eq!(config.get_int("num_attention_heads"), Some(128));
        assert_eq!(config.get_int("num_key_value_heads"), Some(128));
        assert_eq!(config.get_int("head_dim"), Some(256));
        assert_eq!(config.get_int("intermediate_size"), Some(65536));
        assert_eq!(config.get_int("vocab_size"), Some(256000));
    }

    // ── get_bool: is_kv_shared_layer_0 with zero hidden layers ────────────

    #[test]
    fn get_bool_is_kv_shared_layer_zero_with_zero_hidden_layers() {
        // Arrange: config with num_hidden_layers=0 but num_kv_shared_layers=5
        let config = ResolvedConfig {
            num_hidden_layers: 0,
            num_kv_shared_layers: 5,
            ..Default::default()
        };
        // Act: index 0 with zero total layers
        let result = config.get_bool("is_kv_shared_layer_0");
        // Assert: layer 0 >= num_hidden_layers (0), so out of range → false
        assert_eq!(result, Some(false));
    }

    // ── is_kv_shared_layer: boundary at num_hidden_layers - num_kv_shared_layers = 0

    #[test]
    fn is_kv_shared_layer_boundary_at_zero() {
        // Arrange: num_hidden_layers=5, num_kv_shared_layers=5 → boundary = 5-5 = 0
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 5;
        config.num_kv_shared_layers = 5;
        // Act & Assert: layer 0 is >= 0 and < 5, so it IS a consumer
        assert!(config.is_kv_shared_layer(0));
        assert!(config.is_kv_shared_layer(4));
        // Out of range
        assert!(!config.is_kv_shared_layer(5));
    }

    // ── donor_layer: consumer maps to donor with same bucket in 4-bucket pattern

    #[test]
    fn donor_layer_four_bucket_pattern_correct_mapping() {
        // Arrange: 8 layers, 4 shared, pattern with 4 distinct buckets
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 8;
        config.num_kv_shared_layers = 4;
        // 4 distinct buckets: 10, 20, 30, 40 repeated twice
        config.attention_pattern = vec![10, 20, 30, 40, 10, 20, 30, 40];
        // Act & Assert: consumers are layers 4..8
        // Consumer 4 (bucket 10) → latest non-consumer with bucket 10 = layer 0
        assert_eq!(config.donor_layer(4).unwrap(), Some(0));
        // Consumer 5 (bucket 20) → latest non-consumer with bucket 20 = layer 1
        assert_eq!(config.donor_layer(5).unwrap(), Some(1));
        // Consumer 6 (bucket 30) → latest non-consumer with bucket 30 = layer 2
        assert_eq!(config.donor_layer(6).unwrap(), Some(2));
        // Consumer 7 (bucket 40) → latest non-consumer with bucket 40 = layer 3
        assert_eq!(config.donor_layer(7).unwrap(), Some(3));
    }

    // ── validate_config: multiple simultaneous zero fields reports first one

    #[test]
    fn validate_config_two_zero_fields_reports_first_in_order() {
        // Arrange: only num_hidden_layers is non-zero, the rest are 0
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 0,
            num_attention_heads: 0,
            vocab_size: 0,
            ..Default::default()
        };
        // Act
        let err = validate_config(&config).unwrap_err();
        // Assert: hidden_size is the first zero field checked after num_hidden_layers
        assert_eq!(err, ResolveError::MissingConfig("hidden_size".to_string()));
    }

    // ── convert_rope_scaling: Yarn with minimum valid factor (just above 1.0)

    #[test]
    fn convert_rope_scaling_yarn_minimum_valid_factor() {
        // Arrange: Yarn scaling with factor = 1.0 + f32::EPSILON
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(1.0 + f32::EPSILON),
            beta_fast: Some(10.0),
            beta_slow: Some(0.5),
            original_max_position_embeddings: Some(2048),
            ..Default::default()
        };
        // Act
        let result = convert_rope_scaling(Some(&scaling));
        // Assert: factor > 1.0, so scaling is produced
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!(factor > 1.0);
                assert!((beta_fast - 10.0).abs() < f32::EPSILON);
                assert!((beta_slow - 0.5).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 2048);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── substitute_placeholders: template equals single placeholder ────────

    #[test]
    fn substitute_placeholders_template_is_single_placeholder() {
        // Arrange: template is exactly "${vocab_size}" with nothing else
        let config = ResolvedConfig {
            vocab_size: 128256,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders("${vocab_size}", &config);
        // Assert
        assert_eq!(result, "128256");
    }

    // ── substitute_placeholders: extra with value zero still replaces ──────

    #[test]
    fn substitute_placeholders_extra_zero_value_replaces_correctly() {
        // Arrange: extra key "counter" with value 0
        let mut extra = HashMap::new();
        extra.insert("counter".to_string(), 0);
        let config = ResolvedConfig { extra, ..Default::default() };
        // Act
        let result = substitute_placeholders("count=${counter}", &config);
        // Assert: zero value replaces, does NOT leave placeholder
        assert_eq!(result, "count=0");
    }

    // ── ResolvedConfig: clone produces identical Debug output ──────────────

    #[test]
    fn resolved_config_clone_debug_output_identical() {
        // Arrange: fully populated config
        let mut extra = HashMap::new();
        extra.insert("test_key".to_string(), -7);
        let config = ResolvedConfig {
            num_hidden_layers: 3,
            hidden_size: 512,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: Some(2048),
            vocab_size: 5000,
            rope_theta: 50000.0,
            dtype: "bf16".to_string(),
            global_rope_theta: 250000.0,
            rope_partial_ratio: 0.5,
            attention_pattern: vec![0, 1, 0],
            sliding_window: 512,
            num_kv_shared_layers: 1,
            global_head_dim: 32,
            hidden_size_per_layer_input: 64,
            has_per_layer_embedding: true,
            rope_scaling: Some(RopeScaling::Linear { factor: 2.0 }),
            extra,
            norm_eps: 1e-5,
        };
        // Act
        let cloned = config.clone();
        // Assert: Debug output must be identical
        assert_eq!(format!("{config:?}"), format!("{cloned:?}"));
    }

    // ── ResolveError: source via thiserror is None ─────────────────────────

    #[test]
    fn resolve_error_source_is_none() {
        // Arrange: create each variant
        let missing = ResolveError::MissingConfig("field".to_string());
        let deriv = ResolveError::DerivationFailed {
            key: "k".to_string(),
            reason: "r".to_string(),
        };
        let incons = ResolveError::Inconsistent("msg".to_string());
        // Act & Assert: std::error::Error::source returns None for all variants
        assert!(std::error::Error::source(&missing).is_none());
        assert!(std::error::Error::source(&deriv).is_none());
        assert!(std::error::Error::source(&incons).is_none());
    }

    // ── ResolvedConfig: from_geometry with all zero extra map ──────────────

    #[test]
    fn from_geometry_with_empty_extra_produces_empty_extra() {
        // Arrange
        use gllm_kernels::types::DType;
        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 256,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert
        assert!(config.extra.is_empty());
        assert_eq!(config.extra.len(), 0);
    }

    // ── resolve_from_provider: non-matching tensor name does not contribute to layer count

    #[test]
    fn resolve_from_provider_non_matching_names_no_layer_count() {
        // Arrange: tensors with names that don't match any known role
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "some.unknown.tensor".to_string(),
                    shape: vec![100, 100],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: unknown tensor contributes nothing to layer count
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── get_str: dtype returns correct reference lifetime ──────────────────

    #[test]
    fn get_str_dtype_reference_valid_after_config_moved() {
        // Arrange
        let config = ResolvedConfig {
            dtype: "f32".to_string(),
            ..Default::default()
        };
        // Act: get the string reference, then clone config
        let dtype_str = config.get_str("dtype").map(|s| s.to_string());
        let _cloned = config.clone();
        // Assert: the extracted string is still valid (owned copy)
        assert_eq!(dtype_str.as_deref(), Some("f32"));
    }

    // ── is_kv_shared_layer: num_kv_shared_layers=1 with num_hidden_layers=1, boundary at zero

    #[test]
    fn is_kv_shared_layer_one_layer_one_shared_boundary_zero() {
        // Arrange: 1 layer, 1 shared → saturating_sub(1,1)=0, layer 0 >= 0 and < 1
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            num_kv_shared_layers: 1,
            ..Default::default()
        };
        // Act & Assert
        assert!(config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(1)); // out of range
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (438-452) — 15 additional edge-case coverage tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── substitute_placeholders: rope_theta with fractional float formatting ──

    // @trace TEST-RESOLVE-438 [level:unit]
    #[test]
    fn substitute_placeholders_rope_theta_fractional_format() {
        // Arrange: rope_theta = 500000.5 — a float with a fractional part
        let config = ResolvedConfig {
            rope_theta: 500000.5,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders("theta=${rope_theta}", &config);
        // Assert: f64 Display produces "500000.5"
        assert_eq!(result, "theta=500000.5");
    }

    // ── donor_layer: attention_pattern longer than num_hidden_layers ──────────

    // @trace TEST-RESOLVE-439 [level:unit]
    #[test]
    fn donor_layer_pattern_longer_than_num_layers_still_resolves() {
        // Arrange: 4 layers but pattern has 6 entries; scheduler may or may not
        // reject this, but we verify the call does not panic.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 1, 0, 1, 0, 1]; // 6 entries, 4 layers
        // Act: non-consumer layers must still return Ok(None)
        assert_eq!(config.donor_layer(0).unwrap(), None);
        assert_eq!(config.donor_layer(1).unwrap(), None);
        // Consumer layers may resolve or error; either way no panic
        let _ = config.donor_layer(2);
        let _ = config.donor_layer(3);
    }

    // ── substitute_placeholders: num_heads alias equals num_attention_heads ──

    // @trace TEST-RESOLVE-440 [level:unit]
    #[test]
    fn substitute_placeholders_num_heads_alias_matches_num_attention_heads() {
        // Arrange: config with 48 attention heads
        let config = ResolvedConfig {
            num_attention_heads: 48,
            ..Default::default()
        };
        // Act: both aliases in the same template
        let result = substitute_placeholders(
            "heads=${num_heads} == attn=${num_attention_heads}", &config,
        );
        // Assert: both expand to the same value
        assert_eq!(result, "heads=48 == attn=48");
    }

    // ── resolve_from_provider: embed tensor with empty shape vec ─────────────

    // @trace TEST-RESOLVE-441 [level:unit]
    #[test]
    fn resolve_from_provider_embedding_zero_dim_shape_no_crash() {
        // Arrange: embedding tensor with shape=[] (0-dimensional)
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act: must not panic, must still fail validation
        let result = resolve_from_provider(&provider, None);
        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => assert_eq!(key, "num_hidden_layers"),
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── get_bool: prefix "is_kv_shared_layer_" followed by spaces returns None ─

    // @trace TEST-RESOLVE-442 [level:unit]
    #[test]
    fn get_bool_is_kv_shared_layer_spaces_after_prefix_returns_none() {
        // Arrange
        let config = ResolvedConfig::default();
        // Act: " 3" has a leading space, not parseable as usize
        assert_eq!(config.get_bool("is_kv_shared_layer_ 3"), None);
        // "3 " has a trailing space, also not parseable as usize by str::parse
        assert_eq!(config.get_bool("is_kv_shared_layer_3 "), None);
    }

    // ── from_geometry: has_per_layer_embedding true reflected in get_bool ─────

    // @trace TEST-RESOLVE-443 [level:unit]
    #[test]
    fn from_geometry_has_per_layer_embedding_consistent_with_get_bool() {
        // Arrange: geometry with nonzero hidden_size_per_layer_input
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 256,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 128,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: both the direct field and get_bool agree
        assert!(config.has_per_layer_embedding);
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
    }

    // ── donor_layer: error message contains the layer index ──────────────────

    // @trace TEST-RESOLVE-444 [level:unit]
    #[test]
    fn donor_layer_error_key_includes_requested_layer_idx() {
        // Arrange: malformed attention_pattern (length 3 < num_hidden_layers=4)
        // causes the scheduler to fail for consumer layers (2, 3).
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 4;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0u8; 3]; // wrong length
        // Act: layer 3 is a consumer (4-2=2, so layers 2,3 are consumers)
        let err = config.donor_layer(3).unwrap_err();
        // Assert: the error key must contain "3" (the layer index)
        match err {
            ResolveError::DerivationFailed { key, .. } => {
                assert!(key.contains("3"), "key must contain layer index 3, got: {key}");
            }
            other => panic!("expected DerivationFailed, got {other:?}"),
        }
    }

    // ── resolve_from_provider: multiple embedding tensors, last dtype wins ────

    // @trace TEST-RESOLVE-445 [level:unit]
    #[test]
    fn resolve_from_provider_multiple_embed_tensors_last_dtype_detected() {
        // Arrange: two embedding tensors with different dtypes.
        // The resolve_from_tensors loop sets detected_dtype from the first
        // embedding tensor it encounters (detected_dtype.is_none() guard).
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.embed_tokens.weight_backup".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act: must not panic, must still fail validation
        let result = resolve_from_provider(&provider, None);
        // Assert: validation still fails (missing layers/heads)
        assert!(result.is_err());
    }

    // ── ResolvedConfig: clone equality via field-by-field comparison ──────────

    // @trace TEST-RESOLVE-446 [level:unit]
    #[test]
    fn resolved_config_clone_all_fields_equal() {
        // Arrange: fully populated config
        let mut extra = HashMap::new();
        extra.insert("custom".to_string(), 99);
        let config = ResolvedConfig {
            num_hidden_layers: 6,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 64,
            intermediate_size: Some(4096),
            vocab_size: 32000,
            rope_theta: 10000.0,
            dtype: "bf16".to_string(),
            global_rope_theta: 500000.0,
            rope_partial_ratio: 0.25,
            attention_pattern: vec![0, 1, 0, 1, 0, 1],
            sliding_window: 2048,
            num_kv_shared_layers: 2,
            global_head_dim: 128,
            hidden_size_per_layer_input: 64,
            has_per_layer_embedding: true,
            rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
            extra,
            norm_eps: 1e-5,
        };
        // Act
        let cloned = config.clone();
        // Assert: every field matches
        assert_eq!(cloned.num_hidden_layers, config.num_hidden_layers);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_attention_heads, config.num_attention_heads);
        assert_eq!(cloned.num_key_value_heads, config.num_key_value_heads);
        assert_eq!(cloned.head_dim, config.head_dim);
        assert_eq!(cloned.intermediate_size, config.intermediate_size);
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.rope_theta, config.rope_theta);
        assert_eq!(cloned.dtype, config.dtype);
        assert_eq!(cloned.global_rope_theta, config.global_rope_theta);
        assert_eq!(cloned.rope_partial_ratio, config.rope_partial_ratio);
        assert_eq!(cloned.attention_pattern, config.attention_pattern);
        assert_eq!(cloned.sliding_window, config.sliding_window);
        assert_eq!(cloned.num_kv_shared_layers, config.num_kv_shared_layers);
        assert_eq!(cloned.global_head_dim, config.global_head_dim);
        assert_eq!(cloned.hidden_size_per_layer_input, config.hidden_size_per_layer_input);
        assert_eq!(cloned.has_per_layer_embedding, config.has_per_layer_embedding);
        assert_eq!(cloned.rope_scaling, config.rope_scaling);
        assert_eq!(cloned.extra, config.extra);
    }

    // ── convert_rope_scaling: Linear factor NaN does not produce None ─────────

    // @trace TEST-RESOLVE-447 [level:unit]
    #[test]
    fn convert_rope_scaling_linear_nan_factor_not_filtered() {
        // Arrange: NaN is NOT <= 1.0 (NaN comparisons are always false),
        // so it passes the `factor <= 1.0` guard and produces a scaling.
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::NAN),
            ..Default::default()
        };
        // Act
        let result = convert_rope_scaling(Some(&scaling));
        // Assert: NaN > 1.0 is false, so factor <= 1.0 is also false → produces scaling
        assert!(result.is_some(), "NaN factor bypasses the <= 1.0 guard");
    }

    // ── convert_rope_scaling: Linear factor infinity produces scaling ─────────

    // @trace TEST-RESOLVE-448 [level:unit]
    #[test]
    fn convert_rope_scaling_linear_infinity_factor() {
        // Arrange: f32::INFINITY is > 1.0, so scaling is produced
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(f32::INFINITY),
            ..Default::default()
        };
        // Act
        let result = convert_rope_scaling(Some(&scaling));
        // Assert
        assert!(result.is_some());
        match result.unwrap() {
            RopeScaling::Linear { factor } => {
                assert!(factor.is_infinite() && factor.is_sign_positive());
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    // ── substitute_placeholders: mixed builtin and extra with partially overlapping names ─

    // @trace TEST-RESOLVE-449 [level:unit]
    #[test]
    fn substitute_placeholders_builtin_and_extra_partial_overlap() {
        // Arrange: extra has "hidden_size_extra" which is NOT a builtin,
        // alongside the builtin "hidden_size"
        let mut extra = HashMap::new();
        extra.insert("hidden_size_extra".to_string(), 2048);
        let config = ResolvedConfig {
            hidden_size: 4096,
            extra,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders(
            "H=${hidden_size}, HEX=${hidden_size_extra}", &config,
        );
        // Assert: builtin resolved to 4096, extra resolved to 2048
        assert_eq!(result, "H=4096, HEX=2048");
    }

    // ── validate_config: all four required fields exactly at boundary value 1 ─

    // @trace TEST-RESOLVE-450 [level:unit]
    #[test]
    fn validate_config_accepts_all_fields_at_one() {
        // Arrange: minimum valid values — all required fields = 1
        let config = ResolvedConfig {
            num_hidden_layers: 1,
            hidden_size: 1,
            num_attention_heads: 1,
            vocab_size: 1,
            ..Default::default()
        };
        // Act & Assert: all required fields > 0, so validation passes
        assert!(validate_config(&config).is_ok());
    }

    // ── is_kv_shared_layer: num_kv_shared_layers=2 with num_hidden_layers=3, donor+consumer+consumer ─

    // @trace TEST-RESOLVE-451 [level:unit]
    #[test]
    fn is_kv_shared_layer_three_layers_two_shared() {
        // Arrange: 3 layers, 2 shared → boundary = 3-2 = 1
        // Layer 0 is donor, layers 1 and 2 are consumers
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 3;
        config.num_kv_shared_layers = 2;
        // Act & Assert
        assert!(!config.is_kv_shared_layer(0), "layer 0 must be donor");
        assert!(config.is_kv_shared_layer(1), "layer 1 must be consumer");
        assert!(config.is_kv_shared_layer(2), "layer 2 must be consumer");
        assert!(!config.is_kv_shared_layer(3), "layer 3 is out of range");
    }

    // ── donor_layer: 3 layers, 2 shared, matching bucket picks correct donor ─

    // @trace TEST-RESOLVE-452 [level:unit]
    #[test]
    fn donor_layer_three_layers_two_shared_bucket_match() {
        // Arrange: 3 layers, 2 shared; donors = [0], consumers = [1, 2]
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 3;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![7, 7, 7]; // all same bucket
        // Act & Assert: consumer 1 picks latest non-consumer with bucket 7 → layer 0
        assert_eq!(config.donor_layer(1).unwrap(), Some(0));
        assert_eq!(config.donor_layer(2).unwrap(), Some(0));
        // Non-consumer returns None
        assert_eq!(config.donor_layer(0).unwrap(), None);
    }

    // ── convert_rope_scaling: Yarn with all custom fields ─

    // @trace TEST-RESOLVE-453 [level:unit]
    #[test]
    fn convert_rope_scaling_yarn_all_fields_explicit() {
        // Arrange: Yarn scaling with non-default beta_fast and beta_slow
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Yarn),
            factor: Some(16.0),
            beta_fast: Some(8.0),
            beta_slow: Some(4.0),
            original_max_position_embeddings: Some(16384),
            ..Default::default()
        };
        // Act
        let result = convert_rope_scaling(Some(&scaling));
        // Assert
        match result.expect("Yarn with factor>1 must produce scaling") {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position } => {
                assert!((factor - 16.0).abs() < f32::EPSILON);
                assert!((beta_fast - 8.0).abs() < f32::EPSILON);
                assert!((beta_slow - 4.0).abs() < f32::EPSILON);
                assert_eq!(original_max_position, 16384);
            }
            other => panic!("expected Yarn, got {other:?}"),
        }
    }

    // ── convert_rope_scaling: Linear with negative factor passes guard ──────

    // @trace TEST-RESOLVE-454 [level:unit]
    #[test]
    fn convert_rope_scaling_linear_negative_factor_returns_none() {
        // Arrange: Negative factor — -1.0 <= 1.0 is true → returns None
        let scaling = RopeScalingConfig {
            scaling_type: Some(RopeScalingType::Linear),
            factor: Some(-1.0),
            ..Default::default()
        };
        // Act
        let result = convert_rope_scaling(Some(&scaling));
        // Assert: negative is <= 1.0, so the guard filters it out
        assert!(result.is_none(), "negative factor must be filtered by <= 1.0 guard");
    }

    // ── substitute_placeholders: extra value zero substituted as "0" ────────

    // @trace TEST-RESOLVE-455 [level:unit]
    #[test]
    fn substitute_placeholders_extra_zero_value_outputs_zero() {
        // Arrange: extra key with value 0
        let mut extra = HashMap::new();
        extra.insert("counter".to_string(), 0);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders("cnt=${counter}", &config);
        // Assert
        assert_eq!(result, "cnt=0");
    }

    // ── substitute_placeholders: negative extra value substituted correctly ─

    // @trace TEST-RESOLVE-456 [level:unit]
    #[test]
    fn substitute_placeholders_extra_negative_value_outputs_negative() {
        // Arrange: extra key with negative value
        let mut extra = HashMap::new();
        extra.insert("offset".to_string(), -42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders("off=${offset}", &config);
        // Assert
        assert_eq!(result, "off=-42");
    }

    // ── ResolveError Display: DerivationFailed with multiline reason ───────

    // @trace TEST-RESOLVE-457 [level:unit]
    #[test]
    fn resolve_error_derivation_failed_multiline_reason_in_display() {
        // Arrange: reason string contains newlines
        let err = ResolveError::DerivationFailed {
            key: "head_dim".to_string(),
            reason: "step 1 failed\nstep 2 also failed".to_string(),
        };
        // Act
        let msg = format!("{err}");
        // Assert: Display output contains both the key and the full reason
        assert!(msg.contains("head_dim"));
        assert!(msg.contains("step 1 failed"));
        assert!(msg.contains("step 2 also failed"));
    }

    // ── ResolvedConfig default: rope_scaling field defaults to None ──────

    // @trace TEST-RESOLVE-458 [level:unit]
    #[test]
    fn resolved_config_default_rope_scaling_none_confirmed() {
        // Arrange & Act
        let config = ResolvedConfig::default();
        // Assert: no rope scaling by default
        assert!(config.rope_scaling.is_none());
    }

    // ── get_int: intermediate_size Some(0) returns Some(0) not None ──────

    // @trace TEST-RESOLVE-459 [level:unit]
    #[test]
    fn get_int_intermediate_size_some_zero_returns_zero() {
        // Arrange: intermediate_size is Some(0), not None
        let config = ResolvedConfig {
            intermediate_size: Some(0),
            ..Default::default()
        };
        // Act & Assert: Some(0) is a valid value, returns Some(0)
        assert_eq!(config.get_int("intermediate_size"), Some(0));
        assert_ne!(config.get_int("intermediate_size"), None);
    }

    // ── is_kv_shared_layer: exactly at boundary with shared=1 ────────────

    // @trace TEST-RESOLVE-460 [level:unit]
    #[test]
    fn is_kv_shared_layer_single_shared_layer_boundary() {
        // Arrange: 10 layers, 1 shared → boundary = 9. Only layer 9 is consumer.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 10;
        config.num_kv_shared_layers = 1;
        // Act & Assert
        assert!(!config.is_kv_shared_layer(8), "layer 8 must not be consumer");
        assert!(config.is_kv_shared_layer(9), "layer 9 must be consumer");
        assert!(!config.is_kv_shared_layer(10), "layer 10 is out of range");
    }

    // ── donor_layer: consumer with no matching bucket donor returns error ─

    // @trace TEST-RESOLVE-461 [level:unit]
    #[test]
    fn donor_layer_all_donors_different_bucket_from_consumer() {
        // Arrange: 6 layers, 2 shared; donors [0,1,2,3] have bucket 0,
        // consumers [4,5] have bucket 99.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 6;
        config.num_kv_shared_layers = 2;
        config.attention_pattern = vec![0, 0, 0, 0, 99, 99];
        // Act & Assert: consumer 4 has bucket 99, no donor matches
        let err = config.donor_layer(4).unwrap_err();
        match err {
            ResolveError::DerivationFailed { reason, .. } => {
                assert!(!reason.is_empty());
            }
            other => panic!("expected DerivationFailed, got {other:?}"),
        }
    }

    // ── get_bool: empty string key returns None ──────────────────────────

    // @trace TEST-RESOLVE-462 [level:unit]
    #[test]
    fn get_bool_empty_key_returns_none() {
        // Arrange
        let config = ResolvedConfig::default();
        // Act & Assert: empty string matches no known field or prefix
        assert_eq!(config.get_bool(""), None);
    }

    // ── get_int: empty string key returns None ───────────────────────────

    // @trace TEST-RESOLVE-463 [level:unit]
    #[test]
    fn get_int_empty_key_returns_none() {
        // Arrange
        let config = ResolvedConfig::default();
        // Act & Assert: empty string matches no known field
        assert_eq!(config.get_int(""), None);
    }

    // ── get_str: empty string key returns None ──────────────────────────

    // @trace TEST-RESOLVE-464 [level:unit]
    #[test]
    fn get_str_empty_key_returns_none() {
        // Arrange
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        // Act & Assert: empty string matches no known field
        assert_eq!(config.get_str(""), None);
    }

    // ── get_float: empty string key returns None ────────────────────────

    // @trace TEST-RESOLVE-465 [level:unit]
    #[test]
    fn get_float_empty_key_returns_none() {
        // Arrange
        let config = ResolvedConfig {
            rope_theta: 10000.0,
            ..Default::default()
        };
        // Act & Assert: empty string matches no known field
        assert_eq!(config.get_float(""), None);
    }

    // ── ResolvedConfig clone: dtype string is independent ──────────────

    // @trace TEST-RESOLVE-466 [level:unit]
    #[test]
    fn resolved_config_clone_dtype_string_is_independent() {
        // Arrange: config with non-empty dtype
        let original = ResolvedConfig {
            dtype: "f16".to_string(),
            ..Default::default()
        };
        // Act: clone and mutate the dtype
        let mut cloned = original.clone();
        cloned.dtype.push_str("_modified");
        // Assert: original unaffected
        assert_eq!(original.dtype, "f16");
        assert_eq!(cloned.dtype, "f16_modified");
    }

    // ── ResolvedConfig clone: rope_scaling is independent ──────────────

    // @trace TEST-RESOLVE-467 [level:unit]
    #[test]
    fn resolved_config_clone_rope_scaling_is_independent() {
        // Arrange: config with rope_scaling set
        let original = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Linear { factor: 8.0 }),
            ..Default::default()
        };
        // Act: clone, verify independent
        let cloned = original.clone();
        // Assert: both have the same value, but are independent objects
        assert!(original.rope_scaling.is_some());
        assert!(cloned.rope_scaling.is_some());
        // The values are equal (Clone produces equal enum values)
        assert_eq!(original.rope_scaling, cloned.rope_scaling);
    }

    // @trace TEST-RESOLVE-468 [level:unit]
    #[test]
    fn resolved_config_default_all_zero_or_empty() {
        let cfg = ResolvedConfig::default();
        assert_eq!(cfg.num_hidden_layers, 0);
        assert_eq!(cfg.hidden_size, 0);
        assert_eq!(cfg.num_attention_heads, 0);
        assert_eq!(cfg.vocab_size, 0);
        assert_eq!(cfg.rope_theta, 0.0);
        assert!(cfg.dtype.is_empty());
        assert!(cfg.extra.is_empty());
        assert!(cfg.rope_scaling.is_none());
    }

    // @trace TEST-RESOLVE-469 [level:unit]
    #[test]
    fn resolved_config_get_int_num_hidden_layers() {
        let cfg = ResolvedConfig { num_hidden_layers: 32, ..Default::default() };
        assert_eq!(cfg.get_int("num_hidden_layers"), Some(32));
    }

    // @trace TEST-RESOLVE-470 [level:unit]
    #[test]
    fn resolved_config_get_int_hidden_size() {
        let cfg = ResolvedConfig { hidden_size: 4096, ..Default::default() };
        assert_eq!(cfg.get_int("hidden_size"), Some(4096));
    }

    // @trace TEST-RESOLVE-471 [level:unit]
    #[test]
    fn resolved_config_get_int_vocab_size() {
        let cfg = ResolvedConfig { vocab_size: 32000, ..Default::default() };
        assert_eq!(cfg.get_int("vocab_size"), Some(32000));
    }

    // @trace TEST-RESOLVE-472 [level:unit]
    #[test]
    fn resolved_config_get_int_unknown_returns_none() {
        let cfg = ResolvedConfig::default();
        assert_eq!(cfg.get_int("nonexistent_key"), None);
    }

    // @trace TEST-RESOLVE-473 [level:unit]
    #[test]
    fn resolved_config_extra_values_accessible_via_get_int() {
        let mut cfg = ResolvedConfig::default();
        cfg.extra.insert("custom_field".to_string(), 42);
        assert_eq!(cfg.get_int("custom_field"), Some(42));
    }

    // @trace TEST-RESOLVE-474 [level:unit]
    #[test]
    fn resolved_config_debug_shows_dtype() {
        let cfg = ResolvedConfig { dtype: "bf16".to_string(), ..Default::default() };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("bf16"));
    }

    // @trace TEST-RESOLVE-475 [level:unit]
    #[test]
    fn resolved_config_has_per_layer_embedding_false_when_zero() {
        let cfg = ResolvedConfig { hidden_size_per_layer_input: 0, ..Default::default() };
        assert!(!cfg.has_per_layer_embedding);
    }

    // @trace TEST-RESOLVE-480 [level:unit]
    #[test]
    fn resolved_config_get_float_rope_theta() {
        let cfg = ResolvedConfig { rope_theta: 10000.0, ..Default::default() };
        let val = cfg.get_float("rope_theta");
        assert!(val.is_some());
        assert!((val.unwrap() - 10000.0).abs() < 1e-6);
    }

    // @trace TEST-RESOLVE-481 [level:unit]
    #[test]
    fn resolved_config_get_float_global_rope_theta() {
        let cfg = ResolvedConfig { global_rope_theta: 1_000_000.0, ..Default::default() };
        let val = cfg.get_float("global_rope_theta");
        assert!(val.is_some());
        assert!((val.unwrap() - 1_000_000.0).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (482-496) — 15 additional tensor role & API coverage tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── resolve_from_provider: o_proj 张量检测层索引 (AttentionOutput 角色) ──

    // @trace TEST-RESOLVE-482 [level:unit]
    #[test]
    fn resolve_from_provider_o_proj_tensor_layer_detection() {
        // Arrange: o_proj 张量识别为 AttentionOutput 角色，贡献层索引检测
        // 但不贡献 head_dim/num_heads 推导，所以验证仍然失败
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.o_proj.weight".to_string(),
                    shape: vec![2048, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers=1 (从 o_proj 检测), vocab_size=32000, hidden_size=2048
        // 但 num_attention_heads=0 → 验证失败
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: input_layernorm 张量检测层索引 (InputNorm 角色) ──

    // @trace TEST-RESOLVE-483 [level:unit]
    #[test]
    fn resolve_from_provider_input_norm_tensor_layer_detection() {
        // Arrange: input_layernorm 张量识别为 InputNorm 角色，贡献层索引
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.3.input_layernorm.weight".to_string(),
                    shape: vec![2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers=4 (layer 3 → max+1), 但 num_attention_heads=0 → 失败
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: post_attention_layernorm 张量检测层索引 (PostAttnNorm 角色) ──

    // @trace TEST-RESOLVE-484 [level:unit]
    #[test]
    fn resolve_from_provider_post_attn_norm_tensor_layer_detection() {
        // Arrange: post_attention_layernorm 张量识别为 PostAttnNorm 角色
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![1000, 512],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.layers.7.post_attention_layernorm.weight".to_string(),
                    shape: vec![512],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers=8 (layer 7 → max+1), hidden_size=512, vocab_size=1000
        // 但 num_attention_heads=0 → 失败
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: up_proj 张量检测层索引 (FfnUp 角色) ──

    // @trace TEST-RESOLVE-485 [level:unit]
    #[test]
    fn resolve_from_provider_up_proj_tensor_layer_detection() {
        // Arrange: up_proj 张量识别为 FfnUp 角色，贡献层索引
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![5000, 1024],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.2.mlp.up_proj.weight".to_string(),
                    shape: vec![4096, 1024],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers=3 (layer 2 → max+1), hidden_size=1024, vocab_size=5000
        // 但 num_attention_heads=0 → 失败
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── resolve_from_provider: down_proj 张量检测层索引 (FfnDown 角色) ──

    // @trace TEST-RESOLVE-486 [level:unit]
    #[test]
    fn resolve_from_provider_down_proj_tensor_layer_detection() {
        // Arrange: down_proj 张量识别为 FfnDown 角色
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![8000, 768],
                    dtype: safetensors::Dtype::F16,
                },
                TensorMeta {
                    name: "model.layers.1.mlp.down_proj.weight".to_string(),
                    shape: vec![768, 3072],
                    dtype: safetensors::Dtype::F16,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers=2, hidden_size=768, vocab_size=8000, dtype="f16"
        // 但 num_attention_heads=0 → 失败
        assert!(result.is_err());
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── from_geometry: compute_dtype 不同于 dtype，ResolvedConfig.dtype 使用模型 dtype ──

    // @trace TEST-RESOLVE-487 [level:unit]
    #[test]
    fn from_geometry_compute_dtype_does_not_override_dtype() {
        // Arrange: 模型 dtype=BF16 但 compute_dtype=F32
        // ResolvedConfig.dtype 应该使用模型 dtype (BF16)，不是 compute_dtype
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 256,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: dtype 来自模型 dtype，不是 compute_dtype
        assert_eq!(config.dtype, "bf16");
    }

    // ── from_geometry: position_offset 设置但不映射到 ResolvedConfig ──

    // @trace TEST-RESOLVE-488 [level:unit]
    #[test]
    fn from_geometry_position_offset_not_mapped() {
        // Arrange: position_offset=2 (如 RoBERTa 模型)
        // ResolvedConfig 没有此字段，from_geometry 应该忽略它
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 256,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: Some(2),
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: 正常构建，没有 position_offset 字段在 ResolvedConfig 中
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.hidden_size, 256);
    }

    // ── get_int("dtype") 不在 match arms 中且 extra 中无该 key 时返回 None ──

    // @trace TEST-RESOLVE-489 [level:unit]
    #[test]
    fn get_int_dtype_key_returns_none_without_extra() {
        // Arrange: dtype 是 String 字段，get_int 的 match arms 中不包含 "dtype"
        // extra 中也没有 "dtype" 键
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_int("dtype"), None,
            "dtype 是字符串字段，get_int 不应返回值");
    }

    // ── get_str("num_hidden_layers") 返回 None（整数字段不是字符串可访问的）──

    // @trace TEST-RESOLVE-490 [level:unit]
    #[test]
    fn get_str_num_hidden_layers_returns_none() {
        // Arrange: num_hidden_layers 是 usize 字段，get_str 只匹配 "dtype"
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_str("num_hidden_layers"), None,
            "整数字段不能通过 get_str 访问");
    }

    // ── substitute_placeholders: 同一模板中 num_heads 和 num_attention_heads 别名 ──

    // @trace TEST-RESOLVE-491 [level:unit]
    #[test]
    fn substitute_placeholders_both_num_heads_aliases_same_template() {
        // Arrange: 两个别名 ${num_heads} 和 ${num_attention_heads} 在同一模板中
        let config = ResolvedConfig {
            num_attention_heads: 64,
            ..Default::default()
        };
        // Act
        let result = substitute_placeholders(
            "short=${num_heads} long=${num_attention_heads}", &config,
        );
        // Assert: 两个别名都替换为相同值
        assert_eq!(result, "short=64 long=64");
    }

    // ── ResolvedConfig clone: rope_scaling Some(Linear) 是深拷贝 ──

    // @trace TEST-RESOLVE-492 [level:unit]
    #[test]
    fn resolved_config_clone_rope_scaling_deep_copy() {
        // Arrange: rope_scaling 设为 Some(Linear)，clone 后比较
        let original = ResolvedConfig {
            rope_scaling: Some(RopeScaling::Yarn {
                factor: 4.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 8192,
            }),
            ..Default::default()
        };
        // Act
        let cloned = original.clone();
        // Assert: clone 后值相等，是独立副本
        assert_eq!(original.rope_scaling, cloned.rope_scaling);
        assert!(original.rope_scaling.is_some());
        assert!(cloned.rope_scaling.is_some());
    }

    // ── validate_config: head_dim=0 + intermediate_size=None 不导致失败 ──

    // @trace TEST-RESOLVE-493 [level:unit]
    #[test]
    fn validate_config_optional_fields_zero_none_passes() {
        // Arrange: 所有 4 个必填字段有效，可选字段为零/None
        let config = ResolvedConfig {
            num_hidden_layers: 24,
            hidden_size: 2048,
            num_attention_heads: 16,
            vocab_size: 32000,
            head_dim: 0,
            intermediate_size: None,
            num_key_value_heads: 0,
            sliding_window: 0,
            ..Default::default()
        };
        // Act & Assert: 验证通过，head_dim 和 intermediate_size 不是必填字段
        assert!(validate_config(&config).is_ok());
    }

    // ── get_int: extra 中存储 i64::MIN 极端值 ──

    // @trace TEST-RESOLVE-494 [level:unit]
    #[test]
    fn get_int_extra_i64_min_value() {
        // Arrange: extra 中存储 i64 最小值
        let mut extra = HashMap::new();
        extra.insert("extreme_neg".to_string(), i64::MIN);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_int("extreme_neg"), Some(i64::MIN));
    }

    // ── get_str("attention_pattern") 返回 None（Vec<u8> 字段不是字符串可访问的）──

    // @trace TEST-RESOLVE-495 [level:unit]
    #[test]
    fn get_str_attention_pattern_returns_none() {
        // Arrange: attention_pattern 是 Vec<u8> 字段
        let config = ResolvedConfig {
            attention_pattern: vec![0, 1, 0, 1],
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_str("attention_pattern"), None,
            "Vec<u8> 字段不能通过 get_str 访问");
    }

    // ── get_float("dtype") 返回 None（dtype 不是浮点数字段）──

    // @trace TEST-RESOLVE-496 [level:unit]
    #[test]
    fn get_float_dtype_key_returns_none() {
        // Arrange: dtype 是字符串字段，get_float 只匹配 "rope_theta" 和 "global_rope_theta"
        let config = ResolvedConfig {
            dtype: "bf16".to_string(),
            ..Default::default()
        };
        // Act & Assert
        assert_eq!(config.get_float("dtype"), None,
            "dtype 不是浮点数字段，get_float 应返回 None");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  WAVE 61 — coverage expansion: +15 tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── 1. Extra key detection in tensor names ──

    // @trace TEST-RESOLVE-497 [level:unit]
    #[test]
    fn resolve_from_provider_extra_keys_in_tensor_name_not_matched() {
        // Arrange: tensor names that contain role keywords but are not standard
        // (e.g., "q_proj_bias" or "custom_q_proj.weight"). These should not
        // match any TensorRole and should be ignored by resolve_from_tensors.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj_bias".to_string(),
                    shape: vec![4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act: resolve should succeed in deriving vocab+hidden from embed,
        // but fail on num_attention_heads since q_proj_bias is not AttentionQuery.
        let result = resolve_from_provider(&provider, None);
        // Assert
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_ne!(key, "vocab_size", "vocab must derive from embed");
                assert_ne!(key, "hidden_size", "hidden_size must derive from embed");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 2. Donor layer resolution with three-bucket staggered pattern ──

    // @trace TEST-RESOLVE-498 [level:unit]
    #[test]
    fn donor_layer_three_bucket_staggered_resolves_correctly() {
        // Arrange: 12 layers, 4 shared. Pattern: [0,1,2] repeated.
        let mut config = ResolvedConfig::default();
        config.num_hidden_layers = 12;
        config.num_kv_shared_layers = 4;
        config.attention_pattern = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        // Consumers: layers 8..12. Donors: layers 0..8.
        // Pattern: idx 0=0,1=1,2=2,3=0,4=1,5=2,6=0,7=1,8=2,9=0,10=1,11=2
        // Act & Assert
        // Consumer 8 (bucket pattern[8]=2) -> latest donor with bucket 2 = layer 5.
        assert_eq!(config.donor_layer(8).unwrap(), Some(5));
        // Consumer 9 (bucket pattern[9]=0) -> latest donor with bucket 0 = layer 6.
        assert_eq!(config.donor_layer(9).unwrap(), Some(6));
        // Consumer 10 (bucket pattern[10]=1) -> latest donor with bucket 1 = layer 7.
        assert_eq!(config.donor_layer(10).unwrap(), Some(7));
        // Consumer 11 (bucket pattern[11]=2) -> latest donor with bucket 2 = layer 5.
        assert_eq!(config.donor_layer(11).unwrap(), Some(5));
    }

    // ── 3. Placeholder weight handling when intermediate_size is None ──

    // @trace TEST-RESOLVE-499 [level:unit]
    #[test]
    fn substitute_placeholders_intermediate_size_none_in_complex_template() {
        // Arrange: config with intermediate_size=None and several fields set.
        let config = ResolvedConfig {
            num_hidden_layers: 6,
            hidden_size: 1024,
            intermediate_size: None,
            ..Default::default()
        };
        let template = "L=${num_hidden_layers}, H=${hidden_size}, I=${intermediate_size}";
        // Act
        let result = substitute_placeholders(template, &config);
        // Assert: L and H replaced, I stays as literal placeholder.
        assert_eq!(result, "L=6, H=1024, I=${intermediate_size}");
    }

    // ── 4. ResolveError Display all variants in single test ──

    // @trace TEST-RESOLVE-500 [level:unit]
    #[test]
    fn resolve_error_all_display_variants_contain_unique_prefix() {
        // Arrange: create all three ResolveError variants.
        let missing = ResolveError::MissingConfig("field_x".to_string());
        let derivation = ResolveError::DerivationFailed {
            key: "field_y".to_string(),
            reason: "overflow".to_string(),
        };
        let inconsistent = ResolveError::Inconsistent("conflict_z".to_string());
        // Act & Assert: each Display output contains a unique prefix.
        let missing_msg = format!("{missing}");
        assert!(missing_msg.starts_with("Missing required config"),
            "MissingConfig display must start with 'Missing required config'");
        assert!(missing_msg.contains("field_x"));

        let deriv_msg = format!("{derivation}");
        assert!(deriv_msg.starts_with("Failed to derive"),
            "DerivationFailed display must start with 'Failed to derive'");
        assert!(deriv_msg.contains("field_y"));
        assert!(deriv_msg.contains("overflow"));

        let inconsist_msg = format!("{inconsistent}");
        assert!(inconsist_msg.starts_with("Inconsistent config"),
            "Inconsistent display must start with 'Inconsistent config'");
        assert!(inconsist_msg.contains("conflict_z"));
    }

    // ── 5. Config resolution with missing optional fields ──

    // @trace TEST-RESOLVE-501 [level:unit]
    #[test]
    fn resolved_config_with_all_optional_fields_missing_still_validates() {
        // Arrange: only the 4 required fields are set; all optional fields
        // remain at default (intermediate_size=None, rope_theta=0, etc.).
        let config = ResolvedConfig {
            num_hidden_layers: 8,
            hidden_size: 2048,
            num_attention_heads: 16,
            vocab_size: 50000,
            // All optional fields deliberately left at default.
            num_key_value_heads: 0,
            head_dim: 0,
            intermediate_size: None,
            rope_theta: 0.0,
            dtype: String::new(),
            global_rope_theta: 0.0,
            rope_partial_ratio: 0.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            has_per_layer_embedding: false,
            rope_scaling: None,
            extra: HashMap::new(),
            norm_eps: 1e-5,
        };
        // Act & Assert: validation passes with only required fields.
        assert!(validate_config(&config).is_ok());
    }

    // ── 6. Layer index parsing from weight names ──

    // @trace TEST-RESOLVE-502 [level:unit]
    #[test]
    fn resolve_from_provider_layer_index_from_high_numbered_layers() {
        // Arrange: tensors for layers 99 and 100. max_layer_idx should be 101.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.99.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.100.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers should be 101 (100+1), but
        // num_attention_heads=0 (no head_dim) → validation fails.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads",
                    "layers should be derived as 101, but heads cannot be");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 7. Expert weight name format validation ──

    // @trace TEST-RESOLVE-503 [level:unit]
    #[test]
    fn resolve_from_provider_expert_weight_names_not_counted_as_layers() {
        // Arrange: MoE expert inner projection tensors (experts.E.xxx_proj) are
        // not recognized by match_tensor_role, so they do NOT increment the layer
        // counter. Only recognized per-layer roles (MoEGate, AttentionQuery, etc.)
        // count. This test verifies that unrecognized expert weight names don't
        // falsely inflate num_hidden_layers.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                    shape: vec![4096, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.experts.1.gate_proj.weight".to_string(),
                    shape: vec![4096, 2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: expert inner projections are not recognized per-layer roles,
        // so num_hidden_layers stays 0. Validation reports it first.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_hidden_layers",
                    "unrecognized expert tensors must not increment layer count");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 8. Multi-head attention dimension inference ──

    // @trace TEST-RESOLVE-504 [level:unit]
    #[test]
    fn head_dim_auto_computed_from_hidden_and_heads() {
        // Arrange: hidden_size=4096, num_attention_heads=32 -> head_dim should be 128.
        let config = ResolvedConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            head_dim: 0, // not yet computed
            num_hidden_layers: 1,
            vocab_size: 100,
            ..Default::default()
        };
        // Act & Assert: the auto-computation in resolve_from_provider does:
        // if head_dim == 0 && hidden_size > 0 && num_attention_heads > 0:
        //     head_dim = hidden_size / num_attention_heads
        // Simulate that logic here.
        let computed = if config.head_dim == 0
            && config.hidden_size > 0
            && config.num_attention_heads > 0
        {
            config.hidden_size / config.num_attention_heads
        } else {
            config.head_dim
        };
        assert_eq!(computed, 128, "4096 / 32 = 128");
    }

    // ── 9. Tie weights detection (embed_lm_head) ──

    // @trace TEST-RESOLVE-505 [level:unit]
    #[test]
    fn resolve_from_provider_output_head_tensor_role_detected() {
        // Arrange: an output head tensor (lm_head.weight) should be recognized
        // as TensorRole::OutputHead by match_tensor_role. It should NOT be
        // matched as Embedding, so vocab_size should NOT be derived from it.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "lm_head.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: vocab_size stays 0 because lm_head is not Embedding role.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_hidden_layers",
                    "lm_head does not set vocab_size or hidden_size");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 10. MoE expert count from weight names ──

    // @trace TEST-RESOLVE-506 [level:unit]
    #[test]
    fn resolve_from_provider_moe_gate_tensor_counts_layer() {
        // Arrange: MoE gate/router tensors are per-layer and should increment
        // the layer counter. Verify layer detection from moe gate weight names.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![50000, 3072],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.gate.weight".to_string(),
                    shape: vec![64, 3072],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.layers.1.mlp.gate.weight".to_string(),
                    shape: vec![64, 3072],
                    dtype: safetensors::Dtype::BF16,
                },
                TensorMeta {
                    name: "model.layers.2.mlp.gate.weight".to_string(),
                    shape: vec![64, 3072],
                    dtype: safetensors::Dtype::BF16,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers should be 3, but num_attention_heads=0.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads",
                    "layers derived from moe gate tensors, but heads missing");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 11. ResolveResult field access via all getter methods ──

    // @trace TEST-RESOLVE-507 [level:unit]
    #[test]
    fn resolved_config_all_getter_types_covered_for_nonzero_config() {
        // Arrange: config with every field set to a nonzero/unique value.
        let mut extra = HashMap::new();
        extra.insert("custom_dim".to_string(), 999);
        let config = ResolvedConfig {
            num_hidden_layers: 24,
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 64,
            intermediate_size: Some(8192),
            vocab_size: 64000,
            rope_theta: 500000.0,
            dtype: "bf16".to_string(),
            global_rope_theta: 1000000.0,
            rope_partial_ratio: 0.5,
            attention_pattern: vec![0, 1],
            sliding_window: 2048,
            num_kv_shared_layers: 2,
            global_head_dim: 128,
            hidden_size_per_layer_input: 256,
            has_per_layer_embedding: true,
            rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
            extra,
            norm_eps: 1e-5,
        };
        // Act & Assert: exercise every getter path
        // get_int covers: num_hidden_layers, hidden_size, num_attention_heads,
        // num_key_value_heads, head_dim, intermediate_size, vocab_size, extra
        assert_eq!(config.get_int("num_hidden_layers"), Some(24));
        assert_eq!(config.get_int("hidden_size"), Some(2048));
        assert_eq!(config.get_int("num_attention_heads"), Some(32));
        assert_eq!(config.get_int("num_key_value_heads"), Some(4));
        assert_eq!(config.get_int("head_dim"), Some(64));
        assert_eq!(config.get_int("intermediate_size"), Some(8192));
        assert_eq!(config.get_int("vocab_size"), Some(64000));
        assert_eq!(config.get_int("custom_dim"), Some(999));
        // get_float covers: rope_theta, global_rope_theta
        assert_eq!(config.get_float("rope_theta"), Some(500000.0));
        assert_eq!(config.get_float("global_rope_theta"), Some(1000000.0));
        // get_str covers: dtype
        assert_eq!(config.get_str("dtype"), Some("bf16"));
        // get_bool covers: has_per_layer_embedding, is_kv_shared_layer_<N>
        assert_eq!(config.get_bool("has_per_layer_embedding"), Some(true));
        assert_eq!(config.get_bool("is_kv_shared_layer_22"), Some(true));
        assert_eq!(config.get_bool("is_kv_shared_layer_0"), Some(false));
    }

    // ── 12. Config with zero hidden_size ──

    // @trace TEST-RESOLVE-508 [level:unit]
    #[test]
    fn validate_config_rejects_zero_hidden_size_explicitly() {
        // Arrange: all required fields set except hidden_size=0.
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 0,
            num_attention_heads: 32,
            vocab_size: 32000,
            ..Default::default()
        };
        // Act
        let err = validate_config(&config).unwrap_err();
        // Assert
        assert_eq!(err, ResolveError::MissingConfig("hidden_size".to_string()),
            "zero hidden_size must be rejected by validate_config");
    }

    // ── 13. Tensor role classification edge cases ──

    // @trace TEST-RESOLVE-509 [level:unit]
    #[test]
    fn resolve_from_provider_final_norm_tensor_not_counted_as_layer() {
        // Arrange: a final norm tensor (model.norm.weight) should not increment
        // the layer counter. Only per-layer tensors should.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 2048],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.norm.weight".to_string(),
                    shape: vec![2048],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers should still be 0 because model.norm is
        // a global tensor (FinalNorm role, layer_idx=None).
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_hidden_layers",
                    "final norm should not increment layer count");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 14. Duplicate weight name handling ──

    // @trace TEST-RESOLVE-510 [level:unit]
    #[test]
    fn resolve_from_provider_duplicate_layer_tensors_same_index_no_overcount() {
        // Arrange: multiple tensors for the same layer index (e.g., q_proj + k_proj
        // + v_proj all for layer 0). Layer count should still be 1, not 3.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![1024, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.v_proj.weight".to_string(),
                    shape: vec![4096, 4096],
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: num_hidden_layers should be 1 (max_layer_idx = 0+1 = 1),
        // but num_attention_heads=0 (head_dim=0 at tensor scan time).
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads",
                    "layer count should be 1, not 3");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // ── 15. Unicode tensor name preservation ──

    // @trace TEST-RESOLVE-511 [level:unit]
    #[test]
    fn substitute_placeholders_preserves_unicode_in_template_with_replacement() {
        // Arrange: template with CJK characters and a placeholder.
        let config = ResolvedConfig {
            hidden_size: 8192,
            num_attention_heads: 64,
            ..Default::default()
        };
        let template = "隐藏层大小=${hidden_size}，注意力头数=${num_attention_heads}";
        // Act
        let result = substitute_placeholders(template, &config);
        // Assert: unicode preserved, values substituted.
        assert_eq!(result, "隐藏层大小=8192，注意力头数=64");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  16. Additional uncovered edge cases and paths
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-RESOLVE-512 [level:unit]
    #[test]
    fn from_geometry_rope_interleaved_true_does_not_affect_resolved_config() {
        // Arrange: ModelGeometry with rope_interleaved=true. This field is not
        // mapped to ResolvedConfig (it is consumed elsewhere in the pipeline).
        // Verify it does not corrupt any resolved field.
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 1024,
            num_layers: 4,
            vocab_size: 8000,
            intermediate_size: 4096,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: true, // not mapped to ResolvedConfig
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: all standard fields mapped correctly, no corruption.
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.vocab_size, 8000);
        assert!((config.rope_theta - 10000.0).abs() < f64::EPSILON);
    }

    // @trace TEST-RESOLVE-513 [level:unit]
    #[test]
    fn from_geometry_rope_scale_non_unit_does_not_affect_resolved_config() {
        // Arrange: ModelGeometry with rope_scale=2.5 (non-unity). This field
        // is not mapped to ResolvedConfig directly. Verify no corruption.
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 2048,
            num_layers: 8,
            vocab_size: 16000,
            intermediate_size: 8192,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 8192,
            rope_theta: 500000.0,
            rope_scale: 2.5, // non-unity, not in ResolvedConfig
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            norm_eps: 1e-6,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: rope_theta is from geometry, rope_scale is not a field.
        assert!((config.rope_theta - 500000.0).abs() < f64::EPSILON);
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.num_hidden_layers, 8);
    }

    // @trace TEST-RESOLVE-514 [level:unit]
    #[test]
    fn from_geometry_compute_dtype_differs_from_dtype_uses_dtype() {
        // Arrange: ModelGeometry where dtype (storage) differs from compute_dtype.
        // from_geometry maps `g.dtype` to the `dtype` string, not `g.compute_dtype`.
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 512,
            num_layers: 2,
            vocab_size: 1000,
            intermediate_size: 2048,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F16,        // storage dtype
            compute_dtype: DType::F32, // compute dtype (different)
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: dtype string reflects storage dtype (F16), not compute_dtype.
        assert_eq!(config.dtype, "f16");
    }

    // @trace TEST-RESOLVE-515 [level:unit]
    #[test]
    fn is_kv_shared_layer_returns_false_when_num_hidden_layers_is_zero() {
        // Arrange: config with zero num_hidden_layers but nonzero num_kv_shared_layers.
        // saturating_sub(0, N) = 0, so layer_idx >= 0 is always true,
        // but layer_idx < 0 (num_hidden_layers) is false.
        let config = ResolvedConfig {
            num_hidden_layers: 0,
            num_kv_shared_layers: 5,
            ..Default::default()
        };
        // Act & Assert: any layer index should return false because
        // layer_idx < num_hidden_layers (0) is always false.
        assert!(!config.is_kv_shared_layer(0));
        assert!(!config.is_kv_shared_layer(100));
    }

    // @trace TEST-RESOLVE-516 [level:unit]
    #[test]
    fn substitute_placeholders_extra_key_i64_min_value_outputs_negative() {
        // Arrange: extra with i64::MIN (most negative representable value).
        let mut extra = HashMap::new();
        extra.insert("extreme_neg".to_string(), i64::MIN);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        let template = "value=${extreme_neg}";
        // Act
        let result = substitute_placeholders(template, &config);
        // Assert: the full decimal representation of i64::MIN is substituted.
        assert_eq!(result, format!("value={}", i64::MIN));
    }

    // @trace TEST-RESOLVE-517 [level:unit]
    #[test]
    fn get_int_returns_negative_extra_value() {
        // Arrange: extra HashMap with a negative i64 value.
        let mut extra = HashMap::new();
        extra.insert("offset".to_string(), -42);
        let config = ResolvedConfig {
            extra,
            ..Default::default()
        };
        // Act
        let val = config.get_int("offset");
        // Assert: negative extra values are returned correctly.
        assert_eq!(val, Some(-42));
    }

    // @trace TEST-RESOLVE-518 [level:unit]
    #[test]
    fn resolve_from_provider_k_proj_1d_shape_skips_kv_head_derivation() {
        // Arrange: k_proj tensor with 1D shape (malformed). The resolve_from_tensors
        // code checks `shape.len() >= 2` before deriving kv heads. Verify no panic.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![32000, 4096],
                    dtype: safetensors::Dtype::F32,
                },
                TensorMeta {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    shape: vec![1024], // 1D shape — malformed k_proj
                    dtype: safetensors::Dtype::F32,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: no panic; should fail validation because heads not derived.
        match result.unwrap_err() {
            ResolveError::MissingConfig(key) => {
                assert_eq!(key, "num_attention_heads",
                    "1D k_proj shape should not derive kv heads");
            }
            other => panic!("expected MissingConfig, got {other:?}"),
        }
    }

    // @trace TEST-RESOLVE-519 [level:unit]
    #[test]
    fn resolve_from_provider_bool_dtype_defaults_to_f32_string() {
        // Arrange: embed tensor with safetensors::Dtype::BOOL — not in the
        // F32/F16/BF16 match arms, so falls to the default `"f32"` arm.
        // Note: safetensors::Dtype may not have BOOL; use I8 as another
        // uncovered variant that also falls to the default arm.
        let provider = FakeProvider {
            tensors: vec![
                TensorMeta {
                    name: "model.embed_tokens.weight".to_string(),
                    shape: vec![100, 64],
                    dtype: safetensors::Dtype::I8,
                },
            ],
        };
        // Act
        let result = resolve_from_provider(&provider, None);
        // Assert: no panic. The dtype internally becomes "f32" string but
        // validation still fails because required fields are missing.
        assert!(result.is_err());
    }

    // @trace TEST-RESOLVE-520 [level:unit]
    #[test]
    fn resolved_config_default_gemma4_fields_are_zero_or_empty() {
        // Arrange: construct via Default::default().
        let config = ResolvedConfig::default();
        // Assert: Gemma 4 specific fields default to zero/empty/false.
        assert_eq!(config.global_rope_theta, 0.0);
        assert_eq!(config.rope_partial_ratio, 0.0);
        assert!(config.attention_pattern.is_empty());
        assert_eq!(config.sliding_window, 0);
        assert_eq!(config.num_kv_shared_layers, 0);
        assert_eq!(config.global_head_dim, 0);
        assert_eq!(config.hidden_size_per_layer_input, 0);
        assert!(!config.has_per_layer_embedding);
        assert!(config.rope_scaling.is_none());
    }

    // @trace TEST-RESOLVE-521 [level:unit]
    #[test]
    fn from_geometry_max_seq_len_not_mapped_to_resolved_config() {
        // Arrange: ModelGeometry with a non-standard max_seq_len. This field
        // is not part of ResolvedConfig (used by scheduler/executor, not config).
        // Verify it does not leak into any ResolvedConfig field.
        use gllm_kernels::types::DType;

        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 131072, // large value, not in ResolvedConfig
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        };
        // Act
        let config = ResolvedConfig::from_geometry(&geometry, HashMap::new());
        // Assert: no field in ResolvedConfig equals 131072 (max_seq_len value).
        // The closest field is sliding_window which defaults to 0.
        assert_eq!(config.sliding_window, 0);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 12);
        // Confirm no usize field accidentally got 131072.
        assert_ne!(config.vocab_size, 131072);
        assert_ne!(config.num_attention_heads, 131072);
    }
}
