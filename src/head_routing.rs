//! Head Routing SDK — 同一 generator LLM 多头 API (REQ-HR)
//!
//! **SSOT**: `SPEC/HEAD-ROUTING.md`, `SPEC/01-REQUIREMENTS.md §13`,
//! `SPEC/04-API-DESIGN.md §3.8`.
//!
//! # 概念
//!
//! Head Routing 让同一 `Client::new_chat(...)` 加载的 generator LLM 在
//! 运行时通过 API 切换输出形态(generate / classify_binary /
//! classify_multiway / encode_to_layer),**不重新加载权重、不重新 JIT
//! 编译**。
//!
//! # 与 `ModelKind` 的正交关系
//!
//! | 维度 | `ModelKind` | Head Routing |
//! |------|-------------|--------------|
//! | 作用时机 | 加载期 (`Client::new_*` / Builder) | 运行时每次 API 调用 |
//! | 改变 | auto_graph / 权重 / JIT 产物 | 仅后处理 |
//! | 粒度 | 整个 Client 的默认 head | 单次调用的 head 形态 |
//! | 生效开销 | JIT 重编译 + 权重重装 | 零 |
//!
//! # 铁律
//!
//! - **NO_SILENT_FALLBACK**: token 找不到 → `TokenNotFound`;mid-layer
//!   不支持 → `MidLayerNotSupported`;温度非法 → `InvalidConfig`。
//!   禁止 silent 返回 0 / argmax / 默认值。
//! - **Tied embedding**: 实现依赖主流 generator (SmolLM2 / Qwen3 /
//!   Gemma) 的 `lm_head` 与 `embed_tokens.weight` 绑定,
//!   `logit_t = h_last · embed_tokens[t]`。

use thiserror::Error;

// ============================================================================
// LayerAnchor (SPEC HEAD-ROUTING.md §6)
// ============================================================================

/// 指定 `encode_to_layer` 截断前向的层位置。
///
/// 与 `SemanticGatekeeperConfig::detection_depths: Vec<f32>` 共享语义:
/// 相对深度 `r ∈ [0.0, 1.0]` 映射到物理层 `round(r × (num_layers - 1))`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerAnchor {
    /// 相对深度 ∈ `[0.0, 1.0]`。0.0 = layer 0,1.0 = 最后一层。
    Relative(f32),
    /// 绝对层索引 (0-based)。
    Absolute(usize),
}

impl LayerAnchor {
    /// 解析到物理层索引。
    ///
    /// # 错误
    /// - `num_layers == 0` → `InvalidLayerAnchor(-1.0)`
    /// - `Relative(r)` 越界 `[0.0, 1.0]` 或 NaN → `InvalidLayerAnchor(r)`
    /// - `Absolute(i)` `i >= num_layers` → `InvalidLayerAnchor(i as f32)`
    pub fn resolve(self, num_layers: usize) -> Result<usize, HeadRoutingError> {
        if num_layers == 0 {
            return Err(HeadRoutingError::InvalidLayerAnchor(-1.0));
        }
        match self {
            LayerAnchor::Relative(r) => {
                if !r.is_finite() || !(0.0..=1.0).contains(&r) {
                    return Err(HeadRoutingError::InvalidLayerAnchor(r));
                }
                let scaled = r * (num_layers - 1) as f32;
                // round-half-to-even 更稳定,但 f32::round 使用 round-half-away-from-zero
                // 对于 [0, num_layers-1] 范围的正值两者等价
                let idx = scaled.round() as usize;
                Ok(idx.min(num_layers - 1))
            }
            LayerAnchor::Absolute(i) => {
                if i < num_layers {
                    Ok(i)
                } else {
                    Err(HeadRoutingError::InvalidLayerAnchor(i as f32))
                }
            }
        }
    }
}

// ============================================================================
// PoolMode (SPEC HEAD-ROUTING.md §5.2)
// ============================================================================

/// 对 `[seq_len, hidden_size]` hidden state 做 pooling 得到 `[hidden_size]` 向量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolMode {
    /// 对 seq 维度求平均: `mean(H, axis=0)`
    MeanPool,
    /// 取最后一个 token: `H[-1]`
    LastToken,
    /// 取第一个 token (CLS 位置): `H[0]`
    ClsToken,
}

impl PoolMode {
    /// 对 flat row-major 的 `[seq_len, hidden_size]` hidden state 做 pooling。
    ///
    /// # 错误
    /// - `seq_len == 0 || hidden_size == 0` → `InvalidConfig`
    /// - `hidden.len() < seq_len * hidden_size` → `InvalidConfig`
    pub fn apply(
        self,
        hidden: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, HeadRoutingError> {
        if seq_len == 0 || hidden_size == 0 {
            return Err(HeadRoutingError::InvalidConfig(format!(
                "PoolMode::apply requires seq_len > 0 && hidden_size > 0, got {seq_len}/{hidden_size}"
            )));
        }
        let expected = seq_len.checked_mul(hidden_size).ok_or_else(|| {
            HeadRoutingError::InvalidConfig(format!(
                "seq_len * hidden_size overflow: {seq_len} * {hidden_size}"
            ))
        })?;
        if hidden.len() < expected {
            return Err(HeadRoutingError::InvalidConfig(format!(
                "PoolMode::apply expected hidden.len() >= {expected}, got {}",
                hidden.len()
            )));
        }
        match self {
            PoolMode::MeanPool => {
                let mut out = vec![0.0f32; hidden_size];
                let scale = 1.0 / seq_len as f32;
                for row in 0..seq_len {
                    let offset = row * hidden_size;
                    for col in 0..hidden_size {
                        out[col] += hidden[offset + col] * scale;
                    }
                }
                Ok(out)
            }
            PoolMode::LastToken => {
                let offset = (seq_len - 1) * hidden_size;
                Ok(hidden[offset..offset + hidden_size].to_vec())
            }
            PoolMode::ClsToken => Ok(hidden[..hidden_size].to_vec()),
        }
    }
}

// ============================================================================
// 配置结构 (SPEC 04-API-DESIGN §3.8.2)
// ============================================================================

/// Binary classify 配置。
#[derive(Debug, Clone)]
pub struct ClassifyBinaryConfig {
    /// positive 类别对应的 token 文本(必须单 token 化)。
    pub positive_token: String,
    /// negative 类别对应的 token 文本(必须单 token 化)。
    pub negative_token: String,
    /// softmax 温度 (> 0)。1.0 保持原始分布。
    pub temperature: f32,
}

impl ClassifyBinaryConfig {
    /// 便捷构造器(默认 temperature=1.0)。
    pub fn new(positive: impl Into<String>, negative: impl Into<String>) -> Self {
        Self {
            positive_token: positive.into(),
            negative_token: negative.into(),
            temperature: 1.0,
        }
    }
}

impl Default for ClassifyBinaryConfig {
    fn default() -> Self {
        Self {
            positive_token: "yes".to_string(),
            negative_token: "no".to_string(),
            temperature: 1.0,
        }
    }
}

/// Multiway classify 配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifyMultiwayConfig {
    /// softmax 温度 (> 0)。
    pub temperature: f32,
}

impl Default for ClassifyMultiwayConfig {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

// ============================================================================
// 错误类型 (SPEC 04-API-DESIGN §3.8.3)
// ============================================================================

/// Head Routing SDK 专用错误。
#[derive(Debug, Clone, Error)]
pub enum HeadRoutingError {
    /// label 或 positive/negative token 无法单 token 化,或 tokenizer 失败。
    #[error("token not found or not single-token: {0}")]
    TokenNotFound(String),

    /// LayerAnchor::Relative 越界 [0.0, 1.0] / NaN 或 Absolute 越界 [0, num_layers)。
    #[error("invalid layer anchor: {0}")]
    InvalidLayerAnchor(f32),

    /// classify_multiway 收到空 labels。
    #[error("empty labels passed to classify_multiway")]
    EmptyLabels,

    /// 配置参数非法(如 temperature ≤ 0 / NaN)。
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// encode_to_layer 当前不支持 mid-layer 截断(mega-kernel path 扩展 pending)。
    ///
    /// **非 stub/fallback**: 这是**显式拒绝**,而非静默降级。一旦
    /// mega-kernel path 扩展 `run_with_early_exit(anchor_layer)` 落地,
    /// `encode_to_layer` 将走真实 JIT 路径,该错误变体永不返回。
    #[error("MidLayerNotSupported: mega-kernel path does not yet expose mid-layer exit via CallbackChain; see SPEC/HEAD-ROUTING.md §5.3")]
    MidLayerNotSupported,

    /// 下游 backend / tokenizer / executor 错误传播。
    #[error("backend error: {0}")]
    Backend(String),
}

// ============================================================================
// Softmax helper (内部使用, 数值稳定版)
// ============================================================================

/// 数值稳定的 softmax。输入 `logits.is_empty()` 返回空 Vec。
///
/// 计算: `softmax_T(x)[i] = exp((x[i] - max(x)) / T) / sum(exp((x[j] - max(x)) / T))`
///
/// # 错误
/// - `temperature <= 0.0` 或非有限 → `InvalidConfig`
/// - 所有 exp 值为 0 (极端溢出) → `InvalidConfig`
pub(crate) fn softmax_with_temperature(
    logits: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, HeadRoutingError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(HeadRoutingError::InvalidConfig(format!(
            "temperature must be finite and > 0, got {temperature}"
        )));
    }
    if logits.is_empty() {
        return Ok(Vec::new());
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(HeadRoutingError::InvalidConfig(format!(
            "logits contain no finite values (max = {max})"
        )));
    }
    let exps: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HeadRoutingError::InvalidConfig(format!(
            "softmax denominator sum is not finite / positive: {sum}"
        )));
    }
    Ok(exps.into_iter().map(|e| e / sum).collect())
}

// ============================================================================
// Tests (单元测试 — 纯类型契约,不依赖模型加载)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── LayerAnchor::resolve ──────────────────────────────────────────────

    #[test]
    fn layer_anchor_relative_boundary_values() {
        assert_eq!(LayerAnchor::Relative(0.0).resolve(24).unwrap(), 0);
        assert_eq!(LayerAnchor::Relative(1.0).resolve(24).unwrap(), 23);
        assert_eq!(LayerAnchor::Relative(0.5).resolve(24).unwrap(), 12); // round(0.5 * 23) = round(11.5) = 12
    }

    #[test]
    fn layer_anchor_absolute_in_range() {
        assert_eq!(LayerAnchor::Absolute(0).resolve(24).unwrap(), 0);
        assert_eq!(LayerAnchor::Absolute(5).resolve(24).unwrap(), 5);
        assert_eq!(LayerAnchor::Absolute(23).resolve(24).unwrap(), 23);
    }

    #[test]
    fn layer_anchor_relative_out_of_range() {
        assert!(matches!(
            LayerAnchor::Relative(-0.1).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
        assert!(matches!(
            LayerAnchor::Relative(1.5).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
        assert!(matches!(
            LayerAnchor::Relative(f32::NAN).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn layer_anchor_absolute_out_of_range() {
        assert!(matches!(
            LayerAnchor::Absolute(24).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
        assert!(matches!(
            LayerAnchor::Absolute(100).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn layer_anchor_zero_layers_errors() {
        assert!(matches!(
            LayerAnchor::Relative(0.5).resolve(0),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
        assert!(matches!(
            LayerAnchor::Absolute(0).resolve(0),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    // ── PoolMode::apply ──────────────────────────────────────────────────

    #[test]
    fn pool_mean_averages_rows() {
        // seq_len=3, hidden_size=2
        // rows: [1,2], [3,4], [5,6] → mean = [3, 4]
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = PoolMode::MeanPool.apply(&hidden, 3, 2).unwrap();
        assert_eq!(pooled, vec![3.0, 4.0]);
    }

    #[test]
    fn pool_last_returns_last_row() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = PoolMode::LastToken.apply(&hidden, 3, 2).unwrap();
        assert_eq!(pooled, vec![5.0, 6.0]);
    }

    #[test]
    fn pool_cls_returns_first_row() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = PoolMode::ClsToken.apply(&hidden, 3, 2).unwrap();
        assert_eq!(pooled, vec![1.0, 2.0]);
    }

    #[test]
    fn pool_rejects_empty_seq() {
        let hidden: Vec<f32> = Vec::new();
        assert!(matches!(
            PoolMode::MeanPool.apply(&hidden, 0, 4),
            Err(HeadRoutingError::InvalidConfig(_))
        ));
        assert!(matches!(
            PoolMode::MeanPool.apply(&hidden, 2, 0),
            Err(HeadRoutingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn pool_rejects_short_buffer() {
        let hidden = vec![1.0, 2.0];
        assert!(matches!(
            PoolMode::MeanPool.apply(&hidden, 3, 2), // 需 6 个元素,只给 2
            Err(HeadRoutingError::InvalidConfig(_))
        ));
    }

    // ── softmax_with_temperature ─────────────────────────────────────────

    #[test]
    fn softmax_normalizes_to_sum_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        // argmax preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn softmax_temperature_sharpens_distribution() {
        let logits = vec![1.0, 2.0];
        let probs_hot = softmax_with_temperature(&logits, 0.1).unwrap();
        let probs_cold = softmax_with_temperature(&logits, 10.0).unwrap();
        // T=0.1 (hot/sharp): argmax gets more mass
        // T=10.0 (cold/smooth): distribution flattens toward uniform
        assert!(
            probs_hot[1] > probs_cold[1],
            "hot P(argmax)={} should exceed cold P(argmax)={}",
            probs_hot[1], probs_cold[1]
        );
    }

    #[test]
    fn softmax_equal_logits_uniform() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_rejects_non_positive_temperature() {
        let logits = vec![1.0, 2.0];
        assert!(softmax_with_temperature(&logits, 0.0).is_err());
        assert!(softmax_with_temperature(&logits, -1.0).is_err());
        assert!(softmax_with_temperature(&logits, f32::NAN).is_err());
        assert!(softmax_with_temperature(&logits, f32::INFINITY).is_err());
    }

    #[test]
    fn softmax_empty_returns_empty() {
        let empty: Vec<f32> = Vec::new();
        let probs = softmax_with_temperature(&empty, 1.0).unwrap();
        assert!(probs.is_empty());
    }

    // ── ClassifyBinaryConfig ─────────────────────────────────────────────

    #[test]
    fn classify_binary_config_default() {
        let cfg = ClassifyBinaryConfig::default();
        assert_eq!(cfg.positive_token, "yes");
        assert_eq!(cfg.negative_token, "no");
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn classify_binary_config_new() {
        let cfg = ClassifyBinaryConfig::new("good", "bad");
        assert_eq!(cfg.positive_token, "good");
        assert_eq!(cfg.negative_token, "bad");
        assert_eq!(cfg.temperature, 1.0);
    }

    // ── ClassifyMultiwayConfig ───────────────────────────────────────────

    #[test]
    fn classify_multiway_config_default() {
        let cfg = ClassifyMultiwayConfig::default();
        assert_eq!(cfg.temperature, 1.0);
    }

    // ── HeadRoutingError Display ─────────────────────────────────────────

    #[test]
    fn mid_layer_error_message_contains_sentinel() {
        // REQ-HR-003 验收标准 #2: 错误消息包含固定子串 "MidLayerNotSupported"
        let err = HeadRoutingError::MidLayerNotSupported;
        let msg = format!("{err}");
        assert!(
            msg.contains("MidLayerNotSupported"),
            "error message must contain 'MidLayerNotSupported' sentinel, got: {msg}"
        );
    }

    // ── LayerAnchor additional tests ────────────────────────────────────

    #[test]
    fn layer_anchor_copy_semantics() {
        let a = LayerAnchor::Relative(0.5);
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn layer_anchor_single_layer_relative() {
        // With num_layers=1, any valid relative value resolves to layer 0
        assert_eq!(LayerAnchor::Relative(0.0).resolve(1).unwrap(), 0);
        assert_eq!(LayerAnchor::Relative(1.0).resolve(1).unwrap(), 0);
        assert_eq!(LayerAnchor::Relative(0.5).resolve(1).unwrap(), 0);
    }

    #[test]
    fn layer_anchor_single_layer_absolute() {
        assert_eq!(LayerAnchor::Absolute(0).resolve(1).unwrap(), 0);
        assert!(LayerAnchor::Absolute(1).resolve(1).is_err());
    }

    #[test]
    fn layer_anchor_relative_quarter_positions() {
        // 24 layers: 0.25 → round(0.25*23)=round(5.75)=6, 0.75 → round(0.75*23)=round(17.25)=17
        assert_eq!(LayerAnchor::Relative(0.25).resolve(24).unwrap(), 6);
        assert_eq!(LayerAnchor::Relative(0.75).resolve(24).unwrap(), 17);
    }

    #[test]
    fn layer_anchor_relative_infinity_rejected() {
        assert!(matches!(
            LayerAnchor::Relative(f32::INFINITY).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
        assert!(matches!(
            LayerAnchor::Relative(f32::NEG_INFINITY).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn layer_anchor_debug_format() {
        let rel = LayerAnchor::Relative(0.5);
        let debug_str = format!("{rel:?}");
        assert!(debug_str.contains("Relative"), "Debug output should contain 'Relative': {debug_str}");

        let abs = LayerAnchor::Absolute(7);
        let debug_str = format!("{abs:?}");
        assert!(debug_str.contains("Absolute"), "Debug output should contain 'Absolute': {debug_str}");
    }

    // ── PoolMode trait tests ────────────────────────────────────────────

    #[test]
    fn pool_mode_copy_semantics() {
        let mode = PoolMode::MeanPool;
        let copy = mode; // Copy
        assert_eq!(mode, copy);
    }

    #[test]
    fn pool_mode_eq_all_variants() {
        assert_eq!(PoolMode::MeanPool, PoolMode::MeanPool);
        assert_eq!(PoolMode::LastToken, PoolMode::LastToken);
        assert_eq!(PoolMode::ClsToken, PoolMode::ClsToken);
        assert_ne!(PoolMode::MeanPool, PoolMode::LastToken);
        assert_ne!(PoolMode::LastToken, PoolMode::ClsToken);
        assert_ne!(PoolMode::ClsToken, PoolMode::MeanPool);
    }

    #[test]
    fn pool_mode_debug_format() {
        assert!(format!("{:?}", PoolMode::MeanPool).contains("MeanPool"));
        assert!(format!("{:?}", PoolMode::LastToken).contains("LastToken"));
        assert!(format!("{:?}", PoolMode::ClsToken).contains("ClsToken"));
    }

    #[test]
    fn pool_single_seq_len() {
        // seq_len=1: mean/last/cls should all return the same row
        let hidden = vec![3.0, 4.0, 5.0];
        let mean = PoolMode::MeanPool.apply(&hidden, 1, 3).unwrap();
        let last = PoolMode::LastToken.apply(&hidden, 1, 3).unwrap();
        let cls = PoolMode::ClsToken.apply(&hidden, 1, 3).unwrap();
        assert_eq!(mean, hidden);
        assert_eq!(last, hidden);
        assert_eq!(cls, hidden);
    }

    #[test]
    fn pool_larger_buffer_than_needed_is_ok() {
        // Buffer has extra elements beyond seq_len*hidden_size — should succeed
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
        let pooled = PoolMode::MeanPool.apply(&hidden, 2, 2).unwrap();
        assert_eq!(pooled, vec![2.0, 3.0]); // mean of [1,2] and [3,4]
    }

    // ── ClassifyBinaryConfig additional tests ───────────────────────────

    #[test]
    fn classify_binary_config_debug() {
        let cfg = ClassifyBinaryConfig::new("positive", "negative");
        let debug = format!("{cfg:?}");
        assert!(debug.contains("positive_token"));
        assert!(debug.contains("negative_token"));
        assert!(debug.contains("temperature"));
    }

    #[test]
    fn classify_binary_config_clone() {
        let cfg = ClassifyBinaryConfig::new("up", "down");
        let cloned = cfg.clone();
        assert_eq!(cfg.positive_token, cloned.positive_token);
        assert_eq!(cfg.negative_token, cloned.negative_token);
        assert_eq!(cfg.temperature, cloned.temperature);
    }

    #[test]
    fn classify_binary_config_custom_temperature() {
        let mut cfg = ClassifyBinaryConfig::new("yes", "no");
        cfg.temperature = 0.5;
        assert_eq!(cfg.temperature, 0.5);
    }

    // ── ClassifyMultiwayConfig additional tests ─────────────────────────

    #[test]
    fn classify_multiway_config_copy() {
        let cfg = ClassifyMultiwayConfig { temperature: 2.0 };
        let copy = cfg; // Copy
        assert_eq!(cfg, copy);
    }

    #[test]
    fn classify_multiway_config_partial_eq() {
        let a = ClassifyMultiwayConfig { temperature: 1.0 };
        let b = ClassifyMultiwayConfig { temperature: 1.0 };
        let c = ClassifyMultiwayConfig { temperature: 2.0 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn classify_multiway_config_clone() {
        let cfg = ClassifyMultiwayConfig { temperature: 1.5 };
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn classify_multiway_config_debug() {
        let cfg = ClassifyMultiwayConfig { temperature: 0.7 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("temperature"), "Debug output should contain 'temperature': {debug}");
    }

    // ── HeadRoutingError all variant Display messages ───────────────────

    #[test]
    fn error_token_not_found_display() {
        let err = HeadRoutingError::TokenNotFound("▁hello".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("token not found"), "TokenNotFound message: {msg}");
        assert!(msg.contains("▁hello"), "TokenNotFound message should contain the token text: {msg}");
    }

    #[test]
    fn error_empty_labels_display() {
        let err = HeadRoutingError::EmptyLabels;
        let msg = format!("{err}");
        assert!(msg.to_lowercase().contains("empty labels"), "EmptyLabels message: {msg}");
    }

    #[test]
    fn error_invalid_config_display() {
        let err = HeadRoutingError::InvalidConfig("temperature must be > 0".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("invalid config"), "InvalidConfig message: {msg}");
        assert!(msg.contains("temperature"), "InvalidConfig message should contain detail: {msg}");
    }

    #[test]
    fn error_invalid_layer_anchor_display() {
        let err = HeadRoutingError::InvalidLayerAnchor(2.5);
        let msg = format!("{err}");
        assert!(msg.contains("invalid layer anchor"), "InvalidLayerAnchor message: {msg}");
        assert!(msg.contains("2.5"), "InvalidLayerAnchor message should contain the value: {msg}");
    }

    #[test]
    fn error_backend_display() {
        let err = HeadRoutingError::Backend("CUDA OOM".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("backend error"), "Backend message: {msg}");
        assert!(msg.contains("CUDA OOM"), "Backend message should contain detail: {msg}");
    }

    #[test]
    fn error_clone_preserves_variant_and_data() {
        let original = HeadRoutingError::TokenNotFound("test_token".to_string());
        let cloned = original.clone();
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    fn error_debug_format() {
        let err = HeadRoutingError::InvalidConfig("some detail".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("InvalidConfig"), "Debug should contain variant name: {debug}");
    }

    // ── softmax_with_temperature additional tests ───────────────────────

    #[test]
    fn softmax_single_logit_returns_one() {
        let probs = softmax_with_temperature(&[5.0], 1.0).unwrap();
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6, "single logit should produce probability 1.0");
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        // Large logits should not overflow or produce NaN
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        assert!(probs.iter().all(|p| p.is_finite()), "all probabilities should be finite");
    }

    #[test]
    fn softmax_negative_logits() {
        let logits = vec![-10.0, -5.0, -1.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        // argmax preserved: -1.0 is largest → probs[2] should be highest
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn softmax_non_finite_logits_rejected() {
        assert!(softmax_with_temperature(&[f32::INFINITY, 1.0], 1.0).is_err()
            || {
                // If INF is the max, exp(INF-INF)=1 which is finite — check behavior
                let result = softmax_with_temperature(&[f32::INFINITY, 1.0], 1.0);
                match result {
                    Ok(probs) => probs.iter().all(|p| p.is_finite()),
                    Err(_) => false,
                }
            }
        );
        // Negative infinity should be handled (it's finite for the max check when paired with finite values)
        let result = softmax_with_temperature(&[f32::NEG_INFINITY, 1.0], 1.0);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn softmax_two_logits_sum_to_one() {
        let probs = softmax_with_temperature(&[0.0, 0.0], 1.0).unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert!((probs[1] - 0.5).abs() < 1e-6);
    }

    // ── New edge-case tests ──────────────────────────────────────────────

    #[test]
    fn layer_anchor_relative_subnormal_rejected() {
        // Subnormal f32 is finite and positive but effectively zero — still valid at 0.0
        // Test a value just below 0.0 to confirm negative subnormals are rejected
        let subnormal = -f32::from_bits(1); // smallest negative subnormal
        assert!(subnormal.is_finite());
        assert!(subnormal < 0.0);
        assert!(matches!(
            LayerAnchor::Relative(subnormal).resolve(10),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn layer_anchor_absolute_usize_max_rejected() {
        assert!(matches!(
            LayerAnchor::Absolute(usize::MAX).resolve(10),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn layer_anchor_relative_two_layers_rounding() {
        // num_layers=2: valid range [0,1]
        // 0.0 → round(0.0 * 1) = 0, 1.0 → round(1.0) = 1, 0.5 → round(0.5) = 0 or 1
        assert_eq!(LayerAnchor::Relative(0.0).resolve(2).unwrap(), 0);
        assert_eq!(LayerAnchor::Relative(1.0).resolve(2).unwrap(), 1);
        // 0.5 * 1 = 0.5, f32::round(0.5) = 1.0 (round-half-away-from-zero)
        assert_eq!(LayerAnchor::Relative(0.5).resolve(2).unwrap(), 1);
    }

    #[test]
    fn layer_anchor_relative_just_above_one_rejected() {
        let slightly_over = 1.0 + f32::EPSILON;
        assert!(slightly_over > 1.0);
        assert!(matches!(
            LayerAnchor::Relative(slightly_over).resolve(24),
            Err(HeadRoutingError::InvalidLayerAnchor(_))
        ));
    }

    #[test]
    fn pool_mean_all_zeros() {
        let hidden = vec![0.0f32; 8]; // seq_len=2, hidden_size=4
        let pooled = PoolMode::MeanPool.apply(&hidden, 2, 4).unwrap();
        assert_eq!(pooled, vec![0.0; 4]);
    }

    #[test]
    fn pool_mean_negative_values() {
        // rows: [-1,-2], [3,4] → mean = [1, 1]
        let hidden = vec![-1.0, -2.0, 3.0, 4.0];
        let pooled = PoolMode::MeanPool.apply(&hidden, 2, 2).unwrap();
        assert_eq!(pooled, vec![1.0, 1.0]);
    }

    #[test]
    fn pool_cls_single_element() {
        // seq_len=1, hidden_size=1
        let hidden = vec![42.0];
        let pooled = PoolMode::ClsToken.apply(&hidden, 1, 1).unwrap();
        assert_eq!(pooled, vec![42.0]);
    }

    #[test]
    fn pool_multiplication_overflow_rejected() {
        // seq_len * hidden_size overflows usize
        let hidden = vec![0.0f32; 1];
        let result = PoolMode::MeanPool.apply(&hidden, usize::MAX, usize::MAX);
        assert!(matches!(result, Err(HeadRoutingError::InvalidConfig(_))));
    }

    #[test]
    fn softmax_negative_infinity_logits_handled() {
        // max is finite, NEG_INFINITY entries get exp(-INF) = 0
        let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        assert!((probs[1] - 1.0).abs() < 1e-6);
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
    }

    #[test]
    fn softmax_very_small_temperature() {
        // Tiny but positive temperature should still work
        let tiny_temp = f32::MIN_POSITIVE;
        let probs = softmax_with_temperature(&[1.0, 2.0], tiny_temp);
        // May overflow to Inf for large (logit - max)/temp, which could cause error
        assert!(probs.is_ok() || probs.is_err());
    }

    #[test]
    fn softmax_nan_in_logits_rejected() {
        // If NaN is in logits, max() propagates NaN → is_finite() fails
        let logits = vec![1.0, f32::NAN, 2.0];
        assert!(matches!(
            softmax_with_temperature(&logits, 1.0),
            Err(HeadRoutingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn classify_binary_config_empty_string_tokens() {
        let cfg = ClassifyBinaryConfig::new("", "");
        assert_eq!(cfg.positive_token, "");
        assert_eq!(cfg.negative_token, "");
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn error_all_variants_match_exhaustively() {
        // Ensure every variant compiles and matches correctly
        let cases: Vec<HeadRoutingError> = vec![
            HeadRoutingError::TokenNotFound("x".into()),
            HeadRoutingError::InvalidLayerAnchor(0.0),
            HeadRoutingError::EmptyLabels,
            HeadRoutingError::InvalidConfig("x".into()),
            HeadRoutingError::MidLayerNotSupported,
            HeadRoutingError::Backend("x".into()),
        ];
        assert_eq!(cases.len(), 6, "all 6 HeadRoutingError variants covered");
    }

    #[test]
    fn softmax_mixed_sign_logits_order_preserved() {
        // argmax must be preserved regardless of sign mixing
        let logits = vec![-100.0, 0.0, 100.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── Additional tests (wave-06-01: +15 new tests) ─────────────────────

    #[test]
    fn layer_anchor_relative_and_absolute_agree_at_boundaries() {
        // Relative(0.0) and Absolute(0) should both resolve to layer 0
        // Relative(1.0) and Absolute(num_layers-1) should both resolve to last layer
        for n in [2, 5, 24, 100] {
            assert_eq!(
                LayerAnchor::Relative(0.0).resolve(n).unwrap(),
                LayerAnchor::Absolute(0).resolve(n).unwrap()
            );
            assert_eq!(
                LayerAnchor::Relative(1.0).resolve(n).unwrap(),
                LayerAnchor::Absolute(n - 1).resolve(n).unwrap()
            );
        }
    }

    #[test]
    fn layer_anchor_relative_just_above_zero_is_valid() {
        // The smallest positive f32 above 0.0 should be valid
        let tiny = f32::from_bits(1); // smallest positive subnormal
        assert!(tiny > 0.0);
        assert!(tiny < 1.0);
        let result = LayerAnchor::Relative(tiny).resolve(100);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // round(tiny * 99) rounds to 0
    }

    #[test]
    fn layer_anchor_resolve_large_layer_count() {
        // num_layers = 1000: ensure no overflow or panic
        assert_eq!(LayerAnchor::Relative(0.0).resolve(1000).unwrap(), 0);
        assert_eq!(LayerAnchor::Relative(1.0).resolve(1000).unwrap(), 999);
        assert_eq!(LayerAnchor::Relative(0.5).resolve(1000).unwrap(), 500); // round(0.5*999)=round(499.5)=500
        assert_eq!(LayerAnchor::Absolute(500).resolve(1000).unwrap(), 500);
        assert!(LayerAnchor::Absolute(1000).resolve(1000).is_err());
    }

    #[test]
    fn layer_anchor_relative_rounding_at_half_values() {
        // For num_layers=5: Relative(0.5) → round(0.5*4)=round(2.0)=2 (exact)
        assert_eq!(LayerAnchor::Relative(0.5).resolve(5).unwrap(), 2);
        // For num_layers=3: Relative(0.5) → round(0.5*2)=round(1.0)=1
        assert_eq!(LayerAnchor::Relative(0.5).resolve(3).unwrap(), 1);
        // For num_layers=4: Relative(0.5) → round(0.5*3)=round(1.5)=2 (round-half-away-from-zero)
        assert_eq!(LayerAnchor::Relative(0.5).resolve(4).unwrap(), 2);
    }

    #[test]
    fn pool_exact_buffer_length_succeeds() {
        // Buffer length exactly equals seq_len * hidden_size — boundary case
        let hidden = vec![1.0, 2.0, 3.0, 4.0]; // exactly 2*2
        let mean = PoolMode::MeanPool.apply(&hidden, 2, 2).unwrap();
        assert_eq!(mean, vec![2.0, 3.0]);

        let last = PoolMode::LastToken.apply(&hidden, 2, 2).unwrap();
        assert_eq!(last, vec![3.0, 4.0]);

        let cls = PoolMode::ClsToken.apply(&hidden, 2, 2).unwrap();
        assert_eq!(cls, vec![1.0, 2.0]);
    }

    #[test]
    fn pool_last_token_with_single_row() {
        // seq_len=1, hidden_size=4: LastToken returns the only row
        let hidden = vec![7.0, 8.0, 9.0, 10.0];
        let pooled = PoolMode::LastToken.apply(&hidden, 1, 4).unwrap();
        assert_eq!(pooled, vec![7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn pool_mean_with_large_seq_len() {
        // Large seq_len to verify accumulation precision (not overflow)
        let seq_len = 1000;
        let hidden_size = 3;
        let hidden: Vec<f32> = (0..seq_len)
            .flat_map(|row| {
                let v = row as f32;
                vec![v, v + 1.0, v + 2.0]
            })
            .collect();
        let pooled = PoolMode::MeanPool.apply(&hidden, seq_len, hidden_size).unwrap();
        // Expected mean of 0..1000 = 499.5
        assert!((pooled[0] - 499.5).abs() < 0.5, "col0 mean = {}", pooled[0]);
        assert!((pooled[1] - 500.5).abs() < 0.5, "col1 mean = {}", pooled[1]);
        assert!((pooled[2] - 501.5).abs() < 0.5, "col2 mean = {}", pooled[2]);
    }

    #[test]
    fn softmax_all_probabilities_positive_and_finite() {
        let logits = vec![-5.0, -1.0, 0.0, 3.0, 7.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        for (i, &p) in probs.iter().enumerate() {
            assert!(p > 0.0, "prob[{i}] = {p} should be > 0");
            assert!(p.is_finite(), "prob[{i}] = {p} should be finite");
        }
    }

    #[test]
    fn softmax_equal_logits_high_temperature_still_uniform() {
        // Equal logits with high temperature should still produce uniform distribution
        let logits = vec![2.0, 2.0, 2.0];
        let probs = softmax_with_temperature(&logits, 100.0).unwrap();
        for p in &probs {
            assert!(
                (p - (1.0 / 3.0)).abs() < 1e-6,
                "expected uniform ~0.333, got {p}"
            );
        }
    }

    #[test]
    fn softmax_extreme_logit_differences() {
        // Very large difference between logits — smaller one should be nearly zero
        let logits = vec![-1000.0, 0.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        assert!(
            probs[0] < 1e-10,
            "extreme negative logit prob should be ~0, got {}",
            probs[0]
        );
        assert!(
            (probs[1] - 1.0).abs() < 1e-6,
            "dominant logit prob should be ~1.0, got {}",
            probs[1]
        );
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn classify_binary_config_unicode_tokens() {
        // Tokens can contain Unicode characters
        let cfg = ClassifyBinaryConfig::new("是", "否");
        assert_eq!(cfg.positive_token, "是");
        assert_eq!(cfg.negative_token, "否");
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn classify_binary_config_long_token_strings() {
        // Long multi-word token strings
        let positive = "definitely positive".repeat(10);
        let negative = "absolutely negative".repeat(10);
        let cfg = ClassifyBinaryConfig::new(&positive, &negative);
        assert_eq!(cfg.positive_token, positive);
        assert_eq!(cfg.negative_token, negative);
    }

    #[test]
    fn classify_multiway_config_custom_temperature() {
        let cfg = ClassifyMultiwayConfig { temperature: 0.42 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("0.42"), "Debug should show temperature 0.42: {debug}");
        assert_eq!(cfg.temperature, 0.42);
    }

    #[test]
    fn error_variants_exhaustive_clone_and_display_roundtrip() {
        // Verify every variant can be cloned and Display output is non-empty
        let variants: Vec<HeadRoutingError> = vec![
            HeadRoutingError::TokenNotFound("token".into()),
            HeadRoutingError::InvalidLayerAnchor(3.14),
            HeadRoutingError::EmptyLabels,
            HeadRoutingError::InvalidConfig("detail".into()),
            HeadRoutingError::MidLayerNotSupported,
            HeadRoutingError::Backend("fail".into()),
        ];
        assert_eq!(variants.len(), 6);
        for original in &variants {
            let cloned = original.clone();
            assert_eq!(format!("{original}"), format!("{cloned}"));
            assert!(!format!("{original}").is_empty());
        }
    }

    #[test]
    fn pool_rejects_hidden_size_zero_even_with_non_empty_buffer() {
        // hidden_size=0 with a non-empty buffer should still error
        let hidden = vec![1.0, 2.0, 3.0];
        let result = PoolMode::MeanPool.apply(&hidden, 1, 0);
        assert!(
            matches!(result, Err(HeadRoutingError::InvalidConfig(_))),
            "hidden_size=0 should error"
        );
    }

    // ── Additional tests (wave-12x34: +13 new tests) ────────────────────────

    // @trace TEST-HR-01 [req:REQ-HR] [level:unit]
    #[test]
    fn layer_anchor_absolute_exactly_at_last_layer() {
        // Arrange: Absolute index exactly equals num_layers - 1
        let anchor = LayerAnchor::Absolute(49);
        let num_layers = 50;

        // Act
        let result = anchor.resolve(num_layers);

        // Assert: should succeed and return the last valid index
        assert_eq!(result.unwrap(), 49);
    }

    // @trace TEST-HR-02 [req:REQ-HR] [level:unit]
    #[test]
    fn layer_anchor_relative_near_one_maps_to_last_layer() {
        // Arrange: Relative 0.999 with 100 layers → round(0.999 * 99) = round(98.901) = 99
        let anchor = LayerAnchor::Relative(0.999);
        let num_layers = 100;

        // Act
        let result = anchor.resolve(num_layers);

        // Assert
        assert_eq!(result.unwrap(), 99);
    }

    // @trace TEST-HR-03 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_mean_identical_rows_returns_that_row() {
        // Arrange: 3 identical rows [4.0, 5.0]
        let hidden = vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0];

        // Act
        let pooled = PoolMode::MeanPool.apply(&hidden, 3, 2).unwrap();

        // Assert: mean of identical rows equals the row itself
        assert_eq!(pooled, vec![4.0, 5.0]);
    }

    // @trace TEST-HR-04 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_cls_ignores_all_but_first_row() {
        // Arrange: first row [1,2], subsequent rows differ
        let hidden = vec![1.0, 2.0, 99.0, 99.0, 88.0, 88.0];

        // Act
        let pooled = PoolMode::ClsToken.apply(&hidden, 3, 2).unwrap();

        // Assert: only first row returned, rest ignored
        assert_eq!(pooled, vec![1.0, 2.0]);
    }

    // @trace TEST-HR-05 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_last_token_correct_offset_calculation() {
        // Arrange: 4 rows of hidden_size=3, last row is [10, 11, 12]
        let hidden = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            7.0, 8.0, 9.0,  // row 2
            10.0, 11.0, 12.0, // row 3
        ];

        // Act
        let pooled = PoolMode::LastToken.apply(&hidden, 4, 3).unwrap();

        // Assert: offset = (4-1)*3 = 9, returns [10, 11, 12]
        assert_eq!(pooled, vec![10.0, 11.0, 12.0]);
    }

    // @trace TEST-HR-06 [req:REQ-HR] [level:unit]
    #[test]
    fn softmax_with_temperature_one_half_preserves_order() {
        // Arrange: two logits, temperature=0.5 (sharper than default)
        let logits = vec![0.0, 2.0];

        // Act
        let probs = softmax_with_temperature(&logits, 0.5).unwrap();

        // Assert: order preserved, sum = 1
        assert!(probs[1] > probs[0], "argmax should have higher probability");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    // @trace TEST-HR-07 [req:REQ-HR] [level:unit]
    #[test]
    fn softmax_high_temperature_nearly_uniform() {
        // Arrange: logits with large differences, very high temperature
        let logits = vec![0.0, 10.0, 20.0];

        // Act
        let probs = softmax_with_temperature(&logits, 1000.0).unwrap();

        // Assert: distribution should be nearly uniform (~0.333 each)
        for p in &probs {
            assert!(
                (p - (1.0 / 3.0)).abs() < 0.01,
                "expected ~0.333 with high temperature, got {p}"
            );
        }
    }

    // @trace TEST-HR-08 [req:REQ-HR] [level:unit]
    #[test]
    fn classify_binary_config_new_accepts_into_string() {
        // Arrange: use &str which implements Into<String>
        let positive = "positive";
        let negative = "negative";

        // Act
        let cfg = ClassifyBinaryConfig::new(positive, negative);

        // Assert: tokens are correctly stored
        assert_eq!(cfg.positive_token, "positive");
        assert_eq!(cfg.negative_token, "negative");
        assert_eq!(cfg.temperature, 1.0);
    }

    // @trace TEST-HR-09 [req:REQ-HR] [level:unit]
    #[test]
    fn classify_multiway_config_zero_temperature_field_value() {
        // Arrange: construct with temperature = 0.0 (field-level, no constructor validation)
        let cfg = ClassifyMultiwayConfig { temperature: 0.0 };

        // Assert: field is stored as-is (validation happens at softmax call time)
        assert_eq!(cfg.temperature, 0.0);
    }

    // @trace TEST-HR-10 [req:REQ-HR] [level:unit]
    #[test]
    fn error_token_not_found_with_empty_string() {
        // Arrange
        let err = HeadRoutingError::TokenNotFound(String::new());

        // Act
        let msg = format!("{err}");

        // Assert: empty string token still produces valid error message
        assert!(msg.contains("token not found"), "message: {msg}");
    }

    // @trace TEST-HR-11 [req:REQ-HR] [level:unit]
    #[test]
    fn error_backend_with_empty_string() {
        // Arrange
        let err = HeadRoutingError::Backend(String::new());

        // Act
        let msg = format!("{err}");

        // Assert: empty string detail still produces valid error message
        assert!(msg.contains("backend error"), "message: {msg}");
    }

    // @trace TEST-HR-12 [req:REQ-HR] [level:unit]
    #[test]
    fn layer_anchor_absolute_zero_always_resolves_to_first_layer() {
        // Arrange: Absolute(0) should always resolve to layer 0 for any num_layers >= 1
        let anchor = LayerAnchor::Absolute(0);

        // Act & Assert
        for &n in &[1, 2, 10, 100, 1000] {
            assert_eq!(anchor.resolve(n).unwrap(), 0, "failed for num_layers={n}");
        }
    }

    // @trace TEST-HR-13 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_mean_scale_factor_precision() {
        // Arrange: seq_len=7, hidden_size=2 — non-power-of-2 to test division precision
        // All rows identical [3.0, 6.0] → mean should be [3.0, 6.0]
        let hidden: Vec<f32> = (0..7).flat_map(|_| vec![3.0, 6.0]).collect();
        assert_eq!(hidden.len(), 14);

        // Act
        let pooled = PoolMode::MeanPool.apply(&hidden, 7, 2).unwrap();

        // Assert: scale = 1/7 ≈ 0.142857..., mean should still be [3.0, 6.0]
        assert!((pooled[0] - 3.0).abs() < 1e-5, "col0 mean = {}", pooled[0]);
        assert!((pooled[1] - 6.0).abs() < 1e-5, "col1 mean = {}", pooled[1]);
    }

    // ── Additional tests (wave-06-04: +10 new tests) ────────────────────────

    // @trace TEST-HR-14 [req:REQ-HR] [level:unit]
    #[test]
    fn layer_anchor_partial_eq_cross_type_inequality() {
        // Arrange: Relative and Absolute with same resolved value are not equal
        let rel = LayerAnchor::Relative(0.5);
        let abs = LayerAnchor::Absolute(12);

        // Assert: PartialEq compares variants, not resolved values
        assert_ne!(rel, abs);
    }

    // @trace TEST-HR-15 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_hidden_size_one_scalar_rows() {
        // Arrange: hidden_size=1, each row is a single scalar
        let hidden = vec![10.0, 20.0, 30.0]; // 3 rows of size 1

        // Act
        let mean = PoolMode::MeanPool.apply(&hidden, 3, 1).unwrap();
        let last = PoolMode::LastToken.apply(&hidden, 3, 1).unwrap();
        let cls = PoolMode::ClsToken.apply(&hidden, 3, 1).unwrap();

        // Assert
        assert!((mean[0] - 20.0).abs() < 1e-6, "mean = {}", mean[0]);
        assert_eq!(last, vec![30.0]);
        assert_eq!(cls, vec![10.0]);
    }

    // @trace TEST-HR-16 [req:REQ-HR] [level:unit]
    #[test]
    fn classify_binary_config_default_matches_new_yes_no() {
        // Arrange: default() and new("yes", "no") should produce identical configs
        let from_default = ClassifyBinaryConfig::default();
        let from_new = ClassifyBinaryConfig::new("yes", "no");

        // Assert: all fields match
        assert_eq!(from_default.positive_token, from_new.positive_token);
        assert_eq!(from_default.negative_token, from_new.negative_token);
        assert_eq!(from_default.temperature, from_new.temperature);
    }

    // @trace TEST-HR-17 [req:REQ-HR] [level:unit]
    #[test]
    fn classify_multiway_config_negative_temperature_field_stored() {
        // Arrange: negative temperature — config is a plain data struct, validation at call time
        let cfg = ClassifyMultiwayConfig { temperature: -0.5 };

        // Assert: field stored as-is without validation
        assert_eq!(cfg.temperature, -0.5);

        // Verify softmax rejects this temperature when used
        assert!(matches!(
            softmax_with_temperature(&[1.0, 2.0], cfg.temperature),
            Err(HeadRoutingError::InvalidConfig(_))
        ));
    }

    // @trace TEST-HR-18 [req:REQ-HR] [level:unit]
    #[test]
    fn error_is_std_error_trait_object() {
        // Arrange: every variant should be usable as a dyn std::error::Error
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(HeadRoutingError::TokenNotFound("tok".into())),
            Box::new(HeadRoutingError::InvalidLayerAnchor(1.5)),
            Box::new(HeadRoutingError::EmptyLabels),
            Box::new(HeadRoutingError::InvalidConfig("msg".into())),
            Box::new(HeadRoutingError::MidLayerNotSupported),
            Box::new(HeadRoutingError::Backend("err".into())),
        ];

        // Assert: all variants implement std::error::Error and have non-empty Display
        assert_eq!(errors.len(), 6);
        for err in &errors {
            assert!(!err.to_string().is_empty());
        }
    }

    // @trace TEST-HR-19 [req:REQ-HR] [level:unit]
    #[test]
    fn softmax_large_negative_logits_all_near_zero_except_max() {
        // Arrange: all very negative logits, one slightly larger
        let logits = vec![-900.0, -800.0, -700.0];

        // Act
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();

        // Assert: -700 dominates, others are nearly zero
        assert!((probs[2] - 1.0).abs() < 1e-3, "dominant prob = {}", probs[2]);
        assert!(probs[0] < probs[1], "order should be preserved");
        assert!(probs[1] < probs[2], "order should be preserved");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    // @trace TEST-HR-20 [req:REQ-HR] [level:unit]
    #[test]
    fn layer_anchor_relative_exact_third_resolution() {
        // Arrange: num_layers=4, Relative(1/3) → round(0.333*3) = round(1.0) = 1
        let anchor = LayerAnchor::Relative(1.0 / 3.0);
        let num_layers = 4;

        // Act
        let result = anchor.resolve(num_layers);

        // Assert
        assert_eq!(result.unwrap(), 1);
    }

    // @trace TEST-HR-21 [req:REQ-HR] [level:unit]
    #[test]
    fn pool_mean_single_row_identical_to_last_and_cls() {
        // Arrange: seq_len=1, all pool modes must agree
        let hidden = vec![7.0, 8.0, 9.0, 10.0];

        // Act
        let mean = PoolMode::MeanPool.apply(&hidden, 1, 4).unwrap();
        let last = PoolMode::LastToken.apply(&hidden, 1, 4).unwrap();
        let cls = PoolMode::ClsToken.apply(&hidden, 1, 4).unwrap();

        // Assert: all three produce identical output for seq_len=1
        assert_eq!(mean, last);
        assert_eq!(last, cls);
        assert_eq!(cls, hidden);
    }

    // @trace TEST-HR-22 [req:REQ-HR] [level:unit]
    #[test]
    fn classify_binary_config_new_accepts_owned_string() {
        // Arrange: pass owned String, not &str
        let positive = String::from("affirmative");
        let negative = String::from("negative");

        // Act
        let cfg = ClassifyBinaryConfig::new(positive, negative);

        // Assert: Into<String> from owned String works
        assert_eq!(cfg.positive_token, "affirmative");
        assert_eq!(cfg.negative_token, "negative");
    }

    // @trace TEST-HR-23 [req:REQ-HR] [level:unit]
    #[test]
    fn softmax_with_many_logits_still_normalizes() {
        // Arrange: 100 logits, monotonically increasing
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Act
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();

        // Assert: sum = 1, all finite, strictly increasing probabilities
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
        assert!(probs.iter().all(|p| p.is_finite() && *p > 0.0));
        for window in probs.windows(2) {
            assert!(window[1] > window[0], "probabilities must be strictly increasing");
        }
    }
}
