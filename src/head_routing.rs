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
//! | 改变 | 架构 YAML / 权重 / JIT 产物 | 仅后处理 |
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

    /// encode_to_layer 当前不支持 mid-layer 截断(FusedGraphExecutor 扩展 pending)。
    ///
    /// **非 stub/fallback**: 这是**显式拒绝**,而非静默降级。一旦
    /// FusedGraphExecutor 扩展 `run_with_early_exit(anchor_layer)` 落地,
    /// `encode_to_layer` 将走真实 JIT 路径,该错误变体永不返回。
    #[error("MidLayerNotSupported: FusedGraphExecutor single-forward path does not yet expose mid-layer exit via CallbackChain; see SPEC/HEAD-ROUTING.md §5.3")]
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
}
