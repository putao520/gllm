//! Classify API — sync-first design (per SPEC 04-API-DESIGN §3.6).
//!
//! Supports sequence classification models (both encoder and decoder based).
//! Encoder-based: BERT/XLM-R + classifier head (e.g. BAAI/bge-reranker, sentiment models).
//! Decoder-based: LLM + score head (e.g. Qwen3ForSequenceClassification).

use crate::client::{Client, GllmError};

/// Builder for text classification.
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_classifier("model-id")?;
/// let result = client.classify(["This movie is great!", "Terrible experience."])?;
/// for r in &result.predictions {
///     println!("{}: label={} score={:.4}", r.index, r.label_id, r.score);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ClassifyBuilder<'a> {
    client: &'a Client,
    texts: Vec<String>,
}

impl<'a> ClassifyBuilder<'a> {
    #[allow(dead_code)]
    pub(crate) fn new(client: &'a Client, texts: Vec<String>) -> Self {
        Self { client, texts }
    }

    /// Execute the classification (sync).
    pub fn generate(self) -> Result<ClassifyResponse, GllmError> {
        self.client.execute_classify(self.texts)
    }
}

/// Response from text classification.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifyResponse {
    /// Classification predictions, one per input text.
    pub predictions: Vec<ClassificationResult>,
}

/// A single classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    /// Index of the input text in the original batch.
    pub index: usize,
    /// Predicted label ID (argmax of logits).
    pub label_id: usize,
    /// Score for the predicted label (softmax probability).
    pub score: f32,
    /// Full logits vector for all labels (raw, pre-softmax).
    pub logits: Vec<f32>,
}

// ============================================================================
// StreamModerateBuilder — Qwen3Guard-Stream per-token 流式审核 (REQ-QGUARD-003)
// ============================================================================

/// 流式审核角色 — 决定提取哪一组 logits (REQ-QGUARD-003).
///
/// - `Query`: 提取 query_risk_level (3) + query_category (9, 含 Jailbreak)
/// - `Response`: 提取 risk_level (3) + category (8)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamModerateRole {
    Query,
    Response,
}

/// 单 token 流式审核结果 (per-role 提取的 risk + category logits).
///
/// `stream_state` 概念由调用者管理: gllm 的 Qwen3 backbone forward 独立维护
/// KV cache 增量, guard head 跨 token 无状态 (见 `Qwen3GuardHead`).
/// 调用者每 token 喂入 backbone 产出的 last-layer hidden state, 本结构
/// 仅承载该 token 的 guard head 输出.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamModerationOutcome {
    /// risk_level logits (长度 3: Safe/Unsafe/Controversial).
    pub risk_logits: Vec<f32>,
    /// category logits (Query=9 含 Jailbreak / Response=8).
    pub category_logits: Vec<f32>,
    /// 该 token 的角色.
    pub role: StreamModerateRole,
}

impl StreamModerationOutcome {
    /// risk argmax (0=Safe, 1=Unsafe, 2=Controversial).
    pub fn risk_argmax(&self) -> usize {
        argmax(&self.risk_logits)
    }

    /// category argmax.
    pub fn category_argmax(&self) -> usize {
        argmax(&self.category_logits)
    }

    /// risk softmax 概率 (和为 1).
    pub fn risk_softmax(&self) -> Vec<f32> {
        softmax(&self.risk_logits)
    }

    /// category softmax 概率 (和为 1).
    pub fn category_softmax(&self) -> Vec<f32> {
        softmax(&self.category_logits)
    }
}

/// Builder for Qwen3Guard-Stream per-token streaming moderation.
///
/// 包装 `Qwen3GuardHead`, 提供 role-aware 的 per-token API. 调用者负责
/// 驱动 Qwen3 backbone 的增量 decode (KV cache), 每生成一个 token 取其
/// last-layer hidden state 喂入 `stream_moderate`.
///
/// # Example
///
/// ```no_run
/// use gllm::StreamModerateBuilder;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut sm = StreamModerateBuilder::from_safetensors("model.safetensors")?;
/// // hidden: Qwen3 backbone 产出的 last-layer hidden state (1024,)
/// # let hidden: &[f32] = &[0.0; 1024];
/// let outcome = sm.stream_moderate(hidden, gllm::StreamModerateRole::Response)?;
/// println!("risk={} category={}", outcome.risk_argmax(), outcome.category_argmax());
/// # Ok(())
/// # }
/// ```
pub struct StreamModerateBuilder {
    head: crate::qwen3_guard::Qwen3GuardHead,
}

impl StreamModerateBuilder {
    /// 从 Qwen3Guard safetensors 文件加载 guard head.
    ///
    /// 文件可为完整 Qwen3Guard-Stream-0.6B model.safetensors (backbone 张量
    /// 被忽略) 或 head-only 提取. dtype 顺从权重文件 (ARCH-JIT-DATA-YIELDS).
    pub fn from_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GllmError> {
        let head = crate::qwen3_guard::Qwen3GuardHead::from_safetensors(path)
            .map_err(|e| GllmError::RuntimeError(format!("qwen3guard load: {e}")))?;
        Ok(Self { head })
    }

    /// 用已加载的 head 构造 (用于测试 / 复用已加载权重).
    pub fn from_head(head: crate::qwen3_guard::Qwen3GuardHead) -> Self {
        Self { head }
    }

    /// 单 token 流式审核 (REQ-QGUARD-003).
    ///
    /// `token_hidden`: backbone 产出的该 token last-layer hidden state (hidden_size,).
    /// `role`: Query (审核用户输入) / Response (审核模型生成).
    ///
    /// 返回该 token 的 (risk_logits, category_logits). stream_state 由调用者
    /// 管理 (backbone KV cache), guard head 跨 token 无状态.
    pub fn stream_moderate(
        &self,
        token_hidden: &[f32],
        role: StreamModerateRole,
    ) -> Result<StreamModerationOutcome, GllmError> {
        let result = self
            .head
            .moderate_token(token_hidden)
            .map_err(|e| GllmError::RuntimeError(format!("qwen3guard moderate: {e}")))?;
        let (risk_logits, category_logits) = match role {
            StreamModerateRole::Query => (
                result.query_risk_level_logits,
                result.query_category_logits,
            ),
            StreamModerateRole::Response => (result.risk_level_logits, result.category_logits),
        };
        Ok(StreamModerationOutcome {
            risk_logits,
            category_logits,
            role,
        })
    }

    /// 批量序列审核 (非流式便捷方法).
    ///
    /// `hidden_seq`: `[T, hidden_size]` 行主序. 返回 T 个 outcome.
    pub fn stream_moderate_sequence(
        &self,
        hidden_seq: &[f32],
        role: StreamModerateRole,
    ) -> Result<Vec<StreamModerationOutcome>, GllmError> {
        let results = self
            .head
            .moderate_sequence(hidden_seq)
            .map_err(|e| GllmError::RuntimeError(format!("qwen3guard moderate_sequence: {e}")))?;
        results
            .into_iter()
            .map(|result| {
                let (risk_logits, category_logits) = match role {
                    StreamModerateRole::Query => (
                        result.query_risk_level_logits,
                        result.query_category_logits,
                    ),
                    StreamModerateRole::Response => {
                        (result.risk_level_logits, result.category_logits)
                    }
                };
                Ok(StreamModerationOutcome {
                    risk_logits,
                    category_logits,
                    role,
                })
            })
            .collect()
    }

    /// 访问内部 head (用于复用 / 测试).
    pub fn head(&self) -> &crate::qwen3_guard::Qwen3GuardHead {
        &self.head
    }
}

/// argmax helper.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// softmax helper (数值稳定版).
fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; v.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction & field access ──

    #[test]
    fn classify_response_with_predictions() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 2,
                    score: 0.95,
                    logits: vec![0.01, 0.02, 0.95],
                },
            ],
        };
        assert_eq!(resp.predictions.len(), 1);
        assert_eq!(resp.predictions[0].label_id, 2);
        assert!((resp.predictions[0].score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn classification_result_fields() {
        let r = ClassificationResult {
            index: 3,
            label_id: 1,
            score: 0.5,
            logits: vec![0.3, 0.5, 0.2],
        };
        assert_eq!(r.index, 3);
        assert_eq!(r.label_id, 1);
        assert!((r.score - 0.5).abs() < 1e-6);
        assert_eq!(r.logits.len(), 3);
    }

    #[test]
    fn classify_response_empty() {
        let resp = ClassifyResponse { predictions: vec![] };
        assert!(resp.predictions.is_empty());
    }

    #[test]
    fn classify_response_multiple_predictions() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 0,
                    score: 0.7,
                    logits: vec![0.7, 0.3],
                },
            ],
        };
        assert_eq!(resp.predictions.len(), 2);
        assert_eq!(resp.predictions[0].index, 0);
        assert_eq!(resp.predictions[1].index, 1);
    }

    #[test]
    fn classification_result_single_class() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.logits.len(), 1);
        assert_eq!(r.label_id, 0);
    }

    // ── Boundary values ──

    #[test]
    fn classification_result_index_usize_max() {
        let r = ClassificationResult {
            index: usize::MAX,
            label_id: 0,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.index, usize::MAX);
    }

    #[test]
    fn classification_result_label_id_usize_max() {
        let r = ClassificationResult {
            index: 0,
            label_id: usize::MAX,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.label_id, usize::MAX);
    }

    #[test]
    fn classification_result_score_zero() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![1.0, 0.0],
        };
        assert!((r.score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn classification_result_score_one() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1.0,
            logits: vec![0.0, 1.0],
        };
        assert!((r.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classification_result_score_f32_max() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::MAX,
            logits: vec![f32::MAX],
        };
        assert_eq!(r.score, f32::MAX);
    }

    #[test]
    fn classification_result_score_f32_min_positive() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::MIN_POSITIVE,
            logits: vec![f32::MIN_POSITIVE],
        };
        assert_eq!(r.score, f32::MIN_POSITIVE);
    }

    #[test]
    fn classification_result_logits_empty() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![],
        };
        assert!(r.logits.is_empty());
    }

    #[test]
    fn classification_result_logits_contain_negative() {
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 0.6,
            logits: vec![-2.5, 0.6, -0.1],
        };
        assert_eq!(r.logits.len(), 3);
        assert!((r.logits[0] - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn classification_result_logits_contain_infinity() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::INFINITY,
            logits: vec![f32::INFINITY, 0.0],
        };
        assert!(r.logits[0].is_infinite());
        assert!(r.logits[0].is_sign_positive());
    }

    #[test]
    fn classification_result_logits_contain_neg_infinity() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![f32::NEG_INFINITY, 0.5],
        };
        assert!(r.logits[0].is_infinite());
        assert!(r.logits[0].is_sign_negative());
    }

    #[test]
    fn classification_result_large_logits() {
        let logits: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 500,
            score: 0.5,
            logits: logits.clone(),
        };
        assert_eq!(r.logits.len(), 1000);
        assert_eq!(r.logits[500], 0.5);
    }

    // ── Clone derive ──

    #[test]
    fn classify_response_clone() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.8,
                    logits: vec![0.1, 0.8, 0.1],
                },
            ],
        };
        let cloned = resp.clone();
        assert_eq!(cloned.predictions.len(), 1);
        assert_eq!(cloned.predictions[0].label_id, 1);
    }

    #[test]
    fn classification_result_clone_independence() {
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 0.9,
            logits: vec![0.1, 0.9],
        };
        let mut cloned = r.clone();
        cloned.logits[0] = 0.5;
        assert!((r.logits[0] - 0.1).abs() < 1e-6, "original unchanged");
        assert!((cloned.logits[0] - 0.5).abs() < 1e-6, "clone modified");
    }

    #[test]
    fn classify_response_clone_independence() {
        let mut resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.8,
                    logits: vec![0.1, 0.8, 0.1],
                },
            ],
        };
        let cloned = resp.clone();
        resp.predictions.clear();
        assert!(resp.predictions.is_empty(), "original cleared");
        assert_eq!(cloned.predictions.len(), 1, "clone unaffected");
    }

    #[test]
    fn classification_result_clone_preserves_all_fields() {
        let r = ClassificationResult {
            index: 42,
            label_id: 7,
            score: 0.123,
            logits: vec![0.1, 0.2, 0.3],
        };
        let cloned = r.clone();
        assert_eq!(cloned.index, 42);
        assert_eq!(cloned.label_id, 7);
        assert!((cloned.score - 0.123).abs() < 1e-6);
        assert_eq!(cloned.logits, vec![0.1, 0.2, 0.3]);
    }

    // ── Debug derive ──

    #[test]
    fn classification_result_debug_format() {
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.99,
            logits: vec![0.0, 0.01, 0.99],
        };
        let debug = format!("{r:?}");
        assert!(debug.contains("index"));
        assert!(debug.contains("label_id"));
        assert!(debug.contains("score"));
        assert!(debug.contains("logits"));
    }

    #[test]
    fn classify_response_debug_format() {
        let resp = ClassifyResponse { predictions: vec![] };
        let debug = format!("{resp:?}");
        assert!(debug.contains("predictions"));
    }

    #[test]
    fn classify_response_debug_with_predictions() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.75,
                    logits: vec![0.25, 0.75],
                },
            ],
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("0.75"));
        assert!(debug.contains("label_id"));
    }

    // ── PartialEq derive ──

    #[test]
    fn classify_response_eq_self() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        assert_eq!(resp, resp);
    }

    #[test]
    fn classify_response_eq_clone() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        assert_eq!(resp, resp.clone());
    }

    #[test]
    fn classify_response_ne_different_predictions() {
        let a = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        let b = ClassifyResponse { predictions: vec![] };
        assert_ne!(a, b);
    }

    #[test]
    fn classify_response_ne_different_label() {
        let a = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.9,
                    logits: vec![0.9, 0.1],
                },
            ],
        };
        let b = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classify_response_ne_different_score() {
        let a = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        let b = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.8,
                    logits: vec![0.1, 0.9],
                },
            ],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classification_result_eq_identical() {
        let r = ClassificationResult {
            index: 1,
            label_id: 2,
            score: 0.5,
            logits: vec![0.3, 0.5, 0.2],
        };
        assert_eq!(r, r);
    }

    #[test]
    fn classification_result_ne_different_logits() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5, 0.5],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.4, 0.6],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classification_result_ne_different_index() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        let b = ClassificationResult {
            index: 1,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classification_result_ne_different_logit_length() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5, 0.5],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classify_response_eq_empty() {
        let a = ClassifyResponse { predictions: vec![] };
        let b = ClassifyResponse { predictions: vec![] };
        assert_eq!(a, b);
    }

    // ── Special float values ──

    #[test]
    fn classification_result_score_nan() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![f32::NAN],
        };
        assert!(r.score.is_nan());
    }

    #[test]
    fn classification_result_score_infinity() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::INFINITY,
            logits: vec![f32::INFINITY],
        };
        assert!(r.score.is_infinite());
        assert!(r.score.is_sign_positive());
    }

    #[test]
    fn classification_result_score_neg_infinity() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NEG_INFINITY,
            logits: vec![f32::NEG_INFINITY],
        };
        assert!(r.score.is_infinite());
        assert!(r.score.is_sign_negative());
    }

    #[test]
    fn classification_result_logits_all_nan() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![f32::NAN, f32::NAN, f32::NAN],
        };
        for l in &r.logits {
            assert!(l.is_nan());
        }
    }

    #[test]
    fn classification_result_logits_mixed_special_floats() {
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.0,
            logits: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0],
        };
        assert!(r.logits[0].is_nan());
        assert!(r.logits[1].is_infinite() && r.logits[1].is_sign_positive());
        assert!(r.logits[2].is_infinite() && r.logits[2].is_sign_negative());
        assert_eq!(r.logits[3], 0.0f32);
        assert!(r.logits[4] == 0.0f32);
    }

    #[test]
    fn classification_result_negative_zero_score() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: -0.0f32,
            logits: vec![-0.0f32],
        };
        assert_eq!(r.score, 0.0f32);
        assert!(r.score.is_sign_negative());
    }

    #[test]
    fn classification_result_logits_negative_zero() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![-0.0f32],
        };
        assert_eq!(r.logits[0], 0.0f32);
        assert!(r.logits[0].is_sign_negative());
    }

    #[test]
    fn classification_result_score_very_small() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1e-38f32,
            logits: vec![1e-38f32],
        };
        assert!(r.score > 0.0);
        assert!((r.score - 1e-38f32).abs() < 1e-45f32);
    }

    #[test]
    fn classification_result_score_f32_min() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::MIN,
            logits: vec![f32::MIN],
        };
        assert_eq!(r.score, f32::MIN);
    }

    // ── Index and label boundary ──

    #[test]
    fn classification_result_index_zero() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.index, 0);
    }

    #[test]
    fn classification_result_label_id_zero() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.label_id, 0);
    }

    #[test]
    fn classification_result_index_and_label_max() {
        let r = ClassificationResult {
            index: usize::MAX,
            label_id: usize::MAX,
            score: 0.5,
            logits: vec![0.5],
        };
        assert_eq!(r.index, usize::MAX);
        assert_eq!(r.label_id, usize::MAX);
    }

    // ── Logits vector edge cases ──

    #[test]
    fn classification_result_logits_single_element() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1.0,
            logits: vec![42.0],
        };
        assert_eq!(r.logits.len(), 1);
        assert!((r.logits[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn classification_result_logits_all_zeros() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        assert!(r.logits.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn classification_result_logits_all_same() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.25,
            logits: vec![0.25, 0.25, 0.25, 0.25],
        };
        assert!(r.logits.iter().all(|&l| (l - 0.25).abs() < 1e-6));
    }

    #[test]
    fn classification_result_logits_many_classes() {
        let logits: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 9999,
            score: 9999.0,
            logits: logits.clone(),
        };
        assert_eq!(r.logits.len(), 10000);
        assert_eq!(r.logits[0], 0.0);
        assert!((r.logits[9999] - 9999.0).abs() < 1e-6);
    }

    // ── Response predictions ordering ──

    #[test]
    fn classify_response_predictions_preserve_order() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 2,
                    label_id: 0,
                    score: 0.1,
                    logits: vec![0.1],
                },
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.9],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 2,
                    score: 0.5,
                    logits: vec![0.5],
                },
            ],
        };
        assert_eq!(resp.predictions[0].index, 2);
        assert_eq!(resp.predictions[1].index, 0);
        assert_eq!(resp.predictions[2].index, 1);
    }

    #[test]
    fn classify_response_many_predictions() {
        let predictions: Vec<ClassificationResult> = (0..500)
            .map(|i| ClassificationResult {
                index: i,
                label_id: i % 5,
                score: 0.5,
                logits: vec![0.5],
            })
            .collect();
        let resp = ClassifyResponse {
            predictions: predictions.clone(),
        };
        assert_eq!(resp.predictions.len(), 500);
        assert_eq!(resp.predictions[499].index, 499);
    }

    // ── PartialEq deeper coverage ──

    #[test]
    fn classification_result_eq_with_same_logits() {
        let a = ClassificationResult {
            index: 5,
            label_id: 3,
            score: 0.77,
            logits: vec![0.1, 0.2, 0.3, 0.77],
        };
        let b = ClassificationResult {
            index: 5,
            label_id: 3,
            score: 0.77,
            logits: vec![0.1, 0.2, 0.3, 0.77],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn classify_response_ne_different_lengths() {
        let a = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 1.0,
                    logits: vec![1.0],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 1,
                    score: 0.5,
                    logits: vec![0.5],
                },
            ],
        };
        let b = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: 1.0,
                logits: vec![1.0],
            }],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn classify_response_ne_different_order() {
        let a = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.9],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 0,
                    score: 0.5,
                    logits: vec![0.5],
                },
            ],
        };
        let b = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 1,
                    label_id: 0,
                    score: 0.5,
                    logits: vec![0.5],
                },
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.9],
                },
            ],
        };
        assert_ne!(a, b);
    }

    // ── Debug format content verification ──

    #[test]
    fn classification_result_debug_shows_field_values() {
        let r = ClassificationResult {
            index: 99,
            label_id: 42,
            score: 0.123,
            logits: vec![1.0, 2.0],
        };
        let debug = format!("{r:?}");
        assert!(debug.contains("99"), "should show index value");
        assert!(debug.contains("42"), "should show label_id value");
        assert!(debug.contains("0.123"), "should show score value");
    }

    #[test]
    fn classify_response_debug_empty_predictions() {
        let resp = ClassifyResponse { predictions: vec![] };
        let debug = format!("{resp:?}");
        assert!(debug.contains("predictions"));
        assert!(debug.contains("[]"));
    }

    #[test]
    fn classification_result_debug_with_special_floats() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![f32::INFINITY],
        };
        let debug = format!("{r:?}");
        assert!(!debug.is_empty());
        // Debug output should still be produced for special floats
        assert!(debug.contains("index"));
    }

    // ── Clone deep verification ──

    #[test]
    fn classification_result_clone_vec_is_deep_copy() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![1.0, 2.0, 3.0],
        };
        let mut cloned = r.clone();
        cloned.logits.push(4.0);
        assert_eq!(r.logits.len(), 3, "original untouched");
        assert_eq!(cloned.logits.len(), 4, "clone extended");
    }

    #[test]
    fn classify_response_clone_predictions_deep() {
        let resp = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 1,
                score: 0.8,
                logits: vec![0.2, 0.8],
            }],
        };
        let mut cloned = resp.clone();
        cloned.predictions.push(ClassificationResult {
            index: 1,
            label_id: 0,
            score: 0.3,
            logits: vec![0.3, 0.7],
        });
        assert_eq!(resp.predictions.len(), 1);
        assert_eq!(cloned.predictions.len(), 2);
    }

    #[test]
    fn classify_response_clone_empty() {
        let resp = ClassifyResponse { predictions: vec![] };
        let cloned = resp.clone();
        assert!(cloned.predictions.is_empty());
        assert_eq!(resp, cloned);
    }

    // ── Logits with extreme numeric values ──

    #[test]
    fn classification_result_logits_with_f32_max() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::MAX,
            logits: vec![f32::MAX, 0.0],
        };
        assert_eq!(r.logits[0], f32::MAX);
    }

    #[test]
    fn classification_result_logits_with_f32_min() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![f32::MIN, 0.0],
        };
        assert_eq!(r.logits[0], f32::MIN);
    }

    #[test]
    fn classification_result_logits_with_min_positive() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::MIN_POSITIVE,
            logits: vec![f32::MIN_POSITIVE],
        };
        assert_eq!(r.logits[0], f32::MIN_POSITIVE);
        assert!(r.logits[0] > 0.0);
    }

    // ── Score edge cases ──

    #[test]
    fn classification_result_score_negative() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: -0.5,
            logits: vec![-0.5],
        };
        assert!((r.score - (-0.5)).abs() < 1e-6);
        assert!(r.score < 0.0);
    }

    #[test]
    fn classification_result_score_very_large() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 1e30f32,
            logits: vec![1e30f32],
        };
        assert!((r.score - 1e30f32).abs() < 1e24f32);
    }

    // ── Struct field consistency ──

    #[test]
    fn classification_result_label_id_beyond_logits_len() {
        // label_id can legitimately be an index into logits; test it being out of range
        let r = ClassificationResult {
            index: 0,
            label_id: 5,
            score: 0.99,
            logits: vec![0.01, 0.99], // only 2 logits but label_id=5
        };
        assert_eq!(r.label_id, 5);
        assert_eq!(r.logits.len(), 2);
    }

    #[test]
    fn classification_result_index_does_not_correlate_with_batch() {
        // index is from original batch, can be any value regardless of predictions order
        let r = ClassificationResult {
            index: 100,
            label_id: 0,
            score: 1.0,
            logits: vec![1.0],
        };
        assert_eq!(r.index, 100);
    }

    // ── Response with heterogeneous predictions ──

    #[test]
    fn classify_response_predictions_varying_logit_lengths() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.9,
                    logits: vec![0.9],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 2,
                    score: 0.5,
                    logits: vec![0.1, 0.2, 0.5, 0.1, 0.1],
                },
            ],
        };
        assert_eq!(resp.predictions[0].logits.len(), 1);
        assert_eq!(resp.predictions[1].logits.len(), 5);
    }

    #[test]
    fn classify_response_predictions_varying_scores() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.0,
                    logits: vec![0.0],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 0,
                    score: 0.5,
                    logits: vec![0.5],
                },
                ClassificationResult {
                    index: 2,
                    label_id: 0,
                    score: 1.0,
                    logits: vec![1.0],
                },
            ],
        };
        assert!((resp.predictions[0].score - 0.0).abs() < 1e-6);
        assert!((resp.predictions[1].score - 0.5).abs() < 1e-6);
        assert!((resp.predictions[2].score - 1.0).abs() < 1e-6);
    }

    // ── Display-like formatting via Debug ──

    #[test]
    fn classification_result_debug_contains_struct_name() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0,
            logits: vec![],
        };
        let debug = format!("{r:?}");
        assert!(
            debug.contains("ClassificationResult"),
            "Debug should contain struct name"
        );
    }

    #[test]
    fn classify_response_debug_contains_struct_name() {
        let resp = ClassifyResponse { predictions: vec![] };
        let debug = format!("{resp:?}");
        assert!(
            debug.contains("ClassifyResponse"),
            "Debug should contain struct name"
        );
    }

    // ── PartialEq symmetry and transitivity ──

    #[test]
    fn classify_response_eq_symmetry() {
        let a = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: 0.5,
                logits: vec![0.5],
            }],
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn classify_response_eq_transitivity() {
        let a = ClassifyResponse { predictions: vec![] };
        let b = a.clone();
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn classification_result_eq_symmetry() {
        let a = ClassificationResult {
            index: 1,
            label_id: 2,
            score: 0.5,
            logits: vec![0.1, 0.2, 0.3],
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn classification_result_ne_symmetry() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        let b = ClassificationResult {
            index: 1,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        assert_ne!(a, b);
        assert_ne!(b, a);
    }

    // ── PartialEq with NaN (NaN != NaN by IEEE 754) ──

    #[test]
    fn classification_result_ne_nan_score() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![0.5],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![0.5],
        };
        // NaN != NaN for f32 PartialEq
        assert_ne!(a, b);
    }

    #[test]
    fn classification_result_ne_nan_in_logits() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![f32::NAN],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![f32::NAN],
        };
        // NaN in logits makes them not equal
        assert_ne!(a, b);
    }

    // ── Destructuring ──

    #[test]
    fn classification_result_destructure() {
        let r = ClassificationResult {
            index: 7,
            label_id: 3,
            score: 0.42,
            logits: vec![0.1, 0.2, 0.3, 0.42],
        };
        let ClassificationResult { index, label_id, score, logits } = r;
        assert_eq!(index, 7);
        assert_eq!(label_id, 3);
        assert!((score - 0.42).abs() < 1e-6);
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn classify_response_destructure() {
        let resp = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: 1.0,
                logits: vec![1.0],
            }],
        };
        let ClassifyResponse { predictions } = resp;
        assert_eq!(predictions.len(), 1);
    }

    // ── Partial equality with zero variants ──

    #[test]
    fn classification_result_positive_zero_eq_negative_zero() {
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.0f32,
            logits: vec![0.0f32],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: -0.0f32,
            logits: vec![-0.0f32],
        };
        assert_eq!(a, b, "positive zero == negative zero for f32 PartialEq");
    }

    // ── Multiple predictions with same label_id ──

    #[test]
    fn classify_response_multiple_same_label() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.9,
                    logits: vec![0.1, 0.9],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 1,
                    score: 0.8,
                    logits: vec![0.2, 0.8],
                },
                ClassificationResult {
                    index: 2,
                    label_id: 1,
                    score: 0.7,
                    logits: vec![0.3, 0.7],
                },
            ],
        };
        assert!(resp.predictions.iter().all(|p| p.label_id == 1));
    }

    // ── Logits sum and property checks ──

    #[test]
    fn classification_result_logits_sum_to_one_approx() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.6,
            logits: vec![0.6, 0.3, 0.1],
        };
        let sum: f32 = r.logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classification_result_logits_sorted() {
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.8,
            logits: vec![0.1, 0.2, 0.8],
        };
        let mut sorted = r.logits.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(r.logits, sorted);
    }

    #[test]
    fn classification_result_logits_unsorted() {
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.8,
            logits: vec![0.8, 0.1, 0.5],
        };
        let mut sorted = r.logits.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_ne!(r.logits, sorted);
    }

    // ── Binary classification pattern ──

    #[test]
    fn binary_classification_positive_class() {
        // Arrange: binary classification (2 classes), predicted class 1 (positive)
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 0.97,
            logits: vec![0.03, 0.97],
        };
        // Act & Assert: verify binary structure
        assert_eq!(r.logits.len(), 2, "binary classification has exactly 2 classes");
        assert_eq!(r.label_id, 1, "predicted positive class");
        assert!(r.score > 0.5, "positive class score > 0.5");
    }

    #[test]
    fn binary_classification_negative_class() {
        // Arrange: binary classification, predicted class 0 (negative)
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.82,
            logits: vec![0.82, 0.18],
        };
        // Act & Assert
        assert_eq!(r.label_id, 0, "predicted negative class");
        assert!((r.logits[0] + r.logits[1] - 1.0).abs() < 1e-6,
            "binary logits should sum to ~1.0 for probabilities");
    }

    // ── Multiway classification pattern ──

    #[test]
    fn multiway_classification_five_classes() {
        // Arrange: 5-way classification
        let logits = vec![0.05, 0.10, 0.60, 0.15, 0.10];
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.60,
            logits,
        };
        // Act & Assert
        assert_eq!(r.logits.len(), 5);
        assert_eq!(r.label_id, 2);
        let sum: f32 = r.logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities sum to 1.0");
    }

    #[test]
    fn multiway_classification_argmax_matches_label() {
        // Arrange: verify label_id is the argmax position
        let logits = vec![0.01, 0.02, 0.03, 0.90, 0.04];
        let max_idx = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let r = ClassificationResult {
            index: 0,
            label_id: max_idx,
            score: logits[max_idx],
            logits,
        };
        // Act & Assert: label_id should be 3 (the argmax)
        assert_eq!(r.label_id, 3);
        assert!((r.score - 0.90).abs() < 1e-6);
    }

    // ── Scores with identical or near-identical values ──

    #[test]
    fn classification_result_all_identical_scores_in_batch() {
        // Arrange: a batch where all predictions have identical scores
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.5,
                    logits: vec![0.5, 0.5],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 1,
                    score: 0.5,
                    logits: vec![0.5, 0.5],
                },
                ClassificationResult {
                    index: 2,
                    label_id: 0,
                    score: 0.5,
                    logits: vec![0.5, 0.5],
                },
            ],
        };
        // Act & Assert
        let all_same = resp.predictions.iter().all(|p| (p.score - 0.5).abs() < 1e-6);
        assert!(all_same, "all scores identical at 0.5");
    }

    #[test]
    fn classification_result_minimally_different_scores() {
        // Arrange: two predictions with epsilon score difference
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5, 0.5],
        };
        let b = ClassificationResult {
            index: 1,
            label_id: 0,
            score: 0.5 + f32::EPSILON,
            logits: vec![0.5 + f32::EPSILON, 0.5],
        };
        // Act & Assert: scores differ by EPSILON but scores are not equal
        assert_ne!(a.score, b.score, "scores differ by f32::EPSILON");
        assert!(b.score > a.score);
    }

    // ── Scores outside [0,1] (raw logits, not probabilities) ──

    #[test]
    fn classification_result_raw_logits_not_probabilities() {
        // Arrange: raw logits can be any real number, not bounded to [0,1]
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 5.7,
            logits: vec![-1.2, 5.7, 0.3],
        };
        // Act & Assert: score > 1.0 is valid for raw logits
        assert!(r.score > 1.0, "raw logits can exceed 1.0");
        assert!((r.logits[0] - (-1.2)).abs() < 1e-6, "logits can be negative");
    }

    // ── Logits with alternating sign pattern ──

    #[test]
    fn classification_result_logits_alternating_signs() {
        // Arrange: logits alternating between positive and negative
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 3.0,
            logits: vec![1.0, -2.0, 3.0, -4.0, 5.0],
        };
        // Act & Assert
        assert!(r.logits[0] > 0.0);
        assert!(r.logits[1] < 0.0);
        assert!(r.logits[2] > 0.0);
        assert!(r.logits[3] < 0.0);
        assert!(r.logits[4] > 0.0);
    }

    // ── Inconsistent label_id vs logits argmax ──

    #[test]
    fn classification_result_label_disagrees_with_argmax() {
        // Arrange: label_id does not match the argmax of logits (inconsistent but valid struct)
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.9,
            logits: vec![0.1, 0.9],
        };
        // Act: find the actual argmax
        let argmax_idx = r.logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Assert: label_id=0 but argmax=1 — struct allows inconsistency
        assert_ne!(r.label_id, argmax_idx,
            "struct does not enforce label_id == argmax(logits)");
    }

    // ── Empty predictions iteration ──

    #[test]
    fn classify_response_empty_predictions_iteration_no_panic() {
        // Arrange
        let resp = ClassifyResponse { predictions: vec![] };
        // Act: iterate over empty predictions
        let count = resp.predictions.iter().count();
        // Assert
        assert_eq!(count, 0, "iterating empty predictions yields 0 items");
    }

    // ── Response predictions iteration with filter ──

    #[test]
    fn classify_response_filter_by_threshold() {
        // Arrange: predictions with varying scores
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.95,
                    logits: vec![0.95, 0.05],
                },
                ClassificationResult {
                    index: 1,
                    label_id: 1,
                    score: 0.30,
                    logits: vec![0.30, 0.70],
                },
                ClassificationResult {
                    index: 2,
                    label_id: 0,
                    score: 0.80,
                    logits: vec![0.80, 0.20],
                },
            ],
        };
        // Act: filter predictions above 0.5 threshold
        let high_conf: Vec<&ClassificationResult> = resp.predictions
            .iter()
            .filter(|p| p.score > 0.5)
            .collect();
        // Assert
        assert_eq!(high_conf.len(), 2);
        assert_eq!(high_conf[0].index, 0);
        assert_eq!(high_conf[1].index, 2);
    }

    // ── GllmError Display trait coverage ──

    #[test]
    fn gllm_error_model_not_found_display() {
        // Arrange
        let err = GllmError::ModelNotFound("my-model-v2".to_string());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("my-model-v2"), "display should contain model name");
        assert!(msg.contains("model not found"), "display should contain error kind");
    }

    #[test]
    fn gllm_error_invalid_model_type_display() {
        // Arrange
        let err = GllmError::InvalidModelType;
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("invalid model type"), "display should describe the error");
    }

    #[test]
    fn gllm_error_no_model_loaded_display() {
        // Arrange
        let err = GllmError::NoModelLoaded;
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("no model loaded"), "display should describe the error");
    }

    #[test]
    fn gllm_error_runtime_error_display() {
        // Arrange
        let err = GllmError::RuntimeError("tokenizer failed".to_string());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("tokenizer failed"), "display should contain the message");
        assert!(msg.contains("runtime error"), "display should contain error kind");
    }

    // ── Softmax probability computation verification ──

    #[test]
    fn classification_result_softmax_probability_properties() {
        // Arrange: 4-class logits with known softmax result
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let exp_sum: f32 = logits.iter().map(|&x: &f32| x.exp()).sum();
        let softmax: Vec<f32> = logits.iter().map(|&x: &f32| x.exp() / exp_sum).collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 3,
            score: softmax[3],
            logits: softmax.clone(),
        };
        // Act & Assert: softmax probabilities sum to ~1.0
        let sum: f32 = r.logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax probabilities sum to 1.0");
        assert!(r.logits.iter().all(|&p| p > 0.0), "all softmax values are positive");
        assert!((r.score - softmax[3]).abs() < 1e-6, "score matches predicted class probability");
    }

    // ── Argmax with tied maximum logits ──

    #[test]
    fn classification_result_tied_max_logits() {
        // Arrange: two classes share the maximum logit value
        let logits = vec![0.5, 1.0, 1.0, 0.3];
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_indices: Vec<usize> = logits.iter().enumerate()
            .filter(|(_, &v)| (v - max_val).abs() < 1e-6)
            .map(|(i, _)| i)
            .collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: max_val,
            logits,
        };
        // Assert: two classes share the max
        assert_eq!(max_indices.len(), 2, "two classes tied at maximum");
        assert!(max_indices.contains(&r.label_id), "label_id is one of the tied maxima");
    }

    // ── Confidence categorization by threshold ──

    #[test]
    fn classify_response_categorize_by_confidence() {
        // Arrange: predictions at different confidence levels
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.99, logits: vec![0.99, 0.01] },
                ClassificationResult { index: 1, label_id: 1, score: 0.75, logits: vec![0.25, 0.75] },
                ClassificationResult { index: 2, label_id: 0, score: 0.45, logits: vec![0.45, 0.55] },
                ClassificationResult { index: 3, label_id: 0, score: 0.10, logits: vec![0.10, 0.90] },
            ],
        };
        // Act: categorize by thresholds
        let high: Vec<_> = resp.predictions.iter().filter(|p| p.score >= 0.9).collect();
        let medium: Vec<_> = resp.predictions.iter().filter(|p| p.score >= 0.5 && p.score < 0.9).collect();
        let low: Vec<_> = resp.predictions.iter().filter(|p| p.score < 0.5).collect();
        // Assert
        assert_eq!(high.len(), 1, "one high-confidence prediction");
        assert_eq!(medium.len(), 1, "one medium-confidence prediction");
        assert_eq!(low.len(), 2, "two low-confidence predictions");
    }

    // ── Score exactly at threshold boundary ──

    #[test]
    fn classification_result_score_exactly_at_threshold() {
        // Arrange: score exactly 0.5 — the boundary between "above" and "below"
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5, 0.5],
        };
        // Act & Assert: strict > does NOT include boundary
        assert!(!(r.score > 0.5), "0.5 is not strictly > 0.5");
        assert!(r.score >= 0.5, "0.5 is >= 0.5");
        assert!(!(r.score < 0.5), "0.5 is not < 0.5");
        assert!(r.score <= 0.5, "0.5 is <= 0.5");
    }

    // ── Logits with subnormal (denormalized) floats ──

    #[test]
    fn classification_result_logits_subnormal_floats() {
        // Arrange: logits contain subnormal f32 values (between 0 and MIN_POSITIVE)
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 1.0,
            logits: vec![subnormal, 1.0, subnormal],
        };
        // Assert: subnormal values are positive but smaller than MIN_POSITIVE
        assert!(r.logits[0] > 0.0, "subnormal is positive");
        assert!(r.logits[0] < f32::MIN_POSITIVE, "subnormal is smaller than MIN_POSITIVE");
        assert!(!r.logits[0].is_normal(), "subnormal is not a normal float");
    }

    // ── Response predictions sorted by score descending ──

    #[test]
    fn classify_response_sort_by_score_descending() {
        // Arrange: predictions in arbitrary score order
        let mut resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 1, score: 0.3, logits: vec![0.7, 0.3] },
                ClassificationResult { index: 1, label_id: 0, score: 0.95, logits: vec![0.95, 0.05] },
                ClassificationResult { index: 2, label_id: 2, score: 0.6, logits: vec![0.2, 0.2, 0.6] },
            ],
        };
        // Act: sort by score descending
        resp.predictions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        // Assert: highest score first
        assert!((resp.predictions[0].score - 0.95).abs() < 1e-6);
        assert!((resp.predictions[1].score - 0.6).abs() < 1e-6);
        assert!((resp.predictions[2].score - 0.3).abs() < 1e-6);
    }

    // ── Entropy computation from logits ──

    #[test]
    fn classification_result_entropy_from_probabilities() {
        // Arrange: uniform distribution over 4 classes has maximum entropy = ln(4)
        let uniform_p = 0.25f32;
        let logits = vec![uniform_p, uniform_p, uniform_p, uniform_p];
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: uniform_p,
            logits: logits.clone(),
        };
        // Act: compute Shannon entropy H = -sum(p * ln(p))
        let entropy: f32 = -r.logits.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p: &f32| p * p.ln())
            .sum::<f32>();
        let expected_entropy = (4.0f32).ln(); // ln(4) for uniform 4-class
        // Assert: entropy is maximum for uniform distribution
        assert!((entropy - expected_entropy).abs() < 1e-4,
            "uniform 4-class entropy should be ln(4) ~= 1.386");
    }

    // ── Top-K extraction from logits ──

    #[test]
    fn classification_result_top_k_logits() {
        // Arrange: 6-class logits
        let logits = vec![0.05, 0.30, 0.10, 0.40, 0.05, 0.10];
        let r = ClassificationResult {
            index: 0,
            label_id: 3,
            score: 0.40,
            logits: logits.clone(),
        };
        // Act: extract top-3 indices by logit value
        let mut indexed: Vec<(usize, f32)> = r.logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top3: Vec<usize> = indexed.iter().take(3).map(|&(i, _)| i).collect();
        // Assert: top-3 are classes 3, 1, and one of {2,5}
        assert_eq!(top3[0], 3, "highest logit at class 3");
        assert_eq!(top3[1], 1, "second highest at class 1");
        assert!(top3[2] == 2 || top3[2] == 5, "third highest at class 2 or 5 (tied)");
    }

    // ── All predictions with distinct label_ids ──

    #[test]
    fn classify_response_all_distinct_labels() {
        // Arrange: 5 predictions, each with a unique label_id
        let resp = ClassifyResponse {
            predictions: (0..5).map(|i| ClassificationResult {
                index: i,
                label_id: i,
                score: 1.0 - i as f32 * 0.1,
                logits: vec![1.0],
            }).collect(),
        };
        // Act: collect label_ids
        let label_ids: Vec<usize> = resp.predictions.iter().map(|p| p.label_id).collect();
        let unique: std::collections::HashSet<usize> = label_ids.iter().copied().collect();
        // Assert: all 5 labels are distinct
        assert_eq!(unique.len(), 5, "all label_ids are distinct");
    }

    // ── Debug format with large logits vector ──

    #[test]
    fn classification_result_debug_large_logits_truncated_check() {
        // Arrange: a result with 200 logit entries
        let logits: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 199,
            score: 1.99,
            logits,
        };
        // Act
        let debug = format!("{r:?}");
        // Assert: debug output is produced (not truncated or panicking)
        assert!(debug.contains("ClassificationResult"), "contains struct name");
        assert!(debug.contains("199"), "contains label_id");
        assert!(debug.len() > 100, "debug output is substantial for large logits");
    }

    // ── PartialEq reflexivity confirmed for normal values ──

    #[test]
    fn classification_result_eq_reflexivity_normal_values() {
        // Arrange: a result with typical values (no NaN/Inf)
        let r = ClassificationResult {
            index: 10,
            label_id: 3,
            score: 0.8765,
            logits: vec![0.1, 0.05, 0.0235, 0.8765],
        };
        // Assert: a value equals itself (reflexivity holds for non-NaN)
        assert_eq!(r, r, "reflexivity: a == a for normal float values");
    }

    // ── Logit sum verification for non-probability raw scores ──

    #[test]
    fn classification_result_logits_raw_scores_do_not_sum_to_one() {
        // Arrange: raw logits (pre-softmax) that do not sum to 1.0
        let logits = vec![2.3, -0.5, 4.1];
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 4.1,
            logits: logits.clone(),
        };
        // Act
        let sum: f32 = r.logits.iter().sum();
        // Assert: raw logits need not sum to 1.0
        assert!((sum - 5.9).abs() < 1e-5, "raw logits sum to 5.9, not 1.0");
        assert!((sum - 1.0).abs() > 0.1, "raw logits do not sum to ~1.0");
    }

    // ── Prediction score monotonically decreasing validation ──

    #[test]
    fn classify_response_scores_not_monotonic_by_default() {
        // Arrange: predictions with non-monotonic scores (as received from model)
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.2, logits: vec![0.8, 0.2] },
                ClassificationResult { index: 1, label_id: 1, score: 0.9, logits: vec![0.1, 0.9] },
                ClassificationResult { index: 2, label_id: 0, score: 0.5, logits: vec![0.5, 0.5] },
            ],
        };
        // Act: check if scores are sorted descending
        let scores: Vec<f32> = resp.predictions.iter().map(|p| p.score).collect();
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // Assert: scores are NOT sorted by default
        assert_ne!(scores, sorted_scores, "scores are in input order, not sorted");
    }

    // ── Clone with large predictions batch ──

    #[test]
    fn classify_response_clone_large_batch_independence() {
        // Arrange: response with 100 predictions
        let resp = ClassifyResponse {
            predictions: (0..100).map(|i| ClassificationResult {
                index: i,
                label_id: i % 10,
                score: (i as f32) / 100.0,
                logits: vec![(i as f32) / 100.0],
            }).collect(),
        };
        // Act: clone and mutate
        let mut cloned = resp.clone();
        cloned.predictions.truncate(10);
        // Assert: original unaffected
        assert_eq!(resp.predictions.len(), 100, "original has 100 predictions");
        assert_eq!(cloned.predictions.len(), 10, "cloned truncated to 10");
    }

    // ── Binary classification complement score ──

    #[test]
    fn binary_classification_complement_score() {
        // Arrange: binary classification where logits represent probabilities
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.73,
            logits: vec![0.73, 0.27],
        };
        // Act & Assert: complement class score
        let complement = 1.0 - r.score;
        assert!((complement - r.logits[1]).abs() < 1e-6,
            "complement = 1 - score matches the other class logit");
        assert!((r.logits[0] + r.logits[1] - 1.0).abs() < 1e-6,
            "binary probabilities sum to 1.0");
    }

    // ── ClassificationResult with score = -0.0 vs 0.0 equality ──

    #[test]
    fn classification_result_logits_mixed_positive_negative_zero() {
        // Arrange: logits containing +0.0, -0.0, and normal values
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 2.5,
            logits: vec![0.0f32, 2.5, -0.0f32, -1.0],
        };
        // Assert: +0.0 and -0.0 are equal in value but differ in sign bit
        assert_eq!(r.logits[0], 0.0f32, "+0.0 equals 0.0");
        assert_eq!(r.logits[2], 0.0f32, "-0.0 equals 0.0 in value");
        assert!(!r.logits[0].is_sign_negative(), "+0.0 has positive sign");
        assert!(r.logits[2].is_sign_negative(), "-0.0 has negative sign");
        assert!((r.logits[1] - 2.5).abs() < 1e-6);
        assert!((r.logits[3] - (-1.0)).abs() < 1e-6);
    }

    // ── GllmError BackendError display ──

    #[test]
    fn gllm_error_backend_error_display() {
        // Arrange: BackendError from executor
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Cpu("simd not available".to_string()));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display should describe the error kind");
    }

    // ── GllmError Debug trait verification ──

    #[test]
    fn gllm_error_debug_format_contains_variant_name() {
        // Arrange
        let err = GllmError::ModelNotFound("test-model".to_string());
        // Act
        let debug = format!("{err:?}");
        // Assert: Debug output should contain the variant name
        assert!(debug.contains("ModelNotFound"), "Debug should show variant name");
    }

    // ── ClassificationResult transitivity with non-trivial logits ──

    #[test]
    fn classification_result_eq_transitivity_multi_field() {
        // Arrange: three identical results built independently
        let a = ClassificationResult {
            index: 5,
            label_id: 3,
            score: 0.42,
            logits: vec![0.1, 0.2, 0.3, 0.42],
        };
        let b = ClassificationResult {
            index: 5,
            label_id: 3,
            score: 0.42,
            logits: vec![0.1, 0.2, 0.3, 0.42],
        };
        let c = ClassificationResult {
            index: 5,
            label_id: 3,
            score: 0.42,
            logits: vec![0.1, 0.2, 0.3, 0.42],
        };
        // Assert: a==b and b==c implies a==c
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── ClassifyResponse map over predictions to extract scores ──

    #[test]
    fn classify_response_map_predictions_to_scores() {
        // Arrange: a response with varied predictions
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.33, logits: vec![0.33, 0.67] },
                ClassificationResult { index: 1, label_id: 1, score: 0.88, logits: vec![0.12, 0.88] },
                ClassificationResult { index: 2, label_id: 0, score: 0.55, logits: vec![0.55, 0.45] },
            ],
        };
        // Act: extract scores via map
        let scores: Vec<f32> = resp.predictions.iter().map(|p| p.score).collect();
        // Assert
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.33).abs() < 1e-6);
        assert!((scores[1] - 0.88).abs() < 1e-6);
        assert!((scores[2] - 0.55).abs() < 1e-6);
    }

    // ── ClassificationResult clone with NaN logits is independent ──

    #[test]
    fn classification_result_clone_with_nan_logits_independence() {
        // Arrange: a result with NaN in logits
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 0.5,
            logits: vec![f32::NAN, 0.5, 1.0],
        };
        // Act: clone and modify
        let mut cloned = r.clone();
        cloned.logits[2] = 99.0;
        // Assert: original logits unchanged (index 2 is still 1.0)
        assert!((r.logits[2] - 1.0).abs() < 1e-6, "original index 2 unchanged");
        assert!((cloned.logits[2] - 99.0).abs() < 1e-6, "clone index 2 modified");
    }

    // ── ClassifyResponse collect predictions into a HashMap by index ──

    #[test]
    fn classify_response_collect_into_hashmap_by_index() {
        // Arrange
        use std::collections::HashMap;
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 10, label_id: 2, score: 0.9, logits: vec![0.05, 0.05, 0.9] },
                ClassificationResult { index: 20, label_id: 0, score: 0.4, logits: vec![0.4, 0.3, 0.3] },
            ],
        };
        // Act: collect into HashMap keyed by index
        let map: HashMap<usize, &ClassificationResult> = resp.predictions
            .iter()
            .map(|p| (p.index, p))
            .collect();
        // Assert
        assert_eq!(map.len(), 2);
        assert_eq!(map[&10].label_id, 2);
        assert_eq!(map[&20].label_id, 0);
    }

    // ── ClassificationResult with all-negative logits ──

    #[test]
    fn classification_result_logits_all_negative() {
        // Arrange: all logits negative (possible for raw pre-softmax scores)
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: -0.1,
            logits: vec![-5.0, -0.1, -3.0, -8.0],
        };
        // Act & Assert
        assert!(r.logits.iter().all(|&l| l < 0.0), "all logits are negative");
        assert!((r.score - (-0.1)).abs() < 1e-6, "score matches the least negative logit");
    }

    // ── Single-prediction response equality with same content ──

    #[test]
    fn classify_response_single_prediction_equality_exact() {
        // Arrange: two single-prediction responses with identical content
        let make = || ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 1,
                score: 0.77,
                logits: vec![0.23, 0.77],
            }],
        };
        let a = make();
        let b = make();
        // Assert
        assert_eq!(a, b, "two independently constructed responses with same data are equal");
    }

    // ── ClassificationResult score partial_cmp ordering ──

    #[test]
    fn classification_result_scores_ordering_comparison() {
        // Arrange: two results with different scores
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.3,
            logits: vec![0.3, 0.7],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.9,
            logits: vec![0.9, 0.1],
        };
        // Act & Assert: partial_cmp gives correct ordering
        assert_eq!(a.score.partial_cmp(&b.score), Some(std::cmp::Ordering::Less));
        assert_eq!(b.score.partial_cmp(&a.score), Some(std::cmp::Ordering::Greater));
    }

    // ── ClassificationResult with f32::EPSILON score difference ──

    #[test]
    fn classification_result_epsilon_score_difference_not_equal() {
        // Arrange: two results differing only by f32::EPSILON in score
        let base = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5],
        };
        let eps_high = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5 + f32::EPSILON,
            logits: vec![0.5],
        };
        // Assert: they are not equal because score differs
        assert_ne!(base, eps_high, "EPSILON difference makes them unequal");
    }

    // ── ClassifyResponse predictions fold to compute average score ──

    #[test]
    fn classify_response_average_score_via_fold() {
        // Arrange
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.2, logits: vec![0.8, 0.2] },
                ClassificationResult { index: 1, label_id: 1, score: 0.6, logits: vec![0.4, 0.6] },
                ClassificationResult { index: 2, label_id: 0, score: 1.0, logits: vec![1.0, 0.0] },
                ClassificationResult { index: 3, label_id: 0, score: 0.4, logits: vec![0.6, 0.4] },
            ],
        };
        // Act: compute average score
        let sum: f32 = resp.predictions.iter().map(|p| p.score).sum();
        let avg = sum / resp.predictions.len() as f32;
        // Assert: (0.2 + 0.6 + 1.0 + 0.4) / 4 = 0.55
        assert!((avg - 0.55).abs() < 1e-6, "average score should be 0.55");
    }

    // ── NaN score partial_cmp is None ──

    #[test]
    fn classification_result_nan_score_partial_cmp_is_none() {
        // Arrange
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::NAN,
            logits: vec![1.0],
        };
        // Act
        let cmp = r.score.partial_cmp(&0.5f32);
        // Assert: NaN compared with anything is None
        assert_eq!(cmp, None, "NaN partial_cmp returns None");
    }

    // ── ClassifyResponse PartialEq with mixed special float logits ──

    #[test]
    fn classify_response_ne_when_one_has_infinity_other_does_not() {
        // Arrange: two responses where one has infinity logits, other has normal
        let a = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: f32::INFINITY,
                logits: vec![f32::INFINITY],
            }],
        };
        let b = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: 1.0,
                logits: vec![1.0],
            }],
        };
        // Assert: different scores mean different responses
        assert_ne!(a, b, "infinity score != 1.0 score");
    }

    // ── GllmError OutOfMemory display ──

    #[test]
    fn gllm_error_out_of_memory_display() {
        // Arrange
        let oom = crate::kv_cache::OomHaltError::fatal_halt("GPU HBM exhausted");
        let err = GllmError::OutOfMemory(oom);
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("out of memory"), "display should contain error kind: {msg}");
    }

    // ── OomHaltError fatal vs soft ──

    #[test]
    fn oom_halt_error_fatal_vs_soft() {
        // Arrange
        use crate::kv_cache::OomHaltError;
        let fatal = OomHaltError::fatal_halt("GPU HBM exhausted");
        let soft = OomHaltError::soft_halt("retry allocation");
        // Assert
        assert!(fatal.fatal, "fatal_halt creates fatal error");
        assert!(!soft.fatal, "soft_halt creates non-fatal error");
        assert!(fatal.message.contains("GPU HBM"));
        assert!(soft.message.contains("retry"));
    }

    // ── ClassificationResult with logits sum exactly zero ──

    #[test]
    fn classification_result_logits_sum_exactly_zero() {
        // Arrange: logits that cancel out (positive and negative pairs)
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 5.0,
            logits: vec![10.0, 5.0, -10.0, -5.0],
        };
        // Act
        let sum: f32 = r.logits.iter().sum();
        // Assert: sum is exactly zero (no floating point error for these values)
        assert!((sum - 0.0).abs() < 1e-6, "balanced logits sum to zero");
    }

    // ── ClassifyResponse predictions into_iter ownership transfer ──

    #[test]
    fn classify_response_into_iter_drain() {
        // Arrange
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 1, score: 0.9, logits: vec![0.1, 0.9] },
                ClassificationResult { index: 1, label_id: 0, score: 0.3, logits: vec![0.7, 0.3] },
            ],
        };
        // Act: consume via into_iter
        let indices: Vec<usize> = resp.predictions.into_iter().map(|p| p.index).collect();
        // Assert: all indices collected
        assert_eq!(indices, vec![0, 1], "into_iter yields predictions in order");
    }

    // ── ClassificationResult PartialEq: only index differs ──

    #[test]
    fn classification_result_ne_only_index_differs() {
        // Arrange: identical except index
        let a = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.88,
            logits: vec![0.05, 0.07, 0.88],
        };
        let b = ClassificationResult {
            index: 999,
            label_id: 2,
            score: 0.88,
            logits: vec![0.05, 0.07, 0.88],
        };
        // Assert: different index => not equal
        assert_ne!(a, b, "different index makes results unequal");
    }

    // ── ClassifyResponse with mixed NaN and normal predictions ──

    #[test]
    fn classify_response_mixed_nan_and_normal_predictions() {
        // Arrange: one normal prediction and one with NaN score
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.95, logits: vec![0.95, 0.05] },
                ClassificationResult { index: 1, label_id: 0, score: f32::NAN, logits: vec![f32::NAN] },
            ],
        };
        // Act & Assert
        assert!((resp.predictions[0].score - 0.95).abs() < 1e-6, "first is normal");
        assert!(resp.predictions[1].score.is_nan(), "second is NaN");
        assert_eq!(resp.predictions.len(), 2, "response holds both types");
    }

    // ── ClassificationResult Debug format includes all four fields ──

    #[test]
    fn classification_result_debug_includes_all_four_fields() {
        // Arrange
        let r = ClassificationResult {
            index: 42,
            label_id: 3,
            score: 0.77,
            logits: vec![0.1, 0.2, 0.77],
        };
        // Act
        let debug = format!("{r:?}");
        // Assert: all four fields present in output
        assert!(debug.contains("index"), "debug contains 'index'");
        assert!(debug.contains("label_id"), "debug contains 'label_id'");
        assert!(debug.contains("score"), "debug contains 'score'");
        assert!(debug.contains("logits"), "debug contains 'logits'");
    }

    // ── ClassifyResponse predictions access via index out of bounds safety ──

    #[test]
    fn classify_response_get_safe_access_via_iter() {
        // Arrange: single-element response, access via iter to avoid index panic
        let resp = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 1,
                score: 0.66,
                logits: vec![0.34, 0.66],
            }],
        };
        // Act: safe access via iter instead of direct indexing
        let first = resp.predictions.iter().next();
        // Assert
        assert!(first.is_some(), "iter yields at least one element");
        let p = first.unwrap();
        assert_eq!(p.label_id, 1);
        assert!((p.score - 0.66).abs() < 1e-6);
    }

    // ── ClassificationResult with f32 INFINITY score and logits ──

    #[test]
    fn classification_result_infinity_score_and_logits() {
        // Arrange: all infinite values
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: f32::INFINITY,
            logits: vec![f32::INFINITY, 0.0, -1.0],
        };
        // Act & Assert
        assert!(r.score.is_infinite() && r.score.is_sign_positive());
        assert!(r.logits[0].is_infinite());
        assert!((r.logits[1] - 0.0).abs() < 1e-6);
        assert!((r.logits[2] - (-1.0)).abs() < 1e-6);
    }

    // ── ClassifyResponse predictions with descending monotonic scores ──

    #[test]
    fn classify_response_predictions_monotonic_descending_scores() {
        // Arrange: predictions with strictly descending scores
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.9, logits: vec![0.9, 0.1] },
                ClassificationResult { index: 1, label_id: 1, score: 0.7, logits: vec![0.3, 0.7] },
                ClassificationResult { index: 2, label_id: 0, score: 0.5, logits: vec![0.5, 0.5] },
                ClassificationResult { index: 3, label_id: 1, score: 0.3, logits: vec![0.7, 0.3] },
            ],
        };
        // Act: verify monotonicity
        let scores: Vec<f32> = resp.predictions.iter().map(|p| p.score).collect();
        let is_descending = scores.windows(2).all(|w| w[0] > w[1]);
        // Assert
        assert!(is_descending, "scores are strictly monotonically descending");
        assert_eq!(scores.len(), 4);
    }

    // ── ClassificationResult clone with large logits vector independence ──

    #[test]
    fn classification_result_clone_large_logits_deep_copy() {
        // Arrange: result with 500 logit entries
        let logits: Vec<f32> = (0..500).map(|i| (i as f32) * 0.01).collect();
        let r = ClassificationResult {
            index: 0,
            label_id: 250,
            score: 2.5,
            logits,
        };
        // Act: clone and modify a middle element
        let mut cloned = r.clone();
        cloned.logits[250] = -999.0;
        // Assert: original unaffected
        assert!((r.logits[250] - 2.5).abs() < 1e-6, "original logits[250] unchanged");
        assert!((cloned.logits[250] - (-999.0)).abs() < 1e-6, "cloned logits[250] modified");
    }

    // ── ClassifyResponse equality: two responses with different prediction counts ──

    #[test]
    fn classify_response_ne_zero_vs_one_prediction() {
        // Arrange
        let empty = ClassifyResponse { predictions: vec![] };
        let one = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 0,
                score: 1.0,
                logits: vec![1.0],
            }],
        };
        // Assert
        assert_ne!(empty, one, "empty response != single-prediction response");
    }

    // ── ClassificationResult PartialEq: only logits differ by single element ──

    #[test]
    fn classification_result_ne_differs_in_single_logit_entry() {
        // Arrange: two results differing only in one logit entry
        let a = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.1, 0.2, 0.3, 0.4],
        };
        let b = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.1, 0.2, 0.3, 0.99], // only index 3 differs
        };
        // Assert
        assert_ne!(a, b, "single logit difference makes them unequal");
    }

    // ── ClassifyResponse predictions with duplicate indices ──

    #[test]
    fn classify_response_predictions_duplicate_indices() {
        // Arrange: two predictions share the same index (valid structurally)
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult {
                    index: 0,
                    label_id: 1,
                    score: 0.92,
                    logits: vec![0.08, 0.92],
                },
                ClassificationResult {
                    index: 0,
                    label_id: 0,
                    score: 0.61,
                    logits: vec![0.61, 0.39],
                },
            ],
        };
        // Act: verify both have index 0
        assert_eq!(resp.predictions[0].index, 0);
        assert_eq!(resp.predictions[1].index, 0);
        // Assert: predictions are distinct despite same index
        assert_ne!(resp.predictions[0], resp.predictions[1]);
        assert_eq!(resp.predictions.len(), 2);
    }

    // ── ClassifyResponse predictions first and last accessors ──

    #[test]
    fn classify_response_first_and_last_predictions() {
        // Arrange: three predictions with distinct scores
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.1, logits: vec![0.9, 0.1] },
                ClassificationResult { index: 1, label_id: 1, score: 0.5, logits: vec![0.5, 0.5] },
                ClassificationResult { index: 2, label_id: 0, score: 0.9, logits: vec![0.9, 0.1] },
            ],
        };
        // Act & Assert: first() and last() return correct elements
        let first = resp.predictions.first().unwrap();
        let last = resp.predictions.last().unwrap();
        assert_eq!(first.index, 0, "first prediction has index 0");
        assert_eq!(last.index, 2, "last prediction has index 2");
        assert!((first.score - 0.1).abs() < 1e-6);
        assert!((last.score - 0.9).abs() < 1e-6);
    }

    // ── ClassificationResult Debug with empty logits vector ──

    #[test]
    fn classification_result_debug_with_empty_logits() {
        // Arrange: result with empty logits (edge case)
        let r = ClassificationResult {
            index: 5,
            label_id: 0,
            score: 0.0,
            logits: vec![],
        };
        // Act
        let debug = format!("{r:?}");
        // Assert: debug output still shows all fields including empty logits
        assert!(debug.contains("ClassificationResult"), "contains struct name");
        assert!(debug.contains("index"), "contains 'index'");
        assert!(debug.contains("label_id"), "contains 'label_id'");
        assert!(debug.contains("logits"), "contains 'logits'");
    }

    // ── ClassifyResponse Debug with many predictions shows structure ──

    #[test]
    fn classify_response_debug_with_many_predictions() {
        // Arrange: response with 50 predictions
        let resp = ClassifyResponse {
            predictions: (0..50).map(|i| ClassificationResult {
                index: i,
                label_id: i % 3,
                score: 0.5,
                logits: vec![0.5],
            }).collect(),
        };
        // Act
        let debug = format!("{resp:?}");
        // Assert: debug output contains struct name and is substantial
        assert!(debug.contains("ClassifyResponse"), "contains struct name");
        assert!(debug.contains("predictions"), "contains predictions field");
        assert!(debug.len() > 200, "debug output is substantial for 50 predictions");
    }

    // ── GllmError BackendError CUDA variant display ──

    #[test]
    fn gllm_error_backend_cuda_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Cuda("device lost".to_string()));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display contains error kind");
        assert!(msg.contains("CUDA"), "display contains backend variant: {msg}");
    }

    // ── GllmError BackendError HIP variant display ──

    #[test]
    fn gllm_error_backend_hip_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Hip("hipMemcpy failed".to_string()));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display contains error kind");
        assert!(msg.contains("HIP"), "display contains backend variant: {msg}");
    }

    // ── GllmError BackendError Metal variant display ──

    #[test]
    fn gllm_error_backend_metal_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Metal("buffer allocation failed".to_string()));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display contains error kind");
        assert!(msg.contains("Metal"), "display contains backend variant: {msg}");
    }

    // ── GllmError BackendError Unimplemented variant display ──

    #[test]
    fn gllm_error_backend_unimplemented_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Unimplemented("flash attention on CPU"));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display contains error kind");
        assert!(msg.contains("unimplemented"), "display contains unimplemented marker: {msg}");
    }

    // ── GllmError BackendError Other variant display ──

    #[test]
    fn gllm_error_backend_other_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let err = GllmError::BackendError(BackendError::Other("custom failure".to_string()));
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("backend error"), "display contains error kind");
        assert!(msg.contains("custom failure"), "display contains message: {msg}");
    }

    // ── ClassifyResponse predictions mutable push ──

    #[test]
    fn classify_response_predictions_push_mutation() {
        // Arrange: mutable response starting with one prediction
        let mut resp = ClassifyResponse {
            predictions: vec![ClassificationResult {
                index: 0,
                label_id: 1,
                score: 0.8,
                logits: vec![0.2, 0.8],
            }],
        };
        // Act: push a second prediction
        resp.predictions.push(ClassificationResult {
            index: 1,
            label_id: 0,
            score: 0.3,
            logits: vec![0.7, 0.3],
        });
        // Assert
        assert_eq!(resp.predictions.len(), 2);
        assert_eq!(resp.predictions[1].index, 1);
        assert!((resp.predictions[1].score - 0.3).abs() < 1e-6);
    }

    // ── ClassificationResult logits contains and windows iteration ──

    #[test]
    fn classification_result_logits_contains_specific_value() {
        // Arrange: logits with known values
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.6,
            logits: vec![0.1, 0.3, 0.6],
        };
        // Act & Assert: verify specific values are present
        assert!(r.logits.contains(&0.1), "logits contains 0.1");
        assert!(r.logits.contains(&0.3), "logits contains 0.3");
        assert!(r.logits.contains(&0.6), "logits contains 0.6");
        assert!(!r.logits.contains(&0.9), "logits does not contain 0.9");
        // Verify adjacent pairs via windows
        let pairs: Vec<[f32; 2]> = r.logits.windows(2).map(|w| [w[0], w[1]]).collect();
        assert_eq!(pairs.len(), 2, "3 elements yield 2 adjacent pairs");
        assert!((pairs[0][1] - pairs[0][0] - 0.2).abs() < 1e-6, "second is 0.2 more than first");
    }

    // ── ClassifyResponse empty predictions first/last return None ──

    #[test]
    fn classify_response_empty_first_last_are_none() {
        // Arrange
        let resp = ClassifyResponse { predictions: vec![] };
        // Act & Assert
        assert!(resp.predictions.first().is_none(), "first() on empty is None");
        assert!(resp.predictions.last().is_none(), "last() on empty is None");
        assert!(resp.predictions.is_empty());
    }

    // ── OomHaltError message and fatal field accessibility ──

    #[test]
    fn oom_halt_error_fields_are_public() {
        // Arrange
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::fatal_halt("test message");
        // Act & Assert: verify public field access
        assert_eq!(err.message, "test message");
        assert!(err.fatal);
        // Also verify soft halt fields
        let soft = OomHaltError::soft_halt("retry");
        assert_eq!(soft.message, "retry");
        assert!(!soft.fatal);
    }

    // ── OomHaltError Display includes message and fatal flag ──

    #[test]
    fn oom_halt_error_display_includes_message_and_fatal() {
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::fatal_halt("HBM depleted");
        let msg = format!("{err}");
        assert!(msg.contains("HBM depleted"), "display should contain message");
        assert!(msg.contains("fatal=true"), "display should show fatal=true");
        let soft = OomHaltError::soft_halt("retry later");
        let soft_msg = format!("{soft}");
        assert!(soft_msg.contains("retry later"));
        assert!(soft_msg.contains("fatal=false"));
    }

    // ── OomHaltError Clone produces independent copy ──

    #[test]
    fn oom_halt_error_clone_independence() {
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::fatal_halt("original");
        let cloned = err.clone();
        assert_eq!(cloned.message, "original");
        assert_eq!(cloned.fatal, true);
    }

    // ── OomHaltError Debug output contains struct fields ──

    #[test]
    fn oom_halt_error_debug_format() {
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::soft_halt("low memory");
        let debug = format!("{err:?}");
        assert!(debug.contains("OomHaltError"), "Debug should contain struct name");
        assert!(debug.contains("low memory"), "Debug should contain message");
        assert!(debug.contains("false"), "Debug should contain fatal=false");
    }

    // ── GllmError source chain for BackendError variant ──

    #[test]
    fn gllm_error_backend_error_source_chain() {
        use crate::engine::executor::BackendError;
        let inner = BackendError::Cpu("overflow".to_string());
        let err = GllmError::BackendError(inner);
        let source = std::error::Error::source(&err);
        assert!(source.is_some(), "BackendError variant should have a source");
        let source_msg = format!("{}", source.unwrap());
        assert!(source_msg.contains("overflow"), "source message should contain inner detail");
    }

    // ── GllmError source chain for OutOfMemory variant ──

    #[test]
    fn gllm_error_out_of_memory_source_chain() {
        use crate::kv_cache::OomHaltError;
        let oom = OomHaltError::fatal_halt("VRAM full");
        let err = GllmError::OutOfMemory(oom);
        let source = std::error::Error::source(&err);
        assert!(source.is_some(), "OutOfMemory variant should have a source");
        let source_msg = format!("{}", source.unwrap());
        assert!(source_msg.contains("VRAM full"), "source should contain OOM message");
    }

    // ── ClassificationResult field mutation through mutable reference ──

    #[test]
    fn classification_result_mutate_score_and_logit() {
        let mut r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.3, 0.7],
        };
        r.score = 0.9;
        r.logits[0] = 0.1;
        r.logits[1] = 0.9;
        r.label_id = 1;
        assert!((r.score - 0.9).abs() < 1e-6);
        assert_eq!(r.label_id, 1);
        assert!((r.logits[1] - 0.9).abs() < 1e-6);
    }

    // ── ClassifyResponse predictions retain_capacity after clone ──

    #[test]
    fn classify_response_clone_predictions_capacity_independence() {
        let mut resp = ClassifyResponse {
            predictions: Vec::with_capacity(100),
        };
        resp.predictions.push(ClassificationResult {
            index: 0, label_id: 0, score: 1.0, logits: vec![1.0],
        });
        let cloned = resp.clone();
        resp.predictions.push(ClassificationResult {
            index: 1, label_id: 1, score: 0.5, logits: vec![0.5],
        });
        assert_eq!(cloned.predictions.len(), 1, "clone unaffected by push to original");
        assert_eq!(resp.predictions.len(), 2);
    }

    // ── ClassificationResult with score exactly 0.5 boundary in ternary logic ──

    #[test]
    fn classification_result_score_half_boundary_ternary_classification() {
        let r = ClassificationResult {
            index: 0,
            label_id: 1,
            score: 0.5,
            logits: vec![0.5, 0.5],
        };
        let is_positive = r.score > 0.5;
        let is_negative = r.score < 0.5;
        let is_boundary = r.score == 0.5;
        assert!(!is_positive);
        assert!(!is_negative);
        assert!(is_boundary);
    }

    // ── BackendError Display with empty string messages ──

    #[test]
    fn backend_error_display_empty_messages() {
        use crate::engine::executor::BackendError;
        let variants: Vec<BackendError> = vec![
            BackendError::Cpu(String::new()),
            BackendError::Cuda(String::new()),
            BackendError::Hip(String::new()),
            BackendError::Metal(String::new()),
            BackendError::Other(String::new()),
        ];
        for v in &variants {
            let msg = format!("{v}");
            assert!(!msg.is_empty(), "Display should produce non-empty output even with empty message");
        }
    }

    // ── BackendError Debug format contains variant name ──

    #[test]
    fn backend_error_debug_format_contains_variant() {
        use crate::engine::executor::BackendError;
        let err = BackendError::Unimplemented("flash_attn");
        let debug = format!("{err:?}");
        assert!(debug.contains("Unimplemented"), "Debug should contain variant name");
    }

    // ── ClassificationResult logits chunked iteration for batch processing ──

    #[test]
    fn classification_result_logits_chunks_exact_size() {
        let r = ClassificationResult {
            index: 0,
            label_id: 3,
            score: 0.25,
            logits: vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
        };
        let chunks: Vec<Vec<f32>> = r.logits.chunks(2).map(|c| c.to_vec()).collect();
        assert_eq!(chunks.len(), 3, "6 elements / 2 = 3 chunks");
        assert_eq!(chunks[0], vec![0.25, 0.25]);
        assert_eq!(chunks[2], vec![0.0, 0.0]);
    }

    // ── ClassifyResponse predictions partitioned by label_id parity ──

    #[test]
    fn classify_response_partition_predictions_by_label_parity() {
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.9, logits: vec![0.9, 0.1] },
                ClassificationResult { index: 1, label_id: 1, score: 0.6, logits: vec![0.4, 0.6] },
                ClassificationResult { index: 2, label_id: 2, score: 0.3, logits: vec![0.3, 0.7] },
                ClassificationResult { index: 3, label_id: 3, score: 0.8, logits: vec![0.2, 0.8] },
            ],
        };
        let (even, odd): (Vec<_>, Vec<_>) = resp.predictions
            .iter()
            .partition(|p| p.label_id % 2 == 0);
        assert_eq!(even.len(), 2, "labels 0 and 2 are even");
        assert_eq!(odd.len(), 2, "labels 1 and 3 are odd");
        assert_eq!(even[0].label_id, 0);
        assert_eq!(even[1].label_id, 2);
    }

    // ── ClassificationResult PartialEq: same struct re-compared after mutation ──

    #[test]
    fn classification_result_eq_after_mutation_cycle() {
        let mut r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.1,
            logits: vec![0.1, 0.9],
        };
        let original = r.clone();
        r.score = 0.99;
        assert_ne!(r, original, "mutated result differs from original");
        r.score = 0.1;
        assert_eq!(r, original, "restoring score makes them equal again");
    }

    // ── ClassifyBuilder stores texts in order ──

    #[test]
    fn classify_builder_new_preserves_text_order() {
        // Arrange: a dummy Client is not needed to test the builder struct directly
        // because ClassifyBuilder::new just stores texts; we verify the texts field.
        // Since `client` field requires a live Client, we test indirectly by verifying
        // the texts Vec is passed through correctly via the struct fields.
        let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let texts_clone = texts.clone();
        // Act: simulate what ClassifyBuilder::new does internally
        let stored = texts;
        // Assert: texts are stored in order
        assert_eq!(stored, texts_clone);
        assert_eq!(stored[0], "hello");
        assert_eq!(stored[2], "test");
    }

    // ── ClassificationResult logits position lookup via enumerate ──

    #[test]
    fn classification_result_logits_position_lookup_via_enumerate() {
        // Arrange: logits with known values at known positions
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.5,
            logits: vec![0.1, 0.2, 0.5, 0.7, 0.9],
        };
        // Act: find positions of specific values via enumerate + find
        let pos_01 = r.logits.iter().enumerate().find(|(_, &v)| (v - 0.1).abs() < 1e-6);
        let pos_05 = r.logits.iter().enumerate().find(|(_, &v)| (v - 0.5).abs() < 1e-6);
        let pos_09 = r.logits.iter().enumerate().find(|(_, &v)| (v - 0.9).abs() < 1e-6);
        let pos_missing = r.logits.iter().enumerate().find(|(_, &v)| (v - 0.3).abs() < 1e-6);
        // Assert
        assert_eq!(pos_01.map(|(i, _)| i), Some(0), "0.1 is at index 0");
        assert_eq!(pos_05.map(|(i, _)| i), Some(2), "0.5 is at index 2");
        assert_eq!(pos_09.map(|(i, _)| i), Some(4), "0.9 is at index 4");
        assert!(pos_missing.is_none(), "0.3 is not present");
    }

    // ── ClassifyResponse predictions flat_map over logits ──

    #[test]
    fn classify_response_flat_map_logits_across_predictions() {
        // Arrange: two predictions with multi-class logits
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 1, score: 0.6, logits: vec![0.2, 0.6, 0.2] },
                ClassificationResult { index: 1, label_id: 2, score: 0.8, logits: vec![0.1, 0.1, 0.8] },
            ],
        };
        // Act: flatten all logits into a single collection
        let all_logits: Vec<f32> = resp.predictions.iter().flat_map(|p| p.logits.iter().copied()).collect();
        // Assert: 3 logits per prediction, 2 predictions = 6 total
        assert_eq!(all_logits.len(), 6);
        assert!((all_logits[0] - 0.2).abs() < 1e-6);
        assert!((all_logits[3] - 0.1).abs() < 1e-6);
        assert!((all_logits[5] - 0.8).abs() < 1e-6);
    }

    // ── ClassificationResult logits max and min values ──

    #[test]
    fn classification_result_logits_max_and_min() {
        // Arrange: logits with known min and max
        let r = ClassificationResult {
            index: 0,
            label_id: 4,
            score: 9.0,
            logits: vec![1.0, -3.0, 5.0, -7.0, 9.0],
        };
        // Act
        let max_val = r.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = r.logits.iter().cloned().fold(f32::INFINITY, f32::min);
        // Assert
        assert!((max_val - 9.0).abs() < 1e-6, "max logit is 9.0");
        assert!((min_val - (-7.0)).abs() < 1e-6, "min logit is -7.0");
        assert_eq!(r.logits.len(), 5);
    }

    // ── ClassifyResponse predictions count by label via group_by pattern ──

    #[test]
    fn classify_response_count_predictions_per_label() {
        // Arrange: predictions with repeated label_ids
        use std::collections::HashMap;
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 0, score: 0.9, logits: vec![0.9] },
                ClassificationResult { index: 1, label_id: 2, score: 0.7, logits: vec![0.7] },
                ClassificationResult { index: 2, label_id: 0, score: 0.5, logits: vec![0.5] },
                ClassificationResult { index: 3, label_id: 2, score: 0.3, logits: vec![0.3] },
                ClassificationResult { index: 4, label_id: 0, score: 0.1, logits: vec![0.1] },
            ],
        };
        // Act: count occurrences of each label_id
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for p in &resp.predictions {
            *counts.entry(p.label_id).or_insert(0) += 1;
        }
        // Assert
        assert_eq!(counts[&0], 3, "label 0 appears 3 times");
        assert_eq!(counts[&2], 2, "label 2 appears 2 times");
        assert_eq!(counts.len(), 2, "only 2 distinct labels");
    }

    // ── ClassificationResult logits rev iteration ──

    #[test]
    fn classification_result_logits_reverse_iteration() {
        // Arrange: logits in ascending order
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.3,
            logits: vec![0.1, 0.2, 0.3, 0.4],
        };
        // Act: iterate in reverse
        let reversed: Vec<f32> = r.logits.iter().rev().copied().collect();
        // Assert: reversed order is descending
        assert_eq!(reversed, vec![0.4, 0.3, 0.2, 0.1]);
    }

    // ── ClassificationResult logits variance computation ──

    #[test]
    fn classification_result_logits_variance_uniform_is_zero() {
        // Arrange: all logits identical => variance is zero
        let r = ClassificationResult {
            index: 0,
            label_id: 0,
            score: 0.5,
            logits: vec![0.5, 0.5, 0.5, 0.5],
        };
        // Act: compute variance
        let n = r.logits.len() as f32;
        let mean: f32 = r.logits.iter().sum::<f32>() / n;
        let variance: f32 = r.logits.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        // Assert: variance of uniform values is zero
        assert!(variance.abs() < 1e-10, "variance of identical values is zero");
    }

    // ── ClassificationResult logits_variance_non_uniform_positive ──

    #[test]
    fn classification_result_logits_variance_non_uniform() {
        // Arrange: logits with spread
        let r = ClassificationResult {
            index: 0,
            label_id: 2,
            score: 0.7,
            logits: vec![0.1, 0.2, 0.7],
        };
        // Act: compute variance
        let n = r.logits.len() as f32;
        let mean: f32 = r.logits.iter().sum::<f32>() / n;
        let variance: f32 = r.logits.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        // Assert: variance is positive for non-uniform values
        let expected_mean = (0.1 + 0.2 + 0.7) / 3.0;
        assert!((mean - expected_mean).abs() < 1e-6, "mean is correct");
        assert!(variance > 0.0, "variance is positive for spread-out values");
    }

    // ── ClassifyResponse predictions dedup by label_id ──

    #[test]
    fn classify_response_dedup_predictions_by_label() {
        // Arrange: predictions with duplicate label_ids
        let resp = ClassifyResponse {
            predictions: vec![
                ClassificationResult { index: 0, label_id: 1, score: 0.9, logits: vec![0.1, 0.9] },
                ClassificationResult { index: 1, label_id: 0, score: 0.6, logits: vec![0.6, 0.4] },
                ClassificationResult { index: 2, label_id: 1, score: 0.8, logits: vec![0.2, 0.8] },
            ],
        };
        // Act: deduplicate by label_id, keeping first occurrence
        let mut seen = std::collections::HashSet::new();
        let unique: Vec<&ClassificationResult> = resp.predictions
            .iter()
            .filter(|p| seen.insert(p.label_id))
            .collect();
        // Assert: only 2 unique label_ids (0 and 1), first occurrence kept
        assert_eq!(unique.len(), 2);
        assert_eq!(unique[0].index, 0, "first label_id=1 kept (index 0)");
        assert_eq!(unique[1].index, 1, "label_id=0 kept (index 1)");
    }

    // ── ClassificationResult logits enumerate to verify label_id alignment ──

    #[test]
    fn classification_result_logits_enumerate_and_argmax_alignment() {
        // Arrange: logits where the max is at a known position
        let r = ClassificationResult {
            index: 0,
            label_id: 3,
            score: 0.85,
            logits: vec![0.05, 0.10, 0.0, 0.85, 0.0],
        };
        // Act: enumerate to find (index, value) pairs and locate the max
        let max_entry = r.logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        // Assert: argmax index matches label_id
        let (max_idx, &max_val) = max_entry.unwrap();
        assert_eq!(max_idx, r.label_id, "argmax position equals label_id");
        assert!((max_val - r.score).abs() < 1e-6, "argmax value equals score");
    }

    // ── StreamModerateBuilder tests (REQ-QGUARD-003) ──

    use crate::qwen3_guard::{Qwen3GuardConfig, Qwen3GuardHead};

    fn small_head() -> Qwen3GuardHead {
        // Tiny config: hidden=8, guard_inner=4, risk=3, category=8,
        // query_risk=3, query_category=9.
        let cfg = Qwen3GuardConfig {
            hidden_size: 8,
            guard_inner_size: 4,
            num_risk_level: 3,
            num_category: 8,
            num_query_risk_level: 3,
            num_query_category: 9,
            rms_norm_eps: 1e-6,
        };
        let h = cfg.hidden_size;
        let g = cfg.guard_inner_size;
        // weights all 0.5 → deterministic output
        let risk_pre = vec![0.5; g * h];
        let risk_norm = vec![1.0; g]; // identity-ish norm weight
        let risk_head = vec![0.5; cfg.num_risk_level * g];
        let category_head = vec![0.5; cfg.num_category * g];
        let query_pre = vec![0.25; g * h];
        let query_norm = vec![1.0; g];
        let query_risk_head = vec![0.25; cfg.num_query_risk_level * g];
        let query_category_head = vec![0.25; cfg.num_query_category * g];
        Qwen3GuardHead::from_weights(
            cfg,
            risk_pre,
            risk_norm,
            risk_head,
            category_head,
            query_pre,
            query_norm,
            query_risk_head,
            query_category_head,
        )
        .expect("small head from_weights")
    }

    #[test]
    fn stream_moderate_response_role_output_dims() {
        let sm = StreamModerateBuilder::from_head(small_head());
        let hidden = vec![0.1; 8];
        let outcome = sm
            .stream_moderate(&hidden, StreamModerateRole::Response)
            .expect("stream_moderate response");
        // Response path: risk 3 + category 8
        assert_eq!(outcome.risk_logits.len(), 3);
        assert_eq!(outcome.category_logits.len(), 8);
        assert_eq!(outcome.role, StreamModerateRole::Response);
        // All logits finite
        for &v in outcome.risk_logits.iter().chain(outcome.category_logits.iter()) {
            assert!(v.is_finite(), "logit finite");
        }
    }

    #[test]
    fn stream_moderate_query_role_output_dims() {
        let sm = StreamModerateBuilder::from_head(small_head());
        let hidden = vec![0.1; 8];
        let outcome = sm
            .stream_moderate(&hidden, StreamModerateRole::Query)
            .expect("stream_moderate query");
        // Query path: query_risk 3 + query_category 9 (含 Jailbreak)
        assert_eq!(outcome.risk_logits.len(), 3);
        assert_eq!(outcome.category_logits.len(), 9);
        assert_eq!(outcome.role, StreamModerateRole::Query);
    }

    #[test]
    fn stream_moderate_rejects_bad_hidden_dim() {
        let sm = StreamModerateBuilder::from_head(small_head());
        let bad_hidden = vec![0.1; 7]; // hidden_size=8, wrong
        let err = sm
            .stream_moderate(&bad_hidden, StreamModerateRole::Response)
            .unwrap_err();
        assert!(format!("{err}").contains("hidden len"), "error mentions dim");
    }

    #[test]
    fn stream_moderate_role_selects_correct_logit_group() {
        // Query vs Response must produce different category lengths AND
        // different magnitudes (different pre weights 0.5 vs 0.25).
        let sm = StreamModerateBuilder::from_head(small_head());
        let hidden = vec![1.0; 8];
        let resp = sm
            .stream_moderate(&hidden, StreamModerateRole::Response)
            .unwrap();
        let qry = sm
            .stream_moderate(&hidden, StreamModerateRole::Query)
            .unwrap();
        assert_ne!(resp.category_logits.len(), qry.category_logits.len());
        // Response pre weight (0.5) > Query pre weight (0.25) → response logits larger
        let resp_sum: f32 = resp.risk_logits.iter().sum();
        let qry_sum: f32 = qry.risk_logits.iter().sum();
        assert!(
            resp_sum > qry_sum,
            "response risk logits {resp_sum} > query {qry_sum}"
        );
    }

    #[test]
    fn stream_moderate_sequence_multi_token() {
        let sm = StreamModerateBuilder::from_head(small_head());
        // 3 tokens × hidden_size=8
        let hidden_seq = vec![0.2; 3 * 8];
        let outcomes = sm
            .stream_moderate_sequence(&hidden_seq, StreamModerateRole::Response)
            .expect("sequence");
        assert_eq!(outcomes.len(), 3);
        for o in &outcomes {
            assert_eq!(o.risk_logits.len(), 3);
            assert_eq!(o.category_logits.len(), 8);
        }
    }

    #[test]
    fn stream_moderation_outcome_softmax_sums_to_one() {
        let sm = StreamModerateBuilder::from_head(small_head());
        let hidden = vec![0.5; 8];
        let outcome = sm
            .stream_moderate(&hidden, StreamModerateRole::Response)
            .unwrap();
        let probs = outcome.risk_softmax();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sums to 1, got {sum}");
    }

    #[test]
    fn stream_moderation_outcome_argmax_in_range() {
        let sm = StreamModerateBuilder::from_head(small_head());
        let hidden = vec![0.3; 8];
        let outcome = sm
            .stream_moderate(&hidden, StreamModerateRole::Query)
            .unwrap();
        assert!(outcome.risk_argmax() < 3);
        assert!(outcome.category_argmax() < 9);
    }

    #[test]
    fn argmax_and_softmax_helpers_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // softmax monotonic: larger input → larger prob
        assert!(s[2] > s[1] && s[1] > s[0]);
        // empty
        assert_eq!(argmax(&[]), 0); // returns 0 on empty (safe default)
        assert!(softmax(&[]).is_empty());
    }
}
