//! Intent Recall SDK — mid-layer hidden state extraction for lightweight
//! intent classification / retrieval-augmented dispatch.
//!
//! **SSOT**: `SPEC/INTENT.md`, `SPEC/01-REQUIREMENTS.md §15`,
//! `SPEC/04-API-DESIGN.md §3.10`.
//!
//! **REQ 覆盖**:
//! - REQ-INTENT-001 — `encode_intent` 返回 shape 正确、全 finite 的向量
//! - REQ-INTENT-002 — `PoolMode` (Last/Mean/CLS) 影响输出向量内容
//! - REQ-INTENT-003 — `encode_intent` 与 `encode_to_layer` 语义等价 (DRY delegate)
//!
//! # Rationale
//!
//! Many downstream tasks (intent routing, RAG query understanding) need a
//! compact semantic vector of a user's input but do not require full LLM
//! generation. Running only the first ~50% of decoder layers yields a
//! hidden state rich enough for cosine-nearest-neighbor / linear SVM
//! classification at a fraction of the compute cost.
//!
//! Intent Recall wraps [`Client::encode_to_layer`](crate::client::Client::encode_to_layer)
//! with semantically clearer naming. Both paths share the same underlying
//! implementation (FusedGraphExecutor + `run_with_callbacks` +
//! `MidLayerEncodeCallback`). `encode_intent` exists mainly so
//! intent-classification call sites read intuitively — there is zero
//! behavioral difference versus `encode_to_layer`.
//!
//! # 与 HR 的关系
//!
//! - HR `encode_to_layer(text, anchor, pool)`: 原始底层 API,描述"截断前向到
//!   第 N 层, 按 pool 汇聚"。
//! - Intent `encode_intent(text, anchor, pool)`: 语义包装,鼓励用户将该 API
//!   用于意图识别等下游轻量分类任务。
//!
//! 两者在代码层完全等价 (`encode_intent` 内部直接 delegate 给
//! `encode_to_layer`), 禁止复制实现逻辑 (DRY).

use thiserror::Error;

use crate::head_routing::{HeadRoutingError, PoolMode};

/// `encode_intent` 的输出.
#[derive(Debug, Clone)]
pub struct IntentEncoding {
    /// Pooled hidden-state vector (`hidden_size`).
    pub embedding: Vec<f32>,
    /// 实际落点的物理层索引 (LayerAnchor::resolve 结果).
    pub actual_layer: usize,
    /// 使用的池化模式.
    pub pool: PoolMode,
}

impl IntentEncoding {
    /// 向量维度 (= model hidden_size).
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }

    /// L2 范数 (调试 / 健康检查).
    pub fn l2_norm(&self) -> f32 {
        self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Intent SDK 错误类型.
///
/// 多数错误直接从底层 `HeadRoutingError` 透传 (encode_to_layer 的错误集)。
#[derive(Debug, Error)]
pub enum IntentError {
    /// `LayerAnchor::Relative` 越界 / Absolute 越界 / NaN.
    #[error("invalid layer anchor: {0}")]
    InvalidLayerAnchor(String),

    /// 下游 head_routing / executor 错误.
    #[error("encode failed: {0}")]
    EncodeFailed(String),

    /// 客户端未加载模型.
    #[error("no model loaded")]
    NoModelLoaded,
}

impl From<HeadRoutingError> for IntentError {
    fn from(err: HeadRoutingError) -> Self {
        match err {
            HeadRoutingError::InvalidLayerAnchor(v) => IntentError::InvalidLayerAnchor(format!("{v}")),
            other => IntentError::EncodeFailed(format!("{other}")),
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_encoding_dim_and_norm() {
        let enc = IntentEncoding {
            embedding: vec![3.0, 4.0],
            actual_layer: 5,
            pool: PoolMode::MeanPool,
        };
        assert_eq!(enc.dim(), 2);
        assert!((enc.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn intent_error_from_head_routing_preserves_anchor_variant() {
        let hr = HeadRoutingError::InvalidLayerAnchor(1.5);
        let ie: IntentError = hr.into();
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
    }

    #[test]
    fn intent_error_from_head_routing_maps_other_to_encode_failed() {
        let hr = HeadRoutingError::EmptyLabels;
        let ie: IntentError = hr.into();
        assert!(matches!(ie, IntentError::EncodeFailed(_)));
    }
}
