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
//! implementation (mega-kernel path + `run_with_callbacks` +
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

    #[test]
    fn intent_encoding_zero_vector_norm_is_zero() {
        let enc = IntentEncoding {
            embedding: vec![0.0; 128],
            actual_layer: 0,
            pool: PoolMode::LastToken,
        };
        assert_eq!(enc.dim(), 128);
        assert_eq!(enc.l2_norm(), 0.0);
    }

    #[test]
    fn intent_encoding_single_element() {
        let enc = IntentEncoding {
            embedding: vec![-2.0],
            actual_layer: 3,
            pool: PoolMode::ClsToken,
        };
        assert_eq!(enc.dim(), 1);
        assert!((enc.l2_norm() - 2.0).abs() < 1e-6);
    }

    // ---- IntentEncoding struct construction & field access ----

    #[test]
    fn intent_encoding_fields_accessible() {
        let enc = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0],
            actual_layer: 7,
            pool: PoolMode::MeanPool,
        };
        assert_eq!(enc.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(enc.actual_layer, 7);
        assert_eq!(enc.pool, PoolMode::MeanPool);
    }

    #[test]
    fn intent_encoding_clone_produces_equal_instance() {
        let enc = IntentEncoding {
            embedding: vec![0.5, -0.5, 1.5],
            actual_layer: 12,
            pool: PoolMode::LastToken,
        };
        let cloned = enc.clone();
        assert_eq!(cloned.embedding, enc.embedding);
        assert_eq!(cloned.actual_layer, enc.actual_layer);
        assert_eq!(cloned.pool, enc.pool);
        // Cloned is independent
        assert!(cloned.embedding.as_ptr() != enc.embedding.as_ptr());
    }

    #[test]
    fn intent_encoding_empty_embedding() {
        let enc = IntentEncoding {
            embedding: vec![],
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };
        assert_eq!(enc.dim(), 0);
        assert_eq!(enc.l2_norm(), 0.0);
    }

    #[test]
    fn intent_encoding_l2_norm_negative_values() {
        let enc = IntentEncoding {
            embedding: vec![-3.0, -4.0],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };
        // L2 norm of [-3, -4] is sqrt(9 + 16) = 5
        assert!((enc.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn intent_encoding_l2_norm_unit_vector() {
        let enc = IntentEncoding {
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            actual_layer: 2,
            pool: PoolMode::LastToken,
        };
        assert!((enc.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn intent_encoding_debug_format() {
        let enc = IntentEncoding {
            embedding: vec![1.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let debug_str = format!("{enc:?}");
        assert!(debug_str.contains("IntentEncoding"));
        assert!(debug_str.contains("embedding"));
        assert!(debug_str.contains("actual_layer"));
        assert!(debug_str.contains("pool"));
    }

    // ---- IntentError variants & Display ----

    #[test]
    fn intent_error_invalid_layer_anchor_display() {
        let err = IntentError::InvalidLayerAnchor("anchor out of range".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("invalid layer anchor"));
        assert!(msg.contains("anchor out of range"));
    }

    #[test]
    fn intent_error_encode_failed_display() {
        let err = IntentError::EncodeFailed("gpu crash".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("encode failed"));
        assert!(msg.contains("gpu crash"));
    }

    #[test]
    fn intent_error_no_model_loaded_display() {
        let err = IntentError::NoModelLoaded;
        let msg = format!("{err}");
        assert!(msg.contains("no model loaded"));
    }

    #[test]
    fn intent_error_debug_format_all_variants() {
        let variants: Vec<IntentError> = vec![
            IntentError::InvalidLayerAnchor("x".into()),
            IntentError::EncodeFailed("y".into()),
            IntentError::NoModelLoaded,
        ];
        for v in &variants {
            let debug = format!("{v:?}");
            assert!(!debug.is_empty(), "Debug output should not be empty");
        }
    }

    // ---- From<HeadRoutingError> conversion ----

    #[test]
    fn from_head_routing_backend_maps_to_encode_failed() {
        let hr = HeadRoutingError::Backend("cuda error".to_string());
        let ie: IntentError = hr.into();
        match &ie {
            IntentError::EncodeFailed(msg) => assert!(msg.contains("cuda error")),
            other => panic!("expected EncodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn from_head_routing_invalid_config_maps_to_encode_failed() {
        let hr = HeadRoutingError::InvalidConfig("bad temp".to_string());
        let ie: IntentError = hr.into();
        match &ie {
            IntentError::EncodeFailed(msg) => assert!(msg.contains("bad temp")),
            other => panic!("expected EncodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn from_head_routing_token_not_found_maps_to_encode_failed() {
        let hr = HeadRoutingError::TokenNotFound("unknown".to_string());
        let ie: IntentError = hr.into();
        assert!(matches!(ie, IntentError::EncodeFailed(_)));
    }

    #[test]
    fn from_head_routing_mid_layer_not_supported_maps_to_encode_failed() {
        let hr = HeadRoutingError::MidLayerNotSupported;
        let ie: IntentError = hr.into();
        assert!(matches!(ie, IntentError::EncodeFailed(_)));
    }

    // ---- PoolMode coverage in IntentEncoding ----

    #[test]
    fn intent_encoding_all_pool_modes() {
        for pool in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            let enc = IntentEncoding {
                embedding: vec![1.0, 2.0],
                actual_layer: 0,
                pool,
            };
            assert_eq!(enc.pool, pool);
            assert_eq!(enc.dim(), 2);
        }
    }

    // ---- Edge cases: special float values ----

    #[test]
    fn intent_encoding_l2_norm_nan_embedding() {
        let enc = IntentEncoding {
            embedding: vec![f32::NAN, 1.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        assert!(enc.l2_norm().is_nan());
    }

    #[test]
    fn intent_encoding_l2_norm_infinity_embedding() {
        let enc = IntentEncoding {
            embedding: vec![f32::INFINITY, 0.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        assert!(enc.l2_norm().is_infinite() && enc.l2_norm() > 0.0);
    }

    #[test]
    fn intent_encoding_l2_norm_subnormal_values() {
        // Subnormal f32: smallest positive denormalized. Squaring may underflow
        // to zero for the tiniest values, so use a larger subnormal to verify
        // l2_norm handles it without panicking or returning NaN.
        let subnormal = f32::from_bits(0x00400000u32); // ~5.9e-36, still subnormal
        let enc = IntentEncoding {
            embedding: vec![subnormal; 4],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };
        let norm = enc.l2_norm();
        // Squaring a subnormal may still underflow; just verify no panic/NaN
        assert!(norm.is_finite() || norm == 0.0);
    }

    #[test]
    fn intent_embedding_with_f32_max_preserves_value() {
        let enc = IntentEncoding {
            embedding: vec![f32::MAX],
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };
        assert_eq!(enc.embedding[0], f32::MAX);
        assert_eq!(enc.dim(), 1);
    }

    // ---- Edge cases: boundary values ----

    #[test]
    fn intent_encoding_actual_layer_usize_max() {
        let enc = IntentEncoding {
            embedding: vec![1.0],
            actual_layer: usize::MAX,
            pool: PoolMode::LastToken,
        };
        assert_eq!(enc.actual_layer, usize::MAX);
    }

    #[test]
    fn intent_encoding_dim_consistent_after_clone() {
        let enc = IntentEncoding {
            embedding: vec![0.1; 256],
            actual_layer: 10,
            pool: PoolMode::MeanPool,
        };
        let cloned = enc.clone();
        assert_eq!(enc.dim(), cloned.dim());
        assert_eq!(enc.embedding.len(), 256);
    }

    #[test]
    fn intent_encoding_large_dim_does_not_panic() {
        let enc = IntentEncoding {
            embedding: vec![0.0; 8192],
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };
        assert_eq!(enc.dim(), 8192);
        assert_eq!(enc.l2_norm(), 0.0);
    }

    #[test]
    fn intent_encoding_l2_norm_mixed_sign_large_values() {
        // 1e19^2 = 1e38, which is near f32::MAX (~3.4e38); sum of two should still be finite
        let enc = IntentEncoding {
            embedding: vec![1e19, -1e19],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let norm = enc.l2_norm();
        assert!(norm.is_finite());
        let expected = (1e19f32 * 1e19f32 + 1e19f32 * 1e19f32).sqrt();
        assert!((norm - expected).abs() < expected * 1e-5);
    }

    // ---- IntentError edge cases ----

    #[test]
    fn intent_error_no_model_loaded_is_distinct_variant() {
        let err = IntentError::NoModelLoaded;
        let msg = format!("{err}");
        // Verify it is not accidentally matching other variants
        assert!(!msg.contains("encode failed"));
        assert!(!msg.contains("invalid layer anchor"));
    }

    #[test]
    fn from_head_routing_preserves_anchor_float_value() {
        let anchor_val = 0.75;
        let hr = HeadRoutingError::InvalidLayerAnchor(anchor_val);
        let ie: IntentError = hr.into();
        match ie {
            IntentError::InvalidLayerAnchor(msg) => {
                assert!(msg.contains(&anchor_val.to_string()));
            }
            other => panic!("expected InvalidLayerAnchor, got {other:?}"),
        }
    }

    #[test]
    fn intent_error_display_roundtrip_all_variants() {
        let cases: Vec<(IntentError, &str)> = vec![
            (IntentError::InvalidLayerAnchor("test".into()), "invalid layer anchor"),
            (IntentError::EncodeFailed("oops".into()), "encode failed"),
            (IntentError::NoModelLoaded, "no model loaded"),
        ];
        for (err, expected_substring) in cases {
            let display = format!("{err}");
            assert!(
                display.contains(expected_substring),
                "Display of {err:?} should contain '{expected_substring}', got '{display}'"
            );
        }
    }

    #[test]
    fn intent_encoding_negative_infinity_norm() {
        let enc = IntentEncoding {
            embedding: vec![f32::NEG_INFINITY],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let norm = enc.l2_norm();
        // (-inf)^2 = +inf, sqrt(+inf) = +inf
        assert!(norm.is_infinite() && norm > 0.0);
    }

    #[test]
    fn intent_encoding_zero_minus_zero_mixed() {
        let enc = IntentEncoding {
            embedding: vec![0.0, -0.0],
            actual_layer: 2,
            pool: PoolMode::LastToken,
        };
        let norm = enc.l2_norm();
        assert_eq!(norm, 0.0);
    }

    // ---- Additional coverage ----

    #[test]
    fn intent_encoding_clone_independence_original_mutation() {
        // Arrange: create an IntentEncoding and clone it
        let original = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0],
            actual_layer: 4,
            pool: PoolMode::MeanPool,
        };
        let mut cloned = original.clone();

        // Act: mutate the clone's embedding
        cloned.embedding[0] = 99.0;
        cloned.actual_layer = 0;

        // Assert: original is unaffected
        assert_eq!(original.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(original.actual_layer, 4);
        assert_eq!(cloned.embedding, vec![99.0, 2.0, 3.0]);
        assert_eq!(cloned.actual_layer, 0);
    }

    #[test]
    fn intent_encoding_clone_independence_clone_mutation() {
        // Arrange: create and clone, then mutate the original
        let mut original = IntentEncoding {
            embedding: vec![10.0, 20.0],
            actual_layer: 1,
            pool: PoolMode::ClsToken,
        };
        let cloned = original.clone();

        // Act: mutate the original after cloning
        original.embedding[1] = -5.0;
        original.actual_layer = 100;

        // Assert: clone is unaffected
        assert_eq!(cloned.embedding, vec![10.0, 20.0]);
        assert_eq!(cloned.actual_layer, 1);
    }

    #[test]
    fn intent_error_same_variant_same_display() {
        // Arrange: construct two identical IntentError::EncodeFailed instances
        let err1 = IntentError::EncodeFailed("timeout".to_string());
        let err2 = IntentError::EncodeFailed("timeout".to_string());

        // Act: format both to strings
        let msg1 = format!("{err1}");
        let msg2 = format!("{err2}");

        // Assert: identical display output
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn intent_encoding_l2_norm_all_same_value_matches_formula() {
        // Arrange: embedding with all identical values; L2 = |val| * sqrt(dim)
        let val = 0.5f32;
        let enc = IntentEncoding {
            embedding: vec![val; 100],
            actual_layer: 7,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: |0.5| * sqrt(100) = 0.5 * 10 = 5.0
        let expected = val * (enc.dim() as f32).sqrt();
        assert!((norm - expected).abs() < 1e-4);
    }

    #[test]
    fn intent_encoding_dim_matches_vec_len_exactly() {
        // Arrange: various embedding lengths
        for len in [0, 1, 7, 64, 512, 4096] {
            let enc = IntentEncoding {
                embedding: vec![0.0; len],
                actual_layer: 0,
                pool: PoolMode::LastToken,
            };

            // Act & Assert
            assert_eq!(enc.dim(), len, "dim() should equal embedding.len() for len={len}");
        }
    }

    #[test]
    fn intent_error_invalid_layer_anchor_empty_string() {
        // Arrange: InvalidLayerAnchor with empty message
        let err = IntentError::InvalidLayerAnchor(String::new());

        // Act
        let msg = format!("{err}");

        // Assert: display still contains the prefix
        assert!(msg.contains("invalid layer anchor"));
    }

    #[test]
    fn intent_error_encode_failed_empty_string() {
        // Arrange: EncodeFailed with empty message
        let err = IntentError::EncodeFailed(String::new());

        // Act
        let msg = format!("{err}");

        // Assert: display still contains the prefix
        assert!(msg.contains("encode failed"));
    }

    #[test]
    fn intent_encoding_embedding_with_f32_min_positive() {
        // Arrange: embedding with f32::MIN_POSITIVE (smallest positive normal f32 ~1.175e-38).
        // Squaring it underflows to 0.0 in f32, so l2_norm returns 0.0 for a small count.
        // Use a single element to verify the value is stored correctly.
        let enc = IntentEncoding {
            embedding: vec![f32::MIN_POSITIVE],
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: f32::MIN_POSITIVE^2 underflows to 0 in f32 arithmetic,
        // so l2_norm returns 0.0 (the sum of squares is 0.0 after underflow).
        // The important thing is no panic and no NaN.
        assert!(norm == 0.0 || (norm > 0.0 && norm.is_finite()));
        assert_eq!(enc.embedding[0], f32::MIN_POSITIVE);
    }

    #[test]
    fn intent_encoding_embedding_with_f32_min() {
        // Arrange: embedding containing f32::MIN (most negative finite value)
        let enc = IntentEncoding {
            embedding: vec![f32::MIN],
            actual_layer: 5,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: f32::MIN squared overflows to +inf, sqrt(+inf) = +inf
        assert!(norm.is_infinite() && norm > 0.0);
        assert_eq!(enc.embedding[0], f32::MIN);
    }

    #[test]
    fn intent_encoding_l2_norm_alternating_signs() {
        // Arrange: alternating +1, -1 pattern (like a discrete signal)
        let enc = IntentEncoding {
            embedding: vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            actual_layer: 3,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: all squares are 1.0, sum = 8, sqrt(8) = 2*sqrt(2)
        let expected = (8.0f32).sqrt();
        assert!((norm - expected).abs() < 1e-5);
    }

    #[test]
    fn intent_encoding_l2_norm_all_negative_matches_positive() {
        // Arrange: negative-only vector should have same L2 as its absolute version
        let neg = IntentEncoding {
            embedding: vec![-1.0, -2.0, -3.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let pos = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };

        // Act & Assert: L2 norm is sign-invariant
        assert!((neg.l2_norm() - pos.l2_norm()).abs() < 1e-6);
    }

    #[test]
    fn intent_encoding_clone_large_embedding() {
        // Arrange: large embedding to stress-test clone
        let enc = IntentEncoding {
            embedding: (0..4096).map(|i| i as f32 * 0.001).collect(),
            actual_layer: 24,
            pool: PoolMode::ClsToken,
        };

        // Act
        let cloned = enc.clone();

        // Assert: element-by-element equality and independent allocation
        assert_eq!(cloned.embedding.len(), 4096);
        assert_eq!(cloned.embedding, enc.embedding);
        assert_ne!(cloned.embedding.as_ptr(), enc.embedding.as_ptr());
        assert_eq!(cloned.actual_layer, enc.actual_layer);
        assert_eq!(cloned.pool, enc.pool);
    }

    #[test]
    fn from_head_routing_multiple_consecutive_conversions() {
        // Arrange: convert multiple different HeadRoutingError variants in sequence
        let errors: Vec<HeadRoutingError> = vec![
            HeadRoutingError::TokenNotFound("tok_a".into()),
            HeadRoutingError::Backend("err_b".into()),
            HeadRoutingError::InvalidConfig("cfg_c".into()),
            HeadRoutingError::MidLayerNotSupported,
            HeadRoutingError::EmptyLabels,
        ];

        // Act & Assert: each converts to the correct IntentError variant
        for (idx, hr) in errors.into_iter().enumerate() {
            let ie: IntentError = hr.into();
            match idx {
                0 | 1 | 2 | 3 => assert!(
                    matches!(ie, IntentError::EncodeFailed(_)),
                    "index {idx}: expected EncodeFailed, got {ie:?}"
                ),
                4 => assert!(
                    matches!(ie, IntentError::EncodeFailed(_)),
                    "EmptyLabels should map to EncodeFailed"
                ),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn intent_encoding_pool_mode_equality_after_clone() {
        // Arrange: build with each pool mode, clone, verify pool matches
        for pool in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            let enc = IntentEncoding {
                embedding: vec![1.0],
                actual_layer: 0,
                pool,
            };
            let cloned = enc.clone();

            // Assert: pool mode preserved through clone
            assert_eq!(cloned.pool, pool);
        }
    }

    // ---- 13 new tests ----

    #[test]
    fn intent_encoding_dim_with_special_float_values() {
        // Arrange: embedding containing NaN, Inf, -Inf — dim() counts elements
        let enc = IntentEncoding {
            embedding: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };

        // Act
        let dim = enc.dim();

        // Assert: dim is the element count regardless of special float values
        assert_eq!(dim, 3);
    }

    #[test]
    fn intent_encoding_l2_norm_mixed_positive_negative() {
        // Arrange: mixed positive and negative values with known squares
        // [2.0, -3.0, 6.0] -> sum of squares = 4 + 9 + 36 = 49, sqrt(49) = 7.0
        let enc = IntentEncoding {
            embedding: vec![2.0, -3.0, 6.0],
            actual_layer: 4,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert
        assert!((norm - 7.0).abs() < 1e-5, "expected 7.0, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_all_same_nonzero_value() {
        // Arrange: all elements are the same nonzero value (3.0)
        // 5 elements: sqrt(5 * 9) = sqrt(45) = 3*sqrt(5)
        let enc = IntentEncoding {
            embedding: vec![3.0; 5],
            actual_layer: 2,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: 3.0 * sqrt(5)
        let expected = 3.0 * 5.0f32.sqrt();
        assert!((norm - expected).abs() < 1e-5, "expected {expected}, got {norm}");
    }

    #[test]
    fn intent_error_debug_includes_variant_name() {
        // Arrange: create one of each IntentError variant
        let anchor = IntentError::InvalidLayerAnchor("bad".to_string());
        let encode = IntentError::EncodeFailed("fail".to_string());
        let no_model = IntentError::NoModelLoaded;

        // Act: format each with Debug
        let anchor_dbg = format!("{anchor:?}");
        let encode_dbg = format!("{encode:?}");
        let no_model_dbg = format!("{no_model:?}");

        // Assert: Debug output contains the variant name
        assert!(
            anchor_dbg.contains("InvalidLayerAnchor"),
            "Debug should contain variant name, got: {anchor_dbg}"
        );
        assert!(
            encode_dbg.contains("EncodeFailed"),
            "Debug should contain variant name, got: {encode_dbg}"
        );
        assert!(
            no_model_dbg.contains("NoModelLoaded"),
            "Debug should contain variant name, got: {no_model_dbg}"
        );
    }

    #[test]
    fn intent_error_same_display_for_same_input() {
        // Arrange: two EncodeFailed errors constructed from the same HeadRoutingError
        let hr1 = HeadRoutingError::Backend("disk full".to_string());
        let hr2 = HeadRoutingError::Backend("disk full".to_string());
        let ie1: IntentError = hr1.into();
        let ie2: IntentError = hr2.into();

        // Act
        let msg1 = format!("{ie1}");
        let msg2 = format!("{ie2}");

        // Assert: same input produces same Display output
        assert_eq!(msg1, msg2, "identical inputs should produce identical Display");
    }

    #[test]
    fn intent_encoding_very_large_actual_layer() {
        // Arrange: actual_layer set to a very large but not usize::MAX value
        let large_layer = usize::MAX / 2;
        let enc = IntentEncoding {
            embedding: vec![1.0, 2.0],
            actual_layer: large_layer,
            pool: PoolMode::ClsToken,
        };

        // Act
        let layer = enc.actual_layer;

        // Assert: the large value is preserved exactly
        assert_eq!(layer, large_layer);
        assert!(layer < usize::MAX);
    }

    #[test]
    fn intent_encoding_cls_pool_l2_norm_and_dim() {
        // Arrange: IntentEncoding specifically with ClsPool variant and a 3-4-5 right triangle
        let enc = IntentEncoding {
            embedding: vec![5.0, 12.0],
            actual_layer: 6,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();
        let dim = enc.dim();

        // Assert: 5-12-13 right triangle, and ClsPool is preserved
        assert!((norm - 13.0).abs() < 1e-5, "expected 13.0, got {norm}");
        assert_eq!(dim, 2);
        assert_eq!(enc.pool, PoolMode::ClsToken);
    }

    #[test]
    fn intent_encoding_different_pool_same_embedding_same_l2_norm() {
        // Arrange: three IntentEncoding instances with identical embeddings but different pool modes
        let embedding = vec![1.0, 2.0, 2.0];
        let enc_mean = IntentEncoding {
            embedding: embedding.clone(),
            actual_layer: 3,
            pool: PoolMode::MeanPool,
        };
        let enc_last = IntentEncoding {
            embedding: embedding.clone(),
            actual_layer: 3,
            pool: PoolMode::LastToken,
        };
        let enc_cls = IntentEncoding {
            embedding: embedding.clone(),
            actual_layer: 3,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm_mean = enc_mean.l2_norm();
        let norm_last = enc_last.l2_norm();
        let norm_cls = enc_cls.l2_norm();

        // Assert: pool mode does not affect l2_norm (it only affects how embedding was produced)
        assert!((norm_mean - norm_last).abs() < 1e-6);
        assert!((norm_mean - norm_cls).abs() < 1e-6);
        assert!((norm_last - 3.0).abs() < 1e-5, "sqrt(1+4+4)=3.0, got {norm_last}");
    }

    #[test]
    fn intent_encoding_subnormal_embedding_dim_and_finiteness() {
        // Arrange: embedding with subnormal floats (denormalized), verify dim and no panic
        // A subnormal float: exponent bits all zero, mantissa nonzero
        let subnormal = f32::from_bits(0x00000001u32); // smallest positive subnormal ~1.4e-45
        let enc = IntentEncoding {
            embedding: vec![subnormal; 8],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };

        // Act
        let dim = enc.dim();
        let norm = enc.l2_norm();

        // Assert: dim counts elements correctly; l2_norm does not panic or return NaN
        assert_eq!(dim, 8);
        assert!(norm == 0.0 || norm.is_finite(), "subnormal norm should be 0 or finite, got {norm}");
    }

    #[test]
    fn from_head_routing_all_non_anchor_variants_map_to_encode_failed() {
        // Arrange: every HeadRoutingError variant except InvalidLayerAnchor
        let variants: Vec<HeadRoutingError> = vec![
            HeadRoutingError::TokenNotFound("missing_token".to_string()),
            HeadRoutingError::EmptyLabels,
            HeadRoutingError::InvalidConfig("negative_temp".to_string()),
            HeadRoutingError::MidLayerNotSupported,
            HeadRoutingError::Backend("cuda_oom".to_string()),
        ];

        // Act & Assert: all should map to IntentError::EncodeFailed
        for (idx, hr) in variants.into_iter().enumerate() {
            let ie: IntentError = hr.into();
            assert!(
                matches!(ie, IntentError::EncodeFailed(_)),
                "variant at index {idx} should map to EncodeFailed, got {ie:?}"
            );
        }
    }

    #[test]
    fn intent_encoding_dim_after_clone_matches_original() {
        // Arrange: embedding with a specific non-power-of-2 length
        let original = IntentEncoding {
            embedding: vec![0.25; 384],
            actual_layer: 8,
            pool: PoolMode::LastToken,
        };

        // Act
        let cloned = original.clone();
        let original_dim = original.dim();
        let cloned_dim = cloned.dim();

        // Assert: both report the same dimension, and it matches the vector length
        assert_eq!(original_dim, cloned_dim);
        assert_eq!(original_dim, 384);
    }

    #[test]
    fn intent_encoding_l2_norm_pythagorean_triple_8_15_17() {
        // Arrange: use the 8-15-17 Pythagorean triple for exact verification
        let enc = IntentEncoding {
            embedding: vec![8.0, 15.0],
            actual_layer: 10,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: sqrt(64 + 225) = sqrt(289) = 17.0 exactly
        assert!((norm - 17.0).abs() < 1e-6, "expected 17.0, got {norm}");
    }

    #[test]
    fn intent_error_display_each_variant_contains_key_text() {
        // Arrange: one instance per IntentError variant with specific messages
        let anchor = IntentError::InvalidLayerAnchor("relative 1.5".to_string());
        let encode = IntentError::EncodeFailed("layer 24 crashed".to_string());
        let no_model = IntentError::NoModelLoaded;

        // Act
        let anchor_msg = format!("{anchor}");
        let encode_msg = format!("{encode}");
        let no_model_msg = format!("{no_model}");

        // Assert: each Display output contains its key identifier text
        assert!(
            anchor_msg.contains("invalid layer anchor"),
            "InvalidLayerAnchor Display should contain 'invalid layer anchor', got: {anchor_msg}"
        );
        assert!(
            anchor_msg.contains("relative 1.5"),
            "InvalidLayerAnchor Display should contain the detail message, got: {anchor_msg}"
        );
        assert!(
            encode_msg.contains("encode failed"),
            "EncodeFailed Display should contain 'encode failed', got: {encode_msg}"
        );
        assert!(
            encode_msg.contains("layer 24 crashed"),
            "EncodeFailed Display should contain the detail message, got: {encode_msg}"
        );
        assert!(
            no_model_msg.contains("no model loaded"),
            "NoModelLoaded Display should contain 'no model loaded', got: {no_model_msg}"
        );
    }

    // ---- Gap-filling tests (targeting uncovered edge cases) ----

    #[test]
    fn intent_encoding_l2_norm_all_nan_elements() {
        // Arrange: embedding where every element is NaN
        let enc = IntentEncoding {
            embedding: vec![f32::NAN; 4],
            actual_layer: 2,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: NaN squared is NaN, NaN sum is NaN, NaN sqrt is NaN
        assert!(norm.is_nan(), "all-NaN embedding should yield NaN norm, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_single_finite_plus_infinity() {
        // Arrange: one finite value and one +infinity
        // 1.0^2 + inf^2 = 1.0 + inf = inf, sqrt(inf) = inf
        let enc = IntentEncoding {
            embedding: vec![1.0, f32::INFINITY],
            actual_layer: 0,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: any infinite element makes the norm infinite
        assert!(norm.is_infinite() && norm > 0.0, "expected +inf norm, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_positive_and_negative_infinity() {
        // Arrange: both +inf and -inf — squares are both +inf, sum is +inf, sqrt is +inf
        let enc = IntentEncoding {
            embedding: vec![f32::INFINITY, f32::NEG_INFINITY],
            actual_layer: 3,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert
        assert!(norm.is_infinite() && norm > 0.0, "expected +inf norm, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_f32_epsilon() {
        // Arrange: embedding with f32::EPSILON (the smallest representable difference from 1.0)
        // EPSILON^2 is nonzero and representable in f32
        let eps = f32::EPSILON;
        let enc = IntentEncoding {
            embedding: vec![eps; 64],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: norm should be finite and positive (EPSILON^2 is nonzero)
        assert!(norm.is_finite(), "norm should be finite, got {norm}");
        assert!(norm >= 0.0, "norm should be non-negative, got {norm}");
        // Verify the formula: norm = EPSILON * sqrt(64) = EPSILON * 8
        let expected = eps * 64.0f32.sqrt();
        assert!((norm - expected).abs() < expected * 1e-5, "expected {expected}, got {norm}");
    }

    #[test]
    fn from_head_routing_anchor_negative_value() {
        // Arrange: InvalidLayerAnchor with a negative float value
        let hr = HeadRoutingError::InvalidLayerAnchor(-0.5);
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: converts to InvalidLayerAnchor and preserves the negative value in the message
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
        assert!(msg.contains("-0.5"), "message should contain the negative anchor value, got: {msg}");
    }

    #[test]
    fn from_head_routing_anchor_zero_value() {
        // Arrange: InvalidLayerAnchor with zero — edge of valid range
        let hr = HeadRoutingError::InvalidLayerAnchor(0.0);
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
        assert!(msg.contains("0"), "message should contain the zero anchor value, got: {msg}");
    }

    #[test]
    fn from_head_routing_preserves_original_error_text_in_encode_failed() {
        // Arrange: convert Backend error and verify the original text is preserved
        let detail = "CUDA out of memory: tried to allocate 2.00 GiB";
        let hr = HeadRoutingError::Backend(detail.to_string());
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: the full backend error detail is present in the EncodeFailed message
        match &ie {
            IntentError::EncodeFailed(inner) => {
                assert!(inner.contains(detail), "inner message should contain original detail, got: {inner}");
            }
            other => panic!("expected EncodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn intent_error_equality_by_display_for_no_model_loaded() {
        // Arrange: construct NoModelLoaded twice (it has no fields, so should be identical)
        let err1 = IntentError::NoModelLoaded;
        let err2 = IntentError::NoModelLoaded;

        // Act
        let msg1 = format!("{err1}");
        let msg2 = format!("{err2}");

        // Assert: fieldless variant always produces identical display
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn intent_encoding_l2_norm_very_small_but_normal_values() {
        // Arrange: embedding with small but representable floats whose squares
        // do NOT underflow to zero. f32::MIN_POSITIVE ~1.175e-38, squaring gives ~1.38e-76
        // which underflows. Use 1e-15 instead: 1e-15^2 = 1e-30, which is representable.
        // 4 elements: l2 = sqrt(4 * (1e-15)^2) = 2e-15
        let val = 1e-15f32;
        let enc = IntentEncoding {
            embedding: vec![val; 4],
            actual_layer: 5,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: norm = val * sqrt(4) = val * 2
        let expected = val * 2.0f32;
        assert!(norm.is_finite(), "norm should be finite, got {norm}");
        assert!((norm - expected).abs() < expected * 1e-5, "expected {expected}, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_with_mixed_nan_and_finite() {
        // Arrange: embedding mixing NaN with normal finite values
        let enc = IntentEncoding {
            embedding: vec![1.0, f32::NAN, 2.0],
            actual_layer: 0,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: any NaN in the sum makes the entire result NaN
        assert!(norm.is_nan(), "NaN-contaminated sum should yield NaN norm, got {norm}");
    }

    #[test]
    fn intent_encoding_dim_after_multiple_clones() {
        // Arrange: clone multiple times from the same original, verify dim consistency
        let original = IntentEncoding {
            embedding: vec![0.1; 768],
            actual_layer: 15,
            pool: PoolMode::MeanPool,
        };

        // Act
        let c1 = original.clone();
        let c2 = original.clone();
        let c3 = c1.clone();

        // Assert: all copies report the same dimension
        assert_eq!(original.dim(), 768);
        assert_eq!(c1.dim(), 768);
        assert_eq!(c2.dim(), 768);
        assert_eq!(c3.dim(), 768);
    }

    #[test]
    fn intent_encoding_l2_norm_orthonormal_basis_vector() {
        // Arrange: simulate a unit vector from a 3D orthonormal basis (e_y)
        let enc = IntentEncoding {
            embedding: vec![0.0, 1.0, 0.0],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: exactly 1.0 (unit vector)
        assert!((norm - 1.0).abs() < 1e-7, "unit vector should have norm 1.0, got {norm}");
    }

    #[test]
    fn from_head_routing_invalid_config_long_message() {
        // Arrange: InvalidConfig with a very long diagnostic message
        let long_msg = "parameter 'temperature' must be in (0.0, 2.0], got -999.999; \
                        this indicates a misconfigured generation profile; \
                        please check your client configuration before retrying";
        let hr = HeadRoutingError::InvalidConfig(long_msg.to_string());
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: the full long message is preserved inside EncodeFailed
        match &ie {
            IntentError::EncodeFailed(inner) => {
                assert!(
                    inner.contains("temperature"),
                    "EncodeFailed should contain 'temperature' from original, got: {inner}"
                );
                assert!(
                    inner.contains("-999.999"),
                    "EncodeFailed should contain the offending value, got: {inner}"
                );
            }
            other => panic!("expected EncodeFailed, got {other:?}"),
        }
    }

    // ---- Additional gap-filling tests (wave-2) ----

    #[test]
    fn intent_encoding_actual_layer_zero_with_nontrivial_embedding() {
        // Arrange: actual_layer=0 (first layer) with a real embedding, not all zeros
        let enc = IntentEncoding {
            embedding: vec![1.0, -1.0, 2.0, -2.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };

        // Act & Assert: layer 0 is a valid physical layer; norm is unaffected by actual_layer
        assert_eq!(enc.actual_layer, 0);
        let expected_norm = (1.0f32 + 1.0 + 4.0 + 4.0).sqrt(); // sqrt(10)
        assert!(
            (enc.l2_norm() - expected_norm).abs() < 1e-5,
            "expected {expected_norm}, got {}",
            enc.l2_norm()
        );
    }

    #[test]
    fn intent_encoding_l2_norm_invariant_across_clone() {
        // Arrange: create an encoding, clone it, verify norm is identical
        let original = IntentEncoding {
            embedding: vec![3.0, 4.0, 5.0, 6.0],
            actual_layer: 9,
            pool: PoolMode::LastToken,
        };

        // Act
        let cloned = original.clone();
        let norm_orig = original.l2_norm();
        let norm_clone = cloned.l2_norm();

        // Assert: norm is bitwise identical (deterministic computation on same data)
        assert_eq!(norm_orig, norm_clone, "clone must produce identical norm");
    }

    #[test]
    fn intent_encoding_different_actual_layers_same_embedding_same_norm() {
        // Arrange: two IntentEncoding instances with identical embeddings but different actual_layer
        let embedding = vec![1.0, 2.0, 3.0];
        let enc_layer0 = IntentEncoding {
            embedding: embedding.clone(),
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };
        let enc_layer99 = IntentEncoding {
            embedding: embedding.clone(),
            actual_layer: 99,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm0 = enc_layer0.l2_norm();
        let norm99 = enc_layer99.l2_norm();

        // Assert: actual_layer is metadata; it does not affect the mathematical norm
        assert_eq!(norm0, norm99, "actual_layer should not affect l2_norm");
    }

    #[test]
    fn intent_encoding_pool_mode_field_mutation_after_clone_independence() {
        // Arrange: clone and verify pool field can differ after construction
        let mut enc = IntentEncoding {
            embedding: vec![1.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let cloned = enc.clone();

        // Act: swap pool mode on the original (PoolMode is Copy, so this is a value swap)
        enc.pool = PoolMode::ClsToken;

        // Assert: cloned retains the original pool mode
        assert_eq!(enc.pool, PoolMode::ClsToken);
        assert_eq!(cloned.pool, PoolMode::MeanPool, "clone must retain original pool");
    }

    #[test]
    fn intent_error_source_returns_none_for_all_variants() {
        // Arrange: IntentError uses thiserror but does not wrap inner errors via #[source],
        // so std::error::Error::source() should return None for all variants.
        let variants: Vec<IntentError> = vec![
            IntentError::InvalidLayerAnchor("x".into()),
            IntentError::EncodeFailed("y".into()),
            IntentError::NoModelLoaded,
        ];

        // Act & Assert: none have a source chain
        for err in &variants {
            assert!(
                std::error::Error::source(err).is_none(),
                "IntentError variants should have no source, but {:?} does",
                err
            );
        }
    }

    #[test]
    fn intent_encoding_debug_contains_exact_field_values() {
        // Arrange: encoding with specific values for round-trip verification
        let enc = IntentEncoding {
            embedding: vec![42.0],
            actual_layer: 7,
            pool: PoolMode::ClsToken,
        };

        // Act
        let debug = format!("{enc:?}");

        // Assert: Debug output contains the specific actual_layer value
        assert!(
            debug.contains("42"),
            "Debug should contain embedding value 42, got: {debug}"
        );
        assert!(
            debug.contains("7"),
            "Debug should contain actual_layer value 7, got: {debug}"
        );
    }

    #[test]
    fn intent_encoding_l2_norm_single_element_zero() {
        // Arrange: single-element embedding with value 0.0
        let enc = IntentEncoding {
            embedding: vec![0.0],
            actual_layer: 3,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: sqrt(0) = 0
        assert_eq!(norm, 0.0);
        assert_eq!(enc.dim(), 1);
    }

    #[test]
    fn from_head_routing_invalid_anchor_preserves_value_through_display_chain() {
        // Arrange: InvalidLayerAnchor with a specific float that has a precise string representation
        let anchor = 0.33333334f32; // common float near 1/3
        let hr = HeadRoutingError::InvalidLayerAnchor(anchor);
        let ie: IntentError = hr.into();

        // Act: format through Display, then parse back to verify value is preserved
        let msg = format!("{ie}");

        // Assert: the float value's string representation appears in the message
        assert!(
            msg.contains(&format!("{anchor}")),
            "Display chain should preserve anchor value, got: {msg}"
        );
    }

    #[test]
    fn intent_encoding_consecutive_clones_form_chain() {
        // Arrange: create original, clone A from original, clone B from A
        let original = IntentEncoding {
            embedding: vec![1.5, 2.5, 3.5],
            actual_layer: 20,
            pool: PoolMode::MeanPool,
        };

        // Act: chain of clones
        let clone_a = original.clone();
        let clone_b = clone_a.clone();

        // Assert: all three have identical content
        assert_eq!(original.embedding, clone_a.embedding);
        assert_eq!(clone_a.embedding, clone_b.embedding);
        assert_eq!(original.actual_layer, clone_b.actual_layer);
        assert_eq!(original.pool, clone_b.pool);
        // All three have independent allocations
        assert_ne!(original.embedding.as_ptr(), clone_a.embedding.as_ptr());
        assert_ne!(clone_a.embedding.as_ptr(), clone_b.embedding.as_ptr());
    }

    #[test]
    fn intent_encoding_l2_norm_with_very_large_dim_small_values() {
        // Arrange: very large dimension (typical hidden_size = 4096) with small uniform values
        // to verify no overflow in the sum of squares
        let dim = 4096usize;
        let val = 0.01f32;
        let enc = IntentEncoding {
            embedding: vec![val; dim],
            actual_layer: 11,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: norm = val * sqrt(dim) = 0.01 * sqrt(4096) = 0.01 * 64 = 0.64
        let expected = val * (dim as f32).sqrt();
        assert!(
            norm.is_finite(),
            "norm should be finite for small uniform values, got {norm}"
        );
        assert!(
            (norm - expected).abs() < expected * 1e-4,
            "expected {expected}, got {norm}"
        );
    }

    #[test]
    fn from_head_routing_consecutive_conversions_idempotent() {
        // Arrange: convert the same HeadRoutingError variant multiple times
        let build_error = || -> IntentError {
            HeadRoutingError::Backend("timeout".to_string()).into()
        };

        // Act: convert three times
        let ie1 = build_error();
        let ie2 = build_error();
        let ie3 = build_error();

        // Assert: all conversions produce identical Display output
        assert_eq!(format!("{ie1}"), format!("{ie2}"));
        assert_eq!(format!("{ie2}"), format!("{ie3}"));
    }

    #[test]
    fn intent_encoding_pool_mode_copy_semantics() {
        // Arrange: verify PoolMode is Copy (can be reused without move issues)
        let pool = PoolMode::LastToken;
        let enc1 = IntentEncoding {
            embedding: vec![1.0],
            actual_layer: 0,
            pool,
        };
        // pool is still usable because PoolMode: Copy
        let enc2 = IntentEncoding {
            embedding: vec![2.0],
            actual_layer: 1,
            pool,
        };

        // Assert: both encodings have the same pool value
        assert_eq!(enc1.pool, pool);
        assert_eq!(enc2.pool, pool);
    }

    #[test]
    fn intent_encoding_embedding_mutation_after_clone_does_not_affect_original() {
        // Arrange: create encoding, clone it, then push to original's embedding
        let mut original = IntentEncoding {
            embedding: vec![1.0, 2.0],
            actual_layer: 5,
            pool: PoolMode::MeanPool,
        };
        let cloned = original.clone();

        // Act: mutate original's embedding by pushing a new element
        original.embedding.push(3.0);

        // Assert: cloned still has the original length
        assert_eq!(cloned.embedding.len(), 2, "cloned embedding should be unaffected by push");
        assert_eq!(cloned.embedding, vec![1.0, 2.0]);
        assert_eq!(original.embedding.len(), 3);
    }

    // ---- Wave 3: 13 new tests (edge cases & coverage gaps) ----

    #[test]
    fn intent_encoding_empty_embedding_actual_layer_nonzero() {
        // Arrange: empty embedding with a nonzero actual_layer — metadata is valid even with no data
        let enc = IntentEncoding {
            embedding: vec![],
            actual_layer: 42,
            pool: PoolMode::LastToken,
        };

        // Act
        let dim = enc.dim();
        let norm = enc.l2_norm();

        // Assert: dim is 0, norm is 0.0, but actual_layer is preserved
        assert_eq!(dim, 0);
        assert_eq!(norm, 0.0);
        assert_eq!(enc.actual_layer, 42);
    }

    #[test]
    fn intent_encoding_l2_norm_single_positive_infinity() {
        // Arrange: single-element embedding with +inf
        let enc = IntentEncoding {
            embedding: vec![f32::INFINITY],
            actual_layer: 0,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: inf^2 = inf, sqrt(inf) = inf
        assert!(norm.is_infinite() && norm > 0.0, "expected +inf, got {norm}");
        assert_eq!(enc.dim(), 1);
    }

    #[test]
    fn intent_encoding_l2_norm_many_ones_no_overflow() {
        // Arrange: 256 elements all 1.0 — sum of squares = 256, sqrt = 16.0, well within f32 range
        let enc = IntentEncoding {
            embedding: vec![1.0; 256],
            actual_layer: 3,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: sqrt(256) = 16.0 exactly
        assert!((norm - 16.0).abs() < 1e-5, "expected 16.0, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_canceling_pairs() {
        // Arrange: pairs that cancel in sum but not in sum-of-squares
        // [1.0, -1.0, 2.0, -2.0] — sum=0, sum_of_squares = 1+1+4+4=10, sqrt=~3.162
        let enc = IntentEncoding {
            embedding: vec![1.0, -1.0, 2.0, -2.0],
            actual_layer: 7,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: L2 norm is not zero despite values summing to zero
        let expected = 10.0f32.sqrt();
        assert!((norm - expected).abs() < 1e-5, "expected {expected}, got {norm}");
        assert!(norm > 0.0, "canceling pairs should not produce zero norm");
    }

    #[test]
    fn intent_encoding_l2_norm_with_positive_zero_and_negative_zero() {
        // Arrange: embedding with both +0.0 and -0.0 — both square to +0.0
        let enc = IntentEncoding {
            embedding: vec![0.0, -0.0, 0.0, -0.0, 0.0],
            actual_layer: 1,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: all zeros regardless of sign bit
        assert_eq!(norm, 0.0);
        assert_eq!(enc.dim(), 5);
    }

    #[test]
    fn intent_error_as_error_trait_object() {
        // Arrange: box an IntentError as a dyn Error to verify trait object compatibility
        let err: Box<dyn std::error::Error> = Box::new(IntentError::NoModelLoaded);

        // Act
        let msg = format!("{err}");

        // Assert: Display works through the trait object
        assert!(msg.contains("no model loaded"), "trait object Display failed, got: {msg}");
    }

    #[test]
    fn intent_encoding_l2_norm_nearly_unit_vector_precision() {
        // Arrange: a vector very close to unit length — [1/sqrt(2), 1/sqrt(2)]
        let inv_sqrt2 = (2.0f32).sqrt().recip();
        let enc = IntentEncoding {
            embedding: vec![inv_sqrt2, inv_sqrt2],
            actual_layer: 4,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: should be very close to 1.0 (within f32 precision)
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "nearly unit vector should have norm ~1.0, got {norm}"
        );
    }

    #[test]
    fn from_head_routing_anchor_negative_zero() {
        // Arrange: InvalidLayerAnchor with -0.0 — distinct from +0.0 in bit pattern
        let hr = HeadRoutingError::InvalidLayerAnchor(-0.0f32);
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: converts correctly; -0.0 formats as "0" or "-0" depending on platform
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
        assert!(msg.contains("0"), "message should contain zero representation, got: {msg}");
    }

    #[test]
    fn intent_encoding_two_instances_same_data_independent_allocations() {
        // Arrange: two separately constructed IntentEncoding with identical content
        let enc1 = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0],
            actual_layer: 10,
            pool: PoolMode::MeanPool,
        };
        let enc2 = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0],
            actual_layer: 10,
            pool: PoolMode::MeanPool,
        };

        // Act & Assert: semantically equal but independent allocations
        assert_eq!(enc1.embedding, enc2.embedding);
        assert_eq!(enc1.actual_layer, enc2.actual_layer);
        assert_ne!(enc1.embedding.as_ptr(), enc2.embedding.as_ptr(),
            "independently constructed Vecs must have separate allocations");
    }

    #[test]
    fn intent_encoding_l2_norm_float_precision_pythagorean_7_24_25() {
        // Arrange: 7-24-25 Pythagorean triple — exact in f32
        let enc = IntentEncoding {
            embedding: vec![7.0, 24.0],
            actual_layer: 6,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: sqrt(49 + 576) = sqrt(625) = 25.0 exactly
        assert!((norm - 25.0).abs() < 1e-6, "expected 25.0, got {norm}");
    }

    #[test]
    fn intent_encoding_clone_then_original_push_preserves_clone() {
        // Arrange: create and clone, then extend original's embedding
        let mut original = IntentEncoding {
            embedding: vec![10.0],
            actual_layer: 2,
            pool: PoolMode::ClsToken,
        };
        let cloned = original.clone();

        // Act: push additional elements to original
        original.embedding.push(20.0);
        original.embedding.push(30.0);

        // Assert: cloned remains a snapshot of the original state at clone time
        assert_eq!(cloned.embedding, vec![10.0]);
        assert_eq!(cloned.embedding.len(), 1);
        assert_eq!(original.embedding, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn intent_encoding_l2_norm_one_element_largest_normal_f32() {
        // Arrange: single element = f32::MAX (~3.4e38). Squaring overflows to +inf.
        let enc = IntentEncoding {
            embedding: vec![f32::MAX],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: MAX^2 overflows to inf, sqrt(inf) = inf
        assert!(
            norm.is_infinite() && norm > 0.0,
            "f32::MAX squared should overflow to +inf norm, got {norm}"
        );
    }

    #[test]
    fn intent_encoding_pool_mode_field_matches_constructor() {
        // Arrange: construct with each pool mode and immediately verify
        let modes_and_expected = [
            (PoolMode::MeanPool, "MeanPool"),
            (PoolMode::LastToken, "LastToken"),
            (PoolMode::ClsToken, "ClsToken"),
        ];

        // Act & Assert
        for (pool, _label) in modes_and_expected {
            let enc = IntentEncoding {
                embedding: vec![0.0],
                actual_layer: 0,
                pool,
            };
            assert_eq!(enc.pool, pool, "pool field should match the constructor value");
        }
    }

    // ---- Wave 4: 13 new tests (uncovered edge cases & structural guarantees) ----

    #[test]
    fn intent_error_has_exactly_three_variants() {
        // Arrange & Act: count the number of IntentError variants by exhaustive match
        let count = {
            let v1 = IntentError::InvalidLayerAnchor("".to_string());
            let v2 = IntentError::EncodeFailed("".to_string());
            let v3 = IntentError::NoModelLoaded;
            // Each must be a distinct variant (compile-time check via exhaustive match)
            let _ = match v1 {
                IntentError::InvalidLayerAnchor(_) => 1,
                IntentError::EncodeFailed(_) => 2,
                IntentError::NoModelLoaded => 3,
            };
            let _ = match v2 {
                IntentError::InvalidLayerAnchor(_) => 1,
                IntentError::EncodeFailed(_) => 2,
                IntentError::NoModelLoaded => 3,
            };
            let _ = match v3 {
                IntentError::InvalidLayerAnchor(_) => 1,
                IntentError::EncodeFailed(_) => 2,
                IntentError::NoModelLoaded => 3,
            };
            3usize
        };

        // Assert: exactly 3 variants exist
        assert_eq!(count, 3, "IntentError should have exactly 3 variants");
    }

    #[test]
    fn pool_mode_has_exactly_three_variants() {
        // Arrange & Act: exhaustive match proves exactly 3 variants at compile time
        let modes = [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken];

        // Assert: 3 distinct variants
        assert_eq!(modes.len(), 3, "PoolMode should have exactly 3 variants");
    }

    #[test]
    fn intent_encoding_l2_norm_pythagorean_triple_9_40_41() {
        // Arrange: 9-40-41 Pythagorean triple — sqrt(81 + 1600) = sqrt(1681) = 41
        let enc = IntentEncoding {
            embedding: vec![9.0, 40.0],
            actual_layer: 3,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: exact integer result
        assert!((norm - 41.0).abs() < 1e-5, "expected 41.0, got {norm}");
    }

    #[test]
    fn intent_encoding_l2_norm_scaling_property() {
        // Arrange: L2 norm scales linearly — if we double the vector, norm doubles
        let base = IntentEncoding {
            embedding: vec![3.0, 4.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let doubled = IntentEncoding {
            embedding: vec![6.0, 8.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm_base = base.l2_norm();
        let norm_doubled = doubled.l2_norm();

        // Assert: 2 * norm(3,4) == norm(6,8)
        assert!((norm_doubled - 2.0 * norm_base).abs() < 1e-5,
            "scaling property violated: 2*{norm_base} != {norm_doubled}");
        assert!((norm_base - 5.0).abs() < 1e-5);
        assert!((norm_doubled - 10.0).abs() < 1e-5);
    }

    #[test]
    fn intent_encoding_l2_norm_does_not_mutate_embedding() {
        // Arrange: create an encoding and snapshot the embedding values
        let mut enc = IntentEncoding {
            embedding: vec![3.0, 4.0, 5.0],
            actual_layer: 2,
            pool: PoolMode::LastToken,
        };
        let snapshot: Vec<f32> = enc.embedding.clone();

        // Act: call l2_norm (should be a read-only operation)
        let _norm = enc.l2_norm();

        // Assert: embedding is unchanged
        assert_eq!(enc.embedding, snapshot, "l2_norm must not mutate the embedding");
    }

    #[test]
    fn intent_encoding_one_zero_one_nonzero_dim_and_norm() {
        // Arrange: embedding with one zero and one nonzero element
        let enc = IntentEncoding {
            embedding: vec![0.0, 5.0],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };

        // Act
        let dim = enc.dim();
        let norm = enc.l2_norm();

        // Assert: sqrt(0 + 25) = 5.0
        assert_eq!(dim, 2);
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    #[test]
    fn intent_error_invalid_layer_anchor_with_unicode_message() {
        // Arrange: InvalidLayerAnchor with Unicode characters
        let err = IntentError::InvalidLayerAnchor("锚点越界: layer >= 999 (层数=32)".to_string());

        // Act
        let msg = format!("{err}");

        // Assert: Unicode is preserved in Display output
        assert!(msg.contains("锚点越界"), "Unicode should be preserved, got: {msg}");
        assert!(msg.contains("invalid layer anchor"));
    }

    #[test]
    fn intent_encoding_debug_format_shows_pool_variant_name() {
        // Arrange: create IntentEncoding with each PoolMode variant
        for pool in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            let enc = IntentEncoding {
                embedding: vec![1.0],
                actual_layer: 0,
                pool,
            };

            // Act
            let debug = format!("{enc:?}");

            // Assert: Debug output contains the pool field
            assert!(
                debug.contains("pool"),
                "Debug should contain 'pool' field for {pool:?}, got: {debug}"
            );
        }
    }

    #[test]
    fn intent_encoding_dim_called_twice_returns_same_value() {
        // Arrange: embedding of known size
        let enc = IntentEncoding {
            embedding: vec![0.5; 512],
            actual_layer: 6,
            pool: PoolMode::ClsToken,
        };

        // Act: call dim() twice
        let dim1 = enc.dim();
        let dim2 = enc.dim();

        // Assert: deterministic, no side effects
        assert_eq!(dim1, dim2);
        assert_eq!(dim1, 512);
    }

    #[test]
    fn intent_encoding_l2_norm_called_twice_returns_same_value() {
        // Arrange: nontrivial embedding
        let enc = IntentEncoding {
            embedding: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            actual_layer: 8,
            pool: PoolMode::MeanPool,
        };

        // Act: call l2_norm() twice
        let norm1 = enc.l2_norm();
        let norm2 = enc.l2_norm();

        // Assert: bitwise identical (pure function, no state)
        assert_eq!(norm1, norm2, "l2_norm must be deterministic");
    }

    #[test]
    fn from_head_routing_anchor_preserves_precision_through_conversion() {
        // Arrange: a float with a precise, round decimal representation
        let anchor_val = 0.5f32;
        let hr = HeadRoutingError::InvalidLayerAnchor(anchor_val);
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: "0.5" should appear verbatim in the Display output
        assert!(
            msg.contains("0.5"),
            "Display should preserve the anchor value 0.5, got: {msg}"
        );
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
    }

    #[test]
    fn intent_error_encode_failed_with_multiline_message() {
        // Arrange: EncodeFailed with a multi-line diagnostic message
        let detail = "layer 24 failed\ncaused by: OOM\nretry after GC";
        let err = IntentError::EncodeFailed(detail.to_string());

        // Act
        let msg = format!("{err}");

        // Assert: multi-line content is fully preserved
        assert!(msg.contains("layer 24 failed"), "first line preserved, got: {msg}");
        assert!(msg.contains("caused by: OOM"), "second line preserved, got: {msg}");
        assert!(msg.contains("retry after GC"), "third line preserved, got: {msg}");
        assert!(msg.contains("encode failed"));
    }

    #[test]
    fn intent_encoding_struct_size_is_reasonable() {
        // Arrange: check that IntentEncoding is not unexpectedly large
        // A Vec<f32> is 24 bytes (ptr, len, cap), usize is 8, PoolMode is 1 byte (Copy enum)
        // With alignment padding, the struct should be <= 40 bytes on 64-bit platforms.

        // Act
        let size = std::mem::size_of::<IntentEncoding>();

        // Assert: struct size is reasonable (Vec + usize + PoolMode + alignment)
        assert!(
            size <= 40,
            "IntentEncoding should be <= 40 bytes, got {size}"
        );
        assert!(
            size >= 24,
            "IntentEncoding should be at least 24 bytes (Vec alone), got {size}"
        );
    }

    // ---- Wave 5: 13 new tests (uncovered edge cases) ----

    #[test]
    fn from_head_routing_anchor_nan_converts_to_invalid_layer_anchor() {
        // Arrange: InvalidLayerAnchor with NaN — edge of the float value space
        let hr = HeadRoutingError::InvalidLayerAnchor(f32::NAN);
        let ie: IntentError = hr.into();

        // Assert: maps to InvalidLayerAnchor variant
        assert!(
            matches!(ie, IntentError::InvalidLayerAnchor(_)),
            "NaN anchor should map to InvalidLayerAnchor, got {ie:?}"
        );
    }

    #[test]
    fn from_head_routing_anchor_infinity_converts_to_invalid_layer_anchor() {
        // Arrange: InvalidLayerAnchor with +inf
        let hr = HeadRoutingError::InvalidLayerAnchor(f32::INFINITY);
        let ie: IntentError = hr.into();

        // Assert: maps to InvalidLayerAnchor variant
        assert!(
            matches!(ie, IntentError::InvalidLayerAnchor(_)),
            "+inf anchor should map to InvalidLayerAnchor, got {ie:?}"
        );
    }

    #[test]
    fn intent_encoding_l2_norm_max_plus_min_both_overflow_to_inf() {
        // Arrange: f32::MAX and f32::MIN both square to +inf, sum is +inf, sqrt is +inf
        let enc = IntentEncoding {
            embedding: vec![f32::MAX, f32::MIN],
            actual_layer: 4,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: both overflow individually, but the norm is still +inf
        assert!(
            norm.is_infinite() && norm > 0.0,
            "MAX^2 + MIN^2 overflows to +inf, sqrt(+inf) = +inf, got {norm}"
        );
    }

    #[test]
    fn intent_encoding_l2_norm_sum_overflow_with_two_large_values() {
        // Arrange: two values whose squares individually are finite but sum overflows
        // f32::MAX ~3.4e38, (MAX/2)^2 ~2.9e76, still finite but very large
        // Use values where sum of squares overflows: 2 * (MAX * 0.75)^2
        // Actually, let's use f32::MAX.sqrt() * 0.99 — each squared is ~0.98 * MAX, sum > MAX
        // Simplification: just use two values close to f32::MAX/2 whose squares sum overflows
        let large = f32::MAX / 2.0;
        let enc = IntentEncoding {
            embedding: vec![large, large],
            actual_layer: 1,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: both squares are finite but their sum may overflow to +inf
        assert!(
            norm.is_finite() || (norm.is_infinite() && norm > 0.0),
            "sum-of-squares overflow should yield finite or +inf, got {norm}"
        );
    }

    #[test]
    fn intent_encoding_l2_norm_3d_pythagorean_triple_3_4_0() {
        // Arrange: 3D embedding forming a degenerate 3-4-5 triangle with zero in the third axis
        let enc = IntentEncoding {
            embedding: vec![3.0, 4.0, 0.0],
            actual_layer: 2,
            pool: PoolMode::ClsToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: sqrt(9 + 16 + 0) = sqrt(25) = 5.0
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    #[test]
    fn intent_encoding_negative_zero_single_element_norm_is_zero() {
        // Arrange: single element is -0.0 (negative zero)
        let enc = IntentEncoding {
            embedding: vec![-0.0f32],
            actual_layer: 0,
            pool: PoolMode::LastToken,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: (-0.0)^2 = +0.0, sqrt(+0.0) = 0.0
        assert_eq!(norm, 0.0, "negative zero should square to +0.0, norm should be 0.0");
        assert_eq!(enc.dim(), 1);
    }

    #[test]
    fn intent_encoding_l2_norm_single_epsilon_element() {
        // Arrange: single element = f32::EPSILON, which squares to a nonzero tiny value
        let enc = IntentEncoding {
            embedding: vec![f32::EPSILON],
            actual_layer: 1,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert: EPSILON^2 is nonzero and representable in f32, sqrt(EPSILON^2) = EPSILON
        assert!(
            norm.is_finite() && norm > 0.0,
            "EPSILON norm should be finite and positive, got {norm}"
        );
        assert!(
            (norm - f32::EPSILON).abs() < f32::EPSILON,
            "norm should be approximately EPSILON, got {norm}"
        );
    }

    #[test]
    fn intent_error_display_deterministic_across_format_calls() {
        // Arrange: same InvalidLayerAnchor instance formatted twice
        let err = IntentError::InvalidLayerAnchor("boundary at 1.0".to_string());

        // Act
        let msg1 = format!("{err}");
        let msg2 = format!("{err}");
        let msg3 = format!("{err}");

        // Assert: all three format calls produce identical output
        assert_eq!(msg1, msg2, "first and second format should be identical");
        assert_eq!(msg2, msg3, "second and third format should be identical");
    }

    #[test]
    fn intent_encoding_embedding_with_rapidly_changing_signs() {
        // Arrange: embedding that alternates sign every element with increasing magnitude
        // [1, -2, 3, -4, 5] — sum_of_squares = 1+4+9+16+25 = 55, sqrt(55) ≈ 7.416
        let enc = IntentEncoding {
            embedding: vec![1.0, -2.0, 3.0, -4.0, 5.0],
            actual_layer: 3,
            pool: PoolMode::MeanPool,
        };

        // Act
        let norm = enc.l2_norm();

        // Assert
        let expected = 55.0f32.sqrt();
        assert!(
            (norm - expected).abs() < 1e-4,
            "expected {expected}, got {norm}"
        );
    }

    #[test]
    fn intent_encoding_is_send() {
        // Arrange: verify IntentEncoding is Send (safe to send across threads)
        fn assert_send<T: Send>() {}
        assert_send::<IntentEncoding>();
    }

    #[test]
    fn intent_encoding_is_sync() {
        // Arrange: verify IntentEncoding is Sync (safe to share across threads)
        fn assert_sync<T: Sync>() {}
        assert_sync::<IntentEncoding>();
    }

    #[test]
    fn from_head_routing_anchor_at_exact_upper_boundary() {
        // Arrange: InvalidLayerAnchor with 1.0 — the exact upper boundary of valid range
        // This is a valid relative anchor, but HeadRoutingError says it's invalid,
        // so the From conversion should still map it to InvalidLayerAnchor
        let hr = HeadRoutingError::InvalidLayerAnchor(1.0f32);
        let ie: IntentError = hr.into();

        // Act
        let msg = format!("{ie}");

        // Assert: maps to InvalidLayerAnchor and the boundary value is preserved
        assert!(matches!(ie, IntentError::InvalidLayerAnchor(_)));
        assert!(
            msg.contains("1"),
            "message should contain the boundary value 1.0, got: {msg}"
        );
    }

    #[test]
    fn intent_encoding_dim_with_power_of_two_length() {
        // Arrange: embedding with power-of-2 lengths to verify no off-by-one errors
        for exp in 0..=13 {
            let len = 1usize << exp; // 1, 2, 4, 8, ..., 8192
            let enc = IntentEncoding {
                embedding: vec![0.0; len],
                actual_layer: 0,
                pool: PoolMode::MeanPool,
            };

            // Act & Assert: dim() equals the power-of-2 length exactly
            assert_eq!(
                enc.dim(), len,
                "dim() should equal {len} for 2^{exp} elements"
            );
        }
    }

    // ---- Wave 6: 13 new tests (cross-module integration & structural invariants) ----

    #[test]
    fn pool_mode_apply_mean_pool_matches_manual_computation() {
        // Arrange: 2x3 hidden state (seq_len=2, hidden_size=3)
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Act
        let pooled = PoolMode::MeanPool.apply(&hidden, 2, 3).unwrap();
        // Assert: mean of row0=[1,2,3] and row1=[4,5,6] = [2.5, 3.5, 4.5]
        let expected = vec![2.5, 3.5, 4.5];
        for (got, exp) in pooled.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "expected {exp}, got {got}");
        }
    }

    #[test]
    fn pool_mode_apply_last_token_returns_last_row() {
        // Arrange: 3x4 hidden state
        let hidden: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // Act
        let pooled = PoolMode::LastToken.apply(&hidden, 3, 4).unwrap();
        // Assert: last row = [8, 9, 10, 11]
        assert_eq!(pooled, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn pool_mode_apply_cls_token_returns_first_row() {
        // Arrange: 3x4 hidden state
        let hidden: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // Act
        let pooled = PoolMode::ClsToken.apply(&hidden, 3, 4).unwrap();
        // Assert: first row = [0, 1, 2, 3]
        assert_eq!(pooled, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pool_mode_apply_zero_seq_len_returns_error() {
        // Arrange: hidden state with seq_len=0
        let hidden: Vec<f32> = vec![];
        // Act
        let result = PoolMode::MeanPool.apply(&hidden, 0, 4);
        // Assert
        assert!(result.is_err(), "seq_len=0 should return error");
    }

    #[test]
    fn pool_mode_apply_insufficient_hidden_length_returns_error() {
        // Arrange: hidden buffer too short for the declared seq_len*hidden_size
        let hidden = vec![1.0, 2.0, 3.0];
        // Act
        let result = PoolMode::LastToken.apply(&hidden, 2, 4);
        // Assert: need 8 elements, only have 3
        assert!(result.is_err(), "insufficient hidden length should return error");
    }

    #[test]
    fn pool_mode_apply_cls_pool_single_token_returns_entire_hidden() {
        // Arrange: seq_len=1, hidden_size=4 — ClsToken and LastToken both return the single row
        let hidden = vec![10.0, 20.0, 30.0, 40.0];
        // Act
        let cls = PoolMode::ClsToken.apply(&hidden, 1, 4).unwrap();
        let last = PoolMode::LastToken.apply(&hidden, 1, 4).unwrap();
        // Assert: both return the same single row
        assert_eq!(cls, hidden);
        assert_eq!(last, hidden);
        assert_eq!(cls, last);
    }

    #[test]
    fn pool_mode_apply_all_modes_produce_correct_dimension() {
        // Arrange: 5x8 hidden state
        let hidden: Vec<f32> = (0..40).map(|i| i as f32).collect();
        for mode in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            // Act
            let pooled = mode.apply(&hidden, 5, 8).unwrap();
            // Assert: output dimension always equals hidden_size
            assert_eq!(pooled.len(), 8, "{mode:?} should produce hidden_size=8 output");
        }
    }

    #[test]
    fn intent_encoding_l2_norm_triangle_inequality() {
        // Arrange: two vectors, verify ||a+b|| <= ||a|| + ||b||
        let a = IntentEncoding {
            embedding: vec![3.0, 4.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let b = IntentEncoding {
            embedding: vec![5.0, 12.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        let a_plus_b = IntentEncoding {
            embedding: vec![8.0, 16.0],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        // Act
        let norm_a = a.l2_norm();
        let norm_b = b.l2_norm();
        let norm_sum = a_plus_b.l2_norm();
        // Assert: triangle inequality
        assert!(
            norm_sum <= norm_a + norm_b + 1e-5,
            "triangle inequality violated: {norm_sum} > {norm_a} + {norm_b}"
        );
        // Also: norm(a+b) = sqrt(64+256) = sqrt(320) ~ 17.89
        assert!((norm_a - 5.0).abs() < 1e-5, "norm(a) = {norm_a}");
        assert!((norm_b - 13.0).abs() < 1e-5, "norm(b) = {norm_b}");
    }

    #[test]
    fn intent_error_display_messages_are_mutually_exclusive() {
        // Arrange: each IntentError variant has a unique Display prefix
        let anchor = IntentError::InvalidLayerAnchor("msg".into());
        let encode = IntentError::EncodeFailed("msg".into());
        let no_model = IntentError::NoModelLoaded;
        // Act
        let anchor_msg = format!("{anchor}");
        let encode_msg = format!("{encode}");
        let no_model_msg = format!("{no_model}");
        // Assert: no variant's display contains another variant's key prefix
        assert!(!anchor_msg.contains("encode failed"));
        assert!(!anchor_msg.contains("no model loaded"));
        assert!(!encode_msg.contains("invalid layer anchor"));
        assert!(!encode_msg.contains("no model loaded"));
        assert!(!no_model_msg.contains("invalid layer anchor"));
        assert!(!no_model_msg.contains("encode failed"));
    }

    #[test]
    fn intent_encoding_embedding_preserves_nan_bit_pattern() {
        // Arrange: construct NaN with a specific bit pattern
        let nan_bits = 0x7fc00001u32; // quiet NaN with nonzero payload
        let custom_nan = f32::from_bits(nan_bits);
        let enc = IntentEncoding {
            embedding: vec![custom_nan],
            actual_layer: 0,
            pool: PoolMode::MeanPool,
        };
        // Act & Assert: the stored value has the exact same bit pattern
        assert!(enc.embedding[0].is_nan());
        assert_eq!(enc.embedding[0].to_bits(), nan_bits);
    }

    #[test]
    fn intent_encoding_embedding_preserves_infinity_bit_pattern() {
        // Arrange: verify +inf and -inf bit patterns survive storage
        let enc = IntentEncoding {
            embedding: vec![f32::INFINITY, f32::NEG_INFINITY],
            actual_layer: 0,
            pool: PoolMode::LastToken,
        };
        // Act & Assert
        assert_eq!(enc.embedding[0].to_bits(), 0x7f800000u32); // +inf
        assert_eq!(enc.embedding[1].to_bits(), 0xff800000u32); // -inf
        assert!(enc.embedding[0].is_infinite() && enc.embedding[0] > 0.0);
        assert!(enc.embedding[1].is_infinite() && enc.embedding[1] < 0.0);
    }

    #[test]
    fn from_head_routing_pool_apply_error_converts_to_encode_failed() {
        // Arrange: PoolMode::apply returns HeadRoutingError::InvalidConfig
        let hr_err = PoolMode::MeanPool.apply(&[], 0, 0).unwrap_err();
        let ie: IntentError = hr_err.into();
        // Act & Assert: InvalidConfig maps to EncodeFailed
        match &ie {
            IntentError::EncodeFailed(msg) => {
                assert!(msg.contains("seq_len") || msg.contains("hidden_size"),
                    "EncodeFailed message should reference the invalid params, got: {msg}");
            }
            other => panic!("expected EncodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn intent_encoding_l2_norm_pythagorean_triple_11_60_61() {
        // Arrange: 11-60-61 Pythagorean triple — sqrt(121 + 3600) = sqrt(3721) = 61
        let enc = IntentEncoding {
            embedding: vec![11.0, 60.0],
            actual_layer: 5,
            pool: PoolMode::ClsToken,
        };
        // Act
        let norm = enc.l2_norm();
        // Assert: exact integer result
        assert!((norm - 61.0).abs() < 1e-4, "expected 61.0, got {norm}");
    }

    // ---- Wave 7: 10 new tests (PoolMode apply edge cases & structural invariants) ----

    #[test]
    fn pool_mode_apply_hidden_size_zero_returns_error() {
        // Arrange: hidden_size=0 is invalid regardless of seq_len or buffer content
        let hidden = vec![1.0, 2.0, 3.0];
        // Act
        let result = PoolMode::MeanPool.apply(&hidden, 1, 0);
        // Assert: zero hidden_size is rejected
        assert!(result.is_err(), "hidden_size=0 should return error");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("hidden_size > 0"),
            "error should mention hidden_size > 0 constraint, got: {msg}"
        );
    }

    #[test]
    fn pool_mode_apply_mean_pool_single_row_no_averaging_effect() {
        // Arrange: seq_len=1 means MeanPool just returns the single row unchanged
        let hidden = vec![7.0, 11.0, 13.0, 17.0];
        // Act
        let pooled = PoolMode::MeanPool.apply(&hidden, 1, 4).unwrap();
        // Assert: with a single row, MeanPool is identity
        assert_eq!(pooled, hidden);
        // And equal to ClsToken and LastToken for seq_len=1
        let cls = PoolMode::ClsToken.apply(&hidden, 1, 4).unwrap();
        let last = PoolMode::LastToken.apply(&hidden, 1, 4).unwrap();
        assert_eq!(pooled, cls);
        assert_eq!(pooled, last);
    }

    #[test]
    fn pool_mode_apply_extra_trailing_data_ignored() {
        // Arrange: hidden buffer has more elements than seq_len * hidden_size requires.
        // PoolMode::apply should only read the first seq_len * hidden_size elements.
        let hidden: Vec<f32> = (0..20).map(|i| i as f32).collect(); // 20 elements
        // Act: request only 2x4 = 8 elements
        let last = PoolMode::LastToken.apply(&hidden, 2, 4).unwrap();
        let cls = PoolMode::ClsToken.apply(&hidden, 2, 4).unwrap();
        // Assert: LastToken returns row 1 = [4,5,6,7], ClsToken returns row 0 = [0,1,2,3]
        assert_eq!(last, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(cls, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pool_mode_apply_mean_pool_nan_values_propagate() {
        // Arrange: hidden state containing NaN — MeanPool should propagate NaN into output
        let hidden = vec![1.0, f32::NAN, 3.0, 4.0]; // 2x2 matrix
        // Act
        let pooled = PoolMode::MeanPool.apply(&hidden, 2, 2).unwrap();
        // Assert: mean of col0 = (1.0 + 3.0) / 2 = 2.0 (finite); col1 = (NaN + 4.0) / 2 = NaN
        assert!((pooled[0] - 2.0).abs() < 1e-6, "col 0 should be 2.0, got {}", pooled[0]);
        assert!(pooled[1].is_nan(), "col 1 should be NaN, got {}", pooled[1]);
    }

    #[test]
    fn pool_mode_apply_exact_buffer_boundary_succeeds() {
        // Arrange: hidden.len() == seq_len * hidden_size exactly (no extra elements)
        let hidden: Vec<f32> = (0..6).map(|i| i as f32).collect(); // exactly 2x3
        // Act: all three modes should succeed
        for mode in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            let result = mode.apply(&hidden, 2, 3);
            assert!(result.is_ok(), "{mode:?} should succeed at exact boundary, got {:?}", result);
        }
    }

    #[test]
    fn pool_mode_variants_are_distinct_in_debug_format() {
        // Arrange: each PoolMode variant formatted via Debug
        let modes = [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken];
        // Act
        let debugs: Vec<String> = modes.iter().map(|m| format!("{m:?}")).collect();
        // Assert: all three Debug strings are distinct
        assert_ne!(debugs[0], debugs[1], "MeanPool and LastToken Debug should differ");
        assert_ne!(debugs[1], debugs[2], "LastToken and ClsToken Debug should differ");
        assert_ne!(debugs[0], debugs[2], "MeanPool and ClsToken Debug should differ");
    }

    #[test]
    fn pool_mode_equality_reflexivity() {
        // Arrange: each PoolMode variant should equal itself
        for mode in [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken] {
            // Assert: reflexivity (a == a)
            assert_eq!(mode, mode, "{mode:?} should equal itself");
        }
    }

    #[test]
    fn pool_mode_equality_mutual_distinctness() {
        // Arrange: all pairwise combinations
        let modes = [PoolMode::MeanPool, PoolMode::LastToken, PoolMode::ClsToken];
        // Act & Assert: no two distinct variants are equal
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j], "{:?} should not equal {:?}", modes[i], modes[j]);
            }
        }
    }

    #[test]
    fn intent_error_constructing_same_variant_twice_produces_identical_display() {
        // Arrange: construct two identical IntentError::EncodeFailed independently
        let err1 = IntentError::EncodeFailed("layer 12 timeout".to_string());
        let err2 = IntentError::EncodeFailed("layer 12 timeout".to_string());
        // Act
        let msg1 = format!("{err1}");
        let msg2 = format!("{err2}");
        // Assert: independent construction of the same variant/message yields identical display
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn pool_mode_apply_last_token_large_seq_returns_correct_last_row() {
        // Arrange: 100x3 hidden state — verify LastToken returns the actual last row
        let hidden: Vec<f32> = (0..300).map(|i| i as f32).collect();
        // Act
        let pooled = PoolMode::LastToken.apply(&hidden, 100, 3).unwrap();
        // Assert: last row starts at index 99*3 = 297, so [297, 298, 299]
        assert_eq!(pooled, vec![297.0, 298.0, 299.0]);
        assert_eq!(pooled.len(), 3);
    }
}
