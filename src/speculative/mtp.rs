//! MTP (Multi-Token Prediction) — 模型内置多 token 预测
//!
//! 某些模型 (DeepSeek V3, Qwen3) 在训练时使用 MTP loss，
//! 推理时可一次前向输出 K 个未来 token 的 logits。
//! 相当于内置 draft model，不需要额外权重。

use crate::engine::executor::BackendError;

/// MTP Head 配置
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MtpConfig {
    /// 预测深度 (通常 2-4)
    pub depth: usize,
    /// 词表大小
    pub vocab_size: usize,
    /// 隐藏维度
    pub hidden_size: usize,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            vocab_size: 0,
            hidden_size: 0,
        }
    }
}

/// MTP Head 权重 (每个 depth 一组投影权重)
#[derive(Debug, Clone, PartialEq)]
pub struct MtpHead {
    /// [depth] 个投影层，每个 [hidden_size, vocab_size]
    pub projections: Vec<Vec<f32>>,
    pub config: MtpConfig,
}

/// MTP draft: 一次前向输出 K 个 token 的 logits
///
/// 对每个 depth 级别，将 hidden_state 通过对应的投影矩阵映射到 vocab 空间:
///   logits[k][v] = sum_h(hidden[h] * projections[k][v * hidden_size + h])
///
/// 返回 `depth` 个 logits 向量，每个长度为 `vocab_size`。
pub fn mtp_draft(
    hidden_state: &[f32],
    head: &MtpHead,
) -> Result<Vec<Vec<f32>>, BackendError> {
    let hidden_size = head.config.hidden_size;
    let vocab_size = head.config.vocab_size;

    if hidden_size == 0 || vocab_size == 0 {
        return Err(BackendError::Other(
            "MTP config has zero hidden_size or vocab_size".into(),
        ));
    }
    if hidden_state.len() != hidden_size {
        return Err(BackendError::Other(format!(
            "hidden_state length {} != config.hidden_size {}",
            hidden_state.len(),
            hidden_size
        )));
    }
    if head.projections.len() != head.config.depth {
        return Err(BackendError::Other(format!(
            "projections count {} != config.depth {}",
            head.projections.len(),
            head.config.depth
        )));
    }

    let projected_elems = hidden_size * vocab_size;
    for (k, proj) in head.projections.iter().enumerate() {
        if proj.len() != projected_elems {
            return Err(BackendError::Other(format!(
                "projection[{}] length {} != hidden_size({}) * vocab_size({})",
                k,
                proj.len(),
                hidden_size,
                vocab_size
            )));
        }
    }

    // hidden_state @ projection^T for each depth level
    Ok(head
        .projections
        .iter()
        .map(|proj| {
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let base = v * hidden_size;
                let mut sum = 0.0f32;
                for h in 0..hidden_size {
                    sum += hidden_state[h] * proj[base + h];
                }
                logits[v] = sum;
            }
            logits
        })
        .collect())
}

/// 从 MTP logits 中提取 top-1 候选 token 序列
///
/// # Panics
///
/// - If any depth level has empty logits (len=0). MTP must produce at least
///   1 candidate per depth; empty logits always indicates a bug.
/// - If any logit contains NaN, which indicates severe numerical error
pub fn mtp_candidates(mtp_logits: &[Vec<f32>]) -> Vec<u32> {
    mtp_logits
        .iter()
        .enumerate()
        .map(|(depth, logits)| {
            if logits.iter().any(|l| l.is_nan()) {
                panic!(
                    "NaN in logits at MTP depth {} — numerical error in speculative decoding, cannot select token",
                    depth
                );
            }
            if logits.is_empty() {
                panic!(
                    "mtp_candidates: empty logits at depth {} — MTP must produce at least 1 candidate per depth",
                    depth
                );
            }
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .expect("max_by on non-empty iterator always returns Some")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_config_default() {
        let cfg = MtpConfig::default();
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.vocab_size, 0);
        assert_eq!(cfg.hidden_size, 0);
    }

    #[test]
    fn test_mtp_draft_rejects_zero_config() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig::default(),
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_rejects_hidden_size_mismatch() {
        let head = MtpHead {
            projections: vec![vec![0.0; 6]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0, 3.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_rejects_projection_count_mismatch() {
        let head = MtpHead {
            projections: vec![vec![0.0; 6]],
            config: MtpConfig {
                depth: 3,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_rejects_wrong_projection_shape() {
        let head = MtpHead {
            projections: vec![vec![0.0; 4]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_single_depth_identity() {
        // projection = identity [1, 0; 0, 1], hidden = [3.0, 5.0]
        // logits[0] = 3*1 + 5*0 = 3, logits[1] = 3*0 + 5*1 = 5
        let head = MtpHead {
            projections: vec![vec![1.0, 0.0, 0.0, 1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[3.0, 5.0], &head).unwrap();
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 2);
        assert!((logits[0][0] - 3.0).abs() < 1e-5);
        assert!((logits[0][1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_multi_depth() {
        // depth=2, vocab_size=3, hidden_size=2
        // projection[0] = [[1,0],[0,1],[1,1]] → logits = [h0, h1, h0+h1]
        // projection[1] = [[0,1],[1,0],[2,0]] → logits = [h1, h0, 2*h0]
        let head = MtpHead {
            projections: vec![
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
            ],
            config: MtpConfig {
                depth: 2,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let hidden = [2.0, 3.0];
        let logits = mtp_draft(&hidden, &head).unwrap();
        assert_eq!(logits.len(), 2);

        // depth 0: [2*1+3*0, 2*0+3*1, 2*1+3*1] = [2, 3, 5]
        assert!((logits[0][0] - 2.0).abs() < 1e-5);
        assert!((logits[0][1] - 3.0).abs() < 1e-5);
        assert!((logits[0][2] - 5.0).abs() < 1e-5);

        // depth 1: [2*0+3*1, 2*1+3*0, 2*2+3*0] = [3, 2, 4]
        assert!((logits[1][0] - 3.0).abs() < 1e-5);
        assert!((logits[1][1] - 2.0).abs() < 1e-5);
        assert!((logits[1][2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_argmax_end_to_end() {
        // depth=2, vocab_size=4, hidden_size=2
        // projection[0] designed so argmax=2, projection[1] so argmax=0
        let head = MtpHead {
            projections: vec![
                vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            config: MtpConfig {
                depth: 2,
                vocab_size: 4,
                hidden_size: 2,
            },
        };
        let hidden = [1.0, 1.0];
        let logits = mtp_draft(&hidden, &head).unwrap();
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![2, 0]);
    }

    #[test]
    fn test_mtp_candidates_argmax() {
        let logits = vec![
            vec![0.1, 0.9, 0.2],  // argmax = 1
            vec![0.5, 0.3, 0.8],  // argmax = 2
            vec![1.0, 0.0, 0.0],  // argmax = 0
        ];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![1, 2, 0]);
    }

    #[test]
    fn test_mtp_candidates_empty() {
        let logits: Vec<Vec<f32>> = vec![];
        let candidates = mtp_candidates(&logits);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_mtp_candidates_single_element() {
        let logits = vec![vec![42.0]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![0]);
    }

    // ---- New tests below ----

    #[test]
    fn test_mtp_config_partial_eq_same_values() {
        let a = MtpConfig {
            depth: 3,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        let b = MtpConfig {
            depth: 3,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_mtp_config_partial_eq_different_depth() {
        let a = MtpConfig {
            depth: 2,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        let b = MtpConfig {
            depth: 3,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_mtp_config_clone_independent() {
        let original = MtpConfig {
            depth: 4,
            vocab_size: 128256,
            hidden_size: 7168,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // Cloned is independent — modifying fields on clone doesn't affect original
        // (usize is Copy, so this is trivially true, but we verify Clone compiles)
    }

    #[test]
    fn test_mtp_config_debug_format() {
        let cfg = MtpConfig {
            depth: 2,
            vocab_size: 1024,
            hidden_size: 512,
        };
        let debug_str = format!("{:?}", cfg);
        assert!(debug_str.contains("depth"));
        assert!(debug_str.contains("vocab_size"));
        assert!(debug_str.contains("hidden_size"));
    }

    #[test]
    fn test_mtp_head_clone_produces_equal_copy() {
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0, 3.0, 4.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let cloned = head.clone();
        assert_eq!(cloned.config, head.config);
        assert_eq!(cloned.projections, head.projections);
    }

    #[test]
    fn test_mtp_head_debug_format() {
        let head = MtpHead {
            projections: vec![vec![1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let debug_str = format!("{:?}", head);
        assert!(debug_str.contains("MtpHead"));
        assert!(debug_str.contains("projections"));
        assert!(debug_str.contains("config"));
    }

    #[test]
    fn test_mtp_draft_error_display_contains_context() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig::default(),
        };
        let result = mtp_draft(&[1.0], &head);
        let err = result.unwrap_err();
        let display = format!("{}", err);
        assert!(
            display.contains("zero hidden_size or vocab_size"),
            "expected context about zero config, got: {}",
            display
        );
    }

    #[test]
    fn test_mtp_draft_error_is_backend_error_other() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig::default(),
        };
        let result = mtp_draft(&[1.0], &head);
        match result.unwrap_err() {
            BackendError::Other(msg) => {
                assert!(msg.contains("zero hidden_size or vocab_size"));
            }
            other => panic!("expected BackendError::Other, got: {:?}", other),
        }
    }

    #[test]
    fn test_mtp_draft_rejects_empty_hidden_state() {
        let head = MtpHead {
            projections: vec![vec![0.0; 0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let result = mtp_draft(&[], &head);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("hidden_state length"));
    }

    #[test]
    fn test_mtp_draft_zero_hidden_state_with_valid_config() {
        // All-zero hidden_state should still produce valid (zero) logits
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0, 3.0, 4.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[0.0, 0.0], &head).unwrap();
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0], vec![0.0, 0.0]);
    }

    #[test]
    fn test_mtp_draft_negative_hidden_state() {
        // Negative hidden values should produce correct negative logits
        let head = MtpHead {
            projections: vec![vec![1.0, 0.0, 0.0, 1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[-3.0, -5.0], &head).unwrap();
        assert!((logits[0][0] - (-3.0)).abs() < 1e-5);
        assert!((logits[0][1] - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_large_depth() {
        // depth=5 with uniform identity projections
        let identity_proj: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let head = MtpHead {
            projections: vec![identity_proj; 5],
            config: MtpConfig {
                depth: 5,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 1.0], &head).unwrap();
        assert_eq!(logits.len(), 5);
        for k in 0..5 {
            assert_eq!(logits[k].len(), 2);
            assert!((logits[k][0] - 1.0).abs() < 1e-5);
            assert!((logits[k][1] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mtp_candidates_tie_breaks_by_last_occurrence() {
        // All equal values — max_by scans left to right and keeps the last maximum seen
        // because partial_cmp returns Equal for equal f32 and max_by prefers the later element
        let logits = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![3]);
    }

    #[test]
    #[should_panic(expected = "NaN in logits at MTP depth")]
    fn test_mtp_candidates_with_nan_panics() {
        // NaN in logits must be detected and reported, not silently defaulted to token 0
        let logits = vec![vec![f32::NAN, 5.0, f32::NAN, 5.0]];
        let _candidates = mtp_candidates(&logits);
    }

    #[test]
    fn test_mtp_candidates_with_negative_infinity() {
        let logits = vec![vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![1]);
    }

    #[test]
    fn test_mtp_draft_then_candidates_e2e_pipeline() {
        // Build a head where depth=3, vocab_size=3, hidden_size=2
        // projection[0]: identity-like, argmax at index 2
        // projection[1]: reversed, argmax at index 0
        // projection[2]: double-weight, argmax at index 2
        let head = MtpHead {
            projections: vec![
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0], // [0, 0, h0+h1]
                vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], // [h0+h1, 0, 0]
                vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0], // [0, 0, 2*h0+2*h1]
            ],
            config: MtpConfig {
                depth: 3,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 1.0], &head).unwrap();
        let candidates = mtp_candidates(&logits);

        // depth 0: [0, 0, 2] → argmax=2
        // depth 1: [2, 0, 0] → argmax=0
        // depth 2: [0, 0, 4] → argmax=2
        assert_eq!(candidates, vec![2, 0, 2]);
    }

    #[test]
    fn test_mtp_config_default_then_override() {
        let mut cfg = MtpConfig::default();
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.vocab_size, 0);
        assert_eq!(cfg.hidden_size, 0);

        cfg.depth = 4;
        cfg.vocab_size = 128000;
        cfg.hidden_size = 6144;
        assert_eq!(cfg.depth, 4);
        assert_eq!(cfg.vocab_size, 128000);
        assert_eq!(cfg.hidden_size, 6144);
    }

    #[test]
    fn test_mtp_draft_projection_count_mismatch_error_message() {
        let head = MtpHead {
            projections: vec![vec![0.0; 4], vec![0.0; 4]],
            config: MtpConfig {
                depth: 5, // mismatch: 2 projections vs depth=5
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("projections count") && msg.contains("config.depth"),
            "error message should mention both projections count and config.depth, got: {}",
            msg
        );
    }

    // ---- Trait tests ----

    #[test]
    fn test_mtp_config_copy_trait() {
        let cfg = MtpConfig {
            depth: 3,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        // Copy: assignment copies the value, not a move
        let copy: MtpConfig = cfg;
        assert_eq!(cfg, copy); // cfg still usable after assignment to copy
    }

    #[test]
    fn test_mtp_config_hash_equal_values() {
        use std::collections::HashSet;
        let cfg1 = MtpConfig {
            depth: 2,
            vocab_size: 100,
            hidden_size: 64,
        };
        let cfg2 = MtpConfig {
            depth: 2,
            vocab_size: 100,
            hidden_size: 64,
        };
        let mut set = HashSet::new();
        set.insert(cfg1);
        assert!(set.contains(&cfg2));
    }

    #[test]
    fn test_mtp_config_hash_different_values() {
        use std::collections::HashSet;
        let cfg1 = MtpConfig {
            depth: 2,
            vocab_size: 100,
            hidden_size: 64,
        };
        let cfg2 = MtpConfig {
            depth: 3,
            vocab_size: 100,
            hidden_size: 64,
        };
        let mut set = HashSet::new();
        set.insert(cfg1);
        assert!(!set.contains(&cfg2));
    }

    #[test]
    fn test_mtp_config_eq_trait() {
        let a = MtpConfig {
            depth: 1,
            vocab_size: 10,
            hidden_size: 5,
        };
        let b = MtpConfig {
            depth: 1,
            vocab_size: 10,
            hidden_size: 5,
        };
        assert!(a == b); // Eq implies stricter PartialEq

        let c = MtpConfig {
            depth: 1,
            vocab_size: 20,
            hidden_size: 5,
        };
        assert!(a != c);
    }

    #[test]
    fn test_mtp_config_field_access() {
        let cfg = MtpConfig {
            depth: 7,
            vocab_size: 65536,
            hidden_size: 8192,
        };
        assert_eq!(cfg.depth, 7);
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.hidden_size, 8192);
    }

    #[test]
    fn test_mtp_head_partial_eq_same() {
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            config: MtpConfig {
                depth: 2,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        let same = MtpHead {
            projections: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            config: MtpConfig {
                depth: 2,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        assert_eq!(head, same);
    }

    #[test]
    fn test_mtp_head_partial_eq_different_projections() {
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        let diff = MtpHead {
            projections: vec![vec![9.0, 9.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        assert_ne!(head, diff);
    }

    #[test]
    fn test_mtp_head_partial_eq_different_config() {
        let head = MtpHead {
            projections: vec![vec![1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let diff = MtpHead {
            projections: vec![vec![1.0]],
            config: MtpConfig {
                depth: 2,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        assert_ne!(head, diff);
    }

    #[test]
    fn test_mtp_head_field_access() {
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0, 3.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 1,
            },
        };
        assert_eq!(head.projections.len(), 1);
        assert_eq!(head.projections[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(head.config.depth, 1);
        assert_eq!(head.config.vocab_size, 3);
        assert_eq!(head.config.hidden_size, 1);
    }

    // ---- Function boundary tests ----

    #[test]
    fn test_mtp_candidates_with_negative_values() {
        let logits = vec![vec![-5.0, -1.0, -3.0]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![1]); // -1.0 is largest
    }

    #[test]
    fn test_mtp_candidates_with_infinity() {
        let logits = vec![vec![1.0, f32::INFINITY, 3.0]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![1]); // infinity wins
    }

    #[test]
    fn test_mtp_candidates_with_all_neg_infinity() {
        let logits = vec![vec![f32::NEG_INFINITY; 4]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![3]); // all equal, last index wins
    }

    #[test]
    #[should_panic(expected = "mtp_candidates: empty logits at depth")]
    fn test_mtp_candidates_empty_inner_vector() {
        let logits = vec![vec![]];
        let _candidates = mtp_candidates(&logits);
    }

    #[test]
    fn test_mtp_candidates_multiple_vectors_mixed_signs() {
        let logits = vec![
            vec![-100.0, -200.0],    // argmax = 0
            vec![-0.001, -0.0001],   // argmax = 1
        ];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![0, 1]);
    }

    #[test]
    fn test_mtp_draft_depth_zero_rejects() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig {
                depth: 0,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        // projections count 0 != depth 0 => they match, but all-zero projection matrix
        // produces empty Ok(vec![]) since depth=0 means no projections
        assert!(result.is_ok());
        let logits = result.unwrap();
        assert!(logits.is_empty());
    }

    #[test]
    fn test_mtp_draft_vocab_size_zero_rejects() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig {
                depth: 1,
                vocab_size: 0,
                hidden_size: 4,
            },
        };
        let result = mtp_draft(&[1.0, 2.0, 3.0, 4.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_hidden_size_zero_rejects() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig {
                depth: 1,
                vocab_size: 4,
                hidden_size: 0,
            },
        };
        let result = mtp_draft(&[], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_all_zero_projections() {
        let head = MtpHead {
            projections: vec![vec![0.0; 6]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 2.0], &head).unwrap();
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mtp_draft_hidden_state_mismatch_error_message() {
        let head = MtpHead {
            projections: vec![vec![0.0; 4]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0], &head);
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("hidden_state length") && msg.contains("config.hidden_size"),
            "expected hidden_state mismatch context, got: {}",
            msg
        );
    }

    #[test]
    fn test_mtp_draft_wrong_single_projection_shape_error() {
        // Two projections: first is correct, second is wrong shape
        let head = MtpHead {
            projections: vec![
                vec![0.0; 6], // correct: 3*2=6
                vec![0.0; 4], // wrong: should be 6
            ],
            config: MtpConfig {
                depth: 2,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("projection[1]") && msg.contains("length"),
            "expected projection shape error at index 1, got: {}",
            msg
        );
    }

    #[test]
    fn test_mtp_draft_large_values() {
        // Verify no overflow/underflow with large f32 values
        let head = MtpHead {
            projections: vec![vec![1e30, 0.0, 0.0, 1e30]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 1.0], &head).unwrap();
        assert!((logits[0][0] - 1e30).abs() < 1e25);
        assert!((logits[0][1] - 1e30).abs() < 1e25);
    }

    #[test]
    fn test_mtp_draft_single_vocab_single_hidden() {
        let head = MtpHead {
            projections: vec![vec![3.5]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let logits = mtp_draft(&[2.0], &head).unwrap();
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 1);
        assert!((logits[0][0] - 7.0).abs() < 1e-5);
    }

    // =========================================================================
    // New tests (target: 51 → 101+)
    // =========================================================================

    // ---- MtpConfig construction & mutation ----

    #[test]
    fn test_mtp_config_zero_depth() {
        let cfg = MtpConfig {
            depth: 0,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        assert_eq!(cfg.depth, 0);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_mtp_config_large_values() {
        let cfg = MtpConfig {
            depth: 100,
            vocab_size: usize::MAX,
            hidden_size: usize::MAX,
        };
        assert_eq!(cfg.depth, 100);
        assert_eq!(cfg.vocab_size, usize::MAX);
        assert_eq!(cfg.hidden_size, usize::MAX);
    }

    #[test]
    fn test_mtp_config_partial_eq_different_vocab_size() {
        let a = MtpConfig {
            depth: 2,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        let b = MtpConfig {
            depth: 2,
            vocab_size: 128000,
            hidden_size: 4096,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_mtp_config_partial_eq_different_hidden_size() {
        let a = MtpConfig {
            depth: 2,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        let b = MtpConfig {
            depth: 2,
            vocab_size: 32000,
            hidden_size: 8192,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_mtp_config_clone_then_modify_independent() {
        let mut original = MtpConfig {
            depth: 3,
            vocab_size: 5000,
            hidden_size: 1024,
        };
        let cloned = original.clone();
        original.depth = 10;
        assert_ne!(original.depth, cloned.depth);
        assert_eq!(cloned.depth, 3);
    }

    // ---- MtpConfig Hash in maps ----

    #[test]
    fn test_mtp_config_used_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let key = MtpConfig {
            depth: 2,
            vocab_size: 1000,
            hidden_size: 256,
        };
        map.insert(key, "model_a");
        let lookup = MtpConfig {
            depth: 2,
            vocab_size: 1000,
            hidden_size: 256,
        };
        assert_eq!(map.get(&lookup), Some(&"model_a"));
    }

    #[test]
    fn test_mtp_config_hashset_insert_and_remove() {
        use std::collections::HashSet;
        let cfg = MtpConfig {
            depth: 4,
            vocab_size: 50000,
            hidden_size: 2048,
        };
        let mut set = HashSet::new();
        assert!(set.insert(cfg));
        assert!(!set.insert(cfg)); // duplicate
        assert!(set.remove(&cfg));
        assert!(!set.contains(&cfg));
    }

    // ---- MtpConfig Debug ----

    #[test]
    fn test_mtp_config_debug_contains_all_field_values() {
        let cfg = MtpConfig {
            depth: 7,
            vocab_size: 32000,
            hidden_size: 4096,
        };
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("7"), "depth value should appear");
        assert!(debug.contains("32000"), "vocab_size value should appear");
        assert!(debug.contains("4096"), "hidden_size value should appear");
    }

    #[test]
    fn test_mtp_config_debug_default() {
        let cfg = MtpConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("MtpConfig"), "type name should appear");
    }

    // ---- MtpHead construction ----

    #[test]
    fn test_mtp_head_empty_projections() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig {
                depth: 0,
                vocab_size: 10,
                hidden_size: 5,
            },
        };
        assert!(head.projections.is_empty());
        assert_eq!(head.config.depth, 0);
    }

    #[test]
    fn test_mtp_head_clone_deep_copy() {
        let head = MtpHead {
            projections: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            config: MtpConfig {
                depth: 2,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        let mut cloned = head.clone();
        // Modify clone, verify original unchanged
        cloned.projections[0][0] = 99.0;
        assert_eq!(head.projections[0][0], 1.0);
        assert_eq!(cloned.projections[0][0], 99.0);
    }

    #[test]
    fn test_mtp_head_debug_contains_projection_data() {
        let head = MtpHead {
            projections: vec![vec![42.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let debug = format!("{:?}", head);
        assert!(debug.contains("42.0"), "projection values should appear");
    }

    #[test]
    fn test_mtp_head_partial_eq_different_projection_count() {
        let a = MtpHead {
            projections: vec![vec![1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let b = MtpHead {
            projections: vec![vec![1.0], vec![2.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        assert_ne!(a, b);
    }

    // ---- mtp_draft additional validation tests ----

    #[test]
    fn test_mtp_draft_rejects_hidden_state_one_element_short() {
        let head = MtpHead {
            projections: vec![vec![0.0; 6]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        // hidden_state has 1 element, config expects 2
        let result = mtp_draft(&[1.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_rejects_hidden_state_one_element_too_many() {
        let head = MtpHead {
            projections: vec![vec![0.0; 6]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        // hidden_state has 3 elements, config expects 2
        let result = mtp_draft(&[1.0, 2.0, 3.0], &head);
        assert!(result.is_err());
    }

    #[test]
    fn test_mtp_draft_error_unwrapped_message_format() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig::default(),
        };
        let result = mtp_draft(&[], &head);
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.starts_with("backend error:"), "got: {}", msg);
    }

    // ---- mtp_draft computation correctness ----

    #[test]
    fn test_mtp_draft_mixed_sign_projection() {
        // projection with negative weights
        let head = MtpHead {
            projections: vec![vec![-1.0, 2.0, 3.0, -4.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        // hidden = [2.0, 3.0]
        // logits[0] = 2*(-1) + 3*2 = -2+6 = 4
        // logits[1] = 2*3 + 3*(-4) = 6-12 = -6
        let logits = mtp_draft(&[2.0, 3.0], &head).unwrap();
        assert!((logits[0][0] - 4.0).abs() < 1e-5);
        assert!((logits[0][1] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_projection_all_ones() {
        let head = MtpHead {
            projections: vec![vec![1.0; 4]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        // hidden = [3.0, 4.0]
        // logits[0] = 3+4 = 7
        // logits[1] = 3+4 = 7
        let logits = mtp_draft(&[3.0, 4.0], &head).unwrap();
        assert!((logits[0][0] - 7.0).abs() < 1e-5);
        assert!((logits[0][1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_very_small_values() {
        let head = MtpHead {
            projections: vec![vec![1e-38, 0.0, 0.0, 1e-38]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 1.0], &head).unwrap();
        assert!((logits[0][0] - 1e-38).abs() < 1e-33);
        assert!((logits[0][1] - 1e-38).abs() < 1e-33);
    }

    #[test]
    fn test_mtp_draft_negative_projection_and_hidden() {
        // projection = [-2.0], hidden = [-3.0], logits[0] = -2*-3 = 6
        let head = MtpHead {
            projections: vec![vec![-2.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let logits = mtp_draft(&[-3.0], &head).unwrap();
        assert!((logits[0][0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_three_depth_all_different() {
        let head = MtpHead {
            projections: vec![
                vec![1.0, 0.0, 0.0, 1.0], // identity: [h0, h1]
                vec![2.0, 0.0, 0.0, 2.0], // 2x identity: [2*h0, 2*h1]
                vec![0.0, 1.0, 1.0, 0.0], // swap: [h1, h0]
            ],
            config: MtpConfig {
                depth: 3,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let hidden = [3.0, 7.0];
        let logits = mtp_draft(&hidden, &head).unwrap();

        // depth 0: [3, 7]
        assert!((logits[0][0] - 3.0).abs() < 1e-5);
        assert!((logits[0][1] - 7.0).abs() < 1e-5);

        // depth 1: [6, 14]
        assert!((logits[1][0] - 6.0).abs() < 1e-5);
        assert!((logits[1][1] - 14.0).abs() < 1e-5);

        // depth 2: [7, 3]
        assert!((logits[2][0] - 7.0).abs() < 1e-5);
        assert!((logits[2][1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_orthogonal_projection() {
        // vocab=1, hidden=2: projection shape = 1*2 = 2 elements
        let head = MtpHead {
            projections: vec![vec![0.0, 1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 1,
                hidden_size: 2,
            },
        };
        // hidden = [3.0, 7.0], logits[0] = 3*0 + 7*1 = 7
        let logits = mtp_draft(&[3.0, 7.0], &head).unwrap();
        assert!((logits[0][0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_draft_result_length_matches_depth() {
        let head = MtpHead {
            projections: vec![
                vec![0.0; 4],
                vec![0.0; 4],
                vec![0.0; 4],
                vec![0.0; 4],
            ],
            config: MtpConfig {
                depth: 4,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let logits = mtp_draft(&[1.0, 2.0], &head).unwrap();
        assert_eq!(logits.len(), 4);
        for layer in &logits {
            assert_eq!(layer.len(), 2);
        }
    }

    // ---- mtp_candidates additional tests ----

    #[test]
    fn test_mtp_candidates_single_depth_returns_single_token() {
        let logits = vec![vec![0.1, 0.5, 0.3]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], 1);
    }

    #[test]
    fn test_mtp_candidates_preserves_order() {
        // Each depth level is independent
        let logits = vec![
            vec![10.0, 0.0],  // argmax=0
            vec![0.0, 10.0],  // argmax=1
            vec![10.0, 10.0], // tie, last=1
        ];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![0, 1, 1]);
    }

    #[test]
    fn test_mtp_candidates_with_positive_infinity() {
        let logits = vec![vec![f32::INFINITY, f32::INFINITY]];
        let candidates = mtp_candidates(&logits);
        // Both INFINITY, max_by picks last
        assert_eq!(candidates, vec![1]);
    }

    #[test]
    #[should_panic(expected = "NaN in logits at MTP depth")]
    fn test_mtp_candidates_with_mixed_nan_panics() {
        // NaN in logits must be detected and reported, not silently accepted
        let logits = vec![vec![f32::NAN, 5.0, f32::NAN, 3.0]];
        let _candidates = mtp_candidates(&logits);
    }

    #[test]
    fn test_mtp_candidates_all_zeros() {
        let logits = vec![vec![0.0, 0.0, 0.0]];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![2]); // last index wins on tie
    }

    #[test]
    fn test_mtp_candidates_large_vocab() {
        let mut logits_vec = vec![0.0f32; 100000];
        logits_vec[50000] = 1.0; // peak in the middle
        let logits = vec![logits_vec];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates[0], 50000);
    }

    #[test]
    fn test_mtp_candidates_many_depth_levels() {
        let logits: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0; 10];
                v[i % 10] = 1.0;
                v
            })
            .collect();
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates.len(), 100);
        for (i, &c) in candidates.iter().enumerate() {
            assert_eq!(c, (i % 10) as u32);
        }
    }

    // ---- E2E pipeline tests ----

    #[test]
    fn test_mtp_draft_then_candidates_single_vocab() {
        let head = MtpHead {
            projections: vec![vec![5.0], vec![-3.0]],
            config: MtpConfig {
                depth: 2,
                vocab_size: 1,
                hidden_size: 1,
            },
        };
        let logits = mtp_draft(&[1.0], &head).unwrap();
        let candidates = mtp_candidates(&logits);
        // Both have vocab_size=1, so both argmax = 0
        assert_eq!(candidates, vec![0, 0]);
    }

    #[test]
    fn test_mtp_draft_then_candidates_consistent_with_manual_argmax() {
        // vocab=3, hidden=2: projection layout is [v0_h0, v0_h1, v1_h0, v1_h1, v2_h0, v2_h1]
        // Put 100.0 at v1_h0 so that logits[1] = hidden[0]*100 + hidden[1]*0 = 100
        let head = MtpHead {
            projections: vec![
                vec![0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            ],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let hidden = [1.0, 1.0];
        let logits = mtp_draft(&hidden, &head).unwrap();
        // logits = [0, 100, 0] -> argmax = 1
        assert!((logits[0][1] - 100.0).abs() < 1e-5);
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![1]);
    }

    #[test]
    fn test_mtp_draft_then_candidates_zero_logits_all_zero_argmax() {
        // Zero projections produce zero logits -> all tie -> last index
        let head = MtpHead {
            projections: vec![vec![0.0; 9]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 3,
            },
        };
        let logits = mtp_draft(&[0.0, 0.0, 0.0], &head).unwrap();
        let candidates = mtp_candidates(&logits);
        // All zeros, last index wins
        assert_eq!(candidates, vec![2]);
    }

    // ---- BackendError integration ----

    #[test]
    fn test_backend_error_display_cuda() {
        let err = BackendError::Cuda("device not found".into());
        let msg = format!("{}", err);
        assert!(msg.contains("CUDA error:"));
        assert!(msg.contains("device not found"));
    }

    #[test]
    fn test_backend_error_display_hip() {
        let err = BackendError::Hip("ROCm failure".into());
        let msg = format!("{}", err);
        assert!(msg.contains("HIP error:"));
        assert!(msg.contains("ROCm failure"));
    }

    #[test]
    fn test_backend_error_display_metal() {
        let err = BackendError::Metal("GPU timeout".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Metal error:"));
        assert!(msg.contains("GPU timeout"));
    }

    #[test]
    fn test_backend_error_display_cpu() {
        let err = BackendError::Cpu("illegal instruction".into());
        let msg = format!("{}", err);
        assert!(msg.contains("CPU error:"));
        assert!(msg.contains("illegal instruction"));
    }

    #[test]
    fn test_backend_error_display_unimplemented() {
        let err = BackendError::Unimplemented("flash attention");
        let msg = format!("{}", err);
        assert!(msg.contains("unimplemented:"));
        assert!(msg.contains("flash attention"));
    }

    #[test]
    fn test_backend_error_display_other() {
        let err = BackendError::Other("something went wrong".into());
        let msg = format!("{}", err);
        assert!(msg.contains("backend error:"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_backend_error_debug_format() {
        let err = BackendError::Other("test".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Other"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_backend_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::Other("test".into()));
        let msg = format!("{}", err);
        assert!(msg.contains("backend error:"));
    }

    #[test]
    fn test_backend_error_clone() {
        let err = BackendError::Other("original".into());
        let cloned = err.clone();
        assert_eq!(format!("{}", err), format!("{}", cloned));
    }

    // ---- Edge case: projection with exactly one extra element ----

    #[test]
    fn test_mtp_draft_projection_one_extra_element() {
        let head = MtpHead {
            projections: vec![vec![0.0; 7]], // should be 3*2=6
            config: MtpConfig {
                depth: 1,
                vocab_size: 3,
                hidden_size: 2,
            },
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("projection[0]"));
    }

    // ---- MtpConfig: verify all fields independently mutable ----

    #[test]
    fn test_mtp_config_mutable_depth() {
        let mut cfg = MtpConfig::default();
        cfg.depth = 99;
        assert_eq!(cfg.depth, 99);
    }

    #[test]
    fn test_mtp_config_mutable_vocab_size() {
        let mut cfg = MtpConfig::default();
        cfg.vocab_size = 99999;
        assert_eq!(cfg.vocab_size, 99999);
    }

    #[test]
    fn test_mtp_config_mutable_hidden_size() {
        let mut cfg = MtpConfig::default();
        cfg.hidden_size = 8192;
        assert_eq!(cfg.hidden_size, 8192);
    }

    // ---- mtp_draft with subnormal float values ----

    #[test]
    fn test_mtp_draft_subnormal_hidden_state() {
        let head = MtpHead {
            projections: vec![vec![1.0, 0.0, 0.0, 1.0]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let logits = mtp_draft(&[subnormal, subnormal], &head).unwrap();
        assert!(logits[0][0] >= 0.0);
        assert!(logits[0][1] >= 0.0);
    }

    // ---- mtp_candidates with alternating max pattern ----

    #[test]
    fn test_mtp_candidates_alternating_max() {
        let logits = vec![
            vec![1.0, 0.0, 0.0], // max at 0
            vec![0.0, 1.0, 0.0], // max at 1
            vec![0.0, 0.0, 1.0], // max at 2
        ];
        let candidates = mtp_candidates(&logits);
        assert_eq!(candidates, vec![0, 1, 2]);
    }

    // ---- mtp_draft returns correct number of elements per depth ----

    #[test]
    fn test_mtp_draft_vocab_size_matches_output_length() {
        let head = MtpHead {
            projections: vec![vec![0.0; 20]],
            config: MtpConfig {
                depth: 1,
                vocab_size: 5,
                hidden_size: 4,
            },
        };
        let logits = mtp_draft(&[0.0; 4], &head).unwrap();
        assert_eq!(logits[0].len(), 5);
    }

    // ---- Multiple projections with different weight patterns ----

    #[test]
    fn test_mtp_draft_depths_produce_different_logits() {
        let head = MtpHead {
            projections: vec![
                vec![1.0, 0.0, 0.0, 1.0], // identity
                vec![0.0, 1.0, 1.0, 0.0], // swap
            ],
            config: MtpConfig {
                depth: 2,
                vocab_size: 2,
                hidden_size: 2,
            },
        };
        let hidden = [5.0, 10.0];
        let logits = mtp_draft(&hidden, &head).unwrap();

        // depth 0: [5, 10]
        assert!((logits[0][0] - 5.0).abs() < 1e-5);
        assert!((logits[0][1] - 10.0).abs() < 1e-5);

        // depth 1: [10, 5]
        assert!((logits[1][0] - 10.0).abs() < 1e-5);
        assert!((logits[1][1] - 5.0).abs() < 1e-5);

        // Verify the two depth levels produce different results
        assert_ne!(logits[0], logits[1]);
    }
}
