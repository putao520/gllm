//! EAGLE Draft Head — 轻量 draft 模型加速推测解码
//!
//! EAGLE 在目标模型最后一层 hidden state 上添加 1-2 层轻量 transformer，
//! 输入 hidden_state + prev_embedding，输出 next token logits。
//! EAGLE-2 根据 draft logits 置信度动态调整推测树宽度。

use crate::engine::executor::BackendError;

/// EAGLE Draft Head 配置
#[derive(Debug, Clone, PartialEq)]
pub struct EagleConfig {
    /// Draft head 层数 (通常 1-2)
    pub num_draft_layers: usize,
    /// 隐藏维度 (与主模型相同)
    pub hidden_size: usize,
    /// Draft 推测 token 数量
    pub num_draft_tokens: usize,
    /// EAGLE-2 置信度阈值 (高于此值单分支深探，低于此值多分支宽探)
    pub confidence_threshold: f32,
}

impl Default for EagleConfig {
    fn default() -> Self {
        Self {
            num_draft_layers: 1,
            hidden_size: 0, // 从主模型推导
            num_draft_tokens: 5,
            confidence_threshold: 0.7,
        }
    }
}

/// EAGLE Draft Head 权重
#[derive(Debug, Clone, PartialEq)]
pub struct EagleHead {
    /// 融合层权重: [hidden_size * 2, hidden_size]
    pub fc_weight: Vec<f32>,
    /// Draft transformer 层权重 (简化为 FFN)
    pub draft_layers: Vec<DraftLayerWeights>,
    /// 是否共享主模型的 lm_head
    pub share_lm_head: bool,
}

/// Draft transformer 层的 FFN 权重
#[derive(Debug, Clone, PartialEq)]
pub struct DraftLayerWeights {
    pub up_weight: Vec<f32>,
    pub down_weight: Vec<f32>,
}

/// EAGLE draft 前向: hidden_state + embedding → draft logits
///
/// P3 骨架 — 完整实现需要 JIT 编译 draft head 图
pub fn eagle_draft(
    _hidden_state: &[f32],
    _prev_embedding: &[f32],
    _head: &EagleHead,
    _config: &EagleConfig,
) -> Result<Vec<f32>, BackendError> {
    Err(BackendError::Other(
        "EAGLE draft head forward not yet implemented".into(),
    ))
}

/// EAGLE-2: 根据 draft logits 置信度动态构建推测树
///
/// 高置信 (max_prob > threshold): 单分支深探 — 只取 top-1
/// 低置信 (max_prob <= threshold): 多分支宽探 — 取 top-3
pub fn build_eagle_tree(
    draft_logits: &[f32],
    vocab_size: usize,
    config: &EagleConfig,
) -> Vec<u32> {
    if draft_logits.is_empty() || vocab_size == 0 {
        return vec![];
    }

    // Numerically stable softmax: subtract max before exp
    let max_logit = draft_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = draft_logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();

    let max_prob = if sum > 0.0 {
        probs.iter().cloned().fold(0.0f32, f32::max) / sum
    } else {
        0.0
    };

    if max_prob > config.confidence_threshold {
        // 高置信: 单分支 — 只取 top-1
        let top = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        vec![top]
    } else {
        // 低置信: 多分支 — 取 top-3
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(3);
        indexed.iter().map(|(i, _)| *i as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eagle_config_default() {
        let config = EagleConfig::default();
        assert_eq!(config.num_draft_layers, 1);
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.num_draft_tokens, 5);
        assert!((config.confidence_threshold - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_eagle_draft_returns_err() {
        let head = EagleHead {
            fc_weight: vec![0.0; 16],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let config = EagleConfig::default();
        let result = eagle_draft(&[1.0, 2.0], &[3.0, 4.0], &head, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_eagle_tree_empty_input() {
        let config = EagleConfig::default();
        assert!(build_eagle_tree(&[], 1000, &config).is_empty());
        assert!(build_eagle_tree(&[1.0], 0, &config).is_empty());
    }

    #[test]
    fn test_build_eagle_tree_high_confidence_single_branch() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // One logit dominates → high confidence → single branch
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0); // index 0 has highest logit
    }

    #[test]
    fn test_build_eagle_tree_low_confidence_multi_branch() {
        let config = EagleConfig {
            confidence_threshold: 0.99, // very high threshold → forces multi-branch
            ..EagleConfig::default()
        };
        // Uniform logits → low confidence → multi-branch top-3
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = build_eagle_tree(&logits, 5, &config);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_eagle_tree_fewer_than_three_tokens() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        // Only 2 logits → multi-branch but truncated to 2
        let logits = vec![1.0, 2.0];
        let result = build_eagle_tree(&logits, 2, &config);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1); // index 1 has highest logit
    }

    // ── Additional tests ──

    #[test]
    fn test_eagle_config_manual_construction() {
        let config = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 4096,
            num_draft_tokens: 8,
            confidence_threshold: 0.85,
        };
        assert_eq!(config.num_draft_layers, 2);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_draft_tokens, 8);
        assert!((config.confidence_threshold - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_eagle_config_clone_is_independent() {
        let original = EagleConfig {
            num_draft_layers: 3,
            hidden_size: 2048,
            num_draft_tokens: 10,
            confidence_threshold: 0.6,
        };
        let cloned = original.clone();
        // Verify all fields match
        assert_eq!(original.num_draft_layers, cloned.num_draft_layers);
        assert_eq!(original.hidden_size, cloned.hidden_size);
        assert_eq!(original.num_draft_tokens, cloned.num_draft_tokens);
        assert!((original.confidence_threshold - cloned.confidence_threshold).abs() < 1e-6);
    }

    #[test]
    fn test_eagle_config_debug_trait() {
        let config = EagleConfig {
            num_draft_layers: 1,
            hidden_size: 512,
            num_draft_tokens: 5,
            confidence_threshold: 0.7,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("EagleConfig"));
        assert!(debug_str.contains("num_draft_layers"));
        assert!(debug_str.contains("hidden_size"));
        assert!(debug_str.contains("num_draft_tokens"));
        assert!(debug_str.contains("confidence_threshold"));
    }

    #[test]
    fn test_eagle_head_construction_no_layers() {
        let head = EagleHead {
            fc_weight: vec![1.0, 2.0, 3.0, 4.0],
            draft_layers: vec![],
            share_lm_head: false,
        };
        assert_eq!(head.fc_weight.len(), 4);
        assert_eq!(head.fc_weight[2], 3.0);
        assert!(head.draft_layers.is_empty());
        assert!(!head.share_lm_head);
    }

    #[test]
    fn test_eagle_head_construction_with_layers() {
        let layer = DraftLayerWeights {
            up_weight: vec![0.1; 64],
            down_weight: vec![0.2; 64],
        };
        let head = EagleHead {
            fc_weight: vec![0.0; 128],
            draft_layers: vec![layer],
            share_lm_head: true,
        };
        assert_eq!(head.draft_layers.len(), 1);
        assert_eq!(head.draft_layers[0].up_weight.len(), 64);
        assert_eq!(head.draft_layers[0].down_weight[0], 0.2);
        assert!(head.share_lm_head);
    }

    #[test]
    fn test_draft_layer_weights_debug_trait() {
        let layer = DraftLayerWeights {
            up_weight: vec![1.0, 2.0],
            down_weight: vec![3.0, 4.0],
        };
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("DraftLayerWeights"));
        assert!(debug_str.contains("up_weight"));
        assert!(debug_str.contains("down_weight"));
    }

    #[test]
    fn test_eagle_head_debug_trait() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let debug_str = format!("{:?}", head);
        assert!(debug_str.contains("EagleHead"));
        assert!(debug_str.contains("share_lm_head"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_eagle_draft_error_message_content() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let config = EagleConfig::default();
        let err = eagle_draft(&[], &[], &head, &config).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("EAGLE draft head forward not yet implemented"));
    }

    #[test]
    fn test_build_eagle_tree_negative_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // All negative logits; index 2 has the highest (-1.0)
        let logits = vec![-10.0, -5.0, -1.0, -8.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_build_eagle_tree_single_token_vocabulary() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Single token → high confidence → single branch
        let logits = vec![2.0];
        let result = build_eagle_tree(&logits, 1, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_build_eagle_tree_confidence_at_exact_threshold() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Two logits with equal probability: each 0.5 → max_prob == threshold → multi-branch
        let logits = vec![1.0, 1.0];
        let result = build_eagle_tree(&logits, 2, &config);
        // max_prob == threshold → goes to else branch (multi-branch, top-3)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_build_eagle_tree_multi_branch_preserves_ranking() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        // Five logits with clear ranking: index 4 > 2 > 0 > 3 > 1
        let logits = vec![1.0, 0.1, 3.0, 0.5, 5.0];
        let result = build_eagle_tree(&logits, 5, &config);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 4); // highest
        assert_eq!(result[1], 2); // second
        assert_eq!(result[2], 0); // third
    }

    #[test]
    fn test_build_eagle_tree_extreme_logit_dominance() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Index 1 has enormous logit → should dominate with high confidence
        let logits = vec![0.0, 1000.0, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 5, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_eagle_head_multiple_draft_layers() {
        let layer0 = DraftLayerWeights {
            up_weight: vec![1.0; 16],
            down_weight: vec![2.0; 16],
        };
        let layer1 = DraftLayerWeights {
            up_weight: vec![3.0; 16],
            down_weight: vec![4.0; 16],
        };
        let head = EagleHead {
            fc_weight: vec![0.5; 32],
            draft_layers: vec![layer0, layer1],
            share_lm_head: false,
        };
        assert_eq!(head.draft_layers.len(), 2);
        assert_eq!(head.draft_layers[0].up_weight[0], 1.0);
        assert_eq!(head.draft_layers[1].down_weight[0], 4.0);
    }

    // ── PartialEq tests ──

    #[test]
    fn test_eagle_config_partial_eq_equal() {
        let a = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 4096,
            num_draft_tokens: 5,
            confidence_threshold: 0.8,
        };
        let b = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 4096,
            num_draft_tokens: 5,
            confidence_threshold: 0.8,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_eagle_config_partial_eq_not_equal() {
        let a = EagleConfig {
            num_draft_layers: 1,
            hidden_size: 1024,
            num_draft_tokens: 5,
            confidence_threshold: 0.7,
        };
        let b = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 1024,
            num_draft_tokens: 5,
            confidence_threshold: 0.7,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_head_partial_eq_equal() {
        let a = EagleHead {
            fc_weight: vec![1.0, 2.0],
            draft_layers: vec![DraftLayerWeights {
                up_weight: vec![0.5],
                down_weight: vec![0.3],
            }],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![1.0, 2.0],
            draft_layers: vec![DraftLayerWeights {
                up_weight: vec![0.5],
                down_weight: vec![0.3],
            }],
            share_lm_head: true,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_eagle_head_partial_eq_share_lm_head_differs() {
        let a = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_draft_layer_weights_partial_eq() {
        let a = DraftLayerWeights {
            up_weight: vec![1.0, 2.0],
            down_weight: vec![3.0],
        };
        let b = DraftLayerWeights {
            up_weight: vec![1.0, 2.0],
            down_weight: vec![3.0],
        };
        assert_eq!(a, b);

        let c = DraftLayerWeights {
            up_weight: vec![1.0, 9.0],
            down_weight: vec![3.0],
        };
        assert_ne!(a, c);
    }

    // ── Clone tests ──

    #[test]
    fn test_eagle_head_clone_independence() {
        let original = EagleHead {
            fc_weight: vec![1.0, 2.0, 3.0],
            draft_layers: vec![DraftLayerWeights {
                up_weight: vec![4.0],
                down_weight: vec![5.0],
            }],
            share_lm_head: true,
        };
        let cloned = original.clone();

        assert_eq!(original, cloned);

        // Cloning produces independent memory; the two are equal but distinct
        assert_eq!(original.fc_weight.as_ptr(), original.fc_weight.as_ptr());
        assert_ne!(original.fc_weight.as_ptr(), cloned.fc_weight.as_ptr());
    }

    #[test]
    fn test_draft_layer_weights_clone_independence() {
        let original = DraftLayerWeights {
            up_weight: vec![10.0, 20.0],
            down_weight: vec![30.0],
        };
        let cloned = original.clone();

        assert_eq!(original, cloned);
        assert_ne!(original.up_weight.as_ptr(), cloned.up_weight.as_ptr());
    }

    // ── Debug trait comprehensive tests ──

    #[test]
    fn test_eagle_config_debug_contains_field_values() {
        let config = EagleConfig {
            num_draft_layers: 7,
            hidden_size: 999,
            num_draft_tokens: 12,
            confidence_threshold: 0.33,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("7"));
        assert!(debug.contains("999"));
        assert!(debug.contains("12"));
        assert!(debug.contains("0.33"));
    }

    #[test]
    fn test_eagle_head_debug_contains_share_lm_head_false() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let debug = format!("{:?}", head);
        assert!(debug.contains("false"));
        assert!(!debug.contains("true"));
    }

    // ── EagleConfig edge values ──

    #[test]
    fn test_eagle_config_zero_hidden_size() {
        let config = EagleConfig {
            num_draft_layers: 0,
            hidden_size: 0,
            num_draft_tokens: 0,
            confidence_threshold: 0.0,
        };
        assert_eq!(config.num_draft_layers, 0);
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.num_draft_tokens, 0);
        assert_eq!(config.confidence_threshold, 0.0);
    }

    #[test]
    fn test_eagle_config_max_values() {
        let config = EagleConfig {
            num_draft_layers: usize::MAX,
            hidden_size: usize::MAX,
            num_draft_tokens: usize::MAX,
            confidence_threshold: f32::MAX,
        };
        assert_eq!(config.num_draft_layers, usize::MAX);
        assert_eq!(config.hidden_size, usize::MAX);
        assert_eq!(config.num_draft_tokens, usize::MAX);
    }

    // ── EagleHead edge values ──

    #[test]
    fn test_eagle_head_empty_weights() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        assert!(head.fc_weight.is_empty());
        assert!(head.draft_layers.is_empty());
        assert!(!head.share_lm_head);
    }

    #[test]
    fn test_eagle_head_large_fc_weight() {
        let head = EagleHead {
            fc_weight: vec![0.0; 1_000_000],
            draft_layers: vec![],
            share_lm_head: true,
        };
        assert_eq!(head.fc_weight.len(), 1_000_000);
    }

    // ── DraftLayerWeights edge values ──

    #[test]
    fn test_draft_layer_weights_empty_vectors() {
        let layer = DraftLayerWeights {
            up_weight: vec![],
            down_weight: vec![],
        };
        assert!(layer.up_weight.is_empty());
        assert!(layer.down_weight.is_empty());
    }

    // ── eagle_draft error path ──

    #[test]
    fn test_eagle_draft_with_nonempty_inputs_still_errors() {
        let head = EagleHead {
            fc_weight: vec![1.0; 256],
            draft_layers: vec![DraftLayerWeights {
                up_weight: vec![0.5; 64],
                down_weight: vec![0.3; 64],
            }],
            share_lm_head: false,
        };
        let config = EagleConfig {
            num_draft_layers: 1,
            hidden_size: 256,
            num_draft_tokens: 5,
            confidence_threshold: 0.7,
        };
        let hidden = vec![0.1; 256];
        let embedding = vec![0.2; 256];
        let result = eagle_draft(&hidden, &embedding, &head, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("EAGLE"));
        assert!(msg.contains("not yet implemented"));
    }

    #[test]
    fn test_eagle_draft_error_is_backend_error_other_variant() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let config = EagleConfig::default();
        let err = eagle_draft(&[], &[], &head, &config).unwrap_err();
        match err {
            BackendError::Other(msg) => {
                assert!(msg.contains("EAGLE draft head forward"));
            }
            other => panic!("expected BackendError::Other, got {:?}", other),
        }
    }

    // ── build_eagle_tree edge cases ──

    #[test]
    fn test_build_eagle_tree_all_zero_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // All zero logits → uniform distribution → max_prob = 1/4 = 0.25 < 0.5 → multi-branch
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_eagle_tree_threshold_zero_always_high_confidence() {
        let config = EagleConfig {
            confidence_threshold: 0.0,
            ..EagleConfig::default()
        };
        // Any non-zero max_prob > 0.0 → high confidence single branch
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_eagle_tree_threshold_one_always_multi_branch() {
        let config = EagleConfig {
            confidence_threshold: 1.0,
            ..EagleConfig::default()
        };
        // Even a perfectly peaked distribution has max_prob < 1.0 (or == 1.0)
        // For uniform, max_prob = 0.2 < 1.0 → multi-branch
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = build_eagle_tree(&logits, 5, &config);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_eagle_tree_exactly_three_logits_low_confidence() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        // Exactly 3 logits, uniform → multi-branch returns all 3
        let logits = vec![1.0, 1.0, 1.0];
        let result = build_eagle_tree(&logits, 3, &config);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_eagle_tree_two_logits_high_confidence() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // One logit dominates → single branch
        let logits = vec![100.0, 0.0];
        let result = build_eagle_tree(&logits, 2, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_build_eagle_tree_large_vocabulary() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let mut logits = vec![0.0; 1000];
        logits[42] = 10.0; // one spike
        logits[100] = 8.0;
        logits[500] = 6.0;
        let result = build_eagle_tree(&logits, 1000, &config);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 42);
        assert_eq!(result[1], 100);
        assert_eq!(result[2], 500);
    }

    #[test]
    fn test_build_eagle_tree_mixed_positive_negative() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![-5.0, 3.0, -2.0, 1.0];
        let result = build_eagle_tree(&logits, 4, &config);
        // index 1 (3.0) dominates → high confidence → single branch
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_build_eagle_tree_result_indices_are_valid() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let logits = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let result = build_eagle_tree(&logits, 5, &config);
        for &idx in &result {
            assert!(idx < 5, "index {} out of range", idx);
        }
    }

    // ── Round 2: 36 additional tests ──

    // ── EagleConfig field mutation ──

    #[test]
    fn test_eagle_config_field_update_num_draft_layers() {
        let mut config = EagleConfig::default();
        assert_eq!(config.num_draft_layers, 1);
        config.num_draft_layers = 4;
        assert_eq!(config.num_draft_layers, 4);
        // Other fields unchanged
        assert_eq!(config.num_draft_tokens, 5);
    }

    #[test]
    fn test_eagle_config_field_update_hidden_size() {
        let mut config = EagleConfig::default();
        config.hidden_size = 8192;
        assert_eq!(config.hidden_size, 8192);
    }

    #[test]
    fn test_eagle_config_field_update_confidence_threshold() {
        let mut config = EagleConfig::default();
        config.confidence_threshold = 0.123;
        assert!((config.confidence_threshold - 0.123).abs() < 1e-7);
    }

    #[test]
    fn test_eagle_config_default_then_customized() {
        let config = EagleConfig {
            hidden_size: 4096,
            ..EagleConfig::default()
        };
        // Only hidden_size overridden; rest stay default
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_draft_layers, 1);
        assert_eq!(config.num_draft_tokens, 5);
        assert!((config.confidence_threshold - 0.7).abs() < 1e-6);
    }

    // ── EagleConfig PartialEq per-field difference ──

    #[test]
    fn test_eagle_config_ne_differs_by_hidden_size() {
        let a = EagleConfig {
            hidden_size: 1024,
            ..EagleConfig::default()
        };
        let b = EagleConfig {
            hidden_size: 2048,
            ..EagleConfig::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_config_ne_differs_by_num_draft_tokens() {
        let a = EagleConfig {
            num_draft_tokens: 3,
            ..EagleConfig::default()
        };
        let b = EagleConfig {
            num_draft_tokens: 7,
            ..EagleConfig::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_config_ne_differs_by_confidence_threshold() {
        let a = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let b = EagleConfig {
            confidence_threshold: 0.6,
            ..EagleConfig::default()
        };
        assert_ne!(a, b);
    }

    // ── EagleHead PartialEq per-field ──

    #[test]
    fn test_eagle_head_ne_differs_by_fc_weight_content() {
        let a = EagleHead {
            fc_weight: vec![1.0, 2.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![1.0, 9.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_head_ne_differs_by_fc_weight_length() {
        let a = EagleHead {
            fc_weight: vec![1.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![1.0, 2.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_head_ne_differs_by_draft_layers_count() {
        let layer = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![1.0],
        };
        let a = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let b = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![layer],
            share_lm_head: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_head_ne_differs_by_draft_layer_content() {
        let layer_a = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![1.0],
        };
        let layer_b = DraftLayerWeights {
            up_weight: vec![2.0],
            down_weight: vec![1.0],
        };
        let a = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![layer_a],
            share_lm_head: false,
        };
        let b = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![layer_b],
            share_lm_head: false,
        };
        assert_ne!(a, b);
    }

    // ── DraftLayerWeights PartialEq per-field ──

    #[test]
    fn test_draft_layer_weights_ne_differs_by_up_weight_length() {
        let a = DraftLayerWeights {
            up_weight: vec![1.0, 2.0],
            down_weight: vec![3.0],
        };
        let b = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![3.0],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_draft_layer_weights_ne_differs_by_down_weight_content() {
        let a = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![3.0],
        };
        let b = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![9.0],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_draft_layer_weights_eq_same_values_different_allocation() {
        let a = DraftLayerWeights {
            up_weight: vec![1.0, 2.0, 3.0],
            down_weight: vec![4.0, 5.0],
        };
        let b = DraftLayerWeights {
            up_weight: vec![1.0, 2.0, 3.0],
            down_weight: vec![4.0, 5.0],
        };
        // Equal values even though different heap allocations
        assert_eq!(a, b);
        assert_ne!(a.up_weight.as_ptr(), b.up_weight.as_ptr());
    }

    // ── Clone mutation independence ──

    #[test]
    fn test_eagle_head_clone_mutation_independence() {
        let mut original = EagleHead {
            fc_weight: vec![1.0, 2.0, 3.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let cloned = original.clone();
        original.fc_weight[0] = 99.0;
        // Cloned must remain unaffected
        assert!((cloned.fc_weight[0] - 1.0).abs() < 1e-7);
        assert_ne!(original, cloned);
    }

    #[test]
    fn test_draft_layer_weights_clone_mutation_independence() {
        let mut original = DraftLayerWeights {
            up_weight: vec![10.0, 20.0],
            down_weight: vec![30.0],
        };
        let cloned = original.clone();
        original.up_weight[0] = 999.0;
        assert!((cloned.up_weight[0] - 10.0).abs() < 1e-7);
        assert_ne!(original, cloned);
    }

    #[test]
    fn test_eagle_config_clone_mutation_independence() {
        let mut original = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 1024,
            num_draft_tokens: 7,
            confidence_threshold: 0.9,
        };
        let cloned = original.clone();
        {
            let mut o = original;
            o.num_draft_tokens = 20;
            o.confidence_threshold = 0.1;
            assert_eq!(o.num_draft_tokens, 20);
        }
        assert_eq!(cloned.num_draft_tokens, 7);
        assert!((cloned.confidence_threshold - 0.9).abs() < 1e-7);
    }

    // ── build_eagle_tree: softmax numerical stability ──

    #[test]
    fn test_build_eagle_tree_large_positive_logits_no_overflow() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // These would overflow exp() without the max subtraction trick.
        // After subtracting max, f32 precision causes these to collapse,
        // so the result may vary — we only verify it doesn't panic and returns valid indices.
        let logits = vec![1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 - 1.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    #[test]
    fn test_build_eagle_tree_very_negative_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // After subtracting max, most exp values underflow to 0.0.
        // Function must not panic and must return valid indices.
        let logits = vec![-1e10, -1e10 + 1.0, -1e10 + 2.0, -1e10 - 1.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    #[test]
    fn test_build_eagle_tree_very_small_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Very small logits behave like uniform after exp; function must not panic.
        let logits = vec![1e-30, 2e-30, 3e-30, 1e-30];
        let result = build_eagle_tree(&logits, 4, &config);
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    // ── build_eagle_tree: result properties ──

    #[test]
    fn test_build_eagle_tree_multi_branch_no_duplicate_indices() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let logits = vec![3.0, 5.0, 1.0, 4.0, 2.0];
        let result = build_eagle_tree(&logits, 5, &config);
        // All indices must be unique
        let mut seen = std::collections::HashSet::new();
        for &idx in &result {
            assert!(seen.insert(idx), "duplicate index {} found", idx);
        }
    }

    #[test]
    fn test_build_eagle_tree_high_conf_result_is_argmax() {
        let config = EagleConfig {
            confidence_threshold: 0.01, // very low threshold → always high confidence
            ..EagleConfig::default()
        };
        let logits = vec![0.5, 3.0, 1.0, 0.2, 4.5];
        let result = build_eagle_tree(&logits, 5, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 4);
    }

    #[test]
    fn test_build_eagle_tree_multi_branch_sorted_by_probability_desc() {
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let result = build_eagle_tree(&logits, 5, &config);
        assert!(result.len() >= 2);
        // Verify sorted by probability (which corresponds to logit magnitude)
        for i in 1..result.len() {
            assert!(
                logits[result[i - 1] as usize] >= logits[result[i] as usize],
                "results not sorted by probability at position {}",
                i
            );
        }
    }

    // ── build_eagle_tree: threshold boundary ──

    #[test]
    fn test_build_eagle_tree_threshold_very_small_positive() {
        let config = EagleConfig {
            confidence_threshold: 1e-10,
            ..EagleConfig::default()
        };
        // Almost anything will be high confidence
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_eagle_tree_threshold_just_below_one() {
        let config = EagleConfig {
            confidence_threshold: 0.999,
            ..EagleConfig::default()
        };
        // One dominant logit; exp(10)/sum ≈ 0.999... may still exceed 0.999
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 10, &config);
        // With one logit at 10.0 and rest at 0.0, softmax of index 0 is very high
        assert!(!result.is_empty());
        assert_eq!(result[0], 0);
    }

    // ── build_eagle_tree: functional properties ──

    #[test]
    fn test_build_eagle_tree_result_count_never_exceeds_three() {
        let config = EagleConfig {
            confidence_threshold: 1.0, // force multi-branch
            ..EagleConfig::default()
        };
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = build_eagle_tree(&logits, 10, &config);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_build_eagle_tree_single_logit_always_high_confidence() {
        let config = EagleConfig {
            confidence_threshold: 0.999,
            ..EagleConfig::default()
        };
        // Single logit → softmax = 1.0 → exceeds any threshold < 1.0
        let logits = vec![42.0];
        let result = build_eagle_tree(&logits, 1, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_build_eagle_tree_two_equal_logits_at_threshold() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Two equal logits: each prob = 0.5 → max_prob == threshold → multi-branch
        let logits = vec![1.0, 1.0];
        let result = build_eagle_tree(&logits, 2, &config);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_build_eagle_tree_preserves_argmax_under_shift() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // Shifting all logits by a constant should not change the argmax
        let base = vec![1.0, 5.0, 3.0, 2.0];
        let shifted: Vec<f32> = base.iter().map(|x| x + 100.0).collect();

        let result_base = build_eagle_tree(&base, 4, &config);
        let result_shifted = build_eagle_tree(&shifted, 4, &config);

        assert_eq!(result_base[0], result_shifted[0]);
    }

    // ── EagleHead structural ──

    #[test]
    fn test_eagle_head_with_many_layers() {
        let layers: Vec<DraftLayerWeights> = (0..10)
            .map(|i| DraftLayerWeights {
                up_weight: vec![i as f32; 32],
                down_weight: vec![(i + 1) as f32; 32],
            })
            .collect();
        let head = EagleHead {
            fc_weight: vec![0.0; 64],
            draft_layers: layers,
            share_lm_head: true,
        };
        assert_eq!(head.draft_layers.len(), 10);
        // Verify each layer's weights are distinct
        for i in 0..10 {
            assert!((head.draft_layers[i].up_weight[0] - i as f32).abs() < 1e-6);
            assert!((head.draft_layers[i].down_weight[0] - (i + 1) as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_eagle_head_fc_weight_len_matches_hidden_size_product() {
        let hidden = 64;
        // fc_weight should be hidden_size * 2 * hidden_size for a fused layer
        let expected_len = hidden * 2 * hidden;
        let head = EagleHead {
            fc_weight: vec![0.0; expected_len],
            draft_layers: vec![],
            share_lm_head: false,
        };
        assert_eq!(head.fc_weight.len(), expected_len);
    }

    // ── DraftLayerWeights with different weight patterns ──

    #[test]
    fn test_draft_layer_weights_all_ones() {
        let layer = DraftLayerWeights {
            up_weight: vec![1.0; 100],
            down_weight: vec![1.0; 100],
        };
        assert!(layer.up_weight.iter().all(|&w| (w - 1.0).abs() < 1e-7));
        assert!(layer.down_weight.iter().all(|&w| (w - 1.0).abs() < 1e-7));
    }

    #[test]
    fn test_draft_layer_weights_negative_values() {
        let layer = DraftLayerWeights {
            up_weight: vec![-0.5, -1.0, -1.5],
            down_weight: vec![-2.0, -3.0],
        };
        assert!(layer.up_weight.iter().all(|&w| w < 0.0));
        assert!(layer.down_weight.iter().all(|&w| w < 0.0));
    }

    #[test]
    fn test_draft_layer_weights_mixed_signs() {
        let layer = DraftLayerWeights {
            up_weight: vec![-1.0, 0.0, 1.0],
            down_weight: vec![0.5, -0.5],
        };
        assert!(layer.up_weight[0] < 0.0);
        assert!((layer.up_weight[1]).abs() < 1e-7);
        assert!(layer.up_weight[2] > 0.0);
    }

    // ── eagle_draft: various input sizes ──

    #[test]
    fn test_eagle_draft_empty_head_still_errors() {
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let config = EagleConfig::default();
        let result = eagle_draft(&[], &[], &head, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_eagle_draft_mismatched_input_lengths_still_errors() {
        let head = EagleHead {
            fc_weight: vec![0.0; 16],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let config = EagleConfig {
            hidden_size: 4,
            ..EagleConfig::default()
        };
        let result = eagle_draft(&[1.0, 2.0], &[3.0, 4.0, 5.0], &head, &config);
        assert!(result.is_err());
    }

    // ── build_eagle_tree: repeated calls idempotency ──

    #[test]
    fn test_build_eagle_tree_same_input_same_output() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![2.0, 5.0, 1.0, 3.0];
        let r1 = build_eagle_tree(&logits, 4, &config);
        let r2 = build_eagle_tree(&logits, 4, &config);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_build_eagle_tree_different_config_different_result() {
        let high_thresh = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let low_thresh = EagleConfig {
            confidence_threshold: 0.01,
            ..EagleConfig::default()
        };
        let logits = vec![2.0, 5.0, 1.0, 3.0];
        let r_high = build_eagle_tree(&logits, 4, &high_thresh);
        let r_low = build_eagle_tree(&logits, 4, &low_thresh);
        // Low threshold → high confidence → single branch
        assert_eq!(r_low.len(), 1);
        // High threshold → likely multi-branch
        assert!(r_high.len() >= 1);
        // The first choice should be the same (argmax)
        assert_eq!(r_low[0], r_high[0]);
    }

    // ── Debug trait: exhaustive field coverage ──

    #[test]
    fn test_draft_layer_weights_debug_shows_internal_values() {
        let layer = DraftLayerWeights {
            up_weight: vec![7.5],
            down_weight: vec![3.2],
        };
        let debug = format!("{:?}", layer);
        assert!(debug.contains("7.5"));
        assert!(debug.contains("3.2"));
    }

    #[test]
    fn test_eagle_config_debug_roundtrip_identity() {
        let config = EagleConfig {
            num_draft_layers: 3,
            hidden_size: 512,
            num_draft_tokens: 10,
            confidence_threshold: 0.85,
        };
        let debug = format!("{:?}", config);
        // Debug output should contain all field names
        assert!(debug.contains("num_draft_layers: 3"));
        assert!(debug.contains("hidden_size: 512"));
        assert!(debug.contains("num_draft_tokens: 10"));
    }

    // ── build_eagle_tree: f32 NaN input ──

    #[test]
    fn test_build_eagle_tree_handles_nan_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        // NaN mixed in — the function should not panic
        let logits = vec![1.0, f32::NAN, 2.0, 3.0];
        let result = build_eagle_tree(&logits, 4, &config);
        // Result must not be empty and indices must be valid
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    #[test]
    fn test_build_eagle_tree_handles_inf_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::INFINITY, 0.0, 0.0, 0.0];
        let result = build_eagle_tree(&logits, 4, &config);
        assert!(!result.is_empty());
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_build_eagle_tree_handles_neg_inf_logits() {
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let result = build_eagle_tree(&logits, 4, &config);
        assert!(!result.is_empty());
        assert_eq!(result[0], 1);
    }

    // ── Round 3: 15 additional tests ──

    #[test]
    fn test_build_eagle_tree_all_nan_logits_returns_valid_indices() {
        // Arrange: all-NaN logits should not panic; max_logit = NEG_INFINITY,
        // so all exp values become 0.0, sum=0.0, max_prob=0.0.
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::NAN, f32::NAN, f32::NAN];

        // Act
        let result = build_eagle_tree(&logits, 3, &config);

        // Assert: must not panic, result contains valid indices
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 3, "index {} out of range", idx);
        }
    }

    #[test]
    fn test_build_eagle_tree_all_neg_inf_logits() {
        // Arrange: all -inf logits → max_logit = -inf, exp(-inf - (-inf)) = exp(0) = 1.0
        // so all probs equal 1.0, uniform → multi-branch.
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::NEG_INFINITY; 5];

        // Act
        let result = build_eagle_tree(&logits, 5, &config);

        // Assert: uniform → max_prob = 1/5 = 0.2 < 0.5 → multi-branch top-3
        assert_eq!(result.len(), 3);
        for &idx in &result {
            assert!((idx as usize) < 5);
        }
    }

    #[test]
    fn test_build_eagle_tree_subnormal_logits() {
        // Arrange: subnormal f32 values (tiny positive, denormalized)
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![1e-45f32, 2e-45f32, 3e-45f32, 1e-45f32];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: no panic, valid indices
        assert!(!result.is_empty());
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    #[test]
    fn test_build_eagle_tree_all_positive_inf() {
        // Arrange: all +inf → max_logit = +inf, exp(inf - inf) = exp(0) = 1.0
        // all probs equal → uniform → multi-branch.
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::INFINITY; 4];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: uniform → multi-branch
        assert!(!result.is_empty());
        assert!(result.len() <= 3);
        for &idx in &result {
            assert!((idx as usize) < 4);
        }
    }

    #[test]
    fn test_build_eagle_tree_strict_greater_than_threshold() {
        // Arrange: confidence_threshold = 0.25, and 4 uniform logits →
        // each prob = 0.25, max_prob == 0.25 == threshold.
        // The check is `max_prob > threshold`, so 0.25 > 0.25 is false → multi-branch.
        let config = EagleConfig {
            confidence_threshold: 0.25,
            ..EagleConfig::default()
        };
        let logits = vec![1.0, 1.0, 1.0, 1.0];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: max_prob == threshold → NOT strictly greater → multi-branch
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_eagle_tree_confidence_just_above_threshold() {
        // Arrange: construct logits where one token has slightly more than
        // 50% probability. Three logits: [10.0, 0.0, 0.0].
        // Softmax: p0 ≈ 0.9999... > 0.5 → high confidence → single branch.
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![10.0, 0.0, 0.0];

        // Act
        let result = build_eagle_tree(&logits, 3, &config);

        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_eagle_draft_error_display_is_human_readable() {
        // Arrange
        let head = EagleHead {
            fc_weight: vec![1.0; 4],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let config = EagleConfig::default();

        // Act
        let err = eagle_draft(&[1.0], &[2.0], &head, &config).unwrap_err();
        let displayed = format!("{}", err);

        // Assert: BackendError::Other Display format is "backend error: {msg}"
        assert!(displayed.starts_with("backend error:"));
        assert!(displayed.contains("EAGLE draft head forward not yet implemented"));
    }

    #[test]
    fn test_backend_error_other_variant_is_std_error() {
        // Arrange: verify BackendError implements std::error::Error
        let head = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let config = EagleConfig::default();

        // Act
        let err = eagle_draft(&[], &[], &head, &config).unwrap_err();

        // Assert: can be used as a dyn std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_draft_layer_weights_single_element_vectors() {
        // Arrange: minimal valid DraftLayerWeights with single-element vecs
        let layer = DraftLayerWeights {
            up_weight: vec![42.0],
            down_weight: vec![-7.0],
        };

        // Act & Assert: construction, access, equality
        assert!((layer.up_weight[0] - 42.0).abs() < 1e-6);
        assert!((layer.down_weight[0] - (-7.0)).abs() < 1e-6);
        assert_eq!(layer.up_weight.len(), 1);
        assert_eq!(layer.down_weight.len(), 1);

        let clone = layer.clone();
        assert_eq!(layer, clone);
    }

    #[test]
    fn test_eagle_head_asymmetric_layer_weight_sizes() {
        // Arrange: layers with different up/down weight sizes
        let layer0 = DraftLayerWeights {
            up_weight: vec![1.0; 64],
            down_weight: vec![2.0; 32],
        };
        let layer1 = DraftLayerWeights {
            up_weight: vec![3.0; 16],
            down_weight: vec![4.0; 128],
        };
        let head = EagleHead {
            fc_weight: vec![0.0; 256],
            draft_layers: vec![layer0, layer1],
            share_lm_head: true,
        };

        // Act & Assert
        assert_eq!(head.draft_layers[0].up_weight.len(), 64);
        assert_eq!(head.draft_layers[0].down_weight.len(), 32);
        assert_eq!(head.draft_layers[1].up_weight.len(), 16);
        assert_eq!(head.draft_layers[1].down_weight.len(), 128);
    }

    #[test]
    fn test_build_eagle_tree_deterministic_across_vocab_sizes() {
        // Arrange: same relative logits pattern but with extra padding zeros
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let small_logits = vec![5.0, 3.0, 1.0];
        let mut large_logits = vec![5.0, 3.0, 1.0];
        large_logits.extend(vec![0.0; 997]); // pad to 1000

        // Act
        let result_small = build_eagle_tree(&small_logits, 3, &config);
        let result_large = build_eagle_tree(&large_logits, 1000, &config);

        // Assert: top-3 indices should be the same since relative ordering preserved
        assert_eq!(result_small.len(), 3);
        assert_eq!(result_large.len(), 3);
        assert_eq!(result_small[0], result_large[0]);
        assert_eq!(result_small[1], result_large[1]);
        assert_eq!(result_small[2], result_large[2]);
    }

    #[test]
    fn test_eagle_config_f32_nan_threshold() {
        // Arrange: NaN threshold — max_prob > NaN is always false → multi-branch
        let config = EagleConfig {
            confidence_threshold: f32::NAN,
            ..EagleConfig::default()
        };
        let logits = vec![100.0, 0.0, 0.0];

        // Act
        let result = build_eagle_tree(&logits, 3, &config);

        // Assert: NaN comparison → always multi-branch since max_prob > NaN is false
        assert!(!result.is_empty());
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_eagle_config_clone_then_original_modified() {
        // Arrange
        let mut config = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 1024,
            num_draft_tokens: 5,
            confidence_threshold: 0.8,
        };
        let snapshot = config.clone();

        // Act: modify original
        config.num_draft_layers = 99;
        config.hidden_size = 0;
        config.num_draft_tokens = 0;
        config.confidence_threshold = 0.0;

        // Assert: snapshot is unaffected
        assert_eq!(snapshot.num_draft_layers, 2);
        assert_eq!(snapshot.hidden_size, 1024);
        assert_eq!(snapshot.num_draft_tokens, 5);
        assert!((snapshot.confidence_threshold - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_build_eagle_tree_two_logits_one_zero() {
        // Arrange: one logit is zero, one is positive → softmax of positive ≈ 1.0
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![0.0, 5.0];

        // Act
        let result = build_eagle_tree(&logits, 2, &config);

        // Assert: high confidence → single branch → index 1
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_eagle_head_clone_with_layers_deep_copy() {
        // Arrange: head with layers, clone, mutate original's layer content
        let layer = DraftLayerWeights {
            up_weight: vec![1.0, 2.0, 3.0],
            down_weight: vec![4.0, 5.0],
        };
        let mut original = EagleHead {
            fc_weight: vec![10.0, 20.0],
            draft_layers: vec![layer],
            share_lm_head: false,
        };
        let cloned = original.clone();

        // Act: mutate original
        original.fc_weight[0] = 999.0;
        original.draft_layers[0].up_weight[0] = 888.0;

        // Assert: cloned is unaffected (deep copy)
        assert!((cloned.fc_weight[0] - 10.0).abs() < 1e-6);
        assert!((cloned.draft_layers[0].up_weight[0] - 1.0).abs() < 1e-6);
    }

    // ── Round 4: 13 additional tests ──

    #[test]
    fn test_eagle_config_infinity_confidence_threshold_forces_multi_branch() {
        // Arrange: threshold = +inf → max_prob > inf is always false → multi-branch
        let config = EagleConfig {
            confidence_threshold: f32::INFINITY,
            ..EagleConfig::default()
        };
        let logits = vec![100.0, 0.0, 0.0, 0.0];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: even a dominant logit can't exceed +inf → multi-branch
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_eagle_config_partial_eq_with_nan_confidence_threshold() {
        // Arrange: NaN != NaN by IEEE 754, so PartialEq should return false
        let a = EagleConfig {
            confidence_threshold: f32::NAN,
            ..EagleConfig::default()
        };
        let b = EagleConfig {
            confidence_threshold: f32::NAN,
            ..EagleConfig::default()
        };

        // Act & Assert: derived PartialEq uses f32's eq, NaN != NaN
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_config_debug_with_negative_confidence_threshold() {
        // Arrange
        let config = EagleConfig {
            confidence_threshold: -0.5,
            ..EagleConfig::default()
        };

        // Act
        let debug = format!("{:?}", config);

        // Assert: negative threshold should appear in debug output
        assert!(debug.contains("-0.5") || debug.contains("confidence_threshold"));
    }

    #[test]
    fn test_eagle_head_partial_eq_differs_by_layer_order() {
        // Arrange: same layers in different order should not be equal
        let layer_x = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![2.0],
        };
        let layer_y = DraftLayerWeights {
            up_weight: vec![9.0],
            down_weight: vec![8.0],
        };
        let a = EagleHead {
            fc_weight: vec![0.0],
            draft_layers: vec![layer_x.clone(), layer_y.clone()],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![0.0],
            draft_layers: vec![layer_y, layer_x],
            share_lm_head: true,
        };

        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_head_clone_share_lm_head_independence() {
        // Arrange: clone an EagleHead and verify share_lm_head is copied, not shared
        let original = EagleHead {
            fc_weight: vec![],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let mut cloned = original.clone();

        // Act: modify cloned's boolean field
        cloned.share_lm_head = false;

        // Assert: original unaffected
        assert!(original.share_lm_head);
        assert!(!cloned.share_lm_head);
    }

    #[test]
    fn test_draft_layer_weights_partial_eq_empty_vs_nonempty_up() {
        // Arrange: empty up_weight vs non-empty up_weight
        let a = DraftLayerWeights {
            up_weight: vec![],
            down_weight: vec![1.0],
        };
        let b = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![1.0],
        };

        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn test_draft_layer_weights_debug_with_empty_down_weight() {
        // Arrange
        let layer = DraftLayerWeights {
            up_weight: vec![5.0, 6.0],
            down_weight: vec![],
        };

        // Act
        let debug = format!("{:?}", layer);

        // Assert: should contain struct name and up_weight content
        assert!(debug.contains("DraftLayerWeights"));
        assert!(debug.contains("up_weight"));
        assert!(debug.contains("down_weight"));
    }

    #[test]
    fn test_build_eagle_tree_alternating_high_low_logits() {
        // Arrange: alternating large and small logits
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![100.0, 0.0, 90.0, 0.0, 80.0, 0.0];

        // Act
        let result = build_eagle_tree(&logits, 6, &config);

        // Assert: index 0 dominates, high confidence → single branch
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_build_eagle_tree_single_neg_inf_among_finite() {
        // Arrange: one -inf, rest finite
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: index 3 (3.0) dominates → high confidence → single branch
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 3);
    }

    #[test]
    fn test_build_eagle_tree_threshold_zero_with_uniform_logits() {
        // Arrange: threshold 0.0, uniform logits → max_prob = 1/4 = 0.25 > 0.0 → high confidence
        let config = EagleConfig {
            confidence_threshold: 0.0,
            ..EagleConfig::default()
        };
        let logits = vec![1.0, 1.0, 1.0, 1.0];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: 0.25 > 0.0 → high confidence → single branch (first index)
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_eagle_tree_result_indices_within_vocab_bounds() {
        // Arrange: vocab_size=3 but provide 3 logits
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let logits = vec![5.0, 3.0, 8.0];

        // Act
        let result = build_eagle_tree(&logits, 3, &config);

        // Assert: all returned indices must be < vocab_size
        for &idx in &result {
            assert!((idx as usize) < 3, "index {} >= vocab_size 3", idx);
        }
        // Index 2 should be the top pick
        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_build_eagle_tree_dominant_negative_vs_uniform_negative() {
        // Arrange: one less-negative value among equally more-negative values
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![-1.0, -5.0, -5.0, -5.0, -5.0];

        // Act
        let result = build_eagle_tree(&logits, 5, &config);

        // Assert: index 0 (-1.0) strongly dominates → high confidence → single branch
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_eagle_draft_error_with_custom_config() {
        // Arrange: custom config with non-zero hidden_size
        let head = EagleHead {
            fc_weight: vec![0.5; 128],
            draft_layers: vec![DraftLayerWeights {
                up_weight: vec![1.0; 64],
                down_weight: vec![2.0; 64],
            }],
            share_lm_head: false,
        };
        let config = EagleConfig {
            num_draft_layers: 2,
            hidden_size: 4096,
            num_draft_tokens: 10,
            confidence_threshold: 0.9,
        };

        // Act
        let result = eagle_draft(&[0.1; 4096], &[0.2; 4096], &head, &config);

        // Assert: still returns error regardless of config
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not yet implemented"));
    }

    // ── Round 5: 10 additional tests ──

    #[test]
    fn test_build_eagle_tree_negative_threshold_always_high_confidence() {
        // Arrange: negative threshold means any non-negative max_prob > negative → high confidence
        let config = EagleConfig {
            confidence_threshold: -1.0,
            ..EagleConfig::default()
        };
        // Uniform logits → max_prob = 0.25 > -1.0 → high confidence → single branch
        let logits = vec![1.0, 1.0, 1.0, 1.0];

        // Act
        let result = build_eagle_tree(&logits, 4, &config);

        // Assert: even uniform distribution exceeds -1.0 threshold → single branch
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_eagle_tree_max_logit_at_last_index() {
        // Arrange: the dominant token is at the very last position
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let mut logits = vec![0.0; 100];
        logits[99] = 50.0; // last index dominates

        // Act
        let result = build_eagle_tree(&logits, 100, &config);

        // Assert: high confidence → single branch → last index
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 99);
    }

    #[test]
    fn test_build_eagle_tree_two_logits_low_confidence_returns_both() {
        // Arrange: two equal logits with threshold > 0.5 → multi-branch returns both
        let config = EagleConfig {
            confidence_threshold: 0.99,
            ..EagleConfig::default()
        };
        let logits = vec![3.0, 3.0];

        // Act
        let result = build_eagle_tree(&logits, 2, &config);

        // Assert: multi-branch, but only 2 available → returns both, sorted
        assert_eq!(result.len(), 2);
        // Both indices must be present
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_draft_layer_weights_with_nan_values() {
        // Arrange: weights containing NaN — construction and access must not panic
        let layer = DraftLayerWeights {
            up_weight: vec![f32::NAN, 1.0, f32::NAN],
            down_weight: vec![2.0, f32::NAN],
        };

        // Act & Assert: access and check structure
        assert_eq!(layer.up_weight.len(), 3);
        assert_eq!(layer.down_weight.len(), 2);
        assert!(layer.up_weight[0].is_nan());
        assert!((layer.up_weight[1] - 1.0).abs() < 1e-6);
        assert!((layer.down_weight[0] - 2.0).abs() < 1e-6);
        assert!(layer.down_weight[1].is_nan());
    }

    #[test]
    fn test_draft_layer_weights_with_inf_values() {
        // Arrange: weights with positive and negative infinity
        let layer = DraftLayerWeights {
            up_weight: vec![f32::INFINITY, f32::NEG_INFINITY],
            down_weight: vec![f32::INFINITY],
        };

        // Act & Assert
        assert!(layer.up_weight[0].is_infinite() && layer.up_weight[0].is_sign_positive());
        assert!(layer.up_weight[1].is_infinite() && layer.up_weight[1].is_sign_negative());
        assert!(layer.down_weight[0].is_infinite());
        assert_eq!(layer.down_weight.len(), 1);
    }

    #[test]
    fn test_draft_layer_weights_clone_down_weight_independence() {
        // Arrange: verify down_weight vec is independently allocated after clone
        let mut original = DraftLayerWeights {
            up_weight: vec![1.0],
            down_weight: vec![10.0, 20.0, 30.0],
        };
        let cloned = original.clone();

        // Act: mutate original's down_weight
        original.down_weight[1] = 999.0;

        // Assert: cloned unaffected
        assert!((cloned.down_weight[1] - 20.0).abs() < 1e-6);
        assert_ne!(original.down_weight.as_ptr(), cloned.down_weight.as_ptr());
    }

    #[test]
    fn test_eagle_head_partial_eq_fc_weight_same_len_differs_at_last_element() {
        // Arrange: fc_weight differs only at the last element
        let a = EagleHead {
            fc_weight: vec![1.0, 2.0, 3.0],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let b = EagleHead {
            fc_weight: vec![1.0, 2.0, 9.0],
            draft_layers: vec![],
            share_lm_head: true,
        };

        // Act & Assert: single element difference at end → not equal
        assert_ne!(a, b);
    }

    #[test]
    fn test_eagle_config_negative_confidence_threshold_debug_output() {
        // Arrange: verify debug output includes the negative value literally
        let config = EagleConfig {
            num_draft_layers: 1,
            hidden_size: 256,
            num_draft_tokens: 3,
            confidence_threshold: -0.42,
        };

        // Act
        let debug = format!("{:?}", config);

        // Assert: must contain the negative value and field name
        assert!(debug.contains("confidence_threshold"));
        assert!(debug.contains("-0.42"));
        assert!(debug.contains("256"));
    }

    #[test]
    fn test_eagle_head_multiple_layers_individual_clone_independence() {
        // Arrange: head with 3 layers; clone and mutate a middle layer
        let layer0 = DraftLayerWeights {
            up_weight: vec![1.0; 4],
            down_weight: vec![2.0; 4],
        };
        let mut layer1 = DraftLayerWeights {
            up_weight: vec![5.0; 4],
            down_weight: vec![6.0; 4],
        };
        let layer2 = DraftLayerWeights {
            up_weight: vec![9.0; 4],
            down_weight: vec![10.0; 4],
        };
        let mut original = EagleHead {
            fc_weight: vec![0.0; 8],
            draft_layers: vec![layer0, layer1, layer2],
            share_lm_head: false,
        };
        let cloned = original.clone();

        // Act: mutate original's middle layer
        original.draft_layers[1].up_weight[2] = 777.0;

        // Assert: cloned's middle layer unaffected
        assert!((cloned.draft_layers[1].up_weight[2] - 5.0).abs() < 1e-6);
        // First and last layers also unaffected
        assert!((cloned.draft_layers[0].up_weight[0] - 1.0).abs() < 1e-6);
        assert!((cloned.draft_layers[2].up_weight[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_eagle_tree_three_logits_with_two_equal_and_one_dominant() {
        // Arrange: two equal lower logits and one dominant
        let config = EagleConfig {
            confidence_threshold: 0.5,
            ..EagleConfig::default()
        };
        let logits = vec![1.0, 1.0, 100.0];

        // Act
        let result = build_eagle_tree(&logits, 3, &config);

        // Assert: index 2 dominates strongly → high confidence → single branch
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 2);
    }
}
