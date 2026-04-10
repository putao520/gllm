//! EAGLE Draft Head — 轻量 draft 模型加速推测解码
//!
//! EAGLE 在目标模型最后一层 hidden state 上添加 1-2 层轻量 transformer，
//! 输入 hidden_state + prev_embedding，输出 next token logits。
//! EAGLE-2 根据 draft logits 置信度动态调整推测树宽度。

use crate::engine::executor::BackendError;

/// EAGLE Draft Head 配置
#[derive(Debug, Clone)]
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
#[derive(Debug)]
pub struct EagleHead {
    /// 融合层权重: [hidden_size * 2, hidden_size]
    pub fc_weight: Vec<f32>,
    /// Draft transformer 层权重 (简化为 FFN)
    pub draft_layers: Vec<DraftLayerWeights>,
    /// 是否共享主模型的 lm_head
    pub share_lm_head: bool,
}

/// Draft transformer 层的 FFN 权重
#[derive(Debug)]
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
}
