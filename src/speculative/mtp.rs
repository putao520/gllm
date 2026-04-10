//! MTP (Multi-Token Prediction) — 模型内置多 token 预测
//!
//! 某些模型 (DeepSeek V3, Qwen3) 在训练时使用 MTP loss，
//! 推理时可一次前向输出 K 个未来 token 的 logits。
//! 相当于内置 draft model，不需要额外权重。

use crate::engine::executor::BackendError;

/// MTP Head 配置
#[derive(Debug, Clone)]
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
#[derive(Debug)]
pub struct MtpHead {
    /// [depth] 个投影层，每个 [hidden_size, vocab_size]
    pub projections: Vec<Vec<f32>>,
    pub config: MtpConfig,
}

/// MTP draft: 一次前向输出 K 个 token 的 logits
pub fn mtp_draft(
    _hidden_state: &[f32],
    _head: &MtpHead,
) -> Result<Vec<Vec<f32>>, BackendError> {
    Err(BackendError::Other(
        "MTP draft forward not yet implemented".into(),
    ))
}

/// 从 MTP logits 中提取 top-1 候选 token 序列
pub fn mtp_candidates(mtp_logits: &[Vec<f32>]) -> Vec<u32> {
    mtp_logits
        .iter()
        .map(|logits| {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
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
    fn test_mtp_draft_returns_err() {
        let head = MtpHead {
            projections: vec![],
            config: MtpConfig::default(),
        };
        let result = mtp_draft(&[1.0, 2.0], &head);
        assert!(result.is_err());
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
}
