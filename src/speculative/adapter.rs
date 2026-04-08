//! §17.2 Adapter — 零参数 Draft 投影头
//!
//! Adapter 是附在 L2_hot 变体末尾的微型投影图，将中间 hidden state 映射到 vocab 空间。
//! 图结构: RmsNorm(hidden_size) → MatMul(normed, lm_head.weight^T) → logits
//!
//! **零额外参数**: 直接复用 lm_head.weight (Phase A)
//! **可选蒸馏**: Phase B 添加 residual_delta, 前 100 步用 full model logits 蒸馏

use std::sync::Arc;

/// Draft Adapter 配置
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Hidden size (from model config)
    pub hidden_size: usize,
    /// Vocab size (from model config)
    pub vocab_size: usize,
    /// LayerNorm epsilon
    pub rms_norm_eps: f32,
    /// Phase B 可选: 是否启用蒸馏残差
    pub enable_distillation: bool,
    /// 蒸馏步数 (Phase B)
    pub distillation_steps: usize,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            vocab_size: 0,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        }
    }
}

/// Draft Adapter 运行时
///
/// §17.2: Adapter 将浅层变体的 hidden state 投影到 vocab 空间。
/// Phase A: 直接复用 lm_head.weight (零额外参数)
/// Phase B: 添加 residual_delta 通过在线蒸馏学习
pub struct DraftAdapter {
    config: AdapterConfig,
    /// Phase A: 共享的 lm_head.weight 引用 [vocab_size, hidden_size]
    /// Phase B: lm_head.weight + residual_delta
    weight: Arc<Vec<f32>>,
    /// Phase B: residual_delta — 在线蒸馏学习的修正项
    /// 当 enable_distillation=true 时，前 distillation_steps 步用 full model logits 蒸馏
    residual_delta: Option<Vec<f32>>,
    /// 已完成的蒸馏步数
    distillation_step: usize,
    /// 最后一个 norm 层的权重 [hidden_size] — 复用模型最后一层的 norm 权重
    norm_weight: Arc<Vec<f32>>,
}

impl std::fmt::Debug for DraftAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftAdapter")
            .field("hidden_size", &self.config.hidden_size)
            .field("vocab_size", &self.config.vocab_size)
            .field("distillation_step", &self.distillation_step)
            .field("has_residual_delta", &self.residual_delta.is_some())
            .finish()
    }
}

impl DraftAdapter {
    /// 创建 Phase A Adapter (零额外参数, 共享 lm_head.weight)
    ///
    /// # Arguments
    /// * `config` - Adapter 配置
    /// * `lm_head_weight` - lm_head.weight 张量 [vocab_size, hidden_size] (共享引用)
    /// * `norm_weight` - 最后一层 RmsNorm 权重 [hidden_size] (共享引用)
    pub fn new_phase_a(
        config: AdapterConfig,
        lm_head_weight: Arc<Vec<f32>>,
        norm_weight: Arc<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            weight: lm_head_weight,
            residual_delta: None,
            distillation_step: 0,
            norm_weight,
        }
    }

    /// 创建 Phase B Adapter (带蒸馏残差)
    ///
    /// Phase B 在 Phase A 基础上添加 `residual_delta` 参数，
    /// 通过前 `distillation_steps` 步用 full model 的真实 logits 蒸馏学习。
    /// 额外参数: vocab_size × hidden_size (≈0.1% 总模型大小)
    pub fn new_phase_b(
        config: AdapterConfig,
        lm_head_weight: Arc<Vec<f32>>,
        norm_weight: Arc<Vec<f32>>,
    ) -> Self {
        let residual_delta = vec![0.0f32; config.vocab_size * config.hidden_size];
        Self {
            config,
            weight: lm_head_weight,
            residual_delta: Some(residual_delta),
            distillation_step: 0,
            norm_weight,
        }
    }

    /// Adapter 前向传播: hidden_state → logits
    ///
    /// §17.2 图结构:
    /// 1. RmsNorm(hidden_state, norm_weight)
    /// 2. MatMul(normed, weight^T) → logits
    ///
    /// # Arguments
    /// * `hidden` - Hidden state [hidden_size] (来自 L2_hot 变体输出)
    ///
    /// # Returns
    /// Logits 向量 [vocab_size]
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        assert_eq!(hidden.len(), hidden_size, "hidden size mismatch");

        // Step 1: RmsNorm
        let normed = self.rms_norm(hidden);

        // Step 2: MatMul(normed, weight^T) → logits[vocab_size]
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                let w = self.weight[v * hidden_size + h];
                let delta = self.residual_delta
                    .as_ref()
                    .map(|d| d[v * hidden_size + h])
                    .unwrap_or(0.0);
                sum += normed[h] * (w + delta);
            }
            logits[v] = sum;
        }

        logits
    }

    /// 批量 Adapter 前向: 多个 hidden states → 多组 logits
    ///
    /// 用于 EqSpec batch 场景 (§17.4): 多个 sequence 同时 draft
    pub fn forward_batch(&self, hiddens: &[&[f32]]) -> Vec<Vec<f32>> {
        hiddens.iter().map(|h| self.forward(h)).collect()
    }

    /// 在线蒸馏更新 (Phase B only)
    ///
    /// §17.2 Phase B: 用 full model 的真实 logits 做蒸馏信号，
    /// SGD 微调 residual_delta 参数。
    ///
    /// # Arguments
    /// * `draft_logits` - Adapter 产生的 logits [vocab_size]
    /// * `target_logits` - Full model 产生的 logits [vocab_size] (ground truth)
    /// * `hidden` - 输入 hidden state [hidden_size]
    /// * `learning_rate` - SGD 学习率
    pub fn distill_step(
        &mut self,
        draft_logits: &[f32],
        target_logits: &[f32],
        hidden: &[f32],
        learning_rate: f32,
    ) -> f32 {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        // Compute gradient and normed hidden before mutable borrow of delta
        let grad = softmax_diff(draft_logits, target_logits, vocab_size);
        let normed = rms_norm_free(hidden, &self.norm_weight, eps);

        let delta = self.residual_delta.as_mut().expect("Phase B required for distillation");
        let mut total_loss = 0.0f32;
        for v in 0..vocab_size {
            total_loss += grad[v] * grad[v];
            for h in 0..hidden_size {
                delta[v * hidden_size + h] -= learning_rate * grad[v] * normed[h];
            }
        }

        self.distillation_step += 1;
        total_loss / vocab_size as f32
    }

    /// 检查蒸馏是否完成
    pub fn is_distillation_complete(&self) -> bool {
        self.distillation_step >= self.config.distillation_steps
    }

    /// 获取蒸馏进度
    pub fn distillation_progress(&self) -> (usize, usize) {
        (self.distillation_step, self.config.distillation_steps)
    }

    /// Adapter 参数量 (bytes)
    pub fn parameter_bytes(&self) -> usize {
        let base = 0; // Phase A: 零额外参数 (共享 lm_head)
        let delta = self.residual_delta
            .as_ref()
            .map(|d| d.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        base + delta
    }

    // ---- Internal helpers ----

    /// RmsNorm: x / rms(x) * weight
    fn rms_norm(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let eps = self.config.rms_norm_eps;
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        x.iter()
            .zip(self.norm_weight.iter())
            .map(|(&xi, &wi)| xi * inv_rms * wi)
            .collect()
    }

    /// Compute logits gradient for distillation
    fn logits_gradient(&self, draft: &[f32], target: &[f32]) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        // Softmax difference as gradient signal
        let (draft_probs, target_probs) = (softmax(draft), softmax(target));
        let mut grad = vec![0.0f32; vocab_size];
        for i in 0..vocab_size {
            grad[i] = draft_probs[i] - target_probs[i];
        }
        grad
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    crate::moe::routing::softmax(x)
}

/// Free function for softmax difference (avoids borrow conflicts in distill_step)
fn softmax_diff(draft: &[f32], target: &[f32], vocab_size: usize) -> Vec<f32> {
    let (draft_probs, target_probs) = (softmax(draft), softmax(target));
    let mut grad = vec![0.0f32; vocab_size];
    for i in 0..vocab_size {
        grad[i] = draft_probs[i] - target_probs[i];
    }
    grad
}

/// Free function for RmsNorm (avoids borrow conflicts in distill_step)
fn rms_norm_free(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter(hidden_size: usize, vocab_size: usize) -> DraftAdapter {
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        DraftAdapter::new_phase_a(config, lm_head, norm_weight)
    }

    #[test]
    fn test_adapter_forward_output_size() {
        let adapter = make_adapter(64, 1000);
        let hidden = vec![1.0f32; 64];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 1000);
    }

    #[test]
    fn test_adapter_batch_forward() {
        let adapter = make_adapter(64, 1000);
        let h1 = vec![1.0f32; 64];
        let h2 = vec![0.5f32; 64];
        let results = adapter.forward_batch(&[&h1, &h2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1000);
        assert_eq!(results[1].len(), 1000);
    }

    #[test]
    fn test_adapter_phase_b_distillation() {
        let hidden_size = 32;
        let vocab_size = 100;
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 5,
        };
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);

        assert!(!adapter.is_distillation_complete());

        let hidden = vec![1.0f32; hidden_size];
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            let target = vec![0.0f32; vocab_size]; // dummy target
            let _loss = adapter.distill_step(&draft, &target, &hidden, 0.01);
        }

        assert!(adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (5, 5));
    }

    #[test]
    fn test_adapter_phase_a_zero_extra_params() {
        let adapter = make_adapter(64, 1000);
        // Phase A: 零额外参数
        assert_eq!(adapter.parameter_bytes(), 0);
    }

    #[test]
    fn test_adapter_phase_b_has_delta_params() {
        let config = AdapterConfig {
            hidden_size: 32,
            vocab_size: 100,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.1f32; 100 * 32]);
        let norm_weight = Arc::new(vec![1.0f32; 32]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        // Phase B: vocab_size * hidden_size * 4 bytes
        assert_eq!(adapter.parameter_bytes(), 100 * 32 * 4);
    }
}
