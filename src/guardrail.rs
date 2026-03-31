//! Generation Hook 飞行安全护栏系统
//!
//! per SPEC 04-API-DESIGN §7.4, §9.2
//! per SPEC 02-ARCHITECTURE §16.4 "零延迟飞行巡航审查"

use crate::generation::{GenerationHook, HookDecision};
use crate::intent::{GuardProbe, SafetyPolicy};
use crate::knowledge::LayerTarget;

/// 安全护栏探针运行器
///
/// per SPEC 04-API-DESIGN §7.4 — 从 safetensors 加载分类器权重
/// 并在生成过程中实时检测输出安全性。
///
/// 线性分类器: score = sigmoid(w · x + b)
/// 当 score > threshold 时触发 Veto。
#[derive(Debug)]
pub struct GuardProbeRunner {
    /// 探针名称
    probe_name: String,
    /// 触发阈值
    threshold: f32,
    /// 线性分类器权重 (shape: [hidden_dim])
    weight: Vec<f32>,
    /// 线性分类器偏置 (shape: [1])
    bias: f32,
    /// 隐藏层维度
    hidden_dim: usize,
    /// 目标审查层
    target_layer: LayerTarget,
}

impl GuardProbeRunner {
    /// 从 SafetyPolicy 配置创建探针运行器
    ///
    /// per SPEC 04-API-DESIGN §9.2
    pub fn from_policy(
        probe: GuardProbe,
        target_layer: LayerTarget,
        policy: SafetyPolicy,
    ) -> Result<Self, GuardProbeError> {
        let (probe_name, threshold) = match policy {
            SafetyPolicy::HaltAndVeto { threshold } => {
                let name = match &probe {
                    GuardProbe::FromSafetensors { path } => format!("safetensors:{}", path),
                    GuardProbe::FromModel { model_id } => format!("model:{}", model_id),
                };
                (name, threshold)
            }
            SafetyPolicy::SoftWarn { threshold } => {
                let name = match &probe {
                    GuardProbe::FromSafetensors { path } => format!("safetensors:{}", path),
                    GuardProbe::FromModel { model_id } => format!("model:{}", model_id),
                };
                (name, threshold)
            }
        };

        // 从 safetensors 加载权重 (per SPEC 04-API-DESIGN §7.4)
        let (weight, bias, hidden_dim) = load_probe_weights(&probe)?;

        Ok(Self {
            probe_name,
            threshold,
            weight,
            bias,
            hidden_dim,
            target_layer,
        })
    }

    /// 从 safetensors 路径创建探针（便捷方法）
    pub fn from_safetensors(
        path: impl Into<String>,
        target_layer: LayerTarget,
        policy: SafetyPolicy,
    ) -> Result<Self, GuardProbeError> {
        let probe = GuardProbe::from_safetensors(path);
        Self::from_policy(probe, target_layer, policy)
    }

    /// 从模型 ID 创建探针（便捷方法）
    pub fn from_model(
        model_id: impl Into<String>,
        target_layer: LayerTarget,
        policy: SafetyPolicy,
    ) -> Result<Self, GuardProbeError> {
        let probe = GuardProbe::from_model(model_id);
        Self::from_policy(probe, target_layer, policy)
    }

    /// 获取探针名称
    pub fn name(&self) -> &str {
        &self.probe_name
    }

    /// 获取目标审查层
    pub fn target_layer(&self) -> LayerTarget {
        self.target_layer
    }

    /// 线性分类器前向传播: score = sigmoid(w · logits + b)
    ///
    /// per SPEC 02-ARCHITECTURE §16.4 — 在模型深度层物理强插入极小安全审查头。
    /// 当前作用于 logits（词表维度），分类器权重维度应与词表大小匹配。
    ///
    /// 若权重维度小于输入维度（常见的降维分类器），使用部分点积。
    fn classify(&self, logits: &[f32]) -> f32 {
        // 线性分类器: score = w · x + b
        // 仅计算到 min(weight.len(), logits.len()) 的点积
        let dot_len = self.weight.len().min(logits.len());
        let dot_product: f32 = self.weight[..dot_len]
            .iter()
            .zip(logits[..dot_len].iter())
            .map(|(w, x)| w * x)
            .sum();

        // sigmoid 激活: score ∈ (0, 1)
        let raw_score = dot_product + self.bias;
        1.0 / (1.0 + (-raw_score).exp())
    }
}

impl GenerationHook for GuardProbeRunner {
    /// 每步生成后调用
    ///
    /// per SPEC 02-ARCHITECTURE §16.4 — 在模型深度层物理强插入极小安全审查头
    fn post_step(&self, logits: &[f32], generated_tokens: &[u32]) -> HookDecision {
        let score = self.classify(logits);

        if score > self.threshold {
            HookDecision::Veto(format!(
                "Guardrail {} vetoed: score {:.4} exceeds threshold {:.4}",
                self.probe_name, score, self.threshold
            ))
        } else if generated_tokens.len() > 1000 {
            // 防止无限生成
            HookDecision::Terminate
        } else {
            HookDecision::Continue
        }
    }
}

/// 护栏探针错误
#[derive(Debug, thiserror::Error)]
pub enum GuardProbeError {
    #[error("probe load failed: {0}")]
    ProbeLoadFailed(String),
    #[error("invalid layer target")]
    InvalidLayerTarget,
    #[error("safetensors probe error: {0}")]
    SafetensorsError(String),
    #[error("policy configuration error: {0}")]
    PolicyError(String),
    #[error("IO error: {0}")]
    IoError(String),
}

/// 从 safetensors 或模型加载探针权重
///
/// per SPEC 04-API-DESIGN §7.4 — 从 safetensors 加载分类器权重
///
/// 期望的 tensor 名称:
/// - `weight` 或 `classifier.weight`: 分类器权重 (shape: [hidden_dim] 或 [1, hidden_dim])
/// - `bias` 或 `classifier.bias`: 分类器偏置 (shape: [1])
///
/// 若 safetensors 文件不存在或格式不正确，返回明确的 Err (NO_SILENT_FALLBACK)
fn load_probe_weights(
    probe: &GuardProbe,
) -> Result<(Vec<f32>, f32, usize), GuardProbeError> {
    match probe {
        GuardProbe::FromSafetensors { path } => {
            use std::path::Path;
            let file_path = Path::new(path);

            // 检查文件存在性
            if !file_path.exists() {
                return Err(GuardProbeError::ProbeLoadFailed(format!(
                    "safetensors probe file not found: {}", path
                )));
            }

            // 使用 MappedSafetensors 加载权重
            let loader = crate::loader::safetensors::MappedSafetensors::open(file_path)
                .map_err(|e| GuardProbeError::SafetensorsError(format!(
                    "failed to open safetensors '{}': {}", path, e
                )))?;

            // 尝试多种 tensor 名称加载分类器权重
            let weight_tensor = loader.tensor("weight")
                .or_else(|_| loader.tensor("classifier.weight"))
                .map_err(|e| GuardProbeError::SafetensorsError(format!(
                    "no 'weight' or 'classifier.weight' tensor found in '{}': {}", path, e
                )))?;

            // 提取 f32 权重数据
            let weight_data = weight_tensor.as_f32()
                .map_err(|e| GuardProbeError::SafetensorsError(format!(
                    "weight tensor is not f32 in '{}': {}", path, e
                )))?
                .into_owned();

            // 计算隐藏层维度（支持 [hidden_dim] 和 [1, hidden_dim] 两种形状）
            let hidden_dim = match weight_tensor.shape.as_slice() {
                [d] => *d,
                [1, d] => *d,
                [n, d] if *n == 1 => *d,
                shape => {
                    // 对于多维权重（如 [num_classes, hidden_dim]），取最后一维
                    *shape.last().ok_or_else(|| GuardProbeError::SafetensorsError(
                        "weight tensor has empty shape".to_string()
                    ))?
                }
            };

            // 若权重是 2D [num_classes, hidden_dim]，取第一行（二分类器的正类权重）
            let weight_vec = if weight_data.len() > hidden_dim && hidden_dim > 0 {
                weight_data[..hidden_dim].to_vec()
            } else {
                weight_data
            };

            // 加载偏置（可选）
            let bias: f32 = loader.tensor("bias")
                .or_else(|_| loader.tensor("classifier.bias"))
                .and_then(|t| t.as_f32())
                .ok()
                .and_then(|b| b.first().copied())
                .unwrap_or(0.0);

            Ok((weight_vec, bias, hidden_dim))
        }
        GuardProbe::FromModel { model_id } => {
            // TODO: 从 HuggingFace Hub 下载模型 safetensors 并加载
            // 完整实现需要: 1) HF Hub API 下载 2) 缓存管理 3) 权重提取
            Err(GuardProbeError::ProbeLoadFailed(format!(
                "model probe loading not yet implemented: model_id={}. \
                 Use GuardProbe::FromSafetensors with a local safetensors file instead.",
                model_id
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_probe_runner_from_safetensors_nonexistent() {
        // 文件不存在时应返回 Err
        let result = GuardProbeRunner::from_safetensors(
            "nonexistent_classifier.safetensors",
            LayerTarget::DeepLogic,
            SafetyPolicy::HaltAndVeto { threshold: 0.95 },
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, GuardProbeError::ProbeLoadFailed(ref msg) if msg.contains("not found")),
            "Expected ProbeLoadFailed with 'not found', got: {:?}", err
        );
    }

    #[test]
    fn test_guard_probe_runner_from_model_unimplemented() {
        // FromModel 当前未实现，应返回 Err
        let result = GuardProbeRunner::from_model(
            "toxicity-detector-v1",
            LayerTarget::MidSemantic,
            SafetyPolicy::SoftWarn { threshold: 0.8 },
        );

        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), GuardProbeError::ProbeLoadFailed(_)),
            "Expected ProbeLoadFailed for unimplemented model probe"
        );
    }

    #[test]
    fn test_classify_with_manual_weights() {
        // 手动构造 GuardProbeRunner 测试 classify 逻辑
        // 使用一个简单的权重: weight = [1.0, 0.0], bias = 0.0
        // 对于 logits = [1.0, 0.0], score = sigmoid(1.0) ≈ 0.731
        let runner = GuardProbeRunner {
            probe_name: "test".to_string(),
            threshold: 0.5,
            weight: vec![1.0, 0.0],
            bias: 0.0,
            hidden_dim: 2,
            target_layer: LayerTarget::MidSemantic,
        };

        let score = runner.classify(&[1.0, 0.0]);
        let expected = 1.0 / (1.0 + (-1.0f32).exp()); // sigmoid(1.0)
        assert!(
            (score - expected).abs() < 1e-5,
            "sigmoid(1.0) expected {}, got {}", expected, score
        );
    }

    #[test]
    fn test_classify_with_bias() {
        // 测试偏置效果: weight = [1.0], bias = -5.0
        // logits = [0.0] → score = sigmoid(-5.0) ≈ 0.0067
        let runner = GuardProbeRunner {
            probe_name: "test_bias".to_string(),
            threshold: 0.5,
            weight: vec![1.0],
            bias: -5.0,
            hidden_dim: 1,
            target_layer: LayerTarget::MidSemantic,
        };

        let score = runner.classify(&[0.0]);
        let expected = 1.0 / (1.0 + 5.0f32.exp()); // sigmoid(-5.0)
        assert!(
            (score - expected).abs() < 1e-5,
            "sigmoid(-5.0) expected {}, got {}", expected, score
        );
        assert!(score < 0.5, "sigmoid(-5.0) should be close to 0");
    }

    #[test]
    fn test_classify_veto_threshold() {
        // 高分触发 Veto: weight = [10.0], bias = 0.0, threshold = 0.5
        // logits = [1.0] → score = sigmoid(10.0) ≈ 0.9999 > 0.5 → Veto
        let runner = GuardProbeRunner {
            probe_name: "test_veto".to_string(),
            threshold: 0.5,
            weight: vec![10.0],
            bias: 0.0,
            hidden_dim: 1,
            target_layer: LayerTarget::MidSemantic,
        };

        let decision = runner.post_step(&[1.0], &[1, 2, 3]);
        assert!(
            matches!(decision, HookDecision::Veto(_)),
            "Expected Veto for high score"
        );
    }

    #[test]
    fn test_classify_continue_below_threshold() {
        // 低分继续: weight = [-10.0], bias = 0.0, threshold = 0.5
        // logits = [1.0] → score = sigmoid(-10.0) ≈ 0.000045 < 0.5 → Continue
        let runner = GuardProbeRunner {
            probe_name: "test_continue".to_string(),
            threshold: 0.5,
            weight: vec![-10.0],
            bias: 0.0,
            hidden_dim: 1,
            target_layer: LayerTarget::MidSemantic,
        };

        let decision = runner.post_step(&[1.0], &[1, 2, 3]);
        assert!(
            matches!(decision, HookDecision::Continue),
            "Expected Continue for low score"
        );
    }

    #[test]
    fn test_classify_terminate_on_long_generation() {
        // 超过 1000 tokens 时终止（防止无限生成）
        let runner = GuardProbeRunner {
            probe_name: "test_terminate".to_string(),
            threshold: 0.99,
            weight: vec![0.0], // 零权重 → score ≈ 0.5
            bias: 0.0,
            hidden_dim: 1,
            target_layer: LayerTarget::MidSemantic,
        };

        let long_tokens: Vec<u32> = (0..1001).collect();
        let decision = runner.post_step(&[0.0], &long_tokens);
        assert!(
            matches!(decision, HookDecision::Terminate),
            "Expected Terminate for >1000 tokens"
        );
    }
}
