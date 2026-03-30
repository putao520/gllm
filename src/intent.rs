//! Intent SDK — 意图提取与安全护栏 API
//!
//! per SPEC 04-API-DESIGN §7.3, §7.4

use crate::client::GllmError;
use crate::knowledge::LayerTarget;

/// 意图编码配置 (per SPEC 04-API-DESIGN §7.3)
#[derive(Debug, Clone)]
pub struct IntentConfig {
    pub target: LayerTarget,
    pub truncate_at_target: bool,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            target: LayerTarget::MidSemantic,
            truncate_at_target: true,
        }
    }
}

impl IntentConfig {
    pub fn new(target: LayerTarget) -> Self {
        Self {
            target,
            truncate_at_target: true,
        }
    }
}

/// 意图编码结果
#[derive(Debug, Clone)]
pub struct IntentEncoding {
    pub embedding: Vec<f32>,
    pub actual_layer: usize,
}

/// 护栏挂载结果 (per SPEC 04-API-DESIGN §7.4)
#[derive(Debug, Clone)]
pub struct GuardrailAttachment {
    /// 挂载的实际物理层
    pub actual_layer: usize,
    /// 探针标识符
    pub probe_id: String,
}

/// 安全策略 (per SPEC 04-API-DESIGN §7.4, §9)
#[derive(Debug, Clone, Copy)]
pub enum SafetyPolicy {
    HaltAndVeto { threshold: f32 },
    SoftWarn { threshold: f32 },
}

/// 安全护栏探针 (per SPEC 04-API-DESIGN §7.4)
#[derive(Debug, Clone)]
pub enum GuardProbe {
    FromSafetensors { path: String },
    FromModel { model_id: String },
}

impl GuardProbe {
    pub fn from_safetensors(path: impl Into<String>) -> Self {
        Self::FromSafetensors {
            path: path.into(),
        }
    }

    pub fn from_model(model_id: impl Into<String>) -> Self {
        Self::FromModel {
            model_id: model_id.into(),
        }
    }
}

/// 安全策略配置 (per SPEC 04-API-DESIGN §9.2)
#[derive(Debug, Clone)]
pub struct SafetyPolicyConfig {
    pub global_guardrail_enabled: bool,
    pub halt_and_veto_threshold: f32,
    pub target_layer: LayerTarget,
}

impl Default for SafetyPolicyConfig {
    fn default() -> Self {
        Self {
            global_guardrail_enabled: false,
            halt_and_veto_threshold: 0.95,
            target_layer: LayerTarget::DeepLogic,
        }
    }
}

impl SafetyPolicyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_guardrail(mut self, enabled: bool) -> Self {
        self.global_guardrail_enabled = enabled;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.halt_and_veto_threshold = threshold;
        self
    }

    pub fn with_target_layer(mut self, target: LayerTarget) -> Self {
        self.target_layer = target;
        self
    }
}

/// Intent SDK 错误
#[derive(Debug, thiserror::Error)]
pub enum IntentError {
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("probe load failed: {0}")]
    ProbeLoadFailed(String),
    #[error("invalid layer target")]
    InvalidLayerTarget,
}

impl From<IntentError> for GllmError {
    fn from(err: IntentError) -> Self {
        GllmError::RuntimeError(format!("intent error: {}", err))
    }
}

/// 编码意图 (per SPEC 04-API-DESIGN §7.3)
pub fn encode_intent(
    _text: &str,
    _target: LayerTarget,
) -> Result<IntentEncoding, IntentError> {
    Err(IntentError::NotImplemented(
        "encode_intent requires executor integration".into(),
    ))
}

/// 挂载安全护栏 (per SPEC 04-API-DESIGN §7.4)
pub fn attach_guardrail(
    _probe: GuardProbe,
    _target: LayerTarget,
    _policy: SafetyPolicy,
) -> Result<(), IntentError> {
    Err(IntentError::NotImplemented(
        "attach_guardrail requires executor integration".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_target_discriminant() {
        assert_ne!(LayerTarget::ShallowSyntax as u8, LayerTarget::MidSemantic as u8);
        assert_ne!(LayerTarget::MidSemantic as u8, LayerTarget::DeepLogic as u8);
    }

    #[test]
    fn test_safety_policy_config_builder() {
        let config = SafetyPolicyConfig::new()
            .with_guardrail(true)
            .with_threshold(0.9)
            .with_target_layer(LayerTarget::ShallowSyntax);

        assert!(config.global_guardrail_enabled);
        assert_eq!(config.halt_and_veto_threshold, 0.9);
        assert_eq!(config.target_layer, LayerTarget::ShallowSyntax);
    }

    #[test]
    fn test_guard_probe_from_safetensors() {
        let probe = GuardProbe::from_safetensors("toxicity.safetensors");
        match probe {
            GuardProbe::FromSafetensors { path } => {
                assert_eq!(path, "toxicity.safetensors");
            }
            _ => panic!("Wrong variant"),
        }
    }
}
