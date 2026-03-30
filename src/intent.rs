//! Intent SDK (per SPEC 04-API-DESIGN §7.3-§7.4)
//!
//! 提供意图编码、安全护栏和全局安全策略配置。
//! 当前为骨架实现，底层依赖 SPEC §9-§16 的 Mega-Kernel 架构。

/// 意图编码结果
#[derive(Debug, Clone)]
pub struct IntentEncoding {
    /// 意图特征向量（低维嵌入）
    pub embedding: Vec<f32>,
    /// 置信度 [0.0, 1.0]
    pub confidence: f32,
    /// 意图标签（可选）
    pub label: Option<String>,
}

/// 安全护栏动作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailAction {
    /// 允许继续
    Allow,
    /// 拦截并返回固定回复
    Veto,
    /// 标记但允许继续（用于日志记录）
    Flag,
}

/// 护栏挂载结果
#[derive(Debug, Clone)]
pub struct GuardrailAttachment {
    /// 护栏 ID
    pub guard_id: String,
    /// 挂载的目标层
    pub target_layer: crate::knowledge::LayerTarget,
    /// 置信度阈值
    pub threshold: f32,
}

/// 意图 SDK 错误类型
#[derive(Debug, thiserror::Error)]
pub enum IntentError {
    #[error("intent encoding failed: {0}")]
    EncodingFailed(String),
    #[error("guardrail attachment failed: {0}")]
    GuardrailFailed(String),
    #[error("no model loaded")]
    NoModelLoaded,
}

/// 全局安全策略配置 (per SPEC 04-API-DESIGN §9.2)
#[derive(Debug, Clone)]
pub struct SafetyPolicyConfig {
    /// 是否启用全局护栏
    pub enabled: bool,
    /// 拦截置信度阈值 [0.0, 1.0]
    pub veto_threshold: f32,
    /// 最大生成 token 数限制（安全兜底）
    pub max_tokens_limit: usize,
}

impl Default for SafetyPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            veto_threshold: 0.95,
            max_tokens_limit: 4096,
        }
    }
}

impl SafetyPolicyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.veto_threshold = threshold;
        self
    }

    pub fn with_max_tokens(mut self, limit: usize) -> Self {
        self.max_tokens_limit = limit;
        self
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            veto_threshold: 1.0,
            max_tokens_limit: usize::MAX,
        }
    }
}

/// 意图配置
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// 目标意图标签（可选，用于有监督编码）
    pub target_label: Option<String>,
    /// 编码维度
    pub embedding_dim: usize,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            target_label: None,
            embedding_dim: 256,
        }
    }
}

impl IntentConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.target_label = Some(label.into());
        self
    }

    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }
}

/// 意图编码（骨架实现）
///
/// 将输入文本编码为低维意图向量，用于下游分类或路由决策。
/// 当前返回空向量骨架，待 SPEC §9-§16 Mega-Kernel 架构实现后接入。
pub fn encode_intent(
    _text: &str,
    _config: &IntentConfig,
) -> Result<IntentEncoding, IntentError> {
    // 骨架实现：待底层架构就绪后实现
    Ok(IntentEncoding {
        embedding: vec![0.0; _config.embedding_dim],
        confidence: 0.0,
        label: _config.target_label.clone(),
    })
}

/// 附加安全护栏（骨架实现）
///
/// 将安全分类器挂载到指定层，超过阈值时触发熔断。
pub fn attach_guardrail(
    _model_path: &str,
    _target: crate::knowledge::LayerTarget,
    _policy: &SafetyPolicyConfig,
) -> Result<GuardrailAttachment, IntentError> {
    // 骨架实现：待底层架构就绪后实现
    Ok(GuardrailAttachment {
        guard_id: format!("guard_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()),
        target_layer: _target,
        threshold: _policy.veto_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_policy_default() {
        let policy = SafetyPolicyConfig::default();
        assert!(policy.enabled);
        assert_eq!(policy.veto_threshold, 0.95);
        assert_eq!(policy.max_tokens_limit, 4096);
    }

    #[test]
    fn test_safety_policy_builder() {
        let policy = SafetyPolicyConfig::new()
            .with_threshold(0.8)
            .with_max_tokens(2048);
        assert_eq!(policy.veto_threshold, 0.8);
        assert_eq!(policy.max_tokens_limit, 2048);
    }

    #[test]
    fn test_safety_policy_disabled() {
        let policy = SafetyPolicyConfig::disabled();
        assert!(!policy.enabled);
    }

    #[test]
    fn test_intent_config() {
        let config = IntentConfig::new().with_label("cancel_subscription").with_dim(128);
        assert_eq!(config.target_label, Some("cancel_subscription".to_string()));
        assert_eq!(config.embedding_dim, 128);
    }

    #[test]
    fn test_encode_intent_skeleton() {
        let config = IntentConfig::new().with_dim(64);
        let result = encode_intent("Cancel my subscription", &config).unwrap();
        assert_eq!(result.embedding.len(), 64);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_attach_guardrail_skeleton() {
        let policy = SafetyPolicyConfig::default();
        let result = attach_guardrail(
            "toxicity.safetensors",
            crate::knowledge::LayerTarget::DeepLogic,
            &policy,
        ).unwrap();
        assert_eq!(result.target_layer, crate::knowledge::LayerTarget::DeepLogic);
        assert_eq!(result.threshold, 0.95);
    }
}
