//! Intent SDK — 意图提取与安全护栏 API
//!
//! # 概述
//!
//! 本模块提供高阶商业特性（Residual Bus）开发者接口，包括：
//!
//! - **Multi-Intent 降维提取**: 物理截断模型以加速判别式任务
//! - **In-Flight Guardrail**: 底层挂载分类器，实现零延迟熔断
//!
//! # SPEC 对应
//!
//! - `SPEC/04-API-DESIGN.md` §7 意图提取与知识外挂 (API-KNOWLEDGE-INJECTION)
//! - `SPEC/04-API-DESIGN.md` §9 全局引擎 API 配置 (API-GLOBAL-CONFIG)
//!
//! # 使用示例
//!
//! ```no_run
//! use gllm::Client;
//! use gllm::intent::SafetyPolicyConfig;
//! use gllm::knowledge::LayerTarget;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
//!
//! // 意图提取：仅计算至中层语义区即截断
//! let intent_embedding = client.encode_intent(
//!     "Cancel my subscription",
//!     LayerTarget::MidSemantic,
//! )?;
//!
//! // 挂载安全护栏
//! client.attach_guardrail(
//!     "toxicity_classifier_v1.safetensors",
//!     LayerTarget::DeepLogic,
//!     SafetyPolicyConfig::halt_and_veto(0.95),
//! )?;
//! # Ok(())
//! # }
//! ```

use crate::client::GllmError;

// Re-export LayerTarget from knowledge module for convenience
pub use crate::knowledge::LayerTarget;

/// 意图提取配置
///
/// 控制意图提取行为的高级参数。
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// 截断锚点 — 在哪一层停止计算
    pub layer_target: LayerTarget,

    /// 是否归一化输出向量
    pub normalize: bool,

    /// 输出维度 — 0 表示使用模型默认隐藏维度
    pub output_dim: usize,
}

impl IntentConfig {
    /// 创建默认配置
    pub fn new(layer_target: LayerTarget) -> Self {
        Self {
            layer_target,
            normalize: true,
            output_dim: 0,
        }
    }

    /// 创建浅层词法区配置
    pub fn shallow() -> Self {
        Self::new(LayerTarget::ShallowSyntax)
    }

    /// 创建中层语义区配置
    pub fn mid_semantic() -> Self {
        Self::new(LayerTarget::MidSemantic)
    }

    /// 创建深层逻辑区配置
    pub fn deep_logic() -> Self {
        Self::new(LayerTarget::DeepLogic)
    }

    /// 设置是否归一化
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// 设置输出维度
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = dim;
        self
    }
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self::mid_semantic()
    }
}

/// 意图编码结果
///
/// 包含提取的特征向量和元数据。
#[derive(Debug, Clone)]
pub struct IntentEncoding {
    /// 特征向量 — 可直接用于外部轻量分类器
    pub embedding: Vec<f32>,

    /// 实际截断的物理层号
    pub actual_layer: usize,

    /// 使用的锚点类型
    pub layer_target: LayerTarget,
}

/// 护栏触发后的行为模式
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GuardrailAction {
    /// 立即中止并拒绝生成
    HaltAndVeto,

    /// 用安全文本替换输出
    ReplaceWithSafeText { text: String },

    /// 记录但继续生成
    LogOnly,
}

/// 安全策略配置
///
/// 控制护栏探针行为的全局配置。
///
/// # SPEC 对应
///
/// `SPEC/04-API-DESIGN.md` §9.2 飞行安全护栏全局守护
#[derive(Debug, Clone)]
pub struct SafetyPolicyConfig {
    /// 是否在运行时为所有请求挂载护栏探针
    pub global_guardrail_enabled: bool,

    /// 当探针发现毒性特征概率 > 该阈值时，硬件级物理阻断当前请求
    pub halt_and_veto_threshold: f32,

    /// 指定探针挂载的锚点深度
    pub target_layer: LayerTarget,

    /// 触发护栏时的行为模式
    pub action: GuardrailAction,
}

impl SafetyPolicyConfig {
    /// 创建"立即中止并拒绝"策略
    pub fn halt_and_veto(threshold: f32) -> Self {
        Self {
            global_guardrail_enabled: true,
            halt_and_veto_threshold: threshold,
            target_layer: LayerTarget::DeepLogic,
            action: GuardrailAction::HaltAndVeto,
        }
    }

    /// 创建"替换文本"策略
    pub fn replace_with_safe_text(threshold: f32, text: impl Into<String>) -> Self {
        Self {
            global_guardrail_enabled: true,
            halt_and_veto_threshold: threshold,
            target_layer: LayerTarget::DeepLogic,
            action: GuardrailAction::ReplaceWithSafeText {
                text: text.into(),
            },
        }
    }

    /// 创建"仅记录"策略
    pub fn log_only(threshold: f32) -> Self {
        Self {
            global_guardrail_enabled: true,
            halt_and_veto_threshold: threshold,
            target_layer: LayerTarget::DeepLogic,
            action: GuardrailAction::LogOnly,
        }
    }

    /// 设置目标锚点
    pub fn with_target_layer(mut self, layer: LayerTarget) -> Self {
        self.target_layer = layer;
        self
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), GllmError> {
        if self.halt_and_veto_threshold < 0.0 || self.halt_and_veto_threshold > 1.0 {
            return Err(GllmError::Executor(
                crate::engine::executor::ExecutorError::Scheduler(format!(
                    "halt_and_veto_threshold must be in [0.0, 1.0], got {}",
                    self.halt_and_veto_threshold
                )),
            ));
        }
        Ok(())
    }
}

impl Default for SafetyPolicyConfig {
    fn default() -> Self {
        Self::halt_and_veto(0.95)
    }
}

/// 护栏探针挂载结果
#[derive(Debug, Clone)]
pub struct GuardrailAttachment {
    /// 探针唯一标识
    pub probe_id: String,

    /// 挂载的物理层号
    pub mounted_layer: usize,

    /// 配置的策略
    pub policy: SafetyPolicyConfig,
}

/// Intent SDK 错误
#[derive(Debug, thiserror::Error)]
pub enum IntentError {
    #[error("intent encoding not implemented: {0}")]
    NotImplemented(String),
    #[error("guard probe load failed: {0}")]
    ProbeLoadFailed(String),
    #[error("invalid layer target for current model")]
    InvalidLayerTarget,
}

impl From<IntentError> for GllmError {
    fn from(err: IntentError) -> Self {
        GllmError::Executor(crate::engine::executor::ExecutorError::Scheduler(err.to_string()))
    }
}

/// 编码意图 — Multi-Intent 降维提取
///
/// 物理砍断后续层以加速判别式任务。仅计算至指定锚点层
/// 即截断，强制返回特征向量。
///
/// # 参数
///
/// - `text`: 输入文本
/// - `layer_target`: 截断锚点
///
/// # 返回
///
/// 意图编码结果，包含特征向量和元数据。
///
/// # SPEC 对应
///
/// `SPEC/04-API-DESIGN.md` §7.3 Multi-Intent 降维提取 API
///
/// # 示例
///
/// ```no_run
/// # use gllm::Client;
/// # use gllm::knowledge::LayerTarget;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_chat("model")?;
/// let intent = client.encode_intent("Cancel my subscription", LayerTarget::MidSemantic)?;
/// // intent.embedding 可直接用于外部轻量分类器
/// # Ok(())
/// # }
/// ```
pub fn encode_intent(
    _text: &str,
    layer_target: LayerTarget,
) -> Result<IntentEncoding, GllmError> {
    // 骨架实现：返回零向量
    // TODO: P1-4 完整实现需要集成到 executor 的前向传播
    let embedding_dim = 4096; // 默认隐藏维度
    let actual_layer = match layer_target {
        LayerTarget::ShallowSyntax => 8,
        LayerTarget::MidSemantic => 16,
        LayerTarget::DeepLogic => 24,
    };
    Ok(IntentEncoding {
        embedding: vec![0.0f32; embedding_dim],
        actual_layer,
        layer_target,
    })
}

/// 挂载安全护栏 — In-Flight Guardrail
///
/// 底层挂载极小分类器，实现零延迟熔断。当探针检测到
/// 不安全内容时，根据配置的策略执行相应操作。
///
/// # 参数
///
/// - `probe_path`: 护栏探针权重文件路径（safetensors 格式）
/// - `layer_target`: 挂载锚点深度
/// - `policy`: 安全策略配置
///
/// # 返回
///
/// 护栏挂载结果，包含探针 ID 和挂载信息。
///
/// # SPEC 对应
///
/// `SPEC/04-API-DESIGN.md` §7.4 In-Flight Guardrail
///
/// # 示例
///
/// ```no_run
/// # use gllm::Client;
/// # use gllm::intent::SafetyPolicyConfig;
/// # use gllm::knowledge::LayerTarget;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_chat("model")?;
/// let attachment = client.attach_guardrail(
///     "toxicity_classifier_v1.safetensors",
///     LayerTarget::DeepLogic,
///     SafetyPolicyConfig::halt_and_veto(0.95),
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn attach_guardrail(
    _probe_path: &str,
    layer_target: LayerTarget,
    policy: SafetyPolicyConfig,
) -> Result<GuardrailAttachment, GllmError> {
    // 验证配置
    policy.validate()?;

    // 骨架实现：返回虚拟挂载信息
    // TODO: P1-4 完整实现需要加载探针权重并注册到 executor
    let mounted_layer = match layer_target {
        LayerTarget::ShallowSyntax => 8,
        LayerTarget::MidSemantic => 16,
        LayerTarget::DeepLogic => 24,
    };
    Ok(GuardrailAttachment {
        probe_id: format!("probe_{}", uuid_simulator()),
        mounted_layer,
        policy,
    })
}

/// 简单的 UUID 模拟器（骨架实现用）
fn uuid_simulator() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_target_default() {
        let target = LayerTarget::default();
        assert_eq!(target, LayerTarget::MidSemantic);
    }

    #[test]
    fn test_intent_config_default() {
        let config = IntentConfig::default();
        assert_eq!(config.layer_target, LayerTarget::MidSemantic);
        assert!(config.normalize);
        assert_eq!(config.output_dim, 0);
    }

    #[test]
    fn test_intent_config_builder() {
        let config = IntentConfig::new(LayerTarget::DeepLogic)
            .with_normalize(false)
            .with_output_dim(2048);

        assert_eq!(config.layer_target, LayerTarget::DeepLogic);
        assert!(!config.normalize);
        assert_eq!(config.output_dim, 2048);
    }

    #[test]
    fn test_safety_policy_config_halt_and_veto() {
        let policy = SafetyPolicyConfig::halt_and_veto(0.95);
        assert!(policy.global_guardrail_enabled);
        assert_eq!(policy.halt_and_veto_threshold, 0.95);
        assert_eq!(policy.target_layer, LayerTarget::DeepLogic);
        assert_eq!(policy.action, GuardrailAction::HaltAndVeto);
    }

    #[test]
    fn test_safety_policy_config_replace_text() {
        let policy = SafetyPolicyConfig::replace_with_safe_text(0.9, "[内容已屏蔽]");
        assert_eq!(policy.halt_and_veto_threshold, 0.9);
        match policy.action {
            GuardrailAction::ReplaceWithSafeText { ref text } => {
                assert_eq!(text, "[内容已屏蔽]");
            }
            _ => panic!("Expected ReplaceWithSafeText"),
        }
    }

    #[test]
    fn test_safety_policy_config_validate() {
        // 有效阈值
        let policy = SafetyPolicyConfig::halt_and_veto(0.5);
        assert!(policy.validate().is_ok());

        // 无效阈值（超出范围）
        let policy_invalid = SafetyPolicyConfig {
            halt_and_veto_threshold: 1.5,
            ..Default::default()
        };
        assert!(policy_invalid.validate().is_err());
    }

    #[test]
    fn test_encode_intent_skeleton() {
        let result = encode_intent("test input", LayerTarget::MidSemantic);
        assert!(result.is_ok());

        let encoding = result.unwrap();
        assert_eq!(encoding.embedding.len(), 4096);
        assert_eq!(encoding.actual_layer, 16);
        assert_eq!(encoding.layer_target, LayerTarget::MidSemantic);
    }

    #[test]
    fn test_attach_guardrail_skeleton() {
        let policy = SafetyPolicyConfig::halt_and_veto(0.95);
        let result = attach_guardrail("probe.safetensors", LayerTarget::DeepLogic, policy);

        assert!(result.is_ok());

        let attachment = result.unwrap();
        assert!(!attachment.probe_id.is_empty());
        assert_eq!(attachment.mounted_layer, 24);
        assert_eq!(attachment.policy.target_layer, LayerTarget::DeepLogic);
    }

    #[test]
    fn test_attach_guardrail_invalid_threshold() {
        let policy = SafetyPolicyConfig {
            halt_and_veto_threshold: -0.1,
            ..Default::default()
        };
        let result = attach_guardrail("probe.safetensors", LayerTarget::DeepLogic, policy);
        assert!(result.is_err());
    }

    #[test]
    fn test_guardrail_action_equality() {
        assert_eq!(
            GuardrailAction::HaltAndVeto,
            GuardrailAction::HaltAndVeto
        );
        assert_ne!(
            GuardrailAction::HaltAndVeto,
            GuardrailAction::LogOnly
        );
    }

    #[test]
    fn test_layer_target_discriminant() {
        // 确保 LayerTarget 各变体判别值不同（用于 FFI 映射）
        assert_ne!(LayerTarget::ShallowSyntax as u8, LayerTarget::MidSemantic as u8);
        assert_ne!(LayerTarget::MidSemantic as u8, LayerTarget::DeepLogic as u8);
    }

    #[test]
    fn test_intent_config_factory_methods() {
        let shallow = IntentConfig::shallow();
        assert_eq!(shallow.layer_target, LayerTarget::ShallowSyntax);

        let mid = IntentConfig::mid_semantic();
        assert_eq!(mid.layer_target, LayerTarget::MidSemantic);

        let deep = IntentConfig::deep_logic();
        assert_eq!(deep.layer_target, LayerTarget::DeepLogic);
    }
}
