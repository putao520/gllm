//! Guardrail SDK — Safety veto via mid-layer probe classification.
//!
//! **SSOT**: `SPEC/GUARDRAIL.md`, `SPEC/01-REQUIREMENTS.md §14`,
//! `SPEC/04-API-DESIGN.md §3.9`.
//!
//! # 架构
//!
//! Guardrail 是**钩在推理前向中途的安全分类器**:
//!
//! 1. 用户通过 `Client::attach_guardrail(probe, anchor, policy)` 加载一份
//!    二元分类线性头权重 (safetensors) + 指定锚点层 + 安全策略.
//! 2. 注册后, 每次 forward 在 `anchor` 层的 post_node 回调中:
//!    - 从 hidden_state 抽出最后一个 token 的向量 h_last
//!    - 计算 `score = sigmoid(w · h_last + b)`
//!    - 按 `SafetyPolicy` 决策 (HaltAndVeto / LogOnly / SampleDowngrade)
//!
//! # 铁律
//!
//! - **NO_SILENT_FALLBACK**: 权重加载失败 / 形状不匹配 / dtype 错误 → 显式
//!   `Err(GuardrailError::...)`, 禁止静默 OK.
//! - **NO_ISLAND_MODULE**: `GuardrailProbeCallback` 注册到 `CallbackChain`,
//!   由 `Client::classify_binary` / `encode_intent` / `generate` 路径真实触发.
//! - **正交于 SG / HR**: Guardrail callback 使用独立 priority=40, 与 SG(90)
//!   / HR(读最终 hidden) 互不干涉.
//! - **LayerAnchor 复用**: 不重新定义层索引语义, 复用
//!   `crate::head_routing::LayerAnchor`.

use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};

use thiserror::Error;

use crate::head_routing::LayerAnchor;

// ============================================================================
// GuardProbe — 探针权重来源
// ============================================================================

/// 安全探针权重来源 (SPEC/GUARDRAIL.md §4, REQ-GR-001).
///
/// 当前支持 safetensors 文件加载. 将来可扩展 HuggingFace Hub / 内存字节等来源.
#[derive(Debug, Clone)]
pub enum GuardProbe {
    /// 从本地 safetensors 文件加载线性分类器权重 (weight + bias).
    ///
    /// 期望 tensor 名:
    /// - `weight` 或 `classifier.weight`: `[hidden_dim]` 或 `[1, hidden_dim]`
    /// - `bias` 或 `classifier.bias`: `[1]` (缺失时默认 0.0)
    FromSafetensors { path: String },
}

impl GuardProbe {
    /// 便捷构造器 — safetensors 路径.
    pub fn from_safetensors(path: impl Into<String>) -> Self {
        Self::FromSafetensors { path: path.into() }
    }

    /// 便捷构造器 — 直接用内存权重 (用于测试 / 运行时生成的探针).
    ///
    /// 返回 `GuardProbeWeights` 而非 `GuardProbe` — 绕过 safetensors 加载
    /// 路径, 直接进入 `GuardrailProbeCallback::from_weights`.
    pub fn from_inline(weight: Vec<f32>, bias: f32) -> GuardProbeWeights {
        GuardProbeWeights { weight, bias }
    }
}

/// 解码后的线性探针权重 (`score = sigmoid(w · h + b)`).
#[derive(Debug, Clone)]
pub struct GuardProbeWeights {
    /// 线性分类器权重, 长度应等于 hidden_size (或小于, 做部分点积).
    pub weight: Vec<f32>,
    /// 线性分类器偏置.
    pub bias: f32,
}

impl GuardProbeWeights {
    /// 输入维度 (权重向量长度).
    pub fn input_dim(&self) -> usize {
        self.weight.len()
    }

    /// 对给定 hidden 向量计算 sigmoid 分类分数.
    ///
    /// 当权重长度 < hidden 长度时使用部分点积 (前 N 维),
    /// 支持降维探针 (如 probe trained on first 256 dims of 768-dim hidden).
    /// 当权重长度 > hidden 长度时截断权重以匹配 hidden.
    pub fn score(&self, hidden: &[f32]) -> f32 {
        let n = self.weight.len().min(hidden.len());
        if n == 0 {
            return 0.5; // 无可用维度 → 中立分数 (非 fallback, 输入维度为 0 不应发生)
        }
        let dot: f32 = self.weight[..n]
            .iter()
            .zip(hidden[..n].iter())
            .map(|(&w, &h)| w * h)
            .sum();
        let raw = dot + self.bias;
        1.0 / (1.0 + (-raw).exp())
    }
}

// ============================================================================
// SafetyPolicy
// ============================================================================

/// Guardrail 触发后的动作策略 (SPEC/GUARDRAIL.md §5, REQ-GR-002 HaltAndVeto,
/// REQ-GR-003 LogOnly, REQ-GR-004 SampleDowngrade).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyPolicy {
    /// 分数 `> threshold` 时设置 veto 标志 + 提前退出前向
    /// (ExitEarly 返回空 logits, 上层生成循环识别 veto 后终止).
    HaltAndVeto { threshold: f32 },

    /// 仅记录分数, 不改变生成 (遥测 / 审计场景).
    LogOnly,

    /// 分数超标时向共享状态写入 `min_temperature`, 由上层采样器读取后下调
    /// 采样温度 (SPEC §5.3). 不中断前向.
    SampleDowngrade { min_temperature: f32 },
}

// ============================================================================
// GuardrailAttachment — Client 返回的句柄
// ============================================================================

/// `Client::attach_guardrail` 成功后返回的句柄.
///
/// 用户可通过 `attachment.id()` 保存句柄, 通过 `Client::detach_guardrail(id)`
/// 解绑. `last_score()` / `last_veto_reason()` 查询最新一次触发结果
/// (REQ-GR-005: 多探针并发挂载 / 独立查询).
#[derive(Debug, Clone)]
pub struct GuardrailAttachment {
    /// 内部唯一 id (由 Client 分配, 用于 `detach_guardrail`).
    pub id: u64,
    /// 注册的物理层索引 (由 `LayerAnchor::resolve` 计算得出).
    pub actual_layer: usize,
    /// 探针名字 / 标识符 (用于日志).
    pub probe_name: String,
    /// 共享的运行时状态 (score / veto flag / downgrade temperature).
    pub(crate) shared: Arc<GuardrailSharedState>,
}

impl GuardrailAttachment {
    /// 当前挂载句柄的 id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 最新一次 probe 分类分数 (0.0..=1.0), `None` 表示尚未触发过.
    pub fn last_score(&self) -> Option<f32> {
        let bits = self.shared.last_score_bits.load(Ordering::Acquire);
        if bits == u32::MAX {
            None
        } else {
            Some(f32::from_bits(bits))
        }
    }

    /// 最新一次触发的 veto 原因 (`HaltAndVeto` 且超阈值时设置).
    pub fn last_veto_reason(&self) -> Option<String> {
        self.shared
            .last_veto_reason
            .lock()
            .ok()
            .and_then(|g| g.clone())
    }

    /// 是否已被 veto (HaltAndVeto 触发).
    pub fn is_vetoed(&self) -> bool {
        self.shared.vetoed.load(Ordering::Acquire)
    }

    /// 最新一次 `SampleDowngrade` 记录的最低温度 (`None` 表示未触发).
    pub fn downgraded_temperature(&self) -> Option<f32> {
        let bits = self.shared.downgrade_temp_bits.load(Ordering::Acquire);
        if bits == u32::MAX {
            None
        } else {
            Some(f32::from_bits(bits))
        }
    }

    /// 重置共享状态 (在同一 attachment 上复用前清空上次状态).
    pub fn reset(&self) {
        self.shared.reset();
    }
}

/// `GuardrailProbeCallback` 与 `GuardrailAttachment` 共享的运行时状态.
///
/// Arc 共享, 回调侧写入 (score / veto flag), API 侧读取.
#[derive(Debug)]
pub(crate) struct GuardrailSharedState {
    /// 最近一次 probe 分数 (f32.to_bits, MAX = 未设置).
    pub(crate) last_score_bits: AtomicU32,
    /// Veto 标志 (HaltAndVeto 且分数 > threshold 时设置).
    pub(crate) vetoed: AtomicBool,
    /// Veto 原因 (触发后设置; 被 `detach_guardrail` 或 `reset` 清除).
    pub(crate) last_veto_reason: Mutex<Option<String>>,
    /// SampleDowngrade 记录的最低温度.
    pub(crate) downgrade_temp_bits: AtomicU32,
}

impl GuardrailSharedState {
    pub(crate) fn new() -> Self {
        Self {
            last_score_bits: AtomicU32::new(u32::MAX),
            vetoed: AtomicBool::new(false),
            last_veto_reason: Mutex::new(None),
            downgrade_temp_bits: AtomicU32::new(u32::MAX),
        }
    }

    pub(crate) fn record_score(&self, score: f32) {
        self.last_score_bits.store(score.to_bits(), Ordering::Release);
    }

    pub(crate) fn trigger_veto(&self, reason: String) {
        self.vetoed.store(true, Ordering::Release);
        if let Ok(mut guard) = self.last_veto_reason.lock() {
            *guard = Some(reason);
        }
    }

    pub(crate) fn record_downgrade(&self, min_temperature: f32) {
        self.downgrade_temp_bits
            .store(min_temperature.to_bits(), Ordering::Release);
    }

    pub(crate) fn reset(&self) {
        self.last_score_bits.store(u32::MAX, Ordering::Release);
        self.vetoed.store(false, Ordering::Release);
        self.downgrade_temp_bits.store(u32::MAX, Ordering::Release);
        if let Ok(mut guard) = self.last_veto_reason.lock() {
            *guard = None;
        }
    }
}

// ============================================================================
// 错误类型
// ============================================================================

/// Guardrail SDK 错误.
#[derive(Debug, Error)]
pub enum GuardrailError {
    /// Safetensors 文件不存在 / 打不开 / 损坏.
    #[error("probe load failed: {0}")]
    ProbeLoadFailed(String),

    /// 权重形状不匹配 (如 2D 但不是 `[num_classes, hidden_dim]` 形式).
    #[error("probe weight shape invalid: {0}")]
    InvalidShape(String),

    /// SafetyPolicy 参数非法 (如 threshold 不在 [0,1] / min_temperature <= 0).
    #[error("invalid safety policy: {0}")]
    InvalidPolicy(String),

    /// LayerAnchor 无法解析 (越界 / NaN).
    #[error("invalid layer anchor: {0}")]
    InvalidAnchor(String),

    /// 客户端未加载模型.
    #[error("no model loaded")]
    NoModelLoaded,

    /// 句柄未找到 (detach 时).
    #[error("guardrail id {0} not found")]
    NotFound(u64),

    /// 下游 IO / 解析错误.
    #[error("IO error: {0}")]
    Io(String),
}

// ============================================================================
// Safetensors 权重加载
// ============================================================================

/// 从 safetensors 文件读取线性探针权重.
///
/// 搜索路径:
/// 1. `weight` + `bias` (标准命名)
/// 2. `classifier.weight` + `classifier.bias` (HuggingFace 分类头约定)
///
/// 形状校验: weight 必须是 `[hidden_dim]` / `[1, hidden_dim]` /
/// `[num_classes, hidden_dim]` (取第一行). bias 必须是 `[1]` 或 `[num_classes]` (取第一个).
pub fn load_probe_weights(probe: &GuardProbe) -> Result<GuardProbeWeights, GuardrailError> {
    match probe {
        GuardProbe::FromSafetensors { path } => load_from_safetensors(Path::new(path)),
    }
}

fn load_from_safetensors(path: &Path) -> Result<GuardProbeWeights, GuardrailError> {
    if !path.exists() {
        return Err(GuardrailError::ProbeLoadFailed(format!(
            "safetensors probe file not found: {}",
            path.display()
        )));
    }

    let loader = crate::loader::safetensors::MappedSafetensors::open(path).map_err(|e| {
        GuardrailError::ProbeLoadFailed(format!(
            "failed to open safetensors '{}': {}",
            path.display(),
            e
        ))
    })?;

    // 查找 weight tensor (多种常见命名).
    const WEIGHT_ALIASES: &[&str] = &["weight", "classifier.weight", "guard_probe.weight"];
    let mut weight_view = None;
    let mut found_alias = "";
    for alias in WEIGHT_ALIASES {
        if let Ok(t) = loader.tensor(alias) {
            weight_view = Some(t);
            found_alias = alias;
            break;
        }
    }
    let weight_tensor = weight_view.ok_or_else(|| {
        GuardrailError::ProbeLoadFailed(format!(
            "no weight tensor found in '{}' — tried: {:?}",
            path.display(),
            WEIGHT_ALIASES
        ))
    })?;

    let weight_data = weight_tensor.as_f32().map_err(|e| {
        GuardrailError::ProbeLoadFailed(format!(
            "weight tensor '{}' is not f32 in '{}': {}",
            found_alias,
            path.display(),
            e
        ))
    })?;

    // 解析 hidden_dim.
    let hidden_dim = match weight_tensor.shape.as_slice() {
        [d] => *d,
        [1, d] => *d,
        [n, d] if *n >= 1 => {
            // [num_classes, hidden_dim] — 取第一行 (正类权重)
            *d
        }
        shape => {
            return Err(GuardrailError::InvalidShape(format!(
                "weight tensor shape {:?} not supported (expected [H] / [1,H] / [N,H])",
                shape
            )));
        }
    };

    if hidden_dim == 0 {
        return Err(GuardrailError::InvalidShape(
            "weight tensor hidden_dim is 0".to_string(),
        ));
    }

    // 若 2D [N, H] 且 N > 1, 取第一行.
    let weight_vec = if weight_data.len() > hidden_dim {
        weight_data[..hidden_dim].to_vec()
    } else {
        weight_data.into_owned()
    };

    if weight_vec.len() != hidden_dim {
        return Err(GuardrailError::InvalidShape(format!(
            "weight vector length {} != declared hidden_dim {}",
            weight_vec.len(),
            hidden_dim
        )));
    }

    // 查找 bias tensor (可选).
    const BIAS_ALIASES: &[&str] = &["bias", "classifier.bias", "guard_probe.bias"];
    let mut bias_val: f32 = 0.0;
    for alias in BIAS_ALIASES {
        if let Ok(t) = loader.tensor(alias) {
            if let Ok(b) = t.as_f32() {
                if let Some(&v) = b.first() {
                    bias_val = v;
                    break;
                }
            }
        }
    }

    Ok(GuardProbeWeights {
        weight: weight_vec,
        bias: bias_val,
    })
}

// ============================================================================
// Policy 校验
// ============================================================================

/// 校验 SafetyPolicy 参数合法性.
pub fn validate_policy(policy: &SafetyPolicy) -> Result<(), GuardrailError> {
    match policy {
        SafetyPolicy::HaltAndVeto { threshold } => {
            if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
                return Err(GuardrailError::InvalidPolicy(format!(
                    "HaltAndVeto threshold must be finite and in [0,1], got {threshold}"
                )));
            }
        }
        SafetyPolicy::LogOnly => {}
        SafetyPolicy::SampleDowngrade { min_temperature } => {
            if !min_temperature.is_finite() || *min_temperature <= 0.0 {
                return Err(GuardrailError::InvalidPolicy(format!(
                    "SampleDowngrade min_temperature must be finite and > 0, got {min_temperature}"
                )));
            }
        }
    }
    Ok(())
}

/// 解析 LayerAnchor 到物理层索引, 包装错误.
pub fn resolve_anchor(anchor: LayerAnchor, num_layers: usize) -> Result<usize, GuardrailError> {
    anchor
        .resolve(num_layers)
        .map_err(|e| GuardrailError::InvalidAnchor(format!("{e}")))
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_weights_sigmoid_zero_weight_returns_half() {
        let p = GuardProbeWeights {
            weight: vec![0.0; 4],
            bias: 0.0,
        };
        let score = p.score(&[1.0, 2.0, 3.0, 4.0]);
        assert!((score - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {score}");
    }

    #[test]
    fn probe_weights_positive_bias_pushes_score_high() {
        let p = GuardProbeWeights {
            weight: vec![0.0; 4],
            bias: 10.0,
        };
        let score = p.score(&[1.0, 2.0, 3.0, 4.0]);
        assert!(score > 0.99, "sigmoid(10) should be ~1.0, got {score}");
    }

    #[test]
    fn probe_weights_negative_bias_pushes_score_low() {
        let p = GuardProbeWeights {
            weight: vec![0.0; 4],
            bias: -10.0,
        };
        let score = p.score(&[1.0, 2.0, 3.0, 4.0]);
        assert!(score < 0.01, "sigmoid(-10) should be ~0.0, got {score}");
    }

    #[test]
    fn probe_weights_partial_dim_dot_product() {
        // probe trained on 2-dim, hidden is 4-dim → use first 2 dims.
        let p = GuardProbeWeights {
            weight: vec![1.0, 1.0],
            bias: 0.0,
        };
        let score = p.score(&[2.0, 3.0, 999.0, -999.0]);
        // dot = 1*2 + 1*3 = 5 → sigmoid(5) ≈ 0.993
        let expected = 1.0 / (1.0 + (-5.0f32).exp());
        assert!((score - expected).abs() < 1e-5, "got {score}, expected {expected}");
    }

    #[test]
    fn validate_policy_rejects_out_of_range_threshold() {
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: -0.1 }).is_err());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 1.1 }).is_err());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: f32::NAN }).is_err());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 0.0 }).is_ok());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 1.0 }).is_ok());
    }

    #[test]
    fn validate_policy_rejects_non_positive_temperature() {
        assert!(validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: 0.0 }).is_err());
        assert!(validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: -0.5 }).is_err());
        assert!(validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: f32::NAN }).is_err());
        assert!(validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: 0.3 }).is_ok());
    }

    #[test]
    fn validate_policy_accepts_log_only() {
        assert!(validate_policy(&SafetyPolicy::LogOnly).is_ok());
    }

    #[test]
    fn resolve_anchor_maps_relative_to_physical_layer() {
        let layer = resolve_anchor(LayerAnchor::Relative(0.5), 30).unwrap();
        // round(0.5 * 29) = round(14.5) = 14 (f32 rounds half to even) OR 15.
        // 实际 f32::round 是 half away from zero, 14.5 → 15.
        assert!(layer == 14 || layer == 15);
    }

    #[test]
    fn resolve_anchor_rejects_out_of_range() {
        let err = resolve_anchor(LayerAnchor::Relative(1.5), 30);
        assert!(matches!(err, Err(GuardrailError::InvalidAnchor(_))));
    }

    #[test]
    fn shared_state_records_and_resets() {
        let s = GuardrailSharedState::new();
        assert!(s.last_score_bits.load(Ordering::Acquire) == u32::MAX);
        s.record_score(0.75);
        let bits = s.last_score_bits.load(Ordering::Acquire);
        assert!((f32::from_bits(bits) - 0.75).abs() < 1e-6);
        s.trigger_veto("toxicity".to_string());
        assert!(s.vetoed.load(Ordering::Acquire));
        s.reset();
        assert!(!s.vetoed.load(Ordering::Acquire));
        assert_eq!(s.last_score_bits.load(Ordering::Acquire), u32::MAX);
    }

    #[test]
    fn guard_probe_from_inline_builds_weights() {
        let w = GuardProbe::from_inline(vec![1.0, 2.0, 3.0], 0.5);
        assert_eq!(w.weight, vec![1.0, 2.0, 3.0]);
        assert_eq!(w.bias, 0.5);
        assert_eq!(w.input_dim(), 3);
    }

    #[test]
    fn load_probe_weights_from_nonexistent_file_errors() {
        let err = load_probe_weights(&GuardProbe::from_safetensors("/tmp/__nonexistent__.safetensors"));
        assert!(matches!(err, Err(GuardrailError::ProbeLoadFailed(_))));
    }
}
