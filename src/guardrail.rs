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
//! - **NO_ISLAND_MODULE**: Guardrail 探针通过 JIT Op (GuardrailScore +
//!   GuardrailCheck) 注册到前向路径, 由 `Client::classify_binary` /
//!   `encode_intent` / `generate` 路径真实触发.
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
    /// 路径, 直接用于 `attach_guardrail_inline`.
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

/// Guardrail shared runtime state for callback ↔ attachment communication.
///
/// Arc shared, callback side writes (score / veto flag), API side reads.
#[derive(Debug)]
#[allow(dead_code)]
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

#[allow(dead_code)]
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

    // ========================================================================
    // GuardProbe constructors
    // ========================================================================

    #[test]
    fn guard_probe_from_safetensors_stores_path() {
        let probe = GuardProbe::from_safetensors("/data/probe.safetensors");
        match &probe {
            GuardProbe::FromSafetensors { path } => {
                assert_eq!(path, "/data/probe.safetensors");
            }
        }
    }

    #[test]
    fn guard_probe_from_safetensors_with_string() {
        let path = String::from("/models/classifier.safetensors");
        let probe = GuardProbe::from_safetensors(path);
        match &probe {
            GuardProbe::FromSafetensors { path } => {
                assert_eq!(path, "/models/classifier.safetensors");
            }
        }
    }

    // ========================================================================
    // GuardProbeWeights — input_dim
    // ========================================================================

    #[test]
    fn probe_weights_input_dim_matches_weight_length() {
        let w = GuardProbeWeights {
            weight: vec![0.1; 256],
            bias: 0.0,
        };
        assert_eq!(w.input_dim(), 256);
    }

    #[test]
    fn probe_weights_input_dim_zero_when_empty() {
        let w = GuardProbeWeights {
            weight: vec![],
            bias: 1.0,
        };
        assert_eq!(w.input_dim(), 0);
    }

    // ========================================================================
    // GuardProbeWeights::score — edge cases
    // ========================================================================

    #[test]
    fn probe_weights_score_empty_weight_returns_neutral() {
        let p = GuardProbeWeights {
            weight: vec![],
            bias: 5.0,
        };
        let score = p.score(&[1.0, 2.0, 3.0]);
        assert!((score - 0.5).abs() < 1e-6, "no dims available, got {score}");
    }

    #[test]
    fn probe_weights_score_empty_hidden_returns_neutral() {
        let p = GuardProbeWeights {
            weight: vec![1.0, 2.0],
            bias: 0.0,
        };
        let score = p.score(&[]);
        assert!((score - 0.5).abs() < 1e-6, "no hidden dims, got {score}");
    }

    #[test]
    fn probe_weights_score_truncates_weight_when_longer_than_hidden() {
        // weight has 6 dims, hidden has only 3 → use first 3 dims of weight
        let p = GuardProbeWeights {
            weight: vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            bias: 0.0,
        };
        let score = p.score(&[1.0, 1.0, 1.0]);
        // dot = 1*1 + 2*1 + 3*1 = 6 → sigmoid(6) ≈ 0.9975
        let expected = 1.0f32 / (1.0 + (-6.0f32).exp());
        assert!(
            (score - expected).abs() < 1e-4,
            "got {score}, expected {expected}"
        );
    }

    #[test]
    fn probe_weights_score_exact_dot_product() {
        let p = GuardProbeWeights {
            weight: vec![0.5, -0.3, 0.8],
            bias: 0.1,
        };
        let hidden = [1.0, 2.0, 3.0];
        // dot = 0.5*1 + (-0.3)*2 + 0.8*3 = 0.5 - 0.6 + 2.4 = 2.3
        // raw = 2.3 + 0.1 = 2.4
        let expected = 1.0f32 / (1.0 + (-2.4f32).exp());
        let score = p.score(&hidden);
        assert!((score - expected).abs() < 1e-5, "got {score}, expected {expected}");
    }

    #[test]
    fn probe_weights_score_symmetry_with_negated_weights() {
        let p_pos = GuardProbeWeights {
            weight: vec![1.0, 1.0],
            bias: 0.0,
        };
        let p_neg = GuardProbeWeights {
            weight: vec![-1.0, -1.0],
            bias: 0.0,
        };
        let hidden = [1.0, 1.0];
        let s_pos = p_pos.score(&hidden);
        let s_neg = p_neg.score(&hidden);
        // sigmoid(x) + sigmoid(-x) = 1.0
        assert!(
            (s_pos + s_neg - 1.0).abs() < 1e-5,
            "symmetry broken: {s_pos} + {s_neg} = {}",
            s_pos + s_neg
        );
    }

    #[test]
    fn probe_weights_score_large_positive_input_near_one() {
        let p = GuardProbeWeights {
            weight: vec![100.0],
            bias: 0.0,
        };
        let score = p.score(&[1.0]);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "sigmoid(100) should be ~1.0, got {score}"
        );
    }

    #[test]
    fn probe_weights_score_large_negative_input_near_zero() {
        let p = GuardProbeWeights {
            weight: vec![-100.0],
            bias: 0.0,
        };
        let score = p.score(&[1.0]);
        assert!(
            score < 1e-6,
            "sigmoid(-100) should be ~0.0, got {score}"
        );
    }

    // ========================================================================
    // SafetyPolicy — PartialEq and Copy semantics
    // ========================================================================

    #[test]
    fn safety_policy_equality() {
        let a = SafetyPolicy::HaltAndVeto { threshold: 0.8 };
        let b = SafetyPolicy::HaltAndVeto { threshold: 0.8 };
        assert_eq!(a, b);

        let c = SafetyPolicy::HaltAndVeto { threshold: 0.9 };
        assert_ne!(a, c);

        assert_eq!(SafetyPolicy::LogOnly, SafetyPolicy::LogOnly);

        let d = SafetyPolicy::SampleDowngrade { min_temperature: 0.5 };
        let e = SafetyPolicy::SampleDowngrade { min_temperature: 0.5 };
        assert_eq!(d, e);
    }

    #[test]
    fn safety_policy_copy_semantics() {
        let original = SafetyPolicy::HaltAndVeto { threshold: 0.7 };
        let copied = original;
        assert_eq!(original, copied);
    }

    // ========================================================================
    // validate_policy — boundary values
    // ========================================================================

    #[test]
    fn validate_policy_halt_and_veto_accepts_boundary_values() {
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 0.0 }).is_ok());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 1.0 }).is_ok());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 0.5 }).is_ok());
    }

    #[test]
    fn validate_policy_halt_and_veto_rejects_infinity() {
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: f32::INFINITY }).is_err());
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: f32::NEG_INFINITY }).is_err());
    }

    #[test]
    fn validate_policy_sample_downgrade_rejects_infinity() {
        assert!(
            validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: f32::INFINITY })
                .is_err()
        );
        assert!(
            validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: f32::NEG_INFINITY })
                .is_err()
        );
    }

    #[test]
    fn validate_policy_sample_downgrade_accepts_small_positive() {
        assert!(
            validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: 0.001 }).is_ok()
        );
    }

    #[test]
    fn validate_policy_returns_correct_error_type_for_threshold() {
        let err = validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 2.0 });
        match err {
            Err(GuardrailError::InvalidPolicy(msg)) => {
                assert!(msg.contains("threshold"), "message should mention threshold: {msg}");
                assert!(msg.contains("2"), "message should contain the value: {msg}");
            }
            other => panic!("expected InvalidPolicy error, got {other:?}"),
        }
    }

    #[test]
    fn validate_policy_returns_correct_error_type_for_temperature() {
        let err = validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: -1.0 });
        match err {
            Err(GuardrailError::InvalidPolicy(msg)) => {
                assert!(
                    msg.contains("min_temperature"),
                    "message should mention min_temperature: {msg}"
                );
            }
            other => panic!("expected InvalidPolicy error, got {other:?}"),
        }
    }

    // ========================================================================
    // resolve_anchor — Absolute and edge cases
    // ========================================================================

    #[test]
    fn resolve_anchor_absolute_within_range() {
        assert_eq!(resolve_anchor(LayerAnchor::Absolute(0), 10).unwrap(), 0);
        assert_eq!(resolve_anchor(LayerAnchor::Absolute(5), 10).unwrap(), 5);
        assert_eq!(resolve_anchor(LayerAnchor::Absolute(9), 10).unwrap(), 9);
    }

    #[test]
    fn resolve_anchor_absolute_out_of_range() {
        let err = resolve_anchor(LayerAnchor::Absolute(10), 10);
        assert!(matches!(err, Err(GuardrailError::InvalidAnchor(_))));
    }

    #[test]
    fn resolve_anchor_absolute_zero_layers_errors() {
        let err = resolve_anchor(LayerAnchor::Absolute(0), 0);
        assert!(matches!(err, Err(GuardrailError::InvalidAnchor(_))));
    }

    #[test]
    fn resolve_anchor_relative_zero_gives_first_layer() {
        let layer = resolve_anchor(LayerAnchor::Relative(0.0), 10).unwrap();
        assert_eq!(layer, 0);
    }

    #[test]
    fn resolve_anchor_relative_one_gives_last_layer() {
        let layer = resolve_anchor(LayerAnchor::Relative(1.0), 10).unwrap();
        assert_eq!(layer, 9);
    }

    #[test]
    fn resolve_anchor_relative_negative_errors() {
        let err = resolve_anchor(LayerAnchor::Relative(-0.1), 10);
        assert!(matches!(err, Err(GuardrailError::InvalidAnchor(_))));
    }

    #[test]
    fn resolve_anchor_relative_nan_errors() {
        let err = resolve_anchor(LayerAnchor::Relative(f32::NAN), 10);
        assert!(matches!(err, Err(GuardrailError::InvalidAnchor(_))));
    }

    #[test]
    fn resolve_anchor_relative_single_layer() {
        let layer = resolve_anchor(LayerAnchor::Relative(0.5), 1).unwrap();
        assert_eq!(layer, 0);
    }

    // ========================================================================
    // GuardrailSharedState — concurrent-ish operations
    // ========================================================================

    #[test]
    fn shared_state_initial_values() {
        let s = GuardrailSharedState::new();
        assert_eq!(s.last_score_bits.load(Ordering::Acquire), u32::MAX);
        assert!(!s.vetoed.load(Ordering::Acquire));
        assert_eq!(s.downgrade_temp_bits.load(Ordering::Acquire), u32::MAX);
        assert!(s.last_veto_reason.lock().unwrap().is_none());
    }

    #[test]
    fn shared_state_record_score_preserves_value() {
        let s = GuardrailSharedState::new();
        let values = [0.0, 0.25, 0.5, 0.75, 1.0, -1.5, 42.0, f32::MIN_POSITIVE];
        for &v in &values {
            s.record_score(v);
            let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
            assert!(
                (loaded - v).abs() < 1e-6 || (v.is_normal() && loaded.is_normal()),
                "expected {v}, got {loaded}"
            );
        }
    }

    #[test]
    fn shared_state_trigger_veto_sets_reason() {
        let s = GuardrailSharedState::new();
        s.trigger_veto("harmful content detected".to_string());
        assert!(s.vetoed.load(Ordering::Acquire));
        let reason = s.last_veto_reason.lock().unwrap().clone();
        assert_eq!(reason.as_deref(), Some("harmful content detected"));
    }

    #[test]
    fn shared_state_record_downgrade_preserves_value() {
        let s = GuardrailSharedState::new();
        s.record_downgrade(0.3);
        let loaded = f32::from_bits(s.downgrade_temp_bits.load(Ordering::Acquire));
        assert!((loaded - 0.3).abs() < 1e-6, "got {loaded}");
    }

    #[test]
    fn shared_state_reset_clears_veto_reason() {
        let s = GuardrailSharedState::new();
        s.trigger_veto("test".to_string());
        assert!(s.last_veto_reason.lock().unwrap().is_some());
        s.reset();
        assert!(s.last_veto_reason.lock().unwrap().is_none());
    }

    #[test]
    fn shared_state_reset_clears_downgrade() {
        let s = GuardrailSharedState::new();
        s.record_downgrade(0.5);
        s.reset();
        assert_eq!(s.downgrade_temp_bits.load(Ordering::Acquire), u32::MAX);
    }

    #[test]
    fn shared_state_multiple_veto_overwrites_reason() {
        let s = GuardrailSharedState::new();
        s.trigger_veto("first".to_string());
        s.trigger_veto("second".to_string());
        let reason = s.last_veto_reason.lock().unwrap().clone();
        assert_eq!(reason.as_deref(), Some("second"));
    }

    // ========================================================================
    // GuardrailAttachment — API surface
    // ========================================================================

    #[test]
    fn attachment_id_returns_configured_id() {
        let att = GuardrailAttachment {
            id: 42,
            actual_layer: 7,
            probe_name: "toxicity_v1".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert_eq!(att.id(), 42);
    }

    #[test]
    fn attachment_last_score_none_before_recording() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert!(att.last_score().is_none());
    }

    #[test]
    fn attachment_last_score_some_after_recording() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(0.85);
        let score = att.last_score().unwrap();
        assert!((score - 0.85).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn attachment_is_vetoed_false_initially() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert!(!att.is_vetoed());
    }

    #[test]
    fn attachment_is_vetoed_true_after_trigger() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.trigger_veto("dangerous".to_string());
        assert!(att.is_vetoed());
    }

    #[test]
    fn attachment_last_veto_reason_none_initially() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert!(att.last_veto_reason().is_none());
    }

    #[test]
    fn attachment_last_veto_reason_some_after_trigger() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.trigger_veto("hate speech".to_string());
        assert_eq!(att.last_veto_reason().as_deref(), Some("hate speech"));
    }

    #[test]
    fn attachment_downgraded_temperature_none_initially() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert!(att.downgraded_temperature().is_none());
    }

    #[test]
    fn attachment_downgraded_temperature_some_after_record() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_downgrade(0.15);
        let temp = att.downgraded_temperature().unwrap();
        assert!((temp - 0.15).abs() < 1e-6, "got {temp}");
    }

    #[test]
    fn attachment_reset_clears_all_state() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 3,
            probe_name: "probe".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(0.9);
        att.shared.trigger_veto("bad".to_string());
        att.shared.record_downgrade(0.1);

        att.reset();

        assert!(att.last_score().is_none());
        assert!(!att.is_vetoed());
        assert!(att.last_veto_reason().is_none());
        assert!(att.downgraded_temperature().is_none());
    }

    #[test]
    fn attachment_cloned_shares_state() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        let cloned = att.clone();
        att.shared.record_score(0.42);

        // Cloned attachment sees the same shared state
        let score = cloned.last_score().unwrap();
        assert!((score - 0.42).abs() < 1e-6, "cloned got {score}");
    }

    // ========================================================================
    // GuardrailError — Display and variants
    // ========================================================================

    #[test]
    fn error_display_probe_load_failed() {
        let err = GuardrailError::ProbeLoadFailed("file not found".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("probe load failed"), "got: {msg}");
        assert!(msg.contains("file not found"), "got: {msg}");
    }

    #[test]
    fn error_display_invalid_shape() {
        let err = GuardrailError::InvalidShape("expected [H]".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("probe weight shape invalid"), "got: {msg}");
        assert!(msg.contains("expected [H]"), "got: {msg}");
    }

    #[test]
    fn error_display_invalid_policy() {
        let err = GuardrailError::InvalidPolicy("bad threshold".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("invalid safety policy"), "got: {msg}");
        assert!(msg.contains("bad threshold"), "got: {msg}");
    }

    #[test]
    fn error_display_invalid_anchor() {
        let err = GuardrailError::InvalidAnchor("layer out of range".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("invalid layer anchor"), "got: {msg}");
        assert!(msg.contains("layer out of range"), "got: {msg}");
    }

    #[test]
    fn error_display_no_model_loaded() {
        let err = GuardrailError::NoModelLoaded;
        let msg = format!("{err}");
        assert!(msg.contains("no model loaded"), "got: {msg}");
    }

    #[test]
    fn error_display_not_found() {
        let err = GuardrailError::NotFound(123);
        let msg = format!("{err}");
        assert!(msg.contains("123"), "got: {msg}");
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[test]
    fn error_display_io() {
        let err = GuardrailError::Io("read error".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("IO error"), "got: {msg}");
        assert!(msg.contains("read error"), "got: {msg}");
    }

    #[test]
    fn error_debug_includes_variant_name() {
        let err = GuardrailError::ProbeLoadFailed("x".to_string());
        let dbg = format!("{err:?}");
        assert!(dbg.contains("ProbeLoadFailed"), "got: {dbg}");
    }

    // ========================================================================
    // GuardProbe Debug/Clone
    // ========================================================================

    #[test]
    fn guard_probe_debug_format() {
        let probe = GuardProbe::FromSafetensors {
            path: "/tmp/probe.safetensors".to_string(),
        };
        let dbg = format!("{probe:?}");
        assert!(dbg.contains("FromSafetensors"), "got: {dbg}");
        assert!(dbg.contains("/tmp/probe.safetensors"), "got: {dbg}");
    }

    #[test]
    fn guard_probe_clone_preserves_path() {
        let probe = GuardProbe::from_safetensors("/data/model.safetensors");
        let cloned = probe.clone();
        match (&probe, &cloned) {
            (
                GuardProbe::FromSafetensors { path: p1 },
                GuardProbe::FromSafetensors { path: p2 },
            ) => assert_eq!(p1, p2),
        }
    }

    // ========================================================================
    // GuardProbeWeights Debug/Clone
    // ========================================================================

    #[test]
    fn probe_weights_clone_is_independent() {
        let original = GuardProbeWeights {
            weight: vec![1.0, 2.0],
            bias: 0.5,
        };
        let mut cloned = original.clone();
        cloned.weight[0] = 99.0;
        // Original unaffected
        assert_eq!(original.weight[0], 1.0);
    }

    #[test]
    fn probe_weights_debug_includes_fields() {
        let w = GuardProbeWeights {
            weight: vec![1.0],
            bias: 2.0,
        };
        let dbg = format!("{w:?}");
        assert!(dbg.contains("GuardProbeWeights"), "got: {dbg}");
    }

    // ========================================================================
    // SafetyPolicy Debug
    // ========================================================================

    #[test]
    fn safety_policy_debug_halt_and_veto() {
        let p = SafetyPolicy::HaltAndVeto { threshold: 0.8 };
        let dbg = format!("{p:?}");
        assert!(dbg.contains("HaltAndVeto"), "got: {dbg}");
        assert!(dbg.contains("0.8"), "got: {dbg}");
    }

    #[test]
    fn safety_policy_debug_log_only() {
        let p = SafetyPolicy::LogOnly;
        let dbg = format!("{p:?}");
        assert!(dbg.contains("LogOnly"), "got: {dbg}");
    }

    #[test]
    fn safety_policy_debug_sample_downgrade() {
        let p = SafetyPolicy::SampleDowngrade { min_temperature: 0.3 };
        let dbg = format!("{p:?}");
        assert!(dbg.contains("SampleDowngrade"), "got: {dbg}");
        assert!(dbg.contains("0.3"), "got: {dbg}");
    }

    // ========================================================================
    // load_probe_weights — error messages contain path
    // ========================================================================

    #[test]
    fn load_probe_weights_error_contains_path() {
        let path = "/tmp/__nonexistent_abc123__.safetensors";
        let err = load_probe_weights(&GuardProbe::from_safetensors(path));
        match err {
            Err(GuardrailError::ProbeLoadFailed(msg)) => {
                assert!(msg.contains(path), "error should contain path, got: {msg}");
            }
            other => panic!("expected ProbeLoadFailed, got {other:?}"),
        }
    }

    // ========================================================================
    // GuardProbeWeights::score — mathematical properties
    // ========================================================================

    #[test]
    fn probe_weights_score_monotonic_in_hidden() {
        // Increasing hidden magnitude with positive weights should increase score.
        let p = GuardProbeWeights {
            weight: vec![1.0, 1.0],
            bias: 0.0,
        };
        let s1 = p.score(&[0.5, 0.5]);
        let s2 = p.score(&[1.0, 1.0]);
        let s3 = p.score(&[2.0, 2.0]);
        assert!(s1 < s2, "expected {s1} < {s2}");
        assert!(s2 < s3, "expected {s2} < {s3}");
    }

    #[test]
    fn probe_weights_score_monotonic_in_bias() {
        // Increasing bias should increase score for non-negative dot products.
        let hidden = [1.0, 1.0];
        let s1 = GuardProbeWeights { weight: vec![1.0, 1.0], bias: -5.0 }.score(&hidden);
        let s2 = GuardProbeWeights { weight: vec![1.0, 1.0], bias: 0.0 }.score(&hidden);
        let s3 = GuardProbeWeights { weight: vec![1.0, 1.0], bias: 5.0 }.score(&hidden);
        assert!(s1 < s2, "expected {s1} < {s2}");
        assert!(s2 < s3, "expected {s2} < {s3}");
    }

    #[test]
    fn probe_weights_score_range_is_unit_interval() {
        let p = GuardProbeWeights {
            weight: vec![1.0, -1.0, 0.5],
            bias: 2.0,
        };
        let test_cases: &[&[f32]] = &[
            &[0.0, 0.0, 0.0],
            &[100.0, -100.0, 50.0],
            &[-1000.0, 1000.0, -500.0],
            &[0.001, -0.001, 0.0005],
        ];
        for hidden in test_cases {
            let score = p.score(hidden);
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} out of [0,1] for hidden {:?}",
                hidden
            );
        }
    }

    #[test]
    fn probe_weights_score_orthogonal_inputs_zero_bias() {
        // Weight and hidden are orthogonal → dot = 0 → sigmoid(0) = 0.5.
        let p = GuardProbeWeights {
            weight: vec![1.0, 0.0],
            bias: 0.0,
        };
        let score = p.score(&[0.0, 1.0]);
        assert!((score - 0.5).abs() < 1e-6, "orthogonal input should give 0.5, got {score}");
    }

    #[test]
    fn probe_weights_score_negative_weight_inverts() {
        // Same hidden magnitude, negated weight → negated raw → complementary score.
        let p_pos = GuardProbeWeights { weight: vec![2.0, 3.0], bias: 0.0 };
        let p_neg = GuardProbeWeights { weight: vec![-2.0, -3.0], bias: 0.0 };
        let hidden = [1.0, 1.0];
        let s_pos = p_pos.score(&hidden);
        let s_neg = p_neg.score(&hidden);
        assert!((s_pos + s_neg - 1.0).abs() < 1e-5, "s_pos={s_pos}, s_neg={s_neg}");
    }

    #[test]
    fn probe_weights_score_scale_invariant_for_uniform_hidden() {
        // Uniform hidden with uniform weights: score depends on scale*sum(weights).
        let p = GuardProbeWeights {
            weight: vec![1.0, 1.0, 1.0],
            bias: 0.0,
        };
        let s1 = p.score(&[1.0, 1.0, 1.0]);
        // Same as 3.0 dot with single-element weight [1.0]
        let p2 = GuardProbeWeights { weight: vec![3.0], bias: 0.0 };
        let s2 = p2.score(&[1.0]);
        assert!((s1 - s2).abs() < 1e-5, "expected {s1} ≈ {s2}");
    }

    #[test]
    fn probe_weights_score_single_dim_positive() {
        let p = GuardProbeWeights { weight: vec![5.0], bias: 0.0 };
        let score = p.score(&[1.0]);
        let expected = 1.0f32 / (1.0 + (-5.0f32).exp());
        assert!((score - expected).abs() < 1e-5, "got {score}, expected {expected}");
    }

    #[test]
    fn probe_weights_score_single_dim_negative() {
        let p = GuardProbeWeights { weight: vec![-5.0], bias: 0.0 };
        let score = p.score(&[1.0]);
        let expected = 1.0f32 / (1.0 + 5.0f32.exp());
        assert!((score - expected).abs() < 1e-5, "got {score}, expected {expected}");
    }

    #[test]
    fn probe_weights_score_bias_only_dominates_with_zero_weights() {
        let p = GuardProbeWeights { weight: vec![0.0, 0.0, 0.0], bias: 3.0 };
        let score = p.score(&[100.0, 200.0, 300.0]);
        let expected = 1.0f32 / (1.0 + (-3.0f32).exp());
        assert!((score - expected).abs() < 1e-5, "bias-only should be sigmoid(3), got {score}");
    }

    #[test]
    fn probe_weights_score_high_dimensional_consistency() {
        // Verify dot product computation with many dimensions.
        let n = 1024;
        let weight: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let hidden: Vec<f32> = (0..n).map(|_| 1.0f32).collect();
        let p = GuardProbeWeights { weight: weight.clone(), bias: 0.0 };
        let score = p.score(&hidden);
        let expected_dot: f32 = weight.iter().sum();
        let expected = 1.0f32 / (1.0 + (-expected_dot).exp());
        assert!((score - expected).abs() < 1e-3, "got {score}, expected {expected}");
    }

    #[test]
    fn probe_weights_input_dim_matches_score_truncation() {
        // When weight is shorter than hidden, input_dim reports weight length.
        let p = GuardProbeWeights { weight: vec![1.0], bias: 0.0 };
        assert_eq!(p.input_dim(), 1);
        let score = p.score(&[10.0, 999.0]);
        let expected = 1.0f32 / (1.0 + (-10.0f32).exp());
        assert!((score - expected).abs() < 1e-4, "got {score}, expected {expected}");
    }

    // ========================================================================
    // SafetyPolicy — variant distinction
    // ========================================================================

    #[test]
    fn safety_policy_different_variants_not_equal() {
        let halt = SafetyPolicy::HaltAndVeto { threshold: 0.5 };
        let log = SafetyPolicy::LogOnly;
        let downgrade = SafetyPolicy::SampleDowngrade { min_temperature: 0.5 };
        assert_ne!(halt, log);
        assert_ne!(halt, downgrade);
        assert_ne!(log, downgrade);
    }

    #[test]
    fn safety_policy_copy_independent() {
        let a = SafetyPolicy::SampleDowngrade { min_temperature: 0.7 };
        let b = a;
        // Both usable independently (Copy trait).
        assert_eq!(a, b);
    }

    #[test]
    fn safety_policy_clone_preserves_all_variants() {
        let halt = SafetyPolicy::HaltAndVeto { threshold: 0.9 };
        assert_eq!(halt.clone(), halt);
        assert_eq!(SafetyPolicy::LogOnly.clone(), SafetyPolicy::LogOnly);
        let sd = SafetyPolicy::SampleDowngrade { min_temperature: 0.3 };
        assert_eq!(sd.clone(), sd);
    }

    // ========================================================================
    // validate_policy — additional edge cases
    // ========================================================================

    #[test]
    fn validate_policy_halt_and_veto_rejects_nan() {
        let result = validate_policy(&SafetyPolicy::HaltAndVeto { threshold: f32::NAN });
        assert!(result.is_err());
    }

    #[test]
    fn validate_policy_sample_downgrade_rejects_nan() {
        let result = validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: f32::NAN });
        assert!(result.is_err());
    }

    #[test]
    fn validate_policy_halt_and_veto_accepts_very_small_positive() {
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 1e-30 }).is_ok());
    }

    #[test]
    fn validate_policy_halt_and_veto_rejects_negative() {
        assert!(validate_policy(&SafetyPolicy::HaltAndVeto { threshold: -0.001 }).is_err());
    }

    #[test]
    fn validate_policy_sample_downgrade_rejects_zero() {
        assert!(
            validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: 0.0 }).is_err()
        );
    }

    #[test]
    fn validate_policy_sample_downgrade_accepts_large_finite() {
        assert!(
            validate_policy(&SafetyPolicy::SampleDowngrade { min_temperature: 1000.0 }).is_ok()
        );
    }

    #[test]
    fn validate_policy_log_only_always_ok() {
        // Call multiple times to confirm it never fails.
        assert!(validate_policy(&SafetyPolicy::LogOnly).is_ok());
        assert!(validate_policy(&SafetyPolicy::LogOnly).is_ok());
    }

    // ========================================================================
    // resolve_anchor — additional edge cases
    // ========================================================================

    #[test]
    fn resolve_anchor_absolute_boundary_max_index() {
        // Maximum valid absolute index for 10 layers is 9.
        assert_eq!(resolve_anchor(LayerAnchor::Absolute(9), 10).unwrap(), 9);
    }

    #[test]
    fn resolve_anchor_relative_quarter_depth() {
        let layer = resolve_anchor(LayerAnchor::Relative(0.25), 20).unwrap();
        // round(0.25 * 19) = round(4.75) = 5
        assert_eq!(layer, 5);
    }

    #[test]
    fn resolve_anchor_relative_three_quarters_depth() {
        let layer = resolve_anchor(LayerAnchor::Relative(0.75), 20).unwrap();
        // round(0.75 * 19) = round(14.25) = 14
        assert_eq!(layer, 14);
    }

    #[test]
    fn resolve_anchor_relative_infinity_errors() {
        assert!(matches!(
            resolve_anchor(LayerAnchor::Relative(f32::INFINITY), 10),
            Err(GuardrailError::InvalidAnchor(_))
        ));
        assert!(matches!(
            resolve_anchor(LayerAnchor::Relative(f32::NEG_INFINITY), 10),
            Err(GuardrailError::InvalidAnchor(_))
        ));
    }

    #[test]
    fn resolve_anchor_absolute_max_boundary() {
        // Exactly at num_layers should error.
        assert!(resolve_anchor(LayerAnchor::Absolute(10), 10).is_err());
        // One below should work.
        assert!(resolve_anchor(LayerAnchor::Absolute(9), 10).is_ok());
    }

    #[test]
    fn resolve_anchor_two_layers() {
        assert_eq!(resolve_anchor(LayerAnchor::Relative(0.0), 2).unwrap(), 0);
        assert_eq!(resolve_anchor(LayerAnchor::Relative(1.0), 2).unwrap(), 1);
        assert_eq!(resolve_anchor(LayerAnchor::Relative(0.5), 2).unwrap(), 1);
    }

    // ========================================================================
    // GuardProbeWeights — from_inline convenience
    // ========================================================================

    #[test]
    fn guard_probe_from_inline_empty_weight() {
        let w = GuardProbe::from_inline(vec![], 0.0);
        assert_eq!(w.input_dim(), 0);
        assert_eq!(w.bias, 0.0);
    }

    #[test]
    fn guard_probe_from_inline_large_weight() {
        let weight = vec![1.0; 4096];
        let w = GuardProbe::from_inline(weight.clone(), 1.5);
        assert_eq!(w.input_dim(), 4096);
        assert_eq!(w.bias, 1.5);
        assert_eq!(w.weight.len(), 4096);
    }

    // ========================================================================
    // GuardrailSharedState — recording special f32 values
    // ========================================================================

    #[test]
    fn shared_state_record_score_zero() {
        let s = GuardrailSharedState::new();
        s.record_score(0.0);
        let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!(loaded == 0.0, "got {loaded}");
    }

    #[test]
    fn shared_state_record_score_negative() {
        let s = GuardrailSharedState::new();
        s.record_score(-2.5);
        let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!((loaded - (-2.5)).abs() < 1e-6, "got {loaded}");
    }

    #[test]
    fn shared_state_record_score_overwrites_previous() {
        let s = GuardrailSharedState::new();
        s.record_score(0.1);
        s.record_score(0.9);
        let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!((loaded - 0.9).abs() < 1e-6, "got {loaded}");
    }

    #[test]
    fn shared_state_record_downgrade_overwrites_previous() {
        let s = GuardrailSharedState::new();
        s.record_downgrade(0.5);
        s.record_downgrade(0.1);
        let loaded = f32::from_bits(s.downgrade_temp_bits.load(Ordering::Acquire));
        assert!((loaded - 0.1).abs() < 1e-6, "got {loaded}");
    }

    #[test]
    fn shared_state_veto_and_downgrade_coexist() {
        let s = GuardrailSharedState::new();
        s.record_score(0.95);
        s.trigger_veto("test reason".to_string());
        s.record_downgrade(0.05);

        assert!(s.vetoed.load(Ordering::Acquire));
        let score = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!((score - 0.95).abs() < 1e-6);
        let temp = f32::from_bits(s.downgrade_temp_bits.load(Ordering::Acquire));
        assert!((temp - 0.05).abs() < 1e-6);
        assert_eq!(s.last_veto_reason.lock().unwrap().as_deref(), Some("test reason"));
    }

    #[test]
    fn shared_state_reset_after_multiple_operations() {
        let s = GuardrailSharedState::new();
        s.record_score(0.99);
        s.trigger_veto("first".to_string());
        s.record_downgrade(0.01);
        s.trigger_veto("second".to_string());
        s.reset();

        assert_eq!(s.last_score_bits.load(Ordering::Acquire), u32::MAX);
        assert!(!s.vetoed.load(Ordering::Acquire));
        assert_eq!(s.downgrade_temp_bits.load(Ordering::Acquire), u32::MAX);
        assert!(s.last_veto_reason.lock().unwrap().is_none());
    }

    // ========================================================================
    // GuardrailAttachment — multiple attachments, independent state
    // ========================================================================

    #[test]
    fn attachment_two_attachments_independent_state() {
        let att_a = GuardrailAttachment {
            id: 1,
            actual_layer: 3,
            probe_name: "a".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        let att_b = GuardrailAttachment {
            id: 2,
            actual_layer: 7,
            probe_name: "b".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };

        att_a.shared.record_score(0.9);
        att_a.shared.trigger_veto("a veto".to_string());
        att_b.shared.record_score(0.2);

        assert!(att_a.is_vetoed());
        assert!(!att_b.is_vetoed());
        assert!((att_a.last_score().unwrap() - 0.9).abs() < 1e-6);
        assert!((att_b.last_score().unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn attachment_actual_layer_stored() {
        let att = GuardrailAttachment {
            id: 5,
            actual_layer: 12,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert_eq!(att.actual_layer, 12);
    }

    #[test]
    fn attachment_probe_name_stored() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "toxicity_classifier_v2".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert_eq!(att.probe_name, "toxicity_classifier_v2");
    }

    #[test]
    fn attachment_record_then_reset_then_record_again() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };

        att.shared.record_score(0.8);
        att.shared.trigger_veto("first run".to_string());
        att.reset();

        assert!(att.last_score().is_none());
        assert!(!att.is_vetoed());

        att.shared.record_score(0.3);
        assert!((att.last_score().unwrap() - 0.3).abs() < 1e-6);
        assert!(!att.is_vetoed());
    }

    #[test]
    fn attachment_clone_then_original_reset_affects_both() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        let cloned = att.clone();
        att.shared.record_score(0.6);

        att.reset();

        assert!(att.last_score().is_none());
        assert!(cloned.last_score().is_none());
    }

    #[test]
    fn attachment_id_zero() {
        let att = GuardrailAttachment {
            id: 0,
            actual_layer: 0,
            probe_name: "first".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert_eq!(att.id(), 0);
    }

    #[test]
    fn attachment_id_max_u64() {
        let att = GuardrailAttachment {
            id: u64::MAX,
            actual_layer: 99,
            probe_name: "max".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        assert_eq!(att.id(), u64::MAX);
    }

    // ========================================================================
    // GuardrailError — variant discrimination
    // ========================================================================

    #[test]
    fn error_variants_are_distinct() {
        let errors = vec![
            GuardrailError::ProbeLoadFailed("a".to_string()),
            GuardrailError::InvalidShape("b".to_string()),
            GuardrailError::InvalidPolicy("c".to_string()),
            GuardrailError::InvalidAnchor("d".to_string()),
            GuardrailError::NoModelLoaded,
            GuardrailError::NotFound(42),
            GuardrailError::Io("e".to_string()),
        ];
        // Each error should produce a unique display string prefix.
        let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
        for i in 0..displays.len() {
            for j in (i + 1)..displays.len() {
                assert_ne!(displays[i], displays[j], "duplicate display: {} vs {}", displays[i], displays[j]);
            }
        }
    }

    #[test]
    fn error_not_found_includes_id() {
        let err = GuardrailError::NotFound(u64::MAX);
        let msg = format!("{err}");
        assert!(msg.contains(&u64::MAX.to_string()), "got: {msg}");
    }

    #[test]
    fn error_probe_load_failed_empty_string() {
        let err = GuardrailError::ProbeLoadFailed(String::new());
        let msg = format!("{err}");
        assert!(msg.contains("probe load failed"), "got: {msg}");
    }

    // ========================================================================
    // GuardProbe — path with special characters
    // ========================================================================

    #[test]
    fn guard_probe_from_safetensors_path_with_spaces() {
        let probe = GuardProbe::from_safetensors("/path/with spaces/probe.safetensors");
        match &probe {
            GuardProbe::FromSafetensors { path } => {
                assert_eq!(path, "/path/with spaces/probe.safetensors");
            }
        }
    }

    #[test]
    fn guard_probe_from_safetensors_empty_path() {
        let probe = GuardProbe::from_safetensors("");
        match &probe {
            GuardProbe::FromSafetensors { path } => {
                assert_eq!(path, "");
            }
        }
    }

    // ========================================================================
    // GuardProbeWeights — score combined with attachment workflow
    // ========================================================================

    #[test]
    fn score_then_record_via_attachment() {
        let probe = GuardProbe::from_inline(vec![1.0, -1.0], 0.0);
        let hidden = [1.0, 0.0];
        let score = probe.score(&hidden);

        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 5,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(score);

        let retrieved = att.last_score().unwrap();
        assert!((retrieved - score).abs() < 1e-6, "expected {score}, got {retrieved}");
        // score for [1.0, 0.0] with weight [1.0, -1.0] = sigmoid(1.0) > 0.5
        assert!(retrieved > 0.5, "score should be > 0.5, got {retrieved}");
    }

    #[test]
    fn score_triggers_veto_workflow() {
        let probe = GuardProbe::from_inline(vec![10.0], 0.0);
        let hidden = [1.0];
        let score = probe.score(&hidden);

        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 3,
            probe_name: "safety".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(score);

        let threshold = 0.8;
        if score > threshold {
            att.shared.trigger_veto("score exceeded threshold".to_string());
        }

        assert!(att.is_vetoed());
        assert!(att.last_score().unwrap() > 0.99);
        assert_eq!(
            att.last_veto_reason().as_deref(),
            Some("score exceeded threshold")
        );
    }

    #[test]
    fn score_triggers_downgrade_workflow() {
        let probe = GuardProbe::from_inline(vec![5.0], 0.0);
        let hidden = [1.0];
        let score = probe.score(&hidden);

        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 2,
            probe_name: "downgrade_test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(score);

        let threshold = 0.7;
        if score > threshold {
            att.shared.record_downgrade(0.1);
        }

        assert!(att.last_score().unwrap() > threshold);
        let temp = att.downgraded_temperature().unwrap();
        assert!((temp - 0.1).abs() < 1e-6, "got {temp}");
    }

    // ========================================================================
    // SafetyPolicy — threshold values are preserved
    // ========================================================================

    #[test]
    fn safety_policy_halt_and_veto_preserves_threshold() {
        let p = SafetyPolicy::HaltAndVeto { threshold: 0.123456 };
        if let SafetyPolicy::HaltAndVeto { threshold } = p {
            assert!((threshold - 0.123456).abs() < 1e-6);
        }
    }

    #[test]
    fn safety_policy_sample_downgrade_preserves_temperature() {
        let p = SafetyPolicy::SampleDowngrade { min_temperature: 0.77 };
        if let SafetyPolicy::SampleDowngrade { min_temperature } = p {
            assert!((min_temperature - 0.77).abs() < 1e-6);
        }
    }

    // ========================================================================
    // GuardrailSharedState — rapid reset cycles
    // ========================================================================

    #[test]
    fn shared_state_rapid_record_reset_cycles() {
        let s = GuardrailSharedState::new();
        for i in 0..100 {
            let score = i as f32 / 100.0;
            s.record_score(score);
            s.reset();
            assert_eq!(s.last_score_bits.load(Ordering::Acquire), u32::MAX);
            assert!(!s.vetoed.load(Ordering::Acquire));
        }
    }

    #[test]
    fn shared_state_record_many_scores_retains_last() {
        let s = GuardrailSharedState::new();
        let final_score = 0.42;
        for i in 0..50 {
            s.record_score(i as f32 * 0.01);
        }
        s.record_score(final_score);
        let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!((loaded - final_score).abs() < 1e-6, "got {loaded}");
    }

    // ========================================================================
    // Additional edge-case tests
    // ========================================================================

    #[test]
    fn probe_weights_score_large_values_do_not_panic() {
        // Extremely large weight and hidden values must not panic;
        // sigmoid saturates rather than overflowing.
        let p = GuardProbeWeights {
            weight: vec![f32::MAX, f32::MIN],
            bias: f32::MAX,
        };
        let score = p.score(&[f32::MAX, f32::MIN]);
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1], got {score}"
        );
    }

    #[test]
    fn probe_weights_score_mixed_sign_weights_and_hidden() {
        // Mixed positive/negative weights and hidden values.
        // w=[1,-2,3], h=[-1,2,-3] => dot = -1-4-9 = -14
        // bias=1 => raw=-13 => sigmoid(-13) ≈ ~0
        let p = GuardProbeWeights {
            weight: vec![1.0, -2.0, 3.0],
            bias: 1.0,
        };
        let score = p.score(&[-1.0, 2.0, -3.0]);
        assert!(score < 1e-5, "expected near-zero score, got {score}");
    }

    #[test]
    fn probe_weights_score_deterministic() {
        let p = GuardProbeWeights {
            weight: vec![0.7, -0.3, 0.5],
            bias: 0.1,
        };
        let hidden = [1.2, 3.4, -0.8];
        let s1 = p.score(&hidden);
        let s2 = p.score(&hidden);
        assert_eq!(s1.to_bits(), s2.to_bits(), "identical calls must yield identical results");
    }

    #[test]
    fn attachment_downgraded_temperature_keeps_last_value() {
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_downgrade(0.8);
        att.shared.record_downgrade(0.2);
        att.shared.record_downgrade(0.5);
        let temp = att.downgraded_temperature().unwrap();
        assert!(
            (temp - 0.5).abs() < 1e-6,
            "expected last recorded value 0.5, got {temp}"
        );
    }

    #[test]
    fn attachment_multiple_clones_share_state() {
        let att = GuardrailAttachment {
            id: 7,
            actual_layer: 4,
            probe_name: "multi_clone".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        let c1 = att.clone();
        let c2 = att.clone();
        let c3 = c1.clone();

        att.shared.record_score(0.33);
        att.shared.trigger_veto("cloned veto".to_string());

        // All clones must see the same shared state.
        for (label, clone) in [("c1", &c1), ("c2", &c2), ("c3", &c3)] {
            let score = clone.last_score().unwrap_or_else(|| panic!("{label} score is None"));
            assert!((score - 0.33).abs() < 1e-6, "{label} got {score}");
            assert!(clone.is_vetoed(), "{label} should be vetoed");
        }
    }

    #[test]
    fn shared_state_record_score_max_f32() {
        let s = GuardrailSharedState::new();
        s.record_score(f32::MAX);
        let loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!(loaded == f32::MAX, "got {loaded}");
    }

    #[test]
    fn shared_state_record_downgrade_min_positive() {
        let s = GuardrailSharedState::new();
        s.record_downgrade(f32::MIN_POSITIVE);
        let loaded = f32::from_bits(s.downgrade_temp_bits.load(Ordering::Acquire));
        assert!(loaded == f32::MIN_POSITIVE, "got {loaded}");
    }

    #[test]
    fn error_probe_load_vs_io_have_distinct_prefixes() {
        let e1 = GuardrailError::ProbeLoadFailed("details".to_string());
        let e2 = GuardrailError::Io("details".to_string());
        let m1 = format!("{e1}");
        let m2 = format!("{e2}");
        // Both contain "details" but must differ in prefix.
        assert_ne!(m1, m2, "error messages must be distinct");
        assert!(m1.contains("probe load failed"), "got: {m1}");
        assert!(m2.contains("IO error"), "got: {m2}");
    }

    #[test]
    fn validate_policy_halt_and_veto_accepts_exact_midpoint() {
        assert!(
            validate_policy(&SafetyPolicy::HaltAndVeto { threshold: 0.5 }).is_ok(),
            "threshold 0.5 must be accepted"
        );
    }

    #[test]
    fn resolve_anchor_relative_boundary_many_layers() {
        // With 100 layers, Relative(0.0) -> layer 0, Relative(1.0) -> layer 99.
        assert_eq!(resolve_anchor(LayerAnchor::Relative(0.0), 100).unwrap(), 0);
        assert_eq!(resolve_anchor(LayerAnchor::Relative(1.0), 100).unwrap(), 99);
    }

    #[test]
    fn probe_weights_score_all_zero_hidden_with_bias() {
        // hidden all zeros => dot=0, score = sigmoid(bias).
        let p = GuardProbeWeights {
            weight: vec![3.0, -2.0, 1.5],
            bias: 4.0,
        };
        let score = p.score(&[0.0, 0.0, 0.0]);
        let expected = 1.0f32 / (1.0 + (-4.0f32).exp());
        assert!((score - expected).abs() < 1e-5, "got {score}, expected {expected}");
    }

    #[test]
    fn validate_policy_halt_and_veto_very_small_epsilon_threshold() {
        // Threshold just above 0.0 must be accepted.
        assert!(
            validate_policy(&SafetyPolicy::HaltAndVeto { threshold: f32::EPSILON }).is_ok(),
            "epsilon threshold must be accepted"
        );
    }

    #[test]
    fn probe_weights_score_alternating_sign_weights() {
        // Weights alternate sign: [1, -1, 1, -1] with hidden [1, 1, 1, 1].
        // dot = 1-1+1-1 = 0, bias=0 => sigmoid(0) = 0.5.
        let p = GuardProbeWeights {
            weight: vec![1.0, -1.0, 1.0, -1.0],
            bias: 0.0,
        };
        let score = p.score(&[1.0, 1.0, 1.0, 1.0]);
        assert!((score - 0.5).abs() < 1e-6, "alternating cancellation should yield 0.5, got {score}");
    }

    // ========================================================================
    // Additional uncovered edge cases
    // ========================================================================

    #[test]
    fn probe_weights_score_subnormal_weight_and_hidden() {
        // Subnormal (denormalized) f32 values must not panic and must stay in [0,1].
        let sub = f32::from_bits(1); // smallest positive subnormal
        let p = GuardProbeWeights {
            weight: vec![sub, -sub],
            bias: sub,
        };
        let score = p.score(&[sub, sub]);
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1] for subnormal inputs, got {score}"
        );
        assert!(score.is_finite(), "score must be finite, got {score}");
    }

    #[test]
    fn probe_weights_score_negative_hidden_with_positive_weight() {
        // weight=[1,2,3], hidden=[-1,-2,-3], bias=0
        // dot = -1 -4 -9 = -14 => sigmoid(-14) ≈ ~0
        let p = GuardProbeWeights {
            weight: vec![1.0, 2.0, 3.0],
            bias: 0.0,
        };
        let score = p.score(&[-1.0, -2.0, -3.0]);
        assert!(score < 1e-5, "expected near-zero for fully negative dot product, got {score}");
        assert!(score > 0.0, "sigmoid is strictly positive, got {score}");
    }

    #[test]
    fn probe_weights_score_weight_longer_than_hidden_uses_min_length() {
        // weight has 5 elements, hidden has 2 -> n = min(5, 2) = 2
        let p = GuardProbeWeights {
            weight: vec![2.0, 3.0, 999.0, 999.0, 999.0],
            bias: 1.0,
        };
        let score = p.score(&[1.0, 1.0]);
        // dot = 2*1 + 3*1 = 5, raw = 5 + 1 = 6, sigmoid(6) ≈ 0.9975
        let expected = 1.0f32 / (1.0 + (-6.0f32).exp());
        assert!((score - expected).abs() < 1e-4, "got {score}, expected {expected}");
    }

    #[test]
    fn attachment_last_score_records_negative_value() {
        // GuardrailSharedState stores arbitrary f32 via to_bits/from_bits,
        // including negative scores (can happen with unusual probe weights).
        let att = GuardrailAttachment {
            id: 1,
            actual_layer: 0,
            probe_name: "negative_test".to_string(),
            shared: Arc::new(GuardrailSharedState::new()),
        };
        att.shared.record_score(-0.42);
        let score = att.last_score().unwrap();
        assert!((score - (-0.42)).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn validate_policy_sample_downgrade_accepts_very_small_positive() {
        // min_temperature just above 0.0 must be accepted.
        let result = validate_policy(&SafetyPolicy::SampleDowngrade {
            min_temperature: f32::MIN_POSITIVE,
        });
        assert!(result.is_ok(), "smallest positive f32 should be valid, got {result:?}");
    }

    #[test]
    fn resolve_anchor_absolute_one_layer() {
        // With exactly 1 layer, only Absolute(0) is valid.
        assert_eq!(resolve_anchor(LayerAnchor::Absolute(0), 1).unwrap(), 0);
        assert!(resolve_anchor(LayerAnchor::Absolute(1), 1).is_err());
    }

    #[test]
    fn shared_state_record_score_and_downgrade_independent() {
        // Recording a score does not affect downgrade temperature and vice versa.
        let s = GuardrailSharedState::new();
        s.record_score(0.75);
        assert_eq!(s.downgrade_temp_bits.load(Ordering::Acquire), u32::MAX,
            "score recording should not affect downgrade bits");

        s.record_downgrade(0.25);
        let score_loaded = f32::from_bits(s.last_score_bits.load(Ordering::Acquire));
        assert!((score_loaded - 0.75).abs() < 1e-6,
            "downgrade recording should not affect score bits, got {score_loaded}");

        let temp_loaded = f32::from_bits(s.downgrade_temp_bits.load(Ordering::Acquire));
        assert!((temp_loaded - 0.25).abs() < 1e-6, "got {temp_loaded}");
    }

    #[test]
    fn guardrail_error_not_found_zero_id() {
        let err = GuardrailError::NotFound(0);
        let msg = format!("{err}");
        assert!(msg.contains("0"), "message should contain id 0, got: {msg}");
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[test]
    fn probe_weights_score_single_element_with_large_bias() {
        // Single-dimension probe with bias dominating the computation.
        let p = GuardProbeWeights {
            weight: vec![0.001],
            bias: 100.0,
        };
        let score = p.score(&[1.0]);
        // raw = 0.001 + 100 = 100.001, sigmoid ≈ 1.0
        assert!(score > 0.99, "large positive bias should push score near 1.0, got {score}");
    }

    #[test]
    fn attachment_three_attachments_independent_full_lifecycle() {
        // Three attachments go through score, veto, downgrade, reset independently.
        let atts: Vec<GuardrailAttachment> = (0..3)
            .map(|i| GuardrailAttachment {
                id: i,
                actual_layer: (i * 5) as usize,
                probe_name: format!("probe_{i}"),
                shared: Arc::new(GuardrailSharedState::new()),
            })
            .collect();

        // Attachment 0: veto
        atts[0].shared.record_score(0.95);
        atts[0].shared.trigger_veto("unsafe".to_string());
        // Attachment 1: downgrade
        atts[1].shared.record_score(0.6);
        atts[1].shared.record_downgrade(0.1);
        // Attachment 2: only score, no veto or downgrade
        atts[2].shared.record_score(0.3);

        assert!(atts[0].is_vetoed());
        assert!(atts[0].last_veto_reason().is_some());
        assert!(!atts[1].is_vetoed());
        assert!(atts[1].downgraded_temperature().is_some());
        assert!(!atts[2].is_vetoed());
        assert!(atts[2].downgraded_temperature().is_none());
        assert!((atts[2].last_score().unwrap() - 0.3).abs() < 1e-6);

        // Reset attachment 0, verify others unaffected.
        atts[0].reset();
        assert!(!atts[0].is_vetoed());
        assert!(atts[0].last_score().is_none());
        assert!(atts[1].is_vetoed() == false);
        assert!(atts[1].downgraded_temperature().is_some());
        assert!((atts[2].last_score().unwrap() - 0.3).abs() < 1e-6);
    }
}
