//! IntentBias — 用户意图偏好配置 (REQ-IB-001~003)
//!
//! 提供场景偏好、通信重叠偏好、以及三个数值旋钮，用于调制
//! StrategyArbiter 的自动推导结果。所有字段均有 Default 值，
//! 用户可选择性覆盖。

use crate::engine::arbiter::InferenceMode;

// ── ScenarioHint (REQ-IB-002) ──────────────────────────────────────────────

/// ScenarioHint — 场景偏好枚举 (REQ-IB-002)
///
/// 用户可指定推理场景偏好，影响 InferenceMode 基线选择及
/// 各资源维度的调制系数。`Auto` 保持现有自动推导行为。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScenarioHint {
    #[default]
    Auto,
    LatencyCritical,
    ThroughputOptimal,
    LongContext,
    DistributedHeavy,
    MemoryConstrained,
}

// @trace REQ-IB-002 [entity:ENT-INTENT-BIAS] [api:POST /internal/intent/bias]
impl ScenarioHint {
    /// 映射到 InferenceMode 基线
    pub fn inference_mode_baseline(&self) -> InferenceMode {
        match self {
            Self::Auto => InferenceMode::Latency, // 保持现有推导
            Self::LatencyCritical | Self::LongContext => InferenceMode::Latency,
            Self::ThroughputOptimal | Self::DistributedHeavy | Self::MemoryConstrained => {
                InferenceMode::Throughput
            }
        }
    }

    /// kv_cache_budget_scale 调制系数
    pub fn kv_cache_budget_scale_mod(&self) -> f64 {
        match self {
            Self::Auto => 1.0,
            Self::LatencyCritical => 1.0,
            Self::ThroughputOptimal => 1.0,
            Self::LongContext => 1.5,
            Self::DistributedHeavy => 1.0,
            Self::MemoryConstrained => 0.7,
        }
    }

    /// quantization_aggressiveness 调制系数
    pub fn quantization_aggressiveness_mod(&self) -> f64 {
        match self {
            Self::Auto => 1.0,
            Self::LatencyCritical => 1.0,
            Self::ThroughputOptimal => 1.0,
            Self::LongContext => 0.7,
            Self::DistributedHeavy => 1.0,
            Self::MemoryConstrained => 1.5,
        }
    }

    /// expert_prefetch_priority 调制系数
    pub fn expert_prefetch_priority_mod(&self) -> f64 {
        match self {
            Self::Auto => 1.0,
            Self::LatencyCritical => 1.0,
            Self::ThroughputOptimal => 1.0,
            Self::LongContext => 1.0,
            Self::DistributedHeavy => 2.0,
            Self::MemoryConstrained => 1.0,
        }
    }

    /// pipeline_cost_scale 调制系数
    pub fn pipeline_cost_scale_mod(&self) -> f64 {
        match self {
            Self::Auto => 1.0,
            Self::LatencyCritical => 1.0,
            Self::ThroughputOptimal => 1.0,
            Self::LongContext => 1.0,
            Self::DistributedHeavy => 0.6,
            Self::MemoryConstrained => 1.0,
        }
    }

    /// expert_eviction_aggressiveness 调制系数
    pub fn expert_eviction_aggressiveness_mod(&self) -> f64 {
        match self {
            Self::Auto => 1.0,
            Self::LatencyCritical => 1.0,
            Self::ThroughputOptimal => 1.0,
            Self::LongContext => 1.0,
            Self::DistributedHeavy => 1.0,
            Self::MemoryConstrained => 1.5,
        }
    }
}

// ── OverlapHint (REQ-IB-003) ───────────────────────────────────────────────

/// OverlapHint — 通信重叠偏好枚举 (REQ-IB-003)
///
/// 用户可指定通信-计算重叠策略偏好，影响 Mega-Kernel variant 选择。
/// `Auto` 保持现有自动推导行为。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OverlapHint {
    #[default]
    Auto,
    PreferOverlap,
    PreferIsolated,
    ForceDoubleBuffer,
    ForceFlux,
}

// @trace REQ-IB-003 [entity:ENT-INTENT-BIAS] [api:POST /internal/intent/bias]
impl OverlapHint {
    /// SM90+ 映射的 MkVariant 名
    pub fn mk_variant_sm90plus(&self) -> &'static str {
        match self {
            Self::Auto => "select_mk_variant()",
            Self::PreferOverlap => "Cluster5x3",
            Self::PreferIsolated => "Cluster6x2",
            Self::ForceDoubleBuffer | Self::ForceFlux => "select_mk_variant()",
        }
    }

    /// SM70-89 映射的 MkVariant 名
    pub fn mk_variant_sm70_89(&self) -> &'static str {
        match self {
            Self::Auto | Self::ForceDoubleBuffer | Self::ForceFlux => "select_mk_variant()",
            Self::PreferOverlap | Self::PreferIsolated => "GridSync",
        }
    }

    /// SM<60 映射的 MkVariant 名
    pub fn mk_variant_sm_below60(&self) -> &'static str {
        match self {
            Self::Auto | Self::ForceDoubleBuffer | Self::ForceFlux => "select_mk_variant()",
            Self::PreferIsolated | Self::PreferOverlap => "Serial",
        }
    }

    /// SM60-69 映射的 MkVariant 名 (REQ-IB-003)
    pub fn mk_variant_sm60_69(&self) -> &'static str {
        match self {
            Self::PreferOverlap | Self::PreferIsolated => "GridSync",
            Self::Auto | Self::ForceDoubleBuffer | Self::ForceFlux => "select_mk_variant()",
        }
    }

    /// 解析通信重叠偏好，处理单 GPU 降级 (REQ-IB-003)
    ///
    /// ForceFlux 在单 GPU 环境下无意义，自动降级为 Auto 并发出警告。
    pub fn resolve_overlap(&self, is_single_gpu: bool) -> OverlapHint {
        if *self == Self::ForceFlux && is_single_gpu {
            log::warn!("ForceFlux requires multi-GPU; falling back to Auto on single GPU");
            Self::Auto
        } else {
            *self
        }
    }
}

// ── IntentBias (REQ-IB-001) ────────────────────────────────────────────────

/// IntentBias — 用户意图偏好配置 (REQ-IB-001)
///
/// 5 个必填字段全部有 Default 值，用户可选择性覆盖。
/// `scenario` 和 `comm_overlap` 是枚举偏好，三个数值旋钮
/// (`decode_sm_ratio` / `kv_budget_scale` / `quant_aggression`)
/// 为 `Option` 类型：`None` 时不覆盖自动推导值。
#[derive(Debug, Clone, PartialEq)]
pub struct IntentBias {
    /// 场景偏好枚举，默认 Auto
    pub scenario: ScenarioHint,
    /// 通信重叠偏好枚举，默认 Auto
    pub comm_overlap: OverlapHint,
    /// decode CTA 占比旋钮 (0.1~0.9)，None 时不覆盖自动推导值
    pub decode_sm_ratio: Option<f32>,
    /// KV cache 预算缩放旋钮 (>0)，None 时不覆盖自动推导值
    pub kv_budget_scale: Option<f32>,
    /// 量化激进度旋钮 (0.0~2.0)，None 时不覆盖自动推导值
    pub quant_aggression: Option<f32>,
}

// @trace REQ-IB-001 [entity:ENT-INTENT-BIAS] [api:POST /internal/intent/bias]
impl Default for IntentBias {
    fn default() -> Self {
        Self {
            scenario: ScenarioHint::Auto,
            comm_overlap: OverlapHint::Auto,
            decode_sm_ratio: None,
            kv_budget_scale: None,
            quant_aggression: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IntentBias (REQ-IB-001) ────────────────────────────────────────────

    #[test]
    fn intent_bias_default_values() {
        let bias = IntentBias::default();
        assert_eq!(bias.scenario, ScenarioHint::Auto);
        assert_eq!(bias.comm_overlap, OverlapHint::Auto);
        assert!(bias.decode_sm_ratio.is_none());
        assert!(bias.kv_budget_scale.is_none());
        assert!(bias.quant_aggression.is_none());
    }

    #[test]
    fn intent_bias_custom_values() {
        let bias = IntentBias {
            scenario: ScenarioHint::LatencyCritical,
            comm_overlap: OverlapHint::PreferOverlap,
            decode_sm_ratio: Some(0.5),
            kv_budget_scale: Some(1.2),
            quant_aggression: Some(0.8),
        };
        assert_eq!(bias.scenario, ScenarioHint::LatencyCritical);
        assert_eq!(bias.comm_overlap, OverlapHint::PreferOverlap);
        assert!((bias.decode_sm_ratio.unwrap() - 0.5).abs() < f32::EPSILON);
        assert!((bias.kv_budget_scale.unwrap() - 1.2).abs() < f32::EPSILON);
        assert!((bias.quant_aggression.unwrap() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn intent_bias_equality() {
        let a = IntentBias::default();
        let b = IntentBias::default();
        assert_eq!(a, b);
    }

    #[test]
    fn intent_bias_inequality_scenario() {
        let a = IntentBias::default();
        let b = IntentBias {
            scenario: ScenarioHint::ThroughputOptimal,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn intent_bias_clone_independence() {
        let mut bias = IntentBias {
            scenario: ScenarioHint::LongContext,
            comm_overlap: OverlapHint::ForceDoubleBuffer,
            decode_sm_ratio: Some(0.3),
            kv_budget_scale: None,
            quant_aggression: Some(1.5),
        };
        let cloned = bias.clone();
        bias.decode_sm_ratio = Some(0.9);
        assert!((cloned.decode_sm_ratio.unwrap() - 0.3).abs() < f32::EPSILON);
    }

    // ── ScenarioHint (REQ-IB-002) ──────────────────────────────────────────

    #[test]
    fn scenario_hint_default_is_auto() {
        assert_eq!(ScenarioHint::default(), ScenarioHint::Auto);
    }

    #[test]
    fn scenario_hint_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            ScenarioHint::Auto,
            ScenarioHint::LatencyCritical,
            ScenarioHint::ThroughputOptimal,
            ScenarioHint::LongContext,
            ScenarioHint::DistributedHeavy,
            ScenarioHint::MemoryConstrained,
        ];
        let set: HashSet<ScenarioHint> = all.into_iter().collect();
        assert_eq!(set.len(), 6);
    }

    #[test]
    fn scenario_hint_inference_mode_baseline() {
        assert_eq!(
            ScenarioHint::Auto.inference_mode_baseline(),
            InferenceMode::Latency
        );
        assert_eq!(
            ScenarioHint::LatencyCritical.inference_mode_baseline(),
            InferenceMode::Latency
        );
        assert_eq!(
            ScenarioHint::LongContext.inference_mode_baseline(),
            InferenceMode::Latency
        );
        assert_eq!(
            ScenarioHint::ThroughputOptimal.inference_mode_baseline(),
            InferenceMode::Throughput
        );
        assert_eq!(
            ScenarioHint::DistributedHeavy.inference_mode_baseline(),
            InferenceMode::Throughput
        );
        assert_eq!(
            ScenarioHint::MemoryConstrained.inference_mode_baseline(),
            InferenceMode::Throughput
        );
    }

    #[test]
    fn scenario_hint_kv_cache_budget_scale_mod() {
        assert!((ScenarioHint::Auto.kv_cache_budget_scale_mod() - 1.0).abs() < f64::EPSILON);
        assert!(
            (ScenarioHint::LongContext.kv_cache_budget_scale_mod() - 1.5).abs() < f64::EPSILON
        );
        assert!(
            (ScenarioHint::MemoryConstrained.kv_cache_budget_scale_mod() - 0.7).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn scenario_hint_quantization_aggressiveness_mod() {
        assert!(
            (ScenarioHint::Auto.quantization_aggressiveness_mod() - 1.0).abs() < f64::EPSILON
        );
        assert!(
            (ScenarioHint::LongContext.quantization_aggressiveness_mod() - 0.7).abs()
                < f64::EPSILON
        );
        assert!(
            (ScenarioHint::MemoryConstrained.quantization_aggressiveness_mod() - 1.5).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn scenario_hint_expert_prefetch_priority_mod() {
        assert!(
            (ScenarioHint::Auto.expert_prefetch_priority_mod() - 1.0).abs() < f64::EPSILON
        );
        assert!(
            (ScenarioHint::DistributedHeavy.expert_prefetch_priority_mod() - 2.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn scenario_hint_pipeline_cost_scale_mod() {
        assert!((ScenarioHint::Auto.pipeline_cost_scale_mod() - 1.0).abs() < f64::EPSILON);
        assert!(
            (ScenarioHint::DistributedHeavy.pipeline_cost_scale_mod() - 0.6).abs() < f64::EPSILON
        );
    }

    #[test]
    fn scenario_hint_expert_eviction_aggressiveness_mod() {
        assert!(
            (ScenarioHint::Auto.expert_eviction_aggressiveness_mod() - 1.0).abs() < f64::EPSILON
        );
        assert!(
            (ScenarioHint::MemoryConstrained.expert_eviction_aggressiveness_mod() - 1.5).abs()
                < f64::EPSILON
        );
        assert!(
            (ScenarioHint::DistributedHeavy.expert_eviction_aggressiveness_mod() - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn scenario_hint_copy_trait() {
        let original = ScenarioHint::LongContext;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn scenario_hint_debug_trait() {
        let debug = format!("{:?}", ScenarioHint::MemoryConstrained);
        assert!(debug.contains("MemoryConstrained"));
    }

    // ── OverlapHint (REQ-IB-003) ───────────────────────────────────────────

    #[test]
    fn overlap_hint_default_is_auto() {
        assert_eq!(OverlapHint::default(), OverlapHint::Auto);
    }

    #[test]
    fn overlap_hint_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            OverlapHint::Auto,
            OverlapHint::PreferOverlap,
            OverlapHint::PreferIsolated,
            OverlapHint::ForceDoubleBuffer,
            OverlapHint::ForceFlux,
        ];
        let set: HashSet<OverlapHint> = all.into_iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn overlap_hint_mk_variant_sm90plus() {
        assert_eq!(OverlapHint::Auto.mk_variant_sm90plus(), "select_mk_variant()");
        assert_eq!(OverlapHint::PreferOverlap.mk_variant_sm90plus(), "Cluster5x3");
        assert_eq!(OverlapHint::PreferIsolated.mk_variant_sm90plus(), "Cluster6x2");
        assert_eq!(
            OverlapHint::ForceDoubleBuffer.mk_variant_sm90plus(),
            "select_mk_variant()"
        );
        assert_eq!(OverlapHint::ForceFlux.mk_variant_sm90plus(), "select_mk_variant()");
    }

    #[test]
    fn overlap_hint_mk_variant_sm70_89() {
        assert_eq!(OverlapHint::Auto.mk_variant_sm70_89(), "select_mk_variant()");
        assert_eq!(OverlapHint::PreferOverlap.mk_variant_sm70_89(), "GridSync");
        assert_eq!(OverlapHint::PreferIsolated.mk_variant_sm70_89(), "GridSync");
    }

    #[test]
    fn overlap_hint_mk_variant_sm_below60() {
        assert_eq!(OverlapHint::Auto.mk_variant_sm_below60(), "select_mk_variant()");
        assert_eq!(OverlapHint::PreferIsolated.mk_variant_sm_below60(), "Serial");
        assert_eq!(OverlapHint::PreferOverlap.mk_variant_sm_below60(), "Serial");
    }

    #[test]
    fn overlap_hint_copy_trait() {
        let original = OverlapHint::ForceDoubleBuffer;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn overlap_hint_debug_trait() {
        let debug = format!("{:?}", OverlapHint::ForceFlux);
        assert!(debug.contains("ForceFlux"));
    }

    #[test]
    fn overlap_hint_mk_variant_sm60_69() {
        assert_eq!(OverlapHint::PreferOverlap.mk_variant_sm60_69(), "GridSync");
        assert_eq!(OverlapHint::PreferIsolated.mk_variant_sm60_69(), "GridSync");
        assert_eq!(OverlapHint::Auto.mk_variant_sm60_69(), "select_mk_variant()");
        assert_eq!(
            OverlapHint::ForceDoubleBuffer.mk_variant_sm60_69(),
            "select_mk_variant()"
        );
        assert_eq!(OverlapHint::ForceFlux.mk_variant_sm60_69(), "select_mk_variant()");
    }

    #[test]
    fn resolve_overlap_force_flux_single_gpu_downgrades() {
        let result = OverlapHint::ForceFlux.resolve_overlap(true);
        assert_eq!(result, OverlapHint::Auto);
    }

    #[test]
    fn resolve_overlap_force_flux_multi_gpu_unchanged() {
        let result = OverlapHint::ForceFlux.resolve_overlap(false);
        assert_eq!(result, OverlapHint::ForceFlux);
    }

    #[test]
    fn resolve_overlap_other_variants_unchanged() {
        for hint in [
            OverlapHint::Auto,
            OverlapHint::PreferOverlap,
            OverlapHint::PreferIsolated,
            OverlapHint::ForceDoubleBuffer,
        ] {
            assert_eq!(hint.resolve_overlap(true), hint, "single GPU should not affect {:?}", hint);
            assert_eq!(hint.resolve_overlap(false), hint, "multi GPU should not affect {:?}", hint);
        }
    }
}
