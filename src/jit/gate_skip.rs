//! Gate-First Skip Integration — FFN 死神经元跳过 (SPEC §13.1, §13.5)
//!
//! ## 核心职责
//! 将 GateFirstSkipDetector (决策层) 与 FFN 图执行管线集成:
//! - 从 TelemetryAggregator / KvPageHeader 读取死神经元比例
//! - 做出 GateSkipDecision (FullCompute / MaskedCompute / Skip)
//! - 为图构建器提供 Gate-First Skip 配置
//! - 跟踪逐层决策历史
//!
//! ## 数据流
//! ```
//! KvPageHeader.dead_neuron_ratio ──→ GateFirstSkipDetector.decide()
//!                                         ↓
//!                               GateSkipDecision
//!                            ┌──────────┼──────────┐
//!                            ↓          ↓          ↓
//!                       FullCompute  MaskedCompute  Skip
//!                       (正常FFN)   (压缩FFN)   (跳过FFN)
//! ```
//!
//! ## 图构建器集成点
//! FFN 图中 SwiGlu → gemm_down 之间插入 GateMask:
//! ```
//! SwiGlu → [GateMask: 检测死神经元, 生成mask] → MaskedGemm(gemm_down) → ResidualWithTelemetry
//! ```

use super::epilogue::{GateFirstSkipConfig, GateSkipDecision, GateFirstSkipDetector, TelemetryAggregator};
use crate::kv_cache::{KvPageHeader, dead_ratio_to_f32};

/// 逐层 Gate-First Skip 决策记录
#[derive(Debug, Clone)]
pub struct LayerSkipRecord {
    /// 层索引
    pub layer_idx: usize,
    /// 该层的死神经元比例
    pub dead_neuron_ratio: f32,
    /// 决策结果
    pub decision: GateSkipDecision,
}

/// Gate-First Skip 层级集成器
///
/// 管理多层 FFN 的 Gate-First Skip 决策流程:
/// - 维护 GateFirstSkipDetector 实例
/// - 从 TelemetryAggregator / KvPageHeader 提取信号
/// - 逐层做出跳过决策
/// - 记录决策历史
pub struct GateFirstSkipLayer {
    /// 决策器
    detector: GateFirstSkipDetector,
    /// 逐层决策记录 (layer_idx → record)
    history: Vec<LayerSkipRecord>,
    /// 总层数 (从模型配置获取)
    num_layers: usize,
    /// 累计跳过的层数
    total_skipped_layers: usize,
    /// 累计掩码计算的层数
    total_masked_layers: usize,
    /// 累计完整计算的层数
    total_full_layers: usize,
}

impl GateFirstSkipLayer {
    /// 创建新的 Gate-First Skip 层级集成器
    pub fn new(config: GateFirstSkipConfig, num_layers: usize) -> Self {
        Self {
            detector: GateFirstSkipDetector::new(config),
            history: Vec::with_capacity(num_layers),
            num_layers,
            total_skipped_layers: 0,
            total_masked_layers: 0,
            total_full_layers: 0,
        }
    }

    /// 从死神经元比例做出当前层的跳过决策
    pub fn decide_for_layer(&mut self, layer_idx: usize, dead_ratio: f32) -> GateSkipDecision {
        let decision = self.detector.decide(dead_ratio);
        self.record_decision(layer_idx, dead_ratio, decision);
        decision
    }

    /// 从 TelemetryAggregator 做出当前层的跳过决策
    pub fn decide_from_aggregator(
        &mut self,
        layer_idx: usize,
        agg: &TelemetryAggregator,
    ) -> GateSkipDecision {
        let decision = self.detector.decide_from_telemetry(agg);
        self.record_decision(layer_idx, agg.dead_neuron_ratio(), decision);
        decision
    }

    /// 从 KvPageHeader 做出当前层的跳过决策
    ///
    /// KvPageHeader 由 Epilogue STG 指令写入 (§9.5)，
    /// 包含上一层的死神经元比例。
    pub fn decide_from_page_header(
        &mut self,
        layer_idx: usize,
        header: &KvPageHeader,
    ) -> GateSkipDecision {
        let dead_ratio = dead_ratio_to_f32(header.dead_ratio);
        self.decide_for_layer(layer_idx, dead_ratio)
    }

    /// 记录决策到历史
    fn record_decision(
        &mut self,
        layer_idx: usize,
        dead_ratio: f32,
        decision: GateSkipDecision,
    ) {
        // 更新统计
        match decision {
            GateSkipDecision::FullCompute => self.total_full_layers += 1,
            GateSkipDecision::MaskedCompute => self.total_masked_layers += 1,
            GateSkipDecision::Skip => self.total_skipped_layers += 1,
        }

        // 记录历史 (保持 history 按 layer_idx 排序)
        let record = LayerSkipRecord {
            layer_idx,
            dead_neuron_ratio: dead_ratio,
            decision,
        };

        // 如果该层已有记录，更新；否则新增
        if let Some(existing) = self.history.iter_mut().find(|r| r.layer_idx == layer_idx) {
            *existing = record;
        } else {
            self.history.push(record);
        }
    }

    /// 获取某层的决策记录
    pub fn layer_record(&self, layer_idx: usize) -> Option<&LayerSkipRecord> {
        self.history.iter().find(|r| r.layer_idx == layer_idx)
    }

    /// 获取配置引用
    pub fn config(&self) -> &GateFirstSkipConfig {
        &self.detector.config
    }

    /// 是否启用 Gate-First Skip
    pub fn is_enabled(&self) -> bool {
        self.detector.config.enabled
    }

    /// 总层数
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// 累计跳过的层数
    pub fn total_skipped_layers(&self) -> usize {
        self.total_skipped_layers
    }

    /// 累计掩码计算的层数
    pub fn total_masked_layers(&self) -> usize {
        self.total_masked_layers
    }

    /// 累计完整计算的层数
    pub fn total_full_layers(&self) -> usize {
        self.total_full_layers
    }

    /// 总决策次数
    pub fn total_decisions(&self) -> usize {
        self.total_skipped_layers + self.total_masked_layers + self.total_full_layers
    }

    /// 跳过率 (跳过 + 掩码 的比例)
    pub fn skip_rate(&self) -> f32 {
        if self.total_decisions() == 0 {
            return 0.0;
        }
        (self.total_skipped_layers + self.total_masked_layers) as f32
            / self.total_decisions() as f32
    }

    /// 重置统计数据 (保留配置)
    pub fn reset_stats(&mut self) {
        self.history.clear();
        self.total_skipped_layers = 0;
        self.total_masked_layers = 0;
        self.total_full_layers = 0;
    }

    /// 获取全部决策历史
    pub fn history(&self) -> &[LayerSkipRecord] {
        &self.history
    }

    /// 返回最近 N 层的决策记录 (从后往前)
    pub fn recent_decisions(&self, n: usize) -> Vec<&LayerSkipRecord> {
        self.history.iter().rev().take(n).collect()
    }
}

/// 批次级 Gate-First Skip 决策汇总
///
/// 用于 build_batch() 阶段 (§9.6.3 Dispatch-Time)，
/// 汇总整个 batch 的 Gate-First Skip 状态，决定 Variant 选择。
#[derive(Debug, Clone)]
pub struct BatchSkipSummary {
    /// batch 中每个请求的最近一层决策
    pub per_request_decisions: Vec<GateSkipDecision>,
    /// batch 级别的平均死神经元比例
    pub avg_dead_neuron_ratio: f32,
    /// batch 级别建议 (全计算 / 部分掩码 / 全跳过)
    pub batch_advice: BatchSkipAdvice,
}

/// 批次级 Gate-First Skip 建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSkipAdvice {
    /// 所有请求完整计算 — 使用标准 Dense FFN Variant
    FullCompute,
    /// 部分请求需要掩码 — 使用 MaskedGemm Variant
    PartialMask,
    /// 大部分请求可跳过 — 考虑 Compact→Execute→Scatter
    MostlySkip,
}

impl BatchSkipSummary {
    /// 从多个 GateFirstSkipLayer 的决策生成批次汇总
    ///
    /// 在 build_batch() 阶段调用，
    /// 用于决定使用哪个预编译 Variant。
    pub fn from_decisions(decisions: &[GateSkipDecision]) -> Self {
        let total = decisions.len();
        if total == 0 {
            return Self {
                per_request_decisions: vec![],
                avg_dead_neuron_ratio: 0.0,
                batch_advice: BatchSkipAdvice::FullCompute,
            };
        }

        let skip_count = decisions.iter().filter(|&&d| d == GateSkipDecision::Skip).count();
        let masked_count = decisions
            .iter()
            .filter(|&&d| d == GateSkipDecision::MaskedCompute)
            .count();
        let full_count = decisions
            .iter()
            .filter(|&&d| d == GateSkipDecision::FullCompute)
            .count();

        // 死神经元比例估算: Skip=0.8, Masked=0.35, Full=0.1
        let avg_ratio = (skip_count as f32 * 0.8 + masked_count as f32 * 0.35
            + full_count as f32 * 0.1)
            / total as f32;

        // 批次建议规则:
        // - >50% Skip → MostlySkip (触发 Compact→Execute→Scatter)
        // - >25% Masked/Skip → PartialMask (使用 MaskedGemm Variant)
        // - 其余 → FullCompute (标准 Dense Variant)
        let batch_advice = if skip_count * 2 > total {
            BatchSkipAdvice::MostlySkip
        } else if (skip_count + masked_count) * 4 > total {
            BatchSkipAdvice::PartialMask
        } else {
            BatchSkipAdvice::FullCompute
        };

        Self {
            per_request_decisions: decisions.to_vec(),
            avg_dead_neuron_ratio: avg_ratio,
            batch_advice,
        }
    }

    /// 返回批次建议是否需要 MaskedGemm Variant
    pub fn needs_masked_variant(&self) -> bool {
        matches!(self.batch_advice, BatchSkipAdvice::PartialMask | BatchSkipAdvice::MostlySkip)
    }

    /// 返回批次建议是否需要 Compact→Execute→Scatter
    pub fn needs_compact(&self) -> bool {
        matches!(self.batch_advice, BatchSkipAdvice::MostlySkip)
    }
}

/// 将 GateSkipDecision 映射到 VariantRegistry 的 MechanismId
///
/// 用于 §18.4 变体选择决策:
/// - FullCompute → 不包含 GateFirstSkip 机制
/// - MaskedCompute → 包含 GateFirstSkip + RaggedCompaction 机制
/// - Skip → 包含 GateFirstSkip + RaggedCompaction + 深度跳过
impl From<GateSkipDecision> for Option<super::variant_registry::MechanismId> {
    fn from(decision: GateSkipDecision) -> Self {
        match decision {
            GateSkipDecision::FullCompute => None,
            GateSkipDecision::MaskedCompute => {
                Some(super::variant_registry::MechanismId::GateFirstSkip)
            }
            GateSkipDecision::Skip => {
                Some(super::variant_registry::MechanismId::GateFirstSkip)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_first_skip_layer_basic() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // Layer 0-3: 不允许跳过 (min_skip_layer=4)
        for i in 0..4 {
            let decision = layer.decide_for_layer(i, 0.8);
            // 即使死神经元 80%, 前几层仍然由 detector 处理
            // (detector 本身不检查 min_skip_layer, 那是 ResidualBypass)
            assert!(matches!(decision, GateSkipDecision::Skip));
        }

        // Layer 10: 死神经元 60% → Skip
        let decision = layer.decide_for_layer(10, 0.6);
        assert_eq!(decision, GateSkipDecision::Skip);

        // Layer 15: 死神经元 30% → MaskedCompute
        let decision = layer.decide_for_layer(15, 0.3);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);

        // Layer 20: 死神经元 10% → FullCompute
        let decision = layer.decide_for_layer(20, 0.1);
        assert_eq!(decision, GateSkipDecision::FullCompute);

        assert_eq!(layer.total_skipped_layers(), 5); // layers 0-3 + 10
        assert_eq!(layer.total_masked_layers(), 1); // layer 15
        assert_eq!(layer.total_full_layers(), 1); // layer 20
    }

    #[test]
    fn test_gate_first_skip_layer_from_aggregator() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();

        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.6 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::Skip);
        assert_eq!(layer.total_skipped_layers(), 1);
    }

    #[test]
    fn test_gate_first_skip_layer_from_page_header() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.35);

        let decision = layer.decide_from_page_header(15, &header);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
        assert_eq!(layer.total_masked_layers(), 1);
    }

    #[test]
    fn test_gate_first_skip_layer_disabled() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let decision = layer.decide_for_layer(10, 0.9);
        assert_eq!(decision, GateSkipDecision::FullCompute);
        assert!(!layer.is_enabled());
        assert_eq!(layer.total_full_layers(), 1);
    }

    #[test]
    fn test_batch_skip_summary_full_compute() {
        let decisions = vec![
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!(!summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_partial_mask() {
        let decisions = vec![
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::Skip,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // 1 skip + 1 masked = 2, 2*4=8 > 4 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!(summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_mostly_skip() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // 3 skip * 2 = 6 > 4 → MostlySkip
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.needs_masked_variant());
        assert!(summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_empty() {
        let summary = BatchSkipSummary::from_decisions(&[]);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert_eq!(summary.avg_dead_neuron_ratio, 0.0);
    }

    #[test]
    fn test_layer_history_tracking() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.6);
        layer.decide_for_layer(10, 0.3);
        layer.decide_for_layer(15, 0.1);

        assert_eq!(layer.history().len(), 3);

        // 更新 layer 10 的决策
        layer.decide_for_layer(10, 0.7);
        assert_eq!(layer.history().len(), 3); // 仍然是 3 条记录

        // 验证更新后的值
        let record = layer.layer_record(10).unwrap();
        assert!((record.dead_neuron_ratio - 0.7).abs() < 0.001);
        assert_eq!(record.decision, GateSkipDecision::Skip);
    }

    #[test]
    fn test_recent_decisions() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.6);
        layer.decide_for_layer(10, 0.3);
        layer.decide_for_layer(15, 0.1);

        let recent = layer.recent_decisions(2);
        assert_eq!(recent.len(), 2);
        // 最新的两条记录 (按插入顺序倒序)
        assert_eq!(recent[0].layer_idx, 15);
        assert_eq!(recent[1].layer_idx, 10);
    }

    #[test]
    fn test_skip_rate() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.6);  // Skip
        layer.decide_for_layer(10, 0.3); // MaskedCompute
        layer.decide_for_layer(15, 0.1); // FullCompute

        let rate = layer.skip_rate();
        // (1 Skip + 1 Masked) / 3 = 2/3 ≈ 0.667
        assert!((rate - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_reset_stats() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.6);
        layer.decide_for_layer(10, 0.3);
        assert_eq!(layer.total_decisions(), 2);

        layer.reset_stats();
        assert_eq!(layer.total_decisions(), 0);
        assert!(layer.history().is_empty());
        assert!(layer.is_enabled()); // 配置保持不变
    }

    #[test]
    fn test_mechanism_id_mapping() {
        let mapping: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::MaskedCompute.into();
        assert_eq!(
            mapping,
            Some(super::super::variant_registry::MechanismId::GateFirstSkip)
        );

        let mapping: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::FullCompute.into();
        assert!(mapping.is_none());
    }

    // ── LayerSkipRecord ──

    #[test]
    fn layer_skip_record_debug() {
        let record = LayerSkipRecord {
            layer_idx: 7,
            dead_neuron_ratio: 0.45,
            decision: GateSkipDecision::MaskedCompute,
        };
        let debug = format!("{record:?}");
        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("dead_neuron_ratio"));
    }

    #[test]
    fn layer_skip_record_clone() {
        let record = LayerSkipRecord {
            layer_idx: 3,
            dead_neuron_ratio: 0.2,
            decision: GateSkipDecision::FullCompute,
        };
        let cloned = record.clone();
        assert_eq!(record.layer_idx, cloned.layer_idx);
        assert_eq!(record.dead_neuron_ratio, cloned.dead_neuron_ratio);
        assert_eq!(record.decision, cloned.decision);
    }

    // ── GateFirstSkipLayer accessors ──

    #[test]
    fn layer_num_layers() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 64);
        assert_eq!(layer.num_layers(), 64);
    }

    #[test]
    fn layer_config_access() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.7,
            ..Default::default()
        };
        let layer = GateFirstSkipLayer::new(config, 32);
        assert!(layer.config().enabled);
        assert!((layer.config().skip_threshold - 0.7).abs() < 1e-6);
    }

    #[test]
    fn layer_record_none_for_missing() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);
        assert!(layer.layer_record(0).is_none());
        assert!(layer.layer_record(99).is_none());
    }

    #[test]
    fn skip_rate_zero_when_no_decisions() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);
        assert_eq!(layer.skip_rate(), 0.0);
    }

    #[test]
    fn skip_rate_all_full_compute_is_zero() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.0);
        layer.decide_for_layer(1, 0.05);
        assert_eq!(layer.skip_rate(), 0.0);
    }

    #[test]
    fn skip_rate_all_skip_is_one() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.9);
        layer.decide_for_layer(1, 0.95);
        let rate = layer.skip_rate();
        assert!((rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn recent_decisions_n_exceeds_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.6);
        let recent = layer.recent_decisions(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn recent_decisions_zero_returns_empty() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.6);
        let recent = layer.recent_decisions(0);
        assert!(recent.is_empty());
    }

    #[test]
    fn reset_preserves_num_layers_and_enabled() {
        let config = GateFirstSkipConfig { enabled: true, ..Default::default() };
        let mut layer = GateFirstSkipLayer::new(config, 48);
        layer.decide_for_layer(10, 0.8);
        layer.reset_stats();
        assert_eq!(layer.num_layers(), 48);
        assert!(layer.is_enabled());
        assert_eq!(layer.total_decisions(), 0);
    }

    // ── BatchSkipSummary ──

    #[test]
    fn batch_skip_summary_debug() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::Skip]);
        let debug = format!("{summary:?}");
        assert!(debug.contains("per_request_decisions"));
        assert!(debug.contains("avg_dead_neuron_ratio"));
        assert!(debug.contains("batch_advice"));
    }

    #[test]
    fn batch_skip_summary_clone() {
        let summary = BatchSkipSummary::from_decisions(&[
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
        ]);
        let cloned = summary.clone();
        assert_eq!(summary.per_request_decisions.len(), cloned.per_request_decisions.len());
        assert_eq!(summary.batch_advice, cloned.batch_advice);
    }

    #[test]
    fn batch_skip_summary_avg_ratio_all_skip() {
        let decisions = vec![GateSkipDecision::Skip; 4];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!((summary.avg_dead_neuron_ratio - 0.8).abs() < 0.01);
    }

    #[test]
    fn batch_skip_summary_avg_ratio_all_full() {
        let decisions = vec![GateSkipDecision::FullCompute; 4];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!((summary.avg_dead_neuron_ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn batch_skip_advice_boundary_mostly_skip() {
        // 3/5 = 60% skip → skip_count*2=6 > 5 → MostlySkip
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
    }

    #[test]
    fn batch_skip_advice_just_below_mostly_skip() {
        // 2/5 = 40% skip → skip_count*2=4 ≤ 5 → not MostlySkip
        // (skip+masked)*4 = (2+0)*4=8 > 5 → PartialMask
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_skip_advice_no_skip_no_masked_is_full() {
        let decisions = vec![GateSkipDecision::FullCompute; 10];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
    }

    // ── BatchSkipAdvice ──

    #[test]
    fn batch_skip_advice_equality() {
        assert_eq!(BatchSkipAdvice::FullCompute, BatchSkipAdvice::FullCompute);
        assert_ne!(BatchSkipAdvice::FullCompute, BatchSkipAdvice::PartialMask);
        assert_ne!(BatchSkipAdvice::PartialMask, BatchSkipAdvice::MostlySkip);
    }

    #[test]
    fn batch_skip_advice_copy() {
        let a = BatchSkipAdvice::MostlySkip;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn batch_skip_advice_debug() {
        let debug = format!("{:?}", BatchSkipAdvice::PartialMask);
        assert!(debug.contains("PartialMask"));
    }

    // ── GateSkipDecision → MechanismId: Skip variant ──

    #[test]
    fn mechanism_id_skip_maps_to_gate_first_skip() {
        let mapping: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::Skip.into();
        assert_eq!(
            mapping,
            Some(super::super::variant_registry::MechanismId::GateFirstSkip)
        );
    }

    // ── Boundary: dead_ratio exactly at thresholds ──

    #[test]
    fn decide_at_skip_threshold_is_not_skip() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // skip_threshold default = 0.5 — strict > comparison: 0.5 is NOT > 0.5
        let decision = layer.decide_for_layer(10, 0.5);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn decide_just_below_skip_threshold() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.49 < 0.5 (skip_threshold) but >= 0.2 (masked_threshold) → MaskedCompute
        let decision = layer.decide_for_layer(10, 0.49);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn decide_just_below_masked_threshold() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.19 < 0.2 (masked_threshold) → FullCompute
        let decision = layer.decide_for_layer(10, 0.19);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_zero_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 0.0);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_full_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 1.0);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    // ── LayerSkipRecord: field access ──

    #[test]
    fn layer_skip_record_field_access_all_variants() {
        let rec_full = LayerSkipRecord {
            layer_idx: 0,
            dead_neuron_ratio: 0.05,
            decision: GateSkipDecision::FullCompute,
        };
        assert_eq!(rec_full.layer_idx, 0);
        assert!((rec_full.dead_neuron_ratio - 0.05).abs() < 1e-6);
        assert_eq!(rec_full.decision, GateSkipDecision::FullCompute);

        let rec_masked = LayerSkipRecord {
            layer_idx: 100,
            dead_neuron_ratio: 0.3,
            decision: GateSkipDecision::MaskedCompute,
        };
        assert_eq!(rec_masked.layer_idx, 100);
        assert!((rec_masked.dead_neuron_ratio - 0.3).abs() < 1e-6);
        assert_eq!(rec_masked.decision, GateSkipDecision::MaskedCompute);

        let rec_skip = LayerSkipRecord {
            layer_idx: usize::MAX,
            dead_neuron_ratio: 0.99,
            decision: GateSkipDecision::Skip,
        };
        assert_eq!(rec_skip.layer_idx, usize::MAX);
        assert!((rec_skip.dead_neuron_ratio - 0.99).abs() < 1e-6);
        assert_eq!(rec_skip.decision, GateSkipDecision::Skip);
    }

    #[test]
    fn layer_skip_record_zero_ratio() {
        let record = LayerSkipRecord {
            layer_idx: 0,
            dead_neuron_ratio: 0.0,
            decision: GateSkipDecision::FullCompute,
        };
        assert_eq!(record.dead_neuron_ratio, 0.0);
    }

    #[test]
    fn layer_skip_record_max_ratio() {
        let record = LayerSkipRecord {
            layer_idx: 255,
            dead_neuron_ratio: 1.0,
            decision: GateSkipDecision::Skip,
        };
        assert!((record.dead_neuron_ratio - 1.0).abs() < 1e-6);
    }

    // ── BatchSkipAdvice: exhaustive variant coverage ──

    #[test]
    fn batch_skip_advice_all_variants_distinct() {
        let variants = [
            BatchSkipAdvice::FullCompute,
            BatchSkipAdvice::PartialMask,
            BatchSkipAdvice::MostlySkip,
        ];
        // Verify all three are pairwise distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn batch_skip_advice_clone() {
        let a = BatchSkipAdvice::PartialMask;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── BatchSkipSummary: per_request_decisions field ──

    #[test]
    fn batch_skip_summary_preserves_decisions() {
        let decisions = vec![
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::Skip,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.per_request_decisions.len(), 3);
        assert_eq!(summary.per_request_decisions[0], GateSkipDecision::FullCompute);
        assert_eq!(summary.per_request_decisions[1], GateSkipDecision::MaskedCompute);
        assert_eq!(summary.per_request_decisions[2], GateSkipDecision::Skip);
    }

    #[test]
    fn batch_skip_summary_single_decision_full() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute]);
        assert_eq!(summary.per_request_decisions.len(), 1);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!((summary.avg_dead_neuron_ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn batch_skip_summary_single_decision_skip() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::Skip]);
        // 1 skip, total=1, skip*2=2 > 1 → MostlySkip
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.needs_compact());
        assert!(summary.needs_masked_variant());
    }

    #[test]
    fn batch_skip_summary_single_decision_masked() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::MaskedCompute]);
        // 0 skip, 1 masked, total=1, (skip+masked)*4=4 > 1 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!(!summary.needs_compact());
        assert!(summary.needs_masked_variant());
    }

    #[test]
    fn batch_skip_summary_mixed_avg_ratio() {
        let decisions = vec![
            GateSkipDecision::Skip,        // 0.8
            GateSkipDecision::MaskedCompute, // 0.35
            GateSkipDecision::FullCompute,  // 0.1
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        let expected = (0.8 + 0.35 + 0.1) / 3.0;
        assert!((summary.avg_dead_neuron_ratio - expected).abs() < 0.01);
    }

    #[test]
    fn batch_skip_summary_all_masked() {
        let decisions = vec![GateSkipDecision::MaskedCompute; 4];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // 0 skip, 4 masked, total=4, (0+4)*4=16 > 4 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!((summary.avg_dead_neuron_ratio - 0.35).abs() < 0.01);
    }

    // ── GateFirstSkipLayer: num_layers = 0 ──

    #[test]
    fn layer_new_with_zero_layers() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 0);
        assert_eq!(layer.num_layers(), 0);
        assert_eq!(layer.total_decisions(), 0);
        assert_eq!(layer.skip_rate(), 0.0);
    }

    #[test]
    fn layer_decide_with_zero_num_layers_still_records() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 0);
        let decision = layer.decide_for_layer(0, 0.8);
        assert_eq!(decision, GateSkipDecision::Skip);
        assert_eq!(layer.total_decisions(), 1);
    }

    // ── GateFirstSkipLayer: NaN dead_ratio ──

    #[test]
    fn decide_nan_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, f32::NAN);
        // NaN comparisons are false, so neither > skip_threshold nor > skip_threshold*0.5
        // → FullCompute
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: negative dead_ratio ──

    #[test]
    fn decide_negative_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, -0.5);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: custom thresholds ──

    #[test]
    fn decide_custom_high_skip_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.9,
            dead_neuron_epsilon: 1e-3,
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.6 < 0.9 → not Skip; 0.6 > 0.9*0.5=0.45 → MaskedCompute
        assert_eq!(layer.decide_for_layer(10, 0.6), GateSkipDecision::MaskedCompute);
        assert_eq!(layer.decide_for_layer(11, 0.95), GateSkipDecision::Skip);
    }

    #[test]
    fn decide_custom_very_low_skip_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.1,
            dead_neuron_epsilon: 1e-3,
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.15 > 0.1 → Skip
        assert_eq!(layer.decide_for_layer(10, 0.15), GateSkipDecision::Skip);
        // 0.06 > 0.1*0.5=0.05 → MaskedCompute
        assert_eq!(layer.decide_for_layer(11, 0.06), GateSkipDecision::MaskedCompute);
        // 0.03 < 0.05 → FullCompute
        assert_eq!(layer.decide_for_layer(12, 0.03), GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: history update overwrites correctly ──

    #[test]
    fn history_update_changes_decision_type() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // First decision: FullCompute
        layer.decide_for_layer(5, 0.1);
        assert_eq!(layer.total_full_layers(), 1);
        assert_eq!(layer.total_skipped_layers(), 0);

        // Update same layer: Skip
        layer.decide_for_layer(5, 0.9);
        assert_eq!(layer.history().len(), 1); // still 1 record
        let record = layer.layer_record(5).unwrap();
        assert_eq!(record.decision, GateSkipDecision::Skip);
        assert!((record.dead_neuron_ratio - 0.9).abs() < 0.001);

        // Counts reflect both decisions
        assert_eq!(layer.total_full_layers(), 1);
        assert_eq!(layer.total_skipped_layers(), 1);
    }

    #[test]
    fn history_many_layers_all_recorded() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 100);

        for i in 0..100 {
            layer.decide_for_layer(i, 0.9); // all Skip (> skip_threshold 0.5)
        }
        assert_eq!(layer.history().len(), 100);
        assert_eq!(layer.total_skipped_layers(), 100);
        assert_eq!(layer.total_decisions(), 100);
    }

    // ── GateFirstSkipLayer: decide_from_aggregator with various ratios ──

    #[test]
    fn decide_from_aggregator_zero_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.0 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: decide_from_page_header with boundary u8 values ──

    #[test]
    fn decide_from_page_header_zero_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.dead_ratio = 0; // 0/255 = 0.0
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_from_page_header_max_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.dead_ratio = 255; // 255/255 = 1.0
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    // ── GateFirstSkipLayer: total_decisions consistency ──

    #[test]
    fn total_decisions_equals_sum_of_categories() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.1);  // FullCompute
        layer.decide_for_layer(1, 0.3);  // MaskedCompute
        layer.decide_for_layer(2, 0.8);  // Skip
        layer.decide_for_layer(3, 0.05); // FullCompute

        assert_eq!(
            layer.total_decisions(),
            layer.total_full_layers() + layer.total_masked_layers() + layer.total_skipped_layers()
        );
        assert_eq!(layer.total_decisions(), 4);
        assert_eq!(layer.total_full_layers(), 2);
        assert_eq!(layer.total_masked_layers(), 1);
        assert_eq!(layer.total_skipped_layers(), 1);
    }

    // ── GateFirstSkipLayer: reset_stats clears counters but not config ──

    #[test]
    fn reset_stats_full_clear() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.7,
            dead_neuron_epsilon: 1e-4,
        };
        let mut layer = GateFirstSkipLayer::new(config, 16);
        layer.decide_for_layer(5, 0.8);
        layer.decide_for_layer(6, 0.3);
        layer.decide_for_layer(7, 0.1);

        layer.reset_stats();

        assert_eq!(layer.total_decisions(), 0);
        assert_eq!(layer.total_skipped_layers(), 0);
        assert_eq!(layer.total_masked_layers(), 0);
        assert_eq!(layer.total_full_layers(), 0);
        assert!(layer.history().is_empty());
        assert!(layer.is_enabled());
        assert_eq!(layer.num_layers(), 16);
        assert!((layer.config().skip_threshold - 0.7).abs() < 1e-6);
    }

    // ── BatchSkipSummary: large input ──

    #[test]
    fn batch_skip_summary_large_input() {
        let decisions = vec![GateSkipDecision::FullCompute; 999];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.per_request_decisions.len(), 999);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
    }

    // ── GateSkipDecision: PartialEq and Copy ──

    #[test]
    fn gate_skip_decision_equality() {
        assert_eq!(GateSkipDecision::FullCompute, GateSkipDecision::FullCompute);
        assert_eq!(GateSkipDecision::MaskedCompute, GateSkipDecision::MaskedCompute);
        assert_eq!(GateSkipDecision::Skip, GateSkipDecision::Skip);
        assert_ne!(GateSkipDecision::FullCompute, GateSkipDecision::MaskedCompute);
        assert_ne!(GateSkipDecision::MaskedCompute, GateSkipDecision::Skip);
    }

    #[test]
    fn gate_skip_decision_copy() {
        let a = GateSkipDecision::MaskedCompute;
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn gate_skip_decision_clone() {
        let a = GateSkipDecision::Skip;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── MechanismId boundary: all three decisions ──

    #[test]
    fn mechanism_id_full_compute_is_none() {
        let mapping: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::FullCompute.into();
        assert!(mapping.is_none());
    }

    // ── BatchSkipSummary: needs_masked_variant and needs_compact consistency ──

    #[test]
    fn full_compute_advice_neither_masked_nor_compact() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute; 10]);
        assert!(!summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    #[test]
    fn mostly_skip_advice_both_masked_and_compact() {
        // 3/4 skip → MostlySkip
        let summary = BatchSkipSummary::from_decisions(&[
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
        ]);
        assert!(summary.needs_masked_variant());
        assert!(summary.needs_compact());
    }

    // ── LayerSkipRecord: decision field can hold all variants ──

    #[test]
    fn layer_skip_record_decision_all_variants() {
        let full = LayerSkipRecord {
            layer_idx: 0,
            dead_neuron_ratio: 0.0,
            decision: GateSkipDecision::FullCompute,
        };
        let masked = LayerSkipRecord {
            layer_idx: 1,
            dead_neuron_ratio: 0.3,
            decision: GateSkipDecision::MaskedCompute,
        };
        let skip = LayerSkipRecord {
            layer_idx: 2,
            dead_neuron_ratio: 0.8,
            decision: GateSkipDecision::Skip,
        };
        assert_eq!(full.decision, GateSkipDecision::FullCompute);
        assert_eq!(masked.decision, GateSkipDecision::MaskedCompute);
        assert_eq!(skip.decision, GateSkipDecision::Skip);
    }

    // ════════════════════════════════════════════════════════════════
    //  NEW TESTS — targeting 30+ additions
    // ════════════════════════════════════════════════════════════════

    // ── GateFirstSkipLayer: multi-pass decide over same layers ──

    #[test]
    fn decide_two_passes_same_layer_accumulates_counts() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // Pass 1: layer 5 → FullCompute
        layer.decide_for_layer(5, 0.05);
        assert_eq!(layer.total_full_layers(), 1);

        // Pass 2: layer 5 → Skip (update)
        layer.decide_for_layer(5, 0.9);
        assert_eq!(layer.total_full_layers(), 1); // first decision still counted
        assert_eq!(layer.total_skipped_layers(), 1); // second decision counted

        // History still has 1 record (updated, not appended)
        assert_eq!(layer.history().len(), 1);
        let rec = layer.layer_record(5).unwrap();
        assert_eq!(rec.decision, GateSkipDecision::Skip);
    }

    #[test]
    fn decide_three_passes_same_layer_all_categories() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(3, 0.05); // FullCompute
        layer.decide_for_layer(3, 0.35); // MaskedCompute
        layer.decide_for_layer(3, 0.80); // Skip

        assert_eq!(layer.total_full_layers(), 1);
        assert_eq!(layer.total_masked_layers(), 1);
        assert_eq!(layer.total_skipped_layers(), 1);
        assert_eq!(layer.total_decisions(), 3);
        assert_eq!(layer.history().len(), 1);
    }

    // ── GateFirstSkipLayer: multiple distinct layers ──

    #[test]
    fn decide_many_distinct_layers_monotonic_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let ratios = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
        for (i, &r) in ratios.iter().enumerate() {
            layer.decide_for_layer(i, r);
        }
        assert_eq!(layer.history().len(), 10);
        assert_eq!(layer.total_decisions(), 10);

        // Verify skip_rate is in a reasonable range given these ratios
        let rate = layer.skip_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn decide_layer_zero_always_works() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // layer_idx = 0 should not be special — no min_skip_layer check in GateFirstSkipLayer
        let decision = layer.decide_for_layer(0, 0.9);
        assert_eq!(decision, GateSkipDecision::Skip);

        let rec = layer.layer_record(0);
        assert!(rec.is_some());
    }

    #[test]
    fn decide_layer_max_usize_works() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 1);

        let decision = layer.decide_for_layer(usize::MAX, 0.8);
        assert_eq!(decision, GateSkipDecision::Skip);
        assert!(layer.layer_record(usize::MAX).is_some());
    }

    // ── GateFirstSkipLayer: history update preserves insertion order ──

    #[test]
    fn history_insertion_order_preserved() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.1);
        layer.decide_for_layer(10, 0.3);
        layer.decide_for_layer(15, 0.8);

        let history = layer.history();
        assert_eq!(history[0].layer_idx, 5);
        assert_eq!(history[1].layer_idx, 10);
        assert_eq!(history[2].layer_idx, 15);
    }

    #[test]
    fn history_update_does_not_reorder() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.1);
        layer.decide_for_layer(10, 0.3);
        // Update layer 5 (should stay at position 0)
        layer.decide_for_layer(5, 0.8);

        let history = layer.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].layer_idx, 5);
        assert_eq!(history[1].layer_idx, 10);
        assert_eq!(history[0].decision, GateSkipDecision::Skip);
    }

    // ── GateFirstSkipLayer: recent_decisions edge cases ──

    #[test]
    fn recent_decisions_after_reset_returns_empty() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.6);
        layer.decide_for_layer(1, 0.3);
        layer.reset_stats();

        assert!(layer.recent_decisions(10).is_empty());
    }

    #[test]
    fn recent_decisions_returns_most_recent_first() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(2, 0.8);
        layer.decide_for_layer(5, 0.3);
        layer.decide_for_layer(8, 0.1);

        let recent = layer.recent_decisions(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].layer_idx, 8);
        assert_eq!(recent[1].layer_idx, 5);
    }

    // ── GateFirstSkipLayer: skip_rate with single category ──

    #[test]
    fn skip_rate_only_masked_is_one() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // 0.3 > 0.25 (=0.5*0.5) but <= 0.5 → MaskedCompute
        layer.decide_for_layer(0, 0.3);
        layer.decide_for_layer(1, 0.35);
        let rate = layer.skip_rate();
        assert!((rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn skip_rate_mixed_skip_and_masked_only() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.8); // Skip
        layer.decide_for_layer(1, 0.3); // MaskedCompute
        let rate = layer.skip_rate();
        // (1 skip + 1 masked) / 2 = 1.0
        assert!((rate - 1.0).abs() < 0.01);
    }

    // ── GateFirstSkipLayer: reset then re-decide ──

    #[test]
    fn reset_then_redecide_starts_fresh() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.8);
        assert_eq!(layer.total_skipped_layers(), 1);

        layer.reset_stats();
        assert_eq!(layer.total_decisions(), 0);

        layer.decide_for_layer(5, 0.1);
        assert_eq!(layer.total_full_layers(), 1);
        assert_eq!(layer.total_skipped_layers(), 0);
    }

    // ── GateFirstSkipLayer: disabled config ──

    #[test]
    fn disabled_config_all_ratios_yield_full_compute() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);

        for &r in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let decision = layer.decide_for_layer(0, r);
            assert_eq!(decision, GateSkipDecision::FullCompute,
                "expected FullCompute for ratio={r} when disabled");
        }
        assert_eq!(layer.total_full_layers(), 7);
        assert_eq!(layer.total_decisions(), 7);
    }

    #[test]
    fn disabled_config_preserves_history() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.9);
        let rec = layer.layer_record(5).unwrap();
        assert_eq!(rec.decision, GateSkipDecision::FullCompute);
        assert_eq!(layer.history().len(), 1);
    }

    // ── BatchSkipSummary: exact boundary for PartialMask vs MostlySkip ──

    #[test]
    fn batch_boundary_mostly_skip_exactly_half_plus_one() {
        // 6 skip, 4 full, total=10: skip_count*2=12 > 10 → MostlySkip
        let mut decisions = vec![GateSkipDecision::Skip; 6];
        decisions.extend(vec![GateSkipDecision::FullCompute; 4]);
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
    }

    #[test]
    fn batch_boundary_mostly_skip_exactly_half() {
        // 5 skip, 5 full, total=10: skip_count*2=10 = 10 → NOT MostlySkip (> is strict)
        // (skip+masked)*4 = (5+0)*4=20 > 10 → PartialMask
        let mut decisions = vec![GateSkipDecision::Skip; 5];
        decisions.extend(vec![GateSkipDecision::FullCompute; 5]);
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_boundary_partial_mask_exactly_quarter() {
        // 1 skip, 0 masked, 3 full, total=4: skip*2=2 ≤ 4; (skip+masked)*4=4 = 4 → NOT PartialMask (> is strict)
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
    }

    #[test]
    fn batch_boundary_partial_mask_just_above_quarter() {
        // 1 skip, 1 masked, 2 full, total=4: (1+1)*4=8 > 4 → PartialMask
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_all_masked_is_partial_mask() {
        let decisions = vec![GateSkipDecision::MaskedCompute; 8];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // 0 skip → not MostlySkip; (0+8)*4=32 > 8 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_single_full_compute_is_not_mostly_skip() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute]);
        // skip=0, masked=0, total=1 → FullCompute
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!(!summary.needs_masked_variant());
    }

    // ── BatchSkipSummary: avg_dead_neuron_ratio weighted correctness ──

    #[test]
    fn batch_avg_ratio_two_skip_two_full() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // (0.8*2 + 0.1*2) / 4 = 1.8/4 = 0.45
        let expected = (0.8 * 2.0 + 0.1 * 2.0) / 4.0;
        assert!((summary.avg_dead_neuron_ratio - expected).abs() < 0.01);
    }

    #[test]
    fn batch_avg_ratio_all_masked() {
        let decisions = vec![GateSkipDecision::MaskedCompute; 3];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!((summary.avg_dead_neuron_ratio - 0.35).abs() < 0.01);
    }

    // ── BatchSkipSummary: per_request_decisions order preserved ──

    #[test]
    fn batch_decisions_order_preserved() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::Skip,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.per_request_decisions[0], GateSkipDecision::Skip);
        assert_eq!(summary.per_request_decisions[1], GateSkipDecision::FullCompute);
        assert_eq!(summary.per_request_decisions[2], GateSkipDecision::MaskedCompute);
        assert_eq!(summary.per_request_decisions[3], GateSkipDecision::Skip);
    }

    // ── BatchSkipSummary: large input with mixed decisions ──

    #[test]
    fn batch_large_mixed_decisions() {
        // 600 skip, 300 full, 100 masked = 1000 total
        let mut decisions = vec![GateSkipDecision::Skip; 600];
        decisions.extend(vec![GateSkipDecision::FullCompute; 300]);
        decisions.extend(vec![GateSkipDecision::MaskedCompute; 100]);

        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.per_request_decisions.len(), 1000);
        // skip=600, skip*2=1200 > 1000 → MostlySkip
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.needs_compact());
    }

    // ── BatchSkipSummary: clone independence ──

    #[test]
    fn batch_summary_clone_independence() {
        let decisions = vec![
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        let cloned = summary.clone();

        // Verify fields match
        assert_eq!(summary.per_request_decisions, cloned.per_request_decisions);
        assert!((summary.avg_dead_neuron_ratio - cloned.avg_dead_neuron_ratio).abs() < 1e-10);
        assert_eq!(summary.batch_advice, cloned.batch_advice);
    }

    // ── KvPageHeader round-trip: f32 → u8 → f32 precision ──

    #[test]
    fn page_header_round_trip_zero() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.0);

        let decision = layer.decide_from_page_header(10, &header);
        // 0.0 should map to FullCompute
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn page_header_round_trip_high_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.9);

        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn page_header_round_trip_mid_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.35);

        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    // ── TelemetryAggregator: multiple signals ──

    #[test]
    fn decide_from_aggregator_multiple_ingests_last_wins() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();

        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.1 });
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.8 });

        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn decide_from_aggregator_records_in_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();

        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.4 });
        layer.decide_from_aggregator(7, &agg);

        let rec = layer.layer_record(7).unwrap();
        assert_eq!(rec.layer_idx, 7);
        assert!((rec.dead_neuron_ratio - 0.4).abs() < 0.01);
        assert_eq!(rec.decision, GateSkipDecision::MaskedCompute);
    }

    // ── GateSkipDecision → MechanismId: exhaustive ──

    #[test]
    fn mechanism_id_all_variants_consistency() {
        // FullCompute → None
        let fc: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::FullCompute.into();
        assert!(fc.is_none());

        // MaskedCompute → Some(GateFirstSkip)
        let mc: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::MaskedCompute.into();
        assert_eq!(mc, Some(super::super::variant_registry::MechanismId::GateFirstSkip));

        // Skip → Some(GateFirstSkip)
        let sk: Option<super::super::variant_registry::MechanismId> =
            GateSkipDecision::Skip.into();
        assert_eq!(sk, Some(super::super::variant_registry::MechanismId::GateFirstSkip));
    }

    // ── GateFirstSkipConfig: custom epsilon ──

    #[test]
    fn config_custom_epsilon_preserved() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.6,
            dead_neuron_epsilon: 0.005,
        };
        let layer = GateFirstSkipLayer::new(config, 16);
        assert!((layer.config().skip_threshold - 0.6).abs() < 1e-6);
        assert!((layer.config().dead_neuron_epsilon - 0.005).abs() < 1e-8);
    }

    // ── LayerSkipRecord: dead_neuron_ratio is f32 precision ──

    #[test]
    fn layer_skip_record_ratio_precision() {
        let record = LayerSkipRecord {
            layer_idx: 42,
            dead_neuron_ratio: 0.123456789,
            decision: GateSkipDecision::MaskedCompute,
        };
        // f32 has ~7 decimal digits of precision
        assert!((record.dead_neuron_ratio - 0.123456789_f32 as f32).abs() < 1e-6);
    }

    // ── BatchSkipAdvice: exhaustive variant checks ──

    #[test]
    fn batch_skip_advice_ordering() {
        // Verify Copy + PartialEq works correctly
        let variants = [
            BatchSkipAdvice::FullCompute,
            BatchSkipAdvice::PartialMask,
            BatchSkipAdvice::MostlySkip,
        ];
        // All distinct
        assert_ne!(variants[0], variants[1]);
        assert_ne!(variants[1], variants[2]);
        assert_ne!(variants[0], variants[2]);

        // Self-equality
        for v in &variants {
            assert_eq!(*v, *v);
        }
    }

    // ── GateFirstSkipLayer: disabled + reset interaction ──

    #[test]
    fn disabled_layer_reset_preserves_disabled() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 8);

        layer.decide_for_layer(0, 0.9);
        layer.reset_stats();

        assert!(!layer.is_enabled());
        assert_eq!(layer.total_decisions(), 0);
    }

    // ── GateFirstSkipLayer: config accessor returns correct defaults ──

    #[test]
    fn config_default_values_via_layer() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);

        assert!(layer.config().enabled);
        assert!((layer.config().skip_threshold - 0.5).abs() < 1e-6);
        assert!((layer.config().dead_neuron_epsilon - 1e-3).abs() < 1e-10);
    }

    // ── BatchSkipSummary: needs_masked_variant across all advice types ──

    #[test]
    fn needs_masked_variant_full_compute_is_false() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute; 4]);
        assert!(!summary.needs_masked_variant());
    }

    #[test]
    fn needs_masked_variant_partial_mask_is_true() {
        // 2 masked + 2 full, total=4: skip=0 → not MostlySkip
        // (0+2)*4=8 > 4 → PartialMask
        let summary = BatchSkipSummary::from_decisions(&[
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ]);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!(summary.needs_masked_variant());
    }

    #[test]
    fn needs_compact_only_mostly_skip() {
        // FullCompute → false
        let s1 = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute]);
        assert!(!s1.needs_compact());

        // PartialMask → false
        let s2 = BatchSkipSummary::from_decisions(&[GateSkipDecision::MaskedCompute]);
        assert!(!s2.needs_compact());

        // MostlySkip → true
        let s3 = BatchSkipSummary::from_decisions(&[GateSkipDecision::Skip]);
        assert!(s3.needs_compact());
    }

    // ── GateFirstSkipLayer: high num_layers value ──

    #[test]
    fn layer_large_num_layers() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 1000);
        assert_eq!(layer.num_layers(), 1000);
        assert_eq!(layer.total_decisions(), 0);
    }

    // ── GateFirstSkipLayer: decide exactly at masked_threshold boundary ──

    #[test]
    fn decide_exactly_at_masked_threshold() {
        // skip_threshold=0.5, masked_threshold=0.25 (=0.5*0.5)
        // dead_ratio=0.25: 0.25 > 0.25 is false → FullCompute (strict >)
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 0.25);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_just_above_masked_threshold() {
        // 0.251 > 0.25 → MaskedCompute
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 0.251);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    // ── GateFirstSkipLayer: decide_with_inf ──

    #[test]
    fn decide_inf_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, f32::INFINITY);
        // inf > 0.5 → Skip
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn decide_neg_inf_dead_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, f32::NEG_INFINITY);
        // -inf > anything is false → FullCompute
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── LayerSkipRecord: layer_idx distinct for each entry ──

    #[test]
    fn history_distinct_layer_indices() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(1, 0.1);
        layer.decide_for_layer(2, 0.2);
        layer.decide_for_layer(3, 0.3);

        let indices: Vec<usize> = layer.history().iter().map(|r| r.layer_idx).collect();
        assert_eq!(indices, vec![1, 2, 3]);
    }

    // ── GateFirstSkipLayer: decide_from_page_header records correctly ──

    #[test]
    fn page_header_decision_recorded_in_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.6);

        layer.decide_from_page_header(20, &header);

        let rec = layer.layer_record(20).unwrap();
        assert_eq!(rec.layer_idx, 20);
        assert_eq!(rec.decision, GateSkipDecision::Skip);
    }

    // ── BatchSkipSummary: empty decisions clone ──

    #[test]
    fn batch_empty_summary_clone() {
        let summary = BatchSkipSummary::from_decisions(&[]);
        let cloned = summary.clone();
        assert!(cloned.per_request_decisions.is_empty());
        assert_eq!(cloned.batch_advice, BatchSkipAdvice::FullCompute);
        assert!((cloned.avg_dead_neuron_ratio - 0.0).abs() < 1e-10);
    }

    // ── GateFirstSkipLayer: decide_from_aggregator with empty aggregator ──

    #[test]
    fn decide_from_empty_aggregator() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let agg = TelemetryAggregator::new();

        let decision = layer.decide_from_aggregator(10, &agg);
        // Empty aggregator returns 0.0 dead_neuron_ratio → FullCompute
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: total counters never decrease ──

    #[test]
    fn total_counters_monotonically_increase() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.1);  // FullCompute
        let total_after_1 = layer.total_decisions();

        layer.decide_for_layer(1, 0.3);  // MaskedCompute
        let total_after_2 = layer.total_decisions();

        layer.decide_for_layer(2, 0.8);  // Skip
        let total_after_3 = layer.total_decisions();

        assert!(total_after_2 > total_after_1);
        assert!(total_after_3 > total_after_2);
        assert_eq!(total_after_3, 3);
    }

    // ── GateFirstSkipLayer: history updated record has updated ratio ──

    #[test]
    fn history_update_reflects_latest_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(5, 0.1); // FullCompute
        layer.decide_for_layer(5, 0.6); // Skip
        layer.decide_for_layer(5, 0.05); // FullCompute again

        let rec = layer.layer_record(5).unwrap();
        assert!((rec.dead_neuron_ratio - 0.05).abs() < 0.001);
        assert_eq!(rec.decision, GateSkipDecision::FullCompute);

        // All 3 decisions counted
        assert_eq!(layer.total_decisions(), 3);
    }

    // ════════════════════════════════════════════════════════════════
    //  WAVE 2 — ~51 additional tests
    // ════════════════════════════════════════════════════════════════

    // ── GateFirstSkipConfig: extreme threshold values ──

    #[test]
    fn config_zero_skip_threshold_all_skip() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.0,
            dead_neuron_epsilon: 1e-3,
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        assert_eq!(layer.decide_for_layer(10, 0.01), GateSkipDecision::Skip);
        assert_eq!(layer.decide_for_layer(11, 0.001), GateSkipDecision::Skip);
    }

    #[test]
    fn config_unit_skip_threshold_no_skip() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 1.0,
            dead_neuron_epsilon: 1e-3,
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.99 > 1.0 false; 0.99 > 0.5 true → MaskedCompute
        assert_eq!(layer.decide_for_layer(10, 0.99), GateSkipDecision::MaskedCompute);
    }


    #[test]
    fn config_negative_skip_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: -0.5,
            dead_neuron_epsilon: 1e-3,
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        // 0.0 > -0.5 true → Skip
        assert_eq!(layer.decide_for_layer(10, 0.0), GateSkipDecision::Skip);
    }

    #[test]
    fn config_all_fields_zero() {
        let config = GateFirstSkipConfig {
            enabled: false,
            skip_threshold: 0.0,
            dead_neuron_epsilon: 0.0,
        };
        let mut layer = GateFirstSkipLayer::new(config, 8);
        assert_eq!(layer.decide_for_layer(0, 1.0), GateSkipDecision::FullCompute);
        assert!(!layer.is_enabled());
    }

    // ── decide_for_layer: f32 edge values ──

    #[test]
    fn decide_f32_min_positive() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, f32::MIN_POSITIVE);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_f32_max() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, f32::MAX);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn decide_subnormal_f32() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 1.0e-45_f32);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn decide_ratio_just_above_skip_threshold() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let decision = layer.decide_for_layer(10, 0.5001);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    // ── GateSkipDecision: Debug output ──

    #[test]
    fn gate_skip_decision_debug_all_variants() {
        assert!(format!("{:?}", GateSkipDecision::FullCompute).contains("FullCompute"));
        assert!(format!("{:?}", GateSkipDecision::MaskedCompute).contains("MaskedCompute"));
        assert!(format!("{:?}", GateSkipDecision::Skip).contains("Skip"));
    }

    #[test]
    fn layer_skip_record_debug_each_decision() {
        for (decision, name) in [
            (GateSkipDecision::FullCompute, "FullCompute"),
            (GateSkipDecision::MaskedCompute, "MaskedCompute"),
            (GateSkipDecision::Skip, "Skip"),
        ] {
            let record = LayerSkipRecord {
                layer_idx: 0,
                dead_neuron_ratio: 0.5,
                decision,
            };
            let debug = format!("{record:?}");
            assert!(debug.contains(name), "expected {name} in debug output");
        }
    }

    // ── GateFirstSkipLayer: recent_decisions edge cases ──

    #[test]
    fn recent_decisions_fresh_construction() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);
        assert!(layer.recent_decisions(5).is_empty());
    }

    #[test]
    fn recent_decisions_exact_history_size() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.1);
        layer.decide_for_layer(1, 0.3);
        layer.decide_for_layer(2, 0.8);
        let recent = layer.recent_decisions(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn recent_decisions_one_returns_last() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.1);
        layer.decide_for_layer(1, 0.3);
        layer.decide_for_layer(2, 0.8);
        let recent = layer.recent_decisions(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].layer_idx, 2);
    }

    // ── GateFirstSkipLayer: history edge cases ──

    #[test]
    fn history_empty_on_fresh_construction() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);
        assert!(layer.history().is_empty());
    }

    #[test]
    fn history_non_sequential_layer_order() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(7, 0.1);
        layer.decide_for_layer(2, 0.3);
        layer.decide_for_layer(15, 0.8);
        let history = layer.history();
        assert_eq!(history[0].layer_idx, 7);
        assert_eq!(history[1].layer_idx, 2);
        assert_eq!(history[2].layer_idx, 15);
    }

    #[test]
    fn layer_record_correct_among_many() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        for i in 0..10 {
            layer.decide_for_layer(i, 0.1);
        }
        let rec = layer.layer_record(5).unwrap();
        assert_eq!(rec.layer_idx, 5);
        assert!((rec.dead_neuron_ratio - 0.1).abs() < 1e-6);
        assert_eq!(rec.decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn history_count_matches_distinct_layers() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        for i in 0..10 {
            layer.decide_for_layer(i, 0.1);
        }
        layer.decide_for_layer(5, 0.3);
        layer.decide_for_layer(5, 0.8);
        assert_eq!(layer.history().len(), 10);
    }

    // ── GateFirstSkipLayer: double reset ──

    #[test]
    fn double_reset_no_panic() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.8);
        layer.reset_stats();
        layer.reset_stats();
        assert_eq!(layer.total_decisions(), 0);
        assert!(layer.history().is_empty());
    }

    // ── GateFirstSkipLayer: layer_idx > num_layers ──

    #[test]
    fn decide_layer_idx_exceeds_num_layers() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 4);
        let decision = layer.decide_for_layer(100, 0.8);
        assert_eq!(decision, GateSkipDecision::Skip);
        assert!(layer.layer_record(100).is_some());
    }

    // ── GateFirstSkipLayer: is_enabled default ──

    #[test]
    fn is_enabled_default_true() {
        let config = GateFirstSkipConfig::default();
        let layer = GateFirstSkipLayer::new(config, 32);
        assert!(layer.is_enabled());
    }

    // ── decide_from_page_header: specific u8 values ──

    #[test]
    fn page_header_u8_63_is_full_compute() {
        // 63/255 ≈ 0.247 < 0.25 → FullCompute
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 63;
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    #[test]
    fn page_header_u8_64_is_masked_compute() {
        // 64/255 ≈ 0.251 > 0.25 → MaskedCompute
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 64;
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn page_header_u8_128_is_skip() {
        // 128/255 ≈ 0.502 > 0.5 → Skip
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 128;
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn page_header_u8_192_is_skip() {
        // 192/255 ≈ 0.753 > 0.5 → Skip
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 192;
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn page_header_converted_ratio_in_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 128;
        layer.decide_from_page_header(10, &header);
        let rec = layer.layer_record(10).unwrap();
        let expected = 128.0_f32 / 255.0;
        assert!((rec.dead_neuron_ratio - expected).abs() < 0.01);
    }

    // ── decide_from_aggregator: various thresholds ──

    #[test]
    fn decide_from_aggregator_high_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.7 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::Skip);
    }

    #[test]
    fn decide_from_aggregator_mid_ratio() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.4 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::MaskedCompute);
    }

    // ── GateFirstSkipLayer: skip_rate property checks ──

    #[test]
    fn skip_rate_always_in_valid_range() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 64);
        for i in 0..50 {
            let ratio = (i as f32) / 100.0;
            layer.decide_for_layer(i, ratio);
        }
        let rate = layer.skip_rate();
        assert!(rate >= 0.0 && rate <= 1.0, "skip_rate {rate} out of [0,1]");
    }

    #[test]
    fn skip_rate_formula_matches_counts() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.1);  // FullCompute
        layer.decide_for_layer(1, 0.3);  // MaskedCompute
        layer.decide_for_layer(2, 0.8);  // Skip
        layer.decide_for_layer(3, 0.05); // FullCompute
        layer.decide_for_layer(4, 0.6);  // Skip
        let expected = (2.0_f32 + 1.0) / 5.0;
        assert!((layer.skip_rate() - expected).abs() < 0.01);
    }

    // ── GateFirstSkipLayer: history growth ──

    #[test]
    fn history_grows_with_each_new_layer() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        assert_eq!(layer.history().len(), 0);
        layer.decide_for_layer(0, 0.1);
        assert_eq!(layer.history().len(), 1);
        layer.decide_for_layer(1, 0.3);
        assert_eq!(layer.history().len(), 2);
    }

    #[test]
    fn history_interleaved_updates_preserve_count() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.1);
        layer.decide_for_layer(1, 0.3);
        layer.decide_for_layer(0, 0.8);
        layer.decide_for_layer(2, 0.05);
        assert_eq!(layer.history().len(), 3);
    }

    // ── GateFirstSkipLayer: total_decisions after complex sequence ──

    #[test]
    fn total_decisions_complex_update_sequence() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.1);
        layer.decide_for_layer(1, 0.3);
        layer.decide_for_layer(2, 0.8);
        layer.decide_for_layer(0, 0.9);
        layer.decide_for_layer(1, 0.05);
        assert_eq!(layer.total_decisions(), 5);
    }

    // ── GateFirstSkipLayer: reset then complex re-use ──

    #[test]
    fn reset_then_complex_reuse() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(0, 0.8);
        layer.decide_for_layer(1, 0.3);
        layer.reset_stats();
        layer.decide_for_layer(5, 0.1);
        assert_eq!(layer.history().len(), 1);
        assert_eq!(layer.total_full_layers(), 1);
        assert!(layer.layer_record(0).is_none());
    }

    // ── BatchSkipSummary: combinations of two decision types ──

    #[test]
    fn batch_skip_plus_masked_no_full() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::MaskedCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // skip*2=4 = 4 → NOT MostlySkip; (2+2)*4=16 > 4 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_skip_plus_full_no_masked() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // skip*2=4 = 4 → NOT MostlySkip; (2+0)*4=8 > 4 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_masked_plus_full_no_skip() {
        let decisions = vec![
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        // (0+2)*4=8 > 4 → PartialMask
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    // ── BatchSkipSummary: specific avg ratio calculations ──

    #[test]
    fn batch_avg_ratio_single_masked() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::MaskedCompute]);
        assert!((summary.avg_dead_neuron_ratio - 0.35).abs() < 0.01);
    }

    #[test]
    fn batch_avg_ratio_two_of_each() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        let expected = (0.8 * 2.0 + 0.35 * 2.0 + 0.1 * 2.0) / 6.0;
        assert!((summary.avg_dead_neuron_ratio - expected).abs() < 0.01);
    }

    #[test]
    fn batch_avg_ratio_in_valid_range() {
        let cases: Vec<Vec<GateSkipDecision>> = vec![
            vec![GateSkipDecision::FullCompute],
            vec![GateSkipDecision::MaskedCompute],
            vec![GateSkipDecision::Skip],
            vec![GateSkipDecision::Skip, GateSkipDecision::FullCompute],
            vec![GateSkipDecision::MaskedCompute, GateSkipDecision::FullCompute],
        ];
        for decisions in &cases {
            let summary = BatchSkipSummary::from_decisions(decisions);
            assert!(
                summary.avg_dead_neuron_ratio >= 0.0 && summary.avg_dead_neuron_ratio <= 1.0,
                "avg_ratio {} out of [0,1] for {:?}",
                summary.avg_dead_neuron_ratio,
                decisions,
            );
        }
    }

    // ── BatchSkipSummary: per_request_decisions independence ──

    #[test]
    fn batch_per_request_vec_deep_copy() {
        let decisions = vec![GateSkipDecision::FullCompute, GateSkipDecision::Skip];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        let mut copied = summary.per_request_decisions.clone();
        copied[0] = GateSkipDecision::Skip;
        assert_eq!(summary.per_request_decisions[0], GateSkipDecision::FullCompute);
    }

    // ── BatchSkipAdvice: Debug output for each variant ──

    #[test]
    fn batch_advice_debug_full_compute() {
        assert!(format!("{:?}", BatchSkipAdvice::FullCompute).contains("FullCompute"));
    }

    #[test]
    fn batch_advice_debug_partial_mask() {
        assert!(format!("{:?}", BatchSkipAdvice::PartialMask).contains("PartialMask"));
    }

    #[test]
    fn batch_advice_debug_mostly_skip() {
        assert!(format!("{:?}", BatchSkipAdvice::MostlySkip).contains("MostlySkip"));
    }

    // ── BatchSkipSummary: from_decisions with two same-type elements ──

    #[test]
    fn batch_two_skip_is_mostly_skip() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::Skip; 2]);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
    }

    #[test]
    fn batch_two_masked_is_partial_mask() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::MaskedCompute; 2]);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn batch_two_full_is_full_compute() {
        let summary = BatchSkipSummary::from_decisions(&[GateSkipDecision::FullCompute; 2]);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
    }

    // ── GateFirstSkipLayer: config() consistency ──

    #[test]
    fn config_accessor_returns_same_values() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.65,
            dead_neuron_epsilon: 0.002,
        };
        let layer = GateFirstSkipLayer::new(config, 16);
        let c1 = layer.config();
        let c2 = layer.config();
        assert_eq!(c1.enabled, c2.enabled);
        assert!((c1.skip_threshold - c2.skip_threshold).abs() < 1e-10);
        assert!((c1.dead_neuron_epsilon - c2.dead_neuron_epsilon).abs() < 1e-10);
    }

    // ── GateFirstSkipLayer: layer_record after many updates ──

    #[test]
    fn layer_record_latest_after_many_updates() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);
        layer.decide_for_layer(3, 0.1);
        layer.decide_for_layer(3, 0.8);
        layer.decide_for_layer(3, 0.3);
        layer.decide_for_layer(3, 0.95);
        layer.decide_for_layer(3, 0.05);
        let rec = layer.layer_record(3).unwrap();
        assert!((rec.dead_neuron_ratio - 0.05).abs() < 0.001);
        assert_eq!(rec.decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: disabled config with decide_from_page_header ──

    #[test]
    fn disabled_decide_from_page_header() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 255;
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: disabled config with decide_from_aggregator ──

    #[test]
    fn disabled_decide_from_aggregator() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.9 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, GateSkipDecision::FullCompute);
    }

    // ════════════════════════════════════════════════════════════════
    //  WAVE 3 — 13 additional tests (172 → 185)
    // ════════════════════════════════════════════════════════════════

    // ── GateFirstSkipLayer: mixed decide paths (for_layer + from_aggregator) ──

    #[test]
    fn mixed_decide_paths_share_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // Path 1: direct decide_for_layer
        layer.decide_for_layer(5, 0.8);

        // Path 2: decide_from_aggregator
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.3 });
        layer.decide_from_aggregator(10, &agg);

        // Path 3: decide_from_page_header
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.1);
        layer.decide_from_page_header(15, &header);

        // All three paths write to the same history
        assert_eq!(layer.history().len(), 3);
        assert_eq!(layer.total_decisions(), 3);
        assert_eq!(layer.layer_record(5).unwrap().decision, GateSkipDecision::Skip);
        assert_eq!(layer.layer_record(10).unwrap().decision, GateSkipDecision::MaskedCompute);
        assert_eq!(layer.layer_record(15).unwrap().decision, GateSkipDecision::FullCompute);
    }

    // ── GateFirstSkipLayer: skip_rate after reset and partial refill ──

    #[test]
    fn skip_rate_after_partial_refill() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // Fill with all Skip
        layer.decide_for_layer(0, 0.8);
        layer.decide_for_layer(1, 0.9);

        layer.reset_stats();

        // Refill with all FullCompute
        layer.decide_for_layer(5, 0.05);
        layer.decide_for_layer(6, 0.02);

        // Only FullCompute decisions after reset → skip_rate = 0
        assert_eq!(layer.skip_rate(), 0.0);
        assert_eq!(layer.total_full_layers(), 2);
    }

    // ── GateFirstSkipLayer: total counters after reset reflect only new decisions ──

    #[test]
    fn counters_after_reset_reflect_only_new() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.8); // Skip
        layer.decide_for_layer(1, 0.3); // MaskedCompute
        layer.decide_for_layer(2, 0.1); // FullCompute
        assert_eq!(layer.total_decisions(), 3);

        layer.reset_stats();
        assert_eq!(layer.total_skipped_layers(), 0);
        assert_eq!(layer.total_masked_layers(), 0);
        assert_eq!(layer.total_full_layers(), 0);

        layer.decide_for_layer(10, 0.9); // Skip
        assert_eq!(layer.total_skipped_layers(), 1);
        assert_eq!(layer.total_masked_layers(), 0);
        assert_eq!(layer.total_full_layers(), 0);
        assert_eq!(layer.total_decisions(), 1);
    }

    // ── LayerSkipRecord: Debug output contains all fields ──

    #[test]
    fn layer_skip_record_debug_contains_all_fields() {
        let record = LayerSkipRecord {
            layer_idx: 42,
            dead_neuron_ratio: 0.65,
            decision: GateSkipDecision::MaskedCompute,
        };
        let debug = format!("{record:?}");
        assert!(debug.contains("42"), "expected layer_idx 42 in debug output");
        assert!(debug.contains("MaskedCompute"), "expected decision in debug output");
    }

    // ── GateFirstSkipLayer: decide_from_aggregator followed by page_header for same layer ──

    #[test]
    fn aggregator_then_page_header_same_layer_keeps_latest() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // First: aggregator says high ratio → Skip
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.9 });
        layer.decide_from_aggregator(7, &agg);

        // Then: page_header says low ratio → FullCompute
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.05);
        layer.decide_from_page_header(7, &header);

        // Latest decision wins in the history
        let rec = layer.layer_record(7).unwrap();
        assert_eq!(rec.decision, GateSkipDecision::FullCompute);
        assert_eq!(layer.history().len(), 1);
        // Both decisions counted in totals
        assert_eq!(layer.total_decisions(), 2);
    }

    // ── GateFirstSkipLayer: very small epsilon does not affect threshold logic ──

    #[test]
    fn tiny_epsilon_no_effect_on_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.5,
            dead_neuron_epsilon: 1e-38, // extremely small epsilon
        };
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // 0.5 is NOT > 0.5 → MaskedCompute (epsilon does not shift the boundary)
        assert_eq!(layer.decide_for_layer(10, 0.5), GateSkipDecision::MaskedCompute);
        // 0.5001 > 0.5 → Skip
        assert_eq!(layer.decide_for_layer(11, 0.5001), GateSkipDecision::Skip);
    }

    // ── BatchSkipSummary: from_decisions with all three types in specific proportion ──

    #[test]
    fn batch_one_third_each_type() {
        // 1 skip, 1 masked, 1 full → total=3
        // skip*2=2 ≤ 3 → not MostlySkip
        // (1+1)*4=8 > 3 → PartialMask
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        let expected_avg = (0.8 + 0.35 + 0.1) / 3.0;
        assert!((summary.avg_dead_neuron_ratio - expected_avg).abs() < 0.01);
    }

    // ── GateFirstSkipLayer: history remains stable across many updates to same layer ──

    #[test]
    fn history_stable_many_updates_one_layer() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // Add other layers first
        layer.decide_for_layer(0, 0.1);
        layer.decide_for_layer(1, 0.3);

        // Update layer 5 many times
        for i in 0..50 {
            let ratio = (i as f32) / 100.0;
            layer.decide_for_layer(5, ratio);
        }

        // History still has exactly 3 distinct layers
        assert_eq!(layer.history().len(), 3);
        // Total decisions = 2 + 50 = 52
        assert_eq!(layer.total_decisions(), 52);
    }

    // ── GateFirstSkipLayer: decide_for_layer records dead_ratio as given ──

    #[test]
    fn decide_records_exact_ratio_in_history() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let specific_ratio = 0.3721_f32;
        layer.decide_for_layer(10, specific_ratio);

        let rec = layer.layer_record(10).unwrap();
        assert!((rec.dead_neuron_ratio - specific_ratio).abs() < 1e-6);
    }

    // ── GateFirstSkipLayer: all three decide paths produce valid GateSkipDecision ──

    #[test]
    fn all_decide_paths_produce_valid_decision() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        let d1 = layer.decide_for_layer(0, 0.3);
        let valid_variant = matches!(
            d1,
            GateSkipDecision::FullCompute | GateSkipDecision::MaskedCompute | GateSkipDecision::Skip
        );
        assert!(valid_variant);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.6 });
        let d2 = layer.decide_from_aggregator(1, &agg);
        assert!(matches!(d2, GateSkipDecision::FullCompute | GateSkipDecision::MaskedCompute | GateSkipDecision::Skip));

        let mut header = KvPageHeader::new(0);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.1);
        let d3 = layer.decide_from_page_header(2, &header);
        assert!(matches!(d3, GateSkipDecision::FullCompute | GateSkipDecision::MaskedCompute | GateSkipDecision::Skip));
    }

    // ── GateFirstSkipLayer: skip_rate is zero when only full_compute after reset ──

    #[test]
    fn skip_rate_zero_after_reset_with_only_full() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        // First batch: all Skip
        for i in 0..10 {
            layer.decide_for_layer(i, 0.9);
        }
        assert!((layer.skip_rate() - 1.0).abs() < 0.01);

        layer.reset_stats();

        // Second batch: all FullCompute
        for i in 0..5 {
            layer.decide_for_layer(i + 20, 0.01);
        }
        assert_eq!(layer.skip_rate(), 0.0);
        assert_eq!(layer.total_full_layers(), 5);
    }

    // ── BatchSkipSummary: from_decisions with odd number of elements boundary ──

    #[test]
    fn batch_odd_count_boundary_mostly_skip() {
        // 3 skip, 2 full, total=5: skip*2=6 > 5 → MostlySkip
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.needs_compact());
    }

    // ── GateFirstSkipLayer: multiple resets interleaved with decisions ──

    #[test]
    fn multiple_resets_interleaved() {
        let config = GateFirstSkipConfig::default();
        let mut layer = GateFirstSkipLayer::new(config, 32);

        layer.decide_for_layer(0, 0.8);
        assert_eq!(layer.total_decisions(), 1);

        layer.reset_stats();
        assert_eq!(layer.total_decisions(), 0);

        layer.decide_for_layer(1, 0.3);
        assert_eq!(layer.total_masked_layers(), 1);

        layer.reset_stats();
        assert_eq!(layer.total_decisions(), 0);
        assert!(layer.history().is_empty());

        // Config still intact after multiple resets
        assert!(layer.is_enabled());
        assert_eq!(layer.num_layers(), 32);
    }
}
