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
}
