//! Residual Bypass Integration — 层跳过决策 (SPEC §13.3, §13.11)
//!
//! ## 核心职责
//! 将 ResidualBypassDetector (决策层) 与层跳过管线集成:
//! - 从 TelemetryAggregator / KvPageHeader 读取残差信号
//! - 做出 BypassDecision (Execute / Bypass)
//! - 为图构建器提供 Residual Bypass 配置
//! - 跟踪逐层决策历史
//!
//! ## 数据流
//! ```
//! KvPageHeader.residual_delta_rho + residual_cosine ──→ ResidualBypassDetector.decide()
//!                                                            ↓
//!                                                      BypassDecision
//!                                                   ┌──────────┴──────────┐
//!                                                   ↓                    ↓
//!                                              Execute               Bypass
//!                                             (正常执行该层)         (输入直通输出)
//! ```
//!
//! ## 与 Early-Exit 联动
//! - Δρ 接近 1.0 且 cos(θ) > 0.99 → 高置信度跳过
//! - 跳过后的层直接进入下一层，无需计算
//! - 跳过决策写入 KvPageHeader，供后续层参考

use super::epilogue::{ResidualBypassConfig, ResidualBypassDecision, ResidualBypassDetector, TelemetryAggregator};
use crate::kv_cache::{KvPageHeader, f16_bits_to_f32};

/// 逐层 Residual Bypass 决策记录
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBypassRecord {
    /// 层索引
    pub layer_idx: usize,
    /// 该层的残差能量差 Δρ
    pub delta_rho: f32,
    /// 该层的残差方向余弦 cos(θ)
    pub cosine: f32,
    /// 决策结果
    pub decision: ResidualBypassDecision,
}

/// Residual Bypass 层级集成器
///
/// 管理多层残差旁路的决策流程:
/// - 维护 ResidualBypassDetector 实例
/// - 从 TelemetryAggregator / KvPageHeader 提取信号
/// - 逐层做出跳过决策
/// - 记录决策历史
#[allow(dead_code)]
pub struct ResidualBypassLayer {
    /// 决策器
    detector: ResidualBypassDetector,
    /// 逐层决策记录 (layer_idx → record)
    history: Vec<LayerBypassRecord>,
    /// 总层数 (从模型配置获取)
    num_layers: usize,
    /// 累计跳过的层数
    total_bypassed_layers: usize,
    /// 累计执行的层数
    total_executed_layers: usize,
}

impl ResidualBypassLayer {
    /// 创建新的 Residual Bypass 层级集成器
    pub fn new(config: ResidualBypassConfig, num_layers: usize) -> Self {
        Self {
            detector: ResidualBypassDetector::new(config),
            history: Vec::with_capacity(num_layers),
            num_layers,
            total_bypassed_layers: 0,
            total_executed_layers: 0,
        }
    }

    /// 从 Δρ 和 cos(θ) 做出当前层的跳过决策
    pub fn decide_for_layer(
        &mut self,
        layer_idx: usize,
        delta_rho: f32,
        cosine: f32,
    ) -> ResidualBypassDecision {
        let decision = self.detector.decide(layer_idx, delta_rho, cosine);
        self.record_decision(layer_idx, delta_rho, cosine, decision);
        decision
    }

    /// 从 TelemetryAggregator 做出当前层的跳过决策
    pub fn decide_from_aggregator(
        &mut self,
        layer_idx: usize,
        agg: &TelemetryAggregator,
    ) -> ResidualBypassDecision {
        let decision = self.detector.decide_from_telemetry(layer_idx, agg);
        self.record_decision(
            layer_idx,
            agg.residual_delta_rho(),
            agg.residual_cosine(),
            decision,
        );
        decision
    }

    /// 从 KvPageHeader 做出当前层的跳过决策
    ///
    /// KvPageHeader 由 Epilogue STG 指令写入 (§9.5)，
    /// 包含上一层的残差信号。
    ///
    /// 注意: KvPageHeader 不包含 residual_cosine 字段，
    /// 此方法仅使用 delta_rho 做决策 (cosine 默认为 0.99)。
    /// 完整的余弦相似度决策应使用 `decide_from_aggregator()`。
    pub fn decide_from_page_header(
        &mut self,
        layer_idx: usize,
        header: &KvPageHeader,
    ) -> ResidualBypassDecision {
        // KvPageHeader 没有 residual_cosine 字段，使用保守的默认值
        let cosine = 0.99;  // 假设方向一致
        let delta_rho = f16_bits_to_f32(header.delta_rho_avg);
        let decision = self.detector.decide(
            layer_idx,
            delta_rho,
            cosine,
        );
        self.record_decision(
            layer_idx,
            delta_rho,
            cosine,
            decision,
        );
        decision
    }

    /// 记录决策结果
    fn record_decision(
        &mut self,
        layer_idx: usize,
        delta_rho: f32,
        cosine: f32,
        decision: ResidualBypassDecision,
    ) {
        match decision {
            ResidualBypassDecision::Execute => {
                self.total_executed_layers += 1;
            }
            ResidualBypassDecision::Bypass => {
                self.total_bypassed_layers += 1;
            }
        }

        self.history.push(LayerBypassRecord {
            layer_idx,
            delta_rho,
            cosine,
            decision,
        });
    }

    /// 获取历史记录
    pub fn history(&self) -> &[LayerBypassRecord] {
        &self.history
    }

    /// 获取跳过统计
    pub fn stats(&self) -> BypassStats {
        BypassStats {
            total: self.history.len(),
            bypassed: self.total_bypassed_layers,
            executed: self.total_executed_layers,
        }
    }

    /// 重置状态
    pub fn reset(&mut self) {
        self.history.clear();
        self.total_bypassed_layers = 0;
        self.total_executed_layers = 0;
    }
}

/// Bypass 统计
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BypassStats {
    /// 总决策次数
    pub total: usize,
    /// 跳过的层数
    pub bypassed: usize,
    /// 执行的层数
    pub executed: usize,
}

/// 批次级 Bypass 汇总
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BypassSummary {
    /// 批次中的所有 Bypass 建议
    pub advices: Vec<BypassAdvice>,
}

/// 单个请求的 Bypass 建议
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BypassAdvice {
    /// 执行该层
    Execute,
    /// 跳过该层
    Bypass,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::EpilogueSignal;

    #[test]
    fn test_bypass_decision() {
        let config = ResidualBypassConfig::default();
        let mut bypass = ResidualBypassLayer::new(config, 32);

        // 高置信度跳过: Δρ ≈ 1.0 且 cos(θ) > 0.99
        let decision = bypass.decide_for_layer(10, 1.0005, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Bypass);

        // 不跳过: Δρ 偏差大
        let decision = bypass.decide_for_layer(11, 1.05, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Execute);

        // 不跳过: 方向偏移大
        let decision = bypass.decide_for_layer(12, 1.0005, 0.95);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_min_layer_constraint() {
        let config = ResidualBypassConfig::default();
        let mut bypass = ResidualBypassLayer::new(config, 32);

        // 前4层不允许跳过
        for layer in 0..4 {
            let decision = bypass.decide_for_layer(layer, 1.0005, 0.995);
            assert_eq!(decision, ResidualBypassDecision::Execute);
        }

        // 第5层可以跳过
        let decision = bypass.decide_for_layer(4, 1.0005, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    #[test]
    fn test_bypass_disabled() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let mut bypass = ResidualBypassLayer::new(config, 32);

        let decision = bypass.decide_for_layer(10, 1.0005, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_bypass_stats() {
        let config = ResidualBypassConfig::default();
        let mut bypass = ResidualBypassLayer::new(config, 32);

        bypass.decide_for_layer(5, 1.0005, 0.995);  // Bypass
        bypass.decide_for_layer(6, 1.05, 0.995);      // Execute
        bypass.decide_for_layer(7, 1.0005, 0.95);     // Execute

        let stats = bypass.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.bypassed, 1);
        assert_eq!(stats.executed, 2);
    }

    #[test]
    fn test_reset() {
        let config = ResidualBypassConfig::default();
        let mut bypass = ResidualBypassLayer::new(config, 32);

        bypass.decide_for_layer(5, 1.0005, 0.995);
        assert_eq!(bypass.history().len(), 1);

        bypass.reset();
        assert_eq!(bypass.history().len(), 0);
        assert_eq!(bypass.stats().total, 0);
    }

    #[test]
    fn layer_bypass_record_fields() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);

        layer.decide_for_layer(5, 0.98, 0.99);
        let rec = &layer.history()[0];
        assert_eq!(rec.layer_idx, 5);
        assert!((rec.delta_rho - 0.98).abs() < 1e-6);
        assert!((rec.cosine - 0.99).abs() < 1e-6);
        assert_eq!(rec.decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn bypass_stats_default() {
        let stats = BypassStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.bypassed, 0);
        assert_eq!(stats.executed, 0);
    }

    #[test]
    fn bypass_stats_clone_copy() {
        let stats = BypassStats { total: 10, bypassed: 3, executed: 7 };
        let copy = stats;
        assert_eq!(copy.total, 10);
        assert_eq!(copy.bypassed, 3);
    }

    #[test]
    fn bypass_summary_default() {
        let summary = BypassSummary::default();
        assert!(summary.advices.is_empty());
    }

    #[test]
    fn bypass_advice_variants() {
        assert_eq!(BypassAdvice::Execute, BypassAdvice::Execute);
        assert_ne!(BypassAdvice::Execute, BypassAdvice::Bypass);
    }

    #[test]
    fn decide_from_page_header_bypass() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        let mut header = KvPageHeader::default();
        // Δρ close to 1.0 — use exactly 1.0 which survives f16 round-trip
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0);

        // cos(θ)=0.99 (default in decide_from_page_header) + Δρ=1.0
        // cosine_threshold=0.99 requires cosine > 0.99, so default 0.99 is NOT enough.
        // This test verifies the page header path correctly applies the decision logic.
        let decision = layer.decide_from_page_header(6, &header);
        // cos=0.99 is not > 0.99, so Execute is the correct result
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn decide_from_page_header_execute() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.1);

        let decision = layer.decide_from_page_header(6, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn history_preserves_order() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        layer.decide_for_layer(4, 1.0005, 0.995);
        layer.decide_for_layer(5, 1.05, 0.99);
        layer.decide_for_layer(6, 1.0005, 0.995);

        let h = layer.history();
        assert_eq!(h.len(), 3);
        assert_eq!(h[0].layer_idx, 4);
        assert_eq!(h[1].layer_idx, 5);
        assert_eq!(h[2].layer_idx, 6);
    }

    #[test]
    fn stats_after_mixed_decisions() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 32);

        layer.decide_for_layer(4, 1.0005, 0.995); // bypass
        layer.decide_for_layer(5, 1.05, 0.99);     // execute
        layer.decide_for_layer(6, 1.0005, 0.995); // bypass
        layer.decide_for_layer(7, 1.2, 0.98);     // execute

        let stats = layer.stats();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.bypassed, 2);
        assert_eq!(stats.executed, 2);
    }

    #[test]
    fn reset_clears_counters() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 32);

        layer.decide_for_layer(5, 1.0005, 0.995);
        layer.decide_for_layer(6, 1.05, 0.99);
        assert_eq!(layer.stats().bypassed + layer.stats().executed, 2);

        layer.reset();
        assert_eq!(layer.stats().bypassed, 0);
        assert_eq!(layer.stats().executed, 0);
    }

    #[test]
    fn page_header_min_layer_enforced() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0005);

        for i in 0..4 {
            let d = layer.decide_from_page_header(i, &header);
            assert_eq!(d, ResidualBypassDecision::Execute, "layer {} should be forced execute", i);
        }
    }

    #[test]
    fn debug_format_bypass_stats() {
        let stats = BypassStats { total: 5, bypassed: 2, executed: 3 };
        let s = format!("{:?}", stats);
        assert!(s.contains("total"));
        assert!(s.contains("5"));
    }

    // ── 新增测试 ──────────────────────────────────────────────────────

    // 1. LayerBypassRecord PartialEq 验证
    #[test]
    fn layer_bypass_record_partial_eq_equal() {
        let a = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0005,
            cosine: 0.995,
            decision: ResidualBypassDecision::Bypass,
        };
        let b = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0005,
            cosine: 0.995,
            decision: ResidualBypassDecision::Bypass,
        };
        assert_eq!(a, b);
    }

    // 2. LayerBypassRecord PartialEq 不相等字段
    #[test]
    fn layer_bypass_record_partial_eq_different_decision() {
        let a = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0005,
            cosine: 0.995,
            decision: ResidualBypassDecision::Bypass,
        };
        let b = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0005,
            cosine: 0.995,
            decision: ResidualBypassDecision::Execute,
        };
        assert_ne!(a, b);
    }

    // 3. LayerBypassRecord Clone 独立性
    #[test]
    fn layer_bypass_record_clone_independence() {
        let rec = LayerBypassRecord {
            layer_idx: 7,
            delta_rho: 1.05,
            cosine: 0.88,
            decision: ResidualBypassDecision::Execute,
        };
        let cloned = rec.clone();
        assert_eq!(rec, cloned);
    }

    // 4. LayerBypassRecord Debug 输出包含关键字段
    #[test]
    fn layer_bypass_record_debug_format() {
        let rec = LayerBypassRecord {
            layer_idx: 5,
            delta_rho: 1.0005,
            cosine: 0.995,
            decision: ResidualBypassDecision::Bypass,
        };
        let s = format!("{:?}", rec);
        assert!(s.contains("layer_idx"));
        assert!(s.contains("delta_rho"));
        assert!(s.contains("cosine"));
        assert!(s.contains("decision"));
    }

    // 5. BypassStats PartialEq 相等与不等
    #[test]
    fn bypass_stats_partial_eq() {
        let a = BypassStats { total: 10, bypassed: 3, executed: 7 };
        let b = BypassStats { total: 10, bypassed: 3, executed: 7 };
        assert_eq!(a, b);

        let c = BypassStats { total: 10, bypassed: 4, executed: 6 };
        assert_ne!(a, c);
    }

    // 6. BypassStats 零值边界
    #[test]
    fn bypass_stats_zero_values() {
        let stats = BypassStats { total: 0, bypassed: 0, executed: 0 };
        assert_eq!(stats, BypassStats::default());
        assert_eq!(stats.total + stats.bypassed + stats.executed, 0);
    }

    // 7. BypassStats Copy 语义 — 修改 copy 不影响原值
    #[test]
    fn bypass_stats_copy_semantics() {
        let original = BypassStats { total: 8, bypassed: 5, executed: 3 };
        let copy = original;
        // original 仍然有效（Copy trait）
        assert_eq!(original.total, 8);
        assert_eq!(copy.total, 8);
    }

    // 8. BypassSummary PartialEq 和空集合
    #[test]
    fn bypass_summary_partial_eq_empty() {
        let a = BypassSummary::default();
        let b = BypassSummary { advices: vec![] };
        assert_eq!(a, b);
    }

    // 9. BypassSummary 非空集合 PartialEq
    #[test]
    fn bypass_summary_partial_eq_non_empty() {
        let a = BypassSummary { advices: vec![BypassAdvice::Execute, BypassAdvice::Bypass] };
        let b = BypassSummary { advices: vec![BypassAdvice::Execute, BypassAdvice::Bypass] };
        assert_eq!(a, b);

        let c = BypassSummary { advices: vec![BypassAdvice::Bypass, BypassAdvice::Execute] };
        assert_ne!(a, c);
    }

    // 10. BypassAdvice Hash — 可用作 HashSet key
    #[test]
    fn bypass_advice_hash_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BypassAdvice::Execute);
        set.insert(BypassAdvice::Bypass);
        set.insert(BypassAdvice::Execute); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&BypassAdvice::Execute));
        assert!(set.contains(&BypassAdvice::Bypass));
    }

    // 11. BypassAdvice Debug 输出
    #[test]
    fn bypass_advice_debug_format() {
        let s_execute = format!("{:?}", BypassAdvice::Execute);
        let s_bypass = format!("{:?}", BypassAdvice::Bypass);
        assert!(s_execute.contains("Execute"));
        assert!(s_bypass.contains("Bypass"));
    }

    // 12. decide_from_aggregator — Execute 路径 (Δρ 偏差大)
    #[test]
    fn decide_from_aggregator_execute_on_large_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.5 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 13. decide_from_aggregator — min_layer 约束
    #[test]
    fn decide_from_aggregator_min_layer_enforced() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        // 前4层必须 Execute，即使信号满足 Bypass 条件
        for i in 0..4 {
            let d = layer.decide_from_aggregator(i, &agg);
            assert_eq!(d, ResidualBypassDecision::Execute, "layer {} must be forced execute", i);
        }
    }

    // 14. decide_from_aggregator — Bypass 路径 (信号充分)
    #[test]
    fn decide_from_aggregator_bypass_on_stable_signal() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 15. decide_from_aggregator — 记录写入 history
    #[test]
    fn decide_from_aggregator_records_history() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.05 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.88 });

        layer.decide_from_aggregator(5, &agg);

        assert_eq!(layer.history().len(), 1);
        let rec = &layer.history()[0];
        assert_eq!(rec.layer_idx, 5);
        assert!((rec.delta_rho - 1.05).abs() < 1e-6);
        assert!((rec.cosine - 0.88).abs() < 1e-6);
    }

    // 16. decide_from_page_header — 记录写入 history
    #[test]
    fn decide_from_page_header_records_history() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.1);

        layer.decide_from_page_header(5, &header);

        assert_eq!(layer.history().len(), 1);
        let rec = &layer.history()[0];
        assert_eq!(rec.layer_idx, 5);
        // cosine 应为 0.99（page header 默认值）
        assert!((rec.cosine - 0.99).abs() < 1e-6);
    }

    // 17. 自定义 min_skip_layer=0 — 所有层可跳过
    #[test]
    fn custom_min_skip_layer_zero_allows_all_layers() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);

        // 层 0 也应能 Bypass
        let decision = layer.decide_for_layer(0, 1.0005, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 18. 自定义宽松阈值 — 更多层被跳过
    #[test]
    fn custom_loose_threshold_more_bypass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.1,   // 非常宽松
            cosine_threshold: 0.5,      // 非常宽松
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);

        // Δρ=1.05, cos=0.6 — 在默认配置下会 Execute，但在宽松配置下 Bypass
        let decision = layer.decide_for_layer(0, 1.05, 0.6);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // ── 边界值测试 ────────────────────────────────────────────────────

    // 19. num_layers=0 构造 — 空层集成器
    #[test]
    fn new_with_zero_layers() {
        let config = ResidualBypassConfig::default();
        let layer = ResidualBypassLayer::new(config, 0);
        assert_eq!(layer.history().len(), 0);
        assert_eq!(layer.stats().total, 0);
    }

    // 20. delta_rho=0.0 — 能量极低，|0.0 - 1.0| = 1.0 >> threshold
    #[test]
    fn decide_delta_rho_zero() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 0.0, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 21. delta_rho=f32::MAX — 极大偏差
    #[test]
    fn decide_delta_rho_max() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, f32::MAX, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 22. delta_rho=NaN — NaN 比较均为 false，不满足任何条件 → Execute
    #[test]
    fn decide_delta_rho_nan() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, f32::NAN, f32::NAN);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 23. cosine=0.0 — 方向完全不一致
    #[test]
    fn decide_cosine_zero() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0005, 0.0);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 24. cosine=1.0 — 方向完全一致
    #[test]
    fn decide_cosine_one() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0005, 1.0);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 25. cosine 恰好在阈值边界 — cosine_threshold=0.99, 需要 > 0.99
    #[test]
    fn decide_cosine_at_threshold() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        // cosine = 0.99 exactly, not > 0.99
        let decision = layer.decide_for_layer(5, 1.0005, 0.99);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 26. delta_rho 恰好在阈值边界 — 使用精确可表达的浮点数
    #[test]
    fn decide_delta_rho_at_threshold() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.0005,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // delta_rho=1.001, |1.001 - 1.0| = 0.001 > 0.0005 → Execute
        let decision = layer.decide_for_layer(0, 1.001, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 27. delta_rho=1.0 完全相等 — |0| < threshold
    #[test]
    fn decide_delta_rho_exactly_one() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 28. delta_rho < 1.0 — 如 0.9995，|0.9995 - 1.0| = 0.0005 < 0.001
    #[test]
    fn decide_delta_rho_below_one() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 0.9995, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // ── disabled 配置全路径 ────────────────────────────────────────────

    // 29. disabled 时 min_skip_layer 不生效 — 所有层都 Execute
    #[test]
    fn disabled_all_layers_execute() {
        let config = ResidualBypassConfig {
            enabled: false,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.0,  // 极宽松
            min_skip_layer: 0,      // 无最低层限制
        };
        let mut layer = ResidualBypassLayer::new(config, 32);
        for i in 0..32 {
            let d = layer.decide_for_layer(i, 1.0, 1.0);
            assert_eq!(d, ResidualBypassDecision::Execute, "disabled: layer {} should Execute", i);
        }
    }

    // 30. disabled 时从 aggregator 也 Execute
    #[test]
    fn disabled_aggregator_execute() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        let decision = layer.decide_from_aggregator(10, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 31. disabled 时从 page_header 也 Execute
    #[test]
    fn disabled_page_header_execute() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0);
        let decision = layer.decide_from_page_header(10, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // ── KvPageHeader f16 精度边界 ──────────────────────────────────────

    // 32. page_header 默认 delta_rho=0 — |0 - 1.0| = 1.0 → Execute
    #[test]
    fn page_header_default_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let header = KvPageHeader::default();
        let decision = layer.decide_from_page_header(5, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 33. page_header 大 delta_rho
    #[test]
    fn page_header_large_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(100.0);
        let decision = layer.decide_from_page_header(5, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // ── BypassStats Eq trait ────────────────────────────────────────────

    // 34. BypassStats Eq — 相等值产生一致的 hash (需 Hash，但 BypassStats 没有 Hash)
    //     Eq 意味着 reflexivity: a == a
    #[test]
    fn bypass_stats_eq_reflexivity() {
        let stats = BypassStats { total: 5, bypassed: 2, executed: 3 };
        assert_eq!(stats, stats);
    }

    // 35. BypassStats Eq — symmetry: a == b → b == a
    #[test]
    fn bypass_stats_eq_symmetry() {
        let a = BypassStats { total: 1, bypassed: 0, executed: 1 };
        let b = BypassStats { total: 1, bypassed: 0, executed: 1 };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ── BypassAdvice Clone/Copy 验证 ───────────────────────────────────

    // 36. BypassAdvice Clone 独立性
    #[test]
    fn bypass_advice_clone() {
        let original = BypassAdvice::Execute;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // 37. BypassAdvice Copy 语义
    #[test]
    fn bypass_advice_copy() {
        let original = BypassAdvice::Bypass;
        let copy = original;
        assert_eq!(original, BypassAdvice::Bypass);
        assert_eq!(copy, BypassAdvice::Bypass);
    }

    // ── BypassSummary 深度测试 ──────────────────────────────────────────

    // 38. BypassSummary Clone 独立性
    #[test]
    fn bypass_summary_clone_independence() {
        let original = BypassSummary {
            advices: vec![BypassAdvice::Execute, BypassAdvice::Bypass],
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // 修改 clone 不影响 original (Vec 独立)
        drop(cloned);
        assert_eq!(original.advices.len(), 2);
    }

    // 39. BypassSummary Debug 包含字段名
    #[test]
    fn bypass_summary_debug_format() {
        let summary = BypassSummary {
            advices: vec![BypassAdvice::Execute],
        };
        let s = format!("{:?}", summary);
        assert!(s.contains("advices"));
    }

    // 40. BypassSummary 大量 advice
    #[test]
    fn bypass_summary_many_advices() {
        let advices: Vec<BypassAdvice> = (0..100)
            .map(|i| if i % 2 == 0 { BypassAdvice::Execute } else { BypassAdvice::Bypass })
            .collect();
        let summary = BypassSummary { advices };
        assert_eq!(summary.advices.len(), 100);
        assert_eq!(summary.advices[0], BypassAdvice::Execute);
        assert_eq!(summary.advices[1], BypassAdvice::Bypass);
    }

    // ── ResidualBypassDecision 深度测试 ─────────────────────────────────

    // 41. ResidualBypassDecision Clone
    #[test]
    fn bypass_decision_clone() {
        let d = ResidualBypassDecision::Bypass;
        let cloned = d.clone();
        assert_eq!(d, cloned);
    }

    // 42. ResidualBypassDecision Copy 语义
    #[test]
    fn bypass_decision_copy() {
        let original = ResidualBypassDecision::Execute;
        let copy = original;
        assert_eq!(original, ResidualBypassDecision::Execute);
        assert_eq!(copy, ResidualBypassDecision::Execute);
    }

    // 43. ResidualBypassDecision Debug 输出
    #[test]
    fn bypass_decision_debug_format() {
        let s_exec = format!("{:?}", ResidualBypassDecision::Execute);
        let s_bypass = format!("{:?}", ResidualBypassDecision::Bypass);
        assert!(s_exec.contains("Execute"));
        assert!(s_bypass.contains("Bypass"));
    }

    // ── LayerBypassRecord 深度测试 ──────────────────────────────────────

    // 44. LayerBypassRecord 不同 layer_idx 不相等
    #[test]
    fn layer_bypass_record_different_layer_idx() {
        let a = LayerBypassRecord {
            layer_idx: 1,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        let b = LayerBypassRecord {
            layer_idx: 2,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        assert_ne!(a, b);
    }

    // 45. LayerBypassRecord 不同 delta_rho 不相等
    #[test]
    fn layer_bypass_record_different_delta_rho() {
        let a = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        let b = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.1,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        assert_ne!(a, b);
    }

    // 46. LayerBypassRecord 不同 cosine 不相等
    #[test]
    fn layer_bypass_record_different_cosine() {
        let a = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        let b = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: 1.0,
            cosine: 0.95,
            decision: ResidualBypassDecision::Execute,
        };
        assert_ne!(a, b);
    }

    // ── reset 多次调用 ─────────────────────────────────────────────────

    // 47. 连续两次 reset 无副作用
    #[test]
    fn reset_twice_no_side_effects() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);

        layer.decide_for_layer(5, 1.0005, 0.995);
        layer.reset();
        layer.reset(); // 第二次

        assert_eq!(layer.history().len(), 0);
        assert_eq!(layer.stats().total, 0);
        assert_eq!(layer.stats().bypassed, 0);
        assert_eq!(layer.stats().executed, 0);
    }

    // 48. reset 后可继续使用
    #[test]
    fn reset_then_continue() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);

        layer.decide_for_layer(5, 1.0005, 0.995); // Bypass
        layer.reset();

        // reset 后新决策正常记录
        layer.decide_for_layer(5, 1.05, 0.99); // Execute
        assert_eq!(layer.history().len(), 1);
        assert_eq!(layer.stats().total, 1);
        assert_eq!(layer.stats().executed, 1);
        assert_eq!(layer.stats().bypassed, 0);
    }

    // ── history 切片验证 ────────────────────────────────────────────────

    // 49. history() 返回切片反映后续决策
    #[test]
    fn history_slice_reflects_decisions() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        assert!(layer.history().is_empty());

        layer.decide_for_layer(4, 1.0005, 0.995);
        assert_eq!(layer.history().len(), 1);

        layer.decide_for_layer(5, 1.05, 0.99);
        assert_eq!(layer.history().len(), 2);
    }

    // ── decide_from_aggregator 无信号 ───────────────────────────────────

    // 50. 空 TelemetryAggregator (无信号 ingest) — 默认 delta_rho=0.0, cosine=0.0
    #[test]
    fn decide_from_empty_aggregator() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);
        let agg = TelemetryAggregator::new();

        let decision = layer.decide_from_aggregator(5, &agg);
        // 空 aggregator 默认 delta_rho=0.0, cosine=0.0
        // |0.0 - 1.0| = 1.0 >> 0.001 → Execute
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // ── 大量决策压力测试 ────────────────────────────────────────────────

    // 51. 大量连续决策
    #[test]
    fn many_consecutive_decisions() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 100);

        for i in 0..100 {
            layer.decide_for_layer(i, 1.0005, 0.995);
        }
        assert_eq!(layer.history().len(), 100);
        let stats = layer.stats();
        assert_eq!(stats.total, 100);
        assert_eq!(stats.bypassed, 100);
        assert_eq!(stats.executed, 0);
    }

    // ── LayerBypassRecord Execute 变体记录 ──────────────────────────────

    // 52. 记录 Execute 决策的字段正确性
    #[test]
    fn layer_bypass_record_execute_decision() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        layer.decide_for_layer(0, 0.5, 0.3);

        let rec = &layer.history()[0];
        assert_eq!(rec.layer_idx, 0);
        assert!((rec.delta_rho - 0.5).abs() < 1e-6);
        assert!((rec.cosine - 0.3).abs() < 1e-6);
        assert_eq!(rec.decision, ResidualBypassDecision::Execute);
    }

    // ── BypassAdvice 作为 HashMap key ───────────────────────────────────

    // 53. BypassAdvice 可用作 HashMap key
    #[test]
    fn bypass_advice_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(BypassAdvice::Execute, 10);
        map.insert(BypassAdvice::Bypass, 5);
        assert_eq!(map[&BypassAdvice::Execute], 10);
        assert_eq!(map[&BypassAdvice::Bypass], 5);
    }

    // ── BypassStats Debug 包含所有字段名 ────────────────────────────────

    // 54. BypassStats Debug 包含 bypassed 和 executed 字段
    #[test]
    fn bypass_stats_debug_all_fields() {
        let stats = BypassStats { total: 7, bypassed: 3, executed: 4 };
        let s = format!("{:?}", stats);
        assert!(s.contains("total"));
        assert!(s.contains("bypassed"));
        assert!(s.contains("executed"));
    }

    // ── decide_from_page_header 记录 delta_rho 通过 f16 转换 ────────────

    // 55. page_header 记录的 delta_rho 是 f16 解码后的值
    #[test]
    fn page_header_records_f16_decoded_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        let original = 1.05f32;
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(original);

        layer.decide_from_page_header(5, &header);

        let rec = &layer.history()[0];
        // f16 精度约 3 位十进制，1.05 在 f16 范围内
        assert!((rec.delta_rho - f16_bits_to_f32(crate::kv_cache::f32_to_f16_bits(original))).abs() < 1e-3);
    }

    // ── ResidualBypassConfig Default 值验证 ─────────────────────────────

    // 56. ResidualBypassConfig default 字段值符合 SPEC
    #[test]
    fn residual_bypass_config_default_values() {
        let config = ResidualBypassConfig::default();
        assert!(config.enabled);
        assert!((config.delta_rho_threshold - 0.001).abs() < 1e-7);
        assert!((config.cosine_threshold - 0.99).abs() < 1e-7);
        assert_eq!(config.min_skip_layer, 4);
    }

    // ── 混合三种决策路径 ────────────────────────────────────────────────

    // 57. 三种决策路径 (for_layer / aggregator / page_header) 交叉使用
    #[test]
    fn mixed_decision_paths() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 16);

        // 路径 1: decide_for_layer
        layer.decide_for_layer(0, 1.0005, 0.995); // Bypass

        // 路径 2: decide_from_aggregator
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.05 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.88 });
        layer.decide_from_aggregator(1, &agg); // Execute

        // 路径 3: decide_from_page_header
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0);
        layer.decide_from_page_header(2, &header); // cos=0.99 not > 0.99 → Execute

        assert_eq!(layer.history().len(), 3);
        assert_eq!(layer.stats().total, 3);
        assert_eq!(layer.stats().bypassed, 1);
        assert_eq!(layer.stats().executed, 2);
    }

    // ── 极大 layer_idx ──────────────────────────────────────────────────

    // 58. 极大 layer_idx 仍然正常决策
    #[test]
    fn large_layer_idx() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 1000);
        let decision = layer.decide_for_layer(999, 1.0005, 0.995);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
        assert_eq!(layer.history()[0].layer_idx, 999);
    }

    // ── wave-12x50 新增测试 ──────────────────────────────────────────

    // 59. ResidualBypassConfig Clone 独立性
    #[test]
    fn config_clone_independence() {
        let a = ResidualBypassConfig::default();
        let b = a.clone();
        assert!(b.enabled);
        // 修改 clone 不影响 original (Copy fields so just verify equality)
        assert!((a.delta_rho_threshold - b.delta_rho_threshold).abs() < 1e-10);
        assert_eq!(a.min_skip_layer, b.min_skip_layer);
    }

    // 60. ResidualBypassConfig 自定义构造
    #[test]
    fn config_custom_construction() {
        let config = ResidualBypassConfig {
            enabled: false,
            delta_rho_threshold: 0.05,
            cosine_threshold: 0.8,
            min_skip_layer: 10,
        };
        assert!(!config.enabled);
        assert!((config.delta_rho_threshold - 0.05).abs() < 1e-7);
        assert!((config.cosine_threshold - 0.8).abs() < 1e-7);
        assert_eq!(config.min_skip_layer, 10);
    }

    // 61. ResidualBypassConfig Debug 包含字段
    #[test]
    fn config_debug_format() {
        let config = ResidualBypassConfig::default();
        let s = format!("{:?}", config);
        assert!(s.contains("enabled"));
        assert!(s.contains("delta_rho_threshold"));
        assert!(s.contains("cosine_threshold"));
        assert!(s.contains("min_skip_layer"));
    }

    // 62. ResidualBypassDecision Eq 自反性
    #[test]
    fn bypass_decision_eq_reflexivity() {
        let d = ResidualBypassDecision::Execute;
        assert_eq!(d, d);
        let d2 = ResidualBypassDecision::Bypass;
        assert_eq!(d2, d2);
    }

    // 63. ResidualBypassDecision Eq 对称性
    #[test]
    fn bypass_decision_eq_symmetry() {
        let a = ResidualBypassDecision::Execute;
        let b = ResidualBypassDecision::Execute;
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // 64. ResidualBypassDecision PartialEq 不相等
    #[test]
    fn bypass_decision_partial_ne() {
        assert_ne!(ResidualBypassDecision::Execute, ResidualBypassDecision::Bypass);
        assert_ne!(ResidualBypassDecision::Bypass, ResidualBypassDecision::Execute);
    }

    // 65. decide_for_layer 负 delta_rho
    #[test]
    fn decide_negative_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, -1.0, 0.999);
        // |(-1.0) - 1.0| = 2.0 >> 0.001 → Execute
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 66. decide_for_layer 负 cosine
    #[test]
    fn decide_negative_cosine() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0005, -0.5);
        // cosine=-0.5 not > 0.99 → Execute
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 67. decide_for_layer cosine > 1.0 (非物理值但仍正常处理)
    #[test]
    fn decide_cosine_above_one() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0005, 1.5);
        // cosine=1.5 > 0.99 → true; delta_rho ok → Bypass
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 68. decide_for_layer delta_rho=f32::MIN (极大负数)
    #[test]
    fn decide_delta_rho_min() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, f32::MIN, 0.999);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 69. decide_for_layer delta_rho=f32::INFINITY
    #[test]
    fn decide_delta_rho_infinity() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, f32::INFINITY, 0.999);
        // |inf - 1.0| = inf, not < threshold → Execute
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 70. decide_for_layer cosine=f32::INFINITY
    #[test]
    fn decide_cosine_infinity() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, 1.0005, f32::INFINITY);
        // inf > 0.99 → true; delta_rho ok → Bypass
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 71. 零 delta_rho_threshold — 任何非精确1.0 都不跳过
    #[test]
    fn zero_threshold_only_exact_one_bypass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.0,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // delta_rho=1.0 exactly: |1.0-1.0|=0.0 < 0.0 is false (equal, not less)
        let d1 = layer.decide_for_layer(0, 1.0, 0.999);
        assert_eq!(d1, ResidualBypassDecision::Execute);
    }

    // 72. 极大 min_skip_layer — 所有层都 Execute
    #[test]
    fn huge_min_skip_layer_blocks_all() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 100,
        };
        let mut layer = ResidualBypassLayer::new(config, 16);
        for i in 0..16 {
            let d = layer.decide_for_layer(i, 1.0005, 0.995);
            assert_eq!(d, ResidualBypassDecision::Execute, "layer {} should Execute", i);
        }
    }

    // 73. 极宽松 cosine_threshold=0.0
    #[test]
    fn zero_cosine_threshold_all_pass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.0,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // cosine=-1.0 not > 0.0 (strict >)
        let d = layer.decide_for_layer(0, 1.0005, -1.0);
        assert_eq!(d, ResidualBypassDecision::Execute);
        // cosine=0.001 > 0.0, delta_rho ok → Bypass
        let d2 = layer.decide_for_layer(1, 1.0005, 0.001);
        assert_eq!(d2, ResidualBypassDecision::Bypass);
    }

    // 74. stats 一致性: total == bypassed + executed
    #[test]
    fn stats_total_equals_sum() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 32);
        for i in 0..20 {
            layer.decide_for_layer(i, if i % 3 == 0 { 1.0005 } else { 1.05 }, 0.995);
        }
        let stats = layer.stats();
        assert_eq!(stats.total, stats.bypassed + stats.executed);
    }

    // 75. decide_from_aggregator 仅 ingest delta_rho 无 cosine
    #[test]
    fn aggregator_only_delta_rho_no_cosine() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        // cosine defaults to 0.0 → Execute
        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 76. decide_from_aggregator 仅 ingest cosine 无 delta_rho
    #[test]
    fn aggregator_only_cosine_no_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        // delta_rho defaults to 0.0, |0.0-1.0|=1.0 >> threshold → Execute
        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 77. decide_from_aggregator 多次 ingest 覆盖
    #[test]
    fn aggregator_overwrite_signals() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.5 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 });
        // 覆盖为 bypass 条件
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }

    // 78. decide_from_page_header 负 delta_rho f16
    #[test]
    fn page_header_negative_delta_rho() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(-2.0);
        let decision = layer.decide_from_page_header(5, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 79. decide_from_page_header 层0 最低层约束
    #[test]
    fn page_header_layer_zero_forced_execute() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0005);
        let decision = layer.decide_from_page_header(0, &header);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 80. decide_from_page_header history 记录 f16 解码值在合理范围
    #[test]
    fn page_header_delta_rho_in_range() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut header = KvPageHeader::default();
        let original = 0.5f32;
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(original);
        layer.decide_from_page_header(5, &header);
        let rec = &layer.history()[0];
        let decoded = f16_bits_to_f32(crate::kv_cache::f32_to_f16_bits(original));
        assert!((rec.delta_rho - decoded).abs() < 0.01);
    }

    // 81. history 记录数与 stats.total 始终一致
    #[test]
    fn history_len_equals_stats_total() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 32);
        for i in 0..15 {
            layer.decide_for_layer(i, 1.0005, 0.995);
        }
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.05 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.88 });
        layer.decide_from_aggregator(15, &agg);
        assert_eq!(layer.history().len(), layer.stats().total);
    }

    // 82. BypassAdvice PartialEq 对称性
    #[test]
    fn bypass_advice_eq_symmetry() {
        let a = BypassAdvice::Execute;
        let b = BypassAdvice::Execute;
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // 83. BypassAdvice Eq 自反性
    #[test]
    fn bypass_advice_eq_reflexivity() {
        assert_eq!(BypassAdvice::Execute, BypassAdvice::Execute);
        assert_eq!(BypassAdvice::Bypass, BypassAdvice::Bypass);
    }

    // 84. BypassSummary PartialEq 顺序敏感
    #[test]
    fn bypass_summary_order_matters() {
        let a = BypassSummary { advices: vec![BypassAdvice::Execute, BypassAdvice::Bypass] };
        let b = BypassSummary { advices: vec![BypassAdvice::Bypass, BypassAdvice::Execute] };
        assert_ne!(a, b);
    }

    // 85. BypassSummary 单元素
    #[test]
    fn bypass_summary_single_advice() {
        let summary = BypassSummary { advices: vec![BypassAdvice::Execute] };
        assert_eq!(summary.advices.len(), 1);
        assert_eq!(summary.advices[0], BypassAdvice::Execute);
    }

    // 86. BypassStats 大值
    #[test]
    fn bypass_stats_large_values() {
        let stats = BypassStats { total: 100_000, bypassed: 60_000, executed: 40_000 };
        assert_eq!(stats.total, 100_000);
        assert_eq!(stats.bypassed + stats.executed, stats.total);
    }

    // 87. BypassStats PartialEq 不相等 — total 不同
    #[test]
    fn bypass_stats_ne_different_total() {
        let a = BypassStats { total: 5, bypassed: 2, executed: 3 };
        let b = BypassStats { total: 6, bypassed: 2, executed: 3 };
        assert_ne!(a, b);
    }

    // 88. LayerBypassRecord debug 包含所有字段名
    #[test]
    fn layer_bypass_record_debug_all_fields() {
        let rec = LayerBypassRecord {
            layer_idx: 0,
            delta_rho: 0.0,
            cosine: 0.0,
            decision: ResidualBypassDecision::Execute,
        };
        let s = format!("{:?}", rec);
        assert!(s.contains("layer_idx"));
        assert!(s.contains("delta_rho"));
        assert!(s.contains("cosine"));
        assert!(s.contains("decision"));
    }

    // 89. ResidualBypassLayer new 预分配历史容量
    #[test]
    fn new_preallocates_history() {
        let config = ResidualBypassConfig::default();
        let layer = ResidualBypassLayer::new(config, 64);
        assert!(layer.history().is_empty());
        assert_eq!(layer.stats().total, 0);
    }

    // 90. reset 后 stats 全部归零
    #[test]
    fn reset_stats_all_zero() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 32);
        for i in 0..10 {
            layer.decide_for_layer(i, 1.0005, 0.995);
        }
        assert!(layer.stats().total > 0);
        layer.reset();
        let stats = layer.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.bypassed, 0);
        assert_eq!(stats.executed, 0);
    }

    // 91. 三种路径交叉后 history layer_idx 正确
    #[test]
    fn mixed_paths_history_layer_indices() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 16);

        layer.decide_for_layer(10, 1.0005, 0.995);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.05 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.88 });
        layer.decide_from_aggregator(11, &agg);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.1);
        layer.decide_from_page_header(12, &header);

        let h = layer.history();
        assert_eq!(h[0].layer_idx, 10);
        assert_eq!(h[1].layer_idx, 11);
        assert_eq!(h[2].layer_idx, 12);
    }

    // 92. decide_from_aggregator 记录的 delta_rho 来自 aggregator
    #[test]
    fn aggregator_history_records_agg_values() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 0.98 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.77 });
        layer.decide_from_aggregator(5, &agg);
        let rec = &layer.history()[0];
        assert!((rec.delta_rho - 0.98).abs() < 1e-6);
        assert!((rec.cosine - 0.77).abs() < 1e-6);
    }

    // 93. delta_rho 接近1.0 但 cos 恰好在阈值 (cosine_threshold=0.99, cos=0.99)
    #[test]
    fn cosine_exactly_at_threshold_is_execute() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        // cosine=0.99 not > 0.99 (strict >) → Execute
        let d = layer.decide_for_layer(5, 1.0001, 0.99);
        assert_eq!(d, ResidualBypassDecision::Execute);
    }

    // 94. delta_rho 刚好在阈值内 (|drho-1| < threshold)
    #[test]
    fn delta_rho_just_inside_threshold() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.01,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // |1.005 - 1.0| = 0.005 < 0.01 → ok
        let d = layer.decide_for_layer(0, 1.005, 0.999);
        assert_eq!(d, ResidualBypassDecision::Bypass);
    }

    // 95. delta_rho 刚好在阈值外 (|drho-1| > threshold)
    #[test]
    fn delta_rho_just_outside_threshold() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // |1.002 - 1.0| = 0.002 > 0.001 → Execute
        let d = layer.decide_for_layer(0, 1.002, 0.999);
        assert_eq!(d, ResidualBypassDecision::Execute);
    }

    // 96. min_skip_layer=1 — 仅层0被强制 Execute
    #[test]
    fn min_skip_layer_one() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 1,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let d0 = layer.decide_for_layer(0, 1.0005, 0.995);
        assert_eq!(d0, ResidualBypassDecision::Execute);
        let d1 = layer.decide_for_layer(1, 1.0005, 0.995);
        assert_eq!(d1, ResidualBypassDecision::Bypass);
    }

    // 97. 全部 Execute 时 stats.bypassed == 0
    #[test]
    fn all_execute_stats_bypassed_zero() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let mut layer = ResidualBypassLayer::new(config, 16);
        for i in 0..16 {
            layer.decide_for_layer(i, 1.0, 1.0);
        }
        assert_eq!(layer.stats().bypassed, 0);
        assert_eq!(layer.stats().executed, 16);
    }

    // 98. 全部 Bypass 时 stats.executed == 0
    #[test]
    fn all_bypass_stats_executed_zero() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        for i in 0..8 {
            layer.decide_for_layer(i, 1.0005, 0.995);
        }
        assert_eq!(layer.stats().executed, 0);
        assert_eq!(layer.stats().bypassed, 8);
    }

    // 99. page_header 重复调用同一层 — 每次都记录
    #[test]
    fn page_header_same_layer_multiple_times() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let header = KvPageHeader::default();
        layer.decide_from_page_header(5, &header);
        layer.decide_from_page_header(5, &header);
        layer.decide_from_page_header(5, &header);
        assert_eq!(layer.history().len(), 3);
        assert_eq!(layer.stats().total, 3);
    }

    // 100. disabled 配置下三种路径都 Execute
    #[test]
    fn disabled_all_three_paths_execute() {
        let config = ResidualBypassConfig { enabled: false, ..Default::default() };
        let mut layer = ResidualBypassLayer::new(config, 8);

        let d1 = layer.decide_for_layer(10, 1.0005, 0.995);
        assert_eq!(d1, ResidualBypassDecision::Execute);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        let d2 = layer.decide_from_aggregator(10, &agg);
        assert_eq!(d2, ResidualBypassDecision::Execute);

        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0005);
        let d3 = layer.decide_from_page_header(10, &header);
        assert_eq!(d3, ResidualBypassDecision::Execute);
    }

    // 101. LayerBypassRecord 零值字段
    #[test]
    fn layer_bypass_record_zero_fields() {
        let rec = LayerBypassRecord {
            layer_idx: 0,
            delta_rho: 0.0,
            cosine: 0.0,
            decision: ResidualBypassDecision::Execute,
        };
        assert_eq!(rec.layer_idx, 0);
        assert!((rec.delta_rho).abs() < 1e-10);
        assert!((rec.cosine).abs() < 1e-10);
    }

    // 102. BypassStats Default 应与全零字面量相等
    #[test]
    fn bypass_stats_default_matches_zero_literal() {
        let def = BypassStats::default();
        let zero = BypassStats { total: 0, bypassed: 0, executed: 0 };
        assert_eq!(def, zero);
    }

    // 103. history() 切片内容只读
    #[test]
    fn history_slice_readonly() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        layer.decide_for_layer(5, 1.0, 0.999);
        let h = layer.history();
        assert_eq!(h.len(), 1);
        // 读取不修改状态
        let stats_before = layer.stats();
        let _ = h[0].delta_rho;
        assert_eq!(layer.stats(), stats_before);
    }

    // 104. decide_from_aggregator 后 stats 一致性
    #[test]
    fn aggregator_stats_consistency() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        layer.decide_from_aggregator(0, &agg);
        let stats = layer.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.bypassed + stats.executed, 1);
    }

    // 105. reset 不影响 num_layers 配置 (通过再次使用验证)
    #[test]
    fn reset_preserves_capacity() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 32);
        layer.decide_for_layer(5, 1.0005, 0.995);
        layer.reset();
        // 可以继续做大量决策
        for i in 0..32 {
            layer.decide_for_layer(i, 1.0005, 0.995);
        }
        assert_eq!(layer.history().len(), 32);
    }

    // 106. BypassSummary PartialEq 相同内容
    #[test]
    fn bypass_summary_eq_same_content() {
        let a = BypassSummary { advices: vec![BypassAdvice::Bypass] };
        let b = BypassSummary { advices: vec![BypassAdvice::Bypass] };
        assert_eq!(a, b);
    }

    // 107. delta_rho 略小于 1.0 但在阈值内 → Bypass
    #[test]
    fn delta_rho_slightly_below_one_bypass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.01,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // |0.995 - 1.0| = 0.005 < 0.01 → ok
        let d = layer.decide_for_layer(0, 0.995, 0.999);
        assert_eq!(d, ResidualBypassDecision::Bypass);
    }

    // 108. delta_rho=1.0 且 cosine=1.0 → Bypass
    #[test]
    fn perfect_stability_bypass() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let d = layer.decide_for_layer(5, 1.0, 1.0);
        assert_eq!(d, ResidualBypassDecision::Bypass);
    }

    // 109. decide_from_page_header cos 始终为 0.99 (KvPageHeader 无 cosine 字段)
    #[test]
    fn page_header_cosine_always_099() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let header = KvPageHeader::default();
        layer.decide_from_page_header(5, &header);
        let rec = &layer.history()[0];
        assert!((rec.cosine - 0.99).abs() < 1e-6);
    }

    // 110. BypassAdvice Debug 包含变体名
    #[test]
    fn bypass_advice_debug_execute() {
        let s = format!("{:?}", BypassAdvice::Execute);
        assert!(s.contains("Execute"));
        assert!(!s.contains("Bypass"));
    }

    // 111. LayerBypassRecord 不同 decision 但相同其他字段不相等
    #[test]
    fn layer_record_same_fields_different_decision_ne() {
        let a = LayerBypassRecord {
            layer_idx: 5,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Execute,
        };
        let b = LayerBypassRecord {
            layer_idx: 5,
            delta_rho: 1.0,
            cosine: 0.99,
            decision: ResidualBypassDecision::Bypass,
        };
        assert_ne!(a, b);
    }

    // 112. BypassSummary PartialEq 不同长度
    #[test]
    fn bypass_summary_ne_different_length() {
        let a = BypassSummary { advices: vec![BypassAdvice::Execute] };
        let b = BypassSummary { advices: vec![BypassAdvice::Execute, BypassAdvice::Bypass] };
        assert_ne!(a, b);
    }

    // 113. decide_for_layer 连续调用 layer_idx 不要求递增
    #[test]
    fn layer_idx_not_monotonic() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        layer.decide_for_layer(7, 1.0005, 0.995);
        layer.decide_for_layer(3, 1.0005, 0.995);
        layer.decide_for_layer(5, 1.0005, 0.995);
        assert_eq!(layer.history().len(), 3);
        assert_eq!(layer.history()[0].layer_idx, 7);
        assert_eq!(layer.history()[1].layer_idx, 3);
        assert_eq!(layer.history()[2].layer_idx, 5);
    }

    // 114. cosine_threshold=1.0 — 需要 cosine > 1.0 不可能 → 总是 Execute
    #[test]
    fn cosine_threshold_one_always_execute() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 1.0,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let d = layer.decide_for_layer(0, 1.0005, 1.0);
        // 1.0 not > 1.0 → Execute
        assert_eq!(d, ResidualBypassDecision::Execute);
    }

    // 115. delta_rho_threshold=1.0 — 极宽松，几乎所有 delta_rho 都满足
    #[test]
    fn delta_rho_threshold_one_very_loose() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 1.0,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // |0.5 - 1.0| = 0.5 < 1.0 → ok, cosine=0.999 > 0.99 → Bypass
        let d = layer.decide_for_layer(0, 0.5, 0.999);
        assert_eq!(d, ResidualBypassDecision::Bypass);
    }

    // 116. BypassSummary Clone 后修改不影响原
    #[test]
    fn bypass_summary_clone_isolation() {
        let original = BypassSummary {
            advices: vec![BypassAdvice::Execute],
        };
        let mut cloned = original.clone();
        cloned.advices.push(BypassAdvice::Bypass);
        assert_eq!(original.advices.len(), 1);
        assert_eq!(cloned.advices.len(), 2);
    }

    // 117. page_header 最后一层决策
    #[test]
    fn page_header_last_layer() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 64);
        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.0005);
        let d = layer.decide_from_page_header(63, &header);
        // cos=0.99 not > 0.99 → Execute
        assert_eq!(d, ResidualBypassDecision::Execute);
    }

    // 118. decide_from_aggregator 多次调用共享同一 agg
    #[test]
    fn aggregator_shared_across_calls() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        for i in 0..5 {
            let d = layer.decide_from_aggregator(i, &agg);
            assert_eq!(d, ResidualBypassDecision::Bypass, "layer {} should Bypass", i);
        }
        assert_eq!(layer.stats().bypassed, 5);
    }

    // 119. LayerBypassRecord 含 f32 特殊值 (NaN)
    #[test]
    fn layer_record_with_nan_values() {
        let rec = LayerBypassRecord {
            layer_idx: 3,
            delta_rho: f32::NAN,
            cosine: f32::NAN,
            decision: ResidualBypassDecision::Execute,
        };
        assert!(rec.delta_rho.is_nan());
        assert!(rec.cosine.is_nan());
        assert_eq!(rec.layer_idx, 3);
    }

    // 120. BypassStats Eq transitivity: a==b && b==c → a==c
    #[test]
    fn bypass_stats_eq_transitivity() {
        let a = BypassStats { total: 3, bypassed: 1, executed: 2 };
        let b = BypassStats { total: 3, bypassed: 1, executed: 2 };
        let c = BypassStats { total: 3, bypassed: 1, executed: 2 };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── wave-12x54 新增测试 ──────────────────────────────────────────

    // 121. TelemetryAggregator::ingest_from_page_header → decide_from_aggregator 路径
    #[test]
    fn aggregator_from_page_header_then_decide() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 16);

        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.5);

        let mut agg = TelemetryAggregator::new();
        agg.ingest_from_page_header(&header);
        // aggregator 读出 delta_rho ~1.5, cosine=0.0 → Execute
        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 122. 非 residual EpilogueSignal (DeadNeuronRatio) 不影响 bypass 决策
    #[test]
    fn non_residual_signal_does_not_affect_bypass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        // Ingest non-residual signals only
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.5 });
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 2.0 });
        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 1.5 });
        // delta_rho defaults to 0.0, cosine defaults to 0.0 → Execute
        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 123. LayerBypassRecord 含 f32::INFINITY 值
    #[test]
    fn layer_record_with_infinity_values() {
        let rec = LayerBypassRecord {
            layer_idx: 10,
            delta_rho: f32::INFINITY,
            cosine: f32::INFINITY,
            decision: ResidualBypassDecision::Bypass,
        };
        assert!(rec.delta_rho.is_infinite());
        assert!(rec.cosine.is_infinite());
        assert!(!rec.delta_rho.is_nan());
        assert_eq!(rec.layer_idx, 10);
    }

    // 124. BypassStats 字段独立 — total 可以不等于 bypassed + executed
    #[test]
    fn bypass_stats_fields_are_independent() {
        // BypassStats is a plain struct with no invariant enforcement
        let stats = BypassStats { total: 100, bypassed: 30, executed: 60 };
        // 30 + 60 = 90 != 100 — fields are independent, not enforced
        assert_ne!(stats.bypassed + stats.executed, stats.total);
    }

    // 125. delta_rho_threshold 为负值 — 所有 delta_rho 都满足 |drho-1| < negative (false)
    #[test]
    fn negative_delta_rho_threshold_never_bypass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: -0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // |1.0 - 1.0| = 0.0 < -0.001 is false → Execute
        let d = layer.decide_for_layer(0, 1.0, 0.999);
        assert_eq!(d, ResidualBypassDecision::Execute);
    }

    // 126. cosine_threshold 为负值 — 任何 cosine > 负数都满足
    #[test]
    fn negative_cosine_threshold_all_pass() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: -1.0,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // cosine=0.0 > -1.0 → true; delta_rho ok → Bypass
        let d = layer.decide_for_layer(0, 1.0005, 0.0);
        assert_eq!(d, ResidualBypassDecision::Bypass);
    }

    // 127. 多个不同 aggregator 实例分别决策
    #[test]
    fn different_aggregators_for_different_layers() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);

        // Aggregator 1: bypass conditions
        let mut agg1 = TelemetryAggregator::new();
        agg1.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg1.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        // Aggregator 2: execute conditions
        let mut agg2 = TelemetryAggregator::new();
        agg2.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.5 });
        agg2.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 });

        let d1 = layer.decide_from_aggregator(0, &agg1);
        let d2 = layer.decide_from_aggregator(1, &agg2);

        assert_eq!(d1, ResidualBypassDecision::Bypass);
        assert_eq!(d2, ResidualBypassDecision::Execute);
        assert_eq!(layer.history().len(), 2);
    }

    // 128. BypassSummary 全部 Execute advices
    #[test]
    fn bypass_summary_all_execute() {
        let advices = vec![BypassAdvice::Execute; 10];
        let summary = BypassSummary { advices };
        assert_eq!(summary.advices.len(), 10);
        assert!(summary.advices.iter().all(|a| *a == BypassAdvice::Execute));
    }

    // 129. BypassAdvice 可收集为 Vec 并迭代
    #[test]
    fn bypass_advice_collect_and_iterate() {
        let advices: Vec<BypassAdvice> = [BypassAdvice::Execute, BypassAdvice::Bypass, BypassAdvice::Execute]
            .into_iter()
            .collect();
        assert_eq!(advices.len(), 3);
        let execute_count = advices.iter().filter(|a| **a == BypassAdvice::Execute).count();
        assert_eq!(execute_count, 2);
        let bypass_count = advices.iter().filter(|a| **a == BypassAdvice::Bypass).count();
        assert_eq!(bypass_count, 1);
    }

    // 130. ResidualBypassConfig clone 后修改不影响原
    #[test]
    fn config_clone_field_modification() {
        let original = ResidualBypassConfig::default();
        let mut cloned = original.clone();
        cloned.enabled = false;
        cloned.min_skip_layer = 100;
        // original unchanged
        assert!(original.enabled);
        assert_eq!(original.min_skip_layer, 4);
    }

    // 131. decide_for_layer delta_rho=f32::NEG_INFINITY
    #[test]
    fn decide_delta_rho_neg_infinity() {
        let config = ResidualBypassConfig::default();
        let mut layer = ResidualBypassLayer::new(config, 8);
        let decision = layer.decide_for_layer(5, f32::NEG_INFINITY, 0.999);
        // |neg_inf - 1.0| = inf, not < threshold → Execute
        assert_eq!(decision, ResidualBypassDecision::Execute);
    }

    // 132. delta_rho_threshold=f32::EPSILON — 极严格阈值
    #[test]
    fn epsilon_threshold_very_strict() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: f32::EPSILON,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);
        // delta_rho=1.0 exactly: |1.0-1.0|=0.0 < EPSILON → true
        let d1 = layer.decide_for_layer(0, 1.0, 0.999);
        assert_eq!(d1, ResidualBypassDecision::Bypass);
        // delta_rho=1.0001: |1.0001-1.0|=0.0001 >> EPSILON → Execute
        let d2 = layer.decide_for_layer(1, 1.0001, 0.999);
        assert_eq!(d2, ResidualBypassDecision::Execute);
    }

    // 133. decide_from_aggregator 从 page_header ingest 后再叠加 residual 信号覆盖
    #[test]
    fn aggregator_page_header_then_residual_override() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 0,
        };
        let mut layer = ResidualBypassLayer::new(config, 8);

        let mut header = KvPageHeader::default();
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.5);

        let mut agg = TelemetryAggregator::new();
        agg.ingest_from_page_header(&header);
        // Now override with bypass-favorable signals
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0001 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        let decision = layer.decide_from_aggregator(5, &agg);
        assert_eq!(decision, ResidualBypassDecision::Bypass);
    }
}
