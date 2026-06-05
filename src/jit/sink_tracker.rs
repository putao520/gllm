//! Sink Detection + Sharpness Tracking 集成层 (SPEC §13.9, §18.1)
//!
//! ## 核心职责
//! 将 SinkDetector (决策层) 与注意力模式跟踪集成:
//! - 从 TelemetryAggregator / KvPageHeader 读取 Softmax 绌计信息
//! - 做出注意力模式分类 (Normal/Sink/SharpFocus/Diffuse)
//! - 维护 Sink 保护 token 位置列表 (KIVI FP16 保护)
//! - 跟踪逐层注意力模式变化
//! - 提供批次级注意力模式汇总
//!
//! ## 数据流
//! ```
//! KvPageHeader.softmax_max/sharpness ──→ SinkDetector.detect()
//!                                            ↓
//!                                  AttentionPattern
//!                           ┌──────────┼──────────┐
//!                           ↓          ↓          ↓
//!                       Normal      Sink      SharpFocus   Diffuse
//!                       (正常)    (Sink保护)  (紧凑关注)  (均匀分散)
//! ```
//!
//! ## 与 KIVI 集成 (§11.2)
//! Sink token 的 KV 被保护为 FP16，不被量化。 SinkTracker 维护
//! 保护位置列表，当检测到 Sink 模式时标记哪些位置需要保护.

use super::epilogue::{SinkDetectionConfig, AttentionPattern, SinkDetector, TelemetryAggregator};
use crate::kv_cache::{KvPageHeader, f16_bits_to_f32};

/// 逐层注意力模式记录
#[derive(Debug, Clone)]
pub struct LayerAttentionRecord {
    /// 层索引
    pub layer_idx: usize,
    /// Softmax max 值
    pub softmax_max: f32,
    /// Sharpness (max/sum 比值)
    pub sharpness: f32,
    /// 检测到的注意力模式
    pub pattern: AttentionPattern,
}

/// Sink Detection + Sharpness Tracking 层级集成器
///
/// 管理多层注意力模式的检测和跟踪:
/// - 维护 SinkDetector 实例
/// - 从 TelemetryAggregator / KvPageHeader 提取信号
/// - 逐层做出注意力模式分类
/// - 维护 Sink 保护 token 列表
/// - 记录注意力模式历史
pub struct SinkTracker {
    /// 检测器
    detector: SinkDetector,
    /// 逐层注意力模式记录 (layer_idx → record)
    history: Vec<LayerAttentionRecord>,
    /// 总层数
    num_layers: usize,
    /// 累计各模式计数
    normal_count: usize,
    sink_count: usize,
    sharp_focus_count: usize,
    diffuse_count: usize,
}

impl SinkTracker {
    /// 创建新的 SinkTracker
    pub fn new(config: SinkDetectionConfig, num_layers: usize) -> Self {
        Self {
            detector: SinkDetector::new(config),
            history: Vec::with_capacity(num_layers),
            num_layers,
            normal_count: 0,
            sink_count: 0,
            sharp_focus_count: 0,
            diffuse_count: 0,
        }
    }

    /// 从 Softmax max 和 sharpness 检测当前层的注意力模式
    pub fn detect_for_layer(
        &mut self,
        layer_idx: usize,
        max_val: f32,
        sharpness: f32,
    ) -> AttentionPattern {
        let pattern = self.detector.detect(max_val, sharpness);
        self.record_pattern(layer_idx, max_val, sharpness, pattern);
        pattern
    }

    /// 从 TelemetryAggregator 检测当前层的注意力模式
    pub fn detect_from_aggregator(
        &mut self,
        layer_idx: usize,
        agg: &TelemetryAggregator,
    ) -> AttentionPattern {
        let pattern = self.detector.detect_from_telemetry(agg);
        self.record_pattern(
            layer_idx,
            agg.softmax_max(),
            agg.softmax_sharpness(),
            pattern,
        );
        pattern
    }

    /// 从 KvPageHeader 检测当前层的注意力模式
    pub fn detect_from_page_header(
        &mut self,
        layer_idx: usize,
        header: &KvPageHeader,
    ) -> AttentionPattern {
        self.detect_for_layer(layer_idx, f16_bits_to_f32(header.softmax_max_avg), f16_bits_to_f32(header.centroid_pos))
    }

    /// 记录注意力模式到历史
    fn record_pattern(
        &mut self,
        layer_idx: usize,
        max_val: f32,
        sharpness: f32,
        pattern: AttentionPattern,
    ) {
        // 更新统计
        match pattern {
            AttentionPattern::Normal => self.normal_count += 1,
            AttentionPattern::Sink => self.sink_count += 1,
            AttentionPattern::SharpFocus => self.sharp_focus_count += 1,
            AttentionPattern::Diffuse => self.diffuse_count += 1,
        }

        // 记录历史
        let record = LayerAttentionRecord {
            layer_idx,
            softmax_max: max_val,
            sharpness,
            pattern,
        };

        if let Some(existing) = self.history.iter_mut().find(|r| r.layer_idx == layer_idx) {
            *existing = record;
        } else {
            self.history.push(record);
        }
    }

    /// 获取某层的注意力模式记录
    pub fn layer_record(&self, layer_idx: usize) -> Option<&LayerAttentionRecord> {
        self.history.iter().find(|r| r.layer_idx == layer_idx)
    }

    /// 获取配置引用
    pub fn config(&self) -> &SinkDetectionConfig {
        &self.detector.config
    }

    /// 是否启用
    pub fn is_enabled(&self) -> bool {
        self.detector.config.sink_threshold > 0.0
    }

    /// 总层数
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// 总检测次数
    pub fn total_detections(&self) -> usize {
        self.normal_count + self.sink_count
            + self.sharp_focus_count + self.diffuse_count
    }

    /// Sink 模式占比
    pub fn sink_ratio(&self) -> f32 {
        if self.total_detections() == 0 {
            return 0.0;
        }
        self.sink_count as f32 / self.total_detections() as f32
    }

    /// SharpFocus 模式占比
    pub fn sharp_focus_ratio(&self) -> f32 {
        if self.total_detections() == 0 {
            return 0.0;
        }
        self.sharp_focus_count as f32 / self.total_detections() as f32
    }

    /// Diffuse 模式占比
    pub fn diffuse_ratio(&self) -> f32 {
        if self.total_detections() == 0 {
            return 0.0;
        }
        self.diffuse_count as f32 / self.total_detections() as f32
    }

    /// 获取保护 Sink 位置列表
    pub fn protected_sink_positions(&self) -> Vec<usize> {
        (0..self.detector.protected_sink_count()).collect()
    }

    /// 判断某个 token 是否需要 Sink 保护 (KIVI FP16)
    pub fn needs_sink_protection(&self, token_position: usize) -> bool {
        self.detector.is_protected_sink(token_position)
    }

    /// 保护 Sink 数量
    pub fn protected_sink_count(&self) -> usize {
        self.detector.protected_sink_count()
    }

    /// 重置统计数据
    pub fn reset_stats(&mut self) {
        self.history.clear();
        self.normal_count = 0;
        self.sink_count = 0;
        self.sharp_focus_count = 0;
        self.diffuse_count = 0;
    }

    /// 获取全部历史
    pub fn history(&self) -> &[LayerAttentionRecord] {
        &self.history
    }

    /// 返回最近 N 层的注意力模式记录
    pub fn recent_patterns(&self, n: usize) -> Vec<&LayerAttentionRecord> {
        self.history.iter().rev().take(n).collect()
    }
}

/// 批次级注意力模式汇总
///
/// 用于 build_batch() 阶段 (§18.4 Dispatch-Time)，
/// 汇总整个 batch 的注意力模式分布。
#[derive(Debug, Clone)]
pub struct BatchAttentionSummary {
    /// batch 中每个请求最近一层的注意力模式
    pub per_request_patterns: Vec<AttentionPattern>,
    /// 平均 Sharpness
    pub avg_sharpness: f32,
    /// Sink 占比
    pub sink_ratio: f32,
    /// 批次级注意力建议
    pub batch_advice: BatchAttentionAdvice,
}

/// 批次级注意力模式建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAttentionAdvice {
    /// 正常注意力 — 无需特殊处理
    NormalAttention,
    /// Sink 模式为主 — 启用 Sink 保护 + KIVI FP16
    SinkDominant,
    /// 分散注意力 — 考虑启用推测解码
    DiffuseAttention,
}

impl BatchAttentionSummary {
    /// 从多个注意力模式生成批次汇总
    pub fn from_patterns(
        patterns: &[AttentionPattern],
        avg_sharpness: f32,
    ) -> Self {
        let total = patterns.len();
        if total == 0 {
            return Self {
                per_request_patterns: vec![],
                avg_sharpness: 0.0,
                sink_ratio: 0.0,
                batch_advice: BatchAttentionAdvice::NormalAttention,
            };
        }

        let sink_count = patterns
            .iter()
            .filter(|&&p| p == AttentionPattern::Sink)
            .count();
        let diffuse_count = patterns
            .iter()
            .filter(|&&p| p == AttentionPattern::Diffuse)
            .count();

        let sink_ratio = sink_count as f32 / total as f32;

        // 批次建议规则:
        // - >30% Sink → SinkDominant (启用 KIVI Sink 保护)
        // - >40% Diffuse → DiffuseAttention (考虑推测解码)
        // - 其余 → NormalAttention
        let batch_advice = if sink_count * 10 > total * 3 {
            BatchAttentionAdvice::SinkDominant
        } else if diffuse_count * 10 > total * 4 {
            BatchAttentionAdvice::DiffuseAttention
        } else {
            BatchAttentionAdvice::NormalAttention
        };

        Self {
            per_request_patterns: patterns.to_vec(),
            avg_sharpness,
            sink_ratio,
            batch_advice,
        }
    }

    /// 是否需要 Sink 保护 (KIVI FP16)
    pub fn needs_sink_protection(&self) -> bool {
        matches!(self.batch_advice, BatchAttentionAdvice::SinkDominant)
    }

    /// 是否适合推测解码
    pub fn benefits_from_speculation(&self) -> bool {
        matches!(self.batch_advice, BatchAttentionAdvice::DiffuseAttention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sink_tracker_basic() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        // Layer 0: Normal attention
        let pattern = tracker.detect_for_layer(0, 0.3, 0.4);
        assert_eq!(pattern, AttentionPattern::Normal);

        // Layer 5: Sink attention
        let pattern = tracker.detect_for_layer(5, 0.95, 0.9);
        assert_eq!(pattern, AttentionPattern::Sink);

        // Layer 10: SharpFocus
        let pattern = tracker.detect_for_layer(10, 0.5, 0.85);
        assert_eq!(pattern, AttentionPattern::SharpFocus);

        // Layer 15: Diffuse
        let pattern = tracker.detect_for_layer(15, 0.1, 0.05);
        assert_eq!(pattern, AttentionPattern::Diffuse);

        assert_eq!(tracker.normal_count, 1);
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.sharp_focus_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
    }

    #[test]
    fn test_sink_tracker_from_aggregator() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.92,
            sharpness: 0.88,
        });

        let pattern = tracker.detect_from_aggregator(5, &agg);
        assert_eq!(pattern, AttentionPattern::Sink);
        assert_eq!(tracker.sink_count, 1);
    }

    #[test]
    fn test_sink_tracker_from_page_header() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(2.5);
        header.centroid_pos = f32_to_f16_bits(0.85);
        header.softmax_max_avg = f32_to_f16_bits(0.91);
        header.delta_rho_avg = f32_to_f16_bits(0.998);
        header.dead_ratio = 25;

        let pattern = tracker.detect_from_page_header(10, &header);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_sink_protection() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 32);

        assert!(tracker.needs_sink_protection(0));
        assert!(tracker.needs_sink_protection(3));
        assert!(!tracker.needs_sink_protection(4));
        assert!(!tracker.needs_sink_protection(100));
        assert_eq!(tracker.protected_sink_count(), 4);
        assert_eq!(tracker.protected_sink_positions(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sink_tracker_ratios() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        // 2 Normal, 1 Sink, 1 Diffuse
        tracker.detect_for_layer(0, 0.3, 0.4);  // Normal
        tracker.detect_for_layer(5, 0.3, 0.5);  // Normal
        tracker.detect_for_layer(10, 0.95, 0.9); // Sink
        tracker.detect_for_layer(15, 0.1, 0.05); // Diffuse

        assert!((tracker.sink_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.diffuse_ratio() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_batch_attention_summary_normal() {
        let patterns = vec![
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_attention_summary_sink_dominant() {
        // >30% Sink → SinkDominant
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.7);
        // 2/5 = 40% > 30% → SinkDominant
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!(summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_attention_summary_diffuse() {
        // >40% Diffuse → DiffuseAttention
        let patterns = vec![
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.05);
        // 3/5 = 60% > 40% → DiffuseAttention
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
        assert!(summary.benefits_from_speculation());
        assert!(!summary.needs_sink_protection());
    }

    #[test]
    fn test_batch_attention_summary_empty() {
        let summary = BatchAttentionSummary::from_patterns(&[], 0.0);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert_eq!(summary.avg_sharpness, 0.0);
    }

    #[test]
    fn test_sink_tracker_history_tracking() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(10, 0.3, 0.4);
        assert_eq!(tracker.history().len(), 2);

        // Update layer 5
        tracker.detect_for_layer(5, 0.3, 0.4);
        assert_eq!(tracker.history().len(), 2);

        let record = tracker.layer_record(5).unwrap();
        assert_eq!(record.pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_recent_patterns() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(10, 0.3, 0.4);
        tracker.detect_for_layer(15, 0.1, 0.05);

        let recent = tracker.recent_patterns(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].layer_idx, 15);
        assert_eq!(recent[1].layer_idx, 10);
    }

    #[test]
    fn test_reset_stats() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(10, 0.3, 0.4);
        assert_eq!(tracker.total_detections(), 2);

        tracker.reset_stats();
        assert_eq!(tracker.total_detections(), 0);
        assert!(tracker.history().is_empty());
        assert!(tracker.is_enabled());
    }

    // ========================================================================
    // LayerAttentionRecord tests
    // ========================================================================

    #[test]
    fn test_layer_attention_record_debug_clone() {
        let record = LayerAttentionRecord {
            layer_idx: 7,
            softmax_max: 0.92,
            sharpness: 0.85,
            pattern: AttentionPattern::Sink,
        };
        let cloned = record.clone();
        assert_eq!(cloned.layer_idx, 7);
        assert_eq!(cloned.softmax_max, 0.92);
        assert_eq!(cloned.sharpness, 0.85);
        assert_eq!(cloned.pattern, AttentionPattern::Sink);

        let debug_str = format!("{:?}", record);
        assert!(debug_str.contains("layer_idx: 7"));
        assert!(debug_str.contains("Sink"));
    }

    #[test]
    fn test_layer_attention_record_all_patterns() {
        let patterns = [
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
        ];
        for (i, &pattern) in patterns.iter().enumerate() {
            let record = LayerAttentionRecord {
                layer_idx: i,
                softmax_max: 0.5,
                sharpness: 0.5,
                pattern,
            };
            assert_eq!(record.pattern, pattern);
            assert_eq!(record.layer_idx, i);
        }
    }

    // ========================================================================
    // SinkTracker construction and basic state
    // ========================================================================

    #[test]
    fn test_sink_tracker_new_initial_state() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 16);

        assert_eq!(tracker.num_layers(), 16);
        assert_eq!(tracker.total_detections(), 0);
        assert!(tracker.is_enabled());
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.sink_ratio(), 0.0);
        assert_eq!(tracker.sharp_focus_ratio(), 0.0);
        assert_eq!(tracker.diffuse_ratio(), 0.0);
    }

    #[test]
    fn test_sink_tracker_disabled_when_threshold_zero() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.0,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_sink_tracker_enabled_when_threshold_positive() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.001,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_config_returns_reference() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.75,
            protected_sink_count: 8,
            sharp_focus_threshold: 0.7,
            diffuse_threshold: 0.2,
        };
        let tracker = SinkTracker::new(config, 4);
        let cfg = tracker.config();
        assert!((cfg.sink_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(cfg.protected_sink_count, 8);
    }

    // ========================================================================
    // SinkTracker::detect_for_layer — classification boundaries
    // ========================================================================

    #[test]
    fn test_detect_for_layer_sink_boundary_exact_threshold() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);

        // max_val == threshold → NOT Sink (strictly greater)
        let pattern = tracker.detect_for_layer(0, 0.9, 0.5);
        assert_eq!(pattern, AttentionPattern::Normal);

        // max_val > threshold → Sink
        let pattern = tracker.detect_for_layer(1, 0.9001, 0.5);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_for_layer_sharp_focus_boundary() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);

        // sharpness == threshold → NOT SharpFocus (strictly greater)
        let pattern = tracker.detect_for_layer(0, 0.5, 0.8);
        assert_eq!(pattern, AttentionPattern::Normal);

        // sharpness > threshold → SharpFocus
        let pattern = tracker.detect_for_layer(1, 0.5, 0.81);
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_detect_for_layer_diffuse_boundary() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);

        // sharpness == threshold → NOT Diffuse (strictly less)
        let pattern = tracker.detect_for_layer(0, 0.5, 0.1);
        assert_eq!(pattern, AttentionPattern::Normal);

        // sharpness < threshold → Diffuse
        let pattern = tracker.detect_for_layer(1, 0.5, 0.09);
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_for_layer_sink_takes_precedence_over_sharp_focus() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // max_val > sink_threshold AND sharpness > sharp_focus_threshold
        // Sink should win because it's checked first
        let pattern = tracker.detect_for_layer(0, 0.95, 0.95);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_for_layer_sink_takes_precedence_over_diffuse() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // max_val > sink_threshold but sharpness < diffuse_threshold
        // Sink still wins
        let pattern = tracker.detect_for_layer(0, 0.95, 0.05);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_for_layer_normal_in_middle_range() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // sharpness between diffuse_threshold and sharp_focus_threshold,
        // max_val below sink_threshold → Normal
        let pattern = tracker.detect_for_layer(0, 0.5, 0.5);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_for_layer_zero_values() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // Both zero: max=0 < 0.9, sharpness=0 < 0.1 → Diffuse
        let pattern = tracker.detect_for_layer(0, 0.0, 0.0);
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_for_layer_negative_sharpness() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // Negative sharpness < diffuse_threshold → Diffuse
        let pattern = tracker.detect_for_layer(0, 0.5, -0.5);
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_for_layer_negative_max_val() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // Negative max_val < sink_threshold, sharpness in normal range → Normal
        let pattern = tracker.detect_for_layer(0, -1.0, 0.5);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_for_layer_very_high_max_and_sharpness() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        let pattern = tracker.detect_for_layer(0, 1.0, 1.0);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    // ========================================================================
    // SinkTracker history and layer_record
    // ========================================================================

    #[test]
    fn test_history_order_matches_insertion() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        tracker.detect_for_layer(3, 0.5, 0.5);
        tracker.detect_for_layer(7, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.3, 0.05);

        let history = tracker.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].layer_idx, 3);
        assert_eq!(history[1].layer_idx, 7);
        assert_eq!(history[2].layer_idx, 1);
    }

    #[test]
    fn test_layer_record_returns_correct_entry() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);

        tracker.detect_for_layer(5, 0.92, 0.88);
        tracker.detect_for_layer(10, 0.3, 0.4);

        let record = tracker.layer_record(5).unwrap();
        assert_eq!(record.layer_idx, 5);
        assert!((record.softmax_max - 0.92).abs() < 0.01);
        assert!((record.sharpness - 0.88).abs() < 0.01);
        assert_eq!(record.pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_layer_record_missing_layer_returns_none() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);

        tracker.detect_for_layer(0, 0.5, 0.5);
        assert!(tracker.layer_record(99).is_none());
    }

    #[test]
    fn test_layer_record_updates_on_redetection() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);

        // First detection: Sink
        tracker.detect_for_layer(5, 0.95, 0.9);
        assert_eq!(tracker.layer_record(5).unwrap().pattern, AttentionPattern::Sink);

        // Second detection for same layer: Normal
        tracker.detect_for_layer(5, 0.3, 0.4);
        assert_eq!(tracker.layer_record(5).unwrap().pattern, AttentionPattern::Normal);
        assert_eq!(tracker.layer_record(5).unwrap().softmax_max, 0.3);

        // History should still have exactly 1 entry for layer 5
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_recent_patterns_fewer_than_n() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);

        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.5, 0.5);

        // Request 10 but only 2 exist
        let recent = tracker.recent_patterns(10);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_recent_patterns_zero_n() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);

        tracker.detect_for_layer(0, 0.5, 0.5);
        let recent = tracker.recent_patterns(0);
        assert!(recent.is_empty());
    }

    #[test]
    fn test_recent_patterns_empty_history() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 16);
        let recent = tracker.recent_patterns(5);
        assert!(recent.is_empty());
    }

    // ========================================================================
    // SinkTracker statistics and ratios
    // ========================================================================

    #[test]
    fn test_total_detections_accumulates() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        assert_eq!(tracker.total_detections(), 0);

        tracker.detect_for_layer(0, 0.5, 0.5);
        assert_eq!(tracker.total_detections(), 1);

        tracker.detect_for_layer(1, 0.5, 0.5);
        assert_eq!(tracker.total_detections(), 2);

        // Re-detect for layer 0 → still increments
        tracker.detect_for_layer(0, 0.95, 0.9);
        assert_eq!(tracker.total_detections(), 3);
    }

    #[test]
    fn test_ratios_zero_when_no_detections() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 16);
        assert_eq!(tracker.sink_ratio(), 0.0);
        assert_eq!(tracker.sharp_focus_ratio(), 0.0);
        assert_eq!(tracker.diffuse_ratio(), 0.0);
    }

    #[test]
    fn test_sharp_focus_ratio() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        // 1 Normal + 1 SharpFocus + 2 Sink = 4 total
        tracker.detect_for_layer(0, 0.3, 0.4);        // Normal
        tracker.detect_for_layer(1, 0.5, 0.85);        // SharpFocus
        tracker.detect_for_layer(2, 0.95, 0.9);        // Sink
        tracker.detect_for_layer(3, 0.95, 0.9);        // Sink

        assert!((tracker.sharp_focus_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.sink_ratio() - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_all_ratios_sum_to_one() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 32);

        tracker.detect_for_layer(0, 0.3, 0.4);        // Normal
        tracker.detect_for_layer(1, 0.95, 0.9);        // Sink
        tracker.detect_for_layer(2, 0.5, 0.85);        // SharpFocus
        tracker.detect_for_layer(3, 0.1, 0.05);        // Diffuse

        let sum = tracker.sink_ratio()
            + tracker.sharp_focus_ratio()
            + tracker.diffuse_ratio()
            + (1.0 - tracker.sink_ratio() - tracker.sharp_focus_ratio() - tracker.diffuse_ratio());
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_all_same_pattern() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        for i in 0..8 {
            tracker.detect_for_layer(i, 0.95, 0.9); // All Sink
        }

        assert_eq!(tracker.total_detections(), 8);
        assert!((tracker.sink_ratio() - 1.0).abs() < 0.01);
        assert!((tracker.sharp_focus_ratio() - 0.0).abs() < 0.01);
        assert!((tracker.diffuse_ratio() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_reset_stats_preserves_num_layers_and_enabled() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 24);

        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.3, 0.4);
        assert_eq!(tracker.total_detections(), 2);

        tracker.reset_stats();

        assert_eq!(tracker.num_layers(), 24);
        assert!(tracker.is_enabled());
        assert_eq!(tracker.total_detections(), 0);
        assert_eq!(tracker.sink_ratio(), 0.0);
        assert_eq!(tracker.sharp_focus_ratio(), 0.0);
        assert_eq!(tracker.diffuse_ratio(), 0.0);
    }

    #[test]
    fn test_reset_stats_then_detect_again() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();

        // After reset, new detections should work correctly
        tracker.detect_for_layer(5, 0.95, 0.9);
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.history()[0].layer_idx, 5);
    }

    // ========================================================================
    // Sink protection
    // ========================================================================

    #[test]
    fn test_sink_protection_default_count() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 8);

        assert_eq!(tracker.protected_sink_count(), 4);
        assert_eq!(tracker.protected_sink_positions(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sink_protection_custom_count() {
        let config = SinkDetectionConfig {
            protected_sink_count: 2,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);

        assert_eq!(tracker.protected_sink_count(), 2);
        assert_eq!(tracker.protected_sink_positions(), vec![0, 1]);
        assert!(tracker.needs_sink_protection(0));
        assert!(tracker.needs_sink_protection(1));
        assert!(!tracker.needs_sink_protection(2));
    }

    #[test]
    fn test_sink_protection_zero_count() {
        let config = SinkDetectionConfig {
            protected_sink_count: 0,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);

        assert_eq!(tracker.protected_sink_count(), 0);
        assert!(tracker.protected_sink_positions().is_empty());
        assert!(!tracker.needs_sink_protection(0));
    }

    #[test]
    fn test_sink_protection_boundary_position() {
        let config = SinkDetectionConfig {
            protected_sink_count: 4,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);

        // Position 3 is the last protected (0-indexed, < 4)
        assert!(tracker.needs_sink_protection(3));
        // Position 4 is NOT protected
        assert!(!tracker.needs_sink_protection(4));
    }

    // ========================================================================
    // BatchAttentionSummary
    // ========================================================================

    #[test]
    fn test_batch_summary_debug_clone() {
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink],
            avg_sharpness: 0.8,
            sink_ratio: 1.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        let cloned = summary.clone();
        assert_eq!(cloned.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert_eq!(cloned.per_request_patterns.len(), 1);

        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("SinkDominant"));
    }

    #[test]
    fn test_batch_advice_variants_equality() {
        assert_eq!(BatchAttentionAdvice::NormalAttention, BatchAttentionAdvice::NormalAttention);
        assert_eq!(BatchAttentionAdvice::SinkDominant, BatchAttentionAdvice::SinkDominant);
        assert_eq!(BatchAttentionAdvice::DiffuseAttention, BatchAttentionAdvice::DiffuseAttention);

        assert_ne!(BatchAttentionAdvice::NormalAttention, BatchAttentionAdvice::SinkDominant);
        assert_ne!(BatchAttentionAdvice::SinkDominant, BatchAttentionAdvice::DiffuseAttention);
        assert_ne!(BatchAttentionAdvice::DiffuseAttention, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_batch_advice_debug() {
        assert!(format!("{:?}", BatchAttentionAdvice::NormalAttention).contains("NormalAttention"));
        assert!(format!("{:?}", BatchAttentionAdvice::SinkDominant).contains("SinkDominant"));
        assert!(format!("{:?}", BatchAttentionAdvice::DiffuseAttention).contains("DiffuseAttention"));
    }

    #[test]
    fn test_batch_summary_sink_dominant_exact_boundary() {
        // 3 out of 10 = 30% → NOT SinkDominant (need strictly > 30%, i.e. 3*10 > 10*3 → 30 > 30 → false)
        let patterns = vec![
            AttentionPattern::Sink, AttentionPattern::Sink, AttentionPattern::Sink,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_batch_summary_sink_dominant_just_above_boundary() {
        // 4 out of 10 = 40% → SinkDominant (4*10 > 10*3 → 40 > 30)
        let patterns = vec![
            AttentionPattern::Sink, AttentionPattern::Sink, AttentionPattern::Sink, AttentionPattern::Sink,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    #[test]
    fn test_batch_summary_diffuse_exact_boundary() {
        // 4 out of 10 = 40% → NOT Diffuse (need strictly > 40%, 4*10 > 10*4 → 40 > 40 → false)
        let patterns = vec![
            AttentionPattern::Diffuse, AttentionPattern::Diffuse, AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
        ];
        // No sinks so not SinkDominant
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.1);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_batch_summary_diffuse_just_above_boundary() {
        // 5 out of 10 = 50% → Diffuse (5*10 > 10*4 → 50 > 40)
        let patterns = vec![
            AttentionPattern::Diffuse, AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal, AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.1);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
    }

    #[test]
    fn test_batch_summary_sink_takes_precedence_over_diffuse() {
        // Both > 30% sink AND > 40% diffuse → SinkDominant checked first
        let patterns = vec![
            AttentionPattern::Sink, AttentionPattern::Sink, AttentionPattern::Sink, AttentionPattern::Sink,
            AttentionPattern::Diffuse, AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Normal,
        ];
        // 4/10 = 40% sink → SinkDominant (checked first)
        // 5/10 = 50% diffuse → would be DiffuseAttention, but sink check wins
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    #[test]
    fn test_batch_summary_single_pattern_sink() {
        let patterns = vec![AttentionPattern::Sink];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.9);
        // 1/1 = 100% > 30% → SinkDominant
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!(summary.needs_sink_protection());
        assert!((summary.sink_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_single_pattern_normal() {
        let patterns = vec![AttentionPattern::Normal];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
        assert!((summary.sink_ratio - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_preserves_patterns_and_sharpness() {
        let patterns = vec![
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::Diffuse,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.42);
        assert_eq!(summary.per_request_patterns.len(), 3);
        assert!((summary.avg_sharpness - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_sharpfocus_no_special_advice() {
        // All SharpFocus → no SinkDominant or DiffuseAttention → NormalAttention
        let patterns = vec![
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.9);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_mixed_patterns_no_threshold_crossed() {
        // 1/10 Sink (10%), 2/10 Diffuse (20%), rest Normal
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal, AttentionPattern::Normal, AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.3);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    // ========================================================================
    // SinkTracker with custom config
    // ========================================================================

    #[test]
    fn test_custom_config_affects_detection() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.5,  // Lower threshold
            sharp_focus_threshold: 0.9,
            diffuse_threshold: 0.3,
            protected_sink_count: 2,
        };
        let mut tracker = SinkTracker::new(config, 8);

        // max_val=0.6 > 0.5 → Sink (would be Normal with default 0.9 threshold)
        let pattern = tracker.detect_for_layer(0, 0.6, 0.5);
        assert_eq!(pattern, AttentionPattern::Sink);

        // max_val=0.4, sharpness=0.2 < 0.3 → Diffuse
        let pattern = tracker.detect_for_layer(1, 0.4, 0.2);
        assert_eq!(pattern, AttentionPattern::Diffuse);

        assert_eq!(tracker.protected_sink_count(), 2);
        assert!(tracker.needs_sink_protection(1));
        assert!(!tracker.needs_sink_protection(2));
    }

    #[test]
    fn test_tracker_with_single_layer() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 1);

        let pattern = tracker.detect_for_layer(0, 0.95, 0.9);
        assert_eq!(pattern, AttentionPattern::Sink);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.num_layers(), 1);
    }

    #[test]
    fn test_tracker_with_many_layers() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 128);

        for i in 0..128 {
            tracker.detect_for_layer(i, 0.5, 0.5);
        }
        assert_eq!(tracker.total_detections(), 128);
        assert_eq!(tracker.history().len(), 128);
        assert_eq!(tracker.num_layers(), 128);
    }

    // ========================================================================
    // detect_from_page_header tests
    // ========================================================================

    #[test]
    fn test_detect_from_page_header_normal() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut header = KvPageHeader::new(0);
        // softmax_max_avg = 0.5 → not Sink; centroid_pos (used as sharpness) = 0.5 → Normal
        header.softmax_max_avg = f32_to_f16_bits(0.5);
        header.centroid_pos = f32_to_f16_bits(0.5);

        let pattern = tracker.detect_from_page_header(3, &header);
        assert_eq!(pattern, AttentionPattern::Normal);
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_detect_from_page_header_updates_history() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.95);
        header.centroid_pos = f32_to_f16_bits(0.9);

        tracker.detect_from_page_header(7, &header);

        let record = tracker.layer_record(7).unwrap();
        assert_eq!(record.layer_idx, 7);
        assert_eq!(record.pattern, AttentionPattern::Sink);
    }

    // ========================================================================
    // detect_from_aggregator tests
    // ========================================================================

    #[test]
    fn test_detect_from_aggregator_records_history() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.5,
        });

        tracker.detect_from_aggregator(2, &agg);

        let record = tracker.layer_record(2).unwrap();
        assert_eq!(record.layer_idx, 2);
        assert_eq!(record.pattern, AttentionPattern::Normal);
        assert!((record.softmax_max - 0.3).abs() < 0.01);
        assert!((record.sharpness - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_detect_from_aggregator_updates_existing_layer() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        // First: Sink
        let mut agg1 = TelemetryAggregator::new();
        agg1.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.9,
        });
        tracker.detect_from_aggregator(5, &agg1);

        // Second: Diffuse
        let mut agg2 = TelemetryAggregator::new();
        agg2.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.1,
            sharpness: 0.05,
        });
        tracker.detect_from_aggregator(5, &agg2);

        let record = tracker.layer_record(5).unwrap();
        assert_eq!(record.pattern, AttentionPattern::Diffuse);
        assert_eq!(tracker.history().len(), 1);
    }

    // ========================================================================
    // Edge cases: multiple updates to same layer
    // ========================================================================

    #[test]
    fn test_multiple_updates_same_layer_accumulates_counts() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // Same layer detected 5 times, alternating patterns
        tracker.detect_for_layer(0, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(0, 0.3, 0.4);    // Normal
        tracker.detect_for_layer(0, 0.5, 0.85);   // SharpFocus
        tracker.detect_for_layer(0, 0.1, 0.05);   // Diffuse
        tracker.detect_for_layer(0, 0.95, 0.9);  // Sink

        // Only 1 layer in history but 5 detections total
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.total_detections(), 5);
        assert_eq!(tracker.sink_count, 2);
        assert_eq!(tracker.normal_count, 1);
        assert_eq!(tracker.sharp_focus_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
    }

    #[test]
    fn test_history_capacity_grows_beyond_initial() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        // Initial capacity is 4, but detect for 8 layers
        for i in 0..8 {
            tracker.detect_for_layer(i, 0.5, 0.5);
        }
        assert_eq!(tracker.history().len(), 8);
    }

    // ========================================================================
    // AttentionPattern enum completeness
    // ========================================================================

    #[test]
    fn test_attention_pattern_equality() {
        assert_eq!(AttentionPattern::Normal, AttentionPattern::Normal);
        assert_eq!(AttentionPattern::Sink, AttentionPattern::Sink);
        assert_eq!(AttentionPattern::SharpFocus, AttentionPattern::SharpFocus);
        assert_eq!(AttentionPattern::Diffuse, AttentionPattern::Diffuse);

        assert_ne!(AttentionPattern::Normal, AttentionPattern::Sink);
        assert_ne!(AttentionPattern::Sink, AttentionPattern::SharpFocus);
        assert_ne!(AttentionPattern::SharpFocus, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_attention_pattern_debug() {
        let debug_normal = format!("{:?}", AttentionPattern::Normal);
        let debug_sink = format!("{:?}", AttentionPattern::Sink);
        let debug_sharp = format!("{:?}", AttentionPattern::SharpFocus);
        let debug_diffuse = format!("{:?}", AttentionPattern::Diffuse);

        assert!(debug_normal.contains("Normal"));
        assert!(debug_sink.contains("Sink"));
        assert!(debug_sharp.contains("SharpFocus"));
        assert!(debug_diffuse.contains("Diffuse"));
    }

    #[test]
    fn test_attention_pattern_copy_semantics() {
        let p1 = AttentionPattern::Sink;
        let p2 = p1; // Copy
        assert_eq!(p1, p2); // p1 still valid (Copy trait)
    }

    #[test]
    fn test_attention_pattern_clone() {
        let p1 = AttentionPattern::SharpFocus;
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    // ========================================================================
    // SinkDetectionConfig field-level tests
    // ========================================================================

    #[test]
    fn test_config_default_values() {
        let config = SinkDetectionConfig::default();
        assert!((config.sink_threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.protected_sink_count, 4);
        assert!((config.sharp_focus_threshold - 0.8).abs() < f32::EPSILON);
        assert!((config.diffuse_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_debug_format() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.5,
            protected_sink_count: 2,
            sharp_focus_threshold: 0.7,
            diffuse_threshold: 0.3,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("sink_threshold"));
        assert!(debug.contains("protected_sink_count"));
        assert!(debug.contains("sharp_focus_threshold"));
        assert!(debug.contains("diffuse_threshold"));
    }

    #[test]
    fn test_config_clone_is_independent() {
        let config = SinkDetectionConfig::default();
        let cloned = config.clone();
        // Same values
        assert!((cloned.sink_threshold - config.sink_threshold).abs() < f32::EPSILON);
        assert_eq!(cloned.protected_sink_count, config.protected_sink_count);
    }

    #[test]
    fn test_config_with_zero_all_thresholds() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.0,
            protected_sink_count: 0,
            sharp_focus_threshold: 0.0,
            diffuse_threshold: 0.0,
        };
        let tracker = SinkTracker::new(config, 4);
        // sink_threshold == 0 means disabled
        assert!(!tracker.is_enabled());
        assert_eq!(tracker.protected_sink_count(), 0);
    }

    #[test]
    fn test_config_with_large_protected_count() {
        let config = SinkDetectionConfig {
            protected_sink_count: 1000,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 32);
        assert_eq!(tracker.protected_sink_count(), 1000);
        assert!(tracker.needs_sink_protection(999));
        assert!(!tracker.needs_sink_protection(1000));
    }

    #[test]
    fn test_config_with_high_thresholds() {
        let config = SinkDetectionConfig {
            sink_threshold: 2.0,
            sharp_focus_threshold: 5.0,
            diffuse_threshold: -1.0,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // No realistic max_val > 2.0, so never Sink
        let pattern = tracker.detect_for_layer(0, 1.0, 0.0);
        // sharpness=0.0, diffuse_threshold=-1.0, 0.0 > -1.0 → not Diffuse
        // 0.0 not > 5.0 → not SharpFocus → Normal
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    // ========================================================================
    // LayerAttentionRecord edge cases
    // ========================================================================

    #[test]
    fn test_layer_record_with_zero_layer_idx() {
        let record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: 0.0,
            sharpness: 0.0,
            pattern: AttentionPattern::Diffuse,
        };
        assert_eq!(record.layer_idx, 0);
        assert_eq!(record.softmax_max, 0.0);
        assert_eq!(record.sharpness, 0.0);
    }

    #[test]
    fn test_layer_record_with_max_values() {
        let record = LayerAttentionRecord {
            layer_idx: usize::MAX,
            softmax_max: f32::MAX,
            sharpness: f32::MAX,
            pattern: AttentionPattern::Sink,
        };
        assert_eq!(record.layer_idx, usize::MAX);
        assert!(record.softmax_max.is_finite());
        assert!(record.sharpness.is_finite());
    }

    #[test]
    fn test_layer_record_with_negative_sharpness() {
        let record = LayerAttentionRecord {
            layer_idx: 5,
            softmax_max: -0.5,
            sharpness: -1.0,
            pattern: AttentionPattern::Diffuse,
        };
        assert!((record.softmax_max - (-0.5)).abs() < f32::EPSILON);
        assert!((record.sharpness - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layer_record_clone_preserves_all_fields() {
        let record = LayerAttentionRecord {
            layer_idx: 42,
            softmax_max: 0.73,
            sharpness: 0.61,
            pattern: AttentionPattern::SharpFocus,
        };
        let cloned = record.clone();
        assert_eq!(cloned.layer_idx, 42);
        assert!((cloned.softmax_max - 0.73).abs() < f32::EPSILON);
        assert!((cloned.sharpness - 0.61).abs() < f32::EPSILON);
        assert_eq!(cloned.pattern, AttentionPattern::SharpFocus);
    }

    // ========================================================================
    // SinkTracker::new with edge-case num_layers
    // ========================================================================

    #[test]
    fn test_new_with_zero_layers() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 0);
        assert_eq!(tracker.num_layers(), 0);
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.total_detections(), 0);
    }

    #[test]
    fn test_new_with_large_num_layers() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 100_000);
        assert_eq!(tracker.num_layers(), 100_000);
        assert!(tracker.history().is_empty());
    }

    // ========================================================================
    // SinkTracker detection edge cases
    // ========================================================================

    #[test]
    fn test_detect_f32_epsilon_above_sink_threshold() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Just barely above threshold
        let pattern = tracker.detect_for_layer(0, 0.9 + f32::EPSILON, 0.5);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_f32_epsilon_below_sink_threshold() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        let pattern = tracker.detect_for_layer(0, 0.9 - f32::EPSILON, 0.5);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_sink_with_zero_sharpness() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // max_val > 0.9 → Sink regardless of sharpness
        let pattern = tracker.detect_for_layer(0, 0.95, 0.0);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_sharp_focus_with_low_max_val() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // max_val=0.3 < 0.9, sharpness=0.9 > 0.8 → SharpFocus
        let pattern = tracker.detect_for_layer(0, 0.3, 0.9);
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_detect_diffuse_with_high_max_val_not_sink() {
        let config = SinkDetectionConfig {
            sink_threshold: 1.0, // Raise high so max_val=0.95 < threshold
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        let pattern = tracker.detect_for_layer(0, 0.95, 0.05);
        // max_val=0.95 < 1.0 → not Sink; sharpness=0.05 < 0.1 → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_all_patterns_in_sequence() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        let cases = [
            (0.95, 0.9, AttentionPattern::Sink),
            (0.5, 0.85, AttentionPattern::SharpFocus),
            (0.5, 0.05, AttentionPattern::Diffuse),
            (0.5, 0.5, AttentionPattern::Normal),
        ];
        for (i, (max, sharp, expected)) in cases.iter().enumerate() {
            let result = tracker.detect_for_layer(i, *max, *sharp);
            assert_eq!(result, *expected, "Layer {} failed", i);
        }
        assert_eq!(tracker.total_detections(), 4);
    }

    // ========================================================================
    // SinkTracker::protected_sink_positions edge cases
    // ========================================================================

    #[test]
    fn test_protected_positions_are_sequential_from_zero() {
        let config = SinkDetectionConfig {
            protected_sink_count: 5,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        let positions = tracker.protected_sink_positions();
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_protected_positions_large_count() {
        let config = SinkDetectionConfig {
            protected_sink_count: 100,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        let positions = tracker.protected_sink_positions();
        assert_eq!(positions.len(), 100);
        assert_eq!(positions[0], 0);
        assert_eq!(positions[99], 99);
    }

    #[test]
    fn test_needs_protection_usize_max() {
        let config = SinkDetectionConfig::default();
        let tracker = SinkTracker::new(config, 8);
        assert!(!tracker.needs_sink_protection(usize::MAX));
    }

    // ========================================================================
    // SinkTracker::history and recent_patterns edge cases
    // ========================================================================

    #[test]
    fn test_recent_patterns_single_entry() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(5, 0.95, 0.9);

        let recent = tracker.recent_patterns(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].layer_idx, 5);
    }

    #[test]
    fn test_recent_patterns_n_equals_history_len() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.5, 0.5);
        tracker.detect_for_layer(2, 0.5, 0.5);

        let recent = tracker.recent_patterns(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_history_empty_after_reset() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.5, 0.5);
        assert_eq!(tracker.history().len(), 2);

        tracker.reset_stats();
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_layer_record_after_reset_then_redetect() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        tracker.detect_for_layer(3, 0.95, 0.9);
        assert!(tracker.layer_record(3).is_some());

        tracker.reset_stats();
        assert!(tracker.layer_record(3).is_none());

        tracker.detect_for_layer(3, 0.3, 0.4);
        let record = tracker.layer_record(3).unwrap();
        assert_eq!(record.pattern, AttentionPattern::Normal);
    }

    // ========================================================================
    // SinkTracker ratio edge cases
    // ========================================================================

    #[test]
    fn test_sink_ratio_single_detection_sink() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        assert!((tracker.sink_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sharp_focus_ratio_single_detection() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.5, 0.85);
        assert!((tracker.sharp_focus_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_diffuse_ratio_single_detection() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.1, 0.05);
        assert!((tracker.diffuse_ratio() - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // BatchAttentionSummary edge cases
    // ========================================================================

    #[test]
    fn test_batch_summary_single_pattern_sharp_focus() {
        let patterns = vec![AttentionPattern::SharpFocus];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.9);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!((summary.sink_ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_single_pattern_diffuse() {
        let patterns = vec![AttentionPattern::Diffuse];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.05);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
        assert!(summary.benefits_from_speculation());
        assert!((summary.sink_ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_all_sink() {
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Sink,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.95);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!(summary.needs_sink_protection());
        assert!((summary.sink_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_all_diffuse() {
        let patterns = vec![
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.02);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
        assert!(summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_all_normal() {
        let patterns = vec![
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_negative_avg_sharpness() {
        let patterns = vec![AttentionPattern::Normal];
        let summary = BatchAttentionSummary::from_patterns(&patterns, -0.5);
        assert!((summary.avg_sharpness - (-0.5)).abs() < f32::EPSILON);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_batch_summary_preserves_pattern_order() {
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Normal,
            AttentionPattern::Diffuse,
            AttentionPattern::SharpFocus,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert_eq!(summary.per_request_patterns[0], AttentionPattern::Sink);
        assert_eq!(summary.per_request_patterns[1], AttentionPattern::Normal);
        assert_eq!(summary.per_request_patterns[2], AttentionPattern::Diffuse);
        assert_eq!(summary.per_request_patterns[3], AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_batch_summary_two_patterns_equal() {
        // 1 Sink + 1 Diffuse (total=2): sink=50% > 30% → SinkDominant
        let patterns = vec![AttentionPattern::Sink, AttentionPattern::Diffuse];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    // ========================================================================
    // BatchAttentionAdvice exhaustive variant checks
    // ========================================================================

    #[test]
    fn test_batch_advice_all_variants_exhaustive() {
        // Ensure all 3 variants are covered
        let variants = [
            BatchAttentionAdvice::NormalAttention,
            BatchAttentionAdvice::SinkDominant,
            BatchAttentionAdvice::DiffuseAttention,
        ];
        // Verify each variant is distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn test_batch_advice_copy_semantics() {
        let a = BatchAttentionAdvice::SinkDominant;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_batch_advice_clone() {
        let a = BatchAttentionAdvice::DiffuseAttention;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ========================================================================
    // BatchAttentionSummary struct construction
    // ========================================================================

    #[test]
    fn test_batch_summary_direct_construction() {
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::NormalAttention,
        };
        assert!(summary.per_request_patterns.is_empty());
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_direct_construction_sink() {
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink],
            avg_sharpness: 0.9,
            sink_ratio: 1.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        assert!(summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_direct_construction_diffuse() {
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Diffuse],
            avg_sharpness: 0.05,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::DiffuseAttention,
        };
        assert!(!summary.needs_sink_protection());
        assert!(summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_summary_clone_independence() {
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink, AttentionPattern::Normal],
            avg_sharpness: 0.6,
            sink_ratio: 0.5,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        let cloned = summary.clone();
        assert_eq!(cloned.per_request_patterns.len(), 2);
        assert!((cloned.avg_sharpness - 0.6).abs() < f32::EPSILON);
        assert!((cloned.sink_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(cloned.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    // ========================================================================
    // SinkTracker reset + re-detect comprehensive
    // ========================================================================

    #[test]
    fn test_reset_clears_all_pattern_counts() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        tracker.detect_for_layer(0, 0.95, 0.9);   // Sink
        tracker.detect_for_layer(1, 0.5, 0.85);    // SharpFocus
        tracker.detect_for_layer(2, 0.1, 0.05);    // Diffuse
        tracker.detect_for_layer(3, 0.3, 0.4);     // Normal

        tracker.reset_stats();
        assert_eq!(tracker.normal_count, 0);
        assert_eq!(tracker.sink_count, 0);
        assert_eq!(tracker.sharp_focus_count, 0);
        assert_eq!(tracker.diffuse_count, 0);
    }

    #[test]
    fn test_double_reset_is_noop() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);

        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();
        tracker.reset_stats();

        assert_eq!(tracker.total_detections(), 0);
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.num_layers(), 4);
    }

    // ========================================================================
    // SinkTracker config immutability through tracker
    // ========================================================================

    #[test]
    fn test_config_reflects_initial_state() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.6,
            protected_sink_count: 7,
            sharp_focus_threshold: 0.5,
            diffuse_threshold: 0.15,
        };
        let tracker = SinkTracker::new(config, 10);
        let cfg = tracker.config();
        assert!((cfg.sink_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(cfg.protected_sink_count, 7);
        assert!((cfg.sharp_focus_threshold - 0.5).abs() < f32::EPSILON);
        assert!((cfg.diffuse_threshold - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_unchanged_after_detections() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);

        let cfg = tracker.config();
        assert!((cfg.sink_threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(cfg.protected_sink_count, 4);
    }

    // ========================================================================
    // detect_from_page_header with edge-case f16 values
    // ========================================================================

    #[test]
    fn test_detect_from_page_header_with_zero_values() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.0);
        header.centroid_pos = f32_to_f16_bits(0.0);

        let pattern = tracker.detect_from_page_header(0, &header);
        // max=0.0 < 0.9, sharpness=0.0 < 0.1 → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_from_page_header_with_negative_sharpness() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.5);
        header.centroid_pos = f32_to_f16_bits(-0.5);

        let pattern = tracker.detect_from_page_header(0, &header);
        // max=0.5 < 0.9, sharpness=-0.5 < 0.1 → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    // ========================================================================
    // detect_from_aggregator with different signal values
    // ========================================================================

    #[test]
    fn test_detect_from_aggregator_sharp_focus() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.9,
        });

        let pattern = tracker.detect_from_aggregator(3, &agg);
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_detect_from_aggregator_diffuse() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.05,
        });

        let pattern = tracker.detect_from_aggregator(3, &agg);
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    // ========================================================================
    // Cross-method interaction tests
    // ========================================================================

    #[test]
    fn test_detect_for_layer_then_aggregator_different_layers() {
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        tracker.detect_for_layer(0, 0.95, 0.9); // Sink

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.1,
            sharpness: 0.05,
        });
        tracker.detect_from_aggregator(5, &agg); // Diffuse

        assert_eq!(tracker.total_detections(), 2);
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
        assert_eq!(tracker.history().len(), 2);
    }

    #[test]
    fn test_mixed_detect_methods_same_layer() {
        use crate::kv_cache::f32_to_f16_bits;

        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        // First: direct detect → Sink
        tracker.detect_for_layer(3, 0.95, 0.9);
        assert_eq!(tracker.layer_record(3).unwrap().pattern, AttentionPattern::Sink);

        // Second: page header → Normal (overrides)
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.5);
        header.centroid_pos = f32_to_f16_bits(0.5);
        tracker.detect_from_page_header(3, &header);

        assert_eq!(tracker.layer_record(3).unwrap().pattern, AttentionPattern::Normal);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.total_detections(), 2);
    }

    #[test]
    fn test_protected_positions_unchanged_by_detections() {
        let config = SinkDetectionConfig {
            protected_sink_count: 3,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 8);

        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.3, 0.4);

        // Protected positions should not change based on detection results
        assert_eq!(tracker.protected_sink_positions(), vec![0, 1, 2]);
        assert_eq!(tracker.protected_sink_count(), 3);
    }

    // ========================================================================
    // NEW TESTS — 44 additional tests
    // ========================================================================

    // --- SinkTracker::new with various configs ---

    #[test]
    fn test_new_with_default_config_has_correct_thresholds() {
        // Arrange: default config
        let config = SinkDetectionConfig::default();
        // Act
        let tracker = SinkTracker::new(config, 16);
        // Assert: config values match defaults
        assert!((tracker.config().sink_threshold - 0.9).abs() < f32::EPSILON);
        assert!((tracker.config().sharp_focus_threshold - 0.8).abs() < f32::EPSILON);
        assert!((tracker.config().diffuse_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_new_with_extreme_thresholds_sink_never_triggers() {
        // Arrange: sink_threshold so high no f32 can exceed it
        let config = SinkDetectionConfig {
            sink_threshold: f32::MAX,
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
            protected_sink_count: 2,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: even max_val=1.0 cannot exceed f32::MAX
        let pattern = tracker.detect_for_layer(0, 1.0, 0.5);
        // Assert: not Sink
        assert_ne!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_new_with_negative_sink_threshold_is_disabled() {
        // Arrange: negative threshold → is_enabled checks > 0.0, so disabled
        let config = SinkDetectionConfig {
            sink_threshold: -0.1,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 4);
        // Assert: negative threshold means disabled per is_enabled() logic
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_new_history_starts_empty_regardless_of_num_layers() {
        // Arrange
        let config = SinkDetectionConfig::default();
        // Act
        let tracker = SinkTracker::new(config, 64);
        // Assert: history starts empty even with many layers declared
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.total_detections(), 0);
    }

    // --- detect_for_layer: float boundary conditions ---

    #[test]
    fn test_detect_with_nan_max_val() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: NaN comparison with > is always false
        let pattern = tracker.detect_for_layer(0, f32::NAN, 0.5);
        // Assert: NaN > 0.9 is false, 0.5 not > 0.8, 0.5 not < 0.1 → Normal
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_with_nan_sharpness() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: NaN sharpness: NaN > 0.8 false, NaN < 0.1 false → Normal
        let pattern = tracker.detect_for_layer(0, 0.5, f32::NAN);
        // Assert
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_with_both_nan() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, f32::NAN, f32::NAN);
        // Assert: all comparisons false → Normal
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_with_inf_max_val() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: inf > 0.9 → Sink
        let pattern = tracker.detect_for_layer(0, f32::INFINITY, 0.5);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_with_neg_inf_max_val() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: -inf > 0.9 false, 0.5 not > 0.8, not < 0.1 → Normal
        let pattern = tracker.detect_for_layer(0, f32::NEG_INFINITY, 0.5);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    // --- detect_for_layer: sharpness at exact boundaries ---

    #[test]
    fn test_detect_sharpness_exactly_at_diffuse_threshold() {
        // Arrange: sharpness == diffuse_threshold → NOT Diffuse (strictly less)
        let config = SinkDetectionConfig {
            diffuse_threshold: 0.2,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.5, 0.2);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_sharpness_just_below_diffuse_threshold() {
        // Arrange
        let config = SinkDetectionConfig {
            diffuse_threshold: 0.2,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: 0.199 < 0.2 → Diffuse
        let pattern = tracker.detect_for_layer(0, 0.5, 0.199);
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_sharpness_exactly_at_sharp_focus_threshold() {
        // Arrange: sharpness == sharp_focus_threshold → NOT SharpFocus (strictly greater)
        let config = SinkDetectionConfig {
            sharp_focus_threshold: 0.7,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.5, 0.7);
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    // --- record_pattern: history update semantics ---

    #[test]
    fn test_history_update_preserves_latest_pattern_only() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: 3 updates to same layer
        tracker.detect_for_layer(2, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(2, 0.1, 0.05);   // Diffuse
        tracker.detect_for_layer(2, 0.5, 0.85);   // SharpFocus
        // Assert: latest wins
        let record = tracker.layer_record(2).unwrap();
        assert_eq!(record.pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_history_update_preserves_latest_max_val() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(0, 0.3, 0.4);
        // Assert
        let record = tracker.layer_record(0).unwrap();
        assert!((record.softmax_max - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_history_update_preserves_latest_sharpness() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(0, 0.3, 0.15);
        // Assert
        let record = tracker.layer_record(0).unwrap();
        assert!((record.sharpness - 0.15).abs() < f32::EPSILON);
    }

    // --- total_detections accuracy ---

    #[test]
    fn test_total_detections_with_many_redetections() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 2);
        // Act: 10 detections across 2 layers
        for _ in 0..5 {
            tracker.detect_for_layer(0, 0.5, 0.5);
            tracker.detect_for_layer(1, 0.5, 0.5);
        }
        // Assert
        assert_eq!(tracker.total_detections(), 10);
        assert_eq!(tracker.history().len(), 2);
    }

    #[test]
    fn test_total_detections_after_partial_reset_flow() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.5, 0.5);
        // Act: reset then add more
        tracker.reset_stats();
        tracker.detect_for_layer(2, 0.1, 0.05);
        // Assert
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.diffuse_count, 1);
    }

    // --- ratio edge cases ---

    #[test]
    fn test_normal_ratio_via_subtraction() {
        // Arrange: 2 Sink, 1 Normal, 1 SharpFocus = 4 total
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(1, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(2, 0.3, 0.4);   // Normal
        tracker.detect_for_layer(3, 0.5, 0.85);  // SharpFocus
        // Act
        let normal_ratio = 1.0 - tracker.sink_ratio() - tracker.sharp_focus_ratio() - tracker.diffuse_ratio();
        // Assert
        assert!((normal_ratio - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_ratios_non_negative() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.1, 0.05);
        tracker.detect_for_layer(2, 0.5, 0.85);
        // Act & Assert
        assert!(tracker.sink_ratio() >= 0.0);
        assert!(tracker.sharp_focus_ratio() >= 0.0);
        assert!(tracker.diffuse_ratio() >= 0.0);
    }

    #[test]
    fn test_ratios_never_exceed_one() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        for i in 0..8 {
            let max_val = if i % 2 == 0 { 0.95 } else { 0.1 };
            let sharpness = if i % 2 == 0 { 0.9 } else { 0.05 };
            tracker.detect_for_layer(i, max_val, sharpness);
        }
        // Act & Assert
        assert!(tracker.sink_ratio() <= 1.0);
        assert!(tracker.sharp_focus_ratio() <= 1.0);
        assert!(tracker.diffuse_ratio() <= 1.0);
    }

    // --- recent_patterns: reverse order semantics ---

    #[test]
    fn test_recent_patterns_returns_most_recent_first() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.95, 0.9);
        tracker.detect_for_layer(2, 0.1, 0.05);
        // Act: recent 2 → should be layers 2, 1 (most recent first)
        let recent = tracker.recent_patterns(2);
        // Assert
        assert_eq!(recent[0].layer_idx, 2);
        assert_eq!(recent[1].layer_idx, 1);
    }

    #[test]
    fn test_recent_patterns_n_larger_than_history() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        // Act: request 100 but only 1 exists
        let recent = tracker.recent_patterns(100);
        // Assert
        assert_eq!(recent.len(), 1);
    }

    // --- LayerAttentionRecord: Debug output completeness ---

    #[test]
    fn test_layer_record_debug_contains_all_fields() {
        // Arrange
        let record = LayerAttentionRecord {
            layer_idx: 3,
            softmax_max: 0.75,
            sharpness: 0.6,
            pattern: AttentionPattern::SharpFocus,
        };
        // Act
        let debug = format!("{:?}", record);
        // Assert: all fields present in debug output
        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("softmax_max"));
        assert!(debug.contains("sharpness"));
        assert!(debug.contains("pattern"));
    }

    // --- SinkTracker: layer_record for each pattern type ---

    #[test]
    fn test_layer_record_sink_pattern_stored_correctly() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Assert
        assert_eq!(tracker.layer_record(0).unwrap().pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_layer_record_sharp_focus_pattern_stored_correctly() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.5, 0.85);
        // Assert
        assert_eq!(tracker.layer_record(0).unwrap().pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_layer_record_diffuse_pattern_stored_correctly() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.1, 0.05);
        // Assert
        assert_eq!(tracker.layer_record(0).unwrap().pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_layer_record_normal_pattern_stored_correctly() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.3, 0.4);
        // Assert
        assert_eq!(tracker.layer_record(0).unwrap().pattern, AttentionPattern::Normal);
    }

    // --- BatchAttentionSummary: from_patterns ratio calculation ---

    #[test]
    fn test_batch_summary_sink_ratio_with_mixed_patterns() {
        // Arrange: 3 Sink out of 6 → 50%
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert!((summary.sink_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_sink_ratio_with_no_sinks() {
        // Arrange: no sinks
        let patterns = vec![
            AttentionPattern::Normal,
            AttentionPattern::Diffuse,
            AttentionPattern::SharpFocus,
        ];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        // Assert
        assert!((summary.sink_ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_preserves_all_patterns_count() {
        // Arrange
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Normal,
            AttentionPattern::Diffuse,
            AttentionPattern::SharpFocus,
            AttentionPattern::Sink,
        ];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert_eq!(summary.per_request_patterns.len(), 5);
    }

    // --- BatchAttentionSummary: advice boundary with odd/even counts ---

    #[test]
    fn test_batch_summary_31_percent_sink_is_dominant() {
        // Arrange: 4 out of 13 ≈ 30.8% > 30% → SinkDominant
        // 4*10 > 13*3 → 40 > 39 → true
        let mut patterns = vec![
            AttentionPattern::Sink, AttentionPattern::Sink,
            AttentionPattern::Sink, AttentionPattern::Sink,
        ];
        for _ in 0..9 {
            patterns.push(AttentionPattern::Normal);
        }
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    #[test]
    fn test_batch_summary_41_percent_diffuse_is_diffuse() {
        // Arrange: 5 out of 12 ≈ 41.7% > 40% → DiffuseAttention
        // 5*10 > 12*4 → 50 > 48 → true
        let mut patterns = vec![
            AttentionPattern::Diffuse; 5
        ];
        for _ in 0..7 {
            patterns.push(AttentionPattern::Normal);
        }
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.1);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
    }

    // --- Cross-method: all 3 detect methods on different layers ---

    #[test]
    fn test_three_detect_methods_three_layers() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);

        // Act: method 1 — direct
        let p0 = tracker.detect_for_layer(0, 0.95, 0.9);
        // method 2 — aggregator
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        let p1 = tracker.detect_from_aggregator(1, &agg);
        // method 3 — page header
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.1);
        header.centroid_pos = f32_to_f16_bits(0.05);
        let p2 = tracker.detect_from_page_header(2, &header);

        // Assert
        assert_eq!(p0, AttentionPattern::Sink);
        assert_eq!(p1, AttentionPattern::SharpFocus);
        assert_eq!(p2, AttentionPattern::Diffuse);
        assert_eq!(tracker.total_detections(), 3);
        assert_eq!(tracker.history().len(), 3);
    }

    // --- reset_stats does not affect protected positions ---

    #[test]
    fn test_reset_does_not_affect_protected_positions() {
        // Arrange
        let config = SinkDetectionConfig {
            protected_sink_count: 6,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Act
        tracker.reset_stats();
        // Assert: protection config is unchanged
        assert_eq!(tracker.protected_sink_count(), 6);
        assert_eq!(tracker.protected_sink_positions().len(), 6);
        assert!(tracker.needs_sink_protection(5));
        assert!(!tracker.needs_sink_protection(6));
    }

    // --- is_enabled depends only on sink_threshold ---

    #[test]
    fn test_is_enabled_true_with_very_small_positive_threshold() {
        // Arrange
        let config = SinkDetectionConfig {
            sink_threshold: f32::MIN_POSITIVE,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 4);
        // Assert
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_is_enabled_with_zero_threshold() {
        // Arrange
        let config = SinkDetectionConfig {
            sink_threshold: 0.0,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 4);
        // Assert
        assert!(!tracker.is_enabled());
    }

    // --- config() returns immutable reference ---

    #[test]
    fn test_config_fields_match_construction() {
        // Arrange
        let config = SinkDetectionConfig {
            sink_threshold: 0.77,
            protected_sink_count: 12,
            sharp_focus_threshold: 0.66,
            diffuse_threshold: 0.11,
        };
        let expected_threshold = config.sink_threshold;
        let expected_count = config.protected_sink_count;
        let expected_sharp = config.sharp_focus_threshold;
        let expected_diffuse = config.diffuse_threshold;
        // Act
        let tracker = SinkTracker::new(config, 8);
        let cfg = tracker.config();
        // Assert
        assert!((cfg.sink_threshold - expected_threshold).abs() < f32::EPSILON);
        assert_eq!(cfg.protected_sink_count, expected_count);
        assert!((cfg.sharp_focus_threshold - expected_sharp).abs() < f32::EPSILON);
        assert!((cfg.diffuse_threshold - expected_diffuse).abs() < f32::EPSILON);
    }

    // --- history with non-sequential layer indices ---

    #[test]
    fn test_history_with_non_sequential_layer_indices() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 128);
        // Act: use sparse, non-sequential indices
        tracker.detect_for_layer(7, 0.95, 0.9);
        tracker.detect_for_layer(42, 0.5, 0.85);
        tracker.detect_for_layer(99, 0.1, 0.05);
        // Assert
        assert_eq!(tracker.history().len(), 3);
        assert!(tracker.layer_record(7).is_some());
        assert!(tracker.layer_record(42).is_some());
        assert!(tracker.layer_record(99).is_some());
        assert!(tracker.layer_record(0).is_none());
        assert!(tracker.layer_record(50).is_none());
    }

    // --- repeated resets ---

    #[test]
    fn test_triple_reset_preserves_invariants() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Act
        tracker.reset_stats();
        tracker.reset_stats();
        tracker.reset_stats();
        // Assert
        assert_eq!(tracker.total_detections(), 0);
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.num_layers(), 8);
        assert!(tracker.is_enabled());
    }

    // --- BatchAttentionSummary: large pattern vector ---

    #[test]
    fn test_batch_summary_with_large_pattern_vector() {
        // Arrange: 100 patterns — 35 sinks, rest normal
        let mut patterns = vec![AttentionPattern::Sink; 35];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(65));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert: 35/100 = 35% > 30% → SinkDominant
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!((summary.sink_ratio - 0.35).abs() < 0.01);
        assert_eq!(summary.per_request_patterns.len(), 100);
    }

    // --- detect_from_page_header: f16 round-trip precision ---

    #[test]
    fn test_detect_from_page_header_f16_roundtrip_sink() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        // f16 round-trip: 0.95 stays close enough to be > 0.9 threshold
        header.softmax_max_avg = f32_to_f16_bits(0.95);
        header.centroid_pos = f32_to_f16_bits(0.9);
        // Act
        let pattern = tracker.detect_from_page_header(0, &header);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    // --- detect_from_aggregator: multiple ingest signals ---

    #[test]
    fn test_detect_from_aggregator_multiple_ingests() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut agg = TelemetryAggregator::new();
        // Ingest multiple signals — the latest should determine the result
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.4,
        });
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.9,
        });
        // Act
        let _pattern = tracker.detect_from_aggregator(0, &agg);
        // Assert: aggregator averages, but final result should still detect high values
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.history().len(), 1);
    }

    // --- LayerAttentionRecord: various float values ---

    #[test]
    fn test_layer_record_with_subnormal_float() {
        // Arrange
        let record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: f32::MIN_POSITIVE,
            sharpness: f32::MIN_POSITIVE,
            pattern: AttentionPattern::Normal,
        };
        // Assert: subnormal values preserved
        assert!(record.softmax_max > 0.0);
        assert!(record.sharpness > 0.0);
    }

    // --- protected_sink_positions: consistency with needs_sink_protection ---

    #[test]
    fn test_protection_consistency_between_positions_and_needs() {
        // Arrange
        let config = SinkDetectionConfig {
            protected_sink_count: 5,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        let positions = tracker.protected_sink_positions();
        // Act & Assert: every position in the list should be protected
        for &pos in &positions {
            assert!(tracker.needs_sink_protection(pos));
        }
        // Position just beyond the list should not be protected
        assert!(!tracker.needs_sink_protection(5));
    }

    // --- num_layers never changes ---

    #[test]
    fn test_num_layers_constant_after_detections() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 24);
        // Act
        for i in 0..24 {
            tracker.detect_for_layer(i, 0.5, 0.5);
        }
        // Assert
        assert_eq!(tracker.num_layers(), 24);
    }

    #[test]
    fn test_num_layers_constant_after_reset() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 16);
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Act
        tracker.reset_stats();
        // Assert
        assert_eq!(tracker.num_layers(), 16);
    }

    // --- BatchAttentionAdvice: exhaustive match semantics ---

    #[test]
    fn test_batch_advice_needs_sink_protection_only_for_sink_dominant() {
        // Assert: only SinkDominant triggers protection
        assert!(BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        }.needs_sink_protection());
        assert!(!BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::NormalAttention,
        }.needs_sink_protection());
        assert!(!BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::DiffuseAttention,
        }.needs_sink_protection());
    }

    #[test]
    fn test_batch_advice_benefits_from_speculation_only_for_diffuse() {
        // Assert: only DiffuseAttention triggers speculation
        assert!(BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::DiffuseAttention,
        }.benefits_from_speculation());
        assert!(!BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::NormalAttention,
        }.benefits_from_speculation());
        assert!(!BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        }.benefits_from_speculation());
    }

    // --- detect_from_page_header with default header ---

    #[test]
    fn test_detect_from_page_header_default_header() {
        // Arrange: default KvPageHeader has zero f16 values → 0.0, 0.0
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let header = KvPageHeader::new(0);
        // Act: f16_bits_to_f32(0) = 0.0 for both fields
        let pattern = tracker.detect_from_page_header(0, &header);
        // Assert: max=0.0 < 0.9, sharpness=0.0 < 0.1 → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    // --- history: insertion order with interleaved updates ---

    #[test]
    fn test_history_insertion_order_interleaved_updates() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        // Act
        tracker.detect_for_layer(0, 0.5, 0.5);  // Normal
        tracker.detect_for_layer(1, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(0, 0.95, 0.9);  // Update layer 0 → Sink
        tracker.detect_for_layer(2, 0.1, 0.05);  // Diffuse
        // Assert: history order should still be 0, 1, 2 (insertion order preserved)
        let history = tracker.history();
        assert_eq!(history[0].layer_idx, 0);
        assert_eq!(history[1].layer_idx, 1);
        assert_eq!(history[2].layer_idx, 2);
        // Layer 0 was updated to Sink
        assert_eq!(history[0].pattern, AttentionPattern::Sink);
    }

    // --- ratio after reset + re-detect cycle ---

    #[test]
    fn test_ratios_after_reset_redetect_cycle() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();
        // Act: fresh detections with different patterns
        tracker.detect_for_layer(0, 0.5, 0.85);  // SharpFocus
        tracker.detect_for_layer(1, 0.5, 0.85);  // SharpFocus
        // Assert
        assert!((tracker.sharp_focus_ratio() - 1.0).abs() < 0.01);
        assert!((tracker.sink_ratio() - 0.0).abs() < 0.01);
    }

    // --- detect with very small positive values ---

    #[test]
    fn test_detect_with_very_small_positive_max_and_sharpness() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: very small positive values → max < 0.9, sharpness < 0.1 → Diffuse
        let pattern = tracker.detect_for_layer(0, f32::MIN_POSITIVE, f32::MIN_POSITIVE);
        // Assert
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    // --- large number of layers with mixed patterns ---

    #[test]
    fn test_large_mixed_detections_ratios_sum_to_one() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 100);
        // Act: 25 each of Sink, SharpFocus, Diffuse, Normal
        for i in 0..25 {
            tracker.detect_for_layer(i, 0.95, 0.9);       // Sink
        }
        for i in 25..50 {
            tracker.detect_for_layer(i, 0.5, 0.85);       // SharpFocus
        }
        for i in 50..75 {
            tracker.detect_for_layer(i, 0.1, 0.05);       // Diffuse
        }
        for i in 75..100 {
            tracker.detect_for_layer(i, 0.3, 0.4);        // Normal
        }
        // Assert
        let sum = tracker.sink_ratio() + tracker.sharp_focus_ratio() + tracker.diffuse_ratio();
        let normal_ratio = 1.0 - sum;
        assert!((normal_ratio - 0.25).abs() < 0.01);
        assert!((tracker.sink_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.sharp_focus_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.diffuse_ratio() - 0.25).abs() < 0.01);
    }

    // ========================================================================
    // Additional 40 tests — uncovered edges and structural properties
    // ========================================================================

    // --- page header: f16 round-trip precision stored in record ---

    #[test]
    fn test_page_header_f16_precision_in_record() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.75);
        header.centroid_pos = f32_to_f16_bits(0.35);
        // Act
        tracker.detect_from_page_header(0, &header);
        // Assert: f16 round-trip preserves within f16 precision
        let record = tracker.layer_record(0).unwrap();
        assert!((record.softmax_max - 0.75).abs() < 0.05);
        assert!((record.sharpness - 0.35).abs() < 0.05);
    }

    #[test]
    fn test_page_header_same_header_two_layers() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.95);
        header.centroid_pos = f32_to_f16_bits(0.9);
        // Act: same header for two different layers
        let p0 = tracker.detect_from_page_header(0, &header);
        let p1 = tracker.detect_from_page_header(5, &header);
        // Assert
        assert_eq!(p0, AttentionPattern::Sink);
        assert_eq!(p1, AttentionPattern::Sink);
        assert_eq!(tracker.history().len(), 2);
    }

    #[test]
    fn test_page_header_twice_same_layer_updates() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header1 = KvPageHeader::new(0);
        header1.softmax_max_avg = f32_to_f16_bits(0.95);
        header1.centroid_pos = f32_to_f16_bits(0.9);
        tracker.detect_from_page_header(3, &header1);
        assert_eq!(tracker.layer_record(3).unwrap().pattern, AttentionPattern::Sink);
        // Act: second header → Diffuse
        let mut header2 = KvPageHeader::new(0);
        header2.softmax_max_avg = f32_to_f16_bits(0.1);
        header2.centroid_pos = f32_to_f16_bits(0.05);
        tracker.detect_from_page_header(3, &header2);
        // Assert: latest wins, no duplicate history entry
        assert_eq!(tracker.layer_record(3).unwrap().pattern, AttentionPattern::Diffuse);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.total_detections(), 2);
    }

    #[test]
    fn test_page_header_sharp_focus_pattern() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        // max=0.5 < 0.9, sharpness=0.85 > 0.8 → SharpFocus
        header.softmax_max_avg = f32_to_f16_bits(0.5);
        header.centroid_pos = f32_to_f16_bits(0.85);
        // Act
        let pattern = tracker.detect_from_page_header(0, &header);
        // Assert
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    // --- TelemetryAggregator reuse across layers ---

    #[test]
    fn test_aggregator_reused_across_layers() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.9,
        });
        // Act: same aggregator for two layers
        let p0 = tracker.detect_from_aggregator(0, &agg);
        let p1 = tracker.detect_from_aggregator(1, &agg);
        // Assert
        assert_eq!(p0, AttentionPattern::Sink);
        assert_eq!(p1, AttentionPattern::Sink);
        assert_eq!(tracker.history().len(), 2);
        assert_eq!(tracker.sink_count, 2);
    }

    #[test]
    fn test_aggregator_reuse_after_reset() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.9,
        });
        tracker.detect_from_aggregator(0, &agg);
        // Act
        tracker.reset_stats();
        tracker.detect_from_aggregator(0, &agg);
        // Assert: counts restart from 0
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.sink_count, 1);
    }

    #[test]
    fn test_detect_from_aggregator_empty_produces_diffuse() {
        // Arrange: aggregator with no ingested signals
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let agg = TelemetryAggregator::new();
        // Act: empty aggregator → softmax_max/sharpness default to 0.0
        let pattern = tracker.detect_from_aggregator(0, &agg);
        // Assert: max=0.0 < 0.9, sharpness=0.0 < 0.1 → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    // --- Multiple tracker instances independence ---

    #[test]
    fn test_two_trackers_independent() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker_a = SinkTracker::new(config.clone(), 4);
        let mut tracker_b = SinkTracker::new(config, 4);
        // Act: different patterns in each
        tracker_a.detect_for_layer(0, 0.95, 0.9);
        tracker_b.detect_for_layer(0, 0.1, 0.05);
        // Assert: completely independent state
        assert_eq!(tracker_a.sink_count, 1);
        assert_eq!(tracker_a.diffuse_count, 0);
        assert_eq!(tracker_b.sink_count, 0);
        assert_eq!(tracker_b.diffuse_count, 1);
    }

    #[test]
    fn test_two_trackers_different_configs_same_input() {
        // Arrange
        let config_a = SinkDetectionConfig {
            sink_threshold: 0.5,
            ..SinkDetectionConfig::default()
        };
        let config_b = SinkDetectionConfig {
            sink_threshold: 0.95,
            ..SinkDetectionConfig::default()
        };
        let mut tracker_a = SinkTracker::new(config_a, 4);
        let mut tracker_b = SinkTracker::new(config_b, 4);
        // Act: same input values
        let pa = tracker_a.detect_for_layer(0, 0.7, 0.5);
        let pb = tracker_b.detect_for_layer(0, 0.7, 0.5);
        // Assert: different configs → different results
        assert_eq!(pa, AttentionPattern::Sink);    // 0.7 > 0.5
        assert_eq!(pb, AttentionPattern::Normal);  // 0.7 < 0.95
    }

    // --- BatchAttentionSummary: special avg_sharpness values ---

    #[test]
    fn test_batch_summary_inf_sharpness_pass_through() {
        // Arrange
        let patterns = vec![AttentionPattern::Normal];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, f32::INFINITY);
        // Assert
        assert!(summary.avg_sharpness.is_infinite());
        assert!(summary.avg_sharpness.is_sign_positive());
    }

    #[test]
    fn test_batch_summary_nan_sharpness_pass_through() {
        // Arrange
        let patterns = vec![AttentionPattern::Normal];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, f32::NAN);
        // Assert
        assert!(summary.avg_sharpness.is_nan());
    }

    #[test]
    fn test_batch_summary_field_mutation_changes_advice_effect() {
        // Arrange
        let mut summary = BatchAttentionSummary {
            per_request_patterns: vec![],
            avg_sharpness: 0.0,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::NormalAttention,
        };
        assert!(!summary.needs_sink_protection());
        // Act: mutate batch_advice
        summary.batch_advice = BatchAttentionAdvice::SinkDominant;
        // Assert
        assert!(summary.needs_sink_protection());
    }

    // --- BatchAttentionSummary: mixed pattern combinations ---

    #[test]
    fn test_batch_summary_sharpfocus_and_diffuse_only() {
        // Arrange: no Sink, no Normal
        let patterns = vec![
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
        ];
        // Act: 0 sinks, 3/5 = 60% diffuse > 40% → DiffuseAttention
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
        assert!((summary.sink_ratio).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_equal_sink_normal() {
        // Arrange: 50/50 sink vs normal
        let patterns = vec![AttentionPattern::Sink, AttentionPattern::Normal];
        // Act: 1/2 = 50% > 30% → SinkDominant
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.6);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!((summary.sink_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_all_four_patterns_normal_advice() {
        // Arrange: 1 of each pattern
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
            AttentionPattern::Normal,
        ];
        // Act: 1/4 = 25% sink < 30%, 1/4 = 25% diffuse < 40% → NormalAttention
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert_eq!(summary.per_request_patterns.len(), 4);
    }

    #[test]
    fn test_batch_summary_clone_then_mutate_independence() {
        // Arrange
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink],
            avg_sharpness: 0.8,
            sink_ratio: 1.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        // Act
        let mut cloned = summary.clone();
        cloned.per_request_patterns.clear();
        // Assert: original unchanged
        assert_eq!(summary.per_request_patterns.len(), 1);
        assert_eq!(cloned.per_request_patterns.len(), 0);
    }

    #[test]
    fn test_batch_summary_per_request_patterns_mutable() {
        // Arrange
        let mut summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Normal],
            avg_sharpness: 0.5,
            sink_ratio: 0.0,
            batch_advice: BatchAttentionAdvice::NormalAttention,
        };
        // Act: pub field is mutable
        summary.per_request_patterns.push(AttentionPattern::Sink);
        // Assert
        assert_eq!(summary.per_request_patterns.len(), 2);
        assert_eq!(summary.per_request_patterns[1], AttentionPattern::Sink);
    }

    // --- protected_sink_positions with minimal count ---

    #[test]
    fn test_protected_positions_count_one() {
        // Arrange
        let config = SinkDetectionConfig {
            protected_sink_count: 1,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 4);
        // Assert
        assert_eq!(tracker.protected_sink_positions(), vec![0]);
        assert!(tracker.needs_sink_protection(0));
        assert!(!tracker.needs_sink_protection(1));
    }

    // --- Structural invariants: history vs layer_record consistency ---

    #[test]
    fn test_history_and_layer_record_consistent() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(3, 0.5, 0.85);
        tracker.detect_for_layer(7, 0.1, 0.05);
        // Assert: every history entry has a matching layer_record
        for record in tracker.history() {
            let found = tracker.layer_record(record.layer_idx);
            assert!(found.is_some());
            assert_eq!(found.unwrap().pattern, record.pattern);
        }
        // Absent layers return None
        assert!(tracker.layer_record(1).is_none());
        assert!(tracker.layer_record(2).is_none());
        assert!(tracker.layer_record(99).is_none());
    }

    #[test]
    fn test_total_detections_equals_pattern_count_sum() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);    // Sink
        tracker.detect_for_layer(1, 0.5, 0.85);     // SharpFocus
        tracker.detect_for_layer(2, 0.1, 0.05);     // Diffuse
        tracker.detect_for_layer(3, 0.3, 0.4);      // Normal
        tracker.detect_for_layer(0, 0.3, 0.4);      // Normal (re-detect)
        // Assert: sum of individual counts equals total
        assert_eq!(
            tracker.total_detections(),
            tracker.normal_count + tracker.sink_count
                + tracker.sharp_focus_count + tracker.diffuse_count
        );
    }

    #[test]
    fn test_history_length_leq_total_detections() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: same layer detected 5 times
        for _ in 0..5 {
            tracker.detect_for_layer(0, 0.5, 0.5);
        }
        // Assert: history has at most 1 unique entry, but 5 detections
        assert!(tracker.history().len() <= tracker.total_detections());
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.total_detections(), 5);
    }

    // --- Config that forces specific pattern outcomes ---

    #[test]
    fn test_config_forces_all_normal() {
        // Arrange: thresholds that make all inputs Normal
        let config = SinkDetectionConfig {
            sink_threshold: 2.0,
            sharp_focus_threshold: 2.0,
            diffuse_threshold: -1.0,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act & Assert: all inputs result in Normal
        assert_eq!(tracker.detect_for_layer(0, 0.0, 0.0), AttentionPattern::Normal);
        assert_eq!(tracker.detect_for_layer(1, 1.0, 1.0), AttentionPattern::Normal);
        assert_eq!(tracker.detect_for_layer(2, 0.5, 0.5), AttentionPattern::Normal);
        assert_eq!(tracker.detect_for_layer(3, -1.0, -1.0), AttentionPattern::Normal);
    }

    #[test]
    fn test_config_forces_all_diffuse() {
        // Arrange: high sink threshold, high sharp_focus, high diffuse → everything Diffuse
        let config = SinkDetectionConfig {
            sink_threshold: 2.0,
            sharp_focus_threshold: 2.0,
            diffuse_threshold: 1.0,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act & Assert
        assert_eq!(tracker.detect_for_layer(0, 0.5, 0.5), AttentionPattern::Diffuse);
        assert_eq!(tracker.detect_for_layer(1, 0.5, 0.99), AttentionPattern::Diffuse);
    }

    // --- detect_for_layer: specific value boundaries ---

    #[test]
    fn test_detect_max_val_one_with_default_threshold() {
        // Arrange: default sink_threshold = 0.9
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=1.0 > 0.9 → Sink regardless of sharpness
        let pattern = tracker.detect_for_layer(0, 1.0, 0.5);
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_sharpness_one_not_sink() {
        // Arrange: max below threshold, sharpness at max
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=0.5 < 0.9, sharpness=1.0 > 0.8 → SharpFocus
        let pattern = tracker.detect_for_layer(0, 0.5, 1.0);
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    // --- LayerAttentionRecord: special float field values ---

    #[test]
    fn test_layer_record_with_inf_softmax_max() {
        // Arrange
        let record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: f32::INFINITY,
            sharpness: 0.5,
            pattern: AttentionPattern::Sink,
        };
        // Assert
        assert!(record.softmax_max.is_infinite());
        assert!(record.softmax_max.is_sign_positive());
    }

    #[test]
    fn test_layer_record_with_nan_sharpness() {
        // Arrange
        let record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: 0.5,
            sharpness: f32::NAN,
            pattern: AttentionPattern::Normal,
        };
        // Assert
        assert!(record.sharpness.is_nan());
    }

    #[test]
    fn test_layer_record_field_mutation() {
        // Arrange
        let mut record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: 0.5,
            sharpness: 0.5,
            pattern: AttentionPattern::Normal,
        };
        // Act: pub fields are mutable
        record.layer_idx = 10;
        record.softmax_max = 0.95;
        record.sharpness = 0.9;
        record.pattern = AttentionPattern::Sink;
        // Assert
        assert_eq!(record.layer_idx, 10);
        assert!((record.softmax_max - 0.95).abs() < f32::EPSILON);
        assert!((record.sharpness - 0.9).abs() < f32::EPSILON);
        assert_eq!(record.pattern, AttentionPattern::Sink);
    }

    // --- SinkTracker: reset after heavy interleaved load ---

    #[test]
    fn test_reset_after_many_interleaved_detections() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        for _ in 0..10 {
            tracker.detect_for_layer(0, 0.95, 0.9);
            tracker.detect_for_layer(1, 0.5, 0.85);
            tracker.detect_for_layer(2, 0.1, 0.05);
            tracker.detect_for_layer(3, 0.3, 0.4);
        }
        assert_eq!(tracker.total_detections(), 40);
        // Act
        tracker.reset_stats();
        // Assert
        assert_eq!(tracker.total_detections(), 0);
        assert!(tracker.history().is_empty());
        assert_eq!(tracker.num_layers(), 4);
    }

    // --- Ratio boundedness with many detections ---

    #[test]
    fn test_ratios_bounded_after_many_detections() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 100);
        for i in 0..100 {
            let max_val = match i % 4 {
                0 => 0.95,
                1 => 0.5,
                2 => 0.1,
                _ => 0.3,
            };
            let sharpness = match i % 4 {
                0 => 0.9,
                1 => 0.85,
                2 => 0.05,
                _ => 0.4,
            };
            tracker.detect_for_layer(i, max_val, sharpness);
        }
        // Assert: all ratios in [0, 1]
        assert!(tracker.sink_ratio() >= 0.0 && tracker.sink_ratio() <= 1.0);
        assert!(tracker.sharp_focus_ratio() >= 0.0 && tracker.sharp_focus_ratio() <= 1.0);
        assert!(tracker.diffuse_ratio() >= 0.0 && tracker.diffuse_ratio() <= 1.0);
        let sum = tracker.sink_ratio() + tracker.sharp_focus_ratio() + tracker.diffuse_ratio();
        assert!(sum <= 1.0 + 0.01);
    }

    // --- recent_patterns after reset + new detections ---

    #[test]
    fn test_recent_patterns_after_reset_and_new_detects() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.5, 0.5);
        tracker.reset_stats();
        // Act
        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(6, 0.1, 0.05);
        let recent = tracker.recent_patterns(1);
        // Assert: only post-reset entries exist
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].layer_idx, 6);
        assert_eq!(recent[0].pattern, AttentionPattern::Diffuse);
    }

    // --- All three detect methods contribute to pattern counts ---

    #[test]
    fn test_all_detect_methods_record_pattern_counts() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        // Act: method 1 — direct → Sink
        tracker.detect_for_layer(0, 0.95, 0.9);
        // method 2 — aggregator → Normal
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.4,
        });
        tracker.detect_from_aggregator(1, &agg);
        // method 3 — page header → Diffuse
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.1);
        header.centroid_pos = f32_to_f16_bits(0.05);
        tracker.detect_from_page_header(2, &header);
        // Assert
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.normal_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
        assert_eq!(tracker.total_detections(), 3);
    }

    // --- config() stability across operations ---

    #[test]
    fn test_config_stable_across_operations() {
        // Arrange
        let config = SinkDetectionConfig {
            sink_threshold: 0.6,
            protected_sink_count: 3,
            sharp_focus_threshold: 0.4,
            diffuse_threshold: 0.15,
        };
        let mut tracker = SinkTracker::new(config, 8);
        let cfg_before = tracker.config();
        assert!((cfg_before.sink_threshold - 0.6).abs() < f32::EPSILON);
        // Act: perform some operations
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();
        // Assert: config unchanged
        let cfg_after = tracker.config();
        assert!((cfg_after.sink_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(cfg_after.protected_sink_count, 3);
    }

    // --- detect with num_layers = 0 still works ---

    #[test]
    fn test_detect_with_zero_num_layers_still_detects() {
        // Arrange: num_layers=0 is just a hint, detection still functions
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 0);
        // Act
        let pattern = tracker.detect_for_layer(5, 0.95, 0.9);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
        assert_eq!(tracker.num_layers(), 0);
        assert_eq!(tracker.total_detections(), 1);
    }

    // --- recent_patterns with repeated layer updates ---

    #[test]
    fn test_recent_patterns_with_repeated_layer_updates() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(0, 0.1, 0.05);
        tracker.detect_for_layer(0, 0.5, 0.85);
        // Act
        let recent = tracker.recent_patterns(5);
        // Assert: only 1 unique layer, latest pattern is SharpFocus
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].pattern, AttentionPattern::SharpFocus);
    }

    // --- page header updates pattern counts correctly ---

    #[test]
    fn test_page_header_updates_pattern_counts() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.95);
        header.centroid_pos = f32_to_f16_bits(0.9);
        // Act
        tracker.detect_from_page_header(0, &header);
        // Assert
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.total_detections(), 1);
    }

    // --- aggregator detecting two different layers with different patterns ---

    #[test]
    fn test_aggregator_detect_two_layers_different_patterns() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Layer 0: SharpFocus
        let mut agg1 = TelemetryAggregator::new();
        agg1.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        tracker.detect_from_aggregator(0, &agg1);
        // Layer 1: Normal
        let mut agg2 = TelemetryAggregator::new();
        agg2.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.4,
        });
        tracker.detect_from_aggregator(1, &agg2);
        // Assert
        assert_eq!(tracker.sharp_focus_count, 1);
        assert_eq!(tracker.normal_count, 1);
        assert_eq!(tracker.history().len(), 2);
    }

    // --- Consistent pattern for same input across tracker instances ---

    #[test]
    fn test_same_input_same_pattern_across_trackers() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker1 = SinkTracker::new(config.clone(), 4);
        let mut tracker2 = SinkTracker::new(config, 4);
        // Act
        let p1 = tracker1.detect_for_layer(0, 0.7, 0.6);
        let p2 = tracker2.detect_for_layer(0, 0.7, 0.6);
        // Assert: deterministic
        assert_eq!(p1, p2);
    }

    // --- 10 unique layers with explicit expected patterns ---

    #[test]
    fn test_tracker_ten_layers_all_patterns() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 10);
        let cases = [
            (0.95, 0.9, AttentionPattern::Sink),
            (0.5, 0.85, AttentionPattern::SharpFocus),
            (0.1, 0.05, AttentionPattern::Diffuse),
            (0.3, 0.4, AttentionPattern::Normal),
            (0.96, 0.91, AttentionPattern::Sink),
            (0.51, 0.86, AttentionPattern::SharpFocus),
            (0.11, 0.04, AttentionPattern::Diffuse),
            (0.31, 0.41, AttentionPattern::Normal),
            (0.97, 0.92, AttentionPattern::Sink),
            (0.52, 0.87, AttentionPattern::SharpFocus),
        ];
        // Act
        for (i, (max, sharp, expected)) in cases.iter().enumerate() {
            let result = tracker.detect_for_layer(i, *max, *sharp);
            assert_eq!(result, *expected, "Failed at layer {}", i);
        }
        // Assert
        assert_eq!(tracker.history().len(), 10);
        assert_eq!(tracker.total_detections(), 10);
    }

    // ========================================================================
    // Additional 15 tests — remaining uncovered edges
    // ========================================================================

    #[test]
    fn test_detect_inf_sharpness_produces_sharp_focus() {
        // Arrange: max below sink threshold, sharpness = +inf > 0.8 → SharpFocus
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.5, f32::INFINITY);
        // Assert
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_detect_neg_inf_sharpness_produces_diffuse() {
        // Arrange: max below sink threshold, sharpness = -inf < 0.1 → Diffuse
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.5, f32::NEG_INFINITY);
        // Assert
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_batch_summary_clone_then_compare() {
        // Arrange
        let original = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink],
            avg_sharpness: 0.7,
            sink_ratio: 1.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        // Act: Clone — both independent
        let cloned = original.clone();
        // Assert: both are usable and equal
        assert_eq!(original.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert_eq!(cloned.batch_advice, BatchAttentionAdvice::SinkDominant);
    }

    #[test]
    fn test_batch_summary_debug_contains_all_fields() {
        // Arrange
        let summary = BatchAttentionSummary {
            per_request_patterns: vec![AttentionPattern::Sink],
            avg_sharpness: 0.8,
            sink_ratio: 1.0,
            batch_advice: BatchAttentionAdvice::SinkDominant,
        };
        // Act
        let debug = format!("{:?}", summary);
        // Assert: all four struct fields appear in debug output
        assert!(debug.contains("per_request_patterns"));
        assert!(debug.contains("avg_sharpness"));
        assert!(debug.contains("sink_ratio"));
        assert!(debug.contains("batch_advice"));
    }

    #[test]
    fn test_detect_for_layer_layer_idx_exceeds_num_layers() {
        // Arrange: num_layers=4 but detect for layer 1000
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(1000, 0.95, 0.9);
        // Assert: detection works regardless of num_layers
        assert_eq!(pattern, AttentionPattern::Sink);
        assert_eq!(tracker.total_detections(), 1);
        assert!(tracker.layer_record(1000).is_some());
    }

    #[test]
    fn test_detect_for_layer_max_equals_sharpness() {
        // Arrange: both values equal at 0.5 (middle range)
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=0.5 < 0.9, sharpness=0.5: not > 0.8, not < 0.1 → Normal
        let pattern = tracker.detect_for_layer(0, 0.5, 0.5);
        // Assert
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_config_with_identical_thresholds() {
        // Arrange: all three thresholds set to same value
        let config = SinkDetectionConfig {
            sink_threshold: 0.5,
            sharp_focus_threshold: 0.5,
            diffuse_threshold: 0.5,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act & Assert
        // max=0.6 > 0.5 → Sink
        assert_eq!(tracker.detect_for_layer(0, 0.6, 0.5), AttentionPattern::Sink);
        // max=0.4, sharpness=0.6 > 0.5 → SharpFocus
        assert_eq!(tracker.detect_for_layer(1, 0.4, 0.6), AttentionPattern::SharpFocus);
        // max=0.4, sharpness=0.4 < 0.5 → Diffuse
        assert_eq!(tracker.detect_for_layer(2, 0.4, 0.4), AttentionPattern::Diffuse);
    }

    #[test]
    fn test_config_with_sharp_focus_above_sink_threshold() {
        // Arrange: unusual config where sharp_focus > sink
        let config = SinkDetectionConfig {
            sink_threshold: 0.5,
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=0.6 > 0.5 → Sink (sink checked first)
        let pattern = tracker.detect_for_layer(0, 0.6, 0.9);
        // Assert: Sink takes precedence over SharpFocus
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_history_slice_reflects_live_updates() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Act: verify initial state
        {
            let history = tracker.history();
            assert_eq!(history[0].pattern, AttentionPattern::Sink);
        }
        // Re-detect with different pattern
        tracker.detect_for_layer(0, 0.3, 0.4);
        // Assert: history reflects the update
        let history = tracker.history();
        assert_eq!(history[0].pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_batch_summary_29_percent_sink_is_normal() {
        // Arrange: 2 out of 7 ≈ 28.6% → 2*10=20, 7*3=21, 20 < 21 → Normal
        let mut patterns = vec![AttentionPattern::Sink; 2];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(5));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert: <30% sink → NormalAttention
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_batch_summary_39_percent_diffuse_is_normal() {
        // Arrange: 3 out of 8 = 37.5% → 3*10=30, 8*4=32, 30 < 32 → Normal
        let mut patterns = vec![AttentionPattern::Diffuse; 3];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(5));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.1);
        // Assert: <40% diffuse → NormalAttention
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_reset_then_aggregator_detect() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();
        // Act: detect via aggregator after reset
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        tracker.detect_from_aggregator(0, &agg);
        // Assert: only post-reset data exists
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.layer_record(0).unwrap().pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_detect_for_layer_f32_max_produces_sink() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: f32::MAX > 0.9 → Sink
        let pattern = tracker.detect_for_layer(0, f32::MAX, 0.5);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_batch_summary_zero_sharpness_non_empty_patterns() {
        // Arrange
        let patterns = vec![AttentionPattern::Normal, AttentionPattern::Normal];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.0);
        // Assert
        assert!((summary.avg_sharpness - 0.0).abs() < f32::EPSILON);
        assert_eq!(summary.per_request_patterns.len(), 2);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_layer_record_with_all_nan_float_fields() {
        // Arrange
        let record = LayerAttentionRecord {
            layer_idx: 0,
            softmax_max: f32::NAN,
            sharpness: f32::NAN,
            pattern: AttentionPattern::Normal,
        };
        // Assert: NaN fields preserved
        assert!(record.softmax_max.is_nan());
        assert!(record.sharpness.is_nan());
        assert_eq!(record.layer_idx, 0);
        assert_eq!(record.pattern, AttentionPattern::Normal);
    }

    // ========================================================================
    // Additional 15 tests — subnormal, precision, structural edge cases
    // ========================================================================

    #[test]
    fn test_detect_subnormal_max_val_classifies_diffuse() {
        // Arrange: subnormal float < 0.1 threshold for both max and sharpness
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        // Act: max < 0.9, sharpness < 0.1 → Diffuse
        let pattern = tracker.detect_for_layer(0, subnormal, subnormal);
        // Assert
        assert_eq!(pattern, AttentionPattern::Diffuse);
        assert!(tracker.layer_record(0).unwrap().softmax_max > 0.0);
    }

    #[test]
    fn test_detect_sharpness_exactly_zero_one_tenth_boundary() {
        // Arrange: sharpness = 0.1 == diffuse_threshold → NOT Diffuse (strictly less)
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=0.5 < 0.9, sharpness=0.1 == diffuse_threshold → Normal
        let pattern = tracker.detect_for_layer(0, 0.5, 0.1);
        // Assert
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_recent_patterns_with_usize_max_n() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(1, 0.95, 0.9);
        // Act: request usize::MAX patterns — should return all available
        let recent = tracker.recent_patterns(usize::MAX);
        // Assert: returns all 2 entries without panic
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_history_order_after_middle_update_then_new_layer() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(10, 0.1, 0.05);
        // Act: update middle layer
        tracker.detect_for_layer(5, 0.3, 0.4);
        // Add a new layer
        tracker.detect_for_layer(15, 0.5, 0.85);
        // Assert: insertion order preserved: 0, 5, 10, 15
        let history = tracker.history();
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].layer_idx, 0);
        assert_eq!(history[1].layer_idx, 5);
        assert_eq!(history[2].layer_idx, 10);
        assert_eq!(history[3].layer_idx, 15);
        // Layer 5 was updated to Normal
        assert_eq!(history[1].pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_protected_positions_vec_is_independent_clone() {
        // Arrange
        let config = SinkDetectionConfig {
            protected_sink_count: 3,
            ..SinkDetectionConfig::default()
        };
        let tracker = SinkTracker::new(config, 8);
        // Act: get positions and mutate the returned Vec
        let mut positions = tracker.protected_sink_positions();
        positions.push(999);
        // Assert: original tracker unaffected
        assert_eq!(tracker.protected_sink_positions().len(), 3);
        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_batch_summary_one_sink_in_large_batch_sink_ratio_precision() {
        // Arrange: 1 sink in 1000 patterns → 0.1% → well below 30%
        let mut patterns = vec![AttentionPattern::Sink];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(999));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!((summary.sink_ratio - 0.001).abs() < 0.001);
    }

    #[test]
    fn test_batch_summary_neg_inf_sharpness_pass_through() {
        // Arrange
        let patterns = vec![AttentionPattern::Normal];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, f32::NEG_INFINITY);
        // Assert: negative infinity passes through without modification
        assert!(summary.avg_sharpness.is_infinite());
        assert!(summary.avg_sharpness.is_sign_negative());
    }

    #[test]
    fn test_tracker_layer_record_usize_max_index() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: use usize::MAX as layer index
        tracker.detect_for_layer(usize::MAX, 0.95, 0.9);
        // Assert: layer record stored and retrievable
        let record = tracker.layer_record(usize::MAX).unwrap();
        assert_eq!(record.layer_idx, usize::MAX);
        assert_eq!(record.pattern, AttentionPattern::Sink);
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_page_header_with_min_positive_f16_values() {
        // Arrange: very small positive f16 values
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(f32::MIN_POSITIVE);
        header.centroid_pos = f32_to_f16_bits(f32::MIN_POSITIVE);
        // Act
        let pattern = tracker.detect_from_page_header(0, &header);
        // Assert: very small values → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_detect_for_layer_negative_max_negative_sharpness() {
        // Arrange: both values negative
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=-1.0 < 0.9, sharpness=-0.5 < 0.1 → Diffuse
        let pattern = tracker.detect_for_layer(0, -1.0, -0.5);
        // Assert
        assert_eq!(pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_all_detect_methods_accumulate_separate_counts() {
        // Arrange: call each detect method once, verify each contributes
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        // Act: direct → Sink
        tracker.detect_for_layer(0, 0.95, 0.9);
        // aggregator → SharpFocus
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        tracker.detect_from_aggregator(1, &agg);
        // page header → Diffuse
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.1);
        header.centroid_pos = f32_to_f16_bits(0.05);
        tracker.detect_from_page_header(2, &header);
        // Assert: counts match individual patterns
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.sharp_focus_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
        assert_eq!(tracker.normal_count, 0);
        assert_eq!(tracker.total_detections(), 3);
    }

    #[test]
    fn test_batch_summary_30_percent_diffuse_exact_boundary_is_normal() {
        // Arrange: 3 out of 10 = exactly 30% diffuse → 3*10=30, 10*4=40, 30 < 40 → Normal
        let mut patterns = vec![AttentionPattern::Diffuse; 3];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(7));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.1);
        // Assert: exactly 30% diffuse is below 40% threshold → NormalAttention
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_tracker_history_record_has_correct_float_values() {
        // Arrange: verify that stored record has exact float values passed
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let max_val = 0.37;
        let sharpness = 0.62;
        // Act
        tracker.detect_for_layer(3, max_val, sharpness);
        // Assert
        let record = tracker.layer_record(3).unwrap();
        assert!((record.softmax_max - max_val).abs() < f32::EPSILON);
        assert!((record.sharpness - sharpness).abs() < f32::EPSILON);
        assert_eq!(record.layer_idx, 3);
    }

    #[test]
    fn test_detect_for_layer_sharpness_above_one_produces_sharp_focus() {
        // Arrange: sharpness > 1.0 > 0.8 threshold → SharpFocus (max below sink)
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.5, 1.5);
        // Assert: sharpness can exceed 1.0 and still classify correctly
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_batch_summary_large_all_diffuse_batch() {
        // Arrange: 500 all Diffuse → 100% > 40%
        let patterns = vec![AttentionPattern::Diffuse; 500];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.02);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::DiffuseAttention);
        assert!(summary.benefits_from_speculation());
        assert!((summary.sink_ratio).abs() < f32::EPSILON);
        assert_eq!(summary.per_request_patterns.len(), 500);
    }

    // ========================================================================
    // Additional 15 tests — cross-method state, history order, precision edges
    // ========================================================================

    #[test]
    fn test_detect_from_page_header_produces_normal_pattern() {
        // Arrange: page header values that produce Normal (not Sink/Diffuse/SharpFocus)
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut header = KvPageHeader::new(0);
        // max=0.5 < 0.9 (not Sink), sharpness=0.5 not > 0.8 (not SharpFocus), not < 0.1 (not Diffuse) → Normal
        header.softmax_max_avg = f32_to_f16_bits(0.5);
        header.centroid_pos = f32_to_f16_bits(0.5);
        // Act
        let pattern = tracker.detect_from_page_header(0, &header);
        // Assert
        assert_eq!(pattern, AttentionPattern::Normal);
        assert_eq!(tracker.normal_count, 1);
    }

    #[test]
    fn test_aggregator_then_direct_detect_same_layer_preserves_last() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: first via aggregator → SharpFocus
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        tracker.detect_from_aggregator(3, &agg);
        // Then direct detect → Sink
        tracker.detect_for_layer(3, 0.95, 0.9);
        // Assert: last write wins
        assert_eq!(tracker.layer_record(3).unwrap().pattern, AttentionPattern::Sink);
        assert_eq!(tracker.total_detections(), 2);
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_batch_summary_sharpfocus_heavy_with_one_sink_triggers_sink_dominant() {
        // Arrange: 1 Sink + 9 SharpFocus = 10 total, 10% sink < 30% → NormalAttention
        // But test with 4 Sink + 6 SharpFocus = 40% > 30% → SinkDominant
        let mut patterns = vec![AttentionPattern::Sink; 4];
        patterns.extend(std::iter::repeat(AttentionPattern::SharpFocus).take(6));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.8);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!(summary.needs_sink_protection());
        assert!((summary.sink_ratio - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_history_order_after_updating_first_then_appending() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);   // Normal
        tracker.detect_for_layer(3, 0.95, 0.9);   // Sink
        // Act: update layer 0 to Sink
        tracker.detect_for_layer(0, 0.95, 0.9);
        // Add new layer 7
        tracker.detect_for_layer(7, 0.1, 0.05);
        // Assert: order is 0, 3, 7 (insertion order preserved)
        let history = tracker.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].layer_idx, 0);
        assert_eq!(history[1].layer_idx, 3);
        assert_eq!(history[2].layer_idx, 7);
        // Layer 0 was updated to Sink
        assert_eq!(history[0].pattern, AttentionPattern::Sink);
        assert_eq!(history[2].pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_ratios_exactly_quarter_each_after_four_distinct_layers() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: exactly 1 of each pattern
        tracker.detect_for_layer(0, 0.95, 0.9);  // Sink
        tracker.detect_for_layer(1, 0.5, 0.85);   // SharpFocus
        tracker.detect_for_layer(2, 0.1, 0.05);   // Diffuse
        tracker.detect_for_layer(3, 0.3, 0.4);    // Normal
        // Assert: each ratio is exactly 0.25
        assert!((tracker.sink_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.sharp_focus_ratio() - 0.25).abs() < 0.01);
        assert!((tracker.diffuse_ratio() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_recent_patterns_after_interleaved_update_and_append() {
        // Arrange
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.5, 0.5);
        tracker.detect_for_layer(5, 0.95, 0.9);
        tracker.detect_for_layer(10, 0.1, 0.05);
        // Act: update layer 5, then append layer 15
        tracker.detect_for_layer(5, 0.3, 0.4);
        tracker.detect_for_layer(15, 0.5, 0.85);
        // Assert: recent 3 returns layers 15, 10, 5 (reverse insertion order)
        let recent = tracker.recent_patterns(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].layer_idx, 15);
        assert_eq!(recent[1].layer_idx, 10);
        assert_eq!(recent[2].layer_idx, 5);
        // Layer 5 was updated to Normal
        assert_eq!(recent[2].pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_is_enabled_unaffected_by_other_thresholds() {
        // Arrange: negative diffuse_threshold and high sharp_focus_threshold
        // is_enabled() only checks sink_threshold > 0.0
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            sharp_focus_threshold: 10.0,
            diffuse_threshold: -5.0,
            protected_sink_count: 4,
        };
        let tracker = SinkTracker::new(config, 4);
        // Assert: still enabled because sink_threshold > 0
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_batch_summary_100_percent_sharpfocus_is_normal_advice() {
        // Arrange: all SharpFocus → 0% Sink, 0% Diffuse → NormalAttention
        let patterns = vec![
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
            AttentionPattern::SharpFocus,
        ];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.9);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
        assert!((summary.sink_ratio).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_for_layer_max_val_slightly_below_one() {
        // Arrange: max_val = 0.99, still > 0.9 → Sink
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        let pattern = tracker.detect_for_layer(0, 0.99, 0.5);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_detect_for_layer_max_val_one_with_high_sharpness_both_valid() {
        // Arrange: max=1.0 > 0.9 → Sink (takes precedence over SharpFocus)
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: sharpness also high but Sink checked first
        let pattern = tracker.detect_for_layer(0, 1.0, 0.99);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_reset_then_page_header_detect_restores_counts() {
        // Arrange
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.reset_stats();
        // Act: detect via page header after reset
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.95);
        header.centroid_pos = f32_to_f16_bits(0.9);
        tracker.detect_from_page_header(2, &header);
        // Assert: only post-reset data exists
        assert_eq!(tracker.total_detections(), 1);
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.history().len(), 1);
        assert!(tracker.layer_record(0).is_none());
        assert_eq!(tracker.layer_record(2).unwrap().pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_batch_summary_3_sink_in_9_total_triggers_sink_dominant() {
        // Arrange: 3/9 ≈ 33.3% > 30% → 3*10=30, 9*3=27, 30 > 27 → SinkDominant
        let mut patterns = vec![AttentionPattern::Sink; 3];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(6));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!((summary.sink_ratio - 3.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_tracker_history_record_stores_layer_idx_zero_correctly() {
        // Arrange: verify layer 0 is stored and retrievable without ambiguity
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: detect for layer 0
        tracker.detect_for_layer(0, 0.5, 0.85);
        // Assert
        let record = tracker.layer_record(0).unwrap();
        assert_eq!(record.layer_idx, 0);
        assert_eq!(record.pattern, AttentionPattern::SharpFocus);
        // Verify it's not confused with "not found"
        assert!(tracker.layer_record(1).is_none());
    }

    #[test]
    fn test_detect_all_normal_with_extreme_high_thresholds() {
        // Arrange: thresholds so extreme that only max_val=f32::MAX would be Sink
        let config = SinkDetectionConfig {
            sink_threshold: 0.9999999,
            sharp_focus_threshold: 0.9999999,
            diffuse_threshold: -0.0000001,
            protected_sink_count: 2,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act & Assert: typical values all fall into Normal
        assert_eq!(tracker.detect_for_layer(0, 0.95, 0.95), AttentionPattern::Normal);
        assert_eq!(tracker.detect_for_layer(1, 0.99, 0.99), AttentionPattern::Normal);
        assert_eq!(tracker.detect_for_layer(2, 0.5, 0.5), AttentionPattern::Normal);
        // But f32::MAX still triggers Sink
        assert_eq!(tracker.detect_for_layer(3, f32::MAX, 0.5), AttentionPattern::Sink);
    }

    #[test]
    fn test_batch_summary_mixed_6_sink_4_normal_10_total_sink_dominant() {
        // Arrange: 6/10 = 60% > 30% → SinkDominant
        let mut patterns = vec![AttentionPattern::Sink; 6];
        patterns.extend(std::iter::repeat(AttentionPattern::Normal).take(4));
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.7);
        // Assert
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!(summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
        assert!((summary.sink_ratio - 0.6).abs() < 0.01);
        assert_eq!(summary.per_request_patterns.len(), 10);
    }

    // ========================================================================
    // Additional 13 tests — remaining edge cases
    // ========================================================================

    #[test]
    fn test_detect_for_layer_negative_sharpness_exactly_at_threshold() {
        // Arrange: sharpness = -0.1 which equals default diffuse_threshold * -1
        // -0.1 < 0.1 (diffuse_threshold) → Diffuse
        let config = SinkDetectionConfig {
            diffuse_threshold: -0.1,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: sharpness = -0.1, diffuse_threshold = -0.1 → NOT Diffuse (strictly less)
        let pattern = tracker.detect_for_layer(0, 0.5, -0.1);
        // Assert: -0.1 < -0.1 is false → Normal
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_detect_for_layer_max_val_just_below_default_sink_threshold() {
        // Arrange: default sink_threshold = 0.9
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act: 0.89 is clearly below 0.9
        let pattern = tracker.detect_for_layer(0, 0.89, 0.5);
        // Assert: 0.89 < 0.9 → not Sink, 0.5 not > 0.8, not < 0.1 → Normal
        assert_eq!(pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_tracker_history_preserves_sharpness_after_multiple_same_layer_updates() {
        // Arrange: verify the exact sharpness value from the last update is stored
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.5, 0.9);
        tracker.detect_for_layer(0, 0.5, 0.1);
        tracker.detect_for_layer(0, 0.5, 0.77);
        // Assert: last update wins
        let record = tracker.layer_record(0).unwrap();
        assert!((record.sharpness - 0.77).abs() < f32::EPSILON);
        assert!((record.softmax_max - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sink_ratio_after_one_normal_detection() {
        // Arrange: single Normal detection → all ratios should be 0
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        // Act
        tracker.detect_for_layer(0, 0.3, 0.4);
        // Assert
        assert_eq!(tracker.total_detections(), 1);
        assert!((tracker.sink_ratio() - 0.0).abs() < f32::EPSILON);
        assert!((tracker.sharp_focus_ratio() - 0.0).abs() < f32::EPSILON);
        assert!((tracker.diffuse_ratio() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_summary_two_sinks_two_diffuses_two_normals() {
        // Arrange: 2/6 sink = 33.3% > 30%, 2/6 diffuse = 33.3%
        // Sink checked first → SinkDominant
        let patterns = vec![
            AttentionPattern::Sink, AttentionPattern::Sink,
            AttentionPattern::Diffuse, AttentionPattern::Diffuse,
            AttentionPattern::Normal, AttentionPattern::Normal,
        ];
        // Act
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.4);
        // Assert: sink takes precedence
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::SinkDominant);
        assert!((summary.sink_ratio - 2.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_record_debug_output_contains_float_values() {
        // Arrange: verify Debug output includes the numeric field values
        let record = LayerAttentionRecord {
            layer_idx: 5,
            softmax_max: 0.88,
            sharpness: 0.66,
            pattern: AttentionPattern::Normal,
        };
        // Act
        let debug = format!("{:?}", record);
        // Assert: numeric values appear in debug string
        assert!(debug.contains("0.88"));
        assert!(debug.contains("0.66"));
        assert!(debug.contains("layer_idx: 5"));
    }

    #[test]
    fn test_detect_from_aggregator_with_subnormal_values() {
        // Arrange: aggregator with subnormal max and sharpness
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 4);
        let mut agg = TelemetryAggregator::new();
        let subnormal = f32::from_bits(1u32);
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: subnormal,
            sharpness: subnormal,
        });
        // Act
        let pattern = tracker.detect_from_aggregator(0, &agg);
        // Assert: both very small → Diffuse
        assert_eq!(pattern, AttentionPattern::Diffuse);
        assert_eq!(tracker.total_detections(), 1);
    }

    #[test]
    fn test_tracker_detect_mixed_then_verify_each_count() {
        // Arrange: 3 Sink, 2 SharpFocus, 1 Diffuse, 2 Normal = 8 total
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        // Act
        tracker.detect_for_layer(0, 0.95, 0.9);   // Sink
        tracker.detect_for_layer(1, 0.95, 0.9);   // Sink
        tracker.detect_for_layer(2, 0.95, 0.9);   // Sink
        tracker.detect_for_layer(3, 0.5, 0.85);   // SharpFocus
        tracker.detect_for_layer(4, 0.5, 0.85);   // SharpFocus
        tracker.detect_for_layer(5, 0.1, 0.05);   // Diffuse
        tracker.detect_for_layer(6, 0.3, 0.4);    // Normal
        tracker.detect_for_layer(7, 0.3, 0.4);    // Normal
        // Assert: exact counts
        assert_eq!(tracker.sink_count, 3);
        assert_eq!(tracker.sharp_focus_count, 2);
        assert_eq!(tracker.diffuse_count, 1);
        assert_eq!(tracker.normal_count, 2);
        assert_eq!(tracker.total_detections(), 8);
    }

    #[test]
    fn test_config_with_zero_protected_count_detection_still_works() {
        // Arrange: zero protected sinks but detection still functions
        let config = SinkDetectionConfig {
            protected_sink_count: 0,
            ..SinkDetectionConfig::default()
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: detection should still classify correctly
        let pattern = tracker.detect_for_layer(0, 0.95, 0.9);
        // Assert
        assert_eq!(pattern, AttentionPattern::Sink);
        assert_eq!(tracker.protected_sink_count(), 0);
        assert!(!tracker.needs_sink_protection(0));
    }

    #[test]
    fn test_batch_summary_empty_preserves_sharpness_zero() {
        // Arrange: empty patterns with zero sharpness
        // Act
        let summary = BatchAttentionSummary::from_patterns(&[], 0.0);
        // Assert
        assert_eq!(summary.per_request_patterns.len(), 0);
        assert!((summary.avg_sharpness - 0.0).abs() < f32::EPSILON);
        assert!((summary.sink_ratio - 0.0).abs() < f32::EPSILON);
        assert_eq!(summary.batch_advice, BatchAttentionAdvice::NormalAttention);
    }

    #[test]
    fn test_recent_patterns_returns_exact_reverse_order_for_four_layers() {
        // Arrange: 4 layers detected in order 10, 20, 30, 40
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(10, 0.95, 0.9);
        tracker.detect_for_layer(20, 0.5, 0.85);
        tracker.detect_for_layer(30, 0.1, 0.05);
        tracker.detect_for_layer(40, 0.3, 0.4);
        // Act: request 4 recent
        let recent = tracker.recent_patterns(4);
        // Assert: reversed insertion order
        assert_eq!(recent.len(), 4);
        assert_eq!(recent[0].layer_idx, 40);
        assert_eq!(recent[1].layer_idx, 30);
        assert_eq!(recent[2].layer_idx, 20);
        assert_eq!(recent[3].layer_idx, 10);
    }

    #[test]
    fn test_detect_for_layer_sharp_focus_with_max_just_below_sink() {
        // Arrange: max_val slightly below sink_threshold, high sharpness
        let config = SinkDetectionConfig {
            sink_threshold: 0.9,
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
            protected_sink_count: 4,
        };
        let mut tracker = SinkTracker::new(config, 4);
        // Act: max=0.899 < 0.9 (not Sink), sharpness=0.9 > 0.8 → SharpFocus
        let pattern = tracker.detect_for_layer(0, 0.899, 0.9);
        // Assert
        assert_eq!(pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_reset_then_detect_for_layer_via_all_three_methods() {
        // Arrange: fill history, reset, then use all three methods
        use crate::kv_cache::f32_to_f16_bits;
        let config = SinkDetectionConfig::default();
        let mut tracker = SinkTracker::new(config, 8);
        tracker.detect_for_layer(0, 0.95, 0.9);
        tracker.detect_for_layer(1, 0.5, 0.85);
        tracker.reset_stats();
        // Act: all three methods post-reset
        tracker.detect_for_layer(5, 0.95, 0.9);  // Sink via direct
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&super::super::epilogue::EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        tracker.detect_from_aggregator(6, &agg);  // SharpFocus via aggregator
        let mut header = KvPageHeader::new(0);
        header.softmax_max_avg = f32_to_f16_bits(0.1);
        header.centroid_pos = f32_to_f16_bits(0.05);
        tracker.detect_from_page_header(7, &header);  // Diffuse via page header
        // Assert: only post-reset data, 3 entries, correct counts
        assert_eq!(tracker.total_detections(), 3);
        assert_eq!(tracker.history().len(), 3);
        assert_eq!(tracker.sink_count, 1);
        assert_eq!(tracker.sharp_focus_count, 1);
        assert_eq!(tracker.diffuse_count, 1);
        assert!(tracker.layer_record(0).is_none());
        assert!(tracker.layer_record(1).is_none());
    }
}
