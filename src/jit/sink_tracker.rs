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
}
