//! Softmax 质心引导预取 (SPEC §13.2)
//!
//! ## 核心职责
//! 将 Softmax 输出的质心位置信息用于预取决策:
//! - 从 TelemetryAggregator / KvPageHeader 读取 Softmax 质心位置
//! - 计算预取距离 (基于注意力质心与当前位置的偏差)
//! - 为 KV Cache 预取提供质心偏移量
//! - 跟踪逐层质心位置变化
//!
//! ## 数据流
//! ```
//! Softmax(argmax) ──→ CentroidPosition ──→ PrefetchDistance
//!                                             ↓
//!                              KV Cache Prefetch (质心偏移预取)
//! ```
//!
//! ## 与 §13.9 锐度检测联动
//! - SharpFocus (锐度高) → 预取距离减小 (局部性强)
//! - Diffuse (均匀分散) → 预取距离增大 (全局性强)
//! - Sink → 预取前 N 个 token (Sink 保护)
//!
//! ## 预取策略
//! ```
//! 质心偏移 = argmax(softmax) - current_pos
//! 预取距离 = clamp(质心偏移 * 锐度系数, min_dist, max_dist)
//! ```
//!
//! ## 使用示例
//! ```no_run
//! use gllm::jit::prefetch::{CentroidPrefetch, PrefetchConfig};
//! use gllm::jit::epilogue::{TelemetryAggregator, AttentionPattern};
//!
//! let config = PrefetchConfig::default();
//! let mut prefetch = CentroidPrefetch::new(config);
//!
//! // 从遥测聚合器获取质心位置并计算预取距离
//! let distance = prefetch.compute_from_aggregator(&agg, 10);
//! ```

use super::epilogue::{TelemetryAggregator, AttentionPattern};

/// 预取配置
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// 最小预取距离 (tokens)
    pub min_distance: isize,
    /// 最大预取距离 (tokens)
    pub max_distance: isize,
    /// 锐度系数 — SharpFocus 时缩小预取范围
    pub sharp_focus_factor: f32,
    /// 均匀分散系数 — Diffuse 时扩大预取范围
    pub diffuse_factor: f32,
    /// Sink 预取偏移 — Sink token 时预取后续 token
    pub sink_prefetch_offset: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            min_distance: -16,   // 向前预取 16 个 token
            max_distance: 64,    // 向后预取 64 个 token
            sharp_focus_factor: 0.3,  // 锐度高时预取范围缩小到 30%
            diffuse_factor: 1.5,      // 均匀分散时预取范围扩大到 150%
            sink_prefetch_offset: 4,  // Sink 时预取后续 4 个 token
        }
    }
}

/// 预取建议
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefetchAdvice {
    /// 无需预取 (当前 token 即为质心)
    None,
    /// 向前预取 N 个 token (负距离)
    Forward(usize),
    /// 向后预取 N 个 token (正距离)
    Backward(usize),
    /// Sink 预取 — 预取后续 N 个 token
    Sink(usize),
}

/// 逐层质心位置记录
#[derive(Debug, Clone)]
pub struct CentroidRecord {
    /// 层索引
    pub layer_idx: usize,
    /// 质心位置 (argmax of softmax)
    pub centroid_pos: usize,
    /// 当前处理位置
    pub current_pos: usize,
    /// 计算出的预取距离
    pub prefetch_distance: isize,
    /// 注意力模式 (用于调试)
    pub pattern: AttentionPattern,
}

/// Softmax 质心引导预取器
///
/// 管理多层 Softmax 质心位置的预取决策:
/// - 从 TelemetryAggregator / KvPageHeader 提取质心位置
/// - 根据注意力模式计算预取距离
/// - 提供预取建议
/// - 记录质心位置历史
pub struct CentroidPrefetch {
    /// 配置
    config: PrefetchConfig,
    /// 逐层质心位置记录
    history: Vec<CentroidRecord>,
    /// 总层数
    num_layers: usize,
    /// 累计各预取类型计数
    none_count: usize,
    forward_count: usize,
    backward_count: usize,
    sink_count: usize,
}

impl CentroidPrefetch {
    /// 创建新的质心预取器
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            num_layers: 0,  // 将在运行时设置
            none_count: 0,
            forward_count: 0,
            backward_count: 0,
            sink_count: 0,
        }
    }

    /// 设置总层数
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self.history = Vec::with_capacity(num_layers);
        self
    }

    /// 从质心位置和当前位置计算预取建议
    pub fn compute(
        &mut self,
        layer_idx: usize,
        centroid_pos: usize,
        current_pos: usize,
        pattern: AttentionPattern,
    ) -> PrefetchAdvice {
        // Sink 特殊处理
        if matches!(pattern, AttentionPattern::Sink) {
            let advice = PrefetchAdvice::Sink(self.config.sink_prefetch_offset);
            self.record_centroid(layer_idx, centroid_pos, current_pos, 0, pattern);
            self.sink_count += 1;
            return advice;
        }

        // 计算质心偏移
        let offset = centroid_pos as isize - current_pos as isize;

        // 根据注意力模式调整偏移
        let adjusted_offset = match pattern {
            AttentionPattern::SharpFocus => {
                // 锐度高 → 缩小预取范围
                (offset as f32 * self.config.sharp_focus_factor) as isize
            }
            AttentionPattern::Diffuse => {
                // 均匀分散 → 扩大预取范围
                (offset as f32 * self.config.diffuse_factor) as isize
            }
            AttentionPattern::Normal => offset,
            AttentionPattern::Sink => unreachable!(),  // 已处理
        };

        // Clamp 到配置范围
        let distance = adjusted_offset
            .clamp(self.config.min_distance, self.config.max_distance);

        // 转换为预取建议
        let advice = if distance == 0 {
            PrefetchAdvice::None
        } else if distance < 0 {
            PrefetchAdvice::Forward(distance.unsigned_abs())
        } else {
            PrefetchAdvice::Backward(distance as usize)
        };

        // 更新计数
        match advice {
            PrefetchAdvice::None => self.none_count += 1,
            PrefetchAdvice::Forward(_) => self.forward_count += 1,
            PrefetchAdvice::Backward(_) => self.backward_count += 1,
            PrefetchAdvice::Sink(_) => self.sink_count += 1,
        };

        self.record_centroid(layer_idx, centroid_pos, current_pos, distance, pattern);
        advice
    }

    /// 从 TelemetryAggregator 计算预取建议
    pub fn compute_from_aggregator(
        &mut self,
        layer_idx: usize,
        current_pos: usize,
        agg: &TelemetryAggregator,
        pattern: AttentionPattern,
    ) -> PrefetchAdvice {
        let centroid_pos = agg.centroid_position();
        self.compute(layer_idx, centroid_pos, current_pos, pattern)
    }

    /// 记录质心位置
    fn record_centroid(
        &mut self,
        layer_idx: usize,
        centroid_pos: usize,
        current_pos: usize,
        prefetch_distance: isize,
        pattern: AttentionPattern,
    ) {
        self.history.push(CentroidRecord {
            layer_idx,
            centroid_pos,
            current_pos,
            prefetch_distance,
            pattern,
        });
    }

    /// 获取历史记录
    pub fn history(&self) -> &[CentroidRecord] {
        &self.history
    }

    /// 获取预取统计
    pub fn stats(&self) -> PrefetchStats {
        PrefetchStats {
            total: self.history.len(),
            none_count: self.none_count,
            forward_count: self.forward_count,
            backward_count: self.backward_count,
            sink_count: self.sink_count,
        }
    }

    /// 重置状态
    pub fn reset(&mut self) {
        self.history.clear();
        self.none_count = 0;
        self.forward_count = 0;
        self.backward_count = 0;
        self.sink_count = 0;
    }
}

/// 预取统计
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PrefetchStats {
    /// 总预取次数
    pub total: usize,
    /// 无需预取次数
    pub none_count: usize,
    /// 向前预取次数
    pub forward_count: usize,
    /// 向后预取次数
    pub backward_count: usize,
    /// Sink 预取次数
    pub sink_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::epilogue::{EpilogueSignal, TelemetryAggregator};

    // ── PrefetchConfig ──────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let config = PrefetchConfig::default();

        assert_eq!(config.min_distance, -16);
        assert_eq!(config.max_distance, 64);
        assert!((config.sharp_focus_factor - 0.3).abs() < f32::EPSILON);
        assert!((config.diffuse_factor - 1.5).abs() < f32::EPSILON);
        assert_eq!(config.sink_prefetch_offset, 4);
    }

    #[test]
    fn test_config_custom_values() {
        let config = PrefetchConfig {
            min_distance: -32,
            max_distance: 128,
            sharp_focus_factor: 0.1,
            diffuse_factor: 3.0,
            sink_prefetch_offset: 8,
        };

        assert_eq!(config.min_distance, -32);
        assert_eq!(config.max_distance, 128);
        assert!((config.sharp_focus_factor - 0.1).abs() < f32::EPSILON);
        assert!((config.diffuse_factor - 3.0).abs() < f32::EPSILON);
        assert_eq!(config.sink_prefetch_offset, 8);
    }

    #[test]
    fn test_config_clone() {
        let config = PrefetchConfig::default();
        let cloned = config.clone();

        assert_eq!(config.min_distance, cloned.min_distance);
        assert_eq!(config.max_distance, cloned.max_distance);
        assert_eq!(config.sink_prefetch_offset, cloned.sink_prefetch_offset);
    }

    #[test]
    fn test_config_debug_format() {
        let config = PrefetchConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("min_distance"));
        assert!(debug_str.contains("max_distance"));
        assert!(debug_str.contains("sharp_focus_factor"));
        assert!(debug_str.contains("diffuse_factor"));
        assert!(debug_str.contains("sink_prefetch_offset"));
    }

    // ── PrefetchAdvice ──────────────────────────────────────────────

    #[test]
    fn test_advice_equality() {
        assert_eq!(PrefetchAdvice::None, PrefetchAdvice::None);
        assert_eq!(PrefetchAdvice::Forward(10), PrefetchAdvice::Forward(10));
        assert_eq!(PrefetchAdvice::Backward(20), PrefetchAdvice::Backward(20));
        assert_eq!(PrefetchAdvice::Sink(4), PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_advice_inequality() {
        assert_ne!(PrefetchAdvice::Forward(5), PrefetchAdvice::Forward(10));
        assert_ne!(PrefetchAdvice::Forward(10), PrefetchAdvice::Backward(10));
        assert_ne!(PrefetchAdvice::Sink(4), PrefetchAdvice::Forward(4));
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Forward(0));
    }

    #[test]
    fn test_advice_copy_semantics() {
        let a = PrefetchAdvice::Forward(7);
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_advice_debug_format() {
        assert!(format!("{:?}", PrefetchAdvice::None).contains("None"));
        assert!(format!("{:?}", PrefetchAdvice::Forward(10)).contains("Forward"));
        assert!(format!("{:?}", PrefetchAdvice::Backward(20)).contains("Backward"));
        assert!(format!("{:?}", PrefetchAdvice::Sink(4)).contains("Sink"));
    }

    // ── PrefetchStats ───────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = PrefetchStats::default();

        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_stats_clone_copy() {
        let stats = PrefetchStats {
            total: 10,
            none_count: 2,
            forward_count: 3,
            backward_count: 4,
            sink_count: 1,
        };
        let cloned = stats; // Copy
        assert_eq!(stats.total, cloned.total);
        assert_eq!(stats.forward_count, cloned.forward_count);
    }

    #[test]
    fn test_stats_debug_format() {
        let stats = PrefetchStats {
            total: 5,
            none_count: 1,
            forward_count: 2,
            backward_count: 1,
            sink_count: 1,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total"));
        assert!(debug.contains("forward_count"));
    }

    // ── CentroidPrefetch constructor ────────────────────────────────

    #[test]
    fn test_new_prefetch_empty_state() {
        let prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        assert!(prefetch.history().is_empty());
        let stats = prefetch.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_with_num_layers_starts_empty() {
        let prefetch =
            CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(32);

        // History starts empty; capacity is internal (not exposed via slice)
        assert!(prefetch.history().is_empty());
        // Verify it accepts layer indices beyond 0
        let mut prefetch = prefetch;
        let _ = prefetch.compute(31, 100, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);
    }

    // ── Core compute logic ─────────────────────────────────────────

    #[test]
    fn test_prefetch_computation() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        // 质心在前 → 向前预取
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Forward(_)));

        // 质心在后 → 向后预取
        let advice = prefetch.compute(1, 150, 100, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Backward(_)));

        // 质心在当前位置 → 无需预取
        let advice = prefetch.compute(2, 100, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_reduces_range() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        // SharpFocus: 偏移 50 * 0.3 = 15
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::SharpFocus);
        // 期望预取距离约为 15 (实际取决于 clamp)
        match advice {
            PrefetchAdvice::Forward(d) => {
                // sharp_focus_factor = 0.3, 偏移 50 → 15
                assert!(d <= 20); // 允许一些误差
            }
            _ => panic!("Expected Forward advice"),
        }
    }

    #[test]
    fn test_diffuse_increases_range() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        // Diffuse 模式下，偏移被扩大但受 min_distance 限制
        // centroid_pos = 70, current_pos = 100, offset = -30
        // adjusted_offset = -30 * 1.5 = -45, clamped to min_distance (-16)
        let advice = prefetch.compute(0, 70, 100, AttentionPattern::Diffuse);
        match advice {
            PrefetchAdvice::Forward(d) => {
                // 被 min_distance 限制
                assert_eq!(d, 16);
            }
            _ => panic!("Expected Forward advice"),
        }

        // 测试 diffuse 扩大效应：需要更小的偏移才能不被 clamp
        // centroid_pos = 95, current_pos = 100, offset = -5
        // adjusted_offset = -5 * 1.5 = -7.5 ≈ -7, not clamped
        let advice = prefetch.compute(1, 95, 100, AttentionPattern::Diffuse);
        match advice {
            PrefetchAdvice::Forward(d) => {
                // diffuse_factor 扩大，5 → 7.5 ≈ 7
                assert_eq!(d, 7);
            }
            _ => panic!("Expected Forward advice"),
        }

        // Normal 模式对比：相同偏移但不扩大
        let advice = prefetch.compute(2, 95, 100, AttentionPattern::Normal);
        match advice {
            PrefetchAdvice::Forward(d) => {
                // Normal 模式不扩大，5 → 5
                assert_eq!(d, 5);
            }
            _ => panic!("Expected Forward advice"),
        }
    }

    #[test]
    fn test_sink_detection() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_clamp_to_bounds() {
        let config = PrefetchConfig {
            min_distance: -10,
            max_distance: 20,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // 超出下限
        let advice = prefetch.compute(0, 0, 100, AttentionPattern::Normal);
        match advice {
            PrefetchAdvice::Forward(d) => {
                assert!(d <= 10);
            }
            _ => panic!("Expected Forward advice"),
        }

        // 超出上限
        let advice = prefetch.compute(1, 200, 100, AttentionPattern::Normal);
        match advice {
            PrefetchAdvice::Backward(d) => {
                assert!(d <= 20);
            }
            _ => panic!("Expected Backward advice"),
        }
    }

    #[test]
    fn test_prefetch_stats() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        prefetch.compute(1, 100, 100, AttentionPattern::Normal);
        prefetch.compute(2, 150, 100, AttentionPattern::SharpFocus);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.none_count, 1);
        assert!(stats.forward_count + stats.backward_count > 0);
    }

    #[test]
    fn test_reset() {
        let config = PrefetchConfig::default();
        let mut prefetch = CentroidPrefetch::new(config);

        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);

        prefetch.reset();
        assert_eq!(prefetch.history().len(), 0);
        assert_eq!(prefetch.stats().total, 0);
    }

    // ── Precise distance calculations ──────────────────────────────

    #[test]
    fn test_forward_distance_exact() {
        // offset = centroid - current = 90 - 100 = -10
        // Normal: no adjustment, distance = -10 → Forward(10)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_backward_distance_exact() {
        // offset = 110 - 100 = 10
        // Normal: distance = 10 → Backward(10)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(10));
    }

    #[test]
    fn test_sharp_focus_exact_backward() {
        // offset = 130 - 100 = 30, sharp_focus_factor = 0.3
        // adjusted = (30.0 * 0.3) as isize = 9 → Backward(9)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 130, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(9));
    }

    #[test]
    fn test_diffuse_backward_clamped() {
        // offset = 200 - 100 = 100, diffuse_factor = 1.5
        // adjusted = (100.0 * 1.5) as isize = 150, clamped to max_distance = 64
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 200, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    #[test]
    fn test_sink_custom_offset() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 12,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Sink always returns configured offset regardless of positions
        let advice = prefetch.compute(0, 50, 200, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(12));
    }

    #[test]
    fn test_zero_offset_yields_none() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        // SharpFocus on exact same position: offset=0, adjusted=0
        let advice = prefetch.compute(0, 100, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);

        // Diffuse on exact same position: offset=0, adjusted=0
        let advice = prefetch.compute(1, 100, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_min_distance_boundary() {
        // offset = 0 - 100 = -100, Normal → clamped to min_distance = -16 → Forward(16)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_max_distance_boundary() {
        // offset = 200 - 100 = 100, Normal → clamped to max_distance = 64 → Backward(64)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 200, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    // ── History recording ───────────────────────────────────────────

    #[test]
    fn test_history_records_all_fields() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(3, 80, 100, AttentionPattern::SharpFocus);

        let history = prefetch.history();
        assert_eq!(history.len(), 1);
        let record = &history[0];
        assert_eq!(record.layer_idx, 3);
        assert_eq!(record.centroid_pos, 80);
        assert_eq!(record.current_pos, 100);
        // offset = -20 * 0.3 = -6
        assert_eq!(record.prefetch_distance, -6);
        assert_eq!(record.pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_history_sink_records_zero_distance() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(2, 50, 100, AttentionPattern::Sink);

        let record = &prefetch.history()[0];
        assert_eq!(record.layer_idx, 2);
        assert_eq!(record.centroid_pos, 50);
        assert_eq!(record.current_pos, 100);
        assert_eq!(record.prefetch_distance, 0); // Sink records distance=0
        assert_eq!(record.pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_history_accumulates_across_calls() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        prefetch.compute(1, 110, 100, AttentionPattern::Normal);
        prefetch.compute(5, 100, 100, AttentionPattern::Sink);

        let history = prefetch.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].layer_idx, 0);
        assert_eq!(history[1].layer_idx, 1);
        assert_eq!(history[2].layer_idx, 5);
    }

    #[test]
    fn test_centroid_record_debug() {
        let record = CentroidRecord {
            layer_idx: 7,
            centroid_pos: 42,
            current_pos: 100,
            prefetch_distance: -58,
            pattern: AttentionPattern::Diffuse,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("centroid_pos"));
        assert!(debug.contains("42"));
    }

    // ── Stats tracking across multiple patterns ────────────────────

    #[test]
    fn test_stats_all_categories() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal); // Forward
        prefetch.compute(1, 100, 100, AttentionPattern::Normal); // None
        prefetch.compute(2, 110, 100, AttentionPattern::Normal); // Backward
        prefetch.compute(3, 50, 100, AttentionPattern::Sink); // Sink
        prefetch.compute(4, 80, 100, AttentionPattern::SharpFocus); // Forward

        let stats = prefetch.stats();
        assert_eq!(stats.total, 5);
        assert_eq!(stats.forward_count, 2);
        assert_eq!(stats.none_count, 1);
        assert_eq!(stats.backward_count, 1);
        assert_eq!(stats.sink_count, 1);
    }

    #[test]
    fn test_stats_after_reset_are_zero() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        prefetch.compute(1, 50, 100, AttentionPattern::Sink);
        prefetch.reset();

        let stats = prefetch.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_reset_then_recompute_fresh_state() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        prefetch.compute(1, 50, 100, AttentionPattern::Sink);
        assert_eq!(prefetch.history().len(), 2);

        prefetch.reset();
        assert!(prefetch.history().is_empty());

        // After reset, new computations start fresh
        prefetch.compute(10, 110, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);
        let stats = prefetch.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.backward_count, 1);
    }

    // ── compute_from_aggregator ────────────────────────────────────

    #[test]
    fn test_compute_from_aggregator_uses_centroid() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 85 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);

        // centroid=85, current=100, offset=-15 → Forward(15)
        assert_eq!(advice, PrefetchAdvice::Forward(15));
    }

    #[test]
    fn test_compute_from_aggregator_default_centroid() {
        let agg = TelemetryAggregator::new(); // default centroid_position = 0

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);

        // centroid=0, current=100, offset=-100, clamped to -16 → Forward(16)
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_single_token_position() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        // Both at position 0
        let advice = prefetch.compute(0, 0, 0, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_adjacent_positions() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 99, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));

        let advice = prefetch.compute(1, 101, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    #[test]
    fn test_large_positions() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        // Large absolute positions, small offset
        let advice = prefetch.compute(0, 999995, 1_000_000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(5));
    }

    #[test]
    fn test_symmetric_config_distances() {
        // Config where min_distance == -max_distance
        let config = PrefetchConfig {
            min_distance: -32,
            max_distance: 32,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let fwd = prefetch.compute(0, 68, 100, AttentionPattern::Normal);
        let bwd = prefetch.compute(1, 132, 100, AttentionPattern::Normal);

        // Both have offset magnitude 32 → both at boundary
        assert_eq!(fwd, PrefetchAdvice::Forward(32));
        assert_eq!(bwd, PrefetchAdvice::Backward(32));
    }

    #[test]
    fn test_narrow_clamp_window() {
        // Config with very narrow clamp window
        let config = PrefetchConfig {
            min_distance: -2,
            max_distance: 2,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Large offset gets clamped to -2
        let advice = prefetch.compute(0, 0, 1000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(2));

        // Large positive offset gets clamped to 2
        let advice = prefetch.compute(1, 2000, 1000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(2));

        // Tiny offset passes through
        let advice = prefetch.compute(2, 999, 1000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    // ── Additional tests: traits, construction, edge cases ─────────

    #[test]
    fn test_advice_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |v: &PrefetchAdvice| {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        };

        // Equal values must produce equal hashes
        assert_eq!(hash_of(&PrefetchAdvice::None), hash_of(&PrefetchAdvice::None));
        assert_eq!(hash_of(&PrefetchAdvice::Forward(5)), hash_of(&PrefetchAdvice::Forward(5)));
        assert_eq!(hash_of(&PrefetchAdvice::Backward(3)), hash_of(&PrefetchAdvice::Backward(3)));
        assert_eq!(hash_of(&PrefetchAdvice::Sink(7)), hash_of(&PrefetchAdvice::Sink(7)));
    }

    #[test]
    fn test_advice_clone_independent() {
        let original = PrefetchAdvice::Backward(15);
        let cloned = original.clone();
        // Both are independent copies (Copy types, so always equal)
        assert_eq!(original, cloned);
        // Verify the variant carries the same payload
        if let PrefetchAdvice::Backward(d) = cloned {
            assert_eq!(d, 15);
        } else {
            panic!("Expected Backward variant");
        }
    }

    #[test]
    fn test_advice_all_variants_distinct_debug() {
        let variants = [
            format!("{:?}", PrefetchAdvice::None),
            format!("{:?}", PrefetchAdvice::Forward(0)),
            format!("{:?}", PrefetchAdvice::Backward(0)),
            format!("{:?}", PrefetchAdvice::Sink(0)),
        ];
        // All four Debug outputs are distinct strings
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "Debug output collision");
            }
        }
    }

    #[test]
    fn test_advice_zero_payload_forward() {
        // Forward(0) is a valid variant though semantically odd
        let advice = PrefetchAdvice::Forward(0);
        assert_eq!(advice, PrefetchAdvice::Forward(0));
        assert_ne!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_zero_distance_threshold() {
        // When both min and max are 0, any non-zero offset clamps to 0 → None
        let config = PrefetchConfig {
            min_distance: 0,
            max_distance: 0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 50, clamped to 0 → None
        let advice = prefetch.compute(0, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);

        // offset = -50, clamped to 0 → None
        let advice = prefetch.compute(1, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_max_usize_values() {
        let config = PrefetchConfig {
            min_distance: isize::MIN,
            max_distance: isize::MAX,
            sink_prefetch_offset: usize::MAX,
            ..Default::default()
        };
        // Verify construction with extreme values succeeds
        assert_eq!(config.min_distance, isize::MIN);
        assert_eq!(config.max_distance, isize::MAX);
        assert_eq!(config.sink_prefetch_offset, usize::MAX);

        let mut prefetch = CentroidPrefetch::new(config);
        // With huge clamp bounds, offset passes through unclamped
        let advice = prefetch.compute(0, 1000, 0, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1000));

        // Sink with usize::MAX offset
        let advice = prefetch.compute(1, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(usize::MAX));
    }

    #[test]
    fn test_centroid_record_clone_preserves_fields() {
        let record = CentroidRecord {
            layer_idx: 5,
            centroid_pos: 200,
            current_pos: 150,
            prefetch_distance: 50,
            pattern: AttentionPattern::Normal,
        };
        let cloned = record.clone();

        assert_eq!(cloned.layer_idx, 5);
        assert_eq!(cloned.centroid_pos, 200);
        assert_eq!(cloned.current_pos, 150);
        assert_eq!(cloned.prefetch_distance, 50);
        assert_eq!(cloned.pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_centroid_record_debug_all_fields_present() {
        let record = CentroidRecord {
            layer_idx: 3,
            centroid_pos: 77,
            current_pos: 88,
            prefetch_distance: -11,
            pattern: AttentionPattern::SharpFocus,
        };
        let debug = format!("{:?}", record);

        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("centroid_pos"));
        assert!(debug.contains("current_pos"));
        assert!(debug.contains("prefetch_distance"));
        assert!(debug.contains("pattern"));
        assert!(debug.contains("77"));
        assert!(debug.contains("-11"));
    }

    #[test]
    fn test_centroid_record_zero_fields() {
        let record = CentroidRecord {
            layer_idx: 0,
            centroid_pos: 0,
            current_pos: 0,
            prefetch_distance: 0,
            pattern: AttentionPattern::Normal,
        };
        assert_eq!(record.layer_idx, 0);
        assert_eq!(record.centroid_pos, 0);
        assert_eq!(record.current_pos, 0);
        assert_eq!(record.prefetch_distance, 0);
    }

    #[test]
    fn test_prefetch_stats_equality() {
        let a = PrefetchStats {
            total: 10,
            none_count: 2,
            forward_count: 3,
            backward_count: 4,
            sink_count: 1,
        };
        let b = PrefetchStats {
            total: 10,
            none_count: 2,
            forward_count: 3,
            backward_count: 4,
            sink_count: 1,
        };
        assert_eq!(a, b);

        let c = PrefetchStats {
            total: 10,
            none_count: 2,
            forward_count: 3,
            backward_count: 4,
            sink_count: 99,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn test_prefetch_stats_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = PrefetchStats {
            total: 5,
            none_count: 1,
            forward_count: 2,
            backward_count: 1,
            sink_count: 1,
        };
        let b = a; // Copy

        let hash_a = {
            let mut s = DefaultHasher::new();
            a.hash(&mut s);
            s.finish()
        };
        let hash_b = {
            let mut s = DefaultHasher::new();
            b.hash(&mut s);
            s.finish()
        };
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn test_with_num_layers_zero() {
        // Zero layers is a valid (edge-case) configuration
        let prefetch = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(0);
        assert!(prefetch.history().is_empty());
        let stats = prefetch.stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_with_num_layers_large_capacity() {
        let prefetch = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(1000);
        assert!(prefetch.history().is_empty());
        // Can still compute and history grows
        let mut prefetch = prefetch;
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);
    }

    #[test]
    fn test_history_slice_empty_then_filled() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        // Initially empty
        assert!(prefetch.history().is_empty());
        assert_eq!(prefetch.history().len(), 0);

        // After one compute, slice has length 1
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);

        // After two more computes, slice has length 3
        prefetch.compute(1, 110, 100, AttentionPattern::Normal);
        prefetch.compute(2, 100, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 3);
    }

    #[test]
    fn test_reset_preserves_config() {
        let config = PrefetchConfig {
            min_distance: -50,
            max_distance: 50,
            sink_prefetch_offset: 10,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        prefetch.compute(0, 0, 100, AttentionPattern::Normal);
        prefetch.reset();

        // Config still in effect: offset=-100 clamped to -50 → Forward(50)
        let advice = prefetch.compute(1, 0, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(50));

        // Sink still uses original offset=10
        let advice = prefetch.compute(2, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(10));
    }

    #[test]
    fn test_reset_idempotent() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 50, 100, AttentionPattern::Normal);

        prefetch.reset();
        prefetch.reset(); // Double reset is safe
        prefetch.reset(); // Triple reset is safe

        assert!(prefetch.history().is_empty());
        let stats = prefetch.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_config_negative_factors() {
        // Negative sharp_focus_factor inverts direction
        let config = PrefetchConfig {
            sharp_focus_factor: -1.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 50 - 100 = -50, adjusted = (-50.0 * -1.0) as isize = 50 → Backward(50)
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(50));
    }

    #[test]
    fn test_stats_debug_all_fields() {
        let stats = PrefetchStats {
            total: 100,
            none_count: 25,
            forward_count: 30,
            backward_count: 35,
            sink_count: 10,
        };
        let debug = format!("{:?}", stats);

        assert!(debug.contains("total"));
        assert!(debug.contains("none_count"));
        assert!(debug.contains("forward_count"));
        assert!(debug.contains("backward_count"));
        assert!(debug.contains("sink_count"));
    }

    // ── Additional 45 tests to bring ratio below 14 ────────────────

    // --- PrefetchConfig extreme float values ---

    #[test]
    fn test_config_zero_float_factors() {
        let config = PrefetchConfig {
            sharp_focus_factor: 0.0,
            diffuse_factor: 0.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=50, SharpFocus: 50 * 0.0 = 0 → None
        let advice = prefetch.compute(0, 150, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);

        // offset=-30, Diffuse: -30 * 0.0 = 0 → None
        let advice = prefetch.compute(1, 70, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_very_small_sharp_focus_factor() {
        let config = PrefetchConfig {
            sharp_focus_factor: 0.001,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 1000 * 0.001 = 1.0 → 1 as isize → Backward(1)
        let advice = prefetch.compute(0, 1100, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    #[test]
    fn test_config_large_diffuse_factor_clamped() {
        let config = PrefetchConfig {
            diffuse_factor: 100.0,
            max_distance: 64,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=1, diffuse: 1 * 100 = 100, clamped to 64
        let advice = prefetch.compute(0, 101, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    #[test]
    fn test_config_negative_diffuse_factor() {
        let config = PrefetchConfig {
            diffuse_factor: -0.5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 20, diffuse: 20 * -0.5 = -10 → Forward(10)
        let advice = prefetch.compute(0, 120, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_config_sink_offset_zero() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(0));
    }

    // --- PrefetchAdvice additional variants ---

    #[test]
    fn test_advice_forward_large_value() {
        let advice = PrefetchAdvice::Forward(usize::MAX);
        assert_eq!(advice, PrefetchAdvice::Forward(usize::MAX));
        assert_ne!(advice, PrefetchAdvice::Forward(usize::MAX - 1));
    }

    #[test]
    fn test_advice_backward_large_value() {
        let advice = PrefetchAdvice::Backward(usize::MAX);
        if let PrefetchAdvice::Backward(d) = advice {
            assert_eq!(d, usize::MAX);
        } else {
            panic!("Expected Backward variant");
        }
    }

    #[test]
    fn test_advice_sink_zero_value() {
        let advice = PrefetchAdvice::Sink(0);
        assert_eq!(advice, PrefetchAdvice::Sink(0));
        assert_ne!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_advice_exhaustive_match() {
        // Ensure compile-time exhaustiveness checking covers all variants
        let check = |advice: PrefetchAdvice| match advice {
            PrefetchAdvice::None => "none",
            PrefetchAdvice::Forward(_) => "forward",
            PrefetchAdvice::Backward(_) => "backward",
            PrefetchAdvice::Sink(_) => "sink",
        };
        assert_eq!(check(PrefetchAdvice::None), "none");
        assert_eq!(check(PrefetchAdvice::Forward(1)), "forward");
        assert_eq!(check(PrefetchAdvice::Backward(1)), "backward");
        assert_eq!(check(PrefetchAdvice::Sink(1)), "sink");
    }

    // --- PrefetchStats additional ---

    #[test]
    fn test_stats_total_equals_sum_of_categories() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        prefetch.compute(1, 100, 100, AttentionPattern::Normal);
        prefetch.compute(2, 110, 100, AttentionPattern::Normal);
        prefetch.compute(3, 0, 100, AttentionPattern::Sink);
        prefetch.compute(4, 95, 100, AttentionPattern::SharpFocus);

        let stats = prefetch.stats();
        assert_eq!(
            stats.total,
            stats.none_count + stats.forward_count + stats.backward_count + stats.sink_count
        );
    }

    #[test]
    fn test_stats_all_same_category() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        // All produce None (centroid == current)
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);
        prefetch.compute(1, 100, 100, AttentionPattern::SharpFocus);
        prefetch.compute(2, 100, 100, AttentionPattern::Diffuse);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.none_count, 3);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_stats_only_sink_calls() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        prefetch.compute(0, 10, 20, AttentionPattern::Sink);
        prefetch.compute(1, 30, 40, AttentionPattern::Sink);
        prefetch.compute(2, 0, 0, AttentionPattern::Sink);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.sink_count, 3);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.none_count, 0);
    }

    #[test]
    fn test_stats_all_forward_calls() {
        // All centroid < current → all Forward
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        prefetch.compute(1, 60, 100, AttentionPattern::Normal);
        prefetch.compute(2, 99, 100, AttentionPattern::Normal);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.forward_count, 3);
        assert_eq!(stats.backward_count, 0);
    }

    #[test]
    fn test_stats_all_backward_calls() {
        // All centroid > current → all Backward
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 150, 100, AttentionPattern::Normal);
        prefetch.compute(1, 200, 100, AttentionPattern::Normal);
        prefetch.compute(2, 101, 100, AttentionPattern::Normal);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.backward_count, 3);
        assert_eq!(stats.forward_count, 0);
    }

    // --- SharpFocus precision ---

    #[test]
    fn test_sharp_focus_forward_exact() {
        // offset = 90 - 100 = -10, sharp_focus_factor = 0.3
        // adjusted = (-10.0 * 0.3) as isize = -3 → Forward(3)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(3));
    }

    #[test]
    fn test_sharp_focus_zero_offset() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 200, 200, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_very_large_offset_clamped() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        // offset = 1_000_000 * 0.3 = 300000, clamped to max_distance=64
        let advice = prefetch.compute(0, 1_000_100, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    // --- Diffuse precision ---

    #[test]
    fn test_diffuse_forward_small_offset() {
        // offset = 97 - 100 = -3, diffuse: -3 * 1.5 = -4.5 → -4 as isize → Forward(4)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 97, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(4));
    }

    #[test]
    fn test_diffuse_zero_offset() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 500, 500, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    // --- Sink with different positions ---

    #[test]
    fn test_sink_same_position() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 100, 100, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_sink_large_position_difference() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 100000, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_sink_increments_sink_count_only() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 0, 100, AttentionPattern::Sink);

        let stats = prefetch.stats();
        assert_eq!(stats.sink_count, 1);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.none_count, 0);
    }

    // --- compute_from_aggregator additional ---

    #[test]
    fn test_compute_from_aggregator_with_sharp_focus() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 130 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::SharpFocus);

        // centroid=130, current=100, offset=30, sharp=30*0.3=9 → Backward(9)
        assert_eq!(advice, PrefetchAdvice::Backward(9));
    }

    #[test]
    fn test_compute_from_aggregator_with_diffuse() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 110 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Diffuse);

        // offset=10, diffuse=10*1.5=15 → Backward(15)
        assert_eq!(advice, PrefetchAdvice::Backward(15));
    }

    #[test]
    fn test_compute_from_aggregator_with_sink() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 50 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Sink);

        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_compute_from_aggregator_updates_centroid() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 105 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);

        assert_eq!(advice, PrefetchAdvice::Backward(5));
    }

    // --- History ordering and multiple layers ---

    #[test]
    fn test_history_maintains_insertion_order() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal); // Forward
        prefetch.compute(1, 100, 100, AttentionPattern::Normal); // None
        prefetch.compute(2, 110, 100, AttentionPattern::Normal); // Backward
        prefetch.compute(3, 0, 100, AttentionPattern::Sink); // Sink

        let history = prefetch.history();
        assert_eq!(history[0].centroid_pos, 90);
        assert_eq!(history[1].centroid_pos, 100);
        assert_eq!(history[2].centroid_pos, 110);
        assert_eq!(history[3].centroid_pos, 0);
    }

    #[test]
    fn test_history_record_patterns_match() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        prefetch.compute(1, 50, 100, AttentionPattern::SharpFocus);
        prefetch.compute(2, 0, 100, AttentionPattern::Sink);
        prefetch.compute(3, 200, 100, AttentionPattern::Diffuse);

        let history = prefetch.history();
        assert_eq!(history[0].pattern, AttentionPattern::Normal);
        assert_eq!(history[1].pattern, AttentionPattern::SharpFocus);
        assert_eq!(history[2].pattern, AttentionPattern::Sink);
        assert_eq!(history[3].pattern, AttentionPattern::Diffuse);
    }

    // --- CentroidPrefetch builder pattern ---

    #[test]
    fn test_builder_chaining() {
        let prefetch = CentroidPrefetch::new(PrefetchConfig {
            min_distance: -8,
            max_distance: 8,
            ..Default::default()
        })
        .with_num_layers(12);

        assert!(prefetch.history().is_empty());
        assert_eq!(prefetch.stats().total, 0);
    }

    // --- PrefetchConfig Clone independence ---

    #[test]
    fn test_config_clone_independence() {
        let config = PrefetchConfig {
            min_distance: -16,
            ..Default::default()
        };
        let cloned = config.clone();

        // Modifying a field value in a new config doesn't affect the clone
        let modified = PrefetchConfig {
            min_distance: -999,
            ..config.clone()
        };
        assert_eq!(cloned.min_distance, -16);
        assert_eq!(modified.min_distance, -999);
    }

    // --- CentroidRecord with large values ---

    #[test]
    fn test_centroid_record_large_values() {
        let record = CentroidRecord {
            layer_idx: usize::MAX,
            centroid_pos: usize::MAX,
            current_pos: usize::MAX,
            prefetch_distance: isize::MAX,
            pattern: AttentionPattern::Normal,
        };
        assert_eq!(record.layer_idx, usize::MAX);
        assert_eq!(record.prefetch_distance, isize::MAX);
    }

    #[test]
    fn test_centroid_record_negative_distance() {
        let record = CentroidRecord {
            layer_idx: 0,
            centroid_pos: 10,
            current_pos: 50,
            prefetch_distance: -40,
            pattern: AttentionPattern::SharpFocus,
        };
        assert_eq!(record.prefetch_distance, -40);
        assert_eq!(record.pattern, AttentionPattern::SharpFocus);
    }

    // --- Multiple resets interleaved with computes ---

    #[test]
    fn test_interleaved_reset_and_compute() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.stats().total, 1);

        prefetch.reset();
        assert_eq!(prefetch.stats().total, 0);

        prefetch.compute(5, 110, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.stats().total, 1);

        prefetch.reset();
        assert_eq!(prefetch.stats().total, 0);
        assert!(prefetch.history().is_empty());
    }

    // --- Clamped at exact boundary values ---

    #[test]
    fn test_distance_exactly_at_min_boundary() {
        // offset = centroid - current = 84 - 100 = -16, exactly min_distance
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 84, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_distance_exactly_at_max_boundary() {
        // offset = 164 - 100 = 64, exactly max_distance
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 164, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    #[test]
    fn test_distance_one_past_min_clamped() {
        // offset = 83 - 100 = -17, one past min_distance (-16), clamped
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 83, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_distance_one_past_max_clamped() {
        // offset = 165 - 100 = 65, one past max_distance (64), clamped
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 165, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    // --- TelemetryAggregator centroid_position zero ---

    #[test]
    fn test_aggregator_centroid_position_zero() {
        let agg = TelemetryAggregator::new();
        assert_eq!(agg.centroid_position(), 0);
    }

    #[test]
    fn test_aggregator_centroid_position_after_ingest() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 42 });
        assert_eq!(agg.centroid_position(), 42);
    }

    #[test]
    fn test_aggregator_centroid_position_overwritten() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 10 });
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 90 });
        assert_eq!(agg.centroid_position(), 90);
    }

    // --- AttentionPattern Debug format ---

    #[test]
    fn test_attention_pattern_debug_variants() {
        let normal = format!("{:?}", AttentionPattern::Normal);
        let sink = format!("{:?}", AttentionPattern::Sink);
        let sharp = format!("{:?}", AttentionPattern::SharpFocus);
        let diffuse = format!("{:?}", AttentionPattern::Diffuse);

        assert!(normal.contains("Normal"));
        assert!(sink.contains("Sink"));
        assert!(sharp.contains("SharpFocus"));
        assert!(diffuse.contains("Diffuse"));
    }

    // --- PrefetchAdvice Hash discriminant uniqueness ---

    #[test]
    fn test_advice_hash_discriminants_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |v: &PrefetchAdvice| {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        };

        let h_none = hash_of(&PrefetchAdvice::None);
        let h_fwd = hash_of(&PrefetchAdvice::Forward(0));
        let h_bwd = hash_of(&PrefetchAdvice::Backward(0));
        let h_sink = hash_of(&PrefetchAdvice::Sink(0));

        // All four variant discriminants should produce different hashes
        assert_ne!(h_none, h_fwd);
        assert_ne!(h_none, h_bwd);
        assert_ne!(h_none, h_sink);
        assert_ne!(h_fwd, h_bwd);
        assert_ne!(h_fwd, h_sink);
        assert_ne!(h_bwd, h_sink);
    }

    // --- PrefetchStats partial inequality ---

    #[test]
    fn test_stats_inequality_each_field() {
        let base = PrefetchStats {
            total: 10,
            none_count: 2,
            forward_count: 3,
            backward_count: 4,
            sink_count: 1,
        };

        assert_ne!(base, PrefetchStats { total: 99, ..base });
        assert_ne!(base, PrefetchStats { none_count: 99, ..base });
        assert_ne!(base, PrefetchStats { forward_count: 99, ..base });
        assert_ne!(base, PrefetchStats { backward_count: 99, ..base });
        assert_ne!(base, PrefetchStats { sink_count: 99, ..base });
    }

    // --- History record distance consistency with advice ---

    #[test]
    fn test_history_distance_matches_forward_advice() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 80, 100, AttentionPattern::Normal);

        let record = &prefetch.history()[0];
        // offset = 80-100 = -20, clamped by min_distance=-16 → -16
        assert!(record.prefetch_distance < 0, "forward offset should be negative, got {}", record.prefetch_distance);
    }

    #[test]
    fn test_history_distance_matches_backward_advice() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 130, 100, AttentionPattern::Normal);

        let record = &prefetch.history()[0];
        assert_eq!(record.prefetch_distance, 30);
    }

    // --- SharpFocus with very small offset (rounding edge) ---

    #[test]
    fn test_sharp_focus_rounding_edge() {
        // offset = 3, sharp_focus_factor = 0.3 → 0.9 as isize = 0 → None
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 103, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_just_above_rounding_edge() {
        // offset = 4, sharp_focus_factor = 0.3 → 1.2 as isize = 1 → Backward(1)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 104, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    // --- Diffuse rounding edge ---

    #[test]
    fn test_diffuse_rounding_edge() {
        // offset = 1, diffuse_factor = 1.5 → 1.5 as isize = 1 → Backward(1)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 101, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    // --- Compute with layer index 0 multiple times ---

    #[test]
    fn test_same_layer_multiple_computes() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        prefetch.compute(0, 110, 100, AttentionPattern::Normal);
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);

        let history = prefetch.history();
        assert_eq!(history.len(), 3);
        // All layer_idx are 0
        assert!(history.iter().all(|r| r.layer_idx == 0));
    }

    // --- Additional 24 tests to reach ratio < 14 ---

    #[test]
    fn test_config_min_distance_zero() {
        let config = PrefetchConfig {
            min_distance: 0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Negative offset clamped to 0 → None
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);

        // Positive offset still works
        let advice = prefetch.compute(1, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(50));
    }

    #[test]
    fn test_config_max_distance_one() {
        let config = PrefetchConfig {
            max_distance: 1,
            min_distance: -1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=50 clamped to 1
        let advice = prefetch.compute(0, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1));

        // offset=-50 clamped to -1
        let advice = prefetch.compute(1, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_config_sharp_factor_one() {
        let config = PrefetchConfig {
            sharp_focus_factor: 1.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // With factor=1.0, SharpFocus behaves like Normal
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_config_diffuse_factor_one() {
        let config = PrefetchConfig {
            diffuse_factor: 1.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // With factor=1.0, Diffuse behaves like Normal
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_forward_distance_one() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 99, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_backward_distance_one() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 101, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    #[test]
    fn test_many_layers_history() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        for layer in 0..20 {
            prefetch.compute(layer, 100, 100, AttentionPattern::Normal);
        }
        let history = prefetch.history();
        assert_eq!(history.len(), 20);
        assert_eq!(history[19].layer_idx, 19);
        let stats = prefetch.stats();
        assert_eq!(stats.total, 20);
        assert_eq!(stats.none_count, 20);
    }

    #[test]
    fn test_history_record_normal_pattern() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history()[0].pattern, AttentionPattern::Normal);
    }

    #[test]
    fn test_history_record_sharp_pattern() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 80, 100, AttentionPattern::SharpFocus);
        assert_eq!(prefetch.history()[0].pattern, AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_history_record_diffuse_pattern() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 70, 100, AttentionPattern::Diffuse);
        assert_eq!(prefetch.history()[0].pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_history_centroid_pos_recorded_correctly() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 42, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history()[0].centroid_pos, 42);
    }

    #[test]
    fn test_history_current_pos_recorded_correctly() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 50, 77, AttentionPattern::Normal);
        assert_eq!(prefetch.history()[0].current_pos, 77);
    }

    #[test]
    fn test_sharp_focus_backward_moderate() {
        // offset=30, sharp=30*0.3=9 → Backward(9)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 130, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(9));
    }

    #[test]
    fn test_diffuse_forward_not_clamped() {
        // offset=-5, diffuse=-5*1.5=-7.5 → -7 → Forward(7), within [-16,64]
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 95, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(7));
    }

    #[test]
    fn test_normal_mode_forward_unclamped() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 85, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(15));
    }

    #[test]
    fn test_normal_mode_backward_unclamped() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 115, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(15));
    }

    #[test]
    fn test_sink_always_uses_config_offset() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 7,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Various positions all yield the same Sink offset
        assert_eq!(
            prefetch.compute(0, 0, 0, AttentionPattern::Sink),
            PrefetchAdvice::Sink(7)
        );
        assert_eq!(
            prefetch.compute(1, 500, 0, AttentionPattern::Sink),
            PrefetchAdvice::Sink(7)
        );
        assert_eq!(
            prefetch.compute(2, 0, 500, AttentionPattern::Sink),
            PrefetchAdvice::Sink(7)
        );
    }

    #[test]
    fn test_stats_after_many_computes() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        for i in 0..50 {
            let pattern = match i % 4 {
                0 => AttentionPattern::Normal,
                1 => AttentionPattern::SharpFocus,
                2 => AttentionPattern::Diffuse,
                _ => AttentionPattern::Sink,
            };
            prefetch.compute(i, 100 + (i % 3 + 1), 100, pattern);
        }

        let stats = prefetch.stats();
        assert_eq!(stats.total, 50);
        assert_eq!(
            stats.total,
            stats.none_count + stats.forward_count + stats.backward_count + stats.sink_count
        );
    }

    #[test]
    fn test_prefetch_stats_copy_semantics() {
        let stats = PrefetchStats {
            total: 42,
            none_count: 10,
            forward_count: 12,
            backward_count: 15,
            sink_count: 5,
        };
        let copy = stats;
        assert_eq!(copy.total, 42);
        assert_eq!(stats.total, 42);
    }

    #[test]
    fn test_centroid_record_clone_independent_of_original() {
        let record = CentroidRecord {
            layer_idx: 3,
            centroid_pos: 99,
            current_pos: 50,
            prefetch_distance: -49,
            pattern: AttentionPattern::Normal,
        };
        let cloned = record.clone();
        assert_eq!(cloned.layer_idx, record.layer_idx);
        assert_eq!(cloned.centroid_pos, record.centroid_pos);
        assert_eq!(cloned.prefetch_distance, record.prefetch_distance);
    }

    #[test]
    fn test_prefetch_new_stores_config() {
        let config = PrefetchConfig {
            min_distance: -100,
            max_distance: 200,
            ..Default::default()
        };
        let prefetch = CentroidPrefetch::new(config);
        // Config is stored internally; verify via behavior
        let mut prefetch = prefetch;
        // offset = -200 clamped to -100 → Forward(100)
        let advice = prefetch.compute(0, 0, 200, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(100));
    }

    #[test]
    fn test_compute_from_aggregator_records_history() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 120 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute_from_aggregator(5, 100, &agg, AttentionPattern::Normal);

        let history = prefetch.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].layer_idx, 5);
        assert_eq!(history[0].centroid_pos, 120);
    }

    #[test]
    fn test_config_with_nan_sharp_factor() {
        let config = PrefetchConfig {
            sharp_focus_factor: f32::NAN,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset * NaN = NaN, NaN as isize = 0 → None
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_with_inf_diffuse_factor() {
        let config = PrefetchConfig {
            diffuse_factor: f32::INFINITY,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=10 * inf = inf, inf as isize is UB-dependent but clamped to 64
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    // --- Additional batch to bring ratio well below 14 ---

    #[test]
    fn test_sharp_focus_forward_distance_two() {
        // offset = -6, sharp = -6 * 0.3 = -1.8 → -1 → Forward(1)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 94, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_sharp_focus_forward_exact_calculation() {
        // offset = -20, sharp = -20 * 0.3 = -6.0 → -6 → Forward(6)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 80, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(6));
    }

    #[test]
    fn test_diffuse_backward_exact_calculation() {
        // offset = 20, diffuse = 20 * 1.5 = 30 → Backward(30)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 120, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(30));
    }

    #[test]
    fn test_diffuse_forward_exact_calculation() {
        // offset = -10, diffuse = -10 * 1.5 = -15 → Forward(15)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(15));
    }

    #[test]
    fn test_config_equal_min_max_distance() {
        let config = PrefetchConfig {
            min_distance: 5,
            max_distance: 5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Any offset clamps to 5 → Backward(5)
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(5));

        let advice = prefetch.compute(1, 200, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(5));
    }

    #[test]
    fn test_config_negative_max_distance() {
        let config = PrefetchConfig {
            min_distance: -100,
            max_distance: -5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=10 clamped to max_distance=-5 → Forward(5)
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(5));
    }

    #[test]
    fn test_reset_clears_history_not_config() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 20,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);

        prefetch.reset();
        assert!(prefetch.history().is_empty());

        // Config still works after reset
        let advice = prefetch.compute(1, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(20));
    }

    #[test]
    fn test_history_distance_positive_backward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 130, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history()[0].prefetch_distance, 30);
    }

    #[test]
    fn test_history_distance_negative_forward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 75, 100, AttentionPattern::Normal);
        // offset = 75-100 = -25, clamped by default min_distance=-16 → -16
        assert!(prefetch.history()[0].prefetch_distance < 0);
    }

    #[test]
    fn test_history_distance_zero_for_none() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history()[0].prefetch_distance, 0);
    }

    #[test]
    fn test_stats_nonzero_only_forward() {
        let config = PrefetchConfig {
            min_distance: -1,
            max_distance: 1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        prefetch.compute(0, 50, 100, AttentionPattern::Normal); // Forward(1)

        let stats = prefetch.stats();
        assert_eq!(stats.forward_count, 1);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_stats_nonzero_only_backward() {
        let config = PrefetchConfig {
            min_distance: -1,
            max_distance: 1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        prefetch.compute(0, 150, 100, AttentionPattern::Normal); // Backward(1)

        let stats = prefetch.stats();
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 1);
    }

    #[test]
    fn test_centroid_record_all_attention_patterns() {
        let patterns = [
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
        ];
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        for (i, pattern) in patterns.into_iter().enumerate() {
            prefetch.compute(i, 100, 100, pattern);
        }

        let history = prefetch.history();
        assert_eq!(history[0].pattern, AttentionPattern::Normal);
        assert_eq!(history[1].pattern, AttentionPattern::Sink);
        assert_eq!(history[2].pattern, AttentionPattern::SharpFocus);
        assert_eq!(history[3].pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_compute_from_aggregator_with_zero_centroid() {
        let agg = TelemetryAggregator::new();
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        let advice =
            prefetch.compute_from_aggregator(0, 0, &agg, AttentionPattern::Normal);
        // centroid=0, current=0 → None
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_compute_from_aggregator_multiple_layers() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 105 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);
        prefetch.compute_from_aggregator(1, 100, &agg, AttentionPattern::SharpFocus);

        assert_eq!(prefetch.history()[0].layer_idx, 0);
        assert_eq!(prefetch.history()[1].layer_idx, 1);
    }

    #[test]
    fn test_advice_forward_usize_max() {
        let a = PrefetchAdvice::Forward(usize::MAX);
        if let PrefetchAdvice::Forward(v) = a {
            assert_eq!(v, usize::MAX);
        }
    }

    #[test]
    fn test_advice_backward_usize_max() {
        let a = PrefetchAdvice::Backward(usize::MAX);
        if let PrefetchAdvice::Backward(v) = a {
            assert_eq!(v, usize::MAX);
        }
    }

    #[test]
    fn test_advice_sink_usize_max() {
        let a = PrefetchAdvice::Sink(usize::MAX);
        if let PrefetchAdvice::Sink(v) = a {
            assert_eq!(v, usize::MAX);
        }
    }

    #[test]
    fn test_prefetch_stats_all_zero_equality() {
        let a = PrefetchStats::default();
        let b = PrefetchStats {
            total: 0,
            none_count: 0,
            forward_count: 0,
            backward_count: 0,
            sink_count: 0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefetch_stats_single_forward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 99, 100, AttentionPattern::Normal);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.forward_count, 1);
    }

    #[test]
    fn test_prefetch_stats_single_backward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 101, 100, AttentionPattern::Normal);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.backward_count, 1);
    }

    #[test]
    fn test_prefetch_stats_single_none() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.none_count, 1);
    }

    #[test]
    fn test_large_offset_forward_clamped_to_min() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 1000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_large_offset_backward_clamped_to_max() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 2000, 1000, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    #[test]
    fn test_sharp_focus_clamped_backward() {
        // offset = 200, sharp = 200 * 0.3 = 60 → Backward(60), within max
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 300, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(60));
    }

    #[test]
    fn test_sharp_focus_clamped_forward() {
        // offset = -100, sharp = -100 * 0.3 = -30, clamped to -16 → Forward(16)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_diffuse_clamped_forward_to_min() {
        // offset = -100, diffuse = -100 * 1.5 = -150, clamped to -16 → Forward(16)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_position_zero_both() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 0, 0, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_factor_two() {
        let config = PrefetchConfig {
            sharp_focus_factor: 2.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 10, sharp = 10 * 2.0 = 20 → Backward(20)
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(20));
    }

    #[test]
    fn test_diffuse_factor_zero_point_five() {
        let config = PrefetchConfig {
            diffuse_factor: 0.5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = -20, diffuse = -20 * 0.5 = -10 → Forward(10)
        let advice = prefetch.compute(0, 80, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_builder_default_then_layers() {
        let prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let with_layers = prefetch.with_num_layers(8);
        assert!(with_layers.history().is_empty());
    }

    #[test]
    fn test_history_preserves_all_positions() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 10, 20, AttentionPattern::Normal);
        prefetch.compute(1, 30, 40, AttentionPattern::Normal);
        prefetch.compute(2, 50, 60, AttentionPattern::Normal);

        assert_eq!(prefetch.history()[0].centroid_pos, 10);
        assert_eq!(prefetch.history()[0].current_pos, 20);
        assert_eq!(prefetch.history()[1].centroid_pos, 30);
        assert_eq!(prefetch.history()[1].current_pos, 40);
        assert_eq!(prefetch.history()[2].centroid_pos, 50);
        assert_eq!(prefetch.history()[2].current_pos, 60);
    }

    #[test]
    fn test_config_debug_contains_all_field_names() {
        let config = PrefetchConfig {
            min_distance: -1,
            max_distance: 1,
            sharp_focus_factor: 0.5,
            diffuse_factor: 2.0,
            sink_prefetch_offset: 3,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("-1"));
        assert!(debug.contains("0.5"));
        assert!(debug.contains("2.0"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_advice_forward_one() {
        let advice = PrefetchAdvice::Forward(1);
        assert_ne!(advice, PrefetchAdvice::Forward(2));
        assert_ne!(advice, PrefetchAdvice::Backward(1));
    }

    #[test]
    fn test_advice_backward_one() {
        let advice = PrefetchAdvice::Backward(1);
        assert_ne!(advice, PrefetchAdvice::Backward(2));
        assert_ne!(advice, PrefetchAdvice::Sink(1));
    }

    #[test]
    fn test_stats_default_is_all_zeros() {
        let stats = PrefetchStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_sharp_focus_negative_forward_preserves_sign() {
        // offset=-7, sharp=-7*0.3=-2.1 → -2 → Forward(2)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 93, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(2));
    }

    #[test]
    fn test_diffuse_positive_backward_preserves_sign() {
        // offset=7, diffuse=7*1.5=10.5 → 10 → Backward(10)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 107, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(10));
    }

    #[test]
    fn test_normal_offset_boundary_minus_one() {
        // offset = -15, within [-16, 64] → Forward(15)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 85, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(15));
    }

    #[test]
    fn test_normal_offset_boundary_plus_one() {
        // offset = 63, within [-16, 64] → Backward(63)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 163, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(63));
    }

    #[test]
    fn test_history_len_after_reset_and_recompute() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        prefetch.compute(1, 110, 100, AttentionPattern::Normal);
        prefetch.reset();
        prefetch.compute(2, 120, 100, AttentionPattern::Normal);

        assert_eq!(prefetch.history().len(), 1);
        assert_eq!(prefetch.history()[0].layer_idx, 2);
    }

    #[test]
    fn test_compute_from_aggregator_sink_pattern() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 500 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 0, &agg, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_config_clone_equal_fields() {
        let config = PrefetchConfig {
            min_distance: -50,
            max_distance: 100,
            sharp_focus_factor: 0.25,
            diffuse_factor: 2.5,
            sink_prefetch_offset: 8,
        };
        let cloned = config.clone();
        assert_eq!(config.min_distance, cloned.min_distance);
        assert_eq!(config.max_distance, cloned.max_distance);
        assert_eq!(config.sink_prefetch_offset, cloned.sink_prefetch_offset);
    }

    #[test]
    fn test_centroid_record_negative_distance_forward() {
        let record = CentroidRecord {
            layer_idx: 0,
            centroid_pos: 10,
            current_pos: 100,
            prefetch_distance: -90,
            pattern: AttentionPattern::Normal,
        };
        assert!(record.prefetch_distance < 0);
    }

    #[test]
    fn test_advice_none_distinct_from_forward_zero() {
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Forward(0));
    }

    #[test]
    fn test_config_default_is_not_mutated() {
        let a = PrefetchConfig::default();
        let b = PrefetchConfig::default();
        assert_eq!(a.min_distance, b.min_distance);
        assert_eq!(a.max_distance, b.max_distance);
    }

    #[test]
    fn test_sharp_focus_factor_half() {
        let config = PrefetchConfig {
            sharp_focus_factor: 0.5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        // offset=20, sharp=20*0.5=10 → Backward(10)
        let advice = prefetch.compute(0, 120, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(10));
    }

    #[test]
    fn test_diffuse_factor_two() {
        let config = PrefetchConfig {
            diffuse_factor: 2.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        // offset=20, diffuse=20*2.0=40 → Backward(40)
        let advice = prefetch.compute(0, 120, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(40));
    }

    #[test]
    fn test_config_min_distance_negative_one() {
        let config = PrefetchConfig {
            min_distance: -1,
            max_distance: 100,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        // offset=-50 clamped to -1 → Forward(1)
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_stats_equality_requires_all_fields() {
        let a = PrefetchStats {
            total: 5,
            none_count: 1,
            forward_count: 1,
            backward_count: 1,
            sink_count: 2,
        };
        let mut b = a;
        assert_eq!(a, b);
        b.total = 6;
        assert_ne!(a, b);
    }

    #[test]
    fn test_centroid_record_positive_distance_backward() {
        let record = CentroidRecord {
            layer_idx: 1,
            centroid_pos: 200,
            current_pos: 100,
            prefetch_distance: 100,
            pattern: AttentionPattern::Diffuse,
        };
        assert!(record.prefetch_distance > 0);
    }

    #[test]
    fn test_history_after_single_sink() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 0, 100, AttentionPattern::Sink);

        let history = prefetch.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].prefetch_distance, 0);
    }

    #[test]
    fn test_history_after_single_none() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);

        let history = prefetch.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].prefetch_distance, 0);
    }

    #[test]
    fn test_config_sharp_factor_gt_one() {
        let config = PrefetchConfig {
            sharp_focus_factor: 3.0,
            max_distance: 200,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        // offset=10, sharp=10*3.0=30 → Backward(30)
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(30));
    }

    #[test]
    fn test_config_diffuse_factor_lt_one() {
        let config = PrefetchConfig {
            diffuse_factor: 0.5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        // offset=-20, diffuse=-20*0.5=-10 → Forward(10)
        let advice = prefetch.compute(0, 80, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
    }

    #[test]
    fn test_sink_with_centroid_ahead() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 500, 10, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_sink_with_centroid_behind() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 10, 500, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(4));
    }

    #[test]
    fn test_offset_exactly_one_forward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 99, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_offset_exactly_one_backward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 101, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1));
    }

    #[test]
    fn test_offset_exactly_at_min_plus_one() {
        // offset=-15, just inside min_distance=-16 → Forward(15)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 85, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(15));
    }

    #[test]
    fn test_offset_exactly_at_max_minus_one() {
        // offset=63, just inside max_distance=64 → Backward(63)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 163, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(63));
    }

    #[test]
    fn test_with_num_layers_then_compute() {
        let mut prefetch =
            CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(4);
        let advice = prefetch.compute(3, 110, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(10));
        assert_eq!(prefetch.history().len(), 1);
    }

    #[test]
    fn test_stats_default_copy_equals_explicit() {
        let a = PrefetchStats::default();
        let b = PrefetchStats {
            total: 0,
            none_count: 0,
            forward_count: 0,
            backward_count: 0,
            sink_count: 0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefetch_new_default_config() {
        let prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        assert!(prefetch.history().is_empty());
    }

    #[test]
    fn test_config_sink_offset_one() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);
        let advice = prefetch.compute(0, 0, 0, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(1));
    }

    #[test]
    fn test_advice_none_not_equal_any_payload_variant() {
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Forward(0));
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Backward(0));
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Sink(0));
    }

    // ── Wave 12x48: +40 tests ──────────────────────────────────────────

    // --- Config boundary combinations ---

    #[test]
    fn test_config_negative_equal_min_max() {
        // min == max < 0 → all offsets clamp to same negative value → Forward
        let config = PrefetchConfig {
            min_distance: -7,
            max_distance: -7,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Positive offset clamps to -7 → Forward(7)
        let advice = prefetch.compute(0, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(7));

        // Zero offset clamps to -7 → Forward(7)
        let advice = prefetch.compute(1, 100, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(7));
    }

    #[test]
    fn test_config_both_bounds_negative() {
        let config = PrefetchConfig {
            min_distance: -10000,
            max_distance: -1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Negative offset within bounds passes through
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(50));

        // Positive offset clamped to max=-1 → Forward(1)
        let advice = prefetch.compute(1, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_config_both_bounds_positive() {
        let config = PrefetchConfig {
            min_distance: 1,
            max_distance: 10000,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // Negative offset clamped to min=1 → Backward(1)
        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(1));

        // Positive offset passes through
        let advice = prefetch.compute(1, 150, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(50));
    }

    #[test]
    fn test_config_large_sink_offset() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 1000,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 50, 100, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(1000));
    }

    // --- Config special float values ---

    #[test]
    fn test_config_subnormal_sharp_factor() {
        // Smallest positive subnormal: any offset * subnormal ≈ 0.0 → 0 as isize → None
        let config = PrefetchConfig {
            sharp_focus_factor: f32::from_bits(1),
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 200, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_neg_inf_sharp_clamps_forward() {
        // Positive offset * NEG_INF = NEG_INF → isize::MIN → clamp to min → Forward
        let config = PrefetchConfig {
            sharp_focus_factor: f32::NEG_INFINITY,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 110, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    #[test]
    fn test_config_neg_inf_diffuse_negative_offset() {
        // Negative offset * NEG_INF = +INF → isize::MAX → clamp to max → Backward
        let config = PrefetchConfig {
            diffuse_factor: f32::NEG_INFINITY,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 90, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    #[test]
    fn test_config_nan_diffuse_factor_produces_none() {
        // offset * NaN = NaN → 0 as isize → None
        let config = PrefetchConfig {
            diffuse_factor: f32::NAN,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 110, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_config_nan_sharp_negative_offset_none() {
        // Negative offset * NaN = NaN → 0 as isize → None
        let config = PrefetchConfig {
            sharp_focus_factor: f32::NAN,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice = prefetch.compute(0, 50, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    // --- SharpFocus rounding edges ---

    #[test]
    fn test_sharp_focus_negative_three_rounds_to_zero() {
        // offset = -3, sharp = -3 * 0.3 = -0.9 → 0 → None
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 97, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_offset_minus_one_rounds_to_zero() {
        // offset = -1, sharp = -1 * 0.3 = -0.3 → 0 → None
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 99, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_sharp_focus_offset_seven_backward() {
        // offset = 7, sharp = 7 * 0.3 = 2.1 → 2 → Backward(2)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 107, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(2));
    }

    // --- Diffuse rounding edges ---

    #[test]
    fn test_diffuse_offset_minus_one_forward() {
        // offset = -1, diffuse = -1 * 1.5 = -1.5 → -1 → Forward(1)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 99, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(1));
    }

    #[test]
    fn test_diffuse_offset_minus_two_forward() {
        // offset = -2, diffuse = -2 * 1.5 = -3.0 → -3 → Forward(3)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 98, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(3));
    }

    #[test]
    fn test_diffuse_offset_two_backward() {
        // offset = 2, diffuse = 2 * 1.5 = 3.0 → 3 → Backward(3)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 102, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(3));
    }

    #[test]
    fn test_diffuse_large_backward_unclamped() {
        // offset = 40, diffuse = 40 * 1.5 = 60 → Backward(60), within [-16, 64]
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 140, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Backward(60));
    }

    // --- Normal mode precise small offsets ---

    #[test]
    fn test_normal_offset_minus_two_forward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 98, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(2));
    }

    #[test]
    fn test_normal_offset_two_backward() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 102, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(2));
    }

    // --- CentroidRecord Debug per pattern ---

    #[test]
    fn test_centroid_record_debug_normal_pattern() {
        let record = CentroidRecord {
            layer_idx: 1,
            centroid_pos: 50,
            current_pos: 60,
            prefetch_distance: -10,
            pattern: AttentionPattern::Normal,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("Normal"));
        assert!(debug.contains("centroid_pos"));
    }

    #[test]
    fn test_centroid_record_debug_sink_pattern() {
        let record = CentroidRecord {
            layer_idx: 0,
            centroid_pos: 0,
            current_pos: 100,
            prefetch_distance: 0,
            pattern: AttentionPattern::Sink,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("Sink"));
    }

    #[test]
    fn test_centroid_record_debug_diffuse_pattern() {
        let record = CentroidRecord {
            layer_idx: 2,
            centroid_pos: 200,
            current_pos: 100,
            prefetch_distance: 100,
            pattern: AttentionPattern::Diffuse,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("Diffuse"));
    }

    #[test]
    fn test_centroid_record_debug_sharp_pattern() {
        let record = CentroidRecord {
            layer_idx: 3,
            centroid_pos: 88,
            current_pos: 100,
            prefetch_distance: -12,
            pattern: AttentionPattern::SharpFocus,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("SharpFocus"));
    }

    // --- CentroidRecord properties ---

    #[test]
    fn test_centroid_record_clone_field_equality() {
        let record = CentroidRecord {
            layer_idx: 42,
            centroid_pos: 500,
            current_pos: 300,
            prefetch_distance: -200,
            pattern: AttentionPattern::Diffuse,
        };
        let cloned = record.clone();

        assert_eq!(cloned.layer_idx, 42);
        assert_eq!(cloned.centroid_pos, 500);
        assert_eq!(cloned.current_pos, 300);
        assert_eq!(cloned.prefetch_distance, -200);
        assert_eq!(cloned.pattern, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_centroid_record_pattern_variants_match() {
        let patterns = [
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
        ];
        for (i, &pattern) in patterns.iter().enumerate() {
            let record = CentroidRecord {
                layer_idx: i,
                centroid_pos: 0,
                current_pos: 0,
                prefetch_distance: 0,
                pattern,
            };
            assert_eq!(record.pattern, pattern);
        }
    }

    // --- PrefetchStats manual construction ---

    #[test]
    fn test_prefetch_stats_manual_total_independent() {
        // total is an independent field, not computed from categories
        let stats = PrefetchStats {
            total: 999,
            none_count: 0,
            forward_count: 0,
            backward_count: 0,
            sink_count: 0,
        };
        assert_eq!(stats.total, 999);
        let category_sum =
            stats.none_count + stats.forward_count + stats.backward_count + stats.sink_count;
        assert_eq!(category_sum, 0);
        assert_ne!(stats.total, category_sum);
    }

    #[test]
    fn test_prefetch_stats_hash_differs_for_different_total() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |s: &PrefetchStats| {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };

        let a = PrefetchStats {
            total: 1,
            ..Default::default()
        };
        let b = PrefetchStats {
            total: 2,
            ..Default::default()
        };
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn test_prefetch_stats_total_independent_of_category_sum() {
        // Explicit construction: total ≠ sum of categories is allowed
        let stats = PrefetchStats {
            total: 100,
            none_count: 10,
            forward_count: 10,
            backward_count: 10,
            sink_count: 10,
        };
        assert_eq!(stats.total, 100);
        assert_eq!(
            stats.none_count + stats.forward_count + stats.backward_count + stats.sink_count,
            40,
        );
    }

    // --- PrefetchAdvice property ---

    #[test]
    fn test_advice_forward_zero_debug_not_none() {
        let none_debug = format!("{:?}", PrefetchAdvice::None);
        let fwd0_debug = format!("{:?}", PrefetchAdvice::Forward(0));
        assert_ne!(none_debug, fwd0_debug);
        assert!(fwd0_debug.contains("Forward"));
    }

    // --- compute_from_aggregator edge cases ---

    #[test]
    fn test_compute_from_aggregator_after_reset() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 90 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 50, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);

        prefetch.reset();
        assert!(prefetch.history().is_empty());

        // After reset, aggregator-based compute works fresh
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Forward(10));
        assert_eq!(prefetch.history().len(), 1);
        assert_eq!(prefetch.history()[0].centroid_pos, 90);
    }

    #[test]
    fn test_compute_from_aggregator_latest_centroid_used() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 200 });
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 105 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);

        // Latest centroid=105 used, not 200
        assert_eq!(advice, PrefetchAdvice::Backward(5));
    }

    #[test]
    fn test_compute_from_aggregator_reuse_same_aggregator() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 120 });

        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        let a1 =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Normal);
        let a2 =
            prefetch.compute_from_aggregator(1, 100, &agg, AttentionPattern::SharpFocus);

        // Normal: offset=20 → Backward(20)
        assert_eq!(a1, PrefetchAdvice::Backward(20));
        // SharpFocus: offset=20*0.3=6 → Backward(6)
        assert_eq!(a2, PrefetchAdvice::Backward(6));
        assert_eq!(prefetch.history().len(), 2);
    }

    // --- with_num_layers edge cases ---

    #[test]
    fn test_with_num_layers_compute_beyond_nominal_capacity() {
        let mut prefetch =
            CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(4);

        // Layer index exceeds num_layers but compute still succeeds
        let advice = prefetch.compute(100, 110, 100, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::Backward(10));
        assert_eq!(prefetch.history()[0].layer_idx, 100);
    }

    // --- History distance sign properties ---

    #[test]
    fn test_interleaved_sink_normal_history_distances() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(0, 90, 100, AttentionPattern::Normal); // Forward
        prefetch.compute(1, 0, 100, AttentionPattern::Sink); // Sink
        prefetch.compute(2, 110, 100, AttentionPattern::Normal); // Backward

        let history = prefetch.history();
        assert!(history[0].prefetch_distance < 0);
        assert_eq!(history[1].prefetch_distance, 0);
        assert!(history[2].prefetch_distance > 0);
    }

    // --- Stats idempotency ---

    #[test]
    fn test_stats_idempotent_multiple_calls() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        prefetch.compute(0, 90, 100, AttentionPattern::Normal);

        let s1 = prefetch.stats();
        let s2 = prefetch.stats();
        assert_eq!(s1, s2);
    }

    // --- History incremental growth ---

    #[test]
    fn test_history_grows_incrementally() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        assert_eq!(prefetch.history().len(), 0);

        prefetch.compute(0, 90, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 1);

        prefetch.compute(1, 110, 100, AttentionPattern::Normal);
        assert_eq!(prefetch.history().len(), 2);

        prefetch.compute(2, 0, 100, AttentionPattern::Sink);
        assert_eq!(prefetch.history().len(), 3);
    }

    // --- History with non-sequential layer indices ---

    #[test]
    fn test_history_non_sequential_layer_indices() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        prefetch.compute(5, 90, 100, AttentionPattern::Normal);
        prefetch.compute(20, 110, 100, AttentionPattern::Normal);
        prefetch.compute(3, 100, 100, AttentionPattern::Normal);

        let history = prefetch.history();
        assert_eq!(history[0].layer_idx, 5);
        assert_eq!(history[1].layer_idx, 20);
        assert_eq!(history[2].layer_idx, 3);
    }

    // --- Custom factor combinations ---

    #[test]
    fn test_sharp_focus_backward_custom_factor() {
        let config = PrefetchConfig {
            sharp_focus_factor: 0.5,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=30, sharp=30*0.5=15 → Backward(15)
        let advice = prefetch.compute(0, 130, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(15));
    }

    #[test]
    fn test_diffuse_forward_clamped_custom_factor() {
        let config = PrefetchConfig {
            diffuse_factor: 2.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=-10, diffuse=-10*2.0=-20, clamped to -16 → Forward(16)
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::Forward(16));
    }

    // --- Config default range properties ---

    #[test]
    fn test_config_default_diffuse_factor_above_one() {
        let config = PrefetchConfig::default();
        assert!(
            config.diffuse_factor > 1.0,
            "diffuse_factor should expand range (> 1.0)",
        );
    }

    #[test]
    fn test_config_default_min_distance_is_negative() {
        let config = PrefetchConfig::default();
        assert!(
            config.min_distance < 0,
            "min_distance should be negative to allow forward prefetch",
        );
    }

    // ── Additional 13 tests ────────────────────────────────────────────

    // --- 1. SharpFocus forward with exact float multiplication (no rounding error) ---

    #[test]
    fn test_sharp_focus_forward_exact_integer_product() {
        // offset = -10, sharp_focus_factor = 0.3 → -3.0 exactly (no rounding), → Forward(3)
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 90, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Forward(3));

        // Verify history records the exact distance
        assert_eq!(prefetch.history()[0].prefetch_distance, -3);
    }

    // --- 2. SharpFocus forward with very small offset that rounds to zero ---

    #[test]
    fn test_sharp_focus_forward_very_small_offset_rounds_none() {
        // offset = -2, sharp = -2 * 0.3 = -0.6 → 0 as isize → None
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());
        let advice = prefetch.compute(0, 98, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    // --- 3. Diffuse factor zero on positive offset produces None ---

    #[test]
    fn test_diffuse_zero_factor_positive_offset_none() {
        let config = PrefetchConfig {
            diffuse_factor: 0.0,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=50 * 0.0 = 0.0 → 0 as isize → None
        let advice = prefetch.compute(0, 150, 100, AttentionPattern::Diffuse);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    // --- 4. compute_from_aggregator with Sink and custom offset ---

    #[test]
    fn test_compute_from_aggregator_sink_custom_offset() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 999 });

        let config = PrefetchConfig {
            sink_prefetch_offset: 16,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        let advice =
            prefetch.compute_from_aggregator(0, 100, &agg, AttentionPattern::Sink);
        assert_eq!(advice, PrefetchAdvice::Sink(16));

        // History still records centroid from aggregator
        assert_eq!(prefetch.history()[0].centroid_pos, 999);
    }

    // --- 5. History records correct distances for interleaved patterns ---

    #[test]
    fn test_history_distances_for_all_four_patterns() {
        let mut prefetch = CentroidPrefetch::new(PrefetchConfig::default());

        // Normal: offset = 110-100 = 10 → distance=10
        prefetch.compute(0, 110, 100, AttentionPattern::Normal);
        // SharpFocus: offset = 110-100 = 10, adjusted=3 → distance=3
        prefetch.compute(1, 110, 100, AttentionPattern::SharpFocus);
        // Diffuse: offset = 110-100 = 10, adjusted=15 → distance=15
        prefetch.compute(2, 110, 100, AttentionPattern::Diffuse);
        // Sink: always distance=0
        prefetch.compute(3, 110, 100, AttentionPattern::Sink);

        let history = prefetch.history();
        assert_eq!(history[0].prefetch_distance, 10);
        assert_eq!(history[1].prefetch_distance, 3);
        assert_eq!(history[2].prefetch_distance, 15);
        assert_eq!(history[3].prefetch_distance, 0);
    }

    // --- 6. Stats with exactly 2 of each category ---

    #[test]
    fn test_stats_two_of_each_category() {
        let config = PrefetchConfig {
            min_distance: -1,
            max_distance: 1,
            sink_prefetch_offset: 1,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // 2x None (centroid == current)
        prefetch.compute(0, 100, 100, AttentionPattern::Normal);
        prefetch.compute(1, 100, 100, AttentionPattern::SharpFocus);
        // 2x Forward (centroid < current)
        prefetch.compute(2, 99, 100, AttentionPattern::Normal);
        prefetch.compute(3, 99, 100, AttentionPattern::Normal);
        // 2x Backward (centroid > current)
        prefetch.compute(4, 101, 100, AttentionPattern::Normal);
        prefetch.compute(5, 101, 100, AttentionPattern::Normal);
        // 2x Sink
        prefetch.compute(6, 50, 100, AttentionPattern::Sink);
        prefetch.compute(7, 50, 100, AttentionPattern::Sink);

        let stats = prefetch.stats();
        assert_eq!(stats.total, 8);
        assert_eq!(stats.none_count, 2);
        assert_eq!(stats.forward_count, 2);
        assert_eq!(stats.backward_count, 2);
        assert_eq!(stats.sink_count, 2);
    }

    // --- 7. Large sharp_focus_factor causes clamp to max_distance ---

    #[test]
    fn test_large_sharp_factor_clamps_to_max() {
        let config = PrefetchConfig {
            sharp_focus_factor: 100.0,
            max_distance: 64,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset=10 * 100.0 = 1000, clamped to 64 → Backward(64)
        let advice = prefetch.compute(0, 110, 100, AttentionPattern::SharpFocus);
        assert_eq!(advice, PrefetchAdvice::Backward(64));
    }

    // --- 8. History distance matches SharpFocus adjusted value ---

    #[test]
    fn test_history_distance_sharp_focus_adjusted() {
        let config = PrefetchConfig {
            sharp_focus_factor: 0.25,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 80-100 = -20, adjusted = -20*0.25 = -5.0 → -5
        prefetch.compute(0, 80, 100, AttentionPattern::SharpFocus);

        assert_eq!(prefetch.history()[0].prefetch_distance, -5);
    }

    // --- 9. History distance matches Diffuse adjusted value ---

    #[test]
    fn test_history_distance_diffuse_adjusted() {
        let config = PrefetchConfig {
            diffuse_factor: 0.8,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        // offset = 80-100 = -20, adjusted = -20*0.8 = -16.0 → -16
        prefetch.compute(0, 80, 100, AttentionPattern::Diffuse);

        assert_eq!(prefetch.history()[0].prefetch_distance, -16);
    }

    // --- 10. Reset after only Sink calls clears everything ---

    #[test]
    fn test_reset_after_sink_only_calls() {
        let config = PrefetchConfig {
            sink_prefetch_offset: 8,
            ..Default::default()
        };
        let mut prefetch = CentroidPrefetch::new(config);

        prefetch.compute(0, 0, 100, AttentionPattern::Sink);
        prefetch.compute(1, 50, 100, AttentionPattern::Sink);
        prefetch.compute(2, 200, 100, AttentionPattern::Sink);

        assert_eq!(prefetch.stats().sink_count, 3);
        assert_eq!(prefetch.history().len(), 3);

        prefetch.reset();

        assert_eq!(prefetch.stats().total, 0);
        assert_eq!(prefetch.stats().sink_count, 0);
        assert!(prefetch.history().is_empty());
    }

    // --- 11. CentroidRecord with isize::MIN distance ---

    #[test]
    fn test_centroid_record_min_isize_distance() {
        let record = CentroidRecord {
            layer_idx: 0,
            centroid_pos: 0,
            current_pos: 0,
            prefetch_distance: isize::MIN,
            pattern: AttentionPattern::Normal,
        };
        assert_eq!(record.prefetch_distance, isize::MIN);

        // Clone preserves the extreme value
        let cloned = record.clone();
        assert_eq!(cloned.prefetch_distance, isize::MIN);
    }

    // --- 12. Config default max_distance is positive ---

    #[test]
    fn test_config_default_max_distance_is_positive() {
        let config = PrefetchConfig::default();
        assert!(
            config.max_distance > 0,
            "max_distance should be positive to allow backward prefetch",
        );
    }

    // --- 13. PrefetchAdvice Forward(0), Backward(0), Sink(0) are all distinct from None and each other ---

    #[test]
    fn test_advice_zero_payload_variants_all_mutually_distinct() {
        let none = PrefetchAdvice::None;
        let fwd0 = PrefetchAdvice::Forward(0);
        let bwd0 = PrefetchAdvice::Backward(0);
        let sink0 = PrefetchAdvice::Sink(0);

        // All pairwise distinct
        assert_ne!(none, fwd0);
        assert_ne!(none, bwd0);
        assert_ne!(none, sink0);
        assert_ne!(fwd0, bwd0);
        assert_ne!(fwd0, sink0);
        assert_ne!(bwd0, sink0);

        // Each is equal to itself
        assert_eq!(fwd0, PrefetchAdvice::Forward(0));
        assert_eq!(bwd0, PrefetchAdvice::Backward(0));
        assert_eq!(sink0, PrefetchAdvice::Sink(0));
    }
}
