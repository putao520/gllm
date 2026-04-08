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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, Default)]
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
    use crate::kv_cache::KvPageHeader;

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
                assert!(d <= 20);  // 允许一些误差
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
}
