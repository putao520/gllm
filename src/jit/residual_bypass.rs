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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Copy, Default)]
pub struct BypassStats {
    /// 总决策次数
    pub total: usize,
    /// 跳过的层数
    pub bypassed: usize,
    /// 执行的层数
    pub executed: usize,
}

/// 批次级 Bypass 汇总
#[derive(Debug, Clone, Default)]
pub struct BypassSummary {
    /// 批次中的所有 Bypass 建议
    pub advices: Vec<BypassAdvice>,
}

/// 单个请求的 Bypass 建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BypassAdvice {
    /// 执行该层
    Execute,
    /// 跳过该层
    Bypass,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
