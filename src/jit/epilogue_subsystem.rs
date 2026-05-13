//! Epilogue 优化子系统 — 统一接入执行管线 (SPEC §9-§18)
//!
//! ## 核心职责
//! 将所有孤立的 JIT 优化模块打包为一个子系统，接入 Executor::step() 流程:
//! - 从 `SequenceTelemetry` 提取信号 → 喂入 `TelemetryAggregator`
//! - 驱动 GateFirstSkip / SinkTracker / ResidualBypass / ExpertThermal / SpecSchedule 决策
//! - 生成批次级汇总 `EpilogueBatchSummary`
//! - 写入 `KvPageHeader` 遥测字段（供下一步 build_batch 使用）
//!
//! ## 数据流
//! ```text
//! Executor::step()
//!   │
//!   ├─ run_batch_forward() → Vec<SequenceTelemetry>
//!   │
//!   └─ EpilogueSubsystem::ingest_and_decide(batch_telemetry)
//!        │
//!        ├─ SequenceTelemetry → EpilogueSignal → TelemetryAggregator
//!        │
//!        ├─ GateFirstSkipLayer::decide_from_aggregator()  → GateSkipDecision
//!        ├─ SinkTracker::detect_from_aggregator()          → AttentionPattern
//!        ├─ ResidualBypassLayer::decide_from_aggregator()  → BypassDecision
//!        ├─ ExpertThermalTracker::update_from_telemetry()  → ThermalState[]
//!        ├─ SpecScheduleSignal::advise_from_telemetry()    → SpecAdvice
//!        │
//!        └─ EpilogueBatchSummary → 写入 KvPageHeader → 供下一步调度使用
//! ```
//!
//! ## 接入模式
//! 遵循 `MoeSubsystem` 模式，但始终初始化（非 Option）:
//! - `from_loader()` 中构造默认配置
//! - `configure_epilogue()` 可覆盖配置
//! - 内部模块通过 `enabled` 字段独立禁用

use super::epilogue::{
    EpilogueSignal, TelemetryAggregator,
    GateFirstSkipConfig, GateSkipDecision,
    ResidualBypassConfig, ResidualBypassDecision,
    SinkDetectionConfig, AttentionPattern,
    ExpertThermalTracker,
    SpecScheduleAdvice, SpecScheduleSignal,
};
use super::gate_skip::{BatchSkipAdvice, BatchSkipSummary, GateFirstSkipLayer};
use super::residual_bypass::{BypassStats, ResidualBypassLayer};
use super::sink_tracker::{BatchAttentionSummary, SinkTracker};
use super::prefetch::{CentroidPrefetch, PrefetchConfig, PrefetchAdvice};
use crate::kv_cache::{KvPageHeader, f32_to_f16_bits, f32_to_dead_ratio};
use crate::scheduler::telemetry::SequenceTelemetry;

// ============================================================================
// 配置
// ============================================================================

/// Epilogue 子系统配置
#[derive(Debug, Clone)]
pub struct EpilogueConfig {
    /// Gate-First Skip 配置 (§13.1)
    pub gate_skip: GateFirstSkipConfig,
    /// Sink Detection 配置 (§13.9)
    pub sink_detection: SinkDetectionConfig,
    /// Residual Bypass 配置 (§13.3)
    pub residual_bypass: ResidualBypassConfig,
    /// Centroid Prefetch 配置 (§13.2)
    pub prefetch: PrefetchConfig,
    /// 模型层数
    pub num_layers: usize,
    /// MoE 专家数 (0 = dense model)
    pub num_experts: usize,
}

impl Default for EpilogueConfig {
    fn default() -> Self {
        Self {
            gate_skip: GateFirstSkipConfig::default(),
            sink_detection: SinkDetectionConfig::default(),
            residual_bypass: ResidualBypassConfig::default(),
            prefetch: PrefetchConfig::default(),
            num_layers: 32,
            num_experts: 0,
        }
    }
}

// ============================================================================
// 请求级决策结果
// ============================================================================

/// 单个请求的 Epilogue 决策结果
#[derive(Debug, Clone, Copy)]
pub struct RequestEpilogueDecision {
    /// Gate-First Skip 决策
    pub gate_skip: GateSkipDecision,
    /// 注意力模式分类
    pub attention_pattern: AttentionPattern,
    /// Residual Bypass 决策
    pub bypass_decision: ResidualBypassDecision,
    /// 推测解码建议
    pub spec_advice: SpecScheduleAdvice,
    /// Centroid Prefetch 建议 (§13.2)
    pub prefetch_advice: PrefetchAdvice,
}

// ============================================================================
// 批次级汇总
// ============================================================================

/// 批次级 Epilogue 汇总 — 供 Executor 和下一步 build_batch 使用
#[derive(Debug, Clone)]
pub struct EpilogueBatchSummary {
    /// Gate-First Skip 批次汇总
    pub skip_summary: BatchSkipSummary,
    /// Sink Detection 批次汇总
    pub attention_summary: BatchAttentionSummary,
    /// Residual Bypass 统计
    pub bypass_stats: BypassStats,
    /// 是否需要 Compact (Compact→Execute→Scatter)
    pub compact_required: bool,
    /// 批次内请求的浪费比例 (0.0-1.0)
    pub waste_ratio: f32,
    /// 逐请求决策
    pub per_request: Vec<RequestEpilogueDecision>,
}

impl Default for EpilogueBatchSummary {
    fn default() -> Self {
        Self {
            skip_summary: BatchSkipSummary::from_decisions(&[]),
            attention_summary: BatchAttentionSummary::from_patterns(&[], 0.0),
            bypass_stats: BypassStats::default(),
            compact_required: false,
            waste_ratio: 0.0,
            per_request: Vec::new(),
        }
    }
}

// ============================================================================
// EpilogueSubsystem — 核心子系统
// ============================================================================

/// Epilogue 优化子系统
///
/// 打包所有 SPEC §9-§18 遥测驱动的决策模块，提供统一的
/// `ingest_and_decide()` 入口供 Executor::step() 调用。
pub struct EpilogueSubsystem {
    /// 遥测聚合器 — 收集 EpilogueSignal 并提供信号读取
    aggregator: TelemetryAggregator,
    /// Gate-First Skip 层级集成器 (§13.1)
    gate_skip: GateFirstSkipLayer,
    /// Sink Tracker 层级集成器 (§13.9)
    sink_tracker: SinkTracker,
    /// Residual Bypass 层级集成器 (§13.3)
    residual_bypass: ResidualBypassLayer,
    /// Centroid Prefetch 质心预取器 (§13.2)
    prefetch: CentroidPrefetch,
    /// MoE Expert Thermal Tracker (§13.6, §15.4) — None for dense
    expert_thermal: Option<ExpertThermalTracker>,
    /// 推测解码调度信号 (§17.9)
    spec_signal: SpecScheduleSignal,
    /// 模型层数
    num_layers: usize,
}

impl EpilogueSubsystem {
    /// 创建新的 Epilogue 子系统
    pub fn new(config: EpilogueConfig) -> Self {
        let expert_thermal = if config.num_experts > 0 {
            Some(ExpertThermalTracker::new(config.num_experts))
        } else {
            None
        };

        let prefetch = CentroidPrefetch::new(config.prefetch)
            .with_num_layers(config.num_layers);

        Self {
            aggregator: TelemetryAggregator::new(),
            gate_skip: GateFirstSkipLayer::new(config.gate_skip, config.num_layers),
            sink_tracker: SinkTracker::new(config.sink_detection, config.num_layers),
            residual_bypass: ResidualBypassLayer::new(config.residual_bypass, config.num_layers),
            prefetch,
            expert_thermal,
            spec_signal: SpecScheduleSignal::new(),
            num_layers: config.num_layers,
        }
    }

    /// 从 `SequenceTelemetry` 批次驱动全链路决策
    ///
    /// 这是 Executor::step() 在 `run_batch_forward()` 之后调用的核心入口。
    /// 对每个 SequenceTelemetry:
    ///   1. 转换为 EpilogueSignal → 喂入 TelemetryAggregator
    ///   2. 驱动各决策器
    ///   3. 收集 RequestEpilogueDecision
    ///   4. 生成批次级汇总
    pub fn ingest_and_decide(
        &mut self,
        batch_telemetry: &[SequenceTelemetry],
    ) -> EpilogueBatchSummary {
        let mut per_request = Vec::with_capacity(batch_telemetry.len());
        let mut gate_decisions = Vec::with_capacity(batch_telemetry.len());
        let mut attention_patterns = Vec::with_capacity(batch_telemetry.len());
        let mut total_waste = 0.0f32;

        // 逐请求处理
        for (idx, telemetry) in batch_telemetry.iter().enumerate() {
            // Step 1: SequenceTelemetry → EpilogueSignal → TelemetryAggregator
            self.ingest_sequence_telemetry(telemetry);

            // Step 2: 驱动各决策器（使用中间层作为参考层）
            let reference_layer = self.num_layers / 2;

            let gate_skip = self.gate_skip.decide_from_aggregator(reference_layer, &self.aggregator);
            let attention_pattern = self.sink_tracker.detect_from_aggregator(reference_layer, &self.aggregator);
            let bypass_decision = self.residual_bypass.decide_from_aggregator(reference_layer, &self.aggregator);
            let spec_advice = self.spec_signal.advise_from_telemetry(&self.aggregator, 1.0);

            // Step 2b: §13.2 Centroid Prefetch — 从注意力质心计算预取建议
            // 使用 softmax_sharpness 近似质心位置: 高 sharpness → 质心靠近序列起始
            let centroid_pos = if telemetry.per_head_entropy > 0.0 {
                // 低熵 → 高 sharpness → 质心靠前 (Sink/Sharp)
                ((1.0 / (1.0 + telemetry.per_head_entropy)) * reference_layer as f32) as usize
            } else {
                0
            };
            let prefetch_advice = self.prefetch.compute(
                reference_layer,
                centroid_pos,
                reference_layer, // current position ≈ mid-layer
                attention_pattern,
            );

            // Step 3: MoE 专家热度更新
            if let Some(ref mut thermal) = self.expert_thermal {
                thermal.update_from_telemetry(&self.aggregator);
            }

            // Step 4: 计算浪费比 (skip 层贡献的浪费)
            let waste = match gate_skip {
                GateSkipDecision::Skip => 1.0,
                GateSkipDecision::MaskedCompute => 0.3,
                GateSkipDecision::FullCompute => 0.0,
            };
            total_waste += waste;

            gate_decisions.push(gate_skip);
            attention_patterns.push(attention_pattern);

            per_request.push(RequestEpilogueDecision {
                gate_skip,
                attention_pattern,
                bypass_decision,
                spec_advice,
                prefetch_advice,
            });

            // 重置聚合器为下一个请求准备
            self.aggregator = TelemetryAggregator::new();

            // 避免未使用变量警告
            let _ = idx;
        }

        // 生成批次级汇总
        let skip_summary = BatchSkipSummary::from_decisions(&gate_decisions);
        let bypass_stats = self.residual_bypass.stats();
        let avg_sharpness = 0.0; // 近似值 — 完整实现需 GPU Epilogue 直出
        let attention_summary = BatchAttentionSummary::from_patterns(&attention_patterns, avg_sharpness);

        // Compact 判定: Skip 或 MaskedCompute 比例高时触发
        let compact_required = matches!(
            skip_summary.batch_advice,
            BatchSkipAdvice::MostlySkip | BatchSkipAdvice::PartialMask
        );

        let waste_ratio = if batch_telemetry.is_empty() {
            0.0
        } else {
            total_waste / batch_telemetry.len() as f32
        };

        EpilogueBatchSummary {
            skip_summary,
            attention_summary,
            bypass_stats,
            compact_required,
            waste_ratio,
            per_request,
        }
    }

    /// 将 Epilogue 决策结果写入 KvPageHeader
    ///
    /// 在 Executor::step() 结果循环中，为每个请求的当前页面调用。
    /// 写入的遥测字段供下一步的 build_batch 和调度决策使用。
    pub fn write_page_header(
        &self,
        header: &mut KvPageHeader,
        decision: &RequestEpilogueDecision,
        telemetry: &SequenceTelemetry,
    ) {
        // §13.9: Softmax 信号 — 近似映射
        let softmax_max_val = if telemetry.per_head_entropy > 0.0 {
            // 高熵 → 低 max (分散); 低熵 → 高 max (集中)
            1.0 / (1.0 + telemetry.per_head_entropy)
        } else {
            0.0
        };
        header.softmax_max_avg = f32_to_f16_bits(softmax_max_val);
        header.centroid_pos = f32_to_f16_bits(telemetry.per_head_entropy);

        // §13.3: 残差能量差
        header.delta_rho_avg = f32_to_f16_bits(1.0 + telemetry.transform_ratio);

        // §13.5: 死神经元比例
        header.dead_ratio = f32_to_dead_ratio(telemetry.dead_density);

        // §13.8: per-channel scale (placeholder)
        header.v_scale_factor = 0;

        // §7: logits 熵
        header.entropy_avg = f32_to_f16_bits(telemetry.output_entropy);
    }

    /// 将 SequenceTelemetry 的 6 个字段转换为 EpilogueSignal 并喂入聚合器
    fn ingest_sequence_telemetry(&mut self, t: &SequenceTelemetry) {
        // §13.5: dead_density → DeadNeuronRatio
        self.aggregator.ingest(&EpilogueSignal::DeadNeuronRatio {
            ratio: t.dead_density,
        });

        // §13.3: transform_ratio → ResidualDeltaRho
        self.aggregator.ingest(&EpilogueSignal::ResidualDeltaRho {
            delta_rho: 1.0 + t.transform_ratio,
        });

        // §13.9: per_head_entropy → SoftmaxSharpness (近似)
        let max_val = if t.per_head_entropy > 0.0 {
            1.0 / (1.0 + t.per_head_entropy)
        } else {
            0.0
        };
        self.aggregator.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val,
            sharpness: t.per_head_entropy,
        });

        // §17.9: output_entropy → OutputEntropy
        self.aggregator.ingest(&EpilogueSignal::OutputEntropy {
            entropy: t.output_entropy,
        });
    }

    /// 重置所有决策器状态
    pub fn reset(&mut self) {
        self.gate_skip.reset_stats();
        self.sink_tracker.reset_stats();
        self.residual_bypass.reset();
        self.spec_signal.reset();
        self.aggregator = TelemetryAggregator::new();
    }

    /// 获取层数
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// 获取 Gate-First Skip 层（只读）
    pub fn gate_skip(&self) -> &GateFirstSkipLayer {
        &self.gate_skip
    }

    /// 获取 Sink Tracker（只读）
    pub fn sink_tracker(&self) -> &SinkTracker {
        &self.sink_tracker
    }

    /// 获取 Residual Bypass Layer（只读）
    pub fn residual_bypass(&self) -> &ResidualBypassLayer {
        &self.residual_bypass
    }

    /// 获取 Centroid Prefetch（只读）
    pub fn prefetch(&self) -> &CentroidPrefetch {
        &self.prefetch
    }

    /// 获取 Centroid Prefetch（可变）
    pub fn prefetch_mut(&mut self) -> &mut CentroidPrefetch {
        &mut self.prefetch
    }

    /// 获取专家热度追踪器（只读）
    pub fn expert_thermal(&self) -> Option<&ExpertThermalTracker> {
        self.expert_thermal.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{f16_bits_to_f32, dead_ratio_to_f32};

    fn make_telemetry(
        dead_density: f32,
        transform_ratio: f32,
        per_head_entropy: f32,
        output_entropy: f32,
    ) -> SequenceTelemetry {
        SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: false,
            dead_density,
            per_head_entropy,
            transform_ratio,
            output_entropy,
        }
    }

    #[test]
    fn test_epilogue_config_default() {
        let config = EpilogueConfig::default();
        assert!(config.gate_skip.enabled);
        assert!(config.residual_bypass.enabled);
        assert_eq!(config.num_experts, 0);
    }

    #[test]
    fn test_epilogue_subsystem_new() {
        let config = EpilogueConfig {
            num_layers: 12,
            num_experts: 8,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 12);
        assert!(sub.expert_thermal().is_some());
    }

    #[test]
    fn test_ingest_single_telemetry_normal() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 6,
            ..Default::default()
        });

        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);

        assert_eq!(summary.per_request.len(), 1);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
        assert!(!summary.compact_required);
        assert!(summary.waste_ratio < 0.01);
    }

    #[test]
    fn test_ingest_batch_mixed() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let batch = vec![
            make_telemetry(0.6, 0.5, 2.0, 3.0),  // dead_density > 0.5 → Skip
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // Normal
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute (25-50%)
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
        ];
        let summary = sub.ingest_and_decide(&batch);

        assert_eq!(summary.per_request.len(), 4);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::Skip);
        assert_eq!(summary.per_request[1].gate_skip, GateSkipDecision::FullCompute);
        assert_eq!(summary.per_request[3].gate_skip, GateSkipDecision::Skip);
        assert!(summary.waste_ratio > 0.0);
    }

    #[test]
    fn test_compact_triggered_by_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 6,
            ..Default::default()
        });

        // >50% Skip → compact_required
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!(summary.compact_required);
    }

    #[test]
    fn test_sink_protection_from_telemetry() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        // Low entropy → high softmax max → Sink pattern
        let t = make_telemetry(0.1, 0.5, 0.05, 0.5);
        let summary = sub.ingest_and_decide(&[t]);

        assert_eq!(
            summary.per_request[0].attention_pattern,
            AttentionPattern::Sink
        );
    }

    #[test]
    fn test_spec_schedule_from_telemetry() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        // Low output_entropy → EnableSpec
        let t = make_telemetry(0.1, 0.5, 2.0, 1.0);
        let summary = sub.ingest_and_decide(&[t]);

        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::EnableSpec
        );
    }

    #[test]
    fn test_write_page_header() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let mut header = KvPageHeader::default();
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::FullCompute,
            attention_pattern: AttentionPattern::Normal,
            bypass_decision: ResidualBypassDecision::Execute,
            spec_advice: SpecScheduleAdvice::StandardDecode,
            prefetch_advice: PrefetchAdvice::None,
        };
        let t = make_telemetry(0.35, 0.002, 1.5, 2.5);

        sub.write_page_header(&mut header, &decision, &t);

        assert!((dead_ratio_to_f32(header.dead_ratio) - 0.35).abs() < 0.01);
        assert!((f16_bits_to_f32(header.delta_rho_avg) - 1.002).abs() < 0.01);
        assert!((f16_bits_to_f32(header.entropy_avg) - 2.5).abs() < 0.1);
        assert!(f16_bits_to_f32(header.softmax_max_avg) > 0.0);
    }

    #[test]
    fn test_expert_thermal_integration() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            num_experts: 4,
            ..Default::default()
        });

        assert!(sub.expert_thermal().is_some());

        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);
        // ExpertThermalTracker should have been updated
    }

    #[test]
    fn test_disabled_modules() {
        let config = EpilogueConfig {
            gate_skip: GateFirstSkipConfig {
                enabled: false,
                ..Default::default()
            },
            residual_bypass: ResidualBypassConfig {
                enabled: false,
                ..Default::default()
            },
            num_layers: 12,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);

        // Dead density > 50%, but gate_skip disabled → FullCompute
        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);

        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
        assert_eq!(summary.per_request[0].bypass_decision, ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_empty_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let summary = sub.ingest_and_decide(&[]);

        assert_eq!(summary.per_request.len(), 0);
        assert!(!summary.compact_required);
        assert_eq!(summary.waste_ratio, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 6,
            ..Default::default()
        });

        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);

        sub.reset();
        assert_eq!(sub.gate_skip().skip_rate(), 0.0);
        assert_eq!(sub.sink_tracker().total_detections(), 0);
    }
}
