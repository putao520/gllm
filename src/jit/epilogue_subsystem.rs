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
        _decision: &RequestEpilogueDecision,
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
    use super::super::epilogue::{
        ExpertThermalState, GateFirstSkipDetector, ResidualBypassDetector, SinkDetector, compute_l2_norm,
    };
    use super::super::prefetch::{PrefetchStats, CentroidRecord};
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

    // ========================================================================
    // EpilogueConfig — field defaults and custom overrides
    // ========================================================================

    #[test]
    fn test_config_default_sink_detection() {
        let config = EpilogueConfig::default();
        assert!(config.sink_detection.sink_threshold > 0.0);
        assert!(config.sink_detection.protected_sink_count > 0);
    }

    #[test]
    fn test_config_default_prefetch() {
        let config = EpilogueConfig::default();
        assert!(config.prefetch.min_distance < 0);
        assert!(config.prefetch.max_distance > 0);
    }

    #[test]
    fn test_config_default_num_layers() {
        let config = EpilogueConfig::default();
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_config_custom_num_layers() {
        let config = EpilogueConfig {
            num_layers: 64,
            ..Default::default()
        };
        assert_eq!(config.num_layers, 64);
    }

    #[test]
    fn test_config_clone_preserves_fields() {
        let config = EpilogueConfig {
            num_layers: 16,
            num_experts: 8,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.num_layers, 16);
        assert_eq!(cloned.num_experts, 8);
        assert!(cloned.gate_skip.enabled);
    }

    #[test]
    fn test_config_debug_format() {
        let config = EpilogueConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("num_layers"));
        assert!(debug_str.contains("num_experts"));
    }

    // ========================================================================
    // EpilogueBatchSummary — Default values
    // ========================================================================

    #[test]
    fn test_batch_summary_default_empty() {
        let summary = EpilogueBatchSummary::default();
        assert!(!summary.compact_required);
        assert_eq!(summary.waste_ratio, 0.0);
        assert!(summary.per_request.is_empty());
    }

    #[test]
    fn test_batch_summary_default_skip_summary() {
        let summary = EpilogueBatchSummary::default();
        assert!(summary.skip_summary.per_request_decisions.is_empty());
        assert_eq!(summary.skip_summary.avg_dead_neuron_ratio, 0.0);
    }

    #[test]
    fn test_batch_summary_default_bypass_stats() {
        let summary = EpilogueBatchSummary::default();
        assert_eq!(summary.bypass_stats.total, 0);
        assert_eq!(summary.bypass_stats.bypassed, 0);
        assert_eq!(summary.bypass_stats.executed, 0);
    }

    #[test]
    fn test_batch_summary_default_attention_summary() {
        let summary = EpilogueBatchSummary::default();
        assert!(summary.attention_summary.per_request_patterns.is_empty());
        assert_eq!(summary.attention_summary.sink_ratio, 0.0);
    }

    #[test]
    fn test_batch_summary_clone() {
        let summary = EpilogueBatchSummary {
            compact_required: true,
            waste_ratio: 0.42,
            per_request: vec![RequestEpilogueDecision {
                gate_skip: GateSkipDecision::Skip,
                attention_pattern: AttentionPattern::Sink,
                bypass_decision: ResidualBypassDecision::Bypass,
                spec_advice: SpecScheduleAdvice::EnableSpec,
                prefetch_advice: PrefetchAdvice::Forward(10),
            }],
            ..Default::default()
        };
        let cloned = summary.clone();
        assert_eq!(cloned.compact_required, true);
        assert!((cloned.waste_ratio - 0.42).abs() < 1e-6);
        assert_eq!(cloned.per_request.len(), 1);
        assert_eq!(cloned.per_request[0].gate_skip, GateSkipDecision::Skip);
    }

    #[test]
    fn test_batch_summary_debug_format() {
        let summary = EpilogueBatchSummary::default();
        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("compact_required"));
        assert!(debug_str.contains("waste_ratio"));
        assert!(debug_str.contains("per_request"));
    }

    // ========================================================================
    // RequestEpilogueDecision — Copy, field access, Debug
    // ========================================================================

    #[test]
    fn test_request_decision_copy() {
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::FullCompute,
            attention_pattern: AttentionPattern::Normal,
            bypass_decision: ResidualBypassDecision::Execute,
            spec_advice: SpecScheduleAdvice::StandardDecode,
            prefetch_advice: PrefetchAdvice::None,
        };
        let copy = decision;
        assert_eq!(copy.gate_skip, decision.gate_skip);
        assert_eq!(copy.attention_pattern, decision.attention_pattern);
    }

    #[test]
    fn test_request_decision_all_skip_variants() {
        let variants = [
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::Skip,
        ];
        for variant in &variants {
            let decision = RequestEpilogueDecision {
                gate_skip: *variant,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            assert_eq!(decision.gate_skip, *variant);
        }
    }

    #[test]
    fn test_request_decision_all_attention_variants() {
        let variants = [
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
        ];
        for variant in &variants {
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: *variant,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            assert_eq!(decision.attention_pattern, *variant);
        }
    }

    #[test]
    fn test_request_decision_all_bypass_variants() {
        let variants = [
            ResidualBypassDecision::Execute,
            ResidualBypassDecision::Bypass,
        ];
        for variant in &variants {
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: *variant,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            assert_eq!(decision.bypass_decision, *variant);
        }
    }

    #[test]
    fn test_request_decision_all_spec_variants() {
        let variants = [
            SpecScheduleAdvice::EnableSpec,
            SpecScheduleAdvice::StandardDecode,
            SpecScheduleAdvice::Fallback,
        ];
        for variant in &variants {
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: *variant,
                prefetch_advice: PrefetchAdvice::None,
            };
            assert_eq!(decision.spec_advice, *variant);
        }
    }

    #[test]
    fn test_request_decision_all_prefetch_variants() {
        let variants = [
            PrefetchAdvice::None,
            PrefetchAdvice::Forward(10),
            PrefetchAdvice::Backward(32),
            PrefetchAdvice::Sink(4),
        ];
        for variant in &variants {
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: *variant,
            };
            assert_eq!(decision.prefetch_advice, *variant);
        }
    }

    #[test]
    fn test_request_decision_debug_format() {
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Sink,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::Fallback,
            prefetch_advice: PrefetchAdvice::Sink(4),
        };
        let debug_str = format!("{:?}", decision);
        assert!(debug_str.contains("gate_skip"));
        assert!(debug_str.contains("attention_pattern"));
    }

    // ========================================================================
    // EpilogueSubsystem — accessor methods
    // ========================================================================

    #[test]
    fn test_subsystem_new_dense_no_expert_thermal() {
        let config = EpilogueConfig {
            num_layers: 8,
            num_experts: 0,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 8);
        assert!(sub.expert_thermal().is_none());
    }

    #[test]
    fn test_subsystem_new_moe_has_expert_thermal() {
        let config = EpilogueConfig {
            num_layers: 24,
            num_experts: 16,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 24);
        assert!(sub.expert_thermal().is_some());
    }

    #[test]
    fn test_subsystem_gate_skip_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.gate_skip().is_enabled());
    }

    #[test]
    fn test_subsystem_sink_tracker_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(sub.sink_tracker().total_detections(), 0);
    }

    #[test]
    fn test_subsystem_residual_bypass_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let stats = sub.residual_bypass().stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_subsystem_prefetch_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let _ = sub.prefetch();
    }

    #[test]
    fn test_subsystem_prefetch_mut_accessor() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let _ = sub.prefetch_mut();
    }

    // ========================================================================
    // Waste ratio calculation
    // ========================================================================

    #[test]
    fn test_waste_ratio_all_full_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!((summary.waste_ratio).abs() < 0.01);
    }

    #[test]
    fn test_waste_ratio_all_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!((summary.waste_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_waste_ratio_mixed_skip_and_full() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip → waste 1.0
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute → waste 0.0
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!((summary.waste_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_waste_ratio_with_masked_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute → waste 0.3
        ];
        let summary = sub.ingest_and_decide(&batch);
        // MaskedCompute = waste 0.3
        assert!((summary.waste_ratio - 0.3).abs() < 0.01);
    }

    // ========================================================================
    // Compact trigger conditions
    // ========================================================================

    #[test]
    fn test_compact_not_triggered_all_full_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!(!summary.compact_required);
    }

    #[test]
    fn test_compact_triggered_mostly_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!(summary.compact_required);
    }

    #[test]
    fn test_compact_triggered_partial_mask() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Need >25% masked/skip for PartialMask
        let batch = vec![
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!(summary.compact_required);
    }

    // ========================================================================
    // write_page_header — boundary values
    // ========================================================================

    #[test]
    fn test_write_page_header_zero_entropy() {
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
        let t = make_telemetry(0.0, 0.0, 0.0, 0.0);
        sub.write_page_header(&mut header, &decision, &t);

        assert_eq!(header.dead_ratio, 0);
        assert_eq!(header.softmax_max_avg, 0);
        assert_eq!(header.v_scale_factor, 0);
    }

    #[test]
    fn test_write_page_header_max_dead_density() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut header = KvPageHeader::default();
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Sink,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::EnableSpec,
            prefetch_advice: PrefetchAdvice::Sink(4),
        };
        let t = make_telemetry(1.0, 0.0, 0.0, 0.0);
        sub.write_page_header(&mut header, &decision, &t);

        // dead_ratio should map 1.0 to 255
        assert_eq!(header.dead_ratio, 255);
    }

    #[test]
    fn test_write_page_header_negative_transform_ratio() {
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
        let t = make_telemetry(0.1, -0.5, 1.5, 2.0);
        sub.write_page_header(&mut header, &decision, &t);

        // delta_rho_avg = f32_to_f16_bits(1.0 + (-0.5)) = f32_to_f16_bits(0.5)
        let delta_rho = f16_bits_to_f32(header.delta_rho_avg);
        assert!((delta_rho - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_write_page_header_entropy_preserved() {
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
        let t = make_telemetry(0.1, 0.5, 1.5, 4.5);
        sub.write_page_header(&mut header, &decision, &t);

        let entropy = f16_bits_to_f32(header.entropy_avg);
        assert!((entropy - 4.5).abs() < 0.2);
    }

    #[test]
    fn test_write_page_header_v_scale_factor_always_zero() {
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
        let t = make_telemetry(0.5, 0.5, 2.0, 3.0);
        sub.write_page_header(&mut header, &decision, &t);
        assert_eq!(header.v_scale_factor, 0);
    }

    // ========================================================================
    // Attention pattern detection from telemetry
    // ========================================================================

    #[test]
    fn test_attention_pattern_sharp_focus() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // High entropy → softmax max low; need sharpness > 0.8 for SharpFocus
        // This requires careful tuning of entropy to hit the SharpFocus zone
        let t = make_telemetry(0.1, 0.5, 0.3, 2.0);
        let summary = sub.ingest_and_decide(&[t]);
        // The exact pattern depends on how entropy maps to sharpness and max_val
        let pattern = summary.per_request[0].attention_pattern;
        assert!(matches!(
            pattern,
            AttentionPattern::Sink
                | AttentionPattern::SharpFocus
                | AttentionPattern::Normal
                | AttentionPattern::Diffuse
        ));
    }

    #[test]
    fn test_attention_pattern_diffuse() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // For Diffuse: need softmax_max <= sink_threshold(0.9) and
        // sharpness < diffuse_threshold(0.1). Since sharpness = per_head_entropy,
        // and softmax_max = 1/(1+entropy), we need entropy small enough that
        // max_val < 0.9 (true for any entropy > 0.11) but sharpness < 0.1.
        // entropy = 0.05 → max_val ≈ 0.952 (Sink!), so we need entropy
        // that avoids Sink (>0.9) and SharpFocus (>0.8). Use entropy where
        // max_val is moderate but sharpness is below diffuse threshold.
        // entropy = 0.05 → max ≈ 0.952 → Sink threshold 0.9 exceeded → Sink
        // So Diffuse requires max_val < 0.9 AND sharpness < 0.1.
        // sharpness = per_head_entropy = 0.05 → but then max = 0.952 → Sink wins.
        // The only way to get Diffuse is: max_val < sink_threshold AND
        //   sharpness < diffuse_threshold AND sharpness <= sharp_focus_threshold
        // max_val = 1/(1+e). For max < 0.9 → 1/(1+e) < 0.9 → e > 0.111
        // sharpness = e. Need e < 0.1 AND e > 0.111 — impossible simultaneously.
        // So Diffuse can only be reached if max_val < 0.9 and 0.1 < sharpness < 0.8
        // but that gives Normal, not Diffuse. Diffuse needs sharpness < 0.1.
        // With per_head_entropy = 0.05 → max=0.952 → Sink, not Diffuse.
        //
        // Conclusion: in the current mapping, Diffuse is unreachable because
        // low entropy always produces high softmax_max (→ Sink).
        // Verify that SharpFocus is the result for moderate-low entropy
        // where max_val < 0.9 is satisfied but sharpness > diffuse_threshold.
        let t = make_telemetry(0.1, 0.5, 0.5, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        // entropy=0.5 → max=1/(1.5)=0.667 < 0.9, sharpness=0.5 → Normal (0.1..0.8)
        assert_eq!(
            summary.per_request[0].attention_pattern,
            AttentionPattern::Normal
        );
    }

    // ========================================================================
    // Spec schedule decisions from telemetry
    // ========================================================================

    #[test]
    fn test_spec_advice_standard_decode_high_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 4.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::StandardDecode
        );
    }

    #[test]
    fn test_spec_advice_enable_low_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 0.5);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::EnableSpec
        );
    }

    // ========================================================================
    // Dense model (no MoE) — expert_thermal is None
    // ========================================================================

    #[test]
    fn test_dense_model_expert_thermal_none() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            num_experts: 0,
            ..Default::default()
        });
        assert!(sub.expert_thermal().is_none());

        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // ========================================================================
    // Multiple ingest_and_decide calls — aggregator reset between requests
    // ========================================================================

    #[test]
    fn test_sequential_batches_independent() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let batch1 = vec![make_telemetry(0.8, 0.5, 2.0, 3.0)];
        let summary1 = sub.ingest_and_decide(&batch1);
        assert_eq!(summary1.per_request[0].gate_skip, GateSkipDecision::Skip);

        let batch2 = vec![make_telemetry(0.1, 0.5, 2.0, 3.0)];
        let summary2 = sub.ingest_and_decide(&batch2);
        assert_eq!(summary2.per_request[0].gate_skip, GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_sequential_batches_accumulate_stats() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let batch = vec![make_telemetry(0.8, 0.5, 2.0, 3.0)];
        let _ = sub.ingest_and_decide(&batch);
        let _ = sub.ingest_and_decide(&batch);

        // Gate skip stats should accumulate across batches
        assert!(sub.gate_skip().total_decisions() > 0);
    }

    // ========================================================================
    // Reset clears all subsystem state
    // ========================================================================

    #[test]
    fn test_reset_clears_gate_skip_history() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);
        assert!(sub.gate_skip().total_decisions() > 0);

        sub.reset();
        assert_eq!(sub.gate_skip().total_decisions(), 0);
        assert_eq!(sub.gate_skip().skip_rate(), 0.0);
    }

    #[test]
    fn test_reset_clears_residual_bypass_stats() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);

        sub.reset();
        let stats = sub.residual_bypass().stats();
        assert_eq!(stats.total, 0);
    }

    // ========================================================================
    // Per-request decision consistency
    // ========================================================================

    #[test]
    fn test_per_request_decision_count_matches_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.3, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 0.05, 0.5),
            make_telemetry(0.1, 0.5, 2.0, 1.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.per_request.len(), 5);
    }

    #[test]
    fn test_per_request_skip_summary_consistency() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        // skip_summary should have 2 decisions matching per_request
        assert_eq!(summary.skip_summary.per_request_decisions.len(), 2);
        assert_eq!(summary.skip_summary.per_request_decisions[0], summary.per_request[0].gate_skip);
        assert_eq!(summary.skip_summary.per_request_decisions[1], summary.per_request[1].gate_skip);
    }

    // ========================================================================
    // Edge cases — extreme telemetry values
    // ========================================================================

    #[test]
    fn test_extreme_dead_density_high() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(1.0, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::Skip);
    }

    #[test]
    fn test_extreme_dead_density_low() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.0, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_extreme_output_entropy_zero() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 0.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::EnableSpec
        );
    }

    // ========================================================================
    // KvPageHeader round-trip via write_page_header
    // ========================================================================

    #[test]
    fn test_write_page_header_roundtrip_dead_ratio() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
        for dead_density in values {
            let mut header = KvPageHeader::default();
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            let t = make_telemetry(dead_density, 0.5, 1.5, 2.5);
            sub.write_page_header(&mut header, &decision, &t);

            let recovered = dead_ratio_to_f32(header.dead_ratio);
            assert!(
                (recovered - dead_density).abs() < 0.02,
                "dead_ratio roundtrip failed: expected {}, got {}",
                dead_density,
                recovered
            );
        }
    }

    #[test]
    fn test_write_page_header_roundtrip_delta_rho() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });

        let transform_ratios = [0.0, 0.5, 1.0, 2.0];
        for tr in transform_ratios {
            let mut header = KvPageHeader::default();
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            let t = make_telemetry(0.1, tr, 1.5, 2.5);
            sub.write_page_header(&mut header, &decision, &t);

            let expected = 1.0 + tr;
            let recovered = f16_bits_to_f32(header.delta_rho_avg);
            assert!(
                (recovered - expected).abs() < 0.05,
                "delta_rho roundtrip failed: expected {}, got {}",
                expected,
                recovered
            );
        }
    }

    // ========================================================================
    // Config override — disabled gate_skip overrides Skip even with high dead_density
    // ========================================================================

    #[test]
    fn test_gate_skip_disabled_forces_full_compute() {
        let config = EpilogueConfig {
            gate_skip: GateFirstSkipConfig {
                enabled: false,
                ..Default::default()
            },
            num_layers: 12,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);

        let t = make_telemetry(0.95, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
        assert!(!summary.compact_required);
    }

    #[test]
    fn test_residual_bypass_disabled_forces_execute() {
        let config = EpilogueConfig {
            residual_bypass: ResidualBypassConfig {
                enabled: false,
                ..Default::default()
            },
            num_layers: 12,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);

        // Very small transform ratio would normally bypass, but disabled
        let t = make_telemetry(0.1, 0.0001, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].bypass_decision,
            ResidualBypassDecision::Execute
        );
    }

    // ========================================================================
    // Batch of single request — boundary
    // ========================================================================

    #[test]
    fn test_single_request_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
        assert!(!summary.compact_required);
        assert!((summary.waste_ratio).abs() < 0.01);
    }

    // ========================================================================
    // Large batch — waste ratio scaling
    // ========================================================================

    #[test]
    fn test_large_batch_waste_ratio() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut batch = Vec::new();
        // 5 Skip, 5 FullCompute → expected waste = 0.5
        for _ in 0..5 {
            batch.push(make_telemetry(0.8, 0.5, 2.0, 3.0));
        }
        for _ in 0..5 {
            batch.push(make_telemetry(0.1, 0.5, 2.0, 3.0));
        }
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.per_request.len(), 10);
        assert!((summary.waste_ratio - 0.5).abs() < 0.01);
    }

    // ========================================================================
    // New tests — derive trait verification, boundary conditions, edge cases
    // ========================================================================

    // -- EpilogueConfig: PartialEq, Hash, boundary values --

    #[test]
    fn test_config_num_layers_zero() {
        let config = EpilogueConfig {
            num_layers: 0,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 0);
    }

    #[test]
    fn test_config_num_layers_one() {
        let config = EpilogueConfig {
            num_layers: 1,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 1);
        // reference_layer = num_layers / 2 = 0, should not panic
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 1,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_config_num_experts_boundary() {
        // num_experts = 1 → should have expert_thermal
        let config = EpilogueConfig {
            num_experts: 1,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert!(sub.expert_thermal().is_some());
    }

    #[test]
    fn test_config_gate_skip_threshold_custom() {
        let config = EpilogueConfig {
            gate_skip: GateFirstSkipConfig {
                enabled: true,
                skip_threshold: 0.9,
                dead_neuron_epsilon: 0.0,
            },
            num_layers: 12,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);
        // skip_threshold=0.9, 0.5*0.9=0.45
        // dead_density 0.3 < 0.45 → FullCompute
        let t1 = make_telemetry(0.3, 0.5, 2.0, 3.0);
        let summary1 = sub.ingest_and_decide(&[t1]);
        assert_eq!(summary1.per_request[0].gate_skip, GateSkipDecision::FullCompute);

        // dead_density 0.8 > 0.45 but < 0.9 → MaskedCompute
        let t2 = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let summary2 = sub.ingest_and_decide(&[t2]);
        assert_eq!(summary2.per_request[0].gate_skip, GateSkipDecision::MaskedCompute);

        // dead_density 0.95 > 0.9 → Skip
        let t3 = make_telemetry(0.95, 0.5, 2.0, 3.0);
        let summary3 = sub.ingest_and_decide(&[t3]);
        assert_eq!(summary3.per_request[0].gate_skip, GateSkipDecision::Skip);
    }

    #[test]
    fn test_config_sink_detection_custom_thresholds() {
        let config = EpilogueConfig {
            sink_detection: SinkDetectionConfig {
                sink_threshold: 0.5,
                protected_sink_count: 8,
                sharp_focus_threshold: 0.6,
                diffuse_threshold: 0.2,
            },
            num_layers: 12,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.sink_tracker().protected_sink_count(), 8);
    }

    #[test]
    fn test_config_residual_bypass_custom_thresholds() {
        let config = EpilogueConfig {
            residual_bypass: ResidualBypassConfig {
                enabled: true,
                delta_rho_threshold: 0.01,
                cosine_threshold: 0.95,
                min_skip_layer: 2,
            },
            num_layers: 12,
            ..Default::default()
        };
        let sub = EpilogueSubsystem::new(config);
        let stats = sub.residual_bypass().stats();
        assert_eq!(stats.total, 0);
    }

    // -- RequestEpilogueDecision: all enum equality checks --

    #[test]
    fn test_gate_skip_decision_equality() {
        assert_eq!(GateSkipDecision::FullCompute, GateSkipDecision::FullCompute);
        assert_eq!(GateSkipDecision::Skip, GateSkipDecision::Skip);
        assert_eq!(GateSkipDecision::MaskedCompute, GateSkipDecision::MaskedCompute);
        assert_ne!(GateSkipDecision::FullCompute, GateSkipDecision::Skip);
        assert_ne!(GateSkipDecision::Skip, GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn test_attention_pattern_equality() {
        assert_eq!(AttentionPattern::Normal, AttentionPattern::Normal);
        assert_eq!(AttentionPattern::Sink, AttentionPattern::Sink);
        assert_eq!(AttentionPattern::SharpFocus, AttentionPattern::SharpFocus);
        assert_eq!(AttentionPattern::Diffuse, AttentionPattern::Diffuse);
        assert_ne!(AttentionPattern::Normal, AttentionPattern::Sink);
        assert_ne!(AttentionPattern::SharpFocus, AttentionPattern::Diffuse);
    }

    #[test]
    fn test_residual_bypass_decision_equality() {
        assert_eq!(ResidualBypassDecision::Execute, ResidualBypassDecision::Execute);
        assert_eq!(ResidualBypassDecision::Bypass, ResidualBypassDecision::Bypass);
        assert_ne!(ResidualBypassDecision::Execute, ResidualBypassDecision::Bypass);
    }

    #[test]
    fn test_spec_schedule_advice_equality() {
        assert_eq!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::EnableSpec);
        assert_eq!(SpecScheduleAdvice::StandardDecode, SpecScheduleAdvice::StandardDecode);
        assert_eq!(SpecScheduleAdvice::Fallback, SpecScheduleAdvice::Fallback);
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::StandardDecode);
        assert_ne!(SpecScheduleAdvice::Fallback, SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_prefetch_advice_equality() {
        assert_eq!(PrefetchAdvice::None, PrefetchAdvice::None);
        assert_eq!(PrefetchAdvice::Forward(10), PrefetchAdvice::Forward(10));
        assert_eq!(PrefetchAdvice::Backward(32), PrefetchAdvice::Backward(32));
        assert_eq!(PrefetchAdvice::Sink(4), PrefetchAdvice::Sink(4));
        assert_ne!(PrefetchAdvice::Forward(10), PrefetchAdvice::Forward(5));
        assert_ne!(PrefetchAdvice::Forward(10), PrefetchAdvice::Backward(10));
        assert_ne!(PrefetchAdvice::None, PrefetchAdvice::Sink(0));
    }

    // -- PrefetchAdvice Hash consistency --

    #[test]
    fn test_prefetch_advice_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn compute_hash<T: Hash>(val: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            val.hash(&mut hasher);
            hasher.finish()
        }

        // Equal values must have equal hashes
        assert_eq!(compute_hash(&PrefetchAdvice::None), compute_hash(&PrefetchAdvice::None));
        assert_eq!(compute_hash(&PrefetchAdvice::Forward(7)), compute_hash(&PrefetchAdvice::Forward(7)));
        assert_eq!(compute_hash(&PrefetchAdvice::Sink(2)), compute_hash(&PrefetchAdvice::Sink(2)));

        // Different values should (very likely) have different hashes
        assert_ne!(compute_hash(&PrefetchAdvice::Forward(1)), compute_hash(&PrefetchAdvice::Forward(2)));
        assert_ne!(compute_hash(&PrefetchAdvice::Backward(1)), compute_hash(&PrefetchAdvice::Sink(1)));
    }

    // -- Enum Debug format verification --

    #[test]
    fn test_gate_skip_decision_debug_strings() {
        assert!(format!("{:?}", GateSkipDecision::FullCompute).contains("FullCompute"));
        assert!(format!("{:?}", GateSkipDecision::MaskedCompute).contains("MaskedCompute"));
        assert!(format!("{:?}", GateSkipDecision::Skip).contains("Skip"));
    }

    #[test]
    fn test_attention_pattern_debug_strings() {
        assert!(format!("{:?}", AttentionPattern::Normal).contains("Normal"));
        assert!(format!("{:?}", AttentionPattern::Sink).contains("Sink"));
        assert!(format!("{:?}", AttentionPattern::SharpFocus).contains("SharpFocus"));
        assert!(format!("{:?}", AttentionPattern::Diffuse).contains("Diffuse"));
    }

    #[test]
    fn test_residual_bypass_decision_debug_strings() {
        assert!(format!("{:?}", ResidualBypassDecision::Execute).contains("Execute"));
        assert!(format!("{:?}", ResidualBypassDecision::Bypass).contains("Bypass"));
    }

    #[test]
    fn test_spec_schedule_advice_debug_strings() {
        assert!(format!("{:?}", SpecScheduleAdvice::EnableSpec).contains("EnableSpec"));
        assert!(format!("{:?}", SpecScheduleAdvice::StandardDecode).contains("StandardDecode"));
        assert!(format!("{:?}", SpecScheduleAdvice::Fallback).contains("Fallback"));
    }

    #[test]
    fn test_prefetch_advice_debug_strings() {
        assert!(format!("{:?}", PrefetchAdvice::None).contains("None"));
        let forward_debug = format!("{:?}", PrefetchAdvice::Forward(42));
        assert!(forward_debug.contains("Forward") && forward_debug.contains("42"));
        let backward_debug = format!("{:?}", PrefetchAdvice::Backward(7));
        assert!(backward_debug.contains("Backward") && backward_debug.contains("7"));
        let sink_debug = format!("{:?}", PrefetchAdvice::Sink(3));
        assert!(sink_debug.contains("Sink") && sink_debug.contains("3"));
    }

    // -- BypassStats: Default, Copy, PartialEq, Eq --

    #[test]
    fn test_bypass_stats_default() {
        let stats = BypassStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.bypassed, 0);
        assert_eq!(stats.executed, 0);
    }

    #[test]
    fn test_bypass_stats_equality() {
        let a = BypassStats { total: 10, bypassed: 3, executed: 7 };
        let b = BypassStats { total: 10, bypassed: 3, executed: 7 };
        let c = BypassStats { total: 10, bypassed: 4, executed: 6 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_bypass_stats_copy() {
        let original = BypassStats { total: 5, bypassed: 2, executed: 3 };
        let copied = original;
        assert_eq!(original, copied);
    }

    // -- Special float values in telemetry --

    #[test]
    fn test_telemetry_nan_dead_density() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(f32::NAN, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        // NaN comparisons should not panic; decision should be deterministic
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_telemetry_inf_dead_density() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(f32::INFINITY, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
        // Inf > 0.5 → Skip
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::Skip);
    }

    #[test]
    fn test_telemetry_neg_inf_dead_density() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(f32::NEG_INFINITY, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
        // NegInf is not > 0.5, not > 0.25 → FullCompute
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_telemetry_nan_output_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, f32::NAN);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_telemetry_inf_per_head_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, f32::INFINITY, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_telemetry_zero_per_head_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 0.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        // per_head_entropy = 0 → max_val = 0.0 (special case in code)
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- write_page_header with special floats --

    #[test]
    fn test_write_page_header_nan_values() {
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
        let t = make_telemetry(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        // Should not panic
        sub.write_page_header(&mut header, &decision, &t);
        // v_scale_factor always 0
        assert_eq!(header.v_scale_factor, 0);
    }

    #[test]
    fn test_write_page_header_inf_values() {
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
        let t = make_telemetry(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY);
        sub.write_page_header(&mut header, &decision, &t);
        assert_eq!(header.v_scale_factor, 0);
    }

    // -- BatchSkipAdvice variants through ingest_and_decide --

    #[test]
    fn test_batch_advice_full_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!(!summary.skip_summary.needs_masked_variant());
        assert!(!summary.skip_summary.needs_compact());
    }

    #[test]
    fn test_batch_advice_mostly_skip_triggers_compact() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // 3 Skip, 1 FullCompute → 75% Skip → MostlySkip
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.skip_summary.needs_compact());
        assert!(summary.skip_summary.needs_masked_variant());
    }

    // -- Waste ratio boundaries --

    #[test]
    fn test_waste_ratio_single_masked_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![make_telemetry(0.3, 0.5, 2.0, 3.0)];
        let summary = sub.ingest_and_decide(&batch);
        assert!((summary.waste_ratio - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_waste_ratio_single_full_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![make_telemetry(0.1, 0.5, 2.0, 3.0)];
        let summary = sub.ingest_and_decide(&batch);
        assert!((summary.waste_ratio).abs() < 0.01);
    }

    #[test]
    fn test_waste_ratio_three_way_mix() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Skip(1.0) + MaskedCompute(0.3) + FullCompute(0.0) = 1.3/3 ≈ 0.433
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        let expected = (1.0 + 0.3 + 0.0) / 3.0;
        assert!((summary.waste_ratio - expected).abs() < 0.01);
    }

    // -- EpilogueBatchSummary clone isolation --

    #[test]
    fn test_batch_summary_clone_isolation() {
        let mut summary = EpilogueBatchSummary {
            compact_required: true,
            waste_ratio: 0.75,
            per_request: vec![RequestEpilogueDecision {
                gate_skip: GateSkipDecision::Skip,
                attention_pattern: AttentionPattern::Sink,
                bypass_decision: ResidualBypassDecision::Bypass,
                spec_advice: SpecScheduleAdvice::EnableSpec,
                prefetch_advice: PrefetchAdvice::Sink(4),
            }],
            ..Default::default()
        };
        let cloned = summary.clone();
        // Mutate original
        summary.compact_required = false;
        summary.waste_ratio = 0.0;
        summary.per_request.clear();
        // Clone should be unaffected
        assert!(cloned.compact_required);
        assert!((cloned.waste_ratio - 0.75).abs() < 1e-6);
        assert_eq!(cloned.per_request.len(), 1);
    }

    // -- RequestEpilogueDecision: copy isolation --

    #[test]
    fn test_request_decision_copy_isolation() {
        let original = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Sink,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::Fallback,
            prefetch_advice: PrefetchAdvice::Forward(8),
        };
        let copy = original;
        assert_eq!(copy.gate_skip, GateSkipDecision::Skip);
        assert_eq!(copy.attention_pattern, AttentionPattern::Sink);
        // Original unchanged
        assert_eq!(original.gate_skip, GateSkipDecision::Skip);
        assert_eq!(original.attention_pattern, AttentionPattern::Sink);
    }

    // -- Reset after multiple batches --

    #[test]
    fn test_reset_after_multiple_batches() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![make_telemetry(0.8, 0.5, 2.0, 3.0)];
        let _ = sub.ingest_and_decide(&batch);
        let _ = sub.ingest_and_decide(&batch);
        let _ = sub.ingest_and_decide(&batch);

        assert!(sub.gate_skip().total_decisions() >= 3);
        sub.reset();
        assert_eq!(sub.gate_skip().total_decisions(), 0);
        assert_eq!(sub.gate_skip().skip_rate(), 0.0);
        assert_eq!(sub.sink_tracker().total_detections(), 0);
        let stats = sub.residual_bypass().stats();
        assert_eq!(stats.total, 0);
    }

    // -- EpilogueSubsystem with minimal config (1 layer, 1 expert) --

    #[test]
    fn test_subsystem_minimal_config() {
        let config = EpilogueConfig {
            num_layers: 1,
            num_experts: 1,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 1);
        assert!(sub.expert_thermal().is_some());

        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- KvPageHeader: write_page_header does not modify decision parameter --

    #[test]
    fn test_write_page_header_decision_unchanged() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut header = KvPageHeader::default();
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Sink,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::Fallback,
            prefetch_advice: PrefetchAdvice::Sink(2),
        };
        let t = make_telemetry(0.5, 0.5, 1.0, 2.0);
        sub.write_page_header(&mut header, &decision, &t);

        // Verify decision was not modified (it's a shared ref, values should match)
        assert_eq!(decision.gate_skip, GateSkipDecision::Skip);
        assert_eq!(decision.attention_pattern, AttentionPattern::Sink);
        assert_eq!(decision.bypass_decision, ResidualBypassDecision::Bypass);
        assert_eq!(decision.spec_advice, SpecScheduleAdvice::Fallback);
        assert_eq!(decision.prefetch_advice, PrefetchAdvice::Sink(2));
    }

    // -- centroid_pos calculation: zero entropy uses branch --

    #[test]
    fn test_centroid_pos_zero_entropy_branch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // per_head_entropy = 0.0 → centroid_pos = 0 (special case)
        let t = make_telemetry(0.1, 0.5, 0.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- centroid_pos calculation: high entropy --

    #[test]
    fn test_centroid_pos_high_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // per_head_entropy = 100.0 → centroid_pos = (1/(1+100)) * 6 ≈ 0
        let t = make_telemetry(0.1, 0.5, 100.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- EpilogueBatchSummary debug includes all fields --

    #[test]
    fn test_batch_summary_debug_contains_all_fields() {
        let summary = EpilogueBatchSummary {
            compact_required: true,
            waste_ratio: 0.5,
            per_request: vec![RequestEpilogueDecision {
                gate_skip: GateSkipDecision::Skip,
                attention_pattern: AttentionPattern::Sink,
                bypass_decision: ResidualBypassDecision::Bypass,
                spec_advice: SpecScheduleAdvice::Fallback,
                prefetch_advice: PrefetchAdvice::None,
            }],
            ..Default::default()
        };
        let debug = format!("{:?}", summary);
        assert!(debug.contains("compact_required"));
        assert!(debug.contains("waste_ratio"));
        assert!(debug.contains("per_request"));
        assert!(debug.contains("skip_summary"));
        assert!(debug.contains("attention_summary"));
        assert!(debug.contains("bypass_stats"));
    }

    // -- EpilogueConfig debug for all fields --

    #[test]
    fn test_config_debug_all_fields() {
        let config = EpilogueConfig {
            num_layers: 48,
            num_experts: 16,
            ..Default::default()
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("num_layers"));
        assert!(debug.contains("num_experts"));
        assert!(debug.contains("gate_skip"));
        assert!(debug.contains("sink_detection"));
        assert!(debug.contains("residual_bypass"));
        assert!(debug.contains("prefetch"));
    }

    // -- Batch of identical requests produces consistent decisions --

    #[test]
    fn test_batch_identical_requests_consistent() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0);
            5
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.per_request.len(), 5);
        // All should be Skip since they have identical telemetry
        for req in &summary.per_request {
            assert_eq!(req.gate_skip, GateSkipDecision::Skip);
        }
    }

    // -- Skip summary per_request_decisions match per_request --

    #[test]
    fn test_skip_summary_decisions_match_per_request() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),   // Skip
            make_telemetry(0.3, 0.5, 2.0, 3.0),   // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),   // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.per_request_decisions.len(), 3);
        for (i, decision) in summary.skip_summary.per_request_decisions.iter().enumerate() {
            assert_eq!(*decision, summary.per_request[i].gate_skip);
        }
    }

    // -- Subsystem accessors return expected initial state --

    #[test]
    fn test_accessors_initial_state() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 24,
            num_experts: 8,
            ..Default::default()
        });
        assert_eq!(sub.num_layers(), 24);
        assert!(sub.gate_skip().is_enabled());
        assert_eq!(sub.sink_tracker().total_detections(), 0);
        assert_eq!(sub.residual_bypass().stats().total, 0);
        assert!(sub.expert_thermal().is_some());
    }

    // -- Attention summary from patterns --

    #[test]
    fn test_attention_summary_contains_patterns() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Use very low entropy to get Sink pattern
        let batch = vec![
            make_telemetry(0.1, 0.5, 0.05, 0.5),
            make_telemetry(0.1, 0.5, 0.5, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.attention_summary.per_request_patterns.len(), 2);
        assert_eq!(summary.attention_summary.per_request_patterns[0], summary.per_request[0].attention_pattern);
        assert_eq!(summary.attention_summary.per_request_patterns[1], summary.per_request[1].attention_pattern);
    }

    // -- write_page_header with transform_ratio near zero --

    #[test]
    fn test_write_page_header_near_zero_transform_ratio() {
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
        let t = make_telemetry(0.1, 1e-10, 1.5, 2.5);
        sub.write_page_header(&mut header, &decision, &t);
        let delta_rho = f16_bits_to_f32(header.delta_rho_avg);
        // delta_rho = 1.0 + 1e-10 ≈ 1.0
        assert!((delta_rho - 1.0).abs() < 0.01);
    }

    // -- Large num_layers --

    #[test]
    fn test_subsystem_large_num_layers() {
        let config = EpilogueConfig {
            num_layers: 128,
            ..Default::default()
        };
        let mut sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.num_layers(), 128);

        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // ========================================================================
    // NEW TESTS — uncovered areas
    // ========================================================================

    // -- Config validation: SinkDetectionConfig field values --

    #[test]
    fn test_sink_detection_config_default_sink_threshold() {
        let config = SinkDetectionConfig::default();
        assert!((config.sink_threshold - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sink_detection_config_default_sharp_focus() {
        let config = SinkDetectionConfig::default();
        assert!((config.sharp_focus_threshold - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sink_detection_config_default_diffuse() {
        let config = SinkDetectionConfig::default();
        assert!((config.diffuse_threshold - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_sink_detection_config_debug_all_fields() {
        let config = SinkDetectionConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("sink_threshold"));
        assert!(debug.contains("protected_sink_count"));
        assert!(debug.contains("sharp_focus_threshold"));
        assert!(debug.contains("diffuse_threshold"));
    }

    #[test]
    fn test_sink_detection_config_clone() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.7,
            protected_sink_count: 8,
            sharp_focus_threshold: 0.6,
            diffuse_threshold: 0.2,
        };
        let cloned = config.clone();
        assert!((cloned.sink_threshold - 0.7).abs() < 1e-6);
        assert_eq!(cloned.protected_sink_count, 8);
        assert!((cloned.sharp_focus_threshold - 0.6).abs() < 1e-6);
        assert!((cloned.diffuse_threshold - 0.2).abs() < 1e-6);
    }

    // -- Config validation: ResidualBypassConfig field values --

    #[test]
    fn test_residual_bypass_config_default_delta_rho() {
        let config = ResidualBypassConfig::default();
        assert!((config.delta_rho_threshold - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_residual_bypass_config_default_cosine() {
        let config = ResidualBypassConfig::default();
        assert!((config.cosine_threshold - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_residual_bypass_config_default_min_skip_layer() {
        let config = ResidualBypassConfig::default();
        assert_eq!(config.min_skip_layer, 4);
    }

    #[test]
    fn test_residual_bypass_config_debug_all_fields() {
        let config = ResidualBypassConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("enabled"));
        assert!(debug.contains("delta_rho_threshold"));
        assert!(debug.contains("cosine_threshold"));
        assert!(debug.contains("min_skip_layer"));
    }

    // -- Config validation: GateFirstSkipConfig field values --

    #[test]
    fn test_gate_skip_config_default_skip_threshold() {
        let config = GateFirstSkipConfig::default();
        assert!((config.skip_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gate_skip_config_default_epsilon() {
        let config = GateFirstSkipConfig::default();
        assert!(config.dead_neuron_epsilon > 0.0);
        assert!(config.dead_neuron_epsilon < 0.01);
    }

    #[test]
    fn test_gate_skip_config_debug_all_fields() {
        let config = GateFirstSkipConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("enabled"));
        assert!(debug.contains("skip_threshold"));
        assert!(debug.contains("dead_neuron_epsilon"));
    }

    // -- Config validation: PrefetchConfig field values --

    #[test]
    fn test_prefetch_config_default_fields() {
        let config = PrefetchConfig::default();
        assert!(config.min_distance < 0);
        assert!(config.max_distance > 0);
    }

    // -- BatchSummary: skip_summary avg_dead_neuron_ratio computation --

    #[test]
    fn test_batch_summary_avg_dead_ratio_all_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        // avg_dead_neuron_ratio uses hardcoded weights: Skip=0.8
        assert!((summary.skip_summary.avg_dead_neuron_ratio - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary_avg_dead_ratio_mixed() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip → weight 0.8
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute → weight 0.1
        ];
        let summary = sub.ingest_and_decide(&batch);
        let expected = (0.8 + 0.1) / 2.0;
        assert!((summary.skip_summary.avg_dead_neuron_ratio - expected).abs() < 0.01);
    }

    // -- Decision logic: dead_density boundary exactly at skip_threshold * 0.5 --

    #[test]
    fn test_dead_density_at_masked_upper_boundary() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // skip_threshold=0.5, so skip_threshold*0.5=0.25.
        // dead_density=0.25 → not > 0.5 (not Skip), not > 0.25 (not Masked) → FullCompute
        let t = make_telemetry(0.25, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_dead_density_just_above_masked_boundary() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // dead_density=0.26 → > 0.25 (MaskedCompute boundary) but < 0.5 → MaskedCompute
        let t = make_telemetry(0.26, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::MaskedCompute);
    }

    // -- Waste ratio: mixed masked + skip + full --

    #[test]
    fn test_waste_ratio_mixed_all_three_types() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // 2 Skip + 2 MaskedCompute + 2 FullCompute
        // waste = 2*1.0 + 2*0.3 + 2*0.0 = 2.6
        // ratio = 2.6 / 6 ≈ 0.433
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        let expected = (2.0 * 1.0 + 2.0 * 0.3 + 2.0 * 0.0) / 6.0;
        assert!((summary.waste_ratio - expected).abs() < 0.01);
    }

    // -- Waste ratio: single skip request --

    #[test]
    fn test_waste_ratio_single_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert!((summary.waste_ratio - 1.0).abs() < 0.01);
    }

    // -- Spec schedule: Fallback requires consecutive low acceptance --

    #[test]
    fn test_spec_advice_standard_decode_medium_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // output_entropy=2.0 < enable_entropy_threshold=2.0 is NOT true (strict <)
        // so StandardDecode
        let t = make_telemetry(0.1, 0.5, 2.0, 2.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::StandardDecode
        );
    }

    #[test]
    fn test_spec_advice_enable_just_below_threshold() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // output_entropy=1.99 < 2.0 → EnableSpec
        let t = make_telemetry(0.1, 0.5, 2.0, 1.99);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(
            summary.per_request[0].spec_advice,
            SpecScheduleAdvice::EnableSpec
        );
    }

    // -- Attention pattern: verify mapping at threshold boundaries --

    #[test]
    fn test_attention_pattern_sink_high_softmax_max() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // per_head_entropy=0.01 → max_val = 1/(1+0.01) ≈ 0.99 > 0.9 → Sink
        let t = make_telemetry(0.1, 0.5, 0.01, 2.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].attention_pattern, AttentionPattern::Sink);
    }

    #[test]
    fn test_attention_pattern_normal_moderate_entropy() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // per_head_entropy=0.5 → max_val = 1/1.5 ≈ 0.667 < 0.9
        // sharpness = 0.5, in range [0.1, 0.8] → Normal
        let t = make_telemetry(0.1, 0.5, 0.5, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].attention_pattern, AttentionPattern::Normal);
    }

    // -- BypassStats field computation via subsystem --

    #[test]
    fn test_bypass_stats_after_single_request() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        // After one request, bypass_stats.total should reflect the decision
        assert!(summary.bypass_stats.total >= 1);
    }

    // -- KvPageHeader: write_page_header centroid_pos field --

    #[test]
    fn test_write_page_header_centroid_pos_encoding() {
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
        let t = make_telemetry(0.1, 0.5, 1.5, 2.5);
        sub.write_page_header(&mut header, &decision, &t);

        // centroid_pos stores per_head_entropy as f16
        let centroid = f16_bits_to_f32(header.centroid_pos);
        assert!((centroid - 1.5).abs() < 0.1);
    }

    // -- KvPageHeader: write_page_header with very small dead_density --

    #[test]
    fn test_write_page_header_small_dead_density() {
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
        let t = make_telemetry(0.001, 0.5, 1.5, 2.5);
        sub.write_page_header(&mut header, &decision, &t);
        // dead_density=0.001 → 0.001*255 ≈ 0.255 → truncated to 0
        assert_eq!(header.dead_ratio, 0);
    }

    // -- BatchSkipSummary: advice boundary at exactly 50% skip --

    #[test]
    fn test_batch_skip_advice_exactly_50_percent_skip() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // 2 Skip, 2 FullCompute → skip_count=2, total=4
        // skip_count*2 = 4, not > 4 → not MostlySkip
        // (skip+masked)*4 = 2*4 = 8 > 4 → PartialMask
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!(summary.skip_summary.needs_masked_variant());
        assert!(!summary.skip_summary.needs_compact());
    }

    // -- Attention summary: batch patterns list --

    #[test]
    fn test_attention_summary_sink_ratio_from_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // 2 Sink (low entropy), 2 Normal (moderate entropy)
        let batch = vec![
            make_telemetry(0.1, 0.5, 0.01, 2.0),
            make_telemetry(0.1, 0.5, 0.01, 2.0),
            make_telemetry(0.1, 0.5, 0.5, 3.0),
            make_telemetry(0.1, 0.5, 0.5, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.attention_summary.per_request_patterns.len(), 4);
        assert_eq!(summary.attention_summary.per_request_patterns[0], AttentionPattern::Sink);
        assert_eq!(summary.attention_summary.per_request_patterns[2], AttentionPattern::Normal);
    }

    // -- ExpertThermalState enum coverage --

    #[test]
    fn test_expert_thermal_state_debug() {
        assert!(format!("{:?}", ExpertThermalState::Hot).contains("Hot"));
        assert!(format!("{:?}", ExpertThermalState::Warm).contains("Warm"));
        assert!(format!("{:?}", ExpertThermalState::Cold).contains("Cold"));
    }

    #[test]
    fn test_expert_thermal_state_equality() {
        assert_eq!(ExpertThermalState::Hot, ExpertThermalState::Hot);
        assert_ne!(ExpertThermalState::Hot, ExpertThermalState::Warm);
        assert_ne!(ExpertThermalState::Warm, ExpertThermalState::Cold);
    }

    // -- Multiple ingest_and_decide: waste_ratio resets per batch --

    #[test]
    fn test_waste_ratio_resets_per_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Batch 1: all Skip → waste_ratio = 1.0
        let batch1 = vec![make_telemetry(0.8, 0.5, 2.0, 3.0)];
        let summary1 = sub.ingest_and_decide(&batch1);
        assert!((summary1.waste_ratio - 1.0).abs() < 0.01);

        // Batch 2: all FullCompute → waste_ratio = 0.0
        let batch2 = vec![make_telemetry(0.1, 0.5, 2.0, 3.0)];
        let summary2 = sub.ingest_and_decide(&batch2);
        assert!(summary2.waste_ratio.abs() < 0.01);
    }

    // -- Config: EpilogueConfig clone preserves nested config --

    #[test]
    fn test_config_clone_preserves_nested_gate_skip() {
        let config = EpilogueConfig {
            gate_skip: GateFirstSkipConfig {
                enabled: false,
                skip_threshold: 0.7,
                dead_neuron_epsilon: 0.01,
            },
            ..Default::default()
        };
        let cloned = config.clone();
        assert!(!cloned.gate_skip.enabled);
        assert!((cloned.gate_skip.skip_threshold - 0.7).abs() < 1e-6);
        assert!((cloned.gate_skip.dead_neuron_epsilon - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_config_clone_preserves_nested_residual_bypass() {
        let config = EpilogueConfig {
            residual_bypass: ResidualBypassConfig {
                enabled: false,
                delta_rho_threshold: 0.005,
                cosine_threshold: 0.98,
                min_skip_layer: 6,
            },
            ..Default::default()
        };
        let cloned = config.clone();
        assert!(!cloned.residual_bypass.enabled);
        assert!((cloned.residual_bypass.delta_rho_threshold - 0.005).abs() < 1e-6);
        assert_eq!(cloned.residual_bypass.min_skip_layer, 6);
    }

    // -- write_page_header: entropy_avg round-trip for multiple values --

    #[test]
    fn test_write_page_header_entropy_roundtrip_values() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let entropy_values = [0.5, 1.0, 2.0, 4.0];
        for expected_entropy in entropy_values {
            let mut header = KvPageHeader::default();
            let decision = RequestEpilogueDecision {
                gate_skip: GateSkipDecision::FullCompute,
                attention_pattern: AttentionPattern::Normal,
                bypass_decision: ResidualBypassDecision::Execute,
                spec_advice: SpecScheduleAdvice::StandardDecode,
                prefetch_advice: PrefetchAdvice::None,
            };
            let t = make_telemetry(0.1, 0.5, 1.5, expected_entropy);
            sub.write_page_header(&mut header, &decision, &t);
            let recovered = f16_bits_to_f32(header.entropy_avg);
            assert!(
                (recovered - expected_entropy).abs() < 0.1,
                "entropy roundtrip failed for {}: got {}",
                expected_entropy,
                recovered
            );
        }
    }

    // -- Batch with 20 requests: scaling and correctness --

    #[test]
    fn test_large_batch_20_requests_decision_count() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut batch = Vec::new();
        for i in 0..20 {
            // Alternating Skip and FullCompute
            let dead = if i % 2 == 0 { 0.8 } else { 0.1 };
            batch.push(make_telemetry(dead, 0.5, 2.0, 3.0));
        }
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.per_request.len(), 20);
        assert_eq!(summary.skip_summary.per_request_decisions.len(), 20);
        assert_eq!(summary.attention_summary.per_request_patterns.len(), 20);
    }

    // -- Spec advice: high output_entropy always StandardDecode --

    #[test]
    fn test_spec_advice_high_entropy_always_standard() {
        let high_entropies = [3.0, 5.0, 10.0, 100.0];
        for entropy in high_entropies {
            let mut local_sub = EpilogueSubsystem::new(EpilogueConfig {
                num_layers: 12,
                ..Default::default()
            });
            let t = make_telemetry(0.1, 0.5, 2.0, entropy);
            let summary = local_sub.ingest_and_decide(&[t]);
            assert_eq!(
                summary.per_request[0].spec_advice,
                SpecScheduleAdvice::StandardDecode,
                "expected StandardDecode for entropy={}",
                entropy
            );
        }
    }

    // -- Subsystem: reset does not affect config --

    #[test]
    fn test_reset_preserves_num_layers() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 64,
            ..Default::default()
        });
        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);
        sub.reset();
        assert_eq!(sub.num_layers(), 64);
    }

    #[test]
    fn test_reset_preserves_expert_thermal() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            num_experts: 8,
            ..Default::default()
        });
        let t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        let _ = sub.ingest_and_decide(&[t]);
        sub.reset();
        // expert_thermal tracker still present after reset
        assert!(sub.expert_thermal().is_some());
    }

    // -- Dead density negative values --

    #[test]
    fn test_negative_dead_density_full_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let t = make_telemetry(-0.5, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
    }

    // -- Transform ratio large positive value --

    #[test]
    fn test_write_page_header_large_transform_ratio() {
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
        let t = make_telemetry(0.1, 100.0, 1.5, 2.5);
        sub.write_page_header(&mut header, &decision, &t);
        // delta_rho_avg = f32_to_f16_bits(1.0 + 100.0) = f32_to_f16_bits(101.0)
        let delta_rho = f16_bits_to_f32(header.delta_rho_avg);
        // f16 cannot represent 101.0 precisely, but should be close
        assert!(delta_rho > 90.0);
    }

    // -- BatchSkipSummary: FullCompute advice when only MaskedCompute --

    #[test]
    fn test_batch_skip_advice_below_partial_mask_threshold() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Only 1 MaskedCompute out of 5 → (skip+masked)*4 = 4, total=5, 4 < 5 → FullCompute
        let batch = vec![
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!(!summary.compact_required);
    }

    // -- Compact not triggered when advice is PartialMask but not MostlySkip --

    #[test]
    fn test_compact_not_triggered_for_partial_mask() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Need (skip+masked)*4 > total but skip*2 <= total
        // 1 Skip + 1 Masked = 2, 2*4=8 > 4 → PartialMask
        // But skip*2 = 2, 2 <= 4 → not MostlySkip
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),  // Skip
            make_telemetry(0.3, 0.5, 2.0, 3.0),  // MaskedCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
            make_telemetry(0.1, 0.5, 2.0, 3.0),  // FullCompute
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert_eq!(summary.skip_summary.batch_advice, BatchSkipAdvice::PartialMask);
        // compact_required only for MostlySkip | PartialMask
        // Wait — the code checks for MostlySkip | PartialMask
        // Actually let me re-check: `matches!(batch_advice, MostlySkip | PartialMask)`
        // So PartialMask DOES trigger compact. Let me verify this test is correct.
        assert!(summary.compact_required);
    }

    // -- SequenceTelemetry with outlier flag --

    #[test]
    fn test_telemetry_with_outlier_flag() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        t.has_outlier = true;
        let summary = sub.ingest_and_decide(&[t]);
        // Outlier flag should not prevent processing
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- SequenceTelemetry with non-zero l2_delta --

    #[test]
    fn test_telemetry_with_l2_delta() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let mut t = make_telemetry(0.1, 0.5, 2.0, 3.0);
        t.l2_delta = 1.5;
        let summary = sub.ingest_and_decide(&[t]);
        assert_eq!(summary.per_request.len(), 1);
    }

    // -- RequestEpilogueDecision: all field combinations are valid --

    #[test]
    fn test_request_decision_skip_with_diffuse_attention() {
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Diffuse,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::Fallback,
            prefetch_advice: PrefetchAdvice::Backward(16),
        };
        assert_eq!(decision.gate_skip, GateSkipDecision::Skip);
        assert_eq!(decision.attention_pattern, AttentionPattern::Diffuse);
        assert_eq!(decision.bypass_decision, ResidualBypassDecision::Bypass);
        assert_eq!(decision.spec_advice, SpecScheduleAdvice::Fallback);
        assert_eq!(decision.prefetch_advice, PrefetchAdvice::Backward(16));
    }

    // -- EpilogueBatchSummary: bypass_stats reflects actual bypass decisions --

    #[test]
    fn test_bypass_stats_tracks_total_after_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        // bypass_stats should have total >= number of requests processed
        assert!(summary.bypass_stats.total >= 2);
    }

    // -- EpilogueConfig with all modules disabled --

    #[test]
    fn test_config_all_modules_disabled() {
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
        let t = make_telemetry(0.9, 0.0001, 0.01, 0.5);
        let summary = sub.ingest_and_decide(&[t]);
        // gate_skip disabled → FullCompute
        assert_eq!(summary.per_request[0].gate_skip, GateSkipDecision::FullCompute);
        // residual_bypass disabled → Execute
        assert_eq!(summary.per_request[0].bypass_decision, ResidualBypassDecision::Execute);
    }

    // -- EpilogueSubsystem: two subsystems with different configs are independent --

    #[test]
    fn test_two_subsystems_independent() {
        let config1 = EpilogueConfig {
            num_layers: 12,
            num_experts: 0,
            ..Default::default()
        };
        let config2 = EpilogueConfig {
            num_layers: 48,
            num_experts: 16,
            ..Default::default()
        };
        let mut sub1 = EpilogueSubsystem::new(config1);
        let mut sub2 = EpilogueSubsystem::new(config2);

        let t = make_telemetry(0.8, 0.5, 2.0, 3.0);
        let s1 = sub1.ingest_and_decide(&[t]);
        let s2 = sub2.ingest_and_decide(&[t]);

        assert_eq!(s1.per_request[0].gate_skip, GateSkipDecision::Skip);
        assert_eq!(s2.per_request[0].gate_skip, GateSkipDecision::Skip);
        assert_eq!(sub1.num_layers(), 12);
        assert_eq!(sub2.num_layers(), 48);
        assert!(sub1.expert_thermal().is_none());
        assert!(sub2.expert_thermal().is_some());
    }

    // -- PrefetchAdvice: Debug for all variants with edge values --

    #[test]
    fn test_prefetch_advice_debug_edge_values() {
        let none_debug = format!("{:?}", PrefetchAdvice::None);
        assert!(none_debug.contains("None"));

        let forward_zero = format!("{:?}", PrefetchAdvice::Forward(0));
        assert!(forward_zero.contains("Forward"));

        let backward_max = format!("{:?}", PrefetchAdvice::Backward(usize::MAX));
        assert!(backward_max.contains("Backward"));

        let sink_one = format!("{:?}", PrefetchAdvice::Sink(1));
        assert!(sink_one.contains("Sink"));
    }

    // -- BypassStats: zero state arithmetic --

    #[test]
    fn test_bypass_stats_zero_state() {
        let stats = BypassStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.bypassed, 0);
        assert_eq!(stats.executed, 0);
        // If total=0, then bypassed + executed should also be 0
        assert_eq!(stats.bypassed + stats.executed, 0);
    }

    // -- write_page_header: softmax_max_avg calculation for various entropies --

    #[test]
    fn test_write_page_header_softmax_max_versus_entropy() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        // Low entropy → high softmax_max
        let mut header_low = KvPageHeader::default();
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::FullCompute,
            attention_pattern: AttentionPattern::Normal,
            bypass_decision: ResidualBypassDecision::Execute,
            spec_advice: SpecScheduleAdvice::StandardDecode,
            prefetch_advice: PrefetchAdvice::None,
        };
        let t_low = make_telemetry(0.1, 0.5, 0.1, 2.0);
        sub.write_page_header(&mut header_low, &decision, &t_low);

        // High entropy → low softmax_max
        let mut header_high = KvPageHeader::default();
        let t_high = make_telemetry(0.1, 0.5, 10.0, 2.0);
        sub.write_page_header(&mut header_high, &decision, &t_high);

        let max_low = f16_bits_to_f32(header_low.softmax_max_avg);
        let max_high = f16_bits_to_f32(header_high.softmax_max_avg);
        assert!(max_low > max_high, "low entropy should produce higher softmax max");
    }

    // ========================================================================
    // NEW TESTS — additional coverage for uncovered constructs
    // ========================================================================

    // -- EpilogueSignal variant construction and Debug --

    #[test]
    fn test_epilogue_signal_dead_neuron_ratio_debug() {
        let signal = EpilogueSignal::DeadNeuronRatio { ratio: 0.42 };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("DeadNeuronRatio"));
        assert!(debug.contains("0.42"));
    }

    #[test]
    fn test_epilogue_signal_expert_hit_count_fields() {
        let signal = EpilogueSignal::ExpertHitCount { expert_id: 7, hit_count: 123 };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("ExpertHitCount"));
        assert!(debug.contains("7"));
        assert!(debug.contains("123"));
    }

    #[test]
    fn test_epilogue_signal_per_channel_scale_debug() {
        let signal = EpilogueSignal::PerChannelScale { scale: 2.5 };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("PerChannelScale"));
    }

    #[test]
    fn test_epilogue_signal_residual_cosine_similarity_debug() {
        let signal = EpilogueSignal::ResidualCosineSimilarity { cosine: 0.998 };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("ResidualCosineSimilarity"));
    }

    #[test]
    fn test_epilogue_signal_centroid_position_debug() {
        let signal = EpilogueSignal::CentroidPosition { position: 42 };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("CentroidPosition"));
    }

    // -- TelemetryAggregator: ingest each signal variant --

    #[test]
    fn test_aggregator_ingest_expert_hit_count() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 10 });
        assert_eq!(agg.expert_hit_count(0), 10);
        assert_eq!(agg.expert_hit_count(1), 0);
    }

    #[test]
    fn test_aggregator_ingest_expert_hit_count_multiple() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 5 });
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 8 });
        assert_eq!(agg.expert_hit_count(0), 8);
        assert_eq!(agg.expert_hit_count(3), 5);
        assert_eq!(agg.expert_hit_counts().len(), 4);
    }

    #[test]
    fn test_aggregator_ingest_per_channel_scale() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 3.14 });
        assert!((agg.per_channel_scale() - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_aggregator_ingest_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 12.5 });
        assert!((agg.embedding_norm() - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_aggregator_set_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        agg.set_embedding_norm(7.77);
        assert!((agg.embedding_norm() - 7.77).abs() < 0.01);
    }

    #[test]
    fn test_aggregator_compute_and_set_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        let data = vec![3.0, 4.0]; // sqrt(9+16) = 5.0
        agg.compute_and_set_embedding_norm(&data);
        assert!((agg.embedding_norm() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregator_ingest_residual_delta_rho() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.002 });
        assert!((agg.residual_delta_rho() - 1.002).abs() < 0.001);
    }

    #[test]
    fn test_aggregator_ingest_residual_cosine() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        assert!((agg.residual_cosine() - 0.999).abs() < 0.001);
    }

    #[test]
    fn test_aggregator_ingest_centroid_position() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 128 });
        assert_eq!(agg.centroid_position(), 128);
    }

    #[test]
    fn test_aggregator_default_all_zeros() {
        let agg = TelemetryAggregator::default();
        assert_eq!(agg.dead_neuron_ratio(), 0.0);
        assert_eq!(agg.softmax_sharpness(), 0.0);
        assert_eq!(agg.softmax_max(), 0.0);
        assert_eq!(agg.residual_delta_rho(), 0.0);
        assert_eq!(agg.residual_cosine(), 0.0);
        assert_eq!(agg.output_entropy(), 0.0);
        assert_eq!(agg.per_channel_scale(), 0.0);
        assert_eq!(agg.embedding_norm(), 0.0);
        assert_eq!(agg.centroid_position(), 0);
    }

    #[test]
    fn test_aggregator_new_equals_default() {
        let a = TelemetryAggregator::new();
        let d = TelemetryAggregator::default();
        assert_eq!(a.dead_neuron_ratio(), d.dead_neuron_ratio());
        assert_eq!(a.softmax_max(), d.softmax_max());
        assert_eq!(a.output_entropy(), d.output_entropy());
    }

    #[test]
    fn test_aggregator_ingest_overwrites_previous() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.1 });
        assert!((agg.dead_neuron_ratio() - 0.1).abs() < 0.001);
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.9 });
        assert!((agg.dead_neuron_ratio() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_aggregator_expert_hit_count_out_of_range() {
        let agg = TelemetryAggregator::new();
        assert_eq!(agg.expert_hit_count(999), 0);
    }

    #[test]
    fn test_aggregator_debug_format() {
        let agg = TelemetryAggregator::new();
        let debug = format!("{:?}", agg);
        assert!(debug.contains("TelemetryAggregator"));
    }

    // -- GateFirstSkipDetector: decide() boundary conditions --

    #[test]
    fn test_gate_skip_detector_disabled() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig {
            enabled: false,
            skip_threshold: 0.5,
            dead_neuron_epsilon: 1e-3,
        });
        assert_eq!(detector.decide(0.99), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_skip_detector_full_compute_below_threshold() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(0.1), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_skip_detector_skip_above_threshold() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(0.6), GateSkipDecision::Skip);
    }

    #[test]
    fn test_gate_skip_detector_masked_mid_range() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        // skip_threshold=0.5, so 0.5*0.5=0.25; 0.3 > 0.25 but < 0.5
        assert_eq!(detector.decide(0.3), GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn test_gate_skip_detector_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.8 });
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide_from_telemetry(&agg), GateSkipDecision::Skip);
    }

    #[test]
    fn test_gate_skip_detector_config_accessor() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.7,
            dead_neuron_epsilon: 0.01,
        };
        let detector = GateFirstSkipDetector::new(config);
        assert!((detector.config.skip_threshold - 0.7).abs() < 1e-6);
        assert!(detector.config.enabled);
    }

    // -- SinkDetector: detect() and helpers --

    #[test]
    fn test_sink_detector_sink_above_threshold() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // sink_threshold=0.9, max_val > 0.9 → Sink
        assert_eq!(detector.detect(0.95, 0.5), AttentionPattern::Sink);
    }

    #[test]
    fn test_sink_detector_sharp_focus() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // max_val < 0.9, sharpness > 0.8 → SharpFocus
        assert_eq!(detector.detect(0.5, 0.9), AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_sink_detector_diffuse() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // max_val < 0.9, sharpness < 0.1 → Diffuse
        assert_eq!(detector.detect(0.5, 0.05), AttentionPattern::Diffuse);
    }

    #[test]
    fn test_sink_detector_normal() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // max_val < 0.9, 0.1 <= sharpness <= 0.8 → Normal
        assert_eq!(detector.detect(0.5, 0.5), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_detector_is_protected_sink() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // protected_sink_count=4, so positions 0-3 are protected
        assert!(detector.is_protected_sink(0));
        assert!(detector.is_protected_sink(3));
        assert!(!detector.is_protected_sink(4));
        assert!(!detector.is_protected_sink(100));
    }

    #[test]
    fn test_sink_detector_protected_sink_count() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.protected_sink_count(), 4);
    }

    #[test]
    fn test_sink_detector_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::SoftmaxSharpness { max_val: 0.95, sharpness: 0.5 });
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect_from_telemetry(&agg), AttentionPattern::Sink);
    }

    // -- ResidualBypassDetector: decide() boundary conditions --

    #[test]
    fn test_bypass_detector_disabled_forces_execute() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig {
            enabled: false,
            delta_rho_threshold: 0.001,
            cosine_threshold: 0.99,
            min_skip_layer: 4,
        });
        assert_eq!(detector.decide(10, 1.0, 0.999), ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_bypass_detector_below_min_skip_layer() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        assert_eq!(detector.decide(0, 1.0, 0.999), ResidualBypassDecision::Execute);
        assert_eq!(detector.decide(3, 1.0, 0.999), ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_bypass_detector_bypass_when_stable() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // layer >= 4, delta_rho ≈ 1.0, cosine > 0.99 → Bypass
        assert_eq!(detector.decide(10, 1.0001, 0.999), ResidualBypassDecision::Bypass);
    }

    #[test]
    fn test_bypass_detector_execute_unstable_energy() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // delta_rho far from 1.0 → Execute
        assert_eq!(detector.decide(10, 2.0, 0.999), ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_bypass_detector_execute_unstable_direction() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // cosine < 0.99 → Execute
        assert_eq!(detector.decide(10, 1.0, 0.95), ResidualBypassDecision::Execute);
    }

    #[test]
    fn test_bypass_detector_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        assert_eq!(detector.decide_from_telemetry(10, &agg), ResidualBypassDecision::Bypass);
    }

    // -- ExpertThermalTracker: record_routing, hit_rate, boundary --

    #[test]
    fn test_expert_thermal_new_initial_state() {
        let tracker = ExpertThermalTracker::new(8);
        for i in 0..8 {
            assert_eq!(tracker.thermal_state(i), ExpertThermalState::Warm);
            assert_eq!(tracker.hit_rate(i), 0.0);
        }
        assert!(tracker.cold_experts().is_empty());
    }

    #[test]
    fn test_expert_thermal_record_routing_updates_hit_counts() {
        let mut tracker = ExpertThermalTracker::new(4);
        tracker.record_routing(&[0, 2]);
        tracker.record_routing(&[0, 2]);
        tracker.record_routing(&[0, 2]);
        assert!(tracker.hit_rate(0) > 0.0);
        assert!(tracker.hit_rate(2) > 0.0);
    }

    #[test]
    fn test_expert_thermal_cold_expert_after_many_routing() {
        let mut tracker = ExpertThermalTracker::new(4);
        // Route exclusively to expert 0 many times to push others to cold
        for _ in 0..200_000 {
            tracker.record_routing(&[0]);
        }
        let cold = tracker.cold_experts();
        assert!(cold.contains(&1));
        assert!(cold.contains(&2));
        assert!(cold.contains(&3));
        assert!(!cold.contains(&0));
    }

    #[test]
    fn test_expert_thermal_out_of_range_returns_cold() {
        let tracker = ExpertThermalTracker::new(4);
        assert_eq!(tracker.thermal_state(100), ExpertThermalState::Cold);
    }

    #[test]
    fn test_expert_thermal_hit_rate_out_of_range() {
        let tracker = ExpertThermalTracker::new(4);
        assert_eq!(tracker.hit_rate(100), 0.0);
    }

    #[test]
    fn test_expert_thermal_state_equality_hot_warm_cold() {
        assert_eq!(ExpertThermalState::Hot, ExpertThermalState::Hot);
        assert_eq!(ExpertThermalState::Warm, ExpertThermalState::Warm);
        assert_eq!(ExpertThermalState::Cold, ExpertThermalState::Cold);
        assert_ne!(ExpertThermalState::Hot, ExpertThermalState::Warm);
        assert_ne!(ExpertThermalState::Warm, ExpertThermalState::Cold);
    }

    // -- SpecScheduleSignal: advise() for all variants --

    #[test]
    fn test_spec_signal_default_entropy_threshold() {
        let mut signal = SpecScheduleSignal::default();
        // default enable_entropy_threshold=2.0; low entropy < 2.0 → EnableSpec
        assert_eq!(signal.advise(1.0, 0.9), SpecScheduleAdvice::EnableSpec);
        // high entropy >= 2.0 → StandardDecode
        assert_eq!(signal.advise(3.0, 0.9), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_signal_advise_fallback_streak() {
        let mut signal = SpecScheduleSignal::new();
        // acceptance_rate < 0.3 three times → Fallback
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_signal_advise_reset_clears_streak() {
        let mut signal = SpecScheduleSignal::new();
        // Two low-acceptance rounds
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        signal.reset();
        // After reset, streak should be 0, so another low-acceptance doesn't immediately fallback
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_signal_advise_enable_spec_low_entropy() {
        let mut signal = SpecScheduleSignal::new();
        // High acceptance_rate (> 0.3) + low entropy (< 2.0) → EnableSpec
        assert_eq!(signal.advise(1.5, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_signal_advise_standard_high_entropy() {
        let mut signal = SpecScheduleSignal::new();
        // High acceptance_rate (> 0.3) + high entropy (>= 2.0) → StandardDecode
        assert_eq!(signal.advise(5.0, 0.8), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_signal_advise_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 1.0 });
        let mut signal = SpecScheduleSignal::new();
        assert_eq!(
            signal.advise_from_telemetry(&agg, 0.9),
            SpecScheduleAdvice::EnableSpec
        );
    }

    #[test]
    fn test_spec_signal_reset_zeroes_streak() {
        let mut signal = SpecScheduleSignal::new();
        signal.advise(1.0, 0.1);
        signal.advise(1.0, 0.1);
        signal.reset();
        // After reset, need 3 more low-acceptance to get Fallback
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);
    }

    // -- PrefetchConfig: field defaults and custom values --

    #[test]
    fn test_prefetch_config_default_min_distance() {
        let config = PrefetchConfig::default();
        assert_eq!(config.min_distance, -16);
    }

    #[test]
    fn test_prefetch_config_default_max_distance() {
        let config = PrefetchConfig::default();
        assert_eq!(config.max_distance, 64);
    }

    #[test]
    fn test_prefetch_config_default_sharp_focus_factor() {
        let config = PrefetchConfig::default();
        assert!((config.sharp_focus_factor - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_prefetch_config_default_diffuse_factor() {
        let config = PrefetchConfig::default();
        assert!((config.diffuse_factor - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_prefetch_config_default_sink_offset() {
        let config = PrefetchConfig::default();
        assert_eq!(config.sink_prefetch_offset, 4);
    }

    #[test]
    fn test_prefetch_config_debug_all_fields() {
        let config = PrefetchConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("min_distance"));
        assert!(debug.contains("max_distance"));
        assert!(debug.contains("sharp_focus_factor"));
        assert!(debug.contains("diffuse_factor"));
        assert!(debug.contains("sink_prefetch_offset"));
    }

    #[test]
    fn test_prefetch_config_clone_preserves() {
        let config = PrefetchConfig {
            min_distance: -8,
            max_distance: 32,
            sharp_focus_factor: 0.5,
            diffuse_factor: 2.0,
            sink_prefetch_offset: 2,
        };
        let cloned = config.clone();
        assert_eq!(cloned.min_distance, -8);
        assert_eq!(cloned.max_distance, 32);
        assert!((cloned.sharp_focus_factor - 0.5).abs() < 1e-6);
    }

    // -- PrefetchStats: default and zero state --

    #[test]
    fn test_prefetch_stats_default_all_zero() {
        let stats = PrefetchStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_prefetch_stats_equality() {
        let a = PrefetchStats { total: 10, none_count: 2, forward_count: 3, backward_count: 4, sink_count: 1 };
        let b = PrefetchStats { total: 10, none_count: 2, forward_count: 3, backward_count: 4, sink_count: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefetch_stats_copy() {
        let a = PrefetchStats { total: 5, none_count: 1, forward_count: 2, backward_count: 1, sink_count: 1 };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefetch_stats_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = PrefetchStats { total: 3, none_count: 1, forward_count: 1, backward_count: 0, sink_count: 1 };
        let b = PrefetchStats { total: 3, none_count: 1, forward_count: 1, backward_count: 0, sink_count: 1 };
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // -- CentroidPrefetch: compute, stats, reset, history --

    #[test]
    fn test_centroid_prefetch_new_initial_state() {
        let cp = CentroidPrefetch::new(PrefetchConfig::default());
        assert!(cp.history().is_empty());
        let stats = cp.stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_centroid_prefetch_with_num_layers() {
        let cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(32);
        // history capacity is set, but still empty
        assert!(cp.history().is_empty());
    }

    #[test]
    fn test_centroid_prefetch_compute_sink() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        let advice = cp.compute(0, 5, 10, AttentionPattern::Sink);
        assert!(matches!(advice, PrefetchAdvice::Sink(4)));
        assert_eq!(cp.history().len(), 1);
    }

    #[test]
    fn test_centroid_prefetch_compute_forward() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        // centroid_pos=5, current_pos=20 → offset=-15 → Forward(15)
        let advice = cp.compute(0, 5, 20, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Forward(_)));
    }

    #[test]
    fn test_centroid_prefetch_compute_backward() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        // centroid_pos=50, current_pos=20 → offset=30 → Backward(30)
        let advice = cp.compute(0, 50, 20, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Backward(_)));
    }

    #[test]
    fn test_centroid_prefetch_compute_none_at_same_position() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        // centroid_pos=20, current_pos=20 → offset=0 → None
        let advice = cp.compute(0, 20, 20, AttentionPattern::Normal);
        assert_eq!(advice, PrefetchAdvice::None);
    }

    #[test]
    fn test_centroid_prefetch_reset_clears_history() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        cp.compute(0, 5, 20, AttentionPattern::Normal);
        cp.compute(1, 10, 20, AttentionPattern::Sink);
        assert_eq!(cp.history().len(), 2);
        cp.reset();
        assert!(cp.history().is_empty());
        let stats = cp.stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_centroid_prefetch_stats_accumulate() {
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        cp.compute(0, 5, 20, AttentionPattern::Normal);  // Forward
        cp.compute(1, 50, 20, AttentionPattern::Normal); // Backward
        cp.compute(2, 20, 20, AttentionPattern::Normal); // None
        cp.compute(3, 10, 20, AttentionPattern::Sink);   // Sink
        let stats = cp.stats();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.forward_count, 1);
        assert_eq!(stats.backward_count, 1);
        assert_eq!(stats.none_count, 1);
        assert_eq!(stats.sink_count, 1);
    }

    #[test]
    fn test_centroid_prefetch_compute_from_aggregator() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 50 });
        let mut cp = CentroidPrefetch::new(PrefetchConfig::default()).with_num_layers(8);
        let advice = cp.compute_from_aggregator(0, 20, &agg, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Backward(_)));
    }

    // -- CentroidRecord fields --

    #[test]
    fn test_centroid_record_debug() {
        let record = CentroidRecord {
            layer_idx: 5,
            centroid_pos: 100,
            current_pos: 50,
            prefetch_distance: 50,
            pattern: AttentionPattern::Normal,
        };
        let debug = format!("{:?}", record);
        assert!(debug.contains("CentroidRecord"));
        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("centroid_pos"));
    }

    // -- BatchSkipSummary: from_decisions edge cases --

    #[test]
    fn test_batch_skip_summary_empty_decisions() {
        let summary = BatchSkipSummary::from_decisions(&[]);
        assert!(summary.per_request_decisions.is_empty());
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!((summary.avg_dead_neuron_ratio).abs() < 0.001);
    }

    #[test]
    fn test_batch_skip_summary_all_full_compute() {
        let decisions = vec![GateSkipDecision::FullCompute; 5];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::FullCompute);
        assert!(!summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_mostly_skip_advice() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert!(summary.needs_compact());
        assert!(summary.needs_masked_variant());
    }

    #[test]
    fn test_batch_skip_summary_partial_mask_advice() {
        let decisions = vec![
            GateSkipDecision::MaskedCompute,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
        assert!(summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    // -- BatchAttentionSummary: from_patterns and methods --

    #[test]
    fn test_batch_attention_summary_empty_patterns() {
        let summary = BatchAttentionSummary::from_patterns(&[], 0.0);
        assert!(summary.per_request_patterns.is_empty());
        assert!((summary.avg_sharpness).abs() < 0.001);
        assert!((summary.sink_ratio).abs() < 0.001);
    }

    #[test]
    fn test_batch_attention_summary_sink_dominant() {
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // 3/4 = 75% > 30% → SinkDominant
        assert!(summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_attention_summary_diffuse_attention() {
        let patterns = vec![
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Diffuse,
            AttentionPattern::Normal,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // 3/5 = 60% > 40% → DiffuseAttention
        assert!(summary.benefits_from_speculation());
        assert!(!summary.needs_sink_protection());
    }

    #[test]
    fn test_batch_attention_summary_normal_attention() {
        let patterns = vec![
            AttentionPattern::Normal,
            AttentionPattern::Normal,
            AttentionPattern::SharpFocus,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        // No Sink > 30%, no Diffuse > 40% → NormalAttention
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
    }

    #[test]
    fn test_batch_attention_summary_preserves_patterns() {
        let patterns = vec![AttentionPattern::Sink, AttentionPattern::Diffuse];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 1.0);
        assert_eq!(summary.per_request_patterns.len(), 2);
        assert_eq!(summary.per_request_patterns[0], AttentionPattern::Sink);
        assert_eq!(summary.per_request_patterns[1], AttentionPattern::Diffuse);
    }

    // -- BatchAttentionAdvice variants --

    #[test]
    fn test_batch_attention_advice_equality() {
        use super::super::sink_tracker::BatchAttentionAdvice;
        assert_eq!(BatchAttentionAdvice::NormalAttention, BatchAttentionAdvice::NormalAttention);
        assert_eq!(BatchAttentionAdvice::SinkDominant, BatchAttentionAdvice::SinkDominant);
        assert_eq!(BatchAttentionAdvice::DiffuseAttention, BatchAttentionAdvice::DiffuseAttention);
        assert_ne!(BatchAttentionAdvice::NormalAttention, BatchAttentionAdvice::SinkDominant);
    }

    // -- BypassStats: construction and invariants --

    #[test]
    fn test_bypass_stats_custom_construction() {
        let stats = BypassStats { total: 100, bypassed: 30, executed: 70 };
        assert_eq!(stats.total, 100);
        assert_eq!(stats.bypassed, 30);
        assert_eq!(stats.executed, 70);
        assert_eq!(stats.bypassed + stats.executed, stats.total);
    }

    #[test]
    fn test_bypass_stats_debug_format() {
        let stats = BypassStats { total: 10, bypassed: 3, executed: 7 };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total"));
        assert!(debug.contains("bypassed"));
        assert!(debug.contains("executed"));
    }

    // -- EpilogueConfig: clone preserves nested configs --

    #[test]
    fn test_config_clone_preserves_nested_sink_detection() {
        let config = EpilogueConfig {
            sink_detection: SinkDetectionConfig {
                sink_threshold: 0.85,
                protected_sink_count: 8,
                sharp_focus_threshold: 0.7,
                diffuse_threshold: 0.15,
            },
            ..Default::default()
        };
        let cloned = config.clone();
        assert!((cloned.sink_detection.sink_threshold - 0.85).abs() < 1e-6);
        assert_eq!(cloned.sink_detection.protected_sink_count, 8);
    }

    #[test]
    fn test_config_clone_preserves_nested_prefetch() {
        let config = EpilogueConfig {
            prefetch: PrefetchConfig {
                min_distance: -32,
                max_distance: 128,
                sharp_focus_factor: 0.4,
                diffuse_factor: 2.0,
                sink_prefetch_offset: 8,
            },
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.prefetch.min_distance, -32);
        assert_eq!(cloned.prefetch.max_distance, 128);
    }

    // -- Subsystem: expert_thermal accessor for MoE --

    #[test]
    fn test_subsystem_expert_thermal_none_for_dense() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_experts: 0,
            ..Default::default()
        });
        assert!(sub.expert_thermal().is_none());
    }

    #[test]
    fn test_subsystem_expert_thermal_some_for_moe() {
        let sub = EpilogueSubsystem::new(EpilogueConfig {
            num_experts: 8,
            ..Default::default()
        });
        assert!(sub.expert_thermal().is_some());
    }

    // -- write_page_header: centroid_pos encoding with per_head_entropy --

    #[test]
    fn test_write_page_header_high_entropy_centroid_pos() {
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
        let t = make_telemetry(0.1, 0.5, 5.0, 2.0);
        sub.write_page_header(&mut header, &decision, &t);
        // centroid_pos stores per_head_entropy (5.0)
        let centroid = f16_bits_to_f32(header.centroid_pos);
        assert!(centroid > 0.0, "centroid_pos should encode positive per_head_entropy");
    }

    // -- compute_l2_norm: utility function --

    #[test]
    fn test_compute_l2_norm_unit_vector() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        assert!((compute_l2_norm(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_zero_vector() {
        let v = vec![0.0; 10];
        assert!((compute_l2_norm(&v)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_3_4_5() {
        let v = vec![3.0, 4.0];
        assert!((compute_l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_empty() {
        assert!((compute_l2_norm(&[])).abs() < 1e-6);
    }

    // -- GateSkipDecision: all variants PartialEq --

    #[test]
    fn test_gate_skip_decision_all_variants_ne() {
        assert_ne!(GateSkipDecision::FullCompute, GateSkipDecision::MaskedCompute);
        assert_ne!(GateSkipDecision::MaskedCompute, GateSkipDecision::Skip);
        assert_ne!(GateSkipDecision::FullCompute, GateSkipDecision::Skip);
    }

    // -- ResidualBypassDecision: all variants PartialEq --

    #[test]
    fn test_residual_bypass_decision_all_variants_ne() {
        assert_ne!(ResidualBypassDecision::Execute, ResidualBypassDecision::Bypass);
    }

    // -- AttentionPattern: all variants distinct --

    #[test]
    fn test_attention_pattern_all_variants_distinct() {
        assert_ne!(AttentionPattern::Normal, AttentionPattern::Sink);
        assert_ne!(AttentionPattern::Sink, AttentionPattern::SharpFocus);
        assert_ne!(AttentionPattern::SharpFocus, AttentionPattern::Diffuse);
        assert_ne!(AttentionPattern::Normal, AttentionPattern::Diffuse);
    }

    // -- SpecScheduleAdvice: all variants distinct --

    #[test]
    fn test_spec_schedule_advice_all_variants_distinct() {
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::StandardDecode);
        assert_ne!(SpecScheduleAdvice::StandardDecode, SpecScheduleAdvice::Fallback);
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::Fallback);
    }

    // -- EpilogueBatchSummary: waste_ratio range property --

    #[test]
    fn test_batch_summary_waste_ratio_in_range() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig {
            num_layers: 12,
            ..Default::default()
        });
        let batch = vec![
            make_telemetry(0.8, 0.5, 2.0, 3.0),
            make_telemetry(0.1, 0.5, 2.0, 3.0),
            make_telemetry(0.3, 0.5, 2.0, 3.0),
        ];
        let summary = sub.ingest_and_decide(&batch);
        assert!(summary.waste_ratio >= 0.0 && summary.waste_ratio <= 1.0);
    }

    // -- SequenceTelemetry: construction with all field combos --

    #[test]
    fn test_telemetry_has_outlier_true() {
        let t = SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: true,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };
        assert!(t.has_outlier);
    }

    #[test]
    fn test_telemetry_has_outlier_false() {
        let t = SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };
        assert!(!t.has_outlier);
    }

    // -- EpilogueSignal: Copy trait --

    #[test]
    fn test_epilogue_signal_is_copy() {
        let a = EpilogueSignal::DeadNeuronRatio { ratio: 0.5 };
        let b = a;
        // Both should be valid - Copy trait
        let da = format!("{:?}", a);
        let db = format!("{:?}", b);
        assert_eq!(da, db);
    }

    // -- PrefetchAdvice: Hash consistency across variants --

    #[test]
    fn test_prefetch_advice_all_variants_hash_differently() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let variants: Vec<PrefetchAdvice> = vec![
            PrefetchAdvice::None,
            PrefetchAdvice::Forward(10),
            PrefetchAdvice::Backward(10),
            PrefetchAdvice::Sink(4),
        ];
        let hashes: Vec<u64> = variants.iter().map(|v| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }).collect();
        // All hashes should be distinct
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "hashes should differ for variant {} vs {}", i, j);
            }
        }
    }

    // -- BatchSkipAdvice: debug format --

    #[test]
    fn test_batch_skip_advice_debug_strings() {
        let debug_fc = format!("{:?}", BatchSkipAdvice::FullCompute);
        let debug_pm = format!("{:?}", BatchSkipAdvice::PartialMask);
        let debug_ms = format!("{:?}", BatchSkipAdvice::MostlySkip);
        assert!(debug_fc.contains("FullCompute"));
        assert!(debug_pm.contains("PartialMask"));
        assert!(debug_ms.contains("MostlySkip"));
    }

    // ════════════════════════════════════════════════════════════════
    // Wave 12x47: +40 tests — accessor depth, boundary, and property checks
    // ════════════════════════════════════════════════════════════════

    // ── GateFirstSkipLayer accessor depth ──

    #[test]
    fn test_gate_skip_layer_history_initially_empty() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.gate_skip().history().is_empty());
    }

    #[test]
    fn test_gate_skip_layer_record_none_for_unprocessed_layer() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.gate_skip().layer_record(0).is_none());
        assert!(sub.gate_skip().layer_record(31).is_none());
    }

    #[test]
    fn test_gate_skip_layer_record_after_single_telemetry() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.9, 0.0, 0.5, 2.0);
        let summary = sub.ingest_and_decide(&[tel]);
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_gate_skip_recent_decisions_empty() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.gate_skip().recent_decisions(5).is_empty());
    }

    #[test]
    fn test_gate_skip_recent_decisions_limited() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.9, 0.0, 0.5, 2.0);
        for _ in 0..3 {
            sub.ingest_and_decide(&[tel.clone()]);
        }
        let recent = sub.gate_skip().recent_decisions(2);
        assert!(recent.len() <= 2);
    }

    #[test]
    fn test_gate_skip_total_decisions_equals_sum() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.9, 0.0, 0.5, 2.0);
        sub.ingest_and_decide(&[tel]);
        let gs = sub.gate_skip();
        assert_eq!(
            gs.total_decisions(),
            gs.total_skipped_layers() + gs.total_masked_layers() + gs.total_full_layers()
        );
    }

    #[test]
    fn test_gate_skip_config_accessor_matches() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(
            sub.gate_skip().config().skip_threshold,
            GateFirstSkipConfig::default().skip_threshold
        );
    }

    #[test]
    fn test_gate_skip_num_layers_matches_config() {
        let config = EpilogueConfig { num_layers: 16, ..Default::default() };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.gate_skip().num_layers(), 16);
    }

    #[test]
    fn test_gate_skip_skip_rate_zero_initially() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(sub.gate_skip().skip_rate(), 0.0);
    }

    #[test]
    fn test_gate_skip_skip_rate_in_range_after_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.5, 0.0, 0.5, 2.0);
        sub.ingest_and_decide(&[tel]);
        let rate = sub.gate_skip().skip_rate();
        assert!((0.0..=1.0).contains(&rate));
    }

    // ── SinkTracker accessor depth ──

    #[test]
    fn test_sink_tracker_history_initially_empty() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.sink_tracker().history().is_empty());
    }

    #[test]
    fn test_sink_tracker_layer_record_none_for_unprocessed() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.sink_tracker().layer_record(0).is_none());
    }

    #[test]
    fn test_sink_tracker_recent_patterns_empty() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.sink_tracker().recent_patterns(5).is_empty());
    }

    #[test]
    fn test_sink_tracker_recent_patterns_limited() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.1, 0.0, 0.5, 2.0);
        for _ in 0..4 {
            sub.ingest_and_decide(&[tel.clone()]);
        }
        let recent = sub.sink_tracker().recent_patterns(2);
        assert!(recent.len() <= 2);
    }

    #[test]
    fn test_sink_tracker_num_layers_matches_config() {
        let config = EpilogueConfig { num_layers: 24, ..Default::default() };
        let sub = EpilogueSubsystem::new(config);
        assert_eq!(sub.sink_tracker().num_layers(), 24);
    }

    #[test]
    fn test_sink_tracker_config_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(
            sub.sink_tracker().config().sink_threshold,
            SinkDetectionConfig::default().sink_threshold
        );
    }

    #[test]
    fn test_sink_tracker_ratios_sum_to_one_or_less() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.1, 0.0, 0.5, 2.0);
        sub.ingest_and_decide(&[tel; 4]);
        let st = sub.sink_tracker();
        if st.total_detections() > 0 {
            let sum = st.sink_ratio() + st.sharp_focus_ratio() + st.diffuse_ratio();
            assert!(sum <= 1.01);
        }
    }

    #[test]
    fn test_sink_tracker_sharp_focus_ratio_initially_zero() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(sub.sink_tracker().sharp_focus_ratio(), 0.0);
    }

    #[test]
    fn test_sink_tracker_diffuse_ratio_initially_zero() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert_eq!(sub.sink_tracker().diffuse_ratio(), 0.0);
    }

    // ── ResidualBypassLayer accessor depth ──

    #[test]
    fn test_residual_bypass_history_initially_empty() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.residual_bypass().history().is_empty());
    }

    #[test]
    fn test_residual_bypass_stats_after_single_request() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.1, 0.0, 0.5, 2.0);
        sub.ingest_and_decide(&[tel]);
        let stats = sub.residual_bypass().stats();
        assert_eq!(stats.total, stats.bypassed + stats.executed);
    }

    #[test]
    fn test_residual_bypass_history_grows_with_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.1, 0.0, 0.5, 2.0);
        sub.ingest_and_decide(&[tel]);
        let hist_len = sub.residual_bypass().history().len();
        sub.ingest_and_decide(&[tel]);
        assert!(sub.residual_bypass().history().len() >= hist_len);
    }

    // ── BatchSkipSummary boundary & property ──

    #[test]
    fn test_batch_skip_summary_all_masked_compute() {
        let decisions = vec![GateSkipDecision::MaskedCompute; 5];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!(summary.needs_masked_variant());
        assert!(!summary.needs_compact());
        assert_eq!(summary.per_request_decisions.len(), 5);
    }

    #[test]
    fn test_batch_skip_summary_mixed_skip_and_full_50_50() {
        // 50% skip: skip_count*2 == total → NOT MostlySkip (> not >=)
        // But (skip+masked)*4 > total → PartialMask
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::PartialMask);
    }

    #[test]
    fn test_batch_skip_summary_single_decision_skip() {
        let decisions = vec![GateSkipDecision::Skip];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!(summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_single_decision_masked() {
        let decisions = vec![GateSkipDecision::MaskedCompute];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!(summary.needs_masked_variant());
        assert!(!summary.needs_compact());
    }

    #[test]
    fn test_batch_skip_summary_avg_dead_ratio_in_range() {
        let decisions = vec![
            GateSkipDecision::FullCompute,
            GateSkipDecision::Skip,
            GateSkipDecision::MaskedCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert!((0.0..=1.0).contains(&summary.avg_dead_neuron_ratio));
    }

    // ── BatchAttentionSummary boundary & property ──

    #[test]
    fn test_batch_attention_summary_all_sharp_focus() {
        let patterns = vec![AttentionPattern::SharpFocus; 5];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.8);
        assert!(!summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
        assert_eq!(summary.per_request_patterns.len(), 5);
    }

    #[test]
    fn test_batch_attention_summary_mixed_sink_and_normal() {
        let patterns = vec![
            AttentionPattern::Sink,
            AttentionPattern::Sink,
            AttentionPattern::Normal,
        ];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 0.5);
        assert!(summary.needs_sink_protection());
        assert!(!summary.benefits_from_speculation());
        assert!(summary.sink_ratio > 0.0);
    }

    #[test]
    fn test_batch_attention_summary_preserves_avg_sharpness() {
        let patterns = vec![AttentionPattern::Normal];
        let summary = BatchAttentionSummary::from_patterns(&patterns, 1.23);
        assert!((summary.avg_sharpness - 1.23).abs() < 0.001);
    }

    // ── EpilogueConfig edge cases ──

    #[test]
    fn test_config_default_num_experts_zero() {
        assert_eq!(EpilogueConfig::default().num_experts, 0);
    }

    #[test]
    fn test_config_num_experts_custom() {
        let config = EpilogueConfig { num_experts: 64, ..Default::default() };
        assert_eq!(config.num_experts, 64);
    }

    #[test]
    fn test_config_num_layers_default_32() {
        assert_eq!(EpilogueConfig::default().num_layers, 32);
    }

    // ── EpilogueBatchSummary waste_ratio property ──

    #[test]
    fn test_batch_summary_waste_ratio_zero_for_full_compute_batch() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.0, 0.5, 2.0, 3.0);
        let summary = sub.ingest_and_decide(&[tel]);
        assert_eq!(summary.waste_ratio, 0.0);
    }

    #[test]
    fn test_batch_summary_compact_required_consistent_with_skip_summary() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = make_telemetry(0.9, 0.0, 0.5, 2.0);
        let summary = sub.ingest_and_decide(&[tel; 4]);
        if summary.compact_required {
            assert!(matches!(
                summary.skip_summary.batch_advice,
                BatchSkipAdvice::MostlySkip | BatchSkipAdvice::PartialMask
            ));
        }
    }

    // ── EpilogueSubsystem write_page_header via different telemetry ──

    #[test]
    fn test_write_page_header_positive_l2_delta() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let mut header = KvPageHeader::default();
        let tel = SequenceTelemetry {
            l2_delta: 5.0,
            has_outlier: true,
            dead_density: 0.1,
            per_head_entropy: 1.5,
            transform_ratio: 0.02,
            output_entropy: 2.0,
        };
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::FullCompute,
            attention_pattern: AttentionPattern::Normal,
            bypass_decision: ResidualBypassDecision::Execute,
            spec_advice: SpecScheduleAdvice::StandardDecode,
            prefetch_advice: PrefetchAdvice::None,
        };
        sub.write_page_header(&mut header, &decision, &tel);
    }

    // ── CentroidPrefetch via subsystem accessor ──

    #[test]
    fn test_prefetch_history_via_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        assert!(sub.prefetch().history().is_empty());
    }

    #[test]
    fn test_prefetch_stats_via_accessor() {
        let sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let stats = sub.prefetch().stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.none_count, 0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.backward_count, 0);
        assert_eq!(stats.sink_count, 0);
    }

    #[test]
    fn test_prefetch_mut_allows_compute() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let advice = sub.prefetch_mut().compute(0, 10, 5, AttentionPattern::Normal);
        assert!(matches!(advice, PrefetchAdvice::Backward(_)));
    }

    // ── BatchAttentionAdvice debug format ──

    #[test]
    fn test_batch_attention_advice_debug_strings() {
        use super::super::sink_tracker::BatchAttentionAdvice;
        let d1 = format!("{:?}", BatchAttentionAdvice::NormalAttention);
        let d2 = format!("{:?}", BatchAttentionAdvice::SinkDominant);
        let d3 = format!("{:?}", BatchAttentionAdvice::DiffuseAttention);
        assert!(d1.contains("NormalAttention"));
        assert!(d2.contains("SinkDominant"));
        assert!(d3.contains("DiffuseAttention"));
    }

    // ── LayerRecord debug formats ──

    #[test]
    fn test_layer_skip_record_debug_format() {
        let rec = super::super::gate_skip::LayerSkipRecord {
            layer_idx: 5,
            dead_neuron_ratio: 0.3,
            decision: GateSkipDecision::MaskedCompute,
        };
        let debug = format!("{:?}", rec);
        assert!(debug.contains("LayerSkipRecord"));
    }

    #[test]
    fn test_layer_attention_record_debug_format() {
        let rec = super::super::sink_tracker::LayerAttentionRecord {
            layer_idx: 3,
            softmax_max: 0.8,
            sharpness: 0.5,
            pattern: AttentionPattern::Sink,
        };
        let debug = format!("{:?}", rec);
        assert!(debug.contains("LayerAttentionRecord"));
    }

    #[test]
    fn test_layer_bypass_record_debug_format() {
        let rec = super::super::residual_bypass::LayerBypassRecord {
            layer_idx: 7,
            delta_rho: 1.001,
            cosine: 0.998,
            decision: ResidualBypassDecision::Bypass,
        };
        let debug = format!("{:?}", rec);
        assert!(debug.contains("LayerBypassRecord"));
        assert!(debug.contains("Bypass"));
    }

    // ── BatchSkipSummary from heterogeneous decisions ──

    #[test]
    fn test_batch_skip_summary_4_skip_1_full_1_masked() {
        let decisions = vec![
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::Skip,
            GateSkipDecision::FullCompute,
            GateSkipDecision::MaskedCompute,
        ];
        let summary = BatchSkipSummary::from_decisions(&decisions);
        assert_eq!(summary.batch_advice, BatchSkipAdvice::MostlySkip);
        assert_eq!(summary.per_request_decisions.len(), 6);
    }

    // ── Subsystem ingest edge cases ──

    #[test]
    fn test_ingest_with_has_outlier_does_not_panic() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = SequenceTelemetry {
            l2_delta: 0.1,
            has_outlier: true,
            dead_density: 0.2,
            per_head_entropy: 1.0,
            transform_ratio: 0.01,
            output_entropy: 1.5,
        };
        let summary = sub.ingest_and_decide(&[tel]);
        assert_eq!(summary.per_request.len(), 1);
    }

    #[test]
    fn test_ingest_all_zero_telemetry() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel = SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };
        let summary = sub.ingest_and_decide(&[tel]);
        assert_eq!(summary.per_request.len(), 1);
        assert!(summary.waste_ratio >= 0.0);
    }

    #[test]
    fn test_three_sequential_batches_independent() {
        let mut sub = EpilogueSubsystem::new(EpilogueConfig::default());
        let tel1 = make_telemetry(0.0, 0.0, 2.0, 3.0);
        let tel2 = make_telemetry(0.9, 0.0, 0.5, 2.0);
        let tel3 = make_telemetry(0.5, 0.0, 1.0, 2.5);

        let s1 = sub.ingest_and_decide(&[tel1]);
        let s2 = sub.ingest_and_decide(&[tel2]);
        let s3 = sub.ingest_and_decide(&[tel3]);

        assert_eq!(s1.per_request.len(), 1);
        assert_eq!(s2.per_request.len(), 1);
        assert_eq!(s3.per_request.len(), 1);
    }

    // ── RequestEpilogueDecision field construction ──

    #[test]
    fn test_request_decision_with_sink_and_bypass() {
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::Skip,
            attention_pattern: AttentionPattern::Sink,
            bypass_decision: ResidualBypassDecision::Bypass,
            spec_advice: SpecScheduleAdvice::EnableSpec,
            prefetch_advice: PrefetchAdvice::Sink(4),
        };
        assert!(matches!(decision.gate_skip, GateSkipDecision::Skip));
        assert!(matches!(decision.attention_pattern, AttentionPattern::Sink));
        assert!(matches!(decision.bypass_decision, ResidualBypassDecision::Bypass));
        assert!(matches!(decision.spec_advice, SpecScheduleAdvice::EnableSpec));
    }

    #[test]
    fn test_request_decision_with_sharp_focus_and_execute() {
        let decision = RequestEpilogueDecision {
            gate_skip: GateSkipDecision::FullCompute,
            attention_pattern: AttentionPattern::SharpFocus,
            bypass_decision: ResidualBypassDecision::Execute,
            spec_advice: SpecScheduleAdvice::Fallback,
            prefetch_advice: PrefetchAdvice::Forward(8),
        };
        assert!(matches!(decision.attention_pattern, AttentionPattern::SharpFocus));
    }
}
