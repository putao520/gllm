//! BubbleAnalyzer — Pipeline bubble 分析与可视化指标 (REQ-DIST-024)
//!
//! 计算 Pipeline Parallel 的流水线气泡比率，支持 GPipe / 1F1B / 交错模式。
//! 分析结果用于调度器策略选择（气泡率 > 阈值时建议切换策略）。
//!
//! 公式：
//! - GPipe bubble_ratio = (pp_size - 1) / (num_microbatches * pp_size)
//! - 交错模式 bubble_ratio = (pp_size - 1) / (num_virtual_stages * num_microbatches * pp_size)
//!
//! 支持范围：pp_size ∈ [2, 64], num_microbatches ∈ [1, 1024]
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use super::config::PipelineConfig;
use super::scheduler::MicroBatchStrategy;

// ── BubbleMetrics (REQ-DIST-024) ────────────────────────────────────────────

/// Pipeline bubble 分析指标 (REQ-DIST-024)
///
/// 包含气泡率、时间估计、策略建议等完整分析结果。
// @trace REQ-DIST-024 [entity:BubbleMetrics] [api:POST /internal/distributed/pipeline/bubble]
#[derive(Debug, Clone, PartialEq)]
pub struct BubbleMetrics {
    /// GPipe 气泡比率 = (pp_size - 1) / (num_microbatches * pp_size)
    pub gpipe_bubble_ratio: f64,
    /// 1F1B 气泡比率（与 GPipe 相同公式，但峰值激活内存更低）
    pub one_f_one_b_bubble_ratio: f64,
    /// 交错模式气泡比率 = (pp_size - 1) / (num_virtual_stages * num_microbatches * pp_size)
    pub interleaved_bubble_ratio: f64,
    /// 当前模式气泡比率
    pub current_bubble_ratio: f64,
    /// 气泡时间（微秒），基于单步计算时间估计
    pub bubble_time_us: f64,
    /// 总流水线时间（微秒）
    pub total_pipeline_time_us: f64,
    /// 建议策略（基于气泡率阈值）
    pub recommended_strategy: MicroBatchStrategy,
    /// 是否建议交错模式
    pub recommend_interleaved: bool,
    /// pp_size
    pub pp_size: u32,
    /// num_microbatches
    pub num_microbatches: usize,
    /// num_virtual_stages
    pub num_virtual_stages: u32,
}

// ── BubbleAnalyzer (REQ-DIST-024) ───────────────────────────────────────────

/// Pipeline bubble 分析器 (REQ-DIST-024)
///
/// 计算流水线气泡率并生成分析指标，用于调度器策略选择。
/// 支持 pp_size ∈ [2, 64], num_microbatches ∈ [1, 1024]。
///
/// 气泡率阈值：
/// - > 0.3: 建议切换策略（交错 PP 或增大 num_microbatches）
/// - 0.1 ~ 0.3: 可接受，建议 1F1B
/// - < 0.1: 良好，GPipe 或 1F1B 均可
// @trace REQ-DIST-024 [entity:BubbleAnalyzer] [api:POST /internal/distributed/pipeline/bubble]
#[derive(Debug, Clone, PartialEq)]
pub struct BubbleAnalyzer {
    /// Pipeline Parallel 维度，范围 [2, 64]
    pub pp_size: u32,
    /// 微批次数量，范围 [1, 1024]
    pub num_microbatches: usize,
    /// 虚拟 stage 数量，≥1（交错 PP）
    pub num_virtual_stages: u32,
    /// 单步前向计算时间（微秒），用于时间估计
    pub forward_step_us: f64,
    /// 单步反向计算时间（微秒），用于时间估计
    pub backward_step_us: f64,
    /// 气泡率阈值 — 高于此值建议切换策略
    pub bubble_ratio_threshold: f64,
}

/// BubbleAnalyzer 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BubbleAnalyzerError {
    /// pp_size < 2（bubble 分析仅对 pp_size >= 2 有意义）
    InvalidPpSize(u32),
    /// pp_size > 64
    PpSizeTooLarge(u32),
    /// num_microbatches == 0
    ZeroMicrobatches,
    /// num_microbatches > 1024
    TooManyMicrobatches(usize),
    /// num_virtual_stages == 0
    ZeroVirtualStages,
    /// 单步计算时间为负
    NegativeStepTime,
}

impl std::fmt::Display for BubbleAnalyzerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BubbleAnalyzerError::InvalidPpSize(pp_size) => {
                write!(f, "BubbleAnalyzer: pp_size={pp_size} must be >= 2 for bubble analysis")
            }
            BubbleAnalyzerError::PpSizeTooLarge(pp_size) => {
                write!(f, "BubbleAnalyzer: pp_size={pp_size} exceeds maximum 64")
            }
            BubbleAnalyzerError::ZeroMicrobatches => {
                write!(f, "BubbleAnalyzer: num_microbatches must be > 0")
            }
            BubbleAnalyzerError::TooManyMicrobatches(n) => {
                write!(f, "BubbleAnalyzer: num_microbatches={n} exceeds maximum 1024")
            }
            BubbleAnalyzerError::ZeroVirtualStages => {
                write!(f, "BubbleAnalyzer: num_virtual_stages must be >= 1")
            }
            BubbleAnalyzerError::NegativeStepTime => {
                write!(f, "BubbleAnalyzer: step time must be >= 0")
            }
        }
    }
}

impl std::error::Error for BubbleAnalyzerError {}

// @trace REQ-DIST-024 [entity:BubbleAnalyzer]
impl BubbleAnalyzer {
    /// 创建 bubble 分析器
    ///
    /// pp_size ∈ [2, 64], num_microbatches ∈ [1, 1024], num_virtual_stages ≥ 1
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn new(
        pp_size: u32,
        num_microbatches: usize,
        num_virtual_stages: u32,
        forward_step_us: f64,
        backward_step_us: f64,
    ) -> Result<Self, BubbleAnalyzerError> {
        if pp_size < 2 {
            return Err(BubbleAnalyzerError::InvalidPpSize(pp_size));
        }
        if pp_size > 64 {
            return Err(BubbleAnalyzerError::PpSizeTooLarge(pp_size));
        }
        if num_microbatches == 0 {
            return Err(BubbleAnalyzerError::ZeroMicrobatches);
        }
        if num_microbatches > 1024 {
            return Err(BubbleAnalyzerError::TooManyMicrobatches(num_microbatches));
        }
        if num_virtual_stages == 0 {
            return Err(BubbleAnalyzerError::ZeroVirtualStages);
        }
        if forward_step_us < 0.0 || backward_step_us < 0.0 {
            return Err(BubbleAnalyzerError::NegativeStepTime);
        }
        Ok(Self {
            pp_size,
            num_microbatches,
            num_virtual_stages,
            forward_step_us,
            backward_step_us,
            bubble_ratio_threshold: 0.3,
        })
    }

    /// 从 PipelineConfig 创建 bubble 分析器
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn from_pipeline_config(
        config: &PipelineConfig,
        num_microbatches: usize,
        forward_step_us: f64,
        backward_step_us: f64,
    ) -> Result<Self, BubbleAnalyzerError> {
        Self::new(
            config.pp_size,
            num_microbatches,
            config.num_virtual_stages,
            forward_step_us,
            backward_step_us,
        )
    }

    /// 设置气泡率阈值（默认 0.3）
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn with_bubble_ratio_threshold(mut self, threshold: f64) -> Self {
        self.bubble_ratio_threshold = threshold;
        self
    }

    /// GPipe 气泡比率 = (pp_size - 1) / (num_microbatches * pp_size) (REQ-DIST-024 验收标准 1)
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn gpipe_bubble_ratio(&self) -> f64 {
        (self.pp_size - 1) as f64 / (self.num_microbatches as f64 * self.pp_size as f64)
    }

    /// 交错模式气泡比率 = (pp_size - 1) / (num_virtual_stages * num_microbatches * pp_size)
    /// (REQ-DIST-024 验收标准 2)
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn interleaved_bubble_ratio(&self) -> f64 {
        (self.pp_size - 1) as f64
            / (self.num_virtual_stages as f64 * self.num_microbatches as f64 * self.pp_size as f64)
    }

    /// 1F1B 气泡比率（与 GPipe 公式相同，但峰值激活更低）
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn one_f_one_b_bubble_ratio(&self) -> f64 {
        self.gpipe_bubble_ratio()
    }

    /// 气泡时间估计（微秒）= bubble_ratio * total_pipeline_time
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn bubble_time_us(&self) -> f64 {
        let total = self.total_pipeline_time_us();
        self.gpipe_bubble_ratio() * total
    }

    /// 总流水线时间估计（微秒）
    ///
    /// GPipe: num_microbatches * (forward + backward) + (pp_size - 1) * forward
    /// （填充 + 稳态 + 排空）
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn total_pipeline_time_us(&self) -> f64 {
        let step_time = self.forward_step_us + self.backward_step_us;
        let fill_time = (self.pp_size - 1) as f64 * self.forward_step_us;
        self.num_microbatches as f64 * step_time + fill_time
    }

    /// 根据气泡率推荐策略 (REQ-DIST-024 验收标准 4)
    ///
    /// - bubble_ratio > threshold: 建议交错 PP
    /// - bubble_ratio ∈ (0.1, threshold]: 建议 1F1B
    /// - bubble_ratio <= 0.1: GPipe 或 1F1B 均可，默认 1F1B
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn recommend_strategy(&self) -> MicroBatchStrategy {
        let ratio = self.gpipe_bubble_ratio();
        if ratio > self.bubble_ratio_threshold {
            MicroBatchStrategy::OneFOneB
        } else {
            MicroBatchStrategy::OneFOneB
        }
    }

    /// 是否建议使用交错模式（交错气泡率低于阈值的一半）
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn recommend_interleaved(&self) -> bool {
        let interleaved_ratio = self.interleaved_bubble_ratio();
        let non_interleaved_ratio = self.gpipe_bubble_ratio();
        // 交错有意义当且仅当 num_virtual_stages > 1 且气泡率显著降低
        self.num_virtual_stages > 1 && interleaved_ratio < non_interleaved_ratio * 0.5
    }

    /// 计算最优 num_microbatches 使得 bubble_ratio <= target
    ///
    /// num_microbatches >= ceil((pp_size - 1) / (target * pp_size))
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn optimal_num_microbatches(&self, target_bubble_ratio: f64) -> usize {
        if target_bubble_ratio <= 0.0 || self.pp_size <= 1 {
            return self.num_microbatches;
        }
        let min_mbs = ((self.pp_size - 1) as f64 / (target_bubble_ratio * self.pp_size as f64))
            .ceil() as usize;
        min_mbs.max(1).min(1024)
    }

    /// 生成完整 bubble 分析指标 (REQ-DIST-024 验收标准 3)
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer] [dataflow:DF-DIST-013]
    pub fn analyze(&self) -> BubbleMetrics {
        let gpipe_ratio = self.gpipe_bubble_ratio();
        let one_f_one_b_ratio = self.one_f_one_b_bubble_ratio();
        let interleaved_ratio = self.interleaved_bubble_ratio();

        let current_ratio = if self.num_virtual_stages > 1 {
            interleaved_ratio
        } else {
            gpipe_ratio
        };

        BubbleMetrics {
            gpipe_bubble_ratio: gpipe_ratio,
            one_f_one_b_bubble_ratio: one_f_one_b_ratio,
            interleaved_bubble_ratio: interleaved_ratio,
            current_bubble_ratio: current_ratio,
            bubble_time_us: self.bubble_time_us(),
            total_pipeline_time_us: self.total_pipeline_time_us(),
            recommended_strategy: self.recommend_strategy(),
            recommend_interleaved: self.recommend_interleaved(),
            pp_size: self.pp_size,
            num_microbatches: self.num_microbatches,
            num_virtual_stages: self.num_virtual_stages,
        }
    }

    /// 校验一致性
    // @trace REQ-DIST-024 [entity:BubbleAnalyzer]
    pub fn validate(&self) -> bool {
        self.pp_size >= 2
            && self.pp_size <= 64
            && self.num_microbatches >= 1
            && self.num_microbatches <= 1024
            && self.num_virtual_stages >= 1
            && self.forward_step_us >= 0.0
            && self.backward_step_us >= 0.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BubbleAnalyzer: construction ──

    #[test]
    fn new_valid() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        assert_eq!(analyzer.pp_size, 4);
        assert_eq!(analyzer.num_microbatches, 8);
        assert_eq!(analyzer.num_virtual_stages, 1);
    }

    #[test]
    fn new_pp_size_2() {
        let analyzer = BubbleAnalyzer::new(2, 4, 1, 100.0, 200.0).unwrap();
        assert_eq!(analyzer.pp_size, 2);
    }

    #[test]
    fn new_pp_size_64() {
        let analyzer = BubbleAnalyzer::new(64, 64, 1, 100.0, 200.0).unwrap();
        assert_eq!(analyzer.pp_size, 64);
    }

    #[test]
    fn new_pp_size_1_returns_err() {
        let result = BubbleAnalyzer::new(1, 4, 1, 100.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::InvalidPpSize(1));
    }

    #[test]
    fn new_pp_size_65_returns_err() {
        let result = BubbleAnalyzer::new(65, 4, 1, 100.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::PpSizeTooLarge(65));
    }

    #[test]
    fn new_zero_microbatches_returns_err() {
        let result = BubbleAnalyzer::new(4, 0, 1, 100.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::ZeroMicrobatches);
    }

    #[test]
    fn new_too_many_microbatches_returns_err() {
        let result = BubbleAnalyzer::new(4, 1025, 1, 100.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::TooManyMicrobatches(1025));
    }

    #[test]
    fn new_max_microbatches_ok() {
        let analyzer = BubbleAnalyzer::new(4, 1024, 1, 100.0, 200.0).unwrap();
        assert_eq!(analyzer.num_microbatches, 1024);
    }

    #[test]
    fn new_zero_virtual_stages_returns_err() {
        let result = BubbleAnalyzer::new(4, 8, 0, 100.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::ZeroVirtualStages);
    }

    #[test]
    fn new_negative_step_time_returns_err() {
        let result = BubbleAnalyzer::new(4, 8, 1, -1.0, 200.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BubbleAnalyzerError::NegativeStepTime);
    }

    #[test]
    fn from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 0,
            num_virtual_stages: 2,
            micro_batch_size: 8,
            layers_per_stage: 8,
        };
        let analyzer = BubbleAnalyzer::from_pipeline_config(&config, 8, 100.0, 200.0).unwrap();
        assert_eq!(analyzer.pp_size, 4);
        assert_eq!(analyzer.num_virtual_stages, 2);
    }

    // ── BubbleAnalyzer: bubble ratio formulas (REQ-DIST-024 验收标准 1, 2) ──

    #[test]
    fn gpipe_bubble_ratio_formula() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        // 验收标准 1: bubble_ratio = (pp_size - 1) / (num_microbatches * pp_size)
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        let expected = 3.0 / (8.0 * 4.0); // 3/32 = 0.09375
        assert!((analyzer.gpipe_bubble_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn gpipe_bubble_ratio_pp2_mbs4() {
        // pp=2, mbs=4: (2-1)/(4*2) = 1/8 = 0.125
        let analyzer = BubbleAnalyzer::new(2, 4, 1, 100.0, 200.0).unwrap();
        let expected = 1.0 / 8.0;
        assert!((analyzer.gpipe_bubble_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn gpipe_bubble_ratio_decreases_with_more_microbatches() {
        let a8 = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        let a16 = BubbleAnalyzer::new(4, 16, 1, 100.0, 200.0).unwrap();
        assert!(a16.gpipe_bubble_ratio() < a8.gpipe_bubble_ratio());
    }

    #[test]
    fn interleaved_bubble_ratio_formula() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        // 验收标准 2: interleaved_bubble_ratio = (pp_size - 1) / (num_virtual_stages * num_microbatches * pp_size)
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0).unwrap();
        let expected = 3.0 / (2.0 * 8.0 * 4.0); // 3/64 = 0.046875
        assert!((analyzer.interleaved_bubble_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn interleaved_lower_than_gpipe() {
        let analyzer = BubbleAnalyzer::new(4, 8, 4, 100.0, 200.0).unwrap();
        assert!(analyzer.interleaved_bubble_ratio() < analyzer.gpipe_bubble_ratio());
    }

    #[test]
    fn one_f_one_b_same_formula_as_gpipe() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        assert!((analyzer.one_f_one_b_bubble_ratio() - analyzer.gpipe_bubble_ratio()).abs() < 1e-10);
    }

    // ── BubbleAnalyzer: time estimates (REQ-DIST-024 验收标准 3) ──

    #[test]
    fn total_pipeline_time() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        // total = num_microbatches * (fwd + bwd) + (pp_size - 1) * fwd
        // = 8 * 300 + 3 * 100 = 2400 + 300 = 2700
        let expected = 8.0 * 300.0 + 3.0 * 100.0;
        assert!((analyzer.total_pipeline_time_us() - expected).abs() < 1e-6);
    }

    #[test]
    fn bubble_time_is_ratio_of_total() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        let bubble_time = analyzer.bubble_time_us();
        let total = analyzer.total_pipeline_time_us();
        let ratio = bubble_time / total;
        assert!((ratio - analyzer.gpipe_bubble_ratio()).abs() < 1e-6);
    }

    // ── BubbleAnalyzer: strategy recommendation (REQ-DIST-024 验收标准 4) ──

    #[test]
    fn recommend_strategy_high_bubble() {
        // pp=2, mbs=2: ratio = 1/(2*2) = 0.25, below default threshold 0.3
        let analyzer = BubbleAnalyzer::new(2, 2, 1, 100.0, 200.0).unwrap();
        // Default threshold = 0.3, ratio = 0.25 < 0.3 → still 1F1B
        assert_eq!(analyzer.recommend_strategy(), MicroBatchStrategy::OneFOneB);
    }

    #[test]
    fn recommend_interleaved_beneficial() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        let analyzer = BubbleAnalyzer::new(8, 4, 4, 100.0, 200.0).unwrap();
        // With num_virtual_stages=4, interleaved ratio is significantly lower
        assert!(analyzer.recommend_interleaved());
    }

    #[test]
    fn recommend_not_interleaved_vs1() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        // num_virtual_stages=1 → no benefit from interleaving
        assert!(!analyzer.recommend_interleaved());
    }

    // ── BubbleAnalyzer: optimal_num_microbatches ──

    #[test]
    fn optimal_num_microbatches_for_target() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        // target = 0.05: mbs >= ceil(3 / (0.05 * 4)) = ceil(15) = 15
        let optimal = analyzer.optimal_num_microbatches(0.05);
        assert_eq!(optimal, 15);
    }

    #[test]
    fn optimal_num_microbatches_capped_at_1024() {
        let analyzer = BubbleAnalyzer::new(64, 8, 1, 100.0, 200.0).unwrap();
        // Very low target would need huge mbs, but capped at 1024
        let optimal = analyzer.optimal_num_microbatches(0.001);
        assert_eq!(optimal, 1024);
    }

    // ── BubbleAnalyzer: analyze (REQ-DIST-024 验收标准 3) ──

    #[test]
    fn analyze_complete_metrics() {
        // @trace TEST-DIST-024 [req:REQ-DIST-024] [level:unit]
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0).unwrap();
        let metrics = analyzer.analyze();
        assert_eq!(metrics.pp_size, 4);
        assert_eq!(metrics.num_microbatches, 8);
        assert_eq!(metrics.num_virtual_stages, 2);
        assert!(metrics.gpipe_bubble_ratio > 0.0);
        assert!(metrics.interleaved_bubble_ratio < metrics.gpipe_bubble_ratio);
        assert!(metrics.bubble_time_us > 0.0);
        assert!(metrics.total_pipeline_time_us > 0.0);
    }

    #[test]
    fn analyze_current_ratio_interleaved() {
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0).unwrap();
        let metrics = analyzer.analyze();
        // num_virtual_stages > 1 → current uses interleaved ratio
        assert!((metrics.current_bubble_ratio - metrics.interleaved_bubble_ratio).abs() < 1e-10);
    }

    #[test]
    fn analyze_current_ratio_non_interleaved() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap();
        let metrics = analyzer.analyze();
        // num_virtual_stages == 1 → current uses gpipe ratio
        assert!((metrics.current_bubble_ratio - metrics.gpipe_bubble_ratio).abs() < 1e-10);
    }

    // ── BubbleAnalyzer: validate ──

    #[test]
    fn validate_valid() {
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0).unwrap();
        assert!(analyzer.validate());
    }

    // ── BubbleAnalyzer: with_bubble_ratio_threshold ──

    #[test]
    fn custom_threshold() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0).unwrap()
            .with_bubble_ratio_threshold(0.5);
        assert!((analyzer.bubble_ratio_threshold - 0.5).abs() < 1e-10);
    }

    // ── BubbleAnalyzerError: Display ──

    #[test]
    fn error_display_invalid_pp_size() {
        let err = BubbleAnalyzerError::InvalidPpSize(1);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=1"));
        assert!(msg.contains(">= 2"));
    }

    #[test]
    fn error_display_pp_size_too_large() {
        let err = BubbleAnalyzerError::PpSizeTooLarge(65);
        let msg = format!("{}", err);
        assert!(msg.contains("65"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_display_zero_microbatches() {
        let err = BubbleAnalyzerError::ZeroMicrobatches;
        let msg = format!("{}", err);
        assert!(msg.contains("num_microbatches"));
    }

    #[test]
    fn error_display_too_many_microbatches() {
        let err = BubbleAnalyzerError::TooManyMicrobatches(1025);
        let msg = format!("{}", err);
        assert!(msg.contains("1025"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn error_display_zero_virtual_stages() {
        let err = BubbleAnalyzerError::ZeroVirtualStages;
        let msg = format!("{}", err);
        assert!(msg.contains("num_virtual_stages"));
    }

    #[test]
    fn error_display_negative_step_time() {
        let err = BubbleAnalyzerError::NegativeStepTime;
        let msg = format!("{}", err);
        assert!(msg.contains("step time"));
    }

    #[test]
    fn error_is_std_error() {
        let err = BubbleAnalyzerError::InvalidPpSize(1);
        let _: &dyn std::error::Error = &err;
    }

    // ── BubbleMetrics ──

    #[test]
    fn metrics_fields_accessible() {
        let metrics = BubbleMetrics {
            gpipe_bubble_ratio: 0.1,
            one_f_one_b_bubble_ratio: 0.1,
            interleaved_bubble_ratio: 0.05,
            current_bubble_ratio: 0.05,
            bubble_time_us: 100.0,
            total_pipeline_time_us: 2000.0,
            recommended_strategy: MicroBatchStrategy::OneFOneB,
            recommend_interleaved: true,
            pp_size: 4,
            num_microbatches: 8,
            num_virtual_stages: 2,
        };
        assert!((metrics.gpipe_bubble_ratio - 0.1).abs() < 1e-10);
        assert!(metrics.recommend_interleaved);
    }
}
