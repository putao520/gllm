//! AdaptiveMicroBatchSizer — 动态微批次大小调整 (REQ-DIST-025)
//!
//! 根据运行时延迟和带宽指标自适应调整微批次大小：
//! - 延迟 > threshold → 减小 mbs（避免 stage 间空闲）
//! - 带宽 > threshold → 增大 mbs（充分利用通信管道）
//! - 结果约束：user_min <= mbs <= total_batch_size
//!
//! 自适应在第一个 prefill 微批次前完成（验收标准 5）。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use super::config::PipelineConfig;

// ── AdaptiveConfig (REQ-DIST-025) ────────────────────────────────────────────

/// 自适应微批次大小调整配置 (REQ-DIST-025)
///
/// 控制延迟/带宽阈值和调整策略。
// @trace REQ-DIST-025 [entity:AdaptiveConfig] [api:POST /internal/distributed/pipeline/adaptive]
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveConfig {
    /// 延迟阈值（毫秒）：超过此值减小微批次大小
    pub latency_threshold_ms: f64,
    /// 带宽阈值（Gbps）：超过此值增大微批次大小
    pub bandwidth_threshold_gbps: f64,
    /// 用户指定的最小微批次大小
    pub user_min_mbs: usize,
    /// 延迟过高时的缩减因子（0 < factor < 1，默认 0.75）
    pub shrink_factor: f64,
    /// 带宽充足时的增长因子（factor > 1，默认 1.25）
    pub grow_factor: f64,
    /// 单次调整最大变化量（绝对值），防止剧烈波动
    pub max_step_change: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            latency_threshold_ms: 10.0,
            bandwidth_threshold_gbps: 50.0,
            user_min_mbs: 1,
            shrink_factor: 0.75,
            grow_factor: 1.25,
            max_step_change: 4,
        }
    }
}

impl AdaptiveConfig {
    /// 创建自适应配置
    // @trace REQ-DIST-025 [entity:AdaptiveConfig]
    pub fn new(
        latency_threshold_ms: f64,
        bandwidth_threshold_gbps: f64,
        user_min_mbs: usize,
    ) -> Self {
        Self {
            latency_threshold_ms,
            bandwidth_threshold_gbps,
            user_min_mbs: user_min_mbs.max(1),
            ..Default::default()
        }
    }

    /// 设置缩减因子
    // @trace REQ-DIST-025 [entity:AdaptiveConfig]
    pub fn with_shrink_factor(mut self, factor: f64) -> Self {
        self.shrink_factor = factor;
        self
    }

    /// 设置增长因子
    // @trace REQ-DIST-025 [entity:AdaptiveConfig]
    pub fn with_grow_factor(mut self, factor: f64) -> Self {
        self.grow_factor = factor;
        self
    }

    /// 设置最大单步变化量
    // @trace REQ-DIST-025 [entity:AdaptiveConfig]
    pub fn with_max_step_change(mut self, change: usize) -> Self {
        self.max_step_change = change.max(1);
        self
    }

    /// 校验配置一致性
    // @trace REQ-DIST-025 [entity:AdaptiveConfig]
    pub fn validate(&self) -> bool {
        self.latency_threshold_ms > 0.0
            && self.bandwidth_threshold_gbps > 0.0
            && self.user_min_mbs >= 1
            && self.shrink_factor > 0.0
            && self.shrink_factor < 1.0
            && self.grow_factor > 1.0
            && self.max_step_change >= 1
    }
}

// ── AdaptiveMicroBatchSizer (REQ-DIST-025) ────────────────────────────────────

/// 动态微批次大小调整器 (REQ-DIST-025)
///
/// 根据运行时延迟和带宽指标自适应调整微批次大小。
/// 调整策略：
/// - latency_ms > threshold → mbs *= shrink_factor（减小负载降低延迟）
/// - bandwidth_gbps > threshold → mbs *= grow_factor（利用空闲带宽）
/// - 结果约束：user_min_mbs <= mbs <= total_batch_size
///
/// 自适应在第一个 prefill 微批次前完成（验收标准 5）。
// @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer] [api:POST /internal/distributed/pipeline/adaptive]
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveMicroBatchSizer {
    /// 当前微批次大小
    pub current_mbs: usize,
    /// 总批次大小（上限）
    pub total_batch_size: usize,
    /// 自适应配置
    pub config: AdaptiveConfig,
    /// 调整历史（最近 N 次调整的 mbs 值，用于稳定性检测）
    pub history: Vec<usize>,
    /// 是否已完成初始自适应（第一个 prefill 微批次前完成）
    pub adaptation_complete: bool,
}

/// AdaptiveMicroBatchSizer 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveMicroBatchSizerError {
    /// current_mbs == 0
    ZeroMicroBatchSize,
    /// total_batch_size == 0
    ZeroTotalBatchSize,
    /// current_mbs > total_batch_size
    MbsExceedsTotal { mbs: usize, total: usize },
    /// user_min_mbs > total_batch_size
    MinExceedsTotal { min: usize, total: usize },
}

impl std::fmt::Display for AdaptiveMicroBatchSizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdaptiveMicroBatchSizerError::ZeroMicroBatchSize => {
                write!(f, "AdaptiveMicroBatchSizer: current_mbs must be > 0")
            }
            AdaptiveMicroBatchSizerError::ZeroTotalBatchSize => {
                write!(f, "AdaptiveMicroBatchSizer: total_batch_size must be > 0")
            }
            AdaptiveMicroBatchSizerError::MbsExceedsTotal { mbs, total } => {
                write!(f, "AdaptiveMicroBatchSizer: mbs({mbs}) > total_batch_size({total})")
            }
            AdaptiveMicroBatchSizerError::MinExceedsTotal { min, total } => {
                write!(f, "AdaptiveMicroBatchSizer: user_min_mbs({min}) > total_batch_size({total})")
            }
        }
    }
}

impl std::error::Error for AdaptiveMicroBatchSizerError {}

// @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
impl AdaptiveMicroBatchSizer {
    /// 创建自适应微批次大小调整器
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn new(
        current_mbs: usize,
        total_batch_size: usize,
        config: AdaptiveConfig,
    ) -> Result<Self, AdaptiveMicroBatchSizerError> {
        if current_mbs == 0 {
            return Err(AdaptiveMicroBatchSizerError::ZeroMicroBatchSize);
        }
        if total_batch_size == 0 {
            return Err(AdaptiveMicroBatchSizerError::ZeroTotalBatchSize);
        }
        if current_mbs > total_batch_size {
            return Err(AdaptiveMicroBatchSizerError::MbsExceedsTotal {
                mbs: current_mbs,
                total: total_batch_size,
            });
        }
        if config.user_min_mbs > total_batch_size {
            return Err(AdaptiveMicroBatchSizerError::MinExceedsTotal {
                min: config.user_min_mbs,
                total: total_batch_size,
            });
        }
        Ok(Self {
            current_mbs,
            total_batch_size,
            config,
            history: vec![current_mbs],
            adaptation_complete: false,
        })
    }

    /// 从 PipelineConfig 创建调整器
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn from_pipeline_config(
        config: &PipelineConfig,
        total_batch_size: usize,
        adaptive_config: AdaptiveConfig,
    ) -> Result<Self, AdaptiveMicroBatchSizerError> {
        Self::new(config.micro_batch_size, total_batch_size, adaptive_config)
    }

    /// 根据延迟和带宽调整微批次大小 (REQ-DIST-025 验收标准 1, 2, 3)
    ///
    /// - latency_ms > threshold → 减小 mbs
    /// - bandwidth_gbps > threshold → 增大 mbs
    /// - 结果 >= user_min_mbs && <= total_batch_size
    ///
    /// 返回调整后的微批次大小。
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer] [dataflow:DF-DIST-014]
    pub fn adjust(&mut self, latency_ms: f64, bandwidth_gbps: f64) -> usize {
        let prev_mbs = self.current_mbs;

        if latency_ms > self.config.latency_threshold_ms {
            // 延迟过高 → 减小微批次大小 (验收标准 2)
            let shrunk = (self.current_mbs as f64 * self.config.shrink_factor).floor() as usize;
            let delta = prev_mbs.saturating_sub(shrunk).min(self.config.max_step_change);
            self.current_mbs = prev_mbs.saturating_sub(delta);
        } else if bandwidth_gbps > self.config.bandwidth_threshold_gbps {
            // 带宽充足 → 增大微批次大小 (验收标准 3)
            let grown = (self.current_mbs as f64 * self.config.grow_factor).floor() as usize;
            let delta = grown.saturating_sub(prev_mbs).min(self.config.max_step_change);
            self.current_mbs = prev_mbs + delta;
        }

        // 约束范围 (验收标准 1: user_min <= mbs <= total_batch_size)
        self.current_mbs = self.current_mbs
            .max(self.config.user_min_mbs)
            .min(self.total_batch_size);

        // 记录历史
        self.history.push(self.current_mbs);
        if self.history.len() > 16 {
            self.history.remove(0);
        }

        self.current_mbs
    }

    /// 标记自适应完成（在第一个 prefill 微批次前调用）
    ///
    /// 验收标准 5: 自适应在第一个 prefill 微批次前完成。
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn mark_adaptation_complete(&mut self) {
        self.adaptation_complete = true;
    }

    /// 检查自适应是否已完成
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn is_adaptation_complete(&self) -> bool {
        self.adaptation_complete
    }

    /// 重置到初始微批次大小
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn reset(&mut self) {
        if let Some(&initial) = self.history.first() {
            self.current_mbs = initial;
        }
        self.adaptation_complete = false;
    }

    /// 最近调整是否稳定（最近 4 次调整变化 < 10%）
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn is_stable(&self) -> bool {
        if self.history.len() < 4 {
            return false;
        }
        let recent = &self.history[self.history.len() - 4..];
        let min = *recent.iter().min().unwrap_or(&1) as f64;
        let max = *recent.iter().max().unwrap_or(&1) as f64;
        if min == 0.0 {
            return false;
        }
        (max - min) / min < 0.1
    }

    /// 调整次数
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn adjustment_count(&self) -> usize {
        self.history.len().saturating_sub(1)
    }

    /// 校验一致性
    // @trace REQ-DIST-025 [entity:AdaptiveMicroBatchSizer]
    pub fn validate(&self) -> bool {
        self.current_mbs >= 1
            && self.current_mbs <= self.total_batch_size
            && self.current_mbs >= self.config.user_min_mbs
            && self.config.validate()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AdaptiveConfig ──

    #[test]
    fn config_default() {
        let config = AdaptiveConfig::default();
        assert!((config.latency_threshold_ms - 10.0).abs() < 1e-10);
        assert!((config.bandwidth_threshold_gbps - 50.0).abs() < 1e-10);
        assert_eq!(config.user_min_mbs, 1);
        assert!((config.shrink_factor - 0.75).abs() < 1e-10);
        assert!((config.grow_factor - 1.25).abs() < 1e-10);
        assert_eq!(config.max_step_change, 4);
    }

    #[test]
    fn config_new() {
        let config = AdaptiveConfig::new(20.0, 100.0, 2);
        assert!((config.latency_threshold_ms - 20.0).abs() < 1e-10);
        assert!((config.bandwidth_threshold_gbps - 100.0).abs() < 1e-10);
        assert_eq!(config.user_min_mbs, 2);
    }

    #[test]
    fn config_new_min_clamps_to_1() {
        let config = AdaptiveConfig::new(20.0, 100.0, 0);
        assert_eq!(config.user_min_mbs, 1);
    }

    #[test]
    fn config_builder_methods() {
        let config = AdaptiveConfig::default()
            .with_shrink_factor(0.5)
            .with_grow_factor(1.5)
            .with_max_step_change(8);
        assert!((config.shrink_factor - 0.5).abs() < 1e-10);
        assert!((config.grow_factor - 1.5).abs() < 1e-10);
        assert_eq!(config.max_step_change, 8);
    }

    #[test]
    fn config_validate_valid() {
        assert!(AdaptiveConfig::default().validate());
    }

    #[test]
    fn config_validate_invalid_shrink() {
        let config = AdaptiveConfig::default().with_shrink_factor(1.5);
        assert!(!config.validate());
    }

    #[test]
    fn config_validate_invalid_grow() {
        let config = AdaptiveConfig::default().with_grow_factor(0.5);
        assert!(!config.validate());
    }

    // ── AdaptiveMicroBatchSizer: construction ──

    #[test]
    fn new_valid() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        let sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert_eq!(sizer.current_mbs, 8);
        assert_eq!(sizer.total_batch_size, 64);
        assert!(!sizer.adaptation_complete);
    }

    #[test]
    fn new_zero_mbs_returns_err() {
        let result = AdaptiveMicroBatchSizer::new(0, 64, AdaptiveConfig::default());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AdaptiveMicroBatchSizerError::ZeroMicroBatchSize);
    }

    #[test]
    fn new_zero_total_returns_err() {
        let result = AdaptiveMicroBatchSizer::new(8, 0, AdaptiveConfig::default());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AdaptiveMicroBatchSizerError::ZeroTotalBatchSize);
    }

    #[test]
    fn new_mbs_exceeds_total_returns_err() {
        let result = AdaptiveMicroBatchSizer::new(100, 64, AdaptiveConfig::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            AdaptiveMicroBatchSizerError::MbsExceedsTotal { mbs, total } => {
                assert_eq!(mbs, 100);
                assert_eq!(total, 64);
            }
            other => panic!("expected MbsExceedsTotal, got {:?}", other),
        }
    }

    #[test]
    fn new_min_exceeds_total_returns_err() {
        let config = AdaptiveConfig::new(10.0, 50.0, 100);
        let result = AdaptiveMicroBatchSizer::new(8, 64, config);
        assert!(result.is_err());
        match result.unwrap_err() {
            AdaptiveMicroBatchSizerError::MinExceedsTotal { min, total } => {
                assert_eq!(min, 100);
                assert_eq!(total, 64);
            }
            other => panic!("expected MinExceedsTotal, got {:?}", other),
        }
    }

    #[test]
    fn from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 8,
            layers_per_stage: 8,
        };
        let sizer = AdaptiveMicroBatchSizer::from_pipeline_config(
            &config, 64, AdaptiveConfig::default(),
        ).unwrap();
        assert_eq!(sizer.current_mbs, 8);
    }

    // ── AdaptiveMicroBatchSizer: adjust (REQ-DIST-025 验收标准 1, 2, 3) ──

    #[test]
    fn adjust_high_latency_shrinks() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        // 验收标准 2: 延迟 > threshold → 减小 mbs
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        let new_mbs = sizer.adjust(20.0, 30.0); // latency > 10, bandwidth < 50
        assert!(new_mbs < 8, "should shrink when latency is high");
    }

    #[test]
    fn adjust_high_bandwidth_grows() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        // 验收标准 3: 带宽 > threshold → 增大 mbs
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        let new_mbs = sizer.adjust(5.0, 80.0); // latency < 10, bandwidth > 50
        assert!(new_mbs > 8, "should grow when bandwidth is sufficient");
    }

    #[test]
    fn adjust_normal_no_change() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        let new_mbs = sizer.adjust(5.0, 30.0); // latency < 10, bandwidth < 50
        assert_eq!(new_mbs, 8, "no change when conditions are normal");
    }

    #[test]
    fn adjust_clamps_to_user_min() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        // 验收标准 1: 结果 >= user_min_mbs
        let config = AdaptiveConfig::new(10.0, 50.0, 4);
        let mut sizer = AdaptiveMicroBatchSizer::new(4, 64, config).unwrap();
        // Repeatedly shrink — should never go below user_min_mbs=4
        for _ in 0..20 {
            sizer.adjust(100.0, 0.0); // very high latency
        }
        assert!(sizer.current_mbs >= 4, "mbs should never go below user_min_mbs");
    }

    #[test]
    fn adjust_clamps_to_total_batch_size() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        // 验收标准 1: 结果 <= total_batch_size
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 16, AdaptiveConfig::default()).unwrap();
        // Repeatedly grow — should never exceed total_batch_size
        for _ in 0..20 {
            sizer.adjust(0.0, 100.0); // very high bandwidth
        }
        assert!(sizer.current_mbs <= 16, "mbs should never exceed total_batch_size");
    }

    #[test]
    fn adjust_respects_max_step_change() {
        let config = AdaptiveConfig::default().with_max_step_change(2);
        let mut sizer = AdaptiveMicroBatchSizer::new(20, 64, config).unwrap();
        let prev = sizer.current_mbs;
        let new_mbs = sizer.adjust(100.0, 0.0); // high latency → shrink
        let change = prev.saturating_sub(new_mbs);
        assert!(change <= 2, "single step change should not exceed max_step_change");
    }

    // ── AdaptiveMicroBatchSizer: adaptation lifecycle (验收标准 5) ──

    #[test]
    fn adaptation_not_complete_by_default() {
        let sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert!(!sizer.is_adaptation_complete());
    }

    #[test]
    fn mark_adaptation_complete() {
        // @trace TEST-DIST-025 [req:REQ-DIST-025] [level:unit]
        // 验收标准 5: 自适应在第一个 prefill 微批次前完成
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        // Perform initial adaptation
        sizer.adjust(5.0, 80.0);
        sizer.mark_adaptation_complete();
        assert!(sizer.is_adaptation_complete());
    }

    // ── AdaptiveMicroBatchSizer: reset ──

    #[test]
    fn reset_restores_initial() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        sizer.adjust(100.0, 0.0); // shrink
        sizer.reset();
        assert_eq!(sizer.current_mbs, 8);
        assert!(!sizer.adaptation_complete);
    }

    // ── AdaptiveMicroBatchSizer: stability ──

    #[test]
    fn stability_after_convergence() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        // Simulate stable conditions
        for _ in 0..8 {
            sizer.adjust(5.0, 30.0); // no change → stable
        }
        assert!(sizer.is_stable());
    }

    #[test]
    fn not_stable_initially() {
        let sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert!(!sizer.is_stable());
    }

    // ── AdaptiveMicroBatchSizer: adjustment_count ──

    #[test]
    fn adjustment_count() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert_eq!(sizer.adjustment_count(), 0);
        sizer.adjust(5.0, 80.0);
        assert_eq!(sizer.adjustment_count(), 1);
        sizer.adjust(5.0, 30.0);
        assert_eq!(sizer.adjustment_count(), 2);
    }

    // ── AdaptiveMicroBatchSizer: validate ──

    #[test]
    fn validate_valid() {
        let sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert!(sizer.validate());
    }

    // ── AdaptiveMicroBatchSizerError: Display ──

    #[test]
    fn error_display_zero_mbs() {
        let err = AdaptiveMicroBatchSizerError::ZeroMicroBatchSize;
        let msg = format!("{}", err);
        assert!(msg.contains("current_mbs"));
    }

    #[test]
    fn error_display_zero_total() {
        let err = AdaptiveMicroBatchSizerError::ZeroTotalBatchSize;
        let msg = format!("{}", err);
        assert!(msg.contains("total_batch_size"));
    }

    #[test]
    fn error_display_mbs_exceeds_total() {
        let err = AdaptiveMicroBatchSizerError::MbsExceedsTotal { mbs: 100, total: 64 };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_display_min_exceeds_total() {
        let err = AdaptiveMicroBatchSizerError::MinExceedsTotal { min: 100, total: 64 };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_is_std_error() {
        let err = AdaptiveMicroBatchSizerError::ZeroMicroBatchSize;
        let _: &dyn std::error::Error = &err;
    }
}
