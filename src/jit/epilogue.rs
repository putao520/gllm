//! Epilogue 白嫖信号管线 (SPEC §18.1, §13)
//!
//! ## 核心原则
//! 所有机制间的协同通信通过 **零额外计算** 的遥测管线完成。
//! 不存在独立的"机制间通信通道" — 一切寄生在 Epilogue 尾段指令中。
//!
//! ## 信号流向 (§18.1)
//! ```
//! 信号产生 (§13, 寄存器内, ~3-10 条 SIMD):
//!   §13.5  SiLU 死神经元计数     → §13.1 Gate-First Skip
//!   §13.6  MoE Gate 命中计数     → §15.4 Uncommon Trap
//!   §13.7  GEMM 行级 ‖row‖₁+max → §13.5 死神经元判定
//!   §13.8  RmsNorm per-ch scale  → §11.2 KIVI K 量化
//!   §13.9  Softmax 锐度 + Sink   → §11.2 Sink FP16 保护
//!   §13.10 Embedding ‖embed‖₂    → §11.3 RaBitQ 初始修正
//!   §13.3  跨层能量差 Δρ         → §14.3 层跳过决策
//!
//! 信号传输 (§9.5, STG 写入 KV Page Header):
//!   Epilogue 尾段 STG 指令 → 写入 KvPageHeader padding bytes
//!
//! 信号消费:
//!   §9.2  JIT Director Daemon: 冷专家零命中 → Hot JMP 物理封杀
//!   §9.1  Block Routing: Δρ < ε → Thread Block 直接 Thread Exit
//!   §15.4 Uncommon Trap: DEOPT_REQUEST → 主机端微冻结恢复
//!   §17.9 Spec Scheduling: Entropy 低 → 启用推测解码
//! ```

use crate::kv_cache::{KvPageHeader, f16_bits_to_f32, dead_ratio_to_f32};

// ============================================================================
// §13 白嫖信号枚举 — 所有机制信号的统一表示
// ============================================================================

/// Epilogue 白嫖信号 — 寄存器内提取，零额外计算
///
/// 每个信号由 Epilogue 尾段的 1-10 条 SIMD 指令产生，
/// 写入 KvPageHeader 对应字段。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpilogueSignal {
    /// §13.5 SiLU 死神经元比例 → Gate-First Skip
    DeadNeuronRatio {
        /// σ(x) < ε 的神经元占比 [0.0, 1.0]
        ratio: f32,
    },
    /// §13.6 MoE Gate 专家命中计数 → Uncommon Trap
    ExpertHitCount {
        /// 专家 ID
        expert_id: u32,
        /// 累计命中次数
        hit_count: u32,
    },
    /// §13.7 GEMM 行级统计 → 死神经元判定
    RowActivationStats {
        /// 行级 L1 范数
        l1_norm: f32,
        /// 行级最大值
        max_val: f32,
    },
    /// §13.8 RmsNorm per-channel scale → KIVI K 量化
    PerChannelScale {
        /// per-channel |x|_max
        scale: f32,
    },
    /// §13.9 Softmax 锐度 + Sink 检测
    SoftmaxSharpness {
        /// max 值 (Sink 检测: max 极高 → Sink token)
        max_val: f32,
        /// max/sum 比值 (锐度: 接近1 → 尖锐关注; 接近1/n → 均匀分散)
        sharpness: f32,
    },
    /// §13.10 Embedding 范数 → RaBitQ 初始修正
    EmbeddingNorm {
        /// ‖embed‖₂
        norm: f32,
    },
    /// §13.3 跨层残差能量差 → 层跳过
    ResidualDeltaRho {
        /// Δρ = ‖x_out‖ / ‖x_in‖
        delta_rho: f32,
    },
    /// §13.11 残差方向余弦 → Early-Exit 精化指标
    ResidualCosineSimilarity {
        /// cos(θ) = x_in · x_out / (‖x_in‖ ‖x_out‖)
        cosine: f32,
    },
    /// §13.2 Softmax 质心坐标 → Prefetch
    CentroidPosition {
        /// 质心 token 位置 (argmax of softmax)
        position: usize,
    },
    /// 输出分布熵 → Spec Scheduling
    OutputEntropy {
        /// softmax 分布的熵值
        entropy: f32,
    },
}

/// §13.10 计算 L2 范数 ‖x‖₂ = sqrt(Σx²)
///
/// 用于 Embedding Lookup 后的初始范数计算，作为 RaBitQ 修正因子 C_0。
/// 也用于残差总线的能量检测。
pub fn compute_l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// ============================================================================
// §18.1 遥测信号聚合器 — 连接信号产生与消费
// ============================================================================

/// 遥测信号聚合器 — 收集 Epilogue 信号并提供决策接口
///
/// 设计原则:
/// - 聚合器本身不做决策，只提供信号数据
/// - 决策由各消费模块（GateFirstSkip, ResidualBypass 等）自行完成
/// - 聚合器是线程安全的（目前单线程执行，未来 Mega-Kernel 多线程需扩展）
#[derive(Debug, Clone, Default)]
pub struct TelemetryAggregator {
    /// 最近一次 SiLU 死神经元比例
    dead_neuron_ratio: f32,
    /// 最近一次 Softmax 锐度
    softmax_sharpness: f32,
    /// 最近一次 Softmax max
    softmax_max: f32,
    /// 最近一次残差 Δρ
    residual_delta_rho: f32,
    /// 最近一次残差方向余弦
    residual_cosine: f32,
    /// 最近一次输出熵
    output_entropy: f32,
    /// 最近一次 per-channel scale
    per_channel_scale: f32,
    /// 最近一次 Embedding 范数
    embedding_norm: f32,
    /// MoE 专家命中计数 (expert_id → hit_count)
    expert_hit_counts: Vec<u32>,
    /// 最近一次质心位置
    centroid_position: usize,
}

impl TelemetryAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从 Epilogue 信号更新聚合器状态
    pub fn ingest(&mut self, signal: &EpilogueSignal) {
        match signal {
            EpilogueSignal::DeadNeuronRatio { ratio } => {
                self.dead_neuron_ratio = *ratio;
            }
            EpilogueSignal::ExpertHitCount { expert_id, hit_count } => {
                let idx = *expert_id as usize;
                if idx >= self.expert_hit_counts.len() {
                    self.expert_hit_counts.resize(idx + 1, 0);
                }
                self.expert_hit_counts[idx] = *hit_count;
            }
            EpilogueSignal::RowActivationStats { l1_norm: _, max_val: _ } => {
                // Row stats feed into dead neuron detection — consumed at signal generation time
            }
            EpilogueSignal::PerChannelScale { scale } => {
                self.per_channel_scale = *scale;
            }
            EpilogueSignal::SoftmaxSharpness { max_val, sharpness } => {
                self.softmax_max = *max_val;
                self.softmax_sharpness = *sharpness;
            }
            EpilogueSignal::EmbeddingNorm { norm } => {
                self.embedding_norm = *norm;
            }
            EpilogueSignal::ResidualDeltaRho { delta_rho } => {
                self.residual_delta_rho = *delta_rho;
            }
            EpilogueSignal::ResidualCosineSimilarity { cosine } => {
                self.residual_cosine = *cosine;
            }
            EpilogueSignal::CentroidPosition { position } => {
                self.centroid_position = *position;
            }
            EpilogueSignal::OutputEntropy { entropy } => {
                self.output_entropy = *entropy;
            }
        }
    }

    /// 从 KvPageHeader 批量加载遥测数据
    pub fn ingest_from_page_header(&mut self, header: &KvPageHeader) {
        self.dead_neuron_ratio = dead_ratio_to_f32(header.dead_ratio);
        self.softmax_max = f16_bits_to_f32(header.softmax_max_avg);
        self.softmax_sharpness = f16_bits_to_f32(header.centroid_pos);
        self.residual_delta_rho = f16_bits_to_f32(header.delta_rho_avg);
        self.output_entropy = f16_bits_to_f32(header.entropy_avg);
    }

    // === 信号读取接口（供消费模块使用）===

    /// §13.5 死神经元比例
    pub fn dead_neuron_ratio(&self) -> f32 {
        self.dead_neuron_ratio
    }

    /// §13.9 Softmax 锐度 (max/sum)
    pub fn softmax_sharpness(&self) -> f32 {
        self.softmax_sharpness
    }

    /// §13.9 Softmax max (Sink 检测)
    pub fn softmax_max(&self) -> f32 {
        self.softmax_max
    }

    /// §13.3 残差 Δρ
    pub fn residual_delta_rho(&self) -> f32 {
        self.residual_delta_rho
    }

    /// §13.11 残差方向余弦
    pub fn residual_cosine(&self) -> f32 {
        self.residual_cosine
    }

    /// 输出熵
    pub fn output_entropy(&self) -> f32 {
        self.output_entropy
    }

    /// §13.8 per-channel scale
    pub fn per_channel_scale(&self) -> f32 {
        self.per_channel_scale
    }

    /// §13.10 Embedding 范数
    pub fn embedding_norm(&self) -> f32 {
        self.embedding_norm
    }

    /// §13.10 设置 Embedding 范数（由 decoder_forward 在 embedding lookup 后调用）
    pub fn set_embedding_norm(&mut self, norm: f32) {
        self.embedding_norm = norm;
    }

    /// §13.10 从 hidden state 计算 Embedding L2 范数
    ///
    /// 在 embedding lookup 后、第一层 RmsNorm 前调用。
    /// 用于 RaBitQ 修正因子的初始值 C_0。
    pub fn compute_and_set_embedding_norm(&mut self, hidden: &[f32]) {
        let norm = compute_l2_norm(hidden);
        self.embedding_norm = norm;
    }

    /// §13.6 专家命中计数
    pub fn expert_hit_count(&self, expert_id: u32) -> u32 {
        self.expert_hit_counts
            .get(expert_id as usize)
            .copied()
            .unwrap_or(0)
    }

    /// §13.2 质心位置
    pub fn centroid_position(&self) -> usize {
        self.centroid_position
    }

    /// 所有专家命中计数
    pub fn expert_hit_counts(&self) -> &[u32] {
        &self.expert_hit_counts
    }
}

// ============================================================================
// §13.1 Gate-First Skip — FFN 死神经元跳过决策
// ============================================================================

/// Gate-First Skip 配置
#[derive(Debug, Clone)]
pub struct GateFirstSkipConfig {
    /// 是否启用 Gate-First Skip
    pub enabled: bool,
    /// 死神经元占比阈值 — 超过此值触发跳过
    pub skip_threshold: f32,
    /// σ(x) < ε 的死神经元判定阈值
    pub dead_neuron_epsilon: f32,
}

impl Default for GateFirstSkipConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            skip_threshold: 0.5,  // SPEC §13.1: "> 50% 列失效"
            dead_neuron_epsilon: 1e-3,
        }
    }
}

/// Gate-First Skip 决策结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateSkipDecision {
    /// 正常执行完整 FFN (Gate + Up + Down)
    FullCompute,
    /// 跳过 Up + Down GEMM，用掩码输出 (死神经元 < 50% 但 > 25%)
    MaskedCompute,
    /// 完全跳过 FFN，输入直通输出 (死神经元 > 50%)
    Skip,
}

/// Gate-First Skip 决策器
pub struct GateFirstSkipDetector {
    pub(crate) config: GateFirstSkipConfig,
}

impl GateFirstSkipDetector {
    pub fn new(config: GateFirstSkipConfig) -> Self {
        Self { config }
    }

    /// 从死神经元比例做出跳过决策
    pub fn decide(&self, dead_ratio: f32) -> GateSkipDecision {
        if !self.config.enabled {
            return GateSkipDecision::FullCompute;
        }
        if dead_ratio > self.config.skip_threshold {
            GateSkipDecision::Skip
        } else if dead_ratio > self.config.skip_threshold * 0.5 {
            // 25%-50% 死神经元：使用掩码计算（Compact 挤压有效通道）
            GateSkipDecision::MaskedCompute
        } else {
            GateSkipDecision::FullCompute
        }
    }

    /// 从遥测聚合器做出决策
    pub fn decide_from_telemetry(&self, agg: &TelemetryAggregator) -> GateSkipDecision {
        self.decide(agg.dead_neuron_ratio())
    }
}

// ============================================================================
// §13.3 + §13.11 Residual Bypass — 层跳过决策
// ============================================================================

/// Residual Bypass 配置
#[derive(Debug, Clone)]
pub struct ResidualBypassConfig {
    /// 是否启用 Residual Bypass
    pub enabled: bool,
    /// Δρ 阈值 — ‖x_out‖/‖x_in‖ 接近 1.0 时表示层贡献极小
    pub delta_rho_threshold: f32,
    /// 方向余弦阈值 — cos(θ) > 此值时表示方向几乎不变
    pub cosine_threshold: f32,
    /// 最低跳过层 — 前几层通常信息丰富，不允许跳过
    pub min_skip_layer: usize,
}

impl Default for ResidualBypassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            delta_rho_threshold: 0.001,  // SPEC §13.3: Δρ < 0.001
            cosine_threshold: 0.99,       // SPEC §13.11: cos(θ) > 0.99
            min_skip_layer: 4,            // 前4层不允许跳过
        }
    }
}

/// Residual Bypass 决策结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualBypassDecision {
    /// 正常执行该层
    Execute,
    /// 跳过该层（输入直通输出）
    Bypass,
}

/// Residual Bypass 决策器
pub struct ResidualBypassDetector {
    config: ResidualBypassConfig,
}

impl ResidualBypassDetector {
    pub fn new(config: ResidualBypassConfig) -> Self {
        Self { config }
    }

    /// 从 Δρ 和 cos(θ) 做出跳过决策
    ///
    /// SPEC §13.11: "cos(θ) > 0.99 且 Δρ < 0.01 时高置信度跳过"
    pub fn decide(&self, layer: usize, delta_rho: f32, cosine: f32) -> ResidualBypassDecision {
        if !self.config.enabled {
            return ResidualBypassDecision::Execute;
        }
        if layer < self.config.min_skip_layer {
            return ResidualBypassDecision::Execute;
        }

        // SPEC §13.3 + §13.11 联合条件:
        // Δρ 接近 1.0（即 |Δρ - 1.0| < threshold）且 cos(θ) > 0.99
        let energy_stable = (delta_rho - 1.0).abs() < self.config.delta_rho_threshold;
        let direction_stable = cosine > self.config.cosine_threshold;

        if energy_stable && direction_stable {
            ResidualBypassDecision::Bypass
        } else {
            ResidualBypassDecision::Execute
        }
    }

    /// 从遥测聚合器做出决策
    pub fn decide_from_telemetry(
        &self,
        layer: usize,
        agg: &TelemetryAggregator,
    ) -> ResidualBypassDecision {
        self.decide(
            layer,
            agg.residual_delta_rho(),
            agg.residual_cosine(),
        )
    }
}

// ============================================================================
// §13.9 Softmax Sink Detection — Attention Sink 检测
// ============================================================================

/// Sink 检测配置
#[derive(Debug, Clone)]
pub struct SinkDetectionConfig {
    /// Sink token 判定阈值 — softmax_max > 此值视为 Sink
    pub sink_threshold: f32,
    /// 前 N 个 token 强制保护为 FP16 (KIVI Sink 保护, §11.2)
    pub protected_sink_count: usize,
    /// 锐度阈值 — sharpness > 此值时注意力高度集中
    pub sharp_focus_threshold: f32,
    /// 锐度低阈值 — sharpness < 此值时注意力均匀分散
    pub diffuse_threshold: f32,
}

impl Default for SinkDetectionConfig {
    fn default() -> Self {
        Self {
            sink_threshold: 0.9,
            protected_sink_count: 4,  // SPEC §11.2: "前 N 个 Token（默认 N=4）保留 FP16"
            sharp_focus_threshold: 0.8,
            diffuse_threshold: 0.1,
        }
    }
}

/// Sink 检测结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPattern {
    /// 正常注意力分布
    Normal,
    /// Sink token — 单个 token 获得绝大多数注意力
    Sink,
    /// 尖锐关注 — 关注少数几个 token
    SharpFocus,
    /// 均匀分散 — 注意力均匀分布
    Diffuse,
}

/// Sink 检测器
pub struct SinkDetector {
    pub(crate) config: SinkDetectionConfig,
}

impl SinkDetector {
    pub fn new(config: SinkDetectionConfig) -> Self {
        Self { config }
    }

    /// 从 Softmax max 和 sharpness 检测注意力模式
    pub fn detect(&self, max_val: f32, sharpness: f32) -> AttentionPattern {
        if max_val > self.config.sink_threshold {
            AttentionPattern::Sink
        } else if sharpness > self.config.sharp_focus_threshold {
            AttentionPattern::SharpFocus
        } else if sharpness < self.config.diffuse_threshold {
            AttentionPattern::Diffuse
        } else {
            AttentionPattern::Normal
        }
    }

    /// 从遥测聚合器检测
    pub fn detect_from_telemetry(&self, agg: &TelemetryAggregator) -> AttentionPattern {
        self.detect(agg.softmax_max(), agg.softmax_sharpness())
    }

    /// 判断某个 token 位置是否在 Sink 保护范围内
    pub fn is_protected_sink(&self, token_position: usize) -> bool {
        token_position < self.config.protected_sink_count
    }

    /// 返回保护 Sink 数量
    pub fn protected_sink_count(&self) -> usize {
        self.config.protected_sink_count
    }
}

// ============================================================================
// §13.6 MoE Expert Hit Counter — 冷板凳检测
// ============================================================================

/// MoE 专家冷热状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertThermalState {
    /// 热专家 — 频繁命中，权重常驻 L2
    Hot,
    /// 温专家 — 偶尔命中，权重在主存
    Warm,
    /// 冷板凳 — 持续零命中，可触发 Uncommon Trap (§15.4)
    Cold,
}

/// 专家热度追踪器 — JIT Director Daemon 使用
pub struct ExpertThermalTracker {
    /// 每个专家的累计命中次数
    hit_counts: Vec<u64>,
    /// 总 token 数（用于计算命中率）
    total_tokens: u64,
    /// 冷板凳判定阈值 — 命中率低于此值为冷
    cold_threshold: f32,
    /// 温热判定阈值 — 命中率高于此值为热
    hot_threshold: f32,
    /// 冷板凳持续 token 阈值 — 超过此数量零命中才判冷
    cold_zero_streak: u64,
    /// 每个专家的上次命中以来的 token 数
    zero_streak: Vec<u64>,
}

impl ExpertThermalTracker {
    pub fn new(num_experts: usize) -> Self {
        Self {
            hit_counts: vec![0; num_experts],
            total_tokens: 0,
            cold_threshold: 0.001,   // < 0.1% 命中率
            hot_threshold: 0.05,      // > 5% 命中率
            cold_zero_streak: 100000, // SPEC §9.2: "持续数百万 Token" 简化为 10万
            zero_streak: vec![0; num_experts],
        }
    }

    /// 记录一个 token 的专家路由结果
    pub fn record_routing(&mut self, selected_experts: &[usize]) {
        self.total_tokens += 1;

        // 重置被选中专家的零命中计数
        for &expert_id in selected_experts {
            if expert_id < self.hit_counts.len() {
                self.hit_counts[expert_id] += 1;
                self.zero_streak[expert_id] = 0;
            }
        }

        // 递增未选中专家的零命中计数
        for streak in &mut self.zero_streak {
            *streak += 1;
        }
        for &expert_id in selected_experts {
            if expert_id < self.zero_streak.len() {
                self.zero_streak[expert_id] = 0;
            }
        }
    }

    /// 从遥测聚合器更新（批量模式）
    pub fn update_from_telemetry(&mut self, agg: &TelemetryAggregator) {
        for (id, &count) in agg.expert_hit_counts().iter().enumerate() {
            if id < self.hit_counts.len() && count > 0 {
                self.hit_counts[id] += count as u64;
                if id < self.zero_streak.len() {
                    self.zero_streak[id] = 0;
                }
            }
        }
    }

    /// 获取专家的热度状态
    pub fn thermal_state(&self, expert_id: usize) -> ExpertThermalState {
        if expert_id >= self.hit_counts.len() {
            return ExpertThermalState::Cold;
        }

        // 冷板凳优先级最高 — 持续零命中
        if self.zero_streak[expert_id] > self.cold_zero_streak {
            return ExpertThermalState::Cold;
        }

        if self.total_tokens == 0 {
            return ExpertThermalState::Warm;
        }

        let hit_rate = self.hit_counts[expert_id] as f32 / self.total_tokens as f32;
        if hit_rate > self.hot_threshold {
            ExpertThermalState::Hot
        } else if hit_rate < self.cold_threshold {
            ExpertThermalState::Cold
        } else {
            ExpertThermalState::Warm
        }
    }

    /// 返回所有冷板凳专家 ID
    pub fn cold_experts(&self) -> Vec<usize> {
        (0..self.hit_counts.len())
            .filter(|&id| self.thermal_state(id) == ExpertThermalState::Cold)
            .collect()
    }

    /// 返回专家命中率
    pub fn hit_rate(&self, expert_id: usize) -> f32 {
        if self.total_tokens == 0 || expert_id >= self.hit_counts.len() {
            return 0.0;
        }
        self.hit_counts[expert_id] as f32 / self.total_tokens as f32
    }
}

// ============================================================================
// §17.9 Speculative Scheduling Signal — 推测解码调度信号
// ============================================================================

/// 推测解码调度建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecScheduleAdvice {
    /// 建议启用推测解码（低熵 → 高接受率预期）
    EnableSpec,
    /// 建议使用标准解码
    StandardDecode,
    /// 建议回退（连续低接受率）
    Fallback,
}

/// 推测解码调度信号提取器
pub struct SpecScheduleSignal {
    /// 启用推测解码的熵阈值 — 低于此值时建议启用
    enable_entropy_threshold: f32,
    /// 回退的连续低接受轮次阈值
    fallback_streak: u32,
    /// 当前连续低接受轮次计数
    low_acceptance_streak: u32,
}

impl SpecScheduleSignal {
    pub fn new() -> Self {
        Self {
            enable_entropy_threshold: 2.0,  // 低熵 → 高确定性 → 适合推测
            fallback_streak: 3,              // SPEC §17.9: "连续 3 轮 acceptance_rate < 0.3"
            low_acceptance_streak: 0,
        }
    }

    /// 从输出熵和接受率给出调度建议
    pub fn advise(&mut self, output_entropy: f32, acceptance_rate: f32) -> SpecScheduleAdvice {
        if acceptance_rate < 0.3 {
            self.low_acceptance_streak += 1;
            if self.low_acceptance_streak >= self.fallback_streak {
                return SpecScheduleAdvice::Fallback;
            }
            return SpecScheduleAdvice::StandardDecode;
        }

        self.low_acceptance_streak = 0;

        if output_entropy < self.enable_entropy_threshold {
            SpecScheduleAdvice::EnableSpec
        } else {
            SpecScheduleAdvice::StandardDecode
        }
    }

    /// 从遥测聚合器给出调度建议
    pub fn advise_from_telemetry(
        &mut self,
        agg: &TelemetryAggregator,
        acceptance_rate: f32,
    ) -> SpecScheduleAdvice {
        self.advise(agg.output_entropy(), acceptance_rate)
    }

    /// 重置状态
    pub fn reset(&mut self) {
        self.low_acceptance_streak = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_aggregator_ingest() {
        let mut agg = TelemetryAggregator::new();

        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.6 });
        assert!((agg.dead_neuron_ratio() - 0.6).abs() < 0.001);

        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.85,
        });
        assert!((agg.softmax_max() - 0.95).abs() < 0.001);
        assert!((agg.softmax_sharpness() - 0.85).abs() < 0.001);

        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.001 });
        assert!((agg.residual_delta_rho() - 1.001).abs() < 0.0001);

        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.995 });
        assert!((agg.residual_cosine() - 0.995).abs() < 0.001);

        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 1.5 });
        assert!((agg.output_entropy() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_telemetry_ingest_from_page_header() {
        let mut agg = TelemetryAggregator::new();
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = crate::kv_cache::f32_to_f16_bits(2.5);
        header.softmax_max_avg = crate::kv_cache::f32_to_f16_bits(0.8);
        header.dead_ratio = crate::kv_cache::f32_to_dead_ratio(0.3);

        agg.ingest_from_page_header(&header);
        assert!((agg.dead_neuron_ratio() - 0.3).abs() < 0.02);
        assert!((agg.softmax_max() - 0.8).abs() < 0.02);
        assert!((agg.output_entropy() - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_gate_first_skip_decision() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());

        assert_eq!(detector.decide(0.1), GateSkipDecision::FullCompute);
        assert_eq!(detector.decide(0.3), GateSkipDecision::MaskedCompute);
        assert_eq!(detector.decide(0.6), GateSkipDecision::Skip);
        assert_eq!(detector.decide(0.0), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_disabled() {
        let config = GateFirstSkipConfig {
            enabled: false,
            ..Default::default()
        };
        let detector = GateFirstSkipDetector::new(config);
        assert_eq!(detector.decide(0.9), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_residual_bypass_decision() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());

        // 高置信度跳过: Δρ ≈ 1.0 且 cos(θ) > 0.99
        assert_eq!(
            detector.decide(10, 1.0005, 0.995),
            ResidualBypassDecision::Bypass,
        );

        // 不跳过: Δρ 偏差大
        assert_eq!(
            detector.decide(10, 1.05, 0.995),
            ResidualBypassDecision::Execute,
        );

        // 不跳过: cos(θ) 不够高
        assert_eq!(
            detector.decide(10, 1.0005, 0.95),
            ResidualBypassDecision::Execute,
        );

        // 不跳过: 层太浅
        assert_eq!(
            detector.decide(2, 1.0005, 0.995),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0005 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.995 });

        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        assert_eq!(
            detector.decide_from_telemetry(10, &agg),
            ResidualBypassDecision::Bypass,
        );
    }

    #[test]
    fn test_sink_detector() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());

        assert_eq!(detector.detect(0.95, 0.9), AttentionPattern::Sink);
        assert_eq!(detector.detect(0.5, 0.85), AttentionPattern::SharpFocus);
        assert_eq!(detector.detect(0.1, 0.05), AttentionPattern::Diffuse);
        assert_eq!(detector.detect(0.3, 0.4), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_protection() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert!(detector.is_protected_sink(0));
        assert!(detector.is_protected_sink(3));
        assert!(!detector.is_protected_sink(4));
        assert!(!detector.is_protected_sink(100));
        assert_eq!(detector.protected_sink_count(), 4);
    }

    #[test]
    fn test_expert_thermal_tracker() {
        let mut tracker = ExpertThermalTracker::new(4);

        // Expert 0 命中 100 次, Expert 1 命中 0 次
        for _ in 0..100 {
            tracker.record_routing(&[0, 2]);
        }

        assert!(tracker.hit_rate(0) > 0.0);
        assert_eq!(tracker.hit_rate(1), 0.0);

        let state_0 = tracker.thermal_state(0);
        assert!(matches!(state_0, ExpertThermalState::Hot | ExpertThermalState::Warm));
    }

    #[test]
    fn test_expert_cold_detection() {
        let mut tracker = ExpertThermalTracker::new(4);
        tracker.cold_zero_streak = 10; // 简化测试

        // Expert 3 从未命中，超过阈值
        for _ in 0..11 {
            tracker.record_routing(&[0]);
        }

        assert_eq!(tracker.thermal_state(3), ExpertThermalState::Cold);
        let cold = tracker.cold_experts();
        assert!(cold.contains(&3));
    }

    #[test]
    fn test_spec_schedule_signal() {
        let mut signal = SpecScheduleSignal::new();

        // 低熵 → 建议推测
        assert_eq!(signal.advise(1.0, 0.8), SpecScheduleAdvice::EnableSpec);

        // 高熵 → 标准解码
        assert_eq!(signal.advise(3.0, 0.6), SpecScheduleAdvice::StandardDecode);

        // 连续低接受率 → 回退
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::Fallback);

        // 重置后恢复
        signal.reset();
        assert_eq!(signal.advise(1.0, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_expert_hit_counts_from_aggregator() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount {
            expert_id: 2,
            hit_count: 42,
        });
        agg.ingest(&EpilogueSignal::ExpertHitCount {
            expert_id: 5,
            hit_count: 7,
        });

        assert_eq!(agg.expert_hit_count(2), 42);
        assert_eq!(agg.expert_hit_count(5), 7);
        assert_eq!(agg.expert_hit_count(0), 0); // 未记录
    }
}
