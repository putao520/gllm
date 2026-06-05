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

impl Default for SpecScheduleSignal {
    fn default() -> Self {
        Self::new()
    }
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

    // ========================================================================
    // EpilogueSignal — Clone/Copy/PartialEq for all variants
    // ========================================================================

    #[test]
    fn test_epilogue_signal_clone_copy_eq() {
        let signals = vec![
            EpilogueSignal::DeadNeuronRatio { ratio: 0.42 },
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 99 },
            EpilogueSignal::RowActivationStats { l1_norm: 1.5, max_val: 0.8 },
            EpilogueSignal::PerChannelScale { scale: 2.5 },
            EpilogueSignal::SoftmaxSharpness { max_val: 0.7, sharpness: 0.6 },
            EpilogueSignal::EmbeddingNorm { norm: 12.34 },
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.999 },
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.998 },
            EpilogueSignal::CentroidPosition { position: 7 },
            EpilogueSignal::OutputEntropy { entropy: 3.14 },
        ];

        for signal in &signals {
            let cloned = signal.clone();
            assert_eq!(*signal, cloned);
        }
    }

    // ========================================================================
    // compute_l2_norm
    // ========================================================================

    #[test]
    fn test_compute_l2_norm_empty() {
        let norm = compute_l2_norm(&[]);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_compute_l2_norm_single_element() {
        let norm = compute_l2_norm(&[3.0]);
        assert!((norm - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_known_values() {
        // sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3.0
        let norm = compute_l2_norm(&[1.0, 2.0, 2.0]);
        assert!((norm - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_unit_vector() {
        let norm = compute_l2_norm(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(norm, 0.0);
    }

    // ========================================================================
    // TelemetryAggregator — Default, Clone, remaining ingest paths
    // ========================================================================

    #[test]
    fn test_telemetry_aggregator_default_values() {
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
        assert!(agg.expert_hit_counts().is_empty());
    }

    #[test]
    fn test_telemetry_aggregator_clone() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.75 });
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 4.2 });

        let cloned = agg.clone();
        assert!((cloned.dead_neuron_ratio() - 0.75).abs() < 1e-6);
        assert!((cloned.output_entropy() - 4.2).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_ingest_per_channel_scale() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 3.14 });
        assert!((agg.per_channel_scale() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_ingest_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 5.67 });
        assert!((agg.embedding_norm() - 5.67).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_ingest_centroid_position() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 42 });
        assert_eq!(agg.centroid_position(), 42);
    }

    #[test]
    fn test_telemetry_ingest_row_activation_stats_no_panic() {
        // RowActivationStats is consumed at signal generation time, no state change
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::RowActivationStats {
            l1_norm: 1.23,
            max_val: 4.56,
        });
        // All values remain at default
        assert_eq!(agg.dead_neuron_ratio(), 0.0);
    }

    #[test]
    fn test_telemetry_set_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        agg.set_embedding_norm(9.81);
        assert!((agg.embedding_norm() - 9.81).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_compute_and_set_embedding_norm() {
        let mut agg = TelemetryAggregator::new();
        let hidden = vec![3.0, 4.0]; // sqrt(9+16) = 5.0
        agg.compute_and_set_embedding_norm(&hidden);
        assert!((agg.embedding_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_expert_hit_count_out_of_bounds() {
        let agg = TelemetryAggregator::new();
        assert_eq!(agg.expert_hit_count(0), 0);
        assert_eq!(agg.expert_hit_count(999), 0);
    }

    #[test]
    fn test_telemetry_expert_hit_counts_slice_accessor() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 2, hit_count: 10 });
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 4, hit_count: 20 });
        let counts = agg.expert_hit_counts();
        assert_eq!(counts.len(), 5); // resized to index 4 + 1
        assert_eq!(counts[0], 0);
        assert_eq!(counts[2], 10);
        assert_eq!(counts[4], 20);
    }

    #[test]
    fn test_telemetry_expert_hit_count_overwrite() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 1, hit_count: 5 });
        assert_eq!(agg.expert_hit_count(1), 5);
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 1, hit_count: 15 });
        assert_eq!(agg.expert_hit_count(1), 15);
    }

    #[test]
    fn test_telemetry_ingest_all_signal_types_sequential() {
        let mut agg = TelemetryAggregator::new();

        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.1 });
        assert!((agg.dead_neuron_ratio() - 0.1).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 100 });
        assert_eq!(agg.expert_hit_count(0), 100);

        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 1.5 });
        assert!((agg.per_channel_scale() - 1.5).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::SoftmaxSharpness { max_val: 0.5, sharpness: 0.3 });
        assert!((agg.softmax_max() - 0.5).abs() < 1e-6);
        assert!((agg.softmax_sharpness() - 0.3).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 7.0 });
        assert!((agg.embedding_norm() - 7.0).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 0.99 });
        assert!((agg.residual_delta_rho() - 0.99).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 });
        assert!((agg.residual_cosine() - 0.5).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::CentroidPosition { position: 10 });
        assert_eq!(agg.centroid_position(), 10);

        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 2.0 });
        assert!((agg.output_entropy() - 2.0).abs() < 1e-6);
    }

    // ========================================================================
    // GateFirstSkipConfig — Default, Clone
    // ========================================================================

    #[test]
    fn test_gate_first_skip_config_default() {
        let config = GateFirstSkipConfig::default();
        assert!(config.enabled);
        assert!((config.skip_threshold - 0.5).abs() < 1e-6);
        assert!((config.dead_neuron_epsilon - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn test_gate_first_skip_config_clone() {
        let config = GateFirstSkipConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert!((config.skip_threshold - cloned.skip_threshold).abs() < 1e-9);
        assert!((config.dead_neuron_epsilon - cloned.dead_neuron_epsilon).abs() < 1e-9);
    }

    #[test]
    fn test_gate_first_skip_exact_threshold() {
        // At exactly the threshold (0.5): not > 0.5, so MaskedCompute
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(0.5), GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn test_gate_first_skip_quarter_threshold_boundary() {
        // skip_threshold * 0.5 = 0.25 — at exactly 0.25: not > 0.25, so FullCompute
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(0.25), GateSkipDecision::FullCompute);
        assert_eq!(detector.decide(0.25001), GateSkipDecision::MaskedCompute);
    }

    #[test]
    fn test_gate_first_skip_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.7 });

        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(
            detector.decide_from_telemetry(&agg),
            GateSkipDecision::Skip,
        );
    }

    #[test]
    fn test_gate_skip_decision_variants() {
        assert_eq!(GateSkipDecision::FullCompute, GateSkipDecision::FullCompute);
        assert_eq!(GateSkipDecision::MaskedCompute, GateSkipDecision::MaskedCompute);
        assert_eq!(GateSkipDecision::Skip, GateSkipDecision::Skip);
        assert_ne!(GateSkipDecision::FullCompute, GateSkipDecision::Skip);
    }

    // ========================================================================
    // ResidualBypassConfig — Default, Clone, disabled
    // ========================================================================

    #[test]
    fn test_residual_bypass_config_default() {
        let config = ResidualBypassConfig::default();
        assert!(config.enabled);
        assert!((config.delta_rho_threshold - 0.001).abs() < 1e-9);
        assert!((config.cosine_threshold - 0.99).abs() < 1e-9);
        assert_eq!(config.min_skip_layer, 4);
    }

    #[test]
    fn test_residual_bypass_config_clone() {
        let config = ResidualBypassConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert!((config.delta_rho_threshold - cloned.delta_rho_threshold).abs() < 1e-9);
        assert_eq!(config.min_skip_layer, cloned.min_skip_layer);
    }

    #[test]
    fn test_residual_bypass_disabled() {
        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let detector = ResidualBypassDetector::new(config);
        // Even with perfect bypass conditions, disabled means Execute
        assert_eq!(
            detector.decide(10, 1.0, 0.999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_exactly_at_min_skip_layer() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // min_skip_layer=4, so layer 4 is allowed (not < 4)
        assert_eq!(
            detector.decide(4, 1.0, 0.999),
            ResidualBypassDecision::Bypass,
        );
        // layer 3 is blocked
        assert_eq!(
            detector.decide(3, 1.0, 0.999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_energy_only_insufficient() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // Delta_rho stable but cosine too low → Execute
        assert_eq!(
            detector.decide(10, 1.0005, 0.5),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_cosine_only_insufficient() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // Cosine high but delta_rho not stable → Execute
        assert_eq!(
            detector.decide(10, 1.5, 0.999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_decision_variants() {
        assert_eq!(ResidualBypassDecision::Execute, ResidualBypassDecision::Execute);
        assert_eq!(ResidualBypassDecision::Bypass, ResidualBypassDecision::Bypass);
        assert_ne!(ResidualBypassDecision::Execute, ResidualBypassDecision::Bypass);
    }

    // ========================================================================
    // SinkDetectionConfig — Default, Clone
    // ========================================================================

    #[test]
    fn test_sink_detection_config_default() {
        let config = SinkDetectionConfig::default();
        assert!((config.sink_threshold - 0.9).abs() < 1e-6);
        assert_eq!(config.protected_sink_count, 4);
        assert!((config.sharp_focus_threshold - 0.8).abs() < 1e-6);
        assert!((config.diffuse_threshold - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_sink_detection_config_clone() {
        let config = SinkDetectionConfig::default();
        let cloned = config.clone();
        assert!((config.sink_threshold - cloned.sink_threshold).abs() < 1e-9);
        assert_eq!(config.protected_sink_count, cloned.protected_sink_count);
    }

    #[test]
    fn test_sink_detector_sink_priority_over_sharpness() {
        // When max_val > sink_threshold, Sink takes priority regardless of sharpness
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect(0.95, 0.95), AttentionPattern::Sink);
    }

    #[test]
    fn test_sink_detector_boundary_values() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // Exactly at sink threshold: not > 0.9, so check sharpness
        assert_eq!(detector.detect(0.9, 0.85), AttentionPattern::SharpFocus);
        // Exactly at sharp focus threshold: not > 0.8
        assert_eq!(detector.detect(0.5, 0.8), AttentionPattern::Normal);
        // Exactly at diffuse threshold: not < 0.1
        assert_eq!(detector.detect(0.5, 0.1), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_detector_custom_config() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.5,
            protected_sink_count: 2,
            sharp_focus_threshold: 0.6,
            diffuse_threshold: 0.2,
        };
        let detector = SinkDetector::new(config);
        assert_eq!(detector.detect(0.6, 0.5), AttentionPattern::Sink);
        assert_eq!(detector.protected_sink_count(), 2);
    }

    #[test]
    fn test_sink_detector_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.92,
            sharpness: 0.3,
        });

        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect_from_telemetry(&agg), AttentionPattern::Sink);
    }

    #[test]
    fn test_attention_pattern_variants() {
        assert_eq!(AttentionPattern::Normal, AttentionPattern::Normal);
        assert_eq!(AttentionPattern::Sink, AttentionPattern::Sink);
        assert_eq!(AttentionPattern::SharpFocus, AttentionPattern::SharpFocus);
        assert_eq!(AttentionPattern::Diffuse, AttentionPattern::Diffuse);
        assert_ne!(AttentionPattern::Normal, AttentionPattern::Sink);
    }

    // ========================================================================
    // ExpertThermalTracker — boundary conditions
    // ========================================================================

    #[test]
    fn test_expert_thermal_tracker_zero_experts() {
        let tracker = ExpertThermalTracker::new(0);
        assert_eq!(tracker.hit_rate(0), 0.0);
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Cold);
        assert!(tracker.cold_experts().is_empty());
    }

    #[test]
    fn test_expert_thermal_tracker_out_of_bounds() {
        let tracker = ExpertThermalTracker::new(4);
        assert_eq!(tracker.hit_rate(10), 0.0);
        assert_eq!(tracker.thermal_state(10), ExpertThermalState::Cold);
    }

    #[test]
    fn test_expert_thermal_tracker_initial_warm_state() {
        let tracker = ExpertThermalTracker::new(8);
        // No tokens processed → total_tokens == 0 → Warm
        for i in 0..8 {
            assert_eq!(tracker.thermal_state(i), ExpertThermalState::Warm);
        }
    }

    #[test]
    fn test_expert_thermal_tracker_out_of_bounds_routing() {
        let mut tracker = ExpertThermalTracker::new(4);
        // Routing to an out-of-bounds expert should not panic
        tracker.record_routing(&[99]);
        assert_eq!(tracker.hit_rate(0), 0.0);
    }

    #[test]
    fn test_expert_thermal_tracker_update_from_telemetry() {
        let mut tracker = ExpertThermalTracker::new(4);

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 1, hit_count: 50 });
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 10 });

        tracker.update_from_telemetry(&agg);
        assert_eq!(tracker.hit_rate(1), 0.0); // total_tokens is still 0
        assert_eq!(tracker.hit_rate(3), 0.0);

        // After recording tokens, hit_rate reflects accumulated counts
        for _ in 0..100 {
            tracker.record_routing(&[0]);
        }
        assert!(tracker.hit_rate(1) > 0.0);
    }

    #[test]
    fn test_expert_thermal_state_all_variants() {
        let mut tracker = ExpertThermalTracker::new(4);

        // Make expert 0 hot (> 5% hit rate)
        for _ in 0..100 {
            tracker.record_routing(&[0]);
        }
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Hot);

        // Expert 3 never hit, but zero_streak = 100 which is < cold_zero_streak(100000)
        // and hit_rate is 0 < 0.001 → Cold by rate
        assert_eq!(tracker.thermal_state(3), ExpertThermalState::Cold);
    }

    #[test]
    fn test_expert_thermal_cold_experts_list() {
        let mut tracker = ExpertThermalTracker::new(4);
        // Hit only expert 0 for many rounds, others go cold by zero streak
        tracker.cold_zero_streak = 5;
        for _ in 0..6 {
            tracker.record_routing(&[0]);
        }
        let cold = tracker.cold_experts();
        assert!(!cold.contains(&0));
        assert!(cold.contains(&1));
        assert!(cold.contains(&2));
        assert!(cold.contains(&3));
    }

    #[test]
    fn test_expert_hit_counts_empty_accessor() {
        let agg = TelemetryAggregator::new();
        assert!(agg.expert_hit_counts().is_empty());
    }

    // ========================================================================
    // SpecScheduleSignal — Default, reset, advise_from_telemetry
    // ========================================================================

    #[test]
    fn test_spec_schedule_signal_default() {
        let signal = SpecScheduleSignal::default();
        // Default signal should work — just verify construction via advise
        let mut s = signal;
        assert_eq!(s.advise(1.0, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_advice_variants() {
        assert_eq!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::EnableSpec);
        assert_eq!(SpecScheduleAdvice::StandardDecode, SpecScheduleAdvice::StandardDecode);
        assert_eq!(SpecScheduleAdvice::Fallback, SpecScheduleAdvice::Fallback);
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_schedule_fallback_streak_resets_on_good_acceptance() {
        let mut signal = SpecScheduleSignal::new();

        // Two low-acceptance rounds
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);

        // Good acceptance resets streak
        assert_eq!(signal.advise(1.0, 0.8), SpecScheduleAdvice::EnableSpec);

        // Low again — streak is back to 1, not 3
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        // Third one would be Fallback, proving the streak was reset
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_schedule_from_telemetry() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 1.5 });

        let mut signal = SpecScheduleSignal::new();
        let advice = signal.advise_from_telemetry(&agg, 0.9);
        assert_eq!(advice, SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_reset() {
        let mut signal = SpecScheduleSignal::new();
        // Build up a streak
        signal.advise(1.0, 0.2);
        signal.advise(1.0, 0.2);
        signal.reset();
        // After reset, streak is 0, so 3 more low rounds needed
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.2), SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_schedule_high_entropy_standard() {
        let mut signal = SpecScheduleSignal::new();
        assert_eq!(signal.advise(5.0, 0.8), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_schedule_exact_entropy_threshold() {
        let mut signal = SpecScheduleSignal::new();
        // entropy == 2.0 is NOT < 2.0, so StandardDecode
        assert_eq!(signal.advise(2.0, 0.8), SpecScheduleAdvice::StandardDecode);
        // entropy slightly below threshold
        assert_eq!(signal.advise(1.999, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    // ========================================================================
    // GateFirstSkip with custom config
    // ========================================================================

    #[test]
    fn test_gate_first_skip_custom_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.8,
            dead_neuron_epsilon: 1e-4,
        };
        let detector = GateFirstSkipDetector::new(config);

        assert_eq!(detector.decide(0.3), GateSkipDecision::FullCompute);
        assert_eq!(detector.decide(0.5), GateSkipDecision::MaskedCompute); // 0.5 > 0.4 (0.8*0.5)
        assert_eq!(detector.decide(0.9), GateSkipDecision::Skip);
    }

    // ========================================================================
    // ResidualBypass with custom config
    // ========================================================================

    #[test]
    fn test_residual_bypass_custom_config() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.01,
            cosine_threshold: 0.95,
            min_skip_layer: 2,
        };
        let detector = ResidualBypassDetector::new(config);

        // Layer >= 2, delta_rho within 0.01 of 1.0, cosine > 0.95
        assert_eq!(
            detector.decide(2, 1.005, 0.96),
            ResidualBypassDecision::Bypass,
        );
        // Layer 1 blocked by min_skip_layer
        assert_eq!(
            detector.decide(1, 1.005, 0.96),
            ResidualBypassDecision::Execute,
        );
    }

    // ========================================================================
    // SinkDetector protected sink boundary
    // ========================================================================

    #[test]
    fn test_sink_detector_protected_sink_boundary() {
        let config = SinkDetectionConfig {
            protected_sink_count: 2,
            ..Default::default()
        };
        let detector = SinkDetector::new(config);
        assert!(detector.is_protected_sink(0));
        assert!(detector.is_protected_sink(1));
        assert!(!detector.is_protected_sink(2));
    }

    // ========================================================================
    // ExpertThermalTracker — selective expert hitting
    // ========================================================================

    #[test]
    fn test_expert_thermal_hit_rate_calculation() {
        let mut tracker = ExpertThermalTracker::new(2);
        // Route 200 tokens: expert 0 always hit, expert 1 never hit
        for _ in 0..200 {
            tracker.record_routing(&[0]);
        }
        assert!((tracker.hit_rate(0) - 1.0).abs() < 1e-6);
        assert_eq!(tracker.hit_rate(1), 0.0);
    }

    #[test]
    fn test_expert_thermal_multiple_experts_per_routing() {
        let mut tracker = ExpertThermalTracker::new(4);
        for _ in 0..100 {
            tracker.record_routing(&[0, 1]);
        }
        // Both experts 0 and 1 should have high hit rates
        assert!(tracker.hit_rate(0) > 0.05);
        assert!(tracker.hit_rate(1) > 0.05);
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Hot);
        assert_eq!(tracker.thermal_state(1), ExpertThermalState::Hot);
    }

    // ========================================================================
    // New tests — variants, Debug traits, boundary values, special floats
    // ========================================================================

    #[test]
    fn test_epilogue_signal_debug_output_all_variants() {
        // Verify Debug output is non-empty and contains variant names
        let s = EpilogueSignal::DeadNeuronRatio { ratio: 0.5 };
        assert!(format!("{:?}", s).contains("DeadNeuronRatio"));

        let s = EpilogueSignal::ExpertHitCount { expert_id: 7, hit_count: 100 };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("ExpertHitCount"));
        assert!(dbg.contains("7"));
        assert!(dbg.contains("100"));

        let s = EpilogueSignal::RowActivationStats { l1_norm: 1.2, max_val: 3.4 };
        assert!(format!("{:?}", s).contains("RowActivationStats"));

        let s = EpilogueSignal::PerChannelScale { scale: 9.9 };
        assert!(format!("{:?}", s).contains("PerChannelScale"));

        let s = EpilogueSignal::SoftmaxSharpness { max_val: 0.8, sharpness: 0.6 };
        assert!(format!("{:?}", s).contains("SoftmaxSharpness"));

        let s = EpilogueSignal::EmbeddingNorm { norm: 42.0 };
        assert!(format!("{:?}", s).contains("EmbeddingNorm"));

        let s = EpilogueSignal::ResidualDeltaRho { delta_rho: 0.99 };
        assert!(format!("{:?}", s).contains("ResidualDeltaRho"));

        let s = EpilogueSignal::ResidualCosineSimilarity { cosine: 0.98 };
        assert!(format!("{:?}", s).contains("ResidualCosineSimilarity"));

        let s = EpilogueSignal::CentroidPosition { position: 15 };
        assert!(format!("{:?}", s).contains("CentroidPosition"));

        let s = EpilogueSignal::OutputEntropy { entropy: 2.5 };
        assert!(format!("{:?}", s).contains("OutputEntropy"));
    }

    #[test]
    fn test_epilogue_signal_partial_eq_same_variant_different_values() {
        // Same variant, same values → equal
        assert_eq!(
            EpilogueSignal::DeadNeuronRatio { ratio: 0.3 },
            EpilogueSignal::DeadNeuronRatio { ratio: 0.3 },
        );
        // Same variant, different values → not equal
        assert_ne!(
            EpilogueSignal::DeadNeuronRatio { ratio: 0.3 },
            EpilogueSignal::DeadNeuronRatio { ratio: 0.4 },
        );
        // Different variants → not equal
        assert_ne!(
            EpilogueSignal::DeadNeuronRatio { ratio: 0.0 },
            EpilogueSignal::PerChannelScale { scale: 0.0 },
        );
    }

    #[test]
    fn test_telemetry_aggregator_new_eq_default() {
        // new() and default() must produce identical state
        let from_new = TelemetryAggregator::new();
        let from_default = TelemetryAggregator::default();
        assert_eq!(from_new.dead_neuron_ratio(), from_default.dead_neuron_ratio());
        assert_eq!(from_new.softmax_sharpness(), from_default.softmax_sharpness());
        assert_eq!(from_new.softmax_max(), from_default.softmax_max());
        assert_eq!(from_new.residual_delta_rho(), from_default.residual_delta_rho());
        assert_eq!(from_new.residual_cosine(), from_default.residual_cosine());
        assert_eq!(from_new.output_entropy(), from_default.output_entropy());
        assert_eq!(from_new.per_channel_scale(), from_default.per_channel_scale());
        assert_eq!(from_new.embedding_norm(), from_default.embedding_norm());
        assert_eq!(from_new.centroid_position(), from_default.centroid_position());
    }

    #[test]
    fn test_telemetry_accessors_on_fresh_aggregator() {
        // All accessors on a freshly constructed aggregator return zero/empty
        let agg = TelemetryAggregator::new();
        assert_eq!(agg.dead_neuron_ratio(), 0.0);
        assert_eq!(agg.softmax_sharpness(), 0.0);
        assert_eq!(agg.softmax_max(), 0.0);
        assert_eq!(agg.residual_delta_rho(), 0.0);
        assert_eq!(agg.residual_cosine(), 0.0);
        assert_eq!(agg.output_entropy(), 0.0);
        assert_eq!(agg.per_channel_scale(), 0.0);
        assert_eq!(agg.embedding_norm(), 0.0);
        assert_eq!(agg.centroid_position(), 0);
        assert!(agg.expert_hit_counts().is_empty());
        assert_eq!(agg.expert_hit_count(0), 0);
        assert_eq!(agg.expert_hit_count(u32::MAX), 0);
    }

    #[test]
    fn test_telemetry_aggregator_debug_output() {
        let agg = TelemetryAggregator::new();
        let debug_str = format!("{:?}", agg);
        assert!(debug_str.contains("TelemetryAggregator"));
    }

    #[test]
    fn test_compute_l2_norm_negative_values() {
        // sqrt((-3)^2 + (-4)^2) = sqrt(9+16) = 5.0
        let norm = compute_l2_norm(&[-3.0, -4.0]);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_mixed_signs() {
        // sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
        let norm = compute_l2_norm(&[1.0, -2.0, 3.0, -4.0]);
        let expected = (30.0_f32).sqrt();
        assert!((norm - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compute_l2_norm_very_large_values() {
        // f32::MAX squared overflows to infinity, sqrt(infinity) = infinity
        let norm = compute_l2_norm(&[f32::MAX]);
        assert!(norm.is_infinite() && norm.is_sign_positive());
    }

    #[test]
    fn test_compute_l2_norm_inf_input() {
        // Infinity squared is infinity, sqrt(infinity) is infinity
        let norm = compute_l2_norm(&[f32::INFINITY]);
        assert!(norm.is_infinite());
    }

    #[test]
    fn test_compute_l2_norm_nan_input() {
        // NaN propagation: any NaN in input → NaN output
        let norm = compute_l2_norm(&[1.0, f32::NAN, 2.0]);
        assert!(norm.is_nan());
    }

    #[test]
    fn test_gate_skip_decision_debug_clone_copy() {
        // Verify Debug output contains variant names
        assert!(format!("{:?}", GateSkipDecision::FullCompute).contains("FullCompute"));
        assert!(format!("{:?}", GateSkipDecision::MaskedCompute).contains("MaskedCompute"));
        assert!(format!("{:?}", GateSkipDecision::Skip).contains("Skip"));

        // Verify Copy/Clone
        let decision = GateSkipDecision::MaskedCompute;
        let copied = decision;
        assert_eq!(decision, copied);
    }

    #[test]
    fn test_gate_first_skip_decide_with_nan() {
        // NaN comparison: NaN > 0.5 is false, NaN > 0.25 is false → FullCompute
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(f32::NAN), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_decide_negative_ratio() {
        // Negative ratio: -0.5 > 0.5 is false, -0.5 > 0.25 is false → FullCompute
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(-0.5), GateSkipDecision::FullCompute);
        assert_eq!(detector.decide(-1.0), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_decide_one() {
        // dead_ratio = 1.0 → 1.0 > 0.5 → Skip
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(1.0), GateSkipDecision::Skip);
    }

    #[test]
    fn test_gate_first_skip_detector_config_field_access() {
        // Verify that GateFirstSkipDetector exposes config field and can be read
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.75,
            dead_neuron_epsilon: 0.005,
        };
        let detector = GateFirstSkipDetector::new(config);
        assert!(detector.config.enabled);
        assert!((detector.config.skip_threshold - 0.75).abs() < 1e-6);
        assert!((detector.config.dead_neuron_epsilon - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_residual_bypass_decision_debug_clone_copy() {
        assert!(format!("{:?}", ResidualBypassDecision::Execute).contains("Execute"));
        assert!(format!("{:?}", ResidualBypassDecision::Bypass).contains("Bypass"));

        let decision = ResidualBypassDecision::Bypass;
        let copied = decision;
        assert_eq!(decision, copied);
    }

    #[test]
    fn test_attention_pattern_debug_clone_copy() {
        for pattern in &[
            AttentionPattern::Normal,
            AttentionPattern::Sink,
            AttentionPattern::SharpFocus,
            AttentionPattern::Diffuse,
        ] {
            let debug = format!("{:?}", pattern);
            assert!(!debug.is_empty());
            let copied = *pattern;
            assert_eq!(*pattern, copied);
        }
    }

    #[test]
    fn test_expert_thermal_state_debug_clone_copy() {
        // Verify Debug output for all three variants
        assert!(format!("{:?}", ExpertThermalState::Hot).contains("Hot"));
        assert!(format!("{:?}", ExpertThermalState::Warm).contains("Warm"));
        assert!(format!("{:?}", ExpertThermalState::Cold).contains("Cold"));

        // Verify Copy + Clone + PartialEq
        let state = ExpertThermalState::Warm;
        let copied = state;
        assert_eq!(state, copied);
        assert_ne!(state, ExpertThermalState::Hot);
        assert_ne!(state, ExpertThermalState::Cold);
    }

    #[test]
    fn test_spec_schedule_advice_debug_clone_copy() {
        assert!(format!("{:?}", SpecScheduleAdvice::EnableSpec).contains("EnableSpec"));
        assert!(format!("{:?}", SpecScheduleAdvice::StandardDecode).contains("StandardDecode"));
        assert!(format!("{:?}", SpecScheduleAdvice::Fallback).contains("Fallback"));

        let advice = SpecScheduleAdvice::StandardDecode;
        let copied = advice;
        assert_eq!(advice, copied);
    }

    #[test]
    fn test_telemetry_expert_hit_count_large_expert_id() {
        // Ingest a signal with a large expert_id, verify vec resizing and accessor
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount {
            expert_id: 100,
            hit_count: 77,
        });
        assert_eq!(agg.expert_hit_count(100), 77);
        // All indices before 100 should be 0
        assert_eq!(agg.expert_hit_count(0), 0);
        assert_eq!(agg.expert_hit_count(50), 0);
        // Index beyond 100 should also be 0
        assert_eq!(agg.expert_hit_count(101), 0);
        assert_eq!(agg.expert_hit_counts().len(), 101);
    }

    #[test]
    fn test_compute_l2_norm_all_zeros_large_slice() {
        // Larger all-zero slice still yields 0.0
        let zeros = vec![0.0f32; 1000];
        assert_eq!(compute_l2_norm(&zeros), 0.0);
    }

    #[test]
    fn test_telemetry_ingest_overwrites_previous() {
        // Ingesting the same signal type twice overwrites the value
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 0.5 });
        assert!((agg.residual_delta_rho() - 0.5).abs() < 1e-6);
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.2 });
        assert!((agg.residual_delta_rho() - 1.2).abs() < 1e-6);
    }

    // ========================================================================
    // NEW TESTS — uncovered areas
    // ========================================================================

    // ---- TelemetryAggregator: ingest page header with default header ----

    #[test]
    fn test_telemetry_ingest_from_default_page_header() {
        // A freshly created KvPageHeader has all-zero signal fields
        let mut agg = TelemetryAggregator::new();
        let header = KvPageHeader::new(0);
        agg.ingest_from_page_header(&header);
        assert_eq!(agg.dead_neuron_ratio(), 0.0);
        assert_eq!(agg.softmax_max(), 0.0);
        assert_eq!(agg.output_entropy(), 0.0);
    }

    #[test]
    fn test_telemetry_ingest_from_page_header_overwrites_signal() {
        // Page header ingestion overwrites previously ingested signal values
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.9 });
        assert!((agg.dead_neuron_ratio() - 0.9).abs() < 1e-6);

        let header = KvPageHeader::new(0);
        agg.ingest_from_page_header(&header);
        assert_eq!(agg.dead_neuron_ratio(), 0.0);
    }

    #[test]
    fn test_telemetry_ingest_from_page_header_with_delta_rho() {
        let mut agg = TelemetryAggregator::new();
        let mut header = KvPageHeader::new(0);
        header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(1.005);
        agg.ingest_from_page_header(&header);
        assert!((agg.residual_delta_rho() - 1.005).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_ingest_from_page_header_with_centroid_pos() {
        let mut agg = TelemetryAggregator::new();
        let mut header = KvPageHeader::new(0);
        header.centroid_pos = crate::kv_cache::f32_to_f16_bits(0.55);
        agg.ingest_from_page_header(&header);
        assert!((agg.softmax_sharpness() - 0.55).abs() < 0.02);
    }

    #[test]
    fn test_telemetry_compute_and_set_embedding_norm_empty() {
        // Empty hidden slice → norm = 0.0
        let mut agg = TelemetryAggregator::new();
        agg.compute_and_set_embedding_norm(&[]);
        assert_eq!(agg.embedding_norm(), 0.0);
    }

    #[test]
    fn test_telemetry_compute_and_set_embedding_norm_single() {
        let mut agg = TelemetryAggregator::new();
        agg.compute_and_set_embedding_norm(&[5.0]);
        assert!((agg.embedding_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_set_embedding_norm_then_ingest_signal() {
        // set_embedding_norm overrides by direct setter, then signal ingest overwrites again
        let mut agg = TelemetryAggregator::new();
        agg.set_embedding_norm(10.0);
        assert!((agg.embedding_norm() - 10.0).abs() < 1e-6);
        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 3.0 });
        assert!((agg.embedding_norm() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_expert_hit_count_resize_preserves_old() {
        // Ingest expert 0, then expert 5 — expert 0 value preserved
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 11 });
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 5, hit_count: 22 });
        assert_eq!(agg.expert_hit_count(0), 11);
        assert_eq!(agg.expert_hit_count(5), 22);
        // Indices 1-4 are zero-filled during resize
        assert_eq!(agg.expert_hit_count(1), 0);
        assert_eq!(agg.expert_hit_count(4), 0);
    }

    #[test]
    fn test_telemetry_expert_hit_count_multiple_resizes() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 2, hit_count: 5 });
        assert_eq!(agg.expert_hit_counts().len(), 3);
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 7, hit_count: 8 });
        assert_eq!(agg.expert_hit_counts().len(), 8);
        assert_eq!(agg.expert_hit_count(2), 5);
        assert_eq!(agg.expert_hit_count(7), 8);
    }

    #[test]
    fn test_telemetry_aggregator_multiple_signal_interactions() {
        // Ingest multiple different signals and verify each is stored independently
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.33 });
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 4.56 });
        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 7.89 });
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 99 });

        assert!((agg.dead_neuron_ratio() - 0.33).abs() < 1e-6);
        assert!((agg.output_entropy() - 4.56).abs() < 1e-6);
        assert!((agg.per_channel_scale() - 7.89).abs() < 1e-6);
        assert_eq!(agg.centroid_position(), 99);
        // Others remain at default
        assert_eq!(agg.softmax_sharpness(), 0.0);
        assert_eq!(agg.residual_delta_rho(), 0.0);
    }

    #[test]
    fn test_telemetry_aggregator_debug_has_all_fields() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 1.23 });
        let debug = format!("{:?}", agg);
        // Debug output should contain the struct name and be non-empty
        assert!(debug.contains("TelemetryAggregator"));
        assert!(debug.len() > 20);
    }

    // ---- compute_l2_norm: additional edge cases ----

    #[test]
    fn test_compute_l2_norm_very_small_values() {
        // Very small values should not underflow to zero
        let norm = compute_l2_norm(&[1e-20, 1e-20]);
        let expected = (2.0_f32 * 1e-20 * 1e-20).sqrt();
        assert!((norm - expected).abs() < expected * 0.01);
    }

    #[test]
    fn test_compute_l2_norm_negative_infinity() {
        let norm = compute_l2_norm(&[f32::NEG_INFINITY]);
        assert!(norm.is_infinite() && norm.is_sign_positive());
    }

    #[test]
    fn test_compute_l2_norm_single_zero() {
        assert_eq!(compute_l2_norm(&[0.0]), 0.0);
    }

    #[test]
    fn test_compute_l2_norm_two_elements() {
        // 3-4-5 triangle
        let norm = compute_l2_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_l2_norm_large_slice_sum_overflow() {
        // Many large values → sum of squares overflows to infinity
        let values = vec![1e30_f32; 10];
        let norm = compute_l2_norm(&values);
        assert!(norm.is_infinite());
    }

    // ---- GateFirstSkip: additional edge cases ----

    #[test]
    fn test_gate_first_skip_very_small_threshold() {
        let config = GateFirstSkipConfig {
            enabled: true,
            skip_threshold: 0.01,
            dead_neuron_epsilon: 1e-6,
        };
        let detector = GateFirstSkipDetector::new(config);
        // 0.006 > 0.005 (0.01 * 0.5) → MaskedCompute
        assert_eq!(detector.decide(0.006), GateSkipDecision::MaskedCompute);
        // 0.02 > 0.01 → Skip
        assert_eq!(detector.decide(0.02), GateSkipDecision::Skip);
        // 0.003 not > 0.005 → FullCompute
        assert_eq!(detector.decide(0.003), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_zero_dead_ratio() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(0.0), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_infinity_ratio() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(f32::INFINITY), GateSkipDecision::Skip);
    }

    #[test]
    fn test_gate_first_skip_negative_infinity_ratio() {
        let detector = GateFirstSkipDetector::new(GateFirstSkipConfig::default());
        assert_eq!(detector.decide(f32::NEG_INFINITY), GateSkipDecision::FullCompute);
    }

    #[test]
    fn test_gate_first_skip_config_default_epsilon() {
        let config = GateFirstSkipConfig::default();
        assert!((config.dead_neuron_epsilon - 1e-3).abs() < 1e-9);
    }

    // ---- ResidualBypass: additional edge cases ----

    #[test]
    fn test_residual_bypass_delta_rho_exactly_at_threshold_edge() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // delta_rho = 1.0 + threshold (0.001) → |1.001 - 1.0| = 0.001, NOT < 0.001
        assert_eq!(
            detector.decide(10, 1.001, 0.999),
            ResidualBypassDecision::Execute,
        );
        // delta_rho = 1.0 + threshold - epsilon → just under
        assert_eq!(
            detector.decide(10, 0.9991, 0.999),
            ResidualBypassDecision::Bypass,
        );
    }

    #[test]
    fn test_residual_bypass_cosine_exactly_at_threshold() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // cosine = 0.99 exactly: 0.99 > 0.99 is false → Execute
        assert_eq!(
            detector.decide(10, 1.0, 0.99),
            ResidualBypassDecision::Execute,
        );
        // cosine just above threshold
        assert_eq!(
            detector.decide(10, 1.0, 0.9901),
            ResidualBypassDecision::Bypass,
        );
    }

    #[test]
    fn test_residual_bypass_zero_delta_rho() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // delta_rho = 0.0, |0.0 - 1.0| = 1.0, NOT < 0.001
        assert_eq!(
            detector.decide(10, 0.0, 0.999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_very_large_delta_rho() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        assert_eq!(
            detector.decide(10, 100.0, 0.999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_nan_inputs() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // NaN comparisons: NaN.abs() < threshold is false, NaN > cosine_threshold is false
        assert_eq!(
            detector.decide(10, f32::NAN, 0.999),
            ResidualBypassDecision::Execute,
        );
        assert_eq!(
            detector.decide(10, 1.0, f32::NAN),
            ResidualBypassDecision::Execute,
        );
        assert_eq!(
            detector.decide(10, f32::NAN, f32::NAN),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_layer_zero() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        // min_skip_layer=4, layer 0 is always Execute
        assert_eq!(
            detector.decide(0, 1.0, 0.9999),
            ResidualBypassDecision::Execute,
        );
    }

    #[test]
    fn test_residual_bypass_large_layer_number() {
        let detector = ResidualBypassDetector::new(ResidualBypassConfig::default());
        assert_eq!(
            detector.decide(1000, 1.0, 0.999),
            ResidualBypassDecision::Bypass,
        );
    }

    // ---- SinkDetector: additional edge cases ----

    #[test]
    fn test_sink_detector_all_zeros() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // max_val=0, sharpness=0 → Diffuse (no dominant attention)
        assert_eq!(detector.detect(0.0, 0.0), AttentionPattern::Diffuse);
    }

    #[test]
    fn test_sink_detector_max_val_just_below_sink() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // 0.8999 < 0.9 → not Sink
        assert_eq!(detector.detect(0.8999, 0.9), AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_sink_detector_sharpness_just_above_diffuse() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // sharpness = 0.1001 → not < 0.1, not > 0.8 → Normal
        assert_eq!(detector.detect(0.3, 0.1001), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_detector_nan_max_val() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // NaN > 0.9 is false, NaN > 0.8 is false, NaN < 0.1 is false → Normal
        assert_eq!(detector.detect(f32::NAN, 0.5), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_detector_nan_sharpness() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // max_val = 0.5 < 0.9 → not Sink; NaN > 0.8 is false, NaN < 0.1 is false → Normal
        assert_eq!(detector.detect(0.5, f32::NAN), AttentionPattern::Normal);
    }

    #[test]
    fn test_sink_detector_infinity_max_val() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // Infinity > 0.9 → Sink
        assert_eq!(detector.detect(f32::INFINITY, 0.5), AttentionPattern::Sink);
    }

    #[test]
    fn test_sink_detector_negative_max_val() {
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        // Negative max_val → not Sink, check sharpness
        assert_eq!(detector.detect(-1.0, 0.9), AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_sink_detector_protected_sink_zero_count() {
        let config = SinkDetectionConfig {
            protected_sink_count: 0,
            ..Default::default()
        };
        let detector = SinkDetector::new(config);
        assert!(!detector.is_protected_sink(0));
        assert_eq!(detector.protected_sink_count(), 0);
    }

    #[test]
    fn test_sink_detector_config_clone_preserves_all_fields() {
        let config = SinkDetectionConfig {
            sink_threshold: 0.7,
            protected_sink_count: 8,
            sharp_focus_threshold: 0.5,
            diffuse_threshold: 0.15,
        };
        let cloned = config.clone();
        assert!((cloned.sink_threshold - 0.7).abs() < 1e-9);
        assert_eq!(cloned.protected_sink_count, 8);
        assert!((cloned.sharp_focus_threshold - 0.5).abs() < 1e-9);
        assert!((cloned.diffuse_threshold - 0.15).abs() < 1e-9);
    }

    // ---- ExpertThermalTracker: additional edge cases ----

    #[test]
    fn test_expert_thermal_tracker_single_expert() {
        let mut tracker = ExpertThermalTracker::new(1);
        for _ in 0..50 {
            tracker.record_routing(&[0]);
        }
        assert!((tracker.hit_rate(0) - 1.0).abs() < 1e-6);
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Hot);
    }

    #[test]
    fn test_expert_thermal_tracker_routing_empty_selection() {
        let mut tracker = ExpertThermalTracker::new(4);
        tracker.record_routing(&[]);
        assert_eq!(tracker.total_tokens, 1);
        // All zero_streaks incremented
        for i in 0..4 {
            assert_eq!(tracker.zero_streak[i], 1);
        }
    }

    #[test]
    fn test_expert_thermal_tracker_repeated_routing_same_expert() {
        let mut tracker = ExpertThermalTracker::new(4);
        for _ in 0..1000 {
            tracker.record_routing(&[2]);
        }
        assert!((tracker.hit_rate(2) - 1.0).abs() < 1e-6);
        assert_eq!(tracker.thermal_state(2), ExpertThermalState::Hot);
        // Expert 0,1,3 never hit but only 1000 tokens < cold_zero_streak(100000)
        // hit_rate for them is 0 < 0.001 → Cold
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Cold);
        assert_eq!(tracker.thermal_state(1), ExpertThermalState::Cold);
        assert_eq!(tracker.thermal_state(3), ExpertThermalState::Cold);
    }

    #[test]
    fn test_expert_thermal_tracker_update_from_telemetry_with_zero_counts() {
        // Zero-count signals in telemetry should NOT reset zero_streak
        let mut tracker = ExpertThermalTracker::new(4);
        tracker.cold_zero_streak = 5;

        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 0 });
        // hit_count=0 is NOT > 0, so it should not reset zero_streak
        for _ in 0..6 {
            tracker.record_routing(&[1]);
        }
        tracker.update_from_telemetry(&agg);
        // Expert 0 should still be cold (zero_streak from record_routing)
        assert_eq!(tracker.thermal_state(0), ExpertThermalState::Cold);
    }

    #[test]
    fn test_expert_thermal_tracker_update_from_telemetry_oob_expert() {
        let mut tracker = ExpertThermalTracker::new(4);
        let mut agg = TelemetryAggregator::new();
        // Expert 10 is out of tracker bounds (tracker has 4 experts)
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 10, hit_count: 50 });
        tracker.update_from_telemetry(&agg);
        // Should not panic, and hit counts for valid experts remain 0
        assert_eq!(tracker.hit_rate(0), 0.0);
        assert_eq!(tracker.hit_rate(3), 0.0);
    }

    #[test]
    fn test_expert_thermal_tracker_hit_rate_no_tokens() {
        let tracker = ExpertThermalTracker::new(4);
        // total_tokens == 0 → hit_rate returns 0.0
        assert_eq!(tracker.hit_rate(0), 0.0);
        assert_eq!(tracker.hit_rate(3), 0.0);
    }

    #[test]
    fn test_expert_thermal_tracker_cold_then_warm_again() {
        let mut tracker = ExpertThermalTracker::new(4);
        tracker.cold_zero_streak = 5;

        // Expert 3 goes cold
        for _ in 0..6 {
            tracker.record_routing(&[0]);
        }
        assert_eq!(tracker.thermal_state(3), ExpertThermalState::Cold);

        // Now start hitting expert 3 — zero_streak resets
        for _ in 0..20 {
            tracker.record_routing(&[3]);
        }
        // zero_streak was reset, and hit_rate > 0.05 → Hot
        assert_eq!(tracker.thermal_state(3), ExpertThermalState::Hot);
    }

    #[test]
    fn test_expert_thermal_state_ordering() {
        // Verify all pairwise comparisons
        assert_ne!(ExpertThermalState::Hot, ExpertThermalState::Warm);
        assert_ne!(ExpertThermalState::Hot, ExpertThermalState::Cold);
        assert_ne!(ExpertThermalState::Warm, ExpertThermalState::Cold);
    }

    // ---- SpecScheduleSignal: additional edge cases ----

    #[test]
    fn test_spec_schedule_acceptance_exactly_at_boundary() {
        let mut signal = SpecScheduleSignal::new();
        // acceptance_rate = 0.3: NOT < 0.3, so it goes to entropy check
        assert_eq!(signal.advise(1.0, 0.3), SpecScheduleAdvice::EnableSpec);
        // acceptance_rate = 0.2999: < 0.3 → streak increments
        assert_eq!(signal.advise(1.0, 0.2999), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_schedule_entropy_zero() {
        let mut signal = SpecScheduleSignal::new();
        // entropy = 0.0: < 2.0 → EnableSpec
        assert_eq!(signal.advise(0.0, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_negative_entropy() {
        let mut signal = SpecScheduleSignal::new();
        // Negative entropy is mathematically impossible but code handles it
        assert_eq!(signal.advise(-1.0, 0.8), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_nan_entropy() {
        let mut signal = SpecScheduleSignal::new();
        // NaN < 2.0 is false → StandardDecode
        assert_eq!(signal.advise(f32::NAN, 0.8), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_schedule_nan_acceptance_rate() {
        let mut signal = SpecScheduleSignal::new();
        // NaN < 0.3 is false → goes to entropy check, streak resets to 0
        assert_eq!(signal.advise(1.0, f32::NAN), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_zero_acceptance_rate() {
        let mut signal = SpecScheduleSignal::new();
        // 0.0 < 0.3 → streak increments
        assert_eq!(signal.advise(1.0, 0.0), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.0), SpecScheduleAdvice::StandardDecode);
        assert_eq!(signal.advise(1.0, 0.0), SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_schedule_fallback_then_reset_then_fallback_again() {
        let mut signal = SpecScheduleSignal::new();
        // Hit fallback
        signal.advise(1.0, 0.1);
        signal.advise(1.0, 0.1);
        let advice = signal.advise(1.0, 0.1);
        assert_eq!(advice, SpecScheduleAdvice::Fallback);

        // Reset and build up again
        signal.reset();
        signal.advise(1.0, 0.1);
        signal.advise(1.0, 0.1);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_spec_schedule_infinity_acceptance_rate() {
        let mut signal = SpecScheduleSignal::new();
        // Infinity < 0.3 is false → goes to entropy check
        assert_eq!(signal.advise(1.0, f32::INFINITY), SpecScheduleAdvice::EnableSpec);
    }

    #[test]
    fn test_spec_schedule_negative_acceptance_rate() {
        let mut signal = SpecScheduleSignal::new();
        // -0.5 < 0.3 → streak increments
        assert_eq!(signal.advise(1.0, -0.5), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_spec_schedule_fallback_continues_after_threshold() {
        let mut signal = SpecScheduleSignal::new();
        // Reach fallback
        signal.advise(1.0, 0.1);
        signal.advise(1.0, 0.1);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);
        // Still in low acceptance: streak >= 3, keeps returning Fallback
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);
    }

    // ---- EpilogueSignal: PartialEq edge cases for all variants ----

    #[test]
    fn test_signal_partial_eq_expert_hit_count() {
        assert_eq!(
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 10 },
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 10 },
        );
        assert_ne!(
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 10 },
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 11 },
        );
        assert_ne!(
            EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 10 },
            EpilogueSignal::ExpertHitCount { expert_id: 4, hit_count: 10 },
        );
    }

    #[test]
    fn test_signal_partial_eq_row_activation_stats() {
        assert_eq!(
            EpilogueSignal::RowActivationStats { l1_norm: 1.0, max_val: 2.0 },
            EpilogueSignal::RowActivationStats { l1_norm: 1.0, max_val: 2.0 },
        );
        assert_ne!(
            EpilogueSignal::RowActivationStats { l1_norm: 1.0, max_val: 2.0 },
            EpilogueSignal::RowActivationStats { l1_norm: 1.1, max_val: 2.0 },
        );
    }

    #[test]
    fn test_signal_partial_eq_softmax_sharpness() {
        assert_eq!(
            EpilogueSignal::SoftmaxSharpness { max_val: 0.5, sharpness: 0.6 },
            EpilogueSignal::SoftmaxSharpness { max_val: 0.5, sharpness: 0.6 },
        );
        assert_ne!(
            EpilogueSignal::SoftmaxSharpness { max_val: 0.5, sharpness: 0.6 },
            EpilogueSignal::SoftmaxSharpness { max_val: 0.5, sharpness: 0.7 },
        );
    }

    #[test]
    fn test_signal_partial_eq_residual_delta_rho() {
        assert_eq!(
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.99 },
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.99 },
        );
        assert_ne!(
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.99 },
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.98 },
        );
    }

    #[test]
    fn test_signal_partial_eq_residual_cosine() {
        assert_eq!(
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 },
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 },
        );
        assert_ne!(
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.5 },
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.6 },
        );
    }

    #[test]
    fn test_signal_partial_eq_centroid_position() {
        assert_eq!(
            EpilogueSignal::CentroidPosition { position: 5 },
            EpilogueSignal::CentroidPosition { position: 5 },
        );
        assert_ne!(
            EpilogueSignal::CentroidPosition { position: 5 },
            EpilogueSignal::CentroidPosition { position: 6 },
        );
    }

    #[test]
    fn test_signal_partial_eq_output_entropy() {
        assert_eq!(
            EpilogueSignal::OutputEntropy { entropy: 2.0 },
            EpilogueSignal::OutputEntropy { entropy: 2.0 },
        );
        assert_ne!(
            EpilogueSignal::OutputEntropy { entropy: 2.0 },
            EpilogueSignal::OutputEntropy { entropy: 2.1 },
        );
    }

    #[test]
    fn test_signal_cross_variant_inequality() {
        // Every pair of different variants must be unequal
        let signals: Vec<EpilogueSignal> = vec![
            EpilogueSignal::DeadNeuronRatio { ratio: 0.0 },
            EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 0 },
            EpilogueSignal::RowActivationStats { l1_norm: 0.0, max_val: 0.0 },
            EpilogueSignal::PerChannelScale { scale: 0.0 },
            EpilogueSignal::SoftmaxSharpness { max_val: 0.0, sharpness: 0.0 },
            EpilogueSignal::EmbeddingNorm { norm: 0.0 },
            EpilogueSignal::ResidualDeltaRho { delta_rho: 0.0 },
            EpilogueSignal::ResidualCosineSimilarity { cosine: 0.0 },
            EpilogueSignal::CentroidPosition { position: 0 },
            EpilogueSignal::OutputEntropy { entropy: 0.0 },
        ];
        for i in 0..signals.len() {
            for j in (i + 1)..signals.len() {
                assert_ne!(signals[i], signals[j], "variants {} and {} should differ", i, j);
            }
        }
    }

    // ---- ResidualBypassConfig: additional coverage ----

    #[test]
    fn test_residual_bypass_config_min_skip_layer_zero() {
        let config = ResidualBypassConfig {
            enabled: true,
            delta_rho_threshold: 0.01,
            cosine_threshold: 0.95,
            min_skip_layer: 0,
        };
        let detector = ResidualBypassDetector::new(config);
        // min_skip_layer=0 → even layer 0 can be bypassed
        assert_eq!(
            detector.decide(0, 1.0, 0.999),
            ResidualBypassDecision::Bypass,
        );
    }

    #[test]
    fn test_residual_bypass_from_telemetry_with_disabled() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 1.0 });
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.999 });

        let config = ResidualBypassConfig {
            enabled: false,
            ..Default::default()
        };
        let detector = ResidualBypassDetector::new(config);
        assert_eq!(
            detector.decide_from_telemetry(10, &agg),
            ResidualBypassDecision::Execute,
        );
    }

    // ---- SinkDetector: from telemetry with custom config ----

    #[test]
    fn test_sink_detector_from_telemetry_sharp_focus() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.5,
            sharpness: 0.85,
        });
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect_from_telemetry(&agg), AttentionPattern::SharpFocus);
    }

    #[test]
    fn test_sink_detector_from_telemetry_diffuse() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.1,
            sharpness: 0.05,
        });
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect_from_telemetry(&agg), AttentionPattern::Diffuse);
    }

    #[test]
    fn test_sink_detector_from_telemetry_normal() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.3,
            sharpness: 0.4,
        });
        let detector = SinkDetector::new(SinkDetectionConfig::default());
        assert_eq!(detector.detect_from_telemetry(&agg), AttentionPattern::Normal);
    }

    // ========================================================================
    // Additional edge-case tests
    // ========================================================================

    #[test]
    fn test_telemetry_aggregator_clone_preserves_expert_counts() {
        // Arrange: populate expert hit counts
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 3, hit_count: 42 });

        // Act: clone
        let cloned = agg.clone();

        // Assert: expert counts slice is preserved
        assert_eq!(cloned.expert_hit_counts().len(), 4);
        assert_eq!(cloned.expert_hit_count(3), 42);
        assert_eq!(cloned.expert_hit_count(0), 0);
    }

    #[test]
    fn test_telemetry_set_embedding_norm_special_floats() {
        // Arrange
        let mut agg = TelemetryAggregator::new();

        // Act & Assert: infinity
        agg.set_embedding_norm(f32::INFINITY);
        assert!(agg.embedding_norm().is_infinite() && agg.embedding_norm().is_sign_positive());

        // Act & Assert: NaN
        agg.set_embedding_norm(f32::NAN);
        assert!(agg.embedding_norm().is_nan());
    }

    #[test]
    fn test_compute_l2_norm_subnormal_squares_underflow() {
        // Arrange: smallest positive subnormal — its square underflows to zero in f32
        let tiny = f32::from_bits(1u32);
        assert!(tiny.is_subnormal()); // confirm it is subnormal

        // Act: sum of squares underflows, norm = sqrt(0.0) = 0.0
        let norm = compute_l2_norm(&[tiny, tiny]);

        // Assert: f32 arithmetic loses subnormal squares
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_expert_thermal_record_routing_duplicate_experts() {
        // Arrange
        let mut tracker = ExpertThermalTracker::new(4);

        // Act: same expert listed twice in one routing call
        tracker.record_routing(&[1, 1]);

        // Assert: expert 1 hit_count incremented twice
        assert_eq!(tracker.hit_counts[1], 2);
        // total_tokens incremented once per call, not per expert
        assert_eq!(tracker.total_tokens, 1);
    }

    #[test]
    fn test_expert_thermal_total_tokens_increments_per_call() {
        // Arrange
        let mut tracker = ExpertThermalTracker::new(4);

        // Act
        tracker.record_routing(&[0]);
        tracker.record_routing(&[1, 2]);
        tracker.record_routing(&[]);

        // Assert: 3 calls → total_tokens = 3
        assert_eq!(tracker.total_tokens, 3);
    }

    #[test]
    fn test_expert_thermal_update_from_telemetry_then_record() {
        // Arrange
        let mut tracker = ExpertThermalTracker::new(4);
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::ExpertHitCount { expert_id: 0, hit_count: 50 });

        // Act: update from telemetry, then record real routings
        tracker.update_from_telemetry(&agg);
        for _ in 0..50 {
            tracker.record_routing(&[0]);
        }

        // Assert: hit_rate combines telemetry and recorded (50 + 50) / 50 tokens
        let rate = tracker.hit_rate(0);
        assert!(rate > 0.0);
        assert!(rate <= 2.0); // telemetry adds 50 to hits without adding to total_tokens
    }

    #[test]
    fn test_spec_schedule_from_telemetry_high_entropy() {
        // Arrange: high entropy in aggregator
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 5.0 });

        // Act
        let mut signal = SpecScheduleSignal::new();
        let advice = signal.advise_from_telemetry(&agg, 0.9);

        // Assert: high entropy → StandardDecode
        assert_eq!(advice, SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_residual_bypass_config_debug_output() {
        // Arrange
        let config = ResidualBypassConfig::default();

        // Act
        let debug = format!("{:?}", config);

        // Assert: contains struct and field names
        assert!(debug.contains("ResidualBypassConfig"));
        assert!(debug.contains("enabled"));
        assert!(debug.contains("delta_rho_threshold"));
    }

    #[test]
    fn test_gate_first_skip_config_debug_output() {
        // Arrange
        let config = GateFirstSkipConfig::default();

        // Act
        let debug = format!("{:?}", config);

        // Assert
        assert!(debug.contains("GateFirstSkipConfig"));
        assert!(debug.contains("skip_threshold"));
        assert!(debug.contains("enabled"));
    }

    #[test]
    fn test_sink_detection_config_debug_output() {
        // Arrange
        let config = SinkDetectionConfig::default();

        // Act
        let debug = format!("{:?}", config);

        // Assert
        assert!(debug.contains("SinkDetectionConfig"));
        assert!(debug.contains("sink_threshold"));
        assert!(debug.contains("protected_sink_count"));
    }

    #[test]
    fn test_sink_detector_protected_sink_usize_max() {
        // Arrange
        let detector = SinkDetector::new(SinkDetectionConfig::default());

        // Act & Assert: usize::MAX is far beyond protected range
        assert!(!detector.is_protected_sink(usize::MAX));
    }

    #[test]
    fn test_expert_thermal_cold_experts_empty_after_all_hot() {
        // Arrange
        let mut tracker = ExpertThermalTracker::new(3);

        // Act: all experts hit equally
        for _ in 0..100 {
            tracker.record_routing(&[0, 1, 2]);
        }

        // Assert: no cold experts
        assert!(tracker.cold_experts().is_empty());
    }

    #[test]
    fn test_epilogue_signal_copy_semantics() {
        // Arrange: original signal
        let original = EpilogueSignal::CentroidPosition { position: 42 };

        // Act: Copy to new binding, modify concept (Copy types are value types)
        let copied = original;

        // Assert: both are independent and equal
        assert_eq!(original, copied);
        assert_eq!(original, EpilogueSignal::CentroidPosition { position: 42 });
        assert_eq!(copied, EpilogueSignal::CentroidPosition { position: 42 });
    }

    #[test]
    fn test_telemetry_ingest_page_header_then_signal_restores() {
        // Arrange: page header zeros out, then signal restores value
        let mut agg = TelemetryAggregator::new();
        let header = KvPageHeader::new(0);
        agg.ingest_from_page_header(&header);
        assert_eq!(agg.dead_neuron_ratio(), 0.0);

        // Act: ingest signal after page header
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.77 });

        // Assert: signal value takes precedence
        assert!((agg.dead_neuron_ratio() - 0.77).abs() < 1e-6);
    }

    #[test]
    fn test_spec_schedule_fallback_does_not_reset_on_next_good() {
        // Arrange: reach fallback
        let mut signal = SpecScheduleSignal::new();
        signal.advise(1.0, 0.1);
        signal.advise(1.0, 0.1);
        assert_eq!(signal.advise(1.0, 0.1), SpecScheduleAdvice::Fallback);

        // Act: good acceptance resets streak
        let result = signal.advise(1.0, 0.8);

        // Assert: back to EnableSpec (low entropy + good acceptance)
        assert_eq!(result, SpecScheduleAdvice::EnableSpec);
    }
}
