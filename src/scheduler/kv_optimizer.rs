//! KV Cache 智能优化 (per SPEC 19-KV-CACHE-OPTIMIZATION.md §3-§4)
//!
//! 消费 Epilogue 白嫖遥测信号，执行：
//! - §3.1 importance_score 评分 (REQ-KV-OPT-002)
//! - §3.2 PrecisionTier 自动升降级 (REQ-KV-OPT-003)
//! - §4.1 四维交叉决策矩阵
//! - Sink Token 动态保护 (REQ-KV-OPT-006)

use crate::kv_cache::{
    f16_bits_to_f32, KvPageHeader, PrecisionTier,
};

/// Sink token 检测阈值 (softmax_max_avg 的 f16 解码值)
const SINK_THRESHOLD: f32 = 0.8;
/// Head 间 entropy 差值阈值 (超过此值则可能有稀疏 head)
const HEAD_SPARSITY_THRESHOLD: u8 = 100;
/// importance_score sink 加成阈值
const SINK_SCORE_THRESHOLD: u8 = 200;

/// 层位置分区的精度下限 (per SPEC §3.2 KVTuner 启发)
#[derive(Debug, Clone, Copy)]
pub struct LayerTierFloor {
    /// 浅层 [0..L/3] 最低精度
    pub shallow_min: PrecisionTier,
    /// 中层 [L/3..2L/3] 最低精度
    pub mid_min: PrecisionTier,
    /// 深层 [2L/3..L] 最低精度 (Evicted = 无下限)
    pub deep_min: PrecisionTier,
}

impl Default for LayerTierFloor {
    fn default() -> Self {
        Self {
            shallow_min: PrecisionTier::FP8,
            mid_min: PrecisionTier::KIVI4,
            deep_min: PrecisionTier::Evicted,
        }
    }
}

/// importance_score 评分结果
#[derive(Debug, Clone, Copy)]
pub struct ImportanceScore {
    /// 综合评分 [0, 255]
    pub score: u8,
    /// 是否为 sink token
    pub is_sink: bool,
    /// Head 间 entropy 差值
    pub head_spread: u8,
    /// 是否建议标记稀疏 bitmap
    pub should_mark_sparse: bool,
}

/// KV Optimizer — 消费 Epilogue 遥测，输出 tier 决策
pub struct KvOptimizer {
    /// 层位置精度下限配置
    pub tier_floor: LayerTierFloor,
    /// 总层数 (用于计算层位置分区)
    pub num_layers: usize,
}

impl KvOptimizer {
    pub fn new(num_layers: usize) -> Self {
        Self {
            tier_floor: LayerTierFloor::default(),
            num_layers,
        }
    }

    /// §3.1 计算 importance_score (REQ-KV-OPT-002)
    ///
    /// 从 KvPageHeader 的 Epilogue 遥测字段计算综合重要性评分。
    /// 所有信号来自 Epilogue 白嫖，零额外计算开销。
    pub fn compute_importance(&self, header: &KvPageHeader) -> ImportanceScore {
        let entropy_avg = f16_bits_to_f32(header.entropy_avg);
        let softmax_max_avg = f16_bits_to_f32(header.softmax_max_avg);
        let delta_rho_avg = f16_bits_to_f32(header.delta_rho_avg);

        // attention_concentration: 高集中度 = 重要
        // entropy_avg 的 f16 范围约 [0, 10]，max_entropy ≈ 6.93 (ln2048)
        let max_entropy = 6.93_f32;
        let attention_concentration = 1.0 - (entropy_avg / max_entropy).min(1.0);

        // sink indicator: 峰值注意力 = sink token
        let is_sink = softmax_max_avg > SINK_THRESHOLD;

        // stability: 低 Δρ = 稳定 = 可降级
        let stability = 1.0 - delta_rho_avg.min(1.0);

        // active_heads: 头间差异 = 语义丰富度
        let head_spread = header.head_entropy_spread();
        let active_heads_f = head_spread as f32 / 255.0;

        // 加权求和 (per SPEC §3.1 公式)
        let raw_score = attention_concentration * 120.0
            + if is_sink { 80.0 } else { 0.0 }
            + active_heads_f * 30.0
            - stability * 40.0;

        let score = raw_score.clamp(0.0, 255.0) as u8;

        ImportanceScore {
            score,
            is_sink,
            head_spread,
            should_mark_sparse: head_spread > HEAD_SPARSITY_THRESHOLD,
        }
    }

    /// 将 importance_score 写回 KvPageHeader 并更新 sink_mask
    pub fn write_importance(&self, header: &mut KvPageHeader) -> ImportanceScore {
        let result = self.compute_importance(header);
        header.importance_score = result.score;

        // REQ-KV-OPT-006: Sink Token 保护 — 标记 sink_mask
        if result.is_sink || result.score > SINK_SCORE_THRESHOLD {
            header.sink_mask = !0u32; // 标记全页为 sink (保守策略)
        }

        result
    }

    /// §3.2 PrecisionTier 升降级决策 (REQ-KV-OPT-003)
    ///
    /// 基于 importance_score + 层位置 + Pipeline 四维交叉决策。
    /// 返回目标 tier，调用方负责执行 requantize。
    pub fn decide_tier(
        &self,
        header: &KvPageHeader,
        layer_idx: usize,
    ) -> PrecisionTier {
        let score = header.importance_score;

        // REQ-KV-OPT-006: sink_mask 非零 → FP16 锁定，禁止降级
        if header.has_sink_token() {
            return PrecisionTier::FP16;
        }

        // 基于 importance_score 的基础 tier 决策 (per SPEC §3.2)
        let base_tier = if score > 200 {
            PrecisionTier::FP16
        } else if score > 150 {
            PrecisionTier::FP8
        } else if score > 80 {
            PrecisionTier::KIVI4
        } else if score > 40 {
            PrecisionTier::KIVI2
        } else if score > 15 {
            PrecisionTier::Sparse
        } else {
            PrecisionTier::Evicted
        };

        // 层位置精度下限调制 (per SPEC §4.1 维度 3)
        let floor = self.layer_tier_floor(layer_idx);
        apply_tier_floor(base_tier, floor)
    }

    /// 获取指定层的精度下限
    fn layer_tier_floor(&self, layer_idx: usize) -> PrecisionTier {
        let third = self.num_layers / 3;
        if self.num_layers == 0 {
            return PrecisionTier::FP8;
        }
        if layer_idx < third {
            self.tier_floor.shallow_min
        } else if layer_idx < third * 2 {
            self.tier_floor.mid_min
        } else {
            self.tier_floor.deep_min
        }
    }
}

/// 对基础 tier 应用层位置精度下限
fn apply_tier_floor(base: PrecisionTier, floor: PrecisionTier) -> PrecisionTier {
    let base_rank = tier_rank(base);
    let floor_rank = tier_rank(floor);
    if base_rank < floor_rank {
        floor
    } else {
        base
    }
}

/// Tier 精度排序 (越高越精确, rank 越大)
fn tier_rank(tier: PrecisionTier) -> u8 {
    match tier {
        PrecisionTier::Evicted => 0,
        PrecisionTier::Dictionary => 1,
        PrecisionTier::Sparse => 2,
        PrecisionTier::KIVI2 => 3,
        PrecisionTier::KIVI4 => 4,
        PrecisionTier::FP8 => 5,
        PrecisionTier::FP16 => 6,
    }
}

/// 对一批 page header 执行优化决策
pub fn optimize_pages(
    optimizer: &KvOptimizer,
    headers: &mut [KvPageHeader],
    layer_idx: usize,
) {
    for header in headers.iter_mut() {
        if !header.is_active() {
            continue;
        }

        // 计算并写入 importance_score
        let importance = optimizer.write_importance(header);

        // PrecisionTier 决策
        let target_tier = optimizer.decide_tier(header, layer_idx);
        let current_tier = header.precision_tier();

        // 仅在 tier 变化时更新 + 设置 requantize 标志
        if target_tier != current_tier {
            header.set_precision_tier(target_tier);
            // 标记需要 requantize
            header.deopt_flags |= 0x01;

            // 必要时标记稀疏 bitmap
            if importance.should_mark_sparse && target_tier == PrecisionTier::Sparse {
                // channel_bitmap 将在 requantize 时由 Epilogue 填充
            }
        }

        // 更新 tier_age
        header.tier_age = header.tier_age.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{f32_to_f16_bits, f32_to_dead_ratio};

    fn make_header(
        entropy: f32,
        softmax_max: f32,
        delta_rho: f32,
        dead_ratio: f32,
        head_max: u8,
        head_min: u8,
    ) -> KvPageHeader {
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(entropy);
        h.softmax_max_avg = f32_to_f16_bits(softmax_max);
        h.delta_rho_avg = f32_to_f16_bits(delta_rho);
        h.dead_ratio = f32_to_dead_ratio(dead_ratio);
        h.head_entropy_max = head_max;
        h.head_entropy_min = head_min;
        h
    }

    #[test]
    fn test_importance_high_attention() {
        let optimizer = KvOptimizer::new(32);
        let header = make_header(0.5, 0.9, 0.1, 0.1, 200, 50);
        let result = optimizer.compute_importance(&header);
        assert!(result.score > 150, "high attention should have high score, got {}", result.score);
        assert!(result.is_sink);
    }

    #[test]
    fn test_importance_low_attention() {
        let optimizer = KvOptimizer::new(32);
        let header = make_header(5.0, 0.1, 0.9, 0.5, 30, 20);
        let result = optimizer.compute_importance(&header);
        assert!(result.score < 80, "low attention should have low score, got {}", result.score);
        assert!(!result.is_sink);
    }

    #[test]
    fn test_importance_sink_detection() {
        let optimizer = KvOptimizer::new(32);
        let header = make_header(1.0, 0.85, 0.2, 0.1, 100, 80);
        let result = optimizer.compute_importance(&header);
        assert!(result.is_sink);
        assert!(result.score > 100);
    }

    #[test]
    fn test_tier_decision_sink_locked_fp16() {
        let optimizer = KvOptimizer::new(32);
        let mut header = make_header(0.5, 0.9, 0.1, 0.1, 200, 50);
        optimizer.write_importance(&mut header);
        let tier = optimizer.decide_tier(&header, 20); // deep layer
        assert_eq!(tier, PrecisionTier::FP16, "sink should be locked to FP16");
    }

    #[test]
    fn test_tier_decision_shallow_floor() {
        let optimizer = KvOptimizer::new(30);
        // Low importance page in shallow layer (layer 5, < L/3=10)
        let mut header = make_header(5.0, 0.1, 0.9, 0.5, 30, 20);
        header.importance_score = 10; // very low
        let tier = optimizer.decide_tier(&header, 5);
        // Shallow floor is FP8, so even very low score should not go below FP8
        assert!(tier_rank(tier) >= tier_rank(PrecisionTier::FP8),
            "shallow layer should be at least FP8, got {:?}", tier);
    }

    #[test]
    fn test_tier_decision_mid_floor() {
        let optimizer = KvOptimizer::new(30);
        let mut header = make_header(5.0, 0.1, 0.9, 0.5, 30, 20);
        header.importance_score = 20;
        let tier = optimizer.decide_tier(&header, 15); // mid layer [10..20]
        assert!(tier_rank(tier) >= tier_rank(PrecisionTier::KIVI4),
            "mid layer should be at least KIVI4, got {:?}", tier);
    }

    #[test]
    fn test_tier_decision_deep_no_floor() {
        let optimizer = KvOptimizer::new(30);
        let mut header = make_header(5.0, 0.1, 0.9, 0.5, 30, 20);
        header.importance_score = 5; // very low
        let tier = optimizer.decide_tier(&header, 25); // deep layer [20..30]
        assert_eq!(tier, PrecisionTier::Evicted, "deep layer with low score should be evicted");
    }

    #[test]
    fn test_write_importance_marks_sink() {
        let optimizer = KvOptimizer::new(32);
        let mut header = make_header(0.5, 0.9, 0.1, 0.1, 200, 50);
        let result = optimizer.write_importance(&mut header);
        assert!(result.is_sink);
        assert_ne!(header.sink_mask, 0, "sink_mask should be set");
        assert!(header.importance_score > 0);
    }

    #[test]
    fn test_write_importance_no_sink() {
        let optimizer = KvOptimizer::new(32);
        let mut header = make_header(3.0, 0.3, 0.5, 0.3, 80, 60);
        let result = optimizer.write_importance(&mut header);
        assert!(!result.is_sink);
        assert_eq!(header.sink_mask, 0, "sink_mask should not be set");
    }

    #[test]
    fn test_optimize_pages_updates_tier_age() {
        let optimizer = KvOptimizer::new(32);
        let mut headers = vec![make_header(3.0, 0.3, 0.5, 0.3, 80, 60)];
        optimize_pages(&optimizer, &mut headers, 10);
        assert_eq!(headers[0].tier_age, 1);
        assert!(headers[0].importance_score > 0);
    }

    #[test]
    fn test_optimize_pages_skips_inactive() {
        let optimizer = KvOptimizer::new(32);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 0; // inactive
        header.entropy_avg = f32_to_f16_bits(3.0);
        let mut headers = vec![header];
        optimize_pages(&optimizer, &mut headers, 10);
        assert_eq!(headers[0].importance_score, 0, "inactive page should not be scored");
        assert_eq!(headers[0].tier_age, 0);
    }

    #[test]
    fn test_head_spread_sparse_detection() {
        let optimizer = KvOptimizer::new(32);
        let header = make_header(2.0, 0.3, 0.5, 0.2, 250, 10);
        let result = optimizer.compute_importance(&header);
        assert!(result.head_spread > HEAD_SPARSITY_THRESHOLD);
        assert!(result.should_mark_sparse);
    }

    #[test]
    fn test_tier_rank_ordering() {
        assert!(tier_rank(PrecisionTier::FP16) > tier_rank(PrecisionTier::FP8));
        assert!(tier_rank(PrecisionTier::FP8) > tier_rank(PrecisionTier::KIVI4));
        assert!(tier_rank(PrecisionTier::KIVI4) > tier_rank(PrecisionTier::KIVI2));
        assert!(tier_rank(PrecisionTier::KIVI2) > tier_rank(PrecisionTier::Sparse));
        assert!(tier_rank(PrecisionTier::Sparse) > tier_rank(PrecisionTier::Evicted));
    }
}
