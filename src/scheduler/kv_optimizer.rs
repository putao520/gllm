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
    /// 跨层 Index 复用间隔 (REQ-KV-OPT-008 ChunkKV)
    /// 每隔 K 层完整评估 importance_score，中间层复用最近关键层评分
    pub chunk_cross_layer_k: usize,
}

impl KvOptimizer {
    pub fn new(num_layers: usize) -> Self {
        Self {
            tier_floor: LayerTierFloor::default(),
            num_layers,
            chunk_cross_layer_k: 4,
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

    /// REQ-KV-OPT-008: 判断指定层是否为关键层 (ChunkKV)
    ///
    /// 仅关键层完整评估 importance_score，中间层复用最近关键层评分。
    /// K 值由 chunk_cross_layer_k 控制（默认 4）。
    pub fn is_key_layer(&self, layer_idx: usize) -> bool {
        layer_idx % self.chunk_cross_layer_k == 0
    }

    /// REQ-KV-OPT-008: 查找最近的关键层索引
    ///
    /// 对于非关键层，返回向下最近的已评估关键层。
    pub fn nearest_key_layer(&self, layer_idx: usize) -> usize {
        (layer_idx / self.chunk_cross_layer_k) * self.chunk_cross_layer_k
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

/// REQ-KV-OPT-005: Per-Head 稀疏 Bitmap 计算 (MUSTAFAR)
///
/// 基于 head_entropy_max / head_entropy_min 的分布判断哪些 head 活跃。
/// channel_bitmap_lo 中 bit = 1 表示对应 head 的通道活跃（保留），
/// bit = 0 表示低活跃（可跳过 KV 读取）。
///
/// 由于 KvPageHeader 仅存储 head_entropy_max/min 两个值（而非 per-head 数组），
/// 使用一种启发式方法：将 head 0..31 均匀映射到 [min, max] 范围，
/// 低于中位值的 head 标记为非活跃。
fn compute_sparse_bitmap(header: &KvPageHeader, num_kv_heads: usize) -> u32 {
    let h_min = header.head_entropy_min as u32;
    let h_max = header.head_entropy_max as u32;

    // 如果 max 和 min 接近，没有明显稀疏 head
    if h_max.saturating_sub(h_min) < HEAD_SPARSITY_THRESHOLD as u32 {
        return 0xFFFF_FFFF;
    }

    // 将 [min, max] 分成两半，下半部分为低活跃
    let threshold = h_min + (h_max - h_min) / 2;
    let num_heads = num_kv_heads.min(32);
    let mut bitmap = 0u32;

    for i in 0..num_heads {
        // 线性映射 head i → entropy 值
        let val = if num_heads <= 1 {
            h_min
        } else {
            h_min + (h_max - h_min) * i as u32 / (num_heads - 1) as u32
        };
        if val >= threshold {
            bitmap |= 1 << i;
        }
    }

    // 如果 bitmap 中活跃 head 太少（< 25%），保持全活跃避免精度损失
    let active_count = bitmap.count_ones() as usize;
    if active_count < (num_heads + 3) / 4 {
        return 0xFFFF_FFFF;
    }

    bitmap
}

/// 对一批 page header 执行优化决策
pub fn optimize_pages(
    optimizer: &KvOptimizer,
    headers: &mut [KvPageHeader],
    layer_idx: usize,
    num_kv_heads: usize,
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
        }

        // REQ-KV-OPT-005: Per-Head 稀疏 Bitmap (MUSTAFAR)
        // 当 head 间 entropy 差异大时，某些 head 几乎无贡献
        if importance.should_mark_sparse {
            let bitmap = compute_sparse_bitmap(header, num_kv_heads);
            header.channel_bitmap_lo = bitmap;
        } else {
            // 清除 bitmap — 所有 head 活跃
            header.channel_bitmap_lo = 0xFFFF_FFFF;
        }

        // 更新 tier_age
        header.tier_age = header.tier_age.saturating_add(1);
    }
}

/// REQ-KV-OPT-007: System Prompt 压缩复用 (KVzip)
///
/// KvPrefixIndex 匹配的 system prompt 页在首次 prefill 后执行 importance scoring。
/// 高重要性页保持 FP16，低重要性页降级到 KIVI2/Sparse。
/// 标记为 query-agnostic，ref_count 引用管理。
///
/// `system_prompt_pages`: system prompt 占据的 page headers (已 prefill)
/// `num_kv_heads`: KV head 数量，用于稀疏 bitmap 计算
pub fn optimize_system_prompt_pages(
    optimizer: &KvOptimizer,
    system_prompt_pages: &mut [KvPageHeader],
    num_kv_heads: usize,
) {
    for (i, header) in system_prompt_pages.iter_mut().enumerate() {
        if !header.is_active() {
            continue;
        }

        // 计算 importance_score
        let importance = optimizer.write_importance(header);

        // System prompt 页使用更激进的降级策略:
        // 高重要性 → FP16, 低重要性 → KIVI2/Sparse
        // 但保持 sink token 的 FP16 锁定
        let target_tier = if header.has_sink_token() || importance.score > 180 {
            PrecisionTier::FP16
        } else if importance.score > 100 {
            PrecisionTier::KIVI4
        } else {
            PrecisionTier::KIVI2
        };

        let current_tier = header.precision_tier();
        if target_tier != current_tier {
            header.set_precision_tier(target_tier);
            header.deopt_flags |= 0x01;
        }

        // Per-head sparse bitmap
        if importance.should_mark_sparse {
            header.channel_bitmap_lo = compute_sparse_bitmap(header, num_kv_heads);
        } else {
            header.channel_bitmap_lo = 0xFFFF_FFFF;
        }

        // System prompt 页标记为 Conversation 管线 (跨轮保留)
        header.pipeline_id = 0;

        // REQ-KV-OPT-010: 标记为 position-agnostic (CacheSlide)
        // System prompt 页跳过 RoPE 注入，decode 时通过 Correction Attention 补偿
        header.set_position_agnostic(true);

        // tier_age 更新
        let _ = i;
        header.tier_age = header.tier_age.saturating_add(1);
    }
}

/// REQ-KV-OPT-008: 跨层 Index 复用 (ChunkKV)
///
/// 仅在关键层完整评估 importance_score，中间层复用最近关键层评分。
/// `all_layer_headers`: [layer][page] 二维切片
/// `num_kv_heads`: KV head 数量
pub fn optimize_with_cross_layer_reuse(
    optimizer: &KvOptimizer,
    all_layer_headers: &mut [Vec<KvPageHeader>],
    num_kv_heads: usize,
) {
    let num_layers = all_layer_headers.len();
    // 先评估所有关键层，收集评分快照
    let mut key_layer_scores: Vec<Vec<(u8, u32)>> = vec![Vec::new(); num_layers];
    for layer_idx in 0..num_layers {
        if optimizer.is_key_layer(layer_idx) {
            optimize_pages(optimizer, &mut all_layer_headers[layer_idx], layer_idx, num_kv_heads);
            key_layer_scores[layer_idx] = all_layer_headers[layer_idx].iter()
                .map(|h| (h.importance_score, h.sink_mask))
                .collect();
        }
    }

    // 再处理非关键层：复用最近关键层评分
    for layer_idx in 0..num_layers {
        if optimizer.is_key_layer(layer_idx) {
            continue;
        }
        let key_layer = optimizer.nearest_key_layer(layer_idx);
        if key_layer >= num_layers || key_layer_scores[key_layer].is_empty() {
            continue;
        }
        let scores = &key_layer_scores[key_layer];
        let headers = &mut all_layer_headers[layer_idx];
        for (i, header) in headers.iter_mut().enumerate() {
            if !header.is_active() {
                continue;
            }
            if i < scores.len() {
                header.importance_score = scores[i].0;
                header.sink_mask = scores[i].1;
            }
            let target_tier = optimizer.decide_tier(header, layer_idx);
            if target_tier != header.precision_tier() {
                header.set_precision_tier(target_tier);
                header.deopt_flags |= 0x01;
            }
            header.tier_age = header.tier_age.saturating_add(1);
        }
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
        optimize_pages(&optimizer, &mut headers, 10, 32);
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
        optimize_pages(&optimizer, &mut headers, 10, 32);
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

    #[test]
    fn test_sparse_bitmap_high_spread() {
        // High spread: max=250, min=10 → many heads should be filtered
        let header = make_header(2.0, 0.3, 0.5, 0.2, 250, 10);
        let bitmap = compute_sparse_bitmap(&header, 32);
        // Should not be all-ones (some heads filtered)
        assert_ne!(bitmap, 0xFFFF_FFFF, "high spread should produce sparse bitmap");
        // Should have some active heads
        assert!(bitmap != 0, "bitmap should not be all-zero");
    }

    #[test]
    fn test_sparse_bitmap_low_spread() {
        // Low spread: max=60, min=50 → all heads active
        let header = make_header(2.0, 0.3, 0.5, 0.2, 60, 50);
        let bitmap = compute_sparse_bitmap(&header, 32);
        assert_eq!(bitmap, 0xFFFF_FFFF, "low spread should keep all heads active");
    }

    #[test]
    fn test_sparse_bitmap_too_few_active() {
        // Extreme case: max=255, min=0 with 4 heads → should preserve all (too few active)
        let header = make_header(2.0, 0.3, 0.5, 0.2, 255, 0);
        let bitmap = compute_sparse_bitmap(&header, 4);
        // With 4 heads and threshold at midpoint, only 2 would be active = 50% → OK
        // But if it falls below 25% it would be all-ones
        assert!(bitmap.count_ones() >= 1, "should have at least 1 active head");
    }

    #[test]
    fn test_optimize_pages_sets_sparse_bitmap() {
        let optimizer = KvOptimizer::new(32);
        let mut headers = vec![make_header(2.0, 0.3, 0.5, 0.2, 250, 10)];
        optimize_pages(&optimizer, &mut headers, 20, 32);
        // head_spread = 240 > HEAD_SPARSITY_THRESHOLD → should_mark_sparse = true
        // channel_bitmap should be computed (not default)
        assert_ne!(headers[0].channel_bitmap_lo, 0, "sparse bitmap should be set");
    }

    #[test]
    fn test_key_layer_detection() {
        let optimizer = KvOptimizer::new(32);
        assert!(optimizer.is_key_layer(0));
        assert!(optimizer.is_key_layer(4));
        assert!(optimizer.is_key_layer(8));
        assert!(!optimizer.is_key_layer(1));
        assert!(!optimizer.is_key_layer(3));
        assert!(!optimizer.is_key_layer(7));
    }

    #[test]
    fn test_nearest_key_layer() {
        let optimizer = KvOptimizer::new(32);
        assert_eq!(optimizer.nearest_key_layer(0), 0);
        assert_eq!(optimizer.nearest_key_layer(1), 0);
        assert_eq!(optimizer.nearest_key_layer(3), 0);
        assert_eq!(optimizer.nearest_key_layer(4), 4);
        assert_eq!(optimizer.nearest_key_layer(6), 4);
        assert_eq!(optimizer.nearest_key_layer(7), 4);
    }

    #[test]
    fn test_system_prompt_optimization() {
        let optimizer = KvOptimizer::new(32);
        let mut headers = vec![
            make_header(0.5, 0.9, 0.1, 0.1, 200, 50), // sink → FP16
            make_header(3.0, 0.3, 0.5, 0.3, 80, 60),   // normal → KIVI4
            make_header(5.0, 0.1, 0.9, 0.5, 30, 20),   // low → KIVI2
        ];
        optimize_system_prompt_pages(&optimizer, &mut headers, 32);
        assert_eq!(headers[0].precision_tier(), PrecisionTier::FP16);
        assert_eq!(headers[0].pipeline_id, 0); // Conversation pipeline
        assert_eq!(headers[2].pipeline_id, 0);
        // REQ-KV-OPT-010: all system prompt pages should be position-agnostic
        assert!(headers[0].is_position_agnostic());
        assert!(headers[1].is_position_agnostic());
        assert!(headers[2].is_position_agnostic());
    }

    #[test]
    fn test_cross_layer_reuse() {
        let optimizer = KvOptimizer::new(32);
        // 8 layers, 1 page per layer
        let mut all_headers: Vec<Vec<KvPageHeader>> = (0..8)
            .map(|_| vec![make_header(3.0, 0.3, 0.5, 0.3, 80, 60)])
            .collect();

        optimize_with_cross_layer_reuse(&optimizer, &mut all_headers, 32);

        // Layers 0, 4 should be key layers (fully evaluated)
        assert!(all_headers[0][0].importance_score > 0, "key layer 0 should be evaluated");
        assert!(all_headers[4][0].importance_score > 0, "key layer 4 should be evaluated");

        // Non-key layers should reuse scores from their nearest key layer
        assert_eq!(all_headers[1][0].importance_score, all_headers[0][0].importance_score,
            "layer 1 should reuse layer 0 score");
        assert_eq!(all_headers[2][0].importance_score, all_headers[0][0].importance_score,
            "layer 2 should reuse layer 0 score");
        assert_eq!(all_headers[5][0].importance_score, all_headers[4][0].importance_score,
            "layer 5 should reuse layer 4 score");
    }
}
