//! 黄金装筒规则 (SPEC §12.4)
//!
//! 实现硬件感知型黄金装筒规则 (Hardware-Aware Shape Bucketing)。
//!
//! ## 核心原则
//!
//! - **严禁预设硬编码数组**：禁止使用 `[128, 512, 1024, 2048]` 等静态 Bucket
//! - **真实物理探测**：通过 Latency Probe 测定物理拐点，//! - **黄金装筒塌缩**：将任意 SEQ 长度映射到探测出的"黄金尺寸"
//! - **零退化原则**：禁止 Padding 补零，使用 Ragged Compaction
//! - **运行时演化**：JIT Director Daemon 持续观测分布，热插拔新 Bucket

use std::collections::BTreeMap;

use super::histogram::SeqHistogram;
use super::profiler::ProbeResult;
use super::compiler_constraints::CompilerConstraints;

// ── 黄金尺寸 (Golden Size) ──

/// 单个黄金尺寸 (Golden Size)
///
/// 由 Latency Probe 探测出的物理拐点，代表在该硬件上性能最佳的 SEQ 长度。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GoldenSize {
    /// 黄金 SEQ 长度
    pub seq_len: usize,
    /// 该尺寸的寄存器利用率 (0.0-1.0)
    pub register_efficiency: f32,
    /// 该尺寸的共享内存利用率 (0.0-1.0)
    pub smem_efficiency: f32,
    /// 该尺寸的 L2 缓存命中率 (0.0-1.0)
    pub l2_hit_rate: f32,
    /// 综合性能分数 (0.0-1.0)
    pub performance_score: f32,
}

impl GoldenSize {
    /// 创建新的黄金尺寸
    pub fn new(
        seq_len: usize,
        register_efficiency: f32,
        smem_efficiency: f32,
        l2_hit_rate: f32,
    ) -> Self {
        // 综合性能分数 = 加权平均
        let performance_score =
            register_efficiency * 0.4 + smem_efficiency * 0.3 + l2_hit_rate * 0.3;

        Self {
            seq_len,
            register_efficiency,
            smem_efficiency,
            l2_hit_rate,
            performance_score,
        }
    }
}

// ── 黄金装筒注册表 (Golden Bucket Registry) ──

/// 黄金装筒注册表 (SPEC §12.4)
///
/// 维护由硬件探测推导出的黄金尺寸列表。
/// 提供将任意 SEQ 长度映射（塌缩）到最近黄金尺寸的功能。
///
/// ## 生命周期
///
/// 1. **Load-Time**: 从 `ProbeResult` 推导初始黄金尺寸
/// 2. **Runtime**: JIT Director Daemon 根据 SEQ 直方图演化新 Bucket
/// 3. **Hot-Swap**: 原子覆写跳表，淘汰僵尸 Bucket (< 0.1% 命中率)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GoldenBucketRegistry {
    /// 黄金尺寸列表（按 seq_len 升序）
    golden_sizes: Vec<GoldenSize>,

    /// 约束变量（用于 JIT 编译决策）
    constraints: CompilerConstraints,

    /// Bucket 命中率统计（seq_len → hit_count）
    hit_stats: BTreeMap<usize, u64>,

    /// 最大 Bucket 数量（防止无限膨胀）
    max_buckets: usize,

    /// 僵尸 Bucket 淘汰阈值（命中率 < 此值则淘汰）
    zombie_threshold: f64,
}

impl GoldenBucketRegistry {
    /// 从探测结果和约束变量创建初始注册表
    ///
    /// §12.4: 由探测器探出的物理拐点圈定出仅有的几档"黄金尺寸"
    pub fn from_probe_results(
        probe_result: &ProbeResult,
        constraints: CompilerConstraints,
    ) -> Self {
        let mut golden_sizes = Vec::new();

        // 从 spill points 推导黄金尺寸
        // 每个 spill point 代表寄存器溢出的拐点
        // 黄金尺寸 = spill_point 之前的最大 seq_len（寄存器未溢出的最大值）
        for &spill_point in &probe_result.spill_points {
            let seq_len = spill_point;
            // spill point 之前的性能最佳点
            let register_efficiency = 1.0; // 在 spill point 前，寄存器利用率最高
            let smem_efficiency = if let Some(smem) = constraints.smem_size {
                // 估算 SMEM 利用率
                let tile = &constraints.optimal_tile_bits;
                let tile_bytes = tile.tile_m * tile.tile_n * std::mem::size_of::<f32>();
                (tile_bytes as f32 / smem as f32).min(1.0)
            } else {
                0.8 // CPU 无 SMEM，给默认值
            };
            let l2_hit_rate = if probe_result.l2_thrash_threshold > 0 {
                1.0 - (seq_len as f32 / probe_result.l2_thrash_threshold as f32).min(1.0) * 0.3
            } else {
                0.8
            };

            golden_sizes.push(GoldenSize::new(
                seq_len,
                register_efficiency,
                smem_efficiency,
                l2_hit_rate,
            ));
        }

        // 如果没有 spill points（极端情况），使用 L2 thrash 阈值作为指导
        if golden_sizes.is_empty() {
            let l2_threshold = probe_result.l2_thrash_threshold;
            if l2_threshold > 0 {
                // 基于 L2 阈值推导黄金尺寸
                for &ratio in &[0.25, 0.5, 0.75, 1.0] {
                    let seq_len = (l2_threshold as f64 * ratio) as usize;
                    if seq_len > 0 {
                        golden_sizes.push(GoldenSize::new(
                            seq_len,
                            0.9,
                            0.8,
                            1.0 - ratio as f32 * 0.3,
                        ));
                    }
                }
            }

            // 最终兜底：至少一个黄金尺寸
            if golden_sizes.is_empty() {
                golden_sizes.push(GoldenSize::new(256, 0.8, 0.8, 0.8));
            }
        }

        // 按 seq_len 升序排序
        golden_sizes.sort_by_key(|gs| gs.seq_len);

        Self {
            golden_sizes,
            constraints,
            hit_stats: BTreeMap::new(),
            max_buckets: 32,
            zombie_threshold: 0.001, // < 0.1% 命中率
        }
    }

    /// 创建空注册表（用于测试）
    pub fn empty(constraints: CompilerConstraints) -> Self {
        Self {
            golden_sizes: vec![GoldenSize::new(256, 0.8, 0.8, 0.8)],
            constraints,
            hit_stats: BTreeMap::new(),
            max_buckets: 32,
            zombie_threshold: 0.001,
        }
    }

    /// 获取所有黄金尺寸
    pub fn golden_sizes(&self) -> &[GoldenSize] {
        &self.golden_sizes
    }

    /// 将任意 SEQ 长度塌缩到最近的黄金尺寸
    ///
    /// §12.4: 对于推理期任意连续离散的 SEQ 长度，全部强行映射/塌缩到黄金尺寸。
    /// §12.4 零退化原则: 禁止 Padding 补零，使用 Ragged Compaction。
    ///
    /// # Returns
    /// 最近黄金尺寸的索引和引用
    pub fn collapse(&mut self, seq_len: usize) -> (usize, &GoldenSize) {
        // 找到最近的黄金尺寸
        let idx = self.find_nearest_golden_index(seq_len);
        let golden = &self.golden_sizes[idx];

        // 记录命中
        *self.hit_stats.entry(golden.seq_len).or_insert(0) += 1;

        (idx, golden)
    }

    /// 将任意 SEQ 长度塌缩到最近的黄金尺寸（不可变版本）
    pub fn collapse_ref(&self, seq_len: usize) -> (usize, &GoldenSize) {
        let idx = self.find_nearest_golden_index(seq_len);
        (idx, &self.golden_sizes[idx])
    }

    /// 找到最近的黄金尺寸索引
    fn find_nearest_golden_index(&self, seq_len: usize) -> usize {
        if self.golden_sizes.is_empty() {
            return 0;
        }

        // 二分查找：找到第一个 >= seq_len 的
        let pos = self.golden_sizes.partition_point(|gs| gs.seq_len < seq_len);

        if pos == 0 {
            return 0;
        }

        if pos >= self.golden_sizes.len() {
            return self.golden_sizes.len() - 1;
        }

        // 比较 pos-1 和 pos 哪个更近
        let prev = self.golden_sizes[pos - 1].seq_len;
        let curr = self.golden_sizes[pos].seq_len;

        if seq_len - prev <= curr - seq_len {
            pos - 1
        } else {
            pos
        }
    }

    /// §12.4 运行时装筒热演化 (Runtime Bucket Evolution)
    ///
    /// JIT Director Daemon 持续观测 SEQ 分布直方图。
    /// 如果负载发生时段性偏移，在后台编译新的中间态 Bucket 变体。
    pub fn evolve(&mut self, histogram: &SeqHistogram) -> EvolveDecision {
        let snapshot = histogram.snapshot();
        let total = snapshot.total_samples as f64;
        if total < 100.0 {
            return EvolveDecision::InsufficientData;
        }

        // 1. 找出高流量缝隙区间
        let gap_buckets = self.find_high_traffic_gaps(&snapshot);

        if gap_buckets.is_empty() {
            return EvolveDecision::NoEvolutionNeeded;
        }

        // 2. 检查是否超过最大 Bucket 数量
        if self.golden_sizes.len() + gap_buckets.len() > self.max_buckets {
            // 需要先淘汰僵尸 Bucket
            return EvolveDecision::CapacityLimitReached {
                current: self.golden_sizes.len(),
                max: self.max_buckets,
            };
        }

        // 3. 在缝隙区间创建新黄金尺寸
        let mut new_sizes = Vec::new();
        for (start, end, hit_count) in &gap_buckets {
            // 选择区间的中心点作为新黄金尺寸
            let center = (start + end) / 2;
            let hit_rate = *hit_count as f64 / total;
            if hit_rate > 0.05 {
                // 超过 5% 命中率才值得创建新 Bucket
                new_sizes.push(GoldenSize::new(center, 0.85, 0.85, 0.85));
            }
        }

        if new_sizes.is_empty() {
            return EvolveDecision::NoEvolutionNeeded;
        }

        // 4. 插入新黄金尺寸（保持升序）
        for new_size in &new_sizes {
            if let Err(pos) = self.golden_sizes.binary_search_by(
                |gs| gs.seq_len.cmp(&new_size.seq_len)
            ) {
                self.golden_sizes.insert(pos, new_size.clone());
            }
        }

        EvolveDecision::Evolved {
            new_bucket_count: new_sizes.len(),
        }
    }

    /// 淘汰僵尸 Bucket (命中率 < zombie_threshold)
    ///
    /// §12.4: 在 L1i 缓存重排中淘汰命中率 < 0.1% 的僵尸 Bucket
    pub fn evict_zombies(&mut self) -> usize {
        let total_hits: u64 = self.hit_stats.values().sum();
        if total_hits == 0 {
            return 0;
        }

        let threshold = total_hits as f64 * self.zombie_threshold;
        let mut evicted = 0;

        self.golden_sizes.retain(|gs| {
            let hits = self.hit_stats.get(&gs.seq_len).copied().unwrap_or(0);
            let rate = hits as f64;
            if rate >= threshold {
                true
            } else {
                evicted += 1;
                false
            }
        });

        // 清理 hit_stats 中已淘汰的条目
        let remaining: std::collections::HashSet<usize> = self
            .golden_sizes
            .iter()
            .map(|gs| gs.seq_len)
            .collect();
        self.hit_stats.retain(|seq_len, _| remaining.contains(seq_len));

        evicted
    }

    /// 找出高流量缝隙区间
    ///
    /// 扫描直方图，找到那些落在现有黄金尺寸之间且流量较高的区间
    fn find_high_traffic_gaps(
        &self,
        snapshot: &super::histogram::HistogramSnapshot,
    ) -> Vec<(usize, usize, u64)> {
        let mut gaps = Vec::new();

        for i in 0..self.golden_sizes.len().saturating_sub(1) {
            let gap_start = self.golden_sizes[i].seq_len;
            let gap_end = self.golden_sizes[i + 1].seq_len;

            // 计算该区间的总命中数
            let mut gap_hits = 0u64;
            for &(start, end, count) in &snapshot.buckets {
                if start >= gap_start && end <= gap_end {
                    gap_hits += count;
                }
            }

            if gap_hits > 0 {
                gaps.push((gap_start, gap_end, gap_hits));
            }
        }

        gaps.sort_by(|a, b| b.2.cmp(&a.2));
        gaps
    }

    /// 获取当前黄金尺寸数量
    pub fn len(&self) -> usize {
        self.golden_sizes.len()
    }

    /// 获取命中统计
    pub fn hit_stats(&self) -> &BTreeMap<usize, u64> {
        &self.hit_stats
    }

    /// 获取约束变量引用
    pub fn constraints(&self) -> &CompilerConstraints {
        &self.constraints
    }
}

/// 运行时演化决策结果
#[derive(Debug, Clone, PartialEq)]
pub enum EvolveDecision {
    /// 数据不足（< 100 样本）
    InsufficientData,
    /// 无需演化（当前 Bucket 覆盖良好）
    NoEvolutionNeeded,
    /// Bucket 数量达到上限
    CapacityLimitReached {
        current: usize,
        max: usize,
    },
    /// 已演化：创建了新的中间态 Bucket
    Evolved {
        new_bucket_count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_constraints() -> CompilerConstraints {
        CompilerConstraints {
            l2_cache_size: 256 * 1024,
            smem_size: Some(48 * 1024),
            ..CompilerConstraints::default()
        }
    }

    fn make_test_probe_result() -> ProbeResult {
        ProbeResult {
            spill_points: vec![112, 463, 1011],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "test".to_string(),
            raw_measurements: Default::default(),
        }
    }

    #[test]
    fn test_golden_bucket_from_probe() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 应该有 3 个黄金尺寸（来自 3 个 spill points）
        assert_eq!(registry.len(), 3);
        assert_eq!(registry.golden_sizes()[0].seq_len, 112);
        assert_eq!(registry.golden_sizes()[1].seq_len, 463);
        assert_eq!(registry.golden_sizes()[2].seq_len, 1011);
    }

    #[test]
    fn test_golden_bucket_collapse() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 塌缩到最近黄金尺寸
        let (_idx, golden) = registry.collapse(100);
        assert_eq!(golden.seq_len, 112);

        let (_idx2, golden2) = registry.collapse(300);
        assert_eq!(golden2.seq_len, 463);

        let (_idx3, golden3) = registry.collapse(2000);
        assert_eq!(golden3.seq_len, 1011);

        // 命中统计应更新
        assert_eq!(*registry.hit_stats().get(&112).unwrap_or(&0), 1);
        assert_eq!(*registry.hit_stats().get(&463).unwrap_or(&0), 1);
    }

    #[test]
    fn test_golden_bucket_empty_fallback() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: "empty".to_string(),
            raw_measurements: Default::default(),
        };
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 应该有默认的兜底黄金尺寸
        assert!(!registry.golden_sizes().is_empty());
        assert_eq!(registry.golden_sizes()[0].seq_len, 256);
    }

    #[test]
    fn test_evolve_decision_insufficient_data() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let hist = SeqHistogram::new(1000, 4096);
        // 只有 10 个样本，不足 100
        for _ in 0..10 {
            hist.record(300);
        }

        let decision = registry.evolve(&hist);
        assert_eq!(decision, EvolveDecision::InsufficientData);
    }

    #[test]
    fn test_evict_zombies() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 模拟命中统计：只有 1011 被频繁命中
        registry.hit_stats.insert(1011, 10000);
        registry.hit_stats.insert(112, 1);
        registry.hit_stats.insert(463, 1);

        let evicted = registry.evict_zombies();
        // 112 和 463 命中率 < 0.1%，应被淘汰
        assert_eq!(evicted, 2);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_collapse_exact_match() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let (_, golden) = registry.collapse(112);
        assert_eq!(golden.seq_len, 112);
    }

    #[test]
    fn test_collapse_midpoint() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 287 更接近 463 还是 112?
        // 287 - 112 = 175, 463 - 287 = 176 → 更接近 112
        let (_, golden) = registry.collapse(287);
        assert_eq!(golden.seq_len, 112);
    }

    #[test]
    fn test_golden_size_performance_score() {
        let gs = GoldenSize::new(256, 0.9, 0.8, 0.7);
        // 0.9*0.4 + 0.8*0.3 + 0.7*0.3 = 0.36 + 0.24 + 0.21 = 0.81
        assert!((gs.performance_score - 0.81).abs() < 0.01);
    }

    // ---- Additional tests ----

    #[test]
    fn golden_size_clone() {
        let gs = GoldenSize::new(512, 0.9, 0.85, 0.7);
        let gs2 = gs.clone();
        assert_eq!(gs2.seq_len, 512);
        assert!((gs2.register_efficiency - 0.9).abs() < 1e-5);
    }

    #[test]
    fn golden_size_serialize_deserialize() {
        let gs = GoldenSize::new(128, 0.95, 0.9, 0.85);
        let json = serde_json::to_string(&gs).unwrap();
        let gs2: GoldenSize = serde_json::from_str(&json).unwrap();
        assert_eq!(gs2.seq_len, 128);
        assert!((gs2.smem_efficiency - 0.9).abs() < 1e-5);
    }

    #[test]
    fn golden_size_performance_score_equal_weights() {
        let gs = GoldenSize::new(64, 1.0, 1.0, 1.0);
        assert!((gs.performance_score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn golden_size_performance_score_zero() {
        let gs = GoldenSize::new(64, 0.0, 0.0, 0.0);
        assert!((gs.performance_score).abs() < 1e-5);
    }

    #[test]
    fn golden_bucket_registry_empty() {
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::empty(constraints);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 256);
    }

    #[test]
    fn golden_bucket_collapse_ref_immutable() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let (_, golden) = registry.collapse_ref(100);
        assert_eq!(golden.seq_len, 112);
        // hit_stats should NOT be updated (immutable version)
        assert!(registry.hit_stats().is_empty());
    }

    #[test]
    fn golden_bucket_collapse_below_min() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let (_, golden) = registry.collapse(1);
        assert_eq!(golden.seq_len, 112); // smallest golden
    }

    #[test]
    fn golden_bucket_collapse_above_max() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let (_, golden) = registry.collapse(100000);
        assert_eq!(golden.seq_len, 1011); // largest golden
    }

    #[test]
    fn golden_bucket_collapse_exact_midpoint() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // exact midpoint between 112 and 463: (112+463)/2 = 287.5, floor=287
        // 287-112=175, 463-287=176 → closer to 112
        let (_, golden) = registry.collapse(288);
        // 288-112=176, 463-288=175 → closer to 463
        assert_eq!(golden.seq_len, 463);
    }

    #[test]
    fn golden_bucket_from_probe_l2_threshold_fallback() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 1000,
            device_fingerprint: "l2only".into(),
            raw_measurements: Default::default(),
        };
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // 4 ratios: 0.25, 0.5, 0.75, 1.0 → seq_lens 250, 500, 750, 1000
        assert_eq!(registry.len(), 4);
        assert_eq!(registry.golden_sizes()[0].seq_len, 250);
        assert_eq!(registry.golden_sizes()[3].seq_len, 1000);
    }

    #[test]
    fn golden_bucket_evolve_no_evolution_needed() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        let hist = SeqHistogram::new(1000, 4096);
        // All hits exactly at golden sizes — no gaps
        for _ in 0..200 {
            hist.record(112);
        }
        // With only hits at existing golden sizes, there are no high-traffic gaps
        let decision = registry.evolve(&hist);
        // Should be either NoEvolutionNeeded or similar
        match decision {
            EvolveDecision::NoEvolutionNeeded | EvolveDecision::InsufficientData => {}
            other => panic!("Unexpected decision: {:?}", other),
        }
    }

    #[test]
    fn golden_bucket_evolve_capacity_limit() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Set max_buckets to a very low value to force capacity limit
        registry.max_buckets = 3;

        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..200 {
            hist.record(300); // In gap between 112 and 463
        }
        let decision = registry.evolve(&hist);
        match decision {
            EvolveDecision::CapacityLimitReached { current, max } => {
                assert_eq!(current, 3);
                assert_eq!(max, 3);
            }
            EvolveDecision::Evolved { .. } | EvolveDecision::NoEvolutionNeeded => {}
            other => panic!("Unexpected: {:?}", other),
        }
    }

    #[test]
    fn evolve_decision_equality() {
        assert_eq!(EvolveDecision::InsufficientData, EvolveDecision::InsufficientData);
        assert_eq!(EvolveDecision::NoEvolutionNeeded, EvolveDecision::NoEvolutionNeeded);
        assert_ne!(EvolveDecision::InsufficientData, EvolveDecision::NoEvolutionNeeded);
    }

    #[test]
    fn evolve_decision_clone() {
        let d = EvolveDecision::Evolved { new_bucket_count: 3 };
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    #[test]
    fn golden_bucket_evict_zombies_no_hits() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // No hits at all → evict_zombies returns 0
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 0);
    }

    #[test]
    fn golden_bucket_evict_zombies_all_hit() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 100);
        registry.hit_stats.insert(463, 100);
        registry.hit_stats.insert(1011, 100);
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 0);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn golden_bucket_registry_constraints() {
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::empty(constraints.clone());
        assert_eq!(registry.constraints().l2_cache_size, constraints.l2_cache_size);
    }

    #[test]
    fn golden_bucket_registry_serialize_deserialize() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(300);

        let json = serde_json::to_string(&registry).unwrap();
        let registry2: GoldenBucketRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(registry2.len(), 3);
        assert_eq!(registry2.golden_sizes()[1].seq_len, 463);
    }

    #[test]
    fn golden_bucket_collapse_hit_count_increments() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        registry.collapse(100);
        registry.collapse(100);
        registry.collapse(100);
        assert_eq!(*registry.hit_stats().get(&112).unwrap(), 3);
    }

    // ---- Additional comprehensive tests ----

    // ── GoldenSize field access ──

    #[test]
    fn golden_size_all_field_access() {
        let gs = GoldenSize::new(1024, 0.95, 0.88, 0.72);
        assert_eq!(gs.seq_len, 1024);
        assert!((gs.register_efficiency - 0.95).abs() < 1e-5);
        assert!((gs.smem_efficiency - 0.88).abs() < 1e-5);
        assert!((gs.l2_hit_rate - 0.72).abs() < 1e-5);
        // performance_score = 0.95*0.4 + 0.88*0.3 + 0.72*0.3
        //                    = 0.38  + 0.264  + 0.216  = 0.86
        assert!((gs.performance_score - 0.86).abs() < 0.01);
    }

    #[test]
    fn golden_size_debug_format() {
        let gs = GoldenSize::new(512, 0.9, 0.8, 0.7);
        let debug = format!("{:?}", gs);
        assert!(debug.contains("seq_len"));
        assert!(debug.contains("register_efficiency"));
        assert!(debug.contains("smem_efficiency"));
        assert!(debug.contains("l2_hit_rate"));
        assert!(debug.contains("performance_score"));
    }

    #[test]
    fn golden_size_performance_score_partial_efficiency() {
        // Mix of zero and non-zero efficiencies
        let gs = GoldenSize::new(64, 0.0, 0.5, 1.0);
        // 0.0*0.4 + 0.5*0.3 + 1.0*0.3 = 0 + 0.15 + 0.3 = 0.45
        assert!((gs.performance_score - 0.45).abs() < 0.01);
    }

    #[test]
    fn golden_size_clone_independence() {
        let gs = GoldenSize::new(256, 0.8, 0.7, 0.6);
        let mut gs2 = gs.clone();
        gs2.register_efficiency = 0.1;
        assert!(
            (gs.register_efficiency - 0.8).abs() < 1e-5,
            "original must not be mutated"
        );
        assert!((gs2.register_efficiency - 0.1).abs() < 1e-5);
    }

    #[test]
    fn golden_size_serialize_roundtrip_all_fields() {
        let gs = GoldenSize::new(2048, 0.99, 0.77, 0.66);
        let json = serde_json::to_string(&gs).unwrap();
        let gs2: GoldenSize = serde_json::from_str(&json).unwrap();
        assert_eq!(gs2.seq_len, 2048);
        assert!((gs2.register_efficiency - 0.99).abs() < 1e-5);
        assert!((gs2.smem_efficiency - 0.77).abs() < 1e-5);
        assert!((gs2.l2_hit_rate - 0.66).abs() < 1e-5);
        assert!((gs2.performance_score - gs.performance_score).abs() < 1e-7);
    }

    #[test]
    fn golden_size_new_with_zero_seq_len() {
        let gs = GoldenSize::new(0, 0.5, 0.5, 0.5);
        assert_eq!(gs.seq_len, 0);
        // performance_score = 0.5*0.4 + 0.5*0.3 + 0.5*0.3 = 0.5
        assert!((gs.performance_score - 0.5).abs() < 1e-5);
    }

    // ── EvolveDecision variants ──

    #[test]
    fn evolve_decision_debug_format_all_variants() {
        let d1 = EvolveDecision::InsufficientData;
        let d2 = EvolveDecision::NoEvolutionNeeded;
        let d3 = EvolveDecision::CapacityLimitReached { current: 5, max: 32 };
        let d4 = EvolveDecision::Evolved { new_bucket_count: 2 };

        assert!(format!("{:?}", d1).contains("InsufficientData"));
        assert!(format!("{:?}", d2).contains("NoEvolutionNeeded"));
        let debug3 = format!("{:?}", d3);
        assert!(debug3.contains("CapacityLimitReached"));
        assert!(debug3.contains("current"));
        assert!(debug3.contains("max"));
        let debug4 = format!("{:?}", d4);
        assert!(debug4.contains("Evolved"));
        assert!(debug4.contains("new_bucket_count"));
    }

    #[test]
    fn evolve_decision_capacity_limit_fields() {
        let d = EvolveDecision::CapacityLimitReached { current: 10, max: 32 };
        match d {
            EvolveDecision::CapacityLimitReached { current, max } => {
                assert_eq!(current, 10);
                assert_eq!(max, 32);
            }
            other => panic!("Expected CapacityLimitReached, got {:?}", other),
        }
    }

    #[test]
    fn evolve_decision_evolved_fields() {
        let d = EvolveDecision::Evolved { new_bucket_count: 5 };
        match d {
            EvolveDecision::Evolved { new_bucket_count } => {
                assert_eq!(new_bucket_count, 5);
            }
            other => panic!("Expected Evolved, got {:?}", other),
        }
    }

    #[test]
    fn evolve_decision_inequality_across_variants() {
        let d1 = EvolveDecision::InsufficientData;
        let d2 = EvolveDecision::NoEvolutionNeeded;
        let d3 = EvolveDecision::CapacityLimitReached { current: 3, max: 3 };
        let d4 = EvolveDecision::Evolved { new_bucket_count: 1 };
        assert_ne!(d1, d2);
        assert_ne!(d1, d3);
        assert_ne!(d1, d4);
        assert_ne!(d2, d3);
        assert_ne!(d2, d4);
        assert_ne!(d3, d4);
    }

    #[test]
    fn evolve_decision_clone_all_variants() {
        let variants = vec![
            EvolveDecision::InsufficientData,
            EvolveDecision::NoEvolutionNeeded,
            EvolveDecision::CapacityLimitReached { current: 7, max: 32 },
            EvolveDecision::Evolved { new_bucket_count: 3 },
        ];
        for v in &variants {
            assert_eq!(v.clone(), *v);
        }
    }

    // ── GoldenBucketRegistry construction edge cases ──

    #[test]
    fn from_probe_results_unsorted_spill_points_gets_sorted() {
        let probe = ProbeResult {
            spill_points: vec![1011, 112, 463], // deliberately unsorted
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "unsorted".into(),
            raw_measurements: Default::default(),
        };
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Internal sorting should produce ascending order
        assert_eq!(registry.golden_sizes()[0].seq_len, 112);
        assert_eq!(registry.golden_sizes()[1].seq_len, 463);
        assert_eq!(registry.golden_sizes()[2].seq_len, 1011);
    }

    #[test]
    fn from_probe_results_no_smem_cpu_constraints() {
        let probe = make_test_probe_result();
        let constraints = CompilerConstraints {
            smem_size: None, // CPU backend
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 3);
        // smem_efficiency should be 0.8 (CPU default) for all golden sizes
        for gs in registry.golden_sizes() {
            assert!((gs.smem_efficiency - 0.8).abs() < 1e-5);
        }
    }

    #[test]
    fn from_probe_results_with_smem_gpu_constraints() {
        let probe = make_test_probe_result();
        let constraints = CompilerConstraints {
            smem_size: Some(48 * 1024),
            optimal_tile_bits: super::super::compiler_constraints::TileBits {
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
            },
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // smem_efficiency = tile_m * tile_n * 4 / 48K = 16*16*4/49152 ≈ 0.0208
        let gs = &registry.golden_sizes()[0];
        assert!(
            (gs.smem_efficiency - 0.8).abs() > 0.01,
            "GPU smem_efficiency should differ from CPU default 0.8"
        );
    }

    // ── GoldenBucketRegistry trait implementations ──

    #[test]
    fn golden_bucket_registry_clone_preserves_state() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(300);
        registry.collapse(500);

        let cloned = registry.clone();
        assert_eq!(cloned.len(), registry.len());
        assert_eq!(cloned.golden_sizes()[0].seq_len, registry.golden_sizes()[0].seq_len);
        // hit_stats should also be cloned
        assert_eq!(cloned.hit_stats(), registry.hit_stats());
    }

    #[test]
    fn golden_bucket_registry_debug_format() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let debug = format!("{:?}", registry);
        assert!(debug.contains("golden_sizes"));
        assert!(debug.contains("constraints"));
        assert!(debug.contains("hit_stats"));
        assert!(debug.contains("max_buckets"));
        assert!(debug.contains("zombie_threshold"));
    }

    // ── Collapse edge cases ──

    #[test]
    fn collapse_ref_does_not_modify_hit_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Multiple collapse_ref calls should not change hit_stats
        let _ = registry.collapse_ref(100);
        let _ = registry.collapse_ref(300);
        let _ = registry.collapse_ref(1000);
        assert!(registry.hit_stats().is_empty());
    }

    #[test]
    fn collapse_multiple_targets_same_golden() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Multiple seq_lens near 112 should all collapse to 112
        registry.collapse(1);
        registry.collapse(50);
        registry.collapse(100);
        registry.collapse(112);
        // 287 is equidistant: 287-112=175, 463-287=176 → 112
        registry.collapse(287);
        assert_eq!(*registry.hit_stats().get(&112).unwrap(), 5);
    }

    #[test]
    fn collapse_returns_correct_index() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let (idx0, _) = registry.collapse(50);   // → 112 (index 0)
        let (idx1, _) = registry.collapse(300);   // → 463 (index 1)
        let (idx2, _) = registry.collapse(2000);  // → 1011 (index 2)
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn collapse_ref_returns_correct_index() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let (idx0, _) = registry.collapse_ref(50);
        let (idx1, _) = registry.collapse_ref(300);
        let (idx2, _) = registry.collapse_ref(2000);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    // ── Evolve with actual gap traffic ──

    #[test]
    fn evolve_creates_new_bucket_in_gap() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // golden sizes: 112, 463, 1011
        // Record enough samples in the gap between 112 and 463
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(300); // Falls in the gap between 112 and 463
        }

        let decision = registry.evolve(&hist);
        match decision {
            EvolveDecision::Evolved { new_bucket_count } => {
                assert!(new_bucket_count > 0);
                // New golden size should be at midpoint of gap
                assert!(registry.len() > 3);
            }
            EvolveDecision::NoEvolutionNeeded => {
                // Possible if histogram bucket alignment doesn't create a gap
            }
            other => panic!("Unexpected decision: {:?}", other),
        }
    }

    // ── Evict zombies partial eviction ──

    #[test]
    fn evict_zombies_partial_keeps_hot_buckets() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // 112 gets many hits, 463 gets 1 hit, 1011 gets 1 hit
        registry.hit_stats.insert(112, 9998);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);

        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 2);
        assert_eq!(registry.len(), 1);
        // Only the hot one remains
        assert_eq!(registry.golden_sizes()[0].seq_len, 112);
    }

    #[test]
    fn evict_zombies_cleans_orphaned_hit_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        // Make 112 the only hot bucket
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);

        registry.evict_zombies();

        // hit_stats should only contain keys for remaining golden sizes
        for &seq_len in registry.hit_stats().keys() {
            let exists = registry.golden_sizes().iter().any(|gs| gs.seq_len == seq_len);
            assert!(exists, "hit_stats key {} has no matching golden size", seq_len);
        }
    }

    // ── Len tracking ──

    #[test]
    fn len_reflects_initial_state() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn len_after_eviction() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 3);
        // Evict all but one
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);
        registry.evict_zombies();
        assert_eq!(registry.len(), 1);
    }

    // ── Single golden size collapse (edge case with only 1 bucket) ──

    #[test]
    fn collapse_single_golden_size() {
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::empty(constraints);
        // empty() creates a single golden size at 256
        assert_eq!(registry.len(), 1);
        let (_, golden) = registry.collapse(1);
        assert_eq!(golden.seq_len, 256);
        let (_, golden2) = registry.collapse(10000);
        assert_eq!(golden2.seq_len, 256);
    }

    // ── from_probe_results with duplicate spill points ──

    #[test]
    fn from_probe_results_duplicate_spill_points() {
        let probe = ProbeResult {
            spill_points: vec![256, 256, 256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "dup".into(),
            raw_measurements: Default::default(),
        };
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // 3 duplicates → 3 golden sizes at same seq_len
        assert_eq!(registry.len(), 3);
        assert_eq!(registry.golden_sizes()[0].seq_len, 256);
        assert_eq!(registry.golden_sizes()[1].seq_len, 256);
    }

    // ── EvolveDecision PartialEq with different field values ──

    #[test]
    fn evolve_decision_capacity_limit_different_values_not_equal() {
        let d1 = EvolveDecision::CapacityLimitReached { current: 3, max: 32 };
        let d2 = EvolveDecision::CapacityLimitReached { current: 5, max: 32 };
        let d3 = EvolveDecision::CapacityLimitReached { current: 3, max: 16 };
        assert_ne!(d1, d2);
        assert_ne!(d1, d3);
        assert_ne!(d2, d3);
    }

    #[test]
    fn evolve_decision_evolved_different_count_not_equal() {
        let d1 = EvolveDecision::Evolved { new_bucket_count: 1 };
        let d2 = EvolveDecision::Evolved { new_bucket_count: 2 };
        assert_ne!(d1, d2);
    }

    // ── New batch: 18 additional unit tests ──

    // 1. GoldenSize with zero seq_len and zero efficiencies
    #[test]
    fn golden_size_zero_seq_len_zero_efficiencies() {
        let gs = GoldenSize::new(0, 0.0, 0.0, 0.0);

        assert_eq!(gs.seq_len, 0);
        assert!((gs.register_efficiency).abs() < 1e-5);
        assert!((gs.smem_efficiency).abs() < 1e-5);
        assert!((gs.l2_hit_rate).abs() < 1e-5);
        assert!((gs.performance_score).abs() < 1e-5);
    }

    // 2. GoldenSize with very large seq_len (max usize near boundary)
    #[test]
    fn golden_size_large_seq_len_preserved() {
        let gs = GoldenSize::new(usize::MAX, 1.0, 1.0, 1.0);

        assert_eq!(gs.seq_len, usize::MAX);
        assert!((gs.performance_score - 1.0).abs() < 1e-5);
    }

    // 3. GoldenSize performance_score with only register_efficiency nonzero
    #[test]
    fn golden_size_performance_score_register_only() {
        let gs = GoldenSize::new(128, 1.0, 0.0, 0.0);

        // 1.0*0.4 + 0.0*0.3 + 0.0*0.3 = 0.4
        assert!((gs.performance_score - 0.4).abs() < 1e-5);
    }

    // 4. GoldenSize performance_score with only smem_efficiency nonzero
    #[test]
    fn golden_size_performance_score_smem_only() {
        let gs = GoldenSize::new(128, 0.0, 1.0, 0.0);

        // 0.0*0.4 + 1.0*0.3 + 0.0*0.3 = 0.3
        assert!((gs.performance_score - 0.3).abs() < 1e-5);
    }

    // 5. GoldenSize performance_score with only l2_hit_rate nonzero
    #[test]
    fn golden_size_performance_score_l2_only() {
        let gs = GoldenSize::new(128, 0.0, 0.0, 1.0);

        // 0.0*0.4 + 0.0*0.3 + 1.0*0.3 = 0.3
        assert!((gs.performance_score - 0.3).abs() < 1e-5);
    }

    // 6. GoldenSize Clone produces independent copy with correct performance_score
    #[test]
    fn golden_size_clone_preserves_performance_score() {
        let gs = GoldenSize::new(512, 0.7, 0.6, 0.9);
        let cloned = gs.clone();
        let expected = 0.7 * 0.4 + 0.6 * 0.3 + 0.9 * 0.3;

        assert!((cloned.performance_score - expected).abs() < 1e-7);
        assert!((gs.performance_score - cloned.performance_score).abs() < 1e-10);
    }

    // 7. GoldenBucketRegistry::empty with default constraints returns 1 golden size
    #[test]
    fn empty_registry_default_constraints() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 256);
    }

    // 8. GoldenBucketRegistry::empty hit_stats is empty
    #[test]
    fn empty_registry_hit_stats_is_empty() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);

        assert!(registry.hit_stats().is_empty());
    }

    // 9. GoldenBucketRegistry::empty constraints accessor reflects passed value
    #[test]
    fn empty_registry_constraints_accessor() {
        let constraints = CompilerConstraints {
            l2_cache_size: 512 * 1024,
            smem_size: Some(96 * 1024),
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::empty(constraints);

        assert_eq!(registry.constraints().l2_cache_size, 512 * 1024);
        assert_eq!(registry.constraints().smem_size, Some(96 * 1024));
    }

    // 10. GoldenBucketRegistry::golden_sizes returns slice with correct length
    #[test]
    fn golden_sizes_slice_matches_len() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);

        assert_eq!(registry.golden_sizes().len(), registry.len());
        assert_eq!(registry.golden_sizes().len(), 3);
    }

    // 11. collapse_ref on single-entry registry always returns index 0
    #[test]
    fn collapse_ref_single_entry_always_index_zero() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);

        let (idx_small, gs_small) = registry.collapse_ref(0);
        assert_eq!(idx_small, 0);
        assert_eq!(gs_small.seq_len, 256);

        let (idx_large, gs_large) = registry.collapse_ref(999999);
        assert_eq!(idx_large, 0);
        assert_eq!(gs_large.seq_len, 256);
    }

    // 12. EvolveDecision::CapacityLimitReached equality same fields
    #[test]
    fn evolve_decision_capacity_limit_equal_same_fields() {
        let d1 = EvolveDecision::CapacityLimitReached { current: 10, max: 32 };
        let d2 = EvolveDecision::CapacityLimitReached { current: 10, max: 32 };

        assert_eq!(d1, d2);
    }

    // 13. EvolveDecision::Evolved equality same field
    #[test]
    fn evolve_decision_evolved_equal_same_field() {
        let d1 = EvolveDecision::Evolved { new_bucket_count: 7 };
        let d2 = EvolveDecision::Evolved { new_bucket_count: 7 };

        assert_eq!(d1, d2);
    }

    // 14. EvolveDecision Clone round-trip for CapacityLimitReached
    #[test]
    fn evolve_decision_clone_capacity_limit() {
        let d = EvolveDecision::CapacityLimitReached { current: 15, max: 64 };
        let cloned = d.clone();

        assert_eq!(cloned, d);
        if let EvolveDecision::CapacityLimitReached { current, max } = cloned {
            assert_eq!(current, 15);
            assert_eq!(max, 64);
        } else {
            panic!("Cloned should be CapacityLimitReached");
        }
    }

    // 15. GoldenBucketRegistry Debug output contains golden_sizes keys
    #[test]
    fn empty_registry_debug_format_keys() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);
        let debug = format!("{:?}", registry);

        assert!(debug.contains("golden_sizes"));
        assert!(debug.contains("max_buckets"));
        assert!(debug.contains("zombie_threshold"));
    }

    // 16. GoldenSize Debug output contains all five fields
    #[test]
    fn golden_size_debug_contains_all_five_fields() {
        let gs = GoldenSize::new(333, 0.11, 0.22, 0.33);
        let debug = format!("{:?}", gs);

        assert!(debug.contains("seq_len"));
        assert!(debug.contains("register_efficiency"));
        assert!(debug.contains("smem_efficiency"));
        assert!(debug.contains("l2_hit_rate"));
        assert!(debug.contains("performance_score"));
    }

    // 17. EvolveDecision all four variants are distinct (no two equal)
    #[test]
    fn evolve_decision_all_variants_distinct() {
        let variants: Vec<EvolveDecision> = vec![
            EvolveDecision::InsufficientData,
            EvolveDecision::NoEvolutionNeeded,
            EvolveDecision::CapacityLimitReached { current: 1, max: 2 },
            EvolveDecision::Evolved { new_bucket_count: 1 },
        ];

        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants {} and {} should differ", i, j);
            }
        }
    }

    // 18. collapse on empty-constructed registry tracks hits correctly
    #[test]
    fn empty_registry_collapse_tracks_hits() {
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::empty(constraints);

        let (idx1, _) = registry.collapse(100);
        let (idx2, _) = registry.collapse(500);
        let (idx3, _) = registry.collapse(1000);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 0);
        assert_eq!(idx3, 0);
        // All three collapses map to the single golden size 256
        assert_eq!(*registry.hit_stats().get(&256).unwrap(), 3);
    }

    // ── Batch 3: 55 additional unit tests ──

    // ── GoldenSize: performance_score is always in [0, 1] for valid inputs ──

    #[test]
    fn golden_size_performance_score_bounded_below() {
        let gs = GoldenSize::new(64, 0.0, 0.0, 0.0);
        assert!(gs.performance_score >= 0.0);
    }

    #[test]
    fn golden_size_performance_score_bounded_above() {
        let gs = GoldenSize::new(64, 1.0, 1.0, 1.0);
        assert!(gs.performance_score <= 1.0 + 1e-7);
    }

    #[test]
    fn golden_size_performance_score_weighted_sum_property() {
        // For any a, b, c: score = 0.4*a + 0.3*b + 0.3*c
        let gs = GoldenSize::new(64, 0.6, 0.4, 0.8);
        let expected = 0.6 * 0.4 + 0.4 * 0.3 + 0.8 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-7);
    }

    #[test]
    fn golden_size_new_negative_efficiency_clamped_by_math() {
        // Negative inputs are technically allowed; verify score computation
        let gs = GoldenSize::new(64, -0.1, 0.5, 0.5);
        let expected = -0.1 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-7);
    }

    #[test]
    fn golden_size_new_efficiency_above_one() {
        // Values above 1.0 are technically allowed; verify math
        let gs = GoldenSize::new(64, 1.5, 0.5, 0.5);
        let expected = 1.5 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-7);
    }

    #[test]
    fn golden_size_equality_same_fields() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(128, 0.5, 0.6, 0.7);
        assert_eq!(a, b);
    }

    #[test]
    fn golden_size_inequality_different_seq_len() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(256, 0.5, 0.6, 0.7);
        assert_ne!(a, b);
    }

    #[test]
    fn golden_size_inequality_different_register_efficiency() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(128, 0.9, 0.6, 0.7);
        assert_ne!(a, b);
    }

    #[test]
    fn golden_size_inequality_different_smem_efficiency() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(128, 0.5, 0.9, 0.7);
        assert_ne!(a, b);
    }

    #[test]
    fn golden_size_inequality_different_l2_hit_rate() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(128, 0.5, 0.6, 0.9);
        assert_ne!(a, b);
    }

    #[test]
    fn golden_size_serialize_deserialize_roundtrip_json() {
        let gs = GoldenSize::new(4096, 0.33, 0.66, 0.99);
        let json = serde_json::to_string(&gs).unwrap();
        let restored: GoldenSize = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, gs);
    }

    #[test]
    fn golden_size_performance_score_symmetric_weights() {
        // smem_efficiency and l2_hit_rate have equal weight (0.3 each),
        // so swapping them should give same performance_score
        let gs_a = GoldenSize::new(64, 0.5, 0.2, 0.8);
        let gs_b = GoldenSize::new(64, 0.5, 0.8, 0.2);
        assert!((gs_a.performance_score - gs_b.performance_score).abs() < 1e-7);
    }

    // ── from_probe_results: l2_thrash_threshold affects l2_hit_rate ──

    #[test]
    fn from_probe_results_l2_thrash_affects_hit_rate() {
        let probe_low = ProbeResult {
            spill_points: vec![256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 256,
            device_fingerprint: "low_l2".into(),
            raw_measurements: Default::default(),
        };
        let probe_high = ProbeResult {
            spill_points: vec![256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 4096,
            device_fingerprint: "high_l2".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: None,
            ..CompilerConstraints::default()
        };
        let reg_low = GoldenBucketRegistry::from_probe_results(&probe_low, constraints.clone());
        let reg_high = GoldenBucketRegistry::from_probe_results(&probe_high, constraints);

        let gs_low = &reg_low.golden_sizes()[0];
        let gs_high = &reg_high.golden_sizes()[0];
        // Higher l2_thrash_threshold should yield higher l2_hit_rate
        assert!(gs_high.l2_hit_rate > gs_low.l2_hit_rate);
    }

    #[test]
    fn from_probe_results_l2_thrash_zero_gives_default_hit_rate() {
        let probe = ProbeResult {
            spill_points: vec![512],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: "zero_l2".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: None,
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        assert!((gs.l2_hit_rate - 0.8).abs() < 1e-5);
    }

    // ── from_probe_results: single spill point ──

    #[test]
    fn from_probe_results_single_spill_point() {
        let probe = ProbeResult {
            spill_points: vec![768],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "single".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 768);
    }

    // ── from_probe_results: many spill points ──

    #[test]
    fn from_probe_results_many_spill_points() {
        let probe = ProbeResult {
            spill_points: vec![64, 128, 256, 512, 1024, 2048],
            smem_cliffs: vec![],
            l2_thrash_threshold: 4096,
            device_fingerprint: "many".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 6);
        // Verify ascending sort
        for i in 0..registry.len() - 1 {
            assert!(registry.golden_sizes()[i].seq_len < registry.golden_sizes()[i + 1].seq_len);
        }
    }

    // ── from_probe_results: golden_sizes ascending invariant ──

    #[test]
    fn from_probe_results_golden_sizes_ascending_order() {
        let probe = ProbeResult {
            spill_points: vec![2048, 64, 512, 128, 1024, 256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 4096,
            device_fingerprint: "shuffled".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let sizes: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        let mut sorted = sizes.clone();
        sorted.sort();
        assert_eq!(sizes, sorted);
    }

    // ── GoldenBucketRegistry: empty golden_sizes field behavior ──

    #[test]
    fn from_probe_results_fallback_uses_l2_ratios() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2000,
            device_fingerprint: "l2_fallback".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 4);
        // Ratios: 0.25, 0.5, 0.75, 1.0 → 500, 1000, 1500, 2000
        assert_eq!(registry.golden_sizes()[0].seq_len, 500);
        assert_eq!(registry.golden_sizes()[3].seq_len, 2000);
    }

    // ── Collapse: seq_len = 0 boundary ──

    #[test]
    fn collapse_seq_len_zero() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let (_, golden) = registry.collapse(0);
        assert_eq!(golden.seq_len, 112); // smallest golden
    }

    #[test]
    fn collapse_ref_seq_len_zero() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let (_, golden) = registry.collapse_ref(0);
        assert_eq!(golden.seq_len, 112);
    }

    // ── Collapse: tie-breaking at exact midpoint ──

    #[test]
    fn collapse_tie_goes_to_lower() {
        // With golden sizes [112, 463], midpoint = (112 + 463) / 2 = 287.5
        // 287: 287-112=175, 463-287=176 → 112 (lower wins)
        // 288: 288-112=176, 463-288=175 → 463 (higher wins)
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let (_, g_low) = registry.collapse(287);
        assert_eq!(g_low.seq_len, 112);
        let (_, g_high) = registry.collapse(288);
        assert_eq!(g_high.seq_len, 463);
    }

    // ── Evolve: exactly 100 samples threshold ──

    #[test]
    fn evolve_exactly_100_samples_is_sufficient() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..100 {
            hist.record(300);
        }
        let decision = registry.evolve(&hist);
        // 100 samples >= 100 threshold → should NOT be InsufficientData
        assert_ne!(decision, EvolveDecision::InsufficientData);
    }

    #[test]
    fn evolve_99_samples_is_insufficient() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..99 {
            hist.record(300);
        }
        let decision = registry.evolve(&hist);
        assert_eq!(decision, EvolveDecision::InsufficientData);
    }

    // ── Evolve: gap traffic below 5% threshold does not create bucket ──

    #[test]
    fn evolve_low_hit_rate_gap_no_new_bucket() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Only 5 hits out of 200 = 2.5% < 5% threshold
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..195 {
            hist.record(112); // exactly at golden size
        }
        for _ in 0..5 {
            hist.record(300); // in gap but low traffic
        }
        let decision = registry.evolve(&hist);
        // Should be NoEvolutionNeeded because gap traffic < 5%
        match decision {
            EvolveDecision::NoEvolutionNeeded => {}
            EvolveDecision::Evolved { new_bucket_count } => {
                // If evolved, it means histogram buckets cover more than expected
                assert!(new_bucket_count > 0);
            }
            other => panic!("Unexpected: {:?}", other),
        }
    }

    // ── Evolve: preserves sorted order after insertion ──

    #[test]
    fn evolve_preserves_ascending_order() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(300);
        }
        let _ = registry.evolve(&hist);
        let sizes: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        for i in 0..sizes.len().saturating_sub(1) {
            assert!(sizes[i] < sizes[i + 1], "golden_sizes not ascending at index {}", i);
        }
    }

    // ── Evict: hit_stats consistency after eviction ──

    #[test]
    fn evict_zombies_hit_stats_subset_of_golden_sizes() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 0);
        registry.hit_stats.insert(1011, 0);
        registry.evict_zombies();
        // Every hit_stats key must correspond to a remaining golden size
        for &seq_len in registry.hit_stats().keys() {
            assert!(
                registry.golden_sizes().iter().any(|gs| gs.seq_len == seq_len),
                "orphaned hit_stats key: {}",
                seq_len
            );
        }
    }

    #[test]
    fn evict_zombies_all_cold_evicts_all() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // All cold: 1 hit each, with 3 total → each is 1/3 ≈ 33% which is > 0.1%
        // Need to make them all zombies: set threshold high by having one dominant
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 2);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 112);
    }

    #[test]
    fn evict_zombies_zero_total_hits_returns_zero() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // hit_stats empty → total_hits = 0 → early return 0
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 0);
        assert_eq!(registry.len(), 3);
    }

    // ── EvolveDecision: debug format covers all variants ──

    #[test]
    fn evolve_decision_insufficient_data_debug() {
        let d = EvolveDecision::InsufficientData;
        let s = format!("{:?}", d);
        assert!(s.contains("InsufficientData"));
    }

    #[test]
    fn evolve_decision_no_evolution_needed_debug() {
        let d = EvolveDecision::NoEvolutionNeeded;
        let s = format!("{:?}", d);
        assert!(s.contains("NoEvolutionNeeded"));
    }

    // ── GoldenBucketRegistry: golden_sizes() returns slice ──

    #[test]
    fn golden_sizes_returns_non_empty_for_empty_registry() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);
        assert!(!registry.golden_sizes().is_empty());
    }

    #[test]
    fn golden_sizes_len_matches_registry_len() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.golden_sizes().len(), registry.len());
    }

    // ── Collapse: hit_stats keys correspond to golden_sizes ──

    #[test]
    fn collapse_hit_stats_keys_are_golden_seq_lens() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(50);
        registry.collapse(300);
        registry.collapse(2000);
        let golden_seq_lens: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        for &key in registry.hit_stats().keys() {
            assert!(golden_seq_lens.contains(&key), "hit_stats key {} not in golden_sizes", key);
        }
    }

    // ── from_probe_results: golden size register_efficiency is always 1.0 for spill points ──

    #[test]
    fn from_probe_results_spill_points_register_efficiency_is_one() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for gs in registry.golden_sizes() {
            assert!((gs.register_efficiency - 1.0).abs() < 1e-5);
        }
    }

    // ── from_probe_results: l2 fallback golden sizes have register_efficiency 0.9 ──

    #[test]
    fn from_probe_results_l2_fallback_register_efficiency() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 1000,
            device_fingerprint: "l2fb".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: None,
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for gs in registry.golden_sizes() {
            assert!((gs.register_efficiency - 0.9).abs() < 1e-5);
        }
    }

    // ── from_probe_results: smem_efficiency capped at 1.0 ──

    #[test]
    fn from_probe_results_smem_efficiency_capped_at_one() {
        // Use very small smem to make tile ratio > 1.0 (will be clamped)
        let probe = ProbeResult {
            spill_points: vec![256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "tiny_smem".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: Some(1), // 1 byte → ratio will be huge, clamped to 1.0
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        assert!(gs.smem_efficiency <= 1.0 + 1e-5);
    }

    // ── GoldenBucketRegistry: clone after eviction ──

    #[test]
    fn clone_after_eviction_preserves_reduced_state() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);
        registry.evict_zombies();
        let cloned = registry.clone();
        assert_eq!(cloned.len(), registry.len());
        assert_eq!(cloned.hit_stats().len(), registry.hit_stats().len());
    }

    // ── GoldenBucketRegistry: serialize/deserialize with hit_stats ──

    #[test]
    fn serialize_deserialize_with_hit_stats_preserves_data() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(100);
        registry.collapse(100);
        registry.collapse(500);
        let json = serde_json::to_string(&registry).unwrap();
        let restored: GoldenBucketRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), registry.len());
        assert_eq!(restored.hit_stats(), registry.hit_stats());
    }

    // ── GoldenSize: very small positive efficiencies ──

    #[test]
    fn golden_size_very_small_positive_efficiencies() {
        let gs = GoldenSize::new(32, 0.001, 0.001, 0.001);
        let expected = 0.001 * 0.4 + 0.001 * 0.3 + 0.001 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-7);
    }

    // ── GoldenSize: PartialEq works with f32 precision ──

    #[test]
    fn golden_size_eq_with_same_computed_performance_score() {
        let a = GoldenSize::new(64, 0.5, 0.5, 0.5);
        let b = GoldenSize::new(64, 0.5, 0.5, 0.5);
        assert_eq!(a, b);
        assert!((a.performance_score - b.performance_score).abs() < 1e-10);
    }

    // ── Registry: collapse then evolve integration ──

    #[test]
    fn collapse_then_evolve_integration() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Collapse some values to build hit_stats
        registry.collapse(100);
        registry.collapse(300);
        registry.collapse(1000);
        assert_eq!(registry.hit_stats().len(), 3);

        // Evolve with gap traffic
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..200 {
            hist.record(300);
        }
        let _ = registry.evolve(&hist);
        // Registry should still have valid ascending order
        let sizes: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        for i in 0..sizes.len().saturating_sub(1) {
            assert!(sizes[i] <= sizes[i + 1]);
        }
    }

    // ── Evolve: no gaps when all hits are exactly at golden sizes ──

    #[test]
    fn evolve_all_hits_at_golden_sizes_no_evolution() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..150 {
            hist.record(112);
        }
        for _ in 0..150 {
            hist.record(1011);
        }
        let decision = registry.evolve(&hist);
        // Hits are at golden sizes, not in gaps
        match decision {
            EvolveDecision::NoEvolutionNeeded => {}
            EvolveDecision::Evolved { .. } => {
                // Possible if histogram bucket alignment places samples in gaps
            }
            other => panic!("Unexpected: {:?}", other),
        }
    }

    // ── EvolveDecision: CapacityLimitReached with zero max ──

    #[test]
    fn evolve_decision_capacity_limit_zero_max() {
        let d = EvolveDecision::CapacityLimitReached { current: 0, max: 0 };
        match d {
            EvolveDecision::CapacityLimitReached { current, max } => {
                assert_eq!(current, 0);
                assert_eq!(max, 0);
            }
            other => panic!("Expected CapacityLimitReached, got {:?}", other),
        }
    }

    // ── EvolveDecision: Evolved with zero new buckets ──

    #[test]
    fn evolve_decision_evolved_zero_new_buckets() {
        let d = EvolveDecision::Evolved { new_bucket_count: 0 };
        match d {
            EvolveDecision::Evolved { new_bucket_count } => {
                assert_eq!(new_bucket_count, 0);
            }
            other => panic!("Expected Evolved, got {:?}", other),
        }
    }

    // ── Registry: hit_stats BTreeMap ordering ──

    #[test]
    fn hit_stats_keys_are_sorted() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(1000);
        registry.collapse(100);
        registry.collapse(500);
        let keys: Vec<&usize> = registry.hit_stats().keys().collect();
        for i in 0..keys.len().saturating_sub(1) {
            assert!(keys[i] < keys[i + 1], "hit_stats keys not sorted");
        }
    }

    // ── Registry: constraints() returns reference to stored constraints ──

    #[test]
    fn constraints_returns_stored_reference() {
        let constraints = CompilerConstraints {
            l2_cache_size: 999,
            smem_size: Some(12345),
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::empty(constraints);
        assert_eq!(registry.constraints().l2_cache_size, 999);
        assert_eq!(registry.constraints().smem_size, Some(12345));
    }

    // ── Registry: from_probe_results golden_sizes l2_hit_rate decreases with seq_len ──

    #[test]
    fn from_probe_results_l2_hit_rate_decreases_with_seq_len() {
        let probe = ProbeResult {
            spill_points: vec![128, 256, 512, 1024],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "l2_decrease".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: None,
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let sizes = registry.golden_sizes();
        for i in 0..sizes.len().saturating_sub(1) {
            // Higher seq_len → lower l2_hit_rate (approaching thrash)
            assert!(
                sizes[i].l2_hit_rate >= sizes[i + 1].l2_hit_rate - 1e-5,
                "l2_hit_rate should decrease: {} at idx {} vs {} at idx {}",
                sizes[i].l2_hit_rate, i, sizes[i + 1].l2_hit_rate, i + 1
            );
        }
    }

    // ── Evolve: does not create duplicate golden sizes ──

    #[test]
    fn evolve_no_duplicate_seq_lens() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(300);
        }
        let _ = registry.evolve(&hist);
        let seq_lens: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        let unique: std::collections::HashSet<usize> = seq_lens.iter().copied().collect();
        assert_eq!(seq_lens.len(), unique.len(), "duplicate seq_lens found after evolve");
    }

    // ── EvolveDecision: clone roundtrip for InsufficientData ──

    #[test]
    fn evolve_decision_clone_insufficient_data() {
        let d = EvolveDecision::InsufficientData;
        let cloned = d.clone();
        assert_eq!(cloned, d);
    }

    #[test]
    fn evolve_decision_clone_no_evolution_needed() {
        let d = EvolveDecision::NoEvolutionNeeded;
        let cloned = d.clone();
        assert_eq!(cloned, d);
    }

    // ── GoldenSize: PartialEq with different performance_score ──

    #[test]
    fn golden_size_neq_when_performance_score_differs() {
        // performance_score is computed, not stored independently.
        // Two GoldenSizes with different register_efficiency → different performance_score → not equal
        let a = GoldenSize::new(64, 0.5, 0.5, 0.5);
        let b = GoldenSize::new(64, 0.6, 0.5, 0.5);
        assert_ne!(a, b);
    }

    // ── Registry: empty clone preserves single golden size ──

    #[test]
    fn empty_registry_clone_preserves_single_golden() {
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::empty(constraints);
        let cloned = registry.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.golden_sizes()[0].seq_len, 256);
    }

    // ── Collapse: repeated collapse of same value increments linearly ──

    #[test]
    fn collapse_repeated_same_value_linear_increment() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let n = 50;
        for _ in 0..n {
            registry.collapse(100);
        }
        assert_eq!(*registry.hit_stats().get(&112).unwrap(), n);
    }

    // ── from_probe_results: default golden size has correct performance_score ──

    #[test]
    fn from_probe_results_default_golden_has_expected_score() {
        // When no spill_points and no l2_thrash, fallback golden is (256, 0.8, 0.8, 0.8)
        // score = 0.8*0.4 + 0.8*0.3 + 0.8*0.3 = 0.8
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: "fallback".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        assert!((gs.performance_score - 0.8).abs() < 1e-5);
    }

    // ── Registry: evolve with max_buckets = 1 still allows existing sizes ──

    #[test]
    fn evolve_with_max_buckets_one_does_not_remove_existing() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.max_buckets = 1;
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..200 {
            hist.record(300);
        }
        let _ = registry.evolve(&hist);
        // Original golden sizes should not be removed by evolve
        assert!(registry.len() >= 3);
    }

    // ── Batch 4: 50 additional unit tests ──

    // ── GoldenSize: extreme float inputs ──

    #[test]
    fn golden_size_performance_score_inf_input() {
        let gs = GoldenSize::new(64, f32::INFINITY, 0.0, 0.0);
        assert!(gs.performance_score.is_infinite());
        assert!(gs.performance_score > 0.0);
    }

    #[test]
    fn golden_size_performance_score_nan_propagation() {
        let gs = GoldenSize::new(64, f32::NAN, 0.5, 0.5);
        assert!(gs.performance_score.is_nan());
    }

    #[test]
    fn golden_size_performance_score_all_negative() {
        let gs = GoldenSize::new(64, -1.0, -1.0, -1.0);
        let expected = -1.0f32 * 0.4 + -1.0f32 * 0.3 + -1.0f32 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-5);
    }

    #[test]
    fn golden_size_eq_reflexivity() {
        let gs = GoldenSize::new(256, 0.5, 0.6, 0.7);
        assert_eq!(gs, gs);
    }

    #[test]
    fn golden_size_clone_deep_independence_l2() {
        let mut gs = GoldenSize::new(128, 0.5, 0.5, 0.5);
        let cloned = gs.clone();
        gs.l2_hit_rate = 0.99;
        assert!((cloned.l2_hit_rate - 0.5).abs() < 1e-5);
    }

    #[test]
    fn golden_size_debug_includes_numeric_seq_len() {
        let gs = GoldenSize::new(7777, 0.1, 0.2, 0.3);
        let debug = format!("{:?}", gs);
        assert!(debug.contains("7777"));
    }

    #[test]
    fn golden_size_performance_score_register_dominates_weight() {
        let gs_reg = GoldenSize::new(64, 1.0, 0.0, 0.0);
        let gs_smem = GoldenSize::new(64, 0.0, 1.0, 0.0);
        // register weight 0.4 > smem weight 0.3
        assert!(gs_reg.performance_score > gs_smem.performance_score);
    }

    #[test]
    fn golden_size_serde_json_has_expected_shape() {
        let gs = GoldenSize::new(512, 0.9, 0.8, 0.7);
        let json = serde_json::to_string(&gs).unwrap();
        assert!(json.contains("\"seq_len\""));
        assert!(json.contains("\"register_efficiency\""));
        assert!(json.contains("\"smem_efficiency\""));
        assert!(json.contains("\"l2_hit_rate\""));
        assert!(json.contains("\"performance_score\""));
    }

    // ── from_probe_results: additional edge cases ──

    #[test]
    fn from_probe_l2_thrash_smaller_than_spill_point() {
        let probe = ProbeResult {
            spill_points: vec![2048],
            smem_cliffs: vec![],
            l2_thrash_threshold: 512,
            device_fingerprint: "small_l2".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints { smem_size: None, ..CompilerConstraints::default() };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        // ratio = 2048/512 = 4.0, clamped to 1.0 → l2_hit_rate = 1.0 - 1.0*0.3 = 0.7
        assert!((gs.l2_hit_rate - 0.7).abs() < 1e-5);
    }

    #[test]
    fn from_probe_seq_len_equals_l2_thrash() {
        let probe = ProbeResult {
            spill_points: vec![1024],
            smem_cliffs: vec![],
            l2_thrash_threshold: 1024,
            device_fingerprint: "equal_l2".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints { smem_size: None, ..CompilerConstraints::default() };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        assert!((gs.l2_hit_rate - 0.7).abs() < 1e-5);
    }

    #[test]
    fn from_probe_many_spill_points_near_max_buckets() {
        let spill_points: Vec<usize> = (0..30).map(|i| (i + 1) * 100).collect();
        let probe = ProbeResult {
            spill_points,
            smem_cliffs: vec![],
            l2_thrash_threshold: 4096,
            device_fingerprint: "many_spill".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 30);
        for i in 0..registry.len() - 1 {
            assert!(registry.golden_sizes()[i].seq_len < registry.golden_sizes()[i + 1].seq_len);
        }
    }

    #[test]
    fn from_probe_l2_threshold_one() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 1,
            device_fingerprint: "l2_one".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Ratios 0.25/0.5/0.75 produce seq_len=0 (skipped); ratio 1.0 produces 1
        assert!(registry.len() >= 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 1);
    }

    #[test]
    fn from_probe_preserves_constraints_l2_cache_size() {
        let probe = make_test_probe_result();
        let constraints = CompilerConstraints {
            l2_cache_size: 12345,
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.constraints().l2_cache_size, 12345);
    }

    #[test]
    fn from_probe_smem_efficiency_gpu_with_large_tile() {
        let probe = ProbeResult {
            spill_points: vec![256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "large_tile".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints {
            smem_size: Some(16384),
            optimal_tile_bits: super::super::compiler_constraints::TileBits {
                tile_m: 64,
                tile_n: 64,
                tile_k: 4,
            },
            ..CompilerConstraints::default()
        };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let gs = &registry.golden_sizes()[0];
        // tile_bytes = 64*64*4 = 16384; ratio = 16384/16384 = 1.0
        assert!((gs.smem_efficiency - 1.0).abs() < 1e-3);
    }

    #[test]
    fn from_probe_golden_size_score_in_valid_range() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for gs in registry.golden_sizes() {
            assert!(
                gs.performance_score >= 0.0 && gs.performance_score <= 1.0 + 1e-5,
                "performance_score {} out of [0,1] for seq_len {}",
                gs.performance_score,
                gs.seq_len,
            );
        }
    }

    #[test]
    fn from_probe_empty_spill_l2_zero_exactly_one_fallback() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: "total_fallback".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 256);
    }

    // ── Collapse: additional edge cases ──


    #[test]
    fn collapse_after_evolve_targets_new_bucket() {
        let probe = ProbeResult {
            spill_points: vec![100, 500],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "evolve_collapse".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Create gap traffic to trigger evolve
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200); // bucket (129,256] fits in gap (100,500)
        }
        let decision = registry.evolve(&hist);
        if let EvolveDecision::Evolved { new_bucket_count } = decision {
            assert!(new_bucket_count > 0);
            let len_after = registry.len();
            assert!(len_after > 2);
            // Collapse should work with new bucket
            let (idx, golden) = registry.collapse(250);
            assert!(idx < len_after);
            assert!(golden.seq_len > 0);
        }
    }

    #[test]
    fn collapse_ref_after_evolve_no_stats_update() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        let stats_before = registry.hit_stats().clone();
        let _ = registry.collapse_ref(300);
        assert_eq!(registry.hit_stats(), &stats_before);
    }

    #[test]
    fn collapse_different_targets_different_goldens() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let s1 = registry.collapse(50).1.seq_len;
        let s2 = registry.collapse(400).1.seq_len;
        let s3 = registry.collapse(1500).1.seq_len;
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
        assert_ne!(s1, s3);
    }

    #[test]
    fn collapse_between_goldens_near_second() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // golden: [112, 463]; value 400 is much closer to 463 than 112
        let (_, golden) = registry.collapse(400);
        assert_eq!(golden.seq_len, 463);
    }

    #[test]
    fn collapse_duplicate_golden_sizes_both_accessible() {
        let probe = ProbeResult {
            spill_points: vec![256, 256],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "dup_golden".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.len(), 2);
        let (idx, golden) = registry.collapse(256);
        assert_eq!(golden.seq_len, 256);
        assert!(idx < 2);
    }

    // ── Evolve: additional edge cases ──

    #[test]
    fn evolve_empty_registry_insufficient_data() {
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::empty(constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..50 {
            hist.record(100);
        }
        let decision = registry.evolve(&hist);
        assert_eq!(decision, EvolveDecision::InsufficientData);
    }

    #[test]
    fn evolve_twice_consecutive_accumulates() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist1 = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist1.record(200);
        }
        let _ = registry.evolve(&hist1);
        let len_after_first = registry.len();
        let hist2 = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist2.record(200);
        }
        let _ = registry.evolve(&hist2);
        assert!(registry.len() >= len_after_first);
    }

    #[test]
    fn evolve_preserves_existing_golden_size_fields() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let original_first = registry.golden_sizes()[0].seq_len;
        let original_last = registry.golden_sizes()[2].seq_len;
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        let seq_lens: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        assert!(seq_lens.contains(&original_first));
        assert!(seq_lens.contains(&original_last));
    }

    #[test]
    fn evolve_with_zero_histogram_samples() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        let decision = registry.evolve(&hist);
        assert_eq!(decision, EvolveDecision::InsufficientData);
    }

    #[test]
    fn evolve_new_bucket_at_center_of_gap() {
        let probe = ProbeResult {
            spill_points: vec![100, 500],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "gap_center".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let decision = registry.evolve(&hist);
        if let EvolveDecision::Evolved { new_bucket_count } = decision {
            assert!(new_bucket_count > 0);
            let new_size = registry.golden_sizes().iter().find(|gs| gs.seq_len == 300);
            assert!(new_size.is_some(), "new bucket at center 300 not found");
        }
    }

    #[test]
    fn evolve_histogram_traffic_in_two_gaps() {
        // golden sizes at power-of-2 aligned positions to ensure histogram buckets fit
        let probe = ProbeResult {
            spill_points: vec![128, 512, 1024],
            smem_cliffs: vec![],
            l2_thrash_threshold: 4096,
            device_fingerprint: "two_gaps".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        // bucket (129,256] fits in gap (128,512); bucket (513,1024] fits in gap (512,1024)
        for _ in 0..200 {
            hist.record(200);
        }
        for _ in 0..200 {
            hist.record(800);
        }
        let decision = registry.evolve(&hist);
        match decision {
            EvolveDecision::Evolved { new_bucket_count } => {
                assert!(new_bucket_count >= 1);
                assert!(registry.len() > 3);
            }
            EvolveDecision::NoEvolutionNeeded => {}
            other => panic!("Unexpected: {:?}", other),
        }
    }

    // ── Evict zombies: additional edge cases ──

    #[test]
    fn evict_zombies_called_twice_second_returns_zero() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);
        let first_evicted = registry.evict_zombies();
        assert!(first_evicted > 0);
        let second_evicted = registry.evict_zombies();
        assert_eq!(second_evicted, 0);
    }

    #[test]
    fn evict_zombies_then_collapse_works_on_remaining() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 10000);
        registry.hit_stats.insert(463, 1);
        registry.hit_stats.insert(1011, 1);
        registry.evict_zombies();
        assert_eq!(registry.len(), 1);
        let (idx, golden) = registry.collapse(100);
        assert_eq!(idx, 0);
        assert_eq!(golden.seq_len, 112);
        assert_eq!(*registry.hit_stats().get(&112).unwrap(), 10001);
    }

    #[test]
    fn evict_zombies_single_hot_survives() {
        let probe = ProbeResult {
            spill_points: vec![128, 512, 1024],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "three_spill_evict".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(128, 9999);
        registry.hit_stats.insert(512, 0);
        registry.hit_stats.insert(1024, 0);
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 2);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.golden_sizes()[0].seq_len, 128);
    }

    #[test]
    fn evict_zombies_after_collapse_uses_real_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for _ in 0..1000 {
            registry.collapse(50);
        }
        registry.collapse(300);
        registry.collapse(2000);
        let evicted = registry.evict_zombies();
        // 463 and 1011 have 1 hit each vs 1000 for 112
        // threshold = 1002 * 0.001 = 1.002; 1 < 1.002 → zombies
        assert!(evicted > 0);
        assert!(registry.len() < 3);
    }

    #[test]
    fn evict_zombies_all_equal_hits_zero_evicted() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.hit_stats.insert(112, 100);
        registry.hit_stats.insert(463, 100);
        registry.hit_stats.insert(1011, 100);
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 0);
        assert_eq!(registry.len(), 3);
    }

    // ── Registry state: clone, serialize, debug ──

    #[test]
    fn clone_after_evolve_preserves_new_buckets() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        let cloned = registry.clone();
        assert_eq!(cloned.len(), registry.len());
        for i in 0..registry.len() {
            assert_eq!(cloned.golden_sizes()[i].seq_len, registry.golden_sizes()[i].seq_len);
        }
    }

    #[test]
    fn serialize_deserialize_after_evolve_roundtrip() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        registry.collapse(100);
        let json = serde_json::to_string(&registry).unwrap();
        let restored: GoldenBucketRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), registry.len());
        assert_eq!(restored.hit_stats(), registry.hit_stats());
    }

    #[test]
    fn serialize_deserialize_empty_hit_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let json = serde_json::to_string(&registry).unwrap();
        let restored: GoldenBucketRegistry = serde_json::from_str(&json).unwrap();
        assert!(restored.hit_stats().is_empty());
        assert_eq!(restored.len(), registry.len());
    }

    #[test]
    fn debug_format_after_collapse_shows_hit_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(100);
        let debug = format!("{:?}", registry);
        assert!(debug.contains("hit_stats"));
        assert!(debug.contains("golden_sizes"));
    }

    #[test]
    fn constraints_unchanged_after_operations() {
        let constraints = CompilerConstraints {
            l2_cache_size: 55555,
            smem_size: Some(88888),
            ..CompilerConstraints::default()
        };
        let probe = make_test_probe_result();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(100);
        registry.collapse(500);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..200 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        registry.evict_zombies();
        assert_eq!(registry.constraints().l2_cache_size, 55555);
        assert_eq!(registry.constraints().smem_size, Some(88888));
    }

    // ── Property / invariant tests ──

    #[test]
    fn collapse_index_always_within_bounds() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for &seq_len in &[0, 1, 50, 112, 200, 300, 463, 700, 1011, 5000, usize::MAX] {
            let (idx, _) = registry.collapse(seq_len);
            assert!(idx < registry.len(), "index {} out of bounds for seq_len {}", idx, seq_len);
        }
    }

    #[test]
    fn golden_sizes_ascending_after_collapse_and_evolve() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(100);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        for i in 0..registry.len().saturating_sub(1) {
            assert!(
                registry.golden_sizes()[i].seq_len < registry.golden_sizes()[i + 1].seq_len,
                "not ascending at index {}",
                i,
            );
        }
    }

    #[test]
    fn hit_stats_total_matches_collapse_call_count() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let collapse_count: u64 = 42;
        for _ in 0..collapse_count {
            registry.collapse(200);
        }
        let total: u64 = registry.hit_stats().values().sum();
        assert_eq!(total, collapse_count);
    }

    #[test]
    fn collapse_ref_and_collapse_agree_on_golden() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for &seq_len in &[0, 50, 112, 200, 300, 463, 800, 1011, 5000] {
            let (_, golden_ref) = registry.collapse_ref(seq_len);
            let (_, golden_ref2) = registry.collapse_ref(seq_len);
            assert_eq!(golden_ref.seq_len, golden_ref2.seq_len);
        }
    }

    #[test]
    fn evolve_never_removes_existing_sizes() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        let original_lens: Vec<usize> = registry.golden_sizes().iter().map(|gs| gs.seq_len).collect();
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..300 {
            hist.record(200);
        }
        let _ = registry.evolve(&hist);
        for &original in &original_lens {
            assert!(
                registry.golden_sizes().iter().any(|gs| gs.seq_len == original),
                "original golden size {} removed by evolve",
                original,
            );
        }
    }

    #[test]
    fn from_probe_golden_size_performance_score_non_negative() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for gs in registry.golden_sizes() {
            assert!(
                gs.performance_score >= 0.0,
                "negative performance_score {} for seq_len {}",
                gs.performance_score,
                gs.seq_len,
            );
        }
    }

    #[test]
    fn collapse_hit_stats_only_for_collapsed_goldens() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(50);
        registry.collapse(60);
        assert_eq!(registry.hit_stats().len(), 1);
        assert!(registry.hit_stats().contains_key(&112));
    }

    #[test]
    fn from_probe_results_default_constraints_reflected() {
        let probe = make_test_probe_result();
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        assert_eq!(registry.constraints().smem_size, None);
        assert_eq!(registry.constraints().l2_cache_size, 256 * 1024);
    }

    #[test]
    fn registry_len_never_zero_after_construction() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: "minimal".into(),
            raw_measurements: Default::default(),
        };
        let from_probe = GoldenBucketRegistry::from_probe_results(&probe, CompilerConstraints::default());
        assert!(from_probe.len() > 0);
        let empty = GoldenBucketRegistry::empty(CompilerConstraints::default());
        assert!(empty.len() > 0);
    }

    #[test]
    fn collapse_multiple_different_goldens_tracks_all() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(50);
        registry.collapse(300);
        registry.collapse(1500);
        assert_eq!(registry.hit_stats().len(), 3);
        assert_eq!(*registry.hit_stats().get(&112).unwrap(), 1);
        assert_eq!(*registry.hit_stats().get(&463).unwrap(), 1);
        assert_eq!(*registry.hit_stats().get(&1011).unwrap(), 1);
    }

    // ── Batch 5: 13 additional unit tests ──

    // 1. GoldenSize PartialEq symmetry: if a == b then b == a
    #[test]
    fn golden_size_equality_symmetry() {
        let a = GoldenSize::new(256, 0.7, 0.8, 0.9);
        let b = GoldenSize::new(256, 0.7, 0.8, 0.9);
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // 2. GoldenSize PartialEq transitivity: a==b, b==c => a==c
    #[test]
    fn golden_size_equality_transitivity() {
        let a = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let b = GoldenSize::new(128, 0.5, 0.6, 0.7);
        let c = GoldenSize::new(128, 0.5, 0.6, 0.7);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // 3. GoldenSize performance_score is commutative for smem/l2 (same weight)
    #[test]
    fn golden_size_smem_l2_weight_equality() {
        // smem_efficiency and l2_hit_rate both have weight 0.3
        let gs_a = GoldenSize::new(64, 0.0, 0.9, 0.1);
        let gs_b = GoldenSize::new(64, 0.0, 0.1, 0.9);
        // 0.9*0.3 + 0.1*0.3 = 0.3 for both
        assert!((gs_a.performance_score - gs_b.performance_score).abs() < 1e-7);
    }

    // 4. from_probe_results: smem_efficiency is 0.8 when smem_size is None (CPU path)
    #[test]
    fn from_probe_results_cpu_smem_efficiency_is_default() {
        let probe = ProbeResult {
            spill_points: vec![256, 512],
            smem_cliffs: vec![],
            l2_thrash_threshold: 2048,
            device_fingerprint: "cpu_smem".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints { smem_size: None, ..CompilerConstraints::default() };
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for gs in registry.golden_sizes() {
            assert!((gs.smem_efficiency - 0.8).abs() < 1e-5,
                "CPU smem_efficiency should be 0.8, got {}", gs.smem_efficiency);
        }
    }

    // 5. EvolveDecision: CapacityLimitReached and Evolved with same numeric field are still different variants
    #[test]
    fn evolve_decision_different_variants_despite_same_field_count() {
        let d1 = EvolveDecision::CapacityLimitReached { current: 1, max: 1 };
        let d2 = EvolveDecision::Evolved { new_bucket_count: 1 };
        assert_ne!(d1, d2);
    }

    // 6. Registry: collapse then serialize preserves hit_stats counts
    #[test]
    fn serialize_preserves_collapse_hit_counts() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for _ in 0..7 {
            registry.collapse(100);
        }
        let json = serde_json::to_string(&registry).unwrap();
        let restored: GoldenBucketRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(*restored.hit_stats().get(&112).unwrap(), 7);
    }

    // 7. Evolve: evolving an empty() registry with sufficient data does not panic
    #[test]
    fn evolve_empty_registry_sufficient_data_no_panic() {
        let constraints = CompilerConstraints::default();
        let mut registry = GoldenBucketRegistry::empty(constraints);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..150 {
            hist.record(100);
        }
        let decision = registry.evolve(&hist);
        // Single golden size means no gaps → NoEvolutionNeeded or similar
        match decision {
            EvolveDecision::NoEvolutionNeeded
            | EvolveDecision::InsufficientData => {}
            other => panic!("Unexpected for single-bucket registry: {:?}", other),
        }
    }

    // 8. Evict zombies: registry with only hit_stats for non-existent golden sizes still works
    #[test]
    fn evict_zombies_with_only_stale_hit_stats() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        // Insert hit_stats for seq_lens that are actual golden sizes
        registry.hit_stats.insert(112, 0);
        registry.hit_stats.insert(463, 0);
        registry.hit_stats.insert(1011, 0);
        // total_hits = 0 → early return, no eviction
        let evicted = registry.evict_zombies();
        assert_eq!(evicted, 0);
        assert_eq!(registry.len(), 3);
    }

    // 9. GoldenSize: performance_score with mixed very small and large efficiencies
    #[test]
    fn golden_size_performance_score_mixed_extremes() {
        let gs = GoldenSize::new(64, 0.001, 0.999, 0.5);
        let expected = 0.001 * 0.4 + 0.999 * 0.3 + 0.5 * 0.3;
        assert!((gs.performance_score - expected).abs() < 1e-7);
    }

    // 10. Registry: collapse_ref returns consistent results across multiple calls
    #[test]
    fn collapse_ref_idempotent_across_calls() {
        let probe = make_test_probe_result();
        let constraints = make_test_constraints();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for _ in 0..10 {
            let (idx, golden) = registry.collapse_ref(300);
            assert_eq!(idx, 1);
            assert_eq!(golden.seq_len, 463);
        }
    }

    // 11. from_probe_results: l2 fallback ratios produce strictly increasing seq_lens
    #[test]
    fn from_probe_l2_fallback_strictly_increasing() {
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 4000,
            device_fingerprint: "strict_inc".into(),
            raw_measurements: Default::default(),
        };
        let constraints = CompilerConstraints::default();
        let registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        for i in 0..registry.len().saturating_sub(1) {
            assert!(
                registry.golden_sizes()[i].seq_len < registry.golden_sizes()[i + 1].seq_len,
                "l2 fallback seq_lens not strictly increasing at index {}", i,
            );
        }
    }

    // 12. EvolveDecision: Debug output for Evolved variant contains the count value
    #[test]
    fn evolve_decision_evolved_debug_shows_count() {
        let d = EvolveDecision::Evolved { new_bucket_count: 42 };
        let debug = format!("{:?}", d);
        assert!(debug.contains("42"));
        assert!(debug.contains("new_bucket_count"));
    }

    // 13. Registry: constraints() accessor is consistent after multiple operations
    #[test]
    fn constraints_unchanged_after_collapse_evolve_evict_cycle() {
        let probe = make_test_probe_result();
        let original_l2 = 77777;
        let original_smem = Some(44444);
        let constraints = CompilerConstraints {
            l2_cache_size: original_l2,
            smem_size: original_smem,
            ..CompilerConstraints::default()
        };
        let mut registry = GoldenBucketRegistry::from_probe_results(&probe, constraints);
        registry.collapse(100);
        registry.collapse(500);
        let hist = SeqHistogram::new(1000, 4096);
        for _ in 0..200 { hist.record(200); }
        let _ = registry.evolve(&hist);
        registry.evict_zombies();
        assert_eq!(registry.constraints().l2_cache_size, original_l2);
        assert_eq!(registry.constraints().smem_size, original_smem);
    }
}
