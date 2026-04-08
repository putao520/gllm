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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
                let tile_bytes = tile.tile_m * tile.tile_n * 4; // f32
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
    use crate::sensors::MemoryNetworkSensors;
    use gllm_kernels::dispatch::DeviceProfile;

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
        let (idx, golden) = registry.collapse(100);
        assert_eq!(golden.seq_len, 112);

        let (idx2, golden2) = registry.collapse(300);
        assert_eq!(golden2.seq_len, 463);

        let (idx3, golden3) = registry.collapse(2000);
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
}
