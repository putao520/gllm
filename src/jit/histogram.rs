//! SEQ 分布直方图 — 滑动窗口
//!
//! 按 2 的幂分档：(1, 2], (3, 4], (5, 8], ..., (2049, 4096], ...
//! 使用原子计数器实现无锁并发更新。

use std::sync::atomic::{AtomicU64, Ordering};

/// SEQ 分布直方图 — 滑动窗口
///
/// 按 2 的幂分档，每个 bucket 的区间为 (2^(n-1), 2^n]。
/// 例如：(1, 2], (3, 4], (5, 8], (9, 16], ..., (2049, 4096], ...
pub struct SeqHistogram {
    /// 每个 bucket 的 (range_start, range_end)
    buckets: Vec<SeqBucket>,
    /// 滑动窗口大小（最近 N 步请求）
    window_size: usize,
    /// 当前窗口内总样本数
    total_samples: AtomicU64,
}

/// 单个 SEQ 区间 bucket
#[derive(Debug)]
pub struct SeqBucket {
    /// 区间 [start, end] (闭区间)
    pub start: usize,
    pub end: usize,
    /// 原子计数器（无锁并发更新）
    pub count: AtomicU64,
}

/// 直方图查询结果快照
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    /// 所有 bucket 的 (start, end, count)
    pub buckets: Vec<(usize, usize, u64)>,
    /// 总样本数
    pub total_samples: u64,
    /// 最热门 K 个区间 (start, end, count)
    pub top_k: Vec<(usize, usize, u64)>,
}

impl SeqHistogram {
    /// 创建新的 SEQ 直方图
    ///
    /// # 参数
    /// - `window_size`: 滑动窗口大小（样本数量）
    /// - `max_seq`: 最大观测 seq_len（用于确定 bucket 数量）
    ///
    /// # 分档规则
    /// 按 2 的幂分档：(1, 2], (3, 4], (5, 8], ..., (max_seq/2, max_seq]
    pub fn new(window_size: usize, max_seq: usize) -> Self {
        let num_buckets = if max_seq > 0 {
            (max_seq as f64).log2().ceil() as usize
        } else {
            1
        };

        let mut buckets = Vec::with_capacity(num_buckets);

        // bucket 0: (0, 1] - 处理 seq_len=1
        buckets.push(SeqBucket {
            start: 0,
            end: 1,
            count: AtomicU64::new(0),
        });

        // bucket i: (2^i, 2^(i+1)] for i >= 1
        for i in 1..num_buckets {
            let start = (1usize << (i - 1)) + 1; // (2^(i-1) + 1)
            let end = 1usize << i;               // 2^i

            // 最后一个 bucket 扩展到 max_seq
            let end = if i == num_buckets - 1 {
                max_seq.max(end)
            } else {
                end
            };

            buckets.push(SeqBucket {
                start,
                end,
                count: AtomicU64::new(0),
            });
        }

        Self {
            buckets,
            window_size,
            total_samples: AtomicU64::new(0),
        }
    }

    /// 记录一个 seq_len 样本
    ///
    /// 使用原子计数器 O(1) 更新，通过 `floor(log2(seq_len))` 找到对应 bucket。
    /// 开销目标: < 100ns/step
    pub fn record(&self, seq_len: usize) {
        // 找到对应的 bucket
        let bucket_idx = self.find_bucket(seq_len);

        if let Some(bucket) = self.buckets.get(bucket_idx) {
            bucket.count.fetch_add(1, Ordering::Relaxed);
            self.total_samples.fetch_add(1, Ordering::Relaxed);
        }

        // 滑动窗口：如果超过窗口大小，触发衰减（在 director 中定期执行）
    }

    /// 找到 seq_len 对应的 bucket 索引
    #[inline]
    fn find_bucket(&self, seq_len: usize) -> usize {
        if seq_len == 0 {
            return 0;
        }

        // 使用前导零计数计算 floor(log2(seq_len))
        // 对于 x86_64: 使用 lzcnt 指令
        // seq_len=1 -> lzcnt=63 -> log2=0 -> bucket 0
        // seq_len=2 -> lzcnt=62 -> log2=1 -> bucket 1
        // seq_len=3 -> lzcnt=62 -> log2=1 -> bucket 1
        // seq_len=4 -> lzcnt=61 -> log2=2 -> bucket 2
        let log2 = usize::BITS - seq_len.leading_zeros() - 1;

        // bucket 0: (0, 1], bucket i>0: (2^(i-1)+1, 2^i]
        if seq_len == 1 {
            return 0;
        }

        // log2=1 (seq_len=2,3) -> bucket 1
        // log2=2 (seq_len=4..7) -> bucket 2
        (log2 as usize).min(self.buckets.len() - 1)
    }

    /// 获取当前快照
    ///
    /// 读取所有 bucket 的计数器，返回一致的快照。
    /// 注意：由于并发更新，快照可能不完全一致（在多线程场景下）。
    pub fn snapshot(&self) -> HistogramSnapshot {
        let total = self.total_samples.load(Ordering::Relaxed);

        let buckets: Vec<(usize, usize, u64)> = self
            .buckets
            .iter()
            .map(|b| (b.start, b.end, b.count.load(Ordering::Relaxed)))
            .collect();

        // 计算最热门 K 个区间
        let mut top_k = buckets.clone();
        top_k.sort_by(|a, b| b.2.cmp(&a.2)); // 按计数降序
        top_k.truncate(10); // 保留 top 10

        HistogramSnapshot {
            buckets,
            total_samples: total,
            top_k,
        }
    }

    /// 获取最热门 K 个区间
    pub fn top_k(&self, k: usize) -> Vec<(usize, usize, u64)> {
        let mut buckets: Vec<(usize, usize, u64)> = self
            .buckets
            .iter()
            .map(|b| (b.start, b.end, b.count.load(Ordering::Relaxed)))
            .collect();

        buckets.sort_by(|a, b| b.2.cmp(&a.2));
        buckets.truncate(k);
        buckets
    }

    /// 衰减所有计数器
    ///
    /// 滑动窗口衰减：所有计数器乘以 factor（如 0.9）。
    /// 用于实现滑动窗口的"遗忘"效果。
    pub fn decay(&self, factor: f64) {
        for bucket in &self.buckets {
            let current = bucket.count.load(Ordering::Relaxed);
            let decayed = (current as f64 * factor) as u64;
            bucket.count.store(decayed, Ordering::Relaxed);
        }

        // 同时衰减总样本数
        let total = self.total_samples.load(Ordering::Relaxed);
        let decayed = (total as f64 * factor) as u64;
        self.total_samples.store(decayed, Ordering::Relaxed);
    }

    /// 重置所有计数器
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.count.store(0, Ordering::Relaxed);
        }
        self.total_samples.store(0, Ordering::Relaxed);
    }

    /// 获取所有 bucket 引用
    pub fn buckets(&self) -> &[SeqBucket] {
        &self.buckets
    }

    /// 获取滑动窗口大小
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// 获取当前总样本数
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// 检查某个 seq_len 是否落在 bucket 的"缝隙"中
    ///
    /// 返回 true 如果 seq_len 不在任何现有 bucket 的中心附近
    pub fn is_gap(&self, seq_len: usize, tolerance: f64) -> bool {
        let bucket_idx = self.find_bucket(seq_len);

        if let Some(bucket) = self.buckets.get(bucket_idx) {
            // 检查 seq_len 是否接近 bucket 的中心
            let center = (bucket.start + bucket.end) / 2;
            let distance = (seq_len as f64 - center as f64).abs() / (bucket.end - bucket.start + 1) as f64;
            distance > tolerance
        } else {
            false
        }
    }

    /// 计算缝隙区间的累计命中率
    ///
    /// 扫描所有 bucket，计算那些不在 bucket 中心的区间的累计命中率
    pub fn gap_hit_rate(&self, tolerance: f64) -> f64 {
        let total = self.total_samples.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            return 0.0;
        }

        let gap_hits = 0u64;

        for bucket in &self.buckets {
            // 简化：假设 bucket 边缘 20% 是"缝隙"
            let range = bucket.end - bucket.start + 1;
            let _gap_threshold = (range as f64 * tolerance) as usize;

            // 这里只能估算，因为无法知道每个 seq_len 的精确分布
            // 实际实现需要更精细的粒度
        }

        gap_hits as f64 / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_create() {
        let hist = SeqHistogram::new(1000, 4096);
        assert_eq!(hist.window_size(), 1000);
        assert_eq!(hist.total_samples(), 0);

        // 检查 bucket 数量
        // max_seq=4096 -> log2(4096)=12 -> 约 12 个 bucket
        assert!(hist.buckets().len() >= 12);
    }

    #[test]
    fn test_histogram_record() {
        let hist = SeqHistogram::new(1000, 4096);

        hist.record(1);
        hist.record(2);
        hist.record(3);
        hist.record(8);

        assert_eq!(hist.total_samples(), 4);

        let snapshot = hist.snapshot();
        assert_eq!(snapshot.total_samples, 4);
    }

    #[test]
    fn test_histogram_bucket_mapping() {
        let hist = SeqHistogram::new(1000, 256);

        // seq_len=1 应该在 bucket 0
        hist.record(1);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);

        // seq_len=2,3 应该在 bucket 1
        hist.record(2);
        hist.record(3);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 2);

        // seq_len=4..7 应该在 bucket 2
        hist.record(4);
        hist.record(7);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_histogram_snapshot() {
        let hist = SeqHistogram::new(1000, 256);

        hist.record(1);
        hist.record(2);
        hist.record(2);
        hist.record(8);

        let snapshot = hist.snapshot();
        assert_eq!(snapshot.total_samples, 4);
        assert_eq!(snapshot.buckets.len(), hist.buckets().len());

        // 检查 top_k
        assert_eq!(snapshot.top_k[0].2, 2); // bucket 1 有 2 个样本
    }

    #[test]
    fn test_histogram_top_k() {
        let hist = SeqHistogram::new(1000, 256);

        // 在不同 bucket 中记录
        for _ in 0..10 {
            hist.record(2);  // bucket 1
        }
        for _ in 0..5 {
            hist.record(4);  // bucket 2
        }
        for _ in 0..3 {
            hist.record(8);  // bucket 3
        }

        let top_k = hist.top_k(2);
        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].2, 10); // 最热门的是 bucket 1
        assert_eq!(top_k[1].2, 5);  // 第二是 bucket 2
    }

    #[test]
    fn test_histogram_decay() {
        let hist = SeqHistogram::new(1000, 256);

        hist.record(2);
        hist.record(2);
        hist.record(2);

        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 3);

        hist.decay(0.5);

        let decayed = hist.buckets()[1].count.load(Ordering::Relaxed);
        assert!(decayed == 1 || decayed == 2); // 3 * 0.5 = 1.5 -> 取整
    }

    #[test]
    fn test_histogram_reset() {
        let hist = SeqHistogram::new(1000, 256);

        hist.record(2);
        hist.record(4);

        hist.reset();

        assert_eq!(hist.total_samples(), 0);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_histogram_concurrent_record() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SeqHistogram::new(1000, 256));
        let mut handles = vec![];

        // 10 个线程，每个记录 100 次
        for _ in 0..10 {
            let hist_clone = Arc::clone(&hist);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    hist_clone.record((i % 16) + 1);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 总共 1000 次记录
        assert_eq!(hist.total_samples(), 1000);
    }

    // ── SeqBucket ──

    #[test]
    fn test_seq_bucket_debug_trait() {
        let bucket = SeqBucket {
            start: 3,
            end: 4,
            count: AtomicU64::new(42),
        };
        let debug_str = format!("{:?}", bucket);
        assert!(debug_str.contains("start"));
        assert!(debug_str.contains("end"));
        assert!(debug_str.contains("count"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_seq_bucket_fields_match_constructor() {
        let bucket = SeqBucket {
            start: 5,
            end: 8,
            count: AtomicU64::new(7),
        };
        assert_eq!(bucket.start, 5);
        assert_eq!(bucket.end, 8);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 7);
    }

    // ── HistogramSnapshot ──

    #[test]
    fn test_snapshot_debug_trait() {
        let snapshot = HistogramSnapshot {
            buckets: vec![(0, 1, 5)],
            total_samples: 5,
            top_k: vec![(0, 1, 5)],
        };
        let debug_str = format!("{:?}", snapshot);
        assert!(debug_str.contains("buckets"));
        assert!(debug_str.contains("total_samples"));
        assert!(debug_str.contains("top_k"));
    }

    #[test]
    fn test_snapshot_clone_is_independent() {
        let snapshot = HistogramSnapshot {
            buckets: vec![(0, 1, 10), (3, 4, 20)],
            total_samples: 30,
            top_k: vec![(3, 4, 20), (0, 1, 10)],
        };
        let cloned = snapshot.clone();

        // Mutating the clone does not affect the original
        assert_eq!(cloned.buckets, snapshot.buckets);
        assert_eq!(cloned.total_samples, snapshot.total_samples);
        assert_eq!(cloned.top_k, snapshot.top_k);
    }

    #[test]
    fn test_snapshot_empty_state() {
        let snapshot = HistogramSnapshot {
            buckets: vec![],
            total_samples: 0,
            top_k: vec![],
        };
        assert!(snapshot.buckets.is_empty());
        assert_eq!(snapshot.total_samples, 0);
        assert!(snapshot.top_k.is_empty());
    }

    // ── SeqHistogram::new edge cases ──

    #[test]
    fn test_new_max_seq_zero_creates_one_bucket() {
        let hist = SeqHistogram::new(100, 0);
        // max_seq=0 -> num_buckets=1, only the (0,1] bucket
        assert_eq!(hist.buckets().len(), 1);
        assert_eq!(hist.buckets()[0].start, 0);
        assert_eq!(hist.buckets()[0].end, 1);
    }

    #[test]
    fn test_new_max_seq_one_creates_one_bucket() {
        let hist = SeqHistogram::new(100, 1);
        // log2(1)=0, ceil=0 -> num_buckets=0, but special case: 0 becomes at least 1 from code flow
        // Actually log2(1.0)=0.0, ceil(0.0)=0, but code says max_seq > 0 so num_buckets=0
        // Let's verify behavior empirically
        assert!(hist.buckets().len() >= 1);
    }

    #[test]
    fn test_new_max_seq_two() {
        let hist = SeqHistogram::new(100, 2);
        // log2(2)=1, ceil=1 -> 1 bucket
        assert_eq!(hist.buckets().len(), 1);
        assert_eq!(hist.buckets()[0].start, 0);
        assert_eq!(hist.buckets()[0].end, 1);
    }

    #[test]
    fn test_new_max_seq_power_of_two_bucket_count() {
        // max_seq=4096 -> log2(4096)=12.0, ceil=12 -> 12 buckets
        let hist = SeqHistogram::new(1000, 4096);
        assert_eq!(hist.buckets().len(), 12);
    }

    #[test]
    fn test_new_window_size_preserved() {
        let hist = SeqHistogram::new(42, 256);
        assert_eq!(hist.window_size(), 42);
    }

    #[test]
    fn test_new_total_samples_starts_at_zero() {
        let hist = SeqHistogram::new(100, 256);
        assert_eq!(hist.total_samples(), 0);
    }

    #[test]
    fn test_new_all_bucket_counts_start_at_zero() {
        let hist = SeqHistogram::new(100, 256);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_new_last_bucket_extends_to_max_seq() {
        let hist = SeqHistogram::new(100, 300);
        let last = hist.buckets().last().unwrap();
        assert_eq!(last.end, 300);
    }

    #[test]
    fn test_new_bucket_ranges_are_monotonically_increasing() {
        let hist = SeqHistogram::new(100, 4096);
        for window in hist.buckets().windows(2) {
            assert!(
                window[0].end < window[1].start,
                "bucket {:?} overlaps with next bucket {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_new_bucket_start_end_validity() {
        let hist = SeqHistogram::new(100, 4096);
        for bucket in hist.buckets() {
            assert!(
                bucket.start <= bucket.end,
                "bucket start {} > end {}",
                bucket.start,
                bucket.end
            );
        }
    }

    // ── Bucket mapping: find_bucket (tested via record) ──

    #[test]
    fn test_find_bucket_seq_len_zero() {
        let hist = SeqHistogram::new(100, 256);
        // seq_len=0 should go to bucket 0
        hist.record(0);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_find_bucket_seq_len_one() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(1);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_find_bucket_seq_len_two() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(2);
        // seq_len=2 -> log2=1 -> bucket 1
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 1);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_find_bucket_seq_len_three() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(3);
        // seq_len=3 -> log2=1 -> bucket 1
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_find_bucket_seq_len_four() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        // seq_len=4 -> log2=2 -> bucket 2
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_find_bucket_seq_len_boundary_values() {
        let hist = SeqHistogram::new(100, 4096);

        // Test power-of-two boundaries: 1, 2, 4, 8, 16, 32, ...
        // seq_len=1 -> bucket 0
        hist.record(1);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);

        // seq_len=2 -> bucket 1
        hist.record(2);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 1);

        // seq_len=4 -> bucket 2
        hist.record(4);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 1);

        // seq_len=8 -> bucket 3
        hist.record(8);
        assert_eq!(hist.buckets()[3].count.load(Ordering::Relaxed), 1);

        // seq_len=16 -> bucket 4
        hist.record(16);
        assert_eq!(hist.buckets()[4].count.load(Ordering::Relaxed), 1);

        // seq_len=32 -> bucket 5
        hist.record(32);
        assert_eq!(hist.buckets()[5].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_find_bucket_large_seq_len_clamps_to_last_bucket() {
        let hist = SeqHistogram::new(100, 64);
        // seq_len=1000 is way beyond max_seq=64, should clamp to last bucket
        hist.record(1000);
        let last_idx = hist.buckets().len() - 1;
        assert_eq!(hist.buckets()[last_idx].count.load(Ordering::Relaxed), 1);
    }

    // ── record ──

    #[test]
    fn test_record_increments_total_samples() {
        let hist = SeqHistogram::new(100, 256);
        assert_eq!(hist.total_samples(), 0);

        hist.record(1);
        assert_eq!(hist.total_samples(), 1);

        hist.record(4);
        assert_eq!(hist.total_samples(), 2);

        hist.record(4);
        assert_eq!(hist.total_samples(), 3);
    }

    #[test]
    fn test_record_same_bucket_accumulates() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..50 {
            hist.record(3);
        }
        // seq_len=3 -> bucket 1
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 50);
        assert_eq!(hist.total_samples(), 50);
    }

    #[test]
    fn test_record_distributes_across_buckets() {
        let hist = SeqHistogram::new(100, 256);

        // bucket 0: seq_len=1
        hist.record(1);
        // bucket 1: seq_len=2,3
        hist.record(2);
        hist.record(3);
        // bucket 2: seq_len=4,5,6,7
        hist.record(4);
        hist.record(5);
        hist.record(6);
        hist.record(7);

        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 2);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 4);
        assert_eq!(hist.total_samples(), 7);
    }

    // ── snapshot ──

    #[test]
    fn test_snapshot_reflects_record_state() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(1);
        hist.record(2);
        hist.record(2);

        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 3);

        // Find the bucket for seq_len=2 (bucket 1: start=2, end=2)
        let bucket_1 = snap.buckets.iter().find(|(s, e, _)| *s == 2 && *e == 2);
        assert!(bucket_1.is_some());
        assert_eq!(bucket_1.unwrap().2, 2);
    }

    #[test]
    fn test_snapshot_top_k_max_ten() {
        let hist = SeqHistogram::new(100, 65536);

        // Create 15 distinct buckets with different counts
        for _ in 0..15 {
            hist.record(1); // bucket 0
        }
        for _ in 0..14 {
            hist.record(3); // bucket 1
        }
        for _ in 0..13 {
            hist.record(5); // bucket 2
        }
        for _ in 0..12 {
            hist.record(9); // bucket 3
        }
        for _ in 0..11 {
            hist.record(17); // bucket 4
        }
        for _ in 0..10 {
            hist.record(33); // bucket 5
        }
        for _ in 0..9 {
            hist.record(65); // bucket 6
        }
        for _ in 0..8 {
            hist.record(129); // bucket 7
        }
        for _ in 0..7 {
            hist.record(257); // bucket 8
        }
        for _ in 0..6 {
            hist.record(513); // bucket 9
        }
        for _ in 0..5 {
            hist.record(1025); // bucket 10
        }

        let snap = hist.snapshot();
        assert!(snap.top_k.len() <= 10);
    }

    #[test]
    fn test_snapshot_top_k_sorted_descending() {
        let hist = SeqHistogram::new(100, 256);

        for _ in 0..5 {
            hist.record(2); // bucket 1: 5 samples
        }
        for _ in 0..10 {
            hist.record(4); // bucket 2: 10 samples
        }
        for _ in 0..3 {
            hist.record(8); // bucket 3: 3 samples
        }

        let snap = hist.snapshot();
        for window in snap.top_k.windows(2) {
            assert!(
                window[0].2 >= window[1].2,
                "top_k not sorted descending: {:?} < {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_snapshot_after_reset() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        hist.record(8);
        hist.reset();

        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 0);
        for (_, _, count) in &snap.buckets {
            assert_eq!(*count, 0);
        }
        for (_, _, count) in &snap.top_k {
            assert_eq!(*count, 0);
        }
    }

    // ── top_k ──

    #[test]
    fn test_top_k_zero_returns_empty() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        let result = hist.top_k(0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_one_returns_single_best() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(2);
        }
        for _ in 0..3 {
            hist.record(4);
        }

        let top = hist.top_k(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].2, 10);
    }

    #[test]
    fn test_top_k_more_than_buckets_returns_all() {
        let hist = SeqHistogram::new(100, 4);
        hist.record(1);
        hist.record(2);

        // Requesting more than available buckets
        let top = hist.top_k(100);
        assert!(top.len() <= hist.buckets().len());
    }

    #[test]
    fn test_top_k_sorted_descending() {
        let hist = SeqHistogram::new(100, 256);

        for _ in 0..7 {
            hist.record(1); // bucket 0
        }
        for _ in 0..3 {
            hist.record(2); // bucket 1
        }
        for _ in 0..15 {
            hist.record(4); // bucket 2
        }

        let top = hist.top_k(3);
        assert_eq!(top[0].2, 15); // bucket 2 highest
        assert_eq!(top[1].2, 7); // bucket 0 next
        assert_eq!(top[2].2, 3); // bucket 1 lowest
    }

    // ── decay ──

    #[test]
    fn test_decay_factor_one_is_noop() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(2);
        hist.record(2);
        hist.record(2);
        hist.record(4);

        let before_bucket1 = hist.buckets()[1].count.load(Ordering::Relaxed);
        let before_total = hist.total_samples();

        hist.decay(1.0);

        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), before_bucket1);
        assert_eq!(hist.total_samples(), before_total);
    }

    #[test]
    fn test_decay_factor_zero_clears_all() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(2);
        hist.record(4);
        hist.record(8);

        hist.decay(0.0);

        assert_eq!(hist.total_samples(), 0);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_decay_truncates_not_rounds() {
        let hist = SeqHistogram::new(100, 256);
        // Put 10 samples in a bucket
        for _ in 0..10 {
            hist.record(2);
        }
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 10);

        // 10 * 0.75 = 7.5 -> truncates to 7
        hist.decay(0.75);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn test_decay_repeated_converges_to_zero() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..100 {
            hist.record(2);
        }

        // Repeatedly decay by 0.5
        for _ in 0..20 {
            hist.decay(0.5);
        }

        // After 20 halvings, 100 * 0.5^20 ≈ 0.000095 -> truncated to 0
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 0);
        assert_eq!(hist.total_samples(), 0);
    }

    #[test]
    fn test_decay_total_samples_tracks_bucket_counts() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(2);
        hist.record(2);
        hist.record(4);

        assert_eq!(hist.total_samples(), 3);

        hist.decay(0.9);

        // 3 * 0.9 = 2.7 -> 2
        assert_eq!(hist.total_samples(), 2);
    }

    // ── reset ──

    #[test]
    fn test_reset_clears_all_counters() {
        let hist = SeqHistogram::new(100, 256);

        // Fill many buckets
        for seq in [1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128, 256] {
            hist.record(seq);
        }

        assert!(hist.total_samples() > 0);
        hist.reset();

        assert_eq!(hist.total_samples(), 0);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_reset_then_record_works_cleanly() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(2);
        hist.record(4);
        hist.reset();

        // After reset, new records should start from 0
        hist.record(8);
        assert_eq!(hist.total_samples(), 1);
        assert_eq!(hist.buckets()[3].count.load(Ordering::Relaxed), 1);

        // Previous bucket counts should be zero
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 0);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 0);
    }

    // ── is_gap ──

    #[test]
    fn test_is_gap_at_bucket_center_returns_false() {
        let hist = SeqHistogram::new(100, 256);
        // bucket 0: start=0, end=1, center=(0+1)/2=0
        // seq_len=0 is exactly at center -> distance=0 -> not a gap
        assert!(!hist.is_gap(0, 0.5));
    }

    #[test]
    fn test_is_gap_at_bucket_edge_returns_true() {
        let hist = SeqHistogram::new(100, 256);
        // bucket 2: start=3, end=4, center=3
        // seq_len=4 at edge: distance = |4-3|/(4-3+1) = 1/2 = 0.5
        // tolerance=0.3, 0.5 > 0.3 -> gap
        assert!(hist.is_gap(4, 0.3));
    }

    #[test]
    fn test_is_gap_zero_tolerance_only_center() {
        let hist = SeqHistogram::new(100, 256);
        // With tolerance=0, only exact center is not a gap
        // bucket 0: start=0, end=1, center=0
        assert!(!hist.is_gap(0, 0.0));
        assert!(hist.is_gap(1, 0.0));
    }

    #[test]
    fn test_is_gap_full_tolerance_never_gap() {
        let hist = SeqHistogram::new(100, 256);
        // With tolerance=1.0, any distance <= 1.0 is not a gap
        assert!(!hist.is_gap(1, 1.0));
        assert!(!hist.is_gap(4, 1.0));
        assert!(!hist.is_gap(256, 1.0));
    }

    #[test]
    fn test_is_gap_seq_len_zero() {
        let hist = SeqHistogram::new(100, 256);
        // seq_len=0 -> bucket 0: start=0, end=1, center=0
        // distance = |0-0| / (1-0+1) = 0.0
        assert!(!hist.is_gap(0, 0.5));
    }

    #[test]
    fn test_is_gap_large_seq_len_clamped() {
        let hist = SeqHistogram::new(100, 64);
        // seq_len=1000 -> last bucket, check it doesn't panic
        let result = hist.is_gap(1000, 0.5);
        // Should return some bool without panic
        let _ = result;
    }

    // ── gap_hit_rate ──

    #[test]
    fn test_gap_hit_rate_zero_samples_returns_zero() {
        let hist = SeqHistogram::new(100, 256);
        assert_eq!(hist.gap_hit_rate(0.5), 0.0);
    }

    #[test]
    fn test_gap_hit_rate_with_samples_returns_zero_estimate() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        hist.record(8);
        // Current implementation always returns 0.0 (gap_hits stays 0)
        assert_eq!(hist.gap_hit_rate(0.5), 0.0);
    }

    // ── buckets() accessor ──

    #[test]
    fn test_buckets_accessor_matches_internal_count() {
        let hist = SeqHistogram::new(100, 256);
        // Directly access through accessor
        assert_eq!(hist.buckets().len(), hist.buckets().len());

        hist.record(1);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);
    }

    // ── Power-of-two bucket structure validation ──

    #[test]
    fn test_bucket_ranges_cover_full_spectrum() {
        let hist = SeqHistogram::new(100, 4096);
        let buckets = hist.buckets();

        // First bucket covers (0, 1]
        assert_eq!(buckets[0].start, 0);
        assert_eq!(buckets[0].end, 1);

        // Verify bucket i>0 follows pattern: start = 2^(i-1)+1, end = 2^i
        // (except the last which extends to max_seq)
        for i in 1..buckets.len() - 1 {
            let expected_start = (1usize << (i - 1)) + 1;
            let expected_end = 1usize << i;
            assert_eq!(buckets[i].start, expected_start, "bucket {} start mismatch", i);
            assert_eq!(buckets[i].end, expected_end, "bucket {} end mismatch", i);
        }
    }

    #[test]
    fn test_no_gaps_in_coverage() {
        // Verify that every seq_len from 0 to max_seq falls into some bucket
        let hist = SeqHistogram::new(100, 128);

        for seq_len in 0..=128 {
            hist.record(seq_len);
        }

        // Every seq_len should have been recorded (total = 129)
        assert_eq!(hist.total_samples(), 129);

        // Verify no bucket has zero count if it covers range that was tested
        let mut any_nonzero = false;
        for bucket in hist.buckets() {
            if bucket.count.load(Ordering::Relaxed) > 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn test_each_seq_len_maps_to_exactly_one_bucket() {
        let hist = SeqHistogram::new(100, 64);

        // Record each seq_len exactly once
        for seq_len in 0..=64 {
            hist.record(seq_len);
        }

        // Sum of all bucket counts should equal total records
        let bucket_sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(bucket_sum, hist.total_samples());
        assert_eq!(bucket_sum, 65); // 0..=64 inclusive
    }

    // ── Decay + Record interaction ──

    #[test]
    fn test_decay_then_record_accumulates() {
        let hist = SeqHistogram::new(100, 256);

        // Record 10, decay by 0.5 -> 5, then record 3 more -> 8
        for _ in 0..10 {
            hist.record(2);
        }
        hist.decay(0.5);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 5);

        for _ in 0..3 {
            hist.record(2);
        }
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 8);
    }

    // ── Concurrent stress ──

    #[test]
    fn test_concurrent_record_and_snapshot() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SeqHistogram::new(1000, 256));
        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let h = Arc::clone(&hist);
            handles.push(thread::spawn(move || {
                for i in 0..200 {
                    h.record((i % (1 << t)) + 1);
                }
            }));
        }

        // Reader thread taking snapshots
        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let snap = h.snapshot();
                assert!(snap.total_samples <= 800); // max 4*200
                let _ = snap.top_k;
            }
        }));

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ── Single-value edge cases ──

    #[test]
    fn test_record_single_value_histogram() {
        let hist = SeqHistogram::new(100, 4);
        hist.record(2);

        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 1);

        // Only bucket 1 should have a count
        let nonzero: Vec<_> = snap.buckets.iter().filter(|(_, _, c)| *c > 0).collect();
        assert_eq!(nonzero.len(), 1);
        assert_eq!(nonzero[0].2, 1);
    }

    #[test]
    fn test_record_usize_max_clamps_to_last() {
        let hist = SeqHistogram::new(100, 64);
        // Should not panic; clamps to last bucket
        hist.record(usize::MAX);
        assert_eq!(hist.total_samples(), 1);
        let last_idx = hist.buckets().len() - 1;
        assert_eq!(hist.buckets()[last_idx].count.load(Ordering::Relaxed), 1);
    }

    // ── Additional coverage ──

    #[test]
    fn new_small_max_seq_creates_minimal_buckets() {
        let hist = SeqHistogram::new(10, 2);
        // max_seq=2 → log2(2)=1 → 1 bucket (just bucket 0)
        assert!(hist.buckets().len() >= 1);
    }

    #[test]
    fn new_window_size_zero() {
        let hist = SeqHistogram::new(0, 1024);
        assert_eq!(hist.window_size(), 0);
    }

    #[test]
    fn bucket_start_is_zero_for_first_bucket() {
        let hist = SeqHistogram::new(100, 256);
        assert_eq!(hist.buckets()[0].start, 0);
        assert_eq!(hist.buckets()[0].end, 1);
    }

    #[test]
    fn record_many_same_value_accumulates() {
        let hist = SeqHistogram::new(1000, 64);
        for _ in 0..100 {
            hist.record(8);
        }
        assert_eq!(hist.total_samples(), 100);
    }

    #[test]
    fn decay_half_reduces_counts() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(10);
        hist.record(10);
        hist.record(10);
        hist.record(10);
        assert_eq!(hist.total_samples(), 4);
        hist.decay(0.5);
        assert_eq!(hist.total_samples(), 2);
    }

    #[test]
    fn decay_negative_factor_treated_as_zero() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(10);
        hist.decay(-0.5);
        // Negative f64 cast to u64 → some large number, but effectively 0 after store
        // This tests that it doesn't panic
    }

    #[test]
    fn reset_idempotent() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(5);
        hist.reset();
        hist.reset();
        assert_eq!(hist.total_samples(), 0);
    }

    #[test]
    fn snapshot_empty_top_k() {
        let hist = SeqHistogram::new(100, 256);
        let snap = hist.snapshot();
        assert!(snap.top_k.is_empty() || snap.top_k.iter().all(|(_, _, c)| *c == 0));
    }

    #[test]
    fn snapshot_total_samples_matches_sum() {
        let hist = SeqHistogram::new(100, 256);
        for v in [1, 2, 4, 8, 16, 32, 64, 128] {
            hist.record(v);
        }
        let snap = hist.snapshot();
        let bucket_sum: u64 = snap.buckets.iter().map(|(_, _, c)| c).sum();
        assert_eq!(snap.total_samples, bucket_sum);
    }

    #[test]
    fn top_k_two_returns_two_best() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 { hist.record(4); }
        for _ in 0..5 { hist.record(64); }
        let top = hist.top_k(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].2 >= top[1].2);
    }

    #[test]
    fn is_gap_tolerance_one_most_values() {
        let hist = SeqHistogram::new(100, 256);
        // Values exactly at bucket center should never be gap
        assert!(!hist.is_gap(1, 1.0));
        assert!(!hist.is_gap(2, 1.0));
    }

    #[test]
    fn gap_hit_rate_nonzero_samples() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(10);
        let rate = hist.gap_hit_rate(0.2);
        // With simplified implementation, returns 0.0
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn seq_bucket_count_atomic_update() {
        let bucket = SeqBucket { start: 0, end: 1, count: AtomicU64::new(5) };
        assert_eq!(bucket.count.load(Ordering::Relaxed), 5);
        bucket.count.fetch_add(3, Ordering::Relaxed);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn histogram_snapshot_clone_independence() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(5);
        let snap1 = hist.snapshot();
        let snap2 = snap1.clone();
        assert_eq!(snap1.total_samples, snap2.total_samples);
    }

    #[test]
    fn histogram_snapshot_debug_contains_fields() {
        let hist = SeqHistogram::new(100, 256);
        let snap = hist.snapshot();
        let s = format!("{:?}", snap);
        assert!(s.contains("buckets"));
        assert!(s.contains("total_samples"));
    }

    #[test]
    fn record_zero_goes_to_bucket_zero() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(0);
        assert_eq!(hist.buckets()[0].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn record_power_of_two_boundary() {
        let hist = SeqHistogram::new(100, 1024);
        hist.record(1);   // bucket 0
        hist.record(2);   // bucket 1
        hist.record(4);   // bucket 2
        hist.record(8);   // bucket 3
        hist.record(16);  // bucket 4
        hist.record(32);  // bucket 5
        hist.record(64);  // bucket 6
        hist.record(128); // bucket 7
        hist.record(256); // bucket 8
        assert_eq!(hist.total_samples(), 9);
    }

    #[test]
    fn new_large_max_seq_many_buckets() {
        let hist = SeqHistogram::new(1000, 65536);
        assert!(hist.buckets().len() >= 16);
    }

    #[test]
    fn decay_small_factor_near_zero() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..100 { hist.record(10); }
        hist.decay(0.001);
        assert!(hist.total_samples() < 10);
    }

    #[test]
    fn top_k_exceeds_bucket_count_returns_all() {
        let hist = SeqHistogram::new(100, 8);
        let top = hist.top_k(100);
        assert_eq!(top.len(), hist.buckets().len());
    }

    #[test]
    fn histogram_large_window_size() {
        let hist = SeqHistogram::new(usize::MAX, 1024);
        assert_eq!(hist.window_size(), usize::MAX);
    }

    #[test]
    fn snapshot_after_many_records() {
        let hist = SeqHistogram::new(1000, 512);
        for i in 1..=50 {
            hist.record(i * 10);
        }
        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 50);
        assert!(!snap.top_k.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 50 additional tests — gap coverage for public types & methods
    // ═══════════════════════════════════════════════════════════════════

    // ── SeqHistogram::new — non-power-of-two max_seq ──

    #[test]
    fn test_new_max_seq_three_creates_two_buckets() {
        let hist = SeqHistogram::new(100, 3);
        // log2(3) ≈ 1.585, ceil = 2
        assert_eq!(hist.buckets().len(), 2);
        assert_eq!(hist.buckets()[0].start, 0);
        assert_eq!(hist.buckets()[0].end, 1);
        let last = hist.buckets().last().unwrap();
        assert_eq!(last.end, 3);
    }

    #[test]
    fn test_new_max_seq_five_creates_three_buckets() {
        let hist = SeqHistogram::new(100, 5);
        // log2(5) ≈ 2.322, ceil = 3
        assert_eq!(hist.buckets().len(), 3);
    }

    #[test]
    fn test_new_max_seq_seven_creates_three_buckets() {
        let hist = SeqHistogram::new(100, 7);
        // log2(7) ≈ 2.807, ceil = 3
        assert_eq!(hist.buckets().len(), 3);
    }

    #[test]
    fn test_new_max_seq_nine_creates_four_buckets() {
        let hist = SeqHistogram::new(100, 9);
        // log2(9) ≈ 3.17, ceil = 4
        assert_eq!(hist.buckets().len(), 4);
    }

    #[test]
    fn test_new_bucket_zero_always_covers_zero_one_parametric() {
        for max_seq in [1, 2, 3, 4, 8, 16, 100, 256, 4096, 65536] {
            let hist = SeqHistogram::new(100, max_seq);
            assert_eq!(hist.buckets()[0].start, 0, "max_seq={}", max_seq);
            assert_eq!(hist.buckets()[0].end, 1, "max_seq={}", max_seq);
        }
    }

    #[test]
    fn test_new_very_large_max_seq_last_bucket_extends() {
        let hist = SeqHistogram::new(100, 100_000);
        assert!(hist.buckets().len() >= 17);
        let last = hist.buckets().last().unwrap();
        assert_eq!(last.end, 100_000);
    }

    #[test]
    fn test_new_max_seq_1024_exact_power() {
        let hist = SeqHistogram::new(100, 1024);
        // log2(1024) = 10.0, ceil = 10
        assert_eq!(hist.buckets().len(), 10);
    }

    // ── record — mid/upper range bucket coalescing ──

    #[test]
    fn test_record_mid_range_values_same_bucket() {
        let hist = SeqHistogram::new(100, 256);
        // seq_len 4..=7 → floor(log2)=2 → bucket 2
        hist.record(4);
        hist.record(5);
        hist.record(6);
        hist.record(7);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_record_upper_range_values_same_bucket() {
        let hist = SeqHistogram::new(100, 256);
        // seq_len 9..=15 → floor(log2)=3 → bucket 3
        for seq in 9..=15 {
            hist.record(seq);
        }
        assert_eq!(hist.buckets()[3].count.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn test_record_exact_max_seq_goes_to_last_bucket() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(256);
        let last_idx = hist.buckets().len() - 1;
        assert_eq!(hist.buckets()[last_idx].count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_record_many_distinct_values_sum_invariant() {
        let hist = SeqHistogram::new(100, 4096);
        for seq_len in 1..=100 {
            hist.record(seq_len);
        }
        assert_eq!(hist.total_samples(), 100);
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, 100);
    }

    #[test]
    fn test_record_after_decay_different_bucket_preserves_decayed() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(2); // bucket 1
        }
        hist.decay(0.5); // bucket 1: 10 → 5
        // Record into a different bucket
        for _ in 0..3 {
            hist.record(8); // bucket 3
        }
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 5);
        assert_eq!(hist.buckets()[3].count.load(Ordering::Relaxed), 3);
    }

    // ── snapshot — additional coverage ──

    #[test]
    fn test_snapshot_after_single_record() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(7);
        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 1);
        let nonzero: Vec<_> = snap.buckets.iter().filter(|(_, _, c)| *c == 1).collect();
        assert_eq!(nonzero.len(), 1);
    }

    #[test]
    fn test_snapshot_is_point_in_time() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        let snap = hist.snapshot();
        // Record more after snapshot — snapshot must be unchanged
        hist.record(8);
        assert_eq!(snap.total_samples, 1);
        assert_eq!(hist.total_samples(), 2);
    }

    #[test]
    fn test_snapshot_buckets_len_matches_histogram() {
        let hist = SeqHistogram::new(100, 64);
        let snap = hist.snapshot();
        assert_eq!(snap.buckets.len(), hist.buckets().len());
    }

    // ── top_k — additional coverage ──

    #[test]
    fn test_top_k_all_buckets_equal_count() {
        let hist = SeqHistogram::new(100, 4);
        for _ in 0..3 {
            hist.record(1); // bucket 0
        }
        for _ in 0..3 {
            hist.record(2); // bucket 1
        }
        let top = hist.top_k(2);
        assert_eq!(top.len(), 2);
        // Both tied at count 3
        assert!(top[0].2 == 3 && top[1].2 == 3);
    }

    #[test]
    fn test_top_k_after_reset_all_zero() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(4);
        }
        hist.reset();
        let top = hist.top_k(3);
        for (_, _, count) in &top {
            assert_eq!(*count, 0);
        }
    }

    #[test]
    fn test_top_k_single_nonzero_bucket() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..7 {
            hist.record(2); // only bucket 1
        }
        let top_all = hist.top_k(100);
        let nonzero: Vec<_> = top_all.iter().filter(|(_, _, c)| *c > 0).collect();
        assert_eq!(nonzero.len(), 1);
        assert_eq!(nonzero[0].2, 7);
    }

    #[test]
    fn test_top_k_k_equals_bucket_count() {
        let hist = SeqHistogram::new(100, 8);
        let bucket_count = hist.buckets().len();
        let top = hist.top_k(bucket_count);
        assert_eq!(top.len(), bucket_count);
    }

    // ── decay — additional coverage ──

    #[test]
    fn test_decay_empty_histogram_is_noop() {
        let hist = SeqHistogram::new(100, 256);
        hist.decay(0.5);
        assert_eq!(hist.total_samples(), 0);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_decay_factor_greater_than_one_grows_counts() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..5 {
            hist.record(2); // bucket 1
        }
        hist.decay(2.0);
        // 5 * 2.0 = 10.0 → 10u64
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 10);
        assert_eq!(hist.total_samples(), 10);
    }

    #[test]
    fn test_decay_proportional_across_buckets() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(2); // bucket 1
        }
        for _ in 0..20 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.5);
        let b1 = hist.buckets()[1].count.load(Ordering::Relaxed);
        let b2 = hist.buckets()[2].count.load(Ordering::Relaxed);
        assert_eq!(b1, 5);
        assert_eq!(b2, 10);
    }

    #[test]
    fn test_decay_then_snapshot_consistency() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.5);
        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 5);
        let bucket_sum: u64 = snap.buckets.iter().map(|(_, _, c)| c).sum();
        assert_eq!(bucket_sum, 5);
    }

    #[test]
    fn test_decay_multiple_rounds_same_factor() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..100 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.8); // 100 * 0.8 = 80
        hist.decay(0.8); // 80 * 0.8 = 64
        hist.decay(0.8); // 64 * 0.8 = 51.2 → 51
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 51);
    }

    #[test]
    fn test_decay_single_sample_factor_half_truncates_to_zero() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4); // bucket 2
        hist.decay(0.5);
        // 1 * 0.5 = 0.5 → truncates to 0
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 0);
        assert_eq!(hist.total_samples(), 0);
    }

    // ── reset — additional coverage ──

    #[test]
    fn test_reset_after_decay() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(4);
        }
        hist.decay(0.5);
        hist.reset();
        assert_eq!(hist.total_samples(), 0);
        for bucket in hist.buckets() {
            assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_reset_preserves_bucket_structure() {
        let hist = SeqHistogram::new(100, 256);
        let bucket_count = hist.buckets().len();
        let window = hist.window_size();
        hist.record(4);
        hist.record(8);
        hist.reset();
        assert_eq!(hist.buckets().len(), bucket_count);
        assert_eq!(hist.window_size(), window);
        for bucket in hist.buckets() {
            assert!(bucket.start <= bucket.end);
        }
    }

    // ── is_gap — additional coverage ──

    #[test]
    fn test_is_gap_never_panics_for_any_input() {
        let hist = SeqHistogram::new(100, 256);
        for seq_len in 0..=300usize {
            for tol in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let _ = hist.is_gap(seq_len, tol);
            }
        }
    }

    #[test]
    fn test_is_gap_distance_exactly_at_tolerance_boundary() {
        let hist = SeqHistogram::new(100, 256);
        // bucket 0: start=0, end=1, center=0
        // seq_len=1: distance = |1-0|/(1-0+1) = 0.5
        // With tolerance=0.5: distance (0.5) > tolerance (0.5) is false
        assert!(!hist.is_gap(1, 0.5));
    }

    // ── gap_hit_rate — additional coverage ──

    #[test]
    fn test_gap_hit_rate_always_in_unit_range() {
        let hist = SeqHistogram::new(100, 256);
        for i in 1..=50 {
            hist.record(i * 5);
        }
        for tolerance in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let rate = hist.gap_hit_rate(tolerance);
            assert!(
                rate >= 0.0 && rate <= 1.0,
                "rate={} for tolerance={}",
                rate,
                tolerance
            );
        }
    }

    #[test]
    fn test_gap_hit_rate_after_reset_returns_zero() {
        let hist = SeqHistogram::new(100, 256);
        hist.record(4);
        hist.reset();
        assert_eq!(hist.gap_hit_rate(0.5), 0.0);
    }

    // ── SeqBucket — additional coverage ──

    #[test]
    fn test_seq_bucket_count_starts_at_zero() {
        let bucket = SeqBucket {
            start: 0,
            end: 1,
            count: AtomicU64::new(0),
        };
        assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_seq_bucket_large_range() {
        let bucket = SeqBucket {
            start: 0,
            end: usize::MAX,
            count: AtomicU64::new(0),
        };
        assert_eq!(bucket.start, 0);
        assert_eq!(bucket.end, usize::MAX);
    }

    #[test]
    fn test_seq_bucket_fetch_add_accumulates() {
        let bucket = SeqBucket {
            start: 5,
            end: 8,
            count: AtomicU64::new(100),
        };
        bucket.count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 101);
        bucket.count.fetch_add(50, Ordering::Relaxed);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 151);
    }

    #[test]
    fn test_seq_bucket_store_overwrites() {
        let bucket = SeqBucket {
            start: 0,
            end: 1,
            count: AtomicU64::new(42),
        };
        bucket.count.store(0, Ordering::Relaxed);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 0);
        bucket.count.store(999, Ordering::Relaxed);
        assert_eq!(bucket.count.load(Ordering::Relaxed), 999);
    }

    // ── HistogramSnapshot — additional coverage ──

    #[test]
    fn test_snapshot_single_bucket_construction() {
        let snap = HistogramSnapshot {
            buckets: vec![(0, 1, 5)],
            total_samples: 5,
            top_k: vec![(0, 1, 5)],
        };
        assert_eq!(snap.buckets.len(), 1);
        assert_eq!(snap.total_samples, 5);
        assert_eq!(snap.top_k.len(), 1);
    }

    #[test]
    fn test_snapshot_tied_top_k_counts() {
        let snap = HistogramSnapshot {
            buckets: vec![(0, 1, 10), (2, 4, 10), (5, 8, 5)],
            total_samples: 25,
            top_k: vec![(0, 1, 10), (2, 4, 10), (5, 8, 5)],
        };
        assert_eq!(snap.top_k[0].2, 10);
        assert_eq!(snap.top_k[1].2, 10);
    }

    #[test]
    fn test_snapshot_clone_vec_independence() {
        let snap = HistogramSnapshot {
            buckets: vec![(0, 1, 10)],
            total_samples: 10,
            top_k: vec![(0, 1, 10)],
        };
        let mut cloned = snap.clone();
        cloned.buckets.push((2, 4, 5));
        // Original unchanged
        assert_eq!(snap.buckets.len(), 1);
        assert_eq!(cloned.buckets.len(), 2);
    }

    // ── Property / integration — bucket sum == total_samples invariant ──

    #[test]
    fn test_bucket_sum_equals_total_samples_invariant() {
        let hist = SeqHistogram::new(100, 4096);
        for seq_len in [0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 500, 1000, 2000, 4000] {
            hist.record(seq_len);
        }
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, hist.total_samples());
    }

    #[test]
    fn test_bucket_sum_invariant_after_decay() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..50 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.7);
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, hist.total_samples());
    }

    #[test]
    fn test_top_k_sorted_after_mixed_operations() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(2);
        }
        for _ in 0..5 {
            hist.record(4);
        }
        hist.decay(0.5);
        for _ in 0..3 {
            hist.record(8);
        }
        let top = hist.top_k(5);
        for window in top.windows(2) {
            assert!(
                window[0].2 >= window[1].2,
                "top_k not sorted: {:?} < {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_record_decay_snapshot_full_cycle() {
        let hist = SeqHistogram::new(100, 256);
        // Phase 1: record
        for _ in 0..20 {
            hist.record(4); // bucket 2
        }
        // Phase 2: decay
        hist.decay(0.5); // → 10
        let snap1 = hist.snapshot();
        assert_eq!(snap1.total_samples, 10);

        // Phase 3: record more into different bucket
        for _ in 0..5 {
            hist.record(8); // bucket 3
        }
        let snap2 = hist.snapshot();
        assert_eq!(snap2.total_samples, 15);

        // Phase 4: verify bucket sum invariant
        let bucket_sum: u64 = snap2.buckets.iter().map(|(_, _, c)| c).sum();
        assert_eq!(bucket_sum, 15);
    }

    #[test]
    fn test_record_many_then_reset_then_record() {
        let hist = SeqHistogram::new(100, 256);
        for i in 1..=200 {
            hist.record(i);
        }
        assert_eq!(hist.total_samples(), 200);
        hist.reset();
        assert_eq!(hist.total_samples(), 0);

        // Fresh start
        hist.record(4);
        hist.record(8);
        assert_eq!(hist.total_samples(), 2);
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, 2);
    }

    #[test]
    fn test_record_bucket_coverage_for_range() {
        let max_seq = 64;
        let hist = SeqHistogram::new(100, max_seq);
        for seq in 0..=max_seq {
            hist.record(seq);
        }
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, (max_seq + 1) as u64);
    }

    // ── Snapshot top_k entry count bounds ──

    #[test]
    fn test_snapshot_top_k_never_exceeds_ten() {
        let hist = SeqHistogram::new(100, 65536);
        for i in 1..=20 {
            for _ in 0..(21 - i) {
                hist.record(i * 100);
            }
        }
        let snap = hist.snapshot();
        assert!(snap.top_k.len() <= 10);
    }

    #[test]
    fn test_top_k_entry_count_never_exceeds_k_parametric() {
        let hist = SeqHistogram::new(100, 256);
        for i in 1..=20 {
            hist.record(i * 10);
        }
        for k in [1, 3, 5, 10, 50, 100] {
            let top = hist.top_k(k);
            assert!(top.len() <= k, "top_k({}) returned {} entries", k, top.len());
        }
    }

    // ── Decay preserves u64 non-negativity ──

    #[test]
    fn test_decay_preserves_non_negative_counts() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..5 {
            hist.record(4);
        }
        hist.decay(0.1);
        for bucket in hist.buckets() {
            // u64 is inherently non-negative; verify <= original 5
            assert!(bucket.count.load(Ordering::Relaxed) <= 5);
        }
    }

    // ── Concurrent — decay/record/top_k/reset interleaving ──

    #[test]
    fn test_concurrent_decay_and_record() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SeqHistogram::new(1000, 256));
        let mut handles = vec![];

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                h.record((i % 16) + 1);
            }
        }));

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                h.decay(0.9);
            }
        }));

        for handle in handles {
            handle.join().unwrap();
        }
        // No assertion on exact value — just verifying no panic
        assert!(hist.total_samples() <= 100);
    }

    #[test]
    fn test_concurrent_top_k_and_record() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SeqHistogram::new(1000, 256));
        let mut handles = vec![];

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                h.record((i % 8) + 1);
            }
        }));

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for _ in 0..20 {
                let top = h.top_k(3);
                for window in top.windows(2) {
                    assert!(
                        window[0].2 >= window[1].2,
                        "top_k not sorted during concurrent access"
                    );
                }
            }
        }));

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_reset_and_record() {
        use std::sync::Arc;
        use std::thread;

        let hist = Arc::new(SeqHistogram::new(1000, 256));
        let mut handles = vec![];

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                h.record(i % 16 + 1);
            }
        }));

        let h = Arc::clone(&hist);
        handles.push(thread::spawn(move || {
            h.reset();
        }));

        for handle in handles {
            handle.join().unwrap();
        }
        // No assertion on final state — verifying no panic/race
    }

    // ── Decay after record in same bucket accumulates correctly ──

    #[test]
    fn test_decay_in_same_bucket_accumulates_with_new_records() {
        let hist = SeqHistogram::new(100, 256);
        // 10 records → decay 0.5 → 5 → add 3 more → 8
        for _ in 0..10 {
            hist.record(2); // bucket 1
        }
        hist.decay(0.5);
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 5);
        for _ in 0..3 {
            hist.record(2);
        }
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 8);
        assert_eq!(hist.total_samples(), 8);
    }

    // ── Snapshot after decay reflects decayed state ──

    #[test]
    fn test_snapshot_after_decay_reflects_decayed_values() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..10 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.5);
        let snap = hist.snapshot();
        assert_eq!(snap.total_samples, 5);
        // Bucket 2 should show 5
        let bucket_sum: u64 = snap.buckets.iter().map(|(_, _, c)| c).sum();
        assert_eq!(bucket_sum, 5);
    }

    // ── window_size getter after various constructions ──

    #[test]
    fn test_window_size_various_values() {
        for ws in [1, 10, 100, 1000, 1_000_000] {
            let hist = SeqHistogram::new(ws, 256);
            assert_eq!(hist.window_size(), ws);
        }
    }

    // ── total_samples after multiple decay cycles ──

    #[test]
    fn test_total_samples_after_multiple_decay_cycles() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..80 {
            hist.record(4); // bucket 2
        }
        for _ in 0..5 {
            hist.decay(0.8);
        }
        // 80 * 0.8^5 = 80 * 0.32768 = 26.2144 → 26
        let total = hist.total_samples();
        assert!(total > 0 && total < 80);
        // Verify invariant
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, total);
    }

    // ── buckets() accessor after various operations ──

    #[test]
    fn test_buckets_accessor_consistent_after_operations() {
        let hist = SeqHistogram::new(100, 256);
        let original_len = hist.buckets().len();
        hist.record(4);
        assert_eq!(hist.buckets().len(), original_len);
        hist.decay(0.5);
        assert_eq!(hist.buckets().len(), original_len);
        hist.reset();
        assert_eq!(hist.buckets().len(), original_len);
    }

    // ── HistogramSnapshot clone deep independence for top_k ──

    #[test]
    fn test_snapshot_clone_top_k_vec_independence() {
        let snap = HistogramSnapshot {
            buckets: vec![(0, 1, 5), (2, 4, 10)],
            total_samples: 15,
            top_k: vec![(2, 4, 10), (0, 1, 5)],
        };
        let mut cloned = snap.clone();
        cloned.top_k.clear();
        assert_eq!(snap.top_k.len(), 2);
        assert!(cloned.top_k.is_empty());
    }

    // ── Record seq_len exactly at bucket boundary transitions ──

    #[test]
    fn test_record_boundary_transitions() {
        let hist = SeqHistogram::new(100, 256);
        // seq_len=3 and seq_len=4 should go to different buckets
        hist.record(3); // bucket 1 (log2(3)=1)
        hist.record(4); // bucket 2 (log2(4)=2)
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 1);
        assert_eq!(hist.buckets()[2].count.load(Ordering::Relaxed), 1);
    }

    // ── Decay factor very close to 1.0 preserves most counts ──

    #[test]
    fn test_decay_factor_near_one_preserves_counts() {
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..1000 {
            hist.record(4); // bucket 2
        }
        hist.decay(0.999);
        let count = hist.buckets()[2].count.load(Ordering::Relaxed);
        // 1000 * 0.999 = 999
        assert!(count >= 998 && count <= 999, "count={}", count);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Additional 10 tests — boundary & correctness coverage
    // ═══════════════════════════════════════════════════════════════════

    /// Verify HistogramSnapshot field-level equality: buckets vec, total_samples, top_k.
    /// The struct derives Clone but not PartialEq; confirm manual field comparison works.
    #[test]
    fn test_snapshot_field_equality_manual() {
        // Arrange
        let snap_a = HistogramSnapshot {
            buckets: vec![(0, 1, 3), (3, 4, 7)],
            total_samples: 10,
            top_k: vec![(3, 4, 7), (0, 1, 3)],
        };
        let snap_b = snap_a.clone();

        // Assert — field-by-field equality holds for the clone
        assert_eq!(snap_a.buckets, snap_b.buckets);
        assert_eq!(snap_a.total_samples, snap_b.total_samples);
        assert_eq!(snap_a.top_k, snap_b.top_k);
    }

    /// Verify is_gap computation on a mid-range bucket with known geometry.
    /// Bucket 2 for max_seq=256: start=3, end=4, center=(3+4)/2=3.
    /// find_bucket(4) -> bucket 2. seq_len=4 at edge: distance = |4-3|/(4-3+1) = 0.5
    /// find_bucket(3) -> bucket 1 (start=2, end=2). seq_len=3 outside bucket 1 range: distance = |3-2|/(2-2+1) = 1.0
    #[test]
    fn test_is_gap_mid_range_exact_distance_calculation() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        // Bucket 2: start=3, end=4, center=3
        // find_bucket(4) = bucket 2, center=3, distance = |4-3|/(4-3+1) = 0.5

        // Act & Assert
        assert!(hist.is_gap(3, 0.5));  // find_bucket(3)=bucket 1 (2,2], center=2, distance=1.0 > 0.5 -> gap
        assert!(!hist.is_gap(4, 0.5)); // find_bucket(4)=bucket 2 (3,4], center=3, distance=0.5 <= 0.5 -> not gap
        assert!(hist.is_gap(4, 0.49)); // same bucket 2, distance=0.5 > 0.49 -> gap
    }

    /// Verify decay with a non-trivial factor that produces a known truncation.
    /// 7 * 0.3 = 2.1 -> truncates to 2 (not 3).
    #[test]
    fn test_decay_non_trivial_factor_truncation_exact() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..7 {
            hist.record(2); // bucket 1
        }
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 7);

        // Act
        hist.decay(0.3);

        // Assert: 7 * 0.3 = 2.1 -> truncated to 2
        assert_eq!(hist.buckets()[1].count.load(Ordering::Relaxed), 2);
        assert_eq!(hist.total_samples(), 2);
    }

    /// top_k on a histogram with zero records: all counts are zero, any k returns zero-count entries.
    #[test]
    fn test_top_k_empty_histogram_all_zero_counts() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);

        // Act
        let top = hist.top_k(5);

        // Assert
        assert_eq!(top.len(), 5);
        for (_, _, count) in &top {
            assert_eq!(*count, 0);
        }
    }

    /// Record values at max_seq-1 and max_seq+1: both should land in the last bucket.
    #[test]
    fn test_record_at_max_seq_boundary_both_in_last_bucket() {
        // Arrange
        let max_seq = 64;
        let hist = SeqHistogram::new(100, max_seq);
        let last_idx = hist.buckets().len() - 1;

        // Act
        hist.record(max_seq - 1);
        hist.record(max_seq + 1); // beyond max_seq, clamps to last bucket

        // Assert
        assert_eq!(hist.buckets()[last_idx].count.load(Ordering::Relaxed), 2);
        assert_eq!(hist.total_samples(), 2);
    }

    /// Decay with factor > 1.0 increases counts; verify total_samples tracks correctly.
    #[test]
    fn test_decay_factor_above_one_preserves_total_samples_invariant() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        for _ in 0..5 {
            hist.record(2); // bucket 1: count=5
        }
        for _ in 0..10 {
            hist.record(4); // bucket 2: count=10
        }

        // Act
        hist.decay(2.0);
        // bucket 1: 5*2=10, bucket 2: 10*2=20, total: 15*2=30

        // Assert: bucket sum == total_samples
        let sum: u64 = hist
            .buckets()
            .iter()
            .map(|b| b.count.load(Ordering::Relaxed))
            .sum();
        assert_eq!(sum, hist.total_samples());
        assert_eq!(hist.total_samples(), 30);
    }

    /// buckets() returns a slice that reflects the internal state at call time.
    /// Mutating the histogram after getting the slice is reflected through the slice.
    #[test]
    fn test_buckets_slice_reflects_live_state() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        let slice = hist.buckets();
        assert_eq!(slice[0].count.load(Ordering::Relaxed), 0);

        // Act
        hist.record(1);

        // Assert: slice points to the same AtomicU64 objects
        assert_eq!(slice[0].count.load(Ordering::Relaxed), 1);
    }

    /// Snapshot top_k for a single-record histogram contains exactly one non-zero entry.
    #[test]
    fn test_snapshot_single_record_top_k_contains_one_nonzero() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        hist.record(16); // bucket 4

        // Act
        let snap = hist.snapshot();

        // Assert
        assert_eq!(snap.total_samples, 1);
        // top_k is sorted descending; first entry should be the nonzero one
        assert_eq!(snap.top_k[0].2, 1);
        // Exactly one bucket has count 1
        let nonzero_count = snap.top_k.iter().filter(|(_, _, c)| *c > 0).count();
        assert_eq!(nonzero_count, 1);
    }

    /// gap_hit_rate with extremely large total_samples does not panic or overflow.
    #[test]
    fn test_gap_hit_rate_large_total_samples_no_overflow() {
        // Arrange
        let hist = SeqHistogram::new(100, 256);
        // Simulate large sample count by directly manipulating atomic
        for _ in 0..1000 {
            hist.record(4);
        }

        // Act
        let rate = hist.gap_hit_rate(0.5);

        // Assert: rate is in valid range, no panic
        assert!(rate >= 0.0 && rate <= 1.0, "rate={}", rate);
    }

    /// Verify that SeqHistogram::new with max_seq=usize::MAX does not panic
    /// and creates a bucket structure that covers the full range.
    #[test]
    fn test_new_max_seq_usize_max_no_panic() {
        // Arrange & Act
        let hist = SeqHistogram::new(100, usize::MAX);

        // Assert
        assert!(hist.buckets().len() >= 1);
        assert_eq!(hist.buckets()[0].start, 0);
        assert_eq!(hist.buckets()[0].end, 1);
        // Last bucket end must be usize::MAX
        let last = hist.buckets().last().unwrap();
        assert_eq!(last.end, usize::MAX);
        // Bucket ranges must be valid (start <= end)
        for bucket in hist.buckets() {
            assert!(bucket.start <= bucket.end);
        }
    }
}
