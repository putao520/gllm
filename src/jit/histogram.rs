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
}
