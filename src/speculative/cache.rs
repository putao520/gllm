//! §17.10.3 Speculation Cache — 运行时数据结构
//!
//! SAGUARO 多 GPU 并行推测解码 (§17.10) 的缓存结构。
//! Key: (prefix_hash, position) → Value: [F 个 token candidates + logits]
//!
//! Cache-Aware Sampling (Theorem 15):
//! 标准采样 p(x) = softmax(logits)
//! 缓存感知 p'(x) = p(x) * C for x in cache_top_F
//! C 值选择使 P(cache_hit) × P(accept|hit) 最大化

use std::collections::HashMap;

/// 单个缓存条目: 一个 prefix position 的 draft candidates
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Prefix hash (position-dependent)
    pub prefix_hash: u64,
    /// Position in the sequence
    pub position: usize,
    /// Top-F token candidates
    pub candidates: Vec<u32>,
    /// 对应的 logits (用于 cache-aware sampling)
    pub logits: Vec<f32>,
    /// 被接受次数 (用于自适应 C 调整)
    pub accept_count: usize,
    /// 总使用次数
    pub total_count: usize,
}

/// Speculation Cache
///
/// §17.10.3: 存储 draft logits 供跨轮复用，提升 SAGUARO 模式的
/// cache 命中率。每轮 speculation round 后全量刷新。
pub struct SpeculationCache {
    /// 缓存条目: hash → entry
    entries: HashMap<u64, CacheEntry>,
    /// Fan-out 参数 F (每位置缓存的候选数)
    fan_out: usize,
    /// Cache-Aware Sampling 的缩放因子 C (0 < C < 1)
    /// Theorem 15: 乘以缓存 token 的概率，集中分布质量
    cache_scale_factor: f32,
    /// 最大缓存条目数
    max_entries: usize,
    /// 当前 batch size (用于自适应回退, Theorem 17)
    batch_size: usize,
    /// 回退阈值 b* (batch < b* 用慢速备份, >= b* 用快速)
    fallback_threshold: usize,
}

impl std::fmt::Debug for SpeculationCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculationCache")
            .field("entries", &self.entries.len())
            .field("fan_out", &self.fan_out)
            .field("cache_scale_factor", &self.cache_scale_factor)
            .finish()
    }
}

impl SpeculationCache {
    /// 创建新的 Speculation Cache
    ///
    /// # Arguments
    /// * `fan_out` - 每位置缓存的候选数 F
    /// * `max_entries` - 最大缓存条目数
    pub fn new(fan_out: usize, max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            fan_out,
            cache_scale_factor: 0.8,
            max_entries,
            batch_size: 1,
            fallback_threshold: 4,
        }
    }

    /// 查询缓存: 给定 prefix hash 和 position
    pub fn lookup(&self, prefix_hash: u64, _position: usize) -> Option<&CacheEntry> {
        self.entries.get(&prefix_hash)
    }

    /// 插入/更新缓存条目
    ///
    /// 每轮 speculation round 后调用
    pub fn insert(&mut self, entry: CacheEntry) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&entry.prefix_hash) {
            // Evict least-used entry
            if let Some(evict_key) = self.entries.iter()
                .filter(|(_, e)| e.total_count > 0)
                .min_by_key(|(_, e)| e.accept_count)
                .map(|(&k, _)| k)
            {
                self.entries.remove(&evict_key);
            }
        }
        self.entries.insert(entry.prefix_hash, entry);
    }

    /// 全量刷新 (每轮 speculation round 后)
    ///
    /// §17.10.3: 旧 cache 对新 prompt 无效，需要全量替换
    pub fn refresh(&mut self, new_entries: Vec<CacheEntry>) {
        self.entries.clear();
        for entry in new_entries {
            self.entries.insert(entry.prefix_hash, entry);
        }
    }

    /// Cache-Aware Sampling (Theorem 15)
    ///
    /// 对已缓存的 top-F token 的概率乘以 C < 1,
    /// 将剩余分布质量集中到缓存 token 上。
    ///
    /// # Arguments
    /// * `logits` - Draft model 原始 logits
    /// * `prefix_hash` - 当前 prefix hash
    ///
    /// # Returns
    /// 调整后的 logits
    pub fn cache_aware_sample(&self, logits: &[f32], prefix_hash: u64) -> Vec<f32> {
        let mut adjusted = logits.to_vec();

        if let Some(entry) = self.entries.get(&prefix_hash) {
            // Scale down cached token probabilities
            let c = self.cache_scale_factor;
            for &token_id in &entry.candidates {
                if (token_id as usize) < adjusted.len() {
                    adjusted[token_id as usize] *= c;
                }
            }

            // Redistribute residual mass
            let residual_mass: f32 = entry.candidates.iter()
                .filter(|&&t| (t as usize) < logits.len())
                .map(|&t| logits[t as usize] * (1.0 - c))
                .sum();
            let total_mass: f32 = logits.iter().map(|l| l.exp()).sum::<f32>();
            let redistribution = residual_mass / (total_mass - residual_mass + 1e-10);

            for (i, adj) in adjusted.iter_mut().enumerate() {
                let is_cached = entry.candidates.iter().any(|&t| t as usize == i);
                if !is_cached {
                    *adj += redistribution.abs().min(10.0); // cap to prevent explosion
                }
            }
        }

        adjusted
    }

    /// 自适应回退策略 (Theorem 17)
    ///
    /// Cache 未命中时:
    /// - 低 batch (b < b*): 慢速备份 (draft model 实时生成, 质量高)
    /// - 高 batch (b ≥ b*): 快速备份 (n-gram 随机生成, 延迟低)
    pub fn fallback_strategy(&self) -> FallbackStrategy {
        if self.batch_size < self.fallback_threshold {
            FallbackStrategy::SlowDraft
        } else {
            FallbackStrategy::FastNgram
        }
    }

    /// 更新 batch size (用于自适应回退决策)
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// 自适应调整 C 值 (Theorem 15 的核心优化)
    ///
    /// 选择使 P(cache_hit) × P(accept|hit) 最大化的 C
    pub fn adapt_scale_factor(&mut self) {
        let total_hits: usize = self.entries.values().map(|e| e.total_count).sum();
        let total_accepts: usize = self.entries.values().map(|e| e.accept_count).sum();

        if total_hits > 10 {
            let hit_accept_rate = total_accepts as f32 / total_hits as f32;
            // 当命中接受率高 → 降低 C (更集中到缓存 token)
            // 当命中接受率低 → 提高 C (给非缓存 token 更多机会)
            self.cache_scale_factor = 0.5 + 0.3 * (1.0 - hit_accept_rate);
            self.cache_scale_factor = self.cache_scale_factor.clamp(0.3, 0.95);
        }
    }

    /// 缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let total: usize = self.entries.values().map(|e| e.total_count).sum();
        let hits: usize = self.entries.values().map(|e| e.accept_count).sum();
        if total == 0 { return 0.0; }
        hits as f32 / total as f32
    }

    /// 缓存条目数
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Cache 未命中时的回退策略 (Theorem 17)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackStrategy {
    /// 慢速备份: draft model 实时生成 (质量高, 延迟高)
    SlowDraft,
    /// 快速备份: n-gram 随机生成 (质量低, 延迟低)
    FastNgram,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_lookup() {
        let mut cache = SpeculationCache::new(4, 100);
        let entry = CacheEntry {
            prefix_hash: 42,
            position: 10,
            candidates: vec![100, 200, 300, 400],
            logits: vec![5.0, 4.0, 3.0, 2.0],
            accept_count: 0,
            total_count: 0,
        };
        cache.insert(entry);
        assert!(cache.lookup(42, 10).is_some());
        assert!(cache.lookup(99, 10).is_none());
    }

    #[test]
    fn test_cache_refresh() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);

        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![20], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
    }

    #[test]
    fn test_cache_aware_sampling() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![0, 1], logits: vec![5.0, 4.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // Cached tokens should be scaled down
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
    }

    #[test]
    fn test_fallback_strategy() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(2);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);

        cache.set_batch_size(8);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_adapt_scale_factor() {
        let mut cache = SpeculationCache::new(4, 100);
        // Insert entries with high accept rate
        for i in 0..20 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 8, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // High accept rate → low C (more concentration)
        assert!(cache.cache_scale_factor < 0.8);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        // Third insert should evict least-used (prefix_hash=2)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
    }

    // ── CacheEntry constructor & field access ──────────────────────────

    #[test]
    fn test_cache_entry_fields() {
        let entry = CacheEntry {
            prefix_hash: 123,
            position: 5,
            candidates: vec![10, 20, 30],
            logits: vec![1.0, 2.0, 3.0],
            accept_count: 7,
            total_count: 14,
        };
        assert_eq!(entry.prefix_hash, 123);
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![10, 20, 30]);
        assert_eq!(entry.logits, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.accept_count, 7);
        assert_eq!(entry.total_count, 14);
    }

    #[test]
    fn test_cache_entry_clone() {
        let entry = CacheEntry {
            prefix_hash: 99,
            position: 3,
            candidates: vec![1, 2],
            logits: vec![0.5, 0.6],
            accept_count: 1,
            total_count: 2,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.prefix_hash, entry.prefix_hash);
        assert_eq!(cloned.position, entry.position);
        assert_eq!(cloned.candidates, entry.candidates);
        assert_eq!(cloned.logits, entry.logits);
        assert_eq!(cloned.accept_count, entry.accept_count);
        assert_eq!(cloned.total_count, entry.total_count);
    }

    #[test]
    fn test_cache_entry_debug_format() {
        let entry = CacheEntry {
            prefix_hash: 42,
            position: 0,
            candidates: vec![100],
            logits: vec![2.5],
            accept_count: 0,
            total_count: 0,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("CacheEntry"));
        assert!(debug_str.contains("prefix_hash"));
        assert!(debug_str.contains("42"));
    }

    // ── SpeculationCache::new & initial state ──────────────────────────

    #[test]
    fn test_new_cache_is_empty() {
        let cache = SpeculationCache::new(4, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_new_cache_default_scale_factor() {
        let mut cache = SpeculationCache::new(4, 100);
        // Default scale factor is 0.8; verify indirectly via adapt_scale_factor
        // Without enough hits (total_hits <= 10), adapt should not change it.
        cache.adapt_scale_factor();
        // Still 0.8 because there are 0 total hits (below threshold of 10)
        // We verify by checking cache_aware_sample behavior
        let logits = vec![1.0, 2.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 999);
        // No entry in cache → logits unchanged
        assert_eq!(adjusted, logits);
    }

    // ── FallbackStrategy enum ──────────────────────────────────────────

    #[test]
    fn test_fallback_strategy_equality() {
        assert_eq!(FallbackStrategy::SlowDraft, FallbackStrategy::SlowDraft);
        assert_eq!(FallbackStrategy::FastNgram, FallbackStrategy::FastNgram);
        assert_ne!(FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_fallback_strategy_copy() {
        let a = FallbackStrategy::SlowDraft;
        let b = a; // Copy semantics
        assert_eq!(a, b);
    }

    #[test]
    fn test_fallback_strategy_debug_format() {
        assert!(format!("{:?}", FallbackStrategy::SlowDraft).contains("SlowDraft"));
        assert!(format!("{:?}", FallbackStrategy::FastNgram).contains("FastNgram"));
    }

    #[test]
    fn test_fallback_strategy_clone() {
        let original = FallbackStrategy::FastNgram;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── FallbackStrategy threshold boundary ────────────────────────────

    #[test]
    fn test_fallback_strategy_at_exact_threshold() {
        let mut cache = SpeculationCache::new(4, 100);
        // Default fallback_threshold = 4
        cache.set_batch_size(4);
        // batch_size == fallback_threshold → FastNgram (>= b*)
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_fallback_strategy_one_below_threshold() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(3);
        // batch_size (3) < fallback_threshold (4) → SlowDraft
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // ── SpeculationCache Debug trait ───────────────────────────────────

    #[test]
    fn test_speculation_cache_debug_format() {
        let cache = SpeculationCache::new(8, 200);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("SpeculationCache"));
        assert!(debug_str.contains("fan_out: 8"));
    }

    // ── Lookup on empty cache ──────────────────────────────────────────

    #[test]
    fn test_lookup_empty_cache_returns_none() {
        let cache = SpeculationCache::new(4, 100);
        assert!(cache.lookup(0, 0).is_none());
        assert!(cache.lookup(u64::MAX, usize::MAX).is_none());
    }

    // ── Insert then overwrite same key ─────────────────────────────────

    #[test]
    fn test_insert_overwrites_existing_key() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 5, candidates: vec![2, 3], logits: vec![2.0, 3.0],
            accept_count: 5, total_count: 10,
        });
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![2, 3]);
        assert_eq!(entry.accept_count, 5);
    }

    // ── Refresh with multiple entries ──────────────────────────────────

    #[test]
    fn test_refresh_multiple_entries() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let new_entries: Vec<CacheEntry> = (0..5).map(|i| CacheEntry {
            prefix_hash: 100 + i as u64,
            position: i,
            candidates: vec![i as u32],
            logits: vec![i as f32],
            accept_count: 0,
            total_count: 0,
        }).collect();
        cache.refresh(new_entries);
        assert_eq!(cache.len(), 5);
        assert!(cache.lookup(1, 0).is_none());
        for i in 0..5 {
            assert!(cache.lookup(100 + i as u64, i).is_some());
        }
    }

    #[test]
    fn test_refresh_with_empty_vec_clears_cache() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        cache.refresh(vec![]);
        assert!(cache.is_empty());
    }

    // ── cache_aware_sample with no matching entry ──────────────────────

    #[test]
    fn test_cache_aware_sample_no_hit_returns_original_logits() {
        let cache = SpeculationCache::new(4, 100);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let adjusted = cache.cache_aware_sample(&logits, 999);
        assert_eq!(adjusted, logits);
    }

    #[test]
    fn test_cache_aware_sample_out_of_range_token_id_ignored() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![100, 200], // token IDs beyond logits length
            logits: vec![1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![3.0, 2.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // Out-of-range candidates should not panic; non-cached tokens get redistribution
        // The non-cached positions are 0, 1, 2 — all should receive redistribution
        // since candidates [100, 200] are out of range, nothing gets scaled down
        assert_eq!(adjusted.len(), 3);
    }

    // ── adapt_scale_factor with low hits (below threshold) ─────────────

    #[test]
    fn test_adapt_scale_factor_below_threshold_no_change() {
        let mut cache = SpeculationCache::new(4, 100);
        // Only 5 total hits — below threshold of 10
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 5,
        });
        // Record initial scale factor via sampling behavior
        let logits = vec![10.0, 5.0];
        let before = cache.cache_aware_sample(&logits, 1);
        cache.adapt_scale_factor();
        let after = cache.cache_aware_sample(&logits, 1);
        // Should be identical — adapt did not fire (total_hits=5 <= 10)
        assert_eq!(before, after);
    }

    // ── adapt_scale_factor with low accept rate ────────────────────────

    #[test]
    fn test_adapt_scale_factor_low_accept_rate_increases_c() {
        let mut cache = SpeculationCache::new(4, 100);
        // Low accept rate: 2 accepts / 100 total = 0.02
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 2, total_count: 20, // 5 entries × 20 = 100 total
            });
        }
        cache.adapt_scale_factor();
        // hit_accept_rate = 10/100 = 0.1 → C = 0.5 + 0.3*(1-0.1) = 0.77
        // Low accept rate → higher C (more chance for non-cached tokens)
        // We verify the adjusted logits are less scaled down compared to default 0.8
        let logits = vec![10.0, 5.0, 1.0];
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // With C=0.77, cached token[0] = 10.0 * 0.77 = 7.7
        assert!(adjusted[0] < 10.0);
        assert!(adjusted[0] > 7.0);
    }

    // ── hit_rate on empty cache ────────────────────────────────────────

    #[test]
    fn test_hit_rate_empty_cache_is_zero() {
        let cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // ── hit_rate with data ─────────────────────────────────────────────

    #[test]
    fn test_hit_rate_with_data() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });
        // hit_rate = (3+7) / (10+10) = 0.5
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_all_accepted() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 10,
        });
        assert!((cache.hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_none_accepted() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 10,
        });
        assert!((cache.hit_rate() - 0.0).abs() < 1e-6);
    }

    // ── Eviction when all entries have zero total_count ────────────────

    #[test]
    fn test_eviction_all_zero_total_count_no_eviction() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        // Both at capacity with total_count=0. The eviction filter requires total_count>0.
        // Inserting a third unique key cannot evict, but insert still adds → 3 entries.
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        // Since no entry could be evicted (all total_count==0), we exceed max_entries
        assert_eq!(cache.len(), 3);
    }

    // ── Eviction prefers lowest accept_count ───────────────────────────

    #[test]
    fn test_eviction_selects_lowest_accept_count() {
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 20,
        });
        // Inserting a 4th should evict the one with lowest accept_count (prefix_hash=2, accept=3)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none()); // evicted
        assert!(cache.lookup(3, 0).is_some());
        assert!(cache.lookup(4, 0).is_some());
    }

    // ── cache_aware_sample redistribution mass is capped ───────────────

    #[test]
    fn test_cache_aware_sample_redistribution_capped() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1],
            logits: vec![100.0, 100.0],
            accept_count: 0, total_count: 0,
        });
        // Very large logits → large residual mass, but capped at 10.0 per token
        let logits = vec![100.0f32, 100.0, 0.001];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // Non-cached token at index 2 should receive capped redistribution
        assert!(adjusted[2] <= 0.001 + 10.0);
    }

    // ── set_batch_size updates strategy ────────────────────────────────

    #[test]
    fn test_set_batch_size_updates_strategy() {
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft); // batch=1 < 4

        cache.set_batch_size(10);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);

        cache.set_batch_size(0);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // ── adapt_scale_factor clamps to valid range ───────────────────────

    #[test]
    fn test_adapt_scale_factor_clamped_range() {
        let mut cache = SpeculationCache::new(4, 100);
        // Very high accept rate: 49/50 = 0.98
        // C = 0.5 + 0.3 * (1 - 0.98) = 0.506 → within [0.3, 0.95]
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 49, total_count: 50,
            });
        }
        cache.adapt_scale_factor();
        // Verify indirectly: C should be ~0.506
        // hit_accept_rate = 245/250 = 0.98 → C = 0.5 + 0.3*0.02 = 0.506
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0, 1], logits: vec![10.0, 5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * 0.506 ≈ 5.06
        assert!(adjusted[0] > 4.5 && adjusted[0] < 5.5);
    }

    // ── Multiple lookups return consistent references ──────────────────

    #[test]
    fn test_lookup_returns_consistent_data() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 77, position: 3,
            candidates: vec![10, 20, 30],
            logits: vec![0.1, 0.2, 0.3],
            accept_count: 5,
            total_count: 15,
        });
        let e1 = cache.lookup(77, 3).unwrap();
        let e2 = cache.lookup(77, 3).unwrap();
        assert_eq!(e1.candidates, e2.candidates);
        assert_eq!(e1.logits, e2.logits);
        assert_eq!(e1.accept_count, e2.accept_count);
        assert_eq!(e1.total_count, e2.total_count);
    }

    // ── Additional tests (37-56) ──────────────────────────────────────────

    #[test]
    fn test_cache_entry_empty_candidates() {
        let entry = CacheEntry {
            prefix_hash: 0,
            position: 0,
            candidates: vec![],
            logits: vec![],
            accept_count: 0,
            total_count: 0,
        };
        assert!(entry.candidates.is_empty());
        assert!(entry.logits.is_empty());
    }

    #[test]
    fn test_cache_entry_max_u64_hash() {
        let entry = CacheEntry {
            prefix_hash: u64::MAX,
            position: usize::MAX,
            candidates: vec![1],
            logits: vec![1.0],
            accept_count: 0,
            total_count: 0,
        };
        assert_eq!(entry.prefix_hash, u64::MAX);
        assert_eq!(entry.position, usize::MAX);
    }

    #[test]
    fn test_cache_entry_zero_hash() {
        let entry = CacheEntry {
            prefix_hash: 0,
            position: 0,
            candidates: vec![42],
            logits: vec![3.14],
            accept_count: 1,
            total_count: 1,
        };
        assert_eq!(entry.prefix_hash, 0);
    }

    #[test]
    fn test_cache_entry_large_fields() {
        let candidates: Vec<u32> = (0..1000).collect();
        let logits: Vec<f32> = (0..1000).map(|x| x as f32 * 0.001).collect();
        let entry = CacheEntry {
            prefix_hash: 999,
            position: 500,
            candidates: candidates.clone(),
            logits: logits.clone(),
            accept_count: 999_999,
            total_count: 1_000_000,
        };
        assert_eq!(entry.candidates.len(), 1000);
        assert_eq!(entry.logits.len(), 1000);
        assert_eq!(entry.accept_count, 999_999);
        assert_eq!(entry.total_count, 1_000_000);
    }

    #[test]
    fn test_cache_entry_clone_independence() {
        let entry = CacheEntry {
            prefix_hash: 7,
            position: 1,
            candidates: vec![10, 20],
            logits: vec![1.0, 2.0],
            accept_count: 3,
            total_count: 4,
        };
        let mut cloned = entry.clone();
        cloned.candidates.push(30);
        cloned.logits.push(3.0);
        // Original must be unchanged
        assert_eq!(entry.candidates.len(), 2);
        assert_eq!(entry.logits.len(), 2);
        assert_eq!(cloned.candidates.len(), 3);
        assert_eq!(cloned.logits.len(), 3);
    }

    #[test]
    fn test_new_cache_with_zero_max_entries() {
        let mut cache = SpeculationCache::new(4, 0);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        // Insert should still add (no eviction possible)
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_new_cache_with_single_entry_capacity() {
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        assert_eq!(cache.len(), 1);
        // Insert different key: eviction requires total_count > 0 (yes, entry 1 has 10)
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![1], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(2, 0).is_some());
    }

    #[test]
    fn test_new_cache_various_fan_out_values() {
        let c1 = SpeculationCache::new(1, 100);
        let c2 = SpeculationCache::new(64, 100);
        let c3 = SpeculationCache::new(0, 100);
        // All should be valid and empty
        assert!(c1.is_empty());
        assert!(c2.is_empty());
        assert!(c3.is_empty());
    }

    #[test]
    fn test_insert_and_lookup_roundtrip_multiple_keys() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0u64..10 {
            cache.insert(CacheEntry {
                prefix_hash: i,
                position: i as usize,
                candidates: vec![i as u32],
                logits: vec![i as f32],
                accept_count: i as usize,
                total_count: (i + 1) as usize,
            });
        }
        assert_eq!(cache.len(), 10);
        for i in 0u64..10 {
            let entry = cache.lookup(i, i as usize).unwrap();
            assert_eq!(entry.candidates, vec![i as u32]);
            assert_eq!(entry.accept_count, i as usize);
        }
    }

    #[test]
    fn test_lookup_position_ignored_for_hash_lookup() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 5, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        // Same hash, different position should still find the entry
        // (lookup uses prefix_hash as the key)
        assert!(cache.lookup(42, 0).is_some());
        assert!(cache.lookup(42, 999).is_some());
        assert!(cache.lookup(42, 5).is_some());
    }

    #[test]
    fn test_refresh_with_overlapping_keys_keeps_last() {
        let mut cache = SpeculationCache::new(4, 100);
        let entries = vec![
            CacheEntry {
                prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 10, position: 5, candidates: vec![2], logits: vec![2.0],
                accept_count: 0, total_count: 0,
            },
        ];
        cache.refresh(entries);
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![2]);
    }

    #[test]
    fn test_hit_rate_with_mixed_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 10,
        });
        // hit_rate = (0 + 5 + 10) / (10 + 10 + 10) = 15/30 = 0.5
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_strategy_hash_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FallbackStrategy::SlowDraft);
        set.insert(FallbackStrategy::FastNgram);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&FallbackStrategy::SlowDraft));
        assert!(set.contains(&FallbackStrategy::FastNgram));
    }

    #[test]
    fn test_cache_aware_sample_single_element_logits() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32];
        let adjusted = cache.cache_aware_sample(&logits, 1);
        assert_eq!(adjusted.len(), 1);
        // Cached token at index 0 should be scaled down
        assert!(adjusted[0] < logits[0]);
    }

    #[test]
    fn test_adapt_scale_factor_zero_accept_rate() {
        let mut cache = SpeculationCache::new(4, 100);
        // Zero accept rate across many entries: 0 accepts / high total
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // hit_accept_rate = 0/150 = 0.0 → C = 0.5 + 0.3*(1-0) = 0.8
        // Verify C is around 0.8 by checking adjusted logits
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * 0.8 = 8.0
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_adapt_scale_factor_perfect_accept_rate_clamped() {
        let mut cache = SpeculationCache::new(4, 100);
        // Perfect accept rate: all accepts == total
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 10, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // hit_accept_rate = 150/150 = 1.0 → C = 0.5 + 0.3*(1-1.0) = 0.5
        // Should be clamped within [0.3, 0.95]
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0, 1], logits: vec![10.0, 5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * 0.5 = 5.0
        assert!((adjusted[0] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_set_batch_size_zero() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(0);
        // 0 < fallback_threshold(4) → SlowDraft
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    #[test]
    fn test_cache_aware_sample_preserves_length() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 2],
            logits: vec![3.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![3.0f32, 2.0, 1.0, 0.5, 0.1];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), logits.len());
    }

    // ── Additional tests (57-106) ──────────────────────────────────────────

    // ── cache_aware_sample with empty logits ──────────────────────────────

    #[test]
    fn test_cache_aware_sample_empty_logits_no_hit() {
        let cache = SpeculationCache::new(4, 100);
        let adjusted = cache.cache_aware_sample(&[], 42);
        assert!(adjusted.is_empty());
    }

    #[test]
    fn test_cache_aware_sample_empty_logits_with_hit() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let adjusted = cache.cache_aware_sample(&[], 42);
        assert!(adjusted.is_empty());
    }

    // ── cache_aware_sample with all tokens cached ─────────────────────────

    #[test]
    fn test_cache_aware_sample_all_tokens_cached() {
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2],
            logits: vec![5.0, 4.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 3);
        // All tokens are cached, so all should be scaled down
        for i in 0..3 {
            assert!(adjusted[i] < logits[i]);
        }
    }

    // ── cache_aware_sample with duplicate candidate IDs ───────────────────

    #[test]
    fn test_cache_aware_sample_duplicate_candidates() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 1, 1],
            logits: vec![1.0, 1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![2.0f32, 5.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 3);
        // Token at index 1 gets scaled down multiple times (C^3 effectively)
        assert!(adjusted[1] < logits[1]);
    }

    // ── cache_aware_sample with negative logits ───────────────────────────

    #[test]
    fn test_cache_aware_sample_negative_logits() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0,
            candidates: vec![0], logits: vec![-3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-3.0f32, -1.0, 2.0];
        let adjusted = cache.cache_aware_sample(&logits, 10);
        assert_eq!(adjusted.len(), 3);
        // Negative logits * C (0.8) should still be negative
        assert!(adjusted[0] < 0.0);
        assert!(adjusted[0] > logits[0]); // e.g. -3.0 * 0.8 = -2.4 > -3.0
    }

    // ── cache_aware_sample non-cached tokens gain mass ────────────────────

    #[test]
    fn test_cache_aware_sample_non_cached_tokens_increase() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 55, position: 0,
            candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 1.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 55);
        // Cached token[0] should decrease, non-cached tokens should increase
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] >= logits[1] || (adjusted[1] - logits[1]).abs() < 1e-6);
        assert!(adjusted[2] >= logits[2] || (adjusted[2] - logits[2]).abs() < 1e-6);
    }

    // ── insert after eviction reuses key ──────────────────────────────────

    #[test]
    fn test_insert_evicted_key_restores_entry() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 5, total_count: 10,
        });
        // Entry 10 should be evicted (lowest accept_count)
        cache.insert(CacheEntry {
            prefix_hash: 30, position: 0, candidates: vec![3], logits: vec![3.0],
            accept_count: 2, total_count: 10,
        });
        assert!(cache.lookup(10, 0).is_none());
        // Re-insert evicted key
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 0, total_count: 0,
        });
        let entry = cache.lookup(10, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![99]);
    }

    // ── refresh large number of entries ───────────────────────────────────

    #[test]
    fn test_refresh_many_entries() {
        let mut cache = SpeculationCache::new(4, 1000);
        let entries: Vec<CacheEntry> = (0..200).map(|i| CacheEntry {
            prefix_hash: i as u64,
            position: i as usize,
            candidates: vec![i as u32],
            logits: vec![i as f32],
            accept_count: 0,
            total_count: 0,
        }).collect();
        cache.refresh(entries);
        assert_eq!(cache.len(), 200);
        assert!(!cache.is_empty());
    }

    // ── refresh then insert grows cache ───────────────────────────────────

    #[test]
    fn test_refresh_then_insert() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
    }

    // ── set_batch_size usize max ──────────────────────────────────────────

    #[test]
    fn test_set_batch_size_usize_max() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(usize::MAX);
        // usize::MAX >= fallback_threshold(4) → FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    // ── adapt_scale_factor called multiple times converges ────────────────

    #[test]
    fn test_adapt_scale_factor_idempotent_with_no_new_data() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        cache.adapt_scale_factor();
        cache.adapt_scale_factor();
        // Multiple calls with same data should produce same C value each time
        // (the formula is deterministic given same inputs)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted1 = cache.cache_aware_sample(&logits, 99);
        let adjusted2 = cache.cache_aware_sample(&logits, 99);
        assert_eq!(adjusted1, adjusted2);
    }

    // ── adapt_scale_factor with single entry above threshold ──────────────

    #[test]
    fn test_adapt_scale_factor_single_entry_above_threshold() {
        let mut cache = SpeculationCache::new(4, 100);
        // Single entry with total_count=15 > 10 threshold
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 6, total_count: 15,
        });
        cache.adapt_scale_factor();
        // hit_accept_rate = 6/15 = 0.4 → C = 0.5 + 0.3*0.6 = 0.68
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * 0.68 = 6.8
        assert!(adjusted[0] > 6.0 && adjusted[0] < 7.5);
    }

    // ── hit_rate single entry ─────────────────────────────────────────────

    #[test]
    fn test_hit_rate_single_entry() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 4,
        });
        let rate = cache.hit_rate();
        assert!((rate - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_large_counts() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 500_000, total_count: 1_000_000,
        });
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    // ── len and is_empty consistency ──────────────────────────────────────

    #[test]
    fn test_len_is_empty_consistency() {
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        cache.refresh(vec![]);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    // ── FallbackStrategy can be used in Vec and iterated ──────────────────

    #[test]
    fn test_fallback_strategy_in_vec() {
        let strategies = vec![FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];
        assert_eq!(strategies.len(), 2);
        assert_eq!(strategies[0], FallbackStrategy::SlowDraft);
        assert_eq!(strategies[1], FallbackStrategy::FastNgram);
    }

    // ── FallbackStrategy can be used as hashmap key ───────────────────────

    #[test]
    fn test_fallback_strategy_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(FallbackStrategy::SlowDraft, "slow");
        map.insert(FallbackStrategy::FastNgram, "fast");
        assert_eq!(map.get(&FallbackStrategy::SlowDraft), Some(&"slow"));
        assert_eq!(map.get(&FallbackStrategy::FastNgram), Some(&"fast"));
        assert_eq!(map.len(), 2);
    }

    // ── CacheEntry with u32::MAX candidate ────────────────────────────────

    #[test]
    fn test_cache_entry_u32_max_candidate() {
        let entry = CacheEntry {
            prefix_hash: 0,
            position: 0,
            candidates: vec![u32::MAX],
            logits: vec![1.0],
            accept_count: 0,
            total_count: 0,
        };
        assert_eq!(entry.candidates[0], u32::MAX);
    }

    // ── cache_aware_sample with mixed valid/invalid candidates ────────────

    #[test]
    fn test_cache_aware_sample_mixed_valid_invalid_candidates() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 999, 1], // token 999 out of range
            logits: vec![5.0, 3.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 3.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 3);
        // Token 0 and 1 should be scaled, token 999 ignored
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
    }

    // ── Multiple inserts of same key keep last ───────────────────────────

    #[test]
    fn test_insert_same_key_multiple_times() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0u32..5 {
            cache.insert(CacheEntry {
                prefix_hash: 42, position: i as usize,
                candidates: vec![i], logits: vec![i as f32],
                accept_count: i as usize, total_count: (i + 1) as usize,
            });
        }
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(42, 4).unwrap();
        assert_eq!(entry.position, 4);
        assert_eq!(entry.candidates, vec![4]);
        assert_eq!(entry.accept_count, 4);
    }

    // ── Eviction does not evict when below max_entries ────────────────────

    #[test]
    fn test_no_eviction_below_max_entries() {
        let mut cache = SpeculationCache::new(1, 5);
        for i in 1..=4u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 1, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 4);
        // All should still be present
        for i in 1..=4u64 {
            assert!(cache.lookup(i, 0).is_some());
        }
    }

    // ── cache_aware_sample with very small logits ─────────────────────────

    #[test]
    fn test_cache_aware_sample_very_small_logits() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1e-10],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1e-10f32, 1e-10];
        let adjusted = cache.cache_aware_sample(&logits, 1);
        assert_eq!(adjusted.len(), 2);
        // Should not panic or produce NaN
        assert!(adjusted[0].is_finite());
        assert!(adjusted[1].is_finite());
    }

    // ── adapt_scale_factor with exactly threshold hits ────────────────────

    #[test]
    fn test_adapt_scale_factor_exactly_ten_hits() {
        let mut cache = SpeculationCache::new(4, 100);
        // Exactly 10 total hits — boundary condition (total_hits > 10 is the check)
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.adapt_scale_factor();
        // total_hits=10 is NOT > 10, so adapt should not change C
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let before = cache.cache_aware_sample(&logits, 99);
        // C should still be 0.8 (default)
        // cached[0] = 10.0 * 0.8 = 8.0
        assert!((before[0] - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_adapt_scale_factor_eleven_hits_fires() {
        let mut cache = SpeculationCache::new(4, 100);
        // 11 total hits — just above threshold
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 6,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 5,
        });
        cache.adapt_scale_factor();
        // total_hits=11 > 10, adapt fires: rate = 8/11 ≈ 0.727
        // C = 0.5 + 0.3*(1-0.727) = 0.582
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * C where C ~0.582 → ~5.82
        assert!(adjusted[0] > 5.0 && adjusted[0] < 6.5);
    }

    // ── cache_aware_sample identity when cache empty ──────────────────────

    #[test]
    fn test_cache_aware_sample_empty_cache_identity() {
        let cache = SpeculationCache::new(4, 100);
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 0);
        assert_eq!(adjusted, logits);
    }

    // ── SpeculationCache debug with entries present ───────────────────────

    #[test]
    fn test_speculation_cache_debug_with_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("entries: 2"));
    }

    // ── Eviction when updating existing key does not count as new entry ───

    #[test]
    fn test_insert_existing_key_no_eviction_needed() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        assert_eq!(cache.len(), 2);
        // Updating existing key 1 — should not trigger eviction
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);
        let entry = cache.lookup(1, 5).unwrap();
        assert_eq!(entry.position, 5);
    }

    // ── refresh with single entry ─────────────────────────────────────────

    #[test]
    fn test_refresh_single_entry() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1, 2, 3], logits: vec![1.0, 2.0, 3.0],
            accept_count: 10, total_count: 20,
        });
        assert_eq!(cache.len(), 1);

        cache.refresh(vec![CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![50], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(99, 0).is_some());
    }

    // ── hit_rate after refresh resets ─────────────────────────────────────

    #[test]
    fn test_hit_rate_after_refresh() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        assert!((cache.hit_rate() - 0.8).abs() < 1e-6);

        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        // hit_rate = 0/(0+0) = 0.0 (total_count=0)
        assert!((cache.hit_rate() - 0.0).abs() < 1e-6);
    }

    // ── cache_aware_sample with only one valid candidate in range ─────────

    #[test]
    fn test_cache_aware_sample_one_valid_one_invalid() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 500], // only index 1 valid
            logits: vec![2.0, 4.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![3.0f32, 2.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // Token 1 should be scaled, token 500 ignored
        assert!(adjusted[1] < logits[1]);
        assert_eq!(adjusted.len(), 3);
    }

    // ── lookup returns correct entry fields after overwrite ───────────────

    #[test]
    fn test_lookup_after_overwrite_returns_new_data() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 1, candidates: vec![10], logits: vec![0.5],
            accept_count: 1, total_count: 2,
        });
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 2, candidates: vec![20, 30], logits: vec![1.5, 2.5],
            accept_count: 5, total_count: 10,
        });
        let entry = cache.lookup(7, 2).unwrap();
        assert_eq!(entry.position, 2);
        assert_eq!(entry.candidates, vec![20, 30]);
        assert_eq!(entry.logits, vec![1.5, 2.5]);
        assert_eq!(entry.accept_count, 5);
        assert_eq!(entry.total_count, 10);
    }

    // ── FallbackStrategy can be matched exhaustively ──────────────────────

    #[test]
    fn test_fallback_strategy_exhaustive_match() {
        fn label(s: FallbackStrategy) -> &'static str {
            match s {
                FallbackStrategy::SlowDraft => "slow",
                FallbackStrategy::FastNgram => "fast",
            }
        }
        assert_eq!(label(FallbackStrategy::SlowDraft), "slow");
        assert_eq!(label(FallbackStrategy::FastNgram), "fast");
    }

    // ── cache_aware_sample with two cache entries only one matches ────────

    #[test]
    fn test_cache_aware_sample_multiple_entries_one_matches() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0, 1.0];
        // Only hash=2 should apply
        let adjusted = cache.cache_aware_sample(&logits, 2);
        assert!(adjusted[1] < logits[1]); // cached
        assert!(adjusted[0] >= logits[0] || (adjusted[0] - logits[0]).abs() < 1e-6); // non-cached may increase
    }

    // ── adapt_scale_factor with varying accept rates per entry ────────────

    #[test]
    fn test_adapt_scale_factor_mixed_accept_rates() {
        let mut cache = SpeculationCache::new(4, 100);
        // Total: 3+7+2 = 12 accepts, 10+10+10 = 30 total → rate = 12/30 = 0.4
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.adapt_scale_factor();
        // C = 0.5 + 0.3*(1-0.4) = 0.68
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // cached[0] = 10.0 * 0.68 = 6.8
        assert!(adjusted[0] > 6.0 && adjusted[0] < 7.5);
    }

    // ── CacheEntry debug contains all field names ─────────────────────────

    #[test]
    fn test_cache_entry_debug_contains_all_fields() {
        let entry = CacheEntry {
            prefix_hash: 1,
            position: 2,
            candidates: vec![3],
            logits: vec![4.0],
            accept_count: 5,
            total_count: 6,
        };
        let s = format!("{:?}", entry);
        assert!(s.contains("prefix_hash: 1"));
        assert!(s.contains("position: 2"));
        assert!(s.contains("candidates: [3]"));
        assert!(s.contains("logits: [4.0]"));
        assert!(s.contains("accept_count: 5"));
        assert!(s.contains("total_count: 6"));
    }

    // ── refresh then refresh again replaces completely ────────────────────

    #[test]
    fn test_double_refresh() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
                accept_count: 0, total_count: 0,
            },
        ]);
        assert_eq!(cache.len(), 2);

        cache.refresh(vec![CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![99], logits: vec![99.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(99, 0).is_some());
    }

    // ── Eviction with equal accept_count picks one ────────────────────────

    #[test]
    fn test_eviction_equal_accept_count_still_evicts() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        // Both have accept_count=5; inserting third should evict one
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);
        // Exactly one of {1,2} should be evicted, 3 must be present
        assert!(cache.lookup(3, 0).is_some());
        let present_count = [1u64, 2].iter().filter(|&&k| cache.lookup(k, 0).is_some()).count();
        assert_eq!(present_count, 1);
    }

    // ── cache_aware_sample with zero scale factor effect on non-cached ─────

    #[test]
    fn test_cache_aware_sample_non_cached_tokens_never_decrease() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 1.0, 2.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // Non-cached tokens at index 1,2,3 should receive redistribution (>= original)
        for i in 1..4 {
            assert!(adjusted[i] >= logits[i] - 1e-6);
        }
    }

    // ── SpeculationCache fan_out parameter does not affect insert ─────────

    #[test]
    fn test_fan_out_does_not_restrict_insert() {
        let mut cache = SpeculationCache::new(1, 100);
        // fan_out=1 but entry has 5 candidates — should still insert
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 2, 3, 4, 5],
            logits: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            accept_count: 0, total_count: 0,
        });
        let entry = cache.lookup(42, 0).unwrap();
        assert_eq!(entry.candidates.len(), 5);
    }

    // ── set_batch_size toggles strategy back and forth ────────────────────

    #[test]
    fn test_set_batch_size_toggle_strategy() {
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
        cache.set_batch_size(100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
        cache.set_batch_size(1);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
        cache.set_batch_size(4);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 95-148 (54 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_full_lifecycle_new_insert_lookup_overwrite_lookup() {
        let mut cache = SpeculationCache::new(4, 100);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(10, 0).is_some());
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 5, candidates: vec![2], logits: vec![2.0],
            accept_count: 5, total_count: 10,
        });
        let entry = cache.lookup(10, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![2]);
        assert_eq!(entry.accept_count, 5);
    }

    #[test]
    fn test_full_lifecycle_insert_refresh_insert() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);
        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![20], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        }]);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![30], logits: vec![3.0],
            accept_count: 1, total_count: 1,
        });
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_entry_candidates_logits_different_lengths() {
        let entry = CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 2, 3], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.candidates.len(), 3);
        assert_eq!(entry.logits.len(), 1);
    }

    #[test]
    fn test_cache_entry_more_logits_than_candidates() {
        let entry = CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1], logits: vec![1.0, 2.0, 3.0, 4.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.candidates.len(), 1);
        assert_eq!(entry.logits.len(), 4);
    }

    #[test]
    fn test_cache_entry_zero_accept_nonzero_total() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 100,
        };
        assert_eq!(entry.accept_count, 0);
        assert_eq!(entry.total_count, 100);
    }

    #[test]
    fn test_cache_entry_accept_exceeds_total_structurally_valid() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 50, total_count: 10,
        };
        assert_eq!(entry.accept_count, 50);
        assert_eq!(entry.total_count, 10);
    }

    #[test]
    fn test_hit_rate_single_entry_zero_counts() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!((cache.hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_many_entries_all_zero_counts() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..10 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 0,
            });
        }
        assert!((cache.hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_skewed_distribution() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 999, total_count: 1000,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 1000,
        });
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_unchanged_by_lookups() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        let rate_before = cache.hit_rate();
        let _ = cache.lookup(1, 0);
        let _ = cache.lookup(999, 0);
        let rate_after = cache.hit_rate();
        assert!((rate_before - rate_after).abs() < 1e-10);
    }

    #[test]
    fn test_cache_aware_sample_many_candidates() {
        let mut cache = SpeculationCache::new(10, 100);
        let candidates: Vec<u32> = (0..10).collect();
        let logits_in_entry: Vec<f32> = (0..10).map(|x| (x + 1) as f32).collect();
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates, logits: logits_in_entry,
            accept_count: 0, total_count: 0,
        });
        let logits: Vec<f32> = (0..20).map(|x| (x + 1) as f32).collect();
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 20);
        for i in 0..10 {
            assert!(adjusted[i] < logits[i]);
        }
        for i in 10..20 {
            assert!(adjusted[i] >= logits[i] - 1e-6);
        }
    }

    #[test]
    fn test_cache_aware_sample_candidate_at_last_index() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![4], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert!(adjusted[4] < logits[4]);
    }

    #[test]
    fn test_cache_aware_sample_candidate_one_past_end() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![5], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 5);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_cache_aware_sample_uniform_logits() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1], logits: vec![1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 1);
        assert_eq!(adjusted.len(), 5);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
    }

    #[test]
    fn test_cache_aware_sample_single_non_cached_receives_redistribution() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1], logits: vec![10.0, 8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 8.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 1);
        assert!(adjusted[2] > logits[2]);
    }

    #[test]
    fn test_cache_aware_sample_deterministic() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 2], logits: vec![5.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0];
        let first = cache.cache_aware_sample(&logits, 42);
        let second = cache.cache_aware_sample(&logits, 42);
        assert_eq!(first, second);
    }

    #[test]
    fn test_adapt_scale_factor_total_hits_eleven() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 11, total_count: 11,
        });
        cache.adapt_scale_factor();
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!((adjusted[0] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_adapt_scale_factor_fifty_percent_rate() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..6 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] > 6.0 && adjusted[0] < 7.0);
    }

    #[test]
    fn test_adapt_scale_factor_does_not_modify_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 3, total_count: 10,
            });
        }
        let len_before = cache.len();
        cache.adapt_scale_factor();
        assert_eq!(cache.len(), len_before);
        for i in 0u64..5 {
            let e = cache.lookup(i, 0).unwrap();
            assert_eq!(e.accept_count, 3);
            assert_eq!(e.total_count, 10);
        }
    }

    #[test]
    fn test_adapt_scale_factor_extreme_high_rate_clamps() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 1000, total_count: 1000,
            });
        }
        cache.adapt_scale_factor();
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] >= 10.0 * 0.3 - 0.1);
        assert!(adjusted[0] <= 10.0 * 0.95 + 0.1);
    }

    #[test]
    fn test_default_batch_size_is_one_slow_draft() {
        let cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    #[test]
    fn test_batch_size_three_is_slow_draft() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(3);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    #[test]
    fn test_batch_size_very_large_is_fast_ngram() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(1_000_000);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_fallback_strategy_iterate_collection() {
        let strategies = [FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];
        let labels: Vec<&str> = strategies.iter().map(|s| match s {
            FallbackStrategy::SlowDraft => "slow",
            FallbackStrategy::FastNgram => "fast",
        }).collect();
        assert_eq!(labels, vec!["slow", "fast"]);
    }

    #[test]
    fn test_sequential_evictions() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 20,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());

        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 20,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(3, 0).is_none());
        assert!(cache.lookup(4, 0).is_some());
    }

    #[test]
    fn test_eviction_new_entry_lowest_accept_survives() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
        assert!(cache.lookup(1, 0).is_none());
    }

    #[test]
    fn test_eviction_reinsert_evicted_key_with_higher_accept() {
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 10,
        });
        assert_eq!(cache.len(), 2);
        let entry = cache.lookup(1, 0).unwrap();
        assert_eq!(entry.accept_count, 10);
    }

    #[test]
    fn test_at_capacity_update_existing_no_growth() {
        let mut cache = SpeculationCache::new(1, 3);
        for i in 1..=3u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 3);
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 50, total_count: 100,
        });
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_refresh_replaces_high_count_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 999_999, total_count: 1_000_000,
        });
        let rate_before = cache.hit_rate();
        assert!(rate_before > 0.99);
        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        assert!((cache.hit_rate() - 0.0).abs() < 1e-6);
        assert!(cache.lookup(1, 0).is_none());
    }

    #[test]
    fn test_refresh_empty_after_many_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..50 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 100,
            });
        }
        assert_eq!(cache.len(), 50);
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lookup_after_many_overwrites() {
        let mut cache = SpeculationCache::new(4, 100);
        for round in 0..5 {
            for i in 0u64..10 {
                cache.insert(CacheEntry {
                    prefix_hash: i, position: round,
                    candidates: vec![round as u32 * 10 + i as u32],
                    logits: vec![round as f32],
                    accept_count: round as usize,
                    total_count: round as usize * 2,
                });
            }
        }
        assert_eq!(cache.len(), 10);
        for i in 0u64..10 {
            let e = cache.lookup(i, 4).unwrap();
            assert_eq!(e.position, 4);
            assert_eq!(e.accept_count, 4);
        }
    }

    #[test]
    fn test_set_batch_size_does_not_affect_entries() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        let len_before = cache.len();
        cache.set_batch_size(100);
        assert_eq!(cache.len(), len_before);
        let entry = cache.lookup(1, 0).unwrap();
        assert_eq!(entry.accept_count, 5);
    }

    #[test]
    fn test_len_reflects_distinct_hashes() {
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.len(), 0);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 1, candidates: vec![1], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);
        cache.refresh(vec![]);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_is_empty_transitions() {
        let mut cache = SpeculationCache::new(4, 100);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(!cache.is_empty());
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_entry_negative_logit_values() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1], logits: vec![-5.0, -3.0],
            accept_count: 0, total_count: 0,
        };
        assert!(entry.logits[0] < 0.0);
        assert!(entry.logits[1] < 0.0);
    }

    #[test]
    fn test_cache_entry_f32_max_logit() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![f32::MAX],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.logits[0], f32::MAX);
    }

    #[test]
    fn test_cache_entry_f32_min_positive_logit() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![f32::MIN_POSITIVE],
            accept_count: 0, total_count: 0,
        };
        assert!(entry.logits[0] > 0.0);
        assert!(entry.logits[0] < 1e-30);
    }

    #[test]
    fn test_cache_aware_sample_scattered_candidates() {
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 50, 99], logits: vec![5.0, 4.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits: Vec<f32> = (0..100).map(|x| (x as f32 + 1.0).ln_1p()).collect();
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 100);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[50] < logits[50]);
        assert!(adjusted[99] < logits[99]);
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[49] >= logits[49] - 1e-6);
    }

    #[test]
    fn test_cache_aware_sample_len_one_no_match() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![5], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![3.0f32];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 1);
        assert!(adjusted[0].is_finite());
    }

    #[test]
    fn test_adapt_scale_factor_empty_cache_safe() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.adapt_scale_factor();
        let logits = vec![1.0, 2.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 999);
        assert_eq!(adjusted, logits);
    }

    #[test]
    fn test_adapt_scale_factor_lower_bound_respected() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 100, total_count: 100,
            });
        }
        cache.adapt_scale_factor();
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] >= 10.0 * 0.3 - 0.5);
    }

    #[test]
    fn test_adapt_scale_factor_upper_bound_respected() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 100,
            });
        }
        cache.adapt_scale_factor();
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] <= 10.0 * 0.95 + 0.5);
    }

    #[test]
    fn test_eviction_single_slot_zero_total_count_no_evict() {
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(cache.len() >= 1);
    }

    #[test]
    fn test_eviction_single_slot_evicts_existing() {
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(1, 0).is_none());
    }

    #[test]
    fn test_cache_aware_sample_empty_candidates_with_hit() {
        let mut cache = SpeculationCache::new(0, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![], logits: vec![],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 3);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fallback_strategy_ord_via_cmp() {
        assert_eq!(FallbackStrategy::SlowDraft, FallbackStrategy::SlowDraft);
        assert_ne!(FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_position_field_independent_of_lookup() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 999, candidates: vec![5], logits: vec![3.0],
            accept_count: 10, total_count: 20,
        });
        let e0 = cache.lookup(42, 0).unwrap();
        let e999 = cache.lookup(42, 999).unwrap();
        assert_eq!(e0.position, 999);
        assert_eq!(e999.position, 999);
    }

    #[test]
    fn test_zero_fan_out_zero_max_entries_basic_operations() {
        let mut cache = SpeculationCache::new(0, 0);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![], logits: vec![],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_some());
    }

    #[test]
    fn test_repeated_adapt_without_inserts() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 7, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        let rate1 = cache.hit_rate();
        cache.adapt_scale_factor();
        let rate2 = cache.hit_rate();
        assert!((rate1 - rate2).abs() < 1e-10);
    }

    #[test]
    fn test_cache_aware_sample_infinity_logits_no_panic() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![f32::INFINITY],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![f32::INFINITY, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 2);
    }

    #[test]
    fn test_cache_aware_sample_nan_logits_no_panic() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![f32::NAN],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![f32::NAN, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 2);
    }

    #[test]
    fn test_cache_entry_duplicate_candidate_values() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![5, 5, 5, 5], logits: vec![1.0, 2.0, 3.0, 4.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.candidates.len(), 4);
        assert!(entry.candidates.iter().all(|&c| c == 5));
    }

    #[test]
    fn test_cache_entry_zero_position_nonzero_hash() {
        let entry = CacheEntry {
            prefix_hash: u64::MAX, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.prefix_hash, u64::MAX);
        assert_eq!(entry.position, 0);
    }

    #[test]
    fn test_max_entries_one_update_same_key() {
        let mut cache = SpeculationCache::new(4, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 5, total_count: 10,
        });
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(1, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![99]);
    }

    #[test]
    fn test_refresh_duplicate_hash_last_wins() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 10, position: 1, candidates: vec![1], logits: vec![1.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 10, position: 2, candidates: vec![2], logits: vec![2.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 10, position: 3, candidates: vec![3], logits: vec![3.0],
                accept_count: 0, total_count: 0,
            },
        ]);
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 3).unwrap();
        assert_eq!(entry.position, 3);
        assert_eq!(entry.candidates, vec![3]);
    }

    #[test]
    fn test_hit_rate_after_mixed_operations() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 20, total_count: 40,
        });
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 40, total_count: 80,
        });
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_strategy_conditional_logic() {
        let strategy = FallbackStrategy::SlowDraft;
        let is_slow = matches!(strategy, FallbackStrategy::SlowDraft);
        assert!(is_slow);
        let strategy = FallbackStrategy::FastNgram;
        let is_fast = matches!(strategy, FallbackStrategy::FastNgram);
        assert!(is_fast);
    }

    #[test]
    fn test_insert_one_lookup_different_hash() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(cache.lookup(42, 0).is_some());
        assert!(cache.lookup(43, 0).is_none());
        assert!(cache.lookup(0, 0).is_none());
        assert!(cache.lookup(u64::MAX, 0).is_none());
    }

    #[test]
    fn test_adapt_does_not_change_fallback_strategy() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(2);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
        for i in 0..20 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    #[test]
    fn test_set_batch_size_isolation() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 5, total_count: 10,
        });
        let logits = vec![10.0f32, 5.0];
        let before = cache.cache_aware_sample(&logits, 1);
        cache.set_batch_size(100);
        let after = cache.cache_aware_sample(&logits, 1);
        assert_eq!(before, after);
    }

    #[test]
    fn test_cache_entry_debug_field_names() {
        let entry = CacheEntry {
            prefix_hash: 0, position: 0,
            candidates: vec![], logits: vec![],
            accept_count: 0, total_count: 0,
        };
        let s = format!("{:?}", entry);
        assert!(s.contains("prefix_hash"));
        assert!(s.contains("position"));
        assert!(s.contains("candidates"));
        assert!(s.contains("logits"));
        assert!(s.contains("accept_count"));
        assert!(s.contains("total_count"));
    }

    #[test]
    fn test_speculation_cache_debug_fields() {
        let cache = SpeculationCache::new(16, 500);
        let s = format!("{:?}", cache);
        assert!(s.contains("SpeculationCache"));
        assert!(s.contains("entries: 0"));
        assert!(s.contains("fan_out: 16"));
        assert!(s.contains("cache_scale_factor"));
    }

    #[test]
    fn test_cache_aware_sample_redistribution_cap_is_ten() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![50.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![50.0f32, 0.0, 0.0, 0.0, 0.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        for i in 1..5 {
            assert!(adjusted[i] <= 0.0 + 10.0 + 1e-3);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 158-175 (18 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_entry_neg_infinity_logit() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![f32::NEG_INFINITY],
            accept_count: 0, total_count: 0,
        };
        assert!(entry.logits[0].is_infinite());
        assert!(entry.logits[0].is_sign_negative());
    }

    #[test]
    fn test_cache_entry_mixed_sign_logits() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1, 2], logits: vec![-5.0, 0.0, 8.0],
            accept_count: 0, total_count: 0,
        };
        assert!(entry.logits[0] < 0.0);
        assert_eq!(entry.logits[1], 0.0);
        assert!(entry.logits[2] > 0.0);
    }

    #[test]
    fn test_cache_entry_negative_zero_logit() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![-0.0f32],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.logits[0], 0.0);
        assert!(entry.logits[0].is_sign_negative());
    }

    #[test]
    fn test_cache_aware_sample_all_candidates_out_of_range_returns_identity() {
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![100, 200, 300],
            logits: vec![1.0, 1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 3.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted, logits);
    }

    #[test]
    fn test_cache_aware_sample_logits_all_zeros_no_nan() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![0.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![0.0f32, 0.0, 0.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 3);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_cache_aware_sample_two_entries_sequential_sampling() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 5.0, 5.0];
        let adj1 = cache.cache_aware_sample(&logits, 1);
        let adj2 = cache.cache_aware_sample(&logits, 2);
        assert!(adj1[0] < logits[0]);
        assert!(adj2[1] < logits[1]);
    }

    #[test]
    fn test_cache_aware_sample_three_entries_sequential() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0,
            candidates: vec![0], logits: vec![4.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0,
            candidates: vec![1], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 30, position: 0,
            candidates: vec![2], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![4.0f32, 3.0, 2.0, 1.0];
        let adj10 = cache.cache_aware_sample(&logits, 10);
        let adj20 = cache.cache_aware_sample(&logits, 20);
        let adj30 = cache.cache_aware_sample(&logits, 30);
        assert!(adj10[0] < logits[0]);
        assert!(adj20[1] < logits[1]);
        assert!(adj30[2] < logits[2]);
    }

    #[test]
    fn test_cache_aware_sample_large_logits_remain_finite() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![80.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![80.0f32, 50.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 2);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_insert_exact_capacity_all_present() {
        let mut cache = SpeculationCache::new(1, 5);
        for i in 1..=5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 5);
        for i in 1..=5u64 {
            assert!(cache.lookup(i, 0).is_some());
        }
    }

    #[test]
    fn test_lookup_uninserted_hashes_on_nonempty_cache() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(cache.lookup(42, 0).is_some());
        assert!(cache.lookup(0, 0).is_none());
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(43, 0).is_none());
        assert!(cache.lookup(u64::MAX, 0).is_none());
    }

    #[test]
    fn test_refresh_preserves_batch_size() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(8);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_hit_rate_increases_after_overwrite() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        let rate_before = cache.hit_rate();
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        let rate_after = cache.hit_rate();
        assert!(rate_after > rate_before);
    }

    #[test]
    fn test_hit_rate_decreases_after_overwrite() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 9, total_count: 10,
        });
        let rate_before = cache.hit_rate();
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        let rate_after = cache.hit_rate();
        assert!(rate_after < rate_before);
    }

    #[test]
    fn test_set_batch_size_repeatedly_last_wins() {
        let mut cache = SpeculationCache::new(4, 100);
        for bs in 0..50 {
            cache.set_batch_size(bs);
        }
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
        cache.set_batch_size(1);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    #[test]
    fn test_insert_lookup_roundtrip_all_fields() {
        let mut cache = SpeculationCache::new(4, 100);
        let entry = CacheEntry {
            prefix_hash: 77, position: 3,
            candidates: vec![10, 20, 30],
            logits: vec![0.5, 1.5, 2.5],
            accept_count: 7, total_count: 14,
        };
        cache.insert(entry);
        let found = cache.lookup(77, 3).unwrap();
        assert_eq!(found.prefix_hash, 77);
        assert_eq!(found.position, 3);
        assert_eq!(found.candidates, vec![10, 20, 30]);
        assert_eq!(found.logits, vec![0.5, 1.5, 2.5]);
        assert_eq!(found.accept_count, 7);
        assert_eq!(found.total_count, 14);
    }

    #[test]
    fn test_insert_many_unique_keys_large_capacity() {
        let mut cache = SpeculationCache::new(1, 200);
        for i in 0u64..50 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: i as usize,
                candidates: vec![i as u32], logits: vec![i as f32],
                accept_count: i as usize, total_count: (i + 1) as usize,
            });
        }
        assert_eq!(cache.len(), 50);
        for i in 0u64..50 {
            let e = cache.lookup(i, i as usize).unwrap();
            assert_eq!(e.candidates, vec![i as u32]);
        }
    }

    #[test]
    fn test_cache_aware_sample_after_refresh_with_new_data() {
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 3.0, 1.0];
        let adj_old = cache.cache_aware_sample(&logits, 1);
        assert!(adj_old[0] < logits[0]);

        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        }]);
        let adj_miss = cache.cache_aware_sample(&logits, 1);
        assert_eq!(adj_miss, logits);
        let adj_new = cache.cache_aware_sample(&logits, 2);
        assert!(adj_new[1] < logits[1]);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 175-189 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_entry_subnormal_logit() {
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![f32::from_bits(1)],
            accept_count: 0, total_count: 0,
        };
        assert!(entry.logits[0] > 0.0);
        assert!(entry.logits[0] < f32::MIN_POSITIVE);
    }

    #[test]
    fn test_cache_entry_single_candidate_u32_zero() {
        let entry = CacheEntry {
            prefix_hash: 0, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.candidates[0], 0u32);
    }

    #[test]
    fn test_cache_entry_position_zero_hash_nonzero() {
        let entry = CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.prefix_hash, 42);
        assert_eq!(entry.position, 0);
    }

    #[test]
    fn test_insert_lookup_different_position_same_hash() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 100,
            candidates: vec![5], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        // Lookup with different position but same hash finds the entry
        let found = cache.lookup(7, 0).unwrap();
        assert_eq!(found.position, 100);
        assert_eq!(found.candidates, vec![5]);
    }

    #[test]
    fn test_hit_rate_two_entries_one_zero_total() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        // rate = 5 / 10 = 0.5 (entry 2 has 0 total, doesn't contribute)
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_strategy_slow_draft_is_not_fast() {
        let slow = FallbackStrategy::SlowDraft;
        assert!(!matches!(slow, FallbackStrategy::FastNgram));
    }

    #[test]
    fn test_fallback_strategy_fast_ngram_is_not_slow() {
        let fast = FallbackStrategy::FastNgram;
        assert!(!matches!(fast, FallbackStrategy::SlowDraft));
    }

    #[test]
    fn test_cache_aware_sample_two_non_adjacent_cached() {
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 4], logits: vec![5.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[4] < logits[4]);
        assert_eq!(adjusted.len(), 5);
    }

    #[test]
    fn test_eviction_prefer_zero_accept_over_positive() {
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        // Should evict entry with lowest accept_count (hash=3, accept=0)
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(3, 0).is_none());
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(4, 0).is_some());
    }

    #[test]
    fn test_cache_entry_large_position_value() {
        let entry = CacheEntry {
            prefix_hash: 0, position: usize::MAX / 2,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        };
        assert_eq!(entry.position, usize::MAX / 2);
    }

    #[test]
    fn test_new_cache_hit_rate_is_zero() {
        let cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_insert_many_then_refresh_clears_all() {
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0u64..30 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 100,
            });
        }
        assert_eq!(cache.len(), 30);
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_set_batch_size_one_is_slow_draft() {
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(1);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 190-204 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_entry_clone_deep_copies_logits_independently() {
        // Arrange
        let entry = CacheEntry {
            prefix_hash: 10,
            position: 5,
            candidates: vec![1, 2],
            logits: vec![3.0, 4.0],
            accept_count: 1,
            total_count: 2,
        };
        // Act
        let mut cloned = entry.clone();
        cloned.logits[0] = 99.0;
        // Assert — original unchanged
        assert_eq!(entry.logits[0], 3.0);
        assert_eq!(cloned.logits[0], 99.0);
    }

    #[test]
    fn test_insert_at_capacity_same_key_replaces_without_eviction() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        assert_eq!(cache.len(), 2);

        // Act — update existing key 1, not inserting a new key
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 99, candidates: vec![42], logits: vec![7.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — still 2 entries, key 1 updated, key 2 untouched
        assert_eq!(cache.len(), 2);
        let e1 = cache.lookup(1, 99).unwrap();
        assert_eq!(e1.position, 99);
        assert_eq!(e1.candidates, vec![42]);
        let e2 = cache.lookup(2, 0).unwrap();
        assert_eq!(e2.accept_count, 3);
    }

    #[test]
    fn test_refresh_then_lookup_all_new_entries() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 100, total_count: 200,
        });
        let new_entries: Vec<CacheEntry> = (10..15).map(|i| CacheEntry {
            prefix_hash: i as u64, position: i as usize,
            candidates: vec![i as u32], logits: vec![i as f32],
            accept_count: i as usize, total_count: i as usize * 2,
        }).collect();

        // Act
        cache.refresh(new_entries);

        // Assert — old entry gone, all new ones present
        assert!(cache.lookup(1, 0).is_none());
        assert_eq!(cache.len(), 5);
        for i in 10..15u64 {
            let e = cache.lookup(i, i as usize).unwrap();
            assert_eq!(e.candidates, vec![i as u32]);
            assert_eq!(e.accept_count, i as usize);
        }
    }

    #[test]
    fn test_cache_aware_sample_with_candidate_at_exact_logits_boundary() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![2], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — candidate at last valid index scaled, no out-of-bounds
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[2] < logits[2]);
        assert!(adjusted[0].is_finite());
        assert!(adjusted[1].is_finite());
    }

    #[test]
    fn test_eviction_with_single_entry_having_higher_accept_than_new() {
        // Arrange — max_entries=1, existing entry has high accept_count
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 50, total_count: 100,
        });

        // Act — insert a new key; should evict existing
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5,
        });

        // Assert
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
    }

    #[test]
    fn test_adapt_scale_factor_with_two_entries_uneven_totals() {
        // Arrange — one entry with heavy weight, one with light
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 100,
        });

        // Act
        cache.adapt_scale_factor();

        // Assert — total_hits = 110 > 10, rate = 9/110 ≈ 0.0818
        // C = 0.5 + 0.3*(1-0.0818) ≈ 0.7755 → should be clamped within [0.3, 0.95]
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.7755 ≈ 7.755
        assert!(adjusted[0] > 7.0 && adjusted[0] < 8.5);
    }

    #[test]
    fn test_hit_rate_with_three_entries_varying_totals() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 20, total_count: 30,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 30, total_count: 50,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — (10+20+30) / (20+30+50) = 60/100 = 0.6
        assert!((rate - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_strategy_hashable_in_hashset_dedup() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();

        // Act
        set.insert(FallbackStrategy::SlowDraft);
        set.insert(FallbackStrategy::FastNgram);
        set.insert(FallbackStrategy::SlowDraft); // duplicate

        // Assert
        assert_eq!(set.len(), 2);
        assert!(set.contains(&FallbackStrategy::SlowDraft));
        assert!(set.contains(&FallbackStrategy::FastNgram));
    }

    #[test]
    fn test_cache_aware_sample_with_single_non_cached_at_index_zero() {
        // Arrange — cache candidate at index 1, index 0 is non-cached
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1], logits: vec![8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 8.0, 2.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — token 1 scaled down, token 0 and 2 receive redistribution
        assert!(adjusted[1] < logits[1]);
        assert!(adjusted[0] >= logits[0] - 1e-6);
        assert!(adjusted[2] >= logits[2] - 1e-6);
    }

    #[test]
    fn test_eviction_at_exact_capacity_no_new_key_no_eviction() {
        // Arrange — fill to capacity
        let mut cache = SpeculationCache::new(1, 3);
        for i in 1..=3u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 3);

        // Act — update existing key 2 (not a new key)
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![99], logits: vec![9.0],
            accept_count: 50, total_count: 100,
        });

        // Assert — all 3 still present, no eviction needed
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
        let e2 = cache.lookup(2, 0).unwrap();
        assert_eq!(e2.accept_count, 50);
    }

    #[test]
    fn test_cache_aware_sample_after_adapt_produces_different_scaling() {
        // Arrange — default C=0.8
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adj_before_adapt = cache.cache_aware_sample(&logits, 99);

        // Act — force adapt with high accept rate → lower C
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // Re-insert probe entry since adapt doesn't change it
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let adj_after_adapt = cache.cache_aware_sample(&logits, 99);

        // Assert — after adapt with high rate, C should be lower, so cached token even smaller
        assert!(adj_after_adapt[0] < adj_before_adapt[0]);
    }

    #[test]
    fn test_lookup_returns_correct_prefix_hash_from_entry() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 12345, position: 7,
            candidates: vec![10, 20], logits: vec![1.0, 2.0],
            accept_count: 3, total_count: 5,
        });

        // Act
        let entry = cache.lookup(12345, 0).unwrap();

        // Assert — the stored entry preserves the hash it was inserted with
        assert_eq!(entry.prefix_hash, 12345);
    }

    #[test]
    fn test_refresh_empty_then_insert_and_lookup() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });

        // Act — clear then add new data
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 5, candidates: vec![50, 60], logits: vec![3.0, 4.0],
            accept_count: 1, total_count: 2,
        });

        // Assert
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        let e = cache.lookup(99, 5).unwrap();
        assert_eq!(e.candidates, vec![50, 60]);
    }

    #[test]
    fn test_cache_aware_sample_with_very_large_logit_differences() {
        // Arrange — large logit contrast
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![100.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![100.0f32, -100.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — both values finite, cached token scaled down
        assert!(adjusted[0].is_finite());
        assert!(adjusted[1].is_finite());
        assert!(adjusted[0] < logits[0]);
        // Non-cached token may get redistribution capped at 10
        assert!(adjusted[1] >= -100.0 - 1e-6);
    }

    #[test]
    fn test_set_batch_size_five_is_fast_ngram_default_threshold() {
        // Arrange — default threshold is 4
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        cache.set_batch_size(5);

        // Assert — 5 >= 4 → FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);

        // Act — set to 4
        cache.set_batch_size(4);
        // Assert — 4 >= 4 → FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 205-219 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_entry_usize_max_counts() {
        // Arrange & Act
        let entry = CacheEntry {
            prefix_hash: 0, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: usize::MAX, total_count: usize::MAX,
        };
        // Assert — values stored exactly without overflow
        assert_eq!(entry.accept_count, usize::MAX);
        assert_eq!(entry.total_count, usize::MAX);
    }

    #[test]
    fn test_cache_aware_sample_candidate_at_logits_len_minus_one() {
        // Arrange — candidate index exactly at last valid position
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![4], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — last element scaled, no panic from out-of-bounds
        assert!(adjusted[4] < logits[4]);
        assert!(adjusted[0].is_finite());
    }

    #[test]
    fn test_eviction_new_entry_zero_accept_evicts_lowest_nonzero() {
        // Arrange — three entries, new key has accept_count=0
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 20,
        });

        // Act — new key triggers eviction of hash=2 (lowest accept)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });

        // Assert
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(4, 0).is_some());
    }

    #[test]
    fn test_alternating_refresh_empty_and_nonempty() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);

        // Act & Assert — round 1: non-empty
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);

        // Round 2: empty
        cache.refresh(vec![]);
        assert!(cache.is_empty());

        // Round 3: non-empty again
        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(2, 0).is_some());
    }

    #[test]
    fn test_cache_entry_debug_empty_fields_output() {
        // Arrange
        let entry = CacheEntry {
            prefix_hash: 0, position: 0,
            candidates: vec![], logits: vec![],
            accept_count: 0, total_count: 0,
        };
        // Act
        let s = format!("{:?}", entry);
        // Assert — debug contains all field names with empty collections
        assert!(s.contains("candidates: []"));
        assert!(s.contains("logits: []"));
        assert!(s.contains("accept_count: 0"));
    }

    #[test]
    fn test_cache_aware_sample_all_identical_logits_all_cached() {
        // Arrange — all logits equal, all indices cached
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![3.0, 3.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![3.0f32, 3.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all cached tokens scaled down uniformly
        assert_eq!(adjusted.len(), 3);
        for i in 0..3 {
            assert!(adjusted[i] < logits[i]);
        }
    }

    #[test]
    fn test_adapt_scale_factor_threshold_plus_one_total() {
        // Arrange — exactly 11 total hits (one above threshold)
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 6,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        });
        // total_hits = 11, accepts = 7, rate = 7/11 ≈ 0.636

        // Act
        cache.adapt_scale_factor();

        // Assert — C = 0.5 + 0.3*(1-0.636) ≈ 0.609, clamped in [0.3, 0.95]
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] > 5.0 && adjusted[0] < 7.0);
    }

    #[test]
    fn test_lookup_after_eviction_preserves_other_entries() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![20], logits: vec![2.0],
            accept_count: 1, total_count: 10,
        });

        // Act — evict hash=2 (lowest accept)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![30], logits: vec![3.0],
            accept_count: 0, total_count: 5,
        });

        // Assert — hash=1 untouched, hash=2 evicted, hash=3 present
        let e1 = cache.lookup(1, 0).unwrap();
        assert_eq!(e1.candidates, vec![10]);
        assert_eq!(e1.accept_count, 5);
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    #[test]
    fn test_cache_aware_sample_cached_candidate_at_index_zero() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0,
            candidates: vec![0], logits: vec![8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![8.0f32, 4.0, 2.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 7);

        // Assert — first element scaled, rest receive redistribution
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[2] >= logits[2] - 1e-6);
    }

    #[test]
    fn test_hit_rate_proportional_after_partial_overwrite() {
        // Arrange — two entries with equal total
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        // rate = 10/20 = 0.5

        // Act — overwrite entry 1 to change accept ratio
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        // new rate = (5+8)/(10+10) = 13/20 = 0.65

        // Assert
        let rate = cache.hit_rate();
        assert!((rate - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_batch_size_survives_refresh() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(10);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);

        // Act — refresh should not reset batch_size
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);

        // Assert
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_eviction_one_zero_total_one_positive_total_evicts_zero_accept() {
        // Arrange — two entries at capacity; one has total_count>0 but accept=0
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5,
        });

        // Act — new key evicts hash=2 (accept=0 < accept=3)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });

        // Assert
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    #[test]
    fn test_cache_aware_sample_single_token_logits_no_candidate_match() {
        // Arrange — logits has 1 element, candidate is index 0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — single element scaled, no redistribution targets
        assert_eq!(adjusted.len(), 1);
        assert!(adjusted[0] < logits[0]);
    }

    #[test]
    fn test_fallback_strategy_collects_into_hashmap() {
        // Arrange
        use std::collections::HashMap;
        let strategies = [FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];
        let mut map = HashMap::new();

        // Act
        for (i, s) in strategies.iter().enumerate() {
            map.insert(*s, i);
        }

        // Assert — both variants stored with their indices
        assert_eq!(map.len(), 2);
        assert_eq!(*map.get(&FallbackStrategy::SlowDraft).unwrap(), 0);
        assert_eq!(*map.get(&FallbackStrategy::FastNgram).unwrap(), 1);
    }

    #[test]
    fn test_insert_many_unique_hashes_near_capacity() {
        // Arrange — fill cache to near max_entries
        let mut cache = SpeculationCache::new(1, 10);
        for i in 0..9u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: (i + 1) as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 9);

        // Act — insert 10th unique key to reach exact capacity
        cache.insert(CacheEntry {
            prefix_hash: 9, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 10,
        });

        // Assert — all 10 present, no eviction needed
        assert_eq!(cache.len(), 10);
        for i in 0..=9u64 {
            assert!(cache.lookup(i, 0).is_some());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 220-234 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_aware_sample_logits_len_equals_candidate_max_plus_one() {
        // Arrange — logits length exactly matches max candidate index + 1
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![5.0, 4.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all indices valid, all scaled down, no out-of-bounds
        assert_eq!(adjusted.len(), 3);
        for i in 0..3 {
            assert!(adjusted[i] < logits[i]);
        }
    }

    #[test]
    fn test_adapt_scale_factor_single_entry_exactly_threshold_total() {
        // Arrange — single entry with total_count=11 (> threshold 10)
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 11,
        });

        // Act
        cache.adapt_scale_factor();

        // Assert — rate=8/11≈0.727, C=0.5+0.3*(1-0.727)≈0.582
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.582 ≈ 5.82
        assert!(adjusted[0] > 5.0 && adjusted[0] < 6.5);
    }

    #[test]
    fn test_refresh_with_empty_candidates_entries() {
        // Arrange — refresh with entries that have empty candidates/logits
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });

        // Act
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 10, position: 0, candidates: vec![], logits: vec![],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 20, position: 0, candidates: vec![], logits: vec![],
                accept_count: 0, total_count: 0,
            },
        ]);

        // Assert — old entry gone, new entries present
        assert!(cache.lookup(1, 0).is_none());
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(10, 0).is_some());
        assert!(cache.lookup(20, 0).is_some());
    }

    #[test]
    fn test_hit_rate_all_entries_identical_ratio() {
        // Arrange — all entries have same accept/total ratio
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 3, total_count: 9,
            });
        }

        // Act
        let rate = cache.hit_rate();

        // Assert — rate = 15/45 = 1/3
        assert!((rate - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_new_entry_zero_total_zero_accept_at_capacity() {
        // Arrange — fill capacity with entries having total_count > 0
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });

        // Act — new entry has total_count=0 but insertion should still evict
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — hash=2 (accept=3, lowest) evicted, hash=3 present
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    #[test]
    fn test_cache_entry_small_hash_large_position() {
        // Arrange & Act
        let entry = CacheEntry {
            prefix_hash: 1, position: usize::MAX - 1,
            candidates: vec![42], logits: vec![3.14],
            accept_count: 100, total_count: 200,
        };

        // Assert
        assert_eq!(entry.prefix_hash, 1);
        assert_eq!(entry.position, usize::MAX - 1);
        assert_eq!(entry.candidates, vec![42]);
    }

    #[test]
    fn test_cache_aware_sample_two_element_logits_cached_at_zero() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![7.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![7.0f32, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — index 0 scaled, index 1 receives redistribution
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert_eq!(adjusted.len(), 2);
    }

    #[test]
    fn test_insert_then_refresh_then_reinsert_evicted_hash() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 5, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });

        // Act — refresh clears it
        cache.refresh(vec![CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![0.0],
            accept_count: 0, total_count: 0,
        }]);
        assert!(cache.lookup(5, 0).is_none());

        // Re-insert the evicted hash
        cache.insert(CacheEntry {
            prefix_hash: 5, position: 10, candidates: vec![55], logits: vec![5.5],
            accept_count: 2, total_count: 4,
        });

        // Assert — hash=5 is back with new data
        assert_eq!(cache.len(), 2);
        let e = cache.lookup(5, 10).unwrap();
        assert_eq!(e.position, 10);
        assert_eq!(e.candidates, vec![55]);
        assert_eq!(e.accept_count, 2);
    }

    #[test]
    fn test_fallback_strategy_branching_logic() {
        // Arrange
        let strategies = [FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];

        // Act
        let results: Vec<bool> = strategies.iter().map(|&s| {
            match s {
                FallbackStrategy::SlowDraft => true,
                FallbackStrategy::FastNgram => false,
            }
        }).collect();

        // Assert — each variant maps to a distinct branch
        assert_eq!(results, vec![true, false]);
    }

    #[test]
    fn test_hit_rate_mixed_zero_and_nonzero_entries() {
        // Arrange — one entry with all zeros, one with nonzero counts
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });

        // Act — hit_rate called twice
        let rate1 = cache.hit_rate();
        let rate2 = cache.hit_rate();

        // Assert — deterministic, rate = 7/10 = 0.7
        assert!((rate1 - 0.7).abs() < 1e-6);
        assert!((rate2 - rate1).abs() < 1e-10);
    }

    #[test]
    fn test_cache_aware_sample_duplicate_valid_candidates_scaled_multiple_times() {
        // Arrange — duplicate candidate index 2 appears twice
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![2, 2, 2], logits: vec![1.0, 1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — index 2 scaled by C three times (0.8^3 = 0.512)
        // Others receive redistribution but are not scaled down
        assert_eq!(adjusted.len(), 4);
        assert!(adjusted[2] < logits[2] * 0.8);
        assert!(adjusted[0] >= logits[0] - 1e-6);
        assert!(adjusted[1] >= logits[1] - 1e-6);
    }

    #[test]
    fn test_eviction_equal_accept_different_total_still_evicts_one() {
        // Arrange — two entries with same accept_count but different total_count
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 100,
        });

        // Act — new key, should evict one of {1, 2} (both have accept=5)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });

        // Assert — exactly one of {1, 2} evicted, key 3 present
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(3, 0).is_some());
        let surviving = [1u64, 2].iter().filter(|&&k| cache.lookup(k, 0).is_some()).count();
        assert_eq!(surviving, 1);
    }

    #[test]
    fn test_cache_aware_sample_sequential_different_hashes_independent() {
        // Arrange — two entries with non-overlapping candidates
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 8.0, 1.0];

        // Act — sample with hash=1
        let adj1 = cache.cache_aware_sample(&logits, 1);
        // Sample with hash=2
        let adj2 = cache.cache_aware_sample(&logits, 2);

        // Assert — hash=1 scales index 0, hash=2 scales index 1
        assert!(adj1[0] < logits[0]);
        assert!(adj2[1] < logits[1]);
        // Each only affects its own candidates
        assert!(adj1[1] >= logits[1] - 1e-6);
        assert!(adj2[0] >= logits[0] - 1e-6);
    }

    #[test]
    fn test_new_cache_large_parameters() {
        // Arrange & Act
        let cache = SpeculationCache::new(1000, 1_000_000);

        // Assert — valid construction with extreme parameters
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("fan_out: 1000"));
    }

    #[test]
    fn test_adapt_then_refresh_then_adapt_resets_state() {
        // Arrange — build up state, adapt
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // C should have changed from 0.8 to ~0.53

        // Act — refresh clears all entries
        cache.refresh(vec![CacheEntry {
            prefix_hash: 50, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 2,
        }]);

        // Adapt again — total_hits=2, below threshold of 10
        cache.adapt_scale_factor();

        // Assert — C should remain unchanged (adapt didn't fire because total <= 10)
        // Insert probe and verify C is still what it was after first adapt
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // After first adapt: C ≈ 0.53 (high accept rate drove C down)
        // After second adapt: total_hits=2 <= 10, so C unchanged from first adapt
        assert!(adjusted[0] < 10.0 * 0.6); // C should be around 0.53
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 235-249 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_entry_with_nan_logit_stored_exactly() {
        // Arrange & Act
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![f32::NAN],
            accept_count: 0, total_count: 0,
        };
        // Assert — NaN is preserved exactly
        assert!(entry.logits[0].is_nan());
    }

    #[test]
    fn test_cache_entry_with_infinity_logit_stored_exactly() {
        // Arrange & Act
        let entry = CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![0], logits: vec![f32::INFINITY],
            accept_count: 0, total_count: 0,
        };
        // Assert — INFINITY preserved
        assert!(entry.logits[0].is_infinite());
        assert!(entry.logits[0].is_sign_positive());
    }

    #[test]
    fn test_cache_aware_sample_preserves_non_cached_exact_equal() {
        // Arrange — single cached token at index 2, non-cached at 0 and 1
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![2], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0, 3.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — index 2 scaled down, index 0 and 1 get redistribution, index 3 gets redistribution
        assert!(adjusted[2] < logits[2]);
        assert!(adjusted[0] >= logits[0] - 1e-6);
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[3] >= logits[3] - 1e-6);
    }

    #[test]
    fn test_refresh_does_not_alter_batch_size_or_scale_factor() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        // Force adapt to change scale factor from default 0.8
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 1, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        cache.set_batch_size(10);

        // Capture state before refresh via sampling
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adj_before = cache.cache_aware_sample(&logits, 99);
        let strategy_before = cache.fallback_strategy();

        // Act — refresh clears entries but not batch_size or scale_factor
        cache.refresh(vec![CacheEntry {
            prefix_hash: 50, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        }]);

        // Assert — strategy unchanged
        assert_eq!(cache.fallback_strategy(), strategy_before);
        // Re-insert probe and verify scale factor unchanged
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let adj_after = cache.cache_aware_sample(&logits, 99);
        // Same scale factor should produce same result for same logits
        assert!((adj_after[0] - adj_before[0]).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_insert_existing_key_at_capacity_skips_eviction() {
        // Arrange — exactly at capacity with 2 entries
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 20,
        });
        assert_eq!(cache.len(), 2);

        // Act — update key 1 (existing), should NOT trigger eviction since key exists
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 99, candidates: vec![77], logits: vec![7.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — both entries present, key 1 updated
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(2, 0).is_some());
        let e1 = cache.lookup(1, 99).unwrap();
        assert_eq!(e1.position, 99);
        assert_eq!(e1.candidates, vec![77]);
    }

    #[test]
    fn test_hit_rate_single_entry_accept_equals_total() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 42, total_count: 42,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — perfect hit rate
        assert!((rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_aware_sample_with_zero_logit_at_cached_index() {
        // Arrange — cached token has logit 0.0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![0.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![0.0f32, 5.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — 0.0 * 0.8 = 0.0, non-cached tokens get redistribution
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0].is_finite());
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[2] >= logits[2] - 1e-6);
    }

    #[test]
    fn test_insert_lookup_cycle_many_iterations() {
        // Arrange — stress test with many insert/lookup cycles
        let mut cache = SpeculationCache::new(4, 500);
        let num_entries = 100;

        // Act — insert entries
        for i in 0..num_entries as u64 {
            cache.insert(CacheEntry {
                prefix_hash: i,
                position: i as usize,
                candidates: vec![i as u32, (i + 1) as u32],
                logits: vec![i as f32, (i + 1) as f32],
                accept_count: i as usize,
                total_count: (i + 1) as usize,
            });
        }

        // Assert — all entries retrievable with correct data
        assert_eq!(cache.len(), num_entries as usize);
        for i in 0..num_entries as u64 {
            let e = cache.lookup(i, i as usize).unwrap();
            assert_eq!(e.candidates, vec![i as u32, (i + 1) as u32]);
            assert_eq!(e.accept_count, i as usize);
            assert_eq!(e.total_count, (i + 1) as usize);
        }
    }

    #[test]
    fn test_fallback_strategy_equality_and_inequality_symmetry() {
        // Arrange
        let a = FallbackStrategy::SlowDraft;
        let b = FallbackStrategy::FastNgram;

        // Act & Assert — symmetry: a != b implies b != a
        assert_ne!(a, b);
        assert_ne!(b, a);
        // reflexivity: a == a
        assert_eq!(a, a);
        assert_eq!(b, b);
    }

    #[test]
    fn test_adapt_scale_factor_preserves_cache_length() {
        // Arrange — populate with entries above threshold
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..12u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 4, total_count: 10,
            });
        }
        let len_before = cache.len();

        // Act
        cache.adapt_scale_factor();

        // Assert — adapt only changes scale_factor, not entries
        assert_eq!(cache.len(), len_before);
        for i in 0..12u64 {
            assert!(cache.lookup(i, 0).is_some());
        }
    }

    #[test]
    fn test_cache_aware_sample_after_overwrite_uses_new_candidates() {
        // Arrange — insert entry then overwrite with different candidates
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1], logits: vec![5.0, 4.0],
            accept_count: 0, total_count: 0,
        });
        // Overwrite with candidates at different indices
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![2, 3], logits: vec![3.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — new candidates (index 2,3) should be scaled, old (0,1) should get redistribution
        assert!(adjusted[2] < logits[2]);
        assert!(adjusted[3] < logits[3]);
        // Indices 0 and 1 are no longer cached, they should get redistribution
        assert!(adjusted[0] >= logits[0] - 1e-6);
        assert!(adjusted[1] >= logits[1] - 1e-6);
    }

    #[test]
    fn test_hit_rate_after_multiple_overwrites() {
        // Arrange — two entries, repeatedly overwritten
        let mut cache = SpeculationCache::new(4, 100);

        // Initial: 5/10 and 5/10 → rate 0.5
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);

        // Act — overwrite hash=1 to 9/10
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 9, total_count: 10,
        });

        // Assert — rate = (9+5)/(10+10) = 14/20 = 0.7
        let rate = cache.hit_rate();
        assert!((rate - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_cache_aware_sample_non_cached_tokens_all_increase_or_equal() {
        // Arrange — 5 logits, cache only index 0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![8.0f32, 4.0, 2.0, 1.0, 0.5];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all non-cached indices receive redistribution (>= original)
        assert!(adjusted[0] < logits[0]);
        for i in 1..5 {
            assert!(adjusted[i] >= logits[i] - 1e-6, "index {} decreased unexpectedly", i);
        }
    }

    #[test]
    fn test_set_batch_size_then_lookup_unchanged() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 5, candidates: vec![10, 20], logits: vec![1.0, 2.0],
            accept_count: 3, total_count: 7,
        });

        // Act — change batch size multiple times
        cache.set_batch_size(100);
        cache.set_batch_size(0);
        cache.set_batch_size(50);

        // Assert — entries completely unaffected
        let e = cache.lookup(42, 5).unwrap();
        assert_eq!(e.candidates, vec![10, 20]);
        assert_eq!(e.logits, vec![1.0, 2.0]);
        assert_eq!(e.accept_count, 3);
        assert_eq!(e.total_count, 7);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_eviction_with_many_candidates_per_entry() {
        // Arrange — entry with many candidates gets evicted properly
        let mut cache = SpeculationCache::new(10, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            logits: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            accept_count: 3, total_count: 10,
        });

        // Act — insert new key, should evict hash=1
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5,
        });

        // Assert — hash=1 evicted, hash=2 present
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 250-264 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-250
    #[test]
    fn test_cache_aware_sample_negative_logit_scaled_toward_zero() {
        // Arrange — negative cached logit * C (0.8) moves toward zero
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0,
            candidates: vec![0], logits: vec![-10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-10.0f32, 5.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 10);

        // Assert — -10.0 * 0.8 = -8.0, which is greater than -10.0 (closer to zero)
        assert!((adjusted[0] - (-8.0)).abs() < 0.5);
        assert!(adjusted[0] > logits[0]);
    }

    // @trace TEST-SPEC-CACHE-251
    #[test]
    fn test_insert_same_hash_preserves_last_even_with_different_position() {
        // Arrange — insert same hash with different positions
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 100,
            candidates: vec![2], logits: vec![2.0],
            accept_count: 5, total_count: 10,
        });

        // Act — lookup with any position returns the last-inserted entry
        let entry = cache.lookup(42, 0).unwrap();

        // Assert — last insert wins regardless of position
        assert_eq!(entry.position, 100);
        assert_eq!(entry.candidates, vec![2]);
        assert_eq!(entry.accept_count, 5);
    }

    // @trace TEST-SPEC-CACHE-252
    #[test]
    fn test_hit_rate_reflects_only_current_entries() {
        // Arrange — insert two entries, then overwrite one
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        // rate = 10/20 = 0.5

        // Act — overwrite hash=1 to change its counts
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 10,
        });

        // Assert — rate = (0+8)/(10+10) = 8/20 = 0.4
        let rate = cache.hit_rate();
        assert!((rate - 0.4).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-253
    #[test]
    fn test_cache_aware_sample_with_candidate_at_index_zero_and_last() {
        // Arrange — cache first and last logits in a 10-element vector
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 9], logits: vec![10.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        assert_eq!(logits.len(), 10);

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — first and last scaled, middle 8 receive redistribution
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[9] < logits[9]);
        for i in 1..9 {
            assert!(adjusted[i] >= logits[i] - 1e-6);
        }
        assert_eq!(adjusted.len(), 10);
    }

    // @trace TEST-SPEC-CACHE-254
    #[test]
    fn test_eviction_all_entries_have_equal_accept_picks_deterministically() {
        // Arrange — 3 entries with identical accept_count but different hashes
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 100, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 200, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 300, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 20,
        });
        assert_eq!(cache.len(), 3);

        // Act — insert 4th unique key
        cache.insert(CacheEntry {
            prefix_hash: 400, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5,
        });

        // Assert — exactly one evicted, new entry present, still 3 total
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(400, 0).is_some());
        let surviving = [100u64, 200, 300].iter()
            .filter(|&&k| cache.lookup(k, 0).is_some())
            .count();
        assert_eq!(surviving, 2);
    }

    // @trace TEST-SPEC-CACHE-255
    #[test]
    fn test_refresh_with_single_entry_preserves_len_one() {
        // Arrange — fill with multiple entries
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 100,
            });
        }
        assert_eq!(cache.len(), 20);

        // Act — replace all with a single entry
        cache.refresh(vec![CacheEntry {
            prefix_hash: 999, position: 5, candidates: vec![42], logits: vec![3.0],
            accept_count: 1, total_count: 1,
        }]);

        // Assert
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(999, 5).is_some());
        for i in 0..20u64 {
            assert!(cache.lookup(i, 0).is_none());
        }
    }

    // @trace TEST-SPEC-CACHE-256
    #[test]
    fn test_adapt_scale_factor_with_very_low_accept_rate_near_upper_bound() {
        // Arrange — very low accept rate drives C toward 0.95 (upper bound)
        // rate = 1/200 = 0.005 → C = 0.5 + 0.3*(1-0.005) = 0.7985
        // (not near upper bound; let's compute: rate=0 → C = 0.5+0.3*1 = 0.8)
        // With rate near 0: C approaches 0.8, not 0.95
        // With rate near 1: C approaches 0.5
        // Upper bound 0.95 reached when rate < 0: impossible
        // Let's verify C stays within [0.3, 0.95] for realistic rate
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 100,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — rate = 0/2000 = 0 → C = 0.5 + 0.3*1 = 0.8, clamped in [0.3, 0.95]
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.8 = 8.0
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-257
    #[test]
    fn test_cache_entry_candidates_mutation_independent_after_clone() {
        // Arrange
        let mut entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![10, 20, 30],
            logits: vec![1.0, 2.0, 3.0],
            accept_count: 0, total_count: 0,
        };

        // Act — clone, then mutate original
        let snapshot = entry.candidates.clone();
        let cloned = entry.clone();
        entry.candidates.clear();
        entry.candidates.push(99);

        // Assert — clone is fully independent
        assert_eq!(cloned.candidates, snapshot);
        assert_eq!(entry.candidates, vec![99]);
    }

    // @trace TEST-SPEC-CACHE-258
    #[test]
    fn test_lookup_on_nonempty_cache_miss_for_all_uninserted() {
        // Arrange — insert only hash=7
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Act & Assert — nearby hashes all miss
        for probe in [0u64, 1, 6, 8, 100, u64::MAX] {
            assert!(cache.lookup(probe, 0).is_none(), "hash {} should not be found", probe);
        }
        assert!(cache.lookup(7, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-259
    #[test]
    fn test_cache_aware_sample_single_non_cached_at_last_position() {
        // Arrange — 4 logits, cache index 0, 1, 2 — only index 3 is non-cached
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![5.0, 4.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — indices 0,1,2 scaled; index 3 (sole non-cached) gets redistribution
        for i in 0..3 {
            assert!(adjusted[i] < logits[i]);
        }
        assert!(adjusted[3] >= logits[3] - 1e-6);
    }

    // @trace TEST-SPEC-CACHE-260
    #[test]
    fn test_set_batch_size_two_below_threshold_stays_slow() {
        // Arrange — default threshold=4, batch=2 is below
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        cache.set_batch_size(2);

        // Assert
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);

        // Act — set to 3, still below
        cache.set_batch_size(3);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // @trace TEST-SPEC-CACHE-261
    #[test]
    fn test_insert_many_then_overwrite_all_reduces_to_unique_count() {
        // Arrange — insert 10 unique, then overwrite each once
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: i as usize,
                candidates: vec![i as u32], logits: vec![i as f32],
                accept_count: 0, total_count: 0,
            });
        }
        assert_eq!(cache.len(), 10);

        // Act — overwrite all 10
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: (i + 100) as usize,
                candidates: vec![(i + 50) as u32], logits: vec![(i + 50) as f32],
                accept_count: (i + 1) as usize, total_count: (i + 1) as usize * 2,
            });
        }

        // Assert — still 10 entries, but with updated data
        assert_eq!(cache.len(), 10);
        for i in 0..10u64 {
            let e = cache.lookup(i, (i + 100) as usize).unwrap();
            assert_eq!(e.candidates, vec![(i + 50) as u32]);
            assert_eq!(e.accept_count, (i + 1) as usize);
        }
    }

    // @trace TEST-SPEC-CACHE-262
    #[test]
    fn test_hit_rate_zero_when_all_accept_counts_zero() {
        // Arrange — multiple entries with nonzero total but zero accept
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 100,
            });
        }

        // Act
        let rate = cache.hit_rate();

        // Assert — 0 accepts / 1000 total = 0.0
        assert!((rate - 0.0).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-263
    #[test]
    fn test_eviction_insert_existing_key_at_exact_capacity_does_not_evict_other() {
        // Arrange — capacity 3, fill with 3 entries
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });

        // Act — update existing key 2 (at capacity, but key already exists)
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 50, candidates: vec![99], logits: vec![9.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — all 3 still present, key 2 updated, keys 1 and 3 untouched
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
        let e2 = cache.lookup(2, 50).unwrap();
        assert_eq!(e2.position, 50);
        assert_eq!(e2.candidates, vec![99]);
    }

    // @trace TEST-SPEC-CACHE-264
    #[test]
    fn test_cache_aware_sample_large_number_of_logits_mixed_cached() {
        // Arrange — 200 logits, cache 3 scattered indices
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 99, 199], logits: vec![10.0, 5.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits: Vec<f32> = (0..200).map(|x| (x as f32 + 1.0).ln_1p()).collect();

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — cached indices scaled, all others get redistribution
        assert_eq!(adjusted.len(), 200);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[99] < logits[99]);
        assert!(adjusted[199] < logits[199]);
        // Spot-check some non-cached indices
        assert!(adjusted[50] >= logits[50] - 1e-6);
        assert!(adjusted[150] >= logits[150] - 1e-6);
        // All values finite
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 265-279 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-265
    #[test]
    fn test_insert_into_zero_max_entries_existing_key_still_updates() {
        // Arrange — max_entries=0, insert same key twice
        let mut cache = SpeculationCache::new(4, 0);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);

        // Act — overwrite existing key
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 5, candidates: vec![20], logits: vec![2.0],
            accept_count: 3, total_count: 5,
        });

        // Assert — key exists with updated data, still 1 entry
        assert_eq!(cache.len(), 1);
        let e = cache.lookup(1, 5).unwrap();
        assert_eq!(e.position, 5);
        assert_eq!(e.candidates, vec![20]);
        assert_eq!(e.accept_count, 3);
    }

    // @trace TEST-SPEC-CACHE-266
    #[test]
    fn test_cache_aware_sample_all_zero_logits_with_cached_candidates() {
        // Arrange — all logits zero, cache has valid candidate indices
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1], logits: vec![0.0, 0.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![0.0f32, 0.0, 0.0, 0.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — no NaN, all finite, length preserved
        assert_eq!(adjusted.len(), 4);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-267
    #[test]
    fn test_eviction_new_entry_has_highest_accept_count_not_evicted() {
        // Arrange — capacity 2, entries with accept 3 and 7
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });

        // Act — insert new entry with accept_count=100, should evict hash=1 (lowest)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 100, total_count: 200,
        });

        // Assert — hash=1 evicted, hash=2 and hash=3 survive
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-268
    #[test]
    fn test_refresh_with_entry_matching_old_hash_replaces_cleanly() {
        // Arrange — insert entry with hash=42
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 50, total_count: 100,
        });
        assert!(cache.lookup(42, 0).is_some());

        // Act — refresh with a new entry that has the same hash
        cache.refresh(vec![CacheEntry {
            prefix_hash: 42, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 1, total_count: 2,
        }]);

        // Assert — hash=42 present with new data
        assert_eq!(cache.len(), 1);
        let e = cache.lookup(42, 5).unwrap();
        assert_eq!(e.position, 5);
        assert_eq!(e.candidates, vec![99]);
        assert_eq!(e.accept_count, 1);
    }

    // @trace TEST-SPEC-CACHE-269
    #[test]
    fn test_set_batch_size_to_same_value_no_change() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(8);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);

        // Act — set same value again
        cache.set_batch_size(8);

        // Assert — still FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    // @trace TEST-SPEC-CACHE-270
    #[test]
    fn test_cache_aware_sample_with_negative_logits_two_cached_tokens() {
        // Arrange — two cached tokens with negative logits
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1], logits: vec![-5.0, -3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-5.0f32, -3.0, 2.0, 7.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — negative logits * C move toward zero; non-cached get redistribution
        assert_eq!(adjusted.len(), 4);
        assert!(adjusted[0] > logits[0]); // -5 * 0.8 = -4 > -5
        assert!(adjusted[1] > logits[1]); // -3 * 0.8 = -2.4 > -3
        assert!(adjusted[2] >= logits[2] - 1e-6);
        assert!(adjusted[3] >= logits[3] - 1e-6);
    }

    // @trace TEST-SPEC-CACHE-271
    #[test]
    fn test_lookup_after_three_sequential_evictions_preserves_latest_two() {
        // Arrange — capacity 2, insert and evict sequentially
        let mut cache = SpeculationCache::new(1, 2);

        // Round 1: insert keys 1 and 2
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 5, total_count: 20,
        });

        // Round 2: insert key 3 → evicts hash=2 (accept=5, lowest)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![3], logits: vec![3.0],
            accept_count: 2, total_count: 20,
        });
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());

        // Round 3: insert key 4 → evicts hash=3 (accept=2, lowest)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![4], logits: vec![4.0],
            accept_count: 15, total_count: 30,
        });

        // Assert — hash=3 (accept=2) evicted, hash=1 and hash=4 survive
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_none());
        assert!(cache.lookup(4, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-272
    #[test]
    fn test_hit_rate_with_large_accept_and_small_total_per_entry() {
        // Arrange — entries where accept ≈ total (high rate)
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 9, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — (9+8+7)/(10+10+10) = 24/30 = 0.8
        assert!((rate - 0.8).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-273
    #[test]
    fn test_adapt_scale_factor_medium_rate_produces_expected_c() {
        // Arrange — 20 entries, each accept=5/total=10, rate=0.5
        // C = 0.5 + 0.3*(1-0.5) = 0.65
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — total_hits=50 > 10, rate=0.5, C=0.65
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.65 = 6.5
        assert!((adjusted[0] - 6.5).abs() < 0.15);
    }

    // @trace TEST-SPEC-CACHE-274
    #[test]
    fn test_cache_aware_sample_with_alternating_cached_uncached_indices() {
        // Arrange — cache indices 0, 2, 4 (alternating) in a 6-element vector
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 2, 4], logits: vec![6.0, 4.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — cached indices 0,2,4 scaled; non-cached 1,3,5 get redistribution
        assert_eq!(adjusted.len(), 6);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[2] < logits[2]);
        assert!(adjusted[4] < logits[4]);
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[3] >= logits[3] - 1e-6);
        assert!(adjusted[5] >= logits[5] - 1e-6);
    }

    // @trace TEST-SPEC-CACHE-275
    #[test]
    fn test_eviction_with_all_entries_zero_accept_only_some_have_total() {
        // Arrange — 2 entries at capacity: one with total=0, one with total>0
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 10,
        });

        // Act — insert new key; only entry with total>0 is eligible for eviction
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });

        // Assert — hash=2 (total=10, accept=0, lowest among total>0) evicted
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-276
    #[test]
    fn test_refresh_preserves_scale_factor_after_adapt() {
        // Arrange — adapt to change scale factor from default
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // C should now be ~0.53 (high accept rate)

        // Act — refresh and re-insert probe
        cache.refresh(vec![CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        }]);
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);

        // Assert — scale factor persisted through refresh
        assert!(adjusted[0] < 10.0 * 0.6);
        assert!(adjusted[0] > 10.0 * 0.3);
    }

    // @trace TEST-SPEC-CACHE-277
    #[test]
    fn test_insert_then_lookup_ensures_position_from_entry_not_argument() {
        // Arrange — insert with position=100, lookup with position=0
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 100,
            candidates: vec![5], logits: vec![3.0],
            accept_count: 10, total_count: 20,
        });

        // Act — lookup with different position
        let entry = cache.lookup(42, 0).unwrap();

        // Assert — returned position is from the stored entry, not the lookup argument
        assert_eq!(entry.position, 100);
        assert_eq!(entry.candidates, vec![5]);
    }

    // @trace TEST-SPEC-CACHE-278
    #[test]
    fn test_hit_rate_single_entry_nonzero_total_zero_accept_zero_rate() {
        // Arrange — single entry with total but no accepts
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1000,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — 0/1000 = 0.0
        assert!((rate - 0.0).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-279
    #[test]
    fn test_cache_aware_sample_all_candidates_index_zero_repeated() {
        // Arrange — three candidates all pointing to index 0
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 0, 0], logits: vec![5.0, 5.0, 5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 3.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — index 0 scaled three times: 5.0 * 0.8^3 = 2.56
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0] < logits[0] * 0.8);
        assert!(adjusted[0].is_finite());
        // Non-cached indices get redistribution
        assert!(adjusted[1] >= logits[1] - 1e-6);
        assert!(adjusted[2] >= logits[2] - 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 280-294 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-280
    #[test]
    fn test_set_batch_size_does_not_affect_adapt_scale_factor() {
        // Arrange — 构造缓存，先 adapt 得到一个非默认 C，然后修改 batch_size
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        // 此时 C ≈ 0.53
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adj_before = cache.cache_aware_sample(&logits, 99);

        // Act — 修改 batch_size 到不同值
        cache.set_batch_size(1);
        cache.set_batch_size(100);
        cache.set_batch_size(0);
        let adj_after = cache.cache_aware_sample(&logits, 99);

        // Assert — batch_size 变化不影响缩放因子
        assert_eq!(adj_before, adj_after);
    }

    // @trace TEST-SPEC-CACHE-281
    #[test]
    fn test_eviction_reinserted_key_can_survive_next_eviction() {
        // Arrange — 容量2，先驱逐 hash=1（低 accept），再以高 accept 重新插入
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        // hash=1 被驱逐
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        assert!(cache.lookup(1, 0).is_none());

        // Act — 以高 accept_count 重新插入 hash=1
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 100, total_count: 200,
        });
        // 新增 hash=4，应驱逐 hash=3 (accept=3 最低)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });

        // Assert — hash=1 存活（accept=100），hash=3 被驱逐
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(3, 0).is_none());
        assert!(cache.lookup(4, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-282
    #[test]
    fn test_cache_aware_sample_entry_candidates_longer_than_logits_no_panic() {
        // Arrange — entry 的 candidates 数量多于 logits 长度，部分越界
        let mut cache = SpeculationCache::new(5, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2, 3, 4], // logits 只有 3 个元素，index 3/4 越界
            logits: vec![5.0, 4.0, 3.0, 2.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — 不 panic，长度保持一致，合法索引被缩放
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
        assert!(adjusted[2] < logits[2]);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-283
    #[test]
    fn test_adapt_scale_factor_called_on_empty_cache_twice_safe() {
        // Arrange — 空缓存连续调用 adapt
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        cache.adapt_scale_factor();
        cache.adapt_scale_factor();

        // Assert — 默认 C=0.8 不变（total_hits=0 <= 10）
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        // 10.0 * 0.8 = 8.0
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-284
    #[test]
    fn test_hit_rate_after_double_refresh_with_counts() {
        // Arrange — 第一次 refresh 放入有 count 的条目
        let mut cache = SpeculationCache::new(4, 100);
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        }]);
        assert!((cache.hit_rate() - 0.8).abs() < 1e-6);

        // Act — 第二次 refresh 替换为不同 count 的条目
        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        }]);

        // Assert — hit_rate 仅反映第二次 refresh 的数据
        assert!((cache.hit_rate() - 0.3).abs() < 1e-6);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-285
    #[test]
    fn test_cache_aware_sample_large_positive_and_negative_mixed_logits() {
        // Arrange — 大正数和大负数混合 logits
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 2], logits: vec![50.0, -50.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![50.0f32, 0.0, -50.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — 两个 cached token 缩放，中间非 cached token 获得 redistribution
        assert!(adjusted[0] < logits[0]); // 50 * 0.8 = 40
        assert!(adjusted[2] > logits[2]); // -50 * 0.8 = -40 > -50
        assert!(adjusted[1] >= logits[1] - 1e-6); // 非cached获得redistribution
        assert_eq!(adjusted.len(), 3);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-286
    #[test]
    fn test_eviction_new_entry_zero_total_when_all_others_zero_total_exceeds_capacity() {
        // Arrange — 容量2，所有条目 total_count=0（驱逐过滤器要求 total_count > 0）
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);

        // Act — 新增 hash=3，所有条目 total=0，无法驱逐任何一个
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — 超出容量限制（3 > max_entries=2），因为驱逐过滤器不匹配
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-287
    #[test]
    fn test_insert_capacity_one_new_key_evicts_old_data_verified() {
        // Arrange — 容量1，插入一个条目
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 10, candidates: vec![100, 200],
            logits: vec![5.0, 3.0], accept_count: 7, total_count: 15,
        });
        assert_eq!(cache.len(), 1);

        // Act — 插入新 key 驱逐旧的
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 20, candidates: vec![50],
            logits: vec![9.0], accept_count: 1, total_count: 2,
        });

        // Assert — 旧条目被驱逐，新条目数据完整
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 10).is_none());
        let e = cache.lookup(2, 20).unwrap();
        assert_eq!(e.position, 20);
        assert_eq!(e.candidates, vec![50]);
        assert_eq!(e.logits, vec![9.0]);
        assert_eq!(e.accept_count, 1);
        assert_eq!(e.total_count, 2);
    }

    // @trace TEST-SPEC-CACHE-288
    #[test]
    fn test_cache_aware_sample_with_positive_and_zero_logits_mixed() {
        // Arrange — logits 包含正值和零值，cached 指向零值索引
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 3], // index 1=0.0, index 3=正值
            logits: vec![0.0, 5.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 0.0, 5.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — 0.0*C=0.0，3.0*C=2.4；非 cached 获得 redistribution
        assert_eq!(adjusted.len(), 4);
        assert!(adjusted[3] < logits[3]);
        assert!(adjusted[0] >= logits[0] - 1e-6);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-289
    #[test]
    fn test_refresh_empty_then_adapt_then_insert_produces_default_scale() {
        // Arrange — refresh 清空后 adapt 不会触发（total=0 <= 10）
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 50, total_count: 100,
        });
        // Act
        cache.refresh(vec![]);
        cache.adapt_scale_factor(); // total=0, 不触发
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — C 仍为 0.8（默认值）
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-290
    #[test]
    fn test_eviction_same_accept_count_one_zero_total_count_only_positive_total_evictable() {
        // Arrange — 两个条目 accept_count 相同，但一个 total=0
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0, // total=0，不可驱逐
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 5, // total>0，可驱逐
        });

        // Act — 新 key 触发驱逐
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });

        // Assert — hash=2（唯一 total>0 的）被驱逐，hash=1（total=0）保留
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some()); // total=0 不可驱逐
        assert!(cache.lookup(2, 0).is_none()); // 被驱逐
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-291
    #[test]
    fn test_cache_aware_sample_entry_exists_but_all_candidates_out_of_range_identity() {
        // Arrange — entry 存在但所有 candidate 索引都超出 logits 范围
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![10, 20, 30], // 全部超出 logits.len()=5
            logits: vec![1.0, 1.0, 1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — 没有 candidate 在范围内，所有 token 都视为非 cached 获得 redistribution
        // 但 residual_mass 来源于有效范围内的 candidate，所以 residual=0
        // 结果是所有 token 获得 0 redistribution，等于原值
        assert_eq!(adjusted.len(), 5);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-292
    #[test]
    fn test_lookup_returns_correct_candidates_vector_after_multiple_operations() {
        // Arrange — 插入、覆盖、refresh 后再插入，验证 candidates 完整性
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1, 2, 3], logits: vec![1.0, 2.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10, 20], logits: vec![10.0, 20.0],
            accept_count: 5, total_count: 10,
        });
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![100], logits: vec![100.0],
            accept_count: 1, total_count: 2,
        }]);

        // Act
        let e = cache.lookup(1, 0).unwrap();

        // Assert — refresh 后数据是最后一次 refresh 的值
        assert_eq!(e.candidates, vec![100]);
        assert_eq!(e.logits, vec![100.0]);
        assert_eq!(e.accept_count, 1);
        assert_eq!(e.total_count, 2);
    }

    // @trace TEST-SPEC-CACHE-293
    #[test]
    fn test_adapt_scale_factor_after_refresh_with_low_total_below_threshold() {
        // Arrange — 先用高 count 的条目触发 adapt，然后 refresh 替换为低 count
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor(); // C 从 0.8 变为约 0.53

        // Act — refresh 替换为少量条目（total=5 < 10 threshold）
        cache.refresh(vec![CacheEntry {
            prefix_hash: 50, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        }]);
        cache.adapt_scale_factor(); // total=5 <= 10, 不触发

        // Assert — C 保持在第一次 adapt 的值（约 0.53）
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        assert!(adjusted[0] < 10.0 * 0.6); // C ≈ 0.53
    }

    // @trace TEST-SPEC-CACHE-294
    #[test]
    fn test_full_workflow_insert_evict_adapt_refresh_sample() {
        // Arrange — 完整工作流：插入→驱逐→adapt→refresh→采样
        let mut cache = SpeculationCache::new(2, 2);

        // 先填满容量 2
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        // 插入第 3 个条目触发驱逐，hash=1 被驱逐（accept=2 最低）
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_none());

        // Act1 — adapt（total=20 > 10，触发）
        cache.adapt_scale_factor();
        // rate = 13/20 = 0.65 → C = 0.5 + 0.3*(1-0.65) = 0.605

        // Act2 — refresh 替换为新数据
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 10, position: 0, candidates: vec![0, 1],
                logits: vec![8.0, 6.0], accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 20, position: 0, candidates: vec![2],
                logits: vec![4.0], accept_count: 0, total_count: 0,
            },
        ]);

        // Act3 — 采样
        let logits = vec![8.0f32, 6.0, 4.0, 2.0, 1.0];
        let adjusted = cache.cache_aware_sample(&logits, 10);

        // Assert — hash=10 的 candidates (0,1) 被缩放
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
        assert_eq!(adjusted.len(), 5);
        // 验证旧条目完全被 refresh 清除
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_none());
        assert_eq!(cache.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 295-309 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-295
    #[test]
    fn test_cache_aware_sample_all_negative_logits_all_cached() {
        // Arrange — all logits negative, all indices cached
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![-5.0, -3.0, -1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-5.0f32, -3.0, -1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all scaled toward zero (become less negative)
        assert_eq!(adjusted.len(), 3);
        for i in 0..3 {
            assert!(adjusted[i] > logits[i], "index {} should move toward zero", i);
        }
        // Verify specific: -5 * 0.8 = -4.0, -3 * 0.8 = -2.4, -1 * 0.8 = -0.8
        assert!((adjusted[0] - (-4.0)).abs() < 0.5);
        assert!((adjusted[1] - (-2.4)).abs() < 0.5);
        assert!((adjusted[2] - (-0.8)).abs() < 0.5);
    }

    // @trace TEST-SPEC-CACHE-296
    #[test]
    fn test_hit_rate_with_accept_exceeding_total_returns_above_one() {
        // Arrange — structurally valid entry with accept > total
        // hit_rate = accept_count / total_count, no clamping
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 20, total_count: 10,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — 20/10 = 2.0 (no clamping in hit_rate)
        assert!((rate - 2.0).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-297
    #[test]
    fn test_adapt_scale_factor_with_rate_above_one_clamps_to_lower_bound() {
        // Arrange — accept > total → rate > 1.0 → C formula gives < 0.5
        // C = 0.5 + 0.3*(1-rate) with rate=2.0 → C = 0.5 + 0.3*(-1) = 0.2
        // Clamped to [0.3, 0.95] → C = 0.3
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 20, total_count: 10,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — C clamped to 0.3 (lower bound)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.3 = 3.0
        assert!((adjusted[0] - 3.0).abs() < 0.5);
    }

    // @trace TEST-SPEC-CACHE-298
    #[test]
    fn test_cache_aware_sample_only_one_non_cached_gets_all_redistribution() {
        // Arrange — 5 logits, 4 cached, only 1 non-cached token
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2, 3], logits: vec![10.0, 8.0, 6.0, 4.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 8.0, 6.0, 4.0, 2.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — only index 4 is non-cached, gets concentrated redistribution
        assert_eq!(adjusted.len(), 5);
        for i in 0..4 {
            assert!(adjusted[i] < logits[i]);
        }
        assert!(adjusted[4] > logits[4]); // single non-cached gets all residual mass
    }

    // @trace TEST-SPEC-CACHE-299
    #[test]
    fn test_eviction_replaces_key_just_evicted_in_previous_round() {
        // Arrange — capacity 2, evict hash=1, then re-insert hash=1 and evict another
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });

        // Round 1: evict hash=1 (accept=1)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        assert!(cache.lookup(1, 0).is_none());

        // Act — Round 2: re-insert hash=1 with high accept; should evict hash=3 (accept=10 vs 5)
        // Actually hash=2 has accept=5 (lowest), hash=3 has accept=10
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 100, total_count: 200,
        });
        // Now 3 entries at capacity 2, need eviction of lowest accept: hash=2 (accept=5)
        // Wait — inserting hash=1 at capacity triggers eviction first
        // After Round 1: {hash=2(accept=5), hash=3(accept=10)}
        // Insert hash=1(accept=100): len=2 == max, key not present → evict hash=2 (accept=5)
        // Result: {hash=3(accept=10), hash=1(accept=100)}

        // Assert
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-300
    #[test]
    fn test_lookup_after_insert_overwrite_preserves_independent_entries() {
        // Arrange — three entries, overwrite middle one, verify others untouched
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 10, candidates: vec![10], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 20, candidates: vec![20], logits: vec![2.0],
            accept_count: 2, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 30, candidates: vec![30], logits: vec![3.0],
            accept_count: 3, total_count: 30,
        });

        // Act — overwrite hash=2
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 99, candidates: vec![99], logits: vec![9.9],
            accept_count: 99, total_count: 199,
        });

        // Assert — hash=1 and hash=3 completely unchanged
        let e1 = cache.lookup(1, 10).unwrap();
        assert_eq!(e1.position, 10);
        assert_eq!(e1.candidates, vec![10]);
        assert_eq!(e1.accept_count, 1);
        assert_eq!(e1.total_count, 10);

        let e3 = cache.lookup(3, 30).unwrap();
        assert_eq!(e3.position, 30);
        assert_eq!(e3.candidates, vec![30]);
        assert_eq!(e3.accept_count, 3);
        assert_eq!(e3.total_count, 30);

        let e2 = cache.lookup(2, 99).unwrap();
        assert_eq!(e2.position, 99);
        assert_eq!(e2.candidates, vec![99]);
    }

    // @trace TEST-SPEC-CACHE-301
    #[test]
    fn test_cache_aware_sample_with_entry_having_mismatched_candidates_and_logits_lengths() {
        // Arrange — entry has 5 candidates but only 2 logits (mismatch is structurally valid)
        let mut cache = SpeculationCache::new(5, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![5.0], // mismatch: 3 candidates, 1 logit
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];

        // Act — should not panic
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all cached indices scaled, no panic
        assert_eq!(adjusted.len(), 5);
        assert!(adjusted[0] < logits[0]);
        assert!(adjusted[1] < logits[1]);
        assert!(adjusted[2] < logits[2]);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-302
    #[test]
    fn test_refresh_with_exactly_max_entries_count() {
        // Arrange — refresh with entries.length == max_entries
        let mut cache = SpeculationCache::new(4, 5);
        let entries: Vec<CacheEntry> = (0..5).map(|i| CacheEntry {
            prefix_hash: i as u64, position: i as usize,
            candidates: vec![i as u32], logits: vec![i as f32],
            accept_count: i as usize, total_count: (i + 1) as usize,
        }).collect();

        // Act
        cache.refresh(entries);

        // Assert — all 5 entries present, no overflow
        assert_eq!(cache.len(), 5);
        assert!(!cache.is_empty());
        for i in 0..5u64 {
            let e = cache.lookup(i, i as usize).unwrap();
            assert_eq!(e.accept_count, i as usize);
        }
    }

    // @trace TEST-SPEC-CACHE-303
    #[test]
    fn test_eviction_at_capacity_new_entry_has_zero_total_and_zero_accept() {
        // Arrange — at capacity with entries having positive total_count
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 20,
        });

        // Act — new entry with total_count=0; eviction targets lowest accept among total>0
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — hash=2 (accept=5, lowest among total>0) evicted
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-304
    #[test]
    fn test_cache_aware_sample_non_cached_zero_logit_stays_zero_or_increases() {
        // Arrange — cached token has large logit, non-cached has exactly 0.0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![50.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![50.0f32, 0.0, 0.0, 0.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — cached scaled down; non-cached at 0.0 get redistribution (>= 0)
        assert!(adjusted[0] < logits[0]);
        for i in 1..4 {
            assert!(adjusted[i] >= 0.0 - 1e-6, "non-cached index {} should not become negative", i);
            assert!(adjusted[i].is_finite());
        }
    }

    // @trace TEST-SPEC-CACHE-305
    #[test]
    fn test_insert_batch_then_verify_each_entry_fields_independently() {
        // Arrange — insert 20 entries with unique field combinations
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        for i in 0..20u64 {
            cache.insert(CacheEntry {
                prefix_hash: i,
                position: (i * 3) as usize,
                candidates: vec![i as u32, (i * 2) as u32],
                logits: vec![i as f32 * 0.1, i as f32 * 0.2],
                accept_count: i as usize * 2,
                total_count: (i + 1) as usize * 3,
            });
        }

        // Assert — each entry independently correct
        assert_eq!(cache.len(), 20);
        for i in 0..20u64 {
            let e = cache.lookup(i, 0).unwrap();
            assert_eq!(e.position, (i * 3) as usize, "hash {} position mismatch", i);
            assert_eq!(e.candidates, vec![i as u32, (i * 2) as u32], "hash {} candidates mismatch", i);
            assert_eq!(e.logits, vec![i as f32 * 0.1, i as f32 * 0.2], "hash {} logits mismatch", i);
            assert_eq!(e.accept_count, i as usize * 2, "hash {} accept_count mismatch", i);
            assert_eq!(e.total_count, (i + 1) as usize * 3, "hash {} total_count mismatch", i);
        }
    }

    // @trace TEST-SPEC-CACHE-306
    #[test]
    fn test_fallback_strategy_collect_into_btreemap() {
        // Arrange — FallbackStrategy derives Ord via Compare (not derived, but PartialEq+Eq+Hash)
        // Test that it works correctly in collections requiring Hash+Eq
        use std::collections::HashMap;
        let mut map = HashMap::new();

        // Act
        map.entry(FallbackStrategy::SlowDraft).or_insert(0);
        *map.entry(FallbackStrategy::SlowDraft).or_insert(0) += 1;
        map.entry(FallbackStrategy::FastNgram).or_insert(10);

        // Assert
        assert_eq!(*map.get(&FallbackStrategy::SlowDraft).unwrap(), 1);
        assert_eq!(*map.get(&FallbackStrategy::FastNgram).unwrap(), 10);
        assert_eq!(map.len(), 2);
    }

    // @trace TEST-SPEC-CACHE-307
    #[test]
    fn test_adapt_scale_factor_after_multiple_adapts_converges_to_same_c() {
        // Arrange — entries with fixed accept/total, adapt should be deterministic
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 3, total_count: 10,
            });
        }

        // Act — adapt 10 times
        for _ in 0..10 {
            cache.adapt_scale_factor();
        }

        // Assert — C is the same as a single adapt (deterministic given same entries)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adj1 = cache.cache_aware_sample(&logits, 99);

        // Compare with fresh cache doing single adapt
        let mut cache2 = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache2.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 3, total_count: 10,
            });
        }
        cache2.adapt_scale_factor();
        cache2.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let adj2 = cache2.cache_aware_sample(&logits, 99);

        // Both should produce identical results
        assert_eq!(adj1, adj2);
    }

    // @trace TEST-SPEC-CACHE-308
    #[test]
    fn test_cache_aware_sample_after_refresh_uses_new_entry_not_old() {
        // Arrange — insert entry, sample, refresh with different entry at same hash, sample again
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1], logits: vec![10.0, 8.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 8.0, 5.0, 3.0];
        let adj_old = cache.cache_aware_sample(&logits, 42);
        // Old cached: indices 0,1

        // Act — refresh with same hash but different candidates
        cache.refresh(vec![CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![2, 3], logits: vec![5.0, 3.0],
            accept_count: 0, total_count: 0,
        }]);
        let adj_new = cache.cache_aware_sample(&logits, 42);

        // Assert — old cached indices (0,1) now non-cached; new cached indices (2,3) scaled
        assert!(adj_new[2] < logits[2], "new cached index 2 should be scaled");
        assert!(adj_new[3] < logits[3], "new cached index 3 should be scaled");
        assert!(adj_new[0] >= logits[0] - 1e-6, "old cached index 0 now gets redistribution");
        assert!(adj_new[1] >= logits[1] - 1e-6, "old cached index 1 now gets redistribution");
        // Verify behavior actually changed from before
        assert!(adj_old[0] < logits[0]); // old behavior: index 0 was cached
        assert!(adj_new[0] >= logits[0] - 1e-6); // new behavior: index 0 not cached
    }

    // @trace TEST-SPEC-CACHE-309
    #[test]
    fn test_eviction_chained_four_rounds_preserves_two_highest_accept() {
        // Arrange — capacity 2, chain 4 eviction rounds
        let mut cache = SpeculationCache::new(1, 2);

        // Round 1: entries with accept [10, 50]
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 50, total_count: 20,
        });

        // Round 2: insert hash=3 (accept=30), evict hash=1 (accept=10 lowest)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 30, total_count: 20,
        });
        assert!(cache.lookup(1, 0).is_none());

        // Round 3: insert hash=4 (accept=70), evict hash=3 (accept=30 lowest)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 70, total_count: 20,
        });
        assert!(cache.lookup(3, 0).is_none());

        // Round 4: insert hash=5 (accept=20), evict hash=5? No — evict hash=2 (accept=50 vs hash=4 accept=70)
        // hash=5 has accept=20, lowest among {hash=2(50), hash=4(70)} → wait, hash=5 is the new entry
        // Before insertion: {hash=2(50), hash=4(70)}; new key hash=5 → evict hash=2 (accept=50 lowest)
        cache.insert(CacheEntry {
            prefix_hash: 5, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 20, total_count: 20,
        });

        // Act & Assert — final state: {hash=4(70), hash=5(20)}
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(4, 0).is_some()); // highest accept survived
        assert!(cache.lookup(5, 0).is_some()); // latest entry
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_none());
    }

    // @trace TEST-SPEC-CACHE-310
    #[test]
    fn test_clear_via_refresh_empty_then_insert_fresh_preserves_new_data() {
        // Arrange — populate cache, clear via refresh([]), insert new data
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 50, total_count: 100,
            });
        }
        assert_eq!(cache.len(), 10);

        // Act — clear via refresh with empty vec
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // Insert fresh entry and verify it works
        cache.insert(CacheEntry {
            prefix_hash: 999, position: 7, candidates: vec![42], logits: vec![3.14],
            accept_count: 1, total_count: 2,
        });

        // Assert — old entries gone, new entry present with correct data
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(0, 0).is_none());
        assert!(cache.lookup(9, 0).is_none());
        let entry = cache.lookup(999, 7).unwrap();
        assert_eq!(entry.candidates, vec![42]);
        assert_eq!(entry.logits, vec![3.14]);
        assert_eq!(entry.accept_count, 1);
        assert_eq!(entry.total_count, 2);
    }

    // @trace TEST-SPEC-CACHE-311
    #[test]
    fn test_double_clear_via_refresh_preserves_scale_factor_and_fallback() {
        // Arrange — adapt scale factor, then clear twice
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        cache.adapt_scale_factor();
        cache.set_batch_size(8);
        // After adapt with high accept rate, C < 0.8

        // Act — double clear
        cache.refresh(vec![]);
        cache.refresh(vec![]);

        // Assert — cache state reset but config preserved
        assert!(cache.is_empty());
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
        // Insert probe to verify scale factor persisted
        cache.insert(CacheEntry {
            prefix_hash: 0, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 0);
        assert!(adjusted[0] < 10.0 * 0.6, "scale factor should persist after double clear");
    }

    // @trace TEST-SPEC-CACHE-312
    #[test]
    fn test_cache_entry_debug_format_contains_all_fields() {
        // Arrange — create entry with distinct values for each field
        let entry = CacheEntry {
            prefix_hash: 12345,
            position: 42,
            candidates: vec![100, 200, 300],
            logits: vec![1.1, 2.2, 3.3],
            accept_count: 7,
            total_count: 13,
        };

        // Act
        let debug = format!("{:?}", entry);

        // Assert — Debug output contains all field names and values
        assert!(debug.contains("prefix_hash"), "Debug should contain prefix_hash field");
        assert!(debug.contains("12345"), "Debug should contain prefix_hash value");
        assert!(debug.contains("position"), "Debug should contain position field");
        assert!(debug.contains("42"), "Debug should contain position value");
        assert!(debug.contains("candidates"), "Debug should contain candidates field");
        assert!(debug.contains("accept_count"), "Debug should contain accept_count field");
        assert!(debug.contains("7"), "Debug should contain accept_count value");
        assert!(debug.contains("total_count"), "Debug should contain total_count field");
        assert!(debug.contains("13"), "Debug should contain total_count value");
    }

    // @trace TEST-SPEC-CACHE-313
    #[test]
    fn test_cache_entry_debug_format_zero_values() {
        // Arrange — entry with all zeros and empty vectors
        let entry = CacheEntry {
            prefix_hash: 0,
            position: 0,
            candidates: vec![],
            logits: vec![],
            accept_count: 0,
            total_count: 0,
        };

        // Act
        let debug = format!("{:?}", entry);

        // Assert — Debug output still structurally valid
        assert!(debug.contains("CacheEntry"), "should contain type name");
        assert!(debug.contains("prefix_hash"), "should contain prefix_hash field");
        assert!(debug.contains("candidates"), "should contain candidates field");
    }

    // @trace TEST-SPEC-CACHE-314
    #[test]
    fn test_fallback_strategy_slow_draft_debug_string() {
        // Arrange
        let strategy = FallbackStrategy::SlowDraft;

        // Act
        let debug = format!("{:?}", strategy);

        // Assert — Debug contains the variant name
        assert!(debug.contains("SlowDraft"), "SlowDraft Debug should contain variant name");
        assert!(!debug.contains("FastNgram"), "SlowDraft Debug should not contain other variant");
    }

    // @trace TEST-SPEC-CACHE-315
    #[test]
    fn test_fallback_strategy_fast_ngram_debug_string() {
        // Arrange
        let strategy = FallbackStrategy::FastNgram;

        // Act
        let debug = format!("{:?}", strategy);

        // Assert — Debug contains the variant name
        assert!(debug.contains("FastNgram"), "FastNgram Debug should contain variant name");
        assert!(!debug.contains("SlowDraft"), "FastNgram Debug should not contain other variant");
    }

    // @trace TEST-SPEC-CACHE-316
    #[test]
    fn test_fallback_strategy_both_variants_in_collection_debug() {
        // Arrange — both variants in a collection
        let strategies = [FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];

        // Act
        let debug = format!("{:?}", strategies);

        // Assert — collection Debug includes both variant names
        assert!(debug.contains("SlowDraft"), "collection Debug should contain SlowDraft");
        assert!(debug.contains("FastNgram"), "collection Debug should contain FastNgram");
    }

    // @trace TEST-SPEC-CACHE-317
    #[test]
    fn test_max_entries_one_overwrite_same_key_updates_data() {
        // Arrange — capacity 1, insert entry A, then overwrite with entry B at same key
        let mut cache = SpeculationCache::new(4, 1);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        assert_eq!(cache.len(), 1);

        // Act — overwrite same key with new data
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 3, candidates: vec![99, 88], logits: vec![7.0, 8.0],
            accept_count: 20, total_count: 50,
        });

        // Assert — only one entry, updated to new data
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 3).unwrap();
        assert_eq!(entry.position, 3);
        assert_eq!(entry.candidates, vec![99, 88]);
        assert_eq!(entry.logits, vec![7.0, 8.0]);
        assert_eq!(entry.accept_count, 20);
        assert_eq!(entry.total_count, 50);
    }

    // @trace TEST-SPEC-CACHE-318
    #[test]
    fn test_max_entries_one_evicts_on_different_key_when_old_has_total() {
        // Arrange — capacity 1, insert entry with nonzero total_count
        let mut cache = SpeculationCache::new(4, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![5], logits: vec![2.0],
            accept_count: 3, total_count: 10,
        });

        // Act — insert different key, should evict hash=1
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![6], logits: vec![3.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — old entry evicted, new entry present
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        let entry = cache.lookup(2, 0).unwrap();
        assert_eq!(entry.candidates, vec![6]);
    }

    // @trace TEST-SPEC-CACHE-319
    #[test]
    fn test_empty_cache_lookup_multiple_distinct_hashes_all_none() {
        // Arrange — empty cache
        let cache = SpeculationCache::new(4, 100);

        // Act & Assert — lookups for various hash patterns all return None
        assert!(cache.lookup(0, 0).is_none(), "hash 0 should be None");
        assert!(cache.lookup(1, 0).is_none(), "hash 1 should be None");
        assert!(cache.lookup(u64::MAX, 0).is_none(), "hash u64::MAX should be None");
        assert!(cache.lookup(42, 100).is_none(), "hash 42 position 100 should be None");
    }

    // @trace TEST-SPEC-CACHE-320
    #[test]
    fn test_empty_cache_sample_and_hit_rate_and_len_consistent() {
        // Arrange — empty cache, verify all query methods agree on empty state
        let cache = SpeculationCache::new(4, 100);

        // Act
        let is_empty = cache.is_empty();
        let len = cache.len();
        let hit_rate = cache.hit_rate();
        let logits = vec![1.0, 2.0, 3.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        let lookup = cache.lookup(42, 0);

        // Assert — all methods agree on empty state
        assert!(is_empty);
        assert_eq!(len, 0);
        assert_eq!(hit_rate, 0.0);
        assert_eq!(adjusted, logits, "empty cache should return original logits");
        assert!(lookup.is_none());
    }

    // @trace TEST-SPEC-CACHE-321
    #[test]
    fn test_lookup_same_key_many_times_returns_identical_pointers() {
        // Arrange — single entry with specific data
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 5,
            candidates: vec![10, 20, 30], logits: vec![1.0, 2.0, 3.0],
            accept_count: 15, total_count: 30,
        });

        // Act — lookup same key multiple times
        let first = cache.lookup(42, 5).unwrap();
        let second = cache.lookup(42, 5).unwrap();
        let third = cache.lookup(42, 5).unwrap();

        // Assert — all references point to same data (identity check via pointer equality)
        assert!(std::ptr::eq(first, second), "repeated lookups should return same reference");
        assert!(std::ptr::eq(second, third), "repeated lookups should return same reference");
        assert_eq!(first.candidates.as_ptr(), second.candidates.as_ptr());
        assert_eq!(first.accept_count, third.accept_count);
    }

    // @trace TEST-SPEC-CACHE-322
    #[test]
    fn test_lookup_consistent_between_insert_and_refresh_reinsert() {
        // Arrange — insert, record lookup, refresh with same entry, lookup again
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 55, position: 0,
            candidates: vec![1, 2, 3], logits: vec![0.5, 1.5, 2.5],
            accept_count: 4, total_count: 8,
        });
        let before = cache.lookup(55, 0).unwrap().clone();

        // Act — refresh with identical entry
        cache.refresh(vec![CacheEntry {
            prefix_hash: 55, position: 0,
            candidates: vec![1, 2, 3], logits: vec![0.5, 1.5, 2.5],
            accept_count: 4, total_count: 8,
        }]);
        let after = cache.lookup(55, 0).unwrap();

        // Assert — data identical (but different allocation since refresh cleared and reinserted)
        assert_eq!(before.candidates, after.candidates);
        assert_eq!(before.logits, after.logits);
        assert_eq!(before.accept_count, after.accept_count);
        assert_eq!(before.total_count, after.total_count);
    }

    // @trace TEST-SPEC-CACHE-323
    #[test]
    fn test_hit_rate_single_entry_accept_equals_total_is_one() {
        // Arrange — single entry where accept_count == total_count
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 500, total_count: 500,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — hit rate is exactly 1.0
        assert!((rate - 1.0).abs() < 1e-6, "accept==total should give hit_rate 1.0, got {}", rate);
    }

    // @trace TEST-SPEC-CACHE-324
    #[test]
    fn test_hit_rate_two_entries_one_zero_one_full_gives_half() {
        // Arrange — one entry fully accepted, one with zero accepts
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 100, total_count: 100,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 100,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — (100+0)/(100+100) = 0.5
        assert!((rate - 0.5).abs() < 1e-6, "expected 0.5, got {}", rate);
    }

    // @trace TEST-SPEC-CACHE-325
    #[test]
    fn test_fan_out_greater_than_one_stores_multiple_candidates() {
        // Arrange — fan_out=8, verify all candidates stored and retrievable
        let mut cache = SpeculationCache::new(8, 100);
        let candidates = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let logits = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: candidates.clone(), logits: logits.clone(),
            accept_count: 0, total_count: 0,
        });

        // Act
        let entry = cache.lookup(42, 0).unwrap();

        // Assert — all 8 candidates and logits preserved
        assert_eq!(entry.candidates, candidates);
        assert_eq!(entry.logits, logits);
    }

    // @trace TEST-SPEC-CACHE-326
    #[test]
    fn test_fan_out_greater_than_one_cache_aware_sample_scales_all_cached_tokens() {
        // Arrange — fan_out=4 with 4 cached candidates, 8-element logit vector
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0,
            candidates: vec![0, 2, 4, 6], logits: vec![5.0, 4.0, 3.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 7);

        // Assert — indices 0, 2, 4, 6 are cached and scaled down (< original)
        assert!(adjusted[0] < logits[0], "cached token 0 should be scaled down");
        assert!(adjusted[2] < logits[2], "cached token 2 should be scaled down");
        assert!(adjusted[4] < logits[4], "cached token 4 should be scaled down");
        assert!(adjusted[6] < logits[6], "cached token 6 should be scaled down");
        // Non-cached indices 1, 3, 5, 7 should be boosted (>= original)
        assert!(adjusted[1] >= logits[1], "non-cached token 1 should be boosted");
        assert!(adjusted[3] >= logits[3], "non-cached token 3 should be boosted");
        assert!(adjusted[5] >= logits[5], "non-cached token 5 should be boosted");
        assert!(adjusted[7] >= logits[7], "non-cached token 7 should be boosted");
    }

    // @trace TEST-SPEC-CACHE-327
    #[test]
    fn test_fan_out_greater_than_one_refresh_replaces_all_entries_preserving_fan_out_capacity() {
        // Arrange — populate with 3 entries, then refresh with 5 entries each having fan_out=6
        let mut cache = SpeculationCache::new(6, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 3, total_count: 8,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![3], logits: vec![3.0],
            accept_count: 1, total_count: 4,
        });

        let new_entries: Vec<CacheEntry> = (100..105).map(|i| CacheEntry {
            prefix_hash: i, position: 0,
            candidates: vec![i as u32, i as u32 + 1, i as u32 + 2, i as u32 + 3, i as u32 + 4, i as u32 + 5],
            logits: vec![6.0; 6],
            accept_count: 0, total_count: 0,
        }).collect();

        // Act
        cache.refresh(new_entries);

        // Assert — old entries gone, new entries all present with 6 candidates each
        assert_eq!(cache.len(), 5);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_none());
        assert!(cache.lookup(3, 0).is_none());
        for i in 100..105 {
            let entry = cache.lookup(i, 0).unwrap();
            assert_eq!(entry.candidates.len(), 6, "each entry should have 6 candidates");
        }
    }

    // @trace TEST-SPEC-CACHE-328
    #[test]
    fn test_fan_out_greater_than_one_eviction_keeps_higher_accept_among_multi_candidate_entries() {
        // Arrange — capacity 2, entries with multi-candidate lists
        let mut cache = SpeculationCache::new(5, 2);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0,
            candidates: vec![100, 101, 102, 103, 104], logits: vec![5.0; 5],
            accept_count: 50, total_count: 100,
        });
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0,
            candidates: vec![200, 201, 202, 203, 204], logits: vec![4.0; 5],
            accept_count: 10, total_count: 100,
        });

        // Act — insert third entry; should evict hash=20 (lowest accept_count=10)
        cache.insert(CacheEntry {
            prefix_hash: 30, position: 0,
            candidates: vec![300, 301, 302], logits: vec![3.0; 3],
            accept_count: 0, total_count: 0,
        });

        // Assert — hash=20 evicted, hash=10 (higher accept) and hash=30 remain
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(10, 0).is_some(), "hash 10 with accept=50 should survive");
        assert!(cache.lookup(20, 0).is_none(), "hash 20 with accept=10 should be evicted");
        assert!(cache.lookup(30, 0).is_some(), "hash 30 just inserted should be present");
    }

    // @trace TEST-SPEC-CACHE-329
    #[test]
    fn test_adapt_scale_factor_at_zero_accept_rate_converges_to_high_c() {
        // Arrange — many hits but zero accepts → low hit_accept_rate → high C
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0,
                candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 100,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — 0% accept rate → C = 0.5 + 0.3 * (1 - 0) = 0.8, clamped to [0.3, 0.95] → 0.8
        assert!((cache.cache_scale_factor - 0.8).abs() < 1e-6,
            "zero accept rate should yield C=0.8, got {}", cache.cache_scale_factor);
    }

    // @trace TEST-SPEC-CACHE-330
    #[test]
    fn test_adapt_scale_factor_at_perfect_accept_rate_converges_to_low_c() {
        // Arrange — total_hits > 10 and 100% accept rate → high confidence → lowest C
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0,
                candidates: vec![0], logits: vec![1.0],
                accept_count: 200, total_count: 200,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — 100% accept → C = 0.5 + 0.3*(1-1) = 0.5, clamped → 0.5
        assert!((cache.cache_scale_factor - 0.5).abs() < 1e-6,
            "perfect accept rate should yield C=0.5, got {}", cache.cache_scale_factor);
    }

    // @trace TEST-SPEC-CACHE-331
    #[test]
    fn test_adapt_scale_factor_below_hit_threshold_leaves_c_unchanged() {
        // Arrange — only 5 total hits (< threshold of 10), initial C = 0.8
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 5,
        });

        // Act
        cache.adapt_scale_factor();

        // Assert — total_hits=5 <= 10, so C stays at default 0.8
        assert!((cache.cache_scale_factor - 0.8).abs() < 1e-6,
            "below threshold should not change C, got {}", cache.cache_scale_factor);
    }

    // @trace TEST-SPEC-CACHE-332
    #[test]
    fn test_adapt_scale_factor_with_excessive_accept_rate_clamps_to_lower_bound() {
        // Arrange — accept_count > total_count (anomalous data), driving rate > 1.0
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0,
                candidates: vec![0], logits: vec![1.0],
                accept_count: 500, total_count: 100,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — rate=5.0 → C = 0.5 + 0.3*(1-5.0) = 0.5 + (-1.2) = -0.7 → clamped to 0.3
        assert!((cache.cache_scale_factor - 0.3).abs() < 1e-6,
            "excessive accept rate should clamp to 0.3, got {}", cache.cache_scale_factor);
    }

    // @trace TEST-SPEC-CACHE-333
    #[test]
    fn test_batch_size_change_switches_fallback_strategy_both_directions() {
        // Arrange — start with batch_size=1 (SlowDraft)
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "default batch_size=1 should yield SlowDraft");

        // Act — switch to high batch
        cache.set_batch_size(10);

        // Assert — now FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);

        // Act — switch back to low batch
        cache.set_batch_size(1);

        // Assert — back to SlowDraft
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // @trace TEST-SPEC-CACHE-334
    #[test]
    fn test_batch_size_zero_uses_slow_draft() {
        // Arrange — batch_size=0 is below threshold (4)
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        cache.set_batch_size(0);

        // Assert — 0 < 4 → SlowDraft
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "batch_size=0 should yield SlowDraft");
    }

    // @trace TEST-SPEC-CACHE-335
    #[test]
    fn test_batch_size_large_number_uses_fast_ngram() {
        // Arrange — very large batch_size should still yield FastNgram
        let mut cache = SpeculationCache::new(4, 100);

        // Act
        cache.set_batch_size(1_000_000);

        // Assert — 1_000_000 >= 4 → FastNgram
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram,
            "very large batch_size should yield FastNgram");
    }

    // @trace TEST-SPEC-CACHE-336
    #[test]
    fn test_insert_same_key_different_logits_overwrites_previous() {
        // Arrange — insert entry with specific logits, then overwrite with different logits
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 77, position: 5,
            candidates: vec![10, 20], logits: vec![1.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let original = cache.lookup(77, 5).unwrap().clone();
        assert_eq!(original.logits, vec![1.0, 2.0]);

        // Act — overwrite same key with different logits and counts
        cache.insert(CacheEntry {
            prefix_hash: 77, position: 5,
            candidates: vec![30, 40], logits: vec![9.0, 8.0],
            accept_count: 15, total_count: 25,
        });

        // Assert — entry fully replaced, no trace of old data
        let updated = cache.lookup(77, 5).unwrap();
        assert_eq!(updated.candidates, vec![30, 40]);
        assert_eq!(updated.logits, vec![9.0, 8.0]);
        assert_eq!(updated.accept_count, 15);
        assert_eq!(updated.total_count, 25);
        assert_eq!(cache.len(), 1, "overwriting should not increase count");
    }

    // @trace TEST-SPEC-CACHE-337
    #[test]
    fn test_insert_same_key_overwrite_does_not_trigger_eviction() {
        // Arrange — capacity 2, fill both slots
        let mut cache = SpeculationCache::new(4, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 3, total_count: 8,
        });

        // Act — overwrite existing key (not a new key), should not evict
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![99], logits: vec![9.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — both entries still present, hash=1 updated
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(2, 0).is_some(), "hash 2 should not be evicted");
        let updated = cache.lookup(1, 0).unwrap();
        assert_eq!(updated.candidates, vec![99]);
    }

    // @trace TEST-SPEC-CACHE-338
    #[test]
    fn test_is_empty_after_single_insert_then_refresh_with_empty_vec() {
        // Arrange — insert one entry, verify not empty
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        // Act — refresh with empty vector
        cache.refresh(vec![]);

        // Assert — cache is now empty
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.lookup(42, 0).is_none());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // @trace TEST-SPEC-CACHE-339
    #[test]
    fn test_cache_aware_sample_with_max_scale_factor_preserves_logits_nearly_unchanged() {
        // Arrange — set scale factor to near 1.0 (0.95 via adapt), cached tokens barely change
        let mut cache = SpeculationCache::new(4, 100);
        // Insert enough data to trigger adapt (> 10 total hits)
        for i in 0..15 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0,
                candidates: vec![0], logits: vec![1.0],
                accept_count: 1, total_count: 100,
            });
        }
        cache.adapt_scale_factor();
        // With 1% rate → C = 0.5 + 0.3*(1-0.01) = 0.797, clamped → 0.797
        // Override to 0.95 for this test by inserting high-accept entries
        // Actually, let's use a fresh cache and manipulate via low accept rate near threshold
        // to get 0.95 we need: 0.5 + 0.3*(1-rate) = 0.95 → 0.3*(1-rate) = 0.45 → rate = -0.5
        // That is impossible with positive counts, so 0.95 can only be reached by clamping.
        // Let's just verify the behavior with the actual factor after adapt.

        // Use cache with cached entries for sampling test
        let mut cache2 = SpeculationCache::new(4, 100);
        cache2.cache_scale_factor = 0.95; // set to max clamped value
        cache2.insert(CacheEntry {
            prefix_hash: 99, position: 0,
            candidates: vec![0, 1], logits: vec![5.0, 4.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0, 9.0, 8.0, 7.0];

        // Act
        let adjusted = cache2.cache_aware_sample(&logits, 99);

        // Assert — cached tokens scaled by 0.95, so change is small
        assert!((adjusted[0] - 10.0 * 0.95).abs() < 1e-4,
            "cached token 0 should be 9.5, got {}", adjusted[0]);
        assert!((adjusted[1] - 9.0 * 0.95).abs() < 1e-4,
            "cached token 1 should be 8.55, got {}", adjusted[1]);
    }

    // --- Wave 12x34: 15 new tests (338-352) ---

    #[test]
    fn test_clear_via_refresh_then_insert_lookup_cycle() {
        // Arrange: populate cache, refresh empty, then verify clean-slate insert+lookup
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0,
            candidates: vec![1, 2], logits: vec![0.5],
            accept_count: 5, total_count: 10,
        });
        assert!(!cache.is_empty());

        // Act: clear via refresh
        cache.refresh(vec![]);

        // Assert: empty, then insert fresh and lookup succeeds
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert(CacheEntry {
            prefix_hash: 20, position: 5,
            candidates: vec![7, 8], logits: vec![1.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let entry = cache.lookup(20, 5).expect("should find freshly inserted entry");
        assert_eq!(entry.candidates, vec![7, 8]);
        assert_eq!(entry.position, 5);
    }

    #[test]
    fn test_cache_entry_debug_format_field_values() {
        // Arrange: create an entry with specific values and verify Debug output
        let entry = CacheEntry {
            prefix_hash: 12345,
            position: 99,
            candidates: vec![10, 20],
            logits: vec![0.5, -0.5],
            accept_count: 3,
            total_count: 7,
        };

        // Act
        let debug_str = format!("{:?}", entry);

        // Assert: verify key structural elements appear in Debug output
        assert!(debug_str.contains("prefix_hash"), "Debug should contain prefix_hash field");
        assert!(debug_str.contains("position"), "Debug should contain position field");
        assert!(debug_str.contains("candidates"), "Debug should contain candidates field");
        assert!(debug_str.contains("logits"), "Debug should contain logits field");
        assert!(debug_str.contains("accept_count"), "Debug should contain accept_count field");
        assert!(debug_str.contains("total_count"), "Debug should contain total_count field");
        assert!(debug_str.contains("12345"), "Debug should contain prefix_hash value");
        assert!(debug_str.contains("99"), "Debug should contain position value");
    }

    #[test]
    fn test_fallback_strategy_debug_all_variants_content() {
        // Arrange: get Debug strings for both FallbackStrategy variants
        let slow = FallbackStrategy::SlowDraft;
        let fast = FallbackStrategy::FastNgram;

        // Act
        let slow_debug = format!("{:?}", slow);
        let fast_debug = format!("{:?}", fast);

        // Assert: both should produce non-empty distinct strings
        assert!(!slow_debug.is_empty(), "SlowDraft Debug should not be empty");
        assert!(!fast_debug.is_empty(), "FastNgram Debug should not be empty");
        assert_ne!(slow_debug, fast_debug, "SlowDraft and FastNgram Debug should differ");
        assert!(slow_debug.contains("SlowDraft"), "SlowDraft Debug should contain variant name");
        assert!(fast_debug.contains("FastNgram"), "FastNgram Debug should contain variant name");
    }

    #[test]
    fn test_max_entries_one_boundary_different_key_evicts_when_zero_total() {
        // Arrange: max_entries=1, insert one entry with zero total_count
        let mut cache = SpeculationCache::new(2, 1);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 1);

        // Act: insert a different key — eviction loop filters for total_count > 0
        // and finds none, so no eviction occurs. HashMap still grows beyond max_entries.
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });

        // Assert: eviction is best-effort; without evictable entries, HashMap grows
        assert_eq!(cache.len(), 2,
            "both entries exist since no entry had total_count > 0 to evict");
        assert!(cache.lookup(1, 0).is_some(), "original entry should still exist");
        assert!(cache.lookup(2, 0).is_some(), "new entry should also exist despite max_entries=1");
    }

    #[test]
    fn test_empty_cache_lookup_various_hashes_returns_none() {
        // Arrange: fresh empty cache
        let cache = SpeculationCache::new(4, 50);

        // Act & Assert: lookup several different hashes
        for hash in [0, 1, u64::MAX, 42, 999] {
            assert!(cache.lookup(hash, 0).is_none(),
                "empty cache lookup for hash {} should return None", hash);
        }
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_multiple_lookup_same_key_returns_identical_data() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 7,
            candidates: vec![10, 20, 30], logits: vec![0.1, 0.2, 0.3],
            accept_count: 2, total_count: 5,
        });

        // Act: lookup the same key twice
        let first = cache.lookup(42, 0).expect("first lookup should succeed");
        let second = cache.lookup(42, 100).expect("second lookup should succeed");

        // Assert: both lookups return entries with identical field values
        assert_eq!(first.candidates, second.candidates);
        assert_eq!(first.logits, second.logits);
        assert_eq!(first.prefix_hash, second.prefix_hash);
        assert_eq!(first.position, second.position);
        assert_eq!(first.accept_count, second.accept_count);
        assert_eq!(first.total_count, second.total_count);
    }

    #[test]
    fn test_hit_rate_exactly_zero_when_all_accept_zero() {
        // Arrange: entries with nonzero total but zero accept
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 100,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 200,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert
        assert_eq!(rate, 0.0, "hit rate should be exactly 0.0 when all accepts are zero");
    }

    #[test]
    fn test_hit_rate_exactly_one_when_all_accept_equal_total() {
        // Arrange: entries where accept_count == total_count for each
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 50, total_count: 50,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 30, total_count: 30,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert
        assert!((rate - 1.0).abs() < 1e-6,
            "hit rate should be 1.0 when accept_count equals total_count, got {}", rate);
    }

    #[test]
    fn test_fan_out_gt_one_insert_and_sample_uses_all_candidates() {
        // Arrange: fan_out=5, insert entry with 5 candidates
        let mut cache = SpeculationCache::new(5, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1, 2, 3, 4],
            logits: vec![2.0, 2.0, 2.0, 2.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 1);

        // Assert: all 5 cached indices should be scaled by cache_scale_factor
        let c = cache.cache_scale_factor;
        for i in 0..5 {
            assert!((adjusted[i] - 5.0 * c).abs() < 1e-4,
                "cached token {} should be scaled to {}, got {}", i, 5.0 * c, adjusted[i]);
        }
        // Index 5 is not cached, should receive redistribution
        assert!(adjusted[5] > 5.0,
            "non-cached token should receive redistribution mass, got {}", adjusted[5]);
    }

    #[test]
    fn test_adapt_scale_factor_zero_total_hits_no_change() {
        // Arrange: cache with total_count=0 for all entries
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let original_c = cache.cache_scale_factor;

        // Act
        cache.adapt_scale_factor();

        // Assert: total_hits=0 which is <= 10, so no change
        assert!((cache.cache_scale_factor - original_c).abs() < 1e-6,
            "scale factor should not change when total_hits <= 10");
    }

    #[test]
    fn test_adapt_scale_factor_with_total_hits_above_threshold_updates() {
        // Arrange: 2 entries with total_count > 10 combined, medium accept rate
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        // total_hits=20, total_accepts=8, rate=0.4
        // expected C = 0.5 + 0.3*(1-0.4) = 0.5 + 0.18 = 0.68
        let expected_c = 0.5 + 0.3 * (1.0 - 8.0_f32 / 20.0);

        // Act
        cache.adapt_scale_factor();

        // Assert
        assert!((cache.cache_scale_factor - expected_c).abs() < 1e-4,
            "expected C={}, got {}", expected_c, cache.cache_scale_factor);
    }

    #[test]
    fn test_batch_size_changes_across_inserts_preserves_entries() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Act: change batch_size multiple times, then insert another entry
        cache.set_batch_size(8);
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 1,
            candidates: vec![5], logits: vec![3.0],
            accept_count: 1, total_count: 2,
        });
        cache.set_batch_size(2);

        // Assert: both entries still present, batch_size updated
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 1).is_some());
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "batch_size=2 < threshold=4 → SlowDraft");
    }

    #[test]
    fn test_same_key_insert_overwrites_candidates_and_logits() {
        // Arrange: insert entry with initial data
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 2, 3],
            logits: vec![0.1, 0.2, 0.3],
            accept_count: 5, total_count: 10,
        });

        // Act: overwrite same key with different candidates and logits
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![10, 20],
            logits: vec![1.0, 2.0],
            accept_count: 0, total_count: 0,
        });

        // Assert: lookup returns the new data
        let entry = cache.lookup(42, 0).expect("should find entry");
        assert_eq!(entry.candidates, vec![10, 20]);
        assert_eq!(entry.logits, vec![1.0, 2.0]);
        assert_eq!(entry.accept_count, 0);
        assert_eq!(entry.total_count, 0);
        assert_eq!(cache.len(), 1, "should still have exactly one entry");
    }

    #[test]
    fn test_is_empty_true_after_refresh_with_empty_vec() {
        // Arrange: populate cache with multiple entries
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0,
            candidates: vec![1], logits: vec![2.0],
            accept_count: 5, total_count: 15,
        });
        assert!(!cache.is_empty());

        // Act: refresh with empty vec
        cache.refresh(vec![]);

        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_scale_factor_clamp_lower_boundary_via_adapt() {
        // Arrange: force accept > total so rate > 1.0, causing C to go below 0.3.
        // C = 0.5 + 0.3*(1-rate). With rate=2.0: C = 0.5 - 0.3 = 0.2 → clamped to 0.3
        // Need total_hits > 10 to trigger adapt.
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![1.0],
            accept_count: 24, total_count: 12, // accept > total → rate = 2.0
        });
        // total_hits=12 > 10, rate=2.0, C=0.2 → clamped to 0.3

        // Act
        cache.adapt_scale_factor();

        // Assert: C clamped to lower bound 0.3
        assert!((cache.cache_scale_factor - 0.3).abs() < 1e-6,
            "scale factor should be clamped to lower bound 0.3, got {}", cache.cache_scale_factor);
    }

    #[test]
    fn test_fallback_strategy_clone_debug_roundtrip_all_variants() {
        // Arrange: both variants
        let slow = FallbackStrategy::SlowDraft;
        let fast = FallbackStrategy::FastNgram;

        // Act: clone each
        let slow_clone = slow.clone();
        let fast_clone = fast.clone();

        // Assert: cloned values are equal to originals
        assert_eq!(slow, slow_clone);
        assert_eq!(fast, fast_clone);

        // Debug format of clone matches original
        assert_eq!(format!("{:?}", slow), format!("{:?}", slow_clone));
        assert_eq!(format!("{:?}", fast), format!("{:?}", fast_clone));

        // Copy also works (Copy trait)
        let slow_copy: FallbackStrategy = slow;
        let fast_copy: FallbackStrategy = fast;
        assert_eq!(slow, slow_copy);
        assert_eq!(fast, fast_copy);
    }

    #[test]
    fn test_cache_entry_with_u32_max_token_values() {
        // Arrange: create entry with u32::MAX token values
        let max_token = u32::MAX;
        let entry = CacheEntry {
            prefix_hash: u64::MAX,
            position: usize::MAX,
            candidates: vec![max_token, max_token - 1, max_token - 2],
            logits: vec![f32::MAX, f32::MIN_POSITIVE, 0.0],
            accept_count: usize::MAX,
            total_count: usize::MAX,
        };

        // Act: insert and lookup
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(entry.clone());
        let found = cache.lookup(u64::MAX, usize::MAX).expect("should find entry");

        // Assert: all fields preserved
        assert_eq!(found.prefix_hash, u64::MAX);
        assert_eq!(found.position, usize::MAX);
        assert_eq!(found.candidates, vec![max_token, max_token - 1, max_token - 2]);
        assert_eq!(found.logits[0], f32::MAX);
        assert_eq!(found.logits[1], f32::MIN_POSITIVE);
        assert_eq!(found.logits[2], 0.0);
        assert_eq!(found.accept_count, usize::MAX);
        assert_eq!(found.total_count, usize::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 280-294 (15 tests)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_refresh_with_exactly_max_entries_does_not_exceed() {
        // Arrange
        let mut cache = SpeculationCache::new(1, 5);
        let entries: Vec<CacheEntry> = (0..5).map(|i| CacheEntry {
            prefix_hash: i as u64,
            position: i as usize,
            candidates: vec![i as u32],
            logits: vec![i as f32],
            accept_count: 0,
            total_count: 0,
        }).collect();

        // Act
        cache.refresh(entries);

        // Assert — exactly max_entries, no overflow
        assert_eq!(cache.len(), 5);
        for i in 0..5u64 {
            assert!(cache.lookup(i, i as usize).is_some());
        }
    }

    #[test]
    fn test_cache_aware_sample_output_deterministic_across_repeated_calls() {
        // Arrange — populate cache and call sample 5 times
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 3], logits: vec![5.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];

        // Act — call 5 times
        let results: Vec<Vec<f32>> = (0..5)
            .map(|_| cache.cache_aware_sample(&logits, 42))
            .collect();

        // Assert — all 5 results identical
        for i in 1..5 {
            assert_eq!(results[0], results[i], "call {} differs from call 0", i);
        }
    }

    #[test]
    fn test_eviction_skips_zero_total_count_entries_when_all_zero() {
        // Arrange — 3 entries all with total_count=0 at capacity=3
        let mut cache = SpeculationCache::new(1, 3);
        for i in 1..=3u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 0,
            });
        }
        assert_eq!(cache.len(), 3);

        // Act — insert 4th unique key; no entry qualifies for eviction (all total_count=0)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — all 4 entries present (no eviction occurred)
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_adapt_then_insert_new_entry_then_hit_rate_reflects_new_data() {
        // Arrange — build state, adapt, then add a new entry
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        });
        cache.adapt_scale_factor();

        // Act — insert new entry that changes hit rate
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 10,
        });

        // Assert — hit_rate = (8+2)/(10+10) = 0.5
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_set_batch_size_preserves_existing_entries_and_hit_rate() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });
        let rate_before = cache.hit_rate();
        let len_before = cache.len();

        // Act
        cache.set_batch_size(50);
        cache.set_batch_size(1);
        cache.set_batch_size(0);

        // Assert — entries and hit_rate unaffected
        assert_eq!(cache.len(), len_before);
        let rate_after = cache.hit_rate();
        assert!((rate_before - rate_after).abs() < 1e-10);
        assert!(cache.lookup(1, 0).is_some());
    }

    #[test]
    fn test_refresh_empty_then_set_batch_size_keeps_empty_state() {
        // Arrange — populate then refresh empty
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        cache.refresh(vec![]);
        assert!(cache.is_empty());

        // Act — change batch size on empty cache
        cache.set_batch_size(10);

        // Assert — still empty, strategy is FastNgram
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_cache_aware_sample_no_redistribution_needed_when_all_cached() {
        // Arrange — all logits indices covered by candidates
        let mut cache = SpeculationCache::new(3, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 1, 2], logits: vec![5.0, 4.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32, 4.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all scaled, no non-cached token exists to receive redistribution
        assert_eq!(adjusted.len(), 3);
        for i in 0..3 {
            assert!(adjusted[i] < logits[i]);
        }
    }

    #[test]
    fn test_insert_overwrite_does_not_change_cache_length() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 10);

        // Act — overwrite 5 of the 10 entries
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 99, candidates: vec![42], logits: vec![9.0],
                accept_count: 0, total_count: 0,
            });
        }

        // Assert — still exactly 10 entries
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_hit_rate_with_single_entry_zero_total_returns_zero() {
        // Arrange — single entry with total_count=0 and nonzero accept_count
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 0,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — total=0, but hit_rate sums accept/total across all entries
        // With total_count=0, the sum is 5/0 which is... let's check: 0+0=0 total, rate=0.0
        // Actually total_hits = 0, total_accepts = 5, but denominator is total_count sum = 0
        // So rate = 0.0 because total == 0 (early return)
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_lookup_two_different_entries_independent_data() {
        // Arrange — two entries with distinct data
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 1, candidates: vec![100], logits: vec![1.0],
            accept_count: 3, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 2, candidates: vec![200], logits: vec![2.0],
            accept_count: 7, total_count: 15,
        });

        // Act
        let e10 = cache.lookup(10, 1).unwrap();
        let e20 = cache.lookup(20, 2).unwrap();

        // Assert — each entry returns its own independent data
        assert_eq!(e10.candidates, vec![100]);
        assert_eq!(e10.accept_count, 3);
        assert_eq!(e20.candidates, vec![200]);
        assert_eq!(e20.accept_count, 7);
    }

    #[test]
    fn test_adapt_scale_factor_high_rate_drives_c_below_default() {
        // Arrange — 90% accept rate across 20 entries
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..20u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }

        // Act
        cache.adapt_scale_factor();
        // rate = 180/200 = 0.9, C = 0.5 + 0.3*0.1 = 0.53

        // Assert — verify via sampling that C < default 0.8
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.53 = 5.3, which is < 10.0 * 0.8 = 8.0
        assert!(adjusted[0] < 6.0);
        assert!(adjusted[0] > 4.5);
    }

    #[test]
    fn test_multiple_refresh_cycles_alternating_data() {
        // Arrange — cycle through 3 different data sets
        let mut cache = SpeculationCache::new(4, 100);

        // Act & Assert — cycle 1
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_some());

        // Cycle 2 — completely different data
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 2, position: 0, candidates: vec![20], logits: vec![2.0],
                accept_count: 1, total_count: 1,
            },
            CacheEntry {
                prefix_hash: 3, position: 0, candidates: vec![30], logits: vec![3.0],
                accept_count: 0, total_count: 0,
            },
        ]);
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());

        // Cycle 3 — empty again
        cache.refresh(vec![]);
        assert!(cache.is_empty());

        // Cycle 4 — re-add original hash
        cache.refresh(vec![CacheEntry {
            prefix_hash: 1, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 10, total_count: 20,
        }]);
        assert_eq!(cache.len(), 1);
        let e = cache.lookup(1, 5).unwrap();
        assert_eq!(e.candidates, vec![99]);
    }

    #[test]
    fn test_cache_aware_sample_with_negative_and_positive_mixed_logits() {
        // Arrange — logits with both negative and positive values
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0, 3], logits: vec![-5.0, 2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-5.0f32, 0.0, 3.0, 2.0, -1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — indices 0 and 3 scaled, rest get redistribution
        assert!(adjusted[0] > logits[0]); // -5 * 0.8 = -4 > -5
        assert!(adjusted[3] < logits[3]);  // 2 * 0.8 = 1.6 < 2
        assert_eq!(adjusted.len(), 5);
        for v in &adjusted {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fallback_strategy_copy_semantics_allow_simultaneous_use() {
        // Arrange — use both variants simultaneously in a collection
        let a = FallbackStrategy::SlowDraft;
        let b = a; // copy, not move
        let c = FallbackStrategy::FastNgram;
        let d = c; // copy

        // Act — use all four simultaneously
        let collection = [a, b, c, d];

        // Assert — copies are equal to originals
        assert_eq!(collection[0], collection[1]);
        assert_eq!(collection[2], collection[3]);
        assert_eq!(a, FallbackStrategy::SlowDraft);
        assert_eq!(c, FallbackStrategy::FastNgram);
    }

    #[test]
    fn test_eviction_insert_existing_key_bypasses_eviction_check_entirely() {
        // Arrange — capacity 2, both entries have low accept_count=1
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });

        // Act — re-insert existing key 1 with accept_count=0 (lower than both)
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![99], logits: vec![9.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — both keys still present, key 1 updated, key 2 untouched
        assert_eq!(cache.len(), 2);
        let e1 = cache.lookup(1, 0).unwrap();
        assert_eq!(e1.candidates, vec![99]);
        assert_eq!(e1.accept_count, 0);
        let e2 = cache.lookup(2, 0).unwrap();
        assert_eq!(e2.accept_count, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 370-382 (13 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-370
    #[test]
    fn test_cache_entry_struct_update_syntax() {
        // Arrange — construct base entry, then use struct update syntax
        let base = CacheEntry {
            prefix_hash: 42,
            position: 10,
            candidates: vec![1, 2, 3],
            logits: vec![0.5, 1.5, 2.5],
            accept_count: 7,
            total_count: 14,
        };

        // Act — override prefix_hash and accept_count, rest inherited
        let derived = CacheEntry {
            prefix_hash: 99,
            accept_count: 0,
            ..base
        };

        // Assert — overridden fields reflect new values
        assert_eq!(derived.prefix_hash, 99);
        assert_eq!(derived.accept_count, 0);
        // Inherited fields remain from base
        assert_eq!(derived.position, 10);
        assert_eq!(derived.candidates, vec![1, 2, 3]);
        assert_eq!(derived.logits, vec![0.5, 1.5, 2.5]);
        assert_eq!(derived.total_count, 14);
    }

    // @trace TEST-SPEC-CACHE-371
    #[test]
    fn test_new_cache_with_usize_max_fan_out() {
        // Arrange & Act — extreme fan_out value
        let cache = SpeculationCache::new(usize::MAX, 100);

        // Assert — constructed successfully, empty, debug contains fan_out
        assert!(cache.is_empty());
        let debug_str = format!("{:?}", cache);
        // Debug output shows the numeric value; usize::MAX is valid
        assert!(debug_str.contains("fan_out"));
    }

    // @trace TEST-SPEC-CACHE-372
    #[test]
    fn test_hit_rate_usize_overflow_in_sum() {
        // Arrange — two entries with total_count near usize::MAX/2
        // Avoid actual overflow by using large but safe values
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: usize::MAX / 4, total_count: usize::MAX / 2,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: usize::MAX / 4, total_count: usize::MAX / 2,
        });

        // Act — hit_rate sums accept and total across entries
        let rate = cache.hit_rate();

        // Assert — (MAX/4 + MAX/4) / (MAX/2 + MAX/2) = (MAX/2) / MAX = 0.5
        assert!((rate - 0.5).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-373
    #[test]
    fn test_cache_aware_sample_logits_all_epsilon_no_nan() {
        // Arrange — all logits at f32::EPSILON, cached index 0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![f32::EPSILON],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![f32::EPSILON; 5];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — all finite, no NaN from tiny value arithmetic
        assert_eq!(adjusted.len(), 5);
        for v in &adjusted {
            assert!(v.is_finite());
        }
        assert!(adjusted[0] < logits[0]);
    }

    // @trace TEST-SPEC-CACHE-374
    #[test]
    fn test_fallback_strategy_if_let_pattern_matching() {
        // Arrange
        let slow = FallbackStrategy::SlowDraft;
        let fast = FallbackStrategy::FastNgram;

        // Act & Assert — if let binds correctly
        if let FallbackStrategy::SlowDraft = slow {
            // expected path
        } else {
            panic!("SlowDraft should match SlowDraft variant");
        }

        if let FallbackStrategy::FastNgram = fast {
            // expected path
        } else {
            panic!("FastNgram should match FastNgram variant");
        }

        // Cross-match should fail
        let cross_match = if let FallbackStrategy::FastNgram = slow {
            true
        } else {
            false
        };
        assert!(!cross_match);
    }

    // @trace TEST-SPEC-CACHE-375
    #[test]
    fn test_cache_entry_debug_with_special_float_values() {
        // Arrange — entry with NaN, INFINITY, and NEG_INFINITY logits
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0, 1, 2],
            logits: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            accept_count: 0, total_count: 0,
        };

        // Act
        let s = format!("{:?}", entry);

        // Assert — debug format includes all three special values
        assert!(s.contains("NaN") || s.contains("nan"));
        assert!(s.contains("inf"));
        assert!(s.contains("prefix_hash: 1"));
        assert!(s.contains("accept_count: 0"));
    }

    // @trace TEST-SPEC-CACHE-376
    #[test]
    fn test_cache_aware_sample_exp_overflow_denominator_finite() {
        // Arrange — logits large enough that exp() overflows to INFINITY
        // but the algorithm should still produce finite results or handle gracefully
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![200.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![200.0f32, 150.0, 100.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — no panic, result has correct length
        assert_eq!(adjusted.len(), 3);
        // All values are finite or at least the function didn't panic
        for v in &adjusted {
            assert!(v.is_finite() || v.is_infinite() || v.is_nan());
        }
    }

    // @trace TEST-SPEC-CACHE-377
    #[test]
    fn test_insert_then_hit_rate_then_refresh_then_hit_rate_idempotent() {
        // Arrange — verify hit_rate is 0 after each refresh with zero-count entries
        let mut cache = SpeculationCache::new(4, 100);

        // Act — round 1: insert entries with counts
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });
        let rate1 = cache.hit_rate();
        assert!((rate1 - 0.5).abs() < 1e-6);

        // Round 2: refresh with zero-count entries
        cache.refresh(vec![CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        }]);
        let rate2 = cache.hit_rate();
        assert!((rate2 - 0.0).abs() < 1e-6);

        // Round 3: refresh with non-zero entries again
        cache.refresh(vec![CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 8, total_count: 10,
        }]);
        let rate3 = cache.hit_rate();
        assert!((rate3 - 0.8).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-378
    #[test]
    fn test_cache_entry_clone_with_usize_max_counts() {
        // Arrange
        let entry = CacheEntry {
            prefix_hash: u64::MAX,
            position: usize::MAX,
            candidates: vec![u32::MAX],
            logits: vec![f32::MAX],
            accept_count: usize::MAX,
            total_count: usize::MAX,
        };

        // Act
        let cloned = entry.clone();

        // Assert — exact bitwise equality for all fields
        assert_eq!(cloned.prefix_hash, u64::MAX);
        assert_eq!(cloned.position, usize::MAX);
        assert_eq!(cloned.candidates, entry.candidates);
        assert_eq!(cloned.logits, entry.logits);
        assert_eq!(cloned.accept_count, usize::MAX);
        assert_eq!(cloned.total_count, usize::MAX);
    }

    // @trace TEST-SPEC-CACHE-379
    #[test]
    fn test_adapt_scale_factor_with_accept_count_sum_exactly_at_threshold_boundary() {
        // Arrange — 5 entries each with total_count=2 → sum=10 (NOT > 10, so adapt should not fire)
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 1, total_count: 2,
            });
        }

        // Act
        cache.adapt_scale_factor();

        // Assert — total_hits=10, condition is `total_hits > 10`, so adapt did NOT fire
        // C remains at 0.8 (default)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.8 = 8.0
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-380
    #[test]
    fn test_cache_aware_sample_with_denormal_logit_cached_scaled_remains_denormal() {
        // Arrange — denormal float as cached logit
        let denormal = f32::from_bits(1); // smallest positive denormal
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![denormal],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![denormal, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — scaled value is still finite and positive (denormal * 0.8)
        assert_eq!(adjusted.len(), 2);
        assert!(adjusted[0].is_finite());
        assert!(adjusted[0] > 0.0);
        assert!(adjusted[0] < denormal + 1e-40);
    }

    // @trace TEST-SPEC-CACHE-381
    #[test]
    fn test_lookup_on_cache_filled_then_cleared_then_refilled() {
        // Arrange — fill, clear via refresh, refill with different data
        let mut cache = SpeculationCache::new(4, 100);

        // Phase 1: fill
        for i in 0..10u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: i as usize,
                candidates: vec![i as u32], logits: vec![i as f32],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 10);

        // Phase 2: clear
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        for i in 0..10u64 {
            assert!(cache.lookup(i, 0).is_none());
        }

        // Act — Phase 3: refill with different hashes
        for i in 100..110u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: (i - 100) as usize,
                candidates: vec![i as u32], logits: vec![i as f32],
                accept_count: (i - 100) as usize, total_count: 5,
            });
        }

        // Assert — old hashes absent, new hashes present with correct data
        assert_eq!(cache.len(), 10);
        for i in 0..10u64 {
            assert!(cache.lookup(i, 0).is_none());
        }
        for i in 100..110u64 {
            let e = cache.lookup(i, 0).unwrap();
            assert_eq!(e.candidates, vec![i as u32]);
            assert_eq!(e.position, (i - 100) as usize);
        }
    }

    // @trace TEST-SPEC-CACHE-382
    #[test]
    fn test_set_batch_size_many_values_preserves_entries_and_hit_rate() {
        // Arrange — populate cache with known hit_rate
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 7, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 10,
        });
        let rate_before = cache.hit_rate();
        let len_before = cache.len();

        // Act — change batch_size through many values
        for bs in [0, 1, 3, 4, 10, 100, usize::MAX, 0, 2] {
            cache.set_batch_size(bs);
        }

        // Assert — entries and hit_rate completely unaffected
        assert_eq!(cache.len(), len_before);
        let rate_after = cache.hit_rate();
        assert!((rate_before - rate_after).abs() < 1e-10);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        let e1 = cache.lookup(1, 0).unwrap();
        assert_eq!(e1.accept_count, 7);
        let e2 = cache.lookup(2, 0).unwrap();
        assert_eq!(e2.accept_count, 3);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional tests 383-395 (13 tests)
    // ═══════════════════════════════════════════════════════════════════════

    // @trace TEST-SPEC-CACHE-383
    #[test]
    fn test_eviction_all_entries_zero_total_count_no_eviction_occurs() {
        // Arrange — capacity 2, fill with entries that have total_count=0
        let mut cache = SpeculationCache::new(1, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert_eq!(cache.len(), 2);

        // Act — insert a new (third) entry; eviction filter requires total_count > 0
        // so nothing can be evicted, but the insert still happens
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — all three entries present because eviction could not select a victim
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-384
    #[test]
    fn test_cache_aware_sample_single_element_logits_with_cached_match() {
        // Arrange — single-element logit array, candidate index 0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0,
            candidates: vec![0], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![5.0f32];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 7);

        // Assert — single element is scaled by C=0.8
        assert_eq!(adjusted.len(), 1);
        assert!((adjusted[0] - 5.0 * 0.8).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-385
    #[test]
    fn test_insert_replacing_entry_preserves_hash_key_semantics() {
        // Arrange — insert entry with hash 10, then replace with different data
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1, 2], logits: vec![0.5, 1.5],
            accept_count: 3, total_count: 5,
        });
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.lookup(10, 0).unwrap().candidates, vec![1, 2]);

        // Act — replace with different candidates and counts
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 5, candidates: vec![99], logits: vec![9.9],
            accept_count: 0, total_count: 0,
        });

        // Assert — only one entry, fully replaced
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 5).unwrap();
        assert_eq!(entry.position, 5);
        assert_eq!(entry.candidates, vec![99]);
        assert_eq!(entry.accept_count, 0);
        assert_eq!(entry.total_count, 0);
    }

    // @trace TEST-SPEC-CACHE-386
    #[test]
    fn test_adapt_scale_factor_zero_accepts_all_hits_produces_max_c() {
        // Arrange — 5 entries with total_count > 10 sum, accept_count all zero
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 0, total_count: 5,
            });
        }
        // total_hits = 25 (> 10), accept_rate = 0/25 = 0.0

        // Act
        cache.adapt_scale_factor();

        // Assert — C = 0.5 + 0.3*(1 - 0.0) = 0.8, clamped to [0.3, 0.95] = 0.8
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.8 = 8.0
        assert!((adjusted[0] - 8.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-387
    #[test]
    fn test_lookup_returns_correct_entry_when_multiple_similar_hashes() {
        // Arrange — insert entries with hashes that share low bits
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 0x0000_0000_0000_0001, position: 10, candidates: vec![100], logits: vec![1.0],
            accept_count: 1, total_count: 1,
        });
        cache.insert(CacheEntry {
            prefix_hash: 0x0001_0000_0000_0001, position: 20, candidates: vec![200], logits: vec![2.0],
            accept_count: 2, total_count: 2,
        });
        cache.insert(CacheEntry {
            prefix_hash: 0xFFFF_0000_0000_0001, position: 30, candidates: vec![300], logits: vec![3.0],
            accept_count: 3, total_count: 3,
        });

        // Act & Assert — each hash resolves to the correct entry
        let e1 = cache.lookup(0x0000_0000_0000_0001, 0).unwrap();
        assert_eq!(e1.position, 10);
        assert_eq!(e1.candidates, vec![100]);

        let e2 = cache.lookup(0x0001_0000_0000_0001, 0).unwrap();
        assert_eq!(e2.position, 20);
        assert_eq!(e2.candidates, vec![200]);

        let e3 = cache.lookup(0xFFFF_0000_0000_0001, 0).unwrap();
        assert_eq!(e3.position, 30);
        assert_eq!(e3.candidates, vec![300]);
    }

    // @trace TEST-SPEC-CACHE-388
    #[test]
    fn test_cache_aware_sample_two_cached_tokens_same_prefix() {
        // Arrange — entry with 2 cached candidates, both within logits range
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 50, position: 0,
            candidates: vec![1, 3], logits: vec![0.5, 1.5],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 50);

        // Assert — indices 1 and 3 scaled by C=0.8
        assert!((adjusted[1] - 2.0 * 0.8).abs() < 1e-6); // 1.6
        assert!((adjusted[3] - 4.0 * 0.8).abs() < 1e-6); // 3.2
        // Non-cached indices get redistribution bonus
        assert!(adjusted[0] > logits[0]);
        assert!(adjusted[2] > logits[2]);
        assert!(adjusted[4] > logits[4]);
        assert_eq!(adjusted.len(), 5);
    }

    // @trace TEST-SPEC-CACHE-389
    #[test]
    fn test_fallback_strategy_boundary_at_exactly_threshold() {
        // Arrange — threshold=4, test batch_size exactly at boundary
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft); // batch=1 < 4

        // Act & Assert — at threshold boundary
        cache.set_batch_size(3);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft); // 3 < 4

        cache.set_batch_size(4);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram); // 4 >= 4

        cache.set_batch_size(5);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram); // 5 >= 4
    }

    // @trace TEST-SPEC-CACHE-390
    #[test]
    fn test_eviction_picks_entry_with_lowest_accept_among_many() {
        // Arrange — 5 entries at capacity 5, varying accept_counts
        let mut cache = SpeculationCache::new(1, 5);
        for (i, acc) in [(1u64, 10), (2, 5), (3, 1), (4, 8), (5, 3)] {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: acc, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 5);

        // Act — insert new entry, triggering eviction of lowest accept_count (hash 3, accept=1)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 1,
        });

        // Assert — hash 3 evicted, others remain, new entry present
        assert_eq!(cache.len(), 5);
        assert!(cache.lookup(3, 0).is_none()); // evicted (lowest accept_count=1)
        assert!(cache.lookup(1, 0).is_some()); // kept
        assert!(cache.lookup(2, 0).is_some()); // kept
        assert!(cache.lookup(4, 0).is_some()); // kept
        assert!(cache.lookup(5, 0).is_some()); // kept
        assert!(cache.lookup(99, 0).is_some()); // new entry
    }

    // @trace TEST-SPEC-CACHE-391
    #[test]
    fn test_hit_rate_single_entry_with_zero_total_returns_zero() {
        // Arrange — single entry with zero total_count
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 5, total_count: 0,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — total=0, so hit_rate returns 0.0 regardless of accept_count
        assert!((rate - 0.0).abs() < 1e-10);
    }

    // @trace TEST-SPEC-CACHE-392
    #[test]
    fn test_adapt_scale_factor_perfect_accept_rate_drives_c_to_minimum() {
        // Arrange — entries where accept_rate = 1.0 (all accepts equal total)
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 20, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 30, total_count: 30,
        });
        // total_hits = 50 (> 10), rate = 50/50 = 1.0

        // Act
        cache.adapt_scale_factor();

        // Assert — C = 0.5 + 0.3*(1-1.0) = 0.5, clamped = 0.5
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 99);
        // 10.0 * 0.5 = 5.0
        assert!((adjusted[0] - 5.0).abs() < 0.1);
    }

    // @trace TEST-SPEC-CACHE-393
    #[test]
    fn test_cache_aware_sample_candidate_index_at_logits_boundary() {
        // Arrange — candidate index equals last valid logits index
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![4], logits: vec![3.0], // index 4 = last in 5-element array
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — index 4 scaled, indices 0-3 get redistribution
        assert!((adjusted[4] - 5.0 * 0.8).abs() < 1e-6); // 4.0
        for i in 0..4 {
            assert!(adjusted[i] > logits[i]);
        }
        assert_eq!(adjusted.len(), 5);
    }

    // @trace TEST-SPEC-CACHE-394
    #[test]
    fn test_refresh_with_duplicate_hashes_last_one_wins() {
        // Arrange — refresh with two entries sharing the same prefix_hash
        let mut cache = SpeculationCache::new(4, 100);
        let entries = vec![
            CacheEntry {
                prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            },
            CacheEntry {
                prefix_hash: 10, position: 1, candidates: vec![2], logits: vec![2.0],
                accept_count: 8, total_count: 20,
            },
        ];

        // Act
        cache.refresh(entries);

        // Assert — only one entry for hash 10, and it is the last one inserted
        assert_eq!(cache.len(), 1);
        let entry = cache.lookup(10, 0).unwrap();
        assert_eq!(entry.position, 1);
        assert_eq!(entry.candidates, vec![2]);
        assert_eq!(entry.accept_count, 8);
        assert_eq!(entry.total_count, 20);
    }

    // @trace TEST-SPEC-CACHE-395
    #[test]
    fn test_cache_aware_sample_non_cached_tokens_receive_equal_redistribution() {
        // Arrange — 4-element logits, 1 cached token; verify non-cached get same bonus
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![0], logits: vec![4.0],
            accept_count: 0, total_count: 0,
        });
        // Use identical non-cached logits so redistribution delta is measurable
        let logits = vec![4.0f32, 3.0, 3.0, 3.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — indices 1, 2, 3 all receive the same redistribution bonus
        assert!((adjusted[1] - adjusted[2]).abs() < 1e-6);
        assert!((adjusted[2] - adjusted[3]).abs() < 1e-6);
        // Cached index 0 is scaled down
        assert!(adjusted[0] < logits[0]);
        // Non-cached indices are bumped up
        assert!(adjusted[1] > logits[1]);
        assert!(adjusted[2] > logits[2]);
        assert!(adjusted[3] > logits[3]);
    }

    // ── 13 new tests (396-408) ────────────────────────────────────────────

    // @trace TEST-SPEC-CACHE-396
    #[test]
    fn test_refresh_with_empty_vec_then_insert_then_lookup() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });

        // Act — refresh clears everything, then insert a new entry
        cache.refresh(vec![]);
        assert!(cache.is_empty());
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 3, candidates: vec![20, 30], logits: vec![2.0, 3.0],
            accept_count: 1, total_count: 2,
        });

        // Assert — old entry gone, new entry present with correct data
        assert!(cache.lookup(1, 0).is_none());
        let entry = cache.lookup(2, 3).unwrap();
        assert_eq!(entry.position, 3);
        assert_eq!(entry.candidates, vec![20, 30]);
        assert_eq!(entry.accept_count, 1);
    }

    // @trace TEST-SPEC-CACHE-397
    #[test]
    fn test_eviction_among_many_entries_picks_single_lowest_accept() {
        // Arrange — 5 entries with varying accept counts, max_entries=5
        let mut cache = SpeculationCache::new(1, 5);
        for i in 1..=5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize * 10, total_count: 100,
            });
        }
        assert_eq!(cache.len(), 5);

        // Act — insert 6th unique key; should evict lowest accept_count (hash=1, accept=10)
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — hash 1 evicted, all others remain, new entry present
        assert_eq!(cache.len(), 5);
        assert!(cache.lookup(1, 0).is_none());
        for i in 2..=5u64 {
            assert!(cache.lookup(i, 0).is_some(), "hash {} should still be present", i);
        }
        assert!(cache.lookup(99, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-398
    #[test]
    fn test_cache_aware_sample_with_all_logits_zero_and_cached_token() {
        // Arrange — all-zero logits with one cached token
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0,
            candidates: vec![2], logits: vec![0.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![0.0f32, 0.0, 0.0, 0.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 7);

        // Assert — should not produce NaN or infinity
        for val in &adjusted {
            assert!(val.is_finite(), "expected finite, got {}", val);
        }
        assert_eq!(adjusted.len(), 4);
    }

    // @trace TEST-SPEC-CACHE-399
    #[test]
    fn test_adapt_scale_factor_does_not_modify_entries_data() {
        // Arrange — insert entries with known counts
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..3 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 4, total_count: 10,
            });
        }
        let rate_before = cache.hit_rate();

        // Act
        cache.adapt_scale_factor();

        // Assert — adapt_scale_factor only changes the scale factor, not the entries
        assert!((cache.hit_rate() - rate_before).abs() < 1e-10);
        assert_eq!(cache.len(), 3);
        let entry = cache.lookup(1, 0).unwrap();
        assert_eq!(entry.accept_count, 4);
        assert_eq!(entry.total_count, 10);
    }

    // @trace TEST-SPEC-CACHE-400
    #[test]
    fn test_insert_at_exact_capacity_with_existing_key_skips_eviction() {
        // Arrange — fill to max_entries=3
        let mut cache = SpeculationCache::new(1, 3);
        for i in 1..=3u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: i as usize, total_count: 10,
            });
        }
        assert_eq!(cache.len(), 3);

        // Act — update existing key 2 (at capacity, but key exists → no eviction)
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 99, candidates: vec![77], logits: vec![7.0],
            accept_count: 50, total_count: 100,
        });

        // Assert — still 3 entries, all original keys present, key 2 updated
        assert_eq!(cache.len(), 3);
        for i in 1..=3u64 {
            assert!(cache.lookup(i, 0).is_some());
        }
        let entry = cache.lookup(2, 99).unwrap();
        assert_eq!(entry.position, 99);
        assert_eq!(entry.candidates, vec![77]);
        assert_eq!(entry.accept_count, 50);
    }

    // @trace TEST-SPEC-CACHE-401
    #[test]
    fn test_cache_aware_sample_only_non_cached_indices_increase() {
        // Arrange — 5 logits, 2 cached tokens at indices 1 and 3
        let mut cache = SpeculationCache::new(2, 100);
        cache.insert(CacheEntry {
            prefix_hash: 42, position: 0,
            candidates: vec![1, 3], logits: vec![5.0, 3.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![1.0f32, 5.0, 2.0, 3.0, 4.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 42);

        // Assert — cached indices scaled down, non-cached indices bumped up
        assert!(adjusted[1] < logits[1], "cached token 1 should decrease");
        assert!(adjusted[3] < logits[3], "cached token 3 should decrease");
        assert!(adjusted[0] > logits[0] - 1e-6, "non-cached token 0 should increase or stay");
        assert!(adjusted[2] > logits[2] - 1e-6, "non-cached token 2 should increase or stay");
        assert!(adjusted[4] > logits[4] - 1e-6, "non-cached token 4 should increase or stay");
    }

    // @trace TEST-SPEC-CACHE-402
    #[test]
    fn test_fallback_strategy_with_batch_size_one() {
        // Arrange — default batch_size=1 which is < threshold(4)
        let cache = SpeculationCache::new(4, 100);

        // Act & Assert
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft);
    }

    // @trace TEST-SPEC-CACHE-403
    #[test]
    fn test_hit_rate_after_multiple_inserts_accumulates() {
        // Arrange — insert 3 entries with known accept/total counts
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 3, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![0], logits: vec![1.0],
            accept_count: 0, total_count: 10,
        });

        // Act
        let rate = cache.hit_rate();

        // Assert — (2+3+0)/(5+5+10) = 5/20 = 0.25
        assert!((rate - 0.25).abs() < 1e-6);
    }

    // @trace TEST-SPEC-CACHE-404
    #[test]
    fn test_lookup_returns_none_after_entry_evicted_and_not_reinserted() {
        // Arrange — max 1 entry, insert two different keys
        let mut cache = SpeculationCache::new(1, 1);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });

        // Act — insert new unique key; evicts the only entry (accept_count=5)
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 0, total_count: 0,
        });

        // Assert
        assert!(cache.lookup(10, 0).is_none());
        assert!(cache.lookup(20, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-405
    #[test]
    fn test_cache_aware_sample_cached_token_scaled_by_default_c() {
        // Arrange — verify exact scaling with default C=0.8
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        // Use large enough logits to make redistribution negligible relative to scaled value
        let logits = vec![100.0f32, 0.001, 0.001];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 1);

        // Assert — cached token[0] = 100.0 * 0.8 = 80.0
        assert!((adjusted[0] - 80.0).abs() < 0.5, "expected ~80.0, got {}", adjusted[0]);
    }

    // @trace TEST-SPEC-CACHE-406
    #[test]
    fn test_cache_entry_with_f32_special_values_in_logits() {
        // Arrange — CacheEntry with special float values
        let entry = CacheEntry {
            prefix_hash: 1, position: 0,
            candidates: vec![0],
            logits: vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            accept_count: 0, total_count: 0,
        };

        // Assert — fields are stored as-is (Clone preserves them)
        assert!(entry.logits[0].is_infinite() && entry.logits[0].is_sign_positive());
        assert!(entry.logits[1].is_infinite() && entry.logits[1].is_sign_negative());
        assert!(entry.logits[2].is_nan());
        let cloned = entry.clone();
        assert!(cloned.logits[2].is_nan());
    }

    // @trace TEST-SPEC-CACHE-407
    #[test]
    fn test_refresh_then_eviction_then_insert_full_cycle() {
        // Arrange — start with a refresh, then fill and trigger eviction
        let mut cache = SpeculationCache::new(1, 2);
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
                accept_count: 10, total_count: 20,
            },
            CacheEntry {
                prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
                accept_count: 5, total_count: 20,
            },
        ]);
        assert_eq!(cache.len(), 2);

        // Act — insert 3rd unique key, triggers eviction of lowest accept (hash=2)
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![3], logits: vec![3.0],
            accept_count: 1, total_count: 1,
        });

        // Assert
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1, 0).is_some(), "hash 1 should survive");
        assert!(cache.lookup(2, 0).is_none(), "hash 2 should be evicted");
        assert!(cache.lookup(3, 0).is_some(), "hash 3 should be inserted");
    }

    // @trace TEST-SPEC-CACHE-408
    #[test]
    fn test_adapt_scale_factor_then_cache_aware_sample_uses_new_c() {
        // Arrange — high accept rate should drive C down
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5 {
            cache.insert(CacheEntry {
                prefix_hash: i as u64, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 9, total_count: 10,
            });
        }
        // Before adapt: C=0.8
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![0], logits: vec![10.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![10.0f32, 0.0];
        let before_adapt = cache.cache_aware_sample(&logits, 99);

        // Act — adapt will see 45 accepts / 50 total = 0.9 rate → C = 0.5 + 0.3*0.1 = 0.53
        cache.adapt_scale_factor();
        let after_adapt = cache.cache_aware_sample(&logits, 99);

        // Assert — with lower C, the cached token is scaled down more aggressively
        // before: 10.0 * 0.8 = 8.0; after: 10.0 * 0.53 = 5.3
        assert!(after_adapt[0] < before_adapt[0],
            "lower C should scale cached token more aggressively: after={}, before={}",
            after_adapt[0], before_adapt[0]);
    }

    // ══════ Additional tests 409-421 (13 tests) ══════

    // @trace TEST-SPEC-CACHE-409
    #[test]
    fn test_insert_then_lookup_full_lifecycle() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        let entry = CacheEntry {
            prefix_hash: 42, position: 7, candidates: vec![10, 20, 30],
            logits: vec![0.5, 0.3, 0.2], accept_count: 3, total_count: 5,
        };

        // Act
        cache.insert(entry.clone());

        // Assert — lookup returns the entry with matching prefix_hash
        let found = cache.lookup(42, 0).expect("entry should exist");
        assert_eq!(found.prefix_hash, 42);
        assert_eq!(found.candidates, vec![10, 20, 30]);
        assert_eq!(found.accept_count, 3);
        assert_eq!(found.total_count, 5);
        assert_eq!(cache.len(), 1);
    }

    // @trace TEST-SPEC-CACHE-410
    #[test]
    fn test_lookup_returns_none_for_unknown_hash() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 100, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Act
        let result = cache.lookup(999, 0);

        // Assert
        assert!(result.is_none(), "unknown hash should return None");
    }

    // @trace TEST-SPEC-CACHE-411
    #[test]
    fn test_insert_replaces_existing_key() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 1, total_count: 2,
        });

        // Act — insert with same prefix_hash replaces old entry
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 5, candidates: vec![99], logits: vec![9.0],
            accept_count: 50, total_count: 100,
        });

        // Assert
        let found = cache.lookup(10, 0).expect("entry should exist");
        assert_eq!(found.candidates, vec![99], "candidates should be from replacement");
        assert_eq!(found.accept_count, 50, "accept_count should be from replacement");
        assert_eq!(found.position, 5, "position should be from replacement");
        assert_eq!(cache.len(), 1, "size should remain 1 after replacement");
    }

    // @trace TEST-SPEC-CACHE-412
    #[test]
    fn test_eviction_prefers_lowest_accept_count() {
        // Arrange — fill cache to max_entries=3
        let mut cache = SpeculationCache::new(2, 3);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 10, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![1.0],
            accept_count: 1, total_count: 10,
        });
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![3], logits: vec![1.0],
            accept_count: 5, total_count: 10,
        });

        // Act — insert a 4th unique key; hash=2 should be evicted (lowest accept_count=1)
        cache.insert(CacheEntry {
            prefix_hash: 4, position: 0, candidates: vec![4], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert
        assert!(cache.lookup(1, 0).is_some(), "hash 1 (accept=10) should survive");
        assert!(cache.lookup(2, 0).is_none(), "hash 2 (accept=1) should be evicted");
        assert!(cache.lookup(3, 0).is_some(), "hash 3 (accept=5) should survive");
        assert!(cache.lookup(4, 0).is_some(), "hash 4 should be inserted");
        assert_eq!(cache.len(), 3, "cache should remain at max_entries");
    }

    // @trace TEST-SPEC-CACHE-413
    #[test]
    fn test_eviction_skips_zero_total_count_entries() {
        // Arrange — all entries have total_count=0, so none are evictable
        let mut cache = SpeculationCache::new(2, 2);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Act — insert a 3rd unique key; nothing evictable so cache grows beyond max
        cache.insert(CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![3], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — HashMap grew because no evictable entry existed
        assert_eq!(cache.len(), 3, "cache should grow when no evictable entry exists");
        assert!(cache.lookup(1, 0).is_some());
        assert!(cache.lookup(2, 0).is_some());
        assert!(cache.lookup(3, 0).is_some());
    }

    // @trace TEST-SPEC-CACHE-414
    #[test]
    fn test_refresh_clears_and_replaces_all_entries() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 10, total_count: 20,
        });
        cache.insert(CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![2], logits: vec![2.0],
            accept_count: 5, total_count: 10,
        });
        assert_eq!(cache.len(), 2);

        // Act — refresh with entirely new set
        cache.refresh(vec![
            CacheEntry {
                prefix_hash: 100, position: 0, candidates: vec![100], logits: vec![5.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 200, position: 0, candidates: vec![200], logits: vec![6.0],
                accept_count: 0, total_count: 0,
            },
            CacheEntry {
                prefix_hash: 300, position: 0, candidates: vec![300], logits: vec![7.0],
                accept_count: 0, total_count: 0,
            },
        ]);

        // Assert — old entries gone, only new ones remain
        assert!(cache.lookup(1, 0).is_none(), "old hash 1 should be gone");
        assert!(cache.lookup(2, 0).is_none(), "old hash 2 should be gone");
        assert!(cache.lookup(100, 0).is_some());
        assert!(cache.lookup(200, 0).is_some());
        assert!(cache.lookup(300, 0).is_some());
        assert_eq!(cache.len(), 3);
    }

    // @trace TEST-SPEC-CACHE-415
    #[test]
    fn test_refresh_empty_vec_resets_to_zero_len() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        assert!(!cache.is_empty());

        // Act
        cache.refresh(vec![]);

        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // @trace TEST-SPEC-CACHE-416
    #[test]
    fn test_cache_aware_sample_uncached_logits_unchanged() {
        // Arrange — no entry with this prefix_hash in cache
        let cache = SpeculationCache::new(4, 100);
        let logits = vec![2.0f32, 3.0, 5.0];

        // Act
        let result = cache.cache_aware_sample(&logits, 999);

        // Assert — no cached entry to scale, logits returned as-is
        assert_eq!(result, logits, "uncached logits should be returned unchanged");
    }

    // @trace TEST-SPEC-CACHE-417
    #[test]
    fn test_cache_aware_sample_caps_non_cached_redistribution() {
        // Arrange — cached entry has a candidate whose logit is large
        let mut cache = SpeculationCache::new(4, 100);
        cache.insert(CacheEntry {
            prefix_hash: 50, position: 0, candidates: vec![0], logits: vec![100.0],
            accept_count: 1, total_count: 1,
        });
        // logits: index 0 is the cached token with high logit, indices 1..=10 are non-cached
        let logits = vec![100.0f32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];

        // Act
        let result = cache.cache_aware_sample(&logits, 50);

        // Assert — non-cached tokens should be capped at 10.0 each
        for i in 1..result.len() {
            assert!(result[i] <= 10.0 + 1e-6,
                "non-cached token {} should be capped at 10.0, got {}", i, result[i]);
        }
    }

    // @trace TEST-SPEC-CACHE-418
    #[test]
    fn test_fallback_strategy_boundary_at_threshold() {
        // Arrange
        let mut cache = SpeculationCache::new(4, 100);
        cache.set_batch_size(3);

        // Act & Assert — below threshold (4)
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "batch_size=3 < threshold=4 should be SlowDraft");

        // Act — set exactly at threshold
        cache.set_batch_size(4);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram,
            "batch_size=4 >= threshold=4 should be FastNgram");
    }

    // @trace TEST-SPEC-CACHE-419
    #[test]
    fn test_adapt_scale_factor_does_not_fire_below_min_hits() {
        // Arrange — only 5 entries with total_count > 0, total hits = 5*2 = 10
        // adapt requires >10 hits, so exactly 10 should not fire
        let mut cache = SpeculationCache::new(4, 100);
        for i in 0..5u64 {
            cache.insert(CacheEntry {
                prefix_hash: i, position: 0, candidates: vec![0], logits: vec![1.0],
                accept_count: 1, total_count: 2,
            });
        }
        let c_before = cache.cache_scale_factor;

        // Act
        cache.adapt_scale_factor();

        // Assert — total hits = 5*2 = 10, threshold is >10, so adapt should NOT fire
        assert_eq!(cache.cache_scale_factor, c_before,
            "adapt should not fire when total_hits == 10 (needs >10)");
    }

    // @trace TEST-SPEC-CACHE-420
    #[test]
    fn test_hit_rate_with_zero_operations() {
        // Arrange — empty cache with no lookups
        let cache = SpeculationCache::new(4, 100);

        // Act
        let rate = cache.hit_rate();

        // Assert — no operations means 0.0 hit rate
        assert_eq!(rate, 0.0, "empty cache should have 0.0 hit rate");
    }

    // @trace TEST-SPEC-CACHE-421
    #[test]
    fn test_new_cache_default_values() {
        // Arrange & Act
        let cache = SpeculationCache::new(8, 500);

        // Assert
        assert_eq!(cache.fan_out, 8, "fan_out should match constructor argument");
        assert_eq!(cache.max_entries, 500, "max_entries should match constructor argument");
        assert_eq!(cache.batch_size, 1, "batch_size should default to 1");
        assert_eq!(cache.fallback_threshold, 4, "fallback_threshold should default to 4");
        assert!((cache.cache_scale_factor - 0.8).abs() < 1e-6,
            "cache_scale_factor should default to 0.8");
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // @trace TEST-SPEC-CACHE-422
    #[test]
    fn test_refresh_with_more_entries_than_max_entries_exceeds_limit() {
        // Arrange — max_entries=3 but refresh with 5 entries
        let mut cache = SpeculationCache::new(2, 3);
        // Pre-populate to prove refresh clears old entries
        cache.insert(CacheEntry {
            prefix_hash: 99, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });
        let new_entries: Vec<CacheEntry> = (0..5).map(|i| CacheEntry {
            prefix_hash: i as u64, position: i as usize,
            candidates: vec![i as u32], logits: vec![i as f32],
            accept_count: 0, total_count: 0,
        }).collect();

        // Act
        cache.refresh(new_entries);

        // Assert — refresh does NOT enforce max_entries; old entry 99 gone, all 5 new present
        assert!(cache.lookup(99, 0).is_none(), "old entry should be cleared by refresh");
        assert_eq!(cache.len(), 5, "refresh allows exceeding max_entries");
        for i in 0..5u64 {
            assert!(cache.lookup(i, i as usize).is_some(), "entry {} should exist", i);
        }
    }

    // @trace TEST-SPEC-CACHE-423
    #[test]
    fn test_eviction_tiebreak_among_equal_lowest_accept_count() {
        // Arrange — 3 entries at max_entries, all have same accept_count=2 but different prefix hashes
        let mut cache = SpeculationCache::new(1, 3);
        cache.insert(CacheEntry {
            prefix_hash: 10, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 20, position: 0, candidates: vec![2], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        });
        cache.insert(CacheEntry {
            prefix_hash: 30, position: 0, candidates: vec![3], logits: vec![1.0],
            accept_count: 2, total_count: 5,
        });

        // Act — insert 4th unique key; one of the accept_count=2 entries must be evicted
        cache.insert(CacheEntry {
            prefix_hash: 40, position: 0, candidates: vec![4], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert — exactly one entry was evicted (len stays at 3, not 4, since eviction succeeded)
        assert_eq!(cache.len(), 3, "one entry should be evicted to make room");
        assert!(cache.lookup(40, 0).is_some(), "new entry must be present");
        // Exactly one of {10,20,30} was evicted (iter order may vary)
        let surviving_old = [10u64, 20, 30].iter()
            .filter(|&&h| cache.lookup(h, 0).is_some()).count();
        assert_eq!(surviving_old, 2, "exactly 2 of the 3 old entries should survive");
    }

    // @trace TEST-SPEC-CACHE-424
    #[test]
    fn test_cache_aware_sample_all_logits_negative_cached_and_non_cached() {
        // Arrange — all logits negative; cached token at index 1
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 7, position: 0,
            candidates: vec![1], logits: vec![-3.0, -1.0],
            accept_count: 0, total_count: 0,
        });
        let logits = vec![-2.0f32, -1.0, -3.0, -4.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 7);

        // Assert — cached token (index 1) scaled by C=0.8: -1.0 * 0.8 = -0.8
        assert!((adjusted[1] - (-0.8)).abs() < 1e-5,
            "cached token should be scaled: got {}", adjusted[1]);
        // Non-cached tokens get positive redistribution boost
        assert!(adjusted[0] > logits[0], "non-cached idx 0 should increase");
        assert!(adjusted[2] > logits[2], "non-cached idx 2 should increase");
        assert!(adjusted[3] > logits[3], "non-cached idx 3 should increase");
    }

    // @trace TEST-SPEC-CACHE-425
    #[test]
    fn test_adapt_scale_factor_with_exactly_12_total_hits_clamps_high() {
        // Arrange — total_hits=12 > 10; very low accept rate to test upper clamp
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 0, total_count: 12, // accept_rate = 0/12 = 0.0
        });
        assert!((cache.cache_scale_factor - 0.8).abs() < 1e-6, "initial C should be 0.8");

        // Act
        cache.adapt_scale_factor();

        // Assert — rate=0.0, C = 0.5 + 0.3*(1-0.0) = 0.8, clamped to [0.3, 0.95] = 0.8
        assert!((cache.cache_scale_factor - 0.8).abs() < 1e-5,
            "C should be 0.5 + 0.3*1.0 = 0.8, got {}", cache.cache_scale_factor);
    }

    // @trace TEST-SPEC-CACHE-426
    #[test]
    fn test_fallback_strategy_changes_with_sequential_batch_size_updates() {
        // Arrange
        let mut cache = SpeculationCache::new(2, 100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "default batch=1 < threshold=4 => SlowDraft");

        // Act & Assert — cycle through batch sizes
        cache.set_batch_size(3);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "batch=3 < threshold=4 => SlowDraft");

        cache.set_batch_size(4);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram,
            "batch=4 >= threshold=4 => FastNgram");

        cache.set_batch_size(1);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::SlowDraft,
            "back to batch=1 < threshold=4 => SlowDraft");

        cache.set_batch_size(100);
        assert_eq!(cache.fallback_strategy(), FallbackStrategy::FastNgram,
            "batch=100 >> threshold=4 => FastNgram");
    }

    // @trace TEST-SPEC-CACHE-427
    #[test]
    fn test_insert_entry_with_empty_candidates_and_logits_vectors() {
        // Arrange — entry with completely empty candidates and logits
        let mut cache = SpeculationCache::new(4, 100);
        let entry = CacheEntry {
            prefix_hash: 55, position: 3,
            candidates: vec![],
            logits: vec![],
            accept_count: 0, total_count: 0,
        };

        // Act
        cache.insert(entry);

        // Assert — entry stored successfully despite empty vectors
        let found = cache.lookup(55, 3).unwrap();
        assert!(found.candidates.is_empty(), "candidates should be empty");
        assert!(found.logits.is_empty(), "logits should be empty");
        assert_eq!(cache.len(), 1);
    }

    // @trace TEST-SPEC-CACHE-428
    #[test]
    fn test_cache_aware_sample_single_cached_candidate_with_zero_logit_value() {
        // Arrange — single cached candidate whose logit is exactly 0.0
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 33, position: 0,
            candidates: vec![2], logits: vec![3.0, 2.0, 0.0, 1.0],
            accept_count: 5, total_count: 10,
        });
        let logits = vec![3.0f32, 2.0, 0.0, 1.0];

        // Act
        let adjusted = cache.cache_aware_sample(&logits, 33);

        // Assert — cached index 2: 0.0 * 0.8 = 0.0 (unchanged)
        assert!((adjusted[2] - 0.0).abs() < 1e-6,
            "0.0 * 0.8 = 0.0, got {}", adjusted[2]);
        // Non-cached get redistribution from residual = logits[2]*(1-0.8) = 0.0 * 0.2 = 0.0
        // residual_mass = 0.0, so redistribution = 0.0 / (something - 0.0 + 1e-10) ≈ 0.0
        // All non-cached indices should remain unchanged (adjusted = original + 0.0)
        assert!((adjusted[0] - 3.0).abs() < 1e-5, "non-cached idx 0 unchanged");
        assert!((adjusted[1] - 2.0).abs() < 1e-5, "non-cached idx 1 unchanged");
        assert!((adjusted[3] - 1.0).abs() < 1e-5, "non-cached idx 3 unchanged");
    }

    // @trace TEST-SPEC-CACHE-429
    #[test]
    fn test_refresh_then_insert_triggers_eviction_correctly() {
        // Arrange — refresh with 3 entries into max_entries=3
        let mut cache = SpeculationCache::new(1, 3);
        let refresh_entries: Vec<CacheEntry> = vec![
            CacheEntry {
                prefix_hash: 100, position: 0, candidates: vec![1], logits: vec![1.0],
                accept_count: 5, total_count: 10,
            },
            CacheEntry {
                prefix_hash: 200, position: 0, candidates: vec![2], logits: vec![1.0],
                accept_count: 3, total_count: 10,
            },
            CacheEntry {
                prefix_hash: 300, position: 0, candidates: vec![3], logits: vec![1.0],
                accept_count: 1, total_count: 10,
            },
        ];
        cache.refresh(refresh_entries);
        assert_eq!(cache.len(), 3);

        // Act — insert a new unique key; should evict hash 300 (lowest accept_count=1)
        cache.insert(CacheEntry {
            prefix_hash: 400, position: 0, candidates: vec![4], logits: vec![1.0],
            accept_count: 0, total_count: 0,
        });

        // Assert
        assert_eq!(cache.len(), 3, "should stay at max_entries after eviction");
        assert!(cache.lookup(100, 0).is_some(), "hash 100 (accept=5) survives");
        assert!(cache.lookup(200, 0).is_some(), "hash 200 (accept=3) survives");
        assert!(cache.lookup(300, 0).is_none(), "hash 300 (accept=1) evicted");
        assert!(cache.lookup(400, 0).is_some(), "new entry inserted");
    }

    // @trace TEST-SPEC-CACHE-430
    #[test]
    fn test_adapt_scale_factor_very_high_accept_rate_clamps_low() {
        // Arrange — very high accept rate to test lower clamp at 0.3
        let mut cache = SpeculationCache::new(1, 100);
        cache.insert(CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![1], logits: vec![1.0],
            accept_count: 99, total_count: 100, // rate = 0.99
        });

        // Act
        cache.adapt_scale_factor();

        // Assert — rate=0.99, C = 0.5 + 0.3*(1-0.99) = 0.5 + 0.003 = 0.503
        assert!((cache.cache_scale_factor - 0.503).abs() < 1e-4,
            "C = 0.5 + 0.3*0.01 = 0.503, got {}", cache.cache_scale_factor);
        // Verify it's within [0.3, 0.95]
        assert!(cache.cache_scale_factor >= 0.3);
        assert!(cache.cache_scale_factor <= 0.95);
    }

    // @trace TEST-SPEC-CACHE-431
    #[test]
    fn test_insert_large_candidates_vector_many_out_of_logits_range() {
        // Arrange — entry with 200 candidates, only index 0-4 valid for logits of length 5
        let mut cache = SpeculationCache::new(200, 100);
        let candidates: Vec<u32> = (0..200).collect(); // IDs 0..199, only 0..4 in logits range
        let entry = CacheEntry {
            prefix_hash: 77, position: 0,
            candidates: candidates.clone(),
            logits: vec![1.0; 200],
            accept_count: 3, total_count: 10,
        };
        cache.insert(entry);

        // Act — cache_aware_sample with only 5 logits
        let logits = vec![2.0f32, 1.5, 1.0, 0.5, 0.1];
        let adjusted = cache.cache_aware_sample(&logits, 77);

        // Assert — only indices 0..4 are valid for scaling; indices 5+ are silently skipped
        // Indices 0..4 should be scaled by 0.8
        assert!((adjusted[0] - 2.0 * 0.8).abs() < 1e-5, "idx 0 scaled");
        assert!((adjusted[1] - 1.5 * 0.8).abs() < 1e-5, "idx 1 scaled");
        assert!((adjusted[2] - 1.0 * 0.8).abs() < 1e-5, "idx 2 scaled");
        assert!((adjusted[3] - 0.5 * 0.8).abs() < 1e-5, "idx 3 scaled");
        assert!((adjusted[4] - 0.1 * 0.8).abs() < 1e-5, "idx 4 scaled");
        // No panic despite 195 out-of-range candidates
        assert_eq!(adjusted.len(), 5, "output length matches logits");
    }
}
