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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
