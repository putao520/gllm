use crate::kv_cache::KvCompressionStrategy;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub struct PromptCache {
    entries: HashMap<u64, CachedPrompt>,
    max_entries: usize,
}

#[derive(Clone)]
pub(crate) struct PromptCacheSnapshot {
    pub tokens: Vec<u32>,
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
    pub cached_len: usize,
    pub total_len: usize,
    pub last_hidden: Vec<f32>,
    pub compression: KvCompressionStrategy,
}

struct CachedPrompt {
    tokens: Vec<u32>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    cached_len: usize,
    total_len: usize,
    last_hidden: Vec<f32>,
    compression: KvCompressionStrategy,
    last_used: Instant,
}

impl PromptCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
        self.evict_if_needed();
    }

    pub fn lookup(
        &mut self,
        tokens: &[u32],
        compression: KvCompressionStrategy,
    ) -> Option<PromptCacheSnapshot> {
        let key = hash_prompt(tokens, compression);
        let entry = self.entries.get_mut(&key)?;
        if entry.tokens != tokens || entry.compression != compression {
            return None;
        }
        entry.last_used = Instant::now();
        Some(entry.snapshot())
    }

    pub fn insert(
        &mut self,
        tokens: Vec<u32>,
        k_cache: Vec<Vec<f32>>,
        v_cache: Vec<Vec<f32>>,
        cached_len: usize,
        total_len: usize,
        last_hidden: Vec<f32>,
        compression: KvCompressionStrategy,
    ) {
        if self.max_entries == 0 {
            return;
        }
        let key = hash_prompt(&tokens, compression);
        let entry = CachedPrompt {
            tokens,
            k_cache,
            v_cache,
            cached_len,
            total_len,
            last_hidden,
            compression,
            last_used: Instant::now(),
        };
        self.entries.insert(key, entry);
        self.evict_if_needed();
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_entries {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key);
            if let Some(key) = oldest {
                self.entries.remove(&key);
            } else {
                break;
            }
        }
    }
}

impl CachedPrompt {
    fn snapshot(&self) -> PromptCacheSnapshot {
        PromptCacheSnapshot {
            tokens: self.tokens.clone(),
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            cached_len: self.cached_len,
            total_len: self.total_len,
            last_hidden: self.last_hidden.clone(),
            compression: self.compression,
        }
    }
}

fn hash_prompt(tokens: &[u32], compression: KvCompressionStrategy) -> u64 {
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    compression.hash(&mut hasher);
    hasher.finish()
}
