use crate::kv_cache::KvCacheDoubleBuffer;
use crate::kv_cache::KvCacheSlot;
use crate::scheduler::kv_optimizer::KvOptimizer;

use super::super::executor::KvCacheConfig;

pub struct KvCoordinator {
    pub kv_cache: Option<KvCacheDoubleBuffer>,
    pub kv_cache_slot: KvCacheSlot,
    pub kv_cache_config: KvCacheConfig,
    pub paged_kv_pool: Option<crate::compat::cpu_backend::PagedKvPool>,
    pub kv_optimizer: KvOptimizer,
    pub majority_kv_tier: Option<String>,
}
