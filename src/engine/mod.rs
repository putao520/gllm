//! Layer 3: Engine (skeleton).

pub mod attention_contracts;
pub mod executor;
pub mod pipeline;

pub use pipeline::{PipelineError, UnifiedPipeline};

/// 引擎上下文 (per SPEC 04-API-DESIGN §8.1)
///
/// 提供给 `KnowledgeDataSource::materialize()` 的引擎访问接口。
/// 允许数据源在物理化时访问引擎资源（如 KV cache、权重等）。
#[derive(Clone)]
pub struct EngineContext {
    /// 模型总层数（用于语义锚点映射）
    pub num_layers: usize,
    /// 隐藏层维度
    pub hidden_size: usize,
    /// KV cache 页大小
    pub kv_page_size: usize,
}

impl EngineContext {
    /// 创建新的引擎上下文
    pub fn new(num_layers: usize, hidden_size: usize, kv_page_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            kv_page_size,
        }
    }

    /// 从执行器配置创建引擎上下文
    pub fn from_executor_config(config: &GeneratorForwardConfig) -> Self {
        Self {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            kv_page_size: config.paged_kv_page_size,
        }
    }
}

// Re-export scheduler types
pub use crate::scheduler::batcher::ContinuousBatcher;
pub use crate::scheduler::types::{GroupState, RequestKind, SequenceGroup};
pub use crate::scheduler::{PagedScheduler, ScheduledBatch};
pub use crate::scheduler::types::{PageId, PageState, RequestId};

// Re-export engine types
pub use executor::{
    AttentionMaskType, AttentionTopology, BackendError, BatchInput, GeneratorForwardConfig,
    KvCacheConfig, KvCacheHandle, LogitsHandle, PositionEncoding, SamplingConfig, SequenceInput,
    SwapConfig,
};

// Re-export attention contract types (ARCH-ATTN-UNIFIED)
pub use attention_contracts::{
    AttentionSemantics, HeadMode, KvAppendSemantics, KvLayoutContract, KvSplitMode,
    KvStorageKind, KvView, MaskMode, PackingDescriptor, PositionContract, ScalingMode,
    VisibilityMode, WeightBacking, WeightView,
};

// Re-export knowledge injection types at engine::knowledge (per SPEC 04-API-DESIGN §7.2)
pub mod knowledge {
    pub use crate::knowledge::{
        FrozenKvSource, InjectionKind, InjectionScheduler, KnowledgeDataSource, KnowledgeError,
        KnowledgeInjectionConfig, KnowledgeInjectionResult, KvSideloadManager, LayerTarget,
        MaterializedPayload,
    };
}

// Re-export guardrail types at engine::guard (per SPEC 04-API-DESIGN §7.4)
pub mod guard {
    pub use crate::intent::{GuardProbe, SafetyPolicy};
}
