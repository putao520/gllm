//! Layer 3: Engine (skeleton).

pub mod arbiter;
pub mod attention_contracts;
pub mod executor;
pub mod pipeline;
pub mod mega_kernel;
pub mod callbacks;

pub use pipeline::{PipelineError, UnifiedPipeline};

/// 引擎上下文 (per SPEC 04-API-DESIGN §8.1, §7.2, 元数据)
///
/// 提供给 `KnowledgeDataSource::materialize()` 的引擎访问接口。
/// 允许数据源在物理化时访问引擎资源（如 KV cache、权重等）。
/// 所有维度字段从模型配置读取 (Ω1 真实性原则).
#[derive(Clone)]
pub struct EngineContext {
    /// 模型总层数（用于语义锚点映射）
    pub num_layers: usize,
    /// 隐藏层维度
    pub hidden_size: usize,
    /// KV cache 页大小
    pub kv_page_size: usize,
    /// KV cache 头数 (GQA 模型, per SPEC §3.3)
    pub num_kv_heads: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
}

impl EngineContext {
    /// 创建新的引擎上下文
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        kv_page_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            num_layers,
            hidden_size,
            kv_page_size,
            num_kv_heads,
            max_seq_len,
        }
    }

    /// 从执行器配置创建引擎上下文 (Ω1: 从实际模型配置读取)
    pub fn from_executor_config(config: &GeneratorForwardConfig) -> Self {
        Self {
            num_layers: config.num_layers(),
            hidden_size: config.hidden_size(),
            kv_page_size: config.paged_kv.page_size,
            num_kv_heads: config.num_kv_heads(),
            max_seq_len: config.max_seq_len(),
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
    AttentionHeadConfig, AttentionMaskType, AttentionTopology, BackendError, BatchInput,
    GeneratorForwardConfig, KvCacheConfig, KvCacheHandle, LogitsHandle, PagedKvConfig,
    PositionEncoding, RoPEConfig, SamplingConfig, SequenceInput, SwapConfig,
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
        KnowledgeSource, InjectionKind, InjectionScheduler, KnowledgeDataSource, KnowledgeError,
        KnowledgeInjectionConfig, KnowledgeInjectionResult, KvSideloadManager, LayerTarget,
        MaterializedPayload,
    };
}

// Re-export guardrail types at engine::guard (per SPEC 04-API-DESIGN §7.4)
pub mod guard {
    pub use crate::intent::{GuardProbe, SafetyPolicy};
}
