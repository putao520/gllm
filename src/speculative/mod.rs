//! §17 JIT 自适应推测解码 (ARCH-ADAPTIVE-SPEC)
//!
//! 核心架构:
//! - **17.1**: 自推测核心管线 (EESD) — Draft(浅层变体) → Verify(全量模型) → Commit
//! - **17.2**: 零参数 Draft Adapter — 复用 lm_head.weight, 无额外模型权重
//! - **17.3**: 各向异性推测树 — PLD spine + n-gram branches
//! - **17.4**: EqSpec — Batch 正确性三不变量
//! - **17.5**: ADEPT Shadow KV 填充 — Early-Exit 场景专用
//! - **17.6**: 硬件指令级 Batch 合并 — Compact→Execute→Scatter
//! - **17.9**: 自适应调度决策
//! - **17.10**: SAGUARO 多 GPU 并行推测解码
//!
//! **设计原则**:
//! 1. 模型自身的浅层变体充当 Draft Model — 零额外权重
//! 2. EqSpec 三不变量保证 Batch 正确性
//! 3. 硬件级 Compact→Execute→Scatter 消除异构 batch 浪费
//! 4. 自适应回退: 连续低接受率时自动切回标准解码

pub mod adapter;
pub mod cache;
pub mod eagle;
pub mod engine;
pub mod mtp;
#[cfg(feature = "nccl")]
pub mod saguaro;
pub mod tree;
pub mod verify;

// Re-export core types
pub use adapter::{DraftAdapter, AdapterConfig};
pub use cache::{SpeculationCache, CacheEntry, FallbackStrategy};
pub use eagle::{EagleConfig, EagleHead, build_eagle_tree};
pub use engine::{SpecDecodingState, SpecDecodingMode};
pub use mtp::{MtpConfig, MtpHead, mtp_candidates};
pub use tree::{SpecTree, SpecNode, SpecTreeConfig, DraftSource, NgramIndex};
pub use verify::{
    VerifyResult, SequenceVerifyResult, EqSpecInvariant, EqSpecCheckResult,
    KvCommitInstruction, SpeculativePages, generate_kv_commit_instructions,
};

// SAGUARO 分布式推测解码 (REQ-DIST-017) — feature-gated
#[cfg(feature = "nccl")]
pub use saguaro::saguaro::{
    SaguaroPhase, SaguaroConfig, SaguaroResult, SaguaroAcceptanceTracker,
    SaguaroDistSpec, SaguaroPipelineStage,
};
