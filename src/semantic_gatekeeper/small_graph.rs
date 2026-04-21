//! Level Keys 预计算与知识文本编码使用的小 CompilerGraph (SPEC §3.3).
//!
//! 两个小图复用主模型的 FusedGraphExecutor + JIT 编译管线
//! (ARCH-FULL-JIT + ARCH-CPU-GPU-UNIFIED 合规):
//!
//! - **EmbedLookupOnlyGraph**: `Gather(embed_weight)`
//!   用于 (1) Level Keys 预计算的首段,(2) 运行时知识文本编码.
//!
//! - **KProjOnlyGraph@layer_L**: `RmsNorm(input_layernorm_@L) → Gemm(k_proj_@L)`
//!   仅用于 Level Keys 预计算的末段.
//!
//! Phase A 只提供类型标记; 具体的 `CompilerGraph` 构造与编译发生在
//! Phase D (`src/semantic_gatekeeper/level_keys.rs` 的预计算函数和
//! `callback.rs` 的运行时文本编码路径).

/// EmbedLookupOnlyGraph 的类型标记.
///
/// Phase D 将此扩展为持有 `CompiledLayer` 的 newtype.
#[derive(Debug)]
pub struct EmbedLookupOnlyGraph {
    pub(super) hidden_size: usize,
    pub(super) vocab_size: usize,
}

impl EmbedLookupOnlyGraph {
    /// 描述目标小图的形状参数. Phase D 据此构造 `CompilerGraph` + `compile`.
    pub fn describe(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            hidden_size,
            vocab_size,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// KProjOnlyGraph@layer_L 的类型标记.
#[derive(Debug)]
pub struct KProjOnlyGraph {
    pub(super) layer_idx: usize,
    pub(super) hidden_size: usize,
    pub(super) kv_dim: usize,
    pub(super) rms_eps: f32,
}

impl KProjOnlyGraph {
    pub fn describe(layer_idx: usize, hidden_size: usize, kv_dim: usize, rms_eps: f32) -> Self {
        Self {
            layer_idx,
            hidden_size,
            kv_dim,
            rms_eps,
        }
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
    pub fn rms_eps(&self) -> f32 {
        self.rms_eps
    }
}
