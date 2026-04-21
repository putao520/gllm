//! Compatibility shim for gllm_kernels phantom types.
//!
//! This module provides:
//! - `Element` re-export from gllm-kernels
//! - `Backend<E>` trait + `TensorLookup` trait
//! - `CpuBackend<E>` / `CudaBackend<E>` stub implementations
//! - Backward-compatible re-export modules (`kernel_types`, `backend_trait`, `cpu_backend`)

pub(crate) mod weight_helpers;
pub(crate) mod artifact_cache;
pub(crate) mod sampling;

mod gpu_compile;
#[allow(dead_code)]
pub(crate) mod types;
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub(crate) mod scalar_ops;
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) mod jit_helpers;
#[allow(dead_code)]
pub(crate) mod graph_builders;
#[allow(dead_code)]
pub(crate) mod gpu_helpers;
#[macro_use]
mod gpu_backend_macro;
pub mod cpu_backend;
mod cuda_backend;
mod hip_backend;
mod metal_backend;
mod memory;
pub mod audio_forward;
pub mod multimodal;
pub mod vision_forward;

// Re-export the *real* Element trait from gllm-kernels.
pub use gllm_kernels::traits::Element;

// Backward-compatible re-export modules
pub mod kernel_types {
    pub use crate::engine::executor::{
        GeneratorForwardConfig, KvCacheConfig, PositionEncoding, SamplingConfig, SwapConfig,
    };
    pub use crate::scheduler::types::{PageId, PageState, PhysicalId, RequestId, StorageKey};
}

pub mod backend_trait {
    pub use super::Element;
    pub use crate::engine::executor::{
        AttentionTopology, BackendError, BatchInput, GeneratorForwardConfig, KvCacheConfig,
        KvCacheHandle, LogitsHandle, SamplingConfig,
    };
    pub use crate::scheduler::types::{PageId, PageState, StorageKey};
    pub use gllm_kernels::quant::QuantType;

    /// Abstract compute backend.
    pub trait Backend<E: Element>: Send + Sync + 'static + std::fmt::Debug {
        type Tensor: std::fmt::Debug + Clone + Send + Sync + 'static + AsRef<[E]>;

        fn alloc_kv_cache(&self, config: &KvCacheConfig) -> Result<KvCacheHandle, BackendError>;

        fn batch_forward_gpu_pure(
            &self,
            input: &BatchInput,
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            kv_caches: &mut [KvCacheHandle],
            config: &GeneratorForwardConfig,
        ) -> Result<(Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>), BackendError>
        where
            Self: Sized;

        fn sample_from_tensor(
            &self,
            logits: &LogitsHandle,
            topology: &AttentionTopology,
            vocab_size: usize,
            sampling: &SamplingConfig,
        ) -> Result<Vec<u32>, BackendError>;

        fn embedding_forward_gpu_pure(
            &self,
            tokens: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        fn rerank_forward_gpu_pure(
            &self,
            tokens: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        fn classify_forward_gpu_pure(
            &self,
            tokens: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        /// Head Routing SDK — 对给定 tokens 跑一次 generator forward,读取最后
        /// 一个 token 的 hidden state 并与 `embed_tokens.weight` 的指定行做点积,
        /// 返回每个 `target_token_ids[i]` 对应的**原始 logit** (未经 softmax)。
        ///
        /// # 契约
        /// - `tokens.is_empty()` → `BackendError::Other("empty tokens")`
        /// - `target_token_ids.is_empty()` → `Ok(vec![])`
        /// - 模型必须是 decoder generator (tied embedding):否则 `Unimplemented`
        /// - `target_token_ids[i] >= vocab_size` → `Other(format!("..."))`
        /// - 不做 softmax、不做温度缩放(由调用方 client 层处理)
        ///
        /// # 关联
        /// - SPEC/HEAD-ROUTING.md §4.2
        /// - SPEC/04-API-DESIGN.md §3.8
        fn score_tokens_forward_gpu_pure(
            &self,
            tokens: &[u32],
            target_token_ids: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        fn get_memory_pressure(&self) -> Result<f32, BackendError>;

        fn swap_out_pages(
            &self,
            handle: &mut KvCacheHandle,
            mappings: &[(PageId, StorageKey)],
        ) -> Result<(), BackendError>;

        fn swap_in_pages(
            &self,
            handle: &mut KvCacheHandle,
            mappings: &[(PageId, StorageKey)],
        ) -> Result<(), BackendError>;

        fn get_page_states(
            &self,
            handle: &KvCacheHandle,
        ) -> Result<Vec<(PageId, PageState)>, BackendError>;


        fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BackendError>;

        /// Quantized matrix multiplication dispatching to K-Quant/Classic/IQ kernels.
        #[allow(clippy::too_many_arguments)]
        fn quantized_matmul(
            &self,
            _weight_blocks: &[u8],
            _input: &[E],
            _output: &mut [E],
            _quant_type: QuantType,
            _m: usize,
            _n: usize,
            _k: usize,
        ) -> Result<(), BackendError> {
            Err(BackendError::Unimplemented("quantized_matmul"))
        }

        /// Dequantize raw block bytes to f32 output.
        fn dequantize(
            &self,
            _block_data: &[u8],
            _output: &mut [f32],
            _quant_type: QuantType,
        ) -> Result<(), BackendError> {
            Err(BackendError::Unimplemented("dequantize"))
        }
    }

    /// Trait for looking up named tensors in a weight store.
    pub trait TensorLookup<E: Element, B: Backend<E> + ?Sized> {
        fn get_tensor(&self, name: &str) -> Option<&B::Tensor>;
        fn tensor_shape(&self, name: &str) -> Option<&[usize]>;

        /// Returns a quantized tensor by name (if stored in the quantized map).
        fn get_quantized(&self, name: &str) -> Option<&crate::loader::QuantizedTensor> {
            let _ = name;
            None
        }

        /// Returns all available tensor names (for diagnostics).
        fn available_names(&self) -> Vec<String> {
            Vec::new()
        }
    }
}

pub mod cpu_kernels {
    use super::Element;

    /// Marker trait for floating-point Element types.
    pub trait Float: Element {}

    impl Float for f32 {}
    impl Float for half::f16 {}
    impl Float for half::bf16 {}
}

// Root-level re-exports
pub use backend_trait::{Backend, BackendError};
pub use crate::loader::adapter::{DType, PackedBits};
pub use crate::loader::QuantizedTensor;

// Re-export submodule public items
pub use cpu_backend::CpuBackend;
pub use cuda_backend::{CudaBackend, GpuDeviceInfo};
pub use hip_backend::HipBackend;
pub use metal_backend::MetalBackend;

// ARCH-FULL-JIT + ARCH-CPU-GPU-UNIFIED migration:
// Former re-exports (compile_graph_to_ptx, cuda_compile_graph, cuda_launch_graph,
// compile_graph_to_hip, hip_compile_graph, hip_launch_graph, and their *KernelEntry
// companions) have been deleted. The corresponding functions used `gpu_ir` /
// `PtxDialect::with_dtype` / `fuse_with_dag_prebuilt(..., &DeviceProfile)` —
// none of which exist in current gllm-kernels. The replacement path is
// `gllm_kernels::compiler::InferenceCompiler::compile_graph` driving the
// `vm::plan_lower::compile_layer` → CodegenOutput pipeline, integrated via
// `FusedGraphExecutor::run_gpu_with_kv_cache` (not yet implemented).

/// Pooling mode for BERT-family encoder output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PoolingMode {
    /// Mean pooling over all tokens (for embedding models).
    MeanPool,
    /// [CLS] token + classifier head (for reranker/classification models).
    ClsClassifier,
}

// Knowledge injection API (旧 InjectionKind / LayerTarget 已废弃) 已经
// 整体被 src/semantic_gatekeeper/ 模块 (SPEC/SEMANTIC-GATEKEEPER.md) 替代。
// 任何残留引用应指向 crate::semantic_gatekeeper::* 而非此处。
