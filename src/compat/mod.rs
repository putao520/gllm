//! Compatibility shim for gllm_kernels phantom types.
//!
//! This module provides:
//! - `Element` re-export from gllm-kernels
//! - `Backend<E>` trait + `TensorLookup` trait
//! - `CpuBackend<E>` / `CudaBackend<E>` stub implementations
//! - Backward-compatible re-export modules (`kernel_types`, `backend_trait`, `cpu_backend`)

mod weight_helpers;
mod bert_forward;
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
mod decoder_forward;
pub(crate) mod jit_cache;
pub(crate) mod embed_cache;
mod gpu_compile;
#[allow(dead_code)]
pub(crate) mod types;
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub(crate) mod scalar_ops;
#[allow(clippy::too_many_arguments)]
pub(crate) mod jit_helpers;
pub(crate) mod graph_builders;
#[allow(dead_code)]
pub(crate) mod gpu_helpers;
pub mod cpu_backend;
mod cuda_backend;
mod hip_backend;
mod metal_backend;
mod memory;

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

// Re-export pub(crate) GPU compilation helpers
#[cfg(feature = "cuda")]
pub(crate) use gpu_compile::{
    compile_graph_to_ptx, cuda_compile_graph, cuda_launch_graph, GpuKernelEntry,
};
#[cfg(feature = "hip")]
pub(crate) use gpu_compile::{
    compile_graph_to_hip, hip_compile_graph, hip_launch_graph, HipKernelEntry,
};

/// Pooling mode for BERT-family encoder output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PoolingMode {
    /// Mean pooling over all tokens (for embedding models).
    MeanPool,
    /// [CLS] token + classifier head (for reranker/classification models).
    ClsClassifier,
}
