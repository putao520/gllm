//! Compatibility shim for gllm_kernels phantom types.
//!
//! This module provides:
//! - `Element` re-export from gllm-kernels
//! - `Backend<E>` trait + `TensorLookup` trait
//! - `CpuBackend<E>` / `CudaBackend<E>` stub implementations
//! - Backward-compatible re-export modules (`kernel_types`, `backend_trait`, `cpu_backend`)

pub(crate) mod weight_helpers;
pub(crate) mod sampling;

mod gpu_compile;
#[allow(dead_code)]
pub(crate) mod types;
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub(crate) mod scalar_ops;
#[allow(clippy::too_many_arguments)]
pub(crate) mod jit_helpers;
#[allow(dead_code)]
pub(crate) mod gpu_helpers;
#[macro_use]
mod gpu_backend_macro;
pub mod cpu_backend;
mod cuda_backend;
mod hip_backend;
mod metal_backend;
pub mod memory;
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

    /// Weight data placement — determines where uploaded tensor data physically resides.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WeightPlacement {
        /// Device-local memory (GPU VRAM / NPU SRAM).
        /// Accessible by device compute units at highest bandwidth.
        DeviceLocal,
        /// Host memory (CPU RAM). Always accessible from CPU;
        /// GPU access requires PCIe transfer (HtoD/DtoH).
        HostLocal,
    }

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

        /// Intent / HR `encode_to_layer` — run the JIT forward with a
        /// `MidLayerEncodeCallback` attached through
        /// `config.callback_chain`, truncating at `anchor_layer`. Returns
        /// the flattened `[seq_len, hidden_size]` hidden state as f32.
        ///
        /// # Contract
        /// - `tokens.is_empty()` → `BackendError::Other("empty tokens")`
        /// - Model must be a decoder generator: encoder path raises
        ///   `Unimplemented` to avoid silent fallbacks.
        /// - `config.callback_chain` must be non-null and point at a
        ///   `CallbackChain` containing a `MidLayerEncodeCallback`.
        ///
        /// # 关联
        /// - SPEC/HEAD-ROUTING.md §5 mid-layer encode 协议
        /// - SPEC/INTENT.md §2 architecture
        fn encode_at_layer_forward_gpu_pure(
            &self,
            tokens: &[u32],
            anchor_layer: usize,
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        /// Guardrail SDK — run a full generator forward (same as
        /// `batch_forward_gpu_pure` but single-sequence, no KV cache, no
        /// sampling) with `config.callback_chain` driving Guardrail Probe
        /// callbacks. Returns final hidden state `[seq_len, hidden_size]` as
        /// f32 or `Ok(vec![])` when the callback chain raised `ExitEarly`
        /// (veto) before completing.
        ///
        /// # 关联
        /// - SPEC/GUARDRAIL.md §3 Client API
        fn apply_guardrail_probe(
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

        /// Upload weights from a pre-converted f32 Vec, avoiding an extra copy.
        /// Only called when E == f32. CPU backend overrides to take ownership directly.
        fn upload_weights_f32_owned(&self, _data: Vec<f32>) -> Result<Self::Tensor, BackendError> {
            Err(BackendError::Unimplemented("upload_weights_f32_owned: backend does not support zero-copy f32 upload"))
        }

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

        /// Placement-aware weight upload.
        ///
        /// `data` is an f32 weight vector ready for inference.
        /// `placement` determines where the tensor data should reside:
        /// - `DeviceLocal`: upload to device memory (GPU VRAM / NPU SRAM)
        /// - `HostLocal`: keep in host memory (CPU RAM)
        ///
        /// Returns `(Tensor, actual_placement)` where `actual_placement` may differ
        /// from the requested placement (e.g., CPU backend ignores `DeviceLocal`).
        fn upload_weights_with_placement(
            &self,
            data: Vec<f32>,
            placement: WeightPlacement,
        ) -> Result<(Self::Tensor, WeightPlacement), BackendError> {
            let _ = placement;
            let tensor = self.upload_weights_f32_owned(data)?;
            Ok((tensor, WeightPlacement::HostLocal))
        }

        /// Total device memory capacity in bytes (GPU VRAM / NPU SRAM).
        /// Returns 0 for CPU-only backends.
        fn device_memory_capacity(&self) -> usize {
            0
        }

        /// Currently used device memory in bytes.
        /// Returns 0 for CPU-only backends.
        fn device_memory_used(&self) -> usize {
            0
        }

        /// GPU compute capability version (e.g. 80 for sm_80, 90 for sm_90).
        /// Returns 0 for non-GPU backends.
        /// Used by mega-kernel compilation to generate correct PTX/HIP code.
        fn gpu_sm_version(&self) -> u32 {
            0
        }

        /// Store GPU mega-kernel artifacts (PTX/HIP/MSL code + weight blob)
        /// in the backend for subsequent GPU launches.
        ///
        /// Called once during model loading after `MegaKernelExecutor` compilation.
        /// CPU backend: no-op. GPU backend: uploads weight blob to device,
        /// stores PTX in cache.
        ///
        /// `decoder_gpu_code`: mega-kernel GPU code for decoder path (21-param ABI).
        /// `forward_gpu_code`: forward-only GPU code for encoder path (10-param ABI).
        /// `scratchpad_bytes`: scratchpad size needed by the mega-kernel.
        fn prepare_gpu_mega_kernel(
            &self,
            weight_blob: &[u8],
            decoder_gpu_code: Option<&[u8]>,
            forward_gpu_code: Option<&[u8]>,
            scratchpad_bytes: usize,
        ) -> Result<(), BackendError> {
            let _ = (weight_blob, decoder_gpu_code, forward_gpu_code, scratchpad_bytes);
            Ok(())
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
// mega-kernel GPU launch (not yet implemented).

/// Pooling mode for BERT-family encoder output.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub(crate) enum PoolingMode {
    /// Mean pooling over all tokens (for embedding models).
    MeanPool,
    /// [CLS] token + classifier head (for reranker/classification models).
    ClsClassifier,
}

// Knowledge injection API (旧 InjectionKind / LayerTarget 已废弃) 已经
// 整体被 src/semantic_gatekeeper/ 模块 (SPEC/SEMANTIC-GATEKEEPER.md) 替代。
// 任何残留引用应指向 crate::semantic_gatekeeper::* 而非此处。
