use crate::compat::backend_trait::Element;
use crate::compat::{CpuBackend, CudaBackend, HipBackend, MetalBackend};
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

use crate::engine::executor::{Executor, ExecutorError};
use crate::loader::{Loader, LoaderError};
use crate::manifest::{ModelArchitecture, ModelManifest};

pub mod detection;

pub use detection::{
    detect_backend, detect_backend_generic, BackendType, DetectedBackend, DetectedBackendF32,
    DetectedDtype,
};

#[derive(Debug, Error)]
pub enum BackendContextError {
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    Backend(#[from] crate::engine::executor::BackendError),
}

/// Generic backend executor supporting any element type.
///
/// Both CUDA and CPU backends now support full generic execution.
/// CPU backend uses pseudo-SIMD (f32 promotion) for f16/bf16 types.
pub enum BackendExecutor<E: Element = f32> {
    Cuda(Box<Executor<CudaBackend<E>, E>>),
    Rocm(Box<Executor<HipBackend<E>, E>>),
    Metal(Box<Executor<MetalBackend<E>, E>>),
    Cpu(Box<Executor<CpuBackend<E>, E>>),
}

/// Dynamic backend executor that selects precision at runtime based on model dtype.
///
/// This is the recommended entry point for loading models, as it automatically
/// detects the weight dtype and uses the optimal precision for computation.
///
/// # Design Rationale
///
/// LLM inference should use the model's native precision:
/// - F16 models → F16 computation (half memory, tensor core acceleration)
/// - BF16 models → BF16 computation (better range than F16)
/// - F32 models → F32 computation (baseline)
///
/// This enum provides runtime dispatch to the correct precision executor.
pub enum DynBackendExecutor {
    F32(BackendExecutor<f32>),
    F16(BackendExecutor<half::f16>),
    BF16(BackendExecutor<half::bf16>),
}

impl DynBackendExecutor {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DynBackendExecutor::F32(e) => e.backend_type(),
            DynBackendExecutor::F16(e) => e.backend_type(),
            DynBackendExecutor::BF16(e) => e.backend_type(),
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.is_cuda(),
            DynBackendExecutor::F16(e) => e.is_cuda(),
            DynBackendExecutor::BF16(e) => e.is_cuda(),
        }
    }

    pub fn is_gpu(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.is_gpu(),
            DynBackendExecutor::F16(e) => e.is_gpu(),
            DynBackendExecutor::BF16(e) => e.is_gpu(),
        }
    }

    pub fn thinking_head_available(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.thinking_head_available(),
            DynBackendExecutor::F16(e) => e.thinking_head_available(),
            DynBackendExecutor::BF16(e) => e.thinking_head_available(),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.generate(prompt, max_tokens, temperature, top_k, top_p),
            DynBackendExecutor::F16(e) => e.generate(prompt, max_tokens, temperature, top_k, top_p),
            DynBackendExecutor::BF16(e) => {
                e.generate(prompt, max_tokens, temperature, top_k, top_p)
            }
        }
    }

    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: u64,
    ) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
            DynBackendExecutor::F16(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
            DynBackendExecutor::BF16(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
        }
    }

    /// Embedding output is always f32 (API standardization)
    pub fn embed(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.embed(input),
            DynBackendExecutor::F16(e) => e.embed(input),
            DynBackendExecutor::BF16(e) => e.embed(input),
        }
    }

    /// Rerank scores are always f32 (API standardization)
    pub fn rerank(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.rerank(input),
            DynBackendExecutor::F16(e) => e.rerank(input),
            DynBackendExecutor::BF16(e) => e.rerank(input),
        }
    }

    /// Rerank with proper pair encoding (query + document as separate segments)
    pub fn rerank_pair(&mut self, query: &str, document: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.rerank_pair(query, document),
            DynBackendExecutor::F16(e) => e.rerank_pair(query, document),
            DynBackendExecutor::BF16(e) => e.rerank_pair(query, document),
        }
    }

    /// Returns the element type name for debugging/logging
    pub fn dtype_name(&self) -> &'static str {
        match self {
            DynBackendExecutor::F32(_) => "f32",
            DynBackendExecutor::F16(_) => "f16",
            DynBackendExecutor::BF16(_) => "bf16",
        }
    }
}

impl<E: Element> BackendExecutor<E> {
    pub fn backend_type(&self) -> BackendType {
        match self {
            BackendExecutor::Cuda(_) => BackendType::Cuda,
            BackendExecutor::Rocm(_) => BackendType::Rocm,
            BackendExecutor::Metal(_) => BackendType::Metal,
            BackendExecutor::Cpu(_) => BackendType::Cpu,
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, BackendExecutor::Cuda(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, BackendExecutor::Cuda(_) | BackendExecutor::Rocm(_) | BackendExecutor::Metal(_))
    }

    pub fn thinking_head_available(&self) -> bool {
        match self {
            BackendExecutor::Cuda(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Rocm(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Metal(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Cpu(exec) => exec.weights().thinking_head.is_some(),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p)
            }
            BackendExecutor::Rocm(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p)
            }
            BackendExecutor::Metal(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p)
            }
            BackendExecutor::Cpu(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p)
            }
        }
    }

    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: u64,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
            BackendExecutor::Rocm(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
            BackendExecutor::Metal(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
            BackendExecutor::Cpu(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id)
            }
        }
    }

    pub fn embed(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.embed(input),
            BackendExecutor::Rocm(exec) => exec.embed(input),
            BackendExecutor::Metal(exec) => exec.embed(input),
            BackendExecutor::Cpu(exec) => exec.embed(input),
        }
    }

    pub fn rerank(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.rerank(input),
            BackendExecutor::Rocm(exec) => exec.rerank(input),
            BackendExecutor::Metal(exec) => exec.rerank(input),
            BackendExecutor::Cpu(exec) => exec.rerank(input),
        }
    }

    pub fn rerank_pair(&mut self, query: &str, document: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Rocm(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Metal(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Cpu(exec) => exec.rerank_pair(query, document),
        }
    }
}

/// Backward-compatible type alias for f32 backend executor.
pub type BackendExecutorF32 = BackendExecutor<f32>;

pub struct BackendContext {
    model_ref: String,
    manifest: Arc<ModelManifest>,
    weight_paths: Vec<std::path::PathBuf>,
    config_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    executor: Mutex<BackendExecutor<f32>>,
}

impl BackendContext {
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        backend: DetectedBackend<f32>,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();
        // ARCH-ZERO-FALLBACK: No OOM fallback. Executor build failure propagates directly.
        let executor = build_executor(
            backend,
            manifest.clone(),
            &model_ref,
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;
        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn executor(&self) -> MutexGuard<'_, BackendExecutor<f32>> {
        self.executor.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn executor_mut(&self) -> MutexGuard<'_, BackendExecutor<f32>> {
        self.executor()
    }

}

fn build_executor(
    backend: DetectedBackend<f32>,
    manifest: Arc<ModelManifest>,
    _model_ref: &str,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor<f32>, BackendContextError> {
    match backend {
        DetectedBackend::Cuda(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Rocm(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Rocm(Box::new(executor)))
        }
        DetectedBackend::Metal(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Metal(Box::new(executor)))
        }
        DetectedBackend::Cpu(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}

// ============================================================================
// Dynamic Precision Context - Runtime dtype selection
// ============================================================================

/// Dynamic backend context that selects precision at runtime based on model dtype.
///
/// Unlike `BackendContext` which is fixed to f32, this context automatically
/// detects the model's weight dtype and uses the optimal precision.
#[allow(dead_code)] // Fields reserved for future rebuild functionality
pub struct DynBackendContext {
    model_ref: String,
    manifest: Arc<ModelManifest>,
    weight_paths: Vec<std::path::PathBuf>,
    config_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    executor: Mutex<DynBackendExecutor>,
}

impl DynBackendContext {
    /// Create a new dynamic context, auto-detecting the model's dtype.
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();

        // Detect dtype from weights
        let detected_dtype = Self::detect_dtype_from_paths(&weight_paths)?;

        let executor = Self::build_dyn_executor(
            detected_dtype,
            manifest.clone(),
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;

        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    /// Create with explicit dtype (skip auto-detection)
    pub fn with_dtype(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        dtype: DetectedDtype,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();

        let executor = Self::build_dyn_executor(
            dtype,
            manifest.clone(),
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;

        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn executor(&self) -> MutexGuard<'_, DynBackendExecutor> {
        self.executor.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn executor_mut(&self) -> MutexGuard<'_, DynBackendExecutor> {
        self.executor()
    }

    /// Returns the detected dtype
    pub fn dtype(&self) -> &'static str {
        self.executor().dtype_name()
    }

    fn detect_dtype_from_paths(
        weight_paths: &[std::path::PathBuf],
    ) -> Result<DetectedDtype, BackendContextError> {
        // Try to create a temporary loader to detect dtype
        let loader = Loader::from_env()?.with_weights(weight_paths.to_vec());

        if let Ok(Some(size)) = loader.detect_weight_dtype_size() {
            if let Some(dtype) = DetectedDtype::from_size(size) {
                return Ok(dtype);
            }
        }

        // Default to f32 if detection fails
        Ok(DetectedDtype::F32)
    }

    fn build_dyn_executor(
        dtype: DetectedDtype,
        manifest: Arc<ModelManifest>,
        weight_paths: &[std::path::PathBuf],
        config_path: Option<&std::path::Path>,
        tokenizer_path: Option<&std::path::Path>,
    ) -> Result<DynBackendExecutor, BackendContextError> {
        match dtype {
            DetectedDtype::F32 => {
                let executor = build_executor_generic::<f32>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::F32(executor))
            }
            DetectedDtype::F16 => {
                let executor = build_executor_generic::<half::f16>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::F16(executor))
            }
            DetectedDtype::BF16 => {
                let executor = build_executor_generic::<half::bf16>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::BF16(executor))
            }
        }
    }
}

/// Build executor for any element type
fn build_executor_generic<E: Element>(
    manifest: Arc<ModelManifest>,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor<E>, BackendContextError> {
    let backend = detection::detect_backend_generic::<E>()?;

    match backend {
        DetectedBackend::Cuda(cuda_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*cuda_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Rocm(rocm_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*rocm_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Rocm(Box::new(executor)))
        }
        DetectedBackend::Metal(metal_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*metal_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Metal(Box::new(executor)))
        }
        DetectedBackend::Cpu(cpu_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*cpu_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}
