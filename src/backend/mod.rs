use gllm_kernels::{CpuBackend, CudaBackend};
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

use crate::adapter::adapter_for;
use crate::engine::executor::{Executor, ExecutorError};
use crate::loader::{Loader, LoaderError};
use crate::manifest::{ModelArchitecture, ModelManifest};

pub mod detection;
pub mod fallback;

pub use detection::{detect_backend, BackendType, DetectedBackend};
pub use fallback::{FallbackEmbedder, FallbackGenerator, FallbackReranker};

#[derive(Debug, Error)]
pub enum BackendContextError {
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}

pub enum BackendExecutor {
    Cuda(Box<Executor<CudaBackend>>),
    Cpu(Box<Executor<CpuBackend>>),
}

impl BackendExecutor {
    pub fn backend_type(&self) -> BackendType {
        match self {
            BackendExecutor::Cuda(_) => BackendType::Cuda,
            BackendExecutor::Cpu(_) => BackendType::Cpu,
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, BackendExecutor::Cuda(_))
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
            BackendExecutor::Cpu(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p)
            }
        }
    }

    pub fn thinking_head_available(&self) -> bool {
        match self {
            BackendExecutor::Cuda(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Cpu(exec) => exec.weights().thinking_head.is_some(),
        }
    }

    pub fn embed(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.embed(input),
            BackendExecutor::Cpu(exec) => exec.embed(input),
        }
    }

    pub fn rerank(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.rerank(input),
            BackendExecutor::Cpu(exec) => exec.rerank(input),
        }
    }
}

pub struct BackendContext {
    model_ref: String,
    manifest: Arc<ModelManifest>,
    executor: Mutex<BackendExecutor>,
}

impl BackendContext {
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        backend: DetectedBackend,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();
        let backend_type = backend.backend_type();
        let executor = match build_executor(backend, manifest.clone(), &model_ref) {
            Ok(executor) => executor,
            Err(err) => {
                if backend_type == BackendType::Cuda && fallback::is_oom_context_error(&err) {
                    build_cpu_executor(manifest.clone(), &model_ref)?
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            model_ref,
            manifest,
            executor: Mutex::new(executor),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn executor(&self) -> MutexGuard<'_, BackendExecutor> {
        self.executor.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn executor_mut(&self) -> MutexGuard<'_, BackendExecutor> {
        self.executor()
    }

    pub fn rebuild_cpu(&self) -> Result<(), BackendContextError> {
        let mut executor = self.executor_mut();
        if matches!(*executor, BackendExecutor::Cpu(_)) {
            return Ok(());
        }
        let cpu_executor = build_cpu_executor(self.manifest.clone(), &self.model_ref)?;
        *executor = cpu_executor;
        Ok(())
    }
}

fn build_executor(
    backend: DetectedBackend,
    manifest: Arc<ModelManifest>,
    model_ref: &str,
) -> Result<BackendExecutor, BackendContextError> {
    match backend {
        DetectedBackend::Cuda(backend) => {
            let adapter = adapter_for::<CudaBackend>(manifest.as_ref())
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env_with_manifest(model_ref, Some(manifest.as_ref()))?;
            let executor = Executor::from_loader(*backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Cpu(backend) => {
            let adapter = adapter_for::<CpuBackend>(manifest.as_ref())
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env_with_manifest(model_ref, Some(manifest.as_ref()))?;
            let executor = Executor::from_loader(*backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}

fn build_cpu_executor(
    manifest: Arc<ModelManifest>,
    model_ref: &str,
) -> Result<BackendExecutor, BackendContextError> {
    let adapter = adapter_for::<CpuBackend>(manifest.as_ref())
        .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
    let mut loader = Loader::from_env_with_manifest(model_ref, Some(manifest.as_ref()))?;
    let executor = Executor::from_loader(CpuBackend::new(), manifest, adapter, &mut loader)?;
    Ok(BackendExecutor::Cpu(Box::new(executor)))
}
