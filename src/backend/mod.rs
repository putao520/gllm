use gllm_kernels::{CpuBackend, CudaBackend};
use thiserror::Error;

use crate::adapter::{adapter_for, Message};
use crate::engine::executor::{Executor, ExecutorError};
use crate::loader::{Loader, LoaderError};
use crate::manifest::{ModelArchitecture, ModelManifest};

pub mod detection;
pub mod fallback;

pub use detection::{detect_backend, BackendType, DetectedBackend};
pub use fallback::{FallbackEmbedder, FallbackGenerator};

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
    Cuda(Executor<CudaBackend>),
    Cpu(Executor<CpuBackend>),
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

    pub fn apply_chat_template(&self, messages: &[Message]) -> String {
        match self {
            BackendExecutor::Cuda(exec) => exec.apply_chat_template(messages),
            BackendExecutor::Cpu(exec) => exec.apply_chat_template(messages),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.generate(prompt, max_tokens, temperature),
            BackendExecutor::Cpu(exec) => exec.generate(prompt, max_tokens, temperature),
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
    manifest: &'static ModelManifest,
    executor: BackendExecutor,
}

impl BackendContext {
    pub fn new(
        model_ref: impl Into<String>,
        manifest: &'static ModelManifest,
        backend: DetectedBackend,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();
        let backend_type = backend.backend_type();
        let executor = match build_executor(backend, manifest, &model_ref) {
            Ok(executor) => executor,
            Err(err) => {
                if backend_type == BackendType::Cuda
                    && fallback::is_oom_context_error(&err)
                {
                    build_cpu_executor(manifest, &model_ref)?
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            model_ref,
            manifest,
            executor,
        })
    }

    pub fn manifest(&self) -> &'static ModelManifest {
        self.manifest
    }

    pub fn executor(&self) -> &BackendExecutor {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut BackendExecutor {
        &mut self.executor
    }

    pub fn rebuild_cpu(&mut self) -> Result<(), BackendContextError> {
        if matches!(self.executor, BackendExecutor::Cpu(_)) {
            return Ok(());
        }
        let executor = build_cpu_executor(self.manifest, &self.model_ref)?;
        self.executor = executor;
        Ok(())
    }
}

fn build_executor(
    backend: DetectedBackend,
    manifest: &'static ModelManifest,
    model_ref: &str,
) -> Result<BackendExecutor, BackendContextError> {
    match backend {
        DetectedBackend::Cuda(backend) => {
            let adapter = adapter_for::<CudaBackend>(manifest)
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env(model_ref)?;
            let executor = Executor::from_loader(backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cuda(executor))
        }
        DetectedBackend::Cpu(backend) => {
            let adapter = adapter_for::<CpuBackend>(manifest)
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env(model_ref)?;
            let executor = Executor::from_loader(backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cpu(executor))
        }
    }
}

fn build_cpu_executor(
    manifest: &'static ModelManifest,
    model_ref: &str,
) -> Result<BackendExecutor, BackendContextError> {
    let adapter = adapter_for::<CpuBackend>(manifest)
        .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
    let mut loader = Loader::from_env(model_ref)?;
    let executor = Executor::from_loader(CpuBackend::new(), manifest, adapter, &mut loader)?;
    Ok(BackendExecutor::Cpu(executor))
}
