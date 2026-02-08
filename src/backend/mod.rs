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
    weight_paths: Vec<std::path::PathBuf>,
    config_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    executor: Mutex<BackendExecutor>,
}

impl BackendContext {
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        backend: DetectedBackend,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();
        let backend_type = backend.backend_type();
        let executor = match build_executor(
            backend,
            manifest.clone(),
            &model_ref,
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        ) {
            Ok(executor) => executor,
            Err(err) => {
                if backend_type == BackendType::Cuda && fallback::is_oom_context_error(&err) {
                    build_cpu_executor(
                        manifest.clone(),
                        &model_ref,
                        &weight_paths,
                        config_path.as_deref(),
                        tokenizer_path.as_deref(),
                    )?
                } else {
                    return Err(err);
                }
            }
        };
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
        let cpu_executor = build_cpu_executor(
            self.manifest.clone(),
            &self.model_ref,
            &self.weight_paths,
            self.config_path.as_deref(),
            self.tokenizer_path.as_deref(),
        )?;
        *executor = cpu_executor;
        Ok(())
    }
}

fn build_executor(
    backend: DetectedBackend,
    manifest: Arc<ModelManifest>,
    model_ref: &str,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor, BackendContextError> {
    match backend {
        DetectedBackend::Cuda(backend) => {
            let adapter = adapter_for::<CudaBackend>(manifest.as_ref())
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Cpu(backend) => {
            let adapter = adapter_for::<CpuBackend>(manifest.as_ref())
                .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*backend, manifest, adapter, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}

fn build_cpu_executor(
    manifest: Arc<ModelManifest>,
    _model_ref: &str,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor, BackendContextError> {
    let adapter = adapter_for::<CpuBackend>(manifest.as_ref())
        .ok_or(BackendContextError::UnsupportedArchitecture(manifest.arch))?;
    let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
        .with_weights(weight_paths.to_vec());
    if let Some(path) = config_path {
        loader = loader.with_config(path.to_path_buf());
    }
    if let Some(path) = tokenizer_path {
        loader = loader.with_tokenizer(path.to_path_buf());
    }
    let executor = Executor::from_loader(CpuBackend::new(), manifest, adapter, &mut loader)?;
    Ok(BackendExecutor::Cpu(Box::new(executor)))
}
