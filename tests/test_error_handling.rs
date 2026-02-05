mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::backend::{fallback, BackendContextError};
use gllm::engine::executor::{Executor, ExecutorError};
use gllm::loader::{config as loader_config, Loader, LoaderError};
use gllm::manifest::{ModelKind, ModelManifest};
use gllm_kernels::backend_trait::BackendError;
use gllm_kernels::cpu_backend::CpuBackend;
use std::sync::Arc;
use tempfile::TempDir;

fn build_executor(alias: &str, kind: ModelKind, files: &TestModelFiles) -> Executor<CpuBackend> {
    let mut loader = files.loader(alias).expect("loader");
    let manifest = manifest_from_loader(alias, kind, &loader);
    loader.set_manifest_if_missing(&manifest);
    let adapter = adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    Executor::from_loader(backend, Arc::new(manifest.clone()), adapter, &mut loader)
        .expect("executor")
}

fn manifest_from_loader(alias: &str, kind: ModelKind, loader: &Loader) -> ModelManifest {
    let config_path = loader.config_path().expect("config path");
    let config_value = loader_config::load_config_value(config_path).expect("config");
    loader_config::manifest_from_config(alias, &config_value, kind).expect("manifest")
}

#[test]
fn empty_prompt_returns_error() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("Qwen/Qwen3-0.6B", ModelKind::Chat, &files);
    let err = executor.generate("", 1, 0.8).unwrap_err();
    assert!(matches!(err, ExecutorError::EmptyPrompt));
}

#[test]
fn corrupted_weights_surface_loader_errors() {
    let dir = TempDir::new().expect("temp dir");
    let bad_weights = dir.path().join("broken.safetensors");
    std::fs::write(&bad_weights, b"not a tensor").expect("write broken weights");

    let mut loader =
        Loader::from_local_files("Qwen/Qwen3-0.6B", vec![bad_weights], vec![]).expect("loader");
    let backend = CpuBackend::new();
    let err = match loader.upload_weights(&backend) {
        Ok(_) => panic!("expected loader error"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            LoaderError::SafeTensors(_)
                | LoaderError::InvalidQuantization(_)
                | LoaderError::MissingTensor(_)
        ),
        "unexpected loader error: {err:?}"
    );
}

#[test]
fn oom_errors_are_detected_for_fallback() {
    let err =
        BackendContextError::Loader(LoaderError::Backend("CUDA_ERROR_OUT_OF_MEMORY".to_string()));
    assert!(fallback::is_oom_context_error(&err));

    let backend_err = BackendContextError::Executor(ExecutorError::Backend(BackendError::Cuda(
        "device out of memory".into(),
    )));
    assert!(fallback::is_oom_context_error(&backend_err));
}
