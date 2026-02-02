mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::backend::{fallback, BackendContextError};
use gllm::engine::executor::{Executor, ExecutorError};
use gllm::registry;
use gllm::loader::{Loader, LoaderError};
use gllm_kernels::backend_trait::BackendError;
use gllm_kernels::cpu_backend::CpuBackend;
use tempfile::TempDir;

fn build_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let manifest = registry::lookup(alias).expect("manifest");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut loader = files.loader(alias).expect("loader");
    Executor::from_loader(backend, manifest, adapter, &mut loader).expect("executor")
}

#[test]
fn empty_prompt_returns_error() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("qwen3-7b", &files);
    let err = executor.generate("", 1, 0.8).unwrap_err();
    assert!(matches!(err, ExecutorError::EmptyPrompt));
}

#[test]
fn corrupted_weights_surface_loader_errors() {
    let dir = TempDir::new().expect("temp dir");
    let bad_weights = dir.path().join("broken.safetensors");
    std::fs::write(&bad_weights, b"not a tensor").expect("write broken weights");

    let mut loader =
        Loader::from_local_files("qwen3-7b", vec![bad_weights], vec![]).expect("loader");
    let backend = CpuBackend::new();
    let err = match loader.upload_weights(&backend) {
        Ok(_) => panic!("expected loader error"),
        Err(err) => err,
    };
    assert!(
        matches!(
        err,
        LoaderError::SafeTensors(_) | LoaderError::InvalidQuantization(_) | LoaderError::MissingTensor(_)
    ),
        "unexpected loader error: {err:?}"
    );
}

#[test]
fn unsupported_architecture_is_flagged() {
    assert!(
        registry::lookup("unknown-model").is_none(),
        "unknown aliases should be rejected early"
    );
}

#[test]
fn oom_errors_are_detected_for_fallback() {
    let err = BackendContextError::Loader(LoaderError::Backend(
        "CUDA_ERROR_OUT_OF_MEMORY".to_string(),
    ));
    assert!(fallback::is_oom_context_error(&err));

    let backend_err = BackendContextError::Executor(ExecutorError::Backend(
        BackendError::Cuda("device out of memory".into()),
    ));
    assert!(fallback::is_oom_context_error(&backend_err));
}
