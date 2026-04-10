mod common;

use common::TestModelFiles;
use gllm::engine::executor::{Executor, ExecutorError};
use gllm::loader::{Loader, LoaderError};
use gllm::manifest::{
    map_architecture_token, ModelKind, ModelManifest,
    EMPTY_FILE_MAP,
};
use gllm::compat::CpuBackend;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

fn build_executor(
    alias: &str,
    kind: ModelKind,
    files: &TestModelFiles,
) -> Executor<CpuBackend<f32>, f32> {
    let mut loader = files.loader(alias).expect("loader");
    let manifest = manifest_from_loader(alias, kind, &loader);
    loader.set_manifest_if_missing(&manifest);
    let backend = CpuBackend::<f32>::new();
    Executor::from_loader(backend, Arc::new(manifest.clone()), &mut loader).expect("executor")
}

fn manifest_from_loader(alias: &str, kind: ModelKind, loader: &Loader) -> ModelManifest {
    // Ω1: Tensor-driven architecture detection
    let arch = loader
        .gguf_architecture()
        .ok()
        .and_then(map_architecture_token)
        .unwrap_or_else(|| "qwen3".to_string());

    ModelManifest {
        model_id: Cow::Owned(alias.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: HashMap::new(),
    }
}

/// TEST-ERROR-001: 空输入返回错误
///
/// **关联需求**: REQ-TEST-007
/// **测试类型**: 负向测试
///
/// **测试步骤**:
/// 1. 使用空字符串调用 generate()
///
/// **期望结果**: 返回 EmptyPrompt 错误
#[test]
fn empty_prompt_returns_error() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("Qwen/Qwen3-0.6B", ModelKind::Chat, &files);
    let err = executor.generate("", 1, 0.8).unwrap_err();
    assert!(matches!(err, ExecutorError::EmptyPrompt));
}

/// TEST-ERROR-002: 损坏权重上报加载错误
///
/// **关联需求**: REQ-TEST-007
/// **测试类型**: 负向测试
///
/// **测试步骤**:
/// 1. 创建损坏的权重文件
/// 2. 尝试加载权重
///
/// **期望结果**: 返回 SafeTensors/InvalidQuantization/MissingTensor 错误
#[test]
fn corrupted_weights_surface_loader_errors() {
    let dir = TempDir::new().expect("temp dir");
    let bad_weights = dir.path().join("broken.safetensors");
    std::fs::write(&bad_weights, b"not a tensor").expect("write broken weights");

    let mut loader =
        Loader::from_local_files_with_manifest("Qwen/Qwen3-0.6B", vec![bad_weights], vec![], None)
            .expect("loader");
    let backend = CpuBackend::<f32>::new();
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

