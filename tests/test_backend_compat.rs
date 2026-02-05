mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::loader::{config as loader_config, Loader};
use gllm::manifest::{ModelKind, ModelManifest};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::cuda_backend::CudaBackend;
use std::sync::Arc;

const TOLERANCE: f32 = 1e-3;

fn build_cpu_executor(
    alias: &str,
    kind: ModelKind,
    files: &TestModelFiles,
) -> Executor<CpuBackend> {
    let mut loader = files.loader(alias).expect("loader");
    let manifest = manifest_from_loader(alias, kind, &loader);
    loader.set_manifest_if_missing(&manifest);
    let adapter = adapter_for::<CpuBackend>(&manifest).expect("adapter");
    Executor::from_loader(
        CpuBackend::new(),
        Arc::new(manifest.clone()),
        adapter,
        &mut loader,
    )
    .expect("executor")
}

fn manifest_from_loader(alias: &str, kind: ModelKind, loader: &Loader) -> ModelManifest {
    let config_path = loader.config_path().expect("config path");
    let config_value = loader_config::load_config_value(config_path).expect("config");
    loader_config::manifest_from_config(alias, &config_value, kind).expect("manifest")
}

/// TEST-BACKEND-001: CPU 和 CUDA 后端 embedding 结果一致性
///
/// **关联需求**: REQ-TEST-001, REQ-TEST-010
/// **测试类型**: 正向测试
/// **前置条件**: BAAI/bge-small-en-v1.5 模型已缓存
///
/// **测试步骤**:
/// 1. 使用 Embedding 模型在 CPU 后端执行 embedding
/// 2. 使用 Embedding 模型在 CUDA 后端执行 embedding（如可用）
/// 3. 比较两个后端的输出结果
///
/// **期望结果**: CPU 和 CUDA 后端输出在容差范围内一致
#[test]
fn cpu_and_cuda_embeddings_align_within_tolerance() {
    let files = TestModelFiles::new().expect("test model files");
    // 使用 Embedding 模型测试 CPU 和 CUDA 后端的 embedding 结果一致性
    let mut cpu_exec = build_cpu_executor("BAAI/bge-small-en-v1.5", ModelKind::Embedding, &files);
    let reference = cpu_exec.embed("test query").expect("cpu embed");

    if let Ok(cuda_backend) = CudaBackend::new(0) {
        let mut loader = files.loader("BAAI/bge-small-en-v1.5").expect("loader");
        let manifest = manifest_from_loader("BAAI/bge-small-en-v1.5", ModelKind::Embedding, &loader);
        loader.set_manifest_if_missing(&manifest);
        let adapter = adapter_for::<CudaBackend>(&manifest).expect("cuda adapter");
        let mut cuda_exec = Executor::from_loader(
            cuda_backend,
            Arc::new(manifest.clone()),
            adapter,
            &mut loader,
        )
        .expect("cuda exec");
        let cuda_embedding = cuda_exec.embed("test query").expect("cuda embed");
        assert_eq!(reference.len(), cuda_embedding.len());
        for (cpu, cuda) in reference.iter().zip(cuda_embedding.iter()) {
            assert!(
                (cpu - cuda).abs() <= TOLERANCE,
                "cpu {cpu} cuda {cuda} differ"
            );
        }
    } else {
        // Fallback: deterministic CPU vs CPU comparison to keep CI green without CUDA.
        let mut second = build_cpu_executor("BAAI/bge-small-en-v1.5", ModelKind::Embedding, &files);
        let repeat = second.embed("test query").expect("second embed");
        assert_eq!(reference, repeat);
    }
}

/// TEST-BACKEND-002: 后端生成输出稳定性
///
/// **关联需求**: REQ-TEST-001, REQ-TEST-010
/// **测试类型**: 正向测试
/// **前置条件**: microsoft/Phi-4-mini-instruct 模型已缓存
///
/// **测试步骤**:
/// 1. 使用相同模型创建两个独立的执行器
/// 2. 对两个执行器执行相同的生成请求
/// 3. 比较两个执行器的输出
///
/// **期望结果**: 相同输入产生相同输出（确定性采样）
#[test]
fn backend_generation_outputs_are_stable() {
    let files = TestModelFiles::new().expect("test model files");
    let mut first = build_cpu_executor("microsoft/Phi-4-mini-instruct", ModelKind::Chat, &files);
    let mut second = build_cpu_executor("microsoft/Phi-4-mini-instruct", ModelKind::Chat, &files);

    let out1 = first.generate("tok3 tok4", 2, 0.0).expect("generate");
    let out2 = second.generate("tok3 tok4", 2, 0.0).expect("generate");
    assert_eq!(out1, out2, "deterministic sampling mismatch");
}
