mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;
use std::sync::Arc;

fn build_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let manifest = registry::lookup(alias).expect("manifest");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut loader = files.loader(alias).expect("loader");
    Executor::from_loader(backend, Arc::new(manifest.clone()), adapter, &mut loader)
        .expect("executor")
}

/// 使用虚拟小模型进行快速单元测试
///
/// 这些测试使用 mock 权重，不下载真实模型，适合快速 CI 验证
#[test]
fn generator_matrix_covers_required_models() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["Qwen/Qwen3-0.6B", "meta-llama/Llama-4-8B-Instruct", "microsoft/Phi-4-mini-instruct"] {
        let mut executor = build_executor(alias, &files);
        let output = executor.generate("tok1 tok2", 1, 0.0).expect("generate");
        assert!(
            !output.trim().is_empty(),
            "generator output empty for {alias}"
        );
    }
}

#[test]
fn embedding_matrix_covers_required_models() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["Qwen/Qwen3-Embedding", "BAAI/bge-m4"] {
        let mut executor = build_executor(alias, &files);
        let embedding = executor.embed("tok3 tok4").expect("embed");
        assert_eq!(
            embedding.len(),
            4,
            "embedding dimension mismatch for {alias}"
        );
    }
}

#[test]
fn reranker_matrix_covers_required_models() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["Qwen/Qwen3-Reranker", "BAAI/bge-reranker-v3"] {
        let mut executor = build_executor(alias, &files);
        let scores = executor.rerank("tok5 tok6").expect("rerank");
        assert_eq!(scores.len(), 1, "rerank size mismatch for {alias}");
    }
}
