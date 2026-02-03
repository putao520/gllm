mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;

fn build_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let manifest = registry::lookup(alias).expect("manifest");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut loader = files.loader(alias).expect("loader");
    Executor::from_loader(backend, manifest, adapter, &mut loader).expect("executor")
}

#[test]
fn generator_matrix_covers_required_models() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["qwen3-7b", "llama-4-8b", "phi-4-mini"] {
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
    for alias in ["qwen3-embed", "bge-m4"] {
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
    for alias in ["qwen3-rerank", "bge-rerank-v3"] {
        let mut executor = build_executor(alias, &files);
        let scores = executor.rerank("tok5 tok6").expect("rerank");
        assert_eq!(scores.len(), 1, "rerank size mismatch for {alias}");
    }
}
