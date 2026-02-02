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
fn e2e_generator_models_load_and_generate() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["qwen3-7b", "qwen3-moe", "gpt-oss-1.5b"] {
        let mut executor = build_executor(alias, &files);
        let output = executor.generate("tok1 tok2", 2, 0.7).expect("generate");
        assert!(!output.trim().is_empty(), "empty output for {alias}");
    }
}

#[test]
fn e2e_embedding_models_embed() {
    let files = TestModelFiles::new().expect("test model files");
    for alias in ["qwen3-embed", "bge-m3"] {
        let mut executor = build_executor(alias, &files);
        let embedding = executor.embed("tok3 tok4").expect("embed");
        assert_eq!(embedding.len(), 4, "embedding size mismatch for {alias}");
    }
}

#[test]
fn e2e_rerank_models_score() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("qwen3-rerank", &files);
    let scores = executor.rerank("tok5 tok6").expect("rerank");
    assert_eq!(scores.len(), 1, "rerank output size mismatch");
}
