mod common;

use std::time::Instant;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::loader::{config as loader_config, Loader};
use gllm::manifest::ModelManifest;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::Backend;
use std::sync::Arc;

fn build_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let mut loader = files.loader(alias).expect("loader");
    let manifest = manifest_from_loader(alias, &loader);
    loader.set_manifest_if_missing(&manifest);
    let adapter = adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    Executor::from_loader(backend, Arc::new(manifest.clone()), adapter, &mut loader)
        .expect("executor")
}

fn manifest_from_loader(alias: &str, loader: &Loader) -> ModelManifest {
    let overrides = registry::lookup(alias);
    let config_path = loader.config_path().expect("config path");
    let config_value = loader_config::load_config_value(config_path).expect("config");
    loader_config::manifest_from_config(alias, &config_value, overrides).expect("manifest")
}

#[test]
fn performance_harness_reports_throughput_and_latency() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("Qwen/Qwen3-0.6B", &files);

    let mut total_dims = 0usize;
    let start = Instant::now();
    for _ in 0..3 {
        let embedding = executor.embed("tok1 tok2").expect("embed");
        total_dims = total_dims.saturating_add(embedding.len().max(1));
    }
    let elapsed = start.elapsed();
    let throughput = total_dims as f32 / elapsed.as_secs_f32().max(1e-6);

    assert!(throughput.is_finite() && throughput > 0.0);
    assert!(elapsed.as_millis() < 500, "benchmark should stay CI-fast");
}

#[test]
fn performance_harness_checks_memory_pressure() {
    let files = TestModelFiles::new().expect("test model files");
    let executor = build_executor("Qwen/Qwen3-0.6B", &files);
    let pressure = executor
        .backend()
        .get_memory_pressure()
        .expect("memory pressure");
    assert!(
        (0.0..=1.0).contains(&pressure),
        "memory pressure should be normalized"
    );
}
