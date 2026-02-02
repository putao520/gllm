mod common;

use std::time::Instant;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::Backend;

fn build_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let manifest = registry::lookup(alias).expect("manifest");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut loader = files.loader(alias).expect("loader");
    Executor::from_loader(backend, manifest, adapter, &mut loader).expect("executor")
}

#[test]
fn performance_harness_reports_throughput_and_latency() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("qwen3-7b", &files);

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
    let executor = build_executor("qwen3-7b", &files);
    let pressure = executor.backend().get_memory_pressure().expect("memory pressure");
    assert!(
        (0.0..=1.0).contains(&pressure),
        "memory pressure should be normalized"
    );
}
