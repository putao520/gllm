mod common;

use std::time::Instant;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::loader::{config as loader_config, Loader};
use gllm::manifest::{ModelKind, ModelManifest};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::Backend;
use std::sync::Arc;

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

/// TEST-PERF-001: 性能测试报告吞吐量和延迟
///
/// **关联需求**: REQ-TEST-008
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 执行多次 embedding
/// 2. 计算吞吐量
/// 3. 验证吞吐量为有限正值
///
/// **期望结果**: 吞吐量为有限正值，延迟 < 500ms
#[test]
fn performance_harness_reports_throughput_and_latency() {
    let files = TestModelFiles::new().expect("test model files");
    let mut executor = build_executor("Qwen/Qwen3-0.6B", ModelKind::Chat, &files);

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

/// TEST-PERF-002: 性能测试检查内存压力
///
/// **关联需求**: REQ-TEST-008
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 获取后端内存压力
/// 2. 验证压力值在 [0, 1] 范围内
///
/// **期望结果**: 内存压力为归一化值 (0.0 ~ 1.0)
#[test]
fn performance_harness_checks_memory_pressure() {
    let files = TestModelFiles::new().expect("test model files");
    let executor = build_executor("Qwen/Qwen3-0.6B", ModelKind::Chat, &files);
    let pressure = executor
        .backend()
        .get_memory_pressure()
        .expect("memory pressure");
    assert!(
        (0.0..=1.0).contains(&pressure),
        "memory pressure should be normalized"
    );
}
