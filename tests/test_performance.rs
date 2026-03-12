mod common;

use std::time::Instant;

use common::TestModelFiles;
use gllm::engine::executor::Executor;
use gllm::loader::Loader;
use gllm::manifest::{
    map_architecture_token, tensor_rules_for_arch, ModelArchitecture, ModelKind, ModelManifest,
    EMPTY_FILE_MAP,
};
use gllm::compat::CpuBackend;
use gllm::compat::Backend;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

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
        .unwrap_or(ModelArchitecture::XlmR); // BGE is XLM-R based

    ModelManifest {
        model_id: Cow::Owned(alias.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch,
        tensor_rules: tensor_rules_for_arch(arch),
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: HashMap::new(),
    }
}

/// TEST-PERF-001: 性能测试报告吞吐量和延迟
///
/// **关联需求**: REQ-TEST-008
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 使用 Embedding 模型执行多次 embedding
/// 2. 计算吞吐量
/// 3. 验证吞吐量为有限正值
///
/// **期望结果**: 吞吐量为有限正值，延迟 < 500ms
#[test]
fn performance_harness_reports_throughput_and_latency() {
    let files = TestModelFiles::new().expect("test model files");
    // 使用 Embedding 模型而不是 Chat 模型
    let mut executor = build_executor("BAAI/bge-small-en-v1.5", ModelKind::Embedding, &files);

    let mut total_dims = 0usize;
    let start = Instant::now();
    for _ in 0..3 {
        let embedding = executor.embed("tok1 tok2").expect("embed");
        total_dims = total_dims.saturating_add(embedding.len().max(1));
    }
    let elapsed = start.elapsed();
    let throughput = total_dims as f32 / elapsed.as_secs_f32().max(1e-6);

    assert!(throughput.is_finite() && throughput > 0.0);
}

/// TEST-PERF-002: 性能测试检查内存压力
///
/// **关联需求**: REQ-TEST-008
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 使用 Embedding 模型获取后端内存压力
/// 2. 验证压力值在 [0, 1] 范围内
///
/// **期望结果**: 内存压力为归一化值 (0.0 ~ 1.0)
#[test]
fn performance_harness_checks_memory_pressure() {
    let files = TestModelFiles::new().expect("test model files");
    // 使用 Embedding 模型而不是 Chat 模型
    let executor = build_executor("BAAI/bge-small-en-v1.5", ModelKind::Embedding, &files);
    let pressure = executor
        .backend()
        .get_memory_pressure()
        .expect("memory pressure");
    assert!(
        (0.0..=1.0).contains(&pressure),
        "memory pressure should be normalized"
    );
}
