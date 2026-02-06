mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::loader::config as loader_config;
use gllm::manifest::ModelKind;
use gllm_kernels::cpu_backend::CpuBackend;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const DATA_DIR: &str = "tests/e2e_alignment/data";
// Python uses float32, Rust uses float32.
// Expect high precision, but allow small error due to math library differences (e.g. exp/tanh implementation)
const TOLERANCE_FP32: f32 = 1e-4;

fn load_golden(filename: &str, tensor_name: &str) -> Option<Vec<f32>> {
    let path = Path::new(DATA_DIR).join(filename);
    if !path.exists() {
        eprintln!("⚠️ Golden data not found at {:?}. Skipping test.", path);
        return None;
    }
    let file = File::open(&path).expect("failed to open golden file");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("failed to mmap") };
    let tensors = SafeTensors::deserialize(&mmap).expect("failed to parse safetensors");
    let view = tensors.tensor(tensor_name).expect("tensor not found");

    let data = view.data();
    Some(
        data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
    )
}

fn assert_aligned(rust: &[f32], py: &[f32], tolerance: f32, context: &str) {
    assert_eq!(rust.len(), py.len(), "{}: Dimension mismatch", context);
    let mut max_diff = 0.0;
    let mut mismatch_count = 0;

    for (i, (r, p)) in rust.iter().zip(py.iter()).enumerate() {
        let diff = (r - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance {
            if mismatch_count < 5 {
                eprintln!(
                    "❌ {}: Mismatch at {}: rust={}, py={}, diff={}",
                    context, i, r, p, diff
                );
            }
            mismatch_count += 1;
        }
    }

    if mismatch_count > 0 {
        panic!(
            "{}: Alignment failed! {} mismatches. Max diff: {}",
            context, mismatch_count, max_diff
        );
    }
    println!("✅ {} Aligned! Max diff: {:.2e}", context, max_diff);
}

fn build_executor(alias: &str, kind: ModelKind, files: &TestModelFiles) -> Executor<CpuBackend> {
    let mut loader = files.loader(alias).expect("loader");
    let config_path = loader.config_path().expect("config path");
    let config_value = loader_config::load_config_value(config_path).expect("config");
    let manifest =
        loader_config::manifest_from_config(alias, &config_value, kind).expect("manifest");

    loader.set_manifest_if_missing(&manifest);
    let adapter = adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();

    Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader).expect("executor")
}

/// TEST-ALIGN-001: Embedding 对齐 (HuggingFaceTB/SmolLM-135M-Instruct)
///
/// **关联需求**: REQ-TEST-011
/// **测试类型**: 跨语言一致性测试
/// **说明**: 使用 Decoder-only 模型进行 Embedding 对齐测试，因为当前 CPU Backend 仅支持 Causal Attention。
#[test]
fn test_alignment_embedding() {
    let Some(golden) = load_golden("golden_embedding.safetensors", "embeddings") else {
        return;
    };

    let files = TestModelFiles::new().expect("test files");
    // Use SmolLM for embedding test too, as it's a supported architecture (Llama)
    let mut executor = build_executor(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        ModelKind::Chat,
        &files,
    );

    let input = "Hello world from Rust alignment test";
    // executor.embed() returns the last hidden state (before head)
    let rust_emb = executor.embed(input).expect("rust embed failed");

    assert_aligned(&rust_emb, &golden, TOLERANCE_FP32, "Embedding");
}

/// TEST-ALIGN-003: Generation Logits 对齐 (HuggingFaceTB/SmolLM-135M-Instruct)
///
/// **关联需求**: REQ-TEST-011
/// **测试类型**: 跨语言一致性测试
#[test]
fn test_alignment_generation() {
    let Some(golden) = load_golden("golden_generation.safetensors", "logits") else {
        return;
    };

    let files = TestModelFiles::new().expect("test files");
    let mut executor = build_executor(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        ModelKind::Chat,
        &files,
    );

    let prompt = "The capital of France is";
    let tokens = executor.encode_prompt(prompt).expect("encode");

    // 执行 Forward，获取最后一个 token 的 logits
    let logits_handle = executor.forward_step(&tokens).expect("forward");

    // 从 Backend 获取 Logits 数据
    let backend = executor.backend();
    let rust_logits = backend.read_logits(&logits_handle).expect("read logits");

    // Rust backend currently only computes/stores logits for the LAST token (next token prediction)
    // So rust_logits size should be equal to vocab_size
    let vocab_size = executor.model_config().vocab_size;

    assert_eq!(
        rust_logits.len(),
        vocab_size,
        "Logits size mismatch: expected vocab_size"
    );

    // Golden data is also only for the LAST token
    assert_aligned(&rust_logits, &golden, 1e-4, "Generation Logits");
}

/// TEST-ALIGN-002: Rerank Fallback 对齐 (HuggingFaceTB/SmolLM-135M-Instruct)
///
/// **关联需求**: REQ-TEST-011
/// **测试类型**: 跨语言一致性测试
/// **说明**: 验证 Rerank 接口的数据流。由于模型无 Score Head，预期回退到输出 Embedding。
#[test]
fn test_alignment_rerank() {
    let Some(golden) = load_golden("golden_rerank.safetensors", "scores") else {
        return;
    };

    let files = TestModelFiles::new().expect("test files");
    let mut executor = build_executor(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        ModelKind::Reranker,
        &files,
    );

    // Rerank interface usually takes (query, candidates).
    // For this test we manually concatenate to match the Python script's single string.
    // In real usage, the Reranker adapter handles this concatenation.
    // But here we are testing the Executor::rerank -> Backend flow directly.
    // Executor::rerank takes a single string `input`.
    let input = "query hello world doc this is a test";

    // This should call backend.rerank_forward_gpu_pure
    // Which falls back to last_hidden because no score head found
    let rust_emb = executor.rerank(input).expect("rust rerank failed");

    assert_aligned(&rust_emb, &golden, TOLERANCE_FP32, "Rerank Fallback");
}
